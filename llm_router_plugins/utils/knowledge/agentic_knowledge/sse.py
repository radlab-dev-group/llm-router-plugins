"""
Reading the knowledge base, without ever breaking a request.

One call, one short timeout, no retry: the search runs in the path of a live
agent request, and an agent waiting on a knowledge base has already lost.  Any
failure — the library missing, the engine unreachable, a malformed answer — is
a status, not an exception, and the request proceeds as if the knowledge base
were empty.

The engine reports a document's name and its text but not its metadata, so the
two are read back from where the plugin can see them: the name gives the kind
and the retention day, the embedded header gives the failure signature, the
paths and the thread (see
:mod:`llm_router_plugins.utils.knowledge.agentic_knowledge.documents`).

One engine detail shapes how filters may be written: the backend combines
several ``metadata_filters`` with **OR**, whatever the caller meant, so this
module sends at most one — the thread an archived window belongs to.  Isolation
between repositories does not come from a filter at all; it comes from there
being one collection per repository, so a forgotten filter cannot leak another
team's fixes into the prompt.
"""

import logging

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple

__all__ = [
    "KnowledgeStore",
    "SearchRequest",
    "SearchResult",
    "StoreHit",
    "STATUS_DISABLED",
    "STATUS_OK",
    "STATUS_UNAVAILABLE",
]

#: The search answered.
STATUS_OK = "ok"

#: The search is switched off or unconfigured.
STATUS_DISABLED = "disabled"

#: The search was attempted and failed, or its library is not installed.
STATUS_UNAVAILABLE = "unavailable"

_MAX_HIT_CHARS = 1200


@dataclass(frozen=True)
class SearchRequest:
    """
    One search the assembler wants run.

    Parameters
    ----------
    collection : str
        Collection to search.
    query : str
        Query text built for the work mode.
    categories : Tuple[str, ...]
        Document categories to restrict the search to.
    max_results : int
        Most chunks to ask for.
    thread_id : str
        Restrict to one conversation thread; the archive's filter, ``""`` for
        everything else.
    """

    collection: str
    query: str
    categories: Tuple[str, ...] = ()
    max_results: int = 6
    thread_id: str = ""


@dataclass(frozen=True)
class StoreHit:
    """
    One retrieved chunk, with what the plugin can read about it.

    Parameters
    ----------
    collection : str
        Collection the hit came from — the repository it belongs to.
    document_name : str
        Engine-level document name.
    text : str
        The chunk, header not yet removed.
    score : float or None
        Raw retrieval score.
    score_normalized : float or None
        Relevance on the engine's common ``[0, 1]`` scale.
    relative_path : str
        Path the engine associates with the document, when it has one.
    """

    collection: str = ""
    document_name: str = ""
    text: str = ""
    score: Optional[float] = None
    score_normalized: Optional[float] = None
    relative_path: str = ""


@dataclass(frozen=True)
class SearchResult:
    """
    Outcome of one search.

    Parameters
    ----------
    status : str
        :data:`STATUS_OK`, :data:`STATUS_DISABLED` or :data:`STATUS_UNAVAILABLE`.
    hits : Tuple[StoreHit, ...]
        Retrieved chunks; empty unless the search answered.
    reason : str
        Why a search did not answer; a class name, never a payload.
    """

    status: str = STATUS_DISABLED
    hits: Tuple[StoreHit, ...] = ()
    reason: str = ""


class KnowledgeStore:
    """
    Thin, fail-open facade over :class:`sse_lib.SSEClient`.

    Parameters
    ----------
    host : str
        Engine host; empty means the store is unavailable.
    api_prefix : str
        API prefix of the engine.
    token : str
        Bearer token.
    timeout_seconds : float
        Ceiling of one call.
    verify : bool
        Whether TLS certificates are verified.
    min_similarity : float
        Cosine floor applied to every search.
    logger : logging.Logger, optional
        Where to report failures; never with payload content.
    client : Any, optional
        Pre-built client.  Tests pass a fake one here.
    """

    def __init__(
        self,
        host: str = "",
        api_prefix: str = "/api/v1",
        token: str = "",
        timeout_seconds: float = 1.5,
        verify: bool = True,
        min_similarity: float = 0.35,
        logger: Optional[logging.Logger] = None,
        client: Any = None,
    ) -> None:
        """Bind the connection; build no client until the first search."""
        self._host = (host or "").strip()
        self._api_prefix = (api_prefix or "/api/v1").strip()
        self._token = (token or "").strip()
        self._timeout = float(timeout_seconds)
        self._verify = bool(verify)
        self._min_similarity = float(min_similarity)
        self._logger = logger
        self._client = client
        self._unavailable_reason = (
            "" if (client is not None or self._host) else ("no host configured")
        )

    @property
    def available(self) -> bool:
        """Whether a search could be attempted at all."""
        return self._client is not None or bool(self._host)

    @property
    def reason_unavailable(self) -> str:
        """Why the store cannot search, for a single log line."""
        return self._unavailable_reason

    def search(self, request: SearchRequest) -> SearchResult:
        """
        Run *request* against the engine.

        Parameters
        ----------
        request : SearchRequest
            What to look for.

        Returns
        -------
        SearchResult
            The hits, or the reason there are none.
        """
        if not self.available:
            return SearchResult(STATUS_DISABLED, reason=self._unavailable_reason)
        if not request.query.strip():
            return SearchResult(STATUS_OK)
        try:
            client = self._ensure_client()
        except Exception as exc:
            self._warn("knowledge store is unavailable: %s", type(exc).__name__)
            self._unavailable_reason = type(exc).__name__
            return SearchResult(STATUS_UNAVAILABLE, reason="client")
        try:
            response = client.search(
                request.collection,
                request.query,
                _options(request, self._min_similarity),
            )
        except Exception as exc:  # sse_lib.SSEError and anything transport-shaped
            self._warn(
                "knowledge search failed (%s): %s",
                request.collection,
                type(exc).__name__,
            )
            return SearchResult(STATUS_UNAVAILABLE, reason=type(exc).__name__)
        return SearchResult(
            STATUS_OK, hits=tuple(_hits(request.collection, response))
        )

    def close(self) -> None:
        """
        Release the connection, if one was built and knows how.

        Returns
        -------
        None
        """
        client = self._client
        if client is None:
            return
        for name in ("close", "shutdown"):
            hook = getattr(client, name, None)
            if callable(hook):
                try:
                    hook()
                except Exception:  # closing must not raise
                    return
                return

    # -------------------------------------------------------------- helpers -
    def _ensure_client(self) -> Any:
        """Build the engine client on first use."""
        if self._client is not None:
            return self._client
        from sse_lib import SSEClient  # imported lazily: an optional dependency

        self._client = SSEClient(
            self._host,
            api_prefix=self._api_prefix,
            token=self._token,
            timeout=self._timeout,
            verify=self._verify,
        )
        return self._client

    def _warn(self, message: str, *args: Any) -> None:
        """Log a warning when a logger was given."""
        if self._logger is not None:
            self._logger.warning(message, *args)


def _options(request: SearchRequest, min_similarity: float) -> Any:
    """
    Build the search options of one request, as the engine expects them.

    A mapping rather than a ``SearchOptions`` instance, so that building a
    query never requires the optional dependency to be installed; the client
    accepts either.

    Parameters
    ----------
    request : SearchRequest
        What to look for.
    min_similarity : float
        Cosine floor of the vector side of the search.

    Returns
    -------
    Dict[str, Any]
        One accepted ``SearchOptions`` payload.
    """
    options: dict = {
        "max_results": int(request.max_results),
        "hybrid_search": True,
        "rerank_results": True,
        "min_similarity": float(min_similarity),
        # Categories and the metadata filter have to intersect; the engine's
        # default would union them and return the whole collection.
        "use_and_operator": True,
    }
    if request.categories:
        options["categories"] = list(request.categories)
    if request.thread_id:
        options["metadata_filters"] = [
            {"operator": "eq", "field": {"thread_id": request.thread_id}}
        ]
    return options


def _hits(collection: str, response: Any) -> Iterable[StoreHit]:
    """Turn one engine response into hits, tolerating an unusual body."""
    for hit in _iter_hits(response):
        raw = getattr(hit, "raw", None) or {}
        text = getattr(hit, "text", None) or _text_of(raw)
        if not text:
            continue
        yield StoreHit(
            collection=collection,
            document_name=str(getattr(hit, "document_name", "") or ""),
            text=str(text)[:_MAX_HIT_CHARS],
            score=_number(getattr(hit, "score", None)),
            score_normalized=_number(getattr(hit, "score_normalized", None)),
            relative_path=str(getattr(hit, "relative_filepath", "") or ""),
        )


def _iter_hits(response: Any) -> List[Any]:
    """Return the hit objects of a response, whichever shape it came in."""
    hits = getattr(response, "hits", None)
    if callable(hits):
        try:
            return list(hits())
        except Exception:
            return []
    if isinstance(hits, (list, tuple)):
        return list(hits)
    if isinstance(response, dict):
        results = response.get("results") or {}
        detailed = (
            results.get("detailed_results") or response.get("detailed_results") or {}
        )
        entries = (
            list(detailed.values())
            if isinstance(detailed, dict)
            else list(detailed or [])
        )
        return [_flatten(entry) for entry in entries]
    return []


def _flatten(entry: Any) -> Any:
    """Flatten one raw ``detailed_results`` entry into a namespace."""
    if not isinstance(entry, dict):
        return entry
    inner = (entry.get("result") or {}).get("text") or {}
    merged = {**inner, **{k: v for k, v in entry.items() if k != "result"}}
    return _Entry(
        document_name=merged.get("document_name"),
        relative_filepath=merged.get("relative_path"),
        text=merged.get("text_str"),
        score=merged.get("score"),
        score_normalized=merged.get("score_normalized"),
    )


class _Entry:
    """Minimal stand-in for ``sse_lib.models.SearchHit``."""

    def __init__(self, **values: Any) -> None:
        """Copy the given fields onto the instance."""
        for key, value in values.items():
            setattr(self, key, value)
        self.raw = values


def _text_of(raw: Any) -> str:
    """Read chunk text out of a raw hit dictionary."""
    if not isinstance(raw, dict):
        return ""
    return str(raw.get("text_str") or raw.get("text") or "")


def _number(value: Any) -> Optional[float]:
    """Coerce a score to a float, or ``None``."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
