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

Two engine details shape how filters may be written
(``sse_api.engine.controllers.search.semantic``).  ``use_and_operator``
(default ``False``) combines the *filter families* — categories, directly named
documents, query templates, the metadata filter, ``relative_path_contains`` —
by intersection; left alone their document names are unioned.  Within the
metadata family, though, several ``metadata_filters`` are always unioned with
each other: the controller calls its own metadata filter with
``use_and_operator=False`` hardcoded.  This module therefore sets
``use_and_operator: True`` and sends a single ``metadata_filters`` entry — the
thread an archived window belongs to, intersected with the requested categories
rather than added to them, and with nothing in it that would quietly widen the
set again.  Isolation between repositories does not come from a filter at all;
it comes from there being one collection per repository, so a forgotten filter
cannot leak another team's fixes into the prompt.
"""

import logging

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple

__all__ = [
    "KnowledgeStore",
    "SCALE_COSINE",
    "SCALE_RRF",
    "SCALE_RERANKER",
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

#: :attr:`SearchResult.score_scale` — cosine similarity of the vector leg.
SCALE_COSINE = "cosine"

#: :attr:`SearchResult.score_scale` — Reciprocal Rank Fusion of both legs.
SCALE_RRF = "rrf"

#: :attr:`SearchResult.score_scale` — a cross-encoder logit, squashed to
#: :math:`[0, 1]` by the sigmoid when it is normalized.
SCALE_RERANKER = "reranker"

_MAX_HIT_CHARS = 1200

#: The engine's code for "collection not found or access denied"
#: (``sse_api/engine/core/errors.py``, ``COLLECTION_ACCESS_DENIED``).  The
#: engine answers with HTTP 200 and this code in the body, so it arrives as an
#: API error rather than as an empty answer.
_COLLECTION_MISSING_CODE = "engine_001"


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
    phases : Tuple[str, ...]
        Work phases to restrict the search to — the phase an episode was mined
        in, not the document kind.  Honored only without a *thread_id*: the
        engine ORs several ``metadata_filters`` with each other, so a phase
        filter next to a thread filter would widen the archived window to every
        thread of the repository instead of narrowing it.
    modes : Tuple[str, ...]
        Routing modes to restrict the search to — the mode the router resolved
        for the request that produced the episode.  Same single metadata slot
        as *phases*, and it wins over them: one request expresses one metadata
        dimension, never a conjunction of two.
    doc_types : Tuple[str, ...]
        Documentation kinds to restrict the search to — ``readme``, ``guide``,
        ``changelog``.  The third occupant of the same metadata slot, and the
        one the ``doc`` collection is searched with; it is read only when the
        request carries neither a *thread_id* nor *modes*, which is how a
        documentation search is shaped.
    """

    collection: str
    query: str
    categories: Tuple[str, ...] = ()
    max_results: int = 6
    thread_id: str = ""
    phases: Tuple[str, ...] = ()
    modes: Tuple[str, ...] = ()
    doc_types: Tuple[str, ...] = ()


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
        Raw retrieval score, on whatever scale the run used — see
        :attr:`score_scale`.
    score_normalized : float or None
        :attr:`score` mapped onto ``[0, 1]`` by the engine.  The mapping depends
        on :attr:`score_scale` — linear for cosine, a sigmoid for a reranker
        logit, a rescaling for a fused rank — so this is *not* a cosine
        similarity and must not be compared with the configured
        ``min_similarity``, which the engine applies to the vector leg before
        any of that.  Relative order within one result set is the only safe use.
    relative_path : str
        Path the engine associates with the document, when it has one.
    score_scale : str
        ``"reranker"``, ``"rrf"``, ``"cosine"`` or ``""`` when the answer did
        not say; what the two scores mean.
    """

    collection: str = ""
    document_name: str = ""
    text: str = ""
    score: Optional[float] = None
    score_normalized: Optional[float] = None
    relative_path: str = ""
    score_scale: str = ""


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
    score_scale : str
        The scale every hit's scores were produced on, taken from the engine's
        own ``results.score_scale``: :data:`SCALE_RERANKER` under
        ``rerank_results=True``, :data:`SCALE_RRF` in a plain hybrid search,
        :data:`SCALE_COSINE` for a vector-only one, ``""`` when the answer did
        not state it.  Recorded so that no consumer mistakes
        :attr:`StoreHit.score_normalized` for the cosine similarity
        ``min_similarity`` cuts on.
    """

    status: str = STATUS_DISABLED
    hits: Tuple[StoreHit, ...] = ()
    reason: str = ""
    score_scale: str = ""


class KnowledgeStore:
    """
    Thin, fail-open facade over :class:`sse_lib.SSEClient`.

    Parameters
    ----------
    host : str
        Engine host; empty means the store is unavailable.
    api_prefix : str
        API prefix of the engine.  ``""`` — the default — leaves the choice to
        :data:`sse_lib.endpoints.DEFAULT_API_PREFIX`, which is what the
        deployment in front of us actually serves; guessing ``/api/v1`` here
        404s against an unversioned engine and fails open into a silently
        empty knowledge base.
    token : str
        Bearer token.
    username : str
        Login name, for an engine that authenticates by login rather than by a
        pre-issued token.  Used only together with *password*, and only when no
        *token* was configured: the client then logs in on its first call.
    password : str
        The matching secret.  Never logged; a name without a secret is no
        credential at all, and the client is left to stay anonymous.
    timeout_seconds : float
        Ceiling of one call.
    verify : bool
        Whether TLS certificates are verified.
    min_similarity : float
        Cosine floor of the *vector leg only*.  The engine hands it to the
        vector store before the full-text leg runs and before the two are
        fused, so a chunk found by the full-text side never meets it, and it is
        not a floor on anything the caller sees — see :func:`_options`.
    hybrid_search : bool
        Ask the engine to fuse its vector leg with its full-text leg.
    rerank_results : bool
        Ask the engine to re-score what it retrieved with the cross-encoder
        bound to the collection.  Off by default: it costs a model call per
        search, and it decides the scale every score comes back on — see
        :func:`_options`.
    logger : logging.Logger, optional
        Where to report failures; never with payload content.
    client : Any, optional
        Pre-built client.  Tests pass a fake one here.
    """

    def __init__(
        self,
        host: str = "",
        api_prefix: str = "",
        token: str = "",
        timeout_seconds: float = 1.5,
        verify: bool = True,
        min_similarity: float = 0.35,
        hybrid_search: bool = True,
        rerank_results: bool = False,
        logger: Optional[logging.Logger] = None,
        client: Any = None,
        username: str = "",
        password: str = "",
    ) -> None:
        """Bind the connection; build no client until the first search."""
        self._host = (host or "").strip()
        self._api_prefix = (api_prefix or "").strip()
        self._token = (token or "").strip()
        # A login is attempted only with both halves of a credential and no
        # token to use instead; passing ``None`` keeps the client's own
        # behaviour of never logging in.
        self._username = (username or "").strip()
        self._password = (password or "").strip()
        self._timeout = float(timeout_seconds)
        self._verify = bool(verify)
        self._min_similarity = float(min_similarity)
        self._hybrid_search = bool(hybrid_search)
        self._rerank_results = bool(rerank_results)
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
                _options(
                    request,
                    self._min_similarity,
                    hybrid_search=self._hybrid_search,
                    rerank_results=self._rerank_results,
                ),
            )
        except Exception as exc:  # sse_lib.SSEError and anything transport-shaped
            if _collection_missing(exc):
                # A repository whose first episode has not been indexed yet has
                # no collection at all, and the worker creates it on that first
                # event.  To a reader that is the same answer as an empty
                # collection — nothing known yet — so it is reported as an
                # empty result rather than as a failure on every request of a
                # young project.
                self._note(
                    "knowledge base of %s is not indexed yet", request.collection
                )
                return SearchResult(STATUS_OK, reason="no collection")
            self._warn(
                "knowledge search failed (%s): %s",
                request.collection,
                type(exc).__name__,
            )
            return SearchResult(STATUS_UNAVAILABLE, reason=type(exc).__name__)
        scale = _score_scale(response)
        return SearchResult(
            STATUS_OK,
            hits=tuple(_hits(request.collection, response, scale)),
            score_scale=scale,
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
            api_prefix=self._api_prefix or None,
            token=self._token,
            timeout=self._timeout,
            verify=self._verify,
            username=self._username or None,
            password=self._password or None,
        )
        return self._client

    def _warn(self, message: str, *args: Any) -> None:
        """Log a warning when a logger was given."""
        if self._logger is not None:
            self._logger.warning(message, *args)

    def _note(self, message: str, *args: Any) -> None:
        """Log at debug level when a logger was given."""
        if self._logger is not None:
            self._logger.debug(message, *args)


def _collection_missing(exc: Exception) -> bool:
    """
    Return whether *exc* is the engine saying it has no such collection.

    Duck-typed on ``sse_lib.SSEAPIError.errors`` — the payload the engine puts
    behind the message — because the client library is an optional dependency
    and may not be importable here.
    """
    for entry in getattr(exc, "errors", None) or ():
        code = entry.get("error_code") if isinstance(entry, dict) else None
        if code == _COLLECTION_MISSING_CODE:
            return True
    return False


def _options(
    request: SearchRequest,
    min_similarity: float,
    hybrid_search: bool = True,
    rerank_results: bool = False,
) -> Any:
    """
    Build the search options of one request, as the engine expects them.

    A mapping rather than a ``SearchOptions`` instance, so that building a
    query never requires the optional dependency to be installed; the client
    accepts either.

    The two retrieval legs are the operator's, not the caller's: they come from
    the policy, and they decide what the scores mean.  A reranked answer comes
    back on a squashed cross-encoder logit, an unreranked hybrid one on a
    Reciprocal Rank Fusion rank, and a plain vector one on cosine similarity —
    see :attr:`SearchResult.score_scale`.

    Parameters
    ----------
    request : SearchRequest
        What to look for.
    min_similarity : float
        Cosine floor of the vector leg only.  The engine passes it straight to
        the vector store (``semantic.py``, ``self.search(min_similarity=…)``),
        which runs *before* the PostgreSQL full-text leg and the Reciprocal
        Rank Fusion of the two; a chunk found only by the full-text search was
        never measured against it.  It is a pre-rerank cutoff on one leg, not a
        floor on the relevance of what comes back.
    hybrid_search : bool
        Fuse the vector leg with the full-text leg.
    rerank_results : bool
        Re-score the retrieved chunks with the collection's cross-encoder.

    Returns
    -------
    Dict[str, Any]
        One accepted ``SearchOptions`` payload.
    """
    options: dict = {
        "max_results": int(request.max_results),
        "hybrid_search": bool(hybrid_search),
        "rerank_results": bool(rerank_results),
        "min_similarity": float(min_similarity),
        # ``use_and_operator`` is what intersects the *filter families* —
        # categories, named documents, templates, the metadata filter,
        # relative_path_contains.  At the engine default of False their
        # document names are unioned and the categories only widen the set.
        "use_and_operator": True,
    }
    if request.categories:
        options["categories"] = list(request.categories)
    # Exactly one metadata filter, whatever it filters on: the engine unions
    # several ``metadata_filters`` with each other even here
    # (``semantic.py`` calls ``__filter_documents_based_on_metadata`` with
    # ``use_and_operator=False`` hardcoded), so a second one would broaden the
    # archived window instead of narrowing it.
    # One metadata dimension per request: the engine unions several
    # ``metadata_filters`` with each other, so a second dimension would widen
    # the result instead of narrowing it.  Priority: thread, mode, phase,
    # doc_type.
    if request.thread_id:
        options["metadata_filters"] = [
            {"operator": "eq", "field": {"thread_id": request.thread_id}}
        ]
    elif request.modes:
        options["metadata_filters"] = [
            {"operator": "eq", "field": {"mode": mode}} for mode in request.modes
        ]
    elif request.phases:
        # One filter per phase: the engine unions several ``metadata_filters``
        # with each other, which here is exactly "the phase is one of these".
        options["metadata_filters"] = [
            {"operator": "eq", "field": {"phase": phase}} for phase in request.phases
        ]
    elif request.doc_types:
        # Same trick for the documentation's own dimension: one ``eq`` per
        # kind, unioned by the engine into "this document is one of the kinds
        # the mode agreed to read".
        options["metadata_filters"] = [
            {"operator": "eq", "field": {"doc_type": doc_type}}
            for doc_type in request.doc_types
        ]
    return options


def _hits(
    collection: str, response: Any, score_scale: str = ""
) -> Iterable[StoreHit]:
    """
    Turn one engine response into hits, tolerating an unusual body.

    *score_scale* is what the caller read off the same answer; when it is not
    given it is read here, so a hit is never stamped with a scale nobody
    established.
    """
    scale = score_scale or _score_scale(response)
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
            score_scale=scale,
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
    """
    Flatten one raw ``detailed_results`` entry into a namespace.

    Two shapes arrive here and they name the path differently.  The reformatted
    answer the endpoint actually serves — ``reformat_search_results_to_display``,
    which is what :meth:`sse_lib.models.SearchHit.from_api` reads — carries the
    path under ``relative_filepath``, taken from the document's ``relative_path``
    column.  An *un*reformatted entry nests the database row under
    ``result.text``, where the same value is keyed ``relative_path``.  Both are
    accepted, the served spelling first.
    """
    if not isinstance(entry, dict):
        return entry
    inner = (entry.get("result") or {}).get("text") or {}
    merged = {**inner, **{k: v for k, v in entry.items() if k != "result"}}
    return _Entry(
        document_name=merged.get("document_name"),
        relative_filepath=(
            merged.get("relative_filepath") or merged.get("relative_path")
        ),
        text=merged.get("text_str"),
        score=merged.get("score"),
        score_normalized=merged.get("score_normalized"),
    )


def _score_scale(response: Any) -> str:
    """Read ``results.score_scale`` off an answer, or ``""`` when it says none.

    The engine states which scale it scored on — ``"reranker"`` under
    ``rerank_results=True``, ``"rrf"`` for a plain hybrid search,
    ``"cosine"`` for a vector-only one — and the answer is not a hint: the
    number a hit carries means something different under each of them.
    """
    candidates = []
    if isinstance(response, dict):
        results = response.get("results")
        if isinstance(results, dict):
            candidates.append(results.get("score_scale"))
        candidates.append(response.get("score_scale"))
    else:
        results = getattr(response, "results", None)
        if isinstance(results, dict):
            candidates.append(results.get("score_scale"))
        candidates.append(getattr(response, "score_scale", None))
    for value in candidates:
        if value:
            return str(value)
    return ""


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
