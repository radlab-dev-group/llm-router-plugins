"""
Keeping one command generation's context stable.

Codex asks several questions per command: run this, read that, apply a patch.
Each arrives as a separate request, and each would re-search the knowledge base
and — because retrieval is not a pure function of the query — possibly get a
different answer.  An agent whose context changes between two steps of one
thought reads that as the world having changed, and wastes a step working out
whether it did.

So the assembled block is cached for the identity of the generation, and reused
until something in the turn actually moves: the work mode changes, or the agent
produced new phase evidence.  The cache is also where escalation remembers
itself, so a loop that was escalated once is not downgraded by the next request.

Without Redis the plugin does not share state — it degrades to a per-process
counter that bounds how much work one generation may cause, and says so in the
request annotation rather than pretending to be consistent across workers.
"""

import dataclasses
import json
import threading
import time

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from llm_router_plugins.utils.knowledge.agentic_knowledge.assembler import (
    KnowledgeContext,
    KnowledgeHit,
    best_recipe_score,
    deadends_of,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.doc_context import (
    DocContext,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.redisclient import (
    RedisClientHandle,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.state import (
    RedisConnectionSettings,
)

__all__ = [
    "GenerationContextCache",
    "InMemoryContextCache",
    "LookupLimiter",
    "STATE_CACHE_DISABLED",
    "STATE_HIT",
    "STATE_MISS",
    "STATE_UNAVAILABLE",
    "clear_memory_cache",
]

#: A cached block was reused; no search was made.
STATE_HIT = "hit"

#: Nothing cached; the caller will search.
STATE_MISS = "miss"

#: Redis is not configured; the plugin is stateless.
STATE_CACHE_DISABLED = "disabled"

#: Redis was tried and failed.
STATE_UNAVAILABLE = "unavailable"

#: Shape of a serialized entry.  Version ``2`` added the documentation
#: fields of a hit (`source`, `doc_type`, `heading`) and the standing
#: documentation digest, so a cache written before them is not read as if
#: it carried none.
_SCHEMA = 2


@dataclass
class CachedContext:
    """
    A block of knowledge, and the state attached to it.

    Parameters
    ----------
    context : KnowledgeContext
        The assembled block.
    evidence_fingerprint : str
        Identity of the phase evidence the block was built against.
    mode : str
        Work mode the block was built for.
    escalated : bool
        Whether this generation was already escalated, and must stay so.
    escalated_model : str
        Model a previous request of this generation was escalated to, so a
        later request keeps it even if its own evidence no longer repeats the
        loop; ``""`` when the generation was never escalated.
    """

    context: KnowledgeContext = field(default_factory=KnowledgeContext)
    evidence_fingerprint: str = ""
    mode: str = ""
    escalated: bool = False
    escalated_model: str = ""
    #: First failure count at which a qualifying recipe was offered, per failure.
    recipe_trials: Dict[str, int] = field(default_factory=dict)


class GenerationContextCache:
    """
    The generation cache, with Redis as its backing store.

    Parameters
    ----------
    client : Any, optional
        A Redis client.  ``None`` (with no *connection*) makes the cache report
        itself disabled.
    connection : RedisConnectionSettings, optional
        Where to connect when no client is given.  The client is then built on
        first use and rebuilt in a forked process, so a preloaded proxy does
        not hand its workers an inherited connection pool.
    key_prefix : str
        Namespace of this plugin's keys.
    ttl_seconds : int
        How long an entry survives.
    logger : logging.Logger, optional
        Where to report failures.
    """

    def __init__(
        self,
        client: Any = None,
        key_prefix: str = "agentic-kb",
        ttl_seconds: int = 3600,
        logger: Any = None,
        connection: Optional[RedisConnectionSettings] = None,
    ) -> None:
        """Bind the store and the entry lifetime."""
        self._handle = RedisClientHandle(connection, client, logger)
        self._prefix = key_prefix
        self._ttl = int(ttl_seconds)
        self._logger = logger

    @property
    def enabled(self) -> bool:
        """Whether anything can be cached at all."""
        return self._handle.configured

    def get(self, scope: str, generation: str) -> Optional[CachedContext]:
        """
        Return the cached block of one generation.

        Parameters
        ----------
        scope : str
            Repository scope.
        generation : str
            Identity of the command generation.

        Returns
        -------
        CachedContext or None
            The entry, or ``None`` when there is none or Redis did not answer.
        """
        if not scope or not generation:
            return None
        client = self._handle.get()
        if client is None:
            return None
        try:
            raw = client.get(self._key(scope, generation))
        except Exception as exc:
            self._warn("knowledge context cache read failed: %s", type(exc).__name__)
            return None
        if not raw:
            return None
        return _decode(raw)

    def put(
        self,
        scope: str,
        generation: str,
        entry: CachedContext,
    ) -> bool:
        """
        Store the block of one generation.

        Parameters
        ----------
        scope : str
            Repository scope.
        generation : str
            Identity of the command generation.
        entry : CachedContext
            What to remember.

        Returns
        -------
        bool
            ``True`` when the entry was written.
        """
        if not scope or not generation:
            return False
        client = self._handle.get()
        if client is None:
            return False
        try:
            client.set(self._key(scope, generation), _encode(entry), ex=self._ttl)
        except Exception as exc:
            self._warn(
                "knowledge context cache write failed: %s", type(exc).__name__
            )
            return False
        return True

    def mark_escalated(self, scope: str, generation: str) -> None:
        """
        Remember that this generation was escalated.

        A failed write is not worth an error: the cost is one request routed by
        routing again, not a broken request.

        Parameters
        ----------
        scope : str
            Repository scope.
        generation : str
            Identity of the command generation.
        """
        entry = self.get(scope, generation) or CachedContext()
        self.put(
            scope,
            generation,
            dataclasses.replace(entry, escalated=True),
        )

    def is_escalated(self, scope: str, generation: str) -> bool:
        """
        Return whether this generation was already escalated.

        Parameters
        ----------
        scope : str
            Repository scope.
        generation : str
            Identity of the command generation.

        Returns
        -------
        bool
            ``True`` when a previous request of this generation escalated.
        """
        entry = self.get(scope, generation)
        return bool(entry and entry.escalated)

    def _key(self, scope: str, generation: str) -> str:
        """Build one cache key."""
        return f"{self._prefix}:ctx:{scope}:{generation}"

    def _warn(self, message: str, *args: Any) -> None:
        """Log a warning when a logger was given."""
        if self._logger is not None:
            self._logger.warning(message, *args)


class InMemoryContextCache(GenerationContextCache):
    """
    A cache that lives in this process only.

    For tests and for a single-worker deployment.  It implements exactly the
    same contract, which is the point: the plugin cannot tell it from Redis, so
    the code path that ships is the one that is tested — including the two
    things Redis does for free and this class used not to do at all: an entry
    older than the TTL is dropped, and the store stops growing.

    Parameters
    ----------
    ttl_seconds : int
        How long an entry survives, as it does in Redis.
    max_entries : int
        Ceiling of remembered generations; the oldest is evicted to hold it, so
        a long-lived proxy process cannot accumulate a block per request.
    """

    def __init__(self, ttl_seconds: int = 3600, max_entries: int = 2048) -> None:
        """Start with an empty store."""
        super().__init__(client=None, ttl_seconds=ttl_seconds)
        self._entries: Dict[str, Tuple[float, str]] = {}
        self._max_entries = max(1, int(max_entries))
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        """This cache always accepts entries."""
        return True

    def get(self, scope: str, generation: str) -> Optional[CachedContext]:
        """Return the entry stored for one generation, unless it has aged out."""
        if not scope or not generation:
            return None
        key = f"{scope}:{generation}"
        with self._lock:
            stored = self._entries.get(key)
            if stored is None:
                return None
            stored_at, raw = stored
            if self._ttl > 0 and time.time() - stored_at > self._ttl:
                # Reported as a miss and replaced on the next put, exactly as
                # Redis would have expired it.
                del self._entries[key]
                return None
        return _decode(raw) if raw else None

    def put(self, scope: str, generation: str, entry: CachedContext) -> bool:
        """Store an entry for one generation, evicting the oldest when full."""
        if not scope or not generation:
            return False
        key = f"{scope}:{generation}"
        with self._lock:
            self._entries.pop(key, None)
            self._entries[key] = (time.time(), _encode(entry))
            while len(self._entries) > self._max_entries:
                # Insertion order: the generation written longest ago goes first.
                self._entries.pop(next(iter(self._entries)))
        return True


class LookupLimiter:
    """
    A per-process ceiling on how often one generation may search.

    Only a bound, never a guarantee: it holds for the requests that happen to
    land on this worker.  That is stated in the annotation as
    ``cache=disabled`` so an operator reading a dashboard is not told a
    guarantee the plugin cannot make.
    """

    def __init__(self, max_per_generation: int = 20, max_keys: int = 512) -> None:
        """Bind the ceiling and how many generations are remembered."""
        self._limit = int(max_per_generation)
        self._max_keys = int(max_keys)
        self._counts: Dict[str, int] = {}
        self._order: List[str] = []
        self._lock = threading.Lock()

    def allow(self, scope: str, generation: str) -> bool:
        """
        Return whether this generation may search again.

        Parameters
        ----------
        scope : str
            Repository scope.
        generation : str
            Identity of the command generation.

        Returns
        -------
        bool
            ``False`` once the generation already searched :attr:`_limit`
            times on this worker.
        """
        key = f"{scope}:{generation}"
        with self._lock:
            used = self._counts.get(key, 0)
            if used >= self._limit:
                return False
            self._counts[key] = used + 1
            self._order.append(key)
            while len(self._order) > self._max_keys:
                self._counts.pop(self._order.pop(0), None)
            return True

    def reset(self) -> None:
        """Forget every generation; for tests."""
        with self._lock:
            self._counts.clear()
            self._order.clear()


def _encode(entry: CachedContext) -> str:
    """Serialize one cache entry."""
    payload = {
        "schema": _SCHEMA,
        "evidence": entry.evidence_fingerprint,
        "mode": entry.mode,
        "escalated": bool(entry.escalated),
        "escalated_model": str(entry.escalated_model or ""),
        "recipe_trials": dict(entry.recipe_trials),
        "chars": int(entry.context.chars),
        "status": entry.context.status,
        "rendered": entry.context.rendered,
        "hits": [
            {
                "category": hit.category,
                "document_name": hit.document_name,
                "text": hit.text,
                "score": hit.score,
                "weight": hit.weight,
                "created_at": hit.created_at,
                "paths": list(hit.paths),
                "signature": hit.signature,
                "signature_hash": hit.signature_hash,
                "test_command": hit.test_command,
                "source": hit.source,
                "doc_type": hit.doc_type,
                "heading": hit.heading,
            }
            for hit in entry.context.hits
        ],
    }
    digest = entry.context.doc_context
    if digest is not None and not digest.empty:
        payload["doc_context"] = {
            "scope": digest.scope,
            "text": digest.text,
            "files": list(digest.source_files),
            "fingerprint": digest.fingerprint,
            "truncated": digest.truncated,
        }
    return json.dumps(payload, ensure_ascii=False)


def _decode_doc_context(raw: Any) -> Optional[DocContext]:
    """
    Rebuild the standing documentation digest of a cache entry.

    A malformed or absent digest decodes to ``None``, which means the second
    request of a generation is served the block the first one rendered — the
    digest is already inside ``rendered`` — and simply is not reported again in
    the annotation.  The block a request is shown and the block the cache holds
    are never allowed to differ.
    """
    if not isinstance(raw, dict):
        return None
    text = str(raw.get("text") or "").strip()
    if not text:
        return None
    files = raw.get("files")
    return DocContext(
        scope=str(raw.get("scope") or ""),
        text=text,
        source_files=(
            tuple(str(path) for path in files if str(path).strip())
            if isinstance(files, (list, tuple))
            else ()
        ),
        fingerprint=str(raw.get("fingerprint") or ""),
        truncated=bool(raw.get("truncated")),
    )


#: The key namespaces that hold only cached state, each rebuildable by the
#: layer itself: a generation's assembled context, a repository's scanned
#: documentation (the scan state ``docs:<scope>`` and the standing digest
#: ``docs:context:<scope>`` under one pattern), and the publisher's
#: deduplication claims, which would otherwise swallow a re-publication of
#: unchanged documentation as "already seen".  What is deliberately absent is
#: the event stream and the consumer group, which are a queue, not a cache —
#: cleaning them would lose episodes that were published and not yet indexed —
#: and the worker's attempt counters, which belong to stream entries.
CACHE_NAMESPACES: Tuple[str, ...] = ("ctx", "docs", "seen")


def clear_memory_cache(
    client: Any,
    key_prefix: str,
    namespaces: Tuple[str, ...] = CACHE_NAMESPACES,
    batch_size: int = 500,
) -> Dict[str, int]:
    """
    Delete every cached key of one plugin's namespaces and count them.

    The key is built from a literal namespace segment, never from a bare
    ``*`` over the whole prefix: the event stream lives in the same database
    and its name is not a cache, so a wildcard the operator cannot see must
    not reach it.  Deletion runs over ``SCAN`` cursors in batches, which
    neither blocks a busy server the way ``KEYS`` would nor load a large
    keyspace into memory in one call.

    Parameters
    ----------
    client : Any
        A Redis client with ``scan_iter`` and ``delete``.
    key_prefix : str
        The namespace of this plugin's keys; empty input is refused rather
        than scanned, because an empty prefix would widen the match beyond
        what it names.
    namespaces : Tuple[str, ...], optional
        Which namespace segments to clear.
    batch_size : int
        How many keys one ``DELETE`` carries.

    Returns
    -------
    Dict[str, int]
        One count per namespace, in the order asked.

    Raises
    ------
    ValueError
        When *key_prefix* is empty.
    """
    if not str(key_prefix or "").strip():
        raise ValueError("key_prefix must not be empty")
    counts: Dict[str, int] = {}
    for namespace in namespaces:
        pattern = f"{key_prefix}:{namespace}:*"
        removed = 0
        batch: List[str] = []
        for key in client.scan_iter(match=pattern, count=batch_size):
            batch.append(key)
            if len(batch) >= batch_size:
                removed += int(client.delete(*batch) or 0)
                batch = []
        if batch:
            removed += int(client.delete(*batch) or 0)
        counts[namespace] = removed
    return counts


def _decode(raw: Any) -> Optional[CachedContext]:
    """
    Rebuild one cache entry, dropping anything this reader cannot trust.

    Never raises.  A cache value is foreign input: it may have been written by
    an earlier build that shared the schema number, or by anything else with
    access to the key prefix, and a single field of the wrong type would
    otherwise propagate out of the cache read — and out of ``apply()`` — turning
    a knowledge-cache hiccup into a failed request.  An entry that does not
    decode is a miss, nothing more.
    """
    try:
        payload = json.loads(raw) if isinstance(raw, (str, bytes)) else dict(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
        return None
    try:
        stored_hits = payload.get("hits") or ()
        if not isinstance(stored_hits, (list, tuple)):
            stored_hits = ()
        hits = tuple(
            KnowledgeHit(
                category=str(item.get("category", "")),
                document_name=str(item.get("document_name", "")),
                text=str(item.get("text", "")),
                score=float(item.get("score") or 0.0),
                weight=float(item.get("weight") or 0.0),
                created_at=item.get("created_at"),
                paths=tuple(item.get("paths") or ()),
                signature=str(item.get("signature", "")),
                signature_hash=str(item.get("signature_hash") or ""),
                test_command=str(item.get("test_command") or ""),
                source=str(item.get("source", "")),
                doc_type=str(item.get("doc_type", "")),
                heading=str(item.get("heading", "")),
            )
            for item in stored_hits
            if isinstance(item, dict)
        )
        rendered = str(payload.get("rendered", ""))
        doc_context = _decode_doc_context(payload.get("doc_context"))
        context = KnowledgeContext(
            hits=hits,
            rendered=rendered,
            chars=int(payload.get("chars") or len(rendered)),
            # Recomputed, not restored: the escalation policy reads these two,
            # and an entry that came back with a zero score would read as "this
            # repo knows nothing" on the second request of a generation that was
            # shown a strong recipe on the first.
            best_recipe_score=best_recipe_score(hits),
            deadends=deadends_of(hits),
            status=str(payload.get("status", "ok")),
            doc_context=doc_context,
        )
        stored_trials = payload.get("recipe_trials") or {}
        if not isinstance(stored_trials, dict):
            stored_trials = {}
        recipe_trials = {
            key: value
            for key, value in stored_trials.items()
            if isinstance(key, str) and key and type(value) is int and value > 0
        }
        return CachedContext(
            context=context,
            evidence_fingerprint=str(payload.get("evidence", "")),
            mode=str(payload.get("mode", "")),
            escalated=bool(payload.get("escalated")),
            escalated_model=str(payload.get("escalated_model", "")),
            recipe_trials=recipe_trials,
        )
    except (TypeError, ValueError):
        return None
