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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from llm_router_plugins.utils.knowledge.agentic_knowledge.assembler import (
    KnowledgeContext,
    KnowledgeHit,
)

__all__ = [
    "GenerationContextCache",
    "InMemoryContextCache",
    "LookupLimiter",
    "STATE_CACHE_DISABLED",
    "STATE_HIT",
    "STATE_MISS",
    "STATE_UNAVAILABLE",
]

#: A cached block was reused; no search was made.
STATE_HIT = "hit"

#: Nothing cached; the caller will search.
STATE_MISS = "miss"

#: Redis is not configured; the plugin is stateless.
STATE_CACHE_DISABLED = "disabled"

#: Redis was tried and failed.
STATE_UNAVAILABLE = "unavailable"

_SCHEMA = 1


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


class GenerationContextCache:
    """
    The generation cache, with Redis as its backing store.

    Parameters
    ----------
    client : Any, optional
        A Redis client.  Built by the plugin when a connection is configured;
        ``None`` makes the cache report itself disabled.
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
    ) -> None:
        """Bind the store and the entry lifetime."""
        self._client = client
        self._prefix = key_prefix
        self._ttl = int(ttl_seconds)
        self._logger = logger

    @property
    def enabled(self) -> bool:
        """Whether anything can be cached at all."""
        return self._client is not None

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
        if self._client is None or not scope or not generation:
            return None
        try:
            raw = self._client.get(self._key(scope, generation))
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
        if self._client is None or not scope or not generation:
            return False
        try:
            self._client.set(
                self._key(scope, generation), _encode(entry), ex=self._ttl
            )
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
    the code path that ships is the one that is tested.
    """

    def __init__(self, ttl_seconds: int = 3600) -> None:
        """Start with an empty store."""
        super().__init__(client=None, ttl_seconds=ttl_seconds)
        self._entries: Dict[str, str] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        """This cache always accepts entries."""
        return True

    def get(self, scope: str, generation: str) -> Optional[CachedContext]:
        """Return the entry stored for one generation."""
        if not scope or not generation:
            return None
        with self._lock:
            raw = self._entries.get(f"{scope}:{generation}")
        return _decode(raw) if raw else None

    def put(self, scope: str, generation: str, entry: CachedContext) -> bool:
        """Store an entry for one generation."""
        if not scope or not generation:
            return False
        with self._lock:
            self._entries[f"{scope}:{generation}"] = _encode(entry)
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
            }
            for hit in entry.context.hits
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


def _decode(raw: Any) -> Optional[CachedContext]:
    """Rebuild one cache entry, dropping an entry of another schema."""
    try:
        payload = json.loads(raw) if isinstance(raw, (str, bytes)) else dict(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
        return None
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
        )
        for item in payload.get("hits") or ()
        if isinstance(item, dict)
    )
    rendered = str(payload.get("rendered", ""))
    context = KnowledgeContext(
        hits=hits,
        rendered=rendered,
        chars=int(payload.get("chars") or len(rendered)),
        status=str(payload.get("status", "ok")),
    )
    return CachedContext(
        context=context,
        evidence_fingerprint=str(payload.get("evidence", "")),
        mode=str(payload.get("mode", "")),
        escalated=bool(payload.get("escalated")),
        escalated_model=str(payload.get("escalated_model", "")),
    )
