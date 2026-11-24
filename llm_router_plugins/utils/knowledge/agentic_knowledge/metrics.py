"""
The layer's own scoreboard: what the read did, per scope, per day.

The per-request annotation tells you what happened to *this* request; the
numbers a deployment needs to answer "is the knowledge layer earning its
chars?" need the same facts aggregated.  This module keeps those aggregations
in the same Redis the write path already owns, on a per-scope, per-day key,
so a scope's history is a single hash and a period's totals are a handful of
reads.

It is deliberately thin and fire-and-forget: a counter that cannot be written
is dropped, never raised, and never blocks the read.  The read path already
speaks to Redis on the write side (``record_feedback``, the publisher); one
pipelined ``HINCRBY`` at the annotation step is the same cost class, and it
is the only round trip this module adds.
"""

import datetime
import threading
from typing import Any, Dict, Optional

from llm_router_plugins.utils.knowledge.agentic_knowledge.redisclient import (
    RedisClientHandle,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.state import (
    RedisConnectionSettings,
)

__all__ = ["KnowledgeMetrics"]

#: How long a day's counters survive.  A reporting window, not a promise: long
#: enough that a month-over-month read of one scope is still possible, short
#: enough that a repository nobody works in ages out of the keyspace.
DEFAULT_METRICS_TTL_SECONDS = 2_592_000

#: The per-scope, per-day counter hash, ``{prefix}:kpi:{scope}:{YYYYMMDD}``.
_KEY_KIND = "kpi"

#: Fields one scope accumulates per day.  Kept as a constant so the writer and
#: the reader cannot drift apart on a field name.
FIELDS = (
    "turns",
    "injected_turns",
    "injected_docs",
    "failure_turns",
    "loops",
    "escalations",
    "green_after_injection",
    "feedback_up",
    "feedback_down",
    "read_failures",
    "memo_hits",
    "mined_docs",
    "mined_truncated",
)


def _day(now: Optional[datetime.datetime] = None) -> str:
    """The label of the day a counter belongs to, in the scope's local clock."""
    if now is None:
        now = datetime.datetime.now()
    return now.strftime("%Y%m%d")


class KnowledgeMetrics:
    """
    Accumulates the read path's per-scope scoreboard in Redis.

    Parameters
    ----------
    connection : RedisConnectionSettings, optional
        Where to connect.  Unconfigured makes the recorder in-process only: the
        counters still count, and ``snapshot`` reads them, but nothing survives
        a restart — a deployment without Redis measures a run, not a history.
    client : Any, optional
        A client to use as-is (tests, or an application owning its
        connections); it is never rebuilt, only reset after a fork.
    key_prefix : str
        Namespace of the counter keys, normally the plugin's own prefix so the
        counters sit beside the feedback they were built from.
    ttl_seconds : int
        How long a day's hash survives.  A day is a reporting unit, not a
        promise; long enough that a period of history is still readable, short
        enough that a scope the repository stops using ages out of the keyspace.
    logger : Any, optional
        Where to report a dropped write, at debug.
    """

    def __init__(
        self,
        connection: Optional[RedisConnectionSettings] = None,
        client: Any = None,
        key_prefix: str = "agentic-kb",
        ttl_seconds: int = DEFAULT_METRICS_TTL_SECONDS,
        logger: Any = None,
    ) -> None:
        """Bind the store and the day's lifetime; open no connection."""
        # The client is held by the same fork-aware handle the generation cache
        # uses: a proxy that preloads the plugin and forks its workers must not
        # hand two processes one connection pool, and the counters are written
        # from the request path, which is exactly where that bites.
        self._handle = RedisClientHandle(connection, client, logger)
        self._prefix = key_prefix
        self._ttl = int(ttl_seconds)
        self._logger = logger
        self._lock = threading.Lock()
        self._local: Dict[str, Dict[str, int]] = {}

    def _key(self, scope: str, day: str = "") -> str:
        return f"{self._prefix}:{_KEY_KIND}:{scope}:{day or _day()}"

    def record(self, scope: str, **counts: int) -> None:
        """
        Add ``counts`` to *scope*'s day.

        A field that is not one the module owns is ignored rather than raised
        on, because the caller is the request path and a typo in a counter name
        must not turn into a failed request.
        """
        if not scope:
            return
        day = _day()
        key = self._key(scope, day)
        with self._lock:
            local = self._local.setdefault(key, {})
            for name, value in counts.items():
                if name in FIELDS:
                    local[name] = local.get(name, 0) + int(value)
        client = self._handle.get() if self._handle.configured else None
        if client is None:
            return
        try:
            pipe = client.pipeline()
            for name, value in counts.items():
                if name in FIELDS:
                    pipe.hincrby(key, name, int(value))
            pipe.expire(key, self._ttl)
            pipe.execute()
        except (
            Exception
        ) as exc:  # a dropped counter is a missed metric, not an error
            if self._logger is not None:
                self._logger.debug(
                    "knowledge metrics write dropped: %s", type(exc).__name__
                )

    def snapshot(self, scope: str, day: str = "") -> Dict[str, int]:
        """
        Read one scope's counters for one day.

        The local tally is the fallback when no client is bound, and the merge
        point when one is: Redis is the durable half, the local dict the
        in-process half a request wrote before the round trip confirmed, and
        neither should hide the other.
        """
        if not scope:
            return {}
        day = day or _day()
        key = self._key(scope, day)
        local: Dict[str, int] = {}
        with self._lock:
            local = dict(self._local.get(key, {}))
        client = self._handle.get() if self._handle.configured else None
        if client is None:
            return {name: int(local.get(name, 0)) for name in FIELDS}
        try:
            raw = client.hgetall(key) or {}
        except Exception:
            raw = {}
        merged: Dict[str, int] = {}
        for name in FIELDS:
            stored = raw.get(name)
            if stored is not None:
                try:
                    stored = int(stored)
                except (TypeError, ValueError):
                    stored = 0
            # Redis holds the durable count; the local dict holds what this
            # process wrote.  They are the same writes seen from two sides, so
            # the larger of the two is the read, not the sum: summing would
            # count the in-process half twice.
            merged[name] = max(int(stored or 0), int(local.get(name, 0)))
        return merged

    def derived(self, scope: str, day: str = "") -> Dict[str, Any]:
        """
        The KPIs a deployment reads, from one scope's day.

        Ratios are ``None`` where the denominator is zero: a scope that
        injected nothing today did not measure a "0% adoption", it measured
        nothing.  Reporting ``None`` as ``0.0`` would let a quiet day read as a
        bad one.
        """
        snap = self.snapshot(scope, day)

        def ratio(numerator: str, denominator: int):
            if not denominator:
                return None
            return round(snap.get(numerator, 0) / denominator, 4)

        turns = snap.get("turns", 0)
        settled = snap.get("feedback_up", 0) + snap.get("feedback_down", 0)
        return {
            # The scope travels with its own row: a reader that asked for three
            # repositories cannot tell an adoption rate from a loop share apart
            # without knowing which row belongs to which.
            "scope": scope,
            "day": day or _day(),
            "turns": turns,
            "injected_turns": snap.get("injected_turns", 0),
            "injected_docs": snap.get("injected_docs", 0),
            "loops": snap.get("loops", 0),
            "escalations": snap.get("escalations", 0),
            "read_failures": snap.get("read_failures", 0),
            "memo_hits": snap.get("memo_hits", 0),
            # How much the miner wrote, and how much of it had to say it is
            # partial: the number that answers "are the quote budgets cutting
            # real content" without guessing from the documents themselves.
            "mined_docs": snap.get("mined_docs", 0),
            "mined_truncated": snap.get("mined_truncated", 0),
            # Of the turns that were shown a block, how many went on to mine a
            # verified fix: the layer's adoption of its own guidance.
            "adoption_rate": ratio(
                "green_after_injection", snap.get("injected_turns", 0)
            ),
            # Of the injected documents the next turn could judge, how many
            # held: the layer's own help rate, measured on documents rather
            # than on turns.
            "knowledge_help_rate": ratio("feedback_up", settled),
            # How often a turn met a failure at all, and how often it met the
            # same one enough times to be a loop rather than a stumble.  The
            # two are kept apart because the layer's own definition of a loop
            # is the threshold, and a turn that failed once is not stuck.
            "failure_turn_share": ratio("failure_turns", turns),
            "loop_turn_share": ratio("loops", turns),
            # How often a loop reached a model switch: the escalation's reach.
            "escalation_rate": ratio("escalations", turns),
            # Share of published documents whose quoted diff did not fit the
            # quote budget.  None, not 0%, on a day that mined nothing.
            "truncated_share": ratio("mined_truncated", snap.get("mined_docs", 0)),
        }
