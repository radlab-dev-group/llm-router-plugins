"""
Session affinity for agentic routing.

An agent loop issues many requests for one long-lived session.  Re-resolving
the mode from scratch on every turn is wasteful and, more importantly, makes
the model *flip* between turns, which destroys KV/prefix cache locality on the
backend that previously served the session.

This module keeps a small thread-safe TTL+LRU cache mapping
``session_id -> (mode, model)``.  Entries expire after ``ttl_seconds`` and are
refreshed on every hit (sliding expiration), so an active session keeps the
same model while an idle one becomes eligible for re-routing.

The knobs of the layer live in
:class:`llm_router_plugins.utils.routing.agentic_routing.config.AgenticRoutingConfig`
as :class:`SessionAffinitySettings`; this module implements only the cache.
"""

import logging
import threading
import time

from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

__all__ = ["SessionAffinitySettings", "CachedDecision", "SessionAffinityCache"]

DEFAULT_TTL_SECONDS = 900
DEFAULT_MAX_ENTRIES = 1024


@dataclass(frozen=True)
class SessionAffinitySettings:
    """
    Configuration of the session affinity layer.

    Parameters
    ----------
    enabled : bool
        Whether session affinity participates in the routing cascade.
    ttl_seconds : int
        Sliding lifetime of a cached decision, in seconds.
    max_entries : int
        Maximum number of sessions kept before the least recently used one is
        evicted.
    """

    enabled: bool = True
    ttl_seconds: int = DEFAULT_TTL_SECONDS
    max_entries: int = DEFAULT_MAX_ENTRIES


@dataclass(frozen=True)
class CachedDecision:
    """
    A routing decision remembered for one session.

    Parameters
    ----------
    mode_name : str
        Name of the agent work mode chosen for the session.
    model_name : str
        Model name that was selected together with *mode_name*.
    expires_at : float
        Monotonic deadline after which the entry is considered stale.
    """

    mode_name: str
    model_name: str
    expires_at: float


class SessionAffinityCache:
    """
    Thread-safe TTL + LRU cache of per-session routing decisions.

    Attributes
    ----------
    ttl_seconds : int
        Sliding lifetime of an entry, in seconds.
    max_entries : int
        Maximum number of sessions retained.
    """

    def __init__(
        self,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Initialize an empty cache.

        Parameters
        ----------
        ttl_seconds : int
            Sliding lifetime of an entry, in seconds.
        max_entries : int
            Maximum number of sessions retained.
        logger : logging.Logger, optional
            Logger used to report evictions.

        Returns
        -------
        None

        Raises
        ------
        None
        """
        self.ttl_seconds = int(ttl_seconds)
        self.max_entries = int(max_entries)
        self._logger = logger
        self._entries: "OrderedDict[str, CachedDecision]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, session_id: str) -> Optional[CachedDecision]:
        """
        Return the decision cached for *session_id*, refreshing its expiry.

        Expired entries are dropped and reported as a miss.

        Parameters
        ----------
        session_id : str
            Identifier of the session to look up.

        Returns
        -------
        CachedDecision or None
            The refreshed entry, or ``None`` when absent or expired.

        Raises
        ------
        None
        """
        if not session_id:
            return None

        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(session_id)
            if entry is None:
                return None
            if entry.expires_at <= now:
                del self._entries[session_id]
                return None

            refreshed = CachedDecision(
                mode_name=entry.mode_name,
                model_name=entry.model_name,
                expires_at=now + self.ttl_seconds,
            )
            self._entries[session_id] = refreshed
            self._entries.move_to_end(session_id)
            return refreshed

    def set(self, session_id: str, mode_name: str, model_name: str) -> None:
        """
        Remember the routing decision for *session_id*.

        Parameters
        ----------
        session_id : str
            Identifier of the session; an empty identifier is ignored.
        mode_name : str
            Name of the chosen agent work mode.
        model_name : str
            Name of the chosen model.

        Returns
        -------
        None

        Raises
        ------
        None
        """
        if not session_id:
            return

        now = time.monotonic()
        entry = CachedDecision(
            mode_name=mode_name,
            model_name=model_name,
            expires_at=now + self.ttl_seconds,
        )
        with self._lock:
            self._entries[session_id] = entry
            self._entries.move_to_end(session_id)
            while len(self._entries) > self.max_entries:
                evicted, _ = self._entries.popitem(last=False)
                if self._logger:
                    self._logger.debug(
                        "AgenticRouting: evicting session affinity entry '%s'",
                        evicted,
                    )

    def invalidate(self, session_id: str) -> None:
        """
        Drop the decision cached for *session_id*.

        Parameters
        ----------
        session_id : str
            Identifier of the session to forget.

        Returns
        -------
        None

        Raises
        ------
        None
        """
        if not session_id:
            return
        with self._lock:
            self._entries.pop(session_id, None)

    def reset(self) -> None:
        """
        Forget every cached session.

        Returns
        -------
        None

        Raises
        ------
        None
        """
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        """
        Return the number of retained entries, including expired ones.

        Returns
        -------
        int
            Number of entries currently stored.

        Raises
        ------
        None
        """
        with self._lock:
            return len(self._entries)
