"""
Publishing mined knowledge to the indexing worker.

The request path and the indexer live in different processes, and the request
path must not wait for a document to be embedded.  A mined episode is therefore
handed across as one message on a Redis stream: the proxy appends, the worker
claims, masks and indexes.  The stream is the whole contract between them, so
one message carries everything the worker needs and nothing it cannot use.

Two properties keep the request path fast and the store honest.

* **Deduplication.**  Codex replays the whole turn on every request, so the
  same verified fix is mined again and again.  Before appending, the publisher
  takes a short lock on the episode's fingerprint; only the first request of
  an episode gets through, the rest are reported as duplicates and skipped.
* **Fail-open.**  Publishing runs on the way to answering a live agent.  A
  Redis that is missing or slow is a status on the request annotation, never
  an exception, and ``apply()`` never blocks on it: the connection is opened
  with a short timeout and a failed publish is abandoned, not retried here.
  The worker retries on its own side.
"""

import json
import os
import threading

from dataclasses import dataclass
from typing import Any, Sequence

from llm_router_plugins.utils.knowledge.agentic_knowledge.adapters.base import (
    KnowledgeEvent,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.state import (
    RedisConnectionSettings,
)

__all__ = [
    "KnowledgeEventPublisher",
    "PublishStatus",
    "STATUS_OK",
    "STATUS_UNAVAILABLE",
]

#: Every event was either appended or already known.
STATUS_OK = "ok"

#: Redis could not be reached; nothing was appended.
STATUS_UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class PublishStatus:
    """
    The outcome of one :meth:`KnowledgeEventPublisher.publish` call.

    Parameters
    ----------
    status : str
        :data:`STATUS_OK` or :data:`STATUS_UNAVAILABLE`.
    published : int
        Events actually appended to the stream.
    deduplicated : int
        Events skipped because their fingerprint was already published.
    reason : str
        Why the publish did not reach the stream; a class name, never a payload.
    """

    status: str = STATUS_OK
    published: int = 0
    deduplicated: int = 0
    reason: str = ""


class KnowledgeEventPublisher:
    """
    Appends mined episodes to the worker's stream, deduplicating by fingerprint.

    The client is created on first use rather than at construction, so a proxy
    worker forked from a preloaded application opens its own connection instead
    of inheriting its parent's sockets.  A pre-built client (the tests) is used
    as-is and is never rebuilt across forks.

    Parameters
    ----------
    connection : RedisConnectionSettings
        Where the stream lives.  An unconfigured connection makes the publisher
        report itself unavailable instead of raising.
    stream_key : str
        Name of the stream the worker reads.
    key_prefix : str
        Namespace of the deduplication keys.
    dedup_ttl_seconds : int
        How long a fingerprint stays "already published".
    max_stream_len : int
        Approximate ceiling of the stream, trimmed by the stream itself.
    client : Any, optional
        Redis client to use as-is; ``None`` builds one from *connection*.
    logger : Any, optional
        Where failures are reported, without event content.
    """

    def __init__(
        self,
        connection: RedisConnectionSettings,
        stream_key: str,
        key_prefix: str,
        dedup_ttl_seconds: int = 604800,
        max_stream_len: int = 100000,
        client: Any = None,
        logger: Any = None,
    ) -> None:
        """Bind the destination; build no connection until the first publish."""
        self._connection = connection
        self._stream_key = stream_key
        self._key_prefix = key_prefix
        self._dedup_ttl = int(dedup_ttl_seconds)
        self._max_stream_len = int(max_stream_len)
        self._client = client
        self._logger = logger
        self._lock = threading.Lock()
        self._pid = os.getpid()

    @property
    def available(self) -> bool:
        """Whether a publish can be attempted at all."""
        return bool(self._connection.configured)

    def publish(self, events: Sequence[KnowledgeEvent]) -> PublishStatus:
        """
        Append *events* to the stream, once each.

        Parameters
        ----------
        events : Sequence[KnowledgeEvent]
            Episodes mined from the request's turn.

        Returns
        -------
        PublishStatus
        How the publish went; :data:`STATUS_UNAVAILABLE` when Redis was not
        reachable, never an exception.
        """
        if not events:
            return PublishStatus(STATUS_OK)
        if not self.available:
            return PublishStatus(STATUS_UNAVAILABLE, reason="unconfigured")
        client = self._connection_client()
        if client is None:
            return PublishStatus(STATUS_UNAVAILABLE, reason="connection")
        published = 0
        deduplicated = 0
        for event in events:
            if not event.fingerprint:
                # An episode without an identity could not be deduplicated, so
                # it is not appended: a stream of duplicates is worse than a
                # stream that misses one.
                deduplicated += 1
                continue
            if not self._claim(client, event):
                deduplicated += 1
                continue
            if self._append(client, event):
                published += 1
            else:
                # The claim succeeded but the append failed; forget the claim so
                # the next replay of this turn can still publish it.
                self._release(client, event)
        if published + deduplicated < len(events):
            # A claim that lost its append is counted in neither bucket; report
            # the shortfall rather than pretending everything was handled.
            self._warn(
                "knowledge publish partial: %d published, %d deduplicated, "
                "%d of %d events",
                published,
                deduplicated,
                published + deduplicated,
                len(events),
            )
        return PublishStatus(STATUS_OK, published, deduplicated)

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
        for name in ("close", "shutdown", "quit"):
            hook = getattr(client, name, None)
            if callable(hook):
                try:
                    hook()
                except Exception:  # closing must not raise
                    return
                return

    # ------------------------------------------------------------- helpers -
    def _connection_client(self) -> Any:
        """Return the client, building it once per process."""
        pid = os.getpid()
        if self._pid != pid:
            self._lock = threading.Lock()
            if self._client is None:
                self._pid = pid
        if self._client is not None:
            return self._client
        with self._lock:
            if self._client is None:
                try:
                    import redis  # noqa: PLC0415 - optional dependency

                    self._client = redis.Redis(**self._connection.client_kwargs())
                except Exception as exc:
                    self._warn(
                        "knowledge publisher client failed: %s",
                        type(exc).__name__,
                    )
                    return None
            return self._client

    def _seen_key(self, event: KnowledgeEvent) -> str:
        """The deduplication key of one episode."""
        return f"{self._key_prefix}:seen:{event.fingerprint}"

    def _claim(self, client: Any, event: KnowledgeEvent) -> bool:
        """
        Take the one-time lock on an episode.

        Returns
        -------
        bool
            ``True`` when this publish owns the episode, ``False`` when it was
            already published within the deduplication window.
        """
        try:
            taken = client.set(
                self._seen_key(event),
                "1",
                nx=True,
                ex=self._dedup_ttl,
            )
        except Exception as exc:
            self._warn("knowledge dedup failed: %s", type(exc).__name__)
            return False
        return bool(taken)

    def _release(self, client: Any, event: KnowledgeEvent) -> None:
        """Give back a claim whose append did not land."""
        try:
            client.delete(self._seen_key(event))
        except Exception:  # releasing is best-effort
            return

    def _append(self, client: Any, event: KnowledgeEvent) -> bool:
        """Append one event to the stream."""
        try:
            client.xadd(
                self._stream_key,
                {"event": json.dumps(event.to_dict(), ensure_ascii=False)},
                maxlen=self._max_stream_len,
                approximate=True,
            )
        except Exception as exc:
            self._warn("knowledge xadd failed: %s", type(exc).__name__)
            return False
        return True

    def _warn(self, message: str, *args: Any) -> None:
        """Report one failure when a logger was given."""
        if self._logger is not None:
            self._logger.warning(message, *args)
