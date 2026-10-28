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

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from llm_router_plugins.utils.knowledge.agentic_knowledge.adapters.base import (
    KnowledgeEvent,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.redisclient import (
    RedisClientHandle,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.state import (
    RedisConnectionSettings,
)

__all__ = [
    "KnowledgeEventPublisher",
    "PublishStatus",
    "STATUS_OK",
    "STATUS_PARTIAL",
    "STATUS_UNAVAILABLE",
]

#: Every event was either appended or already known.
STATUS_OK = "ok"

#: Some events reached the stream and at least one could not be written.
STATUS_PARTIAL = "partial"

#: Redis could not be reached; nothing further was attempted.
STATUS_UNAVAILABLE = "unavailable"

# What one deduplication claim returned.  A duplicate and a failure look the
# same to a boolean and mean opposite things to an operator.
_CLAIMED = "claimed"
_DUPLICATE = "duplicate"
_ERROR = "error"


@dataclass(frozen=True)
class PublishStatus:
    """
    The outcome of one :meth:`KnowledgeEventPublisher.publish` call.

    Parameters
    ----------
    status : str
        :data:`STATUS_OK`, :data:`STATUS_PARTIAL` or :data:`STATUS_UNAVAILABLE`.
    published : int
        Events actually appended to the stream.
    deduplicated : int
        Events skipped because their fingerprint was already published.  A
        Redis that refused the claim is never counted here — that is an error.
    errors : int
        Events Redis would not take: a refused claim or append.  Counted
        separately so ``deduplicated`` keeps its meaning of "already known" and
        an outage cannot read as a healthy, fully-deduplicated run.
    reason : str
        Why the publish did not reach the stream; a class name, never a payload.
    """

    status: str = STATUS_OK
    published: int = 0
    deduplicated: int = 0
    errors: int = 0
    reason: str = ""


class KnowledgeEventPublisher:
    """
    Appends mined episodes to the worker's stream, deduplicating by fingerprint.

    The client is held by a :class:`RedisClientHandle`: built on first use, and
    rebuilt when a different process asks for it, so a proxy worker forked from
    a preloaded application opens its own connection instead of inheriting its
    parent's sockets.

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
        self._handle = RedisClientHandle(connection, client, logger)
        self._logger = logger

    @property
    def available(self) -> bool:
        """Whether a publish can be attempted at all."""
        return self._handle.configured

    def publish(
        self,
        events: Sequence[KnowledgeEvent],
        *,
        refresh_claims: bool = False,
    ) -> PublishStatus:
        """
        Append *events* to the stream, once each.

        Publishing runs on the way to answering a live agent, so it stops at
        the first transport error rather than walking the rest of the batch:
        a Redis that refused one ``SET`` will refuse them all, each attempt
        costing a socket timeout, and one compaction window can carry three
        hundred events.

        Parameters
        ----------
        events : Sequence[KnowledgeEvent]
            Episodes mined from the request's turn.
        refresh_claims : bool, optional
            Re-publish an event whose claim already exists, refreshing the
            claim instead of treating the event as a duplicate.  Documentation
            needs this: a ``git checkout`` can bring a file back to a version
            whose fingerprint was claimed days ago while its document has been
            retired in the meantime, and an event refused as duplicate then
            leaves the repository with no document for a file it has.  Mined
            episodes keep the strict once-each rule, because a replayed turn
            asks for the same episode on every request.

        Returns
        -------
        PublishStatus
            How the publish went; :data:`STATUS_UNAVAILABLE` when Redis was not
            reachable, :data:`STATUS_PARTIAL` when it took some events and
            refused others — never an exception.
        """
        if not events:
            return PublishStatus(STATUS_OK)
        if not self.available:
            return PublishStatus(STATUS_UNAVAILABLE, reason="unconfigured")
        client = self._handle.get()
        if client is None:
            return PublishStatus(STATUS_UNAVAILABLE, reason="connection")
        published = 0
        deduplicated = 0
        errors = 0
        reason = ""
        for event in events:
            if not event.fingerprint:
                # An episode without an identity could not be deduplicated, so
                # it is not appended: a stream of duplicates is worse than a
                # stream that misses one.
                deduplicated += 1
                continue
            outcome = self._claim(client, event, refresh=refresh_claims)
            if outcome == _ERROR:
                errors += 1
                reason = "claim"
                break
            if outcome == _DUPLICATE:
                deduplicated += 1
                continue
            if self._append(client, event):
                published += 1
                continue
            errors += 1
            reason = "xadd"
            # The claim succeeded but the append failed; forget the claim so the
            # next replay of this turn can still publish it.
            self._release(client, event)
            break
        return self._status(published, deduplicated, errors, reason)

    def replace_docs(
        self, scope: str, events: Sequence[KnowledgeEvent]
    ) -> PublishStatus:
        """Replace one scope's queued documentation, including dedup claims.

        This operator-only operation is not atomic with running consumers or
        publishers: pause them first. A failure may leave a partial replacement;
        retry the forced scan, which clears it before publishing again.
        """
        if not scope or any(
            event.scope != scope or event.kind != "doc" for event in events
        ):
            return PublishStatus(STATUS_UNAVAILABLE, reason="scope")
        client = self._handle.get()
        if client is None:
            return PublishStatus(STATUS_UNAVAILABLE, reason="connection")
        try:
            # Bound the scan to the existing tail; never chase concurrent writes.
            tail = client.xrevrange(self._stream_key, count=1)
            if tail:
                end = tail[0][0]
                start = "-"
                while True:
                    batch = client.xrange(
                        self._stream_key, min=start, max=end, count=200
                    )
                    if not batch:
                        break
                    for entry_id, fields in batch:
                        raw = fields.get("event", fields.get(b"event"))
                        try:
                            event = json.loads(raw)
                        except (TypeError, ValueError, UnicodeError):
                            continue
                        if (
                            not isinstance(event, dict)
                            or event.get("scope") != scope
                        ):
                            continue
                        if event.get("kind") not in ("doc", "docs_prune"):
                            continue
                        client.xdel(self._stream_key, entry_id)
                        fingerprint = event.get("fingerprint")
                        if isinstance(fingerprint, str) and fingerprint:
                            client.delete(f"{self._key_prefix}:seen:{fingerprint}")
                    last = batch[-1][0]
                    if last == end:
                        break
                    # Exclusive XRANGE bounds require Redis 6.2. Advance the
                    # inclusive ID instead, retaining Redis 5 compatibility.
                    milliseconds, sequence = map(
                        int,
                        (
                            last.decode("ascii")
                            if isinstance(last, bytes)
                            else str(last)
                        ).split("-"),
                    )
                    if sequence == 2**64 - 1:
                        start = f"{milliseconds + 1}-0"
                    else:
                        start = f"{milliseconds}-{sequence + 1}"
            # Claims can outlive trimmed stream entries, or the entire stream.
            for event in events:
                client.delete(self._seen_key(event))
        except Exception as exc:
            self._warn(
                "knowledge docs stream replacement failed: %s: %s",
                type(exc).__name__,
                exc,
            )
            return PublishStatus(STATUS_UNAVAILABLE, errors=1, reason="replace")
        return self.publish(events)

    def _status(
        self, published: int, deduplicated: int, errors: int, reason: str
    ) -> PublishStatus:
        """Fold the counters into the status the request annotation reports."""
        if errors:
            self._warn(
                "knowledge publish incomplete: %d published, %d deduplicated, "
                "%d failed (%s)",
                published,
                deduplicated,
                errors,
                reason,
            )
            if published:
                return PublishStatus(
                    STATUS_PARTIAL, published, deduplicated, errors, reason
                )
            return PublishStatus(
                STATUS_UNAVAILABLE, published, deduplicated, errors, reason
            )
        return PublishStatus(STATUS_OK, published, deduplicated)

    def mined_count(self, scope: str) -> Optional[int]:
        """
        Return how many mined documents *scope* has, or ``None`` when unknown.

        The reader asks this before it pays for a search of a bucket a cold
        repository cannot possibly answer.  ``None`` means the question could
        not be asked — no Redis, a timeout, a counter this plugin has never
        written — and the caller searches anyway: a counter that cannot be
        trusted is a reason to do the old thing, never a reason to read less.

        Parameters
        ----------
        scope : str
            Repository whose counter to read.

        Returns
        -------
        int or None
            The count, or ``None`` when it is unknown.
        """
        if not scope or not self.available:
            return None
        client = self._handle.get()
        if client is None:
            return None
        try:
            raw = client.get(f"{self._key_prefix}:mined:{scope}")
        except Exception as exc:
            self._warn("knowledge mined counter failed: %s", type(exc).__name__)
            return None
        if raw is None:
            # A scope this worker has never written a mined document for, and a
            # Redis that has forgotten the counter, are the same answer: there
            # is nothing in the bucket yet.  A deployment predating the counter
            # reads as cold until its first new document, which costs one
            # skipped search and nothing that was ever retrievable.
            return 0
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    def close(self) -> None:
        """
        Release the connection, if one was built and knows how.

        Returns
        -------
        None
        """
        self._handle.close()

    # ------------------------------------------------------------- helpers -
    def _seen_key(self, event: KnowledgeEvent) -> str:
        """The deduplication key of one episode."""
        return f"{self._key_prefix}:seen:{event.fingerprint}"

    def _claim(self, client: Any, event: KnowledgeEvent, *, refresh: bool = False) -> str:
        """
        Take the one-time lock on an episode.

        A claim already taken reads as a duplicate unless *refresh* is set, in
        which case the claim is renewed and the event goes to the stream again;
        a caller that can retire a document behind a claim (documentation)
        cannot let the claim be the last word about what the engine holds.

        Returns
        -------
        str
            :data:`_CLAIMED` when this publish owns the episode,
            :data:`_DUPLICATE` when it was already published within the
            deduplication window, :data:`_ERROR` when Redis did not answer —
            the three are not the same thing and must not be counted alike.
        """
        key = self._seen_key(event)
        try:
            if refresh:
                client.set(key, "1", ex=self._dedup_ttl)
                return _CLAIMED
            taken = client.set(key, "1", nx=True, ex=self._dedup_ttl)
        except Exception as exc:
            self._warn("knowledge dedup failed: %s", type(exc).__name__)
            return _ERROR
        return _CLAIMED if taken else _DUPLICATE

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
