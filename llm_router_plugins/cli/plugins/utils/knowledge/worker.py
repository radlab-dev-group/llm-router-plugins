"""
Indexing worker: the writer's half of the knowledge layer.

The proxy cannot embed a document and answer the agent at the same time, so it
only appends an episode to a Redis stream and moves on.  This process claims
those episodes, masks them, and writes them into the Semantic Search Engine —
in a run whose failures cost a knowledge document, never a request.

The worker owns the whole writing lifecycle:

* it reads the stream with a consumer group, so several workers share the load
  and a crashed one's messages are redelivered, not lost;
* it masks every document (``FastMasker``) and throws the mappings away — the
  knowledge base is read by models, and re-identification is not on its path;
* it creates the target collection on first use, with the embedder, reranker
  and chunking the operator configured for *this* process (the proxy needs none
  of them);
* it retires old documents: archived windows expire on their own clock.

Masking has one rule the rest of the layer depends on: a value the reader
filters on is never masked.  The engine's metadata ``eq`` is exact string
equality and the reader filters with the raw thread id, so a thread id the
masker turned into a token matches nothing — with ``status: ok`` and no hits,
which is the worst way to fail.  See :meth:`KnowledgeIndexWorker._mask_metadata`
and :meth:`KnowledgeIndexWorker._prepare`.

Chunking is worth stating before it surprises anyone.  With
``…_WORKER_CHUNK_TOKENS`` left at its default of ``0`` the worker declares no
chunking and the engine applies its own default (500 tokens per chunk); it also
truncates the *embedded* text at 508 tokens
(``DBSemanticSearchController.max_tokens_in_text``), so a chunk can be stored in
full and embedded only in part.  One knowledge document is therefore often
several chunks — and because the metadata header is the first line of the
document, it rides on the first chunk only; for the rest the reader falls back
to the fields in the document name (see
:mod:`llm_router_plugins.utils.knowledge.agentic_knowledge.documents`).

Every external call is optional-by-construction: the libraries import lazily,
so ``--help`` and the unit tests need neither Redis nor ``sse_lib`` installed.
"""

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time

from datetime import date, timedelta
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

from llm_router_plugins.utils.knowledge.agentic_knowledge.adapters.base import (
    KnowledgeEvent,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
    AgenticKnowledgeConfig,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.documents import (
    KIND_SESSION_ARCHIVE,
    build_document_text,
    document_body,
    document_name,
    is_knowledge_document,
    parse_document_name,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.state import (
    RedisConnectionSettings,
)

__all__ = ["KnowledgeIndexWorker", "main"]

#: Wire version this worker understands.
EVENT_VERSION = 1

#: Slack added to the socket timeout of a worker that blocks on a read.  The
#: connection settings are shared with the request path, whose socket must give
#: up after a second; a worker asks the server to hold a read open for
#: ``worker.block_ms`` (5 s by default), and the socket closed first — every
#: idle wait then surfaced as a ``TimeoutError`` and the loop never blocked.
BLOCK_SOCKET_MARGIN_SECONDS = 2.0

#: Attempts before a message is dropped; a poison message must not stall the
#: group, so after this many failures it is acknowledged with a log.
DEFAULT_MAX_ATTEMPTS = 3

#: How long an entry may sit unacknowledged before this worker claims it back
#: for another attempt, in milliseconds.  Without a reclaim pass a message that
#: fails once is never read again — ``XREADGROUP ``>`` hands out new messages
#: only — so it would sit in the pending list forever.
RECLAIM_IDLE_MS = 60_000

#: Seconds between reclaim passes; a failed entry is retried on this clock.
RECLAIM_INTERVAL_SECONDS = 300

#: Lifetime of an entry's attempt counter, seconds.  Counters outlive their
#: entry by design; Redis trims them, the worker does not have to.
_TRIES_TTL_SECONDS = 86_400

#: Description of a collection this worker creates; the engine rejects an
#: empty one, and an operator who wants a different description pre-creates it.
_COLLECTION_DESCRIPTION = "Agentic knowledge base (auto-created by llm-router)."

#: Metadata fields a reader filters on, written verbatim and never masked.
#: They are the join between the two halves of the layer: the search asks
#: ``{"operator": "eq", "field": {"thread_id": …}}`` with the value the request
#: carried, and the engine matches it with ``==``.  Anything the masker rewrites
#: here is a document that can be written but never read back.
IDENTITY_KEYS: Tuple[str, ...] = (
    "thread_id",
    "window_id",
    "scope",
    "kind",
    "mode",
    "phase",
    "signature_hash",
)


class KnowledgeIndexWorker:
    """
    Consumes knowledge events and indexes them.

    Parameters
    ----------
    config : AgenticKnowledgeConfig
        Resolved policy: stream, group, collection prefix, worker settings.
    logger : logging.Logger, optional
        Where progress and failures are reported, without document content.
    redis_client : Any, optional
        A Redis client; built from the connection settings when omitted.
    sse_client : Any, optional
        An engine client; built from the SSE settings when omitted.
    max_attempts : int
        Delivery attempts after which an unprocessable message is dropped.
    """

    def __init__(
        self,
        config: AgenticKnowledgeConfig,
        logger: Optional[logging.Logger] = None,
        redis_client: Any = None,
        sse_client: Any = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        """Bind the policy; build neither connection until it is needed."""
        self._config = config
        self._logger = logger or logging.getLogger(__name__)
        self._redis = redis_client
        self._sse = sse_client
        self._max_attempts = max(1, int(max_attempts))
        self._stream = config.stream_key
        self._group = config.worker.group
        self._consumer = config.worker.consumer or _default_consumer()
        self._stop = False
        self._last_retention = 0.0
        self._last_reclaim = 0.0
        self._group_ready = False
        self._masker = None
        self._scopes: set = set()

    # ----------------------------------------------------------- lifecycle -
    def run(self) -> None:
        """
        Read, index and retire until stopped.

        Returns
        -------
        None
        """
        self._install_signal_handlers()
        self._ensure_group()
        self._log(
            "info",
            "knowledge worker started (group=%s consumer=%s stream=%s)",
            self._group,
            self._consumer,
            self._stream,
        )
        while not self._stop:
            try:
                processed = self.run_once()
                if processed == 0:
                    # Nothing new arrived: this is when entries that failed
                    # earlier are claimed back for another attempt.
                    self._reclaim_due()
                    self._maybe_retain()
            except Exception as exc:  # a read failure must not end the loop
                self._log(
                    "warning", "knowledge worker read failed: %s", type(exc).__name__
                )
                time.sleep(1.0)
                continue
        self._log("info", "knowledge worker stopped")

    def run_once(self, block_ms: Optional[int] = None) -> int:
        """
        Claim and process one batch.

        The consumer group is created on the first call, so a process asked to
        run a single batch works whether or not a daemon ever ran.

        Parameters
        ----------
        block_ms : int, optional
            How long to wait for messages; the worker setting when omitted,
            ``0`` for no waiting at all.

        Returns
        -------
        int
            Messages processed (indexed or dropped) in this batch.
        """
        self._ensure_group()
        messages = self._read(block_ms)
        processed = 0
        for entry_id, fields in messages:
            if self._stop:
                break
            self._process_entry(entry_id, fields)
            processed += 1
        return processed

    def stop(self) -> None:
        """Ask the loop to stop after the current batch."""
        self._stop = True

    # ---------------------------------------------------------- processing -
    def _process_entry(self, entry_id: Any, fields: Dict[str, Any]) -> bool:
        """Index one stream message, acknowledging it when done or dropped."""
        event = self._decode(fields)
        if event is None or not self._supported(event):
            self._ack(entry_id)
            return False
        try:
            self.index_event(event)
        except Exception as exc:
            if self._drop_if_exhausted(entry_id, event, exc):
                self._forget(entry_id)
            return False
        self._forget(entry_id)
        self._ack(entry_id)
        return True

    def _drop_if_exhausted(
        self, entry_id: Any, event: KnowledgeEvent, exc: Exception
    ) -> bool:
        """
        Count a failed attempt and drop the entry once the attempts run out.

        Returns
        -------
        bool
            ``True`` when the entry was acknowledged and abandoned; ``False``
            when it stays pending for a later reclaim.
        """
        tries = self._note_failure(entry_id)
        if tries < self._max_attempts:
            self._log(
                "warning",
                "knowledge index failed, retrying (fingerprint=%s, attempt %d/%d,"
                " %s)",
                event.fingerprint,
                tries,
                self._max_attempts,
                type(exc).__name__,
            )
            return False
        self._log(
            "error",
            "knowledge index failed, dropping (fingerprint=%s, %s)",
            event.fingerprint,
            type(exc).__name__,
        )
        self._ack(entry_id)
        return True

    # -------------------------------------------------------- attempt counting
    def _tries_key(self, entry_id: Any) -> str:
        """The attempt-counter key of one stream entry."""
        return f"{self._config.key_prefix}:tries:{self._group}:{entry_id}"

    def _note_failure(self, entry_id: Any) -> int:
        """
        Record one failed attempt of an entry and return how many it has.

        Redis reports no delivery count to a consumer reading ``>``, so the
        worker keeps its own: without it a failing entry would never be dropped
        and never be retried, sitting in the pending list for good.  An entry
        whose counter cannot be written is counted as exhausted — dropping one
        document with a log beats an endless cycle of the same failure.
        """
        key = self._tries_key(entry_id)
        try:
            client = self._redis_client()
            tries = int(client.incr(key))
            if tries == 1:
                client.expire(key, _TRIES_TTL_SECONDS)
        except Exception as exc:
            self._log(
                "warning", "knowledge attempt counter failed: %s", type(exc).__name__
            )
            return self._max_attempts
        return max(1, tries)

    def _forget(self, entry_id: Any) -> None:
        """Drop an entry's attempt counter once it is settled."""
        try:
            self._redis_client().delete(self._tries_key(entry_id))
        except Exception:  # a leftover counter expires on its own
            return

    # --------------------------------------------------------------- reclaim -
    def _reclaim_due(self) -> None:
        """Claim back entries left unacknowledged, on the reclaim interval."""
        if RECLAIM_INTERVAL_SECONDS <= 0:
            return
        now = time.time()
        if now - self._last_reclaim < RECLAIM_INTERVAL_SECONDS:
            return
        self._last_reclaim = now
        self._reclaim()
        self._warn_backlog()

    def _reclaim(self) -> int:
        """
        Take over entries idle longer than the reclaim threshold.

        A consumer reading ``>`` only ever sees messages nobody has delivered,
        so an entry whose processing failed is invisible until someone claims it
        back.  This is that someone.

        Returns
        -------
        int
            Entries taken over and processed again.
        """
        client = self._redis_client()
        try:
            claimed = client.xautoclaim(
                self._stream,
                self._group,
                self._consumer,
                RECLAIM_IDLE_MS,
                count=int(self._config.worker.batch),
            )
        except Exception as exc:  # a backend without XAUTOCLAIM simply retries never
            self._log(
                "debug", "knowledge reclaim unavailable: %s", type(exc).__name__
            )
            return 0
        entries = _claimed_entries(claimed)
        for entry_id, fields in entries:
            if self._stop:
                break
            self._process_entry(entry_id, fields)
        return len(entries)

    def _warn_backlog(self) -> None:
        """Say so when the pending list has grown past the configured ceiling."""
        ceiling = int(self._config.worker.max_pending)
        if ceiling <= 0:
            return
        try:
            summary = self._redis_client().xpending(self._stream, self._group)
        except Exception:  # a summary nobody can read is not worth a warning
            return
        pending = _pending_count(summary)
        if pending is None:
            return
        if pending > ceiling:
            self._log(
                "warning",
                "knowledge stream backlog: %d entries pending (threshold %d)",
                pending,
                ceiling,
            )

    def index_event(self, event: KnowledgeEvent) -> None:
        """
        Mask and index one event.

        Parameters
        ----------
        event : KnowledgeEvent
            The episode to write.

        Returns
        -------
        None
        """
        self._scopes.add(event.scope)
        text = self._prepare(event)
        document = self._document(text, event)
        collection = self._config.collection(event.scope, event.kind)
        self._ensure_collection(collection)
        self._sse_client().index_documents(collection, [document])

    def _prepare(self, event: KnowledgeEvent) -> str:
        """
        Return the text to index: masked knowledge under a rebuilt header.

        The header is machine-readable — the reader parses the kind, the day
        and the signature digest out of it — so it is not passed through the
        masker, which rewrites an ISO date into a ``{DATE_…}`` token and an
        email-shaped run into ``{EMAIL_…}``, leaving the reader nothing to
        parse.  The knowledge body is masked, and the header is built again
        from the event's own fields, with only its free-text values masked.

        Parameters
        ----------
        event : KnowledgeEvent
            The episode being written.

        Returns
        -------
        str
            Header line plus masked body.
        """
        metadata = event.metadata or {}
        body = self._mask(document_body(event.text))
        return build_document_text(
            event.kind,
            body,
            date=_day_of(event.created_at),
            signature=self._mask(event.signature),
            signature_hash=event.signature_hash,
            paths=tuple(metadata.get("paths") or ()),
            thread_id=event.thread_id,
            window_id=str(metadata.get("window_id") or ""),
            position=metadata.get("position"),
            agent=event.agent,
            mode=event.mode,
            test_command=self._mask(str(metadata.get("test_command") or "")),
        )

    # ---------------------------------------------------------- collections -
    def _document(self, text: str, event: KnowledgeEvent) -> Dict[str, Any]:
        """
        Build one ``texts[]`` element, mirroring ``sse_lib.text_document``.

        The identity fields are taken from the event, never from the masked
        mapping, so a filter key can only ever hold what the request that
        produced it carried.  The engine appends ``.input_text`` to this name
        when it stores the document; :mod:`…documents` reads both spellings.
        """
        published = _day_of(event.created_at)
        name = document_name(event.kind, published, event.fingerprint)
        metadata = self._mask_metadata(event.metadata)
        metadata.setdefault("created_at", event.created_at)
        metadata["kind"] = event.kind
        metadata["scope"] = event.scope
        metadata["signature_hash"] = event.signature_hash
        return {
            "filepath": name,
            "relative_filepath": name,
            "category": event.kind,
            "pages": [{"page_content": text}],
            "options": metadata,
        }

    def _ensure_collection(self, name: str) -> None:
        """
        Create *name* if the engine does not have it yet.

        Chunking: with ``chunk_tokens`` at its default of ``0`` nothing is
        declared and the engine splits at its own default of 500 tokens, while
        the *embedded* text is cut at 508 tokens however the collection is
        configured.  One knowledge document is then several chunks, and the
        metadata header — the first line of the text — is only on the first of
        them.

        The engine does not check the embedder or the reranker name against
        what it actually serves: it stores whatever it is given and an unknown
        name surfaces only at the first index or the first search, which is
        why :func:`_verify_model_names` runs once at startup.  The creation
        itself, though, is strict about the collection name —
        :meth:`…collection` already reduced it to the engine's character set
        — so the only thing to tolerate here is a concurrent worker that
        created the same name a moment earlier.
        """
        client = self._sse_client()
        if client.collection_exists(name):
            return
        worker = self._config.worker
        chunking = None
        if worker.chunk_tokens:
            chunking = {
                "max_tokens_in_chunk": int(worker.chunk_tokens),
                "number_of_overlap_tokens": int(worker.chunk_overlap),
            }
        try:
            client.create_collection(
                name,
                description=_COLLECTION_DESCRIPTION,
                embedder=worker.embedder,
                reranker=worker.reranker,
                index_type=worker.index_type,
                chunking=chunking,
            )
        except Exception:
            # A concurrent worker may have created it a moment before; that is
            # not a failure, but anything else must surface so the message
            # stays pending and is retried.
            if client.collection_exists(name):
                return
            raise

    # ------------------------------------------------------------- masking -
    def _get_masker(self) -> Any:
        """Build the FastMasker on first use."""
        if self._masker is None:
            from llm_router_plugins.maskers.fast_masker.core.masker import (
                FastMasker,
            )

            self._masker = FastMasker()
        return self._masker

    def _mask(self, text: str) -> str:
        """Mask a document's text; the reverse mappings are discarded."""
        if not text:
            return text
        try:
            masked, _mappings = self._get_masker().mask_payload(text)
        except Exception as exc:  # a masking failure must not skip indexing
            self._log("warning", "knowledge masking failed: %s", type(exc).__name__)
            return text
        return masked

    def _mask_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mask every string in a metadata mapping, keeping its keys.

        The identity fields are not masked, because they are what the read path
        filters on: the engine's metadata ``eq`` compares with exact equality
        (``DBSemanticSearchController.__check_expression_operator_value``) and
        :func:`llm_router_plugins.utils.knowledge.agentic_knowledge.sse._options`
        builds that filter from the raw thread id of the request.  A masked
        ``thread_id`` is a document nothing can ever retrieve — written,
        indexed, and invisible, with the search reporting success.  This is not
        hypothetical for identifier-shaped runs: ``window-1234567890`` comes
        back as ``window-{KRS_…}``, ``thread-2026-10-17`` as
        ``thread-{DATE_…}`` and a digest of eighteen hex digits or more gets a
        ``{POSTAL_CODE_…}`` spliced into it.  This is the same deal the document
        header already makes in :meth:`_prepare`.
        """
        source = dict(metadata or {})
        identity = {key: source.pop(key) for key in IDENTITY_KEYS if key in source}
        try:
            masked, _mappings = self._get_masker().mask_payload(source)
        except Exception as exc:  # a masking failure must not skip indexing
            self._log(
                "warning",
                "knowledge metadata masking failed: %s",
                type(exc).__name__,
            )
            masked = source
        if not isinstance(masked, dict):
            masked = dict(source)
        masked.update(identity)
        return masked

    # ------------------------------------------------------------ retention -
    def _maybe_retain(self) -> None:
        """Run a retention pass if the interval has elapsed."""
        interval = int(self._config.worker.retention_interval_seconds)
        if interval <= 0:
            return
        now = time.time()
        if now - self._last_retention < interval:
            return
        self._last_retention = now
        self._run_retention()

    def run_retention(self) -> Dict[str, int]:
        """
        Run one retention pass over every scope this worker has indexed.

        Returns
        -------
        Dict[str, int]
            The ``deleted`` count of expired archived documents.
        """
        return self._run_retention()

    def _run_retention(self) -> Dict[str, int]:
        """Delete archived documents older than the archive TTL."""
        result = {"deleted": 0}
        if self._sse is None:
            return result
        cutoff = date.today() - timedelta(
            seconds=int(self._config.archive_ttl_seconds)
        )
        if cutoff >= date.today():
            return result
        for collection in self._archive_collections():
            try:
                result["deleted"] += self._delete_older(collection, cutoff)
            except Exception as exc:
                self._log(
                    "warning",
                    "knowledge retention failed on %s: %s",
                    collection,
                    type(exc).__name__,
                )
        return result

    def _archive_collections(self) -> List[str]:
        """
        Every session-archive collection this plugin owns, TTL or no new traffic.

        The scopes seen since this process started are not enough: retention has
        to run on the archive's own clock, and a worker restarted with an empty
        ``_scopes`` would otherwise leave an old window in place until some new
        event happened to touch its scope.  Asking the engine for the
        collections whose name this plugin gives its archives
        (``<prefix>_<scope>_sessions``) closes that gap; the in-process scopes
        are unioned back in so a collection created moments ago, before it shows
        in a listing, is still swept.  When the engine will not answer a listing
        the pass degrades to the in-process scopes rather than doing nothing.
        """
        found: set = set()
        lister = getattr(self._sse, "list_collections", None)
        if callable(lister):
            try:
                # The engine-safe name is ``prefix_<scope>_sessions``; a
                # release before the engine's character set was enforced
                # created ``prefix-<scope>-sessions`` instead, and those
                # archives still age out through the same clock.
                prefix = self._config.collection_prefix
                for entry in lister() or ():
                    name = getattr(entry, "name", "") or str(entry)
                    if (
                        name.startswith(f"{prefix}_")
                        and name.endswith("_sessions")
                    ) or (
                        name.startswith(f"{prefix}-")
                        and name.endswith("-sessions")
                    ):
                        found.add(name)
            except Exception as exc:  # a listing failure narrows, never crashes
                self._log(
                    "warning",
                    "knowledge retention could not list collections: %s",
                    type(exc).__name__,
                )
        found.update(
            self._config.collection(scope, KIND_SESSION_ARCHIVE)
            for scope in self._scopes
        )
        return sorted(found)

    def _delete_older(self, collection: str, cutoff: date) -> int:
        """Delete this plugin's documents in a collection published before *cutoff*."""
        details = self._sse.list_document_details(collection)
        stale: List[str] = []
        for ref in details or ():
            name = getattr(ref, "name", "") or ""
            if not is_knowledge_document(name):
                continue
            meta = parse_document_name(name)
            if meta.date is not None and meta.date < cutoff:
                stale.append(name)
        if not stale:
            return 0
        self._sse.delete_documents(collection, documents=stale)
        return len(stale)

    # --------------------------------------------------------- redis plumbing -
    def _ensure_group(self) -> None:
        """Create the consumer group once, tolerating that it already exists."""
        if self._group_ready:
            return
        client = self._redis_client()
        try:
            # ``id="0"``, not ``"$"``: a group created at the end of the stream
            # never sees what was published before it existed, so every episode
            # the proxy wrote while no worker ran would be skipped silently —
            # the worker would report a clean start and index nothing.
            client.xgroup_create(self._stream, self._group, id="0", mkstream=True)
        except Exception as exc:
            if not _is_busy_group(exc):
                raise
        self._group_ready = True

    def _read(self, block_ms: Optional[int]) -> List[Tuple[Any, Dict[str, Any]]]:
        """Claim a batch of messages from the stream."""
        client = self._redis_client()
        response = client.xreadgroup(
            self._group,
            self._consumer,
            {self._stream: ">"},
            count=int(self._config.worker.batch),
            block=self._block_ms(block_ms),
        )
        return _entries(response)

    def _block_ms(self, requested: Optional[int]) -> Optional[int]:
        """
        Resolve the block timeout, ``None`` meaning do not wait.

        Redis reads ``BLOCK 0`` as "wait forever", so a caller asking for no
        waiting has to be translated into the argument being absent rather than
        zero — the difference is a cron drain that returns and one that hangs.
        """
        value = (
            int(requested)
            if requested is not None
            else int(self._config.worker.block_ms)
        )
        return value if value > 0 else None

    def _ack(self, entry_id: Any) -> None:
        """Acknowledge one processed message."""
        try:
            self._redis_client().xack(self._stream, self._group, entry_id)
        except Exception as exc:  # an ack that failed is a redelivery, not a loss
            self._log("warning", "knowledge xack failed: %s", type(exc).__name__)

    def _redis_client(self) -> Any:
        """Return the Redis client, building it once."""
        if self._redis is None:
            import redis  # noqa: PLC0415 - optional dependency

            self._redis = redis.Redis(
                **_reading_connection(
                    self._config.redis, self._config.worker.block_ms
                ).client_kwargs()
            )
        return self._redis

    def _sse_client(self) -> Any:
        """Return the engine client, building it once."""
        if self._sse is None:
            from sse_lib import SSEClient  # noqa: PLC0415 - optional dependency

            self._sse = SSEClient(
                self._config.sse_host,
                api_prefix=self._config.sse_prefix or None,
                token=self._config.sse_token,
                username=self._config.sse_username or None,
                password=self._config.sse_password or None,
                timeout=max(self._config.sse_timeout_seconds, 10.0),
                verify=self._config.sse_verify,
            )
        return self._sse

    # --------------------------------------------------------------- helpers -
    def _decode(self, fields: Dict[str, Any]) -> Optional[KnowledgeEvent]:
        """Decode one stream message into an event, or ``None``."""
        raw = fields.get("event") if isinstance(fields, dict) else None
        if raw is None:
            return None
        try:
            payload = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
        except (TypeError, ValueError):
            self._log("warning", "knowledge event is not valid JSON, dropping")
            return None
        if not isinstance(payload, dict):
            return None
        try:
            return KnowledgeEvent.from_dict(payload)
        except Exception as exc:
            self._log("warning", "knowledge event malformed: %s", type(exc).__name__)
            return None

    @staticmethod
    def _supported(event: KnowledgeEvent) -> bool:
        """Whether this worker's version understands the event."""
        try:
            version = int(event.version)
        except (TypeError, ValueError):
            return False
        return version <= EVENT_VERSION

    def _install_signal_handlers(self) -> None:
        """Stop cleanly on SIGINT/SIGTERM when running as the main thread."""
        if threading.current_thread() is not threading.main_thread():
            return
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(signum, lambda *_: self.stop())
            except (ValueError, OSError):  # not installable here; loop still ends
                pass

    def _log(self, level: str, message: str, *args: Any) -> None:
        """Emit one log line when a logger was given."""
        if self._logger is not None:
            getattr(self._logger, level, self._logger.info)(message, *args)


def _entries(response: Any) -> List[Tuple[Any, Dict[str, Any]]]:
    """Flatten ``XREADGROUP`` into ``(entry_id, fields)`` pairs."""
    entries: List[Tuple[Any, Dict[str, Any]]] = []
    if not response:
        return entries
    for _stream, messages in response:
        for message in messages or ():
            entry_id, fields = _message(message)
            if entry_id is not None:
                entries.append((entry_id, fields))
    return entries


def _reading_connection(
    settings: RedisConnectionSettings, block_ms: int
) -> RedisConnectionSettings:
    """
    Return *settings* with a socket timeout that outlives the blocking read.

    Parameters
    ----------
    settings : RedisConnectionSettings
        The connection as configured — a request-path timeout, short on
        purpose so a dead Redis cannot stall a routed request.
    block_ms : int
        How long the worker asks the server to hold a read open; ``0`` means
        the read does not block.

    Returns
    -------
    RedisConnectionSettings
        The same settings, with :attr:`socket_timeout` raised when it would
        otherwise be shorter than the wait this worker deliberately starts.
    """
    needed = max(0, int(block_ms)) / 1000.0 + BLOCK_SOCKET_MARGIN_SECONDS
    if settings.socket_timeout >= needed:
        return settings
    return replace(settings, socket_timeout=needed)


def _message(message: Any) -> Tuple[Any, Dict[str, Any]]:
    """Read an entry id and its fields out of one delivered message."""
    if isinstance(message, (tuple, list)) and len(message) == 2:
        entry_id, fields = message
        return entry_id, (fields if isinstance(fields, dict) else {})
    if isinstance(message, dict):
        return message.get("message_id") or message.get("id"), message
    return None, {}


def _claimed_entries(response: Any) -> List[Tuple[Any, Dict[str, Any]]]:
    """Flatten ``XAUTOCLAIM`` into ``(entry_id, fields)`` pairs.

    The reply is ``[next-id, messages, deleted-ids]``, and older clients answer
    with only the first two; whatever the shape, the messages are the second
    element and anything else is ignored.
    """
    if not isinstance(response, (list, tuple)) or len(response) < 2:
        return []
    entries: List[Tuple[Any, Dict[str, Any]]] = []
    for message in response[1] or ():
        entry_id, fields = _message(message)
        if entry_id is not None:
            entries.append((entry_id, fields))
    return entries


def _pending_count(summary: Any) -> Optional[int]:
    """Read the pending depth of an ``XPENDING`` summary, if it states one."""
    if not isinstance(summary, dict):
        return None
    for key in ("pending", b"pending"):
        value = summary.get(key)
        if isinstance(value, int):
            return value
    return None


def _is_busy_group(exc: Exception) -> bool:
    """Whether an exception is Redis's 'group already exists'."""
    text = str(exc)
    return "BUSYGROUP" in text or "already exists" in text


def _day_of(timestamp: float) -> date:
    """The day of a creation timestamp; the retention bucket."""
    try:
        return date.fromtimestamp(float(timestamp))
    except (OverflowError, OSError, ValueError, TypeError):
        return date.today()


def _default_consumer() -> str:
    """A consumer name unique to this process."""
    return f"worker-{int(time.time())}-{os.getpid()}"


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="llm-router-agentic-kb-worker",
        description=(
            "Index knowledge events published by the agentic knowledge layer "
            "into the Semantic Search Engine."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a knowledge policy JSON; defaults to the bundled policy.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process one batch and exit; useful for cron-style draining.",
    )
    parser.add_argument(
        "--no-retain",
        action="store_true",
        help="Skip the periodic retention pass.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ...).",
    )
    return parser


def _build_worker(args: argparse.Namespace) -> KnowledgeIndexWorker:
    """
    Resolve configuration and construct the worker.

    Everything the worker cannot work without is checked here, once, in terms
    an operator can act on.  A process that starts and quietly indexes into the
    wrong place — or nowhere — costs more than one that refuses to start.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line.

    Returns
    -------
    KnowledgeIndexWorker
        A worker wired to the engine it has just been told to use.
    """
    import pathlib

    if args.config:
        config = AgenticKnowledgeConfig.from_file(pathlib.Path(args.config))
    else:
        config = AgenticKnowledgeConfig.from_file()
    config.override_from_env()
    if args.no_retain:
        from dataclasses import replace

        config.worker = replace(config.worker, retention_interval_seconds=0)
    logger = logging.getLogger("agentic-kb-worker")
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    _require_settings(config)
    try:
        client = _engine_client(config)
    except ImportError as exc:
        raise SystemExit(
            "The knowledge worker needs the SSE client library installed "
            f"({exc})."
        ) from exc
    _verify_model_names(client, config, logger)
    return KnowledgeIndexWorker(config=config, logger=logger, sse_client=client)


def _require_settings(config: AgenticKnowledgeConfig) -> None:
    """
    Refuse to start on a configuration that cannot index anything.

    Parameters
    ----------
    config : AgenticKnowledgeConfig
        The resolved configuration.

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        On a missing engine host, a missing embedder or reranker, or a Redis
        that was never configured — each naming the variable to set.
    """
    if not config.sse_host.strip():
        raise SystemExit(
            "SSE host is not configured "
            "(LLM_ROUTER_KNOWLEDGE_AGENTIC_SSE_API_HOST is unset)."
        )
    if not config.worker.embedder or not config.worker.reranker:
        raise SystemExit(
            "The worker needs an embedder and a reranker to create collections "
            "(LLM_ROUTER_KNOWLEDGE_AGENTIC_SSE_EMBEDDER, "
            "LLM_ROUTER_KNOWLEDGE_AGENTIC_SSE_RERANKER)."
        )
    # Without a Redis host the client would fall back to localhost:6379 and the
    # worker would announce itself while reading a stream nobody publishes to.
    if not config.redis.configured:
        problem = f" (discarded: {config.redis_error})" if config.redis_error else ""
        raise SystemExit(
            "The knowledge worker needs the Redis stream it reads "
            f"(LLM_ROUTER_KNOWLEDGE_AGENTIC_REDIS_HOST is unset){problem}."
        )


def _engine_client(config: AgenticKnowledgeConfig) -> Any:
    """Build the engine client the worker will use, from the SSE settings."""
    from sse_lib import SSEClient  # noqa: PLC0415 - optional dependency

    return SSEClient(
        config.sse_host,
        api_prefix=config.sse_prefix or None,
        token=config.sse_token,
        username=config.sse_username or None,
        password=config.sse_password or None,
        timeout=max(config.sse_timeout_seconds, 10.0),
        verify=config.sse_verify,
    )


def _verify_model_names(
    client: Any, config: AgenticKnowledgeConfig, logger: Any
) -> None:
    """
    Check the configured embedder and reranker against the ones the engine has.

    Creating a collection does not validate either name — the engine stores
    whatever it is given and only fails later, at the first index or the first
    search, by which time the collection row exists and is permanently broken.
    Asking once at startup is the difference between a message at boot and a
    knowledge base that silently never fills.

    Fail-open: an engine that does not answer ``embedders``/``rerankers`` —
    too old, or the endpoint not reachable — must not keep the worker down.

    Parameters
    ----------
    client : Any
        The engine client.
    config : AgenticKnowledgeConfig
        The resolved configuration; its worker names the models to check.
    logger : Any
        Where the fail-open is reported.

    Returns
    -------
    None

    Raises
    ------
    SystemExit
        When the engine answers and does not serve one of the two names.
    """
    worker = config.worker
    try:
        served = {
            "embedder": list(client.list_embedders() or ()),
            "reranker": list(client.list_rerankers() or ()),
        }
    except Exception as exc:
        if logger is not None:
            logger.warning(
                "could not list the engine's embedders and rerankers, continuing"
                " unverified: %s",
                type(exc).__name__,
            )
        return
    for kind, wanted in (
        ("embedder", worker.embedder),
        ("reranker", worker.reranker),
    ):
        names = served[kind]
        if names and wanted not in names:
            raise SystemExit(
                f"The SSE engine does not serve the {kind} {wanted!r}; it offers "
                f"{', '.join(sorted(names))}."
            )


def main(argv: Optional[List[str]] = None) -> int:
    """
    Run the worker.

    Parameters
    ----------
    argv : List[str], optional
        Command-line arguments; ``sys.argv`` when omitted.

    Returns
    -------
    int
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    worker = _build_worker(args)
    if args.once:
        logger = logging.getLogger("agentic-kb-worker")
        try:
            processed = worker.run_once(block_ms=0)
        except Exception as exc:
            logger.error(
                "knowledge worker batch failed: %s", type(exc).__name__
            )
            return 1
        # A drained batch is the only evidence a cron-driven worker leaves
        # behind; silence and "it worked" would be indistinguishable.
        logger.info("knowledge worker drained one batch: %d event(s)", processed)
        return 0
    worker.run()
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    sys.exit(main())
