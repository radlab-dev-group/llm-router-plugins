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
from typing import Any, Dict, List, Optional, Tuple

from llm_router_plugins.utils.knowledge.agentic_knowledge.adapters.base import (
    KnowledgeEvent,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
    AgenticKnowledgeConfig,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.documents import (
    KIND_SESSION_ARCHIVE,
    document_name,
    is_knowledge_document,
    parse_document_name,
)

__all__ = ["KnowledgeIndexWorker", "main"]

#: Wire version this worker understands.
EVENT_VERSION = 1

#: Attempts before a message is dropped; a poison message must not stall the
#: group, so after this many failures it is acknowledged with a log.
DEFAULT_MAX_ATTEMPTS = 3

#: Description of a collection this worker creates; the engine rejects an
#: empty one, and an operator who wants a different description pre-creates it.
_COLLECTION_DESCRIPTION = "Agentic knowledge base (auto-created by llm-router)."


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
            except Exception as exc:  # a read failure must not end the loop
                self._log(
                    "warning", "knowledge worker read failed: %s", type(exc).__name__
                )
                time.sleep(1.0)
                continue
            if processed == 0:
                self._maybe_retain()
        self._log("info", "knowledge worker stopped")

    def run_once(self, block_ms: Optional[int] = None) -> int:
        """
        Claim and process one batch.

        Parameters
        ----------
        block_ms : int, optional
            How long to wait for messages; the worker setting when omitted.

        Returns
        -------
        int
            Messages processed (indexed or dropped) in this batch.
        """
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
            if self._attempts(fields) >= self._max_attempts:
                self._log(
                    "error",
                    "knowledge index failed, dropping (fingerprint=%s, %s)",
                    event.fingerprint,
                    type(exc).__name__,
                )
                self._ack(entry_id)
            else:
                self._log(
                    "warning",
                    "knowledge index failed, leaving pending (fingerprint=%s, %s)",
                    event.fingerprint,
                    type(exc).__name__,
                )
            return False
        self._ack(entry_id)
        return True

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
        text = self._mask(event.text)
        document = self._document(text, event)
        collection = self._config.collection(event.scope, event.kind)
        self._ensure_collection(collection)
        self._sse_client().index_documents(collection, [document])

    # ---------------------------------------------------------- collections -
    def _document(self, text: str, event: KnowledgeEvent) -> Dict[str, Any]:
        """Build one ``texts[]`` element, mirroring ``sse_lib.text_document``."""
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
        """Create *name* if the engine does not have it yet."""
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
        """Mask every string in a metadata mapping, keeping its keys."""
        masked, _mappings = self._get_masker().mask_payload(dict(metadata or {}))
        return masked if isinstance(masked, dict) else dict(metadata or {})

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
        for scope in list(self._scopes):
            collection = self._config.collection(scope, KIND_SESSION_ARCHIVE)
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
        """Create the consumer group, tolerating that it already exists."""
        client = self._redis_client()
        try:
            client.xgroup_create(self._stream, self._group, id="$", mkstream=True)
        except Exception as exc:
            if not _is_busy_group(exc):
                raise

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
        """Resolve the block timeout, defaulting to the worker setting."""
        if requested is not None:
            return int(requested)
        return int(self._config.worker.block_ms)

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

            self._redis = redis.Redis(**self._config.redis.client_kwargs())
        return self._redis

    def _sse_client(self) -> Any:
        """Return the engine client, building it once."""
        if self._sse is None:
            from sse_lib import SSEClient  # noqa: PLC0415 - optional dependency

            self._sse = SSEClient(
                self._config.sse_host,
                api_prefix=self._config.sse_prefix,
                token=self._config.sse_token,
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

    @staticmethod
    def _attempts(fields: Dict[str, Any]) -> int:
        """Delivery count Redis reports for the message, if any."""
        if not isinstance(fields, dict):
            return 1
        for key in ("delivery_count", "times_delivered", "delivery"):
            value = fields.get(key)
            if isinstance(value, int):
                return max(1, value)
        return 1

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


def _message(message: Any) -> Tuple[Any, Dict[str, Any]]:
    """Read an entry id and its fields out of one delivered message."""
    if isinstance(message, (tuple, list)) and len(message) == 2:
        entry_id, fields = message
        return entry_id, (fields if isinstance(fields, dict) else {})
    if isinstance(message, dict):
        return message.get("message_id") or message.get("id"), message
    return None, {}


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
    """Resolve configuration and construct the worker."""
    import pathlib

    if args.config:
        config = AgenticKnowledgeConfig.from_file(pathlib.Path(args.config))
    else:
        config = AgenticKnowledgeConfig.from_file()
    config.override_from_env()
    if args.no_retain:
        from dataclasses import replace

        config.worker = replace(config.worker, retention_interval_seconds=0)
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
    logger = logging.getLogger("agentic-kb-worker")
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return KnowledgeIndexWorker(config=config, logger=logger)


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
        try:
            worker.run_once(block_ms=0)
        except Exception as exc:
            logging.getLogger("agentic-kb-worker").error(
                "knowledge worker batch failed: %s", type(exc).__name__
            )
            return 1
        return 0
    worker.run()
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    sys.exit(main())
