"""
The indexing worker: claim, mask, create collections, index, retire.

The worker consumes the stream with a consumer group, masks every document and
throws the mappings away, creates the target collection on first use (embedder
and reranker come from the worker's own ENV), indexes, and retires archived
documents on a schedule.  A poison message is retried a bounded number of times
and then dropped with a log; it never stalls the group.
"""

import dataclasses
import json
import logging
import sys
import time
import datetime
import pathlib
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agentic_knowledge_support import (
    FakeRedis,
    FakeSSE,
    knowledge_config,
    knowledge_event,
    stream_message,
)

from llm_router_plugins.cli.plugins.utils.agentic_knowledge.worker import (
    KnowledgeIndexWorker,
    _engine_client,
    _entries,
    _reading_connection,
    _require_settings,
    _verify_model_names,
)
from llm_router_plugins.cli.plugins.utils.agentic_knowledge import (
    worker as worker_cli,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.documents import (
    KIND_SESSION_ARCHIVE,
    parse_document_text,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.sse import (
    SearchRequest,
    _options,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.state import (
    RedisConnectionSettings,
)


class FakeSSE(FakeSSE):
    """Adapt the shared double to the worker's indexing-options interface."""

    def index_documents(self, collection, documents, indexing_options=None):
        return super().index_documents(collection, documents)


class _SilentLogger:
    """Records the one warning the fail-open path is allowed to write."""

    def __init__(self):
        self.warnings = []

    def warning(self, message, *args):
        self.warnings.append(message % args if args else message)


def worker(redis=None, sse=None, config=None, **overrides):
    config = config or knowledge_config()
    config = dataclasses.replace(config)
    config.sse_host = "http://engine.test"
    config.worker = dataclasses.replace(
        config.worker, embedder="embed", reranker="rerank"
    )
    return KnowledgeIndexWorker(
        config=config,
        redis_client=redis or FakeRedis(),
        sse_client=sse or FakeSSE(),
        **overrides,
    )


def run_event(event, sse=None, redis_messages=None, **overrides):
    redis = FakeRedis(messages=[("1-0", stream_message(event))])
    sse = sse or FakeSSE()
    work = worker(redis=redis, sse=sse, **overrides)
    work._ensure_group()
    processed = work.run_once(block_ms=0)
    return processed, redis, sse


def test_indexes_a_recipe_event_into_the_scope_collection():
    processed, redis, sse = run_event(knowledge_event(kind="recipe", scope="webapp"))
    assert processed == 1
    assert redis.acked == ["1-0"]
    assert len(sse.indexed) == 1
    collection, documents = sse.indexed[0]
    assert collection == "agentic_kb_webapp"
    assert documents[0]["category"] == "recipe"


def test_session_archive_goes_to_the_sessions_collection():
    _processed, _redis, sse = run_event(
        knowledge_event(kind="session_archive", scope="webapp")
    )
    assert sse.indexed[0][0] == "agentic_kb_webapp_sessions"


def test_document_name_carries_kind_date_and_fingerprint():
    event = knowledge_event(kind="recipe", fingerprint="deadbeef", scope="webapp")
    _processed, _redis, sse = run_event(event)
    name = sse.indexed[0][1][0]["filepath"]
    assert name.startswith("recipe:")
    assert name.endswith(":deadbeef")
    # The retention bucket is the day of the event. Asserting today's date
    # instead made this test fail on every day but the one it was written on.
    day = datetime.date.fromtimestamp(event.created_at).strftime("%Y%m%d")
    assert f":{day}:" in name


def test_document_text_and_metadata_are_masked():
    event = knowledge_event(
        kind="recipe",
        text="Contact jane@example.com about the fix",
    )
    event = dataclasses.replace(
        event, metadata={"paths": ["a.py"], "contact": "jane@example.com"}
    )
    _processed, _redis, sse = run_event(event)
    document = sse.indexed[0][1][0]
    assert "jane@example.com" not in document["pages"][0]["page_content"]
    assert "jane@example.com" not in json.dumps(document["options"])


@pytest.mark.parametrize("target", ["body", "signature", "test_command", "metadata"])
@pytest.mark.parametrize("entrypoint", ["batch", "single"])
def test_masking_failure_stays_pending_and_reclaim_retries(target, entrypoint, caplog):
    secret = "jane@example.com"
    event = dataclasses.replace(
        knowledge_event(text=secret if target == "body" else "safe body"),
        signature=secret if target == "signature" else "",
        metadata={target: secret},
    )
    entry = ("1-0", stream_message(event))
    redis = FakeRedis(messages=[entry])
    sse = FakeSSE()
    work = worker(redis=redis, sse=sse)
    masker = work._get_masker()

    class FailingMasker:
        def mask_payload(self, payload):
            if (target == "metadata" and isinstance(payload, dict)) or payload == secret:
                raise RuntimeError(secret)
            return masker.mask_payload(payload)

    work._masker = FailingMasker()
    with caplog.at_level("WARNING"):
        if entrypoint == "batch":
            work.run_once(block_ms=0)
        else:
            assert work._process_entry(*entry) is False
    assert sse.indexed == []
    assert sse.created == []
    assert redis.acked == []
    assert redis.kv[work._tries_key("1-0")] == 1
    assert secret not in caplog.text
    assert "retrying" in caplog.text

    work._masker = masker
    redis.pending.append(entry)
    assert work._reclaim() == 1
    assert redis.acked == ["1-0"]
    assert len(sse.indexed) == 1
    assert secret not in json.dumps(sse.indexed)
    assert work._tries_key("1-0") not in redis.kv


@pytest.mark.parametrize(
    "target,result",
    [
        (target, result)
        for target in ("text", "metadata")
        for result in (
            None, "ab", {"a": 1, "b": 2}, (), ("safe",),
            ("safe", {}, "extra"), (None, {}), (123, {}), ([], {}),
            ("safe", None), ({}, None),
        )
    ] + [("text", ({}, {})), ("metadata", ("safe", {}))],
)
def test_invalid_masker_result_is_not_indexed(target, result):
    redis = FakeRedis()
    sse = FakeSSE()
    work = worker(redis=redis, sse=sse)
    masker = work._get_masker()

    class InvalidMasker:
        def mask_payload(self, payload):
            if isinstance(payload, dict) == (target == "metadata"):
                return result
            return masker.mask_payload(payload)

    work._masker = InvalidMasker()
    event = dataclasses.replace(
        knowledge_event(text="jane@example.com"),
        metadata={"contact": "jane@example.com"},
    )
    work._process_batch([("1-0", stream_message(event))])
    assert sse.indexed == []
    assert sse.created == []
    assert redis.acked == []
    assert redis.kv[work._tries_key("1-0")] == 1


def test_masker_initialization_failure_is_retried_then_dropped(monkeypatch, caplog):
    redis = FakeRedis()
    sse = FakeSSE()
    work = worker(redis=redis, sse=sse, max_attempts=2)

    def fail():
        raise RuntimeError("jane@example.com")

    monkeypatch.setattr(work, "_get_masker", fail)
    entry = ("1-0", stream_message(knowledge_event()))
    with caplog.at_level("WARNING"):
        assert work._process_entry(*entry) is False
        assert redis.acked == []
        assert work._process_entry(*entry) is False
    assert sse.indexed == []
    assert sse.created == []
    assert redis.acked == ["1-0"]
    assert work._dropped == 1
    assert work._tries_key("1-0") not in redis.kv
    assert "dropping" in caplog.text
    assert "jane@example.com" not in caplog.text


def test_masking_failure_does_not_block_other_documents_in_batch():
    redis = FakeRedis()
    sse = FakeSSE()
    work = worker(redis=redis, sse=sse)
    masker = work._get_masker()

    class SelectiveMasker:
        def mask_payload(self, payload):
            if payload == "jane@example.com":
                raise RuntimeError("masking unavailable")
            return masker.mask_payload(payload)

    work._masker = SelectiveMasker()
    bad = knowledge_event(text="jane@example.com", fingerprint="bad")
    good = knowledge_event(text="safe body", fingerprint="good")
    assert work._process_batch([
        ("1-0", stream_message(bad)), ("2-0", stream_message(good)),
    ]) == (2, 1)
    assert redis.acked == ["2-0"]
    assert len(sse.indexed) == 1
    assert len(sse.indexed[0][1]) == 1
    assert sse.indexed[0][1][0]["filepath"].endswith(":good")


# --------------------------------------------------------------------- F1 ---
# The engine matches a metadata ``eq`` with exact equality and the reader builds
# that filter from the raw thread id, so a filter value the masker rewrote is a
# document that is written, indexed, and unreachable — with the search reporting
# ``status: ok`` and no hits.


#: Values the FastMasker demonstrably rewrites today, so the assertions below
#: fail without the guard: a ten-digit run reads as a court-register number, an
#: ISO date as a date, and an eighteen-plus-hex digest as a postal code.
MASKABLE = {
    "window_id": "window-1234567890",
    "thread_id": "thread-2026-10-17",
    "signature_hash": "0a1b2c3d4e5f67890a1b",
}


def test_metadata_filter_keys_are_byte_identical_to_the_reader_s():
    """
    What the worker writes has to be what the reader asks for, byte for byte.

    Not "contains", not "normalises to": the engine compares with ``==``.
    """
    event = dataclasses.replace(
        knowledge_event(kind="session_archive", scope="webapp"),
        thread_id=MASKABLE["thread_id"],
        signature_hash=MASKABLE["signature_hash"],
        metadata={
            "thread_id": MASKABLE["thread_id"],
            "window_id": MASKABLE["window_id"],
            "signature_hash": MASKABLE["signature_hash"],
            "contact": "jane@example.com",
        },
    )
    _processed, _redis, sse = run_event(event)
    stored = sse.indexed[0][1][0]["options"]

    # The filter the reader would build for exactly this document.
    options = _options(
        SearchRequest(
            collection="agentic_kb_webapp_sessions",
            query="anything",
            categories=(KIND_SESSION_ARCHIVE,),
            thread_id=event.thread_id,
        ),
        0.35,
    )
    expression = options["metadata_filters"][0]
    assert expression["operator"] == "eq"
    ((key, wanted),) = expression["field"].items()
    assert key in stored, "the reader filters on a field that was never written"
    assert stored[key] == wanted, "exact equality would find nothing"

    for name, value in MASKABLE.items():
        assert stored[name] == value, f"{name} was rewritten by the masker"
    assert stored["kind"] == KIND_SESSION_ARCHIVE
    assert stored["scope"] == "webapp"
    # Free text is still masked; keeping the keys legible is not a free pass.
    assert "jane@example.com" not in json.dumps(stored)


def test_the_phase_survives_the_masker_like_the_other_filter_keys():
    """
    The read path filters on ``phase`` with ``eq``; the worker masks everything
    that is not an identity key, so the phase has to be one.
    """
    event = dataclasses.replace(
        knowledge_event(kind="recipe"),
        metadata={
            "thread_id": MASKABLE["thread_id"],
            "phase": "test",
            "contact": "jane@example.com",
        },
    )
    _processed, _redis, sse = run_event(event)
    stored = sse.indexed[0][1][0]["options"]
    assert stored["phase"] == "test"
    assert "jane@example.com" not in json.dumps(stored)


def test_the_header_and_the_metadata_agree_on_the_identity():
    """A reader that lost the metadata still reads the header off the text."""
    event = dataclasses.replace(
        knowledge_event(kind="session_archive"),
        thread_id=MASKABLE["thread_id"],
        signature_hash=MASKABLE["signature_hash"],
        metadata={"thread_id": MASKABLE["thread_id"]},
    )
    _processed, _redis, sse = run_event(event)
    document = sse.indexed[0][1][0]
    meta, _body = parse_document_text(document["pages"][0]["page_content"])
    assert meta.thread_id == document["options"]["thread_id"]
    assert meta.signature_hash == document["options"]["signature_hash"]


# --------------------------------------------------------------------- F6 ---
class _Probe:
    """An engine that answers the model-listing calls."""

    def __init__(self, embedders=("bge-m3",), rerankers=("bge-reranker",)):
        self._embedders = embedders
        self._rerankers = rerankers

    def list_embedders(self):
        return list(self._embedders)

    def list_rerankers(self):
        return list(self._rerankers)


class _NoProbe(_Probe):
    def list_embedders(self):
        raise RuntimeError("404")


def _worker_config(embedder="bge-m3", reranker="bge-reranker"):
    config = knowledge_config()
    config.worker = dataclasses.replace(
        config.worker, embedder=embedder, reranker=reranker
    )
    return config


def test_an_unknown_reranker_stops_the_worker_before_it_creates_anything():
    """
    Collection creation never checks the reranker, so a typo is permanent.

    The engine writes the relational row and answers success; the failure only
    shows up at the first search, by which time the broken collection is there.
    """
    config = _worker_config(reranker="bge-rerankr")
    with pytest.raises(SystemExit) as raised:
        _verify_model_names(_Probe(), config, None)
    message = str(raised.value)
    assert "bge-rerankr" in message
    assert "bge-reranker" in message, "the engine's own names should be listed"


def test_an_unknown_embedder_stops_the_worker_too():
    config = _worker_config(embedder="nope")
    with pytest.raises(SystemExit) as raised:
        _verify_model_names(_Probe(), config, None)
    assert "embedder" in str(raised.value)


def test_names_the_engine_serves_are_accepted():
    _verify_model_names(_Probe(), _worker_config(), None)


def test_an_engine_that_does_not_answer_the_lookup_does_not_stop_startup():
    """Fail-open: an old engine must not keep the worker down."""
    logger = _SilentLogger()
    _verify_model_names(_NoProbe(), _worker_config(), logger)
    _verify_model_names(_NoProbe(), _worker_config(), None)
    _verify_model_names(_Probe(embedders=(), rerankers=()), _worker_config(), None)
    assert logger.warnings, "the fail-open should still say it happened"
    assert "unverified" in logger.warnings[0]


# --------------------------------------------------------------------- F7 ---
def test_a_worker_without_a_redis_host_refuses_to_start():
    """
    Otherwise it reads localhost:6379 forever and says it started.

    The gate that matters is the one nobody notices missing: the stream is the
    only source of work, so an unset host means a healthy-looking process that
    indexes nothing at all.
    """
    config = _worker_config()
    config.redis = RedisConnectionSettings()
    with pytest.raises(SystemExit) as raised:
        _require_settings(config)
    assert "REDIS_HOST" in str(raised.value)


def test_a_discarded_malformed_redis_says_why():
    config = _worker_config()
    config.redis = RedisConnectionSettings()
    config.redis_error = "port out of range: 99999"
    with pytest.raises(SystemExit) as raised:
        _require_settings(config)
    assert "port out of range: 99999" in str(raised.value)


def test_a_configured_worker_passes_the_gates():
    _require_settings(_worker_config())


def test_a_missing_sse_host_still_refuses_to_start():
    config = _worker_config()
    config.sse_host = ""
    with pytest.raises(SystemExit) as raised:
        _require_settings(config)
    assert "SSE_API_HOST" in str(raised.value)


def test_a_missing_embedder_still_refuses_to_start():
    config = _worker_config(embedder="")
    with pytest.raises(SystemExit):
        _require_settings(config)


# --------------------------------------------------------------------- F8 ---
def test_the_default_chunking_is_left_to_the_engine():
    """
    ``chunk_tokens`` of ``0`` sends no chunking at all.

    That is the engine's 500-token default, not "no chunking", and the engine
    cuts the *embedded* text at 508 tokens however the collection is declared —
    so one knowledge document is usually several chunks and the header only
    rides on the first.
    """
    sse = FakeSSE()
    config = knowledge_config()
    config.sse_host = "http://engine.test"
    config.worker = dataclasses.replace(
        config.worker, embedder="embed", reranker="rerank", chunk_tokens=0
    )
    work = KnowledgeIndexWorker(
        config=config, redis_client=FakeRedis(), sse_client=sse
    )
    work._ensure_collection("agentic_kb_webapp")
    assert sse.created, "the collection was never created"


def test_an_explicit_chunking_is_declared_as_a_pair():
    seen = {}

    class Recording(FakeSSE):
        def create_collection(self, name, description="", **kwargs):
            seen.update(kwargs)
            super().create_collection(name, description=description, **kwargs)

    config = knowledge_config()
    config.sse_host = "http://engine.test"
    config.worker = dataclasses.replace(
        config.worker,
        embedder="embed",
        reranker="rerank",
        chunk_tokens=900,
        chunk_overlap=64,
    )
    work = KnowledgeIndexWorker(
        config=config, redis_client=FakeRedis(), sse_client=Recording()
    )
    work._ensure_collection("agentic_kb_webapp")
    assert seen["chunking"] == {
        "max_tokens_in_chunk": 900,
        "number_of_overlap_tokens": 64,
    }


def test_collection_is_created_with_embedder_and_reranker():
    _processed, _redis, sse = run_event(knowledge_event())
    assert len(sse.created) == 1
    _name, embedder, reranker = sse.created[0]
    assert (embedder, reranker) == ("embed", "rerank")


def test_existing_collection_is_not_recreated():
    sse = FakeSSE(exists=("agentic_kb_webapp",))
    _processed, _redis, _sse = run_event(knowledge_event(), sse=sse)
    assert _sse.created == []


def test_unprocessable_message_is_retried_then_dropped():
    class FailingSSE(FakeSSE):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.calls = 0

        def index_documents(self, collection, documents, indexing_options=None):
            self.calls += 1
            raise RuntimeError("index down")

    sse = FailingSSE()
    redis = FakeRedis()
    work = worker(redis=redis, sse=sse, max_attempts=2)
    message = stream_message(knowledge_event())
    # Redis reports no delivery count to a consumer reading ">", so the worker
    # counts attempts itself: the first failure leaves the message pending.
    work._process_entry("1-0", message)
    assert redis.acked == []
    # The second failure of the same entry uses the attempts up: drop with ack.
    work._process_entry("1-0", message)
    assert redis.acked == ["1-0"]
    assert sse.calls == 2
    # A settled entry does not leave its counter behind.
    assert not [key for key in redis.kv if ":tries:" in key]


def test_run_once_creates_the_consumer_group_itself():
    """A single-batch run has to work on a stream nobody has grouped yet."""
    redis = FakeRedis(messages=[("1-0", stream_message(knowledge_event()))])
    work = worker(redis=redis, sse=FakeSSE())
    assert work.run_once(block_ms=0) == 1
    assert redis.groups, "XGROUP CREATE ... MKSTREAM was never issued"
    assert redis.acked == ["1-0"]


def test_run_once_without_waiting_passes_no_block():
    """``0`` means do not wait; Redis reads ``BLOCK 0`` as wait forever."""
    redis = FakeRedis()
    work = worker(redis=redis, sse=FakeSSE())
    assert work.run_once(block_ms=0) == 0
    assert redis.reads == [None]


def test_reclaim_takes_over_an_idle_entry():
    """An entry left unacknowledged is claimed back and indexed."""
    pending = [
        ("2-0", stream_message(knowledge_event(kind="recipe", scope="webapp")))
    ]
    redis = FakeRedis(pending=pending)
    sse = FakeSSE()
    work = worker(redis=redis, sse=sse)
    assert work._reclaim() == 1
    assert redis.acked == ["2-0"]
    assert sse.indexed, "the reclaimed entry was never indexed"


def test_backlog_above_the_threshold_is_reported(caplog):
    redis = FakeRedis(pending=[(f"{i}-0", {}) for i in range(3)])
    work = worker(redis=redis, sse=FakeSSE())
    work._config.worker = dataclasses.replace(work._config.worker, max_pending=2)
    with caplog.at_level("WARNING"):
        work._warn_backlog()
    assert "backlog" in caplog.text


def test_backlog_below_the_threshold_is_quiet(caplog):
    redis = FakeRedis(pending=[("1-0", {})])
    work = worker(redis=redis, sse=FakeSSE())
    work._config.worker = dataclasses.replace(work._config.worker, max_pending=100)
    with caplog.at_level("WARNING"):
        work._warn_backlog()
    assert "backlog" not in caplog.text


def test_the_socket_timeout_outlives_the_blocking_read():
    """
    A daemon worker blocks a read for ``worker.block_ms``; the socket timeout
    comes from the request-path settings, where 1 s is a feature.  Left alone
    it is shorter than the block, so every idle wait times out and the loop
    never waits.  The worker must raise its own socket timeout over its block.
    """
    settings = RedisConnectionSettings(host="r", socket_timeout=1.0)
    assert _reading_connection(settings, 5000).socket_timeout == 7.0


def test_a_longer_operator_block_raises_the_socket_with_it():
    settings = RedisConnectionSettings(host="r", socket_timeout=1.0)
    assert _reading_connection(settings, 15000).socket_timeout == 17.0


def test_a_socket_already_long_enough_is_kept():
    settings = RedisConnectionSettings(host="r", socket_timeout=10.0)
    assert _reading_connection(settings, 5000).socket_timeout == 10.0


def test_a_non_blocking_read_needs_only_the_margin():
    settings = RedisConnectionSettings(host="r", socket_timeout=1.0)
    assert _reading_connection(settings, 0).socket_timeout == 2.0


def test_a_drained_batch_says_so(caplog, monkeypatch):
    """
    A cron-run worker leaves no other trace: without a line about the batch it
    processed nothing and did nothing look exactly the same.
    """

    class _CountingWorker:
        def run_once(self, block_ms=None):
            return 3

    monkeypatch.setattr(worker_cli, "_build_worker", lambda args: _CountingWorker())
    with caplog.at_level("INFO", logger="agentic-kb-worker"):
        assert worker_cli.main(["--once"]) == 0
    assert "3 event(s)" in caplog.text


def test_malformed_or_unversioned_message_is_acked_and_skipped():
    redis = FakeRedis()
    work = worker(redis=redis, sse=FakeSSE())
    assert work._process_entry("1-0", {"event": "not json"}) is False
    assert redis.acked == ["1-0"]
    redis2 = FakeRedis()
    work2 = worker(redis=redis2, sse=FakeSSE())
    bad = stream_message(knowledge_event())
    bad = {
        "event": json.dumps(
            {**json.loads(stream_message(knowledge_event())["event"]), "version": 99}
        )
    }
    assert work2._process_entry("2-0", bad) is False
    assert redis2.acked == ["2-0"]


def test_retention_deletes_expired_archived_documents():
    config = knowledge_config()
    config = dataclasses.replace(config)
    config.archive_ttl_seconds = 7 * 86400
    sse = FakeSSE()
    work = worker(sse=sse, config=config)
    work._scopes.add("webapp")
    old = (datetime.date.today() - datetime.timedelta(days=30)).strftime("%Y%m%d")
    recent = datetime.date.today().strftime("%Y%m%d")
    sse.names["agentic_kb_webapp_sessions"] = [
        f"session_archive:{old}:old",
        f"session_archive:{recent}:recent",
        "a-human-document.md",
    ]
    result = work.run_retention()
    assert result["deleted"] == 1
    assert sse.deleted == [
        ("agentic_kb_webapp_sessions", [f"session_archive:{old}:old"])
    ]


def test_retention_never_touches_documents_outside_the_plugin_shape():
    config = knowledge_config()
    config = dataclasses.replace(config)
    config.archive_ttl_seconds = 7 * 86400
    sse = FakeSSE()
    work = worker(sse=sse, config=config)
    work._scopes.add("webapp")
    sse.names["agentic_kb_webapp_sessions"] = ["notes.md", "20160101:legacy:doc"]
    assert work.run_retention()["deleted"] == 0
    assert sse.deleted == []


def test_the_consumer_group_reads_from_the_start_of_the_stream():
    """
    A group created at the end of the stream never sees what was published
    before it, so episodes written while no worker ran would be skipped.
    """
    captured = {}

    class Watching(FakeRedis):
        def xgroup_create(self, stream, group, id="0", mkstream=False):
            captured["id"] = id
            return super().xgroup_create(stream, group, id=id, mkstream=mkstream)

    work = worker(redis=Watching())
    work._ensure_group()
    assert captured["id"] == "0"


def test_an_entry_published_before_the_first_run_is_consumed():
    """A backlog that predates the group is drained, not lost."""
    event = knowledge_event(kind="recipe", scope="webapp")
    redis = FakeRedis(messages=[("0-1", stream_message(event))])
    sse = FakeSSE()
    work = worker(redis=redis, sse=sse)
    processed = work.run_once(block_ms=0)
    assert processed == 1
    assert len(sse.indexed) == 1


def test_ensure_group_tolerates_a_preexisting_group():
    class BusyGroup(FakeRedis):
        def xgroup_create(self, *a, **k):
            raise Exception("BUSYGROUP Consumer Group name already exists")

    work = worker(redis=BusyGroup())
    work._ensure_group()  # must not raise


def test_entries_flattens_xreadgroup_responses():
    response = [("s", [("1-0", {"event": "a"}), ("2-0", {"event": "b"})])]
    entries = _entries(response)
    assert entries == [("1-0", {"event": "a"}), ("2-0", {"event": "b"})]
    assert _entries([]) == []


def test_retention_sweeps_an_archive_this_worker_never_indexed():
    # A worker restarted with empty in-process scopes must still expire windows
    # its predecessor wrote; it discovers its own archive collections from the
    # engine listing, not only from what it happened to index this run.
    config = knowledge_config()
    config = dataclasses.replace(config)
    config.archive_ttl_seconds = 7 * 86400
    # The legacy spelling keeps the test honest: the listing match must catch
    # a name no current release would create.
    old = (datetime.date.today() - datetime.timedelta(days=30)).strftime("%Y%m%d")
    sse = FakeSSE(
        names={
            "agentic_kb_legacy_sessions": [f"session_archive:{old}:old"],
            "agentic_kb-legacy-sessions": [f"session_archive:{old}:legacy"],
        },
        catalogue=(
            "agentic_kb_legacy_sessions",
            "agentic_kb-legacy-sessions",
            "some-human-collection",
        ),
    )
    work = worker(sse=sse, config=config)
    assert work._scopes == set()  # nothing indexed since this process started
    result = work.run_retention()
    assert result["deleted"] == 2
    assert sorted(sse.deleted) == [
        ("agentic_kb-legacy-sessions", [f"session_archive:{old}:legacy"]),
        ("agentic_kb_legacy_sessions", [f"session_archive:{old}:old"]),
    ]


def test_the_engine_client_is_built_with_the_login(monkeypatch):
    """
    The worker writes with the same credential the read side uses.

    An engine behind login answers a token-less client with 401, and the
    worker would report that as an indexing failure rather than as missing
    credentials.
    """
    built = {}

    class _Built:
        def __init__(self, host, **kwargs):
            built["host"] = host
            built.update(kwargs)

    module = types.ModuleType("sse_lib")
    module.SSEClient = _Built
    monkeypatch.setitem(sys.modules, "sse_lib", module)
    config = _worker_config()
    config.sse_username = "default_user"
    config.sse_password = "secret"
    _engine_client(config)
    assert built["username"] == "default_user"
    assert built["password"] == "secret"


def test_retention_ignores_foreign_collections_in_the_listing():
    config = knowledge_config()
    config = dataclasses.replace(config)
    config.archive_ttl_seconds = 7 * 86400
    sse = FakeSSE(catalogue=("documents-team-x", "agentic_kb_webapp"))
    work = worker(sse=sse, config=config)
    # Neither is a `-sessions` archive of ours; nothing is listed for deletion.
    assert work.run_retention()["deleted"] == 0
    assert sse.deleted == []


def test_the_default_chunking_is_declared_on_a_new_collection():
    """A collection created with the defaults carries 400/60, not nothing."""
    seen = {}

    class Recording(FakeSSE):
        def create_collection(self, name, description="", **kwargs):
            seen.update(kwargs)
            super().create_collection(name, description=description, **kwargs)

    config = knowledge_config()
    config.sse_host = "http://engine.test"
    config.worker = dataclasses.replace(
        config.worker, embedder="embed", reranker="rerank"
    )
    work = KnowledgeIndexWorker(
        config=config, redis_client=FakeRedis(), sse_client=Recording()
    )
    work._ensure_collection("agentic_kb_webapp")
    assert seen["chunking"] == {
        "max_tokens_in_chunk": 400,
        "number_of_overlap_tokens": 60,
    }


# --------------------------------------------------------------- verbosity -
class _Recorder:
    """A logger that keeps every line, whatever its level."""

    def __init__(self):
        self.lines = []

    def info(self, message, *args):
        self.lines.append(message % args if args else message)

    @property
    def text(self):
        return "\n".join(self.lines)


def test_the_command_line_level_beats_the_router_switches(monkeypatch):
    monkeypatch.setenv("LLM_ROUTER_IN_DEBUG", "1")
    monkeypatch.setenv("LLM_ROUTER_LOG_LEVEL", "WARNING")
    level, reason = worker_cli._log_level("ERROR")
    assert (level, reason) == (logging.ERROR, "command line")


def test_the_router_debug_switch_debugs_the_worker(monkeypatch):
    """
    The operator turns debug on in one script, the router's, and the wrapper
    exports it here — a worker that stayed at INFO in a debug deployment was
    the whole complaint.
    """
    monkeypatch.delenv("LLM_ROUTER_LOG_LEVEL", raising=False)
    monkeypatch.setenv("LLM_ROUTER_IN_DEBUG", "true")
    assert worker_cli._log_level(None) == (logging.DEBUG, "LLM_ROUTER_IN_DEBUG")


def test_a_router_debug_switch_that_is_off_does_not_debug(monkeypatch):
    monkeypatch.delenv("LLM_ROUTER_LOG_LEVEL", raising=False)
    monkeypatch.setenv("LLM_ROUTER_IN_DEBUG", "0")
    assert worker_cli._log_level(None)[0] == logging.INFO


def test_the_router_log_level_is_honoured(monkeypatch):
    monkeypatch.delenv("LLM_ROUTER_IN_DEBUG", raising=False)
    monkeypatch.setenv("LLM_ROUTER_LOG_LEVEL", "warning")
    assert worker_cli._log_level(None) == (logging.WARNING, "LLM_ROUTER_LOG_LEVEL")


def test_no_switch_at_all_means_info(monkeypatch):
    monkeypatch.delenv("LLM_ROUTER_IN_DEBUG", raising=False)
    monkeypatch.delenv("LLM_ROUTER_LOG_LEVEL", raising=False)
    assert worker_cli._log_level(None) == (logging.INFO, "default")


def test_an_unknown_level_name_falls_back_to_info(monkeypatch):
    monkeypatch.delenv("LLM_ROUTER_IN_DEBUG", raising=False)
    monkeypatch.setenv("LLM_ROUTER_LOG_LEVEL", "chatty")
    assert worker_cli._log_level(None)[0] == logging.INFO


def test_startup_names_the_engine_the_stream_and_the_collections():
    """
    A worker that quietly indexes into the wrong place costs more than one that
    refuses to start, so where it writes is on the console before the first
    event.
    """
    recorder = _Recorder()
    config = knowledge_config()
    config.sse_host = "http://engine.test"
    worker_cli._log_startup(recorder, config, logging.DEBUG, "LLM_ROUTER_IN_DEBUG")
    assert "engine=http://engine.test" in recorder.text
    assert config.stream_key in recorder.text
    assert config.collection_prefix in recorder.text
    assert "logging at DEBUG (from LLM_ROUTER_IN_DEBUG)" in recorder.text


def test_startup_reports_a_credential_without_repeating_it():
    recorder = _Recorder()
    config = knowledge_config()
    config.sse_host = "http://engine.test"
    config.sse_token = "a-secret-token"
    config.sse_password = "a-secret-password"
    worker_cli._log_startup(recorder, config, logging.INFO, "default")
    assert "auth=token" in recorder.text
    assert "a-secret-token" not in recorder.text
    assert "a-secret-password" not in recorder.text


def test_a_batch_says_what_it_indexed(caplog):
    event = knowledge_event(kind="recipe", scope="webapp")
    redis = FakeRedis(messages=[("1-0", stream_message(event))])
    work = worker(redis=redis, sse=FakeSSE())
    with caplog.at_level("INFO"):
        assert work.run_once(block_ms=0) == 1
    assert "knowledge batch: 1 read, 1 indexed, 0 dropped" in caplog.text


def test_a_created_collection_is_announced(caplog):
    work = worker(sse=FakeSSE())
    with caplog.at_level("INFO"):
        work._ensure_collection("agentic_kb_webapp")
    assert "knowledge collection created: agentic_kb_webapp" in caplog.text
    assert "embedder=embed" in caplog.text


def test_an_existing_collection_is_not_announced(caplog):
    work = worker(sse=FakeSSE(exists=("agentic_kb_webapp",)))
    with caplog.at_level("INFO"):
        work._ensure_collection("agentic_kb_webapp")
    assert "collection created" not in caplog.text


def test_retention_says_what_it_deleted(caplog):
    config = dataclasses.replace(knowledge_config())
    config.archive_ttl_seconds = 7 * 86400
    sse = FakeSSE()
    work = worker(sse=sse, config=config)
    work._scopes.add("webapp")
    old = (datetime.date.today() - datetime.timedelta(days=30)).strftime("%Y%m%d")
    sse.names["agentic_kb_webapp_sessions"] = [f"session_archive:{old}:old"]
    with caplog.at_level("INFO"):
        assert work.run_retention()["deleted"] == 1
    assert "retention deleted 1 archived document(s)" in caplog.text
    assert "agentic_kb_webapp_sessions" in caplog.text


def test_an_idle_worker_says_it_is_alive(caplog):
    work = worker()
    work._last_heartbeat -= worker_cli.HEARTBEAT_SECONDS + 1
    with caplog.at_level("INFO"):
        work._heartbeat()
    assert "knowledge worker idle" in caplog.text


def test_a_recently_busy_worker_does_not_echo_its_heartbeat(caplog):
    work = worker()
    with caplog.at_level("INFO"):
        work._heartbeat()
    assert "idle" not in caplog.text


def test_indexing_a_document_resets_the_heartbeat_clock():
    event = knowledge_event(kind="recipe", scope="webapp")
    redis = FakeRedis(messages=[("1-0", stream_message(event))])
    work = worker(redis=redis, sse=FakeSSE())
    work._last_heartbeat -= worker_cli.HEARTBEAT_SECONDS + 1
    work.run_once(block_ms=0)
    assert time.monotonic() - work._last_heartbeat < worker_cli.HEARTBEAT_SECONDS


def test_a_failure_names_the_engine_message_and_not_only_its_class(caplog):
    """``RuntimeError`` says a call failed; the message says which and why."""
    event = knowledge_event(kind="recipe", scope="webapp")
    redis = FakeRedis(messages=[("1-0", stream_message(event))])
    sse = FakeSSE(fail_methods=("index_documents",))
    work = worker(redis=redis, sse=sse, max_attempts=1)
    with caplog.at_level("ERROR"):
        work.run_once(block_ms=0)
    assert "boom:index_documents" in caplog.text


def test_describe_folds_whitespace_and_names_the_class():
    described = worker_cli._describe(RuntimeError(" collection\n  rejected "))
    assert described == "RuntimeError: collection rejected"


def test_describe_clips_a_page_of_html_from_a_proxy():
    described = worker_cli._describe(RuntimeError("<html>" + "x" * 5000))
    assert len(described) <= worker_cli._ERROR_DETAIL_CHARS + 20
    assert described.endswith("…")


def test_describe_on_a_messageless_exception_still_names_it():
    assert worker_cli._describe(RuntimeError()) == "RuntimeError"
