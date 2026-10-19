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
import sys
import datetime
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agentic_knowledge_support import (
    FakeRedis,
    FakeSSE,
    knowledge_config,
    knowledge_event,
    stream_message,
)

from llm_router_plugins.cli.plugins.utils.knowledge.worker import (
    KnowledgeIndexWorker,
    _entries,
    _require_settings,
    _verify_model_names,
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

        def index_documents(self, collection, documents):
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


def test_retention_ignores_foreign_collections_in_the_listing():
    config = knowledge_config()
    config = dataclasses.replace(config)
    config.archive_ttl_seconds = 7 * 86400
    sse = FakeSSE(catalogue=("documents-team-x", "agentic_kb_webapp"))
    work = worker(sse=sse, config=config)
    # Neither is a `-sessions` archive of ours; nothing is listed for deletion.
    assert work.run_retention()["deleted"] == 0
    assert sse.deleted == []
