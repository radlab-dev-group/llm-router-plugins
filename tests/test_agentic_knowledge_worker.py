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
)


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
    assert collection == "agentic-kb-webapp"
    assert documents[0]["category"] == "recipe"


def test_session_archive_goes_to_the_sessions_collection():
    _processed, _redis, sse = run_event(
        knowledge_event(kind="session_archive", scope="webapp")
    )
    assert sse.indexed[0][0] == "agentic-kb-webapp-sessions"


def test_document_name_carries_kind_date_and_fingerprint():
    _processed, _redis, sse = run_event(
        knowledge_event(kind="recipe", fingerprint="deadbeef", scope="webapp")
    )
    name = sse.indexed[0][1][0]["filepath"]
    assert name.startswith("recipe:")
    assert name.endswith(":deadbeef")
    day = datetime.date.today().strftime("%Y%m%d")
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


def test_collection_is_created_with_embedder_and_reranker():
    _processed, _redis, sse = run_event(knowledge_event())
    assert len(sse.created) == 1
    _name, embedder, reranker = sse.created[0]
    assert (embedder, reranker) == ("embed", "rerank")


def test_existing_collection_is_not_recreated():
    sse = FakeSSE(exists=("agentic-kb-webapp",))
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
    # First attempt fails and leaves the message pending (not acked).
    redis.messages = [("1-0", stream_message(knowledge_event()))]
    work._process_entry(
        "1-0", stream_message(knowledge_event()) | {"delivery_count": 1}
    )
    assert redis.acked == []
    # Second (max) attempt drops it with an ack.
    work._process_entry(
        "1-0", stream_message(knowledge_event()) | {"delivery_count": 2}
    )
    assert redis.acked == ["1-0"]
    assert sse.calls == 2


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
    sse.names["agentic-kb-webapp-sessions"] = [
        f"session_archive:{old}:old",
        f"session_archive:{recent}:recent",
        "a-human-document.md",
    ]
    result = work.run_retention()
    assert result["deleted"] == 1
    assert sse.deleted == [
        ("agentic-kb-webapp-sessions", [f"session_archive:{old}:old"])
    ]


def test_retention_never_touches_documents_outside_the_plugin_shape():
    config = knowledge_config()
    config = dataclasses.replace(config)
    config.archive_ttl_seconds = 7 * 86400
    sse = FakeSSE()
    work = worker(sse=sse, config=config)
    work._scopes.add("webapp")
    sse.names["agentic-kb-webapp-sessions"] = ["notes.md", "20160101:legacy:doc"]
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
