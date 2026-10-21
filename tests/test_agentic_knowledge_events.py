"""
Publishing mined episodes to the worker: deduplication, transport, fail-open.

The publisher must append each episode at most once, never block a request on
Redis, and degrade to a status (never an exception) when Redis is missing or
misbehaving.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agentic_knowledge_support import FakeRedis, knowledge_event

from llm_router_plugins.utils.knowledge.agentic_knowledge.events import (
    KnowledgeEventPublisher,
    STATUS_OK,
    STATUS_PARTIAL,
    STATUS_UNAVAILABLE,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.state import (
    RedisConnectionSettings,
)


def publisher(client=None, connection=None, **overrides):
    connection = connection or RedisConnectionSettings(host="cache.test")
    return KnowledgeEventPublisher(
        connection=connection,
        stream_key="kb:events",
        key_prefix="kb",
        client=client,
        **overrides,
    )


def test_publish_appends_event_to_the_stream():
    redis = FakeRedis()
    status = publisher(client=redis).publish([knowledge_event()])
    assert status.status == STATUS_OK
    assert status.published == 1
    assert status.deduplicated == 0
    assert len(redis.stream) == 1


def test_repeated_fingerprint_is_deduplicated_not_republished():
    redis = FakeRedis()
    service = publisher(client=redis)
    first = service.publish([knowledge_event(fingerprint="same")])
    second = service.publish([knowledge_event(fingerprint="same")])
    assert first.published == 1
    assert second.published == 0
    assert second.deduplicated == 1
    assert len(redis.stream) == 1


def test_distinct_fingerprints_are_both_published():
    redis = FakeRedis()
    status = publisher(client=redis).publish(
        [knowledge_event(fingerprint="a"), knowledge_event(fingerprint="b")]
    )
    assert status.published == 2
    assert len(redis.stream) == 2


def test_event_without_fingerprint_is_never_published():
    redis = FakeRedis()
    status = publisher(client=redis).publish([knowledge_event(fingerprint="")])
    assert status.published == 0
    assert status.deduplicated == 1
    assert redis.stream == []


def test_empty_publish_is_ok_and_touches_nothing():
    redis = FakeRedis()
    status = publisher(client=redis).publish([])
    assert status.status == STATUS_OK
    assert redis.stream == []


def test_unconfigured_connection_reports_unavailable():
    status = publisher(connection=RedisConnectionSettings()).publish(
        [knowledge_event()]
    )
    assert status.status == STATUS_UNAVAILABLE
    assert status.published == 0


def test_set_failure_is_an_outage_not_a_duplicate():
    redis = FakeRedis()
    redis.fail_on.add("set")
    status = publisher(client=redis).publish([knowledge_event()])
    # A refused claim is Redis being down, not the episode being known
    # already. Reported as ok, a dead Redis reads as a healthy run whose
    # episodes were all deduplicated — the one outcome an operator cannot
    # tell apart from success.
    assert status.status == STATUS_UNAVAILABLE
    assert status.published == 0
    assert status.deduplicated == 0
    assert status.errors == 1


def test_publish_stops_at_the_first_transport_error():
    class DeadRedis(FakeRedis):
        def __init__(self):
            super().__init__()
            self.set_calls = 0

        def set(self, *args, **kwargs):
            self.set_calls += 1
            raise ConnectionError("redis down")

    redis = DeadRedis()
    events = [knowledge_event(fingerprint=f"f{i}") for i in range(200)]
    status = publisher(client=redis).publish(events)
    # One timeout per request, not one per event: a compaction window can
    # carry hundreds of events and the request path cannot wait on each.
    assert redis.set_calls == 1
    assert status.status == STATUS_UNAVAILABLE
    assert status.errors == 1


def test_partial_publish_is_reported_as_partial():
    class HalfWorking(FakeRedis):
        def xadd(self, *args, **kwargs):
            if len(self.stream) >= 1:
                raise ConnectionError("redis went away")
            return super().xadd(*args, **kwargs)

    redis = HalfWorking()
    events = [knowledge_event(fingerprint=f"p{i}") for i in range(3)]
    status = publisher(client=redis).publish(events)
    assert status.published == 1
    assert status.status == STATUS_PARTIAL
    assert status.errors == 1


def test_append_failure_releases_the_claim_for_a_retry():
    class XaddFailing(FakeRedis):
        def xadd(self, *args, **kwargs):
            raise RuntimeError("xadd boom")

    redis = XaddFailing()
    status = publisher(client=redis).publish([knowledge_event(fingerprint="x")])
    assert status.published == 0
    # The claim was released, so a retry of the same episode can publish again.
    assert "kb:seen:x" not in redis.kv


def test_client_is_built_lazily_from_connection(monkeypatch):
    import types

    built = {}

    class _Client:
        def __init__(self, **kwargs):
            built["kwargs"] = kwargs

        def set(self, *a, **k):
            return True

        def xadd(self, *a, **k):
            return "1-0"

    monkeypatch.setitem(sys.modules, "redis", types.SimpleNamespace(Redis=_Client))
    service = publisher(connection=RedisConnectionSettings(host="cache.test"))
    assert service.available
    status = service.publish([knowledge_event()])
    assert status.status == STATUS_OK
    assert built["kwargs"]["host"] == "cache.test"
