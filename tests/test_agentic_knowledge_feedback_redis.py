"""
Regression: the injection votes live in Redis, not in one worker.

The tracker was built without a client, so a verdict counted in one process
was invisible to the others and gone on restart.  These tests drive the real
plugin with a fake client and pin that the votes leave the process: a second
turn sees what the first one judged, and a "restart" — a new plugin over the
same client — keeps the reputation it counted.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agentic_knowledge_support import (
    FakePublisher,
    FakeStore,
    knowledge_config,
    main_payload,
    store_hit,
)

from llm_router_plugins.utils.knowledge.agentic_knowledge import (
    AgenticKnowledgePlugin,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.cache import (
    InMemoryContextCache,
)


class FakeRedis:
    """The handful of commands the tracker and the cache use, in memory."""

    def __init__(self) -> None:
        self.hashes = {}
        self.values = {}
        self.ttls = {}
        self.calls = []

    def hincrby(self, key, field, amount=1):
        self.calls.append(("hincrby", key, field, amount))
        current = int(self.hashes.setdefault(key, {}).get(field, "0"))
        self.hashes[key][field] = str(current + amount)
        return current + amount

    def hgetall(self, key):
        self.calls.append(("hgetall", key))
        return dict(self.hashes.get(key, {}))

    def hget(self, key, *fields):
        self.calls.append(("hget", key))
        data = self.hashes.get(key, {})
        if len(fields) == 1:
            return data.get(fields[0])
        return [data.get(f) for f in fields]

    def get(self, key):
        self.calls.append(("get", key))
        return self.values.get(key)

    def set(self, key, value, **kwargs):
        self.calls.append(("set", key))
        self.values[key] = value
        return True

    def setex(self, key, ttl, value):
        self.ttls[key] = ttl
        self.values[key] = value
        return True

    def expire(self, key, seconds):
        self.ttls[key] = int(seconds)
        return True


class BrokenRedis(FakeRedis):
    """Every command fails; the layer must not notice."""

    def __getattr__(self, name):
        def _raise(*args, **kwargs):
            raise ConnectionError(f"{name} unavailable")

        return _raise


def make_plugin(client, redis_host="cache.test"):
    """A real plugin over a fake engine, with the given client."""
    from dataclasses import replace

    config = knowledge_config()
    config.redis = replace(config.redis, host=redis_host)
    return AgenticKnowledgePlugin(
        config=config,
        store=FakeStore(hits=(store_hit("recipe", score=0.9),)),
        cache=InMemoryContextCache(),
        publisher=FakePublisher(),
        redis_client=client,
    )


def test_votes_reach_redis_through_the_plugin():
    """Two turns of the real plugin: the second one's downvote is in the client."""
    client = FakeRedis()
    plugin = make_plugin(client)
    plugin.apply(main_payload(turn_id="turn-1", root_turn_id="gen-1"))
    plugin.apply(main_payload(turn_id="turn-2", root_turn_id="gen-2"))
    # The verdict of turn one was judged on turn two: a failure came back,
    # so the document it was about took a downvote, in the shared store.
    feedback_keys = [call[1] for call in client.calls if call[0] == "hincrby"]
    assert feedback_keys, "no vote reached the shared store"
    assert all(key.startswith("agentic-kb:fb:") for key in feedback_keys)
    for key in feedback_keys:
        assert int(client.hashes[key].get("downvotes", "0")) >= 1


def test_a_restart_keeps_the_votes_the_previous_process_counted():
    """A new plugin over the same client reads the reputation the old one kept."""
    client = FakeRedis()
    plugin = make_plugin(client)
    plugin.apply(main_payload(turn_id="turn-1", root_turn_id="gen-1"))
    plugin.apply(main_payload(turn_id="turn-2", root_turn_id="gen-2"))
    # The document this block injected: the one the next turn judged.
    names = sorted(
        key.split("agentic-kb:fb:")[-1]
        for key in client.hashes
        if key.startswith("agentic-kb:fb:")
        and int(client.hashes[key].get("downvotes", "0")) >= 1
    )
    assert names, "the shared store never received a verdict"
    # The restart: a fresh plugin, the same client, the same repository.
    restarted = make_plugin(client)
    for name in names:
        stats = restarted._feedback_tracker.get_feedback(name)
        assert int(stats.get("downvotes", 0)) >= 1, (
            "a restart lost the reputation of %s" % name
        )


def test_a_deployed_redis_down_is_a_memory_only_layer():
    """A client whose commands all fail never costs a request."""
    plugin = make_plugin(BrokenRedis())
    plugin.apply(main_payload(turn_id="turn-1", root_turn_id="gen-1"))
    # …and the next turn still settles, against the state this process kept.
    plugin.apply(main_payload(turn_id="turn-2", root_turn_id="gen-2"))
    assert plugin._feedback_tracker.all_feedback()
