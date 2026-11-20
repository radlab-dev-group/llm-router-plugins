"""
The learned path driven through the real plugin.

The policy tests cover the learner alone; these cover the seams the request
path crosses: the exposure a read writes to the training stream, the graded
label the settle path hands the learner instead of the coarse bit, the
annotation that names the model which served, and the fail-open rule that a
learner that cannot be reached is a request that goes through.
"""

import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agentic_knowledge_support import (
    FakePublisher,
    FakeStore,
    knowledge_config,
    main_payload,
    recipe_payload,
    store_hit,
)

from llm_router_plugins.utils.knowledge.agentic_knowledge import (
    AgenticKnowledgePlugin,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.cache import (
    InMemoryContextCache,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.learning import (
    LearningConfig,
)


class FakeRedis:
    """The handful of commands the learner uses, in memory."""

    def __init__(self) -> None:
        self.hashes = {}
        self.values = {}
        self.streams = {}
        self.ttls = {}
        self.calls = []

    def hset(self, key, field, value):
        self.calls.append(("hset", key, field))
        self.hashes.setdefault(key, {})[field] = str(value)
        return 1

    def hgetall(self, key):
        self.calls.append(("hgetall", key))
        return dict(self.hashes.get(key, {}))

    def hincrby(self, key, field, amount=1):
        self.calls.append(("hincrby", key, field, amount))
        current = int(self.hashes.setdefault(key, {}).get(field, "0"))
        self.hashes[key][field] = str(current + amount)
        return current + amount

    def expire(self, key, seconds):
        self.ttls[key] = int(seconds)
        return True

    def get(self, key):
        self.calls.append(("get", key))
        return self.values.get(key)

    def set(self, key, value):
        self.calls.append(("set", key))
        self.values[key] = value
        return True

    def xadd(self, key, fields, maxlen=0):
        self.calls.append(("xadd", key))
        stream = self.streams.setdefault(key, [])
        stream.append((f"{len(stream) + 1}-0", dict(fields)))
        if maxlen:
            del stream[: max(0, len(stream) - maxlen)]
        return stream[-1][0]

    def xrange(self, key, min="-", max="+", count=0):
        rows = list(self.streams.get(key, []))
        return rows[:count] if count else rows


class BrokenRedis(FakeRedis):
    """Every command fails; the layer must not notice."""

    def __getattr__(self, name):
        def _raise(*args, **kwargs):
            raise ConnectionError(f"{name} unavailable")

        return _raise


def make_learning_plugin(client, enabled, hits=()):
    """A real plugin over a fake engine, with the learner on or off."""
    config = knowledge_config()
    config.learning = LearningConfig(
        enabled=enabled,
        min_samples=1000,
        min_observations=1000,
        explore_rate=0.0,
        half_life_seconds=0.0,
    )
    return AgenticKnowledgePlugin(
        config=config,
        store=FakeStore(hits=tuple(hits)),
        cache=InMemoryContextCache(),
        publisher=FakePublisher(),
        redis_client=client,
    )


def test_an_injection_writes_its_features_to_the_training_stream():
    """The exposure is logged before the verdict, and with learning off."""
    client = FakeRedis()
    plugin = make_learning_plugin(
        client, enabled=False, hits=(store_hit("recipe", score=0.9),)
    )
    plugin.apply(main_payload(turn_id="turn-1", root_turn_id="gen-1"))
    rows = client.streams.get("agentic_kb:training") or []
    assert rows, "an injection wrote no training record"
    record = json.loads(rows[0][1]["d"])
    assert record["kind"] == "injection"
    assert record["label"] == 0.0
    assert record["scope"] == "webapp"
    assert record["doc"]
    # The record carries features, never content.
    for banned in ("def ", "src/", "/home/", "traceback", "assert"):
        assert banned not in json.dumps(record)


def test_a_repeated_failure_blames_the_recipe_the_learner_saw():
    """The graded label of the settle path reaches the learner, not only the counter."""
    client = FakeRedis()
    plugin = make_learning_plugin(
        client, enabled=True, hits=(store_hit("recipe", score=0.9),)
    )
    plugin.apply(main_payload(turn_id="turn-1", root_turn_id="gen-1"))
    doc = plugin._pending_injections["sess-1"][0].document_name
    # The same failure comes back: the injected recipe did not resolve it.
    # Turn two re-injects the block, so the pending entry is the new one —
    # the verdict it settled is already folded into the reputation.
    plugin.apply(main_payload(turn_id="turn-2", root_turn_id="gen-2"))
    stats = plugin._learning.reputation.stats(doc)
    # The posterior moved against the document: a full judgement of failure.
    assert stats["beta"] > 2.0
    assert stats["exposure"] >= 1
    # …and the same verdict is in the stream, graded per document.
    rows = [
        json.loads(payload["d"])
        for _, payload in client.streams.get("agentic_kb:training", [])
    ]
    graded = [row for row in rows if row["label"] < 0]
    assert graded, "no graded label reached the stream"
    assert graded[0]["doc"] == doc
    # The fake hit carries no signature and no paths, so the verdict could
    # only be the block's, marked coarse: the credit is collective, and the
    # model is told so.
    assert graded[0]["kind"] == "coarse"


def test_a_verified_fix_credits_the_recipe_it_names():
    """A turn that mines a recipe credits the document it was shown."""
    client = FakeRedis()
    plugin = make_learning_plugin(
        client, enabled=True, hits=(store_hit("recipe", score=0.9),)
    )
    plugin.apply(main_payload(turn_id="turn-1", root_turn_id="gen-1"))
    doc = plugin._pending_injections["sess-1"][0].document_name
    # The next turn of this session fixes the failure and mines a recipe.
    plugin.apply(recipe_payload(turn_id="turn-2", root_turn_id="gen-2"))
    stats = plugin._learning.reputation.stats(doc)
    assert stats["alpha"] > 2.0
    rows = [
        json.loads(payload["d"])
        for _, payload in client.streams.get("agentic_kb:training", [])
    ]
    credited = [row for row in rows if row["label"] > 0]
    assert credited, "the fix mined this turn credited nothing"
    assert credited[0]["doc"] == doc
    assert credited[0]["kind"] in ("confirmed", "plausible", "coarse")


def test_the_annotation_names_the_model_that_served_the_request():
    client = FakeRedis()
    plugin = make_learning_plugin(
        client, enabled=True, hits=(store_hit("recipe", score=0.9),)
    )
    payload = plugin.apply(main_payload(turn_id="turn-1", root_turn_id="gen-1"))
    learning = payload["routing"]["knowledge"]["learning"]
    assert learning["enabled"] is True
    assert learning["range"] == [0.3, 1.5]
    # Nothing was fitted yet, so the null model serves and says so.
    assert learning["model_id"] == ""
    assert learning["labelled"] == 0
    feedback = payload["routing"]["knowledge"]["metrics"]["feedback"]
    assert feedback["source"] == "redis"


def test_the_annotation_names_the_multiplier_of_each_candidate():
    """Why one recipe ranked above a stronger one is answerable per candidate.

    The request annotation carries the learner's account of each hit it
    weighed: the multiplier it applied, the model it read, and whether the
    read was an exploration.  A candidate the learner did not touch — no
    learner, no scope, or a policy off — carries no entry, and a disabled
    policy carries none for every candidate, because it has no vote to
    explain.
    """
    import random

    from llm_router_plugins.utils.knowledge.agentic_knowledge.learning import (
        publish_snapshot,
        train_ranker,
    )
    from test_agentic_knowledge_learning import corpus

    client = FakeRedis()
    plugin = make_learning_plugin(
        client, enabled=True, hits=(store_hit("recipe", score=0.9),)
    )
    # Fit and publish a model with real signal, so the annotation has a
    # model to name.
    snapshot, _ = train_ranker(
        corpus(300, signal=True, seed=11),
        "webapp",
        config=plugin._config.learning,
        rng=random.Random(11),
    )
    publish_snapshot(client, snapshot, key_prefix=plugin._config.learning.key_prefix)
    payload = plugin.apply(main_payload(turn_id="turn-1", root_turn_id="gen-1"))
    learning = payload["routing"]["knowledge"]["learning"]
    assert learning["model_id"] == snapshot.model_id
    assert learning["labelled"] == 300
    # The hits this block injected are weighed by the learner, and the
    # annotation says so per candidate: the multiplier the vote carried, the
    # model that produced it, and whether the read was an exploration.
    hits = {
        name: detail for name, detail in (learning.get("candidates") or {}).items()
    }
    assert hits, "the candidate the learner weighed is not named in the annotation"
    for name, detail in hits.items():
        assert name.startswith("recipe:")
        # The multiplier the candidate's vote actually carries, within the
        # policy's range, with the reasons for it.
        assert 0.3 <= float(detail["mult"]) <= 1.5
        assert detail["reason"] in ("learned", "reputation")
        assert detail["model_id"] == snapshot.model_id
        assert "explore" in detail


def test_a_disabled_learning_section_is_the_layer_it_always_was():
    client = FakeRedis()
    plugin = make_learning_plugin(
        client, enabled=False, hits=(store_hit("recipe", score=0.9),)
    )
    payload = plugin.apply(main_payload(turn_id="turn-1", root_turn_id="gen-1"))
    learning = payload["routing"]["knowledge"]["learning"]
    assert learning["enabled"] is False
    # Nothing was fitted and nothing moves: the annotation names the null
    # model, and the request path did not read a snapshot to rank the block.
    assert learning["model_id"] == ""
    assert not any(
        call[0] == "get" and call[1].startswith("agentic-kb:model:")
        for call in client.calls
    )


def test_a_broken_redis_does_not_cost_a_request():
    """Every learned read is fail-open: the request is answered, state falls back."""
    plugin = make_learning_plugin(
        BrokenRedis(), enabled=True, hits=(store_hit("recipe", score=0.9),)
    )
    payload = plugin.apply(main_payload(turn_id="turn-1", root_turn_id="gen-1"))
    assert payload["routing"]["knowledge"]["learning"]["enabled"] is True
    # …and the next turn settles against the same in-process state.
    plugin.apply(main_payload(turn_id="turn-2", root_turn_id="gen-2"))
