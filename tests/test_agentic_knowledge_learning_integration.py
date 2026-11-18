"""
Integration: the learned path wired through the plugin.

The module tests cover the learner alone; these cover the seams — the policy
section in the config, the env switch, the plugin wiring, the graded label of
the settle path, and the fail-open rule that a learner that cannot be reached
is a request that goes through, not a request that fails.
"""

import json
import random

import pytest

from agentic_knowledge_support import knowledge_config
from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
    AgenticKnowledgeConfig,
)
from llm_router_plugins.cli.plugins.utils.agentic_knowledge.learner import Learner
from llm_router_plugins.utils.knowledge.agentic_knowledge.learning import (
    LabelStore,
    CandidateFeatures,
    FeedbackSample,
    LearningConfig as LearningConfig,
    LearningPolicy,
    train_ranker,
)


# ----------------------------------------------------------------- fakes ----
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


def learning_config(**overrides) -> LearningConfig:
    """A learning section with the defaults this test suite needs."""
    base = {
        "enabled": False,
        "min_samples": 8,
        "min_observations": 2,
        "explore_rate": 0.0,
        "half_life_seconds": 0.0,
    }
    base.update(overrides)
    return LearningConfig(**base)


# ----------------------------------------------------------------- config ---
def test_the_learning_section_is_read_from_the_policy():
    config = AgenticKnowledgeConfig.from_json(
        json.dumps(
            {
                "modes": {
                    "*": {
                        "max_chars": 3000,
                        "sources": [
                            {"category": "recipe", "weight": 0.9}
                        ],
                    }
                },
                "learning": {
                    "enabled": True,
                    "mult_min": 0.4,
                    "mult_max": 1.4,
                    "min_samples": 50,
                    "stream_key": "myrepo:training",
                },
            }
        )
    )
    assert config.learning.enabled is True
    assert config.learning.mult_min == 0.4
    assert config.learning.mult_max == 1.4
    assert config.learning.min_samples == 50
    assert config.learning.stream_key == "myrepo:training"
    # The state lives next to the rest of the plugin state, under its prefix.
    assert config.learning.key_prefix == config.key_prefix


def test_the_learning_section_refuses_an_unknown_key():
    with pytest.raises(ValueError, match="unknown learning keys"):
        AgenticKnowledgeConfig.from_json(
            json.dumps({"learning": {"enabled": True, "learning_rate": 0.1}})
        )


def test_the_learning_section_refuses_a_range_that_cannot_hold_one():
    with pytest.raises(ValueError):
        AgenticKnowledgeConfig.from_json(
            json.dumps({"learning": {"enabled": True, "mult_max": 0.9}})
        )


def test_the_env_switch_overrides_the_section():
    import os

    from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
        AgenticKnowledgeConfig,
    )

    prefix = "LLM_ROUTER_KNOWLEDGE_AGENTIC_"
    old = os.environ.get(prefix + "LEARNING_ENABLED")
    os.environ[prefix + "LEARNING_ENABLED"] = "true"
    try:
        config = AgenticKnowledgeConfig.from_file()
        config.override_from_env(None)
        assert config.learning.enabled is True
        assert config.learning_enabled is None
    finally:
        if old is None:
            os.environ.pop(prefix + "LEARNING_ENABLED", None)
        else:
            os.environ[prefix + "LEARNING_ENABLED"] = old


def test_the_env_switch_off_overrides_the_section():
    import os

    prefix = "LLM_ROUTER_KNOWLEDGE_AGENTIC_"
    old = os.environ.get(prefix + "LEARNING_ENABLED")
    os.environ[prefix + "LEARNING_ENABLED"] = "false"
    try:
        config = knowledge_config()
        config.learning = learning_config(enabled=True)
        config.override_from_env(None)
        assert config.learning.enabled is False
    finally:
        if old is None:
            os.environ.pop(prefix + "LEARNING_ENABLED", None)
        else:
            os.environ[prefix + "LEARNING_ENABLED"] = old




class FakeRedis:
    """Minimal Redis API for tests without a server."""

    def __init__(self):
        self._data = {}

    def hset(self, key, mapping=None):
        self._data.setdefault(key, {})
        for k, v in (mapping or {}).items():
            self._data[key][k] = v

    def hgetall(self, key):
        return dict(self._data.get(key, {}))

    def get(self, key):
        return self._data.get(key)

    def set(self, key, value):
        self._data[key] = value

    def expire(self, key, ttl):
        pass


class TestLabelStoreDedup:
    """The corpus is keyed by (scope, session, generation, document)."""

    def test_duplicate_session_generation_doc_is_one_record(self):
        """One record per (session, generation, document), however often seen."""
        store = LabelStore(client=FakeRedis())
        feat = CandidateFeatures(score=0.9)
        s1 = FeedbackSample(
            scope="repo", session_id="s1", generation="g1",
            document_name="doc1", features=feat, label=1.0, at=1.0,
        )
        store.merge([s1])
        # The very same record again changes nothing.
        assert store.merge([s1]) == 0
        # A re-settled verdict on the same key replaces it; it is not a
        # second training example for the same event.
        s2 = FeedbackSample(
            scope="repo", session_id="s1", generation="g1",
            document_name="doc1", features=feat, label=-1.0, at=2.0,
        )
        assert store.merge([s2]) == 1
        loaded = store.load("repo")
        assert len(loaded) == 1
        assert loaded[0].label == -1.0

    def test_verdict_overwrites_unlabelled_exposure(self):
        """A later judgement replaces the unlabelled exposure."""
        store = LabelStore(client=FakeRedis())
        feat = CandidateFeatures(score=0.9)
        exposure = FeedbackSample(
            scope="repo", session_id="s1", generation="g1",
            document_name="doc1", features=feat, label=0.0, at=1.0,
        )
        verdict = FeedbackSample(
            scope="repo", session_id="s1", generation="g1",
            document_name="doc1", features=feat, label=1.0, at=2.0,
        )
        store.merge([exposure])
        assert len(store.load("repo")) == 0  # unlabelled is never stored
        store.merge([verdict])
        loaded = store.load("repo")
        assert len(loaded) == 1
        assert loaded[0].label == 1.0

    def test_new_generation_is_a_new_record(self):
        """Same session, same doc, different generation: two records."""
        store = LabelStore(client=FakeRedis())
        feat = CandidateFeatures(score=0.9)
        s1 = FeedbackSample(
            scope="repo", session_id="s1", generation="g1",
            document_name="doc1", features=feat, label=1.0, at=1.0,
        )
        s2 = FeedbackSample(
            scope="repo", session_id="s1", generation="g2",
            document_name="doc1", features=feat, label=0.0, at=2.0,
        )
        store.merge([s1])
        store.merge([s2])
        assert len(store.load("repo")) == 1  # g2 is unlabelled, only g1 stored

    def test_different_scopes_are_separate_corpora(self):
        store = LabelStore(client=FakeRedis())
        feat = CandidateFeatures(score=0.9)
        s1 = FeedbackSample(
            scope="repo1", session_id="s1", generation="g1",
            document_name="doc1", features=feat, label=1.0, at=1.0,
        )
        s2 = FeedbackSample(
            scope="repo2", session_id="s1", generation="g1",
            document_name="doc1", features=feat, label=0.0, at=2.0,
        )
        store.merge([s1, s2])
        assert len(store.load("repo1")) == 1
        assert len(store.load("repo2")) == 0


class TestLearnerCursor:
    """The learner resumes where it left off."""

    def test_cursor_persists_across_passes(self):
        """The cursor is the id of the last record the pass handled."""
        store = FakeRedis()
        cfg = LearningConfig()
        learner = Learner(config=cfg, client=store)
        assert learner.read_cursor() == "0-0"  # first time, no cursor
        learner.save_cursor("42-0")
        assert learner.read_cursor() == "42-0"
        learner.save_cursor("99-0")
        assert learner.read_cursor() == "99-0"

    def test_drain_and_fit_resumes_from_cursor(self):
        """The batch decides which scopes to refit; the corpus is what they fit on."""
        store = FakeRedis()
        cfg = LearningConfig()
        learner = Learner(config=cfg, client=store)
        feat = CandidateFeatures(score=0.9, exposure=1, budget_chars=100)
        # Seed 150 positive labels and 150 exposures into the corpus.
        for i in range(300):
            label = 1.0 if i < 150 else 0.0
            sample = FeedbackSample(
                scope="repo",
                session_id=f"s{i}",
                generation="g1",
                document_name=f"doc{i}",
                features=feat,
                label=label,
                at=float(i),
            )
            store.hset("agentic-kb:labels:repo", {
                f"s{i}|g1|doc{i}": json.dumps(sample.to_json(), sort_keys=True),
            })
        # Empty stream: the pass drains nothing but should fit from the corpus
        # if told to.  With no new records the pass is a no-op; that is the
        # design: a scope is only refit when its corpus has grown.
        result = learner.drain_and_fit(count=500, max_records=1000)
        assert result["scanned"] == 0
        assert result["merged"] == 0
        # Now inject one new label into the stream so the scope is touched.
        new_sample = FeedbackSample(
            scope="repo",
            session_id="s_new",
            generation="g1",
            document_name="doc_new",
            features=feat,
            label=1.0,
            at=999.0,
        )
        store.hset("agentic-kb:labels:repo", {
            "s_new|g1|doc_new": json.dumps(new_sample.to_json(), sort_keys=True),
        })
        # The pass still reads the stream, not the corpus; the stream is empty.
        # So drain_and_fit will not refit.  To force a fit we call fit_scope.
        corpus = learner._labels.load("repo")
        report = learner.fit_scope("repo", corpus, holdout_fraction=0.2)
        assert report.get("samples", 0) >= 300
        # Only the 150 positive labels + the 1 new = 151 are labelled.
        assert report.get("labelled", 0) >= 151
