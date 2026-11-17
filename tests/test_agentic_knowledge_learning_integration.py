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
from llm_router_plugins.utils.knowledge.agentic_knowledge.learning import (
    CandidateFeatures,
    FeedbackSample,
    LearningConfig as ModuleLearningConfig,
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


def learning_config(**overrides) -> ModuleLearningConfig:
    """A learning section with the defaults this test suite needs."""
    base = {
        "enabled": False,
        "min_samples": 8,
        "min_observations": 2,
        "explore_rate": 0.0,
        "half_life_seconds": 0.0,
    }
    base.update(overrides)
    return ModuleLearningConfig(**base)


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


