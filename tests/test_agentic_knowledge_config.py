"""
Configuration: policy loading, validation and readiness.

Connections come from the environment only; the JSON holds policy (modes,
weights, budgets, escalation models).  A config is not "available" until the
engine (read) or the stream (write) is actually configured, and an invalid
policy raises rather than silently falling back.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

from agentic_knowledge_support import knowledge_config

from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
    AGENTIC_KNOWLEDGE_ENV_PREFIX,
    AgenticKnowledgeConfig,
    ModePolicy,
    SourcePolicy,
)


def test_bundled_policy_loads_with_six_modes():
    config = AgenticKnowledgeConfig.from_file()
    assert set(config.modes) == {
        "debug",
        "test",
        "implement",
        "plan",
        "git_review",
        "review",
    }
    assert config.escalation_enabled is False


def test_policy_for_returns_the_mode_or_none():
    config = knowledge_config()
    assert config.policy_for("debug") is not None
    assert config.policy_for("does-not-exist") is None


def test_collection_names_separate_the_session_archive():
    config = knowledge_config()
    assert config.collection("webapp", "recipe") == "agentic-kb-webapp"
    assert (
        config.collection("webapp", "session_archive")
        == "agentic-kb-webapp-sessions"
    )


def test_read_is_available_only_with_an_engine():
    config = knowledge_config()
    assert config.read_available
    no_engine = AgenticKnowledgeConfig.from_file()
    assert not no_engine.read_available


def test_write_is_available_only_with_a_stream():
    config = knowledge_config()
    assert config.write_available
    from llm_router_plugins.utils.routing.agentic_routing.codex.state import (
        RedisConnectionSettings,
    )

    no_redis = knowledge_config()
    no_redis.redis = RedisConnectionSettings()
    assert not no_redis.write_available
    assert not no_redis.archive_available


def test_unknown_top_level_key_is_rejected():
    with pytest.raises(ValueError):
        AgenticKnowledgeConfig.from_json('{"bogus": 1}')


def test_unknown_adapter_is_rejected():
    with pytest.raises(ValueError):
        AgenticKnowledgeConfig.from_json('{"adapters": ["codex", "nope"]}')


def test_out_of_range_similarity_is_rejected():
    with pytest.raises(ValueError):
        AgenticKnowledgeConfig.from_json('{"read": {"min_similarity": 2.0}}')


def test_a_source_must_allow_at_least_one_item():
    with pytest.raises(ValueError):
        AgenticKnowledgeConfig.from_json(
            '{"modes": {"debug": {"sources": '
            '[{"category": "recipe", "max_items": 0}]}}}'
        )


def test_mode_with_too_small_budget_is_rejected():
    with pytest.raises(ValueError):
        AgenticKnowledgeConfig.from_json(
            '{"modes": {"debug": {"max_chars": 100, "sources": '
            '[{"category": "recipe"}]}}}'
        )


def test_escalation_thresholds_are_validated():
    with pytest.raises(ValueError):
        AgenticKnowledgeConfig.from_json(
            '{"escalation": {"strong_recipe_score": 0.0}}'
        )


def test_env_prefix_is_stable():
    assert AGENTIC_KNOWLEDGE_ENV_PREFIX == "LLM_ROUTER_KNOWLEDGE_AGENTIC_"


def test_mode_and_source_policies_are_frozen():
    policy = ModePolicy(name="debug", sources=(SourcePolicy("recipe", 1.0, 3),))
    with pytest.raises(Exception):
        policy.max_chars = 1  # type: ignore[misc]
