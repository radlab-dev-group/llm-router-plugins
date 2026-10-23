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
    WorkerSettings,
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
    assert config.collection("webapp", "recipe") == "agentic_kb_webapp"
    assert (
        config.collection("webapp", "session_archive")
        == "agentic_kb_webapp_sessions"
    )


def test_collection_names_separate_the_documentation():
    """Documentation lives apart from mined knowledge.

    One search request carries one metadata dimension, so a mode that reads
    only the guides of a repository can restrict by doc_type in one request
    only when nothing else shares the request; giving the documentation its own
    collection is what buys that, and it re-indexes as a set besides.
    """
    config = knowledge_config()
    assert config.collection("webapp", "doc") == "agentic_kb_webapp_docs"
    assert config.collection("", "doc") == "agentic_kb_docs"
    assert config.collection("web-app", "doc") == "agentic_kb_web_app_docs"


def test_a_doc_source_can_list_the_kinds_it_reads():
    config = AgenticKnowledgeConfig.from_json(
        '{"modes": {"implement": {"sources": '
        '[{"category": "doc", "doc_types": ["guide", "readme"]}]}}}'
    )
    source = config.modes["implement"].sources[0]
    assert source.doc_types == ("guide", "readme")


def test_a_source_without_doc_types_reads_every_kind():
    config = AgenticKnowledgeConfig.from_json(
        '{"modes": {"implement": {"sources": [{"category": "doc"}]}}}'
    )
    assert config.modes["implement"].sources[0].doc_types == ()


def test_only_the_doc_source_may_list_doc_types():
    """A mined episode carries no doc_type; the filter would read as empty."""
    with pytest.raises(ValueError, match="doc_types"):
        AgenticKnowledgeConfig.from_json(
            '{"modes": {"debug": {"sources": '
            '[{"category": "recipe", "doc_types": ["guide"]}]}}}'
        )


def test_the_doc_source_cannot_restrict_phases_or_modes():
    """Documentation is searched by kind, in its own request."""
    with pytest.raises(ValueError, match="doc"):
        AgenticKnowledgeConfig.from_json(
            '{"modes": {"plan": {"sources": '
            '[{"category": "doc", "phases": ["test"]}]}}}'
        )
    with pytest.raises(ValueError, match="doc"):
        AgenticKnowledgeConfig.from_json(
            '{"modes": {"plan": {"sources": '
            '[{"category": "doc", "modes": ["debug"]}]}}}'
        )


def test_documentation_does_not_share_the_phase_dimension():
    """The doc source is its own request, so it cannot collide with phases."""
    config = AgenticKnowledgeConfig.from_json(
        '{"modes": {"debug": {"sources": '
        '[{"category": "recipe", "phases": ["test"]}, '
        '{"category": "doc", "doc_types": ["guide"]}]}}}'
    )
    assert len(config.modes["debug"].sources) == 2


def test_the_bundled_modes_restrict_documentation_per_work():
    """What each work reads of the documentation is a decision, not a default."""
    config = AgenticKnowledgeConfig.from_file()
    types = {
        mode: tuple(source.doc_types)
        for mode, policy in config.modes.items()
        for source in policy.sources
        if source.category == "doc"
    }
    assert types["implement"] == ("readme", "guide", "contributing")
    assert "changelog" not in types["implement"]
    assert "changelog" in types["plan"]
    assert types["review"] == ("guide", "contributing", "security")
    assert all("license" not in value for value in types.values())


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


def test_an_empty_env_variable_is_validated_not_ignored(monkeypatch):
    """
    `VAR=` is an override attempt, not a variable that was never set.

    Silently ignoring it is how a deployment ends up with the default budget
    after an operator meant to change it and typed the value into the wrong
    line of the unit file.
    """
    monkeypatch.setenv(f"{AGENTIC_KNOWLEDGE_ENV_PREFIX}MIN_SIMILARITY", "")
    config = AgenticKnowledgeConfig.from_file()
    with pytest.raises(ValueError):
        config.override_from_env()


def test_a_malformed_env_number_names_the_variable(monkeypatch):
    monkeypatch.setenv(f"{AGENTIC_KNOWLEDGE_ENV_PREFIX}LOOP_THRESHOLD", "three")
    config = AgenticKnowledgeConfig.from_file()
    with pytest.raises(ValueError, match="LOOP_THRESHOLD"):
        config.override_from_env()


def test_an_env_collection_prefix_is_cut_like_the_json_one(monkeypatch):
    """The environment must not be a way to reach the collection names raw."""
    monkeypatch.setenv(f"{AGENTIC_KNOWLEDGE_ENV_PREFIX}COLLECTION_PREFIX", "--foo")
    config = AgenticKnowledgeConfig.from_file()
    config.override_from_env()
    assert config.collection_prefix == "foo"


def test_a_collection_prefix_with_hyphen_becomes_engine_safe(monkeypatch):
    """A hyphen in the prefix is a 500 at collection creation on the engine."""
    monkeypatch.setenv(
        f"{AGENTIC_KNOWLEDGE_ENV_PREFIX}COLLECTION_PREFIX", "agentic-kb"
    )
    config = AgenticKnowledgeConfig.from_file()
    config.override_from_env()
    assert config.collection_prefix == "agentic_kb"


def test_a_hyphenated_repo_scope_gets_an_engine_safe_collection(monkeypatch):
    """
    The scope is a directory name; the engine rejects a hyphen in a collection
    name.  Without this, every hyphenated repository (`llm-router`) could never
    have a collection created, so nothing would ever be indexed.
    """
    config = AgenticKnowledgeConfig.from_file()
    assert (
        config.collection("llm-router-plugins", "recipe")
        == "agentic_kb_llm_router_plugins"
    )
    assert (
        config.collection("kb-probe", "session_archive")
        == "agentic_kb_kb_probe_sessions"
    )
    for scope in ("kb-probe", "llm-router", "a--b"):
        name = config.collection(scope, "recipe")
        assert "-" not in name, f"the engine would 500 on {name!r}"
        assert not name[0].isdigit()


def test_two_scopes_differing_only_by_punctuation_share_a_collection():
    """A hyphen and an underscore name the same repository, for a name."""
    config = AgenticKnowledgeConfig.from_file()
    assert config.collection("my-repo", "recipe") == config.collection(
        "my_repo", "recipe"
    )


def test_the_default_collection_prefix_is_engine_safe():
    config = AgenticKnowledgeConfig.from_file()
    for character in config.collection_prefix:
        assert (
            character.isalnum() or character in "._"
        ), f"the engine would refuse the default prefix {character!r}"
    assert not config.collection_prefix[0].isdigit()


def test_engine_credentials_come_from_the_environment(monkeypatch):
    """A login is an alternative to a token, not a third secret scheme."""
    monkeypatch.setenv(
        f"{AGENTIC_KNOWLEDGE_ENV_PREFIX}SSE_API_USERNAME", "default_user"
    )
    monkeypatch.setenv(
        f"{AGENTIC_KNOWLEDGE_ENV_PREFIX}SSE_API_PASSWORD", "top-secret"
    )
    config = AgenticKnowledgeConfig.from_file()
    config.override_from_env()
    assert config.sse_username == "default_user"
    assert config.sse_password == "top-secret"
    assert config.sse_token == ""


def test_engine_credentials_default_to_empty(monkeypatch):
    monkeypatch.delenv(
        f"{AGENTIC_KNOWLEDGE_ENV_PREFIX}SSE_API_USERNAME", raising=False
    )
    monkeypatch.delenv(
        f"{AGENTIC_KNOWLEDGE_ENV_PREFIX}SSE_API_PASSWORD", raising=False
    )
    config = AgenticKnowledgeConfig.from_file()
    config.override_from_env()
    assert config.sse_username == ""
    assert config.sse_password == ""


def test_a_malformed_redis_connection_is_reported(monkeypatch):
    """Redis off is survivable, but the operator has to be able to see why."""
    monkeypatch.setenv(f"{AGENTIC_KNOWLEDGE_ENV_PREFIX}REDIS_PORT", "not-a-port")
    config = AgenticKnowledgeConfig.from_file()
    config.override_from_env()
    assert config.redis.configured is False
    assert config.redis_error


def test_a_working_redis_connection_leaves_no_error(monkeypatch):
    monkeypatch.setenv(f"{AGENTIC_KNOWLEDGE_ENV_PREFIX}REDIS_HOST", "redis.internal")
    config = AgenticKnowledgeConfig.from_file()
    config.override_from_env()
    assert config.redis.configured is True
    assert config.redis_error == ""


def test_mode_and_source_policies_are_frozen():
    policy = ModePolicy(name="debug", sources=(SourcePolicy("recipe", 1.0, 3),))
    with pytest.raises(Exception):
        policy.max_chars = 1  # type: ignore[misc]


def test_a_source_can_restrict_its_phases():
    config = AgenticKnowledgeConfig.from_json(
        '{"modes": {"debug": {"sources": '
        '[{"category": "recipe", "phases": ["test", "debug"]}]}}}'
    )
    source = config.modes["debug"].sources[0]
    assert source.phases == ("test", "debug")


def test_a_source_without_phases_reads_every_phase():
    config = AgenticKnowledgeConfig.from_json(
        '{"modes": {"debug": {"sources": [{"category": "recipe"}]}}}'
    )
    assert config.modes["debug"].sources[0].phases == ()


def test_a_source_can_restrict_its_modes():
    config = AgenticKnowledgeConfig.from_json(
        '{"modes": {"implement": {"sources": '
        '[{"category": "recipe", "modes": ["debug"]}]}}}'
    )
    source = config.modes["implement"].sources[0]
    assert source.modes == ("debug",)


def test_an_archive_source_cannot_restrict_phases_or_modes():
    """The thread filter owns the metadata slot; a second would widen it."""
    with pytest.raises(ValueError, match="session_archive"):
        AgenticKnowledgeConfig.from_json(
            '{"modes": {"debug": {"sources": '
            '[{"category": "session_archive", "phases": ["test"]}]}}}'
        )
    with pytest.raises(ValueError, match="session_archive"):
        AgenticKnowledgeConfig.from_json(
            '{"modes": {"debug": {"sources": '
            '[{"category": "session_archive", "modes": ["debug"]}]}}}'
        )


def test_phases_and_modes_cannot_share_a_request():
    """One metadata dimension per request; the engine unions, not intersects."""
    with pytest.raises(ValueError, match="phases and modes"):
        AgenticKnowledgeConfig.from_json(
            '{"modes": {"debug": {"sources": '
            '[{"category": "recipe", "phases": ["test"]}, '
            '{"category": "deadend", "modes": ["debug"]}]}}}'
        )


def test_phases_must_be_a_list_of_names():
    with pytest.raises(ValueError):
        AgenticKnowledgeConfig.from_json(
            '{"modes": {"debug": {"sources": '
            '[{"category": "recipe", "phases": "test"}]}}}'
        )
    with pytest.raises(ValueError):
        AgenticKnowledgeConfig.from_json(
            '{"modes": {"debug": {"sources": '
            '[{"category": "recipe", "phases": [""]}]}}}'
        )


def test_the_retrieval_legs_are_hybrid_without_a_reranker():
    """The engine fuses both legs; the cross-encoder costs a model call."""
    config = AgenticKnowledgeConfig.from_file()
    assert config.hybrid_search is True
    assert config.rerank_results is False


def test_both_retrieval_legs_are_written_in_the_bundled_policy():
    """A switch an operator cannot see in the policy is a switch they lack."""
    import json
    import pathlib

    path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "llm_router_plugins/resources/knowledge/agentic_knowledge.json"
    )
    read = json.loads(path.read_text(encoding="utf-8"))["read"]
    assert read["hybrid_search"] is True
    assert read["rerank_results"] is False


@pytest.mark.parametrize(
    ("hybrid", "rerank"),
    [("false", "true"), ("true", "true"), ("false", "false")],
)
def test_the_retrieval_legs_are_switchable_from_the_environment(
    monkeypatch, hybrid, rerank
):
    monkeypatch.setenv(f"{AGENTIC_KNOWLEDGE_ENV_PREFIX}HYBRID_SEARCH", hybrid)
    monkeypatch.setenv(f"{AGENTIC_KNOWLEDGE_ENV_PREFIX}RERANK_RESULTS", rerank)
    config = AgenticKnowledgeConfig.from_file()
    config.override_from_env()
    assert config.hybrid_search is (hybrid == "true")
    assert config.rerank_results is (rerank == "true")


def test_the_bundled_policy_can_ask_for_a_reranker():
    monkeypatch_env = True
    raw = '{"read": {"rerank_results": true, "hybrid_search": false}}'
    import os

    os.environ[f"{AGENTIC_KNOWLEDGE_ENV_PREFIX}CONFIG"] = raw
    try:
        config = AgenticKnowledgeConfig.from_file()
        assert config.hybrid_search is False
        assert config.rerank_results is True
    finally:
        del os.environ[f"{AGENTIC_KNOWLEDGE_ENV_PREFIX}CONFIG"]


def test_a_malformed_retrieval_switch_is_an_error(monkeypatch):
    monkeypatch.setenv(f"{AGENTIC_KNOWLEDGE_ENV_PREFIX}RERANK_RESULTS", "sometimes")
    config = AgenticKnowledgeConfig.from_file()
    with pytest.raises(ValueError, match="RERANK_RESULTS"):
        config.override_from_env()


def test_the_worker_declares_a_chunking_by_default():
    """400 tokens with a 60 overlap, under the 508 the embedder cuts at."""
    worker = AgenticKnowledgeConfig.from_file().worker
    assert worker.chunk_tokens == 400
    assert worker.chunk_overlap == 60


def test_the_chunking_is_switchable_from_the_environment(monkeypatch):
    monkeypatch.setenv(f"{AGENTIC_KNOWLEDGE_ENV_PREFIX}SSE_CHUNK_TOKENS", "900")
    monkeypatch.setenv(f"{AGENTIC_KNOWLEDGE_ENV_PREFIX}SSE_CHUNK_OVERLAP", "120")
    config = AgenticKnowledgeConfig.from_file()
    config.override_from_env()
    assert (config.worker.chunk_tokens, config.worker.chunk_overlap) == (900, 120)


def test_zero_chunk_leaves_the_split_to_the_engine(monkeypatch):
    monkeypatch.setenv(f"{AGENTIC_KNOWLEDGE_ENV_PREFIX}SSE_CHUNK_TOKENS", "0")
    config = AgenticKnowledgeConfig.from_file()
    config.override_from_env()
    assert config.worker.chunk_tokens == 0


def test_an_overlap_that_eats_the_chunk_is_an_error(monkeypatch):
    monkeypatch.setenv(f"{AGENTIC_KNOWLEDGE_ENV_PREFIX}SSE_CHUNK_TOKENS", "300")
    monkeypatch.setenv(f"{AGENTIC_KNOWLEDGE_ENV_PREFIX}SSE_CHUNK_OVERLAP", "300")
    config = AgenticKnowledgeConfig.from_file()
    with pytest.raises(ValueError, match="SSE_CHUNK_OVERLAP"):
        config.override_from_env()


# ------------------------------------------------------------- documentation -
def test_documentation_is_on_by_default_with_bounded_scans():
    config = AgenticKnowledgeConfig.from_file()
    assert config.docs_enabled is True
    assert config.docs_auto_index is True
    assert config.docs_patterns == ("*.md", "*.markdown", "*.mdx")
    assert "node_modules" in config.docs_excluded_dirs
    assert config.docs_max_files > 0
    assert config.docs_max_file_bytes >= 1024
    assert config.docs_max_total_bytes >= config.docs_max_file_bytes
    assert config.docs_rescan_interval_seconds > 0
    assert config.docs_state_ttl_seconds > 0


def test_the_docs_section_of_the_policy_is_loaded():
    config = AgenticKnowledgeConfig.from_json(
        '{"docs": {"enabled": false, "auto_index": false, "max_files": 5}}'
    )
    assert config.docs_enabled is False
    assert config.docs_auto_index is False
    assert config.docs_max_files == 5


def test_documentation_settings_come_from_the_environment(monkeypatch):
    monkeypatch.setenv(f"{AGENTIC_KNOWLEDGE_ENV_PREFIX}DOCS_ENABLED", "false")
    monkeypatch.setenv(f"{AGENTIC_KNOWLEDGE_ENV_PREFIX}DOCS_AUTO_INDEX", "false")
    monkeypatch.setenv(f"{AGENTIC_KNOWLEDGE_ENV_PREFIX}DOCS_MAX_FILES", "7")
    monkeypatch.setenv(
        f"{AGENTIC_KNOWLEDGE_ENV_PREFIX}DOCS_RESCAN_INTERVAL_SECONDS", "0"
    )
    config = AgenticKnowledgeConfig.from_file()
    config.override_from_env()
    assert config.docs_enabled is False
    assert config.docs_auto_index is False
    assert config.docs_max_files == 7
    assert config.docs_rescan_interval_seconds == 0


def test_documentation_needs_the_write_path_to_be_available():
    config = knowledge_config()
    assert config.docs_available is True
    config.write_enabled = False
    assert config.docs_available is False
    config.write_enabled = True
    config.docs_enabled = False
    assert config.docs_available is False


def test_a_ceiling_below_its_floor_is_a_configuration_error():
    with pytest.raises(ValueError, match="docs.max_file_bytes"):
        AgenticKnowledgeConfig.from_json('{"docs": {"max_file_bytes": 8}}')
    with pytest.raises(ValueError, match="docs.max_total_bytes"):
        AgenticKnowledgeConfig.from_json(
            '{"docs": {"max_file_bytes": 4096, "max_total_bytes": 1024}}'
        )
    # A file is one document and the engine splits it, so there is no section
    # ceiling left to get wrong: the key is ignored, not validated.
    assert (
        AgenticKnowledgeConfig.from_json('{"docs": {"max_section_chars": 10}}')
        is not None
    )
    with pytest.raises(ValueError, match="Unknown knowledge config keys"):
        AgenticKnowledgeConfig.from_json(
            '{"worker": {"chunk_tokens": 400, "chunk_overlap": 400}}'
        )
    config = AgenticKnowledgeConfig(
        worker=WorkerSettings(chunk_tokens=400, chunk_overlap=400)
    )
    with pytest.raises(ValueError, match="worker.chunk_overlap"):
        config.validate()
