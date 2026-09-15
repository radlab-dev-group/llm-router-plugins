"""
Tests for AgenticRoutingPlugin.

Run with:
    pytest tests/test_agentic_routing.py -v
"""

import importlib.util
import json
import os
import pathlib
import sys

# Ensure the package root is on sys.path so imports work without install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest

from llm_router_plugins.utils.routing.agentic_routing import AgenticRoutingPlugin
from llm_router_plugins.utils.routing.agentic_routing.config import (
    AgenticRoutingConfig,
)

_PREFIX = "LLM_ROUTER_ROUTING_AGENTIC_"

# ---------- derive default values from the JSON config (only for tests) ----------
_JSON_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "llm_router_plugins"
    / "resources"
    / "routing"
    / "agentic_routing.json"
)
_json = json.load(open(_JSON_PATH, "r", encoding="utf-8"))
_settings = _json["settings"]

_DEFAULT_TRIGGER: str = _settings["trigger"][0]
_DEFAULT_FALLBACK: str = _settings["fallback_mode"]
_DEFAULT_THRESHOLD: float = float(_settings["semantic"]["threshold"])
_MODE_NAMES: list = [m["name"] for m in _json["agent_modes"]]
_MODEL_FOR_MODE: dict = {m["name"]: m["model_name"] for m in _json["agent_modes"]}

# Text crafted so that the heuristic layer resolves it to the expected mode.
_MODE_TEXTS: dict = {
    "plan": "Let's plan the roadmap and milestones for the next release",
    "code": "Refactor this function to implement a caching module",
    "review": "Review my pull request for style and maintainability feedback",
    "test": "Write unit tests with pytest and improve coverage assertions",
    "debug": "The app crashes with a traceback, please debug this error",
    "research": "Research how does authentication work and compare the docs",
    "summarize": "Summarize this article and give me a concise recap",
    _DEFAULT_FALLBACK: "Just a general question about various things",
}


# --------------- fixtures
@pytest.fixture(autouse=True)
def clean_routing_env():
    """Clear all routing-related env vars before and after each test."""
    kept: dict[str, str] = {}
    for key in list(os.environ.keys()):
        if key.startswith("LLM_ROUTER_ROUTING"):
            kept[key] = os.environ.pop(key)
    yield
    for key, val in kept.items():
        os.environ[key] = val


def _make_plugin(**env: str) -> AgenticRoutingPlugin:
    """Set env vars (always disabling ML) and create a plugin instance."""
    os.environ[f"{_PREFIX}SEMANTIC_ENABLED"] = "false"
    for key, value in env.items():
        os.environ[f"{_PREFIX}{key}"] = value
    return AgenticRoutingPlugin()


def _payload(text: str, **extra) -> dict:
    """Build a chat payload that triggers the plugin."""
    payload: dict = {
        "model": _DEFAULT_TRIGGER,
        "messages": [{"role": "user", "content": text}],
    }
    payload.update(extra)
    return payload


def _apply(plugin: AgenticRoutingPlugin, text: str, **extra) -> dict:
    return plugin.apply(_payload(text, **extra))


def _minimal_raw_config(**overrides) -> str:
    """Return a raw JSON config string containing every required field."""
    raw: dict = {
        "embedding_model": "test/embedding",
        "settings": {
            "trigger": [_DEFAULT_TRIGGER],
            "fallback_mode": _DEFAULT_FALLBACK,
            "vector_store_path": "",
            "semantic": {
                "enabled": False,
                "threshold": _DEFAULT_THRESHOLD,
                "top_k": 3,
                "chunk_size": 256,
                "chunk_overlap": 64,
            },
        },
        "agent_modes": [
            {
                "name": _DEFAULT_FALLBACK,
                "model_name": "model_fallback",
                "description": "Fallback mode",
            }
        ],
    }
    settings = raw["settings"]
    for key, value in overrides.items():
        if key in settings:
            settings[key] = value
        else:
            raw[key] = value
    return json.dumps(raw)


class _StubRouter:
    """Router stub returning a canned ``route()`` result."""

    def __init__(self, target_name: str, similarity: float) -> None:
        self.target_name = target_name
        self.similarity = similarity
        self.calls: list[str] = []

    def route(self, text: str) -> dict:
        self.calls.append(text)
        return {
            "model_name": _MODEL_FOR_MODE.get(self.target_name, "model_stub"),
            "target_name": self.target_name,
            "similarity": self.similarity,
            "all_scores": {},
        }


# ------------ model passthrough
@pytest.mark.parametrize(
    "model",
    ["gpt-4", "auto", "Agentic", "", None, 123, ["agentic"]],
)
def test_model_not_triggering_passthrough(model):
    plugin = _make_plugin()
    payload = {"messages": [{"role": "user", "content": _MODE_TEXTS["code"]}]}
    if model is not None:
        payload["model"] = model
    expected = dict(payload)

    result = plugin.apply(payload)

    assert result == expected
    assert "agent_mode" not in result
    assert "routing" not in result


def test_trigger_is_case_sensitive():
    plugin = _make_plugin()

    result = plugin.apply({"model": "Agentic", "prompt": _MODE_TEXTS["code"]})

    assert result["model"] == "Agentic"
    assert "routing" not in result


def test_trigger_ignores_surrounding_whitespace():
    plugin = _make_plugin()

    payload = {"model": f"  {_DEFAULT_TRIGGER}  ", "query": _MODE_TEXTS["code"]}
    result = plugin.apply(payload)

    assert result["model"] == _MODEL_FOR_MODE["code"]
    assert result["routing"]["agent_mode"] == "code"


def test_trigger_override_from_env():
    plugin = _make_plugin(TRIGGER="agentx|agent-y")

    hit = plugin.apply({"model": "agentx", "prompt": _MODE_TEXTS["code"]})
    miss = plugin.apply({"model": _DEFAULT_TRIGGER, "prompt": "hello"})

    assert hit["model"] == _MODEL_FOR_MODE["code"]
    assert miss["model"] == _DEFAULT_TRIGGER


# ------------ heuristic detection
@pytest.mark.parametrize("expected_mode", sorted(_MODE_TEXTS))
def test_heuristic_detects_every_mode(expected_mode):
    plugin = _make_plugin()

    result = _apply(plugin, _MODE_TEXTS[expected_mode])

    assert result["agent_mode"] == expected_mode
    assert result["model"] == _MODEL_FOR_MODE[expected_mode]
    assert result["routing"]["source"] == "heuristic"
    assert 0.0 < result["routing"]["similarity"] < 1.0


def test_heuristic_similarity_is_score_over_score_plus_one():
    plugin = _make_plugin()
    text = _MODE_TEXTS["code"]

    mode, score = plugin._detect_heuristic(text)
    result = _apply(plugin, text)

    assert result["routing"]["similarity"] == pytest.approx(score / (score + 1.0))


def test_heuristic_reads_prompt_query_input_and_statement():
    plugin = _make_plugin()
    text = _MODE_TEXTS["summarize"]

    for key in ("user_last_statement", "query", "prompt", "input"):
        result = plugin.apply({"model": _DEFAULT_TRIGGER, key: text})
        assert result["agent_mode"] == "summarize"


def test_heuristic_uses_last_message_only():
    plugin = _make_plugin()

    result = plugin.apply(
        {
            "model": _DEFAULT_TRIGGER,
            "messages": [
                {"role": "user", "content": _MODE_TEXTS["code"]},
                {"role": "assistant", "content": "sure"},
                {"role": "user", "content": _MODE_TEXTS["summarize"]},
            ],
        }
    )

    assert result["agent_mode"] == "summarize"


def test_no_match_falls_back():
    plugin = _make_plugin()

    result = _apply(plugin, "qwerty asdfgh zxcvbn 12345")

    assert result["agent_mode"] == _DEFAULT_FALLBACK
    assert result["model"] == _MODEL_FOR_MODE[_DEFAULT_FALLBACK]
    assert result["routing"]["source"] == "fallback"
    assert result["routing"]["similarity"] == 0.0


def test_empty_payload_uses_empty_text_source():
    plugin = _make_plugin()

    result = plugin.apply({"model": _DEFAULT_TRIGGER})

    assert result["agent_mode"] == _DEFAULT_FALLBACK
    assert result["routing"]["source"] == "empty_text"
    assert result["routing"]["similarity"] == 0.0


def test_explicit_mode_is_honored_without_text():
    """An explicitly declared mode needs no text and must win over fallback."""
    plugin = _make_plugin()

    result = plugin.apply({"model": _DEFAULT_TRIGGER, "agent_mode": "code"})

    assert result["agent_mode"] == "code"
    assert result["model"] == _MODEL_FOR_MODE["code"]
    assert result["routing"]["source"] == "explicit"
    assert result["routing"]["similarity"] == 1.0


def test_unknown_explicit_mode_without_text_uses_empty_text_source():
    plugin = _make_plugin()

    result = plugin.apply({"model": _DEFAULT_TRIGGER, "agent_mode": "no_such_mode"})

    assert result["agent_mode"] == _DEFAULT_FALLBACK
    assert result["routing"]["source"] == "empty_text"
    assert result["routing"]["similarity"] == 0.0


# ------------ explicit mode
def test_explicit_mode_wins_over_text():
    plugin = _make_plugin()

    result = _apply(plugin, _MODE_TEXTS["code"], agent_mode="review")

    assert result["agent_mode"] == "review"
    assert result["model"] == _MODEL_FOR_MODE["review"]
    assert result["routing"]["source"] == "explicit"
    assert result["routing"]["similarity"] == 1.0


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("plan", "plan"),
        ("Plan", "plan"),
        (" PLAN ", "plan"),
        ("Summarize", "summarize"),
    ],
)
def test_explicit_mode_is_normalized(raw, expected):
    plugin = _make_plugin()

    result = _apply(plugin, _MODE_TEXTS["code"], agent_mode=raw)

    assert result["agent_mode"] == expected
    assert result["routing"]["source"] == "explicit"


def test_explicit_mode_unknown_falls_through_to_cascade():
    plugin = _make_plugin()

    result = _apply(plugin, _MODE_TEXTS["debug"], agent_mode="code-review")

    assert result["agent_mode"] == "debug"
    assert result["routing"]["source"] == "heuristic"


def test_explicit_mode_field_priority():
    plugin = _make_plugin()

    result = _apply(
        plugin,
        "hello there",
        agent_mode="test",
        mode="review",
        agent={"mode": "code"},
        metadata={"agent_mode": "plan"},
    )

    assert result["agent_mode"] == "test"


def test_explicit_mode_falls_to_next_field():
    plugin = _make_plugin()

    first = _apply(plugin, "hi", agent_mode="", mode="research")
    second = _apply(plugin, "hi", mode=7, agent={"mode": "summarize"})
    third = _apply(
        plugin, "hi", agent={"mode": None}, metadata={"agent_mode": "plan"}
    )

    assert first["agent_mode"] == "research"
    assert second["agent_mode"] == "summarize"
    assert third["agent_mode"] == "plan"


def test_explicit_mode_non_dict_agent_and_metadata_ignored():
    plugin = _make_plugin()

    result = _apply(plugin, _MODE_TEXTS["test"], agent="plan", metadata="review")

    assert result["agent_mode"] == "test"
    assert result["routing"]["source"] == "heuristic"


# ------------ routing metadata
def test_routing_metadata_structure():
    plugin = _make_plugin()

    result = _apply(plugin, _MODE_TEXTS["plan"])

    assert set(result["routing"]) == {
        "plugin",
        "agent_mode",
        "source",
        "similarity",
    }
    assert result["routing"]["plugin"] == "agentic_routing"
    assert result["routing"]["agent_mode"] == result["agent_mode"] == "plan"
    assert isinstance(result["routing"]["similarity"], float)
    assert set(result) == {"model", "messages", "agent_mode", "routing"}


def test_existing_routing_keys_are_overwritten():
    plugin = _make_plugin()

    result = _apply(
        plugin,
        _MODE_TEXTS["code"],
        agent_mode="stale",
        routing={"plugin": "other"},
    )

    assert result["routing"]["plugin"] == "agentic_routing"
    assert result["agent_mode"] == "code"


# ------------ env overrides
def test_models_env_overrides_model_per_mode():
    plugin = _make_plugin(MODELS="code=custom-code-model|plan=custom-plan-model")

    assert _apply(plugin, _MODE_TEXTS["code"])["model"] == "custom-code-model"
    assert _apply(plugin, _MODE_TEXTS["plan"])["model"] == "custom-plan-model"


def test_models_env_unknown_mode_is_ignored():
    plugin = _make_plugin(MODELS="nonexistent=x")

    assert _apply(plugin, _MODE_TEXTS["code"])["model"] == _MODEL_FOR_MODE["code"]


def test_mode_keywords_env_override():
    plugin = _make_plugin(
        MODE_DEBUG_KEYWORDS="kwonlytoken",
    )

    assert plugin._config.mode_by_name["debug"].keywords == ["kwonlytoken"]
    assert _apply(plugin, "please kwonlytoken now")["agent_mode"] == "debug"
    assert _apply(plugin, "please debug this")["agent_mode"] != "debug"


def test_fallback_mode_env_override():
    plugin = _make_plugin(FALLBACK_MODE="summarize")

    result = _apply(plugin, "qwerty asdfgh zxcvbn")

    assert result["agent_mode"] == "summarize"
    assert result["routing"]["source"] == "fallback"


def test_modes_env_whitelist():
    plugin = _make_plugin(MODES=f"code|{_DEFAULT_FALLBACK}")

    assert _apply(plugin, _MODE_TEXTS["code"])["agent_mode"] == "code"
    assert _apply(plugin, _MODE_TEXTS["plan"])["agent_mode"] == _DEFAULT_FALLBACK


@pytest.mark.parametrize(
    "raw",
    ["1", "true", "True", "yes", "on", " YES "],
)
def test_semantic_enabled_env_accepts_true_values(raw):
    os.environ[f"{_PREFIX}SEMANTIC_ENABLED"] = raw
    config = AgenticRoutingConfig.from_file()
    config._override_from_env()
    assert config.semantic_enabled is True


@pytest.mark.parametrize(
    "raw",
    ["0", "false", "False", "no", "off", " NO "],
)
def test_semantic_enabled_env_accepts_false_values(raw):
    os.environ[f"{_PREFIX}SEMANTIC_ENABLED"] = raw
    config = AgenticRoutingConfig.from_file()
    config._override_from_env()
    assert config.semantic_enabled is False


def test_semantic_enabled_env_unknown_value_keeps_config_value():
    os.environ[f"{_PREFIX}SEMANTIC_ENABLED"] = "maybe"
    config = AgenticRoutingConfig.from_file()
    config._override_from_env()
    assert config.semantic_enabled is _json["settings"]["semantic"]["enabled"]


def test_similarity_threshold_env_override():
    os.environ[f"{_PREFIX}SIMILARITY_THRESHOLD"] = "0.91"
    config = AgenticRoutingConfig.from_file()
    config._override_from_env()
    assert config.similarity_threshold == pytest.approx(0.91)

    os.environ[f"{_PREFIX}SIMILARITY_THRESHOLD"] = "not-a-float"
    config = AgenticRoutingConfig.from_file()
    config._override_from_env()
    assert config.similarity_threshold == pytest.approx(_DEFAULT_THRESHOLD)


def test_config_env_accepts_raw_json():
    os.environ[f"{_PREFIX}CONFIG"] = _minimal_raw_config()
    os.environ[f"{_PREFIX}SEMANTIC_ENABLED"] = "false"

    plugin = AgenticRoutingPlugin()
    result = plugin.apply({"model": _DEFAULT_TRIGGER, "prompt": "anything at all"})

    assert result["agent_mode"] == _DEFAULT_FALLBACK
    assert result["model"] == "model_fallback"


# ------------ validation
def test_validate_rejects_unknown_fallback_mode():
    os.environ[f"{_PREFIX}CONFIG"] = _minimal_raw_config(fallback_mode="nope")
    os.environ[f"{_PREFIX}SEMANTIC_ENABLED"] = "false"

    with pytest.raises(ValueError, match="AgenticRouting"):
        AgenticRoutingPlugin()


def test_validate_rejects_empty_agent_modes():
    raw = json.loads(_minimal_raw_config())
    raw["agent_modes"] = []
    os.environ[f"{_PREFIX}CONFIG"] = json.dumps(raw)
    os.environ[f"{_PREFIX}SEMANTIC_ENABLED"] = "false"

    with pytest.raises(ValueError, match="no agent modes"):
        AgenticRoutingPlugin()


def test_validate_rejects_empty_trigger():
    os.environ[f"{_PREFIX}CONFIG"] = _minimal_raw_config(trigger=[])
    os.environ[f"{_PREFIX}SEMANTIC_ENABLED"] = "false"

    with pytest.raises(ValueError, match="no trigger"):
        AgenticRoutingPlugin()


# ------------ semantic layer (stubbed router)
def test_semantic_hit_above_threshold():
    router = _StubRouter("review", 0.91)
    plugin = AgenticRoutingPlugin(router=router)

    result = plugin.apply(
        {"model": _DEFAULT_TRIGGER, "prompt": "hello there friend"}
    )

    assert router.calls
    assert result["agent_mode"] == "review"
    assert result["model"] == _MODEL_FOR_MODE["review"]
    assert result["routing"]["source"] == "semantic"
    assert result["routing"]["similarity"] == pytest.approx(0.91)


def test_semantic_hit_exactly_at_threshold():
    router = _StubRouter("test", _DEFAULT_THRESHOLD)
    plugin = AgenticRoutingPlugin(router=router)

    result = plugin.apply(
        {"model": _DEFAULT_TRIGGER, "prompt": "hello there friend"}
    )

    assert result["routing"]["source"] == "semantic"
    assert result["agent_mode"] == "test"


def test_semantic_below_threshold_cascades_to_heuristic():
    router = _StubRouter("summarize", _DEFAULT_THRESHOLD - 0.2)
    plugin = AgenticRoutingPlugin(router=router)

    result = plugin.apply(
        {"model": _DEFAULT_TRIGGER, "prompt": _MODE_TEXTS["debug"]}
    )

    assert result["routing"]["source"] == "heuristic"
    assert result["agent_mode"] == "debug"


def test_semantic_below_threshold_with_no_heuristic_falls_back():
    router = _StubRouter("plan", 0.05)
    plugin = AgenticRoutingPlugin(router=router)

    result = plugin.apply(
        {"model": _DEFAULT_TRIGGER, "prompt": "qwerty asdfgh zxcvbn"}
    )

    assert result["routing"]["source"] == "fallback"
    assert result["agent_mode"] == _DEFAULT_FALLBACK


def test_semantic_unknown_target_cascades():
    router = _StubRouter("not-a-mode", 0.99)
    plugin = AgenticRoutingPlugin(router=router)

    result = plugin.apply({"model": _DEFAULT_TRIGGER, "prompt": _MODE_TEXTS["plan"]})

    assert result["routing"]["source"] == "heuristic"
    assert result["agent_mode"] == "plan"


def test_semantic_not_called_when_disabled():
    router = _StubRouter("review", 0.99)
    os.environ[f"{_PREFIX}SEMANTIC_ENABLED"] = "false"
    plugin = AgenticRoutingPlugin(router=router)

    result = plugin.apply(
        {"model": _DEFAULT_TRIGGER, "prompt": "hello there friend"}
    )

    assert router.calls == []
    assert result["routing"]["source"] == "fallback"


def test_explicit_mode_skips_router():
    router = _StubRouter("review", 0.99)
    plugin = AgenticRoutingPlugin(router=router)

    result = plugin.apply(
        {"model": _DEFAULT_TRIGGER, "prompt": "hello", "agent_mode": "plan"}
    )

    assert router.calls == []
    assert result["agent_mode"] == "plan"


# ------------ ML dependencies
def test_semantic_enabled_without_faiss_raises():
    if importlib.util.find_spec("faiss") is not None:
        pytest.skip("faiss is installed — nothing to assert")

    with pytest.raises(ValueError, match="sentence-transformers"):
        AgenticRoutingPlugin()


def test_router_is_built_when_dependencies_available(mock_sentence_transformer):
    """With the ML dependencies available the semantic router is built."""
    pytest.importorskip("faiss")

    plugin = AgenticRoutingPlugin()

    assert plugin._router is not None
    result = plugin.apply({"model": _DEFAULT_TRIGGER, "prompt": _MODE_TEXTS["code"]})
    assert result["routing"]["source"] in {"semantic", "heuristic", "fallback"}


# ------------ registry
def test_plugin_registered_in_main_utils_registry():
    os.environ[f"{_PREFIX}SEMANTIC_ENABLED"] = "false"
    from llm_router_plugins.utils.registry import MAIN_UTILS_REGISTRY

    assert "agentic_routing" in MAIN_UTILS_REGISTRY
    assert MAIN_UTILS_REGISTRY["agentic_routing"] is AgenticRoutingPlugin

    plugin = MAIN_UTILS_REGISTRY["agentic_routing"](logger=None)
    result = plugin.apply(
        {"model": _DEFAULT_TRIGGER, "prompt": _MODE_TEXTS["review"]}
    )

    assert result["agent_mode"] == "review"


# ------------ config loading
def test_default_config_matches_resource_file():
    config = AgenticRoutingConfig.from_file()

    assert config.mode_names == _MODE_NAMES
    assert config.fallback_mode == _DEFAULT_FALLBACK
    assert config.trigger == [_DEFAULT_TRIGGER]
    assert config.similarity_threshold == pytest.approx(_DEFAULT_THRESHOLD)
    assert (
        config.mode_by_name[_DEFAULT_FALLBACK].model_name
        == _MODEL_FOR_MODE[_DEFAULT_FALLBACK]
    )


def test_config_from_file_missing_required_key():
    raw = json.loads(_minimal_raw_config())
    del raw["agent_modes"]
    os.environ[f"{_PREFIX}CONFIG"] = json.dumps(raw)

    with pytest.raises(KeyError, match="agent_modes"):
        AgenticRoutingConfig.from_file()


def test_config_from_json_rejects_empty_string():
    with pytest.raises(ValueError, match="empty config string"):
        AgenticRoutingConfig.from_json("")
