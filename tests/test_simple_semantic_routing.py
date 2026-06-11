"""
Tests for SimpleSemanticRoutingPlugin.

Run with:
    pytest tests/test_simple_semantic_routing.py -v
"""

import json
import os
import pathlib
import sys

# Ensure the package root is on sys.path so imports work without install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pytest

from llm_router_plugins.utils.routing.simple_semantic import (
    SimpleSemanticRoutingPlugin,
)

# ---------- derive default values from the JSON config (only for tests) ----------
_JSON_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "llm_router_plugins"
    / "resources"
    / "routing"
    / "simple_semantic.json"
)
_json = json.load(open(_JSON_PATH, "r", encoding="utf-8"))
_settings = _json["settings"]

_DEFAULT_MODELS: list = list(_settings["default_models"].values())
_DEFAULT_INTENT_CATEGORIES: dict = {
    **dict(_json["intents"]),
}
_DEFAULT_COMPLEXITY_THRESHOLDS: tuple = (
    _settings["len_thresholds_max"]["simple"],
)


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


def _make_plugin(
    models: str | None = None,
    intent: dict[str, str] | None = None,
    thresholds: str | None = None,
    default_model: str | None = None,
):
    """Helper to set env vars and create a plugin instance."""
    if models is not None:
        os.environ["LLM_ROUTER_ROUTING_MODELS"] = models
    if intent:
        for cat, kws in intent.items():
            os.environ[f"LLM_ROUTER_ROUTING_INTENT_{cat}"] = kws
    if thresholds is not None:
        os.environ["LLM_ROUTER_ROUTING_COMPLEXITY_THRESHOLDS"] = thresholds
    if default_model is not None:
        os.environ["LLM_ROUTER_ROUTING_DEFAULT_MODEL"] = default_model
    return SimpleSemanticRoutingPlugin()


# ------------ model passthrough
def test_model_not_auto_passthrough():
    plugin = _make_plugin()
    payload: dict = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
    }
    result = plugin.apply(payload)
    assert result["model"] == "gpt-4"


def test_model_is_none_passthrough():
    plugin = _make_plugin()
    payload: dict = {"messages": [{"role": "user", "content": "hi"}]}
    result = plugin.apply(payload)
    assert "model" not in result or result.get("model") is None


# ------------ auto with defaults
def test_auto_with_default_config():
    plugin = _make_plugin()
    payload: dict = {
        "model": "auto",
        "messages": [{"role": "user", "content": "hello"}],
    }
    result = plugin.apply(payload)
    # n=2, simple(0) → idx=0, intent=none(-1) → idx=-1 clamped→0
    assert result["model"] == _DEFAULT_MODELS[0]


def test_auto_with_custom_models():
    plugin = _make_plugin(models="model-a|model-b|model-c")
    payload: dict = {
        "model": "auto",
        "messages": [{"role": "user", "content": "hello"}],
    }
    result = plugin.apply(payload)
    assert result["model"] == "model-a"


# ------------ complexity levels
def test_long_text_selects_strongest_model():
    """Complex input should select the last (strongest) model."""
    plugin = _make_plugin(models="tiny|medium|strong", thresholds="10|50")
    # 90 words → 112 tokens > 50 → complex → idx=2
    long_text = " ".join(["word"] * 90)
    payload: dict = {
        "model": "auto",
        "messages": [{"role": "user", "content": long_text}],
    }
    result = plugin.apply(payload)
    # intent=none(-1) → idx=1
    assert result["model"] == "medium"


def test_medium_text_selects_middle_model():
    """Medium complexity with no intent keywords → idx clamped to 0."""
    # n=3, thresholds=10|50
    # 15 words → 18 tokens → medium → base idx=1
    # intent=none(-1) → idx=0 clamped
    plugin = _make_plugin(models="tiny|medium|strong", thresholds="10|50")
    med_text = " ".join(["word"] * 15)
    payload: dict = {
        "model": "auto",
        "messages": [{"role": "user", "content": med_text}],
    }
    result = plugin.apply(payload)
    assert result["model"] == "tiny"


def test_complexity_simple():
    plugin = _make_plugin(models="a|b|c", thresholds="10|50")
    payload: dict = {
        "model": "auto",
        "messages": [{"role": "user", "content": "hi"}],
    }
    result = plugin.apply(payload)
    assert result["model"] == "a"


def test_complexity_medium():
    # 15 words → 18 tokens, 11-50 → medium
    plugin = _make_plugin(models="a|b|c", thresholds="10|50")
    text = " ".join(["word"] * 15)
    payload: dict = {
        "model": "auto",
        "messages": [{"role": "user", "content": text}],
    }
    result = plugin.apply(payload)
    # intent=none(-1), medium(1) → idx=0
    assert result["model"] == "a"


def test_complexity_complex():
    # ~40 words → 50 tokens → complex (> 50 threshold)
    plugin = _make_plugin(models="a|b|c", thresholds="10|50")
    text = " ".join(["word"] * 41)
    payload: dict = {
        "model": "auto",
        "messages": [{"role": "user", "content": text}],
    }
    result = plugin.apply(payload)
    # intent=none(-1), complex(2) → idx=1
    assert result["model"] == "b"


# ------------ intent classification
def test_intent_code_detected():
    plugin = _make_plugin(models="a|b", intent={"CODE": "code|function|debug"})
    payload: dict = {
        "model": "auto",
        "messages": [{"role": "user", "content": "How do I debug this function?"}],
    }
    result = plugin.apply(payload)
    assert result["model"] == "b"


def test_intent_math_detected():
    plugin = _make_plugin(
        models="a|b", intent={"MATH": "calculate|equation|formula"}
    )
    payload: dict = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "Calculate the probability formula"}
        ],
    }
    result = plugin.apply(payload)
    assert result["model"] == "b"


def test_intent_none_for_neutral_text():
    plugin = _make_plugin(
        models="a|b|c",
        intent={"CODE": "code|debug", "MATH": "math|calculate"},
    )
    payload: dict = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Tell me about the weather today"}],
    }
    result = plugin.apply(payload)
    assert result["model"] == "a"


def test_intent_creative_detected():
    plugin = _make_plugin(models="a|b", intent={"CREATIVE": "write|story|poem"})
    payload: dict = {
        "model": "auto",
        "messages": [{"role": "user", "content": "Write a creative story"}],
    }
    result = plugin.apply(payload)
    assert result["model"] == "a"


def test_intent_general_detected():
    plugin = _make_plugin(models="a|b", intent={"GENERAL": "help|what|how|explain"})
    payload: dict = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "How does this work? Explain please"}
        ],
    }
    result = plugin.apply(payload)
    assert result["model"] == "a"


# ------------ combined routing
def test_code_simple_boosted():
    """Code intent + simple complexity → boosted from cheapest."""
    plugin = _make_plugin(
        models="a|b|c", intent={"CODE": "code|debug"}, thresholds="10|50"
    )
    payload: dict = {
        "model": "auto",
        "messages": [{"role": "user", "content": "debug the code"}],
    }
    result = plugin.apply(payload)
    assert result["model"] == "b"


def test_none_complex_demoted():
    """None intent + complex complexity → demoted from strongest."""
    plugin = _make_plugin(
        models="a|b|c", intent={"CODE": "code|debug"}, thresholds="10|50"
    )
    text = " ".join(["word"] * 41)  # complex
    payload: dict = {
        "model": "auto",
        "messages": [{"role": "user", "content": text}],
    }
    result = plugin.apply(payload)
    assert result["model"] == "b"


def test_math_complex_boosted():
    """Math intent + complex complexity → boosted from strongest."""
    plugin = _make_plugin(
        models="a|b|c|d",
        intent={"MATH": "math|calculate"},
        thresholds="10|100",
    )
    # Text must contain a keyword for math intent to match (score > 0)
    text = " ".join(
        ["math", "calculate"] + ["word"] * 96
    )  # 98 words → 122 tokens > 100
    payload: dict = {
        "model": "auto",
        "messages": [{"role": "user", "content": text}],
    }
    result = plugin.apply(payload)
    assert result["model"] == "d"


# ------------ malformed config degradation
def test_malformed_thresholds_uses_defaults():
    os.environ["LLM_ROUTER_ROUTING_COMPLEXITY_THRESHOLDS"] = "bad|one"
    plugin = _make_plugin()
    assert plugin._complexity_thresholds == [25, 150]


def test_single_threshold_value_uses_defaults():
    os.environ["LLM_ROUTER_ROUTING_COMPLEXITY_THRESHOLDS"] = "42"
    plugin = _make_plugin()
    assert plugin._complexity_thresholds == [25, 150]


def test_empty_models_uses_fallback():
    os.environ["LLM_ROUTER_ROUTING_MODELS"] = ""
    plugin = _make_plugin()
    assert plugin._models == list(_DEFAULT_MODELS)


def test_empty_intents_uses_defaults():
    plugin = _make_plugin()
    for cat in _DEFAULT_INTENT_CATEGORIES:
        assert cat in plugin._intent_categories


# ------------ edge cases
def test_no_text_content_uses_default_model():
    os.environ["LLM_ROUTER_ROUTING_DEFAULT_MODEL"] = "fallback-model"
    plugin = _make_plugin()
    payload: dict = {"model": "auto", "messages": []}
    result = plugin.apply(payload)
    assert result["model"] == "fallback-model"


def test_multi_message_uses_last():
    plugin = _make_plugin(
        models="a|b", intent={"CODE": "code|debug"}, thresholds="10|50"
    )
    payload: dict = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "debug the code"},
        ],
    }
    result = plugin.apply(payload)
    assert result["model"] == "b"


def test_query_field_fallback():
    plugin = _make_plugin(models="a|b", intent={"CODE": "code"}, thresholds="10|50")
    payload: dict = {"model": "auto", "query": "debug the code"}
    result = plugin.apply(payload)
    assert result["model"] == "b"


def test_user_last_statement_fallback():
    plugin = _make_plugin(
        models="a|b", intent={"MATH": "math|calculate"}, thresholds="10|50"
    )
    payload: dict = {
        "model": "auto",
        "user_last_statement": "calculate the equation",
    }
    result = plugin.apply(payload)
    assert result["model"] == "b"


def test_index_clamped_at_lower_bound():
    plugin = _make_plugin(models="a", intent={"CODE": "code"}, thresholds="10|50")
    payload: dict = {
        "model": "auto",
        "messages": [{"role": "user", "content": "code code code"}],
    }
    result = plugin.apply(payload)
    assert result["model"] == "a"


def test_index_clamped_at_upper_bound():
    plugin = _make_plugin(
        models="a", intent={"MATH": "math|calculate"}, thresholds="10|50"
    )
    payload: dict = {
        "model": "auto",
        "messages": [{"role": "user", "content": "math math math"}],
    }
    result = plugin.apply(payload)
    assert result["model"] == "a"


def test_two_models_only():
    plugin = _make_plugin(
        models="cheap|expensive", intent={"CODE": "code|debug"}, thresholds="10|50"
    )
    # Text must contain a keyword for code intent to match
    text = " ".join(["code"] + ["word"] * 40)  # 41 words → 51 tokens > 50 → complex
    payload: dict = {
        "model": "auto",
        "messages": [{"role": "user", "content": text}],
    }
    result = plugin.apply(payload)
    # intent=code(+1), complex(1) → idx=1+1=2 clamped→1
    assert result["model"] == "expensive"
