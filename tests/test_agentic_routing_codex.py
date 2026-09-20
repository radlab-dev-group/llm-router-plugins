"""
Tests for the Codex CLI routing plugin
(``llm_router_plugins.utils.routing.agentic_routing.codex``).

Covers the payload normalizer, the deterministic scorer, the mode-resolution
cascade, configuration loading/validation/env overrides and the plugin itself.
The deterministic layers of the cascade are asserted exactly.  The optional
embedding cosine layer is exercised through injected stub routers only, so no
embedding model and no network access are ever involved.

Payload builders reproduce the three real request classes captured in
``agents-conversation/codex/conv-01``: main agent turns, system title
generation and context compaction.

Run with:
    pytest tests/test_agentic_routing_codex.py -v
"""

import copy
import dataclasses
import json
import os
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

from llm_router_plugins.utils.registry import MAIN_UTILS_REGISTRY
from llm_router_plugins.utils.routing.agentic_routing.codex import (
    COLLABORATION_MODE_DEFAULT,
    COLLABORATION_MODE_PLAN,
    HEURISTIC_MODES,
    REQUEST_CLASS_AUX_TITLE,
    REQUEST_CLASS_COMPACTION,
    REQUEST_CLASS_MAIN,
    SOURCE_CLASS,
    SOURCE_COLLABORATION_MODE,
    SOURCE_EXPLICIT,
    SOURCE_FALLBACK,
    SOURCE_HEURISTIC,
    SOURCE_SEMANTIC,
    CodexMode,
    CodexRequest,
    CodexSemanticLayer,
    CodexRoutingConfig,
    CodexRoutingPlugin,
    RoutingDecision,
    classify,
    detect_mode,
    parse_codex_payload,
    score_mode,
    score_to_similarity,
)
from llm_router_plugins.utils.routing.agentic_routing.codex import (
    plugin as plugin_module,
)
from llm_router_plugins.utils.routing.constants import AGENTIC_CODEX_ROUTING_PREFIX

_PREFIX = AGENTIC_CODEX_ROUTING_PREFIX
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_CONFIG_PATH = (
    _REPO_ROOT / "llm_router_plugins" / "resources" / "routing"
    / "agentic_routing_codex.json"
)
_ROUTING_KEYS = {
    "plugin",
    "similarity",
    "agent_mode",
    "source",
    "codex_class",
    "collaboration_mode",
    "request_kind",
    "thread_id",
    "turn_id",
}


@pytest.fixture(autouse=True)
def clean_codex_env(monkeypatch):
    """Clear every Codex routing env var so tests never leak into each other."""
    for key in list(os.environ.keys()):
        if key.startswith(_PREFIX):
            monkeypatch.delenv(key)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _raw_config():
    """Return the bundled Codex routing config as a decoded dict."""
    return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))


def _expected_model(mode_name):
    """Look the model of *mode_name* up in the shipped JSON config."""
    for mode in _raw_config()["codex_modes"]:
        if mode["name"] == mode_name:
            return mode["model_name"]
    raise AssertionError(f"mode not present in config: {mode_name}")


def _config():
    """Load and validate the bundled configuration without env overrides."""
    config = CodexRoutingConfig.from_file()
    config.validate_args()
    return config


def _decide(request, payload=None, config=None):
    """Run the classifier cascade for an already parsed request."""
    return classify(
        payload if payload is not None else {},
        request,
        config if config is not None else _config(),
    )


def _rebuild(config, **changes):
    """Return *config* with a rebuilt ``codex_modes`` tuple."""
    return dataclasses.replace(config, **changes)


def _without_mode(config, mode_name):
    """Return *config* with *mode_name* removed from the mode list."""
    modes = tuple(m for m in config.codex_modes if m.name != mode_name)
    return _rebuild(config, codex_modes=modes)


def _mode(name, **kwargs):
    """Build a synthetic mode definition for scorer tests."""
    kwargs.setdefault("model_name", f"test/{name}")
    kwargs.setdefault("description", f"synthetic {name} mode")
    kwargs.setdefault("examples", ())
    return CodexMode(name=name, **kwargs)


# --------------------------------------------------------------------------
# payload builders (mirroring the captured Codex CLI requests)
# --------------------------------------------------------------------------
INSTRUCTIONS = "You are a coding agent running in the Codex CLI."
PLAN_BLOCK = (
    "<collaboration_mode># Plan Mode (Conversational)\n\n"
    "Think, do not edit.</collaboration_mode>"
)
DEFAULT_BLOCK = (
    "<collaboration_mode># Collaboration Mode: Default\n\n"
    "Execute the task.</collaboration_mode>"
)


def _turn_metadata(**overrides):
    """Serialize the ``x-codex-turn-metadata`` header value."""
    base = {
        "session_id": "sess-1",
        "thread_id": "thread-1",
        "root_turn_id": "turn-1",
        "turn_id": "turn-7",
        "window_id": "thread-1:0",
        "context_window_id": "cw-1",
        "agent_name": "/root",
        "request_kind": "turn",
        "thread_source": "user",
        "sandbox_mode": "read-only",
    }
    base.update(overrides)
    return json.dumps(base)


def _developer(text):
    return {
        "type": "message",
        "role": "developer",
        "content": [{"type": "input_text", "text": text}],
    }


def _user(text):
    return {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": text}],
    }


def main_payload(text="Zaimplementuj nowy moduł eksportu.",
                 collaboration=DEFAULT_BLOCK, **metadata_overrides):
    """Build a main agent turn in the flat Responses shape."""
    items = []
    if collaboration:
        items.append(_developer(INSTRUCTIONS + "\n\n" + collaboration))
    items.append(_user("<environment_context>\n  <cwd>/repo</cwd>\n"
                       "</environment_context>"))
    items.append(_user(text))
    return {
        "model": "auto_codex",
        "instructions": INSTRUCTIONS,
        "input": items,
        "tools": [
            {"type": "function", "name": "exec_command", "description": "d",
             "parameters": {}},
            {"type": "web_search", "external_web_access": True},
        ],
        "parallel_tool_calls": True,
        "reasoning": {"effort": "medium", "summary": "auto"},
        "stream": True,
        "client_metadata": {
            "session_id": "sess-1",
            "thread_id": "thread-1",
            "turn_id": "turn-7",
            "root_turn_id": "turn-1",
            "x-codex-window-id": "thread-1:0",
            "x-codex-turn-metadata": _turn_metadata(**metadata_overrides),
        },
    }


def plan_payload(text="Zaplanuj migrację bazy danych."):
    """Build a main turn whose developer block declares Plan Mode."""
    return main_payload(text, collaboration=PLAN_BLOCK)


def title_payload():
    """Build a system title-generation request (no tools, JSON schema)."""
    payload = main_payload("Summarize this thread in one line.",
                           collaboration=None)
    payload["tools"] = []
    payload["text"] = {
        "format": {
            "type": "json_schema",
            "name": "codex_output_schema",
            "schema": {"type": "string"},
            "strict": True,
        },
    }
    payload["client_metadata"]["x-codex-turn-metadata"] = _turn_metadata(
        thread_source="system",
    )
    return payload


def compaction_payload(collaboration=DEFAULT_BLOCK):
    """Build a context-compaction request."""
    return main_payload("Compact the conversation.", collaboration,
                        request_kind="compaction")


def _plugin(config=None):
    return CodexRoutingPlugin(logger=None, config=config)


# --------------------------------------------------------------------------
# trigger detection and payload identity
# --------------------------------------------------------------------------
class TestTriggerMatching:
    """Only the configured trigger model is routed; everything else is not."""

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-4",
            None,
            123,
            "   ",
            "auto_agentic",
            "auto_codex_extra",
            "",
        ],
    )
    def test_non_trigger_payloads_are_returned_unchanged(self, model):
        payload = main_payload()
        if model is None:
            del payload["model"]
        else:
            payload["model"] = model
        before = copy.deepcopy(payload)

        result = _plugin().apply(payload)

        assert result is payload
        assert payload == before
        assert "routing" not in payload
        assert "agent_mode" not in payload

    def test_trigger_model_is_configurable(self):
        config = dataclasses.replace(_config(), trigger_model="auto_custom")
        payload = main_payload()
        payload["model"] = "auto_custom"

        result = _plugin(config).apply(payload)

        assert result["model"] == _expected_model("implement")

    @pytest.mark.parametrize("payload", [None, "x", [1], 5])
    def test_non_dict_payloads_are_returned_identically(self, payload):
        assert _plugin().apply(payload) is payload

    def test_routed_payload_keeps_every_field_but_the_model(self):
        plugin = _plugin()
        payload = main_payload()
        before = copy.deepcopy(payload)

        result = plugin.apply(payload)

        assert result is payload
        assert result["model"] == _expected_model("implement")
        assert set(payload) - set(before) == {"agent_mode", "routing"}
        assert {k for k in before if payload[k] != before[k]} == {"model"}
        for key in ("tools", "input", "instructions", "reasoning", "stream"):
            assert payload[key] == before[key]

    def test_routing_annotation_reports_the_decision(self):
        payload = plan_payload()

        _plugin().apply(payload)

        assert payload["routing"] == {
            "plugin": "agentic_routing_codex",
            "similarity": 1.0,
            "agent_mode": "plan",
            "source": SOURCE_COLLABORATION_MODE,
            "codex_class": REQUEST_CLASS_MAIN,
            "collaboration_mode": "plan",
            "request_kind": "turn",
            "thread_id": "thread-1",
            "turn_id": "turn-7",
        }
        assert payload["agent_mode"] == "plan"

    def test_identifier_fields_are_surfaced(self):
        payload = main_payload()
        payload["client_metadata"]["thread_id"] = "thread-42"
        payload["client_metadata"]["turn_id"] = "turn-99"

        _plugin().apply(payload)

        assert payload["routing"]["thread_id"] == "thread-42"
        assert payload["routing"]["turn_id"] == "turn-99"
        assert payload["routing"]["request_kind"] == "turn"

    def test_apply_accepts_the_shared_plugin_interface_arguments(self):
        result = _plugin().apply(main_payload(), model_config={"x": 1}, foo="bar")

        assert result["agent_mode"] == "implement"

    def test_plugin_name_is_the_registry_key(self):
        assert CodexRoutingPlugin.name == "agentic_routing_codex"
        assert _plugin().name == "agentic_routing_codex"


# --------------------------------------------------------------------------
# collaboration mode extraction (latest block wins)
# --------------------------------------------------------------------------
class TestCollaborationMode:
    """The last ``<collaboration_mode>`` block of the request is authoritative."""

    def test_plan_block_selects_plan(self):
        request = parse_codex_payload(plan_payload())

        assert request.collaboration_mode == COLLABORATION_MODE_PLAN
        assert _decide(request, plan_payload()).mode == "plan"

    def test_default_block_selects_default(self):
        request = parse_codex_payload(main_payload())

        assert request.collaboration_mode == COLLABORATION_MODE_DEFAULT
        assert _decide(request, main_payload()).mode == "implement"

    def test_no_block_yields_empty_collaboration_mode(self):
        request = parse_codex_payload({"input": []})

        assert request.collaboration_mode == ""

    @pytest.mark.parametrize(
        "blocks,expected",
        [
            ([PLAN_BLOCK], "plan"),
            ([DEFAULT_BLOCK], "default"),
            ([PLAN_BLOCK, DEFAULT_BLOCK], "default"),
            ([DEFAULT_BLOCK, PLAN_BLOCK], "plan"),
        ],
    )
    def test_latest_block_wins(self, blocks, expected):
        payload = {"input": [_developer(block) for block in blocks]}

        assert parse_codex_payload(payload).collaboration_mode == expected

    def test_plan_to_default_switch_reclassifies_the_next_turn(self):
        plugin = _plugin()
        first = plan_payload()
        second = main_payload()

        plugin.apply(first)
        plugin.apply(second)

        assert first["agent_mode"] == "plan"
        assert second["agent_mode"] == "implement"

    def test_unknown_heading_is_not_a_known_mode(self):
        payload = {
            "input": [
                _developer(
                    "<collaboration_mode># Something Else\nx</collaboration_mode>"
                ),
            ],
        }

        assert parse_codex_payload(payload).collaboration_mode == ""

    def test_unclosed_block_is_still_parsed(self):
        payload = {
            "input": [
                _developer("<collaboration_mode># Plan Mode (Conversational)\nmore"),
            ],
        }

        assert parse_codex_payload(payload).collaboration_mode == "plan"

    def test_blocks_outside_developer_messages_are_ignored(self):
        payload = {"input": [_user(PLAN_BLOCK)]}

        assert parse_codex_payload(payload).collaboration_mode == ""


# --------------------------------------------------------------------------
# keyword layer
# --------------------------------------------------------------------------
class TestHeuristicClassification:
    """Polish and English keywords specialise main turns in Default mode."""

    @pytest.mark.parametrize(
        "text,expected_mode,expected_score",
        [
            ("napraw testy", "test", 9.0),
            ("run the tests", "test", 7.0),
            ("RUN THE TESTS", "test", 7.0),
            ("Dodaj testy jednostkowe do modułu płatności", "test", 12.0),
            ("Przejrzyj ten katalog i zaproponuj poprawki do modułów", "review", 13.0),
            ("Napraw ten błąd w module auth.", "debug", 5.0),
            ("Czy możesz zdebugować ten błąd w parserze?", "debug", 3.0),
            ("zrób refactor i przejrzyj to", "review", 6.0),
        ],
    )
    def test_prompts_select_a_specialised_mode(self, text, expected_mode,
                                               expected_score):
        payload = main_payload(text)

        decision = _decide(parse_codex_payload(payload), payload)

        assert decision.mode == expected_mode
        assert decision.source == SOURCE_HEURISTIC
        assert decision.score == expected_score
        assert decision.similarity == pytest.approx(
            score_to_similarity(expected_score)
        )

    @pytest.mark.parametrize(
        "text",
        [
            "przygotuj commit",
            "Zaimplementuj nowy plugin routingowy i zarejestruj go.",
            "test",
            "",
        ],
    )
    def test_prompts_without_signals_fall_back_to_implement(self, text):
        payload = main_payload(text)

        decision = _decide(parse_codex_payload(payload), payload)

        assert decision.mode == "implement"
        assert decision.source == SOURCE_FALLBACK
        assert decision.score == 0.0
        assert decision.similarity == 0.0

    def test_only_the_three_specialised_modes_are_scored(self):
        assert HEURISTIC_MODES == ("test", "review", "debug")

    def test_heuristic_can_be_disabled(self):
        payload = main_payload("napraw testy")
        config = dataclasses.replace(_config(), heuristic_enabled=False)

        decision = _decide(parse_codex_payload(payload), payload, config)

        assert decision.mode == "implement"
        assert decision.source == SOURCE_FALLBACK

    def test_minimum_score_rejects_weak_matches(self):
        payload = main_payload("napraw testy")
        config = dataclasses.replace(_config(), heuristic_min_score=100.0)

        decision = _decide(parse_codex_payload(payload), payload, config)

        assert decision.mode == "implement"
        assert decision.source == SOURCE_FALLBACK

    def test_collaboration_mode_outranks_the_keyword_layer(self):
        payload = plan_payload("napraw testy")

        decision = _decide(parse_codex_payload(payload), payload)

        assert decision.mode == "plan"
        assert decision.source == SOURCE_COLLABORATION_MODE

    def test_specialised_mode_reaches_the_payload(self):
        payload = main_payload("napraw testy")

        _plugin().apply(payload)

        assert payload["model"] == _expected_model("test")
        assert payload["agent_mode"] == "test"
        assert payload["routing"]["similarity"] == pytest.approx(0.9)


# --------------------------------------------------------------------------
# request classes: compaction and auxiliary title generation
# --------------------------------------------------------------------------
class TestRequestClassRouting:
    """Non-conversational requests are routed by class, never by keywords."""

    def test_main_turn_is_classified_as_main(self):
        request = parse_codex_payload(main_payload())

        assert request.request_class == REQUEST_CLASS_MAIN
        assert _decide(request, main_payload()).mode == "implement"

    def test_title_request_routes_to_aux_title(self):
        payload = title_payload()
        request = parse_codex_payload(payload)

        assert request.request_class == REQUEST_CLASS_AUX_TITLE
        decision = _decide(request, payload)
        assert decision.mode == "aux_title"
        assert decision.source == SOURCE_CLASS
        assert decision.similarity == 1.0

    def test_compaction_routes_to_compaction(self):
        payload = compaction_payload()
        request = parse_codex_payload(payload)

        assert request.request_class == REQUEST_CLASS_COMPACTION
        decision = _decide(request, payload)
        assert decision.mode == "compaction"
        assert decision.source == SOURCE_CLASS
        assert decision.similarity == 1.0

    def test_compaction_outranks_a_plan_collaboration_block(self):
        payload = compaction_payload(PLAN_BLOCK)
        request = parse_codex_payload(payload)

        assert request.collaboration_mode == "plan"
        assert request.request_class == REQUEST_CLASS_COMPACTION
        assert _decide(request, payload).mode == "compaction"

    def test_compaction_outranks_the_keyword_layer(self):
        payload = main_payload("napraw testy", request_kind="compaction")

        assert _decide(parse_codex_payload(payload), payload).mode == "compaction"

    @pytest.mark.parametrize(
        "builder,removed",
        [
            (compaction_payload, REQUEST_CLASS_COMPACTION),
            (title_payload, REQUEST_CLASS_AUX_TITLE),
            (plan_payload, "plan"),
        ],
    )
    def test_missing_target_mode_falls_back(self, builder, removed):
        payload = builder()
        config = _without_mode(_config(), removed)

        decision = _decide(parse_codex_payload(payload), payload, config)

        assert decision.mode == "implement"
        assert decision.source == SOURCE_FALLBACK

    def test_class_routed_modes_are_never_keyword_scored(self):
        payload = title_payload()
        payload["input"][-1]["content"][0]["text"] = "napraw testy"

        decision = _decide(parse_codex_payload(payload), payload)

        assert decision.mode == "aux_title"
        assert decision.source == SOURCE_CLASS

    def test_aux_title_reaches_the_payload(self):
        payload = title_payload()

        _plugin().apply(payload)

        assert payload["model"] == _expected_model("aux_title")
        assert payload["routing"]["codex_class"] == REQUEST_CLASS_AUX_TITLE

    def test_compaction_reaches_the_payload(self):
        payload = compaction_payload()

        _plugin().apply(payload)

        assert payload["model"] == _expected_model("compaction")
        assert payload["routing"]["codex_class"] == REQUEST_CLASS_COMPACTION


# --------------------------------------------------------------------------
# explicit overrides
# --------------------------------------------------------------------------
class TestExplicitOverrides:
    """A mode named in the payload wins over every other layer."""

    @pytest.mark.parametrize(
        "key,value,expected_mode",
        [
            ("agent_mode", "review", "review"),
            ("codex_mode", "debug", "debug"),
            ("agent_mode", "  test  ", "test"),
            ("agent_mode", "plan", "plan"),
        ],
    )
    def test_payload_override_selects_the_mode(self, key, value, expected_mode):
        payload = plan_payload()
        payload[key] = value

        decision = _decide(parse_codex_payload(payload), payload)

        assert decision.mode == expected_mode
        assert decision.source == SOURCE_EXPLICIT
        assert decision.similarity == 1.0

    def test_metadata_override_selects_the_mode(self):
        payload = plan_payload()
        payload["metadata"] = {"agent_mode": "debug"}

        decision = _decide(parse_codex_payload(payload), payload)

        assert decision.mode == "debug"
        assert decision.source == SOURCE_EXPLICIT

    def test_agent_mode_is_preferred_over_the_alternatives(self):
        payload = plan_payload()
        payload["agent_mode"] = "test"
        payload["codex_mode"] = "debug"
        payload["metadata"] = {"agent_mode": "review"}

        decision = _decide(parse_codex_payload(payload), payload)

        assert decision.mode == "test"
        assert decision.source == SOURCE_EXPLICIT

    @pytest.mark.parametrize("value", ["ghost", 7, None, "", ["test"]])
    def test_unknown_or_malformed_overrides_are_ignored(self, value):
        payload = plan_payload()
        payload["agent_mode"] = value

        decision = _decide(parse_codex_payload(payload), payload)

        assert decision.mode == "plan"
        assert decision.source == SOURCE_COLLABORATION_MODE

    @pytest.mark.parametrize(
        "metadata",
        ["nope", {"agent_mode": 5}, {"other": "test"}, None],
    )
    def test_malformed_metadata_blocks_are_ignored(self, metadata):
        payload = plan_payload()
        payload["metadata"] = metadata

        decision = _decide(parse_codex_payload(payload), payload)

        assert decision.mode == "plan"
        assert decision.source == SOURCE_COLLABORATION_MODE

    def test_override_reaches_the_payload(self):
        payload = main_payload()
        payload["agent_mode"] = "review"

        _plugin().apply(payload)

        assert payload["model"] == _expected_model("review")
        assert payload["routing"]["source"] == SOURCE_EXPLICIT
        assert payload["agent_mode"] == "review"


# --------------------------------------------------------------------------
# passthrough and fail-open behaviour
# --------------------------------------------------------------------------
class TestPassthrough:
    """Routing never breaks a request and never invents a model."""

    def test_mode_without_a_model_is_a_no_op(self):
        modes = tuple(
            dataclasses.replace(mode, model_name="") if mode.name == "plan"
            else mode
            for mode in _config().codex_modes
        )
        plugin = _plugin(_rebuild(_config(), codex_modes=modes))
        payload = plan_payload()
        before = copy.deepcopy(payload)

        result = plugin.apply(payload)

        assert result is payload
        assert payload == before
        assert payload["model"] == "auto_codex"
        assert "routing" not in payload

    def test_unclassified_mode_is_a_no_op(self, monkeypatch):
        unknown = RoutingDecision("ghost", SOURCE_CLASS, 1.0, 1.0)
        monkeypatch.setattr(plugin_module, "classify", lambda *args, **kwargs: unknown)
        payload = main_payload()
        before = copy.deepcopy(payload)

        result = _plugin().apply(payload)

        assert result is payload
        assert payload == before
        assert "routing" not in payload

    def test_classifier_failure_is_swallowed(self, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("classifier exploded")

        monkeypatch.setattr(plugin_module, "classify", boom)
        payload = main_payload()
        before = copy.deepcopy(payload)

        result = _plugin().apply(payload)

        assert result is payload
        assert payload == before

    def test_parser_failure_is_swallowed(self, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("parser exploded")

        monkeypatch.setattr(plugin_module, "parse_codex_payload", boom)
        payload = main_payload()

        result = _plugin().apply(payload)

        assert result is payload
        assert payload["model"] == "auto_codex"

    def test_fail_open_is_logged(self, monkeypatch):
        logged = []

        class RecordingLogger:
            def warning(self, message, *args):
                logged.append(message % args)

            def debug(self, message, *args):
                return None

            def info(self, message, *args):
                return None

        def boom(*args, **kwargs):
            raise RuntimeError("classifier exploded")

        monkeypatch.setattr(plugin_module, "classify", boom)

        result = CodexRoutingPlugin(logger=RecordingLogger()).apply(main_payload())

        assert "routing" not in result
        assert any("Codex routing failed" in entry for entry in logged)

    def test_minimal_trigger_only_payload_is_routed(self):
        payload = {"model": "auto_codex"}

        result = _plugin().apply(payload)

        assert result["agent_mode"] == "implement"
        assert result["model"] == _expected_model("implement")


# --------------------------------------------------------------------------
# keyword scorer
# --------------------------------------------------------------------------
class TestScoring:
    """Signals sum per mode; ties go to the mode declared first in config."""

    @pytest.mark.parametrize(
        "mode_kwargs,text,expected",
        [
            ({"keywords": ("testy",)}, "napraw testy", 1.0),
            ({"keywords": ("TESTY",)}, "napraw testy", 1.0),
            ({"phrases": ("napraw testy",)}, "napraw testy", 2.0),
            ({"phrases": ("napraw testy:3",)}, "napraw testy", 3.0),
            ({"patterns": (r"\bnapraw\b",)}, "napraw testy", 3.0),
            (
                {"keywords": ("testy",), "phrases": ("napraw",),
                 "patterns": (r"\bmodu",)},
                "napraw testy modu",
                6.0,
            ),
            (
                {"keywords": ("pytest",), "weights": {"pytest": 5}},
                "pytest x",
                5.0,
            ),
            (
                {"keywords": ("PyTest",), "weights": {"pytest": 4}},
                "pytest x",
                4.0,
            ),
            (
                {"keywords": ("pytest",), "weights": {"pytest": "lots"}},
                "pytest x",
                1.0,
            ),
            ({"keywords": ("pytest",), "weights": "nope"}, "pytest x", 1.0),
            ({"phrases": ("a:x",)}, "a:x", 2.0),
            ({"phrases": ("a:x",)}, "a", 0.0),
            ({"patterns": (r".*",)}, "", 0.0),
            ({"patterns": ("(",)}, "x", 0.0),
            ({}, "anything at all", 0.0),
        ],
    )
    def test_score_mode(self, mode_kwargs, text, expected):
        assert score_mode(_mode("probe", **mode_kwargs), text.lower()) == expected

    def test_score_to_similarity_saturates(self):
        assert score_to_similarity(0.0) == 0.0
        assert score_to_similarity(-3.0) == 0.0
        assert score_to_similarity(1.0) == 0.5
        assert score_to_similarity(3.0) == 0.75
        assert score_to_similarity(9.0) == 0.9

    def test_detect_mode_returns_mode_and_score(self):
        probe = _mode("probe", keywords=("testy",))

        mode, score = detect_mode("napraw testy", [probe])

        assert mode is probe
        assert score == 1.0

    def test_detect_mode_wins_ties_on_declaration_order(self):
        probe = _mode("probe", keywords=("testy",))
        other = _mode("other", keywords=("przejrzyj",))

        mode, _ = detect_mode("testy przejrzyj", [probe, other])
        assert mode is probe

        mode, _ = detect_mode("przejrzyj przejrzyj", [probe, other])
        assert mode is other

    def test_detect_mode_lowercases_the_text(self):
        probe = _mode("probe", keywords=("testy",))

        assert detect_mode("TESTY", [probe])[0] is probe

    @pytest.mark.parametrize("modes", [[], [_mode("probe", keywords=("testy",))]])
    def test_detect_mode_without_a_match(self, modes):
        mode, score = detect_mode("completely unrelated", modes)

        assert mode is None
        assert score == 0.0

    def test_heuristic_modes_are_the_sub_modes_only(self):
        assert HEURISTIC_MODES == ("test", "review", "debug")


# --------------------------------------------------------------------------
# payload normalization robustness
# --------------------------------------------------------------------------
class TestPayloadRobustness:
    """Malformed or missing payload sections never raise, only degrade."""

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"input": "not-a-list"},
            {"input": None},
            {"tools": None},
            {"tools": [{"type": "web_search"}]},
            {"client_metadata": "nope"},
            {"client_metadata": {"x-codex-turn-metadata": "not json"}},
            {"text": "nope"},
            {"reasoning": "nope"},
            {"input": [{"type": "message", "role": "user",
                        "content": "plain string"}]},
        ],
    )
    def test_malformed_payloads_parse(self, payload):
        assert isinstance(parse_codex_payload(payload), CodexRequest)

    def test_tool_without_a_name_falls_back_to_its_type(self):
        request = parse_codex_payload({"tools": [{"type": "web_search"}]})

        assert request.tool_names == ("web_search",)
        assert request.has_tools is True

    def test_string_content_is_used_as_text(self):
        request = parse_codex_payload(
            {"input": [{"type": "message", "role": "user",
                        "content": "plain string"}]},
        )

        assert request.latest_user_text == "plain string"

    def test_main_turn_exposes_the_decoded_identifiers(self):
        request = parse_codex_payload(main_payload())

        assert request.session_id == "sess-1"
        assert request.thread_id == "thread-1"
        assert request.turn_id == "turn-7"
        assert request.root_turn_id == "turn-1"
        assert request.window_id == "thread-1:0"
        assert request.agent_name == "/root"
        assert request.thread_source == "user"
        assert request.sandbox_mode == "read-only"
        assert request.request_kind == "turn"
        assert request.context_window_id == "cw-1"

    def test_main_turn_exposes_the_request_shape(self):
        request = parse_codex_payload(main_payload())

        assert request.tool_names == ("exec_command", "web_search")
        assert request.has_tools is True
        assert request.reasoning_effort == "medium"
        assert request.parallel_tool_calls is True
        assert request.structured_output is False
        assert request.request_class == REQUEST_CLASS_MAIN
        assert request.collaboration_mode == COLLABORATION_MODE_DEFAULT

    def test_context_size_is_derived_from_the_serialized_body(self):
        request = parse_codex_payload(main_payload())

        assert request.context_chars == 553
        assert request.context_tokens == request.context_chars // 4

    def test_environment_context_is_not_the_user_text(self):
        request = parse_codex_payload(main_payload())

        assert request.latest_user_text == "Zaimplementuj nowy moduł eksportu."
        assert "<environment_context>" not in request.latest_user_text

    def test_title_request_is_structured_and_collaboration_free(self):
        request = parse_codex_payload(title_payload())

        assert request.structured_output is True
        assert request.request_class == REQUEST_CLASS_AUX_TITLE
        assert request.collaboration_mode == ""
        assert request.tool_names == ()

    def test_compaction_request_is_tagged(self):
        request = parse_codex_payload(compaction_payload())

        assert request.request_class == REQUEST_CLASS_COMPACTION

    def test_parsing_never_mutates_the_payload(self):
        payload = main_payload()
        before = copy.deepcopy(payload)

        parse_codex_payload(payload)

        assert payload == before


# --------------------------------------------------------------------------
# configuration loading and validation
# --------------------------------------------------------------------------
class TestConfig:
    """The bundled config loads, is self-consistent and validates loudly."""

    def test_shipped_settings(self):
        config = _config()

        assert config.trigger_model == "auto_codex"
        assert config.fallback_mode == "implement"
        assert config.heuristic_enabled is True
        assert config.heuristic_min_score == 2.0

    def test_shipped_mode_order(self):
        config = _config()

        assert config.mode_names == [
            "plan", "implement", "test", "review", "debug",
            "aux_title", "compaction",
        ]
        assert list(config.mode_by_name.keys()) == config.mode_names

    @pytest.mark.parametrize(
        "mode_name,model_name",
        [
            ("plan", "qwen/Qwen3.8-Flash-Next"),
            ("aux_title", "qwen/Qwen3.8-Flash-Next"),
            ("compaction", "qwen/Qwen3.8-Flash-Next"),
            ("implement", "qwen/Qwen3.8-27B"),
            ("test", "qwen/Qwen3.8-27B"),
            ("review", "qwen/Qwen3.8-27B"),
            ("debug", "qwen/Qwen3.8-27B"),
        ],
    )
    def test_shipped_mode_models(self, mode_name, model_name):
        assert _config().mode_by_name[mode_name].model_name == model_name

    def test_class_routed_modes_carry_no_signals(self):
        config = _config()

        for mode_name in ("aux_title", "compaction"):
            mode = config.mode_by_name[mode_name]
            assert mode.keywords == ()
            assert mode.phrases == ()
            assert mode.patterns == ()

    def test_sub_modes_carry_polish_and_english_signals(self):
        config = _config()

        for mode_name in ("plan", "implement", "test", "review", "debug"):
            mode = config.mode_by_name[mode_name]
            assert mode.keywords
            assert mode.phrases
            assert mode.patterns

    def test_semantic_defaults(self):
        config = _config()

        assert config.semantic_enabled is True
        assert config.similarity_threshold == 0.55
        assert config.top_k == 3
        assert config.chunk_size == 256
        assert config.chunk_overlap == 64
        assert config.embedding_model
        assert config.vector_store_path == ""

    @pytest.mark.parametrize(
        "mutate,fragment",
        [
            (lambda raw: raw.pop("settings"),
             "Missing required top-level key 'settings' in config."),
            (lambda raw: raw.pop("codex_modes"),
             "Missing required top-level key 'codex_modes' in config."),
            (lambda raw: raw["settings"].pop("trigger_model"),
             "Missing required field 'trigger_model' in settings."),
            (lambda raw: raw["settings"].pop("fallback_mode"),
             "Missing required field 'fallback_mode' in settings."),
            (lambda raw: raw["codex_modes"][0].pop("name"),
             "Missing required field 'name' in codex_modes[0]."),
            (lambda raw: raw["codex_modes"][0].pop("model_name"),
             "Missing required field 'model_name' in codex_modes[0]."),
            (lambda raw: raw["codex_modes"][0].pop("description"),
             "Missing required field 'description' in codex_modes[0]."),
        ],
    )
    def test_missing_required_keys_are_reported(self, mutate, fragment):
        raw = copy.deepcopy(_raw_config())
        mutate(raw)

        with pytest.raises(KeyError, match=re.escape(fragment)):
            CodexRoutingConfig._from_raw(raw)

    @pytest.mark.parametrize(
        "changes,fragment",
        [
            ({"trigger_model": ""}, "no trigger configured"),
            ({"codex_modes": ()}, "no Codex modes defined"),
            ({"fallback_mode": "nope"},
             "fallback_mode 'nope' is not a defined Codex mode"),
            ({"semantic_enabled": True, "embedding_model": ""},
             "no embedding_model configured"),
            ({"chunk_size": 0}, "chunk_size must be > 0, got 0"),
            ({"chunk_overlap": -1}, "chunk_overlap must be >= 0, got -1"),
            ({"top_k": 0}, "top_k must be >= 1, got 0"),
        ],
    )
    def test_validate_args_rejects_inconsistent_settings(self, changes, fragment):
        config = _rebuild(_config(), **changes)

        with pytest.raises(ValueError, match=re.escape(fragment)):
            config.validate_args()

    def test_duplicate_mode_names_are_reported(self):
        config = _config()
        config = _rebuild(
            config, codex_modes=config.codex_modes + config.codex_modes[:1]
        )

        with pytest.raises(ValueError,
                           match=re.escape("duplicate Codex mode names ['plan']")):
            config.validate_args()

    def test_embedding_model_is_only_required_when_semantic_is_enabled(self):
        config = _rebuild(_config(), semantic_enabled=False, embedding_model="")

        config.validate_args()

    def test_mode_without_a_model_is_valid(self):
        modes = tuple(
            dataclasses.replace(mode, model_name="") if mode.name == "plan"
            else mode
            for mode in _config().codex_modes
        )

        _rebuild(_config(), codex_modes=modes).validate_args()


# --------------------------------------------------------------------------
# environment overrides
# --------------------------------------------------------------------------
def _load_with_env(monkeypatch, **overrides):
    """Load the bundled config with *overrides* applied through the env."""
    for suffix, value in overrides.items():
        monkeypatch.setenv(f"{_PREFIX}{suffix}", str(value))

    config = CodexRoutingConfig.from_file()
    config.override_from_env()
    config.validate_args()
    return config


class TestEnvironmentOverrides:
    """Every documented env var reshapes the config without touching the file."""

    def test_trigger_is_overridden(self, monkeypatch):
        assert _load_with_env(monkeypatch, TRIGGER="auto_x").trigger_model == "auto_x"

    def test_single_mode_model_is_overridden(self, monkeypatch):
        config = _load_with_env(monkeypatch, MODEL_PLAN="m/plan")

        assert config.mode_by_name["plan"].model_name == "m/plan"

    def test_batch_model_overrides_ignore_unknown_modes(self, monkeypatch):
        config = _load_with_env(monkeypatch, MODELS="plan=x|test=y|nope=z")

        assert config.mode_by_name["plan"].model_name == "x"
        assert config.mode_by_name["test"].model_name == "y"
        assert len(config.codex_modes) == 7

    def test_mode_whitelist(self, monkeypatch):
        config = _load_with_env(monkeypatch, MODES="plan|implement")

        assert config.mode_names == ["plan", "implement"]

    def test_unknown_mode_whitelist_is_ignored(self, monkeypatch):
        config = _load_with_env(monkeypatch, MODES="nope|nada")

        assert len(config.codex_modes) == 7

    def test_fallback_mode_is_overridden(self, monkeypatch):
        config = _load_with_env(monkeypatch, FALLBACK_MODE="review")

        assert config.fallback_mode == "review"

    def test_heuristic_can_be_disabled(self, monkeypatch):
        config = _load_with_env(monkeypatch, HEURISTIC_ENABLED="false")

        assert config.heuristic_enabled is False

    def test_heuristic_threshold_is_overridden(self, monkeypatch):
        config = _load_with_env(monkeypatch, HEURISTIC_MIN_SCORE="4.5")

        assert config.heuristic_min_score == 4.5

    def test_mode_keywords_are_overridden(self, monkeypatch):
        config = _load_with_env(monkeypatch, MODE_test_KEYWORDS="a|b")

        assert config.mode_by_name["test"].keywords == ("a", "b")

    def test_keywords_of_an_unknown_mode_are_ignored(self, monkeypatch):
        config = _load_with_env(monkeypatch, MODE_nope_KEYWORDS="a|b")

        assert len(config.mode_by_name["test"].keywords) == 9

    def test_embedding_settings_are_overridden(self, monkeypatch):
        config = _load_with_env(
            monkeypatch,
            MODEL="m/other",
            SIMILARITY_THRESHOLD="0.9",
            TOP_K="5",
            CHUNK_SIZE="128",
            CHUNK_OVERLAP="16",
        )

        assert config.embedding_model == "m/other"
        assert config.similarity_threshold == 0.9
        assert config.top_k == 5
        assert config.chunk_size == 128
        assert config.chunk_overlap == 16

    def test_semantic_can_be_disabled(self, monkeypatch):
        config = _load_with_env(monkeypatch, SEMANTIC_ENABLED="false")

        assert config.semantic_enabled is False

    def test_persist_dir_is_taken_verbatim(self, monkeypatch, tmp_path):
        config = _load_with_env(monkeypatch, PERSIST_DIR=str(tmp_path))

        assert config.vector_store_path == str(tmp_path)

    def test_env_reaches_the_plugin(self, monkeypatch):
        monkeypatch.setenv(f"{_PREFIX}TRIGGER", "auto_x")
        monkeypatch.setenv(f"{_PREFIX}MODELS", "implement=m/override")

        payload = main_payload()
        payload["model"] = "auto_x"
        result = CodexRoutingPlugin(logger=None).apply(payload)

        assert result["model"] == "m/override"

    def test_env_config_path_is_honoured(self, monkeypatch, tmp_path):
        raw = copy.deepcopy(_raw_config())
        raw["settings"]["trigger_model"] = "auto_from_file"
        custom = tmp_path / "codex_config.json"
        custom.write_text(json.dumps(raw), encoding="utf-8")
        monkeypatch.setenv(f"{_PREFIX}CONFIG", str(custom))

        assert CodexRoutingConfig.from_file().trigger_model == "auto_from_file"


# --------------------------------------------------------------------------
# registry wiring
# --------------------------------------------------------------------------
class TestRegistry:
    """The plugin is discoverable next to the untouched agentic plugin."""

    def test_codex_plugin_is_registered(self):
        assert MAIN_UTILS_REGISTRY["agentic_routing_codex"] is CodexRoutingPlugin

    def test_agentic_plugin_is_still_registered(self):
        assert "agentic_routing" in MAIN_UTILS_REGISTRY

    def test_registered_name_matches_the_class(self):
        assert CodexRoutingPlugin.name == "agentic_routing_codex"


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------
class TestDeterminism:
    """Same request, same decision, same similarity — every time."""

    def test_classification_is_stable_across_repeats(self):
        payload = main_payload("Uruchom testy i napraw błędy.")
        request = parse_codex_payload(payload)

        decisions = [_decide(request, payload) for _ in range(5)]

        dumped = [dataclasses.asdict(item) for item in decisions]
        assert dumped[0] == dumped[-1]
        assert len({json.dumps(item, sort_keys=True) for item in dumped}) == 1

    def test_two_plugins_annotate_identically(self):
        first = _plugin().apply(main_payload("Przejrzyj ten moduł."))
        second = _plugin().apply(main_payload("Przejrzyj ten moduł."))

        assert first["routing"] == second["routing"]
        assert first["model"] == second["model"]

    def test_parsing_is_stable_across_repeats(self):
        payload = plan_payload()
        parsed = [parse_codex_payload(payload) for _ in range(3)]

        assert parsed[0] == parsed[1] == parsed[2]

    def test_similarity_is_reproducible_for_the_same_text(self):
        similarities = set()
        for _ in range(4):
            payload = main_payload("Napraw testy jednostkowe.")
            similarities.add(_decide(parse_codex_payload(payload), payload).similarity)

        assert len(similarities) == 1


# --------------------------------------------------------------------------
# semantic similarity (embeddings + cosine)
# --------------------------------------------------------------------------
class _StubRouter:
    """Stand in for the BiEncoder + FAISS router, without any ML dependency."""

    def __init__(self, target_name="test", similarity=0.9, all_scores=None,
                 fail=False, raw_result=None):
        self.target_name = target_name
        self.similarity = similarity
        self.all_scores = all_scores
        self.fail = fail
        self.raw_result = raw_result
        self.calls = []

    def route(self, text):
        """Record the query and replay a router-shaped result."""
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("index unavailable")
        if self.raw_result is not None:
            return self.raw_result
        result = {
            "model_name": "stub-model",
            "target_name": self.target_name,
            "similarity": self.similarity,
        }
        if self.all_scores is not None:
            result["all_scores"] = self.all_scores
        return result


class _CaptureLogger:
    """Collect log records per level so the fail-open paths are assertable."""

    def __init__(self):
        self.records = {"warning": [], "info": [], "debug": []}

    def _log(self, level, message, *args):
        text = str(message)
        if args:
            text = text % args
        self.records[level].append(text)

    def warning(self, message, *args):
        self._log("warning", message, *args)

    def info(self, message, *args):
        self._log("info", message, *args)

    def debug(self, message, *args):
        self._log("debug", message, *args)

    def joined(self):
        """Return every recorded message, whatever the level, as one string."""
        return " | ".join(
            self.records["warning"] + self.records["info"] + self.records["debug"]
        )


def _semantic_layer(router, threshold=0.5, modes=None, logger=None):
    """Build a semantic layer over *router* for the shipped mode table."""
    return CodexSemanticLayer(
        router,
        threshold,
        modes if modes is not None else _config().mode_by_name,
        logger,
    )


def _classify_semantic(payload, router, threshold=0.5, logger=None):
    """Classify *payload* with a stub-backed semantic layer attached."""
    request = parse_codex_payload(payload)
    return classify(
        payload,
        request,
        _config(),
        semantic=_semantic_layer(router, threshold, logger=logger),
    )


class TestSemanticSimilarity:
    """``similarity`` is embedding cosine, shared with ``agentic_routing``."""

    def test_layer_is_available_only_when_a_router_is_present(self):
        assert _semantic_layer(_StubRouter()).available is True
        assert _semantic_layer(None).available is False

    def test_empty_text_never_reaches_the_router(self):
        router = _StubRouter()

        assert _semantic_layer(router).route("") is None
        assert router.calls == []

    def test_router_result_must_be_a_mapping(self):
        router = _StubRouter(raw_result="not-a-dict")

        assert _semantic_layer(router).route("napraw testy") is None
        assert router.calls == ["napraw testy"]

    def test_threshold_acceptance_is_inclusive(self):
        mode, similarity = _semantic_layer(_StubRouter()).accept(
            {"target_name": "test", "similarity": 0.5}
        )

        assert mode.name == "test"
        assert similarity == 0.5

    def test_similarity_is_read_from_all_scores(self):
        layer = _semantic_layer(_StubRouter())
        result = {"all_scores": [{"target": "test", "similarity": 0.77}]}

        assert layer.similarity_for("test", result) == 0.77

    def test_similarity_falls_back_to_the_top_level_target(self):
        layer = _semantic_layer(_StubRouter())
        result = {"target_name": "test", "similarity": 0.42}

        assert layer.similarity_for("test", result) == 0.42

    def test_similarity_is_none_for_a_mode_the_router_never_scored(self):
        layer = _semantic_layer(_StubRouter())
        result = {"target_name": "test", "similarity": 0.42}

        assert layer.similarity_for("plan", result) is None

    @pytest.mark.parametrize(
        "router, mode_name, result",
        [
            (None, "test", {"target_name": "test", "similarity": 0.4}),
            (_StubRouter(), "", {"target_name": "test", "similarity": 0.4}),
            (_StubRouter(), "test", None),
            (_StubRouter(), "test", {}),
        ],
    )
    def test_similarity_is_none_when_nothing_can_be_reported(
        self, router, mode_name, result
    ):
        assert _semantic_layer(router).similarity_for(mode_name, result) is None

    def test_similarity_ignores_malformed_score_entries(self):
        layer = _semantic_layer(_StubRouter())
        result = {"all_scores": ["x", None, {"target": "debug", "similarity": 0.2}]}

        assert layer.similarity_for("debug", result) == 0.2

    @pytest.mark.parametrize(
        "builder, expected_mode, expected_source",
        [
            (plan_payload, "plan", SOURCE_COLLABORATION_MODE),
            (compaction_payload, "compaction", SOURCE_CLASS),
            (title_payload, "aux_title", SOURCE_CLASS),
        ],
    )
    def test_deterministic_layers_short_circuit_the_router(
        self, builder, expected_mode, expected_source
    ):
        router = _StubRouter()

        decision = _classify_semantic(builder(), router)

        assert decision.mode == expected_mode
        assert decision.source == expected_source
        assert decision.similarity == 1.0
        assert router.calls == []

    def test_explicit_override_short_circuits_the_router(self):
        payload = main_payload()
        payload["agent_mode"] = "review"
        router = _StubRouter()

        decision = _classify_semantic(payload, router)

        assert (decision.mode, decision.source) == ("review", SOURCE_EXPLICIT)
        assert router.calls == []

    def test_semantic_match_decides_a_plain_main_turn(self):
        router = _StubRouter("test", 0.9)

        decision = _classify_semantic(main_payload("x"), router)

        assert decision.mode == "test"
        assert decision.source == SOURCE_SEMANTIC
        assert decision.score == decision.similarity == 0.9
        assert router.calls == ["x"]

    def test_cosine_replaces_the_keyword_derived_similarity(self):
        router = _StubRouter(
            "test", 0.9, all_scores=[{"target": "test", "similarity": 0.77}]
        )

        decision = _classify_semantic(main_payload("napraw testy"), router)

        assert decision.mode == "test"
        assert decision.source == SOURCE_HEURISTIC
        assert decision.score == 9.0
        assert decision.similarity == 0.77

    def test_semantic_match_wins_when_keywords_are_silent(self):
        router = _StubRouter("debug", 0.81)

        decision = _classify_semantic(main_payload("przygotuj commit"), router)

        assert decision.mode == "debug"
        assert decision.source == SOURCE_SEMANTIC
        assert decision.similarity == 0.81

    def test_below_threshold_falls_back_but_still_reports_cosine(self):
        router = _StubRouter(
            "debug", 0.2, all_scores=[{"target": "implement", "similarity": 0.31}]
        )

        decision = _classify_semantic(main_payload("przygotuj commit"), router)

        assert decision.mode == "implement"
        assert decision.source == SOURCE_FALLBACK
        assert decision.score == 0.0
        assert decision.similarity == 0.31

    def test_router_failure_is_fail_open(self):
        logger = _CaptureLogger()

        decision = _classify_semantic(
            main_payload("przygotuj commit"), _StubRouter(fail=True), logger=logger
        )

        assert decision.mode == "implement"
        assert decision.source == SOURCE_FALLBACK
        assert decision.similarity == 0.0
        assert "semantic lookup failed" in logger.joined()

    def test_similarity_without_a_layer_is_derived_from_the_score(self):
        payload = main_payload("napraw testy")

        decision = _decide(parse_codex_payload(payload), payload)

        assert decision.source == SOURCE_HEURISTIC
        assert decision.similarity == score_to_similarity(decision.score) == 0.9

    def test_below_threshold_match_is_logged(self):
        logger = _CaptureLogger()

        _classify_semantic(
            main_payload("przygotuj commit"),
            _StubRouter("debug", 0.2),
            threshold=0.5,
            logger=logger,
        )

        assert "below threshold" in logger.joined()

    def test_the_text_is_embedded_at_most_once_per_request(self):
        router = _StubRouter("test", 0.9, all_scores=[{"target": "test",
                                                       "similarity": 0.9}])

        _classify_semantic(main_payload("napraw testy"), router)

        assert len(router.calls) == 1

    def test_resolve_combines_route_and_accept(self):
        mode, similarity = _semantic_layer(_StubRouter("review", 0.66)).resolve(
            "przejrzyj moduł"
        )

        assert mode.name == "review"
        assert similarity == 0.66

    def test_plugin_reports_the_cosine_similarity_end_to_end(self):
        plugin = CodexRoutingPlugin(
            logger=None, config=_config(), emb_router=_StubRouter("debug", 0.81)
        )

        result = plugin.apply(main_payload("przygotuj commit"))

        assert result["model"] == _expected_model("debug")
        assert result["routing"] == {
            "plugin": "agentic_routing_codex",
            "similarity": 0.81,
            "agent_mode": "debug",
            "source": SOURCE_SEMANTIC,
            "codex_class": REQUEST_CLASS_MAIN,
            "collaboration_mode": COLLABORATION_MODE_DEFAULT,
            "request_kind": "turn",
            "thread_id": "thread-1",
            "turn_id": "turn-7",
        }
        assert set(result["routing"]) == _ROUTING_KEYS

    def test_injected_layer_takes_precedence_over_the_router_argument(self):
        strong, weak = _StubRouter("debug", 0.81), _StubRouter("review", 0.99)
        plugin = CodexRoutingPlugin(
            logger=None,
            config=_config(),
            emb_router=weak,
            semantic=_semantic_layer(strong),
        )

        result = plugin.apply(main_payload("przygotuj commit"))

        assert result["routing"]["agent_mode"] == "debug"
        assert weak.calls == []
        assert strong.calls == ["przygotuj commit"]

    def test_semantic_routing_stays_off_when_disabled_by_env(
        self, monkeypatch
    ):
        monkeypatch.setenv(f"{_PREFIX}SEMANTIC_ENABLED", "false")

        plugin = CodexRoutingPlugin(logger=None, emb_router=_StubRouter())

        assert plugin._semantic is None

    def test_plugin_without_semantic_dependencies_still_routes(self):
        plugin = CodexRoutingPlugin()

        assert plugin._semantic is None

        result = plugin.apply(main_payload("napraw testy"))

        assert result["model"] == _expected_model("test")
        assert result["routing"]["source"] == SOURCE_HEURISTIC

    def test_router_is_built_with_the_shared_embedding_factory(
        self, monkeypatch, tmp_path
    ):
        captured = {}
        router = _StubRouter("test", 0.9)

        def fake_build_router(**kwargs):
            captured.update(kwargs)
            return router

        monkeypatch.setattr(plugin_module, "build_embedding_router",
                            fake_build_router)
        monkeypatch.setattr(plugin_module, "resolve_persist_dir",
                            lambda *_args, **_kwargs: str(tmp_path))
        config = _config()

        plugin = CodexRoutingPlugin(logger=None, config=config)

        assert plugin._semantic is not None
        assert captured["embedding_model"] == config.embedding_model
        assert captured["chunk_size"] == config.chunk_size
        assert captured["chunk_overlap"] == config.chunk_overlap
        assert captured["top_k"] == config.top_k
        assert captured["persist_dir"] == str(tmp_path)
        assert tuple(m.name for m in captured["routing_targets"]) == (
            "plan",
            "implement",
            "test",
            "review",
            "debug",
        )

    def test_unbuildable_router_disables_semantic_routing(self, monkeypatch):
        def explode(**_kwargs):
            raise ValueError("faiss is not installed")

        monkeypatch.setattr(plugin_module, "build_embedding_router", explode)
        logger = _CaptureLogger()

        plugin = CodexRoutingPlugin(logger=logger, config=_config())

        assert plugin._semantic is None
        assert "semantic routing disabled" in logger.joined()
        assert plugin.apply(main_payload("napraw testy"))["routing"][
            "similarity"
        ] == 0.9
