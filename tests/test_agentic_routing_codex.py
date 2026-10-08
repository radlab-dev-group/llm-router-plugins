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
    CLASS_ROUTED_MODES,
    COLLABORATION_MODE_DEFAULT,
    COLLABORATION_MODE_PLAN,
    DEFAULT_CLASSIFY_MAX_CHARS,
    HEURISTIC_MODES,
    REQUEST_CLASS_AUX_TITLE,
    REQUEST_CLASS_COMPACTION,
    REQUEST_CLASS_MAIN,
    SOURCE_CLASS,
    SOURCE_COLLABORATION_MODE,
    SOURCE_EXPLICIT,
    SOURCE_FALLBACK,
    SOURCE_HEURISTIC,
    SOURCE_PHASE,
    SOURCE_SEMANTIC,
    CodexMode,
    CodexModeClassifier,
    CodexRequest,
    CodexPayloadParser,
    CodexSemanticLayer,
    CodexModeScorer,
    CodexRoutingConfig,
    CodexRoutingPlugin,
    RoutingDecision,
)
from llm_router_plugins.utils.routing.agentic_routing.codex import (
    plugin as plugin_module,
)
from llm_router_plugins.utils.routing.agentic_routing.codex import (
    payload as payload_module,
)
from llm_router_plugins.utils.routing.agentic_routing.codex import (
    scoring as scoring_module,
)
from llm_router_plugins.utils.routing.constants import AGENTIC_CODEX_ROUTING_PREFIX

_PREFIX = AGENTIC_CODEX_ROUTING_PREFIX
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_CONFIG_PATH = (
    _REPO_ROOT
    / "llm_router_plugins"
    / "resources"
    / "routing"
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


#: Scorer shared by the scoring tests; its plan cache is exercised directly.
_scorer = CodexModeScorer(
    negation_pattern=_config().heuristic_negation_pattern,
    weights=_config().heuristic_weights,
)


def _parse(payload, max_chars=DEFAULT_CLASSIFY_MAX_CHARS):
    """Normalize *payload* with a parser built for this test."""
    return CodexPayloadParser(max_chars=max_chars).parse(payload)


def _decide(request, payload=None, config=None):
    """Run the classifier cascade for an already parsed request."""
    classifier = CodexModeClassifier(config if config is not None else _config())
    return classifier.classify(payload if payload is not None else {}, request)


def test_phase_is_independent_of_keyword_heuristics():
    config = dataclasses.replace(_config(), heuristic_enabled=False)
    payload = main_payload()
    payload["input"].append(
        {
            "type": "function_call",
            "name": "exec_command",
            "call_id": "tests",
            "arguments": json.dumps({"cmd": "pytest -q"}),
        }
    )
    assert _decide(_parse(payload), payload, config).mode == "test"


def test_parser_preserves_incremental_tool_events_without_user_message():
    payload = main_payload()
    payload["input"] = [
        {
            "type": "function_call_output",
            "call_id": "tests",
            "id": "result",
            "output": "Process exited with code 1\nOutput:\nfailure",
        }
    ]
    request = _parse(payload)
    assert request.latest_user_text == ""
    assert len(request.activity) == 1
    assert request.activity[0].call_id == "tests"
    assert request.activity[0].event_id == "result"
    assert request.activity[0].name == ""


def test_long_structured_test_output_keeps_execution_status():
    payload = main_payload()
    payload["input"].extend(
        [
            {
                "type": "function_call",
                "name": "exec_command",
                "call_id": "tests",
                "arguments": '{"cmd":"pytest"}',
            },
            {
                "type": "function_call_output",
                "call_id": "tests",
                "output": json.dumps({"output": "x" * 10000, "exit_code": 1}),
            },
        ]
    )
    request = _parse(payload)
    assert len(request.activity[-1].text) <= request.classify_max_chars
    decision = _decide(request, payload)
    assert (decision.mode, decision.source) == ("debug", SOURCE_PHASE)


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
#: The title-generation instruction, captured verbatim from the Codex CLI.
TITLE_INSTRUCTION = (
    "Generate a concise, single-line task title of at most 36 characters and "
    "under five words where possible. Start with an imperative verb. Write in "
    "the user's language. Do not answer the request.\n\n"
    "User prompt:\nRefaktoryzuj mapping.py na klasę"
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


def _assistant(text):
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


def main_payload(
    text="Zaimplementuj nowy moduł eksportu.",
    collaboration=DEFAULT_BLOCK,
    **metadata_overrides,
):
    """Build a main agent turn in the flat Responses shape."""
    items = []
    if collaboration:
        items.append(_developer(INSTRUCTIONS + "\n\n" + collaboration))
    items.append(
        _user("<environment_context>\n  <cwd>/repo</cwd>\n" "</environment_context>")
    )
    items.append(_user(text))
    return {
        "model": "auto_codex",
        "instructions": INSTRUCTIONS,
        "input": items,
        "tools": [
            {
                "type": "function",
                "name": "exec_command",
                "description": "d",
                "parameters": {},
            },
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


def title_payload(text=TITLE_INSTRUCTION, **metadata_overrides):
    """
    Build a title-generation request: no tools, JSON-schema output, system thread.

    The turn metadata defaults to the ``thread_source == "system"`` declaration
    of the CLI; *metadata_overrides* replace any single entry of it, so the
    releases that emit this call on another thread stay reproducible.
    """
    metadata = {"thread_source": "system"}
    metadata.update(metadata_overrides)
    payload = main_payload(text, collaboration=None, **metadata)
    payload["tools"] = []
    payload["text"] = {
        "format": {
            "type": "json_schema",
            "name": "codex_output_schema",
            "schema": {"type": "string"},
            "strict": True,
        },
    }
    payload["client_metadata"]["x-codex-turn-metadata"] = _turn_metadata(**metadata)
    return payload


def compaction_payload(collaboration=DEFAULT_BLOCK):
    """Build a context-compaction request."""
    return main_payload(
        "Compact the conversation.", collaboration, request_kind="compaction"
    )


def _plugin(config=None, emb_router=None):
    config = config if config is not None else _config()
    if emb_router is None:
        config = dataclasses.replace(config, semantic_enabled=False)
    return CodexRoutingPlugin(logger=None, config=config, emb_router=emb_router)


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
            "agentic",
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
        config = dataclasses.replace(
            _config(), trigger_model="auto_custom", semantic_enabled=False
        )
        payload = main_payload()
        payload["model"] = "auto_custom"

        result = _plugin(config).apply(payload)

        assert result["model"] == _expected_model("implement")

    @pytest.mark.parametrize("payload", [None, "x", [1], 5])
    def test_non_dict_payloads_are_returned_identically(self, payload):
        assert _plugin().apply(payload) is payload

    def test_routed_payload_keeps_every_field_but_the_model(self):
        plugin = _plugin(_rebuild(_config(), semantic_enabled=False))
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
        result = _plugin(_rebuild(_config(), semantic_enabled=False)).apply(
            main_payload(), model_config={"x": 1}, foo="bar"
        )

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
        request = _parse(plan_payload())

        assert request.collaboration_mode == COLLABORATION_MODE_PLAN
        assert _decide(request, plan_payload()).mode == "plan"

    def test_default_block_selects_default(self):
        request = _parse(main_payload())

        assert request.collaboration_mode == COLLABORATION_MODE_DEFAULT
        assert _decide(request, main_payload()).mode == "implement"

    def test_no_block_yields_empty_collaboration_mode(self):
        request = _parse({"input": []})

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

        assert _parse(payload).collaboration_mode == expected

    def test_plan_to_default_switch_reclassifies_the_next_turn(self):
        plugin = _plugin(_rebuild(_config(), semantic_enabled=False))
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

        assert _parse(payload).collaboration_mode == ""

    def test_unclosed_block_is_still_parsed(self):
        payload = {
            "input": [
                _developer("<collaboration_mode># Plan Mode (Conversational)\nmore"),
            ],
        }

        assert _parse(payload).collaboration_mode == "plan"

    def test_blocks_outside_developer_messages_are_ignored(self):
        payload = {"input": [_user(PLAN_BLOCK)]}

        assert _parse(payload).collaboration_mode == ""


# --------------------------------------------------------------------------
# keyword layer
# --------------------------------------------------------------------------
class TestHeuristicClassification:
    """Polish and English keywords specialise main turns in Default mode."""

    @pytest.mark.parametrize(
        "text,expected_mode,expected_score",
        [
            ("napraw testy", "test", 3.0),
            ("run the tests", "test", 3.0),
            ("RUN THE TESTS", "test", 3.0),
            ("Dodaj testy jednostkowe do modułu płatności", "test", 3.0),
            (
                "Przejrzyj ten katalog i zaproponuj poprawki do modułów",
                "review",
                6.0,
            ),
            ("Napraw ten błąd w module auth.", "debug", 3.0),
            ("Czy możesz zdebugować ten błąd w parserze?", "debug", 4.0),
            ("zrób refactor i przejrzyj to", "review", 5.0),
            ("zrób push na remote", "git_review", 4.0),
        ],
    )
    def test_prompts_select_a_specialised_mode(
        self, text, expected_mode, expected_score
    ):
        payload = main_payload(text)

        decision = _decide(_parse(payload), payload)

        assert decision.mode == expected_mode
        assert decision.source == SOURCE_HEURISTIC
        assert decision.score == expected_score
        assert decision.similarity == pytest.approx(
            _scorer.score_to_similarity(expected_score)
        )

    @pytest.mark.parametrize(
        "text",
        [
            "wyrenderuj pusty stan w widoku",
            "Zaimplementuj nowy moduł eksportu i podłącz go do aplikacji.",
            "deploy the service to the cluster",
            "",
        ],
    )
    def test_prompts_without_signals_fall_back_to_implement(self, text):
        payload = main_payload(text)

        decision = _decide(_parse(payload), payload)

        assert decision.mode == "implement"
        assert decision.source == SOURCE_FALLBACK
        assert decision.score == 0.0
        assert decision.similarity == 0.0

    def test_only_the_four_specialised_modes_are_scored(self):
        assert HEURISTIC_MODES == ("test", "git_review", "review", "debug")

    @pytest.mark.parametrize(
        "text,expected_source",
        [
            ("przejrzyj zmiany na branczu i opisz co się zmieniło", SOURCE_SEMANTIC),
            ("porównaj ten branch z develop", SOURCE_HEURISTIC),
            ("review the diff before I merge", SOURCE_HEURISTIC),
            ("przejrzyj merge request w GitLabie", SOURCE_HEURISTIC),
            ("Sprawdź historię tego pliku przez git blame.", SOURCE_HEURISTIC),
            ("zrób commit z tymi zmianami", SOURCE_HEURISTIC),
            ("resolve the merge conflicts", SOURCE_HEURISTIC),
            ("review this branch", SOURCE_HEURISTIC),
            ("review the branch before merging", SOURCE_SEMANTIC),
            ("przejrzyj ten branch", SOURCE_SEMANTIC),
            ("przejrzyj zmiany na gałęzi", SOURCE_SEMANTIC),
            ("compare branches and summarize the changes", SOURCE_HEURISTIC),
            (
                "Zobacz na commity na tym branczu, podsumuj co tam jest.",
                SOURCE_HEURISTIC,
            ),
        ],
    )
    def test_git_review_prompts_select_git_review(self, text, expected_source):
        payload = main_payload(text)
        router = _StubRouter("git_review", 0.9)

        decision = _classify_semantic(payload, router)

        assert decision.mode == "git_review"
        assert decision.source == expected_source
        assert router.calls == ([text] if expected_source == SOURCE_SEMANTIC else [])

    @pytest.mark.parametrize(
        "text",
        [
            "użyj quicksort do sortowania tabeli",
            "dodaj powiadomienia dla użytkowników do aplikacji",
            "don't blame the user for this input",
            "rozważ inne podejście do projektowania",
            "porównaj dwa podejścia architektoniczne",
            "the prairie fire spread",
            "Dodaj endpoint w tym repozytorium.",
            "Implement the handler in this repository.",
            "Dodaj obsługę GitHub do integracji.",
            "Podsumuj zmiany w README.",
            "Opisz zmiany w konfiguracji.",
            "Przejrzyj zmiany w dokumentacji.",
            "Zaimplementuj historię zmian ustawień użytkownika.",
            "Resolve conflicts between application settings.",
            "Implement branching logic in the parser.",
            "Dodaj endpoint na tym branchu.",
        ],
    )
    def test_non_git_prompts_do_not_select_git_review(self, text):
        payload = main_payload(text)

        decision = _decide(_parse(payload), payload)

        assert decision.mode != "git_review"

    @pytest.mark.parametrize(
        "text",
        [
            "Implement the specification for the API.",
            "Dodaj obsługę specyfikacji OpenAPI.",
            "Implement a mock payment provider.",
            "Dodaj mock do trybu demonstracyjnego.",
            "Add assertions to validate production input.",
            "Implement the fixture importer for the application.",
            "Extend the product suite with a new module.",
            "Show geographic coverage on the map.",
            "Implement testimony collection.",
            "Sprawdź konfigurację i porównaj ją z README.",
        ],
    )
    def test_non_testing_prompts_do_not_select_test(self, text):
        payload = main_payload(text)

        decision = _decide(_parse(payload), payload)

        assert decision.mode != "test"

    @pytest.mark.parametrize(
        "text",
        [
            "Write unit tests using mocks and fixtures.",
            "Przygotuj testy jednostkowe dla modułu wandb.",
            "Uruchom pytest i popraw czerwone przypadki.",
            "Increase coverage of the scoring module.",
            "Zwiększ coverage tego pakietu.",
            "Add a test fixture for the parser.",
            "Run the RSpec specs.",
        ],
    )
    def test_explicit_testing_intent_still_selects_test(self, text):
        payload = main_payload(text)

        decision = _decide(_parse(payload), payload)

        assert decision.mode == "test"
        assert decision.source == SOURCE_HEURISTIC

    def test_explicit_git_review_override_short_circuits_scoring(self):
        payload = main_payload("przygotuj opis zmian")
        payload["agent_mode"] = "git_review"

        decision = _decide(_parse(payload), payload)

        assert decision.mode == "git_review"
        assert decision.source == SOURCE_EXPLICIT
        assert decision.similarity == 1.0

    def test_heuristic_can_be_disabled(self):
        payload = main_payload("napraw testy")
        config = dataclasses.replace(_config(), heuristic_enabled=False)

        decision = _decide(_parse(payload), payload, config)

        assert decision.mode == "implement"
        assert decision.source == SOURCE_FALLBACK

    def test_minimum_score_rejects_weak_matches(self):
        payload = main_payload("napraw testy")
        config = dataclasses.replace(_config(), heuristic_min_score=100.0)

        decision = _decide(_parse(payload), payload, config)

        assert decision.mode == "implement"
        assert decision.source == SOURCE_FALLBACK

    @pytest.mark.parametrize(
        "text",
        [
            "jaki jest rozmiar tego pliku",
            "to nie jest jeszcze gotowe",
            "uruchom jest testy i powiedz wynik",
        ],
    )
    def test_polish_copula_does_not_select_test_mode(self, text):
        payload = main_payload(text)

        decision = _decide(_parse(payload), payload)

        if "uruchom jest" in text:
            assert decision.mode == "test"
        else:
            assert decision.mode == "implement"
            assert decision.source == SOURCE_FALLBACK

    @pytest.mark.parametrize(
        "text",
        [
            "protest przeciw tej zmianie",
            "knowledge base update",
        ],
    )
    def test_keywords_do_not_match_mid_word(self, text):
        payload = main_payload(text)

        decision = _decide(_parse(payload), payload)

        assert decision.mode == "implement"
        assert decision.source == SOURCE_FALLBACK

    def test_collaboration_mode_outranks_the_keyword_layer(self):
        payload = plan_payload("napraw testy")

        decision = _decide(_parse(payload), payload)

        assert decision.mode == "plan"
        assert decision.source == SOURCE_COLLABORATION_MODE

    def test_specialised_mode_reaches_the_payload(self):
        payload = main_payload("napraw testy")

        _plugin().apply(payload)

        assert payload["model"] == _expected_model("test")
        assert payload["agent_mode"] == "test"
        assert payload["routing"]["similarity"] == pytest.approx(3.0 / 4.0)

    def test_candidates_follow_the_heuristic_mode_order(self):
        config = _config()
        swapped = tuple(
            (
                _mode("review", keywords=("alpha",), weights={"alpha": 5})
                if mode.name == "review"
                else (
                    _mode("git_review", keywords=("alpha",), weights={"alpha": 5})
                    if mode.name == "git_review"
                    else mode
                )
            )
            for mode in config.codex_modes
        )
        payload = main_payload("alpha")

        decision = _decide(
            _parse(payload),
            payload,
            _rebuild(config, codex_modes=swapped),
        )

        assert decision.mode == "implement"
        assert decision.source == SOURCE_FALLBACK

    @pytest.mark.parametrize(
        "margin,expected_source", [(1.0, SOURCE_FALLBACK), (0.5, SOURCE_HEURISTIC)]
    )
    def test_minimum_margin_is_configurable_and_inclusive(
        self, margin, expected_source
    ):
        config = _rebuild(
            _config(),
            heuristic_min_margin=margin,
            codex_modes=(
                _mode("implement"),
                _mode("test", keywords=("alpha",), weights={"alpha": 3.5}),
                _mode("review", keywords=("beta",), weights={"beta": 3}),
            ),
        )
        decision = _decide(
            CodexRequest(latest_user_text="alpha beta"), config=config
        )

        assert decision.source == expected_source
        assert decision.mode == (
            "test" if expected_source == SOURCE_HEURISTIC else "implement"
        )

    def test_zero_margin_does_not_accept_a_tie(self):
        config = _rebuild(
            _config(),
            heuristic_min_margin=0,
            codex_modes=(
                _mode("implement"),
                _mode("test", phrases=("alpha:3",)),
                _mode("review", phrases=("beta:3",)),
            ),
        )

        decision = _decide(
            CodexRequest(latest_user_text="alpha beta"), config=config
        )

        assert (decision.mode, decision.source) == ("implement", SOURCE_FALLBACK)

    def test_single_candidate_compares_against_zero(self):
        config = _rebuild(
            _config(),
            codex_modes=(_mode("implement"), _mode("test", phrases=("alpha:3",))),
        )

        decision = _decide(CodexRequest(latest_user_text="alpha"), config=config)

        assert (decision.mode, decision.source) == ("test", SOURCE_HEURISTIC)


# --------------------------------------------------------------------------
# request classes: compaction and auxiliary title generation
# --------------------------------------------------------------------------
class TestRequestClassRouting:
    """Non-conversational requests are routed by class, never by keywords."""

    def test_main_turn_is_classified_as_main(self):
        request = _parse(main_payload())

        assert request.request_class == REQUEST_CLASS_MAIN
        assert _decide(request, main_payload()).mode == "implement"

    def test_title_request_routes_to_aux_title(self):
        payload = title_payload()
        request = _parse(payload)

        assert request.request_class == REQUEST_CLASS_AUX_TITLE
        decision = _decide(request, payload)
        assert decision.mode == "aux_title"
        assert decision.source == SOURCE_CLASS
        assert decision.similarity == 1.0

    @pytest.mark.parametrize("thread_source", ["system", "user", ""])
    def test_title_call_is_recognized_from_its_shape(self, thread_source):
        payload = title_payload(thread_source=thread_source)
        request = _parse(payload)

        assert request.request_class == REQUEST_CLASS_AUX_TITLE
        decision = _decide(request, payload)
        assert decision.mode == "aux_title"
        assert decision.source == SOURCE_CLASS

    def test_title_call_without_client_metadata_is_still_aux_title(self):
        payload = title_payload()
        del payload["client_metadata"]

        assert _parse(payload).request_class == REQUEST_CLASS_AUX_TITLE

    def test_system_thread_without_a_request_kind_is_aux_title(self):
        payload = title_payload(thread_source="system", request_kind="")

        assert _parse(payload).request_class == REQUEST_CLASS_AUX_TITLE

    def test_title_prompt_on_an_agent_turn_stays_a_main_turn(self):
        payload = main_payload("Generate a title for this module.")

        assert _parse(payload).request_class == REQUEST_CLASS_MAIN

    def test_structured_output_without_a_title_prompt_is_a_main_turn(self):
        payload = title_payload(
            "Podsumuj ten wątek w jednym zdaniu.", thread_source="user"
        )
        request = _parse(payload)

        assert request.structured_output is True
        assert request.request_class == REQUEST_CLASS_MAIN

    def test_compaction_outranks_the_title_shape(self):
        payload = title_payload(request_kind="compaction")

        assert _parse(payload).request_class == REQUEST_CLASS_COMPACTION

    def test_compaction_routes_to_compaction(self):
        payload = compaction_payload()
        request = _parse(payload)

        assert request.request_class == REQUEST_CLASS_COMPACTION
        decision = _decide(request, payload)
        assert decision.mode == "compaction"
        assert decision.source == SOURCE_CLASS
        assert decision.similarity == 1.0

    def test_compaction_outranks_a_plan_collaboration_block(self):
        payload = compaction_payload(PLAN_BLOCK)
        request = _parse(payload)

        assert request.collaboration_mode == "plan"
        assert request.request_class == REQUEST_CLASS_COMPACTION
        assert _decide(request, payload).mode == "compaction"

    def test_compaction_outranks_the_keyword_layer(self):
        payload = main_payload("napraw testy", request_kind="compaction")

        assert _decide(_parse(payload), payload).mode == "compaction"

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

        decision = _decide(_parse(payload), payload, config)

        assert decision.mode == "implement"
        assert decision.source == SOURCE_FALLBACK

    def test_class_routed_modes_are_never_keyword_scored(self):
        payload = title_payload()
        payload["input"][-1]["content"][0]["text"] = "napraw testy"

        decision = _decide(_parse(payload), payload)

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

        decision = _decide(_parse(payload), payload)

        assert decision.mode == expected_mode
        assert decision.source == SOURCE_EXPLICIT
        assert decision.similarity == 1.0

    def test_metadata_override_selects_the_mode(self):
        payload = plan_payload()
        payload["metadata"] = {"agent_mode": "debug"}

        decision = _decide(_parse(payload), payload)

        assert decision.mode == "debug"
        assert decision.source == SOURCE_EXPLICIT

    def test_agent_mode_is_preferred_over_the_alternatives(self):
        payload = plan_payload()
        payload["agent_mode"] = "test"
        payload["codex_mode"] = "debug"
        payload["metadata"] = {"agent_mode": "review"}

        decision = _decide(_parse(payload), payload)

        assert decision.mode == "test"
        assert decision.source == SOURCE_EXPLICIT

    @pytest.mark.parametrize("value", ["ghost", 7, None, "", ["test"]])
    def test_unknown_or_malformed_overrides_are_ignored(self, value):
        payload = plan_payload()
        payload["agent_mode"] = value

        decision = _decide(_parse(payload), payload)

        assert decision.mode == "plan"
        assert decision.source == SOURCE_COLLABORATION_MODE

    @pytest.mark.parametrize(
        "metadata",
        ["nope", {"agent_mode": 5}, {"other": "test"}, None],
    )
    def test_malformed_metadata_blocks_are_ignored(self, metadata):
        payload = plan_payload()
        payload["metadata"] = metadata

        decision = _decide(_parse(payload), payload)

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
            dataclasses.replace(mode, model_name="") if mode.name == "plan" else mode
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

        class GhostClassifier:
            def __init__(self, *args, **kwargs):
                return None

            def classify(self, payload, request):
                return unknown

        monkeypatch.setattr(plugin_module, "CodexModeClassifier", GhostClassifier)
        payload = main_payload()
        before = copy.deepcopy(payload)

        result = _plugin().apply(payload)

        assert result is payload
        assert payload == before
        assert "routing" not in payload

    def test_classifier_failure_is_swallowed(self, monkeypatch):
        class ExplodingClassifier:
            def __init__(self, *args, **kwargs):
                return None

            def classify(self, payload, request):
                raise RuntimeError("classifier exploded")

        monkeypatch.setattr(
            plugin_module, "CodexModeClassifier", ExplodingClassifier
        )
        payload = main_payload()
        before = copy.deepcopy(payload)

        result = _plugin().apply(payload)

        assert result is payload
        assert payload == before

    def test_parser_failure_is_swallowed(self, monkeypatch):
        class ExplodingParser:
            def __init__(self, *args, **kwargs):
                return None

            def parse(self, payload):
                raise RuntimeError("parser exploded")

        monkeypatch.setattr(plugin_module, "CodexPayloadParser", ExplodingParser)
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

        class ExplodingClassifier:
            def __init__(self, *args, **kwargs):
                return None

            def classify(self, payload, request):
                raise RuntimeError("classifier exploded")

        monkeypatch.setattr(
            plugin_module, "CodexModeClassifier", ExplodingClassifier
        )

        config = dataclasses.replace(_config(), semantic_enabled=False)
        result = CodexRoutingPlugin(
            logger=RecordingLogger(),
            config=config,
        ).apply(main_payload())

        assert "routing" not in result
        assert any("Codex routing failed" in entry for entry in logged)

    def test_minimal_trigger_only_payload_is_routed(self):
        payload = {"model": "auto_codex"}

        result = _plugin(_rebuild(_config(), semantic_enabled=False)).apply(payload)

        assert result["agent_mode"] == "implement"
        assert result["model"] == _expected_model("implement")


# --------------------------------------------------------------------------
# keyword scorer
# --------------------------------------------------------------------------
class TestScoring:
    """Independent evidence adds weight; overlapping evidence and ties abstain."""

    def test_overlapping_signals_count_once(self):
        mode = _mode(
            "test",
            keywords=("test", "testy"),
            phrases=("napisz testy:3",),
            patterns=(r"\btest\w*", r"\bnapisz\b.{0,30}\btesty\b"),
            weights={"testy": 3},
        )

        assert _scorer.score_mode(mode, "napisz testy") == 3.0

    def test_conflicting_modes_abstain_regardless_of_order(self):
        test = _mode("test", phrases=("write tests:3",))
        review = _mode("review", phrases=("review code:3",))

        for modes in ([test, review], [review, test]):
            assert _scorer.detect_mode("write tests and review code", modes) == (
                None,
                3.0,
            )

    def test_forbidden_test_run_is_not_testing_intent(self):
        mode = _config().mode_by_name["test"]

        assert (
            _scorer.score_mode(mode, "nie uruchamiaj testów, popraw dokumentację")
            == 0.0
        )

    def test_forbidden_run_does_not_suppress_writing_tests(self):
        mode = _config().mode_by_name["test"]

        assert (
            _scorer.score_mode(mode, "nie uruchamiaj testów i napisz testy") == 3.0
        )

    @pytest.mark.parametrize(
        "text",
        [
            "nie uruchamiaj pytest, popraw dokumentację",
            "do not run the tests; edit the README",
            "don't write tests. Implement the handler",
            "without running pytest, update docs",
            "bez uruchamiania testów popraw dokumentację",
        ],
    )
    def test_local_forbidden_actions_are_not_positive_signals(self, text):
        assert _scorer.score_mode(_config().mode_by_name["test"], text) == 0.0

    @pytest.mark.parametrize(
        "text",
        [
            "nie uruchamiaj testów, napisz testy",
            "do not run tests but write unit tests",
            "write unit tests and do not run them",
            "without running tests; add tests",
        ],
    )
    def test_positive_testing_clause_survives_a_local_ban(self, text):
        assert _scorer.score_mode(_config().mode_by_name["test"], text) == 3.0

    def test_bug_negation_is_not_an_action_ban(self):
        assert (
            _scorer.score_mode(
                _config().mode_by_name["debug"], "dlaczego nie działa"
            )
            == 3.0
        )

    def test_custom_negation_rule_and_empty_rule(self):
        mode = _mode("test", phrases=("run tests:3",))

        assert (
            CodexModeScorer(
                negation_pattern=r"ignore:.*", weights=_config().heuristic_weights
            ).score_mode(mode, "ignore: run tests")
            == 0.0
        )
        assert (
            CodexModeScorer(
                negation_pattern="", weights=_config().heuristic_weights
            ).score_mode(mode, "do not run tests")
            == 3.0
        )

    def test_repeated_signals_do_not_amplify_the_score(self):
        mode = _mode(
            "test",
            keywords=("test", "test"),
            phrases=("test:2",),
            patterns=(r"\btest\b", r"\btest\b"),
        )

        assert _scorer.score_mode(mode, "test test test") == 3.0

    def test_ranking_includes_zero_scores_and_non_overlapping_evidence(self):
        mode = _mode(
            "test", keywords=("testy",), phrases=("napisz testy:3", "coverage:4")
        )
        text = "napisz testy; coverage"

        ranking = _scorer.rank_modes(text, [_mode("review"), mode])

        assert [(item.mode.name, item.score) for item in ranking] == [
            ("test", 7.0),
            ("review", 0.0),
        ]
        assert ranking[0].matches == (
            scoring_module.SignalMatch("literal:napisz testy", 0, 12, 3.0),
            scoring_module.SignalMatch("literal:coverage", 14, 22, 4.0),
        )
        assert ranking[1].matches == ()

    def test_zero_width_patterns_are_not_evidence(self):
        assert _scorer.score_mode(_mode("test", patterns=(r"\b",)), "x") == 0.0

    @pytest.mark.parametrize(
        "mode_kwargs,text,expected",
        [
            ({"keywords": ("testy",)}, "napraw testy", 1.0),
            ({"keywords": ("TESTY",)}, "napraw testy", 1.0),
            ({"phrases": ("napraw testy",)}, "napraw testy", 2.0),
            ({"phrases": ("napraw testy:3",)}, "napraw testy", 3.0),
            ({"patterns": (r"\bnapraw\b",)}, "napraw testy", 3.0),
            (
                {
                    "keywords": ("testy",),
                    "phrases": ("napraw",),
                    "patterns": (r"\bmodu",),
                },
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
            ({"keywords": ("test",)}, "protest x", 0.0),
            ({"keywords": ("testów",)}, "testów dla modułu", 1.0),
            ({"keywords": ("testów",)}, "testu dla modułu", 0.0),
            ({"phrases": ("a:x",)}, "a:x", 2.0),
            ({"phrases": ("a:x",)}, "a", 0.0),
            ({"patterns": (r".*",)}, "", 0.0),
            ({"patterns": ("(",)}, "x", 0.0),
            ({}, "anything at all", 0.0),
        ],
    )
    def test_score_mode(self, mode_kwargs, text, expected):
        mode = _mode("probe", **mode_kwargs)

        assert _scorer.score_mode(mode, text.lower()) == expected

    def test_score_to_similarity_saturates(self):
        assert _scorer.score_to_similarity(0.0) == 0.0
        assert _scorer.score_to_similarity(-3.0) == 0.0
        assert _scorer.score_to_similarity(1.0) == 0.5
        assert _scorer.score_to_similarity(3.0) == 0.75
        assert _scorer.score_to_similarity(9.0) == 0.9

    def test_detect_mode_returns_mode_and_score(self):
        probe = _mode("probe", keywords=("testy",))

        mode, score = _scorer.detect_mode("napraw testy", [probe])

        assert mode is probe
        assert score == 1.0

    def test_detect_mode_wins_ties_on_declaration_order(self):
        probe = _mode("probe", keywords=("testy",))
        other = _mode("other", keywords=("przejrzyj",))

        mode, score = _scorer.detect_mode("testy przejrzyj", [probe, other])
        assert mode is None
        assert score == 1.0

        mode, _ = _scorer.detect_mode("przejrzyj przejrzyj", [probe, other])
        assert mode is other

    def test_detect_mode_lowercases_the_text(self):
        probe = _mode("probe", keywords=("testy",))

        assert _scorer.detect_mode("TESTY", [probe])[0] is probe

    @pytest.mark.parametrize("modes", [[], [_mode("probe", keywords=("testy",))]])
    def test_detect_mode_without_a_match(self, modes):
        mode, score = _scorer.detect_mode("completely unrelated", modes)

        assert mode is None
        assert score == 0.0

    def test_heuristic_modes_are_the_sub_modes_only(self):
        assert HEURISTIC_MODES == ("test", "git_review", "review", "debug")

    @staticmethod
    def _regex_score(mode, text_lower):
        """Regex-based reference with independent, non-overlapping evidence."""
        if not text_lower:
            return 0.0

        weights = mode.weights if isinstance(mode.weights, dict) else {}
        candidates = []
        for keyword in mode.keywords:
            needle = keyword.strip().lower()
            if not needle:
                continue
            for match in re.finditer(r"(?<!\w)" + re.escape(needle), text_lower):
                candidates.append(
                    (
                        _scorer._keyword_weight(
                            keyword, weights, _config().heuristic_weights["keyword"]
                        ),
                        match.start(),
                        match.end(),
                        "literal:" + needle,
                    )
                )
        for phrase in mode.phrases:
            needle, weight = _scorer._signal_weight(phrase, 2.0)
            if needle:
                for match in re.finditer(r"(?<!\w)" + re.escape(needle), text_lower):
                    candidates.append(
                        (
                            weight,
                            match.start(),
                            match.end(),
                            "literal:" + needle,
                        )
                    )
        for pattern in mode.patterns:
            try:
                for match in re.finditer(pattern, text_lower):
                    if match.start() < match.end():
                        candidates.append(
                            (
                                _config().heuristic_weights["pattern"],
                                match.start(),
                                match.end(),
                                "pattern:" + pattern,
                            )
                        )
            except re.error:
                continue
        accepted = []
        used = set()
        blocked = [
            match.span()
            for match in re.finditer(
                _config().heuristic_negation_pattern, text_lower
            )
        ]
        first_matches = {}
        for weight, start, end, signal in candidates:
            if any(start < right and left < end for left, right in blocked):
                continue
            first_matches.setdefault((signal, weight), (weight, start, end, signal))
        for weight, start, end, signal in sorted(
            first_matches.values(),
            key=lambda item: (-item[0], item[1] - item[2], item[1], item[3]),
        ):
            if signal in used or weight <= 0:
                continue
            if any(start < right and left < end for left, right in blocked):
                continue
            if all(end <= left or start >= right for _, left, right in accepted):
                accepted.append((weight, start, end))
                used.add(signal)
        return sum(weight for weight, _, _ in accepted)

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "   ",
            "test",
            "protest",
            "testów",
            "testow",
            "  test   testy ",
            "napraw testy i zrób review diffa przed commitem",
            "RUN THE TESTS",
            "root cause",
            "  nie  działa  ",
            "plan działania na jutro",
            "_test",
            "8test",
            "test\u0301y",
            "slowo " * 900 + " pytest",
            "nic ciekawego " * 900,
        ],
    )
    def test_fast_path_matches_the_regex_reference(self, text):
        """The substring fast path reports exactly the regex matches."""
        text_lower = text.lower()

        for mode in _config().codex_modes:
            assert _scorer.score_mode(mode, text_lower) == self._regex_score(
                mode, text_lower
            )

    def test_word_boundary_rules_are_preserved(self):
        mode = _mode("probe", keywords=("test",))

        assert _scorer.score_mode(mode, "testów") == 1.0
        assert _scorer.score_mode(mode, "protest") == 0.0
        assert _scorer.score_mode(mode, "_test") == 0.0
        assert _scorer.score_mode(mode, "8test") == 0.0
        assert _scorer.score_mode(mode, "a_test") == 0.0

    @pytest.mark.parametrize(
        "text",
        [
            "test",
            "testów",
            "protest",
            "_test",
            "8test",
            "a_test",
            "CODE review now",
            "code-review",
            "\u2019code review",
            "kolor\u0301test",
            "te\u0301st",
            "  test  ",
            "test\ncode review",
            "",
        ],
    )
    def test_literal_scan_is_equivalent_to_a_word_start_regex(self, text):
        lowered = text.lower()

        for needle in ("test", "testow", "code review", "code", "x"):
            expected = bool(re.search(r"(?<!\w)" + re.escape(needle), lowered))

            assert _scorer._literal_matches(lowered, needle) is expected

    def test_invalid_patterns_are_skipped_and_valid_ones_count(self):
        mode = _mode("probe", keywords=("test",), patterns=("(", r"test\w*"))

        assert (
            _scorer.score_mode(mode, "testy")
            == _config().heuristic_weights["pattern"]
        )

    def test_plans_are_cached_per_signal_signature(self):
        _scorer.clear_cache()
        mode = _mode("cache", keywords=("alpha",), weights={"alpha": 2})
        same = _mode("cache", keywords=("alpha",), weights={"alpha": 2})
        changed = _mode("cache", keywords=("alpha", "beta"), weights={"alpha": 2})

        plan = _scorer._mode_plan(mode)

        assert _scorer._mode_plan(same) is plan
        assert _scorer._mode_plan(changed) is not plan

    def test_plans_are_not_shared_between_scorers(self):
        mode = _mode("isolation", keywords=("alpha",))

        assert CodexModeScorer(
            negation_pattern="", weights=_config().heuristic_weights
        )._mode_plan(mode) is not CodexModeScorer(
            negation_pattern="", weights=_config().heuristic_weights
        )._mode_plan(
            mode
        )

    def test_the_cascade_scores_through_the_injected_scorer(self):
        queried = []

        class SpyScorer(CodexModeScorer):
            def rank_modes(self, text, modes):
                queried.append(text)
                return super().rank_modes(text, modes)

        config = _config()
        classifier = CodexModeClassifier(
            config,
            scorer=SpyScorer(
                negation_pattern=config.heuristic_negation_pattern,
                weights=config.heuristic_weights,
            ),
        )

        decision = classifier.classify({}, _parse(main_payload("napraw testy")))

        assert queried == ["napraw testy"]
        assert decision.source == SOURCE_HEURISTIC


# --------------------------------------------------------------------------
# classified-text budget
# --------------------------------------------------------------------------
class TestActiveWorkPhase:
    """Clear current work can supersede the original task's topic."""

    @staticmethod
    def _call(name, arguments):
        return {
            "type": "function_call",
            "name": name,
            "call_id": "active",
            "arguments": arguments,
        }

    @pytest.mark.parametrize(
        "task,name,arguments,expected",
        [
            ("Dodaj endpoint", "exec_command", '{"cmd":"pytest tests/"}', "test"),
            (
                "Dodaj CHANGELOG dla commitów",
                "exec_command",
                '{"cmd":"git log main..HEAD"}',
                "git_review",
            ),
            (
                "Przejrzyj commity",
                "apply_patch",
                "*** Begin Patch\n*** Update File: CHANGELOG.md\n@@\n+Added endpoint\n*** End Patch",
                "implement",
            ),
        ],
    )
    def test_current_phase_precedes_user_heuristics(
        self, task, name, arguments, expected
    ):
        payload = main_payload(task)
        payload["input"].append(self._call(name, arguments))
        router = _StubRouter()
        decision = _classify_semantic(payload, router)

        assert (decision.mode, decision.source) == (expected, SOURCE_PHASE)
        assert router.calls == []

    def test_neutral_git_status_does_not_create_a_git_phase(self):
        payload = main_payload("Dodaj endpoint")
        payload["input"].append(self._call("exec_command", '{"cmd":"git status"}'))

        assert _decide(_parse(payload), payload).mode == "implement"
        assert _decide(_parse(payload), payload).source == SOURCE_FALLBACK

    def test_advertised_tools_are_not_executed_activity(self):
        request = _parse(main_payload("Dodaj endpoint"))

        assert request.has_tools
        assert request.activity == ()
        assert _decide(request).source == SOURCE_FALLBACK

    def test_custom_patch_call_is_normalized_and_changes_the_phase(self):
        payload = main_payload("Przejrzyj commity")
        payload["input"].extend(
            [
                {
                    "type": "custom_tool_call",
                    "name": "apply_patch",
                    "call_id": "patch",
                    "input": "*** Begin Patch\n*** Update File: CHANGELOG.md\n@@\n+Entry\n*** End Patch",
                },
                {
                    "type": "custom_tool_call_output",
                    "call_id": "patch",
                    "output": "Success",
                },
            ]
        )
        request = _parse(payload)

        assert request.activity[0].kind == "function_call"
        assert request.activity[1].kind == "function_call_output"
        assert request.activity[1].name == "apply_patch"
        decision = _decide(request, payload)
        assert (decision.mode, decision.source) == ("implement", SOURCE_PHASE)

    @pytest.mark.parametrize("override", ["explicit", "plan", "compaction", "title"])
    def test_structural_layers_keep_priority(self, override):
        payload = {
            "explicit": main_payload(),
            "plan": plan_payload(),
            "compaction": compaction_payload(),
            "title": title_payload(),
        }[override]
        if override == "explicit":
            payload["agent_mode"] = "review"
        payload["input"].append(self._call("exec_command", '{"cmd":"pytest"}'))
        expected = {
            "explicit": ("review", SOURCE_EXPLICIT),
            "plan": ("plan", SOURCE_COLLABORATION_MODE),
            "compaction": ("compaction", SOURCE_CLASS),
            "title": ("aux_title", SOURCE_CLASS),
        }[override]

        decision = _decide(_parse(payload), payload)

        assert (decision.mode, decision.source) == expected

    def test_unconfigured_phase_does_not_select_a_missing_mode(self):
        payload = main_payload("Dodaj endpoint")
        payload["input"].append(self._call("exec_command", '{"cmd":"pytest"}'))

        decision = _decide(
            _parse(payload), payload, config=_without_mode(_config(), "test")
        )

        assert decision.mode == "implement"

    def test_disabling_heuristics_also_disables_rule_based_phase_detection(self):
        payload = main_payload("Dodaj endpoint")
        payload["input"].append(self._call("exec_command", '{"cmd":"pytest"}'))
        config = _rebuild(_config(), heuristic_enabled=False)

        decision = _decide(_parse(payload), payload, config=config)
        assert (decision.mode, decision.source) == ("test", SOURCE_PHASE)
        config = dataclasses.replace(
            config,
            phase=dataclasses.replace(config.phase, enabled=False),
        )
        assert (
            _decide(_parse(payload), payload, config=config).source
            == SOURCE_FALLBACK
        )

    def test_phase_routing_changes_only_routing_keys_end_to_end(self):
        payload = main_payload("Przejrzyj commity")
        payload["input"].append(
            self._call(
                "apply_patch",
                "*** Begin Patch\n*** Update File: CHANGELOG.md\n@@\n+Entry\n*** End Patch",
            )
        )
        before = copy.deepcopy(payload)
        router = _StubRouter()
        plugin = CodexRoutingPlugin(config=_config(), emb_router=router)

        result = plugin.apply(payload)

        assert result is payload
        assert result["model"] == _expected_model("implement")
        assert result["routing"]["source"] == SOURCE_PHASE
        assert result["agent_mode"] == "implement"
        assert {
            key: value
            for key, value in result.items()
            if key not in ("model", "routing", "agent_mode")
        } == {key: value for key, value in before.items() if key != "model"}
        assert router.calls == []


class TestClassifyTextBudget:
    """Current command and bounded optional history are kept separate."""

    @staticmethod
    def _history(*texts):
        """Build a payload carrying one user message per entry of *texts*."""
        payload = main_payload(collaboration=None)
        payload["input"] = [_user(text) for text in texts]
        return payload

    def test_messages_are_assembled_newest_first(self):
        payload = self._history("pierwsza sprawa", "druga sprawa", "ostatnia")

        request = _parse(payload)

        assert request.latest_user_text == "ostatnia"
        assert request.user_history == ("druga sprawa", "pierwsza sprawa")
        assert request.intent_text == "ostatnia"

    def test_the_newest_message_always_survives_the_budget(self):
        payload = self._history("stara sprawa", "N" * 50)

        assert _parse(payload, max_chars=10).latest_user_text == "N" * 50

    def test_older_messages_stop_at_the_budget(self):
        payload = self._history("M1", "M2", "M3")

        assert _parse(payload, max_chars=4).latest_user_text == "M3"
        assert _parse(payload, max_chars=4).user_history == ("M2",)
        assert _parse(payload, max_chars=6).user_history == ("M2", "M1")

    def test_history_is_bounded_even_when_one_message_is_huge(self):
        payload = self._history("X" * 100, "nowe zadanie")

        assert _parse(payload, max_chars=10).user_history == ("X" * 10,)

    @pytest.mark.parametrize(
        "reply", ["tak, zrób to", "Yes, do it!", "kontynuuj", "ok"]
    )
    def test_short_confirmation_resolves_only_the_nearest_task(self, reply):
        payload = self._history("git diff", "Przygotuj testy jednostkowe", reply)
        request = _parse(payload)

        assert request.latest_user_text == reply
        assert request.intent_text == reply + "\n\nPrzygotuj testy jednostkowe"
        assert _decide(request, payload).mode == "test"

    @pytest.mark.parametrize("old_task", ["Przygotuj testy", "Przejrzyj git diff"])
    def test_new_task_does_not_inherit_old_testing_or_git_intent(self, old_task):
        payload = self._history(old_task, "Dodaj sekcję instalacji w README")
        request = _parse(payload)

        assert request.intent_text == "Dodaj sekcję instalacji w README"
        assert _decide(request, payload).mode == "implement"

    def test_independent_instruction_starting_with_yes_is_not_a_confirmation(self):
        payload = self._history("Przygotuj testy", "Tak, teraz popraw dokumentację")

        assert _parse(payload).intent_text == "Tak, teraz popraw dokumentację"

    def test_environment_context_messages_stay_out(self):
        request = _parse(main_payload("krótka odpowiedź"))

        assert request.latest_user_text == "krótka odpowiedź"
        assert "<environment_context>" not in request.latest_user_text

    def test_budget_defaults_and_comes_from_the_env(self, monkeypatch):
        assert _config().classify_max_chars == DEFAULT_CLASSIFY_MAX_CHARS
        assert (
            _load_with_env(monkeypatch, CLASSIFY_MAX_CHARS="1234").classify_max_chars
            == 1234
        )

    def test_a_non_positive_budget_is_rejected(self):
        with pytest.raises(ValueError, match="classify_max_chars"):
            _rebuild(_config(), classify_max_chars=0).validate_args()

    def test_the_plugin_passes_the_budget_to_the_parser(self, monkeypatch):
        captured = {}
        parser = plugin_module.CodexPayloadParser

        class SpyParser(parser):
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(plugin_module, "CodexPayloadParser", SpyParser)
        config = _config()
        config.classify_max_chars = 321

        _plugin(config).apply(main_payload("napraw testy"))

        assert captured == {"max_chars": 321}


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
            {
                "input": [
                    {"type": "message", "role": "user", "content": "plain string"}
                ]
            },
        ],
    )
    def test_malformed_payloads_parse(self, payload):
        assert isinstance(_parse(payload), CodexRequest)

    def test_tool_without_a_name_falls_back_to_its_type(self):
        request = _parse({"tools": [{"type": "web_search"}]})

        assert request.tool_names == ("web_search",)
        assert request.has_tools is True

    def test_string_content_is_used_as_text(self):
        request = _parse(
            {
                "input": [
                    {"type": "message", "role": "user", "content": "plain string"}
                ]
            },
        )

        assert request.latest_user_text == "plain string"

    def test_main_turn_exposes_the_decoded_identifiers(self):
        request = _parse(main_payload())

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
        request = _parse(main_payload())

        assert request.tool_names == ("exec_command", "web_search")
        assert request.has_tools is True
        assert request.reasoning_effort == "medium"
        assert request.parallel_tool_calls is True
        assert request.structured_output is False
        assert request.request_class == REQUEST_CLASS_MAIN
        assert request.collaboration_mode == COLLABORATION_MODE_DEFAULT

    def test_context_size_counts_the_content_characters(self):
        request = _parse(main_payload())

        assert request.context_chars == 421
        assert request.context_tokens == request.context_chars // 4

    def test_context_size_grows_with_the_transcript(self):
        payload = main_payload()
        before = _parse(payload).context_chars

        payload["input"].append(_user("Dodaj jeszcze jeden komunikat."))

        assert _parse(payload).context_chars > before

    def test_parsing_never_serializes_the_payload(self, monkeypatch):
        payload = main_payload()
        serialized = []
        real_dumps = json.dumps

        def spy(value, *args, **kwargs):
            serialized.append(value)
            return real_dumps(value, *args, **kwargs)

        monkeypatch.setattr(payload_module.json, "dumps", spy)

        request = _parse(payload)

        assert serialized == []
        assert request.context_chars > 0

    def test_content_length_of_the_plain_fragments(self):
        assert CodexPayloadParser._content_length(None) == 0
        assert CodexPayloadParser._content_length("abc") == 3
        assert CodexPayloadParser._content_length(12345) == 5
        assert CodexPayloadParser._content_length(True) == 4

    def test_content_length_counts_nested_values_and_keys(self):
        assert CodexPayloadParser._content_length({"ab": ["cde", 99]}) == 7
        assert CodexPayloadParser._content_length(("ab", "cd")) == 4
        assert CodexPayloadParser._content_length({1: "abc"}) == 3

    def test_content_length_survives_a_self_referencing_payload(self):
        node = {"text": "abc"}
        node["self"] = node

        assert CodexPayloadParser._content_length(node) == 11

    def test_environment_context_is_not_the_user_text(self):
        request = _parse(main_payload())

        assert request.latest_user_text == "Zaimplementuj nowy moduł eksportu."
        assert "<environment_context>" not in request.latest_user_text

    def test_title_request_is_structured_and_collaboration_free(self):
        request = _parse(title_payload())

        assert request.structured_output is True
        assert request.request_class == REQUEST_CLASS_AUX_TITLE
        assert request.collaboration_mode == ""
        assert request.tool_names == ()

    def test_compaction_request_is_tagged(self):
        request = _parse(compaction_payload())

        assert request.request_class == REQUEST_CLASS_COMPACTION

    def test_parsing_never_mutates_the_payload(self):
        payload = main_payload()
        before = copy.deepcopy(payload)

        _parse(payload)

        assert payload == before

    def test_assistant_messages_are_extracted_as_copies(self):
        payload = main_payload()
        payload["input"].append(
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "naprawiam testy"},
                    {"type": "refusal"},
                ],
            },
        )
        before = copy.deepcopy(payload)

        request = _parse(payload)

        assert payload == before
        assert request.assistant_messages == [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "naprawiam testy"}],
            }
        ]
        request.assistant_messages[0]["content"][0]["text"] = "zmieniona kopia"
        assert payload == before

    def test_old_assistant_and_tools_are_excluded_after_a_new_command(self):
        payload = main_payload("Przygotuj testy")
        payload["input"].extend(
            [
                _assistant("Teraz uruchomię pytest."),
                {
                    "type": "function_call",
                    "name": "exec_command",
                    "call_id": "old",
                    "arguments": '{"cmd":"pytest"}',
                },
                _user("Popraw dokumentację"),
                {
                    "type": "function_call_output",
                    "call_id": "old",
                    "output": "Traceback",
                },
            ]
        )
        request = _parse(payload)

        assert request.assistant_messages == []
        assert len(request.activity) == 1
        assert request.activity[0].name == ""
        assert _decide(request, payload).mode == "implement"

    def test_environment_only_message_does_not_reset_the_active_turn(self):
        payload = main_payload("Dodaj endpoint")
        payload["input"].extend(
            [
                _assistant("Teraz uruchomię pytest."),
                _user("<environment_context>cwd=/repo</environment_context>"),
            ]
        )

        assert len(_parse(payload).activity) == 1

    def test_environment_context_with_a_command_keeps_the_command(self):
        payload = main_payload(
            "<environment_context>cwd=/repo</environment_context>\nPopraw README"
        )

        assert _parse(payload).latest_user_text == "Popraw README"

    def test_active_tool_outputs_are_linked_to_their_calls(self):
        payload = main_payload("Dodaj endpoint")
        payload["input"].extend(
            [
                {
                    "type": "function_call",
                    "name": "exec_command",
                    "call_id": "new",
                    "arguments": '{"cmd":"pytest"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "new",
                    "output": "passed",
                },
            ]
        )
        before = copy.deepcopy(payload)
        request = _parse(payload)

        assert tuple(event.kind for event in request.activity) == (
            "function_call",
            "function_call_output",
        )
        assert request.activity[-1].name == "exec_command"
        assert request.activity[-1].call_id == "new"
        assert payload == before

    def test_assistant_message_without_text_is_dropped(self):
        payload = main_payload()
        payload["input"].insert(
            -1,
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text"}],
            },
        )

        assert _parse(payload).assistant_messages == []


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
        assert config.heuristic_min_score == 3.0
        assert config.heuristic_min_margin == 1.0
        assert (
            config.heuristic_negation_pattern
            == _raw_config()["settings"]["heuristic_negation_pattern"]
        )

    @pytest.mark.parametrize(
        "field",
        [
            "heuristic_min_margin",
            "heuristic_negation_pattern",
            "heuristic_weights",
        ],
    )
    def test_missing_heuristic_settings_are_reported(self, field):
        raw = _raw_config()
        del raw["settings"][field]

        with pytest.raises(KeyError, match=field):
            CodexRoutingConfig._from_raw(raw)

    def test_custom_heuristic_weights_reach_classifier(self):
        raw = _raw_config()
        raw["settings"]["heuristic_weights"] = {
            "keyword": 4.0,
            "phrase": 5.0,
            "pattern": 7.0,
        }
        config = CodexRoutingConfig._from_raw(raw)
        config.validate_args()
        classifier = CodexModeClassifier(config)
        mode = CodexMode(
            name="test",
            model_name="test-model",
            description="test",
            examples=(),
            patterns=(r"\bunique\b",),
        )

        assert config.heuristic_weights["keyword"] == 4.0
        assert classifier._scorer.score_mode(mode, "unique") == 7.0
        assert (
            classifier._scorer.score_mode(
                _mode("test", keywords=("unique",)), "unique"
            )
            == 4.0
        )
        assert (
            classifier._scorer.score_mode(
                _mode("test", phrases=("unique",)), "unique"
            )
            == 5.0
        )

    @pytest.mark.parametrize(
        "weights",
        [
            None,
            [],
            {"unknown": 2},
            {"pattern": 3},
            {"keyword": -1, "phrase": 2, "pattern": 3},
            {"keyword": 1, "phrase": float("nan"), "pattern": 3},
            {"keyword": 1, "phrase": 2, "pattern": True},
            {"keyword": 1, "phrase": 2, "pattern": "3"},
        ],
    )
    def test_invalid_heuristic_weights_are_rejected(self, weights):
        raw = _raw_config()
        raw["settings"]["heuristic_weights"] = weights

        with pytest.raises(ValueError, match="heuristic_weights"):
            CodexRoutingConfig._from_raw(raw)

    def test_custom_config_does_not_read_packaged_rules(self, monkeypatch):
        raw = _raw_config()
        raw["settings"]["heuristic_negation_pattern"] = r"ignored:.*"
        monkeypatch.setattr(
            pathlib.Path,
            "open",
            lambda *args, **kwargs: pytest.fail("Unexpected file read"),
        )

        config = CodexRoutingConfig._from_raw(raw)
        classifier = CodexModeClassifier(config)
        mode = _mode("test", phrases=("run tests:3",))

        assert classifier._scorer.score_mode(mode, "ignored: run tests") == 0.0
        assert classifier._scorer.score_mode(mode, "do not run tests") == 3.0

    @pytest.mark.parametrize(
        "field,value",
        [
            ("heuristic_min_score", -1),
            ("heuristic_min_score", float("nan")),
            ("heuristic_min_margin", -1),
            ("heuristic_min_margin", float("inf")),
            ("heuristic_min_margin", float("nan")),
            ("heuristic_negation_pattern", None),
            ("heuristic_negation_pattern", "("),
        ],
    )
    def test_invalid_heuristic_settings_are_rejected(self, field, value):
        config = _rebuild(_config(), **{field: value})

        with pytest.raises(ValueError, match="heuristic"):
            config.validate_args()

    def test_custom_heuristic_settings_are_loaded_from_json(self):
        raw = _raw_config()
        raw["settings"]["heuristic_min_margin"] = 2.5
        raw["settings"]["heuristic_negation_pattern"] = r"ignored:.*"

        config = CodexRoutingConfig._from_raw(raw)
        config.validate_args()

        assert config.heuristic_min_margin == 2.5
        assert config.heuristic_negation_pattern == r"ignored:.*"

    def test_shipped_mode_order(self):
        config = _config()

        assert config.mode_names == [
            "plan",
            "implement",
            "test",
            "git_review",
            "review",
            "debug",
            "aux_title",
            "compaction",
        ]
        assert list(config.mode_by_name.keys()) == config.mode_names

    @pytest.mark.parametrize(
        "mode_name,model_name",
        [
            ("plan", "qwen/Qwen3.8-Flash-Next"),
            ("aux_title", "qwen/Qwen3.8-27B"),
            ("compaction", "qwen/Qwen3.8-27B"),
            ("implement", "qwen/Qwen3.8-Flash-Next"),
            ("test", "qwen/Qwen3.8-27B"),
            ("git_review", "qwen/Qwen3.8-27B"),
            ("review", "qwen/Qwen3.8-Flash-Next"),
            ("debug", "qwen/Qwen3.8-Flash-Next"),
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

    def test_class_routed_modes_are_the_non_conversation_classes(self):
        assert CLASS_ROUTED_MODES == (
            REQUEST_CLASS_COMPACTION,
            REQUEST_CLASS_AUX_TITLE,
        )

    def test_sub_modes_carry_polish_and_english_signals(self):
        config = _config()

        for mode_name in (
            "plan",
            "implement",
            "test",
            "git_review",
            "review",
            "debug",
        ):
            mode = config.mode_by_name[mode_name]
            assert mode.keywords
            assert mode.phrases
            assert mode.patterns

    def test_semantic_defaults(self):
        config = _config()

        assert config.semantic_enabled is True
        assert config.similarity_threshold == 0.51
        assert config.top_k == 3
        assert config.chunk_size == 256
        assert config.chunk_overlap == 64
        assert config.embedding_model
        assert config.vector_store_path == ""

    @pytest.mark.parametrize(
        "mutate,fragment",
        [
            (
                lambda raw: raw.pop("settings"),
                "Missing required top-level key 'settings' in config.",
            ),
            (
                lambda raw: raw.pop("codex_modes"),
                "Missing required top-level key 'codex_modes' in config.",
            ),
            (
                lambda raw: raw["settings"].pop("trigger_model"),
                "Missing required field 'trigger_model' in settings.",
            ),
            (
                lambda raw: raw["settings"].pop("fallback_mode"),
                "Missing required field 'fallback_mode' in settings.",
            ),
            (
                lambda raw: raw["codex_modes"][0].pop("name"),
                "Missing required field 'name' in codex_modes[0].",
            ),
            (
                lambda raw: raw["codex_modes"][0].pop("model_name"),
                "Missing required field 'model_name' in codex_modes[0].",
            ),
            (
                lambda raw: raw["codex_modes"][0].pop("description"),
                "Missing required field 'description' in codex_modes[0].",
            ),
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
            (
                {"fallback_mode": "nope"},
                "fallback_mode 'nope' is not a defined Codex mode",
            ),
            (
                {"semantic_enabled": True, "embedding_model": ""},
                "no embedding_model configured",
            ),
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

        with pytest.raises(
            ValueError, match=re.escape("duplicate Codex mode names ['plan']")
        ):
            config.validate_args()

    def test_embedding_model_is_only_required_when_semantic_is_enabled(self):
        config = _rebuild(_config(), semantic_enabled=False, embedding_model="")

        config.validate_args()

    def test_mode_without_a_model_is_valid(self):
        modes = tuple(
            dataclasses.replace(mode, model_name="") if mode.name == "plan" else mode
            for mode in _config().codex_modes
        )

        _rebuild(_config(), codex_modes=modes).validate_args()


# --------------------------------------------------------------------------
# startup lint of the configured signals
# --------------------------------------------------------------------------
class TestSignalLint:
    """Signals that can never score are reported instead of silently dropped."""

    @staticmethod
    def _with_patterns(patterns):
        """Return the bundled config with *patterns* added to the ``plan`` mode."""
        modes = tuple(
            (
                dataclasses.replace(mode, patterns=mode.patterns + patterns)
                if mode.name == "plan"
                else mode
            )
            for mode in _config().codex_modes
        )
        return _rebuild(_config(), codex_modes=modes)

    def test_the_shipped_config_lints_clean(self):
        logger = _CaptureLogger()

        _config().lint_signals(logger)

        assert logger.records["warning"] == []

    def test_an_uncompilable_pattern_is_reported(self):
        logger = _CaptureLogger()

        self._with_patterns(("(",)).lint_signals(logger)

        assert "unusable pattern" in logger.joined()

    def test_an_uppercase_pattern_is_reported(self):
        logger = _CaptureLogger()

        self._with_patterns((r"\bREADME\b",)).lint_signals(logger)

        assert "can never match" in logger.joined()

    def test_usable_signals_are_not_reported(self):
        logger = _CaptureLogger()

        self._with_patterns((r"readme\w*",)).lint_signals(logger)

        assert logger.records["warning"] == []

    def test_a_mode_without_a_model_is_reported(self):
        logger = _CaptureLogger()
        config = _rebuild(_config(), codex_modes=(_mode("probe", model_name=""),))

        config.lint_signals(logger)

        assert "no model_name" in logger.joined()

    def test_an_overlap_not_below_chunk_size_is_reported(self):
        logger = _CaptureLogger()

        _rebuild(_config(), chunk_size=128, chunk_overlap=128).lint_signals(logger)

        assert "chunk_overlap" in logger.joined()

    def test_the_lint_needs_no_logger_and_never_raises(self):
        self._with_patterns(("(",)).lint_signals()

    def test_the_plugin_lints_when_constructed(self):
        logger = _CaptureLogger()
        config = _rebuild(self._with_patterns(("(",)), semantic_enabled=False)

        CodexRoutingPlugin(logger=logger, config=config)

        assert "unusable pattern" in logger.joined()


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
        assert (
            _load_with_env(monkeypatch, TRIGGER="auto_x").trigger_model == "auto_x"
        )

    def test_single_mode_model_is_overridden(self, monkeypatch):
        config = _load_with_env(monkeypatch, MODEL_PLAN="m/plan")

        assert config.mode_by_name["plan"].model_name == "m/plan"

    def test_batch_model_overrides_ignore_unknown_modes(self, monkeypatch):
        config = _load_with_env(monkeypatch, MODELS="plan=x|test=y|nope=z")

        assert config.mode_by_name["plan"].model_name == "x"
        assert config.mode_by_name["test"].model_name == "y"
        assert len(config.codex_modes) == 8

    def test_mode_whitelist(self, monkeypatch):
        config = _load_with_env(monkeypatch, MODES="plan|implement")

        assert config.mode_names == ["plan", "implement"]

    def test_unknown_mode_whitelist_is_ignored(self, monkeypatch):
        config = _load_with_env(monkeypatch, MODES="nope|nada")

        assert len(config.codex_modes) == 8

    def test_fallback_mode_is_overridden(self, monkeypatch):
        config = _load_with_env(monkeypatch, FALLBACK_MODE="review")

        assert config.fallback_mode == "review"

    def test_heuristic_can_be_disabled(self, monkeypatch):
        config = _load_with_env(monkeypatch, HEURISTIC_ENABLED="false")

        assert config.heuristic_enabled is False

    def test_heuristic_threshold_is_overridden(self, monkeypatch):
        config = _load_with_env(monkeypatch, HEURISTIC_MIN_SCORE="4.5")

        assert config.heuristic_min_score == 4.5

    def test_heuristic_margin_is_overridden(self, monkeypatch):
        config = _load_with_env(monkeypatch, HEURISTIC_MIN_MARGIN="2.5")

        assert config.heuristic_min_margin == 2.5

    def test_mode_keywords_are_overridden(self, monkeypatch):
        config = _load_with_env(monkeypatch, MODE_test_KEYWORDS="a|b")

        assert config.mode_by_name["test"].keywords == ("a", "b")

    def test_keywords_of_an_unknown_mode_are_ignored(self, monkeypatch):
        untouched = _config().mode_by_name["test"].keywords

        config = _load_with_env(monkeypatch, MODE_nope_KEYWORDS="a|b")

        assert config.mode_by_name["test"].keywords == untouched
        assert "nope" not in config.mode_by_name

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
        result = CodexRoutingPlugin(
            logger=None,
            config=_rebuild(_config(), semantic_enabled=False),
        ).apply(payload)

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
    """The plugin is discoverable in the shared utils registry."""

    def test_codex_plugin_is_registered(self):
        assert MAIN_UTILS_REGISTRY["agentic_routing_codex"] is CodexRoutingPlugin

    def test_removed_agentic_plugin_is_not_registered(self):
        assert "agentic_routing" not in MAIN_UTILS_REGISTRY

    def test_registered_name_matches_the_class(self):
        assert CodexRoutingPlugin.name == "agentic_routing_codex"


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------
class TestDeterminism:
    """Same request, same decision, same similarity — every time."""

    def test_classification_is_stable_across_repeats(self):
        payload = main_payload("Uruchom testy i napraw błędy.")
        request = _parse(payload)

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
        parsed = [_parse(payload) for _ in range(3)]

        assert parsed[0] == parsed[1] == parsed[2]

    def test_similarity_is_reproducible_for_the_same_text(self):
        similarities = set()
        for _ in range(4):
            payload = main_payload("Napraw testy jednostkowe.")
            similarities.add(_decide(_parse(payload), payload).similarity)

        assert len(similarities) == 1


# --------------------------------------------------------------------------
# semantic similarity (embeddings + cosine)
# --------------------------------------------------------------------------
class _StubRouter:
    """Stand in for the BiEncoder + FAISS router, without any ML dependency."""

    def __init__(
        self,
        target_name="test",
        similarity=0.9,
        all_scores=None,
        fail=False,
        raw_result=None,
    ):
        self.target_name = target_name
        self.similarity = similarity
        self.all_scores = all_scores
        self.fail = fail
        self.raw_result = raw_result
        self.calls = []

    def route(self, request):
        """Record the query and replay a router-shaped result."""
        self.calls.append(getattr(request, "latest_user_text", request))
        if self.fail:
            raise RuntimeError("index unavailable")
        if self.raw_result is not None:
            return self.raw_result
        result = {
            "model_name": "stub-model",
            "target_name": self.target_name,
            "similarity": self.similarity,
        }
        result["all_scores"] = (
            self.all_scores
            if self.all_scores is not None
            else [
                {
                    "target": mode.name,
                    "similarity": (
                        self.similarity if mode.name == self.target_name else -0.5
                    ),
                }
                for mode in _config().codex_modes
                if mode.name not in CLASS_ROUTED_MODES
            ]
        )
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


def _semantic_layer(
    router,
    threshold=0.5,
    modes=None,
    logger=None,
    intent_max_chars=None,
    phase_max_chars=None,
):
    """Build a semantic layer over *router* for the shipped mode table."""
    config = _config()
    return CodexSemanticLayer(
        router,
        threshold,
        modes if modes is not None else config.mode_by_name,
        logger,
        min_margin=config.semantic_min_margin,
        intent_max_chars=(
            config.semantic_intent_max_chars
            if intent_max_chars is None
            else intent_max_chars
        ),
        phase_max_chars=(
            config.semantic_phase_max_chars
            if phase_max_chars is None
            else phase_max_chars
        ),
    )


def _classify_semantic(payload, router, threshold=0.5, logger=None):
    """Classify *payload* with a stub-backed semantic layer attached."""
    request = _parse(payload)
    classifier = CodexModeClassifier(
        _config(), semantic=_semantic_layer(router, threshold, logger=logger)
    )
    return classifier.classify(payload, request)


class TestSemanticSimilarity:
    """``similarity`` is embedding cosine, shared with ``agentic_routing``."""

    def test_layer_is_available_only_when_a_router_is_present(self):
        assert _semantic_layer(_StubRouter()).available is True
        assert _semantic_layer(None).available is False

    def test_empty_text_never_reaches_the_router(self):
        router = _StubRouter()
        request = CodexRequest(latest_user_text="")

        assert _semantic_layer(router).route(request) is None
        assert router.calls == []

    def test_router_result_must_be_a_mapping(self):
        router = _StubRouter(raw_result="not-a-dict")
        request = CodexRequest(latest_user_text="napraw testy")

        assert _semantic_layer(router).route(request) is None
        assert router.calls == ["napraw testy"]

    def test_threshold_acceptance_is_inclusive(self):
        mode, similarity = _semantic_layer(_StubRouter()).accept(
            _StubRouter("test", 0.5).route("query")
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

    def test_a_keyword_win_never_asks_the_vector_store(self):
        router = _StubRouter(
            "test", 0.9, all_scores=[{"target": "test", "similarity": 0.77}]
        )

        decision = _classify_semantic(main_payload("napraw testy"), router)

        assert decision.mode == "test"
        assert decision.source == SOURCE_HEURISTIC
        assert decision.score == 3.0
        assert decision.similarity == _scorer.score_to_similarity(3.0)
        assert router.calls == []

    @pytest.mark.parametrize("scores", [(3.0, 3.0), (3.5, 3.0)])
    def test_tied_and_close_scores_defer_to_semantics_once(self, scores):
        config = _rebuild(
            _config(),
            codex_modes=(
                _mode("implement"),
                _mode("test", keywords=("alpha",), weights={"alpha": scores[0]}),
                _mode("review", keywords=("beta",), weights={"beta": scores[1]}),
            ),
        )
        router = _StubRouter(
            "review",
            0.9,
            all_scores=[
                {
                    "target": mode.name,
                    "similarity": 0.9 if mode.name == "review" else 0.2,
                }
                for mode in config.codex_modes
            ],
        )
        payload = main_payload("alpha beta")
        classifier = CodexModeClassifier(
            config, semantic=_semantic_layer(router, modes=config.mode_by_name)
        )

        decision = classifier.classify(payload, _parse(payload))

        assert (decision.mode, decision.source) == ("review", SOURCE_SEMANTIC)
        assert router.calls == ["alpha beta"]

    def test_negated_test_request_reaches_semantics_without_a_test_veto(self):
        text = "Nie uruchamiaj testów, popraw dokumentację"
        router = _StubRouter("implement", 0.9)

        decision = _classify_semantic(main_payload(text), router)

        assert (decision.mode, decision.source) == ("implement", SOURCE_SEMANTIC)
        assert router.calls == [text]

    def test_classifier_uses_configured_negation_rule(self):
        config = _rebuild(_config(), heuristic_negation_pattern=r"ignored:.*")

        decision = _decide(
            CodexRequest(latest_user_text="ignored: run the tests"), config=config
        )

        assert (decision.mode, decision.source) == ("implement", SOURCE_FALLBACK)

    def test_semantic_match_wins_when_keywords_are_silent(self):
        router = _StubRouter("debug", 0.81)

        decision = _classify_semantic(
            main_payload("wyrenderuj pusty stan w widoku"),
            router,
        )

        assert decision.mode == "debug"
        assert decision.source == SOURCE_SEMANTIC
        assert decision.similarity == 0.81

    def test_below_threshold_falls_back_but_still_reports_cosine(self):
        router = _StubRouter(
            "debug", 0.2, all_scores=[{"target": "implement", "similarity": 0.31}]
        )

        decision = _classify_semantic(
            main_payload("wyrenderuj pusty stan w widoku"),
            router,
        )

        assert decision.mode == "implement"
        assert decision.source == SOURCE_FALLBACK
        assert decision.score == 0.0
        assert decision.similarity == 0.31

    def test_router_failure_is_fail_open(self):
        logger = _CaptureLogger()

        decision = _classify_semantic(
            main_payload("wyrenderuj pusty stan w widoku"),
            _StubRouter(fail=True),
            logger=logger,
        )

        assert decision.mode == "implement"
        assert decision.source == SOURCE_FALLBACK
        assert decision.similarity == 0.0
        assert "semantic lookup failed" in logger.joined()

    def test_context_built_from_the_latest_agent_utterance_only(self):
        payload = main_payload("wyrenderuj pusty stan w widoku", collaboration=None)
        payload["input"].append(_assistant("stary temat: przegląd kodu"))
        payload["input"].append(_assistant("przechodzę do testów"))
        router = _StubRouter()

        result = _semantic_layer(router).route(_parse(payload))

        assert result["target_name"] == "test"
        assert router.calls == [
            "wyrenderuj pusty stan w widoku\nprzechodzę do testów"
        ]

    def test_new_task_excludes_old_assistant_and_user_from_semantic_query(self):
        payload = main_payload("Przygotuj testy")
        payload["input"].extend(
            [_assistant("Testy są gotowe"), _user("Popraw README")]
        )
        router = _StubRouter()

        _semantic_layer(router).route(_parse(payload))

        assert router.calls == ["Popraw README"]

    def test_phase_has_a_reserved_budget_after_a_long_command(self):
        payload = main_payload("X" * 1000)
        payload["input"].append(_assistant("Sprawdzam zgodność konfiguracji."))
        request = _parse(payload, max_chars=100)
        router = _StubRouter()

        _semantic_layer(router, intent_max_chars=50, phase_max_chars=49).route(
            request
        )

        assert len(router.calls[0]) <= 100
        assert router.calls[0].endswith("Sprawdzam zgodność konfiguracji.")
        assert request.latest_user_text == "X" * 1000

    def test_semantic_query_describes_the_action_without_the_tool_output(self):
        payload = main_payload("Dodaj endpoint")
        payload["input"].extend(
            [
                TestActiveWorkPhase._call("exec_command", '{"cmd":"git status"}'),
                {
                    "type": "function_call_output",
                    "call_id": "active",
                    "output": "clean",
                },
            ]
        )
        router = _StubRouter()

        _semantic_layer(router).route(_parse(payload))

        assert router.calls == ["Dodaj endpoint\ncalled exec_command"]
        assert "clean" not in router.calls[0]
        assert '"cmd"' not in router.calls[0]

    def test_long_assistant_does_not_exclude_the_latest_tool_from_semantics(self):
        payload = main_payload("Dodaj endpoint")
        payload["input"].extend(
            [
                _assistant("X" * 1000),
                TestActiveWorkPhase._call("exec_command", '{"cmd":"git status"}'),
            ]
        )
        router = _StubRouter()

        _semantic_layer(router, intent_max_chars=100, phase_max_chars=99).route(
            _parse(payload, max_chars=200)
        )

        assert len(router.calls[0]) <= 200
        assert "exec_command" in router.calls[0]

    def test_malformed_assistant_message_does_not_break_routing(self):
        payload = main_payload("wyrenderuj pusty stan w widoku")
        payload["input"].append(_assistant("dobrze"))
        payload["input"].append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text"}],
            }
        )
        router = _StubRouter("debug", 0.81)
        logger = _CaptureLogger()

        decision = _classify_semantic(payload, router, logger=logger)

        assert decision.mode == "debug"
        assert decision.source == SOURCE_SEMANTIC
        assert "context building failed" not in logger.joined()

    def test_broken_context_building_is_fail_open(self):
        logger = _CaptureLogger()
        payload = main_payload("wyrenderuj pusty stan w widoku")
        router = _StubRouter()
        layer = _semantic_layer(router, logger=logger)

        def boom(request, intent_max_chars, phase_max_chars):
            raise ValueError("nope")

        layer.__dict__["_build_semantic_parts"] = boom

        assert layer.route(_parse(payload)) is None
        assert router.calls == []
        assert "context building failed" in logger.joined()

    def test_similarity_without_a_layer_is_derived_from_the_score(self):
        payload = main_payload("napraw testy")

        decision = _decide(_parse(payload), payload)

        assert decision.source == SOURCE_HEURISTIC
        assert decision.similarity == _scorer.score_to_similarity(decision.score)

    def test_below_threshold_match_is_logged(self):
        logger = _CaptureLogger()

        _classify_semantic(
            main_payload("wyrenderuj pusty stan w widoku"),
            _StubRouter("debug", 0.2),
            threshold=0.5,
            logger=logger,
        )

        assert "below threshold" in logger.joined()

    def test_the_text_is_embedded_at_most_once_per_request(self):
        router = _StubRouter(
            "test", 0.9, all_scores=[{"target": "test", "similarity": 0.9}]
        )

        _classify_semantic(main_payload("wyrenderuj pusty stan w widoku"), router)

        assert len(router.calls) == 1

    def test_resolve_combines_route_and_accept(self):
        request = CodexRequest(latest_user_text="przejrzyj moduł")
        mode, similarity = _semantic_layer(_StubRouter("review", 0.66)).resolve(
            request
        )

        assert mode.name == "review"
        assert similarity == 0.66

    def test_plugin_reports_the_cosine_similarity_end_to_end(self):
        plugin = CodexRoutingPlugin(
            logger=None, config=_config(), emb_router=_StubRouter("debug", 0.81)
        )

        result = plugin.apply(main_payload("wyrenderuj pusty stan w widoku"))

        assert result["model"] == _expected_model("debug")
        assert result["routing"] == {
            "plugin": "agentic_routing_codex",
            "similarity": 0.81,
            "agent_mode": "debug",
            "source": SOURCE_SEMANTIC,
            "semantic": "accepted",
            "codex_class": REQUEST_CLASS_MAIN,
            "collaboration_mode": COLLABORATION_MODE_DEFAULT,
            "request_kind": "turn",
            "thread_id": "thread-1",
            "turn_id": "turn-7",
        }
        assert set(result["routing"]) == _ROUTING_KEYS | {"semantic"}

    def test_injected_layer_takes_precedence_over_the_router_argument(self):
        strong, weak = _StubRouter("debug", 0.81), _StubRouter("review", 0.99)
        plugin = CodexRoutingPlugin(
            logger=None,
            config=_config(),
            emb_router=weak,
            semantic=_semantic_layer(strong),
        )

        result = plugin.apply(main_payload("wyrenderuj pusty stan w widoku"))

        assert result["routing"]["agent_mode"] == "debug"
        assert weak.calls == []
        assert strong.calls == ["wyrenderuj pusty stan w widoku"]

    def test_semantic_routing_stays_off_when_disabled_by_env(self, monkeypatch):
        monkeypatch.setenv(f"{_PREFIX}SEMANTIC_ENABLED", "false")

        plugin = CodexRoutingPlugin(logger=None, emb_router=_StubRouter())

        assert plugin._semantic is None

    def test_plugin_without_semantic_dependencies_still_routes(self):
        plugin = CodexRoutingPlugin(emb_router=_StubRouter())

        assert plugin._semantic is not None

        result = plugin.apply(main_payload("napraw testy"))

        assert result["model"] == _expected_model("test")
        assert result["routing"]["source"] in (SOURCE_HEURISTIC, SOURCE_SEMANTIC)

    def test_router_is_built_with_the_shared_embedding_factory(
        self, monkeypatch, tmp_path
    ):
        captured = {}
        router = _StubRouter("test", 0.9)

        def fake_build_router(**kwargs):
            captured.update(kwargs)
            return router

        monkeypatch.setattr(
            plugin_module, "build_embedding_router", fake_build_router
        )
        config = _config()
        config.vector_store_path = str(tmp_path)

        plugin = CodexRoutingPlugin(logger=None, config=config)

        assert plugin._semantic is not None
        assert captured["embedding_model"] == config.embedding_model
        assert captured["chunk_size"] == config.chunk_size
        assert captured["chunk_overlap"] == config.chunk_overlap
        assert captured["top_k"] == config.top_k
        assert captured["aggregation"] == config.semantic_aggregation
        assert captured["persist_dir"] == str(tmp_path)
        assert tuple(m.name for m in captured["routing_targets"]) == (
            "plan",
            "implement",
            "test",
            "git_review",
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
        ] == pytest.approx(3.0 / 4.0)
        assert (
            plugin.apply(main_payload("Dodaj pole"))["routing"]["semantic"]
            == "unavailable"
        )
