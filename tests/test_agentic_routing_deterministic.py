"""
Unit tests for the deterministic layers of AgenticRoutingPlugin.

The routing cascade is ordered so that deterministic decisions are taken
before any semantic (embedding based) matching happens:

    explicit -> rules -> affinity -> heuristic -> semantic -> fallback

These tests cover the deterministic building blocks in isolation
(signals, rules, capabilities, session affinity, heuristics) plus the
plugin-level ordering guarantees. No ML dependencies are required.

Run with:
    pytest tests/test_agentic_routing_deterministic.py -v
"""

import json
import os
import pathlib
import sys

# Ensure the package root is on sys.path so imports work without install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from concurrent.futures import ThreadPoolExecutor

import pytest

from llm_router_plugins.utils.routing.agentic_routing import (
    AgenticRoutingPlugin,
    SemanticLayer,
    SessionAffinityCache,
    describe_rule,
    detect_heuristic,
    escalate,
    filter_modes,
    match_rule,
    missing_capabilities,
    parse_rules,
    requirements_from_signals,
    satisfies,
    scored_modes,
)
from llm_router_plugins.utils.routing.agentic_routing import session_affinity
from llm_router_plugins.utils.routing.agentic_routing.heuristics import (
    PATTERN_WEIGHT,
    score_mode,
    score_to_similarity,
)
from llm_router_plugins.utils.routing.agentic_routing.config import AgentMode
from llm_router_plugins.utils.routing.agentic_routing.rules import RoutingRule
from llm_router_plugins.utils.routing.agentic_routing.signals import RequestSignals

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


class _CaptureLogger:
    """Minimal logger double capturing formatted messages per level."""

    def __init__(self) -> None:
        self.records: dict = {"warning": [], "info": [], "debug": []}

    def _log(self, level: str, msg, *args) -> None:
        text = str(msg)
        if args:
            text = text % args
        self.records[level].append(text)

    def warning(self, msg, *args) -> None:
        self._log("warning", msg, *args)

    def info(self, msg, *args) -> None:
        self._log("info", msg, *args)

    def debug(self, msg, *args) -> None:
        self._log("debug", msg, *args)

    def __getitem__(self, level: str) -> list:
        return self.records[level]

    def joined(self) -> str:
        return "\n".join(
            chunk for level in self.records for chunk in self.records[level]
        )


def _mode(name: str, capabilities=None, model_name=None, **kwargs) -> AgentMode:
    """Build an AgentMode with the smallest possible footprint."""
    return AgentMode(
        name=name,
        model_name=model_name or "model_%s" % name,
        description=name,
        examples=(),
        capabilities=dict(capabilities or {}),
        **kwargs,
    )


@pytest.fixture(autouse=True)
def clean_routing_env():
    """Clear all routing-related env vars before and after each test."""
    kept: dict = {}
    for key in list(os.environ.keys()):
        if key.startswith("LLM_ROUTER_ROUTING"):
            kept[key] = os.environ.pop(key)
    yield
    for key, val in kept.items():
        os.environ[key] = val


def _make_plugin(**env: str) -> AgenticRoutingPlugin:
    """Set env vars (always disabling ML) and create a plugin instance."""
    os.environ["%sSEMANTIC_ENABLED" % _PREFIX] = "false"
    for key, value in env.items():
        os.environ["%s%s" % (_PREFIX, key)] = value
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


def _routing(result: dict) -> dict:
    return result["routing"]


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


def _make_config_plugin(**overrides) -> AgenticRoutingPlugin:
    """Create a plugin driven by a hand written raw config string."""
    os.environ["%sCONFIG" % _PREFIX] = _minimal_raw_config(**overrides)
    return AgenticRoutingPlugin()


class _StubRouter:
    """Router stub recording every ``route()`` call it receives."""

    def __init__(self, target_name: str, similarity: float) -> None:
        self.target_name = target_name
        self.similarity = similarity
        self.calls: list = []

    def route(self, text: str) -> dict:
        self.calls.append(text)
        return {
            "model_name": _MODEL_FOR_MODE.get(self.target_name, "model_stub"),
            "target_name": self.target_name,
            "similarity": self.similarity,
            "all_scores": {},
        }


def _make_plugin_with_router(router, **env: str):
    """Create a plugin with an injected router and semantic matching on."""
    os.environ["%sSEMANTIC_ENABLED" % _PREFIX] = "true"
    for key, value in env.items():
        os.environ["%s%s" % (_PREFIX, key)] = value
    return AgenticRoutingPlugin(router=router), router


# ===================== signals: deterministic request extraction =====================
def test_signals_from_agent_aware_payload():
    """The documented agent-aware payload is extracted without any ML."""
    payload = {
        "agent": {"name": "CodeReview"},
        "session_id": " s-42 ",
        "task": "Review",
        "tools": [{"type": "function"}, {"type": "function"}],
        "reasoning": "high",
        "context_tokens": 24000,
        "metadata": {"tenant": "acme"},
    }
    signals = RequestSignals.from_payload(payload, "please review")
    assert signals.agent == "codereview"
    assert signals.session_id == "s-42"
    assert signals.task == "review"
    assert signals.tools is True
    assert signals.tool_count == 2
    assert signals.reasoning is True
    assert signals.context_tokens == 24000
    assert signals.metadata["tenant"] == "acme"


def test_signals_task_is_never_read_from_text():
    """Free text must not create a task signal - tasks come from the payload."""
    signals = RequestSignals.from_payload({}, "review my pull request")
    assert signals.task == ""


def test_signals_precedence_top_level_over_metadata_and_agent():
    """Top-level keys win over metadata, which wins over the agent object."""
    payload = {
        "task": "coding",
        "metadata": {"task": "planning"},
        "agent": {"task": "research"},
    }
    assert RequestSignals.from_payload(payload).task == "coding"

    payload = {"metadata": {"task": "planning"}, "agent": {"task": "research"}}
    assert RequestSignals.from_payload(payload).task == "planning"


def test_signals_agent_name_normalization_and_fallbacks():
    """Agent identifiers are normalized and read from several shapes."""
    assert RequestSignals.from_payload({"agent": "Code-Review "}).agent == (
        "code_review"
    )
    assert RequestSignals.from_payload({"agent": {"id": "Ops"}}).agent == "ops"
    assert RequestSignals.from_payload({"agent_name": "Code-Review"}).agent == (
        "code_review"
    )
    assert RequestSignals.from_payload({"metadata": {"agent": "BATCH"}}).agent == (
        "batch"
    )


def test_signals_session_id_aliases():
    """session_id falls back to conversation_id and thread_id."""
    assert RequestSignals.from_payload({"conversation_id": "c1"}).session_id == "c1"
    assert RequestSignals.from_payload({"thread_id": "t1"}).session_id == "t1"
    assert (
        RequestSignals.from_payload(
            {"session_id": "s1", "conversation_id": "c1"}
        ).session_id
        == "s1"
    )


@pytest.mark.parametrize(
    "tools,expected_active,expected_count",
    [
        (True, True, 1),
        (False, False, 0),
        ([], False, 0),
        ([{}, {}], True, 2),
        ({"a": 1, "b": 2, "c": 3}, True, 3),
        ("yes", True, 1),
    ],
)
def test_signals_tools_shapes(tools, expected_active, expected_count):
    """Tool declarations of any shape collapse to a flag plus a count."""
    signals = RequestSignals.from_payload({"tools": tools})
    assert signals.tools is expected_active
    assert signals.tool_count == expected_count


def test_signals_tool_count_only_raises_the_count():
    """A bare count enables the flag and can lift an existing count."""
    signals = RequestSignals.from_payload({"tool_count": 5})
    assert signals.tools is True
    assert signals.tool_count == 5

    signals = RequestSignals.from_payload({"tools": [{}, {}], "tool_count": 1})
    assert signals.tool_count == 2


@pytest.mark.parametrize(
    "value,expected",
    [
        ("high", True),
        ("none", False),
        ("", False),
        (False, False),
        (True, True),
    ],
)
def test_signals_reasoning_values(value, expected):
    """Reasoning hints map to a boolean flag."""
    assert RequestSignals.from_payload({"reasoning": value}).reasoning is expected


def test_signals_reasoning_alias():
    """thinking and reasoning_effort are accepted aliases."""
    assert RequestSignals.from_payload({"thinking": True}).reasoning is True
    assert (
        RequestSignals.from_payload({"reasoning_effort": "medium"}).reasoning is True
    )


def test_signals_structured_output_values():
    """response_format output formats are reduced to a boolean."""
    assert (
        RequestSignals.from_payload(
            {"response_format": {"type": "json_schema"}}
        ).structured_output
        is True
    )
    assert (
        RequestSignals.from_payload({"response_format": "text"}).structured_output
        is False
    )
    assert (
        RequestSignals.from_payload({"output_format": "json"}).structured_output
        is True
    )


def test_signals_parallel_tools_alias():
    """parallel_tool_calls and parallel_tools both set the flag."""
    assert (
        RequestSignals.from_payload({"parallel_tool_calls": True}).parallel_tools
        is True
    )
    assert (
        RequestSignals.from_payload({"parallel_tools": False}).parallel_tools
        is False
    )


def test_signals_vision_from_image_content_part():
    """An image content part in the last message enables vision."""
    payload = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "what is this"}]},
        ]
    }
    assert RequestSignals.from_payload(payload).vision is False

    payload["messages"].append(
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "http://x/y"}}],
        }
    )
    assert RequestSignals.from_payload(payload).vision is True
    assert RequestSignals.from_payload({"vision": True}).vision is True


def test_signals_context_tokens_explicit_value():
    """An explicit context_tokens value is used verbatim."""
    assert RequestSignals.from_payload({"context_tokens": 900}).context_tokens == 900
    assert RequestSignals.from_payload(
        {"context_tokens": "1200"}
    ).context_tokens == (1200)
    assert RequestSignals.from_payload({"context_tokens": -5}).context_tokens != -5
    assert RequestSignals.from_payload({"context_tokens": True}).context_tokens == 0


def test_signals_context_tokens_estimated_from_text():
    """Without an explicit hint the context is estimated at four chars/token."""
    text = "x" * 400
    signals = RequestSignals.from_payload({}, text)
    assert signals.context_tokens == 100


def test_signals_context_tokens_estimate_includes_history():
    """Earlier message content contributes to the estimated context."""
    payload = {
        "messages": [
            {"role": "user", "content": "y" * 400},
            {"role": "user", "content": ""},
        ]
    }
    assert RequestSignals.from_payload(payload).context_tokens == 100


def test_signals_context_tokens_estimate_from_content_parts():
    """Structured content parts of earlier messages contribute their text."""
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "z" * 800}],
            },
            {"role": "assistant", "content": "a" * 400},
            "raw string message" * 25,
            {"role": "user", "content": ""},
        ]
    }
    expected = (800 + 400 + len("raw string message" * 25)) // 4
    assert RequestSignals.from_payload(payload).context_tokens == expected


def test_signals_non_mapping_payload_is_safe():
    """A non-object payload yields default signals, text estimate only."""
    signals = RequestSignals.from_payload("not a dict", "w" * 80)
    assert signals == RequestSignals(context_tokens=20)
    assert signals.agent == ""
    assert signals.session_id == ""


# ===================== rules: declarative deterministic routing =====================
def _sig(**kwargs) -> RequestSignals:
    return RequestSignals(**kwargs)


def test_parse_rules_none_and_empty():
    """A missing or empty rule set is a valid no-op."""
    assert parse_rules(None, ["code"]) == ()
    assert parse_rules([], ["code"]) == ()


def test_parse_rules_sorts_by_priority_desc_and_keeps_json_order_on_ties():
    """Highest priority wins; equal priorities keep their declared order."""
    raw = [
        {"id": "low", "priority": 1, "when": {"task": "coding"}, "mode": "code"},
        {"id": "high", "priority": 90, "when": {"task": "coding"}, "mode": "plan"},
        {"id": "mid", "priority": 50, "when": {"task": "coding"}, "mode": "code"},
        {"id": "tie-a", "priority": 50, "when": {"task": "coding"}, "mode": "code"},
    ]
    rules = parse_rules(raw, ["code", "plan"])
    assert [rule.id for rule in rules] == ["high", "mid", "tie-a", "low"]


def test_parse_rules_defaults_id_and_priority():
    """Missing id/priority fall back to positional id and zero priority."""
    rules = parse_rules([{"when": {"task": "coding"}, "mode": "code"}], ["code"])
    assert rules[0].id == "rule-0"
    assert rules[0].priority == 0


def test_parse_rules_reads_mode_from_then_or_flat_key():
    """Both ``then.mode`` and a flat ``mode`` are accepted."""
    rules = parse_rules(
        [
            {"id": "a", "when": {"task": "x"}, "then": {"mode": "code"}},
            {"id": "b", "when": {"task": "y"}, "mode": "plan"},
        ],
        ["code", "plan"],
    )
    assert [rule.mode for rule in rules] == ["code", "plan"]


def test_parse_rules_normalizes_mode_name():
    """Mode references are normalized the same way as the mode definitions."""
    rules = parse_rules([{"when": {"task": "x"}, "mode": "Code-Gen"}], ["code_gen"])
    assert rules[0].mode == "code_gen"


def test_parse_rules_catch_all_warns():
    """An empty condition set matches everything and is reported."""
    logger = _CaptureLogger()
    rules = parse_rules(
        [{"id": "catch", "when": {}, "mode": "code"}], ["code"], logger
    )
    assert len(rules) == 1
    assert "catch" in logger.joined()
    assert "catch-all" in logger.joined()

    logger = _CaptureLogger()
    rules = parse_rules([{"id": "nokey", "mode": "code"}], ["code"], logger)
    assert rules[0].matches(_sig())
    assert "catch-all" in logger.joined()


@pytest.mark.parametrize(
    "raw,substring",
    [
        ({"not": "a list"}, "'rules' must be a list of rule objects"),
        ([7], "rules[0] must be an object, got int"),
        ([{"id": "", "mode": "code"}], "must be a non-empty string"),
        (
            [
                {"id": "dup", "mode": "code"},
                {"id": "dup", "mode": "code"},
            ],
            "duplicate rule id 'dup'",
        ),
        ([{"mode": "nope"}], "references unknown mode 'nope'"),
        ([{"when": {"bogus": 1}, "mode": "code"}], "unsupported rule condition(s)"),
        (
            [{"when": {"min_context_tokens": "x"}, "mode": "code"}],
            "must be an integer",
        ),
        ([{"when": {"metadata": 5}, "mode": "code"}], "metadata must be an object"),
        ([{"priority": "x", "mode": "code"}], "priority must be an integer"),
        ([{"priority": True, "mode": "code"}], "priority must be an integer"),
    ],
)
def test_parse_rules_rejects_malformed_input(raw, substring):
    """Configuration mistakes fail fast with an actionable message."""
    with pytest.raises(ValueError) as excinfo:
        parse_rules(raw, ["code"])
    assert substring in str(excinfo.value)


def test_rule_matches_agent_and_task_lists_use_or():
    """Agent/task conditions accept a scalar or a list and never match unset."""
    rule = parse_rules(
        [
            {
                "id": "r",
                "when": {"agent": ["batch", "etl"], "task": ["coding"]},
                "mode": "code",
            }
        ],
        ["code"],
    )[0]
    assert rule.matches(_sig(agent="etl", task="coding"))
    assert rule.matches(_sig(agent="batch", task=["coding"])) is False
    assert not rule.matches(_sig(agent="etl", task="planning"))
    assert not rule.matches(_sig(task="coding"))
    assert not rule.matches(_sig(agent="batch"))


def test_rule_requires_conditions_treat_false_as_unset():
    """``requires_*: false`` asserts that the capability is not requested."""
    rule_true = parse_rules(
        [{"id": "r", "when": {"requires_tools": True}, "mode": "code"}], ["code"]
    )[0]
    rule_false = parse_rules(
        [{"id": "r", "when": {"requires_tools": False}, "mode": "code"}], ["code"]
    )[0]
    assert rule_true.matches(_sig(tools=True, tool_count=3))
    assert not rule_true.matches(_sig())
    assert rule_false.matches(_sig())
    assert not rule_false.matches(_sig(tools=True))


@pytest.mark.parametrize(
    "flag,when_key",
    [
        ("reasoning", "requires_reasoning"),
        ("vision", "requires_vision"),
        ("structured_output", "requires_structured_output"),
        ("parallel_tools", "requires_parallel_tools"),
    ],
)
def test_rule_requires_conditions_for_every_capability(flag, when_key):
    """Each capability flag is enforceable from a rule."""
    rule = parse_rules(
        [{"id": "r", "when": {when_key: True}, "mode": "code"}], ["code"]
    )[0]
    assert rule.matches(_sig(**{flag: True}))
    assert not rule.matches(_sig())


def test_rule_context_window_bounds_are_inclusive():
    """min/max context token bounds include their endpoints."""
    rule = parse_rules(
        [
            {
                "id": "r",
                "when": {"min_context_tokens": 100, "max_context_tokens": 200},
                "mode": "code",
            }
        ],
        ["code"],
    )[0]
    assert rule.matches(_sig(context_tokens=100))
    assert rule.matches(_sig(context_tokens=150))
    assert rule.matches(_sig(context_tokens=200))
    assert not rule.matches(_sig(context_tokens=99))
    assert not rule.matches(_sig(context_tokens=201))


def test_rule_text_contains_any_is_case_insensitive_or():
    """text_contains_any matches any needle, ignoring case."""
    rule = parse_rules(
        [
            {
                "id": "r",
                "when": {"text_contains_any": ["K8s", "kube*rl"]},
                "mode": "code",
            }
        ],
        ["code"],
    )[0]
    assert rule.matches(_sig(), "roll it out with k8s")
    assert rule.matches(_sig(), "roll it out via KUBE*RL")
    assert not rule.matches(_sig(), "roll it out manually")
    assert not rule.matches(_sig(), "")


def test_rule_text_matches_is_a_regex_and_conditions_are_anded():
    """text_matches uses ``re.search`` and combines with other conditions."""
    rule = parse_rules(
        [
            {
                "id": "r",
                "when": {
                    "text_contains_any": ["k8s"],
                    "text_matches": r"deploy\s+to\s+\w+",
                },
                "mode": "code",
            }
        ],
        ["code"],
    )[0]
    assert rule.matches(_sig(), "Please DEPLOY TO staging with k8s")
    assert not rule.matches(_sig(), "deploy to production")
    assert not rule.matches(_sig(), "check k8s only")


def test_rule_invalid_regex_never_matches():
    """A broken pattern is inert instead of raising at request time."""
    rule = parse_rules(
        [{"id": "r", "when": {"text_matches": "([unclosed"}, "mode": "code"}],
        ["code"],
    )[0]
    assert not rule.matches(_sig(), "anything ([unclosed")


def test_rule_metadata_conditions_require_every_pair():
    """metadata conditions AND together over exact values."""
    rule = parse_rules(
        [
            {
                "id": "r",
                "when": {"metadata": {"env": "prod", "tier": "gold"}},
                "mode": "code",
            }
        ],
        ["code"],
    )[0]
    assert rule.matches(_sig(metadata={"env": "prod", "tier": "gold", "x": 1}))
    assert not rule.matches(_sig(metadata={"env": "prod"}))
    assert not rule.matches(_sig())


def test_match_rule_returns_highest_priority_match():
    """``match_rule`` returns the first hit in evaluation order."""
    rules = parse_rules(
        [
            {"id": "generic", "priority": 10, "when": {}, "mode": "code"},
            {
                "id": "specific",
                "priority": 80,
                "when": {"task": "coding"},
                "mode": "plan",
            },
        ],
        ["code", "plan"],
    )
    assert match_rule(rules, _sig(task="coding")).id == "specific"
    assert match_rule(rules, _sig(task="research")).id == "generic"
    assert match_rule([], _sig(task="coding")) is None
    assert match_rule(rules[:1:1][:0], _sig()) is None


def test_describe_rule_is_log_friendly():
    """Rule descriptions carry id, priority and target mode."""
    rule = RoutingRule(id="r1", priority=42, when={}, mode="code")
    assert describe_rule(rule) == "r1(p=42) -> code"


# ===================== capabilities: hard requirement filtering =====================
def _cap_modes() -> list:
    return [
        _mode("plan", {"reasoning": True, "context_window": 131072}),
        _mode(
            "code", {"tool_calling": True, "vision": False, "context_window": 131072}
        ),
        _mode("review", {"reasoning": True, "context_window": 32768}),
        _mode("visionary", {"vision": True}),
        _mode("fb"),
    ]


def test_requirements_only_contain_active_signals():
    """Inactive capabilities produce no requirement at all."""
    assert requirements_from_signals(RequestSignals()) == {}
    assert requirements_from_signals(RequestSignals(tools=True)) == {
        "tool_calling": True
    }
    assert requirements_from_signals(
        RequestSignals(reasoning=True, vision=True)
    ) == {
        "reasoning": True,
        "vision": True,
    }


def test_requirements_context_window_only_when_positive():
    """The context requirement exists only for a positive token hint."""
    assert "context_window" not in requirements_from_signals(
        RequestSignals(context_tokens=0)
    )
    assert requirements_from_signals(RequestSignals(context_tokens=4096)) == {
        "context_window": 4096
    }


def test_missing_capabilities_ignores_undeclared_keys():
    """A capability that is not declared is not treated as unsupported."""
    assert missing_capabilities(None, {"vision": True}) == []
    assert missing_capabilities({}, {"vision": True}) == []
    assert missing_capabilities({"tool_calling": True}, {"vision": True}) == []
    assert missing_capabilities({"vision": False}, {"vision": True}) == ["vision"]
    assert missing_capabilities({"vision": False}, {"vision": False}) == []


def test_missing_capabilities_context_window_is_numeric():
    """context_window is violated only when the declared window is smaller."""
    caps = {"context_window": 32768}
    assert missing_capabilities(caps, {"context_window": 60000}) == [
        "context_window"
    ]
    assert missing_capabilities(caps, {"context_window": 32768}) == []
    assert missing_capabilities(caps, {"context_window": 1000}) == []
    assert missing_capabilities({}, {"context_window": 60000}) == []


def test_satisfies_is_the_negation_of_missing():
    """``satisfies`` is a readable shorthand."""
    assert satisfies({"reasoning": True}, {"reasoning": True})
    assert not satisfies({"reasoning": False}, {"reasoning": True})


def test_filter_modes_keeps_configuration_order():
    """Filtering preserves declaration order and drops incapable modes."""
    modes = _cap_modes()
    modes.append(_mode("narrow", {"reasoning": False}))
    kept = [mode.name for mode in filter_modes(modes, {"reasoning": True})]
    assert kept == ["plan", "code", "review", "visionary", "fb"]
    assert [mode.name for mode in filter_modes(modes, None)] == [
        mode.name for mode in modes
    ]
    assert [mode.name for mode in filter_modes(modes, {})] == [
        mode.name for mode in modes
    ]


def test_escalate_is_a_noop_without_requirements():
    """Nothing to enforce means the original mode is kept."""
    modes = _cap_modes()
    result = escalate(modes[2], modes, None, logger=_CaptureLogger())
    assert result.mode is modes[2]
    assert result.changed is False
    assert result.reason == ""


def test_escalate_picks_the_widest_context_window():
    """Escalation prefers the alternative with the largest declared window."""
    modes = _cap_modes()
    logger = _CaptureLogger()
    result = escalate(modes[2], modes, {"context_window": 60000}, None, logger)
    assert result.mode.name == "plan"
    assert result.changed is True
    assert result.reason == "missing: context_window"
    assert "escalating 'review' -> 'plan'" in logger.joined()


def test_escalate_never_returns_the_fallback_mode():
    """The configured fallback is excluded even when it would qualify."""
    modes = [
        _mode("code", {"parallel_tool_calls": False}),
        _mode("plan", {"parallel_tool_calls": True, "context_window": 128}),
        _mode("fb", {"parallel_tool_calls": True}),
    ]
    logger = _CaptureLogger()
    result = escalate(modes[0], modes, {"parallel_tool_calls": True}, "fb", logger)
    assert result.mode.name == "plan"
    assert result.changed is True


def test_escalate_falls_back_to_the_fallback_when_only_capable():
    """When only the fallback can serve the request it is allowed."""
    modes = [
        _mode("code", {"vision": False}),
        _mode("plan", {"vision": False}),
        _mode("fb", {"vision": True}),
    ]
    result = escalate(modes[0], modes, {"vision": True}, "fb", _CaptureLogger())
    assert result.mode.name == "fb"
    assert result.changed is True


def test_escalate_keeps_mode_and_warns_when_nothing_can_serve():
    """An unsatisfiable request keeps the decision but is loudly reported."""
    modes = [
        _mode("code", {"vision": False}),
        _mode("plan", {"vision": False}),
        _mode("fb", {"vision": False}),
    ]
    logger = _CaptureLogger()
    result = escalate(modes[0], modes, {"vision": True}, "fb", logger)
    assert result.mode.name == "code"
    assert result.changed is False
    assert "cannot serve" in logger.joined() or "no capable alternative" in (
        logger.joined()
    )


# ===================== session affinity: sticky deterministic routing =========
def test_affinity_stores_and_returns_a_decision():
    """A stored decision is replayed with its mode and model."""
    cache = SessionAffinityCache(ttl_seconds=60)
    cache.set("s1", "code", "model_code")
    decision = cache.get("s1")
    assert decision is not None
    assert decision.mode_name == "code"
    assert decision.model_name == "model_code"
    assert len(cache) == 1


def test_affinity_ignores_empty_session_ids():
    """Requests without a session id never touch the cache."""
    cache = SessionAffinityCache(ttl_seconds=60)
    cache.set("", "code", "model_code")
    assert len(cache) == 0
    assert cache.get("") is None


def test_affinity_entry_expires_after_ttl(monkeypatch):
    """Entries older than the TTL are dropped and reported as a miss."""
    now = [1000.0]
    monkeypatch.setattr(session_affinity.time, "monotonic", lambda: now[0])
    cache = SessionAffinityCache(ttl_seconds=60)
    cache.set("s1", "code", "model_code")
    now[0] += 59
    assert len(cache) == 1
    now[0] += 1
    assert cache.get("s1") is None
    assert len(cache) == 0


def test_affinity_hit_refreshes_the_ttl(monkeypatch):
    """A cache hit slides the expiry window forward."""
    now = [1000.0]
    monkeypatch.setattr(session_affinity.time, "monotonic", lambda: now[0])
    cache = SessionAffinityCache(ttl_seconds=60)
    cache.set("s1", "code", "model_code")
    now[0] += 50
    assert cache.get("s1") is not None
    now[0] += 50
    assert cache.get("s1") is not None
    now[0] += 61
    assert cache.get("s1") is None


def test_affinity_evicts_least_recently_used_with_debug(monkeypatch):
    """The oldest entry leaves first and says so on debug level."""
    now = [1000.0]
    monkeypatch.setattr(session_affinity.time, "monotonic", lambda: now[0])
    logger = _CaptureLogger()
    cache = SessionAffinityCache(ttl_seconds=60, max_entries=2, logger=logger)
    cache.set("s1", "code", "model_code")
    cache.set("s2", "plan", "model_plan")
    assert cache.get("s1") is not None
    cache.set("s3", "review", "model_review")
    assert len(cache) == 2
    assert cache.get("s2") is None
    assert cache.get("s1") is not None
    assert cache.get("s3") is not None
    assert "evicting session affinity entry 's2'" in "\n".join(logger["debug"])


def test_affinity_invalidate_and_reset():
    """Entries can be dropped individually or wholesale."""
    cache = SessionAffinityCache(ttl_seconds=60)
    cache.set("s1", "code", "model_code")
    cache.set("s2", "plan", "model_plan")
    cache.invalidate("s1")
    assert cache.get("s1") is None
    assert cache.get("s2") is not None
    cache.invalidate("missing")
    cache.reset()
    assert len(cache) == 0
    assert cache.get("s2") is None


def test_affinity_overwrites_an_existing_entry():
    """Re-routing a session replaces the sticky decision."""
    cache = SessionAffinityCache(ttl_seconds=60)
    cache.set("s1", "code", "model_code")
    cache.set("s1", "plan", "model_plan")
    assert len(cache) == 1
    assert cache.get("s1").mode_name == "plan"


def test_affinity_is_thread_safe():
    """Concurrent access to distinct sessions does not corrupt the cache."""
    cache = SessionAffinityCache(ttl_seconds=60, max_entries=512)

    def worker(index: int) -> None:
        session = "session-%d" % index
        cache.set(session, "code", "model_code")
        cache.get(session)
        cache.get("session-%d" % ((index + 1) % 32))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(worker, range(32)))
    assert len(cache) == 32


# ===================== heuristics: keyword scoring without ML ===================
def test_score_to_similarity_is_monotonic_and_bounded():
    """Scores map into (0, 1) and never decrease with a higher score."""
    assert score_to_similarity(0.0) == 0.0
    low = score_to_similarity(1.0)
    high = score_to_similarity(5.0)
    assert 0.0 < low < high < 1.0
    assert low == pytest.approx(0.5)


def test_score_mode_counts_keywords_with_weights():
    """Keywords score their configured weight, defaulting to one."""
    mode = _mode(
        "code",
        keywords=["refactor", "cache", "python"],
        weights={"python": 4},
    )
    assert score_mode(mode, "refactor the cache") == 2.0
    assert score_mode(mode, "python refactor") == 5.0
    assert score_mode(mode, "nothing relevant") == 0.0


def test_score_mode_counts_phrases_with_optional_weight():
    """Phrases weigh two by default and accept a ``text:weight`` suffix."""
    mode = _mode("plan", phrases=["road map", "mile:stone:2.5", "break down:5"])
    assert score_mode(mode, "build a road map please") == 2.0
    assert score_mode(mode, "a mile:stone moment") == 2.5
    assert score_mode(mode, "let's break down the work") == 5.0
    assert score_mode(mode, "break down") == 5.0


def test_score_mode_falls_back_for_an_unparseable_phrase_weight():
    """A non-numeric weight suffix keeps the whole phrase at the default."""
    mode = _mode("plan", phrases=["end to end:heavy"])
    assert score_mode(mode, "an end to end:heavy review") == 2.0
    assert score_mode(mode, "an end to end review") == 0.0


def test_score_mode_counts_patterns_with_a_fixed_weight():
    """Each regex match adds the pattern weight; broken patterns are skipped."""
    mode = _mode("debug", patterns=[r"traceback", r"error:.*"])
    assert score_mode(mode, "traceback and error: boom") == 2 * PATTERN_WEIGHT
    broken = _mode("debug", patterns=[r"([unclosed"])
    assert score_mode(broken, "anything ([unclosed") == 0.0


def test_detect_heuristic_prefers_highest_score_then_configuration_order():
    """The strongest match wins, ties resolve to the first configured mode."""
    strong = _mode("code", keywords=["refactor"], phrases=["refactor the cache"])
    weak = _mode("review", keywords=["refactor"])
    assert detect_heuristic("REFACTOR THE CACHE", [weak, strong]) == (strong, 3.0)
    first = _mode("code", keywords=["refactor"])
    second = _mode("review", keywords=["refactor"])
    assert detect_heuristic("please refactor", [first, second])[0].name == "code"


def test_detect_heuristic_requires_text_and_a_match():
    """No text or no signal means no deterministic answer."""
    mode = _mode("code", keywords=["refactor"])
    assert detect_heuristic("", [mode]) == (None, 0.0)
    assert detect_heuristic("   ", [mode]) == (None, 0.0)
    assert detect_heuristic("a nice day", [mode]) == (None, 0.0)
    assert detect_heuristic("refactor", []) == (None, 0.0)


def test_scored_modes_is_sorted_and_positive_only():
    """The ranked view exposes only modes that actually matched."""
    code = _mode("code", keywords=["refactor"], weights={"refactor": 5})
    review = _mode("review", keywords=["refactor"])
    other = _mode("test", keywords=["pytest"])
    ranked = scored_modes("refactor this", [other, review, code])
    assert [mode.name for mode, _ in ranked] == ["code", "review"]
    assert ranked[0][1] > ranked[1][1] > 0
    assert scored_modes("nothing here", [code, review]) == []


# ============ semantic layer: the last, non-deterministic resort ===============
def test_semantic_layer_is_available_only_with_a_router():
    """A missing router makes the layer unavailable, never an error."""
    assert SemanticLayer(_StubRouter("code", 0.9), 0.5, {}).available is True
    assert SemanticLayer(None, 0.5, {}).available is False


def test_semantic_layer_never_calls_the_router_without_text():
    """Empty text is filtered before any embedding lookup."""
    router = _StubRouter("code", 0.9)
    layer = SemanticLayer(router, 0.5, {"code": _mode("code")})
    assert layer.resolve("") == (None, 0.0)
    assert router.calls == []


def test_semantic_layer_resolves_a_confident_known_target():
    """A known target above the threshold is accepted with its similarity."""
    code = _mode("code")
    layer = SemanticLayer(_StubRouter("code", 0.83), 0.5, {"code": code})
    mode, similarity = layer.resolve("please refactor")
    assert mode is code
    assert similarity == pytest.approx(0.83)


def test_semantic_layer_accepts_the_threshold_exactly():
    """The threshold is inclusive: ``similarity >= threshold`` wins."""
    code = _mode("code")
    layer = SemanticLayer(_StubRouter("code", 0.5), 0.5, {"code": code})
    assert layer.resolve("please refactor")[0] is code


def test_semantic_layer_rejects_a_weak_match_with_a_note():
    """Weak matches are dropped and reported so the cascade can continue."""
    logger = _CaptureLogger()
    layer = SemanticLayer(
        _StubRouter("code", 0.4), 0.5, {"code": _mode("code")}, logger
    )
    mode, similarity = layer.resolve("please refactor")
    assert mode is None
    assert similarity == pytest.approx(0.4)
    assert "below threshold" in logger.joined()


def test_semantic_layer_rejects_a_target_that_is_not_configured():
    """A router may only select modes that exist in the configuration."""
    layer = SemanticLayer(_StubRouter("ghost", 0.99), 0.5, {"code": _mode("code")})
    mode, similarity = layer.resolve("anything")
    assert mode is None
    assert similarity == pytest.approx(0.99)


# ======= plugin: the cascade answers deterministically whenever it can =========
def test_plugin_returns_fallback_for_unmatched_text():
    """Text that moves no layer lands on the configured fallback."""
    result = _apply(_make_plugin(), "Hello there")
    assert result["agent_mode"] == _DEFAULT_FALLBACK
    assert _routing(result)["source"] == "fallback"
    assert _routing(result)["similarity"] == 0.0


def test_plugin_marks_requests_without_text_as_empty_text():
    """A request without usable text gets its own distinguishable source."""
    result = _apply(_make_plugin(), "   ")
    assert _routing(result)["source"] == "empty_text"
    assert result["agent_mode"] == _DEFAULT_FALLBACK


def test_plugin_routes_the_documented_task_values():
    """Every shipped ``task`` value maps to its mode through ``rules``."""
    expected = {
        "coding": "code",
        "planning": "plan",
        "review": "review",
        "testing": "test",
        "debugging": "debug",
        "research": "research",
        "summarize": "summarize",
    }
    for task, mode in expected.items():
        result = _apply(_make_plugin(), "Hello there", task=task)
        routing = _routing(result)
        assert routing["agent_mode"] == mode, task
        assert routing["source"] == "rules", task
        assert routing["rule_id"] == "task-%s" % task, task
        assert routing["similarity"] == 1.0


def test_plugin_routes_capability_signals_before_any_text_analysis():
    """Tools and reasoning are payload facts, not text guesses."""
    tools = _apply(_make_plugin(), "Hello there", tools=True)
    assert _routing(tools)["agent_mode"] == "code"
    assert _routing(tools)["rule_id"] == "needs-tools"

    reasoning = _apply(_make_plugin(), "Hello there", reasoning=True)
    assert _routing(reasoning)["agent_mode"] == "plan"
    assert _routing(reasoning)["rule_id"] == "needs-reasoning"


def test_plugin_prefers_the_task_rule_over_the_capability_rule():
    """Higher priority rules win, independently of rule declaration order."""
    result = _apply(_make_plugin(), "Hello there", task="coding", tools=True)
    assert _routing(result)["rule_id"] == "task-coding"


def test_plugin_cascade_is_reproducible():
    """The same payload always produces the same mode, layer and model."""
    plugin = _make_plugin()

    def decide() -> tuple:
        result = _apply(plugin, "Hello there", task="review", context_tokens=60000)
        routing = _routing(result)
        return (
            routing["agent_mode"],
            routing["source"],
            routing["rule_id"],
            routing.get("escalated_from"),
            result["model"],
        )

    assert len({decide() for _ in range(5)}) == 1
    assert decide() == (
        "plan",
        "rules",
        "task-review",
        "review",
        _MODEL_FOR_MODE["plan"],
    )


def test_plugin_decisions_do_not_depend_on_request_order():
    """Interleaving unrelated requests must not disturb a repeated payload."""
    plugin = _make_plugin()
    baseline = _routing(_apply(plugin, "Please refactor this module"))

    for other in ("Hello there", "summarize the report", "fix the stack trace"):
        _apply(plugin, other, session_id="noise-%s" % other[:4])

    after = _routing(_apply(plugin, "Please refactor this module"))
    assert after["agent_mode"] == baseline["agent_mode"]
    assert after["source"] == baseline["source"]


# ================== plugin: capabilities and escalation ========================
def test_plugin_escalates_a_rule_hit_that_cannot_serve_the_request():
    """A rule still decides; capabilities only upgrade the chosen mode."""
    result = _apply(
        _make_plugin(), "Hello there", task="review", context_tokens=60000
    )
    routing = _routing(result)
    assert routing["agent_mode"] == "plan"
    assert routing["source"] == "rules"
    assert routing["rule_id"] == "task-review"
    assert routing["escalated"] is True
    assert routing["escalated_from"] == "review"


def test_plugin_escalates_a_heuristic_hit_that_cannot_serve_the_request():
    """Escalation is layer agnostic, not limited to rules."""
    result = _apply(_make_plugin(), "Please refactor this module", vision=True)
    routing = _routing(result)
    assert routing["source"] == "heuristic"
    assert routing["escalated"] is True
    assert routing["escalated_from"] == "code"


def test_plugin_keeps_the_mode_when_capabilities_are_satisfied():
    """Nothing is escalated when the selected mode can do the job."""
    result = _apply(_make_plugin(), "Hello there", task="review")
    assert result["agent_mode"] == "review"
    assert "escalated" not in _routing(result)


@pytest.mark.parametrize("flag", ["ESCALATION_ENABLED", "CAPABILITIES_ENABLED"])
def test_plugin_honours_the_capability_feature_flags(flag: str):
    """Both switches suppress escalation, each at its own level."""
    result = _apply(
        _make_plugin(**{flag: "false"}),
        "Hello there",
        task="review",
        context_tokens=60000,
    )
    assert result["agent_mode"] == "review"
    assert "escalated" not in _routing(result)


def test_plugin_never_escalates_an_explicit_choice():
    """An explicit mode is respected even when it is not capable."""
    logger = _CaptureLogger()
    os.environ["%sSEMANTIC_ENABLED" % _PREFIX] = "false"
    plugin = AgenticRoutingPlugin(logger=logger)
    result = _apply(plugin, "Hello there", agent_mode="review", context_tokens=60000)
    assert result["agent_mode"] == "review"
    assert _routing(result)["source"] == "explicit"
    assert "escalated" not in _routing(result)
    assert "does not satisfy" in logger.joined()


# ================ plugin: explicit mode and feature flags ======================
def test_explicit_agent_mode_short_circuits_the_cascade():
    """A caller-provided mode beats a matching rule."""
    result = _apply(_make_plugin(), "Hello there", agent_mode="plan", task="coding")
    assert result["agent_mode"] == "plan"
    assert _routing(result)["source"] == "explicit"
    assert "rule_id" not in _routing(result)


def test_unknown_explicit_agent_mode_falls_through_the_cascade():
    """A typo must not break routing - it warns and continues detection."""
    logger = _CaptureLogger()
    os.environ["%sSEMANTIC_ENABLED" % _PREFIX] = "false"
    plugin = AgenticRoutingPlugin(logger=logger)
    result = _apply(plugin, "Hello there", agent_mode="does-not-exist")
    assert result["agent_mode"] == _DEFAULT_FALLBACK
    assert _routing(result)["source"] == "fallback"
    assert "unknown agent_mode 'does_not_exist'" in logger.joined()


def test_rules_can_be_disabled_without_losing_configuration():
    """``RULES_ENABLED=false`` skips the layer but keeps the parsed rules."""
    plugin = _make_plugin(RULES_ENABLED="false")
    assert plugin.config.rules_enabled is False
    assert len(plugin.config.rules) == 9
    result = _apply(plugin, "Hello there", task="coding")
    assert result["agent_mode"] == _DEFAULT_FALLBACK
    assert _routing(result)["source"] == "fallback"


# ======================== plugin: session affinity =============================
def test_session_affinity_reuses_the_first_decision():
    """A follow-up with no signal keeps the mode chosen earlier."""
    plugin = _make_plugin()
    first = _apply(plugin, "please refactor the python module", session_id="s1")
    assert _routing(first)["source"] == "heuristic"
    assert _routing(first)["session"]["reused"] is False

    second = _apply(plugin, "Hello there", session_id="s1")
    routing = _routing(second)
    assert routing["agent_mode"] == first["agent_mode"]
    assert routing["source"] == "affinity"
    assert routing["similarity"] == 1.0
    assert routing["session"]["reused"] is True


def test_session_affinity_never_overrides_rules():
    """Deterministic rules keep priority over the sticky session mode."""
    plugin = _make_plugin()
    _apply(plugin, "please refactor the python module", session_id="s1")
    result = _apply(plugin, "Hello there", session_id="s1", task="review")
    routing = _routing(result)
    assert routing["source"] == "rules"
    assert routing["rule_id"] == "task-review"
    assert routing["session"]["reused"] is False


def test_session_affinity_skips_fallback_decisions():
    """A fallback says nothing about the session, so nothing is cached."""
    plugin = _make_plugin()
    first = _apply(plugin, "Hello there", session_id="s2")
    assert _routing(first)["session"]["reused"] is False
    second = _apply(plugin, "please refactor the python module", session_id="s2")
    assert _routing(second)["source"] == "heuristic"


def test_reset_sessions_forces_a_fresh_decision():
    """``reset_sessions`` drops affinity without touching the config."""
    plugin = _make_plugin()
    _apply(plugin, "please refactor the python module", session_id="s1")
    assert _routing(_apply(plugin, "Hello there", session_id="s1"))["source"] == (
        "affinity"
    )
    plugin.reset_sessions()
    assert (
        _routing(_apply(plugin, "Hello there", session_id="s1"))["source"]
        == "fallback"
    )


def test_session_affinity_can_be_turned_off():
    """Disabling affinity removes the ``session`` annotation entirely."""
    plugin = _make_plugin(SESSION_AFFINITY_ENABLED="false")
    _apply(plugin, "please refactor the python module", session_id="s1")
    result = _apply(plugin, "Hello there", session_id="s1")
    assert _routing(result)["source"] == "fallback"
    assert "session" not in _routing(result)


def test_session_cache_settings_come_from_the_environment():
    """TTL and size of the affinity cache are configurable by env vars."""
    plugin = _make_plugin(SESSION_TTL_SECONDS="42", SESSION_MAX_ENTRIES="7")
    settings = plugin.config.session_affinity
    assert settings.enabled is True
    assert settings.ttl_seconds == 42
    assert settings.max_entries == 7


# =============== plugin: semantic matching really is last ======================
def test_deterministic_layers_do_not_touch_the_router():
    """Rules, explicit modes and affinity never reach the embedding layer."""
    cases = (
        {"text": "Hello there", "extra": {"task": "coding"}},
        {"text": "Hello there", "extra": {"agent_mode": "plan"}},
        {"text": "Hello there", "extra": {"tools": True}},
    )
    for case in cases:
        plugin, router = _make_plugin_with_router(_StubRouter("research", 0.99))
        _apply(plugin, case["text"], **case["extra"])
        assert router.calls == [], case


def test_affinity_hit_short_circuits_the_router():
    """A sticky session keeps semantic work off the hot path."""
    plugin, router = _make_plugin_with_router(_StubRouter("research", 0.99))
    _apply(plugin, "please research the market trends", session_id="s9")
    result = _apply(plugin, "Hello there", session_id="s9")
    assert _routing(result)["source"] == "affinity"
    assert router.calls == []


def test_semantic_layer_decides_only_when_nothing_else_matched():
    """Unmatched text is the single path that pays for an embedding lookup."""
    plugin, router = _make_plugin_with_router(_StubRouter("research", 0.99))
    result = _apply(plugin, "complete unrelated gibberish zzz")
    routing = _routing(result)
    assert routing["agent_mode"] == "research"
    assert routing["source"] == "semantic"
    assert routing["similarity"] == pytest.approx(0.99)
    assert router.calls == ["complete unrelated gibberish zzz"]


def test_semantic_layer_respects_the_threshold_and_falls_back():
    """A weak semantic match must not override the fallback mode."""
    plugin, router = _make_plugin_with_router(_StubRouter("research", 0.1))
    result = _apply(plugin, "complete unrelated gibberish zzz")
    assert result["agent_mode"] == _DEFAULT_FALLBACK
    assert _routing(result)["source"] == "fallback"
    assert len(router.calls) == 1


def test_semantic_matching_can_be_turned_off():
    """With semantic disabled no router call happens at all."""
    plugin, router = _make_plugin_with_router(_StubRouter("research", 0.99))
    os.environ["%sSEMANTIC_ENABLED" % _PREFIX] = "false"
    fresh = AgenticRoutingPlugin(router=router)
    result = _apply(fresh, "complete unrelated gibberish zzz")
    assert _routing(result)["source"] == "fallback"
    assert router.calls == []


# ==================== plugin: response annotation contract =====================
@pytest.mark.parametrize(
    "text,extra,expected_mode,expected_source",
    [
        ("Hello there", {}, _DEFAULT_FALLBACK, "fallback"),
        ("Hello there", {"task": "coding"}, "code", "rules"),
        ("Hello there", {"agent_mode": "plan"}, "plan", "explicit"),
        ("Please refactor this module", {}, "code", "heuristic"),
    ],
)
def test_routing_annotation_is_consistent(
    text: str, extra: dict, expected_mode: str, expected_source: str
):
    """Every decision reports the same mode, model and layer."""
    result = _apply(_make_plugin(), text, **extra)
    routing = _routing(result)
    assert routing["agent_mode"] == expected_mode
    assert routing["source"] == expected_source
    assert routing["plugin"] == "agentic_routing"
    assert result["agent_mode"] == expected_mode
    assert result["model"] == _MODEL_FOR_MODE[expected_mode]
    assert set(routing) >= {"plugin", "similarity", "agent_mode", "source"}
