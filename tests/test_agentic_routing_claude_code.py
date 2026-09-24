"""
Tests for the Claude Code model swap plugin
(``llm_router_plugins.utils.routing.agentic_routing.claude_code``).

Covers the model-name normalizer, the four-layer match cascade (literal,
normalized exact, wildcard prefix, family token), configuration
loading/validation/env overrides and the plugin itself.  Every assertion is
exact: the plugin has no scoring, no embeddings and no network access, so a
model name either resolves to a mode or the payload comes back as the same
object it went in as.

The shipped config is asserted against the model IDs Claude Code emits today
(``claude-fable-5-1``, ``claude-opus-5-5``, ``claude-sonnet-5``,
``claude-haiku-4-5`` and friends) and against third-party spellings of them.

Run with:
    pytest tests/test_agentic_routing_claude_code.py -v
"""

import copy
import json
import logging
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

from llm_router_plugins.utils.registry import MAIN_UTILS_REGISTRY
from llm_router_plugins.utils.routing.agentic_routing.claude_code import (
    DEFAULT_MODEL_FIELDS,
    MATCH_EXACT,
    MATCH_FAMILY,
    MATCH_LITERAL,
    MATCH_WILDCARD,
    ClaudeCodeMode,
    ClaudeCodeRoutingConfig,
    ClaudeCodeRoutingPlugin,
    ModelMatcher,
    find_ambiguous_wildcards,
    find_duplicate_literals,
    is_wildcard,
    model_family,
    normalize_model_name,
    validate_pattern,
)
from llm_router_plugins.utils.routing.constants import (
    AGENTIC_CLAUDE_CODE_ROUTING_PREFIX,
)

_PREFIX = AGENTIC_CLAUDE_CODE_ROUTING_PREFIX
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_CONFIG_PATH = (
    _REPO_ROOT
    / "llm_router_plugins"
    / "resources"
    / "routing"
    / "agentic_routing_claude_code.json"
)
_ROUTING_KEYS = {
    "plugin",
    "similarity",
    "mode",
    "original_model",
    "matched_model",
    "match_type",
    "field",
}


@pytest.fixture(autouse=True)
def clean_claude_code_env(monkeypatch):
    """Remove every plugin env var so the ambient shell cannot leak in."""
    for key in list(__import__("os").environ):
        if key.startswith(_PREFIX):
            monkeypatch.delenv(key)


@pytest.fixture(name="logger")
def logger_fixture():
    """A logger the plugin writes to, captured by ``caplog``."""
    return logging.getLogger("test_agentic_routing_claude_code")


def _mode(name, **kwargs):
    """Build a mode with a sensible default target model."""
    return ClaudeCodeMode(
        name=name,
        model_name=kwargs.pop("model_name", f"target/{name}"),
        models=tuple(kwargs.pop("models", ())),
        description=kwargs.pop("description", ""),
    )


def _config(modes=None, **changes):
    """Build a config over ``modes``, defaulting to one tier per family."""
    if modes is None:
        modes = (
            _mode("opus", models=("claude-opus-5-5", "claude-opus-*")),
            _mode("sonnet", models=("claude-sonnet-5", "claude-sonnet-*")),
            _mode("haiku", model_name="target/fast", models=("claude-haiku-*",)),
        )
    return ClaudeCodeRoutingConfig(
        enabled=changes.get("enabled", True),
        match_families=changes.get("match_families", True),
        model_fields=changes.get("model_fields", DEFAULT_MODEL_FIELDS),
        provider_prefixes=changes.get(
            "provider_prefixes",
            ("us.anthropic.", "eu.anthropic.", "anthropic.", "anthropic/"),
        ),
        modes=tuple(modes),
    )


def _plugin(config=None, logger=None):
    """Build a plugin over *config* (a default one when omitted)."""
    return ClaudeCodeRoutingPlugin(
        logger=logger, config=config if config is not None else _config()
    )


def _shipped_config():
    """Load the config that ships in ``resources/routing``."""
    return ClaudeCodeRoutingConfig.from_file(_CONFIG_PATH)


class TestNormalizeModelName:
    """The lossy reduction that lets one config entry cover many spellings."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("claude-sonnet-5", "claude-sonnet-5"),
            ("  Claude-Sonnet-5  ", "claude-sonnet-5"),
            ("claude-opus-5-5[1m]", "claude-opus-5-5"),
            ("claude-sonnet-4-5-20250929", "claude-sonnet-4-5"),
            ("claude-sonnet-4-5-20250929-v1", "claude-sonnet-4-5"),
            ("us.anthropic.claude-sonnet-4-5-20250929-v1:0", "claude-sonnet-4-5"),
            ("eu.anthropic.claude-opus-4-8:2", "claude-opus-4-8"),
            ("anthropic.claude-haiku-4-5", "claude-haiku-4-5"),
            ("anthropic/claude-haiku-4-5", "claude-haiku-4-5"),
            ("models/claude-sonnet-4-5", "claude-sonnet-4-5"),
            ("claude-sonnet-5@20250929", "claude-sonnet-5"),
            ("qwen/Qwen3.8-27B", "qwen3.8-27b"),
        ],
    )
    def test_spellings_collapse(self, raw, expected):
        assert normalize_model_name(raw) == expected

    @pytest.mark.parametrize("raw", ["", "   ", None, 12, ["claude-opus-5-5"]])
    def test_nothing_to_compare(self, raw):
        assert normalize_model_name(raw) == ""

    def test_provider_prefixes_are_configurable(self):
        assert (
            normalize_model_name(
                "us.anthropic.claude-opus-5-5", provider_prefixes=("anthropic.",)
            )
            == "us.anthropic.claude-opus-5-5"
        )
        assert (
            normalize_model_name(
                "us.anthropic.claude-opus-5-5", provider_prefixes=()
            )
            == "us.anthropic.claude-opus-5-5"
        )

    def test_longest_prefix_wins(self):
        assert (
            normalize_model_name(
                "us.anthropic.claude-opus-5-5",
                provider_prefixes=("anthropic.", "us.anthropic."),
            )
            == "claude-opus-5-5"
        )


class TestModelFamily:
    """Family tokens, including the legacy names that put them mid-string."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("claude-opus-5-5", "opus"),
            ("claude-sonnet-5", "sonnet"),
            ("claude-haiku-4-5", "haiku"),
            ("claude-fable-5-1", "fable"),
            ("claude-3-5-haiku-20241022", "haiku"),
            ("claude-2-0-sonnet", "sonnet"),
            ("CLAUDE-OPUS-4-8", "opus"),
            ("gpt-5", None),
            ("qwen/Qwen3.8-27B", None),
            ("opusculum", None),
            ("", None),
            (None, None),
        ],
    )
    def test_families(self, raw, expected):
        assert model_family(raw) == expected


class TestValidatePattern:
    def test_plain_pattern_trimmed(self):
        assert validate_pattern("  claude-opus-5-5 ") == "claude-opus-5-5"

    def test_trailing_wildcard_accepted(self):
        assert validate_pattern("claude-opus-*") == "claude-opus-*"

    @pytest.mark.parametrize("bad", ["", "   ", None, 7, ["claude-opus-*"]])
    def test_empty_pattern_rejected(self, bad):
        with pytest.raises(ValueError):
            validate_pattern(bad)

    @pytest.mark.parametrize("bad", ["claude-*opus", "cl*ude-opus", "**"])
    def test_embedded_wildcard_rejected(self, bad):
        with pytest.raises(ValueError):
            validate_pattern(bad)

    def test_bare_wildcard_rejected(self):
        with pytest.raises(ValueError, match="matches every model"):
            validate_pattern("*")

    def test_is_wildcard(self):
        assert is_wildcard("claude-opus-*") is True
        assert is_wildcard("claude-opus-5-5") is False


class TestMatcherCascade:
    """Literal → exact → wildcard → family, most specific first."""

    def test_literal_beats_everything(self):
        matcher = ModelMatcher(
            [
                ("sonnet", "claude-sonnet-*"),
                ("opus", "claude-sonnet-4-5"),
            ]
        )
        assert matcher.match("claude-sonnet-4-5").mode_name == "opus"
        assert matcher.match("claude-sonnet-4-5").kind == MATCH_LITERAL

    def test_normalized_exact_beats_wildcard(self):
        matcher = ModelMatcher(
            [("sonnet", "claude-sonnet-*"), ("opus", "claude-sonnet-4-5")]
        )
        # Dated/prefixed spelling of the pinned name still hits the pin.
        found = matcher.match("us.anthropic.claude-sonnet-4-5-20250929-v1:0")
        assert (found.mode_name, found.kind) == ("opus", MATCH_EXACT)

    def test_longest_wildcard_prefix_wins(self):
        matcher = ModelMatcher(
            [
                ("family", "claude-*"),
                ("line", "claude-opus-4-*"),
                ("tier", "claude-opus-*"),
            ]
        )
        assert matcher.match("claude-opus-4-8").mode_name == "line"
        assert matcher.match("claude-opus-5-5").mode_name == "tier"
        assert matcher.match("claude-sonnet-5").mode_name == "family"

    def test_family_layer_covers_legacy_ordering(self):
        matcher = ModelMatcher([("haiku", "claude-haiku-*")])
        found = matcher.match("claude-3-5-haiku-20241022")
        assert (found.mode_name, found.kind) == ("haiku", MATCH_FAMILY)

    def test_family_layer_needs_a_wildcard_declaration(self):
        # An exact pin does not silently claim a whole family.
        matcher = ModelMatcher([("haiku", "claude-3-5-haiku-20241022")])
        assert matcher.match("claude-haiku-4-5") is None

    def test_reported_kind_matches_the_layer(self):
        matcher = ModelMatcher(
            [("sonnet", "claude-sonnet-4-5"), ("opus", "claude-opus-*")]
        )
        assert matcher.match("claude-sonnet-4-5").kind == MATCH_LITERAL
        assert matcher.match("claude-sonnet-4-5[1m]").kind == MATCH_EXACT
        assert matcher.match("claude-opus-5-5").kind == MATCH_WILDCARD

    def test_wildcard_matches_the_prefixed_spelling(self):
        matcher = ModelMatcher([("opus", "claude-opus-*")])
        assert matcher.match("us.anthropic.claude-opus-4-8:2").mode_name == "opus"

    def test_matching_is_case_insensitive(self):
        matcher = ModelMatcher([("opus", "Claude-Opus-*")])
        assert matcher.match("CLAUDE-OPUS-4-8").mode_name == "opus"
        assert matcher.match("claude-opus-4-8").mode_name == "opus"

    def test_first_declaration_wins_on_a_tie(self):
        matcher = ModelMatcher([("first", "claude-*"), ("second", "claude-*")])
        assert matcher.match("claude-sonnet-5").mode_name == "first"

    def test_no_entry_matches(self):
        matcher = ModelMatcher([("opus", "claude-opus-*")])
        assert matcher.match("gpt-5") is None
        assert matcher.match("qwen/Qwen3.8-27B") is None
        assert matcher.match("claude") is None

    @pytest.mark.parametrize("bad", ["", "   ", None, 5, ["claude-opus-5-5"]])
    def test_unusable_names_never_match(self, bad):
        matcher = ModelMatcher([("opus", "claude-opus-*")])
        assert matcher.match(bad) is None

    def test_families_disabled_is_exact_only(self):
        matcher = ModelMatcher(
            [("opus", "claude-opus-5-5"), ("sonnet", "claude-sonnet-*")],
            families_enabled=False,
        )
        assert matcher.families_enabled is False
        assert matcher.match("claude-opus-5-5").mode_name == "opus"
        assert matcher.match("claude-opus-9-9") is None
        assert matcher.match("claude-sonnet-5") is None
        assert matcher.match("claude-3-5-haiku-20241022") is None

    def test_invalid_pattern_rejected_at_build_time(self):
        with pytest.raises(ValueError):
            ModelMatcher([("opus", "claude-*opus")])


class TestPatternAudits:
    def test_duplicate_literals_across_modes(self):
        found = find_duplicate_literals(
            [("opus", "claude-opus-5-5"), ("sonnet", "Claude-Opus-5-5")]
        )
        assert found == {"claude-opus-5-5": ("opus", "sonnet")}

    def test_duplicate_after_normalization(self):
        found = find_duplicate_literals(
            [
                ("opus", "us.anthropic.claude-opus-4-8-20260101-v1:0"),
                ("sonnet", "claude-opus-4-8"),
            ]
        )
        assert "claude-opus-4-8" in found

    def test_same_mode_declaring_twice_is_not_a_conflict(self):
        assert find_duplicate_literals([("opus", "claude-opus-5-5")] * 2) == {}

    def test_wildcards_are_not_literals(self):
        entries = [("opus", "claude-*"), ("sonnet", "claude-*")]
        assert find_duplicate_literals(entries) == {}

    def test_ambiguous_wildcards_reported_in_order(self):
        found = find_ambiguous_wildcards(
            [
                ("opus", "claude-*"),
                ("sonnet", "claude-*"),
                ("haiku", "claude-haiku-*"),
            ]
        )
        assert found == {"claude-*": ("opus", "sonnet")}


def _raw_config():
    """A minimal but complete config document for parsing tests."""
    return {
        "description": "test config",
        "settings": {
            "enabled": True,
            "match_families": True,
            "model_fields": ["model", "model_name"],
            "provider_prefixes": ["us.anthropic."],
        },
        "claude_code_modes": [
            {
                "name": "opus",
                "model_name": "target/opus",
                "description": "Opus tier",
                "models": ["claude-opus-5-5", "claude-opus-*"],
            },
            {
                "name": "haiku",
                "model_name": "target/haiku",
                "models": ["claude-haiku-*"],
            },
        ],
    }


class TestConfigLoading:
    def test_bundled_config_loads(self):
        config = ClaudeCodeRoutingConfig.from_file()
        assert config.modes
        assert config.enabled is True
        config.validate_args()

    def test_bundled_config_path_resolves(self):
        assert ClaudeCodeRoutingConfig._DEFAULT_CONFIG_PATH == _CONFIG_PATH
        assert _CONFIG_PATH.is_file()

    def test_from_file_explicit_path(self, tmp_path):
        path = tmp_path / "claude_code.json"
        path.write_text(json.dumps(_raw_config()), encoding="utf-8")
        config = ClaudeCodeRoutingConfig.from_file(path)
        assert config.mode_names == ["opus", "haiku"]
        assert config.provider_prefixes == ("us.anthropic.",)
        assert config.patterns == [
            ("opus", "claude-opus-5-5"),
            ("opus", "claude-opus-*"),
            ("haiku", "claude-haiku-*"),
        ]

    def test_from_env_var_raw_json(self, monkeypatch):
        monkeypatch.setenv(f"{_PREFIX}CONFIG", json.dumps(_raw_config()))
        assert ClaudeCodeRoutingConfig.from_file().mode_names == ["opus", "haiku"]

    def test_from_env_var_file_path(self, monkeypatch, tmp_path):
        path = tmp_path / "claude_code.json"
        path.write_text(json.dumps(_raw_config()), encoding="utf-8")
        monkeypatch.setenv(f"{_PREFIX}CONFIG", str(path))
        assert ClaudeCodeRoutingConfig.from_file().mode_names == ["opus", "haiku"]

    def test_from_json(self):
        config = ClaudeCodeRoutingConfig.from_json(json.dumps(_raw_config()))
        assert config.mode_by_name["opus"].model_name == "target/opus"
        assert config.mode_by_name["haiku"].description == ""

    def test_missing_modes_key(self, monkeypatch):
        monkeypatch.setenv(
            f"{_PREFIX}CONFIG", json.dumps({"settings": {"enabled": True}})
        )
        with pytest.raises(KeyError, match="claude_code_modes"):
            ClaudeCodeRoutingConfig.from_file()

    def test_mode_without_name(self, monkeypatch):
        raw = _raw_config()
        del raw["claude_code_modes"][0]["name"]
        monkeypatch.setenv(f"{_PREFIX}CONFIG", json.dumps(raw))
        with pytest.raises(KeyError, match="name"):
            ClaudeCodeRoutingConfig.from_file()

    def test_settings_are_optional(self, monkeypatch):
        raw = _raw_config()
        del raw["settings"]
        monkeypatch.setenv(f"{_PREFIX}CONFIG", json.dumps(raw))
        config = ClaudeCodeRoutingConfig.from_file()
        assert config.enabled is True
        assert config.match_families is True
        assert config.model_fields == DEFAULT_MODEL_FIELDS

    def test_patterns_validated_on_load(self, monkeypatch):
        raw = _raw_config()
        raw["claude_code_modes"][0]["models"] = ["claude-*opus"]
        monkeypatch.setenv(f"{_PREFIX}CONFIG", json.dumps(raw))
        with pytest.raises(ValueError, match="wildcard"):
            ClaudeCodeRoutingConfig.from_file()


class TestConfigValidation:
    def test_valid_config_passes(self):
        _config().validate_args()

    def test_no_modes(self):
        with pytest.raises(ValueError, match="no modes defined"):
            _config(modes=()).validate_args()

    def test_empty_mode_name(self):
        with pytest.raises(ValueError, match="empty name"):
            _config(modes=(_mode("", models=("claude-opus-*",)),)).validate_args()

    def test_duplicate_mode_names(self):
        with pytest.raises(ValueError, match="duplicate mode names"):
            _config(modes=(_mode("opus"), _mode("opus"))).validate_args()

    def test_empty_model_fields(self):
        with pytest.raises(ValueError, match="model_fields"):
            _config(model_fields=()).validate_args()

    def test_blank_model_field(self):
        with pytest.raises(ValueError, match="model_fields"):
            _config(model_fields=("model", "  ")).validate_args()

    def test_two_modes_claiming_one_model(self):
        config = _config(
            modes=(
                _mode("opus", models=("claude-opus-5-5",)),
                _mode("sonnet", models=("claude-opus-5-5",)),
            )
        )
        with pytest.raises(ValueError, match="claimed by modes"):
            config.validate_args()

    def test_mode_without_target_is_valid(self):
        config = _config(modes=(_mode("opus", model_name="", models=("claude-*",)),))
        config.validate_args()

    def test_hand_built_config_is_whitespace_insensitive(self):
        config = ClaudeCodeRoutingConfig(
            model_fields=(" model ", "model_name"),
            provider_prefixes=(" us.anthropic. ", ""),
            modes=(ClaudeCodeMode("opus", "target/opus", ("claude-*",)),),
        )
        assert config.model_fields == ("model", "model_name")
        assert config.provider_prefixes == ("us.anthropic.",)


class TestSignalLint:
    def test_mode_without_target_warns(self, caplog):
        config = _config(modes=(_mode("opus", model_name="", models=("claude-*",)),))
        with caplog.at_level(logging.WARNING):
            config.lint_signals(logging.getLogger("lint"))
        assert "no model_name" in caplog.text

    def test_mode_without_models_warns(self, caplog):
        config = _config(modes=(_mode("opus"),))
        with caplog.at_level(logging.WARNING):
            config.lint_signals(logging.getLogger("lint"))
        assert "can never match" in caplog.text

    def test_ambiguous_wildcard_warns(self, caplog):
        config = _config(
            modes=(
                _mode("opus", models=("claude-*",)),
                _mode("sonnet", models=("claude-*",)),
            )
        )
        with caplog.at_level(logging.WARNING):
            config.lint_signals(logging.getLogger("lint"))
        assert "only 'opus' is ever selected" in caplog.text

    def test_lint_is_silent_without_a_logger(self):
        _config(modes=(_mode("opus", model_name="", models=()),)).lint_signals(None)

    def test_bundled_config_lints_clean(self, caplog, logger):
        with caplog.at_level(logging.WARNING):
            ClaudeCodeRoutingPlugin(logger=logger, config=_shipped_config())
        assert caplog.records == []


class TestEnvironmentOverrides:
    def _load(self, monkeypatch, **env):
        for key, value in env.items():
            monkeypatch.setenv(f"{_PREFIX}{key}", value)
        config = ClaudeCodeRoutingConfig.from_file(_CONFIG_PATH)
        config.override_from_env(logging.getLogger("test_claude_code_env"))
        return config

    def test_enabled_false(self, monkeypatch):
        assert self._load(monkeypatch, ENABLED="false").enabled is False

    def test_match_families_false(self, monkeypatch):
        assert self._load(monkeypatch, MATCH_FAMILIES="0").match_families is False

    def test_fields(self, monkeypatch):
        assert self._load(monkeypatch, FIELDS="model|model_name").model_fields == (
            "model",
            "model_name",
        )
        assert self._load(monkeypatch, FIELDS="model, model").model_fields == (
            "model",
            "model",
        )

    def test_model_per_mode(self, monkeypatch):
        config = self._load(monkeypatch, MODEL_OPUS="some/model")
        assert config.mode_by_name["opus"].model_name == "some/model"

    def test_models_mapping(self, monkeypatch):
        config = self._load(
            monkeypatch, MODELS="opus=model-a|haiku=model-b|ghost=model-c"
        )
        assert config.mode_by_name["opus"].model_name == "model-a"
        assert config.mode_by_name["haiku"].model_name == "model-b"

    def test_models_mapping_warns_on_unknown_mode(self, monkeypatch, caplog):
        with caplog.at_level(logging.WARNING):
            self._load(monkeypatch, MODELS="ghost=model-c")
        assert "unknown mode 'ghost'" in caplog.text

    def test_modes_whitelist(self, monkeypatch):
        config = self._load(monkeypatch, MODES="opus|haiku")
        assert config.mode_names == ["opus", "haiku"]
        assert "sonnet" not in config.mode_names

    def test_mode_models_list_replaced(self, monkeypatch):
        config = self._load(
            monkeypatch,
            MODE_SONNET_MODELS="claude-sonnet-9|us.anthropic.claude-sonnet-8",
        )
        assert config.mode_by_name["sonnet"].models == (
            "claude-sonnet-9",
            "us.anthropic.claude-sonnet-8",
        )

    def test_mode_models_unknown_mode_warns(self, monkeypatch, caplog):
        with caplog.at_level(logging.WARNING):
            self._load(monkeypatch, MODE_GHOST_MODELS="claude-x-*")
        assert "unknown mode 'ghost'" in caplog.text

    def test_override_survives_into_the_matcher(self, monkeypatch):
        config = self._load(monkeypatch, MODE_HAIKU_MODELS="claude-fake-*")
        plugin = ClaudeCodeRoutingPlugin(config=config)
        assert plugin.resolve("claude-fake-1") is not None
        assert plugin.resolve("claude-haiku-4-5") is None


class TestPluginSwap:
    def test_model_swapped(self):
        result = _plugin().apply({"model": "claude-sonnet-5", "messages": ["x"]})
        assert result["model"] == "target/sonnet"
        assert result["messages"] == ["x"]

    def test_model_name_swapped_without_inventing_model(self):
        result = _plugin().apply({"model_name": "claude-opus-5-5", "messages": []})
        assert result["model_name"] == "target/opus"
        assert "model" not in result

    def test_both_fields_swapped(self):
        result = _plugin().apply(
            {"model": "claude-opus-5-5", "model_name": "claude-opus-5-5"}
        )
        assert result["model"] == result["model_name"] == "target/opus"

    def test_model_wins_when_it_is_read_first(self):
        result = _plugin().apply(
            {"model": "claude-haiku-4-5", "model_name": "other"}
        )
        assert result["model"] == result["model_name"] == "target/fast"

    def test_routing_block(self):
        result = _plugin().apply({"model": "claude-opus-9-9"})
        assert set(result["routing"]) == _ROUTING_KEYS
        assert result["routing"]["plugin"] == "agentic_routing_claude_code"
        assert result["routing"]["similarity"] == 1.0
        assert result["routing"]["mode"] == "opus"
        assert result["routing"]["original_model"] == "claude-opus-9-9"
        assert result["routing"]["matched_model"] == "claude-opus-*"
        assert result["routing"]["match_type"] == MATCH_WILDCARD
        assert result["routing"]["field"] == "model"

    def test_third_party_spelling_reported_verbatim_as_original(self):
        raw = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        result = _plugin().apply({"model": raw})
        assert result["routing"]["original_model"] == raw
        assert result["model"] == "target/sonnet"

    def test_swapped_payload_is_the_same_object(self):
        payload = {"model": "claude-opus-5-5"}
        assert _plugin().apply(payload) is payload

    def test_match_logs_one_info_line(self, logger, caplog):
        with caplog.at_level(logging.INFO):
            _plugin(logger=logger).apply({"model": "claude-opus-5-5"})
        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.INFO
        assert "Claude Code model swap" in caplog.text

    @pytest.mark.parametrize(
        "model", ["gpt-5", "qwen/Qwen3.8-27B", "claude", "", "   "]
    )
    def test_unmatched_model_is_silent_passthrough(self, model, logger, caplog):
        payload = {"model": model, "messages": ["x"]}
        with caplog.at_level(logging.DEBUG):
            result = _plugin(logger=logger).apply(payload)
        assert result is payload
        assert caplog.records == []

    def test_payload_without_model_fields(self, logger, caplog):
        payload = {"messages": ["x"], "model": None}
        with caplog.at_level(logging.DEBUG):
            result = _plugin(logger=logger).apply(payload)
        assert result is payload
        assert caplog.records == []

    @pytest.mark.parametrize("payload", [None, [], "text", 7, {"model": 5}])
    def test_unusable_payloads_pass_through(self, payload):
        assert _plugin().apply(payload) is payload

    def test_disabled_plugin_is_inert(self):
        plugin = _plugin(config=_config(enabled=False))
        payload = {"model": "claude-opus-5-5"}
        assert plugin.apply(payload) is payload

    def test_mode_without_target_is_inert(self, logger, caplog):
        plugin = _plugin(
            config=_config(
                modes=(_mode("opus", model_name="", models=("claude-*",)),)
            )
        )
        payload = {"model": "claude-opus-5-5"}
        with caplog.at_level(logging.DEBUG):
            assert plugin.apply(payload) is payload
        assert caplog.records == []

    def test_model_fields_narrow_which_keys_are_read(self):
        plugin = _plugin(config=_config(model_fields=("model_name",)))
        # "model" is no longer read, so nothing matches and nothing is swapped.
        payload = {"model": "claude-opus-5-5", "model_name": "untouched"}
        assert plugin.apply(payload) is payload

    def test_keys_outside_model_fields_keep_their_value(self):
        plugin = _plugin(config=_config(model_fields=("model_name",)))
        payload = {"model": "not-mine", "model_name": "claude-opus-5-5"}
        result = plugin.apply(payload)
        assert result["model"] == "not-mine"
        assert result["model_name"] == "target/opus"

    def test_families_disabled_leaves_a_wildcard_only_name_alone(self):
        plugin = _plugin(config=_config(match_families=False))
        payload = {"model": "claude-opus-4-1"}
        assert plugin.apply(payload) is payload
        assert plugin.apply({"model": "claude-opus-5-5"})["model"] == "target/opus"

    def test_injected_matcher_is_used(self):
        plugin = ClaudeCodeRoutingPlugin(
            config=_config(), matcher=ModelMatcher([("sonnet", "claude-opus-*")])
        )
        assert plugin.apply({"model": "claude-opus-5-5"})["model"] == "target/sonnet"

    def test_matcher_naming_an_unconfigured_mode_passes_through(
        self, logger, caplog
    ):
        plugin = ClaudeCodeRoutingPlugin(
            logger=logger,
            config=_config(),
            matcher=ModelMatcher([("ghost", "claude-*")]),
        )
        payload = {"model": "claude-opus-5-5"}
        with caplog.at_level(logging.WARNING):
            assert plugin.apply(payload) is payload
        assert "not configured" in caplog.text

    def test_broken_matcher_fails_open(self, logger, caplog):
        class _Boom:
            @staticmethod
            def match(_value):
                raise RuntimeError("boom")

        plugin = ClaudeCodeRoutingPlugin(
            logger=logger, config=_config(), matcher=_Boom()
        )
        payload = {"model": "claude-opus-5-5"}
        with caplog.at_level(logging.WARNING):
            assert plugin.apply(payload) is payload
        assert "swap failed" in caplog.text

    def test_accepts_model_config_kwarg(self):
        assert _plugin().apply({"model": "claude-opus-5-5"}, model_config=object())

    def test_apply_is_deterministic(self):
        plugin = _plugin()
        first = plugin.apply({"model": "claude-opus-5-5"})
        second = plugin.apply({"model": "claude-opus-5-5"})
        assert first["model"] == second["model"]
        assert first["routing"] == second["routing"]

    def test_resolve_helper(self):
        plugin = _plugin()
        assert plugin.resolve("claude-haiku-4-5").mode_name == "haiku"
        assert plugin.resolve("gpt-5") is None


class TestShippedConfig:
    """The bundled config must cover the IDs Claude Code emits today."""

    @pytest.mark.parametrize(
        ("model", "mode"),
        [
            ("claude-fable-5-1", "fable"),
            ("claude-fable-5", "fable"),
            ("claude-opus-5-5", "opus"),
            ("claude-opus-4-8", "opus"),
            ("claude-opus-4-6", "opus"),
            ("opusplan", "plan"),
            ("claude-sonnet-5", "sonnet"),
            ("claude-sonnet-4-6", "sonnet"),
            ("claude-sonnet-4-5", "sonnet"),
            ("claude-haiku-4-5", "haiku"),
            ("claude-3-5-haiku-20241022", "haiku"),
            # A future release of a covered family still resolves.
            ("claude-opus-7-7", "opus"),
            ("claude-sonnet-9", "sonnet"),
            # Provider spellings of the same tiers.
            ("claude-sonnet-5[1m]", "sonnet"),
            ("claude-opus-5-5[1m]", "opus"),
            ("us.anthropic.claude-sonnet-4-5-20250929-v1:0", "sonnet"),
            ("eu.anthropic.claude-opus-4-8:1", "opus"),
            ("models/claude-haiku-4-5", "haiku"),
        ],
    )
    def test_tier_mapping(self, model, mode):
        assert _shipped_config().build_matcher().match(model).mode_name == mode

    @pytest.mark.parametrize(
        "model",
        ["gpt-5", "qwen/Qwen3.8-27B", "auto", "auto_codex", "claude", "haikuu"],
    )
    def test_foreign_models_are_left_alone(self, model):
        assert _shipped_config().build_matcher().match(model) is None

    def test_every_shipped_mode_has_a_target(self):
        config = _shipped_config()
        assert all(mode.model_name for mode in config.modes)
        assert all(mode.models for mode in config.modes)

    def test_shipped_payload_round_trip(self):
        plugin = ClaudeCodeRoutingPlugin(config=_shipped_config())
        result = plugin.apply(
            {
                "model": "claude-sonnet-5",
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": "hello"}],
            }
        )
        assert result["model"] == "qwen/Qwen3.8-Flash-Next"
        assert result["max_tokens"] == 4096
        assert result["messages"] == [{"role": "user", "content": "hello"}]
        assert result["routing"]["mode"] == "sonnet"

    def test_shipped_config_is_json_serializable_data(self):
        raw = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        assert {mode["name"] for mode in raw["claude_code_modes"]} == {
            "fable",
            "opus",
            "plan",
            "sonnet",
            "haiku",
        }


class TestRegistry:
    def test_plugin_is_registered(self):
        assert "agentic_routing_claude_code" in MAIN_UTILS_REGISTRY
        assert MAIN_UTILS_REGISTRY["agentic_routing_claude_code"] is (
            ClaudeCodeRoutingPlugin
        )

    def test_coexists_with_codex_plugin(self):
        assert MAIN_UTILS_REGISTRY["agentic_routing_codex"].name != (
            ClaudeCodeRoutingPlugin.name
        )

    def test_registry_construction_uses_the_bundled_config(self):
        plugin = MAIN_UTILS_REGISTRY["agentic_routing_claude_code"](logger=None)
        assert plugin.config.mode_names == _shipped_config().mode_names
        assert plugin.resolve("claude-sonnet-5").mode_name == "sonnet"


class TestCopySafety:
    """A payload copy behaves exactly like the original."""

    def test_copy_of_a_payload(self):
        payload = {"model": "claude-sonnet-5", "messages": ["x"]}
        result = _plugin().apply(copy.deepcopy(payload))
        assert result["model"] == "target/sonnet"
        assert payload["model"] == "claude-sonnet-5"
