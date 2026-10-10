"""Validated phase rules from the configuration supplied to the router."""

import re
from dataclasses import dataclass
from typing import Optional, Pattern, Tuple

#: Phase fields a configuration may omit.  Each has a safe default that keeps
#: the behaviour of the version that predates it, so an older config file keeps
#: working untouched; nothing is merged in from the shipped default file.
OPTIONAL_FIELDS = frozenset(
    {
        "neutral_filters",
        "announcement_followup_max_chars",
        "neutral_executables",
        "implement_write_patterns",
        "activity_description_limit",
        "evidence_window",
    }
)

#: A rule that names the work the agent is doing: running tests, inspecting
#: history, editing files.
STRENGTH_STRONG = "strong"

#: A rule that only accompanies the work — a linter, a type checker — and
#: therefore never overrides a strong signal of the same turn.
STRENGTH_WEAK = "weak"

#: Strengths a command rule may declare.
STRENGTHS = frozenset({STRENGTH_STRONG, STRENGTH_WEAK})


@dataclass(frozen=True)
class PhaseCommandRule:
    executable: Pattern[str]
    args_prefix: Tuple[str, ...]
    mode: Optional[str]
    strength: str = "strong"


@dataclass(frozen=True)
class CodexPhaseConfig:
    enabled: bool
    announcement_prefix: Pattern[str]
    uncertain: Pattern[str]
    announcements: Tuple[Tuple[str, Pattern[str]], ...]
    commands: Tuple[PhaseCommandRule, ...]
    command_tools: Tuple[str, ...]
    patch_tools: Tuple[str, ...]
    test_directories: Tuple[str, ...]
    test_filename_prefixes: Tuple[str, ...]
    test_filename_pattern: Pattern[str]
    test_mode: str
    implement_mode: str
    failure_mode: str
    neutral_filters: Tuple[Pattern[str], ...] = ()
    announcement_followup_max_chars: int = 0
    neutral_executables: Tuple[Pattern[str], ...] = ()
    implement_write_patterns: Tuple[Pattern[str], ...] = ()
    activity_description_limit: int = 6
    evidence_window: int = 1

    @staticmethod
    def _followup_budget(value):
        """A non-negative cap on the prose that may follow an announcement."""
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                "settings.phase.announcement_followup_max_chars must be a "
                "non-negative integer"
            )
        return value

    @staticmethod
    def _description_limit(value):
        """Number of distinct actions a semantic activity description names."""
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                "settings.phase.activity_description_limit must be a "
                "non-negative integer"
            )
        return value

    @staticmethod
    def _window(value):
        """How many recent signals decide; one signal is the latest action."""
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                "settings.phase.evidence_window must be a positive integer"
            )
        return value

    @classmethod
    def from_raw(cls, raw, mode_names=None):
        """Validate supplied rules; never read or merge another configuration."""
        if not isinstance(raw, dict):
            raise ValueError("settings.phase must be an object")
        fields = set(cls.__dataclass_fields__)
        unknown = raw.keys() - fields
        if unknown:
            raise ValueError(f"Unknown settings.phase fields: {sorted(unknown)}")
        missing = fields - raw.keys() - OPTIONAL_FIELDS
        if missing:
            raise ValueError(f"Missing settings.phase fields: {sorted(missing)}")
        data = raw

        def strings(value, label):
            if not isinstance(value, list) or any(
                not isinstance(item, str) or not item for item in value
            ):
                raise ValueError(
                    f"settings.phase.{label} must be a list of nonempty strings"
                )
            return tuple(value)

        def pattern(value, label, flags=0):
            if not isinstance(value, str) or not value:
                raise ValueError(f"settings.phase.{label} must be a nonempty regex")
            try:
                return re.compile(value, flags)
            except re.error as exc:
                raise ValueError(
                    f"Invalid settings.phase.{label} regex: {exc}"
                ) from exc

        def mode(value, label, nullable=False):
            if value is None and nullable:
                return None
            if not isinstance(value, str) or not value:
                raise ValueError(f"settings.phase.{label} must name a mode")
            return value

        if type(data["enabled"]) is not bool:
            raise ValueError("settings.phase.enabled must be a boolean")
        announcements = data["announcements"]
        if not isinstance(announcements, dict):
            raise ValueError("settings.phase.announcements must be an object")
        compiled_announcements = tuple(
            (
                mode(name, "announcements"),
                pattern(regex, f"announcements.{name}", re.IGNORECASE),
            )
            for name, regex in announcements.items()
        )
        commands = data["commands"]
        if not isinstance(commands, list):
            raise ValueError("settings.phase.commands must be a list")
        compiled_commands = []
        for index, rule in enumerate(commands):
            label = f"commands[{index}]"
            if not isinstance(rule, dict) or not {
                "executable",
                "args_prefix",
                "mode",
            }.issubset(rule) or set(rule) - {
                "executable",
                "args_prefix",
                "mode",
                "strength",
            }:
                raise ValueError(
                    f"settings.phase.{label} requires executable, args_prefix and mode"
                )
            strength = rule.get("strength", "strong")
            if strength not in STRENGTHS:
                raise ValueError(
                    f"settings.phase.{label}.strength must be one of "
                    f"{sorted(STRENGTHS)}"
                )
            compiled_commands.append(
                PhaseCommandRule(
                    pattern(rule["executable"], label + ".executable"),
                    strings(rule["args_prefix"], label + ".args_prefix"),
                    mode(rule["mode"], label + ".mode", nullable=True),
                    strength,
                )
            )
        result = cls(
            enabled=data["enabled"],
            announcement_prefix=pattern(
                data["announcement_prefix"], "announcement_prefix", re.IGNORECASE
            ),
            uncertain=pattern(data["uncertain"], "uncertain", re.IGNORECASE),
            announcements=compiled_announcements,
            commands=tuple(compiled_commands),
            command_tools=strings(data["command_tools"], "command_tools"),
            patch_tools=strings(data["patch_tools"], "patch_tools"),
            test_directories=strings(data["test_directories"], "test_directories"),
            test_filename_prefixes=strings(
                data["test_filename_prefixes"], "test_filename_prefixes"
            ),
            test_filename_pattern=pattern(
                data["test_filename_pattern"], "test_filename_pattern", re.IGNORECASE
            ),
            test_mode=mode(data["test_mode"], "test_mode"),
            implement_mode=mode(data["implement_mode"], "implement_mode"),
            failure_mode=mode(data["failure_mode"], "failure_mode"),
            neutral_filters=tuple(
                pattern(value, "neutral_filters")
                for value in strings(
                    data.get("neutral_filters", []), "neutral_filters"
                )
            ),
            announcement_followup_max_chars=cls._followup_budget(
                data.get("announcement_followup_max_chars", 0)
            ),
            neutral_executables=tuple(
                pattern(value, "neutral_executables")
                for value in strings(
                    data.get("neutral_executables", []), "neutral_executables"
                )
            ),
            implement_write_patterns=tuple(
                pattern(value, "implement_write_patterns", re.IGNORECASE)
                for value in strings(
                    data.get("implement_write_patterns", []),
                    "implement_write_patterns",
                )
            ),
            activity_description_limit=cls._description_limit(
                data.get("activity_description_limit", 6)
            ),
            evidence_window=cls._window(data.get("evidence_window", 1)),
        )
        if mode_names is not None:
            references = set()
            if "announcements" in raw:
                references.update(name for name, _ in result.announcements)
            if "commands" in raw:
                references.update(
                    rule.mode for rule in result.commands if rule.mode is not None
                )
            references.update(
                raw[key]
                for key in ("test_mode", "implement_mode", "failure_mode")
                if key in raw
            )
            unknown_modes = references - set(mode_names)
            if unknown_modes:
                raise ValueError(
                    f"Unknown settings.phase modes: {sorted(unknown_modes)}"
                )
        return result
