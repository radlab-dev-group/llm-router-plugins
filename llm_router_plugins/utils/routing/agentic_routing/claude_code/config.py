"""
Configuration dataclass for the Claude Code model swap plugin.

JSON structure::

    {
      "description": "Claude Code model swap configuration — ...",
      "settings": {
        "enabled": true,
        "match_families": true,
        "model_fields": ["model", "model_name"],
        "provider_prefixes": ["us.anthropic.", "anthropic."]
      },
      "claude_code_modes": [
        {
          "name": "opus",
          "model_name": "qwen/Qwen3.8-Flash-Next",
          "description": "Opus tier: the complex-reasoning model the user picked",
          "models": [
            "claude-opus-5-5",
            "claude-opus-4-8",
            "claude-opus-*"
          ]
        }
      ]
    }

A mode is a Claude Code tier (``fable`` / ``opus`` / ``plan`` / ``sonnet`` /
``haiku``): ``models`` are the names the CLI sends for that tier and
``model_name`` is what the plugin writes into the payload instead.  Entries in
``models`` may end with ``*`` to cover a whole family or version line; see
:mod:`~llm_router_plugins.utils.routing.agentic_routing.claude_code.mapping`.
"""

import logging
import os
import pathlib
import re

from dataclasses import dataclass, replace
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from llm_router_plugins.utils.routing.agentic_routing.claude_code.mapping import (
    DEFAULT_MODEL_FIELDS,
    DEFAULT_PROVIDER_PREFIXES,
    ModelMatcher,
    find_ambiguous_wildcards,
    find_duplicate_literals,
    iter_mode_entries,
    validate_pattern,
)
from llm_router_plugins.utils.routing.common import (
    RoutingConfigBase,
    env_bool,
)
from llm_router_plugins.utils.routing.constants import (
    AGENTIC_CLAUDE_CODE_ROUTING_PREFIX,
)

__all__ = [
    "ClaudeCodeRoutingConfig",
    "ClaudeCodeMode",
]

# Separators accepted in list-typed environment variables.
_LIST_SEPARATOR_RE = re.compile(r"[|,]")


@dataclass(frozen=True)
class ClaudeCodeMode:
    """
    Definition of a single Claude Code tier.

    Parameters
    ----------
    name : str
        Unique identifier of the tier, reported as ``routing.mode``.
    model_name : str
        The model written into the payload when a request matches this tier.
        An empty value is valid and makes matching requests pass through.
    models : Tuple[str, ...]
        Model names this tier answers.  Each entry is either an exact name
        (compared literally and after normalization) or a trailing-wildcard
        pattern such as ``"claude-opus-*"``.
    description : str
        Human-readable note; documentation only, never matched against.
    """

    name: str
    model_name: str = ""
    models: Tuple[str, ...] = ()
    description: str = ""


@dataclass
class ClaudeCodeRoutingConfig(RoutingConfigBase):
    """
    Snapshot of the Claude Code swap configuration.

    Mutable on purpose: :meth:`_override_from_env` applies environment
    variable overrides in place before the configuration is consumed.

    Parameters
    ----------
    enabled : bool
        Master switch; when ``False`` the plugin passes every payload through.
    match_families : bool
        Whether trailing-wildcard and family matching participate.  With
        ``False`` only exact names match.
    model_fields : Tuple[str, ...]
        Payload keys read for the model name and rewritten with the target.
    provider_prefixes : Tuple[str, ...]
        Vendor prefixes stripped from a model name before comparison.
    modes : Tuple[ClaudeCodeMode, ...]
        Immutable sequence of tier definitions, in configuration order.
    """

    # RoutingConfigBase hooks (ClassVar — not dataclass fields)
    _ENV_PREFIX: ClassVar[str] = AGENTIC_CLAUDE_CODE_ROUTING_PREFIX
    _DEFAULT_CONFIG_PATH: ClassVar[Optional[pathlib.Path]] = (
        pathlib.Path(__file__).resolve().parents[4]
        / "resources"
        / "routing"
        / "agentic_routing_claude_code.json"
    )

    enabled: bool = True
    match_families: bool = True
    model_fields: Tuple[str, ...] = DEFAULT_MODEL_FIELDS
    provider_prefixes: Tuple[str, ...] = DEFAULT_PROVIDER_PREFIXES
    modes: Tuple[ClaudeCodeMode, ...] = ()

    @property
    def mode_names(self) -> List[str]:
        """
        Return the names of all configured tiers.

        Returns
        -------
        List[str]
            Tier names in configuration order.
        """
        return [mode.name for mode in self.modes]

    @property
    def mode_by_name(self) -> Dict[str, ClaudeCodeMode]:
        """
        Return a mapping from tier name to :class:`ClaudeCodeMode`.

        Returns
        -------
        Dict[str, ClaudeCodeMode]
            A dictionary mapping each tier name to its configuration.
        """
        return {mode.name: mode for mode in self.modes}

    @property
    def patterns(self) -> List[Tuple[str, str]]:
        """
        Return the flattened ``(mode_name, pattern)`` pairs of all tiers.

        Returns
        -------
        List[Tuple[str, str]]
            Every configured model pattern, tagged with its tier name.
        """
        return iter_mode_entries(self.modes)

    def build_matcher(self) -> ModelMatcher:
        """
        Build the matcher that resolves a model name to a tier.

        Returns
        -------
        ModelMatcher
            A matcher over the configured patterns and provider prefixes.
        """
        return ModelMatcher(
            self.patterns,
            provider_prefixes=self.provider_prefixes,
            families_enabled=self.match_families,
        )

    def override_from_env(self, logger: Optional[logging.Logger] = None) -> None:
        """Apply the environment variable overrides documented below."""
        self._override_from_env(logger=logger)

    def validate_args(self) -> None:
        """Validate the configuration; raises ``ValueError`` when unusable."""
        self._validate_args()

    def lint_signals(self, logger: Optional[logging.Logger] = None) -> None:
        """
        Report suspicious configuration as warnings, changing nothing.

        A tier without ``model_name`` cannot swap anything, a tier without
        ``models`` can never be reached, and a wildcard declared by two tiers
        only ever serves the first of them.  None of these is an error, so they
        are reported instead of raised.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger used to report the findings.  When ``None`` this does
            nothing.

        Returns
        -------
        None
        """
        if logger is None:
            return

        for mode in self.modes:
            if not mode.model_name:
                logger.warning(
                    "ClaudeCodeRouting: mode '%s' has no model_name — requests "
                    "matching it are passed through unchanged",
                    mode.name,
                )
            if not mode.models:
                logger.warning(
                    "ClaudeCodeRouting: mode '%s' declares no models — it can "
                    "never match a request",
                    mode.name,
                )

        for pattern, owners in find_ambiguous_wildcards(self.patterns).items():
            logger.warning(
                "ClaudeCodeRouting: wildcard %r is declared by modes %s — only "
                "'%s' is ever selected",
                pattern,
                ", ".join(owners),
                owners[0],
            )

    @classmethod
    def _from_raw(cls, raw: Dict[str, Any]) -> "ClaudeCodeRoutingConfig":
        """Parse and validate the decoded JSON dict (``RoutingConfigBase`` hook).

        Raises
        ------
        KeyError
            If ``claude_code_modes`` is missing, or a mode has no ``name``.
        ValueError
            If a model pattern is empty or embeds a wildcard.
        """
        if "claude_code_modes" not in raw:
            raise KeyError(
                "Missing required top-level key 'claude_code_modes' in config. "
                f"Available keys: {list(raw.keys())}"
            )

        settings = raw.get("settings") or {}
        model_fields = _as_tuple(settings.get("model_fields"), DEFAULT_MODEL_FIELDS)
        provider_prefixes = _as_tuple(
            settings.get("provider_prefixes"), DEFAULT_PROVIDER_PREFIXES
        )

        modes = tuple(
            ClaudeCodeMode(
                name=str(mode["name"]).strip(),
                model_name=str(mode.get("model_name", "") or "").strip(),
                models=_as_tuple(mode.get("models"), ()),
                description=str(mode.get("description", "") or ""),
            )
            for mode in raw["claude_code_modes"]
        )
        for mode in modes:
            for pattern in mode.models:
                validate_pattern(pattern)

        return ClaudeCodeRoutingConfig(
            enabled=bool(settings.get("enabled", True)),
            match_families=bool(settings.get("match_families", True)),
            model_fields=model_fields,
            provider_prefixes=provider_prefixes,
            modes=modes,
        )

    def _override_from_env(self, logger: Optional[logging.Logger] = None) -> None:
        """
        Apply environment variable overrides to the configuration **in place**.

        Supported environment variables (prefix
        ``LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CLAUDE_CODE_``):

        - ``CONFIG`` — the full configuration, as a raw JSON string or a path
        - ``ENABLED`` — ``1/0``, ``true/false``, ``yes/no``, ``on/off``
        - ``MATCH_FAMILIES`` — turn wildcard and family matching on/off
        - ``FIELDS`` — payload keys to read and rewrite, e.g. ``model|model_name``
        - ``MODEL_<MODE>`` — target model of one tier, e.g. ``MODEL_OPUS``
        - ``MODELS`` — per-tier targets, e.g. ``opus=model_a|haiku=model_b``
        - ``MODES`` — pipe-separated whitelist of tier names
        - ``MODE_<name>_MODELS`` — names served by one tier, replacing the
          configured list, e.g.
          ``MODE_SONNET_MODELS=claude-sonnet-5|claude-sonnet-*``

        Unknown tier names are logged as warnings and otherwise ignored.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger used to report applied overrides.

        Returns
        -------
        None
        """
        self._override_switches(logger)
        self._override_mode_names(logger)
        self._override_mode_targets(logger)
        self._override_mode_filter(logger)

    def _override_switches(self, logger: Optional[logging.Logger]) -> None:
        """
        Apply ``ENABLED``, ``MATCH_FAMILIES`` and ``FIELDS``.

        Covers the master switch, the wildcard/family switch and the payload
        keys the plugin reads and rewrites.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger used to report applied overrides.

        Returns
        -------
        None
        """
        enabled = env_bool(AGENTIC_CLAUDE_CODE_ROUTING_PREFIX, "ENABLED", logger)
        if enabled is not None:
            self.enabled = enabled

        match_families = env_bool(
            AGENTIC_CLAUDE_CODE_ROUTING_PREFIX, "MATCH_FAMILIES", logger
        )
        if match_families is not None:
            self.match_families = match_families

        fields_env = os.getenv(f"{AGENTIC_CLAUDE_CODE_ROUTING_PREFIX}FIELDS")
        if fields_env and _split_list(fields_env):
            self.model_fields = _split_list(fields_env)
            if logger:
                logger.info(
                    "Overriding Claude Code model fields: %s",
                    "|".join(self.model_fields),
                )

    def _override_mode_names(self, logger: Optional[logging.Logger]) -> None:
        """
        Replace per-tier name lists from ``MODE_<name>_MODELS`` variables.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger used to report applied overrides and unknown tiers.

        Returns
        -------
        None
        """
        prefix = f"{AGENTIC_CLAUDE_CODE_ROUTING_PREFIX}MODE_"
        suffix = "_MODELS"
        for env_name, env_value in os.environ.items():
            if not env_name.startswith(prefix) or not env_name.endswith(suffix):
                continue
            mode_name = env_name[len(prefix) : -len(suffix)].lower()
            if mode_name not in self.mode_by_name:
                if logger:
                    logger.warning(
                        "Ignoring %s override for unknown mode '%s'",
                        env_name,
                        mode_name,
                    )
                continue
            self._replace_mode(
                mode_name, logger, "models", models=_split_list(env_value)
            )

    def _override_mode_targets(self, logger: Optional[logging.Logger]) -> None:
        """
        Replace per-tier target models from ``MODEL_<MODE>`` and ``MODELS``.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger used to report applied overrides and unknown tiers.

        Returns
        -------
        None
        """
        for mode in self.modes:
            model_env = os.getenv(
                f"{AGENTIC_CLAUDE_CODE_ROUTING_PREFIX}MODEL_{mode.name.upper()}"
            )
            if model_env and model_env.strip():
                self._replace_mode(
                    mode.name, logger, "model", model_name=model_env.strip()
                )

        models_env = os.getenv(f"{AGENTIC_CLAUDE_CODE_ROUTING_PREFIX}MODELS")
        if not models_env:
            return

        known = self.mode_by_name
        for pair in models_env.split("|"):
            if "=" not in pair:
                continue
            mode_name, _, target = pair.partition("=")
            mode_name, target = mode_name.strip().lower(), target.strip()
            if not mode_name or not target:
                continue
            if mode_name not in known:
                if logger:
                    logger.warning(
                        "Ignoring MODELS override for unknown mode '%s'",
                        mode_name,
                    )
                continue
            self._replace_mode(mode_name, logger, "model", model_name=target)

    def _override_mode_filter(self, logger: Optional[logging.Logger]) -> None:
        """
        Keep only the tiers listed in ``MODES``.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger used to report applied overrides.

        Returns
        -------
        None
        """
        modes_env = os.getenv(f"{AGENTIC_CLAUDE_CODE_ROUTING_PREFIX}MODES")
        if not modes_env:
            return

        allowed = set(self.mode_names)
        selected = [name for name in _split_list(modes_env) if name in allowed]
        if selected and selected != self.mode_names:
            keep = set(selected)
            self.modes = tuple(mode for mode in self.modes if mode.name in keep)
            if logger:
                logger.info(
                    "Overriding Claude Code modes: %s",
                    "|".join(selected),
                )

    def _replace_mode(
        self,
        mode_name: str,
        logger: Optional[logging.Logger],
        label: str,
        **changes: Any,
    ) -> None:
        """
        Replace one tier in :attr:`modes` and log the applied override.

        Parameters
        ----------
        mode_name : str
            Name of the tier to rebuild; it must already be configured.
        logger : logging.Logger, optional
            Logger used to report the applied override.
        label : str
            What was overridden, e.g. ``"model"`` or ``"models"``.
        **changes : Any
            Fields to overwrite on the mode (a single field per call).

        Returns
        -------
        None
        """
        self.modes = tuple(
            replace(mode, **changes) if mode.name == mode_name else mode
            for mode in self.modes
        )
        if logger:
            logger.info(
                "Overriding %s for mode '%s': %s",
                label,
                mode_name,
                next(iter(changes.values())),
            )

    def _validate_args(self) -> None:
        """
        Validate that the configuration is usable after env overrides.

        An empty ``model_name`` is valid: matching requests are passed through
        instead of rewritten, and :meth:`lint_signals` says so.

        Raises
        ------
        ValueError
            If no tier is defined, a tier name is empty or duplicated, a model
            pattern is empty or embeds a wildcard, no payload key is
            configured, or two tiers claim the same exact model name.
        """
        if not self.modes:
            raise ValueError(
                "ClaudeCodeRouting: no modes defined — check 'claude_code_modes' "
                "in the JSON config or the "
                f"{AGENTIC_CLAUDE_CODE_ROUTING_PREFIX}MODES environment variable"
            )

        names = self.mode_names
        if not all(name for name in names):
            raise ValueError(
                "ClaudeCodeRouting: a mode has an empty name — check "
                "'claude_code_modes' in the JSON config"
            )

        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"ClaudeCodeRouting: duplicate mode names {duplicates} — check "
                "'claude_code_modes' in the JSON config"
            )

        if not self.model_fields or not all(
            isinstance(name, str) and name.strip() for name in self.model_fields
        ):
            raise ValueError(
                "ClaudeCodeRouting: model_fields must list at least one payload "
                "key — check 'settings.model_fields' in the JSON config or the "
                f"{AGENTIC_CLAUDE_CODE_ROUTING_PREFIX}FIELDS environment variable"
            )

        for mode in self.modes:
            for pattern in mode.models:
                validate_pattern(pattern)

        claimed = find_duplicate_literals(
            self.patterns, provider_prefixes=self.provider_prefixes
        )
        if claimed:
            detail = ", ".join(
                f"'{pattern}' claimed by modes '{first}' and '{second}'"
                for pattern, (first, second) in sorted(claimed.items())
            )
            raise ValueError(
                f"ClaudeCodeRouting: {detail} — check 'claude_code_modes'"
            )

    def __post_init__(self) -> None:
        # Normalise whitespace so that a hand-built config matches the JSON one.
        self.model_fields = tuple(
            name.strip() for name in self.model_fields if isinstance(name, str)
        )
        self.provider_prefixes = tuple(
            prefix.strip()
            for prefix in self.provider_prefixes
            if isinstance(prefix, str) and prefix.strip()
        )


def _as_tuple(value: Any, default: Tuple[str, ...]) -> Tuple[str, ...]:
    """
    Coerce a JSON value into a tuple of strings.

    Parameters
    ----------
    value : Any
        A list of strings, a single string, or ``None``.
    default : Tuple[str, ...]
        Fallback used when *value* is ``None`` or holds no usable entry.

    Returns
    -------
    Tuple[str, ...]
        The stripped, non-empty entries, or *default*.
    """
    if value is None:
        return default
    candidates = value if isinstance(value, (list, tuple)) else [value]
    cleaned = tuple(str(item).strip() for item in candidates if str(item).strip())
    return cleaned or default


def _split_list(value: str) -> Tuple[str, ...]:
    """
    Split an environment-variable list into stripped entries.

    Both ``|`` and ``,`` separate entries, so that a model name containing a
    comma never has to be quoted.

    Parameters
    ----------
    value : str
        The raw environment-variable value.

    Returns
    -------
    Tuple[str, ...]
        The non-empty entries, in the order they appear.
    """
    return tuple(
        item.strip() for item in _LIST_SEPARATOR_RE.split(value) if item.strip()
    )
