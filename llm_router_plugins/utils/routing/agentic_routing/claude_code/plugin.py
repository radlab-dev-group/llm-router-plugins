"""
Claude Code model swap plugin.

Rewrites the model Claude Code asked for into the model the operator wants to
serve it.  Claude Code resolves its own ``opus`` / ``sonnet`` / ``haiku`` /
``fable`` aliases client-side and sends a versioned model ID, so pointing it at
a non-Anthropic backend used to mean overriding one environment variable per
tier — ``ANTHROPIC_DEFAULT_FABLE_MODEL``, ``ANTHROPIC_DEFAULT_OPUS_MODEL``,
``ANTHROPIC_DEFAULT_SONNET_MODEL``, ``ANTHROPIC_DEFAULT_HAIKU_MODEL``,
``CLAUDE_CODE_SUBAGENT_MODEL`` — and repeating that on every host.  Here the
tier-to-model mapping lives in one configuration file instead: each mode names
the model IDs it answers and the model to serve them with.

Only the model name changes.  ``payload["model"]`` and every other configured
key from ``settings.model_fields`` receives the tier's target model, and the
payload gains a ``routing`` block describing the decision.  Messages, tools,
system prompts and headers are forwarded untouched.

Nothing here classifies or inspects request content: the incoming model name
selects the tier through :class:`~llm_router_plugins.utils.routing.
agentic_routing.claude_code.mapping.ModelMatcher`, which is why an unmatched
request costs one dictionary lookup and no log line.

Example
-------
::

    plugin = ClaudeCodeRoutingPlugin(logger)
    result = plugin.apply({"model": "claude-sonnet-5", "messages": [...]})
    result["model"]                 # "qwen/Qwen3.8-Flash-Next"
    result["routing"]["mode"]       # "sonnet"
    result["routing"]["original_model"]  # "claude-sonnet-5"
"""

import logging

from typing import Any, Dict, Optional, Tuple

from llm_router_plugins.plugin_interface import PluginInterface
from llm_router_plugins.utils.routing.agentic_routing.claude_code.config import (
    ClaudeCodeMode,
    ClaudeCodeRoutingConfig,
)
from llm_router_plugins.utils.routing.agentic_routing.claude_code.mapping import (
    ModelMatch,
    ModelMatcher,
)
from llm_router_plugins.utils.routing.common import annotate_routing

__all__ = ["ClaudeCodeRoutingPlugin"]

# The decision is deterministic, not a similarity score: 1.0 keeps the
# "routing" block shaped like the semantic plugins' for downstream consumers.
_ROUTING_SIMILARITY = 1.0

# Key that always carries the model name in an Anthropic Messages request body.
_MODEL_FIELD = "model"

# What ``_resolve`` hands back: the winning tier, the match that selected it,
# the payload key it was read from, and the value as it appeared there.
_Resolution = Tuple[ClaudeCodeMode, ModelMatch, str, str]


class ClaudeCodeRoutingPlugin(PluginInterface):
    """
    Model swap plugin for requests emitted by the Claude Code CLI.

    Attributes
    ----------
    name : str
        Plugin identifier (``"agentic_routing_claude_code"``).
    """

    name = "agentic_routing_claude_code"

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        config: Optional[ClaudeCodeRoutingConfig] = None,
        matcher: Optional[ModelMatcher] = None,
    ) -> None:
        """
        Initialize the plugin, loading and validating configuration.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger used for diagnostics.  When ``None`` the plugin stays quiet.
        config : ClaudeCodeRoutingConfig, optional
            Preloaded configuration.  When ``None`` the bundled JSON config is
            loaded (or the
            ``LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CLAUDE_CODE_CONFIG`` env var).
        matcher : ModelMatcher, optional
            Pre-built matcher.  Injecting one skips :meth:`
            ClaudeCodeRoutingConfig.build_matcher`, which is what keeps the
            plugin testable without a config file.

        Raises
        ------
        ValueError
            If the configuration is inconsistent, for instance no mode is
            defined or two modes claim the same exact model name.
        KeyError
            If the loaded JSON config is missing required keys.
        """
        super().__init__(logger=logger)
        self._config: ClaudeCodeRoutingConfig = (
            config if config is not None else ClaudeCodeRoutingConfig.from_file()
        )
        self._config.override_from_env(self._logger)
        self._config.validate_args()
        self._config.lint_signals(self._logger)

        self._matcher: ModelMatcher = (
            matcher if matcher is not None else self._config.build_matcher()
        )

    @property
    def config(self) -> ClaudeCodeRoutingConfig:
        """
        Return the configuration the plugin runs on.

        Returns
        -------
        ClaudeCodeRoutingConfig
            The loaded, env-overridden and validated configuration.
        """
        return self._config

    def resolve(self, model_name: Any) -> Optional[ModelMatch]:
        """
        Resolve a model name to the mode that would serve it.

        Useful for verifying a configuration without building a payload.

        Parameters
        ----------
        model_name : Any
            A model name as it would appear in the payload.

        Returns
        -------
        ModelMatch or None
            The winning match, or ``None`` when the name matches no mode.
        """
        return self._matcher.match(model_name)

    def apply(
        self,
        payload: Any,
        model_config: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Swap the requested Claude Code model for the configured target.

        The plugin activates when one of the configured payload keys carries a
        string that a mode claims, and rewrites every configured key the payload
        carries.  Anything else — a non-dict payload, a plugin switched off, a
        model no mode claims, a mode with no target model — is returned as the
        very same object it came in as, without a word.

        Parameters
        ----------
        payload : Any
            The incoming request payload.
        model_config : Any, optional
            Accepted for interface compatibility; unused by this plugin.
        **kwargs : Any
            Accepted for interface compatibility; unused by this plugin.

        Returns
        -------
        Any
            The payload.  Rewritten payloads have the target model in every
            configured key they carried plus a ``routing`` block; unrouted
            payloads are the very same object that was passed in.

        Raises
        ------
        None
        """
        if not isinstance(payload, dict):
            return payload
        if not self._config.enabled:
            return payload

        try:
            resolution = self._resolve(payload)
            if resolution is None:
                return payload
            if not resolution[0].model_name:
                # Configured but not swappable: the lint already said so.
                return payload

            return self._swap(payload, resolution)
        except Exception as exc:  # a swap bug must never break a request
            self._warn(
                "Claude Code model swap failed, passing the request through: %s", exc
            )
            return payload

    def _resolve(self, payload: Dict[str, Any]) -> Optional[_Resolution]:
        """
        Find the mode that claims the model this payload asks for.

        Parameters
        ----------
        payload : dict
            The incoming request payload.

        Returns
        -------
        Tuple[ClaudeCodeMode, ModelMatch, str, str] or None
            The winning mode, the match that selected it, the payload key the
            name was read from and the name as it appeared there; ``None`` when
            no configured key carries a claimed model.
        """
        for field_name in self._config.model_fields:
            raw = payload.get(field_name)
            if not isinstance(raw, str) or not raw.strip():
                continue
            match = self._matcher.match(raw)
            if match is None:
                continue
            mode = self._config.mode_by_name.get(match.mode_name)
            if mode is None:
                self._warn(
                    "ClaudeCodeRouting: matcher resolved mode '%s', which is not "
                    "configured — passing the request through",
                    match.mode_name,
                )
                return None
            return mode, match, field_name, raw.strip()
        return None

    def _swap(
        self, payload: Dict[str, Any], resolution: _Resolution
    ) -> Dict[str, Any]:
        """
        Write the tier's target model into *payload* and annotate the decision.

        Parameters
        ----------
        payload : dict
            The payload to rewrite, in place.
        resolution : Tuple[ClaudeCodeMode, ModelMatch, str, str]
            The winning tier, the match that selected it, the payload key the
            model name was read from and that name as it appeared there.

        Returns
        -------
        dict
            The rewritten payload.
        """
        mode, match, source_field, original = resolution
        # ``annotate_routing`` always writes payload["model"], which is right
        # only when this payload names its model that way: remember what the
        # key held so a contract that spells it differently keeps its value.
        had_model_field = _MODEL_FIELD in payload
        original_model_field_value = payload.get(_MODEL_FIELD)

        # Only keys the payload already carries are rewritten, so a gateway
        # contract that names the model one way is never widened.
        fields = tuple(name for name in self._config.model_fields if name in payload)

        annotated = annotate_routing(
            payload,
            self.name,
            mode.model_name,
            _ROUTING_SIMILARITY,
            mode=mode.name,
            original_model=original,
            matched_model=match.pattern,
            match_type=match.kind,
            field=source_field,
        )
        if _MODEL_FIELD not in fields:
            if had_model_field:
                annotated[_MODEL_FIELD] = original_model_field_value
            else:
                annotated.pop(_MODEL_FIELD, None)
        for field_name in fields:
            annotated[field_name] = mode.model_name

        self._info(
            "Claude Code model swap: %s -> %s (mode=%s, matched=%s, "
            "match_type=%s, field=%s)",
            original,
            mode.model_name,
            mode.name,
            match.pattern,
            match.kind,
            source_field,
        )
        return annotated

    def _info(self, message: str, *args: Any) -> None:
        """
        Log an info line when a logger is available.

        Parameters
        ----------
        message : str
            The log message, optionally with ``%`` placeholders.
        *args : Any
            Arguments for the ``%`` placeholders.

        Returns
        -------
        None
        """
        if self._logger is not None:
            self._logger.info(message, *args)

    def _warn(self, message: str, *args: Any) -> None:
        """
        Log a warning when a logger is available.

        Parameters
        ----------
        message : str
            The log message, optionally with ``%`` placeholders.
        *args : Any
            Arguments for the ``%`` placeholders.

        Returns
        -------
        None
        """
        if self._logger is not None:
            self._logger.warning(message, *args)
