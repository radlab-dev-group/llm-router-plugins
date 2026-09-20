"""
Codex CLI routing plugin.

Routes OpenAI-Responses-style requests emitted by the Codex CLI coding agent.
When ``payload["model"]`` equals the configured trigger (default
``"auto_codex"``) the plugin classifies the request, resolves the Codex work
mode and rewrites ``payload["model"]`` to the model configured for that mode.

The decision is written back by :func:`~llm_router_plugins.utils.routing.
common.annotate_routing`, so the payload gains a ``routing`` block describing
the outcome and ``payload["agent_mode"]`` is set to the resolved mode.  No
other field is touched: ``tools``, ``instructions``, ``input``, ``reasoning``,
``text`` and ``stream`` are forwarded unchanged.

Detection runs as a deterministic-first cascade — request-class rules, the
``<collaboration_mode>`` block declared by the CLI and keyword scoring — so an
identical request is answered identically, offline and for free.  An optional
semantic layer then contributes embedding cosine similarity over the mode
descriptions and examples, reusing the same BiEncoder + FAISS router as the
``agentic_routing`` plugin.  When the layer is disabled or its dependencies are
missing it silently steps aside and routing stays purely deterministic.

The plugin fails open: a payload it cannot parse or classify is returned
untouched rather than turning a routing bug into a request failure.

Example
-------
::

    plugin = CodexRoutingPlugin(logger)
    result = plugin.apply({"model": "auto_codex", "input": [...]})
    result["model"]           # "qwen/Qwen3.8-Flash-Next"
    result["routing"]["plugin"]  # "agentic_routing_codex"
"""

import logging

from typing import Any, Optional

from llm_router_plugins.plugin_interface import PluginInterface
from llm_router_plugins.utils.routing.agentic_routing.codex.classifier import classify
from llm_router_plugins.utils.routing.agentic_routing.codex.config import (
    CodexRoutingConfig,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.payload import (
    parse_codex_payload,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.semantic import (
    CodexSemanticLayer,
)
from llm_router_plugins.utils.routing.common import (
    annotate_routing,
    build_embedding_router,
    resolve_persist_dir,
    should_route,
)
from llm_router_plugins.utils.routing.constants import AGENTIC_CODEX_ROUTING_PREFIX

__all__ = ["CodexRoutingPlugin"]

_MISSING_DEPENDENCIES_MESSAGE = (
    "CodexRouting: semantic routing is enabled but the sentence-transformers "
    "/ FAISS dependencies are not installed — install them or set "
    f"semantic.enabled=false / {AGENTIC_CODEX_ROUTING_PREFIX}SEMANTIC_ENABLED=false"
)


class CodexRoutingPlugin(PluginInterface):
    """
    Work-mode routing plugin for Codex CLI requests.

    Attributes
    ----------
    name : str
        Plugin identifier (``"agentic_routing_codex"``).
    """

    name = "agentic_routing_codex"

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        config: Optional[CodexRoutingConfig] = None,
        emb_router: Optional[Any] = None,
        semantic: Optional[CodexSemanticLayer] = None,
    ) -> None:
        """
        Initialize the plugin, loading and validating configuration.

        The semantic layer is optional and fail-open: a missing embedding
        model or an absent ``faiss`` / ``sentence_transformers`` dependency
        disables it and nothing else, so the plugin is always constructible.
        Configuration stays authoritative: with ``semantic.enabled`` ``false``
        no layer is built and an injected *emb_router* is ignored.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger used for diagnostics.  When ``None`` the plugin stays quiet.
        config : CodexRoutingConfig, optional
            Preloaded configuration.  When ``None`` the bundled JSON config is
            loaded (or the
            ``LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_CONFIG`` env var).
        emb_router : Any, optional
            Pre-built embedding router exposing ``route(text)``.  Injected
            routers are used as-is and are never re-initialized.
        semantic : CodexSemanticLayer, optional
            Pre-built semantic layer.  Takes precedence over *emb_router*.

        Raises
        ------
        None
        """
        super().__init__(logger=logger)
        self._config: CodexRoutingConfig = (
            config if config is not None else CodexRoutingConfig.from_file()
        )
        self._config.override_from_env(self._logger)
        self._config.validate_args()

        self._semantic: Optional[CodexSemanticLayer] = semantic
        if self._semantic is None and self._config.semantic_enabled:
            router = emb_router
            if router is None:
                try:
                    router = self._build_router()
                except Exception as exc:
                    self._warn("Codex semantic routing disabled: %s", exc)
                    router = None
            if router is not None:
                self._semantic = CodexSemanticLayer(
                    router=router,
                    threshold=self._config.similarity_threshold,
                    mode_by_name=self._config.mode_by_name,
                    logger=self._logger,
                )

    def apply(
        self,
        payload: Any,
        model_config: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Route a Codex request to the model of its resolved work mode.

        Parameters
        ----------
        payload : Any
            The incoming request payload.  Anything that is not a dict, does
            not carry the trigger model, or cannot be classified is returned
            unchanged.
        model_config : Any, optional
            Accepted for interface compatibility; unused by this plugin.
        **kwargs : Any
            Accepted for interface compatibility; unused by this plugin.

        Returns
        -------
        Any
            The payload.  Routed payloads have ``"model"``, ``"routing"`` and
            ``"agent_mode"`` set; unrouted payloads are the very same object
            that was passed in.

        Raises
        ------
        None
        """
        if not isinstance(payload, dict):
            return payload

        if not should_route(payload, {self._config.trigger_model}):
            return payload

        try:
            request = parse_codex_payload(payload)
            decision = classify(
                payload, request, self._config, semantic=self._semantic
            )
        except Exception as exc:  # routing must never break a request
            self._warn("Codex routing failed, passing the request through: %s", exc)
            return payload


        mode = self._config.mode_by_name.get(decision.mode)
        if mode is None:
            self._warn(
                "Resolved Codex mode '%s' is not configured, passing the "
                "request through.",
                decision.mode,
            )
            return payload
        if not mode.model_name:
            self._warn(
                "Codex mode '%s' has no model configured, passing the request "
                "through.",
                mode.name,
            )
            return payload

        self._info(
            "Codex routing: mode=%s source=%s model=%s similarity=%.3f class=%s",
            mode.name,
            decision.source,
            mode.model_name,
            decision.similarity,
            request.request_class,
        )
        annotated = annotate_routing(
            payload,
            self.name,
            mode.model_name,
            decision.similarity,
            agent_mode=mode.name,
            source=decision.source,
            codex_class=request.request_class,
            collaboration_mode=request.collaboration_mode,
            request_kind=request.request_kind,
            thread_id=request.thread_id,
            turn_id=request.turn_id,
        )
        annotated["agent_mode"] = mode.name
        return annotated

    def _build_router(self) -> Any:
        """
        Build and initialize the semantic (BiEncoder + FAISS) router.

        Returns
        -------
        Any
            An initialized :class:`EmbeddingRouter` over the configured Codex
            work modes.

        Raises
        ------
        ValueError
            If ``faiss`` / ``sentence_transformers`` are not importable, or if
            the resulting index contains no vectors.
        """
        persist_dir = resolve_persist_dir(
            AGENTIC_CODEX_ROUTING_PREFIX,
            self._config.vector_store_path,
            logger=self._logger,
        )

        return build_embedding_router(
            embedding_model=self._config.embedding_model,
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
            top_k=self._config.top_k,
            routing_targets=tuple(
                mode
                for mode in self._config.codex_modes
                if mode.name not in ("aux_title", "compaction")
            ),
            logger=self._logger,
            persist_dir=persist_dir,
            missing_deps_hint=_MISSING_DEPENDENCIES_MESSAGE,
        )

    def _info(self, message: str, *args: Any) -> None:
        """
        Log a info when a logger is available.

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

    def _debug(self, message: str, *args: Any) -> None:
        """
        Log a debug message when a logger is available.

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
            self._logger.debug(message, *args)
