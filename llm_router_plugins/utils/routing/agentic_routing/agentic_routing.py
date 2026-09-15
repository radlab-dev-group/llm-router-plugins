"""
AgenticRoutingPlugin — model selection driven by the agent work mode.

Instead of classifying the *topic* of a request, this plugin detects in which
**work mode** the agent is currently operating (planning, coding, reviewing,
testing, debugging, research, summarizing) and selects the model configured
for that mode.

The plugin activates only when ``payload["model"]`` is a string whose trimmed
value is listed in ``settings.trigger`` (by default ``["agentic"]``).

Mode detection runs as a cascade — the first layer that produces an answer
wins:

1. **Explicit** — a mode declared in the payload itself, looked up in
   ``agent_mode``, ``mode``, ``agent.mode`` and ``metadata.agent_mode``.
2. **Semantic** — a BiEncoder + FAISS lookup reusing
   :class:`EmbeddingRouter`; every agent mode acts as a routing target built
   from its description and examples.  A match is accepted when the cosine
   similarity is greater than or equal to ``settings.semantic.threshold``.
3. **Heuristic** — weighted keyword, phrase and regex scoring over the mode
   definitions.
4. **Fallback** — the mode named in ``settings.fallback_mode``.

The plugin only replaces ``payload["model"]`` and adds routing metadata; it
never touches temperature, token limits, prompts or tools.

Configuration is loaded from
``llm_router_plugins/resources/routing/agentic_routing.json``
and can be overridden by environment variables:

    LLM_ROUTER_ROUTING_AGENTIC_CONFIG
        - full configuration as a raw JSON string or a path to a config file
    LLM_ROUTER_ROUTING_AGENTIC_TRIGGER
        - pipe-separated list of ``payload["model"]`` values that activate the plugin
    LLM_ROUTER_ROUTING_AGENTIC_MODEL
        - override the embedding model name
    LLM_ROUTER_ROUTING_AGENTIC_MODELS
        - per-mode model mapping, e.g. ``plan=model_a|code=model_b``
    LLM_ROUTER_ROUTING_AGENTIC_MODES
        - pipe-separated whitelist of mode names
    LLM_ROUTER_ROUTING_AGENTIC_SEMANTIC_ENABLED
        - ``1/0``, ``true/false``, ``yes/no``, ``on/off``
    LLM_ROUTER_ROUTING_AGENTIC_SIMILARITY_THRESHOLD
        - minimum similarity accepted for a semantic match
    LLM_ROUTER_ROUTING_AGENTIC_TOP_K
        - number of nearest neighbours used when routing
    LLM_ROUTER_ROUTING_AGENTIC_CHUNK_SIZE
        - override chunk size
    LLM_ROUTER_ROUTING_AGENTIC_CHUNK_OVERLAP
        - override chunk overlap
    LLM_ROUTER_ROUTING_AGENTIC_PERSIST_DIR
        - directory for FAISS index persistence
    LLM_ROUTER_ROUTING_AGENTIC_FALLBACK_MODE
        - name of the mode used when nothing else matches
    LLM_ROUTER_ROUTING_AGENTIC_MODE_<NAME>_KEYWORDS
        - pipe-separated keyword list for a single mode

Example JSON configuration::

    {
      "embedding_model": "google/embeddinggemma-300m",
      "settings": {
        "trigger": ["agentic"],
        "fallback_mode": "fallback",
        "vector_store_path": "",
        "semantic": {
          "enabled": true,
          "threshold": 0.55,
          "top_k": 3,
          "chunk_size": 256,
          "chunk_overlap": 64
        }
      },
      "agent_modes": [
        {
          "name": "plan",
          "model_name": "qwen3.6:35b",
          "description": "Agent works in planning mode: ...",
          "examples": ["Plan the migration of this service...", ...],
          "keywords": ["planning", "roadmap", ...],
          "phrases": ["zaplanuj pracę:5", ...],
          "patterns": ["\\\\bplan\\\\b", ...],
          "weights": { "planning": 3, "roadmap": 3, ... }
        }
      ]
    }
"""

import logging
import re

from typing import Any, Dict, Optional, Tuple

from llm_router_plugins.plugin_interface import PluginInterface
from llm_router_plugins.utils.text_extractor import extract_user_text
from llm_router_plugins.utils.routing.common import (
    annotate_routing,
    build_embedding_router,
    resolve_persist_dir,
    should_route,
)
from llm_router_plugins.utils.routing.agentic_routing.config import (
    AgentMode,
    AgenticRoutingConfig,
)
from llm_router_plugins.utils.routing.constants import AGENTIC_ROUTING_PREFIX

_MISSING_DEPENDENCIES_MESSAGE = (
    "AgenticRouting: semantic routing is enabled but the sentence-transformers "
    "/ FAISS dependencies are not installed — install them or set "
    f"semantic.enabled=false / {AGENTIC_ROUTING_PREFIX}SEMANTIC_ENABLED=false"
)


class AgenticRoutingPlugin(PluginInterface):
    """
    Work-mode based routing plugin.

    When ``payload["model"]`` matches a configured trigger (default
    ``"agentic"``) the plugin detects the current agent work mode and replaces
    the model with the one configured for that mode.

    Attributes
    ----------
    name : str
        Plugin identifier (``"agentic_routing"``).
    """

    name = "agentic_routing"

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        router: Optional[Any] = None,
        config: Optional[AgenticRoutingConfig] = None,
    ) -> None:
        """
        Initialize the plugin: load config, validate it and build the router.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger instance.  If ``None``, logging is skipped.
        router : Any, optional
            Pre-built router exposing ``route(text)``.  Injected routers are
            used as-is and are never re-initialised.
        config : AgenticRoutingConfig, optional
            Pre-loaded configuration.  When omitted the default resource file
            (or ``LLM_ROUTER_ROUTING_AGENTIC_CONFIG``) is loaded.

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist at the expected path.
        KeyError
            If the configuration file is missing required fields.
        ValueError
            If the configuration is inconsistent, or if semantic routing is
            enabled but the ML dependencies are missing or produced an empty
            index.
        """
        super().__init__(logger=logger)

        self._config = (
            config if config is not None else AgenticRoutingConfig.from_file()
        )
        self._config.override_from_env(self._logger)
        self._config.validate_args()

        self._router = router
        if self._router is None and self._config.semantic_enabled:
            self._router = self._build_router()

    def _build_router(self) -> Any:
        """
        Build and initialize the semantic (BiEncoder + FAISS) router.

        Returns
        -------
        Any
            An initialized :class:`EmbeddingRouter` over the configured agent
            modes.

        Raises
        ------
        ValueError
            If ``faiss`` / ``sentence_transformers`` are not importable, or if
            the resulting index contains no vectors.
        """
        persist_dir = resolve_persist_dir(
            AGENTIC_ROUTING_PREFIX,
            self._config.vector_store_path,
            logger=self._logger,
        )

        return build_embedding_router(
            embedding_model=self._config.embedding_model,
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
            top_k=self._config.top_k,
            routing_targets=self._config.agent_modes,
            logger=self._logger,
            persist_dir=persist_dir,
            missing_deps_hint=_MISSING_DEPENDENCIES_MESSAGE,
        )

    def apply(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process *payload*, selecting the model for the detected agent mode.

        Text is extracted from the payload, the work mode is resolved via the
        explicit → semantic → heuristic → fallback cascade, and the payload is
        annotated with the resulting model and routing metadata.

        Parameters
        ----------
        payload : dict
            The incoming payload.  Routing activates only when
            ``payload["model"]`` is a string listed in the configured triggers.

        Returns
        -------
        dict
            The modified payload.  When the trigger does not match the payload
            is returned unchanged.  Otherwise ``payload["model"]`` holds the
            selected model name, ``payload["agent_mode"]`` the detected mode,
            and ``payload["routing"]`` contains:

            - ``"plugin"`` (str): ``"agentic_routing"``
            - ``"agent_mode"`` (str): name of the detected mode
            - ``"source"`` (str): ``"explicit"``, ``"semantic"``,
              ``"heuristic"``, ``"fallback"`` or ``"empty_text"``
            - ``"similarity"`` (float): confidence score of the decision

        Raises
        ------
        None
        """
        if not should_route(payload, self._config.trigger):
            return payload

        text = extract_user_text(payload)
        mode, source, similarity = self._resolve_mode(text, payload)
        return self._annotate(payload, mode, source, similarity, text)

    def _resolve_mode(
        self, text: str, payload: Dict[str, Any]
    ) -> Tuple[AgentMode, str, float]:
        """
        Resolve the agent work mode for *text* using the detection cascade.

        Parameters
        ----------
        text : str
            The extracted user text used for detection.  When it is empty the
            semantic and heuristic layers are skipped, unless the payload
            declares an explicit mode.
        payload : dict
            The payload, consulted for an explicitly declared mode.

        Returns
        -------
        Tuple[AgentMode, str, float]
            The resolved mode, the name of the layer that resolved it, and the
            associated confidence score.

        Raises
        ------
        None
        """
        explicit = self._get_explicit_mode_name(payload)
        if explicit:
            mode = self._config.mode_by_name.get(explicit)
            if mode is not None:
                return mode, "explicit", 1.0
            if self._logger:
                self._logger.warning(
                    "AgenticRouting: unknown agent_mode '%s', "
                    "continuing with detection cascade",
                    explicit,
                )

        if not text:
            if self._logger:
                self._logger.warning(
                    "AgenticRouting: no text content found, using fallback mode"
                )
            return (
                self._config.mode_by_name[self._config.fallback_mode],
                "empty_text",
                0.0,
            )

        if self._config.semantic_enabled and self._router is not None:
            result = self._router.route(text)
            similarity = float(result["similarity"])
            target = str(result.get("target_name", ""))
            mode = self._config.mode_by_name.get(target)
            if mode is not None and similarity >= self._config.similarity_threshold:
                return mode, "semantic", similarity
            if self._logger:
                self._logger.info(
                    "AgenticRouting: semantic match '%s' similarity=%.4f is "
                    "below threshold %.4f, falling back to heuristics",
                    target or "unknown",
                    similarity,
                    self._config.similarity_threshold,
                )

        mode, score = self._detect_heuristic(text)
        if mode is not None and score > 0:
            return mode, "heuristic", score / (score + 1.0)

        return self._config.mode_by_name[self._config.fallback_mode], "fallback", 0.0

    def _get_explicit_mode_name(self, payload: Dict[str, Any]) -> str:
        """
        Return the normalized mode name explicitly declared in *payload*.

        The following locations are consulted in priority order:
        ``agent_mode``, ``mode``, ``agent.mode`` and ``metadata.agent_mode``.
        The first non-empty string wins and is normalized by lower-casing,
        trimming, and converting hyphens and spaces to underscores.

        Parameters
        ----------
        payload : dict
            The payload to inspect.

        Returns
        -------
        str
            The normalized mode name, or an empty string when none is declared.

        Raises
        ------
        None
        """
        agent = payload.get("agent")
        agent_mode = agent.get("mode") if isinstance(agent, dict) else None

        metadata = payload.get("metadata")
        metadata_mode = (
            metadata.get("agent_mode") if isinstance(metadata, dict) else None
        )

        for value in (
            payload.get("agent_mode"),
            payload.get("mode"),
            agent_mode,
            metadata_mode,
        ):
            if isinstance(value, str) and value.strip():
                return value.strip().lower().replace("-", "_").replace(" ", "_")
        return ""

    def _detect_heuristic(self, text: str) -> Tuple[Optional[AgentMode], float]:
        """
        Detect the mode by scoring keywords, phrases and regex patterns.

        Each mode accumulates a score from three sources:

        1. **Keywords** — case-insensitive substring match, weight taken from
           ``weights`` (default 1).
        2. **Phrases** — multi-word expression with optional ``":weight"``
           suffix (default weight 2.0).
        3. **Patterns** — regex match; each match adds **3.0**.

        Parameters
        ----------
        text : str
            The text to classify.

        Returns
        -------
        Tuple[Optional[AgentMode], float]
            The highest-scoring mode and its score, or ``(None, 0.0)`` when no
            mode scores above zero.  Ties are won by the mode defined first.

        Raises
        ------
        None
        """
        text_lower = text.lower()
        best_mode: Optional[AgentMode] = None
        best_score = 0.0

        for mode in self._config.agent_modes:
            score = 0.0

            for kw in mode.keywords:
                w = mode.weights.get(kw, 1)
                if kw in text_lower:
                    score += w

            for phrase in mode.phrases:
                if isinstance(phrase, str) and ":" in phrase:
                    parts = phrase.rsplit(":", 1)
                    p_text, w = parts[0].strip().lower(), float(parts[1].strip())
                else:
                    p_text, w = phrase.lower(), 2.0
                if p_text in text_lower:
                    score += w

            for pat in mode.patterns:
                try:
                    if re.search(pat, text_lower):
                        score += 3.0
                except re.error:
                    pass

            if score > best_score:
                best_score = score
                best_mode = mode

        return best_mode, best_score

    def _annotate(
        self,
        payload: Dict[str, Any],
        mode: AgentMode,
        source: str,
        similarity: float,
        text: str,
    ) -> Dict[str, Any]:
        """
        Write the routing decision into *payload*.

        Parameters
        ----------
        payload : dict
            The payload to annotate.
        mode : AgentMode
            The resolved agent work mode.
        source : str
            Name of the cascade layer that resolved the mode.
        similarity : float
            Confidence score of the decision.
        text : str
            The extracted text, used for logging only.

        Returns
        -------
        dict
            The annotated payload with ``model``, ``agent_mode`` and
            ``routing`` keys set.

        Raises
        ------
        None
        """
        payload["agent_mode"] = mode.name
        annotate_routing(
            payload,
            self.name,
            mode.model_name,
            similarity,
            agent_mode=mode.name,
            source=source,
        )

        if self._logger:
            self._logger.info(
                "AgenticRouting: text='%s' mode='%s' source=%s "
                "similarity=%.4f -> model=%s",
                text[:80],
                mode.name,
                source,
                float(similarity),
                mode.model_name,
            )

        return payload
