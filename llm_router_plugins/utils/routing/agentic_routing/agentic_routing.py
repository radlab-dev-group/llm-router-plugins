"""
AgenticRoutingPlugin — model selection driven by the agent work mode.

Instead of classifying the *topic* of a request, this plugin detects in which
**work mode** the agent is currently operating (planning, coding, reviewing,
testing, debugging, research, summarizing) and selects the model configured
for that mode.

The plugin activates only when ``payload["model"]`` is a string whose trimmed
value is listed in ``settings.trigger`` (by default ``["agentic"]``).

Mode detection runs as a cascade — the first layer that produces an answer
wins.  Every **deterministic** layer is tried before any embedding lookup, so
an identical request is answered identically, offline and for free:

1. **Explicit** (``"explicit"``) — a mode declared in the payload itself,
   looked up in ``agent_mode``, ``mode``, ``agent.mode`` and
   ``metadata.agent_mode``.
2. **Rules** (``"rules"``) — declarative ``rules`` matched against the
   normalized request signals (agent, task, required capabilities, context
   size).  Highest ``priority`` wins.
3. **Session affinity** (``"affinity"``) — the mode already chosen earlier in
   the same conversation, taken from a TTL/LRU cache keyed by ``session_id``.
4. **Heuristic** (``"heuristic"``) — weighted keyword, phrase and regex
   scoring over the mode definitions.
5. **Semantic** (``"semantic"``) — a BiEncoder + FAISS lookup reusing
   :class:`EmbeddingRouter`; every agent mode acts as a routing target built
   from its description and examples.  A match is accepted when the cosine
   similarity is greater than or equal to ``settings.semantic.threshold``.
6. **Fallback** (``"fallback"``, or ``"empty_text"`` when no text could be
   extracted) — the mode named in ``settings.fallback_mode``.

After a layer has selected a mode the capability gate checks it against the
capabilities the request implies (tool calling, parallel tool calls,
reasoning, vision, structured output, context window) and escalates to the
most capable alternative when the selected model cannot serve the request.

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
        - pipe-separated whitelist of mode names (rules pointing at removed
          modes are dropped)
    LLM_ROUTER_ROUTING_AGENTIC_RULES_ENABLED
        - master switch for the declarative rules layer
    LLM_ROUTER_ROUTING_AGENTIC_CAPABILITIES_ENABLED
        - master switch for the capability gate
    LLM_ROUTER_ROUTING_AGENTIC_ESCALATION_ENABLED
        - master switch for escalating to a capable mode
    LLM_ROUTER_ROUTING_AGENTIC_SESSION_AFFINITY_ENABLED
        - master switch for session affinity
    LLM_ROUTER_ROUTING_AGENTIC_SESSION_TTL_SECONDS
        - lifetime of a cached session decision
    LLM_ROUTER_ROUTING_AGENTIC_SESSION_MAX_ENTRIES
        - maximum number of cached sessions
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
        "rules_enabled": true,
        "capabilities_enabled": true,
        "escalation_enabled": true,
        "session_affinity": { "enabled": true, "ttl_seconds": 900 },
        "semantic": {
          "enabled": true,
          "threshold": 0.55,
          "top_k": 3,
          "chunk_size": 256,
          "chunk_overlap": 64
        }
      },
      "rules": [
        { "id": "task-coding", "priority": 100,
          "when": { "task": ["coding", "implement"] }, "then": { "mode": "code" } },
        { "id": "needs-tools", "priority": 80,
          "when": { "requires_tools": true }, "mode": "code" }
      ],
      "agent_modes": [
        {
          "name": "plan",
          "model_name": "qwen3.6:35b",
          "description": "Agent works in planning mode: ...",
          "examples": ["Plan the migration of this service...", ...],
          "keywords": ["planning", "roadmap", ...],
          "phrases": ["zaplanuj pracę:5", ...],
          "patterns": ["\\\\bplan\\\\b", ...],
          "weights": { "planning": 3, "roadmap": 3, ... },
          "capabilities": { "tool_calling": true, "context_window": 131072 }
        }
      ]
    }
"""

import logging

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Optional, Tuple

from llm_router_api.core.model_config import ApiModelConfig

from llm_router_plugins.plugin_interface import PluginInterface
from llm_router_plugins.utils.text_extractor import extract_user_text
from llm_router_plugins.utils.routing.common import (
    annotate_routing,
    build_embedding_router,
    resolve_persist_dir,
    should_route,
)
from llm_router_plugins.utils.routing.agentic_routing.capabilities import (
    escalate,
    missing_capabilities,
    requirements_from_signals,
    satisfies,
)
from llm_router_plugins.utils.routing.agentic_routing.config import (
    AgentMode,
    AgenticRoutingConfig,
)
from llm_router_plugins.utils.routing.agentic_routing.heuristics import (
    detect_heuristic,
    score_to_similarity,
)
from llm_router_plugins.utils.routing.agentic_routing.rules import (
    describe_rule,
    match_rule,
)
from llm_router_plugins.utils.routing.agentic_routing.semantic import SemanticLayer
from llm_router_plugins.utils.routing.agentic_routing.session_affinity import (
    SessionAffinityCache,
)
from llm_router_plugins.utils.routing.agentic_routing.signals import RequestSignals
from llm_router_plugins.utils.routing.constants import AGENTIC_ROUTING_PREFIX

_MISSING_DEPENDENCIES_MESSAGE = (
    "AgenticRouting: semantic routing is enabled but the sentence-transformers "
    "/ FAISS dependencies are not installed — install them or set "
    f"semantic.enabled=false / {AGENTIC_ROUTING_PREFIX}SEMANTIC_ENABLED=false"
)

#: Sources whose decision is not worth remembering for the whole session.
_NON_CACHABLE_SOURCES = ("fallback", "empty_text")


@dataclass(frozen=True)
class _RoutingDecision:
    """
    Result of one cascade pass.

    Parameters
    ----------
    mode : AgentMode
        The selected work mode.
    source : str
        Name of the layer that produced the decision.
    similarity : float
        Confidence of the decision (``1.0`` for the deterministic layers,
        ``0.0`` for the fallback).
    rule_id : str
        Identifier of the matching rule; empty unless ``source == "rules"``.
    escalated_from : str
        Name of the mode replaced by the capability gate; empty when the
        decision needed no escalation.
    session : dict, optional
        Session affinity information added for requests carrying a session id.

    Raises
    ------
    None
    """

    mode: AgentMode
    source: str
    similarity: float
    rule_id: str = ""
    escalated_from: str = ""
    session: Optional[Dict[str, Any]] = None


class AgenticRoutingPlugin(PluginInterface):
    """
    Work-mode-based routing plugin.

    When ``payload["model"]`` matches a configured trigger (default
    ``"agentic"``) the plugin detects the current agent work mode and replaces
    the model with the one configured for that mode.

    The cascade is ordered deterministic-first (explicit mode, declarative
    rules, session affinity, keyword scoring) and semantic-last, so the
    expensive embedding/vector-store lookup runs only when nothing cheaper
    could decide.

    Attributes
    ----------
    name : str
        Plugin identifier (``"agentic_routing"``).
    """

    name = "agentic_routing"

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        emb_router: Optional[Any] = None,
        config: Optional[AgenticRoutingConfig] = None,
    ) -> None:
        """
        Initialize the plugin: load config, validate it and build the router.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger instance.  If ``None``, logging is skipped.
        emb_router : Any, optional
            Pre-built router exposing ``route(text)``.  Injected routers are
            used as-is and are never re-initialized.
        config : AgenticRoutingConfig, optional
            Pre-loaded configuration.  When omitted, the default resource file
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

        self._config: AgenticRoutingConfig = (
            config if config is not None else AgenticRoutingConfig.from_file()
        )
        self._config.override_from_env(self._logger)
        self._config.validate_args()

        self._emb_router = emb_router
        if self._emb_router is None and self._config.semantic_enabled:
            self._emb_router = self._build_router()

        # self._semantic = SemanticLayer(
        #     router=self._emb_router,
        #     threshold=self._config.similarity_threshold,
        #     mode_by_name=self._config.mode_by_name,
        #     logger=self._logger,
        # )
        # self._session_cache = SessionAffinityCache(
        #     ttl_seconds=self._config.session_affinity.ttl_seconds,
        #     max_entries=self._config.session_affinity.max_entries,
        #     logger=self._logger,
        # )
        pass

    #
    # @property
    # def config(self) -> AgenticRoutingConfig:
    #     """
    #     Return the active plugin configuration.
    #
    #     Parameters
    #     ----------
    #     None
    #
    #     Returns
    #     -------
    #     AgenticRoutingConfig
    #         The configuration used by the cascade.
    #
    #     Raises
    #     ------
    #     None
    #     """
    #     return self._config
    #
    # def reset_sessions(self) -> None:
    #     """
    #     Drop every cached session decision.
    #
    #     Useful after a configuration change, so that requests of ongoing
    #     conversations are re-routed through the cascade.
    #
    #     Parameters
    #     ----------
    #     None
    #
    #     Returns
    #     -------
    #     None
    #
    #     Raises
    #     ------
    #     None
    #     """
    #     self._session_cache.reset()
    #
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

    def apply(
        self,
        payload: Dict[str, Any],
        model_config: Optional[ApiModelConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Process *payload*, selecting the model for the detected agent mode.

        Text is extracted from the payload (surrounding whitespace is removed,
        so a whitespace-only request counts as having no text), the work mode
        is resolved via the deterministic-first cascade and the payload is
        annotated with the resulting model and routing metadata.

        Parameters
        ----------
        payload : dict
            The incoming payload.  Routing activates only when
            ``payload["model"]`` is a string listed in the configured triggers.
        model_config :
            LLM Router models configuration.

        Returns
        -------
        dict
            The modified payload.  When the trigger does not match the payload
            is returned unchanged.  Otherwise ``payload["model"]`` holds the
            selected model name, ``payload["agent_mode"]`` the detected mode,
            and ``payload["routing"]`` contains:

            - ``"plugin"`` (str): ``"agentic_routing"``
            - ``"agent_mode"`` (str): name of the detected mode
            - ``"source"`` (str): ``"explicit"``, ``"rules"``, ``"affinity"``,
              ``"heuristic"``, ``"semantic"``, ``"fallback"`` or
              ``"empty_text"``
            - ``"similarity"`` (float): confidence score of the decision
            - ``"rule_id"`` (str): present only for ``source == "rules"``
            - ``"escalated"`` (bool) and ``"escalated_from"`` (str): present
              only when the capability gate replaced the selected mode
            - ``"session"`` (dict): present only when the request carries a
              session id and session affinity is enabled

        Raises
        ------
        None
        """
        if not should_route(payload, self._config.trigger):
            return payload

        print("AGENTIC_ROUTING")
        print("AGENTIC_ROUTING")
        print("AGENTIC_ROUTING")
        print("AGENTIC_ROUTING")
        print("AGENTIC_ROUTING")
        print("AGENTIC_ROUTING")
        print("AGENTIC_ROUTING")
        print("AGENTIC_ROUTING")

        payload["model"] = "qwen/Qwen3.8-27B"

        # text = extract_user_text(payload).strip()
        # decision = self._resolve_mode(text, payload)
        # return self._annotate(payload, decision, text)
        return payload

    #
    # def _resolve_mode(self, text: str, payload: Dict[str, Any]) -> _RoutingDecision:
    #     """
    #     Resolve the agent work mode using the deterministic-first cascade.
    #
    #     Parameters
    #     ----------
    #     text : str
    #         The extracted user text used for detection.  When it is empty the
    #         inference layers are skipped, unless an earlier layer answers.
    #     payload : dict
    #         The payload, source of the request signals.
    #
    #     Returns
    #     -------
    #     _RoutingDecision
    #         The resolved mode together with the layer, confidence and optional
    #         rule/session information.
    #
    #     Raises
    #     ------
    #     None
    #     """
    #     signals = RequestSignals.from_payload(payload, text)
    #     requirements = (
    #         requirements_from_signals(signals)
    #         if self._config.capabilities_enabled
    #         else {}
    #     )
    #
    #     resolvers: Tuple[Callable[[], Optional[_RoutingDecision]], ...] = (
    #         lambda: self._resolve_explicit(payload, requirements),
    #         lambda: self._resolve_rules(signals, text, requirements),
    #         lambda: self._resolve_affinity(signals, requirements),
    #         lambda: self._resolve_inference(text, requirements),
    #     )
    #     for resolve in resolvers:
    #         decision = resolve()
    #         if decision is not None:
    #             return self._attach_session(decision, signals)
    #
    #     if not text:
    #         if self._logger:
    #             self._logger.warning(
    #                 "AgenticRouting: no text content found, using fallback mode"
    #             )
    #         decision = self._fallback_decision("empty_text")
    #     else:
    #         decision = self._fallback_decision("fallback")
    #
    #     return self._attach_session(decision, signals)
    #
    # def _fallback_decision(self, source: str) -> _RoutingDecision:
    #     """
    #     Build the decision pointing at the configured fallback mode.
    #
    #     Parameters
    #     ----------
    #     source : str
    #         ``"fallback"`` for unmatched text, ``"empty_text"`` when the
    #         request carried no text at all.
    #
    #     Returns
    #     -------
    #     _RoutingDecision
    #         A zero-confidence decision for the fallback mode.
    #
    #     Raises
    #     ------
    #     None
    #     """
    #     mode = self._config.mode_by_name[self._config.fallback_mode]
    #     return _RoutingDecision(mode=mode, source=source, similarity=0.0)
    #
    # def _resolve_explicit(
    #     self, payload: Dict[str, Any], requirements: Dict[str, Any]
    # ) -> Optional[_RoutingDecision]:
    #     """
    #     Resolve a mode explicitly declared in *payload*.
    #
    #     An explicit declaration is the intent of the caller: when the named
    #     mode lacks a capability the gate only warns, it never overrides the
    #     choice.
    #
    #     Parameters
    #     ----------
    #     payload : dict
    #         The payload to inspect.
    #     requirements : dict
    #         Capabilities implied by the request, used for warnings only.
    #
    #     Returns
    #     -------
    #     Optional[_RoutingDecision]
    #         The explicit decision, or ``None`` when no known mode is declared.
    #
    #     Raises
    #     ------
    #     None
    #     """
    #     name = self._get_explicit_mode_name(payload)
    #     if not name:
    #         return None
    #
    #     mode = self._config.mode_by_name.get(name)
    #     if mode is None:
    #         if self._logger:
    #             self._logger.warning(
    #                 "AgenticRouting: unknown agent_mode '%s', "
    #                 "continuing with detection cascade",
    #                 name,
    #             )
    #         return None
    #
    #     if requirements and self._logger:
    #         missing = missing_capabilities(mode.capabilities, requirements)
    #         if missing:
    #             self._logger.warning(
    #                 "AgenticRouting: explicitly requested mode '%s' does not "
    #                 "satisfy %s — keeping the explicit choice",
    #                 mode.name,
    #                 ", ".join(missing),
    #             )
    #
    #     return _RoutingDecision(mode=mode, source="explicit", similarity=1.0)
    #
    # def _resolve_rules(
    #     self,
    #     signals: RequestSignals,
    #     text: str,
    #     requirements: Dict[str, Any],
    # ) -> Optional[_RoutingDecision]:
    #     """
    #     Resolve the mode through the declarative rules layer.
    #
    #     Parameters
    #     ----------
    #     signals : RequestSignals
    #         Normalized request signals.
    #     text : str
    #         The extracted user text, used by text-based rule conditions.
    #     requirements : dict
    #         Capabilities implied by the request.
    #
    #     Returns
    #     -------
    #     Optional[_RoutingDecision]
    #         The decision of the first matching rule, or ``None`` when rules
    #         are disabled or none applies.
    #
    #     Raises
    #     ------
    #     None
    #     """
    #     if not self._config.rules_enabled or not self._config.rules:
    #         return None
    #
    #     rule = match_rule(self._config.rules, signals, text)
    #     if rule is None:
    #         return None
    #
    #     mode = self._config.mode_by_name.get(rule.mode)
    #     if mode is None:
    #         if self._logger:
    #             self._logger.warning(
    #                 "AgenticRouting: rule '%s' targets unknown mode '%s', "
    #                 "ignoring it",
    #                 rule.id,
    #                 rule.mode,
    #             )
    #         return None
    #
    #     if self._logger:
    #         self._logger.info(
    #             "AgenticRouting: rule %s matched the request signals",
    #             describe_rule(rule),
    #         )
    #     return self._enforce_capabilities(
    #         mode, "rules", 1.0, requirements, rule_id=rule.id
    #     )
    #
    # def _resolve_affinity(
    #     self, signals: RequestSignals, requirements: Dict[str, Any]
    # ) -> Optional[_RoutingDecision]:
    #     """
    #     Reuse the mode already chosen for the current session.
    #
    #     A cached entry whose mode disappeared from the configuration, or that
    #     can no longer serve the request, is dropped and the cascade continues.
    #
    #     Parameters
    #     ----------
    #     signals : RequestSignals
    #         Normalized request signals; ``session_id`` selects the entry.
    #     requirements : dict
    #         Capabilities implied by the request.
    #
    #     Returns
    #     -------
    #     Optional[_RoutingDecision]
    #         The cached decision, or ``None`` on a miss.
    #
    #     Raises
    #     ------
    #     None
    #     """
    #     if not self._config.session_affinity.enabled or not signals.session_id:
    #         return None
    #
    #     cached = self._session_cache.get(signals.session_id)
    #     if cached is None:
    #         return None
    #
    #     mode = self._config.mode_by_name.get(cached.mode_name)
    #     if mode is None:
    #         if self._logger:
    #             self._logger.info(
    #                 "AgenticRouting: cached session mode '%s' is no longer "
    #                 "configured, re-routing session '%s'",
    #                 cached.mode_name,
    #                 signals.session_id,
    #             )
    #         self._session_cache.invalidate(signals.session_id)
    #         return None
    #
    #     if not satisfies(mode.capabilities, requirements):
    #         if self._logger:
    #             self._logger.warning(
    #                 "AgenticRouting: cached session mode '%s' cannot serve the "
    #                 "request anymore, re-routing session '%s'",
    #                 mode.name,
    #                 signals.session_id,
    #             )
    #         self._session_cache.invalidate(signals.session_id)
    #         return None
    #
    #     return _RoutingDecision(mode=mode, source="affinity", similarity=1.0)
    #
    # def _resolve_inference(
    #     self, text: str, requirements: Dict[str, Any]
    # ) -> Optional[_RoutingDecision]:
    #     """
    #     Resolve the mode from the text: heuristic scoring, then semantics.
    #
    #     Parameters
    #     ----------
    #     text : str
    #         The extracted user text.
    #     requirements : dict
    #         Capabilities implied by the request.
    #
    #     Returns
    #     -------
    #     Optional[_RoutingDecision]
    #         The inferred decision, or ``None`` when neither layer matched.
    #
    #     Raises
    #     ------
    #     None
    #     """
    #     if not text:
    #         return None
    #
    #     mode, score = self._detect_heuristic(text)
    #     if mode is not None and score > 0:
    #         return self._enforce_capabilities(
    #             mode,
    #             "heuristic",
    #             score_to_similarity(score),
    #             requirements,
    #         )
    #
    #     return self._resolve_semantic(text, requirements)
    #
    # def _resolve_semantic(
    #     self, text: str, requirements: Dict[str, Any]
    # ) -> Optional[_RoutingDecision]:
    #     """
    #     Resolve the mode through the embedding / vector-store layer.
    #
    #     Parameters
    #     ----------
    #     text : str
    #         The extracted user text.
    #     requirements : dict
    #         Capabilities implied by the request.
    #
    #     Returns
    #     -------
    #     Optional[_RoutingDecision]
    #         The semantic decision, or ``None`` when the layer is disabled,
    #         unavailable, or below the configured threshold.
    #
    #     Raises
    #     ------
    #     None
    #     """
    #     if not self._config.semantic_enabled or not self._semantic.available:
    #         return None
    #
    #     mode, similarity = self._semantic.resolve(text)
    #     if mode is None:
    #         return None
    #
    #     return self._enforce_capabilities(mode, "semantic", similarity, requirements)
    #
    # def _enforce_capabilities(
    #     self,
    #     mode: AgentMode,
    #     source: str,
    #     similarity: float,
    #     requirements: Dict[str, Any],
    #     rule_id: str = "",
    # ) -> _RoutingDecision:
    #     """
    #     Guard *mode* against the capabilities the request demands.
    #
    #     When the selected model cannot serve the request the decision is
    #     escalated to the most capable alternative — the candidate with the
    #     largest declared ``context_window``.  Escalation keeps the layer
    #     and the confidence of the original decision and only records which
    #     mode was replaced.
    #
    #     Parameters
    #     ----------
    #     mode : AgentMode
    #         The mode selected by a cascade layer.
    #     source : str
    #         Name of that layer.
    #     similarity : float
    #         Confidence of the original decision.
    #     requirements : dict
    #         Capabilities implied by the request; empty means "anything fits".
    #     rule_id : str, optional
    #         Identifier of the matching rule, passed through to the decision.
    #
    #     Returns
    #     -------
    #     _RoutingDecision
    #         The original decision, or the escalated one.
    #
    #     Raises
    #     ------
    #     None
    #     """
    #     base = _RoutingDecision(
    #         mode=mode,
    #         source=source,
    #         similarity=similarity,
    #         rule_id=rule_id,
    #     )
    #     if not requirements or not self._config.escalation_enabled:
    #         return base
    #
    #     escalation = escalate(
    #         mode,
    #         self._config.agent_modes,
    #         requirements,
    #         fallback_mode=self._config.fallback_mode,
    #         logger=self._logger,
    #     )
    #     if not escalation.changed:
    #         return base
    #
    #     return replace(
    #         base,
    #         mode=escalation.mode,
    #         escalated_from=mode.name,
    #     )
    #
    # def _attach_session(
    #     self, decision: _RoutingDecision, signals: RequestSignals
    # ) -> _RoutingDecision:
    #     """
    #     Record *decision* in the session cache and annotate it.
    #
    #     A hit refreshes the entry, a miss stores the decision so that the rest
    #     of the conversation keeps the same model.  Fallback decisions are not
    #     remembered: they carry no information about the session and must not
    #     pin a bad mode to it.
    #
    #     Parameters
    #     ----------
    #     decision : _RoutingDecision
    #         The decision to remember.
    #     signals : RequestSignals
    #         Normalized request signals; ``session_id`` selects the entry.
    #
    #     Returns
    #     -------
    #     _RoutingDecision
    #         The same decision, with the ``session`` field filled in when
    #         session affinity is active.
    #
    #     Raises
    #     ------
    #     None
    #     """
    #     if not self._config.session_affinity.enabled or not signals.session_id:
    #         return decision
    #
    #     reused = decision.source == "affinity"
    #     if decision.source not in _NON_CACHABLE_SOURCES:
    #         self._session_cache.set(
    #             signals.session_id, decision.mode.name, decision.mode.model_name
    #         )
    #
    #     session = {
    #         "session_id": signals.session_id,
    #         "mode": decision.mode.name,
    #         "model": decision.mode.model_name,
    #         "reused": reused,
    #     }
    #     return replace(decision, session=session)
    #
    # def _get_explicit_mode_name(self, payload: Dict[str, Any]) -> str:
    #     """
    #     Return the normalized mode name explicitly declared in *payload*.
    #
    #     The following locations are consulted in priority order:
    #     ``agent_mode``, ``mode``, ``agent.mode`` and ``metadata.agent_mode``.
    #     The first non-empty string wins and is normalized by lower-casing,
    #     trimming, and converting hyphens and spaces to underscores.
    #
    #     Parameters
    #     ----------
    #     payload : dict
    #         The payload to inspect.
    #
    #     Returns
    #     -------
    #     str
    #         The normalized mode name, or an empty string when none is declared.
    #
    #     Raises
    #     ------
    #     None
    #     """
    #     agent = payload.get("agent")
    #     agent_mode = agent.get("mode") if isinstance(agent, dict) else None
    #
    #     metadata = payload.get("metadata")
    #     metadata_mode = (
    #         metadata.get("agent_mode") if isinstance(metadata, dict) else None
    #     )
    #
    #     for value in (
    #         payload.get("agent_mode"),
    #         payload.get("mode"),
    #         agent_mode,
    #         metadata_mode,
    #     ):
    #         if isinstance(value, str) and value.strip():
    #             return value.strip().lower().replace("-", "_").replace(" ", "_")
    #     return ""
    #
    # def _detect_heuristic(self, text: str) -> Tuple[Optional[AgentMode], float]:
    #     """
    #     Detect the mode by scoring keywords, phrases and regex patterns.
    #
    #     Each mode accumulates a score from three sources:
    #
    #     1. **Keywords** — case-insensitive substring match, weight taken from
    #        ``weights`` (default 1).
    #     2. **Phrases** — multi-word expression with optional ``":weight"``
    #        suffix (default weight 2.0).
    #     3. **Patterns** — regex match; each match adds **3.0**.
    #
    #     Parameters
    #     ----------
    #     text : str
    #         The text to classify.
    #
    #     Returns
    #     -------
    #     Tuple[Optional[AgentMode], float]
    #         The highest-scoring mode and its score, or ``(None, 0.0)`` when no
    #         mode scores above zero.  Ties are won by the mode defined first.
    #
    #     Raises
    #     ------
    #     None
    #     """
    #     return detect_heuristic(text, self._config.agent_modes)
    #
    # def _annotate(
    #     self, payload: Dict[str, Any], decision: _RoutingDecision, text: str
    # ) -> Dict[str, Any]:
    #     """
    #     Write the routing decision into *payload*.
    #
    #     Parameters
    #     ----------
    #     payload : dict
    #         The payload to annotate.
    #     decision : _RoutingDecision
    #         The resolved decision.
    #     text : str
    #         The extracted text, used for logging only.
    #
    #     Returns
    #     -------
    #     dict
    #         The annotated payload with ``model``, ``agent_mode`` and
    #         ``routing`` keys set.
    #
    #     Raises
    #     ------
    #     None
    #     """
    #     mode = decision.mode
    #     extras: Dict[str, Any] = {
    #         "agent_mode": mode.name,
    #         "source": decision.source,
    #     }
    #     if decision.rule_id:
    #         extras["rule_id"] = decision.rule_id
    #     if decision.escalated_from:
    #         extras["escalated"] = True
    #         extras["escalated_from"] = decision.escalated_from
    #     if decision.session is not None:
    #         extras["session"] = decision.session
    #
    #     payload["agent_mode"] = mode.name
    #     annotate_routing(
    #         payload,
    #         self.name,
    #         mode.model_name,
    #         decision.similarity,
    #         **extras,
    #     )
    #
    #     if self._logger:
    #         self._logger.info(
    #             "AgenticRouting: text='%s' mode='%s' source=%s "
    #             "similarity=%.4f -> model=%s",
    #             text[:80],
    #             mode.name,
    #             decision.source,
    #             float(decision.similarity),
    #             mode.model_name,
    #         )
    #
    #     return payload
