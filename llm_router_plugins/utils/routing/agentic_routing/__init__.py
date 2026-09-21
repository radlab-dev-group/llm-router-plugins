"""
Backward-compatibility re-exports for the agentic routing packages.

The implementation is split into two sub-packages:

- ``llm_router_plugins.utils.routing.agentic_routing.general`` — the
  :class:`AgenticRoutingPlugin` (plugin name ``agentic_routing``), with its
  signals, rules, session-affinity, heuristic, semantic and capability layers.
- ``llm_router_plugins.utils.routing.agentic_routing.codex`` — the
  :class:`~llm_router_plugins.utils.routing.agentic_routing.codex.plugin.CodexRoutingPlugin`
  (plugin name ``agentic_routing_codex``) for Codex CLI requests.

Import from ``...agentic_routing.general`` (or ``...agentic_routing.codex``)
directly; the names re-exported here are kept for compatibility.
"""

from llm_router_plugins.utils.routing.agentic_routing.general import (  # noqa: F401
    AgenticRoutingConfig,
    AgenticRoutingPlugin,
    AgentMode,
    CachedDecision,
    Escalation,
    REQUIREMENT_KEYS,
    RequestSignals,
    RoutingRule,
    SemanticLayer,
    SessionAffinityCache,
    SessionAffinitySettings,
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

__all__ = [
    "AgenticRoutingConfig",
    "AgenticRoutingPlugin",
    "AgentMode",
    "CachedDecision",
    "Escalation",
    "REQUIREMENT_KEYS",
    "RequestSignals",
    "RoutingRule",
    "SemanticLayer",
    "SessionAffinityCache",
    "SessionAffinitySettings",
    "describe_rule",
    "detect_heuristic",
    "escalate",
    "filter_modes",
    "match_rule",
    "missing_capabilities",
    "parse_rules",
    "requirements_from_signals",
    "satisfies",
    "scored_modes",
]
