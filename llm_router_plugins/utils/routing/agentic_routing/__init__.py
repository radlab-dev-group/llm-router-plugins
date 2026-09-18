"""
Agentic routing — model selection driven by the agent work mode.

The package is split into one module per cascade layer so that every step of
the decision can be read, tested and reused on its own:

- :mod:`~agentic_routing.signals` — request → normalized :class:`RequestSignals`
- :mod:`~agentic_routing.rules` — declarative, deterministic rules
- :mod:`~agentic_routing.session_affinity` — TTL/LRU session cache
- :mod:`~agentic_routing.heuristics` — keyword, phrase and regex scoring
- :mod:`~agentic_routing.semantic` — BiEncoder/FAISS lookup (last resort)
- :mod:`~agentic_routing.capabilities` — capability gate and escalation
- :mod:`~agentic_routing.config` — configuration loading and validation
- :mod:`~agentic_routing.agentic_routing` — the plugin orchestrating the above
"""

from llm_router_plugins.utils.routing.agentic_routing.capabilities import (
    REQUIREMENT_KEYS,
    Escalation,
    escalate,
    filter_modes,
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
    scored_modes,
)
from llm_router_plugins.utils.routing.agentic_routing.agentic_routing import (
    AgenticRoutingPlugin,
)
from llm_router_plugins.utils.routing.agentic_routing.rules import (
    RoutingRule,
    describe_rule,
    match_rule,
    parse_rules,
)
from llm_router_plugins.utils.routing.agentic_routing.semantic import SemanticLayer
from llm_router_plugins.utils.routing.agentic_routing.session_affinity import (
    CachedDecision,
    SessionAffinityCache,
    SessionAffinitySettings,
)
from llm_router_plugins.utils.routing.agentic_routing.signals import RequestSignals

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
