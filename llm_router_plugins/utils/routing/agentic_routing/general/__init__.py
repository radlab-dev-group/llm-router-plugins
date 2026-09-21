"""
Agentic routing — model selection driven by the agent work mode.

The package is split into one module per cascade layer so that every step of
the decision can be read, tested and reused on its own:

- :mod:`~agentic_routing.general.signals` — request → normalized :class:`RequestSignals`
- :mod:`~agentic_routing.general.rules` — declarative, deterministic rules
- :mod:`~agentic_routing.general.session_affinity` — TTL/LRU session cache
- :mod:`~agentic_routing.general.heuristics` — keyword, phrase and regex scoring
- :mod:`~agentic_routing.general.semantic` — BiEncoder/FAISS lookup (last resort)
- :mod:`~agentic_routing.general.capabilities` — capability gate and escalation
- :mod:`~agentic_routing.general.config` — configuration loading and validation
- :mod:`~agentic_routing.general.agentic_routing` — the plugin orchestrating the above
"""

from llm_router_plugins.utils.routing.agentic_routing.general.capabilities import (
    REQUIREMENT_KEYS,
    Escalation,
    escalate,
    filter_modes,
    missing_capabilities,
    requirements_from_signals,
    satisfies,
)
from llm_router_plugins.utils.routing.agentic_routing.general.config import (
    AgentMode,
    AgenticRoutingConfig,
)
from llm_router_plugins.utils.routing.agentic_routing.general.heuristics import (
    detect_heuristic,
    scored_modes,
)
from llm_router_plugins.utils.routing.agentic_routing.general.agentic_routing import (
    AgenticRoutingPlugin,
)
from llm_router_plugins.utils.routing.agentic_routing.general.rules import (
    RoutingRule,
    describe_rule,
    match_rule,
    parse_rules,
)
from llm_router_plugins.utils.routing.agentic_routing.general.semantic import SemanticLayer
from llm_router_plugins.utils.routing.agentic_routing.general.session_affinity import (
    CachedDecision,
    SessionAffinityCache,
    SessionAffinitySettings,
)
from llm_router_plugins.utils.routing.agentic_routing.general.signals import RequestSignals

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
