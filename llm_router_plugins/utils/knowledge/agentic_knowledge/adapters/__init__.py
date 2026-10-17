"""
Payload adapters: one per coding agent.

The knowledge itself does not care which CLI produced a request; the payload
shape does.  An adapter answers four questions about a payload — is this yours,
what did the agent do, where does knowledge go, how is the model changed — and
everything else stays in the plugin.
"""

from llm_router_plugins.utils.knowledge.agentic_knowledge.adapters.base import (
    AgentAdapter,
    AgentTurn,
    KnowledgeEvent,
)

__all__ = ["AgentAdapter", "AgentTurn", "KnowledgeEvent"]
