"""
Knowledge layer for coding agents.

The package gives a coding agent, in the proxy and without the agent's
cooperation, the knowledge its own previous sessions produced: verified fixes
for the failure it is facing, changes already known not to work in this
repository, and the exact text its context held before it was summarized.  It
also learns from the traffic, mining those very episodes out of the requests it
sees and sending them to an indexer.

:mod:`.plugin` holds :class:`AgenticKnowledgePlugin`, the only entry point the
utility pipeline knows about; the rest of the package is its internals — the
adapters that read each agent's payload, the miner, the query and assembly of a
retrieval, the caches, and the stream the knowledge is published on.
"""

from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
    AgenticKnowledgeConfig,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.plugin import (
    AgenticKnowledgePlugin,
)

__all__ = ["AgenticKnowledgePlugin", "AgenticKnowledgeConfig"]
