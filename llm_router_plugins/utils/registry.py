"""
Central registry for utils plugins.
"""

from llm_router_plugins.utils.rag.langchain_plugin import LangchainRAGPlugin
from llm_router_plugins.utils.routing.semantic.default_plugin import (
    DefaultSemanticRoutingPlugin,
)

MAIN_UTILS_REGISTRY = {
    LangchainRAGPlugin.name: LangchainRAGPlugin,
    DefaultSemanticRoutingPlugin.name: DefaultSemanticRoutingPlugin,
}

UTILS_HOSTS_DEFINITION = {}

# Runtime session cache – filled by the registrator.
UTILS_REGISTRY_SESSION = {}
