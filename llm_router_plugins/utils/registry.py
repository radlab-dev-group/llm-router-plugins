"""
Central registry for util plugins.
"""

from llm_router_plugins.utils.rag.langchain_plugin import LangchainRAGPlugin
from llm_router_plugins.utils.routing.semantic_biencoder.semantic_biencoder_routing import (
    SemanticBiEncoderRoutingPlugin,
)
from llm_router_plugins.utils.routing.simple_semantic.simple_semantic_routing import (
    SimpleSemanticRoutingPlugin,
)

MAIN_UTILS_REGISTRY = {
    LangchainRAGPlugin.name: LangchainRAGPlugin,
    SimpleSemanticRoutingPlugin.name: SimpleSemanticRoutingPlugin,
    SemanticBiEncoderRoutingPlugin.name: SemanticBiEncoderRoutingPlugin,
}

UTILS_HOSTS_DEFINITION = {}

# Runtime session cache – filled by the registrator.
UTILS_REGISTRY_SESSION = {}
