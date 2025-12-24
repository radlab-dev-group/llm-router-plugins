"""
Central registry for utils plugins.
"""

from llm_router_plugins.utils.rag.langchain_plugin import LangchainRAGPlugin

MAIN_UTILS_REGISTRY = {
    LangchainRAGPlugin.name: LangchainRAGPlugin,
}

UTILS_HOSTS_DEFINITION = {}

# Runtime session cache – filled by the registrator.
UTILS_REGISTRY_SESSION = {}
