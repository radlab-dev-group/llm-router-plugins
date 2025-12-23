"""
Central registry for utils plugins.
"""

from llm_router_plugins.utils.rag.langchain_plugin import LangchainRAGPlugin

MAIN_UTILS_REGISTRY = {
    # LangchainRAGPlugin is available as a module, not as a service
    LangchainRAGPlugin.name: None,
}

UTILS_HOSTS_DEFINITION = {}

# Runtime session cache – filled by the registrator.
UTILS_REGISTRY_SESSION = {}
