from llm_router_plugins.maskers.fast_masker_plugin import FastMaskerPlugin

MAIN_MASKERS_REGISTRY = {FastMaskerPlugin.name: FastMaskerPlugin}

MASKERS_HOSTS_DEFINITION = {
    # The Masker is available as a module, not as a service
    FastMaskerPlugin.name: None
}

MASKERS_REGISTRY_SESSION = {}
