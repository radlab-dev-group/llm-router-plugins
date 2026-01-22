"""
Masker plugin registry.

This module defines the global registries used by the LLM‑Router to discover
and instantiate masker plugins.  Two dictionaries are exported:

* ``MAIN_MASKERS_REGISTRY`` – a static mapping from plugin names to their
  concrete factory classes.  It is populated at import time and never
  mutated at runtime.
* ``MASKERS_REGISTRY_SESSION`` – a mutable cache that holds instantiated
  plugin objects for the duration of the current application session.
  The :class:`~llm_router_plugins.maskers.plugin_registrator.MaskerRegistry`
  helper populates this cache on demand.

The design allows the router to lazily create plugin instances only when
they are actually requested, while keeping a single source of truth for
available plugins.
"""

from llm_router_plugins.maskers.fast_masker_plugin import FastMaskerPlugin

MAIN_MASKERS_REGISTRY = {FastMaskerPlugin.name: FastMaskerPlugin}

MASKERS_HOSTS_DEFINITION = {
    # The Masker is available as a module, not as a service
    FastMaskerPlugin.name: None
}

MASKERS_REGISTRY_SESSION = {}
