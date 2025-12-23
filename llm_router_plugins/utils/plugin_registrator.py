"""
Utility‑plugin registry.

This module provides a global registry for “utils” plugins.  It works in the
same way as the existing masker registry:

* ``MAIN_UTILS_REGISTRY`` – a mapping from plugin names to factory callables
  (provided by the utils plugin package).
* ``UTILS_REGISTRY_SESSION`` – a mutable mapping that holds instantiated
  plugin objects for the current runtime session.

All methods are static because the registry is global and does not require
instance state.
"""

import logging
from typing import Optional, Dict

from llm_router_plugins.utils.registry import (
    MAIN_UTILS_REGISTRY,
    UTILS_REGISTRY_SESSION,
)


class UtilsRegistry:
    """
    Central registry for utility plugins.

    Provides static helpers to register a plugin, retrieve an instantiated
    plugin by name, and list all plugins that are currently registered in the
    active session.
    """

    @staticmethod
    def register(name: str, logger: Optional[logging.Logger] = None) -> None:
        """
        Register a utility plugin for the current session.

        The function checks that the requested ``name`` exists in the global
        ``MAIN_UTILS_REGISTRY``. If the plugin has already been instantiated in
        ``UTILS_REGISTRY_SESSION`` the call becomes a no‑op. Otherwise a new
        instance is created using the factory callable and stored in the session
        registry.

        Args:
            name:   Identifier of the utility plugin to register.
            logger: Optional logger passed to the plugin’s constructor (if the
                    plugin accepts it).  ``None`` is safe – the plugin can ignore
                    it.

        Raises:
            KeyError: If ``name`` is not present in ``MAIN_UTILS_REGISTRY``.
        """
        if name not in MAIN_UTILS_REGISTRY:
            raise KeyError(
                f"Utility plugin '{name}' not found in registry: "
                f"{list(MAIN_UTILS_REGISTRY.keys())}"
            )

        if name in UTILS_REGISTRY_SESSION:
            # Already instantiated – nothing to do.
            return

        # Factory call – most plugins expect a ``logger`` kw‑arg, but they can
        # ignore it if they don’t need it.
        plugin_instance = MAIN_UTILS_REGISTRY[name](logger=logger)
        UTILS_REGISTRY_SESSION[name] = plugin_instance

        if logger:
            logger.info(
                f"[utils] Registered utility plugin '{name}' as instance "
                f"'{plugin_instance.__class__.__name__}'"
            )

    @staticmethod
    def get(name: str):
        """
        Retrieve a registered utility plugin instance by its name.

        Args:
            name: Identifier of the utility plugin.

        Returns:
            The instantiated plugin object associated with ``name``.

        Raises:
            KeyError: If the plugin is not found in the session registry.
        """
        try:
            return UTILS_REGISTRY_SESSION[name]
        except KeyError as exc:
            raise KeyError(
                f"Utility plugin '{name}' not found in session registry. "
                f"Available plugins: {UtilsRegistry.list_plugins()}"
            ) from exc

    @staticmethod
    def list_plugins() -> list[str]:
        """
        List all utility plugin names that have been registered in the current
        session.

        Returns:
            A list of identifiers currently stored in ``UTILS_REGISTRY_SESSION``.
        """
        return list(UTILS_REGISTRY_SESSION.keys())
