"""
Executable pipeline for utility plugins.

It works like the original guard‑rail pipeline: an ordered list of plugin
identifiers is supplied, each plugin is registered (if not already), and then
their ``apply`` methods are called sequentially on the payload.
"""

import logging
from typing import Tuple, Dict, List

from llm_router_plugins.utils.plugin_registrator import UtilsRegistry


class UtilsPipeline:
    """
    Represents an executable pipeline of utility plugins.

    The pipeline is built from an ordered list of plugin identifiers.
    Calling ``apply(payload)`` invokes each plugin's ``apply`` method
    sequentially, passing the result of one as the input to the next.
    """

    def __init__(self, plugin_names: List[str], logger: logging.Logger):
        """
        Initialise the pipeline.

        Args:
            plugin_names: Ordered identifiers of the plugins to load.
            logger:       Logger instance used for diagnostic output.
        """
        self._logger = logger

        # Ensure every requested plugin is instantiated and cached.
        for name in plugin_names:
            UtilsRegistry.register(name=name, logger=logger)

        # Resolve concrete plugin instances.
        self._plugin_instances = [UtilsRegistry.get(name) for name in plugin_names]

    def apply(self, payload: Dict) -> Tuple[bool, Dict]:
        """
        Execute the pipeline.

        Args:
            payload: Initial data passed to the first utility plugin.

        Returns:
            A tuple ``(is_successful, result)`` where ``is_successful`` is
            ``True`` when all plugins approve the payload; otherwise ``False``
            and ``result`` contains the error information.
        """
        result = payload
        for plugin_instance in self._plugin_instances:
            result = plugin_instance.apply(result)
        return result
