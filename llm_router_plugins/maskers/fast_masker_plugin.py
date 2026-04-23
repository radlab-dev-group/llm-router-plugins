"""
FastMasker plugin entry point.

This module provides the concrete :class:`FastMaskerPlugin` implementation
that integrates the generic ``FastMasker`` core with the LLM‑Router plugin
framework.

Typical usage (handled by the router itself)::

    from llm_router_plugins.maskers.fast_masker_plugin import FastMaskerPlugin

    plugin = FastMaskerPlugin()
    masked_payload = plugin.apply({"text": "User email: alice@example.com"})

The plugin:
* inherits from :class:`llm_router_plugins.plugin_interface.PluginInterface`,
  gaining a standard ``apply`` contract,
* lazily constructs
  a :class:`~llm_router_plugins.maskers.fast_masker.core.FastMasker`
  instance, and
* delegates the actual masking work to ``FastMasker.mask_payload``.
"""

import logging
from typing import Dict, Optional, Tuple, Any

from llm_router_plugins.plugin_interface import PluginInterface
from llm_router_plugins.maskers.fast_masker.core import FastMasker


class FastMaskerPlugin(PluginInterface):
    """
    Plugin wrapper exposing the FastMasker engine.

    Attributes
    ----------
    name : str
        Identifier used by the router registry (``"fast_masker"``).

    _fast_masker : FastMasker
        Core masker instance that holds the rule set and performs the
        transformation of payloads.

    Methods
    -------
    apply(payload: Dict) -> Dict
        Mask the supplied payload using the configured rule pipeline and
        return a new dictionary with all sensitive data redacted.
    """

    name = "fast_masker"

    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger=logger)
        self._fast_masker = FastMasker()

    def apply(self, payload: Dict) -> Tuple[Any, Dict]:
        """
        Mask *payload* and return the redacted version and its mappings.
        """
        return self._fast_masker.mask_payload(payload=payload)
