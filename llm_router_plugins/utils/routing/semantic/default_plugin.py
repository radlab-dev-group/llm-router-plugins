"""
Default semantic routing plugin.

NOTE: This is only a skeleton with a fallback model to test the whole logic.
It is not implemented yet.
"""

from llm_router_plugins.plugin_interface import PluginInterface


class DefaultSemanticRoutingPlugin(PluginInterface):
    """
    Default semantic routing plugin.

    Resolves 'auto' model names to the configured fallback model.
    """

    name = "default_semantic_routing"
    _FALLBACK_MODEL = "gpt-oss:120b"

    def apply(self, payload: dict) -> dict:
        if payload.get("model") == "auto":
            payload["model"] = self._FALLBACK_MODEL
        return payload
