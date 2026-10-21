"""
The plugin is wired into the utility registry under its own name.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from llm_router_plugins.utils.registry import MAIN_UTILS_REGISTRY
from llm_router_plugins.utils.knowledge.agentic_knowledge import (
    AgenticKnowledgePlugin,
)


def test_plugin_is_registered_under_its_name():
    assert AgenticKnowledgePlugin.name == "agentic_knowledge"
    assert MAIN_UTILS_REGISTRY.get("agentic_knowledge") is AgenticKnowledgePlugin


def test_registered_plugin_is_instantiable_without_connections():
    # Constructing the plugin must not require sse_lib or redis installed: it
    # resolves its collaborators lazily.
    plugin = MAIN_UTILS_REGISTRY["agentic_knowledge"]()
    assert plugin.name == "agentic_knowledge"
