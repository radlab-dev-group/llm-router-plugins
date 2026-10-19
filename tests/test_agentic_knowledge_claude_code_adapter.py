"""
The Claude Code adapter: recognize, scope, read the newest command, inject.

This is the basic adapter: it identifies the repository from the declared
working directory, reads the latest user message, resolves the mode, and puts
knowledge into ``system``.  It does not mine episodes or escalate, and a
request it cannot serve comes back unchanged.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agentic_knowledge_support import knowledge_config

from llm_router_plugins.utils.knowledge.agentic_knowledge import (
    AgenticKnowledgePlugin,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.adapters.claude_code import (
    CLAUDE_ROUTING_PLUGIN,
    ClaudeCodeAgentAdapter,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.cache import (
    InMemoryContextCache,
)
from agentic_knowledge_support import FakePublisher, FakeStore, store_hit

MARKER = "<agentic_knowledge>"


def claude_payload(mode=None):
    system = "You are a coding assistant.\nWorking directory: /home/u/dev/webapp"
    messages = [
        {"role": "user", "content": "Refactor the login form"},
        {"role": "assistant", "content": [{"type": "text", "text": "On it."}]},
        {"role": "user", "content": "Now fix the validation errors"},
    ]
    payload = {"model": "claude-sonnet", "system": system, "messages": messages}
    if mode is not None:
        payload["routing"] = {"plugin": CLAUDE_ROUTING_PLUGIN, "mode": mode}
    return payload


def test_matches_routing_plugin_and_anthropic_shape():
    adapter = ClaudeCodeAgentAdapter()
    assert adapter.matches(claude_payload(mode="plan"))
    assert adapter.matches(claude_payload())
    assert not adapter.matches({"model": "m", "input": []})


def test_parse_reads_scope_mode_and_latest_command():
    adapter = ClaudeCodeAgentAdapter()
    turn = adapter.parse(claude_payload())
    assert turn.agent == "claude_code"
    assert turn.scope == "webapp"
    assert turn.mode == "implement"  # default when routing names no mode
    assert turn.latest_user_text == "Now fix the validation errors"
    assert turn.evidence == ()  # no mining for this adapter


def test_parse_applies_the_mode_map():
    adapter = ClaudeCodeAgentAdapter(mode_map={"planning": "plan"})
    turn = adapter.parse(claude_payload(mode="planning"))
    assert turn.mode == "plan"


def test_inject_appends_a_text_block_to_system():
    adapter = ClaudeCodeAgentAdapter()
    payload = claude_payload()
    out = adapter.inject(payload, "SOME KNOWLEDGE")
    assert MARKER in out["system"]
    assert "SOME KNOWLEDGE" in out["system"]


def test_inject_into_a_list_system_appends_a_block():
    adapter = ClaudeCodeAgentAdapter()
    payload = claude_payload()
    payload["system"] = [{"type": "text", "text": "You are helpful."}]
    out = adapter.inject(payload, "SOME KNOWLEDGE")
    assert isinstance(out["system"], list)
    assert MARKER in out["system"][-1]["text"]


def test_inject_is_idempotent():
    adapter = ClaudeCodeAgentAdapter()
    payload = claude_payload()
    out = adapter.inject(payload, "SOME KNOWLEDGE")
    again = adapter.inject(out, "MORE")
    assert again["system"].count(MARKER) == 1


def test_set_model_updates_the_model_field():
    adapter = ClaudeCodeAgentAdapter()
    payload = claude_payload()
    adapter.set_model(payload, "claude-opus")
    assert payload["model"] == "claude-opus"


def test_parse_returns_none_for_unclaimed_payload():
    adapter = ClaudeCodeAgentAdapter()
    assert adapter.parse({"model": "m", "input": []}) is None


def test_plugin_serves_claude_code_without_mining_or_escalation():
    config = knowledge_config(escalation_enabled=True)
    store = FakeStore(hits=(store_hit("doc", text="repo convention"),))
    publisher = FakePublisher()
    plugin = AgenticKnowledgePlugin(
        config=config,
        store=store,
        cache=InMemoryContextCache(),
        publisher=publisher,
        adapters=(ClaudeCodeAgentAdapter(),),
    )
    out = plugin.apply(claude_payload(mode="plan"))
    # 'plan' is a served mode, so knowledge is read and injected into system.
    assert MARKER in out["system"]
    # No episodes are mined by this adapter, so nothing is published.
    assert publisher.published == []
    # No escalation for Claude Code.
    assert "escalation" not in out["routing"]["knowledge"]


def test_generation_is_stable_across_one_user_turn():
    """
    The cache needs an identity that survives the requests of one command.

    Claude Code sends no turn id and the routing annotation carries none, so the
    identity is keyed on the newest user message: unchanged while the agent
    works, new when the user speaks.  Hashing the whole history instead hands
    every request a fresh identity, and the cache then misses every time.
    """
    adapter = ClaudeCodeAgentAdapter()
    first = adapter.parse(claude_payload())
    assert first.generation

    grew = claude_payload()
    grew["messages"].append(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Reading files."}],
        }
    )
    assert adapter.parse(grew).generation == first.generation

    spoke = claude_payload()
    spoke["messages"].append({"role": "user", "content": "Now run the tests"})
    assert adapter.parse(spoke).generation != first.generation


def test_identity_is_not_read_from_the_routing_annotation():
    """The Claude Code router records how it matched a model, not who asked."""
    adapter = ClaudeCodeAgentAdapter()
    turn = adapter.parse(claude_payload(mode="plan"))
    assert turn.turn_id == ""
    assert turn.thread_id == ""
    assert turn.generation, "the turn is unidentified"


def test_session_id_comes_from_the_client_metadata():
    adapter = ClaudeCodeAgentAdapter()
    payload = claude_payload()
    payload["metadata"] = {"user_id": '{"session_id": "sess-42"}'}
    assert adapter.parse(payload).session_id == "sess-42"
