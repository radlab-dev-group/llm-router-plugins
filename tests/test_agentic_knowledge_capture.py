"""
Replay of captured production payloads through the parser and the miner.

Captures are written by the plugin itself when
``LLM_ROUTER_KNOWLEDGE_AGENTIC_DEBUG_DUMP_DIR`` is set; point the same variable
at that directory to replay them here::

    LLM_ROUTER_KNOWLEDGE_AGENTIC_DEBUG_DUMP_DIR=/tmp/kb-capture \
        python -m pytest tests/test_agentic_knowledge_capture.py -q -s

Without the variable, or with an empty directory, the module skips: it is a
replay harness, not a source of fixtures.  What it checks is deliberately
mechanical — the parser must account for every tool call the request carried,
and the miner must derive its events from the evidence the parser produced.
Those two invariants are exactly what "the layer mined nothing" has to satisfy.
"""

import json
import os
from pathlib import Path

import pytest

from llm_router_plugins.utils.knowledge.agentic_knowledge.adapters.codex import (
    CodexAgentAdapter,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.debug_capture import (
    DUMP_DIR_ENV,
    replay,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.miner import EpisodeMiner
from llm_router_plugins.utils.knowledge.agentic_knowledge.scope import ScopeResolver

_CALL_TYPES = frozenset(
    {
        "function_call",
        "custom_tool_call",
        "tool_call_call",
        "tool_call",
        "local_shell_call",
    }
)


def _captures():
    directory = os.getenv(DUMP_DIR_ENV, "").strip()
    if not directory:
        return []
    return sorted(Path(directory).glob("kb-*.json"))


ALL = _captures()
if not ALL:
    pytest.skip(f"no captures in {DUMP_DIR_ENV}", allow_module_level=True)


def _payload(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _adapter():
    return CodexAgentAdapter(scope_resolver=ScopeResolver())


def _raw_call_names(payload):
    """Every tool name the request carried, whatever item type it rode in on."""
    names = []
    items = payload.get("input") if isinstance(payload, dict) else None
    for item in items or ():
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "") in _CALL_TYPES and item.get("call_id"):
            names.append(str(item.get("name") or "?"))
    return names


@pytest.mark.parametrize("path", ALL, ids=lambda path: path.name)
def test_captured_payload_reports(path):
    """The replay report is printed for a human reading the run."""
    print("\n" + replay(Path(path)))
    assert _adapter().parse(_payload(path)) is not None


@pytest.mark.parametrize("path", ALL, ids=lambda path: path.name)
def test_parser_accounts_for_every_tool_call(path):
    """A tool call in the payload must appear in the parsed activity."""
    payload = _payload(path)
    turn = _adapter().parse(payload)
    seen = {item.name for item in turn.activity if item.kind == "function_call"}
    missing = {name for name in _raw_call_names(payload) if name not in seen}
    assert not missing, "payload carried %s but the parser saw only %s" % (
        sorted(set(_raw_call_names(payload)) or ["-"]),
        sorted(seen) or ["-"],
    )


@pytest.mark.parametrize("path", ALL, ids=lambda path: path.name)
def test_events_are_derived_from_evidence(path):
    """A mined recipe needs a failed test and a patch in the same evidence."""
    turn = _adapter().parse(_payload(path))
    result = EpisodeMiner().mine(turn)
    kinds = {str(event.kind) for event in result.events}
    if "recipe" not in kinds:
        return
    evidence = {item.kind for item in turn.evidence}
    failed = any(
        item.kind == "test_failure" or item.succeeded is False
        for item in turn.evidence
    )
    assert failed, f"recipe without a failure: {evidence}"
    assert "patch" in evidence, f"recipe without a patch: {evidence}"
