"""
The Codex adapter: contract, recognition, parsing and injection.

A contract test pins the routing structures the knowledge layer depends on —
if the Codex routing plugin renames them, this test fails here instead of a
recipe quietly never being written again.  The adapter reuses the routing
parser and phase rules, so a request it claims is one it can parse, and a
request it does not claim comes back untouched.
"""

import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agentic_knowledge_support import (
    base_items,
    codex_payload,
    compaction_payload,
    loop_items,
    recipe_items,
)

from llm_router_plugins.utils.knowledge.agentic_knowledge.adapters.codex import (
    KNOWLEDGE_MARKER,
    CodexAgentAdapter,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.payload import (
    CodexActivity,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.phase import (
    PhaseEvidence,
    collect_phase_evidence,
)


def adapter():
    return CodexAgentAdapter()


# --------------------------------------------------------------- contract --
def test_contract_reuses_the_routing_parser_and_phase_rules():
    """The adapter must agree with the routing plugin about what a turn is."""
    payload = codex_payload(recipe_items())
    from llm_router_plugins.utils.routing.agentic_routing.codex.payload import (
        CodexPayloadParser,
    )

    request = CodexPayloadParser().parse(payload)
    turn = adapter().parse(payload)
    # The adapter's activity must be a faithful copy of the routing request's.
    assert len(turn.activity) == len(request.activity)
    assert turn.turn_id == request.turn_id
    assert turn.thread_id == request.thread_id
    # Evidence is derived from the same rules the routing plugin uses.
    rules = adapter().phase_rules
    if rules is not None:
        expected = collect_phase_evidence(request.activity, rules)
        assert [e.kind for e in turn.evidence] == [e.kind for e in expected]


def test_contract_dataclasses_expose_the_fields_the_miner_reads():
    # The miner and query builder read exactly these; a rename must break here.
    for field_name in (
        "kind",
        "text",
        "name",
        "call_id",
        "event_id",
    ):
        assert hasattr(CodexActivity(kind="x", text=""), field_name)
    for field_name in ("kind", "mode", "call_id", "event_id", "succeeded"):
        assert hasattr(PhaseEvidence(mode="m", kind="k", reason="r"), field_name)


# ------------------------------------------------------------- recognition --
def test_matches_a_routing_annotated_codex_request():
    assert adapter().matches(codex_payload(recipe_items()))


def test_matches_structurally_without_routing():
    payload = codex_payload(recipe_items())
    del payload["routing"]
    assert adapter().matches(payload)


def test_does_not_match_an_unrelated_payload():
    assert not adapter().matches({"model": "m", "input": "no metadata here"})


# ------------------------------------------------------------------- parse --
def test_parse_reads_scope_mode_and_identifiers():
    turn = adapter().parse(codex_payload(recipe_items()))
    assert turn.agent == "codex"
    assert turn.scope == "webapp"
    assert turn.repo_root == "/repo/webapp"
    assert turn.mode == "debug"
    assert turn.routing_source == "phase"
    assert turn.request_class == "main"


def test_parse_reads_touched_paths_relative_to_the_repo():
    turn = adapter().parse(codex_payload(recipe_items(paths=("src/export.py",))))
    assert "src/export.py" in turn.touched_paths


# ----------------------------------------------------------- touched paths --
def _call(name, arguments, call_id="c1", event_id="e1"):
    """One ``function_call`` item carrying *arguments* verbatim."""
    return {
        "type": "function_call",
        "name": name,
        "call_id": call_id,
        "id": event_id,
        "arguments": arguments,
    }


def test_touched_paths_reads_json_encoded_apply_patch_arguments():
    """The wire shape of a real patch: one JSON string with escaped breaks."""
    arguments = json.dumps(
        {
            "input": "*** Begin Patch\n"
            "*** Update File: src/service/address.py\n"
            "@@\n"
            "-def validate(value):\n"
            "-    return True\n"
            "+def validate(value):\n"
            "+    return bool(value)\n"
            "*** End Patch\n"
        }
    )
    turn = adapter().parse(
        codex_payload(
            base_items() + [_call("apply_patch", arguments, "address-fix", "pe")]
        )
    )
    assert turn.touched_paths == ("src/service/address.py",)


def test_touched_paths_reads_a_patch_heredoc_inside_exec_command():
    """Codex patches through ``exec_command``, one here-document per patch."""
    patch_one = (
        "*** Begin Patch\n"
        "*** Update File: src/service/address.py\n"
        "@@\n"
        "-def validate(value):\n"
        "+    return bool(value)\n"
        "*** End Patch\n"
    )
    patch_two = (
        "*** Begin Patch\n"
        "*** Update File: CHANGELOG.md\n"
        "@@\n"
        "+# Unreleased\n"
        "+- Address validation\n"
        "*** End Patch\n"
    )
    arguments = json.dumps(
        {
            "cmd": "cd /repo/webapp && apply_patch <<'PATCH'\n"
            f"{patch_one}PATCH\n"
            "apply_patch <<'PATCH'\n"
            f"{patch_two}PATCH",
            "workdir": "/repo/webapp",
        }
    )
    turn = adapter().parse(
        codex_payload(base_items() + [_call("exec_command", arguments)])
    )
    # Both patches of the one call, in the order the agent applied them.
    assert turn.touched_paths == ("src/service/address.py", "CHANGELOG.md")


def test_touched_paths_reads_a_rename_from_a_json_command():
    """A rename names two files: the one patched and the one it becomes."""
    arguments = json.dumps(
        {
            "command": "apply_patch <<'PATCH'\n"
            "*** Begin Patch\n"
            "*** Update File: /repo/webapp/src/old_name.py\n"
            "*** Move to: src/new_name.py\n"
            "@@\n"
            "-old\n"
            "+new\n"
            "*** End Patch\n"
            "PATCH"
        }
    )
    turn = adapter().parse(
        codex_payload(base_items() + [_call("exec_command", arguments)])
    )
    assert turn.touched_paths == ("src/old_name.py", "src/new_name.py")


def test_touched_paths_are_deduplicated_over_the_replayed_turn():
    """Codex replays the whole turn: one file stays named once, still in order."""
    envelope = (
        "*** Begin Patch\n*** Update File: src/a.py\n@@\n-x\n+y\n*** End Patch"
    )
    encoded = json.dumps({"input": envelope})
    turn = adapter().parse(
        codex_payload(
            base_items()
            + [
                _call("apply_patch", encoded, "p1", "e1"),
                _call("apply_patch", envelope, "p2", "e2"),
                _call("apply_patch", json.dumps({"input": envelope}), "p3", "e3"),
            ]
        )
    )
    assert turn.touched_paths == ("src/a.py",)


def test_touched_paths_ignores_a_patch_example_in_another_here_document():
    """A document that shows what a patch looks like patched nothing."""
    arguments = json.dumps(
        {
            "cmd": "cat > docs/patching.md <<'EOF'\n"
            "A Codex patch reads:\n\n"
            "*** Begin Patch\n"
            "*** Update File: src/example.py\n"
            "*** End Patch\n"
            "EOF"
        }
    )
    turn = adapter().parse(
        codex_payload(base_items() + [_call("exec_command", arguments)])
    )
    assert turn.touched_paths == ()


def test_parse_marks_a_compaction_request_and_fills_the_window():
    turn = adapter().parse(compaction_payload())
    assert turn.request_class == "compaction"
    assert turn.window_text.strip()


def test_parse_reads_the_mode_from_agent_mode():
    turn = adapter().parse(codex_payload(loop_items(), mode="test"))
    assert turn.mode == "test"


def test_latest_failure_returns_the_last_test_failure():
    turn = adapter().parse(codex_payload(loop_items(failures=2)))
    failure = turn.latest_failure()
    assert failure is not None
    assert failure.kind == "test_failure"
    # A turn with no failure reports none.
    clean = adapter().parse(
        codex_payload(
            base_items()
            + [
                {
                    "type": "function_call",
                    "name": "exec_command",
                    "call_id": "c",
                    "id": "e",
                    "arguments": json.dumps({"cmd": "pytest"}),
                },
                {
                    "type": "function_call_output",
                    "call_id": "c",
                    "id": "o",
                    "output": "Process exited with code 0\nOutput:\n1 passed",
                },
            ]
        )
    )
    assert clean.latest_failure() is None


# --------------------------------------------------------------- injection --
def test_inject_appends_a_developer_item_at_the_end():
    payload = codex_payload(recipe_items())
    before = len(payload["input"])
    out = adapter().inject(payload, "KNOWLEDGE")
    assert len(out["input"]) == before + 1
    last = out["input"][-1]
    assert last["role"] == "developer"
    assert "KNOWLEDGE" in last["content"][0]["text"]


def test_inject_is_idempotent():
    payload = codex_payload(recipe_items())
    once = adapter().inject(payload, "KNOWLEDGE")
    twice = adapter().inject(once, "MORE")
    # The second injection is detected and not stacked.
    assert len(twice["input"]) == len(once["input"])
    assert json.dumps(twice).count(KNOWLEDGE_MARKER) >= 1


def test_inject_converts_a_string_input_into_a_list():
    payload = {
        "model": "auto_codex",
        "input": "just a prompt",
        "client_metadata": {"x-codex-turn-metadata": "{}"},
        "routing": {"plugin": "agentic_routing_codex"},
    }
    out = adapter().inject(payload, "KNOWLEDGE")
    assert isinstance(out["input"], list)
    assert out["input"][-1]["role"] == "developer"


def test_set_model_updates_the_model_field():
    payload = codex_payload(recipe_items())
    adapter().set_model(payload, "other-model")
    assert payload["model"] == "other-model"
