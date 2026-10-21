"""
The episode miner: recipes, dead ends, loops and the compaction archive.

The miner is stateless — it recomputes episodes from a turn's evidence and
deduplicates downstream — so a recipe is only mined from a failure, a change
and a green test; a dead end from the same failure after a change; and a loop
is a count, not a document.  A compaction turn archives its window instead.
"""

import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agentic_knowledge_support import (
    base_items,
    codex_payload,
    compaction_payload,
    deadend_items,
    heredoc_recipe_items,
    loop_items,
    recipe_items,
)

from llm_router_plugins.utils.knowledge.agentic_knowledge.adapters.codex import (
    CodexAgentAdapter,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.documents import (
    KIND_DEADEND,
    KIND_RECIPE,
    KIND_SESSION_ARCHIVE,
    parse_document_text,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.miner import EpisodeMiner


def mine(payload):
    return EpisodeMiner().mine(CodexAgentAdapter().parse(payload))


def test_recipe_is_mined_from_failure_patch_and_green_test():
    result = mine(codex_payload(recipe_items()))
    assert len(result.events) == 1
    event = result.events[0]
    assert event.kind == KIND_RECIPE
    assert event.fingerprint
    assert event.signature
    # The document carries a header a reader can read back.
    meta, body = parse_document_text(event.text)
    assert meta.kind == KIND_RECIPE
    assert "Diff that made it pass" in body


def test_deadend_is_mined_from_the_same_failure_after_a_patch():
    result = mine(codex_payload(deadend_items()))
    kinds = [event.kind for event in result.events]
    assert KIND_DEADEND in kinds
    deadend = next(event for event in result.events if event.kind == KIND_DEADEND)
    assert deadend.fingerprint


def test_a_loop_is_reported_not_published():
    result = mine(codex_payload(loop_items(failures=4)))
    assert result.loop_count == 4
    assert result.signature_hash
    # A pure loop is a signal, not a document.
    assert result.events == ()


def test_a_clean_turn_mines_nothing():
    payload = codex_payload(
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
    result = mine(payload)
    assert result.events == ()
    assert result.loop_count == 0


def test_repeated_history_yields_a_stable_fingerprint():
    payload = codex_payload(recipe_items())
    first = mine(payload).events
    second = mine(codex_payload(recipe_items())).events
    assert [event.fingerprint for event in first] == [
        event.fingerprint for event in second
    ]


def test_compaction_archives_the_window_in_chunks():
    result = mine(compaction_payload())
    assert result.events
    assert all(event.kind == KIND_SESSION_ARCHIVE for event in result.events)
    positions = sorted(event.metadata["position"] for event in result.events)
    assert positions == list(range(len(positions)))
    # Every fragment names the thread so the archive can be filtered.
    assert all(event.metadata["thread_id"] == "thread-1" for event in result.events)


def test_archive_can_be_disabled():
    miner = EpisodeMiner(archive_enabled=False)
    result = miner.mine(CodexAgentAdapter().parse(compaction_payload()))
    assert result.events == ()


def test_a_recipe_carries_the_phase_of_its_own_episode():
    """
    Routing decided `debug` for the turn; the recipe was made while testing.

    The phase is what a later request filters on, so it is the phase of the
    episode — the mode its evidence names — not the mode of the whole turn.
    """
    event = mine(codex_payload(recipe_items(), mode="debug")).events[0]
    assert event.mode == "debug"
    assert event.metadata["phase"] == "test"
    assert parse_document_text(event.text)[0].phase == "test"


def test_a_deadend_carries_the_phase_it_was_stuck_in():
    event = mine(codex_payload(deadend_items(), mode="debug")).events[0]
    assert event.metadata["phase"] == "debug"
    assert parse_document_text(event.text)[0].phase == "debug"


def test_an_archived_fragment_carries_the_mode_of_its_turn():
    result = mine(compaction_payload(mode="implement"))
    assert result.events
    for event in result.events:
        assert event.metadata["phase"] == "implement"


def test_recipe_is_mined_from_a_patch_piped_through_the_shell():
    """A current Codex CLI patches through ``exec_command``, naming no tool."""
    result = mine(codex_payload(heredoc_recipe_items()))
    kinds = [event.kind for event in result.events]
    assert kinds == [KIND_RECIPE]


def test_recipe_quotes_the_envelope_not_the_command_that_carried_it():
    result = mine(codex_payload(heredoc_recipe_items()))
    event = result.events[0]
    _, body = parse_document_text(event.text)
    assert "*** Begin Patch" in body
    assert "apply_patch <<'PATCH'" not in body
    assert '"cmd"' not in body


def test_a_python_module_invocation_verifies_the_recipe():
    """`python3 -m pytest` is a launcher; its `-m` does not narrow the suite."""
    command = "python3 -m pytest tests/test_a.py"
    result = mine(
        codex_payload(recipe_items(command=command, green_command=command))
    )
    kinds = [event.kind for event in result.events]
    assert kinds == [KIND_RECIPE]


def test_a_marker_filter_still_voids_the_recipe():
    command = "python3 -m pytest tests/test_a.py"
    result = mine(
        codex_payload(
            recipe_items(command=command, green_command=command + ' -m "not slow"')
        )
    )
    assert result.events == ()
