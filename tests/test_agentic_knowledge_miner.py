"""
The episode miner: recipes, procedures, dead ends, loops and the archive.

The miner is stateless — it recomputes episodes from a turn's evidence and
deduplicates downstream — so a recipe is only mined from a failure, a change
and a green test; a dead end from the same failure after a change; a procedure
from a change a green run accepted with no failure in between; and a loop is a
count, not a document.  A compaction turn archives its window instead.
"""

import json
import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agentic_knowledge_support import (
    FAILING,
    GREEN,
    _command,
    _command_output,
    _patch,
    base_items,
    codex_payload,
    compaction_payload,
    deadend_items,
    fix_then_procedure_items,
    heredoc_recipe_items,
    loop_items,
    procedure_items,
    recipe_items,
)

from llm_router_plugins.utils.knowledge.agentic_knowledge.adapters.codex import (
    CodexAgentAdapter,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.documents import (
    KIND_DEADEND,
    KIND_PLAYBOOK,
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


def test_a_change_its_tests_accepted_is_mined_as_a_procedure():
    result = mine(codex_payload(procedure_items(), mode="implement"))
    assert len(result.events) == 1
    event = result.events[0]
    assert event.kind == KIND_PLAYBOOK
    assert event.fingerprint
    # No failure happened, so nothing normalizes into a signature — and the
    # document still says what was changed and what proved it.
    assert event.signature_hash == ""
    assert event.metadata["test_command"] == "pytest tests/test_a.py"
    assert "src/export.py" in event.metadata["paths"]
    meta, body = parse_document_text(event.text)
    assert meta.kind == KIND_PLAYBOOK
    assert "Steps that worked" in body
    assert "fix the failing export test" in body


def test_a_procedure_records_the_phase_it_was_verified_in():
    result = mine(codex_payload(procedure_items(), mode="implement"))
    event = result.events[0]
    # The patch was written while implementing, the run that accepted it was a
    # test run; the episode's own evidence says which one closed it.
    assert event.metadata["phase"] == "test"


def test_a_verified_fix_is_a_recipe_and_not_also_a_procedure():
    kinds = [event.kind for event in mine(codex_payload(recipe_items())).events]
    assert kinds == [KIND_RECIPE]


def test_a_wider_green_run_records_the_change_it_accepted():
    # The second run proved a different scope than the one that failed, so no
    # fix is claimed — but a run accepted the later change, and that is a step
    # worth repeating even though nothing ever broke.
    result = mine(codex_payload(fix_then_procedure_items(), mode="debug"))
    kinds = [event.kind for event in result.events]
    assert kinds == [KIND_PLAYBOOK]
    procedure = result.events[0]
    assert "src/other.py" in procedure.metadata["paths"]
    assert procedure.metadata["test_command"] == "pytest tests/test_b.py"


def test_a_procedure_never_duplicates_the_recipe_that_covers_it():
    # A last, wider test run closes the failure and its later patch together,
    # so the recipe already carries that change; a second document for it
    # would spend the mode's budget saying the same thing twice.
    items = base_items() + [
        _command("c1", "e1", "pytest"),
        _command_output("c1", "o1", FAILING, code=1),
        _patch("p1", "pe1", ("src/a.py",)),
        _command("c2", "e2", "pytest"),
        _command_output("c2", "o2", GREEN, code=0),
        _patch("p3", "pe3", ("src/b.py",)),
        _command("c3", "e3", "pytest"),
        _command_output("c3", "o3", GREEN, code=0),
    ]
    kinds = [event.kind for event in mine(codex_payload(items, mode="debug")).events]
    assert kinds == [KIND_RECIPE]


def test_each_accepted_round_of_a_turn_is_its_own_procedure():
    result = mine(codex_payload(procedure_items(rounds=2), mode="implement"))
    procedures = [e for e in result.events if e.kind == KIND_PLAYBOOK]
    assert len(procedures) == 2
    assert len({event.fingerprint for event in procedures}) == 2


def test_a_procedure_fingerprint_survives_the_turn_replaying():
    first = mine(codex_payload(procedure_items(), mode="implement")).events
    second = mine(codex_payload(procedure_items(), mode="implement")).events
    assert [event.fingerprint for event in first] == [
        event.fingerprint for event in second
    ]


def test_a_change_no_test_run_accepted_is_not_recorded():
    items = base_items() + [_patch("p1", "pe", ("src/a.py",))]
    assert mine(codex_payload(items, mode="implement")).events == ()


def test_a_deadend_turn_records_no_procedure():
    kinds = [event.kind for event in mine(codex_payload(deadend_items())).events]
    assert KIND_PLAYBOOK not in kinds


def test_a_procedure_never_suppresses_an_escalation():
    # A procedure carries no signature, so it cannot be the "strong recipe"
    # that keeps a looping model in place; escalation reads recipes only.
    result = mine(codex_payload(procedure_items(), mode="implement"))
    event = result.events[0]
    meta, _ = parse_document_text(event.text)
    assert meta.signature_hash == ""


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


@pytest.mark.parametrize(
    "command, green_command",
    [
        ("pytest tests/test_a.py tests/test_b.py", "pytest tests/test_a.py"),
        ("pytest tests/test_a.py tests/test_b.py", "pytest tests/test_b.py"),
        ("pytest tests/test_a.py", "pytest tests/test_a.py::test_other"),
        ("pytest tests/test_a.py", "pytest tests/test_a.py.bak"),
        ("pytest tests/test_a.py", "pytest other/tests/test_a.py"),
        ("pytest tests/unit tests/integration", "pytest tests/unit"),
        ("pytest tests/test_a.py", "pytest tests/test_a.py -knot_test_x"),
        ("pytest tests/test_a.py", "pytest tests/test_a.py -mfast"),
        (
            "pytest tests/test_a.py",
            "pytest tests/test_a.py --deselect=tests/test_a.py::test_x",
        ),
        (
            "pytest tests/test_a.py",
            "pytest tests/test_a.py --ignore=tests/test_a.py",
        ),
        ("pytest tests/test_a.py", "pytest tests/test_a.py --ignore-glob='*a.py'"),
        ("pytest tests/test_a.py", "pytest tests/test_a.py --lf"),
        ("pytest tests/test_a.py", "pytest tests/test_a.py --collect-only"),
        ("pytest tests/test_a.py", "pytest tests/test_a.py || true"),
        ("pytest tests/test_a.py", "pytest tests/test_a.py --setup-plan"),
        ("pytest tests/test_a.py", "pytest tests/test_a.py --help"),
        ("pytest tests/test_a.py -k test_x", "pytest tests/test_a.py -k test_x"),
        ("pytest tests/test_a.py", "pytest tests/test_a.py --unknown-selection"),
    ],
)
def test_narrowed_or_unproven_green_scope_does_not_verify(command, green_command):
    result = mine(
        codex_payload(recipe_items(command=command, green_command=green_command))
    )
    assert not any(event.kind == KIND_RECIPE for event in result.events)


@pytest.mark.parametrize(
    "green_command",
    [
        "pytest tests/test_a.py tests/test_b.py",
        "pytest -q tests/test_b.py tests/test_a.py",
        "python3 -m pytest tests/test_a.py tests/test_b.py",
    ],
)
def test_equivalent_green_scope_verifies_all_original_files(green_command):
    result = mine(
        codex_payload(
            recipe_items(
                command="pytest tests/test_a.py tests/test_b.py",
                green_command=green_command,
            )
        )
    )
    assert [event.kind for event in result.events] == [KIND_RECIPE]


def test_narrowing_beyond_command_display_limit_does_not_verify():
    command = "pytest " + " ".join(
        f"tests/test_original_{index}.py" for index in range(12)
    )
    result = mine(
        codex_payload(
            recipe_items(command=command, green_command=command + " -knot_test_x")
        )
    )
    assert result.events == ()


@pytest.mark.parametrize("second_path", ["src/a.py", "src/success.py"])
def test_deadend_freezes_failed_attempt_before_successful_patch(second_path):
    items = deadend_items()
    before = mine(codex_payload(items))
    original = next(event for event in before.events if event.kind == KIND_DEADEND)
    items += [
        {
            "type": "function_call",
            "name": "apply_patch",
            "call_id": "p2",
            "id": "pe2",
            "arguments": f"*** Begin Patch\n*** Update File: {second_path}\n@@\n-x = 2\n+x = 3\n*** End Patch",
        },
        {
            "type": "function_call",
            "name": "exec_command",
            "call_id": "c3",
            "id": "e3",
            "arguments": json.dumps({"cmd": "pytest tests/test_a.py"}),
        },
        {
            "type": "function_call_output",
            "call_id": "c3",
            "id": "o3",
            "output": "Process exited with code 0\nOutput:\n1 passed",
        },
    ]
    result = mine(codex_payload(items))
    deadends = [event for event in result.events if event.kind == KIND_DEADEND]
    recipes = [event for event in result.events if event.kind == KIND_RECIPE]
    assert len(deadends) == len(recipes) == 1
    assert "+x = 2" in deadends[0].text
    assert "+x = 3" not in deadends[0].text
    assert deadends[0].fingerprint == original.fingerprint
    assert deadends[0].text == original.text
    assert deadends[0].metadata == original.metadata
    assert "+x = 2" in recipes[0].text
    assert "+x = 3" in recipes[0].text


def test_pending_patch_is_not_part_of_an_earlier_deadend():
    items = deadend_items() + [
        {
            "type": "function_call",
            "name": "apply_patch",
            "call_id": "p2",
            "id": "pe2",
            "arguments": "*** Begin Patch\n*** Update File: src/a.py\n@@\n-x = 2\n+x = 3\n*** End Patch",
        }
    ]
    result = mine(codex_payload(items))
    assert [event.kind for event in result.events] == [KIND_DEADEND]
    assert "+x = 2" in result.events[0].text
    assert "+x = 3" not in result.events[0].text


def test_each_failed_attempt_has_its_own_deadend():
    items = deadend_items() + [
        _patch("p2", "pe2", ("src/second.py",)),
        _command("c3", "e3"),
        _command_output("c3", "o3", FAILING),
    ]
    result = mine(codex_payload(items))
    deadends = [event for event in result.events if event.kind == KIND_DEADEND]
    assert len(deadends) == 2
    assert deadends[0].fingerprint != deadends[1].fingerprint
    assert "src/second.py" not in deadends[0].text
    assert "*** Update File: src/a.py" not in deadends[1].text
    assert "*** Update File: src/second.py" in deadends[1].text
    assert result.loop_count == 3


def test_interleaved_failure_keeps_attempts_separate():
    other_failure = "FAILED tests/test_b.py::test_y - ValueError: invalid input"
    items = base_items() + [
        _command("c1", "e1"),
        _command_output("c1", "o1", FAILING),
        _patch("p1", "pe1"),
        _command("cb", "eb", "pytest tests/test_b.py"),
        _command_output("cb", "ob", other_failure),
        _patch("pb", "peb", ("src/b.py",)),
        _command("c2", "e2"),
        _command_output("c2", "o2", FAILING),
    ]
    result = mine(codex_payload(items))
    deadends = [event for event in result.events if event.kind == KIND_DEADEND]
    assert len(deadends) == 1
    assert "*** Update File: src/a.py" in deadends[0].text
    assert "src/b.py" not in deadends[0].text


def test_repeated_failure_revokes_green_without_duplicate_attempts():
    items = recipe_items() + [
        _command("c3", "e3"),
        _command_output("c3", "o3", FAILING),
        _command("c4", "e4"),
        _command_output("c4", "o4", FAILING),
    ]
    result = mine(codex_payload(items))
    assert [event.kind for event in result.events] == [KIND_DEADEND]
    assert result.loop_count == 3
