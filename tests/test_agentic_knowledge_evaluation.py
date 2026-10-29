"""
The counterfactual replay: deterministic, and leak-free by construction.

The replay must produce the same report on every run, and a scored turn must
never see a document mined from its own turn or a later one — "would have
helped" must not quietly mean "already knew the answer".
"""

import sys
import pathlib
import copy

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agentic_knowledge_support import knowledge_config

from llm_router_plugins.utils.knowledge.agentic_knowledge.evaluation import (
    VARIANT_ESCALATE,
    VARIANT_NONE,
    VARIANT_READ,
    evaluate,
    load_sessions,
)

DATASET = (
    pathlib.Path(__file__).resolve().parent / "data" / "agentic_knowledge_eval.json"
)


def report_with(escalation=False):
    config = knowledge_config(escalation_enabled=escalation)
    return evaluate(config, load_sessions(DATASET))


def test_report_is_deterministic():
    first = report_with()
    second = report_with()
    assert first == second


def test_report_structure_and_totals():
    report = report_with()
    assert report["sessions"] == 3
    assert report["turns"] == 5
    for variant in (VARIANT_NONE, VARIANT_READ, VARIANT_ESCALATE):
        block = report["variants"][variant]
        for metric in (
            "recipe_hit_rate",
            "turns_potentially_saved",
            "escalations",
            "injected_chars_per_turn",
        ):
            assert metric in block


def test_no_kb_variant_reads_nothing():
    report = report_with()
    none = report["variants"][VARIANT_NONE]
    assert none["injected_chars_per_turn"] == 0.0
    assert none["escalations"] == 0
    assert none["turns_potentially_saved"] == 0
    # The no-knowledge baseline misses the saves the read variant catches.
    assert (
        none["missed_saves"]
        == report["variants"][VARIANT_READ]["turns_potentially_saved"]
    )
    assert report["variants"][VARIANT_READ]["recipe_hit_rate"] > 0


def test_no_future_leak_a_turn_never_sees_its_own_or_later_docs():
    report = report_with()
    for turns in _session_turns(report):
        for index, detail in enumerate(turns):
            # Map mined-at turn ids back to their positions in the session.
            for mined in detail["retrieved_mined_at"]:
                mined_position = next(
                    position
                    for position, other in enumerate(turns)
                    if other["turn"] == mined
                )
                assert mined_position < index, (
                    f"turn {detail['turn']} retrieved a document mined at or "
                    f"after itself: {mined}"
                )


def test_loop_detection_meets_ground_truth():
    report = report_with()
    loop = report["loop_detection"]
    assert loop["false_positive"] == 0
    assert loop["false_negative"] == 0
    assert loop["precision"] == 1.0
    assert loop["recall"] == 1.0


def test_read_escalate_reports_escalations():
    report = report_with(escalation=True)
    assert report["variants"][VARIANT_ESCALATE]["escalations"] >= 1


def test_read_only_never_escalates():
    report = report_with(escalation=False)
    assert report["variants"][VARIANT_READ]["escalations"] == 0
    assert report["variants"][VARIANT_ESCALATE]["escalations"] == 0


def test_replay_ignores_unknown_extra_fields():
    sessions = load_sessions(DATASET)
    sessions = copy.deepcopy(sessions)
    sessions[0]["turns"][0]["bogus_field"] = "ignored"
    config = knowledge_config(escalation_enabled=True)
    report = evaluate(config, sessions)
    assert report["turns"] == 5


# ---------------------------------------------------- synthetic scenarios --
def test_the_synthetic_scenarios_mine_the_kinds_the_read_path_needs():
    """The shapes the corpus lacks, with the kinds they must produce.

    A replay over recorded sessions can only see what the sessions were.
    The shapes a real repository produces — a fix, a fix after a dead
    end, a procedure with no failure, a procedure that later fails, a
    shell edit — are what stage 1 made visible and what the read path
    serves, so they are asserted here, where they can fail.
    """
    from agentic_knowledge_support import (
        fix_then_procedure_items,
        procedure_items,
        recipe_items,
        heredoc_recipe_items,
        codex_payload,
        base_items,
        _command,
        _command_output,
        _patch,
        FAILING,
    )

    from llm_router_plugins.utils.knowledge.agentic_knowledge.miner import (
        EpisodeMiner,
    )
    from llm_router_plugins.utils.knowledge.agentic_knowledge.adapters.codex import (
        CodexAgentAdapter,
    )
    from llm_router_plugins.utils.knowledge.agentic_knowledge.documents import (
        KIND_DEADEND,
        KIND_PLAYBOOK,
        KIND_RECIPE,
    )

    miner = EpisodeMiner()

    def kinds_of(items, mode="debug"):
        turn = CodexAgentAdapter().parse(codex_payload(items, mode=mode))
        return sorted(event.kind for event in miner.mine(turn).events)

    # failure -> patch -> green: a recipe
    assert kinds_of(recipe_items()) == [KIND_RECIPE]
    # failure -> patch -> failure: the attempt is a dead end
    deadend = base_items() + [
        _command("c1", "e1"),
        _command_output("c1", "o1", FAILING, code=1),
        _patch("p1", "pe"),
        _command("c2", "e2"),
        _command_output("c2", "o2", FAILING, code=1),
    ]
    assert kinds_of(deadend) == [KIND_DEADEND]
    # a failure fixed and, afterwards, a wider test run that accepted the
    # whole turn: the recipe covers the later change, so the procedure that
    # would repeat it is not written twice
    assert kinds_of(fix_then_procedure_items(), mode="implement") == [KIND_PLAYBOOK]
    # patch -> green: a procedure
    assert kinds_of(procedure_items(), mode="implement") == [KIND_PLAYBOOK]
    # a shell edit is still an edit: the recipe closes on it
    assert kinds_of(heredoc_recipe_items()) == [KIND_RECIPE]
    # patch -> green -> failure: the work was accepted, then broke; the
    # procedure stands, the later failure has no fix yet
    broke = procedure_items() + [
        _command("c9", "e9"),
        _command_output("c9", "o9", FAILING, code=1),
    ]
    assert kinds_of(broke, mode="implement") == [KIND_PLAYBOOK]


def test_a_replay_with_documentation_counts_the_map_separately():
    """The standing map is a fixed cost, measured apart from the retrieval.

    A session that declares documentation replays with its digest
    injected on every request of the session, and the report measures the
    block a request is shown and the retrieved content in it as two
    numbers, so a deployment can see the map and the earnings apart.
    """
    config = knowledge_config()
    report = evaluate(config, load_sessions(DATASET))
    read = report["variants"][VARIANT_READ]
    assert read["injected_chars_per_turn"] > read["retrieved_chars_per_turn"]
    assert read["turns_with_doc_context"] == 2
    # A session that declared no documentation replays without a map.
    none = report["variants"][VARIANT_NONE]
    assert none["turns_with_doc_context"] == 0
    assert none["retrieved_chars_per_turn"] == 0.0


# ------------------------------------------------------------------ helpers -
def _session_turns(report):
    """Group the per-turn detail back into its session, in order."""
    grouped = {}
    for detail in report["turn_details"]:
        grouped.setdefault(detail["session"], []).append(detail)
    return list(grouped.values())


def _session_order(report):
    """Session ids in the order their turns first appear."""
    order = []
    for detail in report["turn_details"]:
        if detail["session"] not in order:
            order.append(detail["session"])
    return order
