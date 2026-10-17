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
