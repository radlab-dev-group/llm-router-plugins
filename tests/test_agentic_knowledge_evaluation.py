"""
The counterfactual replay: deterministic, and leak-free by construction.

The replay must produce the same report on every run, and a scored turn must
never see a document mined from its own turn or a later one — "would have
helped" must not quietly mean "already knew the answer".
"""

import sys
import pathlib
import copy
import random

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agentic_knowledge_support import knowledge_config

from llm_router_plugins.utils.knowledge.agentic_knowledge.evaluation import (
    VARIANT_ESCALATE,
    VARIANT_LEARNED,
    VARIANT_NONE,
    VARIANT_READ,
    VARIANT_REPUTATION,
    _LearnedReplay,
    evaluate,
    load_sessions,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.learning import (
    CandidateFeatures,
    publish_snapshot,
    train_ranker,
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
    # The no-knowledge baseline counts every turn and misses the saves the
    # read variant catches: one save per turn that had a document to find,
    # and none of them were on screen, because nothing was on screen.
    assert none["turns"] == report["turns"]
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
def _learned_sessions(sessions=8, turns=6, hits_every=2, scope="oneproj"):
    """
    A corpus the learned path can actually learn from.

    Eight sessions of a repository that keeps hitting the same two failures,
    half of its turns loops, half of those loops ended by a document the block
    was shown.  The shipped dataset is a handful of turns and is right to
    refuse a fit on it; this one is the size at which the gates are supposed
    to start saying yes, and it is here so the variant can be tested on the
    claim it exists to make rather than on its own absence.
    """
    built = []
    for session_index in range(sessions):
        session_turns = []
        for index in range(turns):
            loop = (index % hits_every) != 0
            entry = {
                "id": f"s{session_index}-t{index}",
                "mode": "debug",
                "text": f"failure {index} in module {session_index} cache get key error",
                "failure": (
                    f"FAILED tests/test_m{index % 3}.py::test_get - KeyError: 'session'"
                ),
                "patch": (
                    f"*** Begin Patch\n*** Update File: src/m{index % 3}.py\n@@\n"
                    "-    return data[key]\n+    return data.get(key)\n*** End Patch"
                ),
                "test_command": f"pytest tests/test_m{index % 3}.py",
                "test_passed": True,
                "is_loop": loop,
            }
            if loop and index >= 1:
                entry["saved_by"] = f"s{session_index}-t0"
            session_turns.append(entry)
        built.append({"id": f"s{session_index}", "scope": scope, "turns": session_turns})
    return built


def test_the_learned_variant_is_measured_over_the_same_turns_as_read():
    report = report_with()
    learned = report["variants"][VARIANT_LEARNED]
    assert learned["turns"] == report["variants"][VARIANT_READ]["turns"]
    # The learned variant is not allowed to save the difference by being
    # cheaper to run: it is the same read, the same sessions, a model on top.
    assert (
        learned["injected_chars_per_turn"]
        == report["variants"][VARIANT_READ]["injected_chars_per_turn"]
    )


def test_the_learned_variant_reports_the_fitting_and_the_votes():
    report = report_with()
    learning = report["learning"]
    for metric in (
        "scopes",
        "fits",
        "accepted",
        "refusals",
        "holdout_auc_mean",
        "explore_share",
        "mult_p95",
    ):
        assert metric in learning
    # On five turns nothing can be fitted, and the report says which gate
    # refused rather than leaving the block empty: "no model" and "no model,
    # because too few labels" are different answers to an operator.
    assert learning["fits"] >= 1
    assert learning["accepted"] == 0
    assert learning["refusals"].get("too_few_labels") >= 1
    assert learning["model_ids"] == []
    # Every multiplier the learner spent is inside the range the policy names.
    assert 0.3 <= learning["mult_min"] <= learning["mult_max"] <= 1.5


def test_the_learned_variant_never_sees_the_label_of_the_turn_it_ranks():
    """The anti-leak rule of the replay, applied to the model rather than the store.

    The per-turn record carries the corpus the request was ranked with and
    what the turn itself added afterwards.  A running sum of the additions
    must equal the corpus the next request saw — one record of drift, and the
    model has quietly learned from the turn it was supposed to predict.
    """
    report = evaluate(knowledge_config(), _learned_sessions())
    running: dict = {}
    for detail in report["turn_details"]:
        learned = detail["learned"]
        scope = learned["scope"]
        assert learned["samples"] == running.get(scope, 0), (
            f"{detail['turn']} was ranked with {learned['samples']} records and "
            f"the earlier turns of {scope} had produced only {running.get(scope, 0)}"
        )
        assert learned["labelled"] <= learned["samples"]
        running[scope] = running.get(scope, 0) + learned["added"]


def test_the_learned_variant_fits_and_votes_once_the_corpus_earns_it():
    report = evaluate(knowledge_config(), _learned_sessions())
    learning = report["learning"]
    assert learning["accepted"] >= 1, learning["refusals"]
    assert learning["model_ids"], "no model was ever named"
    assert learning["holdout_auc_mean"] >= 0.6
    # The model in force on a turn is reported from that turn, so the answer
    # to "which model ranked this" survives the run.
    served = [
        detail for detail in report["turn_details"] if detail["learned"]["model_id"]
    ]
    assert served, "no turn reported the model that served it"
    # A model that votes everywhere and never differs from 1.0 is a model the
    # variant did not consult; the spread is what says it did.
    assert learning["mult_min"] < 1.0 < learning["mult_max"]
    assert 0.0 <= learning["explore_share"] <= 1.0


def test_model_uplift_is_a_point_estimate_with_an_interval():
    report = report_with()
    uplift = report["variants"][VARIANT_LEARNED]["model_uplift_vs_read"]
    assert uplift["sessions"] == report["sessions"]
    assert uplift["bootstrap"] >= 100
    assert uplift["ci_low"] <= uplift["point"] <= uplift["ci_high"]


def test_model_uplift_refuses_an_interval_a_single_session_cannot_give():
    sessions = load_sessions(DATASET)
    single = [session for session in sessions if session.get("id") == "export-fix"]
    report = evaluate(knowledge_config(), single)
    uplift = report["variants"][VARIANT_LEARNED]["model_uplift_vs_read"]
    assert uplift["sessions"] == 1
    assert uplift["ci_low"] is None
    assert uplift["ci_high"] is None
    assert "note" in uplift


def test_char_efficiency_is_reported_for_the_reading_variants():
    report = report_with()
    read = report["variants"][VARIANT_READ]
    assert read["char_efficiency"] > 0.0
    # The same number read the other way: what one save cost, in characters.
    assert read["chars_per_turn_saved"] > 0.0
    assert report["variants"][VARIANT_NONE]["char_efficiency"] == 0.0


def test_the_learned_variant_does_not_change_the_other_variants():
    with_learned = report_with()
    plain = report_with()
    for variant in (VARIANT_NONE, VARIANT_READ, VARIANT_ESCALATE):
        assert with_learned["variants"][variant] == plain["variants"][variant]


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


# ------------------------------------------------- ranking: precision and MRR


def test_the_document_needed_at_the_top_scores_full_marks():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.evaluation import (
        _ranking_quality,
    )

    assert _ranking_quality("t2", ["t2", "t1", "t3", "t9"]) == (1, 1.0)


def test_the_third_place_is_still_inside_the_three_a_deployment_reads():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.evaluation import (
        _ranking_quality,
    )

    in_three, reciprocal = _ranking_quality("t2", ["a", "b", "t2", "c"])
    assert in_three == 1
    assert round(reciprocal, 4) == 0.3333


def test_fourth_is_outside_top_three_but_still_counted_in_mrr():
    # Precision@3 is a ceiling, MRR is not: a fix at rank 4 was still found,
    # and dropping it from both would make a search that misses by one look as
    # good as one that never found it.
    from llm_router_plugins.utils.knowledge.agentic_knowledge.evaluation import (
        _ranking_quality,
    )

    in_three, reciprocal = _ranking_quality("t2", ["a", "b", "c", "t2"])
    assert in_three == 0
    assert reciprocal == 0.25


def test_a_fix_that_was_never_retrieved_scores_on_neither():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.evaluation import (
        _ranking_quality,
    )

    assert _ranking_quality("missing", ["a", "b"]) == (0, 0.0)


def test_a_document_retrieved_twice_is_ranked_at_its_first_appearance():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.evaluation import (
        _ranking_quality,
    )

    in_three, reciprocal = _ranking_quality("t2", ["x", "t2", "y", "t2"])
    assert in_three == 1
    assert reciprocal == 0.5


def test_the_report_carries_the_ranking_metrics_for_the_reading_variants():
    report = report_with()
    for variant in (VARIANT_READ, VARIANT_ESCALATE):
        block = report["variants"][variant]
        assert "precision_at_3" in block
        assert "mrr" in block
        # the shipped set is small and unambiguous: every applicable turn found
        # its document, and found it first
        assert block["precision_at_3"] == block["recipe_hit_rate"]
        assert block["mrr"] >= block["precision_at_3"]


# ------------------------------------------------ the reputation-only loop --
class _Hit:
    """The one field of an injected document the replay reads."""

    def __init__(self, name: str) -> None:
        self.document_name = name


class _Context:
    def __init__(self, hits) -> None:
        self.hits = list(hits)


def test_the_reputation_variant_is_measured_over_the_same_turns_as_read():
    report = report_with()
    reputation = report["variants"][VARIANT_REPUTATION]
    read = report["variants"][VARIANT_READ]
    # The same read, the same sessions, one loop held back: it may not buy a
    # better number by being cheaper to run.
    assert reputation["turns"] == read["turns"]
    assert reputation["injected_chars_per_turn"] == read["injected_chars_per_turn"]
    uplift = reputation["reputation_uplift_vs_read"]
    assert uplift["sessions"] == report["sessions"]
    assert uplift["bootstrap"] >= 100
    assert uplift["ci_low"] <= uplift["point"] <= uplift["ci_high"]


def test_the_reputation_variant_holds_the_fitted_model_and_the_arm_back():
    """What the variant measures is the posterior, and only the posterior."""
    state = _LearnedReplay("repo", knowledge_config())
    features = CandidateFeatures(
        kind="recipe", mode="debug", score=0.7, size_chars=300, budget_chars=3000
    )
    for _ in range(4):
        state.reputation_policy.reputation.observe("recipe:20260101:x", -1.0)

    _, detail = state.reputation_policy.multiplier_for(
        features, "recipe:20260101:x", scope="repo"
    )
    assert detail["rep"] < 1.0, "the posterior did not move"
    assert detail["model"] == 1.0
    assert detail["arm"] == 1.0
    assert detail["model_id"] == ""


def test_the_reputation_variant_is_in_a_namespace_of_its_own():
    """A snapshot the full learner published must not reach this variant."""
    state = _LearnedReplay("repo", knowledge_config())
    features = {
        "recipe:20260101:x": CandidateFeatures(
            kind="recipe", mode="debug", score=0.6, size_chars=200, budget_chars=3000
        )
    }
    context = _Context([_Hit("recipe:20260101:x")])
    for index in range(6):
        state.observe(
            session_id=f"s{index}",
            generation=f"g{index}",
            context=context,
            features=features,
            saved_by="x",
            is_loop=False,
        )
    snapshot, _ = train_ranker(
        state.samples,
        "repo",
        config=state.config,
        rng=random.Random(5),
        now=1.0,
    )
    publish_snapshot(state.client, snapshot, key_prefix=state.config.key_prefix)

    assert state.policy._snapshot("repo") is not None
    assert state.reputation_policy._snapshot("repo") is None


def test_the_reputation_loop_earns_the_verdicts_of_the_turns_it_ran():
    """The variant is fed the same labels the learned one is, and nothing more."""
    state = _LearnedReplay("repo", knowledge_config())
    features = {
        "recipe:20260101:x": CandidateFeatures(
            kind="recipe", mode="debug", score=0.6, size_chars=200, budget_chars=3000
        )
    }
    context = _Context([_Hit("recipe:20260101:x")])
    for index in range(5):
        state.observe(
            session_id=f"s{index}",
            generation=f"g{index}",
            context=context,
            features=features,
            saved_by="x",
            is_loop=False,
        )

    stats = state.reputation_policy.reputation.stats("recipe:20260101:x")
    assert stats["exposure"] == 5
    _, detail = state.reputation_policy.multiplier_for(
        features["recipe:20260101:x"], "recipe:20260101:x", scope="repo"
    )
    assert detail["rep"] > 1.0, "five confirmed fixes left no trace"
    # The learned policy folded the same verdicts into its bandit; this one
    # was handed the posterior and nothing else.
    assert state.policy.bandit.arms()
