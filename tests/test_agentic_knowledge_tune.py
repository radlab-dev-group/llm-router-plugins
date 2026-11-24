"""
The policy tune: a grid search that answers the question it was asked.

The command measures a policy on a dataset it was not tuned on, so what these
tests can pin is the honesty of the measurement: that a grid point is the
shipped policy with exactly the numbers the search moves and nothing else,
that the objective is the formula the operator named, that the precision floor
is a constraint and not a tie-breaker, and that the report says what it
measured against.  A tune that quietly also changed the character budget, or
that reported a delta against the grid instead of against the shipped policy,
would still produce a confident table — and that is the failure mode to test
for.
"""

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
    AgenticKnowledgeConfig,
    ModePolicy,
    SourcePolicy,
)
from llm_router_plugins.cli.plugins.utils.agentic_knowledge.tune import (
    _apply,
    _escalation_precision,
    _markdown,
    build_parser,
    tune,
)

DATASET = (
    pathlib.Path(__file__).resolve().parent / "data" / "agentic_knowledge_eval.json"
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.evaluation import (
    load_sessions,
)


def base_config() -> AgenticKnowledgeConfig:
    """The shipped policy, the one a tune is asked to improve."""
    config = AgenticKnowledgeConfig.from_file()
    config.validate()
    return config


def test_a_grid_point_moves_the_four_numbers_and_nothing_else():
    config = base_config()
    variant = _apply(
        config,
        min_similarity=0.42,
        strong_recipe_score=0.55,
        loop_threshold=5,
        weights={"recipe": 1.25},
    )
    assert variant.min_similarity == 0.42
    assert variant.strong_recipe_score == 0.55
    assert variant.loop_threshold == 5
    # The weight the search named moved, and the weight it did not name did
    # not: a point that also moved the character budget is a point whose
    # result cannot be attributed to the number the operator asked about.
    for name, policy in variant.modes.items():
        original = config.modes[name]
        assert policy.max_chars == original.max_chars
        for source, before in zip(policy.sources, original.sources):
            expected = (
                1.25
                if source.category == "recipe"
                else before.weight
            )
            assert source.weight == pytest.approx(expected)
    # And the original is untouched: a grid that writes through to the policy
    # it started from measures a different policy on every point.
    assert config.min_similarity != 0.42
    for name, policy in config.modes.items():
        assert all(
            source.weight == before.weight
            for source, before in zip(
                policy.sources, base_config().modes[name].sources
            )
        )


def test_the_objective_is_the_formula_the_operator_named():
    config = base_config()
    sessions = load_sessions(DATASET)
    result = tune(
        config=config,
        sessions=sessions,
        min_similarity=[config.min_similarity],
        strong_recipe_score=[config.strong_recipe_score],
        loop_threshold=[config.loop_threshold],
        weight_factors=[1.0],
        weight_categories=[],
        lam=0.001,
        precision_floor=0.8,
    )
    point = result["points"][0]
    assert point["objective"] == pytest.approx(
        point["recipe_hit_rate"] - 0.001 * point["injected_chars_per_turn"], abs=1e-5
    )
    # A λ of zero is "do not care what it costs", and the objective collapses
    # to the hit rate: the same run at a big λ must not report the same number.
    expensive = tune(
        config=config,
        sessions=sessions,
        min_similarity=[config.min_similarity],
        strong_recipe_score=[config.strong_recipe_score],
        loop_threshold=[config.loop_threshold],
        weight_factors=[1.0],
        weight_categories=[],
        lam=1.0,
        precision_floor=0.8,
    )
    assert expensive["points"][0]["objective"] < point["objective"]


def test_the_chosen_point_beats_the_others_on_the_objective():
    config = base_config()
    sessions = load_sessions(DATASET)
    result = tune(
        config=config,
        sessions=sessions,
        min_similarity=[0.1, 0.35],
        strong_recipe_score=[0.5, 0.9],
        loop_threshold=[2, 3],
        weight_factors=[1.0],
        weight_categories=[],
        lam=0.0001,
        precision_floor=0.8,
    )
    chosen = result["chosen"]
    eligible = [point for point in result["points"] if point["eligible"]]
    assert eligible
    best = max(point["objective"] for point in eligible)
    assert chosen["objective"] == pytest.approx(best)


def test_the_precision_floor_is_a_constraint_not_a_tie_breaker():
    """A point with a better objective that breaks the floor does not win."""
    config = base_config()
    sessions = load_sessions(DATASET)

    def precision_of(point):
        return point["escalation_precision"]

    # A floor above anything the dataset can produce: the search must say so
    # rather than hand back the point it liked best as if it had met the bar.
    impossible = tune(
        config=config,
        sessions=sessions,
        min_similarity=[0.35],
        strong_recipe_score=[0.6],
        loop_threshold=[3],
        weight_factors=[1.0],
        weight_categories=[],
        lam=0.0001,
        precision_floor=1.0001,
    )
    assert impossible["chosen"] is not None
    assert impossible["chosen"]["eligible"] is False
    assert impossible["note"]

    # The same search at a floor everything meets: no note, and every point is
    # eligible.
    met = tune(
        config=config,
        sessions=sessions,
        min_similarity=[0.35],
        strong_recipe_score=[0.6],
        loop_threshold=[3],
        weight_factors=[1.0],
        weight_categories=[],
        lam=0.0001,
        precision_floor=0.0,
    )
    assert met["note"] == ""
    assert all(point["eligible"] for point in met["points"])


def test_a_variant_that_escalates_on_nothing_reports_no_precision():
    """A constraint on escalations is a constraint on the ones made."""
    report = {"variants": {"read_escalate": {"escalations": 0}}, "loop_detection": {}}
    assert _escalation_precision(report) == 1.0
    made = {
        "variants": {
            "read_escalate": {
                "escalations": 4,
                "escalation_precision": 0.5,
            }
        },
        "loop_detection": {},
    }
    assert _escalation_precision(made) == 0.5


def test_the_baseline_is_the_shipped_policy_on_the_same_sessions():
    config = base_config()
    sessions = load_sessions(DATASET)
    result = tune(
        config=config,
        sessions=sessions,
        min_similarity=[0.2, 0.35],
        strong_recipe_score=[0.6],
        loop_threshold=[3],
        weight_factors=[1.0],
        weight_categories=[],
        lam=0.0001,
        precision_floor=0.8,
    )
    baseline = result["baseline"]
    assert baseline["index"] == -1
    assert baseline["min_similarity"] == config.min_similarity
    assert baseline["strong_recipe_score"] == config.strong_recipe_score
    assert baseline["loop_threshold"] == config.loop_threshold
    # The deltas are against that baseline, not against an average of the
    # grid: the number the operator has in production is the one worth
    # comparing to.
    chosen = result["chosen"]
    assert result["deltas"]["recipe_hit_rate"] == pytest.approx(
        chosen["recipe_hit_rate"] - baseline["recipe_hit_rate"], abs=1e-4
    )


def test_the_markdown_justifies_the_choice_with_the_numbers():
    config = base_config()
    sessions = load_sessions(DATASET)
    result = tune(
        config=config,
        sessions=sessions,
        min_similarity=[0.35],
        strong_recipe_score=[0.6],
        loop_threshold=[3],
        weight_factors=[1.0],
        weight_categories=[],
        lam=0.0001,
        precision_floor=0.8,
    )
    text = _markdown(result)
    assert "# Knowledge policy tune" in text
    assert "## Chosen point" in text
    assert "## Against the shipped policy" in text
    # The constraint is named, so a reviewer can see what was held fixed.
    assert "escalation precision" in text
    assert "Nothing is applied by this command" in text


def test_the_json_report_names_the_grid_and_the_winner(tmp_path):
    config = base_config()
    sessions = load_sessions(DATASET)
    out = tmp_path / "report.json"
    code = tune_main(
        [
            "--dataset",
            str(DATASET),
            "--min-similarity",
            "0.35",
            "--strong-recipe-score",
            "0.6",
            "--loop-threshold",
            "3",
            "--weight-factor",
            "1.0",
            "--output",
            str(out),
        ]
    )
    assert code == 0
    report = json.loads(out.read_text())
    assert report["grid"]["min_similarity"] == [0.35]
    assert report["chosen"]["min_similarity"] == 0.35
    assert len(report["points"]) == 1


def tune_main(argv):
    from llm_router_plugins.cli.plugins.utils.agentic_knowledge.tune import main

    return main(argv)


def test_the_parser_reads_a_grid_and_defaults_to_a_sane_one():
    args = build_parser().parse_args(["--dataset", "x.json"])
    assert args.min_similarity == [0.25, 0.35, 0.45]
    assert args.loop_threshold == [2, 3, 4]
    # One shared factor by default: a grid that moves every source by the same
    # factor re-measures the same ordering, and the exponential version costs
    # an hour to say what the linear one says in a minute.
    assert args.per_category_factors is False
    assert args.precision_floor == 0.8


def test_the_tune_never_writes_to_the_policy():
    """The command reports; the operator edits."""
    config_path = pathlib.Path(
        "llm_router_plugins/resources/knowledge/agentic_knowledge.json"
    )
    if not config_path.exists():
        pytest.skip("the bundled policy is not installed")
    before = config_path.read_bytes()
    tune_main(["--dataset", str(DATASET), "--min-similarity", "0.35"])
    assert config_path.read_bytes() == before


# ------------------------------------------------------ the split of the grid
def _tiny_tune(config, sessions, **overrides):
    args = dict(
        min_similarity=[config.min_similarity],
        strong_recipe_score=[config.strong_recipe_score],
        loop_threshold=[config.loop_threshold],
        weight_factors=[1.0],
        weight_categories=[],
        lam=0.001,
        precision_floor=0.8,
    )
    args.update(overrides)
    return tune(config=config, sessions=sessions, **args)


def test_the_holdout_is_never_selected_on():
    """The grid is chosen on the calibration and only judged on the holdout."""
    config = base_config()
    sessions = load_sessions(DATASET)
    names = [str(session.get("id") or "") for session in sessions]
    held = names[:1]

    result = _tiny_tune(config, sessions, holdout_ids=held)

    assert result["split"]["holdout"] == held
    assert result["split"]["holdout_sessions"] == 1
    assert result["split"]["calibration_sessions"] == len(sessions) - 1
    # Every point carries what it earned on the half it was not selected on,
    # and the point the selection used is the calibration number, not that
    # one: the objective is recomputed from the calibration fields.
    for point in result["points"]:
        assert point["holdout"] is not None
    assert result["chosen"]["objective"] == pytest.approx(
        result["chosen"]["recipe_hit_rate"]
        - 0.001 * result["chosen"]["injected_chars_per_turn"],
        abs=1e-5,
    )


def test_the_holdout_delta_is_read_against_the_shipped_policy_on_the_same_half():
    config = base_config()
    sessions = load_sessions(DATASET)
    held = [str(session.get("id") or "") for session in sessions][:1]

    result = _tiny_tune(config, sessions, holdout_ids=held)

    deltas = result["holdout_deltas"]
    assert deltas is not None
    # The calibration delta and the holdout delta are two different numbers:
    # the first was selected for, the second was not.
    assert set(deltas) == set(result["deltas"])


def test_no_holdout_reports_no_holdout_numbers():
    """A run that held nothing out says so instead of a column of zeros."""
    config = base_config()
    sessions = load_sessions(DATASET)

    result = _tiny_tune(config, sessions)

    assert result["split"]["holdout"] == []
    assert result["split"]["note"]
    assert result["holdout_deltas"] is None
    assert result["chosen"]["holdout"] is None


def test_a_holdout_fraction_never_eats_the_calibration():
    """A grid selected on nothing is not a grid, whatever the operator asked."""
    config = base_config()
    sessions = load_sessions(DATASET)

    result = _tiny_tune(config, sessions, holdout_fraction=1.0)

    assert 0 < result["split"]["holdout_sessions"] < len(sessions)
    assert result["split"]["calibration_sessions"] >= 1


def test_the_fraction_split_is_reproducible_from_the_seed():
    config = base_config()
    sessions = load_sessions(DATASET)

    first = _tiny_tune(config, sessions, holdout_fraction=0.4, seed=7)["split"]
    second = _tiny_tune(config, sessions, holdout_fraction=0.4, seed=7)["split"]

    # The same dataset and the same seed split the same way, whatever order
    # the sessions were loaded in: the split is a function of the names.
    assert first["holdout"] == second["holdout"]
    assert first["holdout"]
    assert first["calibration_sessions"] >= 1


def test_an_unknown_holdout_session_is_named_and_refused():
    config = base_config()
    sessions = load_sessions(DATASET)

    with pytest.raises(ValueError, match="no session"):
        _tiny_tune(config, sessions, holdout_ids=["session-that-was-never-recorded"])


def test_the_markdown_shows_the_half_the_selection_never_saw():
    config = base_config()
    sessions = load_sessions(DATASET)
    held = [str(session.get("id") or "") for session in sessions][:1]

    result = _tiny_tune(config, sessions, holdout_ids=held)
    text = _markdown(result)

    assert "holdout" in text.lower()
    assert "which the selection never saw" in text


# ------------------------------------------------------- the recall of loops
def test_the_recall_floor_refuses_a_point_that_escalates_on_nothing():
    """
    A precision floor met by doing nothing is not a constraint met.

    The dataset has two loops.  A loop threshold of 4 never reaches them: the
    point escalates on nothing, so it escalates on nothing wrong, and its
    precision reports as met — which is precisely the degenerate winner a
    floor on precision alone hands to the operator, chosen over points that
    caught the loops.  With a recall floor the same point is refused by name,
    and without one it is still eligible, because the default is the search
    the command shipped with.
    """
    config = base_config()
    sessions = load_sessions(DATASET)

    unguarded = _tiny_tune(
        config, sessions, loop_threshold=[4], recall_floor=0.0
    )
    assert unguarded["points"][0]["eligible"] is True
    # Nothing was escalated, so nothing was escalated wrongly: the precision
    # floor has nothing to refuse, and says so.
    assert unguarded["points"][0]["escalation_precision"] == 1.0
    assert unguarded["points"][0]["escalation_recall"] == 0.0

    guarded = _tiny_tune(
        config, sessions, loop_threshold=[4], recall_floor=0.5
    )
    point = guarded["points"][0]
    assert point["eligible"] is False
    assert "recall" in point["refused_reason"]
    assert guarded["note"]

    # The same point at a threshold that does reach the loops keeps its
    # eligibility: the floor refuses the misses, not the escalating.
    catching = _tiny_tune(config, sessions, loop_threshold=[3], recall_floor=0.5)
    assert catching["points"][0]["escalation_recall"] == 1.0
    assert catching["points"][0]["eligible"] is True
    assert catching["note"] == ""


def test_the_recall_floor_and_the_precision_floor_name_what_each_refused():
    """A point that breaks both floors says so, rather than one of them."""
    config = base_config()
    sessions = load_sessions(DATASET)

    result = _tiny_tune(
        config,
        sessions,
        loop_threshold=[4],
        precision_floor=1.0001,
        recall_floor=0.5,
    )
    reason = result["points"][0]["refused_reason"]
    assert "precision" in reason
    assert "recall" in reason


def test_a_dataset_with_no_loops_meets_the_recall_floor_by_definition():
    """
    Nothing to catch is not a recall of zero.

    The denominator of a recall belongs to the dataset.  When it is empty the
    floor is met — a point cannot be refused for missing loops that were
    never recorded — and the view says so rather than printing a zero that
    reads as a failure.
    """
    from llm_router_plugins.cli.plugins.utils.agentic_knowledge.tune import (
        _escalation_recall,
    )

    assert _escalation_recall({"variants": {"read_escalate": {}}}) == 1.0
    assert (
        _escalation_recall(
            {"variants": {"read_escalate": {"escalation_recall": 0.25}}}
        )
        == 0.25
    )

    config = base_config()
    sessions = [
        session
        for session in load_sessions(DATASET)
        if not any(turn.get("is_loop") for turn in (session.get("turns") or ()))
    ]
    assert sessions
    result = _tiny_tune(config, sessions, recall_floor=1.0)
    assert result["points"][0]["escalation_recall"] == 1.0
    assert result["points"][0]["eligible"] is True


def test_the_recall_travels_with_the_other_deltas_and_the_markdown():
    """The report and its prose both carry the number the floor was set at."""
    config = base_config()
    sessions = load_sessions(DATASET)

    result = _tiny_tune(config, sessions, loop_threshold=[3], recall_floor=0.5)

    assert "escalation_recall" in result["deltas"]
    for point in result["points"]:
        assert "escalation_recall" in point
    text = _markdown(result)
    assert "escalation recall" in text
    assert "0.5" in text

    # And the absence of a floor is said out loud, because a reader who
    # sees a recall number and no floor needs to know the number was free.
    unguarded = _markdown(_tiny_tune(config, sessions, loop_threshold=[3]))
    assert "no recall floor" in unguarded


def test_the_parser_defaults_to_no_recall_floor():
    """The default is the search the command shipped with."""
    parser = build_parser()
    args = parser.parse_args(["--dataset", "x.json"])
    assert args.recall_floor == 0.0
    assert parser.parse_args(
        ["--dataset", "x.json", "--recall-floor", "0.75"]
    ).recall_floor == 0.75


# ------------------------------------------------- the Bayesian search ------
def test_the_bayesian_objective_is_the_multi_criteria_formula_the_plan_named():
    """The search ranks on the constraint, not just on the hit rate.

    A point that cannot keep the precision floor pays a penalty that grows
    without bound as it misses it, so a very high hit rate cannot buy its way
    under the floor: the floor is a constraint the search was given, and a
    number that does not contain it is a number that does not see it.
    """
    from llm_router_plugins.cli.plugins.utils.agentic_knowledge.tune import (
        _bayesian_objective,
    )

    met = _bayesian_objective(0.5, 100.0, 0.001, 0.9, 0.8, 0.9)
    assert met == pytest.approx(0.5 + 0.3 * 0.9 - 0.001 * 100.0)
    # The same point at a missed floor pays the penalty, and a point that
    # misses it harder pays more: the penalty is the floor being a constraint.
    missed = _bayesian_objective(0.5, 100.0, 0.001, 0.9, 0.8, 0.5)
    worse = _bayesian_objective(0.5, 100.0, 0.001, 0.9, 0.8, 0.2)
    assert missed < met
    assert worse < missed


def test_the_bayesian_search_spends_the_budget_and_is_deterministic():
    """The same dataset, budget and seed measure the same points, in order.

    A search that cannot be re-run cannot be checked: the report says the
    search found a point, and the only reason that sentence means anything is
    that a different machine with the same inputs proposes the same points in
    the same order.  The budget is the number of replays, and the report
    names both it and how many points the search measured.
    """
    config = base_config()
    sessions = load_sessions(DATASET)
    first = tune(
        config=config,
        sessions=sessions,
        min_similarity=[0.2, 0.45],
        strong_recipe_score=[0.5, 0.7],
        loop_threshold=[2, 3],
        weight_factors=[0.8, 1.2],
        weight_categories=["recipe"],
        lam=0.0001,
        precision_floor=0.8,
        search="bayesian",
        evaluations=10,
        seed=11,
    )
    again = tune(
        config=config,
        sessions=sessions,
        min_similarity=[0.2, 0.45],
        strong_recipe_score=[0.5, 0.7],
        loop_threshold=[2, 3],
        weight_factors=[0.8, 1.2],
        weight_categories=["recipe"],
        lam=0.0001,
        precision_floor=0.8,
        search="bayesian",
        evaluations=10,
        seed=11,
    )
    assert first["search"]["mode"] == "bayesian"
    assert first["search"]["budget"] == 10
    assert len(first["points"]) == 10
    assert all(
        left == right
        for left, right in zip(
            [dict(point) for point in first["points"]],
            [dict(point) for point in again["points"]],
        )
    )
    # A different seed proposes a different walk of the same space.
    other = tune(
        config=config,
        sessions=sessions,
        min_similarity=[0.2, 0.45],
        strong_recipe_score=[0.5, 0.7],
        loop_threshold=[2, 3],
        weight_factors=[0.8, 1.2],
        weight_categories=["recipe"],
        lam=0.0001,
        precision_floor=0.8,
        search="bayesian",
        evaluations=10,
        seed=12,
    )
    assert [point["min_similarity"] for point in other["points"]] != [
        point["min_similarity"] for point in first["points"]
    ]


def test_the_bayesian_search_keeps_the_precision_floor_a_constraint():
    """The search proposes under the floor, and the report still refuses it."""
    config = base_config()
    sessions = load_sessions(DATASET)
    result = tune(
        config=config,
        sessions=sessions,
        min_similarity=[0.2, 0.45],
        strong_recipe_score=[0.5, 0.7],
        loop_threshold=[2, 3],
        weight_factors=[0.8, 1.2],
        weight_categories=["recipe"],
        lam=0.0001,
        precision_floor=1.0001,
        search="bayesian",
        evaluations=8,
        seed=11,
    )
    # The floor is above anything the dataset can produce: every point is
    # refused, and the report says so rather than handing back the point the
    # search liked best as if it had met the bar.
    assert result["chosen"] is not None
    assert result["chosen"]["eligible"] is False
    assert result["note"]


def test_the_write_policy_flag_writes_a_proposal_the_layer_never_reads(tmp_path):
    """The tune's answer is a diff against the evidence, not a policy edit.

    The file is named ``.tuned.json`` so it is never the one a deployment
    loads by accident, it carries the numbers it claims so the merge into the
    real policy is reviewable against the evidence, and it is written only
    when asked: a tool that edits the policy it measured against has answered
    its own question.
    """
    config = base_config()
    sessions = load_sessions(DATASET)
    out = tmp_path / "report.json"
    code = tune_main(
        [
            "--dataset",
            str(DATASET),
            "--output",
            str(out),
            "--search",
            "bayesian",
            "--evaluations",
            "6",
            "--write-policy",
        ]
    )
    assert code == 0
    proposal = tmp_path / "agentic_knowledge.tuned.json"
    assert proposal.exists()
    written = json.loads(proposal.read_text())
    assert "Not loaded by the layer" in written["_comment"]
    # The proposal names the point it is, and the numbers it claims.
    assert written["read"]["min_similarity"] is not None
    assert written["escalation"]["loop_threshold"] is not None
    assert written["earned"]["recipe_hit_rate"] is not None
    # …and the shipped policy is untouched by the command that measured it.
    assert not (tmp_path / "agentic_knowledge.json").exists()
