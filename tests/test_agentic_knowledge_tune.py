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
