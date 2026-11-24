"""
Grid search over the knowledge policy's thresholds, on a replayed dataset.

The policy ships with numbers the operator chose by hand: how much of a query
a document must cover to be retrieved at all, how strong a recipe must be to
count as known, how many repetitions of a failure are a loop, and what a
character of each source is worth.  This command asks the replay what those
numbers earn, over a grid the operator names, and reports the point that
maximizes

    recipe_hit_rate − λ · injected_chars_per_turn

under the constraint that escalation precision does not fall below the floor
the operator sets.  The report is a justification, not a decision: the JSON
names the winner and the deltas against the shipped policy on the same
sessions, and the markdown says why, with the numbers, so a pull request can
quote it.  Nothing is written to the policy: the file is the operator's to
change, and a tune that edits the policy it measured against has already
answered its own question.

The grid is deliberately small and explicit.  A search that can find a better
policy on the dataset it was searched on has found a policy that fits that
dataset, so the selection is made on the calibration half and the holdout
half is measured, not selected on: the point is picked blind to the sessions
it is then judged on, and a delta claimed on the holdout is read against the
delta the shipped policy claims there.  With no holdout named the whole
dataset is the calibration and the report says so.
"""

import argparse
import hashlib
import itertools
import json
import math
import pathlib
import random
import sys

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
    AgenticKnowledgeConfig,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.evaluation import (
    VARIANT_READ,
    evaluate,
    load_sessions,
)

__all__ = ["build_parser", "main", "tune"]


@dataclass(frozen=True)
class _Point:
    """One grid point and what it earned on the replay."""

    index: int
    min_similarity: float
    strong_recipe_score: float
    loop_threshold: int
    weights: Dict[str, float]
    objective: float
    recipe_hit_rate: float
    injected_chars_per_turn: float
    char_efficiency: float
    escalation_precision: float
    escalation_recall: float
    eligible: bool
    refused_reason: str
    #: The same point measured on the sessions it was not selected on, each
    #: ``None`` when the run had no holdout: an empty number would read as
    #: "it earned zero there", which is a different claim than "nobody asked".
    holdout_objective: Optional[float] = None
    holdout_recipe_hit_rate: Optional[float] = None
    holdout_injected_chars_per_turn: Optional[float] = None
    holdout_char_efficiency: Optional[float] = None
    holdout_escalation_precision: Optional[float] = None
    holdout_escalation_recall: Optional[float] = None


def _objective(
    hit_rate: float, chars_per_turn: float, lam: float
) -> float:
    """What the operator asked to maximize, in the plan's own terms."""
    return round(hit_rate - lam * chars_per_turn, 6)


def _bayesian_objective(
    hit_rate: float,
    chars_per_turn: float,
    lam: float,
    escalation_recall: float,
    precision_floor: float,
    escalation_precision: float,
) -> float:
    """
    The multi-criteria objective the Bayesian search climbs.

    The plan's form: the hit rate the point earns, plus a share of the
    escalation recall it holds, minus the price of the characters it
    injects, minus a penalty that is zero when the precision floor is met
    and grows without bound as it is missed.  The penalty is what makes the
    floor a *constraint* rather than a hope: a point that cannot keep the
    floor is not a point that lost by a little, it is a point that must not
    be selected, and a bounded penalty would let a very high hit rate buy its
    way under the floor.
    """
    penalty = max(0.0, precision_floor - escalation_precision)
    return round(
        hit_rate + 0.3 * escalation_recall - lam * chars_per_turn - 1.5 * penalty,
        6,
    )


def _escalation_precision(report: Dict[str, Any]) -> float:
    """
    The share of the escalations the replay would make that a loop was.

    The variant's escalations are the denominator; the numerator is the ones
    the ground truth marks as loops, which is exactly what the replay counts
    on the variant.  A variant that escalates on nothing reports no precision,
    and the floor treats that as met: a constraint on escalations is a
    constraint on the ones made, not on the ones not made, and reporting the
    empty case as perfect would let a point that escalates on nothing read as
    one that escalates flawlessly.
    """
    escalate = report["variants"].get("read_escalate") or {}
    precision = escalate.get("escalation_precision")
    if precision is None:
        return 1.0
    return float(precision)


def _escalation_recall(report: Dict[str, Any]) -> float:
    """
    The share of the loops that were there to catch the variant caught.

    The denominator is the turns the ground truth marks as loops — a property
    of the dataset, not of the point — and the numerator is the ones the
    variant escalated on.  It exists because a precision floor on its own is
    satisfiable by doing nothing: a point whose threshold is high enough never
    escalates, never escalates wrongly, and reads as flawless on the one
    number the search was constrained by.  Recall is the half of the
    confusion matrix the floor ignored, and with a floor under it a point can
    no longer meet the constraint by declining to act.

    A dataset with no loops reports no recall, and the floor reads that as
    met: there was nothing to catch, and a number invented for it would be a
    claim about a policy that had no opportunity to be wrong.
    """
    escalate = report["variants"].get("read_escalate") or {}
    recall = escalate.get("escalation_recall")
    if recall is None:
        return 1.0
    return float(recall)


def _apply(
    config: AgenticKnowledgeConfig,
    min_similarity: float,
    strong_recipe_score: float,
    loop_threshold: int,
    weights: Dict[str, float],
) -> AgenticKnowledgeConfig:
    """
    The policy a grid point measures under.

    Built by copying the shipped policy and moving the four numbers the search
    owns, so a grid point that moves nothing else is a policy that differs
    from the one in force by exactly the four numbers the operator is asking
    about.  The copy is deep for the one nested thing the search touches — the
    mode's sources — because a frozen dataclass whose sources are shared with
    the original would let one point's weight leak into the next.
    """
    modes: Dict[str, Any] = {}
    for name, policy in config.modes.items():
        sources = tuple(
            replace(source, weight=float(weights.get(source.category, source.weight)))
            for source in policy.sources
        )
        modes[name] = replace(policy, sources=sources)
    return replace(
        config,
        min_similarity=float(min_similarity),
        strong_recipe_score=float(strong_recipe_score),
        loop_threshold=int(loop_threshold),
        modes=modes,
    )


def _holdout_names(
    sessions: Sequence[Dict[str, Any]],
    holdout_fraction: float = 0.0,
    holdout_ids: Sequence[str] = (),
    seed: int = 11,
) -> List[str]:
    """
    The sessions the grid is not allowed to select on.

    Named ids win, because the operator's holdout is a thing they already keep
    and the tool has no right to second-guess it.  A fraction is honoured by a
    hash of ``seed`` and the session id, so the same dataset and the same seed
    split the same way in two runs and in two checkouts, without the split
    depending on the order the sessions were loaded in.

    The split never takes the whole dataset: a grid selected on nothing is not
    a grid, and a point chosen by a coin cannot be reported as a measurement.
    """
    names = [str(session.get("id") or "") for session in sessions]
    known = {name for name in names}
    chosen = {str(name) for name in holdout_ids}
    unknown = sorted(chosen - known)
    if unknown:
        raise ValueError(
            "the dataset has no session(s) named: " + ", ".join(unknown)
        )
    if holdout_fraction > 0.0 and len(known) > 1:
        ranked = sorted(
            known,
            key=lambda name: hashlib.sha256(
                f"{int(seed)}:{name}".encode("utf-8")
            ).hexdigest(),
        )
        take = min(
            max(int(math.ceil(float(holdout_fraction) * len(ranked))), 1),
            len(ranked) - 1,
        )
        chosen |= set(ranked[:take])
    return sorted(chosen)


def _measure(
    config: AgenticKnowledgeConfig,
    sessions: Sequence[Dict[str, Any]],
    lam: float,
    precision_floor: float,
    recall_floor: float = 0.0,
) -> Dict[str, Any]:
    """
    One configuration on one set of sessions: the objective and the gate.

    The calibration and the holdout are measured by this same function, so a
    number on either side is the number the other side would have read: the
    split changes what a point is allowed to be selected on, not what it is
    judged by.

    Two floors gate the point, not one: the precision floor says "when the
    point escalates, it escalates on a loop", and the recall floor says "when
    a loop happens, the point escalates on it".  A point that clears the first
    by never escalating fails the second, and that is the failure mode the
    recall floor exists to catch — a precision floor alone would let a point
    that escalates on nothing read as one that escalates flawlessly, because
    the empty case is reported as perfect rather than as absent.
    """
    if not sessions:
        return {
            "objective": 0.0,
            "recipe_hit_rate": 0.0,
            "injected_chars_per_turn": 0.0,
            "char_efficiency": 0.0,
            "escalation_precision": None,
            "escalation_recall": None,
            "eligible": False,
            "refused_reason": "no sessions to measure on",
            "turns": 0,
        }
    report = evaluate(config, sessions)
    read = report["variants"].get(VARIANT_READ) or {}
    hit_rate = float(read.get("recipe_hit_rate", 0.0))
    chars = float(read.get("injected_chars_per_turn", 0.0))
    precision = _escalation_precision(report)
    recall = _escalation_recall(report)
    eligible = precision >= precision_floor and recall >= recall_floor
    reasons = []
    if precision < precision_floor:
        reasons.append("escalation precision below floor")
    if recall < recall_floor:
        reasons.append("escalation recall below floor")
    return {
        "objective": _objective(hit_rate, chars, lam),
        "recipe_hit_rate": hit_rate,
        "injected_chars_per_turn": chars,
        "char_efficiency": float(read.get("char_efficiency", 0.0)),
        "escalation_precision": precision,
        "escalation_recall": recall,
        "eligible": eligible,
        "refused_reason": "; ".join(reasons),
        "turns": sum(len(session.get("turns") or ()) for session in sessions),
    }


def tune(
    config: AgenticKnowledgeConfig,
    sessions: Sequence[Dict[str, Any]],
    min_similarity: Sequence[float],
    strong_recipe_score: Sequence[float],
    loop_threshold: Sequence[int],
    weight_factors: Sequence[float],
    weight_categories: Sequence[str],
    lam: float,
    precision_floor: float,
    recall_floor: float = 0.0,
    shared_factors: bool = True,
    holdout_fraction: float = 0.0,
    holdout_ids: Sequence[str] = (),
    seed: int = 11,
    search: str = "grid",
    evaluations: int = 40,
) -> Dict[str, Any]:
    """
    Walk the grid, replay every point, and report what the walk found.

    The walk is over the Cartesian product of the ranges the operator named,
    in the order they were named, which is what makes the index of a point in
    the report reproducible: a report that cannot say "point 17 is
    (0.35, 0.55, 3, 1.2)" cannot be re-run, and a tune that cannot be re-run
    is a guess with a table attached.

    The selection is made on the calibration half and the holdout half is only
    measured, which is the plan's criterion for the phase: a point chosen on
    the sessions it is then judged on has been fitted to them, and the delta
    it claims is the shape of the dataset rather than of the policy.  The
    shipped policy is measured on both halves, so a holdout delta is read
    against the delta the policy it replaces claims on the same sessions.
    """
    holdout = _holdout_names(
        sessions,
        holdout_fraction=holdout_fraction,
        holdout_ids=holdout_ids,
        seed=seed,
    )
    calibration = [
        session for session in sessions
        if str(session.get("id") or "") not in holdout
    ]
    held = [
        session for session in sessions
        if str(session.get("id") or "") in holdout
    ]
    shipped_weights = {
        source.category: float(source.weight)
        for policy in config.modes.values()
        for source in policy.sources
    }

    def measure(
        index: int, ms: float, srs: float, lt: int, factors: Dict[str, float]
    ) -> _Point:
        """One point of the space, replayed on both halves of the split."""
        weights = {
            category: round(shipped_weights.get(category, 1.0) * factor, 4)
            for category, factor in factors.items()
        }
        variant = _apply(config, ms, srs, lt, weights)
        earned = _measure(variant, calibration, lam, precision_floor, recall_floor)
        scored = (
            _measure(variant, held, lam, precision_floor, recall_floor)
            if held
            else None
        )
        return _Point(
            index=index,
            min_similarity=ms,
            strong_recipe_score=srs,
            loop_threshold=lt,
            weights=weights,
            objective=earned["objective"],
            recipe_hit_rate=earned["recipe_hit_rate"],
            injected_chars_per_turn=earned["injected_chars_per_turn"],
            char_efficiency=earned["char_efficiency"],
            escalation_precision=earned["escalation_precision"],
            escalation_recall=earned["escalation_recall"],
            eligible=earned["eligible"],
            refused_reason=earned["refused_reason"],
            holdout_objective=(
                scored["objective"] if scored is not None else None
            ),
            holdout_recipe_hit_rate=(
                scored["recipe_hit_rate"] if scored is not None else None
            ),
            holdout_injected_chars_per_turn=(
                scored["injected_chars_per_turn"]
                if scored is not None
                else None
            ),
            holdout_char_efficiency=(
                scored["char_efficiency"] if scored is not None else None
            ),
            holdout_escalation_precision=(
                scored["escalation_precision"]
                if scored is not None
                else None
            ),
            holdout_escalation_recall=(
                scored["escalation_recall"] if scored is not None else None
            ),
        )

    points: List[_Point] = []
    if str(search).lower() == "bayesian":
        points = _bayesian_search(
            measure,
            dimensions=(
                _Dimension(
                    "min_similarity", min(min_similarity), max(min_similarity)
                ),
                _Dimension(
                    "strong_recipe_score",
                    min(strong_recipe_score),
                    max(strong_recipe_score),
                ),
                _Dimension(
                    "loop_threshold",
                    min(loop_threshold),
                    max(loop_threshold),
                    is_int=True,
                ),
                _Dimension("weight_factor", min(weight_factors), max(weight_factors)),
            ),
            categories=tuple(weight_categories),
            budget=max(1, int(evaluations)),
            seed=seed,
            lam=lam,
            precision_floor=precision_floor,
        )
    else:
        for index, (ms, srs, lt, factors) in enumerate(
            itertools.product(
                min_similarity,
                strong_recipe_score,
                loop_threshold,
                _factor_combos(weight_categories, weight_factors, shared_factors),
            )
        ):
            points.append(measure(index, ms, srs, lt, factors))

    eligible = [point for point in points if point.eligible]
    chosen = None
    if eligible:
        chosen = max(
            eligible, key=lambda p: (p.objective, -p.injected_chars_per_turn)
        )
    elif points:
        chosen = max(points, key=lambda p: p.objective)

    # The shipped policy, measured on the same split as the grid, so the
    # deltas are against the numbers the operator already has in production
    # rather than against an average the grid happened to produce.
    baseline_earned = _measure(
        config, calibration, lam, precision_floor, recall_floor
    )
    baseline_scored = (
        _measure(config, held, lam, precision_floor, recall_floor) if held else None
    )
    baseline = _Point(
        index=-1,
        min_similarity=config.min_similarity,
        strong_recipe_score=config.strong_recipe_score,
        loop_threshold=config.loop_threshold,
        weights=dict(shipped_weights),
        objective=baseline_earned["objective"],
        recipe_hit_rate=baseline_earned["recipe_hit_rate"],
        injected_chars_per_turn=baseline_earned["injected_chars_per_turn"],
        char_efficiency=baseline_earned["char_efficiency"],
        escalation_precision=baseline_earned["escalation_precision"],
        escalation_recall=baseline_earned["escalation_recall"],
        eligible=baseline_earned["eligible"],
        refused_reason=baseline_earned["refused_reason"],
        holdout_objective=(
            baseline_scored["objective"] if baseline_scored is not None else None
        ),
        holdout_recipe_hit_rate=(
            baseline_scored["recipe_hit_rate"]
            if baseline_scored is not None
            else None
        ),
        holdout_injected_chars_per_turn=(
            baseline_scored["injected_chars_per_turn"]
            if baseline_scored is not None
            else None
        ),
        holdout_char_efficiency=(
            baseline_scored["char_efficiency"]
            if baseline_scored is not None
            else None
        ),
        holdout_escalation_precision=(
            baseline_scored["escalation_precision"]
            if baseline_scored is not None
            else None
        ),
        holdout_escalation_recall=(
            baseline_scored["escalation_recall"]
            if baseline_scored is not None
            else None
        ),
    )

    return {
        "sessions": len(sessions),
        "turns": sum(len(session.get("turns") or ()) for session in sessions),
        "split": {
            "holdout": list(holdout),
            "calibration_sessions": len(calibration),
            "holdout_sessions": len(held),
            "seed": int(seed),
            "note": (
                ""
                if holdout
                else "no holdout named: every session was calibration, and the "
                     "chosen point was selected on the sessions it reports"
            ),
        },
        "search": {
            "mode": str(search),
            "evaluations": len(points),
            "budget": (
                max(1, int(evaluations))
                if str(search).lower() == "bayesian"
                else len(points)
            ),
        },
        "grid": {
            "min_similarity": list(min_similarity),
            "strong_recipe_score": list(strong_recipe_score),
            "loop_threshold": list(loop_threshold),
            "weight_factors": list(weight_factors),
            "weight_categories": list(weight_categories),
            "shared_factors": bool(shared_factors),
            "lambda": lam,
            "precision_floor": precision_floor,
            "recall_floor": recall_floor,
        },
        "points": [_point_report(point) for point in points],
        "chosen": _point_report(chosen) if chosen is not None else None,
        "baseline": _point_report(baseline),
        "deltas": _deltas(chosen, baseline) if chosen is not None else None,
        "holdout_deltas": (
            _holdout_deltas(chosen, baseline) if chosen is not None else None
        ),
        "note": (
            "no grid point met the precision floor; the best available point "
            "is reported with the floor it failed"
            if not eligible
            else ""
        ),
    }


@dataclass(frozen=True)
class _Dimension:
    """One axis of the search space, with the bounds it is sampled in."""

    name: str
    low: float
    high: float
    is_int: bool = False

    def cast(self, value: float) -> Any:
        """Snap a sample to the dimension's kind and bound it."""
        if self.is_int:
            return int(round(min(max(value, self.low), self.high)))
        return round(min(max(value, self.low), self.high), 4)

    def width(self) -> float:
        """The span a sample can move in; a zero-width axis is inert."""
        return max(0.0, self.high - self.low)


def _parzen_log_density(
    sample: Sequence[float],
    observations: Sequence[Sequence[float]],
    dimensions: Sequence[_Dimension],
) -> float:
    """
    ``log`` of the Parzen-window density of *sample* under *observations*.

    The estimator TPE is named for: the density of the points already seen is
    the mixture of Gaussians centred on each of them, one dimension at a time,
    and the ratio of the density under the good points to the density under
    the rest is the quantity that says "this region has been good and has not
    been explored enough".  A product over dimensions is the assumption that
    the axes are independent, which they are not exactly — the hit rate does
    not care about ``min_similarity`` in the same way at every
    ``strong_recipe_score`` — and the assumption is the one that costs nothing
    to hold and still puts the next sample where the evidence points.

    The bandwidth is a share of the observed span, never zero: a Gaussian on
    a dimension the search has only sampled once has no spread to speak of,
    and a zero-width kernel proposes the same point twice and calls it
    convergence.
    """
    if not observations:
        return 0.0
    total = 0.0
    for position, dimension in enumerate(dimensions):
        values = [float(point[position]) for point in observations]
        span = max(values) - min(values) if len(values) > 1 else dimension.width()
        bandwidth = max(span / max(1.0, float(len(values) - 1)) * 0.75, dimension.width() * 0.02, 1e-6)
        density = 0.0
        for value in values:
            distance = (float(sample[position]) - value) / bandwidth
            density += math.exp(-0.5 * distance * distance)
        total += math.log(max(density / float(len(values)), 1e-12))
    return total


def _tpe_propose(
    dimensions: Sequence[_Dimension],
    seen: Sequence[Tuple[Sequence[float], float]],
    rng: random.Random,
    candidates: int = 24,
) -> List[float]:
    """
    The next point to measure: the candidate the ratio of good to rest likes.

    Half the candidates are drawn from the density of the points that earned
    most — the search exploiting what it has learned — and half from the
    uniform prior, which is the search refusing to believe yet that it has
    learned anything.  Both halves are then ranked by the ratio of the two
    densities, which is TPE's acquisition: a point that looks good and has
    not been looked at is worth a measurement, and a point that looks good
    because it *is* a point already measured is not.

    The uniform half is not a courtesy.  A search that only proposes near its
    best observations stops at the first ridge it climbs, and a replay whose
    landscape has two ridges — one that reads well on the calibration
    sessions and one that reads well everywhere — needs the samples that fall
    off the ridge to know there is another one.
    """
    del candidates  # the split below fixes the count; kept for call compatibility
    if not seen:
        return [
            dimension.cast(rng.uniform(dimension.low, dimension.high))
            for dimension in dimensions
        ]
    ranked = sorted(seen, key=lambda item: item[1], reverse=True)
    top = max(1, math.ceil(len(ranked) / 4))
    good = [list(values) for values, _ in ranked[:top]]
    rest = [list(values) for values, _ in ranked[top:]] or good

    def sample(from_good: bool) -> List[float]:
        origin = rng.choice(good if from_good else rest)
        point = []
        for position, dimension in enumerate(dimensions):
            spread = dimension.width() * 0.15
            point.append(dimension.cast(origin[position] + rng.gauss(0.0, spread)))
        return point

    best_point: Optional[List[float]] = None
    best_score = None
    for index in range(_TPE_CANDIDATES):
        candidate = sample(index % 2 == 0)
        score = _parzen_log_density(candidate, good, dimensions) - _parzen_log_density(
            candidate, rest, dimensions
        )
        if best_score is None or score > best_score:
            best_point, best_score = candidate, score
    return best_point or [
        dimension.cast(rng.uniform(dimension.low, dimension.high))
        for dimension in dimensions
    ]


#: Candidates the TPE proposes per measurement.  Each candidate costs only a
#: density evaluation; the point they win is what costs a replay.
_TPE_CANDIDATES = 24

#: Measurements spent before the model is trusted, whatever it claims.  A
#: density estimated from two observations is a story about two points, and a
#: search that believes it has stopped exploring on the strength of a story
#: has converged on the first ridge it stood on.
_TPE_EXPLORE = 8


def _bayesian_search(
    measure: Any,
    dimensions: Sequence[_Dimension],
    categories: Sequence[str],
    budget: int,
    seed: int,
    lam: float,
    precision_floor: float,
) -> List[_Point]:
    """
    Spend a replay budget where the evidence says to spend it.

    A grid walks every corner of the space the operator named, which is right
    when the space is small enough to walk and wrong when it is not: the
    corners of a five-axis grid multiply past the number of replays an
    operator will wait for, and the ones they do wait for are the corners,
    not the interior where the optimum of a threshold policy usually is.
    This walks the same space on a budget: the first :data:`_TPE_EXPLORE`
    measurements are uniform, and each measurement after that is proposed by
    the density of the measurements that earned most against the density of
    the ones that did not.

    The ranking the proposal learns from is the multi-criteria objective, not
    the plain one: a search that ranks by hit rate alone will happily propose
    a point under the precision floor, because the floor is invisible to a
    number that does not contain it, and the constraint the operator wrote is
    the part of their intention the search is least allowed to lose.

    Deterministic under *seed*: the same dataset, the same budget and the same
    seed propose the same points in the same order, which is the only reason a
    report saying "the search found this" can be checked by anyone but the
    machine that found it.
    """
    rng = random.Random(seed)
    seen: List[Tuple[List[float], float]] = []
    points: List[_Point] = []

    def earned(point: _Point) -> float:
        return _bayesian_objective(
            point.recipe_hit_rate,
            point.injected_chars_per_turn,
            lam,
            point.escalation_recall,
            precision_floor,
            point.escalation_precision,
        )

    def evaluate(values: Sequence[float], index: int) -> _Point:
        by_name = {
            dimension.name: value
            for dimension, value in zip(dimensions, values)
        }
        factor = float(by_name.get("weight_factor", 1.0))
        point = measure(
            index,
            float(by_name.get("min_similarity", 0.35)),
            float(by_name.get("strong_recipe_score", 0.6)),
            int(by_name.get("loop_threshold", 3)),
            {str(category): factor for category in categories},
        )
        points.append(point)
        seen.append((list(values), earned(point)))
        return point

    def uniform() -> List[float]:
        return [
            dimension.cast(rng.uniform(dimension.low, dimension.high))
            for dimension in dimensions
        ]

    for index in range(max(1, int(budget))):
        if index < _TPE_EXPLORE or index * 5 <= budget:
            values = uniform()
        else:
            values = _tpe_propose(dimensions, seen, rng)
            # A proposal that repeats a measured point spends the budget on
            # nothing.  A few resamples; if the model is still proposing a
            # point it has already seen, the space it is modelling has run
            # out, and a uniform draw is the honest next measurement.
            attempts = 0
            while any(
                _same_point(values, known) for known, _ in seen
            ) and attempts < 5:
                values = _tpe_propose(dimensions, seen, rng)
                attempts += 1
            if any(_same_point(values, known) for known, _ in seen):
                values = uniform()
        evaluate(values, index)
    return points


def _same_point(left: Sequence[float], right: Sequence[float]) -> bool:
    """Whether two samples are the same point of the space."""
    if len(left) != len(right):
        return False
    return all(abs(float(a) - float(b)) < 1e-9 for a, b in zip(left, right))


def _factor_combos(
    categories: Sequence[str], factors: Sequence[float], shared: bool = True
) -> List[Dict[str, float]]:
    """
    One factor per category, over the named range.

    The factors are the search's, not the weights': a factor of 1.0 is the
    shipped weight, and a grid that moves a category by 20% is asking "what
    does 20% more of this source earn", which is the question the operator
    can check against the policy they have, and it is the only question a
    small grid is honest about.

    With *shared* the factor moves every named category at once, so the grid
    stays the product of the ranges; without it each category takes its own
    factor, and the grid grows to the factor range to the power of the number
    of categories.  A ranking is decided by the relative weights of the
    sources, so a shared factor across all of them measures the same ordering
    — the mode that wants recipes and dead ends both 20% cheaper is a
    different policy, but one that moves one source by 20% is, and that is
    what a shared sweep measures honestly.
    """
    if not categories:
        return [{}]
    if shared:
        return [{category: factor for category in categories} for factor in factors]
    return [
        dict(zip(categories, combo))
        for combo in itertools.product(factors, repeat=len(categories))
    ]


def _point_report(point: _Point) -> Dict[str, Any]:
    return {
        "index": point.index,
        "min_similarity": point.min_similarity,
        "strong_recipe_score": point.strong_recipe_score,
        "loop_threshold": point.loop_threshold,
        "weights": point.weights,
        "objective": point.objective,
        "recipe_hit_rate": point.recipe_hit_rate,
        "injected_chars_per_turn": point.injected_chars_per_turn,
        "char_efficiency": point.char_efficiency,
        "escalation_precision": point.escalation_precision,
        "escalation_recall": point.escalation_recall,
        "eligible": point.eligible,
        "refused_reason": point.refused_reason,
        "holdout": {
            "objective": point.holdout_objective,
            "recipe_hit_rate": point.holdout_recipe_hit_rate,
            "injected_chars_per_turn": point.holdout_injected_chars_per_turn,
            "char_efficiency": point.holdout_char_efficiency,
            "escalation_precision": point.holdout_escalation_precision,
            "escalation_recall": point.holdout_escalation_recall,
        }
        if point.holdout_objective is not None
        else None,
    }


def _deltas(chosen: _Point, baseline: _Point) -> Dict[str, Any]:
    """
    What the chosen point moves against the shipped policy, on the same data.

    The deltas are the report's point, because the grid's numbers are only
    comparable to the policy the operator is about to replace, and a delta
    against an average of the grid is a number nobody in production has.
    """
    return {
        "objective": round(chosen.objective - baseline.objective, 6),
        "recipe_hit_rate": round(
            chosen.recipe_hit_rate - baseline.recipe_hit_rate, 4
        ),
        "injected_chars_per_turn": round(
            chosen.injected_chars_per_turn - baseline.injected_chars_per_turn, 2
        ),
        "char_efficiency": round(
            chosen.char_efficiency - baseline.char_efficiency, 4
        ),
        "escalation_precision": round(
            chosen.escalation_precision - baseline.escalation_precision, 4
        ),
        "escalation_recall": round(
            chosen.escalation_recall - baseline.escalation_recall, 4
        ),
    }


def _holdout_deltas(chosen: _Point, baseline: _Point) -> Optional[Dict[str, Any]]:
    """
    The same deltas on the sessions the point was not selected on.

    ``None`` when the run had no holdout, rather than a table of zeros: a
    reader who sees ``0.0`` in a holdout column reads a measured absence, and
    what they should read is that nobody measured.
    """
    if chosen.holdout_objective is None or baseline.holdout_objective is None:
        return None
    return {
        "objective": round(
            chosen.holdout_objective - baseline.holdout_objective, 6
        ),
        "recipe_hit_rate": round(
            chosen.holdout_recipe_hit_rate - baseline.holdout_recipe_hit_rate, 4
        ),
        "injected_chars_per_turn": round(
            chosen.holdout_injected_chars_per_turn
            - baseline.holdout_injected_chars_per_turn,
            2,
        ),
        "char_efficiency": round(
            chosen.holdout_char_efficiency - baseline.holdout_char_efficiency, 4
        ),
        "escalation_precision": round(
            chosen.holdout_escalation_precision
            - baseline.holdout_escalation_precision,
            4,
        ),
        "escalation_recall": round(
            chosen.holdout_escalation_recall - baseline.holdout_escalation_recall, 4
        ),
    }


def _markdown(result: Dict[str, Any]) -> str:
    """
    The justification a pull request quotes, in the shape a reviewer reads.

    The numbers, the constraint, and the delta against the shipped policy on
    the same sessions — in that order, because the question a reviewer asks
    is "does it earn more, cost less, and break the escalations less", and
    the order of the answer is the order of the question.
    """
    grid = result["grid"]
    chosen = result.get("chosen") or {}
    baseline = result.get("baseline") or {}
    deltas = result.get("deltas") or {}
    factor_shape = (
        f"× {len(grid['weight_factors'])} shared"
        if grid.get("shared_factors", True)
        else f"^ {len(grid['weight_categories'])} per category"
    )
    lines = [
        "# Knowledge policy tune",
        "",
        f"Dataset: {result['sessions']} sessions, {result['turns']} turns.",
        f"Grid: {len(result['points'])} points "
        f"(`min_similarity` × {len(grid['min_similarity'])}, "
        f"`strong_recipe_score` × {len(grid['strong_recipe_score'])}, "
        f"`loop_threshold` × {len(grid['loop_threshold'])}, "
        f"weight factors {factor_shape}).",
        f"Objective: `recipe_hit_rate − {grid['lambda']} · injected_chars_per_turn` "
        f"under `escalation precision ≥ {grid['precision_floor']}`"
        + (
            f" and `escalation recall ≥ {grid['recall_floor']}`."
            if grid.get("recall_floor")
            else " (no recall floor: a point may meet the precision floor by "
                 "escalating on nothing)."
        ),
        "",
        "## Chosen point",
        "",
        f"- `min_similarity` = {chosen.get('min_similarity')}",
        f"- `strong_recipe_score` = {chosen.get('strong_recipe_score')}",
        f"- `loop_threshold` = {chosen.get('loop_threshold')}",
        f"- weights = {json.dumps(chosen.get('weights') or {}, sort_keys=True)}",
        f"- objective = {chosen.get('objective')}",
        "",
        "## Against the shipped policy, on the same sessions",
        "",
        "| metric | shipped | chosen | Δ |",
        "|---|---|---|---|",
        f"| recipe_hit_rate | {baseline.get('recipe_hit_rate')} | "
        f"{chosen.get('recipe_hit_rate')} | {deltas.get('recipe_hit_rate')} |",
        f"| injected_chars_per_turn | {baseline.get('injected_chars_per_turn')} | "
        f"{chosen.get('injected_chars_per_turn')} | "
        f"{deltas.get('injected_chars_per_turn')} |",
        f"| char_efficiency | {baseline.get('char_efficiency')} | "
        f"{chosen.get('char_efficiency')} | {deltas.get('char_efficiency')} |",
        f"| escalation precision | {baseline.get('escalation_precision')} | "
        f"{chosen.get('escalation_precision')} | "
        f"{deltas.get('escalation_precision')} |",
        f"| escalation recall | {baseline.get('escalation_recall')} | "
        f"{chosen.get('escalation_recall')} | {deltas.get('escalation_recall')} |",
        f"| objective | {baseline.get('objective')} | "
        f"{chosen.get('objective')} | {deltas.get('objective')} |",
    ]
    split = result.get("split") or {}
    holdout_deltas = result.get("holdout_deltas")
    if holdout_deltas:
        chosen_half = chosen.get("holdout") or {}
        baseline_half = baseline.get("holdout") or {}
        lines += [
            "",
            "## On the holdout, which the selection never saw",
            "",
            f"{split.get('calibration_sessions')} calibration sessions, "
            f"{split.get('holdout_sessions')} held out.",
            "",
            "| metric | shipped | chosen | Δ |",
            "|---|---|---|---|",
        ]
        for metric, label in (
            ("recipe_hit_rate", "recipe_hit_rate"),
            ("injected_chars_per_turn", "injected_chars_per_turn"),
            ("char_efficiency", "char_efficiency"),
            ("escalation_precision", "escalation precision"),
            ("escalation_recall", "escalation recall"),
            ("objective", "objective"),
        ):
            lines.append(
                f"| {label} | {baseline_half.get(metric)} | "
                f"{chosen_half.get(metric)} | {holdout_deltas.get(metric)} |"
            )
    elif split.get("note"):
        lines += ["", f"Split: {split['note']}."]
    if result.get("note"):
        lines += ["", f"Note: {result['note']}."]
    lines += [
        "",
        "Nothing is applied by this command: the policy file is the operator's "
        "to change, and the numbers above are the reason to change them, with "
        "the same dataset the report was measured on.",
        "",
    ]
    return "\n".join(lines)


def _write_tuned_policy(
    config: AgenticKnowledgeConfig, result: Dict[str, Any], report_path: Optional[str]
) -> str:
    """
    Write the chosen point next to the report, as a policy the operator can read.

    The file is a proposal with the report's numbers in it, not a policy the
    layer reads: it is named ``.tuned.json`` so it is never the file a
    deployment loads by accident, and it carries the deltas it claims so the
    diff into the real policy is reviewable against the evidence rather than
    against the taste of whoever ran the tune.

    The tune writes it only when asked.  A tool that edits the policy it
    measured against has answered its own question, and a report the tool
    wrote about a policy the tool changed is not evidence about anything.
    """
    chosen = result.get("chosen") or {}
    if not chosen:
        return ""
    directory = (
        pathlib.Path(report_path).parent if report_path else pathlib.Path(".")
    )
    target = pathlib.Path(directory) / "agentic_knowledge.tuned.json"
    payload = {
        "_comment": (
            "Proposed thresholds from llm-router-agentic-kb-tune. Not loaded "
            "by the layer; review and merge into the policy by hand."
        ),
        "read": {
            "min_similarity": chosen.get("min_similarity"),
        },
        "escalation": {
            "strong_recipe_score": chosen.get("strong_recipe_score"),
            "loop_threshold": chosen.get("loop_threshold"),
        },
        "search": result.get("search") or {},
        "earned": {
            "objective": chosen.get("objective"),
            "recipe_hit_rate": chosen.get("recipe_hit_rate"),
            "injected_chars_per_turn": chosen.get("injected_chars_per_turn"),
            "char_efficiency": chosen.get("char_efficiency"),
            "escalation_precision": chosen.get("escalation_precision"),
            "escalation_recall": chosen.get("escalation_recall"),
        },
        "deltas_against_shipped": result.get("deltas") or {},
        "weights": chosen.get("weights") or {},
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(target)


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="llm-router-agentic-kb-tune",
        description=(
            "Grid-search the agentic knowledge policy's thresholds on a "
            "replayed dataset, and report the point that earns most of what "
            "the policy is asked to earn."
        ),
    )
    parser.add_argument(
        "--dataset", required=True, help="Path to the eval JSON."
    )
    parser.add_argument(
        "--config", default=None, help="Path to a policy JSON; the shipped one."
    )
    parser.add_argument(
        "--min-similarity",
        nargs="+",
        type=float,
        default=[0.25, 0.35, 0.45],
        help="Values for the retrieval floor (default: 0.25 0.35 0.45).",
    )
    parser.add_argument(
        "--strong-recipe-score",
        nargs="+",
        type=float,
        default=[0.5, 0.6, 0.7],
        help="Values for the strong-recipe floor (default: 0.5 0.6 0.7).",
    )
    parser.add_argument(
        "--loop-threshold",
        nargs="+",
        type=int,
        default=[2, 3, 4],
        help="Values for the loop threshold (default: 2 3 4).",
    )
    parser.add_argument(
        "--weight-factor",
        nargs="+",
        type=float,
        default=[0.8, 1.0, 1.2],
        help=(
            "Factors applied to each source's shipped weight "
            "(default: 0.8 1.0 1.2)."
        ),
    )
    parser.add_argument(
        "--weight-category",
        action="append",
        default=None,
        help=(
            "A source category to scale; repeatable.  Defaults to every "
            "category the policy's modes read."
        ),
    )
    parser.add_argument(
        "--per-category-factors",
        action="store_true",
        help=(
            "Sweep each weighted category on its own factor instead of one "
            "shared factor for all of them; the grid grows to the factor "
            "range raised to the number of categories, and a ranking decided "
            "by relative weights rarely earns it."
        ),
    )
    parser.add_argument(
        "--lambda",
        dest="lambda_cost",
        type=float,
        default=1e-4,
        help=(
            "The price of a character of injection in hit-rate units; the "
            "objective is recipe_hit_rate − λ·injected_chars_per_turn "
            "(default: 1e-4)."
        ),
    )
    parser.add_argument(
        "--precision-floor",
        type=float,
        default=0.8,
        help=(
            "The constraint: escalation precision must not fall below this "
            "(default: 0.8)."
        ),
    )
    parser.add_argument(
        "--recall-floor",
        type=float,
        default=0.0,
        help=(
            "The other constraint: of the turns the dataset marks as loops, "
            "the share the point must escalate on (default: 0, no floor). A "
            "precision floor alone is satisfiable by escalating never; this "
            "is the floor that says a point must still catch the loops it "
            "was given a chance to catch."
        ),
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.0,
        help=(
            "Share of the sessions held out of the selection, split by a hash "
            "of --seed and the session id so two runs split the same way. The "
            "grid is chosen on the rest and only measured on the holdout "
            "(default: 0, no holdout)."
        ),
    )
    parser.add_argument(
        "--holdout-session",
        action="append",
        default=None,
        metavar="SESSION_ID",
        help=(
            "A session to hold out of the selection; repeatable. Names the "
            "operator keeps win over a fraction: the holdout is theirs, not "
            "the tool's."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=11,
        help="Seed of the holdout split (default: 11).",
    )
    parser.add_argument(
        "--search",
        choices=("grid", "bayesian"),
        default="grid",
        help=(
            "How the points are chosen: the full grid, or the TPE search that "
            "spends --evaluations where the evidence points (default: grid)."
        ),
    )
    parser.add_argument(
        "--evaluations",
        type=int,
        default=40,
        help=(
            "The replay budget of the Bayesian search (default: 40).  Ignored "
            "by the grid, which measures every point it is given."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write the JSON report here; the markdown goes to standard out.",
    )
    parser.add_argument(
        "--write-policy",
        action="store_true",
        help=(
            "Write the chosen point next to the report as "
            "agentic_knowledge.tuned.json, a proposal for review.  The layer "
            "never reads the file; merging it into the policy is the "
            "operator's, and the diff is against the numbers it reports."
        ),
    )
    parser.add_argument(
        "--log-level", default="WARNING", help="Logging level (default: WARNING)."
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """
    Run one tune and print the justification.

    Parameters
    ----------
    argv : List[str], optional
        Command-line arguments; ``sys.argv`` when omitted.

    Returns
    -------
    int
        Process exit code: ``0`` when a report was produced.
    """
    import logging

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.WARNING),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    config = AgenticKnowledgeConfig.from_file(
        pathlib.Path(args.config) if args.config else None
    )
    config.validate()
    sessions = load_sessions(pathlib.Path(args.dataset))
    if not sessions:
        print("the dataset is empty; there is nothing to tune on", file=sys.stderr)
        return 1

    categories = [str(name) for name in (args.weight_category or [])]
    if not categories:
        categories = sorted(
            {
                source.category
                for policy in config.modes.values()
                for source in policy.sources
            }
        )

    try:
        result = tune(
            config=config,
            sessions=sessions,
            min_similarity=[float(v) for v in args.min_similarity],
            strong_recipe_score=[float(v) for v in args.strong_recipe_score],
            loop_threshold=[int(v) for v in args.loop_threshold],
            weight_factors=[float(v) for v in args.weight_factor],
            weight_categories=categories,
            lam=float(args.lambda_cost),
            precision_floor=float(args.precision_floor),
            recall_floor=float(args.recall_floor),
            shared_factors=not args.per_category_factors,
            holdout_fraction=float(args.holdout_fraction),
            holdout_ids=[str(name) for name in (args.holdout_session or [])],
            seed=int(args.seed),
            search=str(getattr(args, "search", "grid")),
            evaluations=int(getattr(args, "evaluations", 40)),
        )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1

    if args.output:
        pathlib.Path(args.output).write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
    if getattr(args, "write_policy", False):
        written = _write_tuned_policy(config, result, args.output)
        if written:
            print(f"tuned policy written: {written}", file=sys.stderr)
    print(_markdown(result))
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    sys.exit(main())
