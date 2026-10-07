"""
Offline Codex routing evaluation; no generation calls or historical labels.

The corpus is a list of *request prefixes* — the payload the router saw at the
moment a decision had to be made — paired with the work mode a human reading
only what came **before** that decision would label.  Nothing here reads the
historical routing decisions recorded in the logs: the report is always a
replay of the *current* implementation, so a change in the cascade shows up as
a change in the numbers instead of being blended with the recorded past.

Variants
--------
``deterministic``
    The cascade without a semantic layer.  Fully reproducible; this is the
    variant a deployment with ``semantic.enabled=false`` actually runs.
``cascade``
    The cascade with the semantic layer, i.e. what a full deployment does.
``semantic_only``
    The semantic layer alone, kept as an upper bound on what the embeddings
    contribute on their own.

Modes, not models, are the unit of measurement
----------------------------------------------
Every metric that describes routing quality is computed on the *mode*.  Models
are mapped back from the modes at the end (``expected_model`` / the variant's
``model``), so reassigning ``model_name`` in the configuration moves the
auxiliary model metrics but leaves the mode metrics — accuracy, confusion and
the transition counters — byte-for-byte identical.
"""

import argparse
import hashlib
import json
import math
import time

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_router_plugins.utils.routing.agentic_routing.codex.classifier import (
    CLASS_ROUTED_MODES,
    CodexModeClassifier,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.config import CodexRoutingConfig
from llm_router_plugins.utils.routing.agentic_routing.codex.payload import CodexPayloadParser
from llm_router_plugins.utils.routing.agentic_routing.codex.semantic import CodexSemanticLayer
from llm_router_plugins.utils.routing.agentic_routing.codex.state import (
    InMemoryRoutingStateStore,
    remember_decision,
    replace as _replace_config,
)
from llm_router_plugins.utils.routing.common import build_embedding_router

#: Variants that never depend on an embedding lookup.
DETERMINISTIC_VARIANT = "deterministic"

#: Variants that require a router and are only reported when one was supplied.
ROUTER_VARIANTS = ("cascade", "semantic_only")

#: The deterministic cascade plus the shared session memory, replayed with a
#: fresh store per session sequence.
STATEFUL_VARIANT = "stateful"

#: Splits a sequence must stay in: calibration and holdout are never mixed.
SPLITS = ("calibration", "holdout")


class CheckedRouter:
    """Keep lookup failures visible even though production routing fails open."""

    def __init__(self, router):
        self.router = router
        self.result = None
        self.error = None

    def route_context(self, parts):
        self.result = None
        self.error = None
        try:
            self.result = self.router.route_context(parts)
            return self.result
        except Exception as exc:
            self.error = exc
            raise


def load_cases(path, config, split):
    """
    Read and validate the corpus, selecting one split.

    A case is one request prefix plus the mode expected *at that point*.
    ``sequence`` groups prefixes that belong to the same running session, so
    transition metrics compare consecutive requests of one session rather than
    unrelated requests that happen to sit next to each other in the file.
    ``metadata`` carries the session/turn headers the real payload had, which
    the stateful variant needs to isolate one session's memory from another's.
    ``ambiguous`` marks a prefix without enough evidence before the decision;
    such cases are reported separately instead of counting for or against.

    Parameters
    ----------
    path : pathlib.Path
        Corpus file.
    config : CodexRoutingConfig
        Config whose modes validate ``expected_mode``.
    split : str
        ``calibration``, ``holdout`` or ``all``.

    Returns
    -------
    list
        The selected cases, in file order.

    Raises
    ------
    ValueError
        On a malformed corpus, an unknown or class-routed expected mode, an
        invalid split, a sequence spanning two splits, or an empty selection.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1 or not isinstance(data.get("cases"), list):
        raise ValueError("Expected schema_version=1 and a cases list")
    seen = set()
    sequence_splits = {}
    cases = []
    for case in data["cases"]:
        if not isinstance(case, dict):
            raise ValueError("Each case must be an object")
        identifier = case.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("Case IDs must be non-empty and unique")
        seen.add(identifier)
        if case.get("expected_mode") not in config.mode_by_name:
            raise ValueError(f"Unknown expected mode in {identifier}")
        if case["expected_mode"] in CLASS_ROUTED_MODES:
            raise ValueError("This corpus evaluates main-turn semantic work modes only")
        if case.get("split") not in SPLITS:
            raise ValueError(f"Invalid split in {identifier}")
        if not isinstance(case.get("input"), list) or not case["input"]:
            raise ValueError(f"Missing Responses input in {identifier}")
        if "sequence" in case and (
            not isinstance(case["sequence"], str) or not case["sequence"]
        ):
            raise ValueError(f"Invalid sequence in {identifier}")
        if "metadata" in case and not isinstance(case["metadata"], dict):
            raise ValueError(f"Invalid metadata in {identifier}")
        if "ambiguous" in case and not isinstance(case["ambiguous"], bool):
            raise ValueError(f"Invalid ambiguous flag in {identifier}")
        sequence = case.get("sequence")
        if sequence:
            known = sequence_splits.setdefault(sequence, case["split"])
            if known != case["split"]:
                raise ValueError(f"Sequence {sequence} spans two splits in {identifier}")
        cases.append(case)
    if split != "all":
        cases = [case for case in cases if case["split"] == split]
    if not cases:
        raise ValueError("No cases selected")
    return cases


def build_payload(case: Dict[str, Any], config: CodexRoutingConfig) -> Dict[str, Any]:
    """
    Rebuild the request payload of *case*, replaying its session metadata.

    Parameters
    ----------
    case : Dict[str, Any]
        A corpus entry.
    config : CodexRoutingConfig
        Config providing the trigger model.

    Returns
    -------
    Dict[str, Any]
        A payload carrying the case input plus the ``client_metadata`` the
        recorded request had, so a replay reaches the same session identity.
    """
    payload: Dict[str, Any] = {"model": config.trigger_model, "input": case["input"]}
    metadata = case.get("metadata")
    if isinstance(metadata, dict) and metadata:
        payload["client_metadata"] = dict(metadata)
    return payload


def _transition_metrics(
    records: List[Dict[str, Any]], prediction: str, field: str = "mode"
) -> Dict[str, Any]:
    """
    Count switch mistakes inside each session sequence, for one field.

    Within a sequence, every consecutive pair contributes a *pair*.  A pair is
    an expected switch when the expected value changes and an actual switch
    when the predicted value changes; the two disagree as *missed* or
    *unnecessary* switches.  When a switch is expected, the delay counts the
    steps the prediction needed to reach the newly expected value — a
    prediction that was already right has delay ``0``, and one that never
    catches up within that run is censored at the run's length.

    Parameters
    ----------
    records : List[Dict[str, Any]]
        One report record per request, in session order.
    prediction : str
        Report key holding the variant, with ``mode``/``model`` inside.

    Returns
    -------
    Dict[str, Any]
        Counters plus ``mean_switch_delay`` over the expected switches.
    """
    counters: Counter = Counter({
        "pairs": 0, "expected_switches": 0, "actual_switches": 0,
        "unnecessary_switches": 0, "missed_switches": 0,
    })
    delays: List[int] = []
    expected_key = "expected_mode" if field == "mode" else "expected_model"
    for sequence in _sequences(records):
        for index in range(1, len(sequence)):
            previous, current = sequence[index - 1], sequence[index]
            expected_switch = (
                previous[expected_key] != current[expected_key]
            )
            actual_switch = (
                previous[prediction][field] != current[prediction][field]
            )
            counters["pairs"] += 1
            counters["expected_switches"] += int(expected_switch)
            counters["actual_switches"] += int(actual_switch)
            counters["unnecessary_switches"] += int(actual_switch and not expected_switch)
            counters["missed_switches"] += int(expected_switch and not actual_switch)
            if not expected_switch:
                continue
            target = current[expected_key]
            delay = 0
            cursor = index
            while (
                cursor < len(sequence)
                and sequence[cursor][expected_key] == target
                and sequence[cursor][prediction][field] != target
            ):
                delay += 1
                cursor += 1
            delays.append(delay)
    metrics = dict(counters)
    metrics["mean_switch_delay"] = sum(delays) / len(delays) if delays else None
    return metrics


def _sequences(records: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Group the records of each session, in request order, newest session last.

    A record without a :attr:`sequence` belongs to no session and contributes
    no transition: two unrelated requests that merely sit next to each other in
    the corpus must not be counted as a mode switch.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        sequence = record.get("sequence")
        if sequence:
            grouped[sequence].append(record)
    return list(grouped.values())


def summarize(records, prediction):
    """
    Aggregate one variant over the report records.

    Mode accuracy, the per-mode confusion matrix and the mode transition
    counters describe routing quality.  The model figures are kept as an
    appendix: they depend on the mapping in the configuration, which is not
    what a routing change moves.  Ambiguous cases are tallied separately and
    excluded from the mode metrics, so an unlabeled prefix can neither inflate
    nor pollute an accuracy number.

    Parameters
    ----------
    records : list
        Report records produced by :func:`evaluate`.
    prediction : str
        Report key holding the variant being summarized.

    Returns
    -------
    dict
        Metrics for one variant.
    """
    scored = [record for record in records if not record.get("ambiguous")]
    count = len(scored)
    correct_modes = 0
    correct_models = 0
    modes = defaultdict(Counter)
    models = defaultdict(Counter)
    confusion = defaultdict(Counter)
    mode_confusion = defaultdict(Counter)
    sources = Counter()
    for record in scored:
        result = record[prediction]
        expected_mode = record["expected_mode"]
        expected_model = record["expected_model"]
        mode_ok = result["mode"] == expected_mode
        model_ok = result["model"] == expected_model
        correct_modes += mode_ok
        correct_models += model_ok
        modes[expected_mode].update(total=1, correct=int(mode_ok))
        modes[result["mode"]].update(predicted=1)
        models[expected_model].update(total=1, correct=int(model_ok))
        confusion[expected_model][result["model"]] += 1
        mode_confusion[expected_mode][result["mode"]] += 1
        sources[result["source"]] += 1
    for counts in modes.values():
        counts["recall"] = counts["correct"] / counts["total"] if counts["total"] else None
        counts["precision"] = counts["correct"] / counts["predicted"] if counts["predicted"] else None
    total = len(records)
    return {
        "count": count,
        "ambiguous_count": total - count,
        "mode_accuracy": correct_modes / count if count else None,
        "model_accuracy": correct_models / count if count else None,
        "per_mode": dict(modes),
        "per_expected_model": dict(models),
        "mode_confusion": dict(mode_confusion),
        "model_confusion": dict(confusion),
        "sources": dict(sources),
        "mode_transitions": _transition_metrics(scored, prediction, "mode"),
        "model_transitions": _transition_metrics(scored, prediction, "model"),
        "mean_routing_ms": (
            sum(r[prediction]["elapsed_ms"] for r in records) / total if total else None
        ),
        "semantic_acceptance_rate": (
            sources["semantic"] / count if count else None
        ),
    }


def evaluate(config, cases, router=None):
    """
    Replay *cases* through the current implementation and report every variant.

    Parameters
    ----------
    config : CodexRoutingConfig
        Configuration under evaluation.
    cases : list
        Cases from :func:`load_cases`.
    router : Any, optional
        Embedding router.  When it is ``None`` only the deterministic variant
        is produced, which is the reproducible half of the comparison.

    Returns
    -------
    dict
        One summary per variant plus the per-case ``records``.

    Raises
    ------
    ValueError
        On a case that is not a non-empty main turn, or an incomplete ranking.
    RuntimeError
        When a supplied router fails: an evaluation that silently degraded to
        the fallback would report a lookup outage as a routing choice.
    """
    checked = CheckedRouter(router) if router is not None else None
    layer = (
        CodexSemanticLayer(
            checked, config.similarity_threshold, config.mode_by_name,
            min_margin=config.semantic_min_margin,
            intent_max_chars=config.semantic_intent_max_chars,
            phase_max_chars=config.semantic_phase_max_chars,
            phase_rules=config.phase,
        )
        if checked is not None
        else None
    )
    parser = CodexPayloadParser(max_chars=config.classify_max_chars)
    deterministic = CodexModeClassifier(config)
    cascade = CodexModeClassifier(config, semantic=layer)
    # Replay never touches a real Redis: one fresh in-memory store per session
    # sequence reproduces what a shared store would hold, without leaking state
    # between sequences or between runs.
    memory_policy = _replace_config(config.memory, enabled=True, backend="memory")
    memory_classifiers: Dict[str, Any] = {}
    expected_targets = set(config.mode_by_name) - set(CLASS_ROUTED_MODES)
    records = []
    for case in cases:
        payload = build_payload(case, config)
        request = parser.parse(payload)
        if request.request_class != "main" or not request.intent_text.strip():
            raise ValueError(f"Expected a non-empty main turn: {case['id']}")
        variants: Dict[str, Dict[str, Any]] = {}
        pairs = [(DETERMINISTIC_VARIANT, deterministic)]
        # A session sequence shares one store, exactly as one Codex session
        # shares one entry in the real one; unrelated cases start from nothing.
        bucket = case.get("sequence") or case["id"]
        pair = memory_classifiers.get(bucket)
        if pair is None:
            store = InMemoryRoutingStateStore(memory_policy)
            pair = memory_classifiers[bucket] = (
                CodexModeClassifier(config, memory=store), store
            )
        memory_classifier, memory_store = pair
        if layer is not None:
            pairs.append(("cascade", cascade))
        for name, classifier in pairs:
            started = time.perf_counter()
            decision = classifier.classify(payload, request)
            variants[name] = _variant(config, decision.mode, decision.source, started)
        started = time.perf_counter()
        memory_decision = memory_classifier.classify(payload, request)
        remember_decision(memory_store, memory_policy, request, memory_decision)
        variants[STATEFUL_VARIANT] = _variant(
            config, memory_decision.mode, memory_decision.source, started
        )
        if layer is None:
            records.append(_record(config, case, variants))
            continue
        checked.error = None
        started = time.perf_counter()
        routed = layer.route(request)
        semantic_ms = (time.perf_counter() - started) * 1000
        if checked.error is not None:
            raise RuntimeError(f"Semantic lookup failed: {case['id']}") from checked.error
        accepted, _ = layer.accept(routed)
        semantic_mode = accepted.name if accepted else config.fallback_mode
        ranking = routed.get("all_scores", []) if routed else []
        if (
            not isinstance(ranking, list) or len(ranking) != len(expected_targets)
            or any(not isinstance(entry, dict) for entry in ranking)
            or {entry.get("target") for entry in ranking} != expected_targets
            or any(
                isinstance(entry.get("similarity"), bool)
                or not isinstance(entry.get("similarity"), (int, float))
                or not math.isfinite(entry["similarity"])
                or not -1 <= entry["similarity"] <= 1
                for entry in ranking
            )
        ):
            raise ValueError(f"Incomplete or invalid semantic ranking: {case['id']}")
        scores = sorted((entry["similarity"] for entry in ranking), reverse=True)
        variants["semantic_only"] = {
            "mode": semantic_mode,
            "model": config.mode_by_name[semantic_mode].model_name,
            "source": "semantic" if accepted else "fallback",
            "elapsed_ms": semantic_ms,
            "all_scores": ranking,
            "margin": scores[0] - scores[1] if len(scores) > 1 else None,
        }
        records.append(_record(config, case, variants))
    report = {
        name: summarize(records, name)
        for name in (
            (DETERMINISTIC_VARIANT, STATEFUL_VARIANT, "cascade", "semantic_only")
            if layer is not None
            else (DETERMINISTIC_VARIANT, STATEFUL_VARIANT)
        )
    }
    report["records"] = records
    return report


def _variant(config: CodexRoutingConfig, mode: str, source: str, started: float) -> Dict[str, Any]:
    """Snapshot one variant's answer, with the model mapped back from the mode."""
    return {
        "mode": mode,
        "model": config.mode_by_name[mode].model_name,
        "source": source,
        "elapsed_ms": (time.perf_counter() - started) * 1000,
    }


def _record(config: CodexRoutingConfig, case: Dict[str, Any], variants: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Assemble one report record from the case and the variants that ran."""
    return {
        "id": case["id"],
        "split": case["split"],
        "sequence": case.get("sequence"),
        "ambiguous": bool(case.get("ambiguous")),
        "expected_mode": case["expected_mode"],
        "expected_model": config.mode_by_name[case["expected_mode"]].model_name,
        **variants,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=(*SPLITS, "all"), default="holdout")
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Replay the deterministic cascade only: no embedding model is loaded.",
    )
    args = parser.parse_args()
    raw = args.config.read_bytes()
    config = CodexRoutingConfig._from_raw(json.loads(raw))
    config.validate_args()
    if not args.no_semantic and (
        not config.semantic_enabled or config.semantic_aggregation != "per_target_top_k"
    ):
        parser.error(
            "Semantic evaluation requires enabled semantic routing and "
            "per_target_top_k; pass --no-semantic for the deterministic replay"
        )
    cases = load_cases(args.dataset, config, args.split)
    started = time.perf_counter()
    router = None
    if not args.no_semantic:
        router = build_embedding_router(
            embedding_model=config.embedding_model,
            chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap,
            top_k=config.top_k,
            routing_targets=tuple(m for m in config.codex_modes if m.name not in CLASS_ROUTED_MODES),
            aggregation=config.semantic_aggregation,
        )
    initialization_ms = (time.perf_counter() - started) * 1000
    report = evaluate(config, cases, router)
    report["metadata"] = {
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "embedding_model": config.embedding_model,
        "threshold": config.similarity_threshold,
        "min_margin": config.semantic_min_margin,
        "split": args.split, "initialization_ms": initialization_ms,
        "environment_overrides": False, "persistent_index": False,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
