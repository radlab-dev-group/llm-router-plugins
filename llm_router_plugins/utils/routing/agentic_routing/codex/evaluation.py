"""Offline Codex routing evaluation; no generation calls or historical labels."""

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

from llm_router_plugins.utils.routing.agentic_routing.codex.classifier import (
    CLASS_ROUTED_MODES,
    CodexModeClassifier,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.config import CodexRoutingConfig
from llm_router_plugins.utils.routing.agentic_routing.codex.payload import CodexPayloadParser
from llm_router_plugins.utils.routing.agentic_routing.codex.semantic import CodexSemanticLayer
from llm_router_plugins.utils.routing.common import build_embedding_router


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
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1 or not isinstance(data.get("cases"), list):
        raise ValueError("Expected schema_version=1 and a cases list")
    seen = set()
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
        if case.get("split") not in ("calibration", "holdout"):
            raise ValueError(f"Invalid split in {identifier}")
        if not isinstance(case.get("input"), list) or not case["input"]:
            raise ValueError(f"Missing Responses input in {identifier}")
        if "sequence" in case and (
            not isinstance(case["sequence"], str) or not case["sequence"]
        ):
            raise ValueError(f"Invalid sequence in {identifier}")
        if split == "all" or split == case["split"]:
            cases.append(case)
    if not cases:
        raise ValueError("No cases selected")
    return cases


def summarize(records, prediction):
    correct_modes = 0
    correct_models = 0
    modes = defaultdict(Counter)
    models = defaultdict(Counter)
    confusion = defaultdict(Counter)
    sources = Counter()
    previous = {}
    transitions = Counter()
    for record in records:
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
        sources[result["source"]] += 1
        sequence = record.get("sequence")
        if sequence:
            old = previous.get(sequence)
            if old is not None:
                expected_switch = old["expected_model"] != expected_model
                actual_switch = old[prediction]["model"] != result["model"]
                transitions.update(
                    pairs=1,
                    expected_switches=int(expected_switch),
                    actual_switches=int(actual_switch),
                    unnecessary_switches=int(actual_switch and not expected_switch),
                    missed_switches=int(expected_switch and not actual_switch),
                )
            previous[sequence] = record
    count = len(records)
    for counts in modes.values():
        counts["recall"] = counts["correct"] / counts["total"] if counts["total"] else None
        counts["precision"] = counts["correct"] / counts["predicted"] if counts["predicted"] else None
    return {
        "count": count,
        "mode_accuracy": correct_modes / count,
        "model_accuracy": correct_models / count,
        "per_mode": dict(modes),
        "per_expected_model": dict(models),
        "model_confusion": dict(confusion),
        "sources": dict(sources),
        "model_transitions": dict(transitions),
        "mean_routing_ms": sum(r[prediction]["elapsed_ms"] for r in records) / count,
        "semantic_acceptance_rate": sources["semantic"] / count,
    }


def evaluate(config, cases, router):
    checked = CheckedRouter(router)
    layer = CodexSemanticLayer(
        checked, config.similarity_threshold, config.mode_by_name,
        min_margin=config.semantic_min_margin,
        intent_max_chars=config.semantic_intent_max_chars,
        phase_max_chars=config.semantic_phase_max_chars,
    )
    parser = CodexPayloadParser(max_chars=config.classify_max_chars)
    classifier = CodexModeClassifier(config, semantic=layer)
    records = []
    for case in cases:
        payload = {"model": config.trigger_model, "input": case["input"]}
        request = parser.parse(payload)
        if request.request_class != "main" or not request.intent_text.strip():
            raise ValueError(f"Expected a non-empty main turn: {case['id']}")
        checked.error = None
        started = time.perf_counter()
        decision = classifier.classify(payload, request)
        cascade_ms = (time.perf_counter() - started) * 1000
        if checked.error is not None:
            raise RuntimeError(f"Semantic lookup failed: {case['id']}") from checked.error
        started = time.perf_counter()
        routed = layer.route(request)
        semantic_ms = (time.perf_counter() - started) * 1000
        if checked.error is not None or routed is None:
            raise RuntimeError(f"Semantic lookup failed: {case['id']}") from checked.error
        accepted, _ = layer.accept(routed)
        semantic_mode = accepted.name if accepted else config.fallback_mode
        ranking = routed.get("all_scores", [])
        expected_targets = set(config.mode_by_name) - set(CLASS_ROUTED_MODES)
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
        records.append({
            "id": case["id"], "split": case["split"],
            "sequence": case.get("sequence"),
            "expected_mode": case["expected_mode"],
            "expected_model": config.mode_by_name[case["expected_mode"]].model_name,
            "cascade": {
                "mode": decision.mode,
                "model": config.mode_by_name[decision.mode].model_name,
                "source": decision.source, "elapsed_ms": cascade_ms,
            },
            "semantic_only": {
                "mode": semantic_mode,
                "model": config.mode_by_name[semantic_mode].model_name,
                "source": "semantic" if accepted else "fallback",
                "elapsed_ms": semantic_ms,
                "all_scores": ranking,
                "margin": scores[0] - scores[1] if len(scores) > 1 else None,
            },
        })
    return {
        "cascade": summarize(records, "cascade"),
        "semantic_only": summarize(records, "semantic_only"),
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", choices=("calibration", "holdout", "all"), default="holdout")
    args = parser.parse_args()
    raw = args.config.read_bytes()
    config = CodexRoutingConfig._from_raw(json.loads(raw))
    config.validate_args()
    if not config.semantic_enabled or config.semantic_aggregation != "per_target_top_k":
        parser.error("Evaluation requires enabled semantic routing and per_target_top_k")
    cases = load_cases(args.dataset, config, args.split)
    started = time.perf_counter()
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