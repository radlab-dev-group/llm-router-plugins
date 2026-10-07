"""Offline evaluation plumbing, not embedding accuracy measurements."""

import copy
import json
import pathlib
import sys
from dataclasses import replace
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

from llm_router_plugins.utils.routing.agentic_routing.codex.evaluation import (
    DETERMINISTIC_VARIANT,
    STATEFUL_VARIANT,
    build_payload,
    evaluate,
    load_cases,
    summarize,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.config import CodexRoutingConfig
from llm_router_plugins.utils.routing.agentic_routing.codex.payload import CodexPayloadParser


_ROOT = pathlib.Path(__file__).resolve().parent.parent
_DATASET = _ROOT / "tests/data/codex_routing_quality.json"


@pytest.fixture
def config():
    path = _ROOT / "llm_router_plugins/resources/routing/agentic_routing_codex.json"
    result = CodexRoutingConfig._from_raw(json.loads(path.read_text(encoding="utf-8")))
    result.validate_args()
    return result


def user_texts(case):
    """The user text of a case, read the way the parser reads it."""
    texts = []
    for item in case["input"]:
        if item.get("role") != "user":
            continue
        content = item["content"]
        parts = [content] if isinstance(content, str) else content
        for part in parts:
            if isinstance(part, str):
                texts.append(part)
            elif part.get("type") == "input_text":
                texts.append(part["text"])
    return texts


def stub_router(config, winner="implement", similarity=0.9):
    """A router that always names *winner*, with a complete, ranked score set."""
    def route_context(parts):
        route_context.calls.append(parts)
        return {
            "target_name": winner, "similarity": similarity,
            "all_scores": [
                {"target": mode.name, "similarity": (
                    similarity if mode.name == winner else similarity - 0.5
                )}
                for mode in config.codex_modes
                if mode.name not in ("aux_title", "compaction")
            ],
        }
    route_context.calls = []
    return SimpleNamespace(route_context=route_context)


def remapped(config, model_name):
    """The same config with every mode pointing at one model."""
    return replace(
        config,
        codex_modes=tuple(
            replace(mode, model_name=model_name) for mode in config.codex_modes
        ),
    )


def test_corpus_cases_use_the_codex_wire_format(config):
    """Every case parses as a real main turn, or the replay measures nothing."""
    parser = CodexPayloadParser(max_chars=config.classify_max_chars)
    for case in load_cases(_DATASET, config, "all"):
        request = parser.parse(build_payload(case, config))
        assert request.request_class == "main", case["id"]
        assert request.intent_text.strip(), case["id"]


def test_corpus_labels_and_holdout_are_separate_from_examples(config):
    cases = load_cases(_DATASET, config, "all")
    assert {case["expected_mode"] for case in cases} == {
        "plan", "implement", "test", "review", "git_review", "debug",
    }
    examples = {text for mode in config.codex_modes for text in mode.examples}
    for case in load_cases(_DATASET, config, "holdout"):
        for text in user_texts(case):
            assert text not in examples
    assert len(load_cases(_DATASET, config, "calibration")) + len(
        load_cases(_DATASET, config, "holdout")
    ) == len(cases)


def test_sessions_stay_inside_their_own_split(config):
    """A sequence split across calibration and holdout would leak the corpus."""
    cases = load_cases(_DATASET, config, "all")
    by_sequence = {}
    for case in cases:
        sequence = case.get("sequence")
        if sequence:
            by_sequence.setdefault(sequence, set()).add(case["split"])
    assert by_sequence
    assert all(len(splits) == 1 for splits in by_sequence.values())


@pytest.mark.parametrize("mutation", [
    "duplicate", "mode", "split", "input", "sequence", "shared_sequence_split",
    "metadata", "ambiguous",
])
def test_invalid_corpus_is_rejected(config, tmp_path, mutation):
    case = {"id": "example", "split": "holdout", "expected_mode": "implement",
            "input": [{"role": "user", "content": "Add a field."}]}
    cases = [case]
    if mutation == "duplicate":
        cases.append(dict(case))
    elif mutation == "mode":
        case["expected_mode"] = "unknown"
    elif mutation == "split":
        case["split"] = "training"
    elif mutation == "input":
        case["input"] = []
    elif mutation == "sequence":
        case["sequence"] = []
    elif mutation == "shared_sequence_split":
        case["sequence"] = "example"
        cases.append({"id": "second", "split": "calibration",
                      "expected_mode": "implement", "sequence": "example",
                      "input": [{"role": "user", "content": "Add a field."}]})
    elif mutation == "metadata":
        case["metadata"] = "session-1"
    else:
        case["ambiguous"] = "yes"
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"schema_version": 1, "cases": cases}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_cases(path, config, "holdout" if mutation != "shared_sequence_split" else "all")


def test_empty_selection_is_rejected(config, tmp_path):
    path = tmp_path / "cases.json"
    path.write_text('{"schema_version":1,"cases":[]}', encoding="utf-8")
    with pytest.raises(ValueError, match="No cases"):
        load_cases(path, config, "holdout")


def test_case_metadata_reaches_the_replayed_payload(config):
    case = {"id": "x", "split": "holdout", "expected_mode": "implement",
            "input": [{"role": "user", "content": "Add a field."}],
            "metadata": {"session_id": "s", "thread_id": "t", "turn_id": "u"}}
    payload = build_payload(case, config)
    assert payload["model"] == config.trigger_model
    assert payload["client_metadata"] == {"session_id": "s", "thread_id": "t", "turn_id": "u"}


def test_mode_errors_are_not_necessarily_model_errors():
    records = [
        {"expected_mode": "review", "expected_model": "large",
         "cascade": {"mode": "implement", "model": "large", "source": "fallback", "elapsed_ms": 1}},
        {"expected_mode": "test", "expected_model": "small",
         "cascade": {"mode": "review", "model": "large", "source": "semantic", "elapsed_ms": 3}},
    ]
    report = summarize(records, "cascade")
    assert report["mode_accuracy"] == 0
    assert report["model_accuracy"] == 0.5
    assert report["model_confusion"]["small"]["large"] == 1
    assert report["mode_confusion"]["test"]["review"] == 1
    assert report["per_mode"]["review"]["precision"] == 0
    assert report["mean_routing_ms"] == 2


def test_ambiguous_cases_are_reported_separately():
    record = {"expected_mode": "implement", "expected_model": "large",
              "cascade": {"mode": "implement", "model": "large", "source": "phase", "elapsed_ms": 0}}
    clear = dict(record, sequence="a")
    uncertain = dict(record, id="u", sequence="b", ambiguous=True)
    report = summarize([clear, uncertain], "cascade")
    assert report["count"] == 1
    assert report["ambiguous_count"] == 1
    assert report["mode_accuracy"] == 1
    assert report["mode_transitions"]["pairs"] == 0


def test_sequence_metrics_track_expected_mode_boundary():
    records = []
    for mode in ("implement", "test", "test", "debug"):
        records.append({
            "sequence": "work", "expected_mode": mode, "expected_model": "m",
            "cascade": {"mode": mode, "model": "m", "source": "phase", "elapsed_ms": 0},
        })
    transitions = summarize(records, "cascade")["mode_transitions"]
    assert transitions == {
        "pairs": 3, "expected_switches": 2, "actual_switches": 2,
        "unnecessary_switches": 0, "missed_switches": 0, "mean_switch_delay": 0.0,
    }


def test_unnecessary_and_missed_mode_switches_are_counted():
    expected = ("implement", "test", "implement", "implement")
    predicted = ("implement", "implement", "implement", "debug")
    records = [
        {"sequence": "work", "expected_mode": want, "expected_model": "m",
         "cascade": {"mode": got, "model": "m", "source": "fallback", "elapsed_ms": 0}}
        for want, got in zip(expected, predicted)
    ]
    transitions = summarize(records, "cascade")["mode_transitions"]
    assert transitions["missed_switches"] == 2
    assert transitions["unnecessary_switches"] == 1


def test_switch_delay_measures_how_late_the_mode_catches_up():
    def records(predicted):
        return [
            {"sequence": "work", "expected_mode": want, "expected_model": "m",
             "cascade": {"mode": got, "model": "m", "source": "phase", "elapsed_ms": 0}}
            for want, got in zip(("implement", "test", "test", "test"), predicted)
        ]
    late = summarize(records(("implement", "implement", "test", "test")), "cascade")
    assert late["mode_transitions"]["mean_switch_delay"] == 1
    never = summarize(records(("implement", "implement", "implement", "implement")), "cascade")
    assert never["mode_transitions"]["mean_switch_delay"] == 3


def test_records_without_a_sequence_are_not_a_transition():
    records = [
        {"expected_mode": "implement", "expected_model": "m",
         "cascade": {"mode": "implement", "model": "m", "source": "phase", "elapsed_ms": 0}},
        {"expected_mode": "debug", "expected_model": "m",
         "cascade": {"mode": "implement", "model": "m", "source": "fallback", "elapsed_ms": 0}},
    ]
    transitions = summarize(records, "cascade")["mode_transitions"]
    assert transitions["pairs"] == 0
    assert transitions["expected_switches"] == 0
    assert transitions["mean_switch_delay"] is None


def test_model_transitions_are_kept_as_an_appendix():
    records = []
    for expected, actual in (("small", "small"), ("large", "small"), ("large", "large")):
        records.append({
            "sequence": "work", "expected_mode": "implement", "expected_model": expected,
            "cascade": {"mode": "implement", "model": actual, "source": "phase", "elapsed_ms": 0},
        })
    transitions = summarize(records, "cascade")["model_transitions"]
    assert transitions["pairs"] == 2
    assert transitions["missed_switches"] == 1
    assert transitions["unnecessary_switches"] == 1


def test_evaluator_runs_semantics_even_when_cascade_uses_phase(config):
    case = next(case for case in load_cases(_DATASET, config, "holdout")
                if case["id"] == "holdout-executed-suite")
    router = stub_router(config)

    report = evaluate(config, [case], router)
    assert report["cascade"]["model_accuracy"] == 1
    assert report["semantic_only"]["model_accuracy"] == 0
    assert report["records"][0]["cascade"]["source"] == "phase"
    assert len(router.route_context.calls) == 1


def test_deterministic_variant_needs_no_router(config):
    cases = load_cases(_DATASET, config, "holdout")
    report = evaluate(config, cases)
    assert list(report) == [DETERMINISTIC_VARIANT, STATEFUL_VARIANT, "records"]
    for record in report["records"]:
        assert set(record) == {
            "id", "split", "sequence", "ambiguous", "expected_mode",
            "expected_model", DETERMINISTIC_VARIANT, STATEFUL_VARIANT,
        }


def test_the_stateful_variant_gets_a_fresh_store_per_sequence(config):
    """One session's memory must not leak into the next case or sequence."""
    report = evaluate(config, load_cases(_DATASET, config, "holdout"))
    stateful = report[STATEFUL_VARIANT]
    assert stateful["count"] == report[DETERMINISTIC_VARIANT]["count"]
    carried = [
        record["id"] for record in report["records"]
        if record[STATEFUL_VARIANT]["source"] == "memory"
    ]
    assert carried, "the stateful variant never carried a phase"
    # Every carried decision belongs to a case that follows one in its sequence.
    sequences = {}
    for record in report["records"]:
        if record["sequence"]:
            sequences.setdefault(record["sequence"], []).append(record["id"])
    inside = {
        identifier for items in sequences.values() for identifier in items[1:]
    }
    assert set(carried) <= inside


def test_the_stateful_variant_never_loses_a_stateless_success(config):
    """Memory may only add, so a case the stateless cascade got right stays right."""
    report = evaluate(config, load_cases(_DATASET, config, "holdout"))
    for record in report["records"]:
        stateless, stateful = record[DETERMINISTIC_VARIANT], record[STATEFUL_VARIANT]
        if stateless["mode"] == record["expected_mode"]:
            assert stateful["mode"] == record["expected_mode"], record["id"]


def test_semantic_layer_only_decides_where_the_deterministic_layers_abstain(config):
    cases = load_cases(_DATASET, config, "all")
    report = evaluate(config, cases, stub_router(config))
    for record in report["records"]:
        cascade, deterministic = record["cascade"], record[DETERMINISTIC_VARIANT]
        if cascade["source"] != "semantic":
            assert cascade["mode"] == deterministic["mode"], record["id"]


def test_replay_baseline_records_the_stateless_cascade(config):
    """The recorded cascade before the phase-evidence work, case by case.

    The file is the reference the later steps are compared against, so it has
    to stay reproducible as data.  A deliberate improvement regenerates it;
    :func:`test_no_case_loses_a_mode_it_had_right_at_the_baseline` is what
    turns an accidental regression red.
    """
    frozen = _frozen_baseline()
    assert frozen["variant"] == DETERMINISTIC_VARIANT
    for split, summary in frozen["summary"].items():
        recorded = [case for case in load_cases(_DATASET, config, "all")
                    if case["split"] == split]
        assert len(recorded) == summary["count"] + summary["ambiguous_count"]


def test_no_case_loses_a_mode_it_had_right_at_the_baseline(config):
    """Every case the baseline resolved correctly stays resolved.

    Improvements are expected and not asserted here; going backwards is not.
    The check runs per case rather than on aggregate accuracy, so one fix
    cannot pay for one regression.
    """
    frozen = _frozen_baseline()
    name = DETERMINISTIC_VARIANT
    for split in frozen["summary"]:
        current = {record["id"]: record for record in
                   evaluate(config, load_cases(_DATASET, config, split))["records"]}
        for identifier, was in frozen["per_case"].items():
            if was["split"] != split or was["mode"] != was["expected_mode"]:
                continue
            assert current[identifier][name]["mode"] == was["expected_mode"], identifier


def _frozen_baseline():
    """Load the recorded baseline replay of the corpus."""
    frozen = json.loads(
        (_ROOT / "tests/data/codex_routing_baseline.json").read_text(encoding="utf-8")
    )
    assert frozen["variant"] == DETERMINISTIC_VARIANT
    return frozen


def test_holdout_baseline_is_the_number_the_next_steps_must_beat(config):
    """Recorded 2026-10-06: the stateless cascade, on the untouched holdout."""
    report = evaluate(config, load_cases(_DATASET, config, "holdout"))
    baseline = report[DETERMINISTIC_VARIANT]
    assert baseline["mode_accuracy"] == 20 / 27
    assert baseline["ambiguous_count"] == 1
    # An incremental session loses the thread twice: once early, once at the
    # switch it should have made.
    assert baseline["mode_transitions"]["missed_switches"] == 1
    assert baseline["mode_transitions"]["unnecessary_switches"] == 1


def test_the_shared_memory_fixes_exactly_the_incremental_sessions(config):
    """The stateful variant is measured on the same holdout, not a new one."""
    report = evaluate(config, load_cases(_DATASET, config, "holdout"))
    stateless = report[DETERMINISTIC_VARIANT]
    stateful = report[STATEFUL_VARIANT]
    assert stateful["mode_accuracy"] > stateless["mode_accuracy"]
    assert stateful["mode_accuracy"] == 22 / 27
    assert stateful["mode_transitions"]["missed_switches"] == 0
    assert stateful["mode_transitions"]["unnecessary_switches"] == 0


def test_mode_decisions_do_not_depend_on_the_model_mapping(config):
    cases = load_cases(_DATASET, config, "all")
    before = evaluate(config, cases)
    after = evaluate(remapped(config, "one/model-for-everything"), cases)
    name = DETERMINISTIC_VARIANT
    assert [r[name]["mode"] for r in before["records"]] == [
        r[name]["mode"] for r in after["records"]
    ]
    assert before[name]["mode_confusion"] == after[name]["mode_confusion"]
    assert before[name]["mode_transitions"] == after[name]["mode_transitions"]
    assert before[name]["sources"] == after[name]["sources"]
    assert all(
        r[name]["model"] == "one/model-for-everything" for r in after["records"]
    )


def test_model_metrics_follow_the_mapping_while_mode_metrics_do_not(config):
    cases = load_cases(_DATASET, config, "holdout")
    name = DETERMINISTIC_VARIANT
    before = evaluate(config, cases)[name]
    after = evaluate(remapped(config, "single/model"), cases)[name]
    assert after["model_accuracy"] == 1
    assert before["mode_accuracy"] == after["mode_accuracy"]


def test_lookup_failure_cannot_be_reported_as_semantic_fallback(config):
    def broken(parts):
        raise RuntimeError("Index unavailable")

    cases = load_cases(_DATASET, config, "holdout")[:1]
    with pytest.raises(RuntimeError, match="Semantic lookup failed"):
        evaluate(config, cases, SimpleNamespace(route_context=broken))


def test_incomplete_ranking_is_not_a_quality_measurement(config):
    router = SimpleNamespace(route_context=lambda parts: {
        "target_name": "implement", "similarity": 0.9,
        "all_scores": [{"target": "implement", "similarity": 0.9}],
    })
    with pytest.raises(ValueError, match="semantic ranking"):
        evaluate(config, load_cases(_DATASET, config, "holdout")[:1], router)


def test_prefix_without_a_user_command_is_not_a_quality_measurement(config):
    """An empty main turn cannot be scored as a work mode."""
    case = copy.deepcopy(load_cases(_DATASET, config, "holdout")[0])
    case["input"] = [{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": "   "}]}]
    with pytest.raises(ValueError, match="non-empty main turn"):
        evaluate(config, [case])
