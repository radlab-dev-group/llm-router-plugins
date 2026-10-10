"""Offline evaluation plumbing, not embedding accuracy measurements."""

import copy
import json
import pathlib
import random
import sys
from dataclasses import replace
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

from llm_router_plugins.utils.routing.agentic_routing.codex.evaluation import (
    DETERMINISTIC_VARIANT,
    STATEFUL_VARIANT,
    build_payload,
    compare_baseline,
    evaluate,
    load_cases,
    main,
    summarize,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.config import (
    CodexRoutingConfig,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.payload import (
    CodexPayloadParser,
)

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_DATASET = _ROOT / "tests/data/codex_routing_quality.json"


@pytest.fixture
def config():
    path = _ROOT / "llm_router_plugins/resources/routing/agentic_routing_codex.json"
    result = CodexRoutingConfig._from_raw(
        json.loads(path.read_text(encoding="utf-8"))
    )
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
            "target_name": winner,
            "similarity": similarity,
            "all_scores": [
                {
                    "target": mode.name,
                    "similarity": (
                        similarity if mode.name == winner else similarity - 0.5
                    ),
                }
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
    """Every case parses as the request class that its label measures."""
    parser = CodexPayloadParser(max_chars=config.classify_max_chars)
    for case in load_cases(_DATASET, config, "all"):
        request = parser.parse(build_payload(case, config))
        expected_class = (
            case["expected_mode"]
            if case["expected_mode"] in ("aux_title", "compaction")
            else "main"
        )
        assert request.request_class == expected_class, case["id"]
        if expected_class == "main":
            assert request.intent_text.strip(), case["id"]


def test_corpus_labels_and_holdout_are_separate_from_examples(config):
    cases = load_cases(_DATASET, config, "all")
    assert {case["expected_mode"] for case in cases} == {
        "plan",
        "implement",
        "test",
        "review",
        "git_review",
        "debug",
        "aux_title",
        "compaction",
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


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "mode",
        "split",
        "input",
        "sequence",
        "shared_sequence_split",
        "metadata",
        "ambiguous",
    ],
)
def test_invalid_corpus_is_rejected(config, tmp_path, mutation):
    case = {
        "id": "example",
        "split": "holdout",
        "expected_mode": "implement",
        "input": [{"role": "user", "content": "Add a field."}],
    }
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
        cases.append(
            {
                "id": "second",
                "split": "calibration",
                "expected_mode": "implement",
                "sequence": "example",
                "input": [{"role": "user", "content": "Add a field."}],
            }
        )
    elif mutation == "metadata":
        case["metadata"] = "session-1"
    else:
        case["ambiguous"] = "yes"
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps({"schema_version": 1, "cases": cases}), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load_cases(
            path, config, "holdout" if mutation != "shared_sequence_split" else "all"
        )


def test_empty_selection_is_rejected(config, tmp_path):
    path = tmp_path / "cases.json"
    path.write_text('{"schema_version":1,"cases":[]}', encoding="utf-8")
    with pytest.raises(ValueError, match="No cases"):
        load_cases(path, config, "holdout")


def test_case_metadata_reaches_the_replayed_payload(config):
    case = {
        "id": "x",
        "split": "holdout",
        "expected_mode": "implement",
        "input": [{"role": "user", "content": "Add a field."}],
        "metadata": {"session_id": "s", "thread_id": "t", "turn_id": "u"},
    }
    payload = build_payload(case, config)
    assert payload["model"] == config.trigger_model
    assert payload["client_metadata"] == {
        "session_id": "s",
        "thread_id": "t",
        "turn_id": "u",
    }


def test_mode_errors_are_not_necessarily_model_errors():
    records = [
        {
            "expected_mode": "review",
            "expected_model": "large",
            "cascade": {
                "mode": "implement",
                "model": "large",
                "source": "fallback",
                "elapsed_ms": 1,
            },
        },
        {
            "expected_mode": "test",
            "expected_model": "small",
            "cascade": {
                "mode": "review",
                "model": "large",
                "source": "semantic",
                "elapsed_ms": 3,
            },
        },
    ]
    report = summarize(records, "cascade")
    assert report["mode_accuracy"] == 0
    assert report["model_accuracy"] == 0.5
    assert report["model_confusion"]["small"]["large"] == 1
    assert report["mode_confusion"]["test"]["review"] == 1
    assert report["per_mode"]["review"]["precision"] == 0
    assert report["mean_routing_ms"] == 2


def test_ambiguous_cases_are_reported_separately():
    record = {
        "expected_mode": "implement",
        "expected_model": "large",
        "cascade": {
            "mode": "implement",
            "model": "large",
            "source": "phase",
            "elapsed_ms": 0,
        },
    }
    clear = dict(record, sequence="a")
    uncertain = dict(record, id="u", sequence="b", ambiguous=True)
    report = summarize([clear, uncertain], "cascade")
    assert report["count"] == 1
    assert report["ambiguous_count"] == 1
    assert report["mode_accuracy"] == 1
    assert report["mode_transitions"]["pairs"] == 0


def test_fallback_and_ambiguous_rates_are_reported():
    """The two refusal rates are the numbers the fallback fix is measured by."""
    record = {
        "expected_mode": "implement",
        "expected_model": "large",
        "cascade": {
            "mode": "implement",
            "model": "large",
            "source": "fallback",
            "elapsed_ms": 0,
        },
    }
    decided = dict(
        record,
        id="d",
        cascade=dict(record["cascade"], source="phase"),
    )
    fell_back = dict(record, id="f", cascade=dict(record["cascade"], reason="no layer answered"))
    uncertain = dict(record, id="u", ambiguous=True)
    report = summarize([decided, fell_back, uncertain], "cascade")
    assert report["fallback_rate"] == pytest.approx(0.5)
    assert report["ambiguous_rate"] == pytest.approx(1 / 3)
    assert report["fallback_reasons"] == {"no layer answered": 1}
    clean = summarize([decided], "cascade")
    assert clean["fallback_rate"] == 0
    assert clean["ambiguous_rate"] == 0


def test_sequence_metrics_track_expected_mode_boundary():
    records = []
    for mode in ("implement", "test", "test", "debug"):
        records.append(
            {
                "sequence": "work",
                "expected_mode": mode,
                "expected_model": "m",
                "cascade": {
                    "mode": mode,
                    "model": "m",
                    "source": "phase",
                    "elapsed_ms": 0,
                },
            }
        )
    transitions = summarize(records, "cascade")["mode_transitions"]
    assert transitions == {
        "pairs": 3,
        "expected_switches": 2,
        "actual_switches": 2,
        "unnecessary_switches": 0,
        "missed_switches": 0,
        "mean_switch_delay": 0.0,
        "censored_switches": 0,
    }


def test_unnecessary_and_missed_mode_switches_are_counted():
    expected = ("implement", "test", "implement", "implement")
    predicted = ("implement", "implement", "implement", "debug")
    records = [
        {
            "sequence": "work",
            "expected_mode": want,
            "expected_model": "m",
            "cascade": {
                "mode": got,
                "model": "m",
                "source": "fallback",
                "elapsed_ms": 0,
            },
        }
        for want, got in zip(expected, predicted)
    ]
    transitions = summarize(records, "cascade")["mode_transitions"]
    assert transitions["missed_switches"] == 2
    assert transitions["unnecessary_switches"] == 1


def test_switch_delay_measures_how_late_the_mode_catches_up():
    def records(predicted):
        return [
            {
                "sequence": "work",
                "expected_mode": want,
                "expected_model": "m",
                "cascade": {
                    "mode": got,
                    "model": "m",
                    "source": "phase",
                    "elapsed_ms": 0,
                },
            }
            for want, got in zip(("implement", "test", "test", "test"), predicted)
        ]

    late = summarize(records(("implement", "implement", "test", "test")), "cascade")
    assert late["mode_transitions"]["mean_switch_delay"] == 1
    never = summarize(
        records(("implement", "implement", "implement", "implement")), "cascade"
    )
    assert never["mode_transitions"]["mean_switch_delay"] == 3


def test_records_without_a_sequence_are_not_a_transition():
    records = [
        {
            "expected_mode": "implement",
            "expected_model": "m",
            "cascade": {
                "mode": "implement",
                "model": "m",
                "source": "phase",
                "elapsed_ms": 0,
            },
        },
        {
            "expected_mode": "debug",
            "expected_model": "m",
            "cascade": {
                "mode": "implement",
                "model": "m",
                "source": "fallback",
                "elapsed_ms": 0,
            },
        },
    ]
    transitions = summarize(records, "cascade")["mode_transitions"]
    assert transitions["pairs"] == 0
    assert transitions["expected_switches"] == 0
    assert transitions["mean_switch_delay"] is None


def test_model_transitions_are_kept_as_an_appendix():
    records = []
    for expected, actual in (
        ("small", "small"),
        ("large", "small"),
        ("large", "large"),
    ):
        records.append(
            {
                "sequence": "work",
                "expected_mode": "implement",
                "expected_model": expected,
                "cascade": {
                    "mode": "implement",
                    "model": actual,
                    "source": "phase",
                    "elapsed_ms": 0,
                },
            }
        )
    transitions = summarize(records, "cascade")["model_transitions"]
    assert transitions["pairs"] == 2
    assert transitions["missed_switches"] == 1
    assert transitions["unnecessary_switches"] == 1


def test_evaluator_runs_semantics_even_when_cascade_uses_phase(config):
    case = next(
        case
        for case in load_cases(_DATASET, config, "holdout")
        if case["id"] == "holdout-executed-suite"
    )
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
            "id",
            "split",
            "sequence",
            "ambiguous",
            "expected_mode",
            "expected_model",
            DETERMINISTIC_VARIANT,
            STATEFUL_VARIANT,
            "provenance",
            "source_session",
        }


def test_the_stateful_variant_gets_a_fresh_store_per_sequence(config):
    """One session's memory must not leak into the next case or sequence."""
    report = evaluate(config, load_cases(_DATASET, config, "holdout"))
    stateful = report[STATEFUL_VARIANT]
    assert stateful["count"] == report[DETERMINISTIC_VARIANT]["count"]
    carried = [
        record["id"]
        for record in report["records"]
        if record[STATEFUL_VARIANT]["source"] == "memory"
    ]
    assert carried, "the stateful variant never carried a phase"
    # Every carried decision belongs to a case that follows one in its sequence.
    sequences = {}
    for record in report["records"]:
        if record["sequence"]:
            sequences.setdefault(record["sequence"], []).append(record["id"])
    inside = {identifier for items in sequences.values() for identifier in items[1:]}
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
    """Frozen decisions stay intact; new cases have no historical prediction."""
    frozen = _frozen_baseline()
    assert frozen["variant"] == DETERMINISTIC_VARIANT
    for split, summary in frozen["summary"].items():
        recorded = [
            case
            for case in load_cases(_DATASET, config, "all")
            if case["split"] == split and case["id"] in frozen["per_case"]
        ]
        assert len(recorded) == summary["count"] + summary["ambiguous_count"]
        for case in recorded:
            previous = frozen["per_case"][case["id"]]
            for key in ("expected_mode", "split", "sequence"):
                assert previous.get(key) == case.get(key)
            assert previous["ambiguous"] == bool(case.get("ambiguous"))
    assert frozen["provenance"]["verified_pre_change"] is False


def test_no_case_loses_a_mode_it_had_right_at_the_baseline(config):
    """Every case the baseline resolved correctly stays resolved.

    Improvements are expected and not asserted here; going backwards is not.
    The check runs per case rather than on aggregate accuracy, so one fix
    cannot pay for one regression.
    """
    frozen = _frozen_baseline()
    name = DETERMINISTIC_VARIANT
    for split in frozen["summary"]:
        current = {
            record["id"]: record
            for record in evaluate(config, load_cases(_DATASET, config, split))[
                "records"
            ]
        }
        for identifier, was in frozen["per_case"].items():
            if (
                was["split"] != split
                or was["ambiguous"]
                or was["mode"] != was["expected_mode"]
            ):
                continue
            for variant in (name, STATEFUL_VARIANT):
                assert (
                    current[identifier][variant]["mode"] == was["expected_mode"]
                ), identifier


def _frozen_baseline():
    """Load the recorded baseline replay of the corpus."""
    frozen = json.loads(
        (_ROOT / "tests/data/codex_routing_baseline.json").read_text(
            encoding="utf-8"
        )
    )
    assert frozen["variant"] == DETERMINISTIC_VARIANT
    return frozen


def test_holdout_baseline_is_the_number_the_next_steps_must_beat(config):
    """Compare the snapshot only on common cases, using one metric definition."""
    report = evaluate(config, load_cases(_DATASET, config, "holdout"))
    comparison = compare_baseline(config, report["records"], _frozen_baseline())
    baseline = comparison["variants"]["baseline"]
    assert comparison["common_count"] == 28
    # The 28 common holdout cases were in the baseline; the added ones are
    # 8 conv-02 prefixes and the 27 minimized conv-03 goal-turn prefixes.
    assert len(comparison["added_case_ids"]) == 35
    assert baseline["mode_accuracy"] == 20 / 27
    assert baseline["ambiguous_count"] == 1
    # An incremental session loses the thread twice: once early, once at the
    # switch it should have made.
    assert baseline["mode_transitions"]["missed_switches"] == 1
    assert baseline["mode_transitions"]["unnecessary_switches"] == 1
    current = comparison["variants"][DETERMINISTIC_VARIANT]
    assert current["mode_accuracy"] >= baseline["mode_accuracy"]
    # The conv-03 goal turn is what the fallback fix targets: every prefix of
    # it is now decided by the phase layer, so no conv-03 request may fall back.
    conv03 = [
        record
        for record in report["records"]
        if record["id"].startswith("conv03-goal-")
    ]
    assert conv03
    assert all(
        record[DETERMINISTIC_VARIANT]["source"] != "fallback" for record in conv03
    )


def test_the_shared_memory_fixes_exactly_the_incremental_sessions(config):
    """The stateful variant is measured on the same holdout, not a new one."""
    report = evaluate(config, load_cases(_DATASET, config, "holdout"))
    stateless = report[DETERMINISTIC_VARIANT]
    stateful = report[STATEFUL_VARIANT]
    assert stateful["main_mode_accuracy"] > stateless["main_mode_accuracy"]
    comparison = compare_baseline(config, report["records"], _frozen_baseline())
    assert comparison["variants"][STATEFUL_VARIANT]["mode_accuracy"] >= 22 / 27
    assert stateful["count"] == stateless["count"]
    assert stateful["ambiguous_count"] == stateless["ambiguous_count"]
    assert stateful["mode_transitions"]["missed_switches"] == 0
    assert stateful["mode_transitions"]["unnecessary_switches"] == 0


def test_mode_decisions_do_not_depend_on_the_model_mapping(config):
    cases = load_cases(_DATASET, config, "all")
    before = evaluate(config, cases)
    after = evaluate(remapped(config, "one/model-for-everything"), cases)
    for name in (DETERMINISTIC_VARIANT, STATEFUL_VARIANT):
        assert [r[name]["mode"] for r in before["records"]] == [
            r[name]["mode"] for r in after["records"]
        ]
        for key in (
            "mode_confusion",
            "mode_transitions",
            "sources",
            "per_mode",
            "mode_accuracy",
            "special_cases",
        ):
            assert before[name][key] == after[name][key]
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
    router = SimpleNamespace(
        route_context=lambda parts: {
            "target_name": "implement",
            "similarity": 0.9,
            "all_scores": [{"target": "implement", "similarity": 0.9}],
        }
    )
    with pytest.raises(ValueError, match="semantic ranking"):
        evaluate(config, load_cases(_DATASET, config, "holdout")[:1], router)


def test_prefix_without_a_user_command_is_not_a_quality_measurement(config):
    """An empty main turn cannot be scored as a work mode."""
    case = copy.deepcopy(load_cases(_DATASET, config, "holdout")[0])
    case["input"] = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "   "}],
        }
    ]
    with pytest.raises(ValueError, match="non-empty main turn"):
        evaluate(config, [case])


def test_session_split_is_checked_even_with_different_sequence_names(
    config, tmp_path
):
    cases = [
        {
            "id": split,
            "split": split,
            "sequence": split,
            "expected_mode": "implement",
            "metadata": {"session_id": "same"},
            "input": [{"role": "user", "content": "Add a field."}],
        }
        for split in ("calibration", "holdout")
    ]
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"schema_version": 1, "cases": cases}))
    with pytest.raises(ValueError, match="Session.*spans two splits"):
        load_cases(path, config, "holdout")


def test_replay_preserves_payload_fields_without_aliasing(config):
    case = {
        "input": [{"role": "user", "content": "Add a field."}],
        "payload": {
            "tools": [{"type": "function", "name": "exec_command"}],
            "text": {"format": {"type": "json_schema"}},
            "client_metadata": {"request_kind": "compact"},
        },
    }
    original = copy.deepcopy(case)
    payload = build_payload(case, config)
    assert payload["tools"] == case["payload"]["tools"]
    assert payload["client_metadata"] == case["payload"]["client_metadata"]
    payload["tools"].clear()
    payload["input"].clear()
    assert case == original


def test_ambiguous_prefix_is_a_transition_boundary_not_a_removed_step():
    records = [
        {
            "sequence": "s",
            "ambiguous": ambiguous,
            "expected_mode": expected,
            "expected_model": "m",
            "cascade": {
                "mode": actual,
                "model": "m",
                "source": "phase",
                "elapsed_ms": 0,
            },
        }
        for ambiguous, expected, actual in (
            (False, "implement", "implement"),
            (True, "implement", "test"),
            (False, "test", "test"),
        )
    ]
    metrics = summarize(records, "cascade")["mode_transitions"]
    assert metrics["pairs"] == 0
    assert metrics["expected_switches"] == 0
    assert metrics["mean_switch_delay"] is None


def test_delay_is_censored_at_ambiguous_boundary():
    records = [
        {
            "sequence": "s",
            "ambiguous": ambiguous,
            "expected_mode": expected,
            "expected_model": "m",
            "cascade": {
                "mode": actual,
                "model": "m",
                "source": "phase",
                "elapsed_ms": 0,
            },
        }
        for ambiguous, expected, actual in (
            (False, "implement", "implement"),
            (False, "test", "implement"),
            (True, "test", "implement"),
            (False, "test", "test"),
        )
    ]
    metrics = summarize(records, "cascade")["mode_transitions"]
    assert metrics["mean_switch_delay"] == 1
    assert metrics["censored_switches"] == 1


def test_holdout_special_requests_preserve_the_main_phase(config):
    cases = [
        case
        for case in load_cases(_DATASET, config, "holdout")
        if case.get("sequence") == "holdout-special-memory"
    ]
    router = stub_router(config)
    report = evaluate(config, cases, router)
    # One semantic-only lookup for the test call; cascade and semantic-only
    # both look up the neutral main continuation. Neither special call does.
    assert len(router.route_context.calls) == 3
    assert router.route_context.calls == [
        ("Dodaj walidację numeru przesyłki.", "ran python -m pytest"),
        ("Dodaj walidację numeru przesyłki.", "Kontynuuję."),
        ("Dodaj walidację numeru przesyłki.", "Kontynuuję."),
    ]
    by_id = {record["id"]: record for record in report["records"]}
    for variant in (
        DETERMINISTIC_VARIANT,
        STATEFUL_VARIANT,
        "cascade",
        "semantic_only",
    ):
        for identifier, mode in (
            ("holdout-special-title", "aux_title"),
            ("holdout-special-compaction", "compaction"),
        ):
            assert by_id[identifier][variant]["mode"] == mode
            assert by_id[identifier][variant]["source"] == "class"
        assert report[variant]["special_cases"]["correct"] == 2
        assert report[variant]["special_cases"]["count"] == 2
        assert report[variant]["mode_transitions"]["pairs"] == 1
    resumed = by_id["holdout-special-resume"][STATEFUL_VARIANT]
    assert resumed["mode"] == "test"
    assert resumed["source"] == "memory"
    assert report[STATEFUL_VARIANT]["mode_transitions"]["actual_switches"] == 0


def test_holdout_explicit_and_plan_priority(config):
    cases = [
        case
        for case in load_cases(_DATASET, config, "holdout")
        if case["id"] in ("holdout-explicit-priority", "holdout-plan-priority")
    ]
    report = evaluate(config, cases, stub_router(config, winner="debug"))
    for record in report["records"]:
        for variant in (DETERMINISTIC_VARIANT, STATEFUL_VARIANT, "cascade"):
            assert record[variant]["mode"] == record["expected_mode"]
            assert record[variant]["source"] == (
                "explicit"
                if record["id"] == "holdout-explicit-priority"
                else "collaboration_mode"
            )


def test_captured_title_metadata_is_preserved(config):
    case = next(
        case
        for case in load_cases(_DATASET, config, "holdout")
        if case["id"] == "holdout-log-title"
    )
    payload = build_payload(case, config)
    request = CodexPayloadParser().parse(payload)
    assert request.session_id == "replay-conv-02"
    assert request.thread_id == "replay-conv-02"
    assert request.turn_id == "turn-review"
    assert request.root_turn_id == "turn-review"
    assert request.agent_name == "/root"
    assert request.window_id == "replay-conv-02:0"
    assert request.context_window_id == "context-1"
    assert request.thread_source == "thread_title"
    assert request.request_kind == "turn"
    assert request.request_class == "aux_title"
    assert payload["tools"] == []
    assert payload["text"]["format"]["type"] == "json_schema"


@pytest.mark.parametrize("identity", ["header", "source_session"])
def test_whole_session_split_includes_header_and_log_origin(
    config, tmp_path, identity
):
    cases = []
    for split in ("calibration", "holdout"):
        case = {
            "id": split,
            "split": split,
            "expected_mode": "implement",
            "input": [{"role": "user", "content": "Add a field."}],
        }
        if identity == "header":
            case["payload"] = {
                "client_metadata": {
                    "x-codex-turn-metadata": json.dumps(
                        {"session_id": "same-session"}
                    )
                }
            }
        else:
            case["source_session"] = "same-log"
        cases.append(case)
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"schema_version": 1, "cases": cases}))
    with pytest.raises(ValueError, match="Session.*spans two splits"):
        load_cases(path, config, "holdout")


def test_baseline_comparison_does_not_invent_added_case_predictions(config):
    records = evaluate(config, load_cases(_DATASET, config, "holdout"))["records"]
    frozen = _frozen_baseline()
    original = copy.deepcopy(frozen)
    comparison = compare_baseline(config, records, frozen)
    assert frozen == original
    assert comparison["baseline_only_case_ids"] == []
    counts = {summary["count"] for summary in comparison["variants"].values()}
    assert counts == {27}
    assert "holdout-log-title" in comparison["added_case_ids"]
    assert comparison["variants"]["baseline"]["mean_routing_ms"] is None
    changed = copy.deepcopy(records)
    changed[0]["expected_mode"] = "debug"
    with pytest.raises(ValueError, match="Baseline case changed"):
        compare_baseline(config, changed, frozen)


def test_baseline_mode_metrics_survive_one_model_mapping(config):
    cases = load_cases(_DATASET, config, "holdout")
    before = compare_baseline(
        config, evaluate(config, cases)["records"], _frozen_baseline()
    )
    other = remapped(config, "one/model")
    after = compare_baseline(
        other, evaluate(other, cases)["records"], _frozen_baseline()
    )
    for variant in before["variants"]:
        for key in (
            "mode_accuracy",
            "mode_transitions",
            "mode_confusion",
            "per_mode",
        ):
            assert (
                before["variants"][variant][key] == after["variants"][variant][key]
            )


def test_fallback_reasons_and_ambiguous_sources_are_separate():
    records = [
        {
            "expected_mode": "implement",
            "expected_model": "m",
            "ambiguous": ambiguous,
            "cascade": {
                "mode": "implement",
                "model": "m",
                "source": "fallback",
                "reason": reason,
                "elapsed_ms": 0,
            },
        }
        for ambiguous, reason in (
            (False, "semantic_disabled"),
            (False, "semantic_ambiguous"),
            (True, "unknown"),
        )
    ]
    metrics = summarize(records, "cascade")
    assert metrics["fallback_reasons"] == {
        "semantic_disabled": 1,
        "semantic_ambiguous": 1,
    }
    assert metrics["ambiguous_sources"] == {"fallback": 1}


def test_cascade_lookup_outage_cannot_be_hidden_by_diagnostic_success(config):
    healthy = stub_router(config)
    calls = []

    def fail_once(parts):
        calls.append(parts)
        if len(calls) == 1:
            raise RuntimeError("Temporary index failure")
        return healthy.route_context(parts)

    case = next(
        case
        for case in load_cases(_DATASET, config, "holdout")
        if case["id"] == "holdout-specification"
    )
    with pytest.raises(RuntimeError, match="Semantic lookup failed"):
        evaluate(config, [case], SimpleNamespace(route_context=fail_once))


def test_seeded_model_permutation_preserves_live_and_baseline_mode_metrics(config):
    rng = random.Random(817)
    models = [mode.model_name for mode in config.codex_modes]
    rng.shuffle(models)
    other = replace(
        config,
        codex_modes=tuple(
            replace(mode, model_name=model)
            for mode, model in zip(config.codex_modes, models)
        ),
    )
    cases = load_cases(_DATASET, config, "holdout")
    before = evaluate(config, cases)
    after = evaluate(other, cases)
    for variant in (DETERMINISTIC_VARIANT, STATEFUL_VARIANT):
        for key in (
            "mode_accuracy",
            "mode_confusion",
            "mode_transitions",
            "per_mode",
        ):
            assert before[variant][key] == after[variant][key]
        for record in after["records"]:
            result = record[variant]
            assert result["model"] == other.mode_by_name[result["mode"]].model_name
    old = compare_baseline(config, before["records"], _frozen_baseline())
    new = compare_baseline(other, after["records"], _frozen_baseline())
    for variant in old["variants"]:
        assert (
            old["variants"][variant]["mode_transitions"]
            == new["variants"][variant]["mode_transitions"]
        )


def test_cli_reports_comparable_baseline_and_replay_metadata(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluation",
            "--config",
            str(
                _ROOT
                / "llm_router_plugins/resources/routing/agentic_routing_codex.json"
            ),
            "--dataset",
            str(_DATASET),
            "--baseline",
            str(_ROOT / "tests/data/codex_routing_baseline.json"),
            "--split",
            "holdout",
            "--no-semantic",
        ],
    )
    main()
    report = json.loads(capsys.readouterr().out)
    assert report["metadata"]["metrics_version"] == 2
    for key in ("config_sha256", "dataset_sha256", "baseline_sha256"):
        assert len(report["metadata"][key]) == 64
    assert report["metadata"]["environment_overrides"] is False
    assert report["baseline_comparison"]["common_count"] == 28
    assert report[STATEFUL_VARIANT]["special_cases"]["count"] == 3
    assert report[STATEFUL_VARIANT]["special_cases"]["correct"] == 3
