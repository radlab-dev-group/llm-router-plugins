import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/codex-tune-eval.sh"


@pytest.fixture
def runner(tmp_path):
    config = tmp_path / "source config.json"
    config.write_text(json.dumps({"settings": {"semantic": {
        "threshold": 0.51, "min_margin": 0.05,
    }}}))
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps({"cases": [
        {"split": split, "expected_mode": "review"}
        for split in ("calibration", "holdout")
    ]}))
    # A process-boundary fixture: controllable scores test Bash selection,
    # argument forwarding and failure handling without loading an ML model.
    interpreter = tmp_path / "fixture interpreter"
    interpreter.write_text('''#!/usr/bin/env bash
set -euo pipefail
shift 2
config=""; split=""; baseline=""
while (($#)); do
    case "$1" in
        --config) config=$2; shift 2 ;;
        --split) split=$2; shift 2 ;;
        --baseline) baseline=$2; shift 2 ;;
        --dataset) shift 2 ;;
        *) exit 98 ;;
    esac
done
printf '%s|%s|%s\\n' "$split" "$config" "$baseline" >> "$CALLS"
if [[ ${FAIL_EVAL:-0} == 1 ]]; then printf 'backend unavailable\\n' >&2; exit 7; fi
jq --arg split "$split" --arg tie "${TIE:-0}" '
    .settings.semantic.threshold as $t
    | {records: [], cascade: {
        main_mode_accuracy: (if $tie == "1" then 0.7
            elif $split == "holdout" then (1 - $t) else $t end),
        mode_accuracy: $t,
        mode_transitions: {missed_switches: 0, unnecessary_switches: 0}}}
    | if $tie == "1" then .cascade.mode_accuracy = 0.7 else . end' "$config"
''')
    interpreter.chmod(0o755)
    output = tmp_path / "output reports"
    calls = tmp_path / "calls"

    def run(*args, **env):
        return subprocess.run(
            ["bash", str(SCRIPT), "--config", str(config),
             "--dataset", str(dataset), "--python", str(interpreter),
             "--output-dir", str(output), "--thresholds", "0.45 0.60",
             "--margins", "0.05", *args],
            cwd=tmp_path, env={**os.environ, "CALLS": str(calls), **env},
            capture_output=True, text=True,
        )

    return run, config, output, calls


def test_selects_using_calibration_only_and_preserves_source(runner):
    run, config, output, calls = runner
    before = config.read_bytes()
    result = run()
    assert result.returncode == 0, result.stderr
    assert config.read_bytes() == before
    selected = json.loads((output / "selected-config.json").read_text())
    assert selected["settings"]["semantic"]["threshold"] == 0.60
    assert [line.split("|")[0] for line in calls.read_text().splitlines()] == [
        "calibration", "calibration", "calibration", "holdout", "holdout",
    ]
    assert (output / "holdout-comparison.json").exists()


def test_exact_tie_keeps_source_config(runner):
    run, config, output, _ = runner
    assert run(TIE="1").returncode == 0
    assert (output / "selected-config.json").read_bytes() == config.read_bytes()


def test_failure_retains_log_and_does_not_publish_selection(runner):
    run, _, output, _ = runner
    result = run(FAIL_EVAL="1")
    assert result.returncode != 0
    assert "backend unavailable" in (output / "calibration-source.log").read_text()
    assert not (output / "selected-config.json").exists()


@pytest.mark.parametrize("arguments", [
    ["--thresholds", "bad"], ["--margins", "-1"],
    ["--thresholds", "1.1"], ["--unknown"], ["--dataset"],
])
def test_invalid_arguments_fail_before_evaluation(runner, arguments):
    run, _, _, calls = runner
    assert run(*arguments).returncode != 0
    assert not calls.exists()


def test_existing_output_is_not_overwritten(runner):
    run, _, output, calls = runner
    output.mkdir()
    marker = output / "keep"
    marker.write_text("unchanged")
    assert run().returncode != 0
    assert marker.read_text() == "unchanged"
    assert not calls.exists()


def test_baseline_is_only_used_on_holdout(runner, tmp_path):
    run, _, _, calls = runner
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{}")
    result = run("--baseline", str(baseline))
    assert result.returncode == 0, result.stderr
    lines = [line.split("|") for line in calls.read_text().splitlines()]
    assert all(not line[2] for line in lines[:3])
    assert all(line[2].endswith("baseline.json") for line in lines[3:])


def test_real_deterministic_evaluation(tmp_path):
    output = tmp_path / "real reports"
    result = subprocess.run(
        ["bash", str(SCRIPT), "--python", sys.executable,
         "--output-dir", str(output), "--no-semantic"],
        cwd=tmp_path, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((output / "holdout-selected.json").read_text())
    assert report["deterministic"]["main_turn_count"] > 0
    assert "cascade" not in report
    assert not (output / "selection.json").exists()