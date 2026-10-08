#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG="$ROOT/llm_router_plugins/resources/routing/agentic_routing_codex.json"
DATASET="$ROOT/tests/data/codex_routing_quality.json"
PYTHON=${PYTHON:-python3}
OUTPUT=""
BASELINE=""
THRESHOLDS="0.40 0.44 0.48 0.52 0.56"
MARGINS="0.001 0.005 0.02 0.05"
NO_SEMANTIC=false
MODULE=llm_router_plugins.utils.routing.agentic_routing.codex.evaluation

usage() {
    printf '%s\n' \
        'Usage: bash scripts/codex-tune-eval.sh [options]' \
        '  --config FILE       Source Codex config (never modified)' \
        '  --dataset FILE      Corpus with calibration and holdout splits' \
        '  --python EXECUTABLE Interpreter with the project and [ml] installed' \
        '  --output-dir DIR    New directory for configs, reports and logs' \
        '  --thresholds LIST   Space-separated values (default: 0.40 0.44 0.48 0.52 0.56)' \
        '  --margins LIST      Space-separated values (default: 0.001 0.005 0.02 0.05)' \
        '  --baseline FILE     Optional frozen baseline, used only on holdout' \
        '  --no-semantic       Deterministic eval only; no tuning or ML required' \
        '  --help              Show this help' \
        '' \
        'Selection: cascade main-mode accuracy, then fewer missed/unnecessary' \
        'mode switches, then overall mode accuracy. Exact ties keep the source' \
        'config (evaluated first). Holdout is NEVER used to select thresholds.' \
        'Each grid point runs a real replay and reloads the embedding model.'
}

die() { printf 'Error: %s\n' "$*" >&2; exit 1; }
while (($#)); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --no-semantic) NO_SEMANTIC=true; shift ;;
        --config|--dataset|--python|--output-dir|--thresholds|--margins|--baseline)
            (($# >= 2)) && [[ -n "$2" ]] || die "Missing value for $1"
            case "$1" in
                --config) CONFIG=$2 ;;
                --dataset) DATASET=$2 ;;
                --python) PYTHON=$2 ;;
                --output-dir) OUTPUT=$2 ;;
                --thresholds) THRESHOLDS=$2 ;;
                --margins) MARGINS=$2 ;;
                --baseline) BASELINE=$2 ;;
            esac
            shift 2 ;;
        *) die "Unknown option: $1" ;;
    esac
done

command -v jq >/dev/null || die 'jq is required'
command -v "$PYTHON" >/dev/null || die "Interpreter not found: $PYTHON"
# Resolve user paths before switching to the repository directory.
PYTHON=$(command -v "$PYTHON")
[[ "$PYTHON" = /* ]] || PYTHON="$PWD/$PYTHON"
CONFIG=$(realpath -e -- "$CONFIG")
DATASET=$(realpath -e -- "$DATASET")
if [[ -n "$BASELINE" ]]; then BASELINE=$(realpath -e -- "$BASELINE"); fi
read -r -a threshold_values <<< "$THRESHOLDS"
read -r -a margin_values <<< "$MARGINS"
if ! "$NO_SEMANTIC"; then
    ((${#threshold_values[@]} && ${#margin_values[@]})) || die 'Empty grid'
    for value in "${threshold_values[@]}" "${margin_values[@]}"; do
        jq -en --arg value "$value" \
            '$value | tonumber | . >= 0 and . <= 1' >/dev/null \
            || die "Grid values must be finite numbers between 0 and 1: $value"
    done
fi
jq -e '.cases | type == "array"' "$DATASET" >/dev/null || die 'Invalid corpus'
for split in calibration holdout; do
    jq -e --arg split "$split" \
        '[.cases[] | select(.split == $split and (.ambiguous != true))
          | select(.expected_mode != "aux_title" and .expected_mode != "compaction")]
         | length > 0' "$DATASET" >/dev/null \
        || die "No labeled main cases in $split"
done

OUTPUT=${OUTPUT:-"$ROOT/codex-eval-$(date +%Y%m%d-%H%M%S)-$$"}
[[ ! -e "$OUTPUT" ]] || die "Output directory already exists: $OUTPUT"
mkdir -p -- "$(dirname -- "$OUTPUT")"
mkdir -- "$OUTPUT"
OUTPUT=$(cd -- "$OUTPUT" && pwd)
cp -- "$CONFIG" "$OUTPUT/source-config.json"
cp -- "$DATASET" "$OUTPUT/dataset.json"
CONFIG="$OUTPUT/source-config.json"
DATASET="$OUTPUT/dataset.json"
if [[ -n "$BASELINE" ]]; then
    cp -- "$BASELINE" "$OUTPUT/baseline.json"
    BASELINE="$OUTPUT/baseline.json"
fi
cd -- "$ROOT"

run_eval() {
    local config=$1 split=$2 name=$3
    local args=(--config "$config" --dataset "$DATASET" --split "$split")
    if "$NO_SEMANTIC"; then args+=(--no-semantic); fi
    if [[ "$split" == holdout && -n "$BASELINE" ]]; then
        args+=(--baseline "$BASELINE")
    fi
    printf 'Evaluating %s (%s)\n' "$name" "$split" >&2
    if ! "$PYTHON" -m "$MODULE" "${args[@]}" \
        > "$OUTPUT/$name.json.partial" 2> "$OUTPUT/$name.log"; then
        die "Evaluation failed; see $OUTPUT/$name.log (run incomplete)"
    fi
    jq -e '.records | type == "array"' "$OUTPUT/$name.json.partial" >/dev/null \
        || die "Invalid evaluation report: $name"
    mv -- "$OUTPUT/$name.json.partial" "$OUTPUT/$name.json"
}

run_eval "$CONFIG" calibration calibration-source
BEST_CONFIG="$CONFIG"
if ! "$NO_SEMANTIC"; then
    append_result() {
        local config=$1 report=$2
        jq -ec --arg config "$config" --arg report "$report" '
            .cascade as $s
            | if ($s.main_mode_accuracy | type) != "number" then
                error("Missing cascade main-mode accuracy") else
                {config: $config, report: $report,
                 main_mode_accuracy: $s.main_mode_accuracy,
                 mode_accuracy: $s.mode_accuracy,
                 missed_switches: $s.mode_transitions.missed_switches,
                 unnecessary_switches: $s.mode_transitions.unnecessary_switches}
              end' "$report" >> "$OUTPUT/candidates.jsonl"
    }
    append_result "$CONFIG" "$OUTPUT/calibration-source.json"
    index=0
    for threshold in "${threshold_values[@]}"; do
        for margin in "${margin_values[@]}"; do
            index=$((index + 1))
            candidate="$OUTPUT/candidate-$index.json"
            jq --argjson threshold "$threshold" --argjson margin "$margin" \
                '.settings.semantic.threshold = $threshold
                 | .settings.semantic.min_margin = $margin' "$CONFIG" > "$candidate"
            run_eval "$candidate" calibration "calibration-$index"
            append_result "$candidate" "$OUTPUT/calibration-$index.json"
        done
    done
    jq -s 'to_entries | sort_by([
        -.value.main_mode_accuracy, .value.missed_switches,
        .value.unnecessary_switches, -.value.mode_accuracy, .key])
        | map(.value)' "$OUTPUT/candidates.jsonl" > "$OUTPUT/selection.json"
    BEST_CONFIG=$(jq -r '.[0].config' "$OUTPUT/selection.json")
fi

# Selection is frozen before looking at either holdout report.
cp -- "$BEST_CONFIG" "$OUTPUT/selected-config.json"
run_eval "$CONFIG" holdout holdout-source
run_eval "$OUTPUT/selected-config.json" holdout holdout-selected
jq -n --slurpfile source "$OUTPUT/holdout-source.json" \
    --slurpfile selected "$OUTPUT/holdout-selected.json" \
    '{source: ($source[0] | del(.records)),
      selected: ($selected[0] | del(.records))}' > "$OUTPUT/holdout-comparison.json"
printf 'Done: %s\nSelected config: %s/selected-config.json\n' "$OUTPUT" "$OUTPUT"