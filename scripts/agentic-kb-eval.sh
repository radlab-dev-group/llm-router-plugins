#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG="$ROOT/llm_router_plugins/resources/knowledge/agentic_knowledge.json"
DATASET="$ROOT/tests/data/agentic_knowledge_eval.json"
PYTHON=${PYTHON:-python3}
OUTPUT=""
MODULE=llm_router_plugins.utils.knowledge.agentic_knowledge.evaluation

usage() {
    printf '%s\n' \
        'Usage: bash scripts/agentic-kb-eval.sh [options]' \
        '  --dataset FILE      Recorded sessions to replay (JSON)' \
        '  --config FILE       Knowledge policy JSON (escalation thresholds, budgets)' \
        '  --python EXECUTABLE Interpreter with the project installed' \
        '  --output FILE       Write the JSON report here' \
        '  --enable-escalation Measure the read+escalate variant with escalation on' \
        '  --help              Show this help' \
        '' \
        'Replays each session through the real read path against an in-memory' \
        'engine, feeding a turn only the knowledge mined from earlier turns. The' \
        'report compares no-knowledge, read-only and read-plus-escalation.'
}

die() { printf 'Error: %s\n' "$*" >&2; exit 1; }

while (($#)); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --enable-escalation) export LLM_ROUTER_KNOWLEDGE_AGENTIC_ESCALATION_ENABLED=true; shift ;;
        --dataset|--config|--python|--output)
            (($# >= 2)) && [[ -n "$2" ]] || die "Missing value for $1"
            case "$1" in
                --dataset) DATASET=$2 ;;
                --config) CONFIG=$2 ;;
                --python) PYTHON=$2 ;;
                --output) OUTPUT=$2 ;;
            esac
            shift 2 ;;
        *) die "Unknown option: $1" ;;
    esac
done

command -v "$PYTHON" >/dev/null || die "Interpreter not found: $PYTHON"
PYTHON=$(command -v "$PYTHON")
[[ -f "$DATASET" ]] || die "Dataset not found: $DATASET"
[[ -f "$CONFIG" ]] || die "Config not found: $CONFIG"

ARGS=(--dataset "$DATASET" --config "$CONFIG")
if [[ -n "$OUTPUT" ]]; then ARGS+=(--output "$OUTPUT"); fi

exec "$PYTHON" -m "$MODULE" "${ARGS[@]}"
