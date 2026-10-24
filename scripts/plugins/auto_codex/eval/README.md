# Codex routing — offline corpus and replay

Tools for evaluating `agentic_routing_codex` against **recorded Codex CLI
traffic** instead of synthetic payloads. Nothing here calls a model provider:
the corpus is replayed through the routing code in the checkout, so a change in
the cascade shows up as a change in the numbers rather than being blended with
the decisions the recording server happened to make.

Recorded traffic lives in `agents-conversation/codex/` (router debug logs
containing full request bodies). The corpus and decision files are generated
artifacts — keep them in `/tmp` or another scratch directory, they are large
(793 requests ≈ 180 MB of JSON).

## Pipeline

```bash
EVAL=scripts/plugins/auto_codex/eval

# 1. Recover payloads and the decision logged for each of them
python3 $EVAL/extract_payloads.py -o /tmp/corpus.jsonl

# 2. Replay the current code over the corpus
python3 $EVAL/replay.py /tmp/corpus.jsonl -o /tmp/decisions.jsonl --determinism

# 3. Compare with the recorded decisions (diagnostic, not a score)
python3 $EVAL/compare_history.py /tmp/decisions.jsonl --recorded --show 5

# 4. Look at a single disagreement
python3 $EVAL/inspect_case.py /tmp/corpus.jsonl /tmp/decisions.jsonl \
    --line 104387
```

`replay.py --semantic` adds the embedding layer (loads the configured
`sentence-transformers` model, ~25 s of startup). Run both variants: the
deterministic one is what a `semantic.enabled=false` deployment ships.

Against a labeled corpus, use the in-tree evaluation instead — it measures
accuracy, while this directory measures what moved:

```bash
python3 -m llm_router_plugins.utils.routing.agentic_routing.codex.evaluation \
    --dataset tests/data/codex_routing_quality.json --split all --no-semantic
```

## What `replay.py` guarantees

Three invariants are reported on every run, because a routing change must never
break them:

- `apply()` raises nothing — routing fails open;
- a payload that was not routed comes back as **the same object**, byte for byte;
- a routed payload gains exactly `agent_mode` and `routing`
  (`annotate_routing` mutates in place, so the pre-call key set is captured
  before the call).

`--determinism` replays the corpus backwards through a second plugin instance
and checks the decisions are identical, which catches any hidden dependence on
request order.

## Reading `compare_history.py`

Roughly half the requests disagree with the recording, and that is expected: the
logs span several versions of the cascade and of the configuration (an earlier
deployment ran with memory and the semantic layer enabled, the shipped config is
stateless). Treat it as a **change-impact and case-mining** tool:

- agreement per source file localises where behaviour moved;
- the `recorded -> current` confusion table names the pairs worth a human look;
- `--list-unrouted` separates a coverage change (the request is no longer
  routed at all) from a wrong mode, so a non-trigger payload paired with a
  neighbouring decision cannot fake the agreement number.

Correctness claims belong to the labeled corpus and to the metrics in
`codex/evaluation.py` — per-mode precision/recall and transition counts.

## Known limits

- Requests are paired with decisions **in stream order**. A payload whose
  decision line never arrived keeps `recorded: null`, and a payload interleaved
  with another request's decision can be matched to the wrong one; the two
  visible cases are reported by `--list-unrouted`.
- `extract_payloads.py` needs router logs at `DEBUG`/`INFO` with the request body
  printed as indented JSON — a one-line body is not recovered.
- Log blocks interrupted by another log line are counted and skipped
  (`unparsable blocks dropped`), never guessed.
