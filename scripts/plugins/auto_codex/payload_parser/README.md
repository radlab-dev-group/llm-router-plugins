# Codex routing — per-request payload export

Turns the recorded router traffic in `agents-conversation/` into **one JSON
document per request**: the request exactly as the Codex CLI sent it, plus the
mode and model the plugin in this checkout decides for it.

Parsing is shared with [`../eval`](../eval/README.md) — one recovery rule for
both entry points — and the decision always comes from the current code, never
from the line the recording server logged.

## Usage

```bash
# whole recorded corpus, deterministic cascade
python3 scripts/plugins/auto_codex/payload_parser/export_payloads.py

# one session, embedding layer on, into a scratch directory
python3 scripts/plugins/auto_codex/payload_parser/export_payloads.py \
    agents-conversation/codex/conv-03 --semantic -o /tmp/payloads

python3 scripts/plugins/auto_codex/payload_parser/export_payloads.py --help
```

Output is `agents-conversation/codex/payloads/codex_payload_NNN.json`,
numbered in recovery order (log order). The directory is gitignored — it is
hundreds of MB of derived data.

## Document shape

```jsonc
{
  "id": 63,
  "source": "agents-conversation/codex/conv-03/llm-router.log",
  "line": 1842,
  "logged_at": "2026-10-09 04:57:14,037",
  "trigger": "auto_codex",
  "routed": true,
  "decision": {                    // what the current code decided
    "mode": "git_review",          // work mode
    "model": "qwen/Qwen3.8-27B",   // chosen model
    "layer": "phase",              // which layer answered
    "similarity": 1.0,
    "class": "main",               // main | aux_title | compaction
    "reason": "git log",
    "evidence": "command",
    "memory": null,
    "semantic": null,
    "error": null                  // set only if routing raised (it must not)
  },
  "routing": { ...raw plugin annotation... },   // same decision, unflattened
  "recorded": { "mode": "implement", "source": "fallback", ... } | null,
  "payload": { ...request as sent, untouched... }
}
```

`payload` is the request **before** routing: the plugin rewrites `model` and
adds `routing`/`agent_mode` in place, so a copy is routed and the original is
what gets stored. Compare `decision.model` with `payload.model` to see the
rewrite. `recorded` is the decision the recording server made, kept for
diffing; `null` when the log had none.

Requests the plugin does not own (model other than the trigger) are written
with `routed: false`, `decision.model` equal to the model they arrived with and
`decision.mode` null; `--only-trigger` skips them.

## Resuming

A run can be interrupted and repeated: file `NNN` is reused unless
`--overwrite` is given, and the stored `source`/`line` are checked against the
request that would be written there. Changing the filters changes the
numbering, so a mismatch aborts with a message instead of silently relabelling
already written documents — pass `--overwrite` or an empty `-o` directory then.
