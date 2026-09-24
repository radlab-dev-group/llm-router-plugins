# Simple Semantic Routing (`simple_semantic_routing`)

Heuristic, two-stage model selection: it classifies the **intent** of the last user message (code / math / creative /
general) with weighted keywords, phrases and regexes, estimates the **complexity** from a rough token count, and picks
a model from an ordered pool. No embedding model, no vector store, no network — the whole decision is a few string
scans, so it costs microseconds and is fully reproducible.

It is the entry-level routing plugin: turn `model: "auto"` into "a sensible model for this request" without any ML
dependencies. When you outgrow keyword heuristics, move to
[`semantic_biencoder_routing`](../semantic_biencoder/README.md), which decides by embedding similarity instead.

- **Plugin name (registry key):** `simple_semantic_routing`
- **Class:** `llm_router_plugins.utils.routing.simple_semantic.simple_semantic_routing.SimpleSemanticRoutingPlugin`
- **Default config:** `llm_router_plugins/resources/routing/simple_semantic.json`
- **Trigger:** `payload["model"] == "auto"` (exact match)
- **Env prefix:** `LLM_ROUTER_ROUTING_SEMANTIC_`
- **Dependencies:** none beyond the base package

---

## What the plugin changes

`apply()` sets exactly one key:

| Key     | Value                                                       |
|---------|--------------------------------------------------------------|
| `model` | the pool entry selected for this intent + complexity          |

Nothing else is added — there is **no `routing` metadata block** in the payload (unlike
`semantic_biencoder_routing` and `agentic_routing_codex`). Everything the plugin knows is written to the log:

```text
SimpleSemanticRouting: intent=code, complexity=simple (10 tokens) -> qwen3.6:35b
```

If no user text can be found, `model` is set to the configured **default model** and a warning is logged; a payload
whose `model` is not exactly `"auto"` is returned untouched.

---

## Quick start

### 1. Install

```bash
# from a checkout of this repository, into the environment the router runs in
pip install -e .
```

No extras needed — this plugin imports only the standard library.

### 2. Load the plugin into the router

```bash
export LLM_ROUTER_UTILS_PLUGINS_PIPELINE="simple_semantic_routing"
```

The router resolves the name in `llm_router_plugins.utils.registry.MAIN_UTILS_REGISTRY`, instantiates it once per
process and runs `apply()` on every prepared payload, before guardrails and before a provider is selected.

### 3. Declare the models in the router model config

Everything the plugin can select must exist in the llm-router model config (`LLM_ROUTER_MODELS_CONFIG`):

- the **`auto`** alias itself, backed by a `builtin` provider — the client asks for it, the plugin replaces it;
- **every model in the pool** (`default_models` values, or the `LLM_ROUTER_ROUTING_SEMANTIC_MODELS` list).

```jsonc
// fragment of the llm-router models config
"semantic_routing": {
  "auto": {
    "providers": [
      { "id": "semantic-routing__auto", "api_type": "builtin", "api_host": null,
        "api_token": "", "model_path": "", "input_size": 0, "weight": 1.0,
        "tool_calling": false }
    ]
  }
}
```

A pool entry the router does not know is a hard downstream error, not a routing no-op.

### 4. Verify

Send a request with `"model": "auto"` and look for the log line, or run the
[verification snippet](#offline-dry-run). Two lines to remember:

```text
SimpleSemanticRouting: intent=code, complexity=simple (10 tokens) -> qwen3.6:35b
No text content found, using default model gpt-oss:120b
```

---

## How it works

```text
payload ──▶ extract text ──▶ stage 1: intent ──▶ stage 2: complexity ──▶ stage 3: pool index ──▶ payload["model"]
```

### Where it runs

Inside the llm-router utils pipeline. The plugin is stateless: no cache, no session, no learned state, so every
replica answers identically.

### Text extraction

Text is located by the shared helper `llm_router_plugins.utils.text_extractor.extract_user_text`, in this priority:

1. `payload["messages"][-1]["content"]` — the last message of a chat history (earlier turns are **not** seen)
2. `payload["user_last_statement"]`
3. `payload["query"]`
4. `payload["prompt"]`
5. `payload["input"]`

Only that one string is scored, so intent detection always reflects the newest user turn.

### Stage 1 — intent classification

Every intent in `intents` accumulates a score from three signal kinds; the highest score above zero wins, ties go to
the intent declared first in the config, and a zero score everywhere yields `none`.

| Signal     | Match                                              | Weight                                  |
|------------|-----------------------------------------------------|------------------------------------------|
| `keywords` | **case-insensitive substring** in the lower-cased text | `weights[keyword]`, else `1`         |
| `phrases`  | substring of the multi-word expression                | `":weight"` suffix, else `2.0`        |
| `patterns` | `re.search(pattern, lower_text)`                     | `3.0` per matching pattern             |

Measured with the shipped config (score contributions listed):

| Input                                              | Winner   | What scored                                                        |
|----------------------------------------------------|----------|---------------------------------------------------------------------|
| `Napisz funkcję def foo(): import os`               | `code`   | `funkcję` 3 + `import` 2 + `napisz funkcję` 2 + 2 patterns 6 = 14.0 |
| the same input, `creative` intent                    | —        | `napisz` 1 + `pisz` 1 + … = 5.0 (outscored)                         |
| `Wyjaśnij czym jest blockchain`                      | `general` | `wyjaśnij` 3 + `czym` 1 = 4.0                                      |
| `hello`                                              | `none`   | nothing scores — greetings are not an intent                        |
| `contest`                                            | `general` | substring `co` → 1.0 (see the substring caveat below)             |

**Keywords are plain substrings, not word-start matches.** That is what makes the Polish lists work — `kod`, `kodu`,
`kodem`, `kodzie` are all listed explicitly — but it also means a short keyword matches inside unrelated words: the
`general` keyword `co` fires inside `contest`, `cost`, `second`, `economy`. Keep ambiguous short words out of the
lists, or give them a phrase with a higher weight.

### Stage 2 — complexity estimation

```python
token_estimate = int(len(text.split()) * 1.25)     # word count, not a tokenizer
```

| `token_estimate`          | Complexity |
|---------------------------|------------|
| ≤ `len_thresholds_max.simple` (25)   | `simple`  |
| ≤ `len_thresholds_max.medium` (150)  | `medium`  |
| > `len_thresholds_max.medium`        | `complex` |

The estimate is a whitespace word count, so code blocks, URLs and CJK text are counted badly (a 400-line pasted file
is scored by its whitespace, not its tokens). Treat the thresholds as "how much did the user write", not "how much
context the model will see".

### Stage 3 — model selection

The pool is the ordered list of model names: `default_models.values()` from the config, or
`LLM_ROUTER_ROUTING_SEMANTIC_MODELS`. Position **is** capability ranking: index `0` is the cheapest/simplest model,
index `n-1` the strongest.

```text
base index    simple → 0        medium → n // 2        complex → n - 1
adjustment    intent_adjustment[intent] → index i of a complexity level
                  defined (non-empty) → index = max(base, i)     # can only escalate
                  empty or unknown     → index = base - 1        # de-escalates by one
final index   clamp(index, 0, n - 1) → pool[index]
```

Measured with the shipped two-model pool `["gpt-oss:120b", "qwen3.6:35b"]`
(`intent_adjustment`: `code`/`math` → `medium`, `creative`/`general` → `simple`, `none` → `""`):

| Intent     | simple          | medium          | complex         |
|------------|-----------------|-----------------|-----------------|
| `code`     | `qwen3.6:35b`   | `qwen3.6:35b`   | `qwen3.6:35b`   |
| `math`     | `qwen3.6:35b`   | `qwen3.6:35b`   | `qwen3.6:35b`   |
| `creative` | `gpt-oss:120b`  | `qwen3.6:35b`   | `qwen3.6:35b`   |
| `general`  | `gpt-oss:120b`  | `qwen3.6:35b`   | `qwen3.6:35b`   |
| `none`     | `gpt-oss:120b`  | `gpt-oss:120b`  | `gpt-oss:120b`  |

With a three-model pool (`LLM_ROUTER_ROUTING_SEMANTIC_MODELS="gpt-mini|qwen-mid|qwen-max"`, `default_models`
unchanged) the same rules produce a real three-step ladder:

| Intent     | simple       | medium       | complex      |
|------------|--------------|--------------|--------------|
| `code`     | `qwen-mid`   | `qwen-mid`   | `qwen-max`   |
| `math`     | `qwen-mid`   | `qwen-mid`   | `qwen-max`   |
| `creative` | `gpt-mini`   | `qwen-mid`   | `qwen-max`   |
| `general`  | `gpt-mini`   | `qwen-mid`   | `qwen-max`   |
| `none`     | `gpt-mini`   | `gpt-mini`   | `qwen-mid`   |

Note what the matrices say about the shipped config: with two models, `code` and `math` are always escalated — a
one-word "debug" reaches the bigger model — while `none` is de-escalated to index 0 at every length.

---

## Configuration

### Config file layout

```jsonc
{
  "settings": {
    "len_thresholds_max": { "simple": 25, "medium": 150 },
    "default_models": {                       // required: complexity → model name
      "simple": "gpt-oss:120b",
      "medium": "qwen3.6:35b"
    },
    "intent_adjustment": {                    // intent → complexity floor ("" = de-escalate)
      "code": "medium", "math": "medium", "creative": "simple", "general": "simple", "none": ""
    }
  },
  "intents": {
    "code": {
      "keywords": ["code", "debug", "kod", "funkcja"],
      "phrases": ["write code:5", "napraw błąd:4"],
      "patterns": ["def\\s+\\w+\\s*\\(", "import\\s+\\w+"],
      "weights": { "debug": 5, "napraw": 4, "kod": 3 }
    }
  },
  "none": { "keywords": ["hello", "cześć"] }   // informational, not used by the plugin
}
```

Required keys: `settings.len_thresholds_max`, `settings.default_models`, `settings.intent_adjustment`, `intents`
(a `KeyError` at startup names the missing one). `default_models` must contain `simple` (it provides the fallback
model) and `medium` (it provides the escalation target for `intent_adjustment`).

The top-level `none` block is read into `RoutingConfig.none_keywords` and **never used**: `none` is what the plugin
reports when nothing scores, and it can also be forced via `intent_adjustment`. Editing that block changes nothing.

### Environment variables

Prefix `LLM_ROUTER_ROUTING_SEMANTIC_`, read once at plugin construction (restart to apply):

| Variable                                                | Effect                                                                      |
|---------------------------------------------------------|------------------------------------------------------------------------------|
| `…_COMPLEXITY_THRESHOLDS`                               | `simple\|medium` token cut-offs, e.g. `25\|150`; malformed input warns and keeps the file values |
| `…_MODELS`                                              | replaces the whole pool, pipe-separated, cheapest first                       |
| `…_DEFAULT_MODEL`                                       | model used when no user text is found                                          |
| `…_INTENT_<name>`                                       | replaces one intent: `kw1\|kw2\|phrase one:3\|phrase two:5`                     |

`…_INTENT_<name>` splits its value on `:` — entries with a colon become phrases (with weights), the rest become
keywords. There is **no `…_CONFIG` variable for this plugin**: the config path is fixed to the bundled
`simple_semantic.json`, so a custom config means editing that file (or bind-mounting your own over it in a container).

### Validation

Startup raises `ValueError` when the pool is empty, the thresholds are not exactly two values, no intent category is
defined, or no default model resolves. A malformed `…_COMPLEXITY_THRESHOLDS` value only warns and falls back to the
file.

---

## Tuning and gotchas

- **Pool order is everything.** The plugin never reads model metadata; index `0` is what it uses for "easy", index
  `n-1` for "hard". Reversing `…_MODELS` inverts your cost curve.
- **`…_MODELS` does not move the default model.** The fallback for text-less requests comes from the config
  (`default_models.simple`), so with `…_MODELS="model-a|model-b"` a request without extractable text is still sent to
  `gpt-oss:120b` — a model that may not even be in your pool. Always set `…_DEFAULT_MODEL` together with `…_MODELS`.
- **`…_INTENT_<name>` replaces, it does not merge.** The override rebuilds the intent with `patterns: []` and
  `weights: {}`: overriding `INTENT_code` silently drops all 31 code regexes and all 15 keyword weights. Add your words
  to the JSON instead.
- **Category names keep the env var's case.** `…_INTENT_CODE=x` creates an intent called `CODE`, next to the existing
  `code` — and `intent_adjustment` will not match it, so it never escalates.
- **Phrase weights are parsed strictly.** A value with two colons (`…_INTENT_X=bad:phrase:x`) raises
  `ValueError: could not convert string to float` inside `apply()` — the request fails, it does not degrade. One colon,
  digits after it.
- **Substring matching bites short keywords.** `co` matches `contest`; `pisz` matches `piszemy` *and* any word that
  contains it. Prefer longer stems or a phrase with a weight over a 2-letter keyword.
- **Patterns are not anchored and are matched against lower-cased text.** An upper-case pattern can never match; an
  invalid regex is skipped silently (`re.error` is swallowed).
- **Only the last message is classified.** A long system prompt with tools and instructions is invisible; so is the
  rest of the conversation. Intent is decided on the newest user turn alone.
- **No `routing` block.** Correlate decisions with the `intent=…, complexity=… (N tokens) -> model` log line; add
  `logger` at INFO level in the router.
- **Both this plugin and `semantic_biencoder_routing` trigger on `auto`.** If both are in the pipeline, the first one
  rewrites `model` and the second sees a concrete model and does nothing. Pick one, or make them complementary by
  keeping only one enabled.
- **Escalation is monotonic on purpose.** `intent_adjustment` can raise the index but never lower it below the
  complexity base; the only de-escalation is the `-1` step for intents with an empty adjustment.

---

## Verifying the plugin

### Offline dry run

```python
"""Public-API check of the Simple Semantic routing plugin."""
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

from llm_router_plugins.utils.routing.simple_semantic.simple_semantic_routing import (
    SimpleSemanticRoutingPlugin,
)

plugin = SimpleSemanticRoutingPlugin(logger=logging.getLogger("routing"))

TEXTS = [
    "hello",
    "Napisz funkcję w Pythonie, która sortuje listę słowników",
    "Oblicz prawdopodobieństwo warunkowe dla zdarzeń A i B",
    "Napisz wiersz o jesieni",
    "Wyjaśnij czym jest blockchain",
]

for text in TEXTS:
    out = plugin.apply({"model": "auto", "messages": [{"role": "user", "content": text}]})
    print(f"{text[:44]!r:48} -> {out['model']}")
```

Expected output with the shipped config:

```text
SimpleSemanticRouting: intent=none, complexity=simple (1 tokens) -> gpt-oss:120b
SimpleSemanticRouting: intent=code, complexity=simple (10 tokens) -> qwen3.6:35b
SimpleSemanticRouting: intent=math, complexity=simple (10 tokens) -> qwen3.6:35b
SimpleSemanticRouting: intent=creative, complexity=simple (5 tokens) -> gpt-oss:120b
SimpleSemanticRouting: intent=general, complexity=simple (5 tokens) -> gpt-oss:120b
'hello'                                          -> gpt-oss:120b
'Napisz funkcję w Pythonie, która sortuje lis'   -> qwen3.6:35b
'Oblicz prawdopodobieństwo warunkowe dla zdar'   -> qwen3.6:35b
'Napisz wiersz o jesieni'                        -> gpt-oss:120b
'Wyjaśnij czym jest blockchain'                  -> gpt-oss:120b
```

### Router smoke test

```bash
curl -sS http://<router-host>:8081/v1/chat/completions \
  -H "Authorization: Bearer $LLM_ROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"napraw błąd w funkcji eksportu"}]}' \
  | head -c 300
```

### Automated tests

```bash
python -m pytest tests/test_simple_semantic_routing.py -q
```

---

## Troubleshooting

| Symptom                                                   | Cause                                                            | Fix                                                          |
|------------------------------------------------------------|-------------------------------------------------------------------|---------------------------------------------------------------|
| Requests never rerouted                                     | plugin not in `LLM_ROUTER_UTILS_PLUGINS_PIPELINE`, or `model` ≠ `auto` (not even `" auto "`) | check the pipeline env var and the exact model string |
| Everything lands on the same model                          | pool too small, or `intent_adjustment` collapses the ladder        | use a 3-model pool, review the matrices above                   |
| Big model chosen for a one-word greeting                    | an intent is scoring on a substring, so `none` is not reached       | shorten/qualify the offending keyword, verify with the dry run   |
| `none` on obvious code requests                             | signals are English-only or too narrow                             | add PL+EN phrases with weights in the JSON                       |
| Text-less request routed to an unexpected model              | `…_MODELS` changed the pool but not `…_DEFAULT_MODEL`              | set both env vars together                                       |
| `ValueError: could not convert string to float`              | a phrase with two colons in an `…_INTENT_<name>` override          | exactly one `:weight` suffix                                     |
| Intent keywords stopped working after an env override        | `…_INTENT_<name>` replaced the intent (patterns and weights cleared) | move the change into the JSON config                          |
| 4xx/5xx from the provider after routing                       | a pool entry is not declared in `LLM_ROUTER_MODELS_CONFIG`         | declare every pool model                                          |
| Long pasted code treated as `simple`                           | token estimate counts whitespace words                             | rely on intent escalation, or lower the thresholds                |

---

## Reference

### Defaults at a glance

| Setting                          | Shipped value                       |
|----------------------------------|--------------------------------------|
| Trigger                          | `payload["model"] == "auto"`          |
| Complexity thresholds            | `simple ≤ 25`, `medium ≤ 150` tokens  |
| Token estimate                    | `int(words * 1.25)`                  |
| Pool                              | `["gpt-oss:120b", "qwen3.6:35b"]`     |
| Default (text-less) model         | `gpt-oss:120b` (`default_models.simple`) |
| Intent categories                 | `code`, `math`, `creative`, `general` |
| Keyword / phrase / pattern weight | `1` / `2.0` / `3.0`                  |
| No-match intent                   | `none` (de-escalates one pool step)   |
| Payload keys written              | `model` only                          |

### Files

| Path                                                                     | Role                                        |
|---------------------------------------------------------------------------|----------------------------------------------|
| `simple_semantic_routing.py`                                               | plugin: three stages, pool selection        |
| `config.py`                                                                | `RoutingConfig` loader for the bundled JSON |
| `../../../resources/routing/simple_semantic.json`                       | intents, thresholds, pool, adjustments      |
| `../../text_extractor.py`                                               | shared user-text extraction                 |

### See also

- [Semantic Routing Plugins reference](../README.md#1-simple-semantic-routing-heuristic) — §1 of the shared routing
  README: architecture diagram, built-in pattern catalogue, weight-tuning guidelines and how to add an intent.
- [`semantic_biencoder`](../semantic_biencoder/README.md) — embedding-similarity routing on the same `auto` trigger.
- [`agentic_routing/codex`](../agentic_routing/codex/README.md) — work-mode routing for Codex CLI traffic.
- [Root README §2.7](../../../../README.md#27-semantic-routing-model-selection) — plugin overview and triggers.
