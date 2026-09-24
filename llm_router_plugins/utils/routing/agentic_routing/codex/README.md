# Codex CLI Routing (`agentic_routing_codex`)

Routing plugin for the **OpenAI-Responses-style** requests emitted by the **Codex CLI** coding agent. It looks at
every request that arrives with the trigger model (`auto_codex` by default), decides *what kind of work the agent is
doing right now*, and rewrites `payload["model"]` to the model configured for that work mode. Planning turns, code
reviews, test runs, debugging, one-line thread titles and context compaction can then each run on the model that fits
them best, while the Codex CLI keeps talking to a single stable model name.

The decision cascade is **deterministic first**: structural signals (request class, the `<collaboration_mode>` block
the CLI injects, keyword scoring) answer most requests offline, for free, and reproducibly. An optional embedding layer
contributes cosine similarity over the mode descriptions and examples only when the cheap layers stay silent.

- **Plugin name (registry key):** `agentic_routing_codex`
- **Class:** `llm_router_plugins.utils.routing.agentic_routing.codex.plugin.CodexRoutingPlugin`
- **Default config:** `llm_router_plugins/resources/routing/agentic_routing_codex.json`
- **Env prefix:** `LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_`

---

## What the plugin changes

`apply()` rewrites the payload in place and touches exactly three keys:

| Key                | Value                                                                     |
|--------------------|---------------------------------------------------------------------------|
| `model`            | `model_name` of the resolved Codex work mode                              |
| `agent_mode`       | name of the resolved mode (`plan`, `implement`, `test`, …)                 |
| `routing`          | decision metadata: `plugin`, `similarity`, `source`, class and ids         |

Everything else — `input`, `instructions`, `tools`, `reasoning`, `text`, `stream`, `parallel_tool_calls` — is forwarded
unchanged. The plugin never rejects a request; it only ever picks a model.

**Fail-open by design.** A payload that is not a dict, does not carry the trigger model, resolves to a mode that is not
configured or to a mode with an empty `model_name`, or that raises anywhere during parsing or classification, is
returned untouched with the trigger model still in place. **Fail-hard on configuration only**: an inconsistent config
raises at plugin construction, so the router refuses to start instead of silently misrouting every request.

---

## Quick start

### 1. Install the plugin package

```bash
# from a checkout of this repository, into the environment the router runs in
pip install -e .              # deterministic routing only
pip install -e ".[ml]"        # + CPU FAISS and sentence-transformers (semantic layer)
pip install -e ".[ml-gpu]"    # + CUDA FAISS instead — never both FAISS builds at once
```

The `[ml]` / `[ml-gpu]` extras install `faiss-cpu`/`faiss-gpu`, `sentence-transformers`, `numpy` and `scipy`. Without
them the plugin still loads and routes deterministically — the semantic layer steps aside with a single warning. In a
container image, add the package (with the extra you want) next to `llm-router` so both import the same
`sentence-transformers`.

### 2. Load the plugin into the router

The plugin is a *utils* plugin: it runs inside the llm-router request pipeline and is selected by a comma-separated
list of plugin identifiers.

```bash
export LLM_ROUTER_UTILS_PLUGINS_PIPELINE="agentic_routing_codex"
```

The router resolves the name against `llm_router_plugins.utils.registry.MAIN_UTILS_REGISTRY`, instantiates it **once per
process** (`UtilsRegistry`), and calls its `apply()` on every prepared request, before guardrails and before a provider
is chosen. Order matters if you wire several utils plugins together: the first plugin that rewrites `model` wins the
downstream dispatch.

> Instantiating the plugin loads the JSON config, applies env overrides, validates them and — when the semantic layer
> is enabled — loads the embedding model and builds (or loads) the FAISS index. Expect a few seconds of extra startup
> time and a couple of hundred MB of RSS for a 300 M embedding model.

### 3. Declare the models in the router

Two things must exist in the llm-router model config (`LLM_ROUTER_MODELS_CONFIG`, JSON):

1. **The trigger** `auto_codex` as a declared model alias backed by a `builtin` provider — the CLI asks for it, the
   router accepts it, and the plugin replaces it before any upstream call is made. It follows the same pattern as the
   `auto` alias used by the semantic routing plugins:

   ```jsonc
   // fragment of the llm-router models config (LLM_ROUTER_MODELS_CONFIG)
   "semantic_routing": {
     "auto_codex": {
       "providers": [
         {
           "id": "semantic-routing__auto_codex",
           "api_type": "builtin",
           "api_host": null,
           "api_token": "",
           "model_path": "",
           "input_size": 0,
           "weight": 1.0,
           "tool_calling": true
         }
       ]
     }
   }
   ```

2. **Every model referenced by `codex_modes[].model_name`** as a normal, active model with a reachable provider
   (`api_type: "vllm"`, `ollama`, `openai`, …). A name the router does not know is a hard downstream failure
   (`model not found`), *not* a routing error: the plugin has already done its job.

### 4. Point the Codex CLI at the router

`~/.codex/config.toml`:

```toml
model = "auto_codex"
model_provider = "llm_router"
model_context_window = 256000
model_max_output_tokens = 32000

[model_providers.llm_router]
name = "LLM Router"
base_url = "http://<router-host>:8081/v1"
wire_api = "responses"
env_key = "LLM_ROUTER_API_KEY"
```

`wire_api = "responses"` is what produces the `/v1/responses` request shape this plugin reads, and `LLM_ROUTER_API_KEY`
must hold a valid router API key. Because `auto_codex` is a stable alias, nothing else changes when you retune the
mode→model mapping — no CLI redeploy, no `git pull` of client configs.

### 5. Verify

Two log lines prove the plugin is loaded and deciding:

```text
[utils] Registered utility plugin 'agentic_routing_codex' as instance 'CodexRoutingPlugin'
Codex routing: mode=plan source=collaboration_mode model=qwen/Qwen3.8-Flash-Next similarity=1.000 class=main
```

For an end-to-end check without the CLI, see [Verifying the plugin](#verifying-the-plugin).

---

## How it works

### Where it runs

```text
Codex CLI ──POST /v1/responses (model=auto_codex)──▶ llm-router
    auth middleware → prepare_payload → UtilsPipeline → guardrails → masking → provider dispatch ──▶ vLLM
                                              │
                                              └── CodexRoutingPlugin.apply(payload)
```

The plugin sees the prepared request payload as a plain `dict` and returns it. It is stateless: no session cache, no
memory between requests, so the same request is routed identically by any replica at any time.

### Step 1 — payload normalization

`parse_codex_payload()` is the only code that knows where Codex metadata lives on the wire; everything downstream works
on an immutable `CodexRequest` snapshot. Reading it never raises — an unreadable value keeps the field default.

| Snapshot field                              | Read from                                                                 |
|---------------------------------------------|---------------------------------------------------------------------------|
| `session_id`, `thread_id`, `turn_id`, `root_turn_id` | `client_metadata.*` (turn metadata as fallback)                      |
| `window_id`                                 | `client_metadata["x-codex-window-id"]`                                    |
| `request_kind`, `thread_source`, `sandbox_mode`, `agent_name`, `context_window_id` | `client_metadata["x-codex-turn-metadata"]`, which is a **JSON-encoded string** |
| `tool_names`, `has_tools`                   | `tools[].name` (falls back to `type` for built-in tools)                  |
| `structured_output`                         | `text.format.type == "json_schema"`                                       |
| `reasoning_effort`, `parallel_tool_calls`   | `reasoning.effort`, `parallel_tool_calls`                                 |
| `context_chars`, `context_tokens`           | content length of `instructions` + `input` (tokens ≈ chars / 4)           |
| `collaboration_mode`                        | `<collaboration_mode>…</collaboration_mode>` block in `developer` messages |
| `latest_user_text`                          | genuine `user` messages, newest first, within `classify_max_chars`         |

Details that matter when you debug routing:

- **Collaboration mode** is read from every `role == "developer"` message; the **last** `<collaboration_mode>` block
  wins, so a Plan → Default switch mid-session is picked up on the next turn. Its first heading decides: `# Plan Mode`
  → `plan`, `# Collaboration Mode: Default` → `default`, anything else → `""`.
- **`latest_user_text`** scans `input` in reverse and keeps only `type == "message"`, `role == "user"` items whose text
  is not just an `<environment_context>` block. The newest message is always kept whole; older ones are appended while
  the `classify_max_chars` budget holds (a non-positive budget means "unbounded"). If the request carries no usable
  user message, the legacy `payload["prompt"]` is used.

### Step 2 — request class

Codex mixes three kinds of request on one endpoint. The structural ones are recognised *before* any prompt text is
read, with strict priority `compaction > aux_title > main`:

| Class        | Condition                                                     | Why it matters                                                    |
|--------------|---------------------------------------------------------------|-------------------------------------------------------------------|
| `compaction` | `request_kind == "compaction"`                                | Carries the whole transcript; wins over every keyword signal       |
| `aux_title`  | `request_kind == "turn"` **and** `thread_source == "system"`   | CLI-generated one-line thread title, no tools, JSON-schema output  |
| `main`       | everything else                                               | A regular agent turn (Plan or Default collaboration mode)          |

### Step 3 — resolution cascade

`classify()` tries layers in a fixed order; the **first layer that can answer wins**. A main turn never falls below the
fallback mode: the keyword and semantic layers can specialise the decision, never weaken it.

| # | Layer                                          | `routing.source`     | Confidence                          |
|---|------------------------------------------------|----------------------|-------------------------------------|
| 1 | Explicit `agent_mode` / `codex_mode` / `metadata.agent_mode` in the payload | `explicit`           | `1.0`             |
| 2 | Request class — `compaction`, then `aux_title` | `class`              | `1.0`                               |
| 3 | `<collaboration_mode>` Plan block from the CLI  | `collaboration_mode` | `1.0`                               |
| 4 | Keyword scoring of the user text                | `heuristic`          | `score / (score + 1)`               |
| 5 | Embedding cosine similarity over the modes      | `semantic`           | cosine of the matched mode          |
| 6 | Configured `fallback_mode` (`implement`)        | `fallback`           | cosine of that mode, else `0.0`     |

Notes:

- An explicit override must name a **configured** mode, otherwise it is ignored and the cascade continues;
  `agent_mode` is checked before `codex_mode`, and both before `metadata.agent_mode`. This is the escape hatch for
  "route this one request by hand": send `"agent_mode": "debug"` in the request body.
- Only four modes compete in the keyword layer, scanned in the order `test`, `git_review`, `review`, `debug`
  (`HEURISTIC_MODES`). `plan` is declared by the CLI and `implement` is the fallback, so their `keywords` / `phrases` /
  `patterns` are never scored — only their `description` and `examples` feed the embedding index. `aux_title` and
  `compaction` are class-routed and are left out of the index too.
- Layers 1–4 never touch the embedding stack. The vector store is queried **at most once per request**, and only after
  the deterministic layers have stayed silent.

### Step 4 — keyword scoring

Matching is case-insensitive (`DEBUG` matches `debug`) and keywords and phrases must start at a word boundary. A
declared stem therefore also hits inflected forms — `test` matches `testy` and `testów`, `napraw` matches
`naprawiłem` — while mid-word occurrences are ignored (`protest` and `kontest` do not match `test`). Letters are
**not** diacritic-folded, which is exactly why the shipped lists carry `błąd` *and* `blad`, `nie działa` *and*
`nie dziala`, `plan rozwiązania` *and* `plan rozwiazania`.

| Signal    | Weight                                     | Example                          |
|-----------|--------------------------------------------|----------------------------------|
| `keywords`| `weights[keyword]`, else `1.0`             | `"debug": 3`                     |
| `phrases` | `":weight"` suffix, else `2.0`             | `"napraw błąd:3"`                |
| `patterns`| `3.0` per match                            | `"\broot\s+cause\b"`             |

Scores of all matched signals are summed per mode; `detect_mode` uses strictly-greater-than while scanning
`HEURISTIC_MODES` in order, so **ties go to the earlier mode** (`test` over `git_review`, `git_review` over `review`).
The best score is accepted only when it reaches `heuristic_min_score` (`3.0` by default), and is reported as
`similarity = score / (score + 1)`.

Measured against the shipped configuration:

| Text                                        | Mode        | Score | Accepted (`≥ 3.0`) |
|---------------------------------------------|-------------|-------|--------------------|
| `napraw testy w tests/`                      | `test`      | 14.0  | yes → `0.933`      |
| `testy`                                      | `test`      | 8.0   | yes → `0.889`      |
| `Dlaczego nie działa ta funkcja?`            | `debug`     | 9.0   | yes → `0.900`      |
| `review this PR`                             | `review`    | 9.0   | yes → `0.900`      |
| `git review przed merge`                     | `git_review`| 11.0  | yes → `0.917`      |
| `Przygotuj plan refactoru`                   | `review`    | 2.0   | no → semantic/fallback |
| `Dodaj nowy endpoint do API`                 | —           | 0.0   | no → semantic/fallback |

The last two rows are the point of the design: **`plan` is never decided by keywords** (the CLI tells you, in Plan
Mode, and a planning *sentence* in Default mode is not enough), and a plain imperative like "add an endpoint" is left to
the semantic layer or to `fallback_mode` — which is exactly what `implement` means.

### Step 5 — semantic similarity (optional)

Enabled with `settings.semantic.enabled`, disabled for a single process with
`LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_SEMANTIC_ENABLED=false`.

- An `EmbeddingRouter` indexes `description` + `examples` of the six non-class-routed modes with a sliding window
  (`chunk_size` 256, `chunk_overlap` 64) using a sentence-transformers BiEncoder.
- Index vectors are L2-normalized and searched with FAISS inner product, which is cosine similarity.
- A query longer than the embedding model's `max_seq_length` is split into overlapping token windows; at most the
  first `MAX_QUERY_WINDOWS = 4` windows are encoded and their unit vectors are averaged and renormalized, so a huge
  prompt costs a bounded amount of compute.
- `top_k` hits are grouped per mode and **averaged**; the best mode wins and is accepted only when its average cosine
  reaches `similarity_threshold` (`0.51` by default).
- One lookup serves both the accept decision and the fallback mode's reported similarity, so a request never embeds the
  same text twice.
- The model is loaded with `device="cpu"` and `trust_remote_code=True` at plugin construction. A router that raises at
  query time (broken index, empty vector) disables only the semantic answer for that request.

### Step 6 — annotating the payload

Before (trimmed real Codex turn, Plan Mode):

```json
{
  "model": "auto_codex",
  "instructions": "You are a coding agent running in the Codex CLI.",
  "input": [
    {"type": "message", "role": "developer", "content": [{"type": "input_text",
      "text": "You are a coding agent running in the Codex CLI.\n\n<collaboration_mode># Plan Mode (Conversational)\n\nThink, do not edit.</collaboration_mode>"}]},
    {"type": "message", "role": "user", "content": [{"type": "input_text",
      "text": "<environment_context>\n  <cwd>/repo</cwd>\n</environment_context>"}]},
    {"type": "message", "role": "user", "content": [{"type": "input_text",
      "text": "Zaplanuj migrację bazy danych."}]}
  ],
  "tools": [{"type": "function", "name": "exec_command", "parameters": {}}],
  "parallel_tool_calls": true,
  "reasoning": {"effort": "medium", "summary": "auto"},
  "stream": true,
  "client_metadata": {
    "thread_id": "thread-1",
    "turn_id": "turn-7",
    "x-codex-window-id": "thread-1:0",
    "x-codex-turn-metadata": "{\"request_kind\":\"turn\",\"thread_source\":\"user\",\"sandbox_mode\":\"danger-full-access\",\"agent_name\":\"/root\"}"
  }
}
```

After (`model` replaced, `agent_mode` and `routing` added):

```json
{
  "model": "qwen/Qwen3.8-Flash-Next",
  "agent_mode": "plan",
  "routing": {
    "plugin": "agentic_routing_codex",
    "similarity": 1.0,
    "agent_mode": "plan",
    "source": "collaboration_mode",
    "codex_class": "main",
    "collaboration_mode": "plan",
    "request_kind": "turn",
    "thread_id": "thread-1",
    "turn_id": "turn-7"
  }
}
```

`routing` is metadata for logs, tracing and cost attribution. If your upstream is strict about unknown fields, strip
it after logging — the plugin itself only guarantees that the request shape Codex expects is preserved.

### Failure behaviour

| Situation                                                        | Result                                  |
|------------------------------------------------------------------|-----------------------------------------|
| payload is not a dict                                             | same object, untouched                  |
| `model` absent, not a string, or ≠ trigger (after `strip()`)       | same object, untouched                  |
| parse or classify raises                                          | same object, untouched + `warning` log  |
| resolved mode is not configured                                    | same object, untouched + `warning` log  |
| resolved mode has `model_name: ""`                                 | same object, untouched + `warning` log  |
| FAISS / sentence-transformers missing while semantic is enabled     | plugin loads, semantic layer off, `warning` log |
| semantic lookup raises at query time                               | that request continues without semantic  |
| empty trigger, no modes, duplicate mode names, unknown `fallback_mode`, semantic on with no embedding model, `chunk_size <= 0`, `chunk_overlap < 0`, `top_k < 1`, `classify_max_chars <= 0` | `ValueError` at construction — router refuses to start |

---

## Configuration

### Config file layout

```jsonc
{
  "description": "what this config is for (documentation only)",
  "embedding_model": "/models/google/embeddinggemma-300m",   // top level, NOT in settings
  "embedding_model_public": "google/embeddinggemma-300m",     // informational, never read
  "settings": {
    "trigger_model": "auto_codex",       // required
    "fallback_mode": "implement",        // required, must name a mode below
    "heuristic_enabled": true,
    "heuristic_min_score": 3.0,
    "classify_max_chars": 4000,
    "vector_store_path": "",             // "" = index lives in memory only
    "semantic": {
      "enabled": true,
      "threshold": 0.51,
      "top_k": 3,
      "chunk_size": 256,
      "chunk_overlap": 64
    }
  },
  "codex_modes": [
    {
      "name": "debug",                    // required
      "model_name": "qwen/Qwen3.8-Flash-Next",  // required; "" = pass through
      "description": "…",                 // required; indexed semantically
      "examples": ["Napraw błąd w …"],    // indexed semantically
      "keywords": ["debug", "crash"],     // scored: 1.0 (or weights[name])
      "phrases": ["root cause:3"],        // scored: 2.0 (or the :weight suffix)
      "patterns": ["\\broot\\s+cause\\b"],// scored: 3.0
      "weights": {"debug": 3}             // per-keyword overrides
    }
  ]
}
```

Required keys: top-level `settings` and `codex_modes`; inside `settings`, `trigger_model` and `fallback_mode`; inside
each mode, `name`, `model_name` and `description`. Everything else has a default (see
[Defaults at a glance](#defaults-at-a-glance)). `description` and `examples` are not decoration — with the semantic
layer enabled they *are* the classifier's training set.

### Environment variables

All overrides use the prefix `LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_` and are applied **in place at plugin
construction** (so after a restart, never mid-session).

| Variable                                                           | Purpose                                              |
|--------------------------------------------------------------------|------------------------------------------------------|
| `…_CONFIG`                                                          | Custom config: raw JSON string **or** path to a file |
| `…_TRIGGER`                                                         | Trigger model value (default `auto_codex`)            |
| `…_MODEL_<MODE>`                                                    | Model for one mode, e.g. `MODEL_PLAN=…`               |
| `…_MODELS`                                                          | Per-mode models, e.g. `plan=model_a\|test=model_b`     |
| `…_MODES`                                                           | Pipe-separated whitelist of mode names to keep        |
| `…_FALLBACK_MODE`                                                   | Mode used when nothing matches                        |
| `…_HEURISTIC_ENABLED`                                               | `1/0`, `true/false`, `yes/no`, `on/off`               |
| `…_HEURISTIC_MIN_SCORE`                                             | Minimum keyword score to accept a heuristic hit       |
| `…_CLASSIFY_MAX_CHARS`                                              | Character budget of the classified user text          |
| `…_MODEL`                                                           | **Embedding** model for the semantic layer            |
| `…_SEMANTIC_ENABLED`                                                | Toggle the embedding layer                            |
| `…_SIMILARITY_THRESHOLD`                                            | Minimum cosine similarity for a semantic hit          |
| `…_TOP_K` / `…_CHUNK_SIZE` / `…_CHUNK_OVERLAP`                       | Embedding router knobs                                |
| `…_PERSIST_DIR`                                                     | Directory holding `index.faiss` + `docstore.pkl`      |
| `…_MODE_<name>_KEYWORDS`                                            | Pipe-separated keyword override for one mode          |

Booleans accept `1/0`, `true/false`, `yes/no`, `on/off`; numeric values are parsed as `int`/`float` and a malformed
value is ignored (with a warning where a logger is present). Unknown mode names in `MODELS`, `MODES`, `MODEL_<MODE>` or
`MODE_<name>_KEYWORDS` are logged as warnings and otherwise ignored — a typo silently does nothing rather than breaking
routing.

`…_MODES` filters the mode list **but does not move the fallback**: if the whitelist excludes the configured
`fallback_mode` (`implement`), set `…_FALLBACK_MODE` too — otherwise validation fails at startup.

### Precedence and linting

`…_CONFIG` (if non-empty) wins over the bundled JSON; there is no silent fall-through, so a missing file or invalid
JSON raises. Environment variables then win over whatever the file said. At construction the plugin also **lints** the
signals and warns about settings that can never score, without changing anything:

- a `patterns` entry that does not compile (it is dropped by the scorer);
- a `patterns` entry with upper-case letters (the text is lower-cased before scoring, so it can never match);
- a mode with an empty `model_name` (requests resolved to it pass through with the trigger model);
- `chunk_overlap` not below `chunk_size` (the router clamps it).

Run the router with the logger at `INFO`/`WARNING` while changing a config: these warnings are the cheapest way to
discover dead signals.

---

## Configuring the models

### The shipped mapping

| Mode         | Model                      | Decided by                                    | Why                                                       |
|--------------|----------------------------|-----------------------------------------------|-----------------------------------------------------------|
| `plan`       | `qwen/Qwen3.8-Flash-Next`  | `<collaboration_mode>` Plan block             | Long-context reasoning, no file edits, latency tolerant    |
| `implement`  | `qwen/Qwen3.8-Flash-Next`  | Fallback for a plain main turn                 | The hot path: cheap per token, tool-calling reliable       |
| `test`       | `qwen/Qwen3.8-27B`         | Keywords                                        | Reading failures and fixing them rewards a dense model     |
| `git_review` | `qwen/Qwen3.8-27B`         | Keywords                                        | Diff/`git log` analysis: precision over speed              |
| `review`     | `qwen/Qwen3.8-Flash-Next`  | Keywords                                        | Mostly reading and describing; short answers               |
| `debug`      | `qwen/Qwen3.8-Flash-Next`  | Keywords                                        | Long traces + hypothesis chains                            |
| `aux_title`  | `qwen/Qwen3.8-27B`         | Request class (system thread)                  | Tiny prompt, structured JSON-schema output                 |
| `compaction` | `qwen/Qwen3.8-27B`         | Request class (`request_kind == "compaction"`) | Whole-transcript summarization: capacity matters          |

The names are what this deployment uses today; the plugin is agnostic to them, it just rewrites strings.

### Model facts that matter here

From the Hugging Face model cards (retrieved 2026-09-24 — re-check for your quantization and serving stack):

| Model                        | Type                                | Native context          | Main caveat                   |
|------------------------------|-------------------------------------|-------------------------|-------------------------------|
| `qwen/Qwen3.8-Flash-Next`    | MoE, 125 B total / **6 B active**   | 262 144, up to 1 000 000 | thinking mode on by default   |
| `qwen/Qwen3.8-27B`           | dense 27 B                          | 262 144, up to 1 000 000 | thinking mode on by default   |
| `google/embeddinggemma-300m` | 300 M BiEncoder, 768-d output       | **2 048**                | gated license, no `float16`   |

Per model:

- **`qwen/Qwen3.8-Flash-Next`** — MoE with 512 experts (10 routed + 1 shared), 48 layers, hidden 2560, plus a 51 B
  n-gram embedding table and a 4 B MTP head: cheap per token, which is why it carries the hot modes. The card
  recommends `temperature=1.0, top_p=0.95, top_k=20` for thinking mode.
- **`qwen/Qwen3.8-27B`** — dense, 64 layers, hidden 5120: more expensive per token, more capacity per pass, which is
  why the transcript-sized auxiliary classes and the failure-analysis modes sit here.
- **Both chat models** expose `enable_thinking`, `preserve_thinking` and `reasoning_effort`, and are vision-language
  models (Codex only ever sends them text). Native context is 262 144 tokens, extensible to 1 000 000 with RoPE
  scaling — but only if the serving stack is configured for it.
- **`google/embeddinggemma-300m`** — sentence-transformers BiEncoder, 768-d output (MRL-truncatable to 512/256/128),
  trained on 100+ languages, which is what makes the Polish + English `examples` lists indexable at all.

Consequences worth internalizing:

- **Both chat models default to thinking mode.** The plugin forwards `reasoning` untouched, so reasoning depth comes
  from the CLI (`model_reasoning_effort`) and the server's chat-template defaults, not from routing config. If a mode
  feels slow, look there first.
- **Allocate enough output tokens.** Thinking needs room; `model_max_output_tokens` that is too tight truncates
  reasoning and looks like a broken agent, not like a routing problem.
- **The embedding model only sees 2 048 tokens.** That is why `chunk_size` is 256 and why over-length queries are
  windowed and averaged; raising `chunk_size` past 2 048 buys nothing.
- **EmbeddingGemma is gated.** Either accept the license and `huggingface-cli login`, or point `embedding_model` at a
  local directory (what this deployment does: a path under `/models/...` instead of the Hub id). Otherwise the semantic
  layer fails to load at startup and quietly disables itself.
- **Embeddings run on CPU.** A semantic decision costs an embedding pass per request that reaches layer 5; the
  deterministic layers exist precisely so most turns never pay it.

### Hard requirements on the router side

- A `model_name` the router does not know is a **downstream 4xx/5xx**, not a routing no-op. Keep the mode→model mapping
  and the router model list in sync; when you rename a deployment, both places change.
- Codex needs **tool calling**: the provider for every mode must expose it (`tool_calling: true` in the model config,
  and a server started with tool parsing enabled). A mode whose model cannot call tools makes the agent hang or emit
  text instead of `exec_command` calls — routing looks "wrong" but is configured correctly.
- **Context window:** `compaction` requests carry the entire transcript. The provider's `input_size`, the server's
  `--max-model-len` and the CLI's `model_context_window` should agree; a 256 000-token Codex window against a 32 000
  provider is a guaranteed failure on the longest turns.
- **Mixed models per session are expected.** Codex keeps history in the CLI and re-sends it every turn, so switching
  models between turns is safe; but a mode change mid-session does mean a different tokenizer and different tool-call
  formatting quirks. Keep modes that must agree on the same model if you observe drift.
- **Do not point the trigger at itself.** `model_name` equal to the trigger model in a hot mode makes routing invisible
  and hides misconfiguration; if you want "no routing for this mode", give it the model you actually want to audit.

### Choosing which mode gets which model

Practical order, cheapest first:

1. Run with the shipped mapping and read `mode=… source=… model=…` from the logs for a day of real work.
2. Move the modes you actually hit (usually `implement`, then `test`) to the model you would like to pay less for and
   compare task success; keep the auxiliary classes (`aux_title`, `compaction`) on whichever model summarizes best —
   they are invisible to you until they are wrong.
3. If a mode is over-triggered, tune the signals (see [Tuning](#tuning-and-gotchas)) before you change models; a wrong
   mode makes any model look bad.

```bash
# one mode, one process, no config edit
export LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_MODEL_TEST="qwen/Qwen3.8-Flash-Next"
# several at once
export LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_MODELS="implement=qwen/A|debug=qwen/B"
```

---

## Tuning and gotchas

- **`heuristic_min_score` is the false-positive dial.** One plain keyword is `1.0`, so the default `3.0` needs a strong
  keyword (`"debug": 3`), a phrase (`2.0`+), or a pattern (`3.0`). Lower it and generic words start stealing turns;
  raise it and specialised modes go quiet. Score math is visible in the logs via `similarity = score/(score+1)`.
- **`plan` and `implement` signals are dead by design.** Their keyword lists exist in the JSON for the embedding layer
  only. To route planning by text you would have to change `HEURISTIC_MODES` in code — prefer Plan Mode in the CLI, or
  an explicit `"agent_mode": "plan"` override.
- **Patterns must be lower-case.** Text is lower-cased before scoring; `"\bTODO\b"` never matches and `lint_signals`
  says so at startup.
- **`weights` only affects declared keywords.** A weight for a keyword that is not in `keywords` is never read.
- **Phrase weights are inline.** `"napraw błąd:3"` is a phrase worth 3; a malformed suffix (`"napraw błąd:x"`) is
  treated as part of the phrase text — a silent no-op.
- **Keep `classify_max_chars` in mind for long first prompts.** The budget applies to *older* messages; the newest one
  is always classified whole. Raising it gives the keyword and semantic layers more history (and more CPU), lowering it
  makes the decision depend on the newest message only.
- **Write Polish signals in both spellings.** There is no diacritic folding and no stemming beyond word-start prefixes:
  `błąd` never matches `blad`. Every signal users may type without Polish characters needs its ASCII twin — as the
  shipped lists already do.
- **Class routing needs Codex metadata.** If a proxy or middleware strips `client_metadata` (or the JSON string in
  `x-codex-turn-metadata`), `compaction` and `aux_title` degrade to `main` and get heuristic/fallback routing — the
  classic "why is my title generation on the coding model" symptom.
- **A persisted FAISS index is not invalidated when you edit the config.** With `PERSIST_DIR` / `vector_store_path`
  set, `index.faiss` and `docstore.pkl` are reloaded as-is. After changing modes, descriptions or examples, delete both
  files (or point at a new directory) or you keep routing against the old index.
- **Missing ML extras are silent by intent.** `semantic.enabled: true` without `faiss`/`sentence-transformers` logs
  `Codex semantic routing disabled: …` and continues deterministically. If you expect layer 5 to work, grep startup
  logs for that line.
- **One trigger per plugin.** `agentic_routing_codex` answers `auto_codex`; the sibling `agentic_routing` answers
  `auto_agentic` (Chat/agent traffic). Both can be enabled in `LLM_ROUTER_UTILS_PLUGINS_PIPELINE` at once — they do not
  interfere as long as the aliases stay distinct.
- **Trigger matching is exact after `strip()`** and case-sensitive: `" Auto_codex"` is not routed.
- **Routing is per request, not per session.** Two turns of one conversation may legitimately land on different
  models. If you need stickiness, that belongs in `agentic_routing` (session affinity), not here.

---

## Verifying the plugin

### Offline dry run (no network, no embedding model)

```python
"""Minimal end-to-end check of the Codex routing plugin (deterministic layers only)."""
import json
import os

os.environ["LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_SEMANTIC_ENABLED"] = "false"

from llm_router_plugins.utils.routing.agentic_routing.codex import CodexRoutingPlugin

INSTRUCTIONS = "You are a coding agent running in the Codex CLI."
PLAN = "<collaboration_mode># Plan Mode (Conversational)\n\nThink first.</collaboration_mode>"
DEFAULT = "<collaboration_mode># Collaboration Mode: Default\n\nExecute.</collaboration_mode>"


def turn(text, collaboration=DEFAULT, **turn_meta):
    meta = {"request_kind": "turn", "thread_source": "user", "thread_id": "t-1", "turn_id": "u-1"}
    meta.update(turn_meta)
    items = []
    if collaboration:
        items.append({"type": "message", "role": "developer",
                      "content": [{"type": "input_text", "text": INSTRUCTIONS + "\n\n" + collaboration}]})
    items.append({"type": "message", "role": "user",
                  "content": [{"type": "input_text", "text": text}]})
    return {
        "model": "auto_codex",
        "instructions": INSTRUCTIONS,
        "input": items,
        "tools": [{"type": "function", "name": "exec_command", "parameters": {}}],
        "client_metadata": {"thread_id": "t-1", "turn_id": "u-1",
                            "x-codex-turn-metadata": json.dumps(meta)},
    }


plugin = CodexRoutingPlugin(logger=None)
cases = {
    "plan mode turn": turn("Zaplanuj migrację bazy danych.", PLAN),
    "keyword turn": turn("napraw testy w tests/"),
    "plain turn": turn("Dodaj endpoint zwracający status usługi."),
    "title generation": turn("Summarize this thread in one line.", None, thread_source="system"),
    "compaction": turn("Compact the conversation.", DEFAULT, request_kind="compaction"),
}
for name, payload in cases.items():
    out = plugin.apply(payload)
    print(f"{name:20} -> model={out['model']:26} agent_mode={out['agent_mode']:11} "
          f"source={out['routing']['source']:19} sim={out['routing']['similarity']:.3f} "
          f"class={out['routing']['codex_class']}")
```

Expected output with the shipped config:

```text
plan mode turn       -> model=qwen/Qwen3.8-Flash-Next    agent_mode=plan        source=collaboration_mode  sim=1.000 class=main
keyword turn         -> model=qwen/Qwen3.8-27B           agent_mode=test        source=heuristic           sim=0.933 class=main
plain turn           -> model=qwen/Qwen3.8-Flash-Next    agent_mode=implement   source=fallback            sim=0.000 class=main
title generation     -> model=qwen/Qwen3.8-27B           agent_mode=aux_title   source=class               sim=1.000 class=aux_title
compaction           -> model=qwen/Qwen3.8-27B           agent_mode=compaction  source=class               sim=1.000 class=compaction
```

### Router smoke test

```bash
curl -sS http://<router-host>:8081/v1/responses \
  -H "Authorization: Bearer $LLM_ROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"auto_codex","input":[{"type":"message","role":"user",
       "content":[{"type":"input_text","text":"napraw testy w tests/"}]}]}' \
  | head -c 400
```

Then confirm the router logged `mode=test source=heuristic model=qwen/Qwen3.8-27B`.

### Automated tests

```bash
python -m pytest tests/test_agentic_routing_codex.py -q      # plugin, cascade, scoring, config
python -m pytest tests/test_routing_common.py -q             # shared routing plumbing
```

The suite asserts the deterministic layers exactly and exercises the semantic layer through stub routers only, so it
runs without a model or network. `agents-conversation/codex/conv-01/` holds the captured real requests (main turn,
title, compaction) the payload builders mirror.

---

## Troubleshooting

| Symptom                                                     | Likely cause                                                     | Fix                                                                    |
|--------------------------------------------------------------|------------------------------------------------------------------|-------------------------------------------------------------------------|
| Nothing is ever rerouted                                      | plugin not in `LLM_ROUTER_UTILS_PLUGINS_PIPELINE`, or trigger mismatch | check the `[utils] Registered utility plugin` log line, then `…_TRIGGER` |
| Every turn is `implement` / `source=fallback`                 | signals too narrow, `heuristic_min_score` too high, semantic off   | lower `…_HEURISTIC_MIN_SCORE`, add phrases, check `…_SEMANTIC_ENABLED`   |
| `plan` never selected in Plan Mode                            | no `<collaboration_mode>` block reached the plugin (stripped/rewritten) | verify the `developer` message survives to the router; `"agent_mode": "plan"` as override |
| Titles or compaction on the coding model                      | `client_metadata` / `x-codex-turn-metadata` missing or unparsable   | stop stripping metadata; `routing.codex_class` in the logs tells you what was seen |
| Wrong specialised mode                                        | overlapping keywords; ties go to `test` → `git_review` → `review` → `debug` | re-weight, or move the ambiguous phrase to a `phrases` entry with a weight |
| `similarity` just under `0.51` on good matches                 | embedding model / threshold mismatch for your phrasing              | lower `…_SIMILARITY_THRESHOLD` gradually (0.45–0.5 is the usual band)     |
| `model not found` after a routing decision                     | `model_name` not declared in the router model config                | add/fix the model in `LLM_ROUTER_MODELS_CONFIG`                            |
| Agent stops calling tools on some turns                        | routed model/provider without tool parsing                          | `tool_calling: true` + a server with tool parsing for that model           |
| Longest turns fail with context errors                        | provider `input_size` < Codex `model_context_window`                 | align them; compaction requests carry the whole transcript                 |
| Semantic layer never used, no explanation at INFO             | extras missing (single `Codex semantic routing disabled: …` warning) | `pip install ".[ml]"` or set `…_SEMANTIC_ENABLED=false` on purpose          |
| Edits to modes/examples change nothing, with a persist dir     | stale `index.faiss` / `docstore.pkl` reloaded as-is                  | delete both files (or change `…_PERSIST_DIR`) and restart                    |
| Startup fails with `CodexRouting: …`                           | invalid config (trigger, modes, fallback, semantic params)           | the message names the offending key or env var                              |

---

## Reference

### Module map

| Module            | Responsibility                                                             |
|-------------------|----------------------------------------------------------------------------|
| `payload.py`      | Codex wire format → immutable `CodexRequest`; request classes; user-text assembly |
| `scoring.py`      | keyword / phrase / regex scoring, `score / (score + 1)` confidence mapping  |
| `classifier.py`   | the six-layer cascade, `HEURISTIC_MODES`, `CLASS_ROUTED_MODES`, `RoutingDecision` |
| `semantic.py`     | optional cosine-similarity layer, acceptance threshold, fail-open lookups   |
| `config.py`       | JSON loading, env overrides, `validate_args`, `lint_signals`                |
| `plugin.py`       | `CodexRoutingPlugin`: trigger gate, router construction, payload annotation  |

### Defaults at a glance

| Key                          | Default                              |
|------------------------------|--------------------------------------|
| `settings.trigger_model`     | `auto_codex`                          |
| `settings.fallback_mode`     | `implement`                           |
| `settings.heuristic_enabled` | `true`                                |
| `settings.heuristic_min_score` | `3.0`                               |
| `settings.classify_max_chars`| `4000`                                |
| `settings.semantic.enabled`  | `true`                                |
| `settings.semantic.threshold`| `0.51`                                |
| `settings.semantic.top_k`    | `3`                                   |
| `settings.semantic.chunk_size` / `chunk_overlap` | `256` / `64`          |
| keyword / phrase / pattern weight | `1.0` / `2.0` / `3.0`            |
| heuristic candidate order    | `test`, `git_review`, `review`, `debug` |
| class-routed modes           | `compaction`, `aux_title`             |
| embedding device             | `cpu`                                 |
| `MAX_QUERY_WINDOWS`          | `4` (over-length query windows)       |

### Reference deployment (example — your hosts will differ)

| Role           | Declaration                                                            | Notes                              |
|----------------|------------------------------------------------------------------------|-------------------------------------|
| trigger        | `auto_codex` → `api_type: "builtin"`                                    | resolved by this plugin, never sent upstream |
| `implement`, `plan`, `review`, `debug` | `qwen/Qwen3.8-Flash-Next` → `api_type: "vllm"`, `http://<vllm-host-1>:7000` | `input_size: 256000`, `tool_calling: true` |
| `test`, `git_review`, `aux_title`, `compaction` | `qwen/Qwen3.8-27B` → `api_type: "vllm"`, `http://<vllm-host-2>:7000` + `http://<vllm-host-3>:7000` | two replicas, balanced by the router |
| embeddings     | local directory with `google/embeddinggemma-300m`                        | CPU, no Hub download at startup     |

On the CLI side this deployment runs `model_context_window = 256000` and `model_max_output_tokens = 32000`, matching
the 256 000-token providers so that even a compaction request stays inside what the backend accepts. The shipped
`embedding_model` is a local path precisely because the Hub build is gated: with no cached weights and no
`huggingface-cli login`, the semantic layer disables itself at startup.

### See also

- [Semantic Routing Plugins reference](../../README.md#312-codex-routing-codex-cli-requests) — §3.12 of the shared
  routing README, plus its [plugin comparison](../../README.md#4-comparison-which-plugin-to-use).
- Sibling plugin `agentic_routing` (`auto_agentic`): capability filtering, declarative rules and session affinity for
  generic agent traffic — described in
  [root README §2.7.3](../../../../../README.md#273-agentic-routing-agent-work-mode).
- `llm_router_plugins/utils/routing/embedder.py`: the shared BiEncoder + FAISS router used by both plugins.
- `llm_router_plugins/resources/routing/agentic_routing_codex.json`: the shipped mode definitions and signal lists.
