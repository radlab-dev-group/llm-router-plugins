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

## Calibration and evaluation guides (PL / EN)

The full guides are stored alongside this README in `llm_router_plugins/utils/routing/agentic_routing/codex/`:

- [CODEX_EVAL_HOWTO_PL.md — Polski](CODEX_EVAL_HOWTO_PL.md)
- [CODEX_EVAL_HOWTO_EN.md — English](CODEX_EVAL_HOWTO_EN.md)

They cover configuration files and ENV overrides, heuristic scoring versus embedding similarity, descriptions/examples
and threshold/margin calibration, session-separated calibration/holdout datasets, the evaluation CLI, report variants,
precision/recall and mode-transition metrics, and safe deployment with index rebuilding. The goal is better **work-mode
selection**, not tuning for particular response-model names.

For automated threshold/margin tuning and evaluation, see
[`scripts/codex-tune-eval.sh`](../../../../../scripts/codex-tune-eval.sh) in the repository root and section 6 of either guide.
The script selects candidates on calibration, then compares source and selected configurations on holdout; it does not
overwrite production configuration or use production Redis. Use `--no-semantic` for deterministic evaluation without tuning.

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
returned untouched with the trigger model still in place. **Fail-hard on required routing configuration**: invalid
triggers, modes or routing rules reject construction. Optional memory is **fail-open**, including invalid memory
policy/ENV, a missing Redis host or client library, and connection/read/write failures: routing remains stateless.

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

The plugin sees the prepared request payload as a plain `dict` and returns it. Routing is stateless by default.
Explicitly enabled Redis memory can carry reliable phase evidence between requests and workers; it is optional,
and the full-history payload remains usable when memory is disabled or unavailable.

### Step 1 — payload normalization

`CodexPayloadParser.parse()` is the only code that knows where Codex metadata lives on the wire; everything downstream
works on an immutable `CodexRequest` snapshot. The parser is configured once (with `classify_max_chars`) and holds no
per-request state; reading never raises — an unreadable value keeps the field default.

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
| `latest_user_text`                          | newest genuine `user` command, kept whole                                 |
| `user_history`                              | earlier user commands, newest first, within `classify_max_chars`           |
| `intent_text`                               | current command; nearest earlier task only for a short confirmation       |
| `assistant_messages`                        | assistant `output_text` messages after the current command                |
| `activity`                                  | ordered assistant utterances, actual tool calls and linked outputs after that command |

Details that matter when you debug routing:

- **Collaboration mode** is read from every `role == "developer"` message; the **last** `<collaboration_mode>` block
  wins, so a Plan → Default switch mid-session is picked up on the next turn. Its first heading decides: `# Plan Mode`
  → `plan`, `# Collaboration Mode: Default` → `default`, anything else → `""`.
- **`latest_user_text`** keeps only the newest real command. Environment-only messages do not start a new task;
  an environment block attached to a command is stripped without dropping that command. With no usable user message,
  the legacy `payload["prompt"]` is used. Earlier commands live separately in bounded **`user_history`**.
- **`intent_text`** uses history only for narrowly recognised confirmations such as `tak, zrób to`, `yes, do it`,
  `ok` or `continue`. It stops at the nearest non-confirmation command; an independent new instruction never inherits
  older test/Git keywords. This is a conservative rule, not general reference resolution.
- **`assistant_messages`** and **`activity`** include only events after the newest real command. Assistant text is
  copied, and tool outputs are linked by `call_id` only to calls in that active turn. An old final answer, old failure,
  or advertised tool therefore cannot establish the new task's phase. Optional memory can supply bounded pending-call
  context for incremental requests in the same identified command generation.

### Step 2 — request class

Codex mixes three kinds of request on one endpoint. The structural ones are recognised *before* any prompt text is
read, with strict priority `compaction > aux_title > main`:

| Class        | Condition                                                       | Why it matters                                                    |
|--------------|-----------------------------------------------------------------|-------------------------------------------------------------------|
| `compaction` | `request_kind == "compaction"`                                  | Carries the whole transcript; wins over every keyword signal       |
| `aux_title`  | `thread_source == "system"` **or** the title-call shape below    | CLI-generated one-line thread title, no tools, JSON-schema output  |
| `main`       | everything else                                                 | A regular agent turn (Plan or Default collaboration mode)          |

The title call is recognised in two ways on purpose. Codex releases disagree about what they declare in
`x-codex-turn-metadata` — some emit the title on a `user` thread, or send no `client_metadata` at all — so when the
`system` thread source is missing the parser falls back to the *shape* of the call, which is what actually makes it
unmistakable: no tool advertised, a `json_schema` response format, and the CLI title instruction as the first
sentence of the user text (`generate` / `produce` / `create` / `write` / `draft` / `suggest` … `title`). A regular
agent turn advertises tools, so it can never match that shape.

### Step 3 — resolution cascade

`CodexModeClassifier.classify()` tries layers in a fixed order; the **first layer that can answer wins**. A main turn never falls below the
fallback mode: the keyword and semantic layers can specialise the decision, never weaken it.

| # | Layer                                          | `routing.source`     | Routing similarity                  |
|---|------------------------------------------------|----------------------|-------------------------------------|
| 1 | Explicit `agent_mode` / `codex_mode` / `metadata.agent_mode` in the payload | `explicit`           | `1.0`             |
| 2 | Request class — `compaction`, then `aux_title` | `class`              | `1.0`                               |
| 3 | `<collaboration_mode>` Plan block from the CLI  | `collaboration_mode` | `1.0`                               |
| 4 | Clear current assistant action or actual tool activity | `phase`         | `1.0` (rule strength, not calibrated probability) |
| 5 | Fresh reliable phase carried by optional session memory | `memory`        | `1.0` (rule strength, not calibrated probability) |
| 6 | Keyword scoring of the current user intent      | `heuristic`          | `score / (score + 1)` (heuristic strength, not calibrated probability) |
| 7 | Embedding cosine similarity over the modes      | `semantic`           | cosine of the matched mode          |
| 8 | Configured `fallback_mode` (`implement`)        | `fallback`           | cosine of that mode, else `0.0`     |

Notes:

- An explicit override must name a **configured** mode, otherwise it is ignored and the cascade continues;
  `agent_mode` is checked before `codex_mode`, and both before `metadata.agent_mode`. This is the escape hatch for
  "route this one request by hand": send `"agent_mode": "debug"` in the request body.
- Only four modes compete in the keyword layer: `test`, `git_review`, `review`, `debug`
  (`HEURISTIC_MODES`). `plan` is declared by the CLI and `implement` is the fallback, so their `keywords` / `phrases` /
  `patterns` are never scored — only their `description` and `examples` feed the embedding index. `aux_title` and
  `compaction` are class-routed and are left out of the index too.
- The phase layer recognises explicit current-action announcements and concrete executed commands/patches, rather
  than arbitrary mentions of tests or Git. Thus commit inspection can route to `git_review`, then editing `CHANGELOG`
  to `implement`; helper commands such as `git status` do not create a Git phase. The phase that dominates the last
  `settings.phase.evidence_window` strong signals wins (a window of `1` means the latest strong action alone); a weak
  signal (a linter) only carries the turn while no strong signal exists, so `ruff` after `git log` does not make the
  turn a style review. Unknown or ambiguous activity leaves the remaining cascade to decide. `heuristic_enabled=false`
  disables only keyword scoring, not phase detection or memory. A phase must name a configured mode.
  `settings.phase.enabled=false` disables phase evidence and carried phases, leaving keyword scoring active.
  Shell noise (redirects, `$(…)` substitutions, `2>&1`, quoted operators) is read as noise, and an executable the
  `neutral_executables` list names (or that the table simply does not name) carries no phase of its own instead of
  vetoes the segments beside it; a here-document fed to an interpreter counts as an edit when its body matches
  `implement_write_patterns`.
- Incremental payloads without a user message retain tool events. A result linked to
  a pending test call may produce fresh `test_failure` evidence; unknown execution
  status does not settle the call, and an older test result cannot replace a newer
  recognized action. Bounding a long structured result preserves its exit status.
- A memory decision uses the version read before classification when persisting.
  A concurrent update is reported as a conflict rather than silently overwriting
  a phase selected by another request.
- Layers 1–6 never touch the embedding stack. The vector store is queried **at most once per request**, and only after
  the deterministic layers have stayed silent.

#### Phase configuration

Phase signals live in `settings.phase` in `agentic_routing_codex.json`, not in Python keyword lists:

- `announcement_prefix`, `uncertain`, `announcements`: case-insensitive regexes for current-action prefixes,
  uncertainty/negation and complete actions mapped to mode names. Actions use full matches, not substring searches.
- `command_tools`, `patch_tools`: names of actual command/patch tools. Advertised tools are never signals.
- `commands`: each rule has an `executable` regex (case-sensitive full match of the executable basename), an
  `args_prefix` list of exact leading arguments, and a `mode`. A `null` mode is neutral (e.g. `git status` or `cd`).
  Unknown commands or conflicting matches make the command ambiguous; rule order does not break ties.
- `test_directories`, `test_filename_prefixes`, `test_filename_pattern`: recognition of test-only patches;
  directory names and prefixes should be lower-case, since paths are compared in lower-case.
- `test_mode`, `implement_mode`, `failure_mode`: modes for test-only patches, other patches and linked test failures.
  Failure transitions apply only to command calls routed to `test_mode`, never to unrelated tool outputs.
- `evidence_window` (optional, default `1`): how many of the most recent strong signals the phase is the mode of;
  the most recent action breaks a tie. Larger windows keep one interleaved commit between edits from flipping the
  whole turn to `git_review`.
- `neutral_executables` (optional): executables that carry no phase of their own (`echo`, `ls`, `rg`, …). They do not
  silence a recognized command beside them, and an executable the table does not name at all is skipped rather than
  treated as a veto.
- `implement_write_patterns` (optional): case-insensitive regexes over a here-document body; a match makes a
  `python3 - <<'EOF' …` interpreter call an `implement` ("scripted edit") instead of no evidence.
- `activity_description_limit` (optional, default `6`): how many of the newest distinct actions the semantic
  activity section names; repeated actions collapse into one `… xN` clause, so a long turn of reads does not spend
  the whole phase budget on one repeated action.
- A `commands` entry may also declare `"strength": "weak"`: a weak rule (a linter, a type checker) only carries the
  turn while it holds no strong signal.

All phase fields must be supplied in the loaded configuration. No fields are inherited from another JSON file.
Empty maps/lists disable those signals. Regexes are compiled when configuration is loaded. Missing fields,
invalid types, regexes, unknown fields or referenced unknown modes reject configuration loading.
Apply configuration changes by reloading/recreating the plugin; rules are not read from disk per request.

To disable phase routing without affecting the keyword layer, set `settings.phase.enabled` to `false` in the
complete phase object.

Shell/patch syntax validation, rejected unsafe shell constructs, Git option parsing and call/output linkage remain
in code. They are parser safeguards, not configurable routing signals. Extending the command list cannot bypass them.

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
| `patterns`| `3.0` per declared rule                    | `"\broot\s+cause\b"`             |

Matches are deduplicated **within each mode**: strongest weight first, then longest span for equal weights,
keeping only non-overlapping matches. A single declared rule contributes its weight **at most once**; repeating
the same keyword, phrase or regex match does not inflate the score. The retained weights are summed per mode.
Each rule supplies only its first non-negated match and never retries a later occurrence after losing
deduplication, so repeated text cannot inflate scores through keyword/phrase/pattern aliases.

`CodexModeScorer.rank_modes(text, modes)` returns a score-descending ranking of `ModeScore(mode, score, matches)`.
Each retained match is a `SignalMatch(signal, start, end, weight)`; offsets refer to the lower-cased text, with
`end` exclusive. `detect_mode()` returns `(None, top_score)` on a tied top score rather than choosing by mode order.
These result types live in `codex.scoring`.

The classifier accepts a heuristic winner only when its score reaches `settings.heuristic_min_score` (`3.0` by
default) **and** its lead over the second-best score is at least `settings.heuristic_min_margin` (`1.0` by default).
Ties are always rejected, even with a zero minimum margin. A conflict, insufficient margin or insufficient score
continues to the optional semantic layer, then fallback; it does not force a heuristic choice.

For API compatibility, heuristic `routing.similarity` remains `score / (score + 1)`: this expresses **heuristic
strength, not a calibrated probability**. For a simple example, `testy` scores `3.0`, hence similarity `0.75`.
This is illustrative score arithmetic, not a report of measurements.

#### Local negation

`settings.heuristic_negation_pattern` is a configurable regex identifying explicit prohibitions of actions in
Polish and English, such as `nie uruchamiaj` / `do not run`, `nie pisz` / `do not write`, and
`bez uruchamiania` / `without running`. It does **not** treat every `nie` as negation: `nie działa` still supplies
debugging evidence.

The regex matches the **entire prohibited fragment** and defines its own scope and boundaries; there is no
separate boundary parser. The default regex ends the span at a sentence boundary, semicolon, comma,
contrastive `ale` / `but`, or a new positive action after `i` / `and`. Thus `nie uruchamiaj testów i napisz testy`
does not suppress the separate request to write tests. An empty regex disables filtering; zero-width spans
are ignored. This is a bounded heuristic, not full NLP and not a veto on semantic routing.

The shipped `test` signals deliberately exclude standalone `spec`, `mock`, `assertion`, `fixture`, `suite` and
`coverage`: these can describe production code rather than testing. Test patterns match bounded test nouns rather
than every `test…` prefix; coverage needs a concrete action such as `increase coverage` or `sprawdź coverage`.
Test-framework names and explicit phrases such as `unit tests` and `test fixture` remain strong signals.

Likewise, repository and hosting names, generic descriptions of changes and configuration conflicts do not by
themselves establish `git_review`. Branch mentions are weak signals; reviewing or comparing a branch supplies the
stronger evidence. Git commands, commit history and pull/merge requests remain explicit Git signals. These are
heuristic safeguards, not a veto on the optional semantic layer or a general-purpose NLP/history parser.

**`plan` is never decided by keywords** (the CLI tells you, in Plan
Mode, and a planning *sentence* in Default mode is not enough), and a plain imperative like "add an endpoint" is left to
the semantic layer or to `fallback_mode` — which is exactly what `implement` means.

### Step 5 — semantic similarity (optional)

Enabled with `settings.semantic.enabled`, disabled for a single process with
`LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_SEMANTIC_ENABLED=false`.

- An `EmbeddingRouter` indexes `description` + `examples` of the six non-class-routed modes with a sliding window
  (`chunk_size` 256, `chunk_overlap` 64) using a sentence-transformers BiEncoder.
- Index vectors are L2-normalized and searched with FAISS inner product, which is cosine similarity.
- With `aggregation: "per_target_top_k"`, FAISS retrieves every indexed fragment. Each mode contributes the same
  number of its best fragments: `min(top_k, smallest indexed mode's fragment count)`. Their cosines are averaged,
  producing a complete `all_scores` ranking, including negative similarities. Larger example collections do not
  contribute more votes. A missing indexed mode rejects the lookup rather than inventing a score.
- Acceptance requires both `threshold` (shipped value `0.44`) and a strictly positive lead over the runner-up of
  at least `min_margin` (shipped value `0.005`). Ties, incomplete rankings, inconsistent winners and malformed scores
  abstain. A single configured semantic mode needs only the threshold. Margin `0.005` is a starting setting,
  not a measured or calibrated optimum.
- `CodexSemanticLayer._build_semantic_parts` separately budgets current `request.intent_text` and phase context
  (last active-turn utterance plus bounded tool action descriptions and linked execution status,
  never the raw content of files or tool output). `intent_max_chars` and `phase_max_chars`
  are both `2000` in the shipped JSON and are independent of the parser's `classify_max_chars` history budget.
  Earlier assistant turns never cross a new user-command boundary. Empty context does not call the router.
- The shared router's `route_context(parts)` encodes those sections separately, then averages and normalizes their
  vectors for one FAISS lookup. Each section has its own bounded token-window budget (`MAX_QUERY_WINDOWS = 4`),
  so a long intent cannot displace the phase before embedding. One lookup serves acceptance and fallback similarity.
- Other routing plugins retain legacy `global_top_k` aggregation unless explicitly opted in. Codex can also select
  that strategy, but an incomplete global ranking cannot meet its acceptance contract. Injected routers exposing
  only `route(text)` receive concatenated, budgeted sections; they must return a complete ranking to be accepted,
  and do not gain independent section encoding automatically.
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
| invalid memory policy/ENV, missing Redis host/client                | plugin loads, stateless routing, `routing.memory=unconfigured` |
| memory connection/read/write fails                                 | stateless routing or preserved decision, `routing.memory=unavailable` |
| memory write exhausts version-conflict retries                      | decision preserved, `routing.memory=conflict` |
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
    "heuristic_min_margin": 1.0,
    // heuristic_negation_pattern: required; use the desired regex or "" to disable
    "heuristic_weights": {"keyword": 1.0, "phrase": 2.0, "pattern": 3.0},
    // phase: required complete object; see the bundled config for an example
    "classify_max_chars": 4000,
    "vector_store_path": "",             // "" = index lives in memory only
    "semantic": {
      "enabled": true,
      "threshold": 0.44,
      "aggregation": "per_target_top_k",  // required
      "min_margin": 0.005,               // required
      "intent_max_chars": 2000,           // required
      "phase_max_chars": 2000,            // required
      "top_k": 4,
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

Required keys: top-level `settings` and `codex_modes`; inside `settings`, `trigger_model`, `fallback_mode`,
`heuristic_min_margin`, `heuristic_negation_pattern`, `heuristic_weights` and complete `phase`; inside
each mode, `name`, `model_name` and `description`. The `semantic` object must explicitly supply `aggregation`,
`min_margin`, `intent_max_chars` and `phase_max_chars`, even when disabled. Other settings retain their existing defaults (see
[Defaults at a glance](#defaults-at-a-glance)). `description` and `examples` are not decoration — with the semantic
layer enabled they *are* the classifier's training set.

`settings.heuristic_min_margin` is the required score lead over the runner-up, not a similarity difference;
zero permits any strictly positive lead but never a tie. `settings.heuristic_negation_pattern` replaces the
local-prohibition regex (escape regex backslashes in JSON). `heuristic_min_margin`, `heuristic_negation_pattern`,
`heuristic_weights` and the complete `phase` object are required in the supplied config. Older custom configs
must add them explicitly; no rules are silently copied from the bundled file. The plugin's standard config
loading chooses the bundled file only when no custom config is supplied.

`settings.heuristic_weights` configures default signal weights: `keyword: 1.0`, `phrase: 2.0`, `pattern: 3.0`.
Per-keyword `weights` and phrase `:weight` suffixes still take precedence. All three weights must be supplied;
values must be finite, nonnegative numbers. Changing them requires only changes to the loaded JSON, not edits
to the scorer.

The new semantic settings are taken only from the supplied JSON, never merged with another config.
Older custom configurations must add all four explicitly. `aggregation` accepts `per_target_top_k` or
`global_top_k`; `min_margin` must be finite in `[0, 2]` (cosine differences), and both section budgets must
be positive integers. Changing aggregation, margin or query budgets does not require rebuilding the index;
changing descriptions, examples, modes or the embedding model does. These new settings have no separate env overrides.

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
| `…_HEURISTIC_MIN_MARGIN`                                            | Minimum score lead over the runner-up (default `1.0`; ties always rejected) |
| `…_CLASSIFY_MAX_CHARS`                                              | Parser's optional history budget                      |
| `…_MODEL`                                                           | **Embedding** model for the semantic layer            |
| `…_SEMANTIC_ENABLED`                                                | Toggle the embedding layer                            |
| `…_SIMILARITY_THRESHOLD`                                            | Minimum cosine similarity for a semantic hit          |
| `…_TOP_K` / `…_CHUNK_SIZE` / `…_CHUNK_OVERLAP`                       | Embedding router knobs                                |
| `…_PERSIST_DIR`                                                     | Directory holding `index.faiss` + `docstore.pkl`      |
| `…_MODE_<name>_KEYWORDS`                                            | Pipe-separated keyword override for one mode          |
| `…_MEMORY_ENABLED`                                          | Turn the shared session memory on (default **off**)   |
| `…_MEMORY_BACKEND`                                          | `redis` (production) or `memory` (tests/replay only)   |
| `…_MEMORY_TTL_SECONDS`                                      | Lifetime of one session record (default `900`)         |
| `…_MEMORY_MAX_SESSIONS`                                     | Sessions kept in this plugin's namespace (default `10000`) |
| `…_MEMORY_MAX_EVENTS` / `…_MEMORY_MAX_CALLS`                 | Per-session identifier caps (defaults `64` / `32`)     |
| `…_MEMORY_KEY_PREFIX`                                       | Installation-specific namespace (default `llm-router:codex-routing`) |
| `…_MEMORY_MAX_RETRIES`                                      | Attempts after a version conflict (default `1`)         |
| `…_REDIS_HOST`                                              | **Empty by default — no connection, memory stays off**  |
| `…_REDIS_PORT` / `…_REDIS_DB` / `…_REDIS_PROTOCOL`           | `6379` / `0` / `3`                                      |
| `…_REDIS_PASSWORD` / `…_REDIS_USERNAME`                      | Empty password means no AUTH; username is optional ACL  |
| `…_REDIS_SSL`                                               | TLS on/off (default off)                                |
| `…_REDIS_SSL_CA_CERTS` / `…_REDIS_SSL_CERTFILE` / `…_REDIS_SSL_KEYFILE` | TLS material                                   |
| `…_REDIS_SSL_CERT_REQS`                                     | `required` (default), `optional` or `none`              |
| `…_REDIS_SOCKET_CONNECT_TIMEOUT` / `…_REDIS_SOCKET_TIMEOUT`  | Short positive timeouts, seconds (default `1.0`)        |

Booleans accept `1/0`, `true/false`, `yes/no`, `on/off`. Memory policy resolves **ENV → explicit JSON → defaults**;
Redis connection settings come exclusively from the plugin's own fixed `…_REDIS_*` variables, never from JSON,
Redis URLs, `AUTH_REDIS_*`, `LLM_ROUTER_AUTH_REDIS_*`, `LLM_ROUTER_REDIS_*` or generic `REDIS_*` variables.
Malformed memory/Redis values disable memory with `unconfigured` diagnostics rather than choosing a guessed value
or interrupting startup. Other numeric routing overrides retain their existing `int`/`float` parsing behaviour.
Unknown mode names in `MODELS`, `MODES`, `MODEL_<MODE>` or
`MODE_<name>_KEYWORDS` are logged as warnings and otherwise ignored — a typo silently does nothing rather than breaking
routing.

`…_MODES` filters the mode list **but does not move the fallback**: if the whitelist excludes the configured
`fallback_mode` (`implement`), set `…_FALLBACK_MODE` too — otherwise validation fails at startup.

### Session memory (shared, optional)

Codex asks one action per model call, and between two calls the user's instruction does not change while the evidence
does: the suite has now run, the patch has landed, the failure arrived. A stateless cascade can only read what the
client happened to send, so an **incremental payload** — the result of a call it already showed, without the call —
carries no evidence of its own and falls through to the fallback.

The session memory carries a **reliable** phase from one request of a command generation to the next. It lives in
**Redis**, shared by every Gunicorn worker pointed at the same instance and namespace; there is deliberately **no
local-process cache**, because two workers holding two ideas of the same session's phase is worse than no memory.

```bash
pip install -e ".[memory]"   # optional Redis client; without it the plugin stays stateless
export LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_REDIS_HOST=cache.internal
export LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_REDIS_PASSWORD=…   # or leave unset
export LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_MEMORY_KEY_PREFIX=installation-a:codex-routing
export LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_MEMORY_ENABLED=true
```

For ACL, set `…_REDIS_USERNAME` and provide `…_REDIS_PASSWORD` through the deployment's secret mechanism.
For TLS, set `…_REDIS_SSL=true` and `…_REDIS_SSL_CA_CERTS` as needed; client certificate/key paths support mutual
TLS. Certificate verification defaults to `required`; enabling TLS does not disable it. `optional`/`none` are
explicit insecure overrides, not production defaults. Connection settings and credentials do not belong in JSON.

**Lifecycle.** A key isolates the plugin's namespace, the session, the thread and the agent; `turn_id` is the
*generation* — the current user command. A new command resets the phase. A record holds the mode, the kind and reason
of the evidence that produced it, bounded event identifiers and a version counter: **no model names, no conversation
text, no tool output, no credentials**. Every write is an atomic versioned compare-and-set executed in a Lua script
alongside the namespace's session index, so concurrent workers cannot interleave a read, a merge and a write; the
loser gets a conflict and continues statelessly rather than overwriting. TTL (`MEMORY_TTL_SECONDS`, default 15 min)
and `MEMORY_MAX_SESSIONS` bound storage; pruning touches only keys inside `MEMORY_KEY_PREFIX`, never anything else in
the instance. Replaying the same history is a no-op — identical evidence in the same generation does not refresh a
record's standing.

**What it never does.** It never outranks a fresh signal: it is consulted only after the current turn's own evidence
stayed silent, and a phase decided from evidence always wins. It never stores a fallback, a weak semantic match, a
title request or a compaction, because those are not facts about what the agent is doing. Without a session and thread
identifier there is nothing to isolate, so routing stays stateless instead of guessing.

**Failure behaviour.** A memory error never fails a request and never activates a process-local fallback.
`routing.memory` distinguishes `unconfigured` (invalid memory policy/ENV, missing host or Redis client), `unavailable`
(failed connection/read/write), `hit`, `miss`, `expired`, `conflict` and `not consulted`; disabled memory may omit the
field. Startup status is retained even when no store exists. Failed writes do not undo a successful routing decision.
Routing diagnostics include a reason and evidence kind, but do not log raw configuration values,
credentials or full session/event/call identifiers. Redis startup, read, write, clear and cleanup failures
also emit a `WARNING` with the exception message and full traceback, even without a supplied logger.
Redis URL credentials and configured Redis username/password are masked in these tracebacks; exception
messages can still contain other operational details, so treat logs as sensitive. A damaged record, an unknown
schema version or a value written by another format is dropped and treated as a miss. Connection and socket timeouts
default to one second per operation; multiple operations/retries can take longer. Restarting a worker keeps the state;
restarting Redis without persistence loses it, which costs only the carried phase.

When Redis memory is enabled, construction checks connectivity with `PING` on a temporary client and closes it.
The operational client is initialized lazily per worker process, avoiding inherited open connections when the
application is preloaded before fork. A failed startup check leaves routing stateless until plugin recreation;
subsequent operation failures also fail open, using the configured short timeouts. Disabled memory and the in-memory
replay backend do not connect to Redis; the latter emits a warning because its state is process-local and cannot
provide production multi-worker consistency. Injected Redis clients are caller-owned and require caller-managed
fork safety and lifecycle.

**Installation isolation.** There is no automatic installation identifier in a session key. Set a distinct,
non-overlapping `MEMORY_KEY_PREFIX` for each independent installation sharing Redis; use the same prefix only for
workers intentionally sharing routing state. The default prefix alone does **not** isolate installations.
To reset memory, remove only that installation's exact versioned session keys and session index, using the correct
Redis DB, ACL and TLS settings. Never flush the database or use a broad prefix that can include another namespace.

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
| `aux_title`  | `qwen/Qwen3.8-27B`         | Request class (system thread / title shape)    | Tiny prompt, structured JSON-schema output                 |
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
- **`heuristic_min_margin` guards ambiguous intent.** The default requires a lead of `1.0` over the runner-up;
  ties always continue to semantic/fallback, even at margin `0`. Repeated rules and overlapping matches do not
  multiply the score. Heuristic similarity is rule strength, not calibrated probability.
- **Negations are local action prohibitions.** Tune `heuristic_negation_pattern` for your phrasing, not for every
  occurrence of `nie` / `not`; a failure report such as `nie działa` must remain debugging evidence.
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
- **Class routing needs Codex metadata — except the title.** `compaction` is decided by `request_kind` alone, so a
  proxy that strips `client_metadata` (or the JSON string in `x-codex-turn-metadata`) degrades it to `main` and to
  heuristic/fallback routing. `aux_title` survives that: the title-call shape (no tools + `json_schema` output + the
  title instruction) is read from the body, which is also why a title request that reaches the keyword layer is a bug
  in the shape match, not in the metadata.
- **A persisted FAISS index is not invalidated when you edit the config.** With `PERSIST_DIR` / `vector_store_path`
  set, `index.faiss` and `docstore.pkl` are reloaded as-is. After changing modes, descriptions or examples, delete both
  files (or point at a new directory) or you keep routing against the old index.
- **Missing ML extras are silent by intent.** `semantic.enabled: true` without `faiss`/`sentence-transformers` logs
  `Codex semantic routing disabled: …` and continues deterministically. If you expect layer 5 to work, grep startup
  logs for that line.
- **One trigger per plugin.** `agentic_routing_codex` answers `auto_codex`; the semantic plugins answer `auto`. All of
  them can be enabled in `LLM_ROUTER_UTILS_PLUGINS_PIPELINE` at once — they do not interfere as long as the aliases
  stay distinct.
- **Trigger matching is exact after `strip()`** and case-sensitive: `" Auto_codex"` is not routed.
- **Routing is per request, not per session.** Two turns of one conversation may legitimately land on different
  models. If you need per-session stickiness, implement it in the caller — this plugin never caches a decision.

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
    "keyword turn": turn("testy"),
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

Illustrative expected output with the shipped config (not a measurement report):

```text
plan mode turn       -> model=qwen/Qwen3.8-Flash-Next    agent_mode=plan        source=collaboration_mode  sim=1.000 class=main
keyword turn         -> model=qwen/Qwen3.8-27B           agent_mode=test        source=heuristic           sim=0.750 class=main
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
python -m pytest tests/test_codex_semantic_ranking.py -q     # balanced ranking, margin, section encoding
python -m pytest tests/test_routing_common.py -q             # shared routing plumbing
```

The suite asserts the deterministic layers exactly and exercises the semantic layer through stub routers only, so it
runs without a model or network. `agents-conversation/codex/conv-01/` holds the captured real requests (main turn,
title, compaction) the payload builders mirror.

### Semantic quality evaluation and calibration

`tests/data/codex_routing_quality.json` is a small, manually labelled main-turn corpus, not a record of the router's
historical choices. It distinguishes Git-related product code from Git inspection, demo doubles from automated tests,
test repair from diagnosing application failures, and README/config consistency review from running tests. It includes
new-task boundaries, short confirmations and `git_review → implement` / `test → debug` phase sequences.
The three log-derived prompts have file/line provenance and reconstructed minimal inputs; synthetic continuations
are marked explicitly. Labels describe the current action, not every eventual deliverable of a compound task.

Run from the repository root with the project's Python environment and optional embedding dependencies:

```bash
python -m llm_router_plugins.utils.routing.agentic_routing.codex.evaluation \
  --config llm_router_plugins/resources/routing/agentic_routing_codex.json \
  --dataset tests/data/codex_routing_quality.json \
  --split calibration > codex-calibration.json
```

Add `--no-semantic` to replay the deterministic cascade and the stateful variant without loading an embedding model at
all — that is the reproducible half of the comparison, and it runs in milliseconds.

Use `--split holdout` for the separate check set (the default), or `all` for diagnostics only. The evaluator loads the
supplied JSON without environment overrides and rebuilds the index in memory; stale persisted embeddings cannot
hide description/example changes. Missing model/dependencies, failed lookups and incomplete rankings stop evaluation
instead of silently measuring disabled semantics. No generation model/provider is called.

The report measures four variants of the same replay:

| Variant         | What ran                                                              |
|-----------------|------------------------------------------------------------------------|
| `deterministic` | the cascade with no embedding layer — what `semantic.enabled=false` runs |
| `stateful`      | the same cascade plus the session memory, one fresh store per session sequence |
| `cascade`       | the cascade with the semantic layer                                    |
| `semantic_only` | semantic ranking for main turns; request-class routing for helpers; not an upper bound |

Every routing metric is computed on the **mode**, not the model: mode accuracy, the per-mode confusion matrix, and
per-sequence mode transitions (`pairs`, `expected_switches`, `actual_switches`, `missed_switches`,
`unnecessary_switches`, `mean_switch_delay`, `censored_switches`). Special requests are
measured separately and never create main-phase transitions. Ambiguous prefixes break
the labeled transition run; unresolved switch delays are censored at its boundary.
Reassigning `model_name` in the configuration moves the auxiliary model
metrics and leaves the mode metrics byte-for-byte identical, which is what makes a routing change measurable at all.
Cases marked `ambiguous` — those where nothing before the decision determines a mode — are counted separately and
excluded from accuracy, so an unlabeled prefix can neither inflate nor pollute a number. Expected models always come
from the supplied mode table, not hard-coded model names. Initialization and routing timings are separate; two diagnostic passes and
cache warm-up mean these are not a production latency benchmark. Config/dataset hashes identify each run.

Tune descriptions/examples and threshold/margin only on `calibration`, then compare an untouched `holdout` run with
the previous configuration using the same embedding weights and dataset. Do not copy evaluation prompts into indexed
examples; if tuning against holdout, retire it and create a new independent set. Prioritize wrong **mode** selections
and false switches, then per-mode recall and abstention. This small corpus is a regression starting point, not proof
of statistical quality. No measured improvement or optimal threshold is claimed by adding it. The embedding model,
model mapping and acceptance thresholds remain unchanged; measuring answer quality, token cost and provider latency
requires a separate generation experiment on both target models. Rebuild any production persisted index after changing
examples/descriptions. `tests/test_codex_routing_quality.py` covers evaluation plumbing only, without loading a model.
`tests/data/codex_routing_baseline.json` is a frozen replay snapshot with unverified
provenance, not proof of the implementation before all phase/memory changes. Use
`--baseline tests/data/codex_routing_baseline.json` to compare all variants only on
shared case IDs and unchanged labels, with the same current metric definitions.
New cases never receive invented historical predictions. The per-case regression
test prevents an aggregate gain from hiding a loss on a formerly correct case.
The final audit results and limits of this small, previously known holdout are in
`improve-codex-work-mode-routing-verification.md`; they are not production-quality claims.

---

## Troubleshooting

| Symptom                                                     | Likely cause                                                     | Fix                                                                    |
|--------------------------------------------------------------|------------------------------------------------------------------|-------------------------------------------------------------------------|
| Nothing is ever rerouted                                      | plugin not in `LLM_ROUTER_UTILS_PLUGINS_PIPELINE`, or trigger mismatch | check the `[utils] Registered utility plugin` log line, then `…_TRIGGER` |
| Every turn is `implement` / `source=fallback`                 | signals too narrow, score/margin too high, tied scores, semantic off | inspect scores and local negations; tune score/margin and phrases; check `…_SEMANTIC_ENABLED` |
| `plan` never selected in Plan Mode                            | no `<collaboration_mode>` block reached the plugin (stripped/rewritten) | verify the `developer` message survives to the router; `"agent_mode": "plan"` as override |
| Titles or compaction on the coding model                      | `client_metadata` / `x-codex-turn-metadata` missing or unparsable   | stop stripping metadata; `routing.codex_class` in the logs tells you what was seen |
| Ambiguous intent reaches semantic/fallback                     | tied heuristic scores or insufficient lead over the runner-up    | add specific signals or re-weight; lowering the margin never accepts a tie |
| `similarity` just under `0.44` on good matches                 | embedding model / threshold mismatch for your phrasing              | lower `…_SIMILARITY_THRESHOLD` gradually (0.40–0.44 is the usual band)     |
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
| `scoring.py`      | `CodexModeScorer`: deduplicated keyword / phrase / regex scoring, local negations, `ModeScore` / `SignalMatch` ranking |
| `classifier.py`   | `CodexModeClassifier` cascade, `HEURISTIC_MODES`, `CLASS_ROUTED_MODES`, `RoutingDecision` |
| `semantic.py`     | optional cosine ranking, threshold + margin, section budgets, fail-open lookups |
| `state.py`        | optional shared session memory: keys, TTL/caps, versioned compare-and-set, Redis and in-memory stores, ENV contract |
| `config.py`       | JSON loading, env overrides, `validate_args`, `lint_signals`                |
| `plugin.py`       | `CodexRoutingPlugin`: trigger gate, router construction, payload annotation  |

### Defaults at a glance

| Key                          | Default                              |
|------------------------------|--------------------------------------|
| `settings.trigger_model`     | `auto_codex`                          |
| `settings.fallback_mode`     | `implement`                           |
| `settings.heuristic_enabled` | `true`                                |
| `settings.heuristic_min_score` | `3.0`                               |
| `settings.heuristic_min_margin` | `1.0`                              |
| `settings.heuristic_weights` | `keyword: 1.0`, `phrase: 2.0`, `pattern: 3.0` |
| `settings.heuristic_negation_pattern` | packaged explicit PL/EN action-prohibition regex |
| `settings.classify_max_chars`| `4000`                                |
| `settings.memory.enabled`    | `false` — routing is stateless unless turned on explicitly |
| `settings.memory.backend`    | `redis`; `memory` is for replay and tests only |
| `settings.memory.ttl_seconds` | `900`                                |
| `…_REDIS_HOST`               | empty — no connection configured      |
| `settings.semantic.enabled`  | `true`                                |
| `settings.semantic.threshold`| `0.44`                                |
| `settings.semantic.aggregation` | `per_target_top_k`                  |
| `settings.semantic.min_margin` | `0.005`                              |
| `settings.semantic.intent_max_chars` / `phase_max_chars` | `2000` / `2000` |
| `settings.semantic.top_k`    | `4`                                   |
| `settings.semantic.chunk_size` / `chunk_overlap` | `256` / `64`          |
| keyword / phrase / pattern weight | `1.0` / `2.0` / `3.0`            |
| heuristic candidate order    | `test`, `git_review`, `review`, `debug` |
| class-routed modes           | `compaction`, `aux_title`             |
| embedding device             | `cpu`                                 |
| `MAX_QUERY_WINDOWS`          | `4` per semantic context section      |

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

- [Semantic Routing Plugins reference](../../README.md#3-codex-routing-codex-cli-requests) — §3 of the shared
  routing README, plus its [plugin comparison](../../README.md#4-comparison-which-plugin-to-use).
- `llm_router_plugins/utils/routing/embedder.py`: the shared BiEncoder + FAISS router used by both plugins.
- `llm_router_plugins/resources/routing/agentic_routing_codex.json`: the shipped mode definitions and signal lists.
