## Overview

The **LLM‑Router** project ships with a modular plugin system that lets you plug‑in **anonymizers** (also called
*maskers*) and **guardrails** into request‑processing pipelines.  
Each plugin implements a tiny, well‑defined interface (`apply`) and can be composed in an ordered list to form a *
*pipeline**. Pipelines are instantiated by the `MaskerPipeline` and `GuardrailPipeline` classes and are driven
automatically by the endpoint logic in `endpoint_i.py`.

---

## 1. Anonymizers (Maskers)

### 1.1 What they do

* **Goal** – Remove or replace personally‑identifiable information (PII) from a payload before it reaches the LLM or an
  external service.
* **Typical strategy** – Run a pipeline of maskers that locate spans corresponding to IDs, emails, IPs, etc., and
  replace each span with a placeholder such as `{{MASKED_ITEM}}`.

### 1.2 Built‑in anonymizer plugins

| Plugin                                         | Description                                                                                                                                       | Technical notes                                                                                                                                                            |
|------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **FastMaskerPlugin** (`fast_masker_plugin.py`) | Thin wrapper around the `FastMasker` utility class. Receives a JSON‑compatible payload and returns the same payload with all detected PII masked. | Implements `PluginInterface`. The heavy lifting is delegated to `FastMasker.mask_payload(payload)`. No extra I/O; the `FastMasker` instance is created once in `__init__`. |

### 1.3 How a masker is used

1. The endpoint (e.g. `EndpointI._do_masking_if_needed`) checks the global flag `FORCE_MASKING`.
2. If enabled, it creates a `MaskerPipeline` with the list of masker plugin identifiers (e.g. `["fast_masker"]`).
3. The pipeline calls each plugin’s `apply` method sequentially, feeding the output of one as the input of the next.
4. The final payload – now stripped of PII – proceeds to the rest of the request flow (guardrails, model dispatch,
   etc.).

---

## 2. Guardrails

### 2.1 What they do

* **Goal** – Verify that a request (or its response) complies with policy rules (e.g. no hateful, illegal, or unsafe
  content).
* **Typical strategy** – Split the payload into manageable text chunks, run a pipeline of guardrails, aggregate
  per‑chunk scores, and decide whether the overall request is safe.

### 2.2 Built‑in guardrail plugins

| Plugin                                             | Description                                                                                                                                                                                                                  | Technical notes                                                                                                                                                                                                                       |
|----------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **NASKGuardPlugin** (`nask_guard_plugin.py`)       | HTTP‑based guardrail that forwards the payload to the external NASK guardrail service (`/nask_guard` endpoint) and returns a boolean *safe* flag together with the raw response.                                             | Inherits from `HttpPluginInterface`. The `apply` method calls `_request(payload)` (provided by the base class) and extracts `results["safe"]`. Errors are caught and logged; on failure the plugin returns `(False, {})`.             |
| **SojkaGuardPlugin** (`sojka_guard_plugin.py`)     | HTTP‑based guardrail that forwards the payload to the **Sójka** guardrail service (`/sojka_guard` endpoint) and returns a safety flag.                                                                                       | Mirrors the design of `NASKGuardPlugin`. The `endpoint_url` is built from the `LLM_ROUTER_GUARDRAIL_SOJKA_GUARD_HOST` environment variable. On success it returns `(True, response)`, otherwise `(False, {})`.                        |
| **(Implicit) GuardrailProcessor** (`processor.py`) | Core logic used by the internal NASK guardrail Flask route (`nask_guardrail`). Tokenises the payload, creates overlapping chunks, runs a Hugging‑Face `text‑classification` pipeline, and produces a detailed safety report. | Handles model loading (`AutoTokenizer`, `pipeline("text‑classification")`), chunking (`_chunk_text`), and scoring thresholds (`MIN_SCORE_FOR_SAFE`, `MIN_SCORE_FOR_NOT_SAFE`). Returns a dict: `{"safe": <bool>, "detailed": [...]}`. |

### 2.3 How a guardrail is used

1. The endpoint calls `_is_request_guardrail_safe(payload)` (or the analogous response guardrail).
2. If `FORCE_GUARDRAIL_REQUEST` is true, a `GuardrailPipeline` is built from the configured plugin IDs (e.g.
   `["nask_guard", "sojka_guard"]`).
3. The pipeline iterates over each guardrail plugin; each `apply` returns `(is_safe, message)`.
4. The first plugin that reports `is_safe=False` short‑circuits the pipeline and the request is rejected with a 400/500
   error payload.

---

## 2.5 ML-Based PII Classification

For cases where regex patterns alone are insufficient (e.g. context-dependent PII detection), the project integrates
with the **anonymizer-model** repository:

**Repository**: [radlab-dev-group/anonymizer-model](https://github.com/radlab-dev-group/anonymizer-model)

| Feature               | Description                                                                                     |
|-----------------------|-------------------------------------------------------------------------------------------------|
| **Approach**          | NER (Named Entity Recognition) model based on Hugging Face `AutoModelForTokenClassification`    |
| **What it does**      | Identifies PII entities in Polish text with context-aware detection (not just pattern matching) |
| **Training pipeline** | Configurable via JSON, logs to W&B, exports best model (F1 macro) to `final_model/`             |
| **REST API**          | Flask service supporting multiple model versions with optional dynamic quantization             |
| **CLI**               | `pii-classifier convert` / `generalise` / `report` for data prep and analysis                   |
| **Inference**         | Sub-token merging, punctuation cleaning, gap preservation for human-readable entity spans       |
| **Web tester**        | HTML/JS interface for real-time PII detection testing                                           |

### How it complements regex maskers

| Approach                              | Strength                                                           | Best for                                                    |
|---------------------------------------|--------------------------------------------------------------------|-------------------------------------------------------------|
| **Regex maskers** (FastMasker)        | Deterministic, zero false negatives for known formats, fast        | Structured IDs (KRS, NIP, PESEL, NRB, REGON, VIN, etc.)     |
| **PII classifier** (anonymizer-model) | Context-aware, handles unknown formats, generalizes across domains | Free-text entities, names, addresses, context-dependent PII |

### Quick start

```bash
# Clone and install
git clone https://github.com/radlab-dev-group/anonymizer-model.git
cd anonymizer-model
pip install .

# Run the API (serves multiple model versions)
python3 -m pii_classification.api.app

# API endpoints
#   GET  /models   — list available models + default
#   POST /predict   — { "text": "...", "model": "optional_name" }
```

---

## 2.6 Polish Identification Regex Patterns

The **FastMaskerPlugin** ships with a comprehensive set of rules for detecting Polish business and personal identifiers.
These rules use regex matching + checksum/form validation to minimize false positives.

### Available Polish rules

| Rule                | Placeholder        | Format                                            | Validation                                           |
|---------------------|--------------------|---------------------------------------------------|------------------------------------------------------|
| **KrsRule**         | `{{KRS}}`          | `1234567890` or `123-456-78-90` / `123 456 78 90` | 10 digits (format-only)                              |
| **NrbRule**         | `{{NRB}}`          | `PL58105012981000009062923173` (26 digits)        | 26 digits                                            |
| **NipRule**         | `{{NIP}}`          | `1234567890` or `123-456-78-90`                   | Checksum with weights `[6,5,7,2,3,4,5,6,7]` mod 11   |
| **PeselRule**       | `{{PESEL}}`        | 11 digits                                         | Checksum with weights `[1,3,7,9,1,3,7,9,1,3]` mod 10 |
| **RegonRule**       | `{{REGON}}`        | 9 or 14 digits                                    | Checksum (different weights per form)                |
| **BankAccountRule** | `{{BANK_ACCOUNT}}` | `PL58 1050 1298 1000 0090 6292 3173`              | Polish IBAN (28 chars), supports masked `XX` groups  |

### How they work

Each rule follows the same pattern:

1. **Regex match** — detects candidate strings (e.g. 10-digit sequences for KRS)
2. **Checksum validation** — validates the candidate against the official algorithm
3. **Placeholder replacement** — valid matches are replaced with `{{PLACEHOLDER}}`; invalid ones are left untouched
4. **Optional anonymizer** — if an `anonymizer_fn` is provided, it's called with `(original, tag_type)` and its result (
   wrapped in `{}`) is used instead

### Adding rules to the pipeline

To enable Polish rules in your masking pipeline, add their plugin IDs to `MASKING_STRATEGY_PIPELINE`:

```python
MASKING_STRATEGY_PIPELINE = ["fast_masker"]
```

All Polish rules are included in the FastMasker plugin by default. For a full list of available rules, see
the [fast_masker README](llm_router_plugins/maskers/fast_masker/README.md).

---

## 2.7 Semantic Routing (Model Selection)

Four routing plugins are available for model selection. The two semantic plugins
(`simple_semantic_routing`, `semantic_biencoder_routing`) activate when `payload["model"] == "auto"`; the agentic
plugin (`agentic_routing`) activates when `payload["model"]` matches its own trigger value (`"agentic"` by default)
and the Codex plugin (`agentic_routing_codex`) activates on `"auto_codex"`, routing the OpenAI-Responses-style
requests emitted by the Codex CLI agent by request class and work mode.

### 2.7.1 Simple Semantic Routing (Heuristic)

The **Simple Semantic Routing plugin** (`simple_semantic_routing`) performs
two-stage heuristic model selection: it classifies the user's intent
(code, math, creative, general) via weighted keywords, multi-word phrases, and
regex patterns, then estimates input complexity (token count) to pick the most
appropriate model from a configured pool.

No embedding model is required — routing is a fast, pure-text classification.

**1. Intent scoring — each intent accumulates a score from three sources:**

| Source       | How it works                                                                                                                                             |
|--------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Keywords** | Each keyword from the JSON has an optional weight. If the keyword is found in the lower-cased input text the score increases by that weight (default 1). |
| **Phrases**  | Multi-word expressions like `"write code:5"` or `"debug:2"`. If the phrase is found the score increases by the specified weight (default 2.0).           |
| **Patterns** | Regex patterns — if a pattern matches the input each match adds **3.0** to the score.                                                                    |

The intent with the **highest total score** wins. If no intent score exceeds zero, the intent is classified as `"none"`.

**2. Complexity estimation — token count:**

The plugin estimates the number of tokens using a simple word-count heuristic:

```
token_estimate = len(input_text.split()) × 1.25
```

This is compared against two thresholds from the config (`simple`, `medium`):

| Tokens               | Complexity |
|----------------------|------------|
| ≤ `simple` threshold | `simple`   |
| ≤ `medium` threshold | `medium`   |
| > `medium` threshold | `complex`  |

**3. Model selection:**

The pool of models is a list defined in `default_models` (e.g. `["gpt-oss:120b", "qwen3.6:35b"]`). The complexity and
intent together determine the index into this pool:

- Complexity maps to a base index: `simple → 0` (first / smallest model), `medium → n // 2` (middle), `complex → n-1` (
  last / largest).
- Intent-adjustment from the config (e.g.
  `{"code": "medium", "math": "medium", "creative": "simple", "general": "simple"}`) can **increase** the base index but
  never decrease it.
- The final index is clamped to `[0, n-1]` and the model at that index is selected.
- If no text is found or the intent is `"none"` the `default_models["simple"]` model is used as fallback.

Configuration is entirely JSON-driven in
[simple_semantic.json](llm_router_plugins/resources/routing/simple_semantic.json)
with intent definitions (keywords, phrases, patterns, weights) and two
complexity thresholds (`simple` / `medium`).

**Environment variable overrides:**

| Env variable                               | Purpose                                        |
|--------------------------------------------|------------------------------------------------|
| `LLM_ROUTER_ROUTING_COMPLEXITY_THRESHOLDS` | Pipe-separated `simple                         |medium` token thresholds |
| `LLM_ROUTER_ROUTING_MODELS`                | Pipe-separated model names for the pool        |
| `LLM_ROUTER_ROUTING_DEFAULT_MODEL`         | Fallback model when no text or no intent match |
| `LLM_ROUTER_ROUTING_INTENT_<name>`         | Override intent keywords (pipe-separated)      |

### 2.7.2 Bi-Encoder Semantic Routing (Model Selection)

The **Bi-Encoder routing plugin** (`semantic_biencoder_routing`) uses a neural embedding model
(**radlab/semantic-euro-bert-encoder-v1**) to compute semantic embeddings for a set of pre-configured routing targets.
Each target has a `name`, a `model_name` (the model to route to), a `description`, and a list of `examples`.
At query time the user message is embedded and matched against all stored target embeddings using FAISS
(`IndexFlatIP` on L2-normalised vectors = cosine similarity). The best-matching target determines the selected model.

**1. Index building (on first load or when the persist directory is missing):**

- For each target, its `description` and `examples` are combined into text.
- The text is split into overlapping **token chunks** using a sliding window (`chunk_size` tokens, `chunk_overlap`
  tokens overlap).
- Each chunk is embedded via the BiEncoder model (e.g. `radlab/semantic-euro-bert-encoder-v1`).
- All embedding vectors are **L2-normalised** to unit length.
- Vectors are inserted into a `faiss.IndexFlatIP` index (inner product).
- A docstore maps each FAISS doc ID to its target name (for reverse lookup).

**2. Routing (query):**

- The user message is embedded and L2-normalised.
- A message longer than the model's `max_seq_length` is first split into overlapping token windows (window =
  `max_seq_length`, overlap = `chunk_overlap` clamped to `max_seq_length // 4`), capped at the first 4 windows from
  the head of the text (`MAX_QUERY_WINDOWS`). The windows are encoded in a single batch and combined into one unit-norm
  query vector — the L2-normalised mean of the per-window unit vectors — so the cosine scale of the lookup is
  unchanged. Each extra window costs roughly ~1.2 s of CPU latency.
- FAISS performs a nearest-neighbor search returning the `top_k` closest chunks.
- Scores are **aggregated per target**: the mean cosine similarity of all chunks belonging to the same target is
  computed.
- The target with the **highest mean similarity** wins and its `model_name` is returned.

**3. Persistence:**

The FAISS index and docstore are saved to disk (files `index.faiss` and `docstore.pkl`) under the configured persist
directory.
On subsequent starts the index is loaded from disk — embeddings are **not recomputed**.
If the embedding model changes (different output dimension) the index is automatically rebuilt.

Configuration is loaded from
[semantic_biencoder.json](llm_router_plugins/resources/routing/semantic_biencoder.json). The embedding model, chunk
size, overlap, and persist directory can all be overridden via environment variables:

| Env variable                                          | Purpose                               |
|-------------------------------------------------------|---------------------------------------|
| `LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_MODEL`         | Override the embedding model name     |
| `LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_TARGETS`       | Pipe-separated list of target names   |
| `LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_CHUNK_SIZE`    | Override chunk size                   |
| `LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_CHUNK_OVERLAP` | Override chunk overlap                |
| `LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_PERSIST_DIR`   | Directory for FAISS index persistence |

### 2.7.3 Agentic Routing (Agent Work Mode)

The **Agentic Routing plugin** (`agentic_routing`) picks the model from the agent's **work mode** (planning, coding,
reviewing, debugging, …) rather than from a generic intent classification. It activates only when `payload["model"]`
is a string whose trimmed value is in the configured trigger list — `"agentic"` by default. Matching is
case-sensitive, so `"Agentic"` is left untouched; `" agentic "` is accepted.

The plugin rewrites `payload["model"]` and adds `payload["agent_mode"]` plus a `payload["routing"]` metadata block.
Nothing else is touched — temperature, `max_tokens`, prompts and tool definitions pass through unchanged.

Selection is **deterministic first, semantic last**: anything that can be derived from declared facts — an explicit
mode, a rule, session memory, a keyword — is decided in plain Python without ML, and embeddings are consulted only
when the deterministic layers stay silent. With `SEMANTIC_ENABLED=false` no ML library is imported at all. See
[routing/README.md §3](llm_router_plugins/utils/routing/README.md) for the full reference.

**1. Agent modes:**

Modes are entirely configuration-driven via
[agentic_routing.json](llm_router_plugins/resources/routing/agentic_routing.json). Each mode declares a `name`, a
`model_name`, a `description`, `examples`, heuristic signals (`keywords`, `phrases`, `patterns`) and optional
`capabilities` (`tool_calling`, `parallel_tool_calls`, `reasoning`, `vision`, `structured_output`,
`context_window`). The shipped default defines eight modes:

| Mode        | Model           | Typical work                                          |
|-------------|-----------------|-------------------------------------------------------|
| `plan`      | `qwen3.6:35b`   | Roadmaps, breaking work into steps, scope, milestones |
| `code`      | `gpt-oss:120b`  | Writing, implementing and refactoring source code     |
| `review`    | `granite3.3:8b` | Merge-request audits, style and security feedback     |
| `test`      | `granite3.3:8b` | Unit tests, fixtures, coverage, assertions            |
| `debug`     | `gpt-oss:120b`  | Tracebacks, exceptions, crashes, reproducing faults   |
| `research`  | `qwen3.6:35b`   | Investigating docs, comparing and analysing sources   |
| `summarize` | `qwen3.6:35b`   | Condensing long documents, threads and logs           |
| `fallback`  | `qwen3.6:35b`   | General-purpose catch-all when nothing matches        |

**2. Request signals (agent-aware payload):**

`signals.py` is the only place that knows where agent metadata lives in a payload; every other layer works on the
normalized `RequestSignals` snapshot. Locations are read in priority order — top level → `metadata` → `agent` — and
the first non-empty declaration wins.

```json
{
  "model": "agentic",
  "agent": "codex",
  "session_id": "abc123",
  "task": "coding",
  "tools": true,
  "reasoning": true,
  "context_tokens": 42000
}
```

Recognized signals: `agent`, `session_id` (also `conversation_id`, `thread_id`), `task`, `tools` (boolean or a list
of tool definitions, from which `tool_count` is derived), `reasoning` (also `thinking`, `reasoning_effort`) and
`vision` (flag, or an image part in the message content). Missing fields simply constrain nothing, and signal
parsing never raises.

**3. Mode detection cascade:**

The mode is resolved strictly in order — the first layer that produces an answer wins. Layers 1–4 are pure Python:

| # | Layer           | `source`                  | How it decides                                                                                     | ML    |
|---|-----------------|---------------------------|----------------------------------------------------------------------------------------------------|-------|
| 1 | **Explicit**    | `explicit`                | `agent_mode` → `mode` → `agent.mode` → `metadata.agent_mode`; first present key wins                | no    |
| 2 | **Rules**       | `rules`                   | Declarative `when`/`then` rules over the signals, highest `priority` first, all conditions AND-ed    | no    |
| 3 | **Affinity**    | `affinity`                | The mode already chosen for this `session_id`, while it is alive in the TTL/LRU cache                | no    |
| 4 | **Heuristic**   | `heuristic`               | Weighted keywords (1), `text:weight` phrases (default 2.0), regex patterns (+3.0); best score above 0 | no   |
| 5 | **Semantic**    | `semantic`                | Bi-encoder + FAISS match against mode descriptions/examples, accepted at `similarity >= threshold`   | yes   |
| 6 | **Fallback**    | `fallback` / `empty_text` | The configured `fallback_mode`                                                                       | no    |

An explicit mode name is normalised (strip, lower-case, `-` and space → `_`), so a custom `deep_research` mode also
matches `"Deep Research"` and `"deep-research"`. An unknown name only logs a warning and the cascade continues instead
of failing. An explicit mode is honoured even when the payload carries no text; whitespace-only text is stripped, so
empty extracted text *without* an explicit mode resolves to the fallback mode with `source = "empty_text"` — blank
text never reaches FAISS.

Rules intentionally **beat** affinity: a rule is the operator's policy for this request, affinity is only the memory
of the conversation. Set `rules_enabled=false` to make sessions stick.

**4. Capabilities and escalation:**

When `capabilities_enabled` is on, the signals become requirement checks (`tools` → `tool_calling`, `reasoning` →
`reasoning`, `vision` → `vision`, `context_tokens` → `context_window`). If the mode selected by any of layers 1–4
does not satisfy them and `escalation_enabled` is on, the request moves to the most capable alternative — the
declared `context_window` ranks candidates — and `routing.escalated` records the move. The reported `agent_mode` is
the escalated mode, the original one stays available as `routing.escalated_from`. A mode with no `capabilities`
block never blocks: absent facts are treated as satisfied, and when no capable alternative exists the original mode
is kept with a warning.

**5. Session affinity:**

A `session_id` keeps its resolved mode for `ttl_seconds` (sliding on every hit, LRU-bounded, thread-safe), so a
multi-turn agent does not ping-pong between models. Fallback and empty-text decisions are never cached, and
`plugin.reset_sessions()` clears the cache. The `session` block appears in `routing` only when the request declared a
session id.

**6. Result metadata:**

```python
payload = {"model": "agentic", "prompt": "Please refactor this function to add caching"}

# after apply():
{
    "model": "gpt-oss:120b",
    "prompt": "Please refactor this function to add caching",
    "agent_mode": "code",
    "routing": {
        "plugin": "agentic_routing",
        "agent_mode": "code",
        "source": "heuristic",
        "similarity": 0.9,
    },
}
```

A rule hit adds `rule_id`, an escalation adds `escalated` and `escalated_from`, and a request carrying a
`session_id` adds a `session` block:

```python
payload = {
    "model": "agentic",
    "agent": "codex",
    "session_id": "abc123",
    "task": "review",
    "tools": True,
    "prompt": "Check this diff for problems",
}

# after apply(): the task-review rule wins, then the capability gate escalates because tools are required
{
    "model": "qwen3.6:35b",
    "agent_mode": "plan",
    "routing": {
        "plugin": "agentic_routing",
        "agent_mode": "plan",
        "source": "rules",
        "rule_id": "task-review",
        "similarity": 1.0,
        "escalated": True,
        "escalated_from": "review",
        "session": {"session_id": "abc123", "mode": "plan", "model": "qwen3.6:35b", "reused": False},
    },
}
```

The next turn that reuses `"session_id": "abc123"` resolves with `source: "affinity"` and `"reused": True`.

`similarity` carries `1.0` for an explicit mode, a rule and an affinity hit, `score / (score + 1)` for a heuristic
hit, the FAISS score on the semantic path and `0.0` for the fallback.

**Environment variable overrides:**

| Env variable                                          | Purpose                                            |
|-------------------------------------------------------|----------------------------------------------------|
| `LLM_ROUTER_ROUTING_AGENTIC_CONFIG`                   | Path to a custom JSON config, or a raw JSON string |
| `LLM_ROUTER_ROUTING_AGENTIC_TRIGGER`                  | Pipe-separated trigger values (default `agentic`)  |
| `LLM_ROUTER_ROUTING_AGENTIC_MODEL`                    | Override the **embedding** model name              |
| `LLM_ROUTER_ROUTING_AGENTIC_MODELS`                   | Per-mode models, e.g. `plan=model_a\|code=model_b` |
| `LLM_ROUTER_ROUTING_AGENTIC_MODES`                    | Whitelist of mode names to keep                    |
| `LLM_ROUTER_ROUTING_AGENTIC_RULES_ENABLED`            | `true`/`false` — toggle the declarative rules layer |
| `LLM_ROUTER_ROUTING_AGENTIC_CAPABILITIES_ENABLED`     | `true`/`false` — derive requirements from signals  |
| `LLM_ROUTER_ROUTING_AGENTIC_ESCALATION_ENABLED`       | `true`/`false` — escalate modes lacking capabilities |
| `LLM_ROUTER_ROUTING_AGENTIC_SESSION_AFFINITY_ENABLED` | `true`/`false` — toggle per-session stickiness     |
| `LLM_ROUTER_ROUTING_AGENTIC_SESSION_TTL_SECONDS`      | Affinity entry lifetime, refreshed on hit (900)    |
| `LLM_ROUTER_ROUTING_AGENTIC_SESSION_MAX_ENTRIES`      | Affinity cache size (default 1024)                 |
| `LLM_ROUTER_ROUTING_AGENTIC_SEMANTIC_ENABLED`         | `true`/`false` — toggle the embedding layer        |
| `LLM_ROUTER_ROUTING_AGENTIC_SIMILARITY_THRESHOLD`     | Minimum cosine similarity for a semantic hit       |
| `LLM_ROUTER_ROUTING_AGENTIC_TOP_K`                    | Chunks retrieved per query                         |
| `LLM_ROUTER_ROUTING_AGENTIC_CHUNK_SIZE`               | Token chunk size used when indexing modes          |
| `LLM_ROUTER_ROUTING_AGENTIC_CHUNK_OVERLAP`            | Token overlap between chunks                       |
| `LLM_ROUTER_ROUTING_AGENTIC_PERSIST_DIR`              | Directory for FAISS index persistence              |
| `LLM_ROUTER_ROUTING_AGENTIC_FALLBACK_MODE`            | Mode used when nothing matches                     |
| `LLM_ROUTER_ROUTING_AGENTIC_MODE_<name>_KEYWORDS`     | Pipe-separated keyword override for a single mode  |

`MODES` filters the mode list but does not move the fallback: if the whitelist excludes the configured
`fallback_mode`, set `FALLBACK_MODE` as well — otherwise configuration validation fails at startup. Whitelisting a
mode also drops the rules that pointed at a removed mode, with a warning.

#### Codex CLI routing (`auto_codex`)

The **Codex Routing plugin** (`agentic_routing_codex`, `utils/routing/agentic_routing/codex/`) routes the
OpenAI-Responses-style requests emitted by the **Codex CLI** coding agent. It activates only when `payload["model"]`
equals its configured trigger — `"auto_codex"` by default — so it coexists with `agentic_routing`, which answers its
own trigger value.

Like its sibling it rewrites `payload["model"]` and adds `payload["agent_mode"]` plus a `payload["routing"]` block, and
nothing else: `instructions`, `input`, `tools`, `reasoning`, `text` and `stream` are forwarded unchanged. Routing
**fails open** — a payload that is not a dict, does not carry the trigger, resolves to an unconfigured mode or to a
mode with an empty `model_name`, or raises anywhere during classification, comes back as the very same object that was
passed in.

**1. Codex work modes:**

Modes ship in [agentic_routing_codex.json](llm_router_plugins/resources/routing/agentic_routing_codex.json) with a
`name`, `model_name`, `description`, `examples` and Polish + English heuristic signals (`keywords`, `phrases`,
`patterns`):

| Mode         | Model                     | Routed by                                      |
| ------------ | ------------------------- | ---------------------------------------------- |
| `plan`       | `qwen/Qwen3.8-Flash-Next` | `<collaboration_mode>` Plan Mode block         |
| `implement`  | `qwen/Qwen3.8-Flash-Next` | Fallback for a plain main turn                 |
| `test`       | `qwen/Qwen3.8-27B`        | Keywords in the latest user message            |
| `git_review` | `qwen/Qwen3.8-27B`        | Keywords in the latest user message            |
| `review`     | `qwen/Qwen3.8-Flash-Next` | Keywords in the latest user message            |
| `debug`      | `qwen/Qwen3.8-Flash-Next` | Keywords in the latest user message            |
| `aux_title`  | `qwen/Qwen3.8-27B`        | Request class — system-thread title generation |
| `compaction` | `qwen/Qwen3.8-27B`        | Request class — context compaction             |

**2. Request classes:**

Codex mixes three kinds of request on one endpoint, and the structural ones are recognised before any prompt text is
read. `client_metadata` carries the ids (`session_id`, `thread_id`, `turn_id`, `root_turn_id`) together with
`x-codex-turn-metadata`, which is itself a **JSON string** holding `request_kind` and `thread_source`:

| Class        | Condition                                                    |
| ------------ | ------------------------------------------------------------ |
| `compaction` | `request_kind == "compaction"`                               |
| `aux_title`  | `request_kind == "turn"` **and** `thread_source == "system"` |
| `main`       | every other request                                          |

**3. Resolution cascade — the first layer that can answer wins:**

| # | Layer                                                        | `source`             | Similarity                         |
| - | ------------------------------------------------------------ | -------------------- | ---------------------------------- |
| 1 | Explicit `agent_mode`, `codex_mode` or `metadata.agent_mode` | `explicit`           | `1.0`                              |
| 2 | Request class — `compaction`, then `aux_title`               | `class`              | `1.0`                              |
| 3 | `<collaboration_mode>` Plan block declared by the CLI        | `collaboration_mode` | `1.0`                              |
| 4 | PL+EN keyword scoring of the latest message                  | `heuristic`          | cosine, else `score / (score + 1)` |
| 5 | Embedding cosine similarity over the mode examples           | `semantic`           | cosine of the matched mode         |
| 6 | Configured `fallback_mode` (`implement`)                     | `fallback`           | cosine of that mode, else `0.0`    |

Only `test`, `git_review`, `review` and `debug` compete in the keyword layer: `plan` is declared by the CLI and
`implement` is the fallback, so neither needs keywords. A keyword hit needs `heuristic_min_score` (default `3.0`) or the
layer stays silent. Keywords and phrases match at a word start, so inflected forms still match (`testów`) while mid-word hits do
not: `protest` never scores the `test` keyword. The `<collaboration_mode>` block is re-read on every request and the **last** occurrence wins, so a
Plan → Default switch mid-session reclassifies the next turn correctly. Keyword scoring is Polish + English because real
prompts are short strings such as `"napraw testy"` or `"Przejrzyj ten katalog i zaproponuj poprawki"`.

**4. Similarity is embedding cosine similarity, exactly like `agentic_routing`:**

`routing.similarity` reports **embedding cosine similarity** produced by the same BiEncoder + FAISS stack
(`build_embedding_router` in `utils/routing/common.py`, `google/embeddinggemma-300m` by default). The six work modes
are indexed once at startup — `aux_title` and `compaction` are excluded because the request class already decides
them — and a request issues **at most one** `route(text)` lookup, whose result is reused by the heuristic, semantic
and fallback layers so the text is never embedded twice. Requests longer than the model's `max_seq_length` are
encoded with the same sliding-window aggregation as the semantic router (at most 4 windows from the head of the
text) before the lookup. The layer is fail-open: without
`sentence-transformers` / `faiss`, with an unloadable model, or with `SEMANTIC_ENABLED=false` it steps aside, the
cascade stays purely deterministic and the plugin remains constructible.

```python
from llm_router_plugins.utils.routing.agentic_routing.codex import CodexRoutingPlugin

plugin = CodexRoutingPlugin(logger)

payload = {
    "model": "auto_codex",
    "client_metadata": {
        "thread_id": "thread-1",
        "turn_id": "turn-7",
        "x-codex-turn-metadata": '{"request_kind":"turn"}',
    },
    "input": [
        {
            "type": "message",
            "role": "developer",
            "content": [{
                "type": "input_text",
                "text": "<collaboration_mode># Plan Mode (Conversational)\n"
                        "Start with `apply_patch`.</collaboration_mode>",
            }],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Zaplanuj migrację bazy danych."}],
        },
    ],
}

# after apply(): the Plan Mode block declared by the CLI decides — no ML involved
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
        "turn_id": "turn-7",
    },
}
```

The same payload with a `# Collaboration Mode: Default` block and `"napraw testy w tests/"` resolves to `test` /
`qwen/Qwen3.8-27B` with `source: "heuristic"`; a title-generation request on a system thread resolves to `aux_title` /
`qwen/Qwen3.8-27B` with `source: "class"`.

**Environment variable overrides** (prefix `LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_`):

| Env variable                                                     | Purpose                                            |
| ---------------------------------------------------------------- | -------------------------------------------------- |
| `LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_CONFIG`               | Path to a custom JSON config, or a raw JSON string |
| `LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_TRIGGER`              | Trigger model value (default `auto_codex`)         |
| `LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_MODEL`                | Override the **embedding** model name              |
| `LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_MODEL_<MODE>`         | Model for a single mode, e.g. `MODEL_PLAN=…`       |
| `LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_MODELS`               | Per-mode models, e.g. `plan=model_a\|test=model_b` |
| `LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_MODES`                | Whitelist of mode names to keep                    |
| `LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_FALLBACK_MODE`        | Mode used when nothing matches                     |
| `LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_HEURISTIC_ENABLED`    | `true`/`false` — toggle the keyword layer          |
| `LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_HEURISTIC_MIN_SCORE`  | Minimum keyword score to accept a match (3.0)      |
| `LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_MODE_<name>_KEYWORDS` | Pipe-separated keyword override for one mode       |
| `LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_SEMANTIC_ENABLED`     | `true`/`false` — toggle the embedding layer        |
| `LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_SIMILARITY_THRESHOLD` | Minimum cosine similarity for a semantic hit       |
| `LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_TOP_K`                | Chunks retrieved per query                         |
| `LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_CHUNK_SIZE`           | Token chunk size used when indexing modes          |
| `LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_CHUNK_OVERLAP`        | Token overlap between chunks                       |
| `LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_PERSIST_DIR`          | Directory for FAISS index persistence              |

`MODES` filters the mode list but does not move the fallback: if the whitelist excludes the configured `fallback_mode`
(`implement`), set `FALLBACK_MODE` as well — otherwise configuration validation fails at startup. The full reference
lives in [routing/README.md §3.12](llm_router_plugins/utils/routing/README.md).

---

## 3. Pipelines

Both masker and guardrail pipelines share the same design pattern:

| Class                                                     | Purpose                                                                            |
|-----------------------------------------------------------|------------------------------------------------------------------------------------|
| **MaskerPipeline** (`pipeline.py` – masker version)       | Executes a list of masker plugins in order, transforming the payload step‑by‑step. |
| **GuardrailPipeline** (`pipeline.py` – guardrail version) | Executes guardrail plugins sequentially, stopping on the first failure.            |

### 3.1 Registration

* Plugins are registered lazily via `MaskerRegistry.register(name, logger)` or
  `GuardrailRegistry.register(name, logger)`.
* The registry maps a string identifier (e.g. `"fast_masker"`) to a concrete plugin class, allowing pipelines to resolve
  the classes at runtime.

### 3.2 Configuration

All plugin identifiers are stored in environment variables or constants such as:

```python
MASKING_STRATEGY_PIPELINE = ["fast_masker"]
GUARDRAIL_STRATEGY_PIPELINE_REQUEST = ["nask_guard", "sojka_guard"]
```

These lists are consumed by the endpoint initialization (`EndpointI._prepare_masker_pipeline`,
`EndpointI._prepare_guardrails_pipeline`).

---

## 4. Adding a New Plugin

1. **Create a subclass** of either `PluginInterface` (for maskers) or `HttpPluginInterface` / a custom guardrail base.
2. **Define a `name` class attribute** – this is the identifier used in pipeline configuration.
3. **Implement `apply(self, payload: Dict) -> Dict`** (masker) **or `apply(self, payload: Dict) -> Tuple[bool, Dict]`
   ** (guardrail).
4. **Register the plugin** – either automatically via the registry’s `register` call in the pipeline constructor, or
   manually by calling `MaskerRegistry.register(name=MyPlugin.name, logger=logger)`.

*Example stub for a new masker:*

```python
# my_custom_masker.py
from llm_router_plugins.maskers.plugin_interface import PluginInterface
import logging
from typing import Dict, Optional


class MyCustomMasker(PluginInterface):
    name = "my_custom_masker"

    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger=logger)
        # Load any heavy resources here (e.g., a spaCy model)

    def apply(self, payload: Dict) -> Dict:
        # Perform your masking logic and return the modified payload
        return payload
```

After placing the file in `llm_router_plugins/maskers/plugins/`, enable it by adding `"my_custom_masker"` to
`MASKING_STRATEGY_PIPELINE`.

---

## 5. Retrieval‑Augmented Generation (RAG) Support

The project now includes a **LangChain‑based RAG plugin** that enables semantic search over user‑provided documents. The
implementation lives in `llm_router_plugins/utils/rag/langchain_plugin.py` and is driven by the helper CLI scripts
located in `scripts/`.

### 5.1 What the plugin does

| Feature           | Description                                                                                                                                                                                                                             |
|-------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Indexing**      | Reads a directory of text‑like files (`.txt`, `.md`, `.html`, `.js`, …), splits them into token‑based windows, embeds each chunk with a configurable transformer model, and stores the vectors in a FAISS (or compatible) vector store. |
| **Searching**     | Given a user query, retrieves the most similar chunks and injects them into the payload (e.g., appends to the last user message) so that downstream LLM calls can use the retrieved context.                                            |
| **Configuration** | All parameters (collection name, embedder model, device, chunk size, overlap, persistence directory) are driven by environment variables prefixed with `LLM_ROUTER_`. See the table below for the full list.                            |
| **CLI helpers**   | Two ready‑to‑use scripts: `scripts/llm-router-rag-langchain-index.sh` (indexes a repository) and `scripts/llm-router-rag-langchain-search.sh` (runs a search or starts an interactive REPL).                                            |

### 5.2 Environment variables

| Variable                                 | Default                                                                        | Meaning                                                     |
|------------------------------------------|--------------------------------------------------------------------------------|-------------------------------------------------------------|
| `LLM_ROUTER_LANGCHAIN_RAG_COLLECTION`    | *must be set*                                                                  | Name of the FAISS collection (e.g. `sample_collection`).    |
| `LLM_ROUTER_LANGCHAIN_RAG_EMBEDDER`      | `/mnt/data2/llms/models/community/google/embeddinggemma-300m`                  | Path or Hugging‑Face identifier of the embedding model.     |
| `LLM_ROUTER_LANGCHAIN_RAG_DEVICE`        | `cuda:2`                                                                       | Torch device (`cpu`, `cuda:0`, …).                          |
| `LLM_ROUTER_LANGCHAIN_RAG_CHUNK_SIZE`    | `1024`                                                                         | Number of tokens per chunk.                                 |
| `LLM_ROUTER_LANGCHAIN_RAG_CHUNK_OVERLAP` | `100`                                                                          | Number of overlapping tokens between consecutive chunks.    |
| `LLM_ROUTER_LANGCHAIN_RAG_PERSIST_DIR`   | `./workdir/plugins/utils/rag/langchain/${LLM_ROUTER_LANGCHAIN_RAG_COLLECTION}` | Directory where the FAISS index and docstore are persisted. |

#### Example export block (add to your shell profile or a `.env` file)

```shell script
export LLM_ROUTER_LANGCHAIN_RAG_COLLECTION="${LLM_ROUTER_LANGCHAIN_RAG_COLLECTION:-sample_collection}"
export LLM_ROUTER_LANGCHAIN_RAG_EMBEDDER="${LLM_ROUTER_LANGCHAIN_RAG_EMBEDDER:-/mnt/data2/llms/models/community/google/embeddinggemma-300m}"
export LLM_ROUTER_LANGCHAIN_RAG_DEVICE="${LLM_ROUTER_LANGCHAIN_RAG_DEVICE:-cuda:2}"
export LLM_ROUTER_LANGCHAIN_RAG_CHUNK_SIZE="${LLM_ROUTER_LANGCHAIN_RAG_CHUNK_SIZE:-1024}"
export LLM_ROUTER_LANGCHAIN_RAG_CHUNK_OVERLAP="${LLM_ROUTER_LANGCHAIN_RAG_CHUNK_OVERLAP:-100}"
export LLM_ROUTER_LANGCHAIN_RAG_PERSIST_DIR="${LLM_ROUTER_LANGCHAIN_RAG_PERSIST_DIR:-./workdir/plugins/utils/rag/langchain/${LLM_ROUTER_LANGCHAIN_RAG_COLLECTION}}"
```

### 5.3 Using the CLI scripts

**Index a repository** (example for the documentation site):

```shell script
scripts/llm-router-rag-langchain-index.sh
# Internally runs:
# llm-router-rag-langchain index --path "../.github/pages/llmrouter.cloud/" --ext .html .js .md
```

**Search** (interactive REPL):

```shell script
scripts/llm-router-rag-langchain-search.sh
# Internally runs:
# llm-router-rag-langchain search
# (you will be prompted for a query, type “exit” to quit)
```

**One‑shot search**:

```shell script
llm-router-rag-langchain search --query "What is Retrieval‑Augmented Generation?" --top_n 5
```

The CLI returns the raw matching chunks together with similarity scores. The `LangchainRAGPlugin` automatically formats
the retrieved text and appends it to the user’s last message, prefixed with:

```
If the context below will help answer the above question, use it.
Context separated with double enter
```
