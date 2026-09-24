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

Three routing plugins are available for model selection. The two semantic plugins
(`simple_semantic_routing`, `semantic_biencoder_routing`) activate when `payload["model"] == "auto"`; the Codex plugin
(`agentic_routing_codex`) activates on `"auto_codex"`, routing the OpenAI-Responses-style requests emitted by the Codex
CLI agent by request class and work mode.

### 2.7.1 Simple Semantic Routing (Heuristic)

The **Simple Semantic Routing plugin** (`simple_semantic_routing`, `utils/routing/simple_semantic/`) activates on
`payload["model"] == "auto"` and rewrites it in three dependency-free stages: **intent** scoring over the last user
message — weighted keywords (case-insensitive substrings), phrases with a `:weight` suffix and regex patterns (+3.0
each) — giving `code` / `math` / `creative` / `general`, or `none` when nothing scores; **complexity** from a word-count
estimate (`int(words * 1.25)`) against the `simple` / `medium` thresholds; and **model selection** from an ordered pool
where index 0 is the cheapest and index `n-1` the strongest. `intent_adjustment` only ever escalates the intents that
declare one; an intent with an empty adjustment is demoted a single pool step. Only `payload["model"]` is written, so
the decision lives in the log:
`SimpleSemanticRouting: intent=code, complexity=simple (10 tokens) -> qwen3.6:35b`.

Configuration ships in
[simple_semantic.json](llm_router_plugins/resources/routing/simple_semantic.json). Overrides:
`LLM_ROUTER_ROUTING_COMPLEXITY_THRESHOLDS` (`simple|medium`), `LLM_ROUTER_ROUTING_MODELS` (the pool),
`LLM_ROUTER_ROUTING_DEFAULT_MODEL` and `LLM_ROUTER_ROUTING_INTENT_<name>` — that last one **replaces** an intent and
clears its patterns and weights, and there is no `..._CONFIG` variable for this plugin.

**Full documentation** — stage-by-stage mechanics, measured intent × complexity → model matrices for two- and
three-model pools, configuration reference, tuning, gotchas and a runnable verification snippet:
[Simple Semantic Routing README](llm_router_plugins/utils/routing/simple_semantic/README.md).

### 2.7.2 Bi-Encoder Semantic Routing (Model Selection)

The **Bi-Encoder routing plugin** (`semantic_biencoder_routing`, `utils/routing/semantic_biencoder/`) also answers
`payload["model"] == "auto"`, but decides by **embedding similarity**. Each routing target is indexed at startup as
`"Target: {name}. {description}"` plus its `examples`, split into overlapping token chunks, embedded by a
sentence-transformers BiEncoder (`google/embeddinggemma-300m` by default), L2-normalized and stored in a
`faiss.IndexFlatIP`; a docstore maps doc IDs back to target names. At request time the last user message is embedded
(text longer than the model's `max_seq_length` is windowed and averaged, capped at `MAX_QUERY_WINDOWS = 4`), the
`top_k` hits are **averaged per target**, and the winning target's `model_name` is selected when its cosine reaches
`similarity_threshold`; the payload gains `routing.target_name` and `routing.similarity`. The index is persisted as
`index.faiss` + `docstore.pkl` and reloaded on the next start, rebuilt automatically when the embedding model outputs a
different dimension. Needs the `[ml]` / `[ml-gpu]` extras — and it fails hard rather than passing traffic through.

Configuration ships in [semantic_biencoder.json](llm_router_plugins/resources/routing/semantic_biencoder.json).
Overrides: `LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_CONFIG` (inline JSON *or* a path), `_MODEL`, `_TARGETS`,
`_CHUNK_SIZE`, `_CHUNK_OVERLAP`, `_PERSIST_DIR`; `similarity_threshold` and `top_k` are config-only. The shipped
`similarity_threshold: 0.0` reroutes **every** request — measured on the shipped targets, real matches score 0.70–0.82
while unrelated or nonsense text still scores ≈0.35–0.37, so `0.5`–`0.6` is the usual production band.

**Full documentation** — startup and query mechanics, measured similarities per target, configuration reference, failure
modes, tuning, gotchas (persisted-index staleness included) and a verification recipe:
[Bi-Encoder Routing README](llm_router_plugins/utils/routing/semantic_biencoder/README.md).

### 2.7.3 Codex CLI routing (`auto_codex`)

The **Codex Routing plugin** (`agentic_routing_codex`, `utils/routing/agentic_routing/codex/`) routes the
OpenAI-Responses-style requests emitted by the **Codex CLI** coding agent. It activates only when `payload["model"]`
equals its configured trigger — `"auto_codex"` by default — so it never intercepts the `"auto"` traffic owned by the
two semantic plugins. It rewrites `payload["model"]` and adds `payload["agent_mode"]` plus a `payload["routing"]`
block, and nothing else; routing **fails open**, so a payload it cannot classify is passed through untouched.

A work mode is resolved by a deterministic cascade — an explicit `agent_mode` override, the request class
(`compaction`, `aux_title`), the `<collaboration_mode>` Plan block injected by the CLI, PL+EN keyword scoring, then an
optional embedding cosine-similarity lookup — and falls back to `implement`:

| Mode         | Model                     | Routed by                                      |
| ------------ | ------------------------- | ---------------------------------------------- |
| `plan`       | `qwen/Qwen3.8-Flash-Next` | `<collaboration_mode>` Plan Mode block         |
| `implement`  | `qwen/Qwen3.8-Flash-Next` | Fallback for a plain main turn                 |
| `test`       | `qwen/Qwen3.8-27B`        | Keywords in the classified user text           |
| `git_review` | `qwen/Qwen3.8-27B`        | Keywords in the classified user text           |
| `review`     | `qwen/Qwen3.8-Flash-Next` | Keywords in the classified user text           |
| `debug`      | `qwen/Qwen3.8-Flash-Next` | Keywords in the classified user text           |
| `aux_title`  | `qwen/Qwen3.8-27B`        | Request class — system-thread title generation |
| `compaction` | `qwen/Qwen3.8-27B`        | Request class — context compaction             |

Modes ship in
[agentic_routing_codex.json](llm_router_plugins/resources/routing/agentic_routing_codex.json) and are overridable with
`LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_*` environment variables (`…_CONFIG`, `…_TRIGGER`, `…_MODEL_<MODE>`,
`…_MODELS`, `…_MODES`, `…_FALLBACK_MODE`, `…_HEURISTIC_ENABLED`, `…_HEURISTIC_MIN_SCORE`, `…_CLASSIFY_MAX_CHARS`,
`…_MODEL`, `…_SEMANTIC_ENABLED`, `…_SIMILARITY_THRESHOLD`, `…_TOP_K`, `…_CHUNK_SIZE`, `…_CHUNK_OVERLAP`,
`…_PERSIST_DIR`, `…_MODE_<name>_KEYWORDS`). `MODES` filters the mode list but does not move the fallback: if the
whitelist excludes the configured `fallback_mode` (`implement`), set `FALLBACK_MODE` as well — otherwise configuration
validation fails at startup.

**Full documentation** — the cascade and scoring model, installation, loading the plugin into the router, model
selection and its pitfalls, tuning, a verification recipe and a troubleshooting table — lives in the
[Codex CLI Routing README](llm_router_plugins/utils/routing/agentic_routing/codex/README.md).

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
