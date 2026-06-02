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

The **Semantic Routing plugin** (`simple_semantic_routing`) performs two-stage heuristic model selection: it classifies
the user's intent
(code, math, creative, general) via weighted keywords, multi-word phrases, and regex patterns, then estimates input
complexity
(token count) to pick the most appropriate model from a configured pool. Configuration is entirely JSON-driven in
[simple.json](llm_router_plugins/resources/routing/semantic/simple.json); full details are in the
[plugin README](llm_router_plugins/utils/routing/semantic/README.md).

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
