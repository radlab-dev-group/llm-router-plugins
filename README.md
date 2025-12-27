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
| **(Implicit) GuardrailProcessor** (`processor.py`) | Core logic used by the internal NASK guardrail Flask route (`nask_guardrail`). Tokenises the payload, creates overlapping chunks, runs a Hugging‑Face `text‑classification` pipeline, and produces a detailed safety report. | Handles model loading (`AutoTokenizer`, `pipeline("text‑classification")`), chunking (`_chunk_text`), and scoring thresholds (`MIN_SCORE_FOR_SAFE`, `MIN_SCORE_FOR_NOT_SAFE`). Returns a dict: `{"safe": <bool>, "detailed": [...]}`. |

### 2.3 How a guardrail is used

1. The endpoint calls `_is_request_guardrail_safe(payload)` (or the analogous response guardrail).
2. If `FORCE_GUARDRAIL_REQUEST` is true, a `GuardrailPipeline` is built from the configured plugin IDs (e.g.
   `["nask_guard"]`).
3. The pipeline iterates over each guardrail plugin; each `apply` returns `(is_safe, message)`.
4. The first plugin that reports `is_safe=False` short‑circuits the pipeline and the request is rejected with a 400/500
   error payload.

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
GUARDRAIL_STRATEGY_PIPELINE_REQUEST = ["nask_guard"]
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

*Index a repository* (example for the documentation site):

```shell script
scripts/llm-router-rag-langchain-index.sh
# Internally runs:
# llm-router-rag-langchain index --path "../.github/pages/llmrouter.cloud/" --ext .html .js .md
```

*Search* (interactive REPL):

```shell script
scripts/llm-router-rag-langchain-search.sh
# Internally runs:
# llm-router-rag-langchain search
# (you will be prompted for a query, type “exit” to quit)
```

*One‑shot search*:

```shell script
llm-router-rag-langchain search --query "What is Retrieval‑Augmented Generation?" --top_n 5
```

The CLI returns the raw matching chunks together with similarity scores. The `LangchainRAGPlugin` automatically formats
the retrieved text and appends it to the user’s last message, prefixed with:

```
If the context below will help answer the above question, use it.
Context separated with double enter:
```

### 5.4 Enabling the RAG plugin in LLM‑Router

1. Add the plugin name to the utils pipeline configuration (usually via an environment variable):

```shell script
export LLM_ROUTER_UTILS_PLUGINS_PIPELINE="${LLM_ROUTER_UTILS_PLUGINS_PIPELINE:-langchain_rag}"
```

2. Ensure the persistence directory exists (the first indexing run will create it).
3. Start the router as usual; the plugin will load the FAISS index on startup and enrich every incoming request with
   relevant context.

*No code changes are required* – the plugin is discovered automatically by the utils‑registry mechanism.

---

## 6. Summary

* **Anonymizers** (`FastMaskerPlugin`, `BANonymizer`) scrub PII from requests.
* **Guardrails** (`NASKGuardPlugin`, internal `GuardrailProcessor`) enforce safety policies.
* **Pipelines** (`MaskerPipeline`, `GuardrailPipeline`) orchestrate sequential execution, short‑circuiting on failure
  for guardrails.
* **RAG Plugin** (`langchain_rag`) adds semantic retrieval capabilities using LangChain, FAISS, and configurable
  embedding models.
* The system is **extensible**: new plugins are just classes that obey a tiny interface contract and can be referenced
  by name in the configuration.

These components together give the LLM‑Router a flexible, policy‑driven request‑processing stack that can be tailored to
any deployment scenario.

---

## 📜 License

See the [LICENSE](LICENSE) file.
