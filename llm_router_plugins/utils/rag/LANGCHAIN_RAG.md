# LangChain RAG Plugin for LLM‑Router

A lightweight **Retrieval‑Augmented Generation (RAG)** plugin that lets you attach any local knowledge base to the
LLM‑Router without touching the original application code. It builds the knowledge base with **LangChain**, stores
vectors in **FAISS** (fallback to Milvus is possible), and injects the retrieved context into every LLM request.

---

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Configuration (environment variables)](#configuration-environment-variables)
4. [CLI – Indexing & Searching](#cli--indexing--searching)
5. [Integrating the plugin into LLM‑Router](#integrating-the-plugin-into-llmrouter)
6. [Example workflow](#example-workflow)
7. [Error handling](#error-handling)
8. [License & Credits](#license--credits)

---

## Features

| Feature                  | Description                                                                                                                                                                                                                |
|--------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Indexing**             | Recursively reads a directory of `.md`, `.txt`, `.html` (or any extensions you pass) → splits documents into overlapping token windows → embeds chunks with a transformer model → stores them in a persistent FAISS index. |
| **Searching**            | Given a user query, retrieves the top‑N most similar chunks (default 10) using cosine similarity.                                                                                                                          |
| **Context injection**    | Retrieved chunks are appended (with a clear prefix) to the user’s last message, so the downstream LLM can use the extra knowledge automatically.                                                                           |
| **Full configurability** | All hyper‑parameters (collection name, embedder model, device, chunk size, overlap, persistence directory) are set via environment variables.                                                                              |
| **Fail‑safe**            | If the RAG subsystem is disabled or mis‑configured, the plugin raises a clear exception while the rest of the router keeps working.                                                                                        |
| **CLI helpers**          | `llm-router-rag-langchain` command provides convenient `index` and `search` sub‑commands, plus an interactive REPL mode.                                                                                                   |

---

## Installation

Only the optional RAG dependencies are required; they are **not** installed with the core LLM‑Router.

```shell script
# Activate your virtualenv / conda env first
pip install faiss-cpu tqdm langchain langchain-community transformers torch
```

> **Note** – If you prefer GPU‑accelerated FAISS, replace `faiss-cpu` with `faiss-gpu`.

---

## Configuration (environment variables)

| Variable                                 | Default       | Description                                                                                                                                                            |
|------------------------------------------|---------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `LLM_ROUTER_LANGCHAIN_RAG_COLLECTION`    | *must be set* | Name of the FAISS collection (e.g. `sample_collection`).                                                                                                               |
| `LLM_ROUTER_LANGCHAIN_RAG_EMBEDDER`      | *must be set* | Hugging Face model identifier for the embedder (e.g. `sentence-transformers/all-MiniLM-L6-v2`).                                                                        |
| `LLM_ROUTER_LANGCHAIN_RAG_DEVICE`        | `cpu`         | Torch device (`cpu`, `cuda:0`, `cuda:1`, …).                                                                                                                           |
| `LLM_ROUTER_LANGCHAIN_RAG_CHUNK_SIZE`    | `1024`        | Number of tokens per chunk.                                                                                                                                            |
| `LLM_ROUTER_LANGCHAIN_RAG_CHUNK_OVERLAP` | `100`         | Overlap in tokens between consecutive chunks.                                                                                                                          |
| `LLM_ROUTER_LANGCHAIN_RAG_PERSIST_DIR`   | *none*        | Directory where the FAISS index and docstore are persisted. If omitted, a folder named after the collection is created under `./workdir/plugins/utils/rag/langchain/`. |

### Example export block (add to your shell profile or a `.env` file)

```shell script
export LLM_ROUTER_LANGCHAIN_RAG_COLLECTION="${LLM_ROUTER_LANGCHAIN_RAG_COLLECTION:-sample_collection}"
export LLM_ROUTER_LANGCHAIN_RAG_EMBEDDER="${LLM_ROUTER_LANGCHAIN_RAG_EMBEDDER:-sentence-transformers/all-MiniLM-L6-v2}"
export LLM_ROUTER_LANGCHAIN_RAG_DEVICE="${LLM_ROUTER_LANGCHAIN_RAG_DEVICE:-cpu}"
export LLM_ROUTER_LANGCHAIN_RAG_CHUNK_SIZE="${LLM_ROUTER_LANGCHAIN_RAG_CHUNK_SIZE:-200}"
export LLM_ROUTER_LANGCHAIN_RAG_CHUNK_OVERLAP="${LLM_ROUTER_LANGCHAIN_RAG_CHUNK_OVERLAP:-50}"
export LLM_ROUTER_LANGCHAIN_RAG_PERSIST_DIR="${LLM_ROUTER_LANGCHAIN_RAG_PERSIST_DIR:-./workdir/plugins/utils/rag/langchain/${LLM_ROUTER_LANGCHAIN_RAG_COLLECTION}}"
```

---

## CLI – Indexing & Searching

The plugin ships with a convenient entry‑point:

```shell script
llm-router-rag-langchain <subcommand> [options]
```

### Indexing

```shell script
# Index Markdown, plain‑text and HTML files from a repository
llm-router-rag-langchain index --path "../llm-router" --ext .md .txt .html
```

You can run the command for any number of directories (plugins, services, utils, web, etc.).  
A helper script `scripts/llm-router-rag-langchain-index.sh` already exports all environment variables and launches the
indexing process from the root of the LLM‑Router repository.

### Searching

```shell script
# One‑shot search
llm-router-rag-langchain --search --query "Explain RAG" --top_n 3

# Interactive REPL (no --query)
llm-router-rag-langchain --search
```

The CLI returns the raw retrieved chunks; the plugin automatically prefixes them with:

```
If the context below will help answer the above question, use it.
Context separated with double enter:
```

---

## Integrating the plugin into LLM‑Router

1. **Enable the plugin** – add it to the router’s pipeline (usually via an environment variable):

```shell script
export LLM_ROUTER_UTILS_PLUGINS_PIPELINE="${LLM_ROUTER_UTILS_PLUGINS_PIPELINE:-langchain_rag}"
```

2. **Make sure the persistence directory exists** (the CLI will create it on first index).
3. **Run the router** – the plugin will load the FAISS index on start and enrich every incoming request with relevant
   context.

No code changes are required in the application that already uses LLM‑Router.

---

## Example workflow

```shell script
# 1️⃣ Set up environment
source .env   # (or manually export the vars shown above)

# 2️⃣ Build the knowledge base
llm-router-rag-langchain index --path "./docs" --ext .md .txt

# 3️⃣ Start the router (example)
export LLM_ROUTER_UTILS_PLUGINS_PIPELINE="${LLM_ROUTER_UTILS_PLUGINS_PIPELINE:-langchain_rag}"
./run-rest-api-gunicorn.sh

# 4️⃣ Send a query (via your existing client)
curl -X POST http://localhost:8000/api/chat -d '{"model":"google/gemma3-12b-it", "messages":[{"role":"user","content":"What is Retrieval‑Augmented Generation?"}]}'
```

The router will automatically fetch the most relevant chunks from the FAISS store and prepend them to the user’s message
before forwarding it to the underlying LLM.

---

## Error handling

* **Missing/invalid environment** – The plugin raises a clear `RuntimeError` during construction, pointing to the
  missing variable.
* **Indexing/search failures** – Exceptions are caught, logged (if a logger is supplied), and the original payload is
  returned unchanged, so the rest of the pipeline continues to work.

---

## License & Credits

The plugin is released under the **Apache 2.0** license, the same as the rest of the LLM‑Router project.

* **LangChain** – https://github.com/hwchase17/langchain (MIT)
* **FAISS** – https://github.com/facebookresearch/faiss (MIT)

Please see the individual repositories for their licensing terms.

---

### Happy augmenting! 🚀

If you encounter any issues or have ideas for improvements, feel free to open an issue or submit a pull request.