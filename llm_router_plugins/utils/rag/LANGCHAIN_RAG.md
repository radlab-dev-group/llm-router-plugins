## LangChain RAG Plugin Overview

The **LangChain RAG** plugin (`langchain_rag`) provides a lightweight Retrieval‑Augmented Generation (RAG) service that
can be plugged into the LLM‑Router pipeline. It leverages:

* **LangChain** – for document handling, chunking, and vector‑store abstractions.
* **FAISS** – an efficient similarity‑search index (fallback to Milvus is possible).
* **Transformer embeddings** – any Hugging Face model that outputs sentence‑level embeddings.

### Core Features

| Feature               | Description                                                                                                                                                                 |
|-----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Indexing**          | Reads a directory of `.txt`/`.md` files, splits each document into overlapping token windows, embeds the chunks, and stores them in a persistent FAISS index.               |
| **Searching**         | Given a user query, retrieves the top‑N most similar chunks (default 10) using cosine similarity.                                                                           |
| **Context Injection** | The retrieved chunks are appended to the user’s last message (or a dedicated field) under a standard instruction prefix, so the downstream LLM can use the extra knowledge. |
| **Configurable**      | All hyper‑parameters (collection name, embedder model, device, chunk size, overlap, persistence directory) are set via environment variables.                               |
| **Fail‑safe**         | If the RAG subsystem is disabled or mis‑configured, the plugin raises a clear exception; the rest of the router continues to work.                                          |

### Environment Variables

| Variable                                 | Default              | Description                                                                                                     |
|------------------------------------------|----------------------|-----------------------------------------------------------------------------------------------------------------|
| `LLM_ROUTER_LANGCHAIN_RAG_COLLECTION`    | *None* (must be set) | Name of the FAISS collection.                                                                                   |
| `LLM_ROUTER_LANGCHAIN_RAG_EMBEDDER`      | *None* (must be set) | Hugging Face model identifier (e.g., `sentence-transformers/all-MiniLM-L6-v2`).                                 |
| `LLM_ROUTER_LANGCHAIN_RAG_DEVICE`        | `cpu`                | Torch device (`cpu`, `cuda:0`, …).                                                                              |
| `LLM_ROUTER_LANGCHAIN_RAG_CHUNK_SIZE`    | `200`                | Number of tokens per chunk.                                                                                     |
| `LLM_ROUTER_LANGCHAIN_RAG_CHUNK_OVERLAP` | `50`                 | Overlap in tokens between consecutive chunks.                                                                   |
| `LLM_ROUTER_LANGCHAIN_RAG_PERSIST_DIR`   | *None*               | Directory where the FAISS index and docstore are saved. If omitted, the collection name is used as folder name. |

### Payload Format

The response is the original payload enriched with a `messages[-1]["content"]` field (or the field containing the user
query) that now contains the retrieved context prefixed by:

```
If the context below will help answer the above question, use it.
Context separated with double enter:
```

### Integration Steps

1. **Install optional dependencies** (only needed if you enable RAG):

```shell script
pip install faiss-cpu tqdm langchain langchain-community transformers torch
```

2. **Set the environment variables** shown above.

3. **Add the plugin to the utils pipeline** (e.g. in your configuration file):

```python
UTILS_PIPELINE = ["langchain_rag"]
```

4. **Use the CLI** (optional) for quick indexing/searching:

```shell script
python -m llm_router_plugins.utils.rag.engine.langchain --index --path ./data --ext .txt .md
   python -m llm_router_plugins.utils.rag.engine.langchain --search --query "Explain RAG" --top_n 3
```

### Error Handling

* If the environment is incomplete, the plugin raises an exception during construction.
* Runtime errors during indexing or searching are caught, logged (if a logger is supplied), and the original payload is
  returned unchanged.

### License & Credits

The RAG helper code is released under the same **Apache 2.0** license as the rest of the project. The underlying
LangChain and FAISS libraries have their own licenses (see their respective repositories).  
