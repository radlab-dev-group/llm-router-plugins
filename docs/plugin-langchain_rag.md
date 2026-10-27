# LangChain RAG plugin guide

`langchain_rag` attaches a local knowledge base to the router: it indexes a
directory with LangChain, stores the vectors in FAISS, and injects the
retrieved chunks into every request.

## What it does

`LangchainRAGPlugin` recursively reads a directory of documents, splits them
into overlapping token windows, embeds the chunks with a transformer model, and
stores them in a persistent FAISS index. On a request it retrieves the top-N
most similar chunks and appends them, with a clear prefix, to the user's last
message so the model can use them.

If the RAG subsystem is disabled or misconfigured, the plugin raises a clear
exception while the rest of the router keeps working.

## CLI

`llm-router-rag-langchain` provides `index` and `search` subcommands and an
interactive REPL, so you can build and probe an index without a live router.

## Configuration

Environment variables under `LLM_ROUTER_`:

- `LLM_ROUTER_LANGCHAIN_RAG_COLLECTION` — the collection name,
- `LLM_ROUTER_LANGCHAIN_RAG_EMBEDDER` — the embedder model,
- `LLM_ROUTER_LANGCHAIN_RAG_DEVICE` — `cpu` or a CUDA device,
- `LLM_ROUTER_LANGCHAIN_RAG_CHUNK_SIZE` — tokens per chunk (default 400),
- `LLM_ROUTER_LANGCHAIN_RAG_CHUNK_OVERLAP` — overlap (default 100),
- `LLM_ROUTER_LANGCHAIN_RAG_PERSIST_DIR` — where the FAISS index is stored.

The optional RAG dependencies (`faiss-cpu`, `tqdm`, `langchain`,
`langchain-community`, `transformers`, `torch`) are not installed with the core
package. See `llm_router_plugins/utils/rag/LANGCHAIN_RAG.md`.

## When to use it

Use `langchain_rag` when you want to ground the model in a local corpus with
no external knowledge service. It is the standalone RAG path; the
`agentic_knowledge` layer is the agentic, repository-aware one.
