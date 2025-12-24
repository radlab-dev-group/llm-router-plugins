#!/usr/bin/env python3
"""
CLI wrapper around the LangChainRAG helper.

Usage examples
--------------
# Index all *.txt and *.md files under ./data
python -m llm_router_plugins.utils.rag.engine.langchain_cli --index --path ./data --ext .txt .md

# Search the previously built index
python -m llm_router_plugins.utils.rag.engine.langchain_cli --search --query "What is LangChain?" --top_n 5
"""

import sys
import argparse

from pathlib import Path

from llm_router_plugins.core.utils import read_files_from_dir
from llm_router_plugins.utils.rag.engine.langchain import USE_LANGCHAIN_RAG

if USE_LANGCHAIN_RAG:
    try:
        from llm_router_plugins.utils.rag.engine.langchain import (
            LangChainRAG,
            LANGCHAIN_RAG_COLLECTION,
            LANGCHAIN_RAG_EMBEDDER,
            LANGCHAIN_RAG_DEVICE,
            LANGCHAIN_RAG_CHUNK_SIZE,
            LANGCHAIN_RAG_CHUNK_OVERLAP,
            USE_LANGCHAIN_RAG,
        )
        from langchain_community.vectorstores import FAISS
    except Exception as exc:  # pragma: no cover
        sys.stderr.write(f"Failed to import LangChainRAG utilities: {exc}\n")
        sys.exit(1)

    from llm_router_plugins.utils.rag.langchain_plugin import LangchainRAGPlugin


def _ensure_rag_enabled() -> None:
    """Exit early if the environment disables the LangChain RAG feature."""
    if not USE_LANGCHAIN_RAG:
        sys.stderr.write(
            "LangChain RAG is disabled (environment variables are incomplete).\n"
        )
        sys.exit(1)


def cmd_index(args: argparse.Namespace) -> None:
    """Handle the ``--index`` command."""
    base_path = Path(args.path).expanduser().resolve()
    if not base_path.is_dir():
        sys.stderr.write(f"Provided path is not a directory: {base_path}\n")
        sys.exit(1)

    extensions = [ext if ext.startswith(".") else f".{ext}" for ext in args.ext]
    texts = read_files_from_dir(base_path, extensions)

    print(len(texts))

    #
    # plugin = LangchainRAGPlugin()
    # success, response = plugin.rag.index_texts(texts)
    #
    # if not success:
    #     sys.stderr.write(f"Indexing failed: {response.get('error')}\n")
    #     sys.exit(1)
    #
    # print(response.get("result"))


def cmd_search(args: argparse.Namespace) -> None:
    """Handle the ``--search`` command."""

    # ------------------------------------------------------------------- #
    # Delegate searching to the plugin.  The plugin validates that an index
    # exists, runs the similarity search, and returns plain‑serialisable dicts.
    # ------------------------------------------------------------------- #
    plugin = LangchainRAGPlugin()
    payload = {
        "action": "search",
        "query": args.query,
        "top_n": args.top_n,
    }
    success, response = plugin.apply(payload)
    if not success:
        sys.stderr.write(f"Search failed: {response.get('error')}\n")
        sys.exit(1)

    results = response.get("result", [])
    if not results:
        print("No results found.")
        return

    for i, doc in enumerate(results, start=1):
        print(f"--- Result {i} ---")
        print(f"Content : {doc['content']}")
        print(f"Metadata: {doc['metadata']}")
        print()


def build_parser() -> argparse.ArgumentParser:
    """Create the top‑level argument parser."""
    parser = argparse.ArgumentParser(
        description="CLI for LangChain‑based Retrieval‑Augmented Generation."
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Command: [index|search]"
    )

    # ------------------------------ index ------------------------------ #
    idx_parser = subparsers.add_parser(
        "index", help="Index a directory of text files."
    )
    idx_parser.add_argument(
        "--path",
        required=True,
        help="Root directory containing files to index.",
    )
    idx_parser.add_argument(
        "--ext",
        nargs="+",
        required=True,
        help="File extensions to include (e.g. .txt .md).",
    )
    idx_parser.set_defaults(func=cmd_index)

    # ------------------------------ search ------------------------------ #
    srch_parser = subparsers.add_parser(
        "search", help="Search the previously built index."
    )
    srch_parser.add_argument(
        "--query",
        required=True,
        help="Search query string.",
    )
    srch_parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="Number of most similar chunks to return (default: 10).",
    )
    srch_parser.set_defaults(func=cmd_search)

    return parser


def main() -> None:
    _ensure_rag_enabled()

    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
