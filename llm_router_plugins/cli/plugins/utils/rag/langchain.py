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

# -------------------------------------------------
# Simple ANSI colour helpers – no external deps needed
# -------------------------------------------------
RESET = "\033[0m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"

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

    plugin = LangchainRAGPlugin()
    plugin.rag.index_texts(texts)


def search_and_show_results(plugin, text_as_query, top_n: int = 10):
    docs_and_scores = plugin.rag.search(text_as_query, top_n=top_n)
    for i, [doc, score] in enumerate(docs_and_scores, start=1):
        print(f"{CYAN}--- [{score}] Result {i} ---{RESET}")
        print(
            f"{GREEN}Content :{RESET} {doc.page_content} (len={len(doc.page_content)})"
        )
        # print(f"{GREEN}Metadata:{RESET} {doc.metadata}\n")


def cmd_search(args: argparse.Namespace) -> None:
    """Handle the ``--search`` command."""
    plugin = LangchainRAGPlugin()

    # Interactive mode when no query is supplied
    if not args.query:
        print(
            f"{CYAN}Entering interactive search mode (type 'exit' to quit).{RESET}"
        )
        try:
            while True:
                query = input(f"{MAGENTA}>>> {RESET}").strip()
                if query.lower() in ("", "exit", "quit"):
                    print(f"{CYAN}Good‑bye!{RESET}")
                    break
                search_and_show_results(plugin, query, args.top_n)
        except KeyboardInterrupt:
            print(f"\n{CYAN}Interrupted – exiting interactive mode.{RESET}")
        return

    # Normal (non‑interactive) execution when a query is supplied
    search_and_show_results(plugin, args.query, args.top_n)


def build_parser() -> argparse.ArgumentParser:
    """Create the top‑level argument parser."""
    parser = argparse.ArgumentParser(
        description="CLI for LangChain‑based Retrieval‑Augmented Generation."
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Command: [index|search]"
    )

    # ---------------------------- index ---------------------------- #
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

    # ---------------------------- search ---------------------------- #
    srch_parser = subparsers.add_parser(
        "search", help="Search the previously built index."
    )
    srch_parser.add_argument(
        "--query",
        required=False,  # <-- made optional
        help="Search query string. If omitted, an interactive REPL starts.",
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
