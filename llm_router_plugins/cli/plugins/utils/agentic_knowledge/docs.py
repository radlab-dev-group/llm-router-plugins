"""
Operator's command for the documentation half of the knowledge layer.

The proxy indexes a repository's documentation on the first request made in it,
which is the right default and the wrong tool: an operator wants to know what
would be indexed *before* a knowledge base fills with the wrong thing, wants to
index a repository no agent has opened yet, and wants a repository whose router
does not run on the same filesystem to have documentation at all.

This is that command. It does the same scan, against the same policy, under the
same configuration, and publishes to the same stream — so the indexing worker
masks and writes what it writes for a mined episode, and there is no second
write path to keep honest.

Nothing here talks to the search engine: the command reads files and publishes
events, optionally replacing queued documentation. A repository can be indexed
from a machine that cannot reach the engine, as long as it reaches the stream.
"""

import argparse
import json
import logging
import os
import sys

from typing import Any, List, Optional

from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
    AgenticKnowledgeConfig,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.docs_ingestor import (
    DocsIndexer,
    scan_repository,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.events import (
    KnowledgeEventPublisher,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.scope import (
    normalize_scope,
    scope_key,
)

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="llm-router-agentic-kb-docs",
        description=(
            "Publish a repository's documentation to the agentic knowledge "
            "stream, where the indexing worker masks and indexes it."
        ),
    )
    parser.add_argument(
        "--path",
        default=".",
        help="Repository to index; defaults to the current directory.",
    )
    parser.add_argument(
        "--scope",
        default=None,
        help=(
            "Knowledge scope to index into. Defaults to what the proxy would "
            "derive from a request made in this directory."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a knowledge policy JSON; defaults to the bundled policy.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be published and exit; reads no Redis, "
        "publishes nothing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace this scope's documentation events in Redis and republish "
        "all files, even unchanged ones, without automatic scan file/byte limits. "
        "Pause workers and other publishers first.",
    )
    parser.add_argument(
        "--no-prune",
        action="store_true",
        help="Publish new and changed files, keep the superseded documents.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the outcome as JSON instead of one line.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ...).",
    )
    return parser


def _load_config(args: argparse.Namespace, logger: logging.Logger) -> Any:
    """Read the policy the proxy uses, with the environment over it."""
    import pathlib

    if args.config:
        config = AgenticKnowledgeConfig.from_file(pathlib.Path(args.config))
    else:
        config = AgenticKnowledgeConfig.from_file()
    config.override_from_env(logger)
    config.validate()
    return config


def _publisher(config: Any, logger: logging.Logger) -> KnowledgeEventPublisher:
    """
    Build the stream publisher, refusing a deployment that has no stream.

    Parameters
    ----------
    config : Any
        The resolved configuration.
    logger : logging.Logger
        Where the refusal is explained before the process stops.

    Returns
    -------
    KnowledgeEventPublisher
        A publisher for the configured stream.

    Raises
    ------
    SystemExit
        When no Redis was configured: this command writes to the stream and
        nowhere else, so falling back to "index directly" would skip the
        masking the worker is responsible for.
    """
    if not config.redis.configured:
        problem = f" (discarded: {config.redis_error})" if config.redis_error else ""
        raise SystemExit(
            "Documentation is published to the knowledge stream, so this "
            "command needs Redis "
            f"(LLM_ROUTER_KNOWLEDGE_AGENTIC_REDIS_HOST is unset){problem}. "
            "Export the same Redis settings used by the worker; this CLI does "
            "not load .env files automatically. Previous indexing does not "
            "persist connection settings."
        )
    return KnowledgeEventPublisher(
        connection=config.redis,
        stream_key=config.stream_key,
        key_prefix=config.key_prefix,
        dedup_ttl_seconds=config.dedup_ttl_seconds,
        max_stream_len=config.max_stream_len,
        logger=logger,
    )


def _report_scan(config: Any, root: str, scope: str, *, force: bool = False) -> int:
    """Print what a scan would publish; return the number of documents."""
    report = scan_repository(
        root,
        patterns=config.docs_patterns,
        excluded_dirs=config.docs_excluded_dirs,
        max_files=None if force else config.docs_max_files,
        max_file_bytes=None if force else config.docs_max_file_bytes,
        max_total_bytes=None if force else config.docs_max_total_bytes,
        scope=scope,
    )
    for record in report.records:
        where = f"{record.source_file}"
        if record.heading:
            where = f"{where} [{record.heading}]"
        print(f"{record.doc_type:<13} {where}  {len(record.text)} chars")
    if report.skipped:
        print(
            f"{report.skipped} file(s)/directory(s) skipped (unreadable or too large)"
        )
        if force:
            print(
                "force would abort: unreadable files/directories; stream unchanged"
            )
    if report.truncated:
        print(
            "scan truncated by the configured ceilings: superseded documents "
            "will be kept, not retired"
        )
    print(
        f"{report.files} file(s), {len(report.records)} document(s)"
        " \u2014 the engine chunks each one"
    )
    return len(report.records)


def main(argv: Optional[List[str]] = None) -> int:
    """
    Publish one repository's documentation.

    Parameters
    ----------
    argv : List[str], optional
        Command-line arguments; ``sys.argv`` when omitted.

    Returns
    -------
    int
        Process exit code: ``0`` when the scan ran, ``1`` when it could not.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger = logging.getLogger("agentic-kb-docs")

    config = _load_config(args, logger)
    root = os.path.abspath(os.path.expanduser(str(args.path)))
    if not os.path.isdir(root):
        logger.error("%s is not a directory", root)
        return 1
    explicit = args.scope is not None and str(args.scope).strip() != ""
    scope = (
        normalize_scope(str(args.scope))
        if explicit
        else scope_key(
            root,
            strategy=config.scope_strategy,
            overrides=config.scope_overrides,
            default=config.scope_default,
        )
    )
    if not scope:
        logger.error(
            "no knowledge scope could be %s; pass --scope with characters a "
            "collection name accepts (latin letters, digits, dot, underscore, "
            "hyphen) or configure scope.default",
            "normalized from --scope" if explicit else "derived",
        )
        return 1
    if not config.docs_enabled:
        logger.error(
            "documentation is disabled by the policy (docs.enabled, "
            "LLM_ROUTER_KNOWLEDGE_AGENTIC_DOCS_ENABLED)"
        )
        return 1
    if not config.docs_patterns:
        logger.error("the policy names no documentation patterns (docs.patterns)")
        return 1

    if args.dry_run:
        print(f"scope={scope} root={root}")
        _report_scan(config, root, scope, force=args.force)
        return 0

    indexer = DocsIndexer(
        config=config,
        publisher=_publisher(config, logger),
        logger=logger,
    )
    outcome = indexer.index(
        scope,
        root,
        agent="cli",
        force=True,
        republish=args.force,
        allow_prune=not args.no_prune,
    )
    if args.json:
        print(json.dumps(outcome.as_dict(), sort_keys=True))
    else:
        print(
            f"scope={scope} status={outcome.status} files={outcome.files} "
            f"published={outcome.published} pruned={outcome.pruned}"
            f" unchanged={outcome.unchanged}"
            + (f" reason={outcome.reason}" if outcome.reason else "")
            + (" truncated=yes" if outcome.truncated else "")
        )
    if outcome.status == "unavailable":
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    sys.exit(main())
