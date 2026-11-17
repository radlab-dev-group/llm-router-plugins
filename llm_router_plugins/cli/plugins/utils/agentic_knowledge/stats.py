"""
Operator's reader for the knowledge layer's scoreboard.

The proxy counts every request it serves onto a per-repository, per-day hash
in Redis; this command reads those counters back and prints the rates they
are meant to answer with: how much of the knowledge the layer showed it, and
how often showing it was followed by a fix.

It reads, and nothing else — no search, no writes, no policy change.  A
deployment can run it against the same Redis the proxy writes, from a machine
that cannot reach the search engine at all.
"""

import argparse
import datetime
import json
import logging
import pathlib
import sys

from typing import Any, Dict, List, Optional, Sequence

from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
    AgenticKnowledgeConfig,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.metrics import (
    KnowledgeMetrics,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.scope import (
    normalize_scope,
    scope_key,
)

__all__ = ["build_parser", "main", "report"]

#: The rates a run is read by, in the order they are printed.
_RATE_ORDER = (
    "adoption_rate",
    "knowledge_help_rate",
    "failure_turn_share",
    "loop_turn_share",
    "escalation_rate",
)


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="llm-router-agentic-kb-stats",
        description=(
            "Read the agentic knowledge layer's scoreboard: what it injected "
            "per repository and how often that was followed by a fix."
        ),
    )
    parser.add_argument(
        "--scope",
        action="append",
        default=None,
        help=(
            "Repository scope to read; repeatable. Omit with --path to derive "
            "it the way the proxy would."
        ),
    )
    parser.add_argument(
        "--path",
        default=".",
        help="Repository to derive the scope from when --scope is not given.",
    )
    parser.add_argument(
        "--day",
        default="",
        help="Day to read as YYYYMMDD; defaults to today.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=1,
        help="How many consecutive days ending at --day to print (default 1).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON rather than a table, for a dashboard to consume.",
    )
    parser.add_argument(
        "--config",
        default="",
        help="Policy file; defaults to the plugin's bundled policy.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="Logging level for this command (default WARNING).",
    )
    return parser


def _days(end: str, count: int) -> List[str]:
    """The ``count`` days ending at *end* (YYYYMMDD), oldest first."""
    days = int(count) if int(count) > 0 else 1
    last = (
        datetime.datetime.strptime(end, "%Y%m%d") if end else datetime.datetime.now()
    )
    return [
        (last - datetime.timedelta(days=offset)).strftime("%Y%m%d")
        for offset in range(days - 1, -1, -1)
    ]


def report(
    metrics: KnowledgeMetrics, scopes: Sequence[str], days: Sequence[str]
) -> List[Dict[str, Any]]:
    """
    Read every scope's counters for every day asked.

    A scope with no counters is reported with zero counts and ``None`` rates:
    it was not a bad day, it was no measurement, and the two must not print
    the same way.
    """
    rows: List[Dict[str, Any]] = []
    for scope in scopes:
        for day in days:
            rows.append(metrics.derived(scope, day))
    return rows


def _format_rate(value: Optional[float]) -> str:
    return "  n/a " if value is None else f"{value:6.1%}"


def _print_text(rows: List[Dict[str, Any]]) -> None:
    print(
        f"{'scope':<28} {'day':<10} {'turns':>6} {'inj':>5} {'docs':>5} "
        f"{'loops':>6} {'esc':>4} {'fail':>5} {'memo':>5}  "
        + "  ".join(
            name.replace("_rate", "").replace("_share", "") for name in _RATE_ORDER
        )
    )
    for row in rows:
        print(
            f"{str(row.get('scope', ''))[:28]:<28} {row['day']:<10} "
            f"{row['turns']:>6} {row['injected_turns']:>5} "
            f"{row['injected_docs']:>5} {row['loops']:>6} "
            f"{row['escalations']:>4} {row['read_failures']:>5} "
            f"{row.get('memo_hits', 0):>5}  "
            + "  ".join(_format_rate(row.get(name)) for name in _RATE_ORDER)
        )


def main(
    argv: Optional[List[str]] = None, metrics: Optional[KnowledgeMetrics] = None
) -> int:
    """
    Print the scoreboard of one or more repositories.

    Parameters
    ----------
    argv : List[str], optional
        Command-line arguments; ``sys.argv`` when omitted.
    metrics : KnowledgeMetrics, optional
        The scoreboard to read.  Supplied by tests; built from the policy's
        Redis connection otherwise.

    Returns
    -------
    int
        ``0`` when a report was produced, ``1`` when no scope or no Redis made
        reading possible.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.WARNING),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger = logging.getLogger("agentic-kb-stats")

    config = _load_config(args, logger)
    scopes = _scopes(args, config, logger)
    if not scopes:
        return 1
    if metrics is None and not config.redis.configured:
        logger.error(
            "no Redis is configured (%s), and the scoreboard lives there; the "
            "counters of a run without Redis exist only inside that process",
            config.redis_error or "empty LLM_ROUTER_KNOWLEDGE_AGENTIC_REDIS_HOST",
        )
        return 1
    recorder = metrics or KnowledgeMetrics(
        connection=config.redis, key_prefix=config.key_prefix
    )
    rows = report(recorder, scopes, _days(str(args.day), int(args.days)))
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        _print_text(rows)
    return 0


def _load_config(args: Any, logger: Any) -> AgenticKnowledgeConfig:
    """
    Read the policy the proxy reads, with the environment over it.

    The same resolution, not an imported one: a stats reader that resolved a
    different prefix or scope than the proxy wrote with would report an empty
    scoreboard as an unused knowledge base, so the two must agree by following
    the same rule rather than by sharing a function either could change.
    """
    if str(args.config):
        config = AgenticKnowledgeConfig.from_file(pathlib.Path(str(args.config)))
    else:
        config = AgenticKnowledgeConfig.from_file()
    config.override_from_env(logger)
    config.validate()
    return config


def _scopes(args: Any, config: AgenticKnowledgeConfig, logger: Any) -> List[str]:
    """Resolve the scopes to read, from --scope or from a repository path."""
    given = [str(item) for item in (args.scope or []) if str(item).strip()]
    if given:
        return [normalize_scope(name) for name in given]
    root = pathlib.Path(str(args.path)).expanduser().resolve()
    scope = scope_key(
        str(root),
        strategy=config.scope_strategy,
        overrides=config.scope_overrides,
        default=config.scope_default,
    )
    if not scope:
        logger.error(
            "no scope given and none could be derived from %s; pass --scope",
            root,
        )
        return []
    return [scope]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
