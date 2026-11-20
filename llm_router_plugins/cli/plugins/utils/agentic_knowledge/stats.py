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
from llm_router_plugins.utils.knowledge.agentic_knowledge.learning import (
    LabelStore,
    SourceBandit,
    load_snapshot,
    snapshot_history,
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
        "--learning",
        action="store_true",
        help=(
            "Also read the learned state of the scopes: the model in force, "
            "the corpus it rests on, the history a rollback could reach, and "
            "the self-reinforcement monitors of the training log."
        ),
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
        payload: Dict[str, Any] = {"scoreboard": rows}
        if args.learning:
            payload["learning"] = [
                learning_view(
                    scope=scope,
                    config=config,
                    client=_redis_client(config),
                )
                for scope in scopes
            ]
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_text(rows)
        if args.learning:
            _print_learning(
                [
                    learning_view(
                        scope=scope,
                        config=config,
                        client=_redis_client(config),
                    )
                    for scope in scopes
                ]
            )
    return 0


def _redis_client(config: AgenticKnowledgeConfig) -> Any:
    """
    The client the learned state is read from, or ``None``.

    Built from the same connection the scoreboard uses, so a reader that
    resolves a different prefix or a different Redis than the proxy wrote
    with would report an empty learned state as a layer that never learned.
    """
    if not config.redis.configured:
        return None
    try:
        from llm_router_plugins.utils.knowledge.agentic_knowledge.redisclient import (
            RedisClientHandle,
        )

        handle = RedisClientHandle(config.redis)
        return handle.get()
    except Exception:
        return None


def learning_view(
    scope: str, config: AgenticKnowledgeConfig, client: Any
) -> Dict[str, Any]:
    """
    One scope's learned state, read without touching the request path.

    The view is what an operator asks before changing anything: what is
    serving, what it rests on, what it could go back to, and — from the
    training log — whether the layer is starting to reinforce itself, the
    mode in which the most injected advice stops being the advice that works
    and becomes the advice that gets injected most.  Every read is fail-open:
    a key that cannot be read is reported as absent, not as an error, because
    the command's job is to answer, and an operator reading a half-error is
    an operator who stops reading.
    """
    learning = config.learning
    prefix = config.key_prefix
    view: Dict[str, Any] = {"scope": scope, "enabled": bool(learning.enabled)}
    if client is None:
        view["note"] = "no Redis is configured; the learned state lives there"
        return view

    # The model in force, and what it rests on.
    snapshot = load_snapshot(
        client, scope, key_prefix=prefix, config=learning
    )
    if snapshot is not None:
        view["model"] = {
            "model_id": snapshot.model_id,
            "trained_at": snapshot.trained_at,
            "labelled": snapshot.labelled,
            "holdout_auc": snapshot.holdout_auc,
            "neutral": snapshot.neutral,
        }
    history = snapshot_history(client, scope, key_prefix=prefix)
    view["history"] = {
        "count": len(history),
        "kept": int(learning.snapshot_history_size),
        "newest": [
            str(row.get("model_id") or "") for row in history[:3]
        ],
    }

    # The corpus the next fit would stand on.
    corpus = 0
    labelled = 0
    try:
        store = LabelStore(client=client, key_prefix=prefix)
        rows = store.load(scope)
        corpus = len(rows)
        labelled = sum(1 for row in rows if row.label != 0.0)
    except Exception:
        pass
    view["corpus"] = {"records": corpus, "labelled": labelled}

    # The arms the bandit has folded, for the value a character of each
    # source earned against the weight the policy still ships.
    try:
        bandit = SourceBandit(client=client, config=learning)
        view["arms"] = bandit.summary()
    except Exception:
        view["arms"] = {}

    # The monitors, from the training log.  The log is bounded by design, so
    # these describe the window it still holds, not the whole deployment —
    # which is exactly the window the next fit would see, and the one a
    # self-reinforcing layer shows its tilt in first.
    view["monitors"] = _monitors(client, learning, scope)
    return view


def _monitors(client: Any, learning: Any, scope: str) -> Dict[str, Any]:
    """
    What the training log says about the shape of the evidence.

    A share of the log that is exposures of documents the layer has shown
    more than fifty times is a share of the corpus the layer has earned by
    repetition rather than by being right, and a p95 of the multipliers that
    the layer has spent on the advice it injects is a meter of how hard the
    learned path is steering.  Both are read from the window the log still
    holds, with the scope filtered out of the answer, because the question is
    "is this layer starting to like its own advice", not "how big is the log".
    """
    try:
        tail = client.xrange(learning.stream_key, min="-", max="+", count=2000)
    except Exception:
        return {"note": "the training log could not be read"}
    exposures: Dict[str, int] = {}
    rows_seen = 0
    for row in tail:
        fields = row[1] if isinstance(row, (list, tuple)) and len(row) > 1 else row
        raw = fields.get("d") if isinstance(fields, dict) else None
        if raw is None:
            continue
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        try:
            record = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if not isinstance(record, dict):
            continue
        if str(record.get("scope") or "") not in ("", scope):
            continue
        rows_seen += 1
        name = str(record.get("doc") or "")
        if name:
            exposures[name] = exposures.get(name, 0) + 1
    heavy = 0
    if exposures:
        heavy = sum(1 for count in exposures.values() if count > 50)
    return {
        "window": rows_seen,
        "docs": len(exposures),
        # The share of the documents the log still names that the layer has
        # shown more than fifty times: a self-reinforcing layer shows its
        # tilt here first, because the advice it likes is the advice it keeps
        # injecting, and the injections are what the log counts.
        "docs_over_50_exposures": round(heavy / len(exposures), 4)
        if exposures
        else 0.0,
        "heaviest": max(exposures.values()) if exposures else 0,
    }


def _print_learning(views: List[Dict[str, Any]]) -> None:
    """The learned state, in the shape an operator reads it by hand."""
    print()
    print("learned state")
    for view in views:
        scope = str(view.get("scope", ""))
        enabled = view.get("enabled")
        model = view.get("model") or {}
        corpus = view.get("corpus") or {}
        history = view.get("history") or {}
        monitors = view.get("monitors") or {}
        model_line = (
            f"{model.get('model_id', '')} (auc {model.get('holdout_auc', 0):.2f}, "
            f"{model.get('labelled', 0)} labels"
            + (", neutral" if model.get("neutral") else "")
            + ")"
            if model
            else "none"
        )
        print(
            f"  {scope[:28]:<28} enabled={str(enabled).lower():<5} "
            f"model={model_line} "
            f"corpus={corpus.get('labelled', 0)}/{corpus.get('records', 0)} "
            f"history={history.get('count', 0)}/{history.get('kept', 0)}"
        )
        newest = history.get("newest") or []
        if newest:
            print(
                f"  {'':<28} rollback: " + ", ".join(newest)
            )
        if monitors:
            print(
                f"  {'':<28} log: {monitors.get('docs', 0)} docs, "
                f"{monitors.get('docs_over_50_exposures', 0):.0%} over 50 "
                f"exposures (heaviest {monitors.get('heaviest', 0)})"
            )


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
