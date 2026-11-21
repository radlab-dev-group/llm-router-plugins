"""
The learner: what the indexing worker does with what the proxy has judged.

The proxy cannot fit a model and answer the agent at the same time, so it logs
one record per injected document — its features and, one turn later, its
verdict — to a Redis stream and moves on.  This process drains that stream,
fits a scope's ranking model on the labels it has gathered and publishes the
fit as a versioned snapshot the request path reads back with a one-key GET.

Nothing here trains on the request path, and nothing here changes the policy:
a fit that cannot separate held-out judgements better than the floor publishes
a neutral snapshot, which reads as ``1.0`` everywhere and behaves exactly like
the layer does today.  The policy's thresholds are the operator's to change;
this process only says, with the numbers, what they would earn if changed.
"""

import argparse
import json
import logging
import pathlib
import random
import sys

from typing import Any, Dict, List, Optional, Sequence, Union

from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
    AgenticKnowledgeConfig,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.learning import (
    acquire_pass_lock,
    compute_meta_prior,
    FeedbackLog,
    FeedbackSample,
    LabelStore,
    LearningConfig,
    load_meta_prior,
    release_pass_lock,
    save_meta_prior,
    SourceBandit,
    load_snapshot,
    publish_snapshot,
    rollback_snapshot,
    snapshot_history,
    train_ranker,
)

__all__ = ["Learner", "main"]


def _load_config(path: Optional[str], logger: Any) -> AgenticKnowledgeConfig:
    """The policy the deployment ships with; the learning section drives us."""
    config = AgenticKnowledgeConfig.from_file(
        pathlib.Path(path) if path else None
    )
    config.override_from_env(logger)
    config.validate()
    return config


class Learner:
    """Fit one scope's model from the settled records the proxy has logged."""

    def __init__(
        self,
        config: Union[AgenticKnowledgeConfig, LearningConfig],
        client: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        """Bind the store; read nothing yet."""
        self._config = config
        # A bare learning section is accepted too: the CLI is the only caller
        # that builds one, and a test that wants to fit a scope should not have
        # to manufacture the rest of the policy to do it.
        self._learning = (
            config if isinstance(config, LearningConfig) else config.learning
        )
        self._key_prefix = str(getattr(config, "key_prefix", "agentic-kb"))
        self._stream_key = self._learning.stream_key
        self._client = client
        self._logger = logger
        self._rng = rng or random.Random()
        self._log = FeedbackLog(
            client=client,
            stream_key=self._stream_key,
            logger=logger,
        )
        # The corpus a scope is fitted on: the accumulation of its settled
        # judgements, deduplicated, outliving the stream's MAXLEN.  The learner
        # fits on this, not on the batch it happened to drain.
        self._labels = LabelStore(
            client=client,
            key_prefix=self._key_prefix,
            ttl_seconds=self._corpus_ttl_seconds(),
            logger=logger,
        )
        self._cursor_key = "%s:learner:cursor" % self._key_prefix
        # The bandit is written by the request path, not here: this process
        # only reads what it has learned, so that a published snapshot can
        # say what a character of each source was worth when the fit shipped.
        self._bandit = SourceBandit(
            client=client, config=self._learning, rng=self._rng
        )

    def _corpus_ttl_seconds(self) -> float:
        """How long a settled label stays in the corpus before it ages out."""
        half_life = float(getattr(self._learning, "half_life_seconds", 0.0) or 0.0)
        return half_life * 4.0

    def _learning_section(self) -> LearningConfig:
        """The learning section of what this learner was built with."""
        return self._learning

    # -------------------------------------------------------------- api -----
    def fit_scope(
        self,
        scope: str,
        samples: List[FeedbackSample],
        now: float = 0.0,
        holdout_fraction: float = 0.2,
        arms: Optional[Dict[str, Dict[str, float]]] = None,
        prior_weights: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """
        Fit one scope and decide whether the fit is worth publishing.

        Returns the report the log line is built from: how much the scope had
        judged, what the fit separated on the holdout, and whether the snapshot
        was published or refused — with the reason, so "the worker did nothing"
        and "the worker refused to publish" are two different sentences.
        """
        snapshot, report = train_ranker(
            samples,
            scope,
            config=self._learning_section(),
            holdout_fraction=holdout_fraction,
            rng=self._rng,
            now=now,
            arms=self._bandit.arms() if arms is None else arms,
            prior_weights=prior_weights,
        )
        published = False
        if self._client is not None:
            published = publish_snapshot(
                self._client,
                snapshot,
                key_prefix=self._key_prefix,
                history_size=int(
                    getattr(self._learning, "snapshot_history_size", 8)
                ),
            )
        report["model_id"] = snapshot.model_id
        report["published"] = published
        report["arms"] = self._bandit.summary()
        return report

    def arm_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        What the source bandit knows: judgements folded and reward expected.

        The arm state belongs to the request path, which folds every settled
        document into it; this process only reads it, so an operator can ask
        the same question the snapshot answers — what was a character of each
        source worth — without reaching into a live counter.
        """
        return self._bandit.summary()

    def _read_meta_prior(self) -> Optional[List[float]]:
        """The cross-repository prior, when the deployment keeps one."""
        if not bool(getattr(self._learning, "meta_prior_enabled", False)):
            return None
        return load_meta_prior(
            self._client,
            key_prefix=self._key_prefix,
            config=self._learning,
        )

    def _last_snapshot(self, scope: str) -> Optional[Any]:
        """The model now serving one scope; the prior is built from these."""
        if self._client is None:
            return None
        try:
            return load_snapshot(
                self._client,
                scope,
                key_prefix=self._key_prefix,
                config=self._learning,
            )
        except Exception:
            return None

    def _refresh_meta_prior(self, accepted: List[Any]) -> None:
        """
        Rewrite the prior from what this pass accepted, if there is enough.

        A pass that accepted nothing leaves the prior alone: the last prior
        that was worth writing still is, and a pass that saw no evidence is
        not a reason to forget the evidence an earlier pass saw.  The same
        rule covers a pass that accepted fewer scopes than the minimum —
        a prior is a statement about what the repositories have taught, and
        this pass has not taught.
        """
        if (
            self._client is None
            or not bool(getattr(self._learning, "meta_prior_enabled", False))
        ):
            return
        minimum = int(getattr(self._learning, "meta_prior_min_scopes", 3))
        weights = compute_meta_prior(accepted, min_scopes=minimum)
        if weights is None:
            return
        save_meta_prior(
            self._client,
            weights,
            key_prefix=self._key_prefix,
            scopes=[item.scope for item in accepted if item.scope],
        )

    def read_cursor(self) -> str:
        """Where the last pass stopped in the training stream; ``0-0`` at first."""
        if self._client is None:
            return "0-0"
        try:
            raw = self._client.get(self._cursor_key)
        except Exception:
            return "0-0"
        if not raw:
            return "0-0"
        return raw.decode() if isinstance(raw, bytes) else str(raw)

    def save_cursor(self, last_id: str) -> None:
        """Remember where this pass stopped, so the next one starts there."""
        if self._client is None or not last_id or last_id == "0-0":
            return
        try:
            self._client.set(self._cursor_key, last_id)
        except Exception:
            # A cursor that will not persist means the next pass re-reads what
            # this one read.  Labels are deduplicated by key, so the corpus is
            # the same; only the work is repeated, which is not a corruption.
            pass

    def drain_and_fit(
        self,
        count: int = 500,
        last_id: Optional[str] = None,
        holdout_fraction: float = 0.2,
        now: float = 0.0,
        max_records: int = 20000,
        corpus_limit: int = 10000,
    ) -> Dict[str, Any]:
        """
        One pass: resume where the last pass stopped, fold what is new into
        every scope's corpus, fit on that corpus, publish, save the cursor.

        The fit is on the corpus and not on the batch.  The stream is a queue
        and a queue is trimmed, so a model fitted on the newest batch would be
        fitted on whatever happened to survive ``MAXLEN`` this hour, and a
        scope with ten thousand labels would be modelled from the last few
        hundred.  The batch decides *which* scopes are worth refitting; the
        corpus decides what the fit says.

        Returns ``{"last_id": ..., "scanned": n, "merged": n, "scopes": {...}}``.
        """
        cursor = last_id if last_id is not None else self.read_cursor()
        drained: List[FeedbackSample] = []
        seen = cursor
        while not drained or seen != cursor:
            budget = max(1, int(max_records)) - len(drained)
            if budget <= 0:
                break
            seen, rows = self._log.drain(count=min(int(count), budget), last_id=seen)
            drained.extend(rows)
            # Without Redis the log answers from a local tail, which is the
            # same handful every time; one read of it is the whole of it.
            if self._client is None or len(rows) < min(int(count), budget):
                break

        merged = self._labels.merge(drained)
        touched = sorted({s.scope for s in drained if s.scope and s.label})
        # A cold scope starts from what the warm ones learned.  The prior is
        # read once per pass; a scope with few labels of its own is fitted
        # from it rather than from nothing, which is the difference between
        # "the layer knows how to rank knowledge" and "this repository has
        # not judged anything yet".
        prior = self._read_meta_prior()
        scopes: Dict[str, Dict[str, Any]] = {}
        accepted: List[Any] = []
        for scope in touched:
            corpus = self._labels.load(scope, limit=corpus_limit)
            report = self.fit_scope(
                scope,
                corpus,
                now=now,
                holdout_fraction=holdout_fraction,
                prior_weights=prior,
            )
            scopes[scope] = report
            serving = self._last_snapshot(scope)
            if serving is not None and serving.ranker.trained:
                accepted.append(serving)
        self.save_cursor(seen)
        self._refresh_meta_prior(accepted)

        if self._logger is not None:
            self._logger.info(
                "learner: drained=%d merged=%d cursor=%s scopes=%d",
                len(drained),
                merged,
                seen,
                len(scopes),
            )
            for scope, report in scopes.items():
                verdict = (
                    "accepted" if report.get("accepted") else report.get("refused")
                )
                self._logger.info(
                    "learner: scope=%s corpus=%d labelled=%d holdout_auc=%.3f "
                    "model_id=%s %s published=%s",
                    scope,
                    report.get("samples", 0),
                    report.get("labelled", 0),
                    float(report.get("holdout_auc", 0.0)),
                    report.get("model_id", ""),
                    verdict or "neutral",
                    report.get("published", False),
                )
        return {
            "last_id": seen,
            "scanned": len(drained),
            "merged": merged,
            "scopes": scopes,
        }


def _build_client(config: AgenticKnowledgeConfig) -> Optional[Any]:
    """A Redis client for the worker; ``None`` when Redis is not configured."""
    if not config.redis.configured:
        return None
    try:
        import redis  # noqa: PLC0415 - optional dependency

        return redis.Redis(**config.redis.client_kwargs())
    except Exception:
        return None


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="llm-router-agentic-kb-learner",
        description=(
            "Fit the agentic knowledge layer's per-scope ranking models from "
            "the settled feedback the proxy has logged."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a knowledge policy JSON; defaults to the bundled policy.",
    )
    parser.add_argument(
        "--scope",
        default=None,
        help="Fit one scope instead of every scope in the batch.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=500,
        help="Records to read per pass; defaults to 500.",
    )
    parser.add_argument(
        "--holdout",
        type=float,
        default=0.2,
        help="Share of the labels held out of the fit; defaults to 0.2.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed of the split; the fit is reproducible from it.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print one JSON report line per scope and exit.",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Logging level; defaults to INFO.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=20000,
        help=(
            "Stop after reading this many stream entries in one pass; "
            "defaults to 20000."
        ),
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10000,
        help=(
            "Maximum corpus size per scope; the oldest labels are "
            "dropped; defaults to 10000."
        ),
    )
    parser.add_argument(
        "--reset-cursor",
        action="store_true",
        help="Start at the head of the stream instead of the last saved cursor.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help=(
            "Print the weight vector of the current snapshot for "
            "each touched scope and exit."
        ),
    )
    parser.add_argument(
        "--arms",
        action="store_true",
        help=(
            "Print the bandit's view of each (mode, source) arm — how many "
            "judgements it has folded and what reward it expects — and exit."
        ),
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help=(
            "Print the snapshots this scope has published and retired, newest "
            "first, with the evidence each was fitted on, and exit."
        ),
    )
    parser.add_argument(
        "--rollback",
        default=None,
        metavar="MODEL_ID",
        help=(
            "Put a retired snapshot back in force instead of fitting. The id "
            "comes from --history; the model it replaces goes into the "
            "history, so a rollback is itself reversible."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Fit even though another pass holds the learner's lock. The lock "
            "is what stops two cron minutes, or an impatient operator and a "
            "still-running pass, from draining the same cursor and publishing "
            "over each other; forcing it means accepting that the last writer "
            "decides what a scope serves."
        ),
    )
    parser.add_argument(
        "--bootstrap-git",
        default=None,
        metavar="REPO_ROOT",
        help=(
            "Seed a scope's knowledge base from the fix commits of a "
            "repository instead of waiting for its first agent: reads the "
            "subjects that say a change repaired something, publishes one "
            "recipe per file each touched, and exits. The cold start this "
            "answers is the one a meta-prior cannot: a first project has no "
            "other projects to learn from, and its own history is the "
            "judgement it has already made."
        ),
    )
    parser.add_argument(
        "--bootstrap-limit",
        type=int,
        default=50,
        help="How many commits a bootstrap reads (default: 50).",
    )
    parser.add_argument(
        "--bootstrap-since",
        default=None,
        metavar="WHEN",
        help=(
            "Only bootstrap commits newer than a git date, e.g. '6 months "
            "ago'. A repository whose history predates its current "
            "architecture should not teach the layer about the architecture "
            "it no longer has."
        ),
    )
    parser.add_argument(
        "--lock-ttl",
        default=None,
        metavar="SECONDS",
        help=(
            "How long the pass lock outlives a crashed learner; defaults to "
            "learning.lock_ttl_seconds."
        ),
    )
    return parser


def _bootstrap_git(
    config: AgenticKnowledgeConfig, args: Any, logger: Any
) -> int:
    """
    Seed a scope from the fix commits its own history already made.

    A scope is required: a bootstrap writes documents, and a document without
    a repository to belong to is a recipe about nowhere.  The publisher is the
    one the proxy uses, so the deduplication is the same one — re-running a
    bootstrap on a repository it already seeded is a no-op, not a doubling of
    the corpus.
    """
    from llm_router_plugins.utils.knowledge.agentic_knowledge.events import (
        KnowledgeEventPublisher,
    )
    from llm_router_plugins.utils.knowledge.agentic_knowledge.miner import (
        mine_commit_fixes,
    )

    scope = str(args.scope or "").strip()
    if not scope:
        logger.error("--bootstrap-git needs --scope")
        return 1
    events = mine_commit_fixes(
        str(args.bootstrap_git),
        scope,
        limit=max(1, int(args.bootstrap_limit)),
        since=getattr(args, "bootstrap_since", None) or None,
    )
    if not events:
        logger.info(
            "bootstrap: %s has no fix commits to mine for %s",
            args.bootstrap_git,
            scope,
        )
        print(json.dumps({"scope": scope, "published": 0, "found": 0}))
        return 0
    publisher = KnowledgeEventPublisher(
        connection=config.redis,
        stream_key=config.stream_key,
        key_prefix=config.key_prefix,
        dedup_ttl_seconds=int(config.dedup_ttl_seconds),
        max_stream_len=int(config.max_stream_len),
        logger=logger,
    )
    status = publisher.publish(events)
    logger.info(
        "bootstrap: %s mined=%d published=%s reason=%s",
        scope,
        len(events),
        getattr(status, "published", "?"),
        getattr(status, "reason", "") or "-",
    )
    print(
        json.dumps(
            {
                "scope": scope,
                "found": len(events),
                "published": int(getattr(status, "published", 0) or 0),
                "status": str(getattr(status, "status", "")),
                "reason": str(getattr(status, "reason", "") or ""),
            },
            sort_keys=True,
        )
    )
    return 0


def _scope_view(
    config: AgenticKnowledgeConfig,
    client: Any,
    learner: Learner,
    args: Any,
    logger: Any,
) -> int:
    """
    Show a scope's published history, or move one snapshot back into force.

    A scope is required for a rollback and taken from the one scope the policy
    has state for when there is only one: an operator who typed `--rollback`
    without a `--scope` on a single-scope deployment means that scope, and an
    operator on a multi-scope deployment is stopped with the list instead of
    having a guess made for them.
    """
    prefix = config.key_prefix
    scopes = _known_scopes(client, prefix)
    scope = args.scope or ""
    if args.rollback:
        if not scope:
            if len(scopes) == 1:
                scope = scopes[0]
            else:
                logger.error(
                    "--rollback needs --scope; histories exist for: %s",
                    ", ".join(scopes) or "none",
                )
                return 1
        moved = rollback_snapshot(
            client,
            scope,
            args.rollback,
            key_prefix=prefix,
            history_size=int(config.learning.snapshot_history_size),
        )
        if not moved:
            logger.error(
                "scope %s has no retired snapshot %s; see --history --scope %s",
                scope,
                args.rollback,
                scope,
            )
            return 1
        print(
            json.dumps(
                {"scope": scope, "rolled_back_to": args.rollback}, sort_keys=True
            )
        )
        return 0
    if not scope:
        print(
            json.dumps(
                {
                    scope: [
                        _row_summary(row)
                        for row in snapshot_history(client, scope, key_prefix=prefix)
                    ]
                    for scope in scopes
                },
                sort_keys=True,
            )
        )
        return 0
    print(
        json.dumps(
            {
                "scope": scope,
                "live": _row_summary(_live_row(client, prefix, scope)),
                "history": [
                    _row_summary(row)
                    for row in snapshot_history(client, scope, key_prefix=prefix)
                ],
            },
            sort_keys=True,
        )
    )
    return 0


def _known_scopes(client: Any, key_prefix: str) -> List[str]:
    """Scopes with a live model or a history, read without a SCAN."""
    scopes: List[str] = []
    try:
        # A scan is the honest way; a client without one reports no scopes,
        # which sends the operator to pass --scope rather than to a wrong one.
        for key in client.scan_iter(match="%s:model:*" % key_prefix, count=512):
            name = str(key)
            if ":model_history:" in name:
                continue
            scopes.append(name.split(":model:", 1)[1])
    except Exception:
        return []
    return sorted(set(scopes))


def _live_row(client: Any, key_prefix: str, scope: str) -> Dict[str, Any]:
    """The live snapshot of a scope as a row, or an empty one."""
    try:
        raw = client.get("%s:model:%s" % (key_prefix, scope))
        if not raw:
            return {}
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _row_summary(row: Dict[str, Any]) -> Dict[str, Any]:
    """The few numbers an operator reads before choosing a snapshot."""
    return {
        "model_id": str(row.get("model_id") or ""),
        "trained_at": float(row.get("trained_at") or 0.0),
        "labelled": int(row.get("labelled") or 0),
        "holdout_auc": float(row.get("holdout_auc") or 0.0),
        "neutral": not bool(row.get("w")),
    }


def main(argv: Optional[List[str]] = None) -> int:
    """
    Run one pass of the learner.

    Parameters
    ----------
    argv : List[str], optional
        Command-line arguments; ``sys.argv`` when omitted.

    Returns
    -------
    int
        Process exit code: ``0`` when the pass ran, ``1`` when it could not.
    """
    args = build_parser().parse_args(argv)
    level = getattr(
        logging, str(args.log_level or "INFO").upper(), logging.INFO
    )
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logger = logging.getLogger("agentic-kb-learner")

    config = _load_config(args.config, logger)
    client = _build_client(config)
    if client is None:
        logger.error(
            "the learner needs Redis to read the training stream; configure "
            "LLM_ROUTER_KNOWLEDGE_AGENTIC_REDIS_HOST (and friends)"
        )
        return 1

    if getattr(args, "bootstrap_git", None):
        return _bootstrap_git(config, args, logger)

    learner = Learner(
        config=config,
        client=client,
        logger=logger,
        rng=random.Random(args.seed),
    )
    # Rollback and history are reads and one copy of a key; neither fits
    # anything, and neither should drain the stream it would then have to
    # pretend it did not consume.
    if args.history or args.rollback:
        return _scope_view(config, client, learner, args, logger)
    lock_ttl = (
        float(args.lock_ttl)
        if args.lock_ttl
        else float(config.learning.lock_ttl_seconds)
    )
    token = acquire_pass_lock(
        client, key_prefix=config.key_prefix, ttl_seconds=lock_ttl
    )
    if not token and args.force:
        token = acquire_pass_lock(
            client,
            key_prefix=config.key_prefix,
            ttl_seconds=lock_ttl,
            force=True,
        )
        if token:
            logger.warning(
                "learner: --force took the pass lock from whatever held it"
            )
    if not token:
        logger.warning(
            "learner: another pass holds the lock (ttl %ss); nothing fitted "
            "this pass — wait for it, or pass --force",
            int(lock_ttl),
        )
        return 0
    last_id = None
    if args.reset_cursor:
        learner.save_cursor("0-0")
        last_id = "0-0"
    try:
        result = learner.drain_and_fit(
            count=max(int(args.count), 1),
            last_id=last_id,
            holdout_fraction=float(args.holdout),
            max_records=max(int(args.max_records), 1),
            corpus_limit=max(int(args.window), 1),
        )
    finally:
        if not release_pass_lock(client, config.key_prefix, token):
            # The lock expiring on its own is a pass that ran longer than its
            # ttl, which is worth one line: it says the next cron minute may
            # fit alongside this one's tail.
            logger.warning("learner: pass lock was no longer ours at release")
    self_prefix = config.key_prefix
    if args.arms:
        print(json.dumps({"arms": learner.arm_summary()}, sort_keys=True))
        return 0
    if args.show:
        for scope, report in (result.get("scopes") or {}).items():
            snapshot = load_snapshot(client, scope, key_prefix=self_prefix)
            if snapshot is not None:
                print(json.dumps({"scope": scope, "weights": list(snapshot.weights)}))
    if args.report:
        print(json.dumps(result, sort_keys=True))
    if args.scope is not None:
        scopes = result.get("scopes") or {}
        if args.scope not in scopes:
            logger.warning("scope %s had no records in the batch", args.scope)
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    sys.exit(main())
