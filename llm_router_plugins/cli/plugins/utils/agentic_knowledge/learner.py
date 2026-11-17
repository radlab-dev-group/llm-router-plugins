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

from typing import Any, Dict, List, Optional, Union

from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
    AgenticKnowledgeConfig,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.learning import (
    FeedbackLog,
    FeedbackSample,
    LearningConfig,
    publish_snapshot,
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
        )
        published = False
        if self._client is not None:
            published = publish_snapshot(
                self._client,
                snapshot,
                key_prefix=self._key_prefix,
            )
        report["model_id"] = snapshot.model_id
        report["published"] = published
        return report

    def drain_and_fit(
        self,
        count: int = 500,
        last_id: str = "0-0",
        holdout_fraction: float = 0.2,
        now: float = 0.0,
    ) -> Dict[str, Any]:
        """
        One pass: read the stream, fit every scope in it, publish, report.

        Returns ``{"last_id": …, "scanned": n, "scopes": {scope: report}}``.
        A scope that appears in the batch with no labelled record publishes
        nothing: labels are the only thing this process can do with, and a
        stream of exposures without verdicts is a log, not a corpus.
        """
        last_id, samples = self._log.drain(count=count, last_id=last_id)
        by_scope: Dict[str, List[FeedbackSample]] = {}
        for sample in samples:
            # A record nobody has judged yet is an exposure, not a corpus: the
            # learner fits on labels, and a scope whose whole batch is unlabelled
            # is a scope that has nothing to fit, which is not the same as a
            # scope that fit nothing.
            if not sample.scope or not sample.label:
                continue
            by_scope.setdefault(sample.scope, []).append(sample)
        scopes: Dict[str, Dict[str, Any]] = {}
        for scope, rows in sorted(by_scope.items()):
            scopes[scope] = self.fit_scope(
                scope,
                rows,
                now=now,
                holdout_fraction=holdout_fraction,
            )
        if self._logger is not None:
            for scope, report in scopes.items():
                verdict = (
                    "accepted" if report.get("accepted") else report.get("refused")
                )
                self._logger.info(
                    "learner: scope=%s scanned=%d labelled=%d holdout_auc=%.3f "
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
            "last_id": last_id,
            "scanned": len(samples),
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
    return parser


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

    learner = Learner(
        config=config,
        client=client,
        logger=logger,
        rng=random.Random(args.seed),
    )
    result = learner.drain_and_fit(
        count=max(int(args.count), 1),
        holdout_fraction=float(args.holdout),
    )
    if args.report:
        print(json.dumps(result, sort_keys=True))
    if args.scope is not None:
        scopes = result.get("scopes") or {}
        if args.scope not in scopes:
            logger.warning("scope %s had no records in the batch", args.scope)
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    sys.exit(main())
