"""
The learned part of the knowledge layer: what the static policy cannot know.

Three numbers decide what an agent is shown today, and all three were chosen
by hand: the source weights of a mode, the bonus a matching signature earns,
and the ``0.3 … 1.5`` range a document's upvotes move it through.  The layer
already measures what happens after an injection — a mined fix or the same
failure returning — so those numbers do not have to stay chosen.  They can be
estimated, and this module is the estimation.

It is built on one rule, which is the reason it is split the way it is:
**nothing trains on the request path.**  A proxy request has a three-second
read budget and must fail open, so the request path only *reads* — a document
reputation from one Redis hash and a model snapshot from one Redis key — and
writes the features it saw to a training stream.  The fitting happens in the
worker, next to the indexing it already does, and what it publishes is a
versioned, clamped, auditable snapshot.  A model that cannot beat the policy
publishes the policy.

Two loops, deliberately of different speed:

* :class:`DocReputation` — online, per document, a Beta posterior with a
  half-life.  Two ``HINCRBY`` and one ``HMGET``; no worker, nothing to fit,
  because a Bernoulli posterior is counted rather than trained.  It replaces
  ``ContextAssembler._feedback_multiplier`` with the same range and the same
  prior, plus decay and Thompson sampling, so a debunked recipe still sinks
  and a proven one still rises — but a recipe nobody has needed for two months
  drifts back to "unknown" instead of staying famous forever.
* :class:`ContextRanker` — offline, per scope, a ridge logistic regression over
  the features a candidate already carries.  Fitted by :func:`train_ranker` in
  the worker, published by :func:`publish_snapshot`, read by
  :class:`LearningPolicy` with a local TTL cache.  The Laplace covariance gives
  each scope its own exploration term, so a repository with eleven labels is
  served mostly by the policy and a repository with eleven thousand is served
  mostly by the model.

:class:`SourceBandit` sits between them: per ``(mode, kind)`` it learns what a
character of budget is worth, which is the number the hand-written ``weight``
in the policy JSON is a guess at.

Everything here is stdlib.  No numpy, no torch, no model files: the state is a
weight vector and two integers per document, both of which fit in an annotation
and both of which an operator can read.
"""

import json
import logging
import math
import random
import threading
import time

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "LearningConfig",
    "CandidateFeatures",
    "FeedbackSample",
    "DocReputation",
    "ContextRanker",
    "ModelSnapshot",
    "SourceBandit",
    "LearningPolicy",
    "FeedbackLog",
    "train_ranker",
    "publish_snapshot",
    "load_snapshot",
    "FEATURE_NAMES",
    "SCALE_RRF",
    "SCALE_RERANKER",
]

#: Scales the engine names on a hit.  A fused rank and a cosine similarity are
#: different numbers that share a field, so the scale is a feature: the model
#: learns how far to trust each of them instead of one constant trusting both.
SCALE_RRF = "rrf"
SCALE_RERANKER = "reranker"

#: What the weight vector holds, in order.  The names are part of the snapshot
#: contract: a snapshot whose names do not match is refused rather than read.
FEATURE_NAMES: Tuple[str, ...] = (
    "bias",
    "score",
    "score_sq",
    "scale_rrf",
    "scale_reranker",
    "signature_match",
    "path_match",
    "path_stem_match",
    "command_match",
    "age_days_log",
    "upvotes_log",
    "downvotes_log",
    "exposure_log",
    "rank",
    "size_share",
)

_N_FEATURES = len(FEATURE_NAMES)

#: The features a document is named by.  A corpus that varies none of them
#: cannot identify a weight, and a weight nobody identified must not be used.
_LEARNABLE = (1, 5, 6, 7, 8, 9, 13)

#: Correlation at which two features are one feature wearing two names, and the
#: second is pruned rather than fitted.
_COLLINEAR_MAX = 0.995

#: How far a Newton step may be halved before the fit settles for what it has.
_BACKTRACK_LIMIT = 24


def _sigmoid(x: float) -> float:
    """Numerically safe logistic."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def _clamp(value: float, low: float, high: float) -> float:
    """Hold a number inside a range, whatever the caller produced."""
    if not math.isfinite(value):
        return low if value < 0 else high
    return low if value < low else high if value > high else value


def _log1p(value: float) -> float:
    """``log1p`` of a non-negative count, floored at zero."""
    return math.log1p(value) if value > 0.0 else 0.0


@dataclass(frozen=True)
class LearningConfig:
    """
    The knobs of the learned path, all of them bounded.

    These belong in the policy JSON under ``learning`` and are validated at
    startup by the same fail-hard rule the rest of the policy uses: a
    misconfigured learner is a configuration error, while a learner that
    cannot read Redis is a Tuesday.

    Parameters
    ----------
    enabled : bool
        Master switch.  Off means every multiplier is ``1.0`` and nothing is
        written, which is how the layer behaves today.
    mult_min, mult_max : float
        The range one learned multiplier may move a vote through.  Defaults to
        the range ``_feedback_multiplier`` already uses, so turning the learner
        on cannot change how much influence learning has — only what it says.
    prior_alpha, prior_beta : float
        The Beta prior of a document nobody has judged: ``Beta(2, 2)``, the
        ``+2`` of the current formula, kept as a prior rather than as an
        inline constant so the policy can widen it.
    half_life_seconds : float
        How long a reputation takes to decay halfway back to the prior.  Zero
        disables decay, which is the behaviour of the current counters.
    min_observations : int
        Labels a document needs before its reputation moves it at all.  Below
        it the multiplier is ``1.0``: an unjudged document must not be cheaper
        than a judged one just because its posterior is wide.
    l2 : float
        Ridge strength of the ranker, **per 100 labelled samples**: the prior
        precision actually used is ``l2 * n / 100``.  Scaling it is what keeps
        the prior a belief rather than a rounding error — an absolute lambda of
        8 against 800 samples is noise, and the fit then runs to the clip on
        whatever feature happens to separate.  With few labels this term is what
        keeps the weights near zero, which is what "be the policy" means
        numerically.
    weight_clip : float
        Ceiling of one trained weight.  A feature that would dominate the vote
        by itself is a fit on noise, and this is where that stops.
    newton_steps : int
        Newton iterations per fit.  Four is plenty for fifteen features.
    min_samples, min_holdout_auc : int, float
        What a scope must show before its snapshot is published.  A scope that
        shows less gets a neutral snapshot, which reads as ``1.0``.
    explore_coeff : float
        Laplace standard deviations of exploration bonus.  Zero is pure
        exploitation; the default is small, because the multiplier is already
        clamped and exploration that costs characters is not free.
    explore_rate : float
        Share of requests served by a sampled rather than a mean snapshot.
    model_ttl_seconds : float
        How long a worker holds a snapshot before reading it again.
    stream_key, key_prefix : str
        Where the training log and the state live.
    """

    enabled: bool = False
    mult_min: float = 0.3
    mult_max: float = 1.5
    prior_alpha: float = 2.0
    prior_beta: float = 2.0
    half_life_seconds: float = 2592000.0
    min_observations: int = 3
    l2: float = 32.0
    weight_clip: float = 3.0
    newton_steps: int = 4
    min_samples: int = 200
    #: Minimum weighted evidence the fit must rest on.  Counterfactual labels
    #: count fully; coarse labels count half.  Below this the fit is published
    #: neutral: "a model that cannot beat the policy publishes the policy."
    min_evidence: float = 100.0
    min_holdout_auc: float = 0.6
    explore_coeff: float = 0.25
    explore_rate: float = 0.1
    model_ttl_seconds: float = 60.0
    stream_key: str = "agentic_kb:training"
    key_prefix: str = "agentic-kb"

    def __post_init__(self) -> None:
        """Refuse a learner that could not be reasoned about."""
        if not 0.0 < self.mult_min <= 1.0 <= self.mult_max:
            raise ValueError(
                "learning.mult_min/mult_max must satisfy 0 < min <= 1 <= max"
            )
        if self.prior_alpha <= 0.0 or self.prior_beta <= 0.0:
            raise ValueError("learning.prior_alpha/prior_beta must be positive")
        if self.half_life_seconds < 0.0:
            raise ValueError("learning.half_life_seconds must not be negative")
        if self.min_observations < 0:
            raise ValueError("learning.min_observations must not be negative")
        if self.l2 <= 0.0:
            raise ValueError("learning.l2 must be positive")
        if self.weight_clip <= 0.0:
            raise ValueError("learning.weight_clip must be positive")
        if self.newton_steps < 1:
            raise ValueError("learning.newton_steps must be at least 1")
        if self.min_samples < 1:
            raise ValueError("learning.min_samples must be at least 1")
        if self.min_evidence <= 0.0:
            raise ValueError("learning.min_evidence must be positive")
        if not 0.0 <= self.min_holdout_auc <= 1.0:
            raise ValueError("learning.min_holdout_auc must be within 0..1")
        if self.explore_coeff < 0.0 or not 0.0 <= self.explore_rate <= 1.0:
            raise ValueError("learning.explore_coeff/explore_rate out of range")
        if self.model_ttl_seconds <= 0.0:
            raise ValueError("learning.model_ttl_seconds must be positive")
        if not self.stream_key.strip() or not self.key_prefix.strip():
            raise ValueError("learning.stream_key/key_prefix must not be empty")

    @classmethod
    def from_dict(
        cls,
        raw: Optional[Dict[str, Any]],
        key_prefix: Optional[str] = None,
        stream_key: Optional[str] = None,
    ) -> "LearningConfig":
        """
        Build the section from policy JSON, refusing unknown keys.

        Mirrors how ``escalation`` is read: an operator's typo is a startup
        failure, not a silently ignored setting.  *key_prefix* and *stream_key*
        bind the section to the deployment it will run in: the state lives next
        to the rest of the plugin state, under the same prefix.
        """
        raw = dict(raw or {})
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"unknown learning keys: {', '.join(unknown)}")
        if key_prefix is not None and raw.get("key_prefix") in (None, ""):
            raw["key_prefix"] = key_prefix
        if stream_key is not None and raw.get("stream_key") in (None, ""):
            raw["stream_key"] = stream_key
        return cls(**raw)


@dataclass(frozen=True)
class CandidateFeatures:
    """
    What one retrieved candidate looks like to the learner.

    Every field is a number the assembler already has in hand when it builds a
    ``_Candidate``: the score it normalized, the scale it normalized it on, the
    metadata bonuses it added by rule, the position the engine ranked it at.
    Nothing here is content — no text, no path, no document body — because this
    record is what goes into the training stream, and the worker masks
    documents precisely so that the layer never carries content it cannot
    unmask.
    """

    kind: str = ""
    mode: str = ""
    score: float = 0.0
    score_scale: str = ""
    signature_match: bool = False
    path_match: int = 0
    path_stem_match: int = 0
    command_match: bool = False
    age_days: float = 0.0
    upvotes: int = 0
    downvotes: int = 0
    exposure: int = 0
    rank: int = 0
    size_chars: int = 0
    budget_chars: int = 0

    def vector(self) -> List[float]:
        """The weight vector's input, in :data:`FEATURE_NAMES` order."""
        size_share = 0.0
        if self.budget_chars > 0:
            size_share = _clamp(
                self.size_chars / float(self.budget_chars), 0.0, 1.0
            )
        return [
            1.0,
            _clamp(self.score, 0.0, 1.0),
            _clamp(self.score, 0.0, 1.0) ** 2,
            1.0 if self.score_scale == SCALE_RRF else 0.0,
            1.0 if self.score_scale == SCALE_RERANKER else 0.0,
            1.0 if self.signature_match else 0.0,
            1.0 if self.path_match > 0 else 0.0,
            _clamp(self.path_stem_match, 0, 4) / 4.0,
            1.0 if self.command_match else 0.0,
            _log1p(max(self.age_days, 0.0)),
            _log1p(self.upvotes),
            _log1p(self.downvotes),
            _log1p(self.exposure),
            _clamp(self.rank, 0, 20) / 20.0,
            size_share,
        ]

    def to_json(self) -> Dict[str, Any]:
        """The stream form: numbers and short names only."""
        return {
            "kind": self.kind,
            "mode": self.mode,
            "score": round(self.score, 6),
            "scale": self.score_scale,
            "sig": bool(self.signature_match),
            "paths": int(self.path_match),
            "stems": int(self.path_stem_match),
            "cmd": bool(self.command_match),
            "age": round(self.age_days, 3),
            "up": int(self.upvotes),
            "down": int(self.downvotes),
            "exp": int(self.exposure),
            "rank": int(self.rank),
            "chars": int(self.size_chars),
            "budget": int(self.budget_chars),
        }

    @classmethod
    def from_json(cls, raw: Any) -> "CandidateFeatures":
        """Read back what :meth:`to_json` wrote, tolerating an older record."""
        raw = raw if isinstance(raw, dict) else {}
        return cls(
            kind=str(raw.get("kind") or ""),
            mode=str(raw.get("mode") or ""),
            score=float(raw.get("score") or 0.0),
            score_scale=str(raw.get("scale") or ""),
            signature_match=bool(raw.get("sig")),
            path_match=int(raw.get("paths") or 0),
            path_stem_match=int(raw.get("stems") or 0),
            command_match=bool(raw.get("cmd")),
            age_days=float(raw.get("age") or 0.0),
            upvotes=int(raw.get("up") or 0),
            downvotes=int(raw.get("down") or 0),
            exposure=int(raw.get("exp") or 0),
            rank=int(raw.get("rank") or 0),
            size_chars=int(raw.get("chars") or 0),
            budget_chars=int(raw.get("budget") or 0),
        )


@dataclass(frozen=True)
class FeedbackSample:
    """
    One settled judgement of one injected document.

    ``label`` is the counterfactual one, not the coarse one: a fix mined in the
    next turn credits the documents that could have caused it — the same
    signature, the same files, the same verification command — and leaves the
    rest unlabelled.  ``label == 0.0`` is not a downvote, it is "this turn
    proves nothing about this document", and it trains nothing while still
    being logged, because the exposure it records is what makes the credit
    assignment auditable later.
    """

    scope: str
    session_id: str
    generation: str
    document_name: str
    features: CandidateFeatures
    label: float
    #: ``counterfactual`` when the label came from evidence, ``coarse`` when it
    #: came from the current "the whole block was on screen" rule.
    label_kind: str = "counterfactual"
    #: The snapshot that was serving when this injection happened.
    model_id: str = ""
    at: float = 0.0

    def to_json(self) -> Dict[str, Any]:
        """The training-stream record."""
        return {
            "scope": self.scope,
            "session": self.session_id,
            "generation": self.generation,
            "doc": self.document_name,
            "f": self.features.to_json(),
            "label": round(float(self.label), 4),
            "kind": self.label_kind,
            "model_id": self.model_id,
            "at": round(float(self.at or time.time()), 3),
        }

    @classmethod
    def from_json(cls, raw: Any) -> "FeedbackSample":
        """Read back a stream record; a malformed one is dropped by the caller."""
        raw = raw if isinstance(raw, dict) else {}
        return cls(
            scope=str(raw.get("scope") or ""),
            session_id=str(raw.get("session") or ""),
            generation=str(raw.get("generation") or ""),
            document_name=str(raw.get("doc") or ""),
            features=CandidateFeatures.from_json(raw.get("f")),
            label=float(raw.get("label") or 0.0),
            label_kind=str(raw.get("kind") or "coarse"),
            model_id=str(raw.get("model_id") or ""),
            at=float(raw.get("at") or 0.0),
        )


class _RedisLike:
    """
    What this module asks Redis for; anything with these methods works.

    A documentation of the contract, not a type the callers must inherit: a
    real caller is either ``redis.Redis`` or a lazy holder that resolves the
    client of its own process on every command, and neither of those is a
    subclass of this.  The parameters are therefore annotated loosely, and a
    test fake that answers these commands is as valid as the client itself.
    """

    def hset(self, key: str, field: str, value: str) -> Any:
        raise NotImplementedError

    def hgetall(self, key: str) -> Any:
        raise NotImplementedError

    def xrange(
        self, key: str, min: str = "-", max: str = "+", count: int = 0
    ) -> Any:
        raise NotImplementedError

    def expire(self, key: str, seconds: int) -> Any:
        raise NotImplementedError

    def get(self, key: str) -> Any:
        raise NotImplementedError

    def set(self, key: str, value: str) -> Any:
        raise NotImplementedError

    def xadd(self, key: str, fields: Any, maxlen: int = 0) -> Any:
        raise NotImplementedError


class DocReputation:
    """
    How well one document has done, as a Beta posterior with a half-life.

    The current ``_feedback_multiplier`` is ``(up + 2) / (up + down + 2)``
    mapped onto ``0.3 … 1.5``.  That is a Beta(2, 2) posterior mean, which is
    the right idea; what it lacks is a clock and a spread.  Without a clock a
    recipe that was right before a refactor stays right forever, and a recipe
    debunked once carries that verdict into a repository that has since
    changed.  Without a spread the layer cannot tell three upvotes from three
    hundred, so it cannot explore.

    Both are cheap.  The clock is a decay of the accumulated evidence toward
    the prior, computed on read from one stored timestamp — no sweeper, no
    scan, nothing running when nobody is asking.  The spread is Thompson
    sampling: draw from ``Beta(alpha, beta)`` instead of taking the mean, and a
    document with two labels moves a lot while a document with two hundred
    barely moves, which is exactly how much trust each has earned.

    State is one Redis hash per document (``a``, ``b``, ``e``, ``t``) with a
    TTL of four half-lives, and a local mirror so a repeated read in one
    generation costs nothing.  No Redis: the mirror is the whole state, which
    is the same degradation the generation cache already accepts.
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        config: Optional[LearningConfig] = None,
        rng: Optional[random.Random] = None,
        time_fn: Any = time.time,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Bind the store and the prior; read nothing yet."""
        self._client = client
        self._config = config or LearningConfig()
        self._rng = rng or random.Random()
        self._time = time_fn
        self._logger = logger
        self._local: Dict[str, Tuple[float, float, float, float]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ api -
    def observe(self, document_name: str, label: float, now: float = 0.0) -> None:
        """
        Fold one judgement into the document's posterior.

        A fractional label is folded as its own weight: ``0.5`` for "plausible
        but not proven" is half an upvote, which is what makes the graded
        credit assignment of the settle path usable without inventing a second
        counter.  The evidence is added **beyond the prior** — the stored state
        already contains it, because the first read returns the prior — so
        ``label`` moves the posterior from where it is, and the prior that
        decay is measured against and the prior a document starts with are the
        same numbers.  Subtracting the prior here would read the evidence as
        "the posterior minus the prior" and add it back, leaving the prior in
        place of the judgement: every document would then sit at ``prior +
        label`` rather than at its own evidence, and the decay below would
        measure the wrong distance.
        """
        if not document_name or not label:
            return
        now = float(now or self._time())
        alpha, beta, exposure, at = self._posterior(document_name, now)
        if label > 0:
            alpha += float(label)
        else:
            beta += float(-label)
        exposure += 1.0
        self._store(document_name, alpha, beta, exposure, now, at)

    def multiplier(self, document_name: str, now: float = 0.0) -> float:
        """
        The vote multiplier this document has earned, ``1.0`` when unknown.

        Below ``min_observations`` the answer is deliberately neutral rather
        than the prior mean: an unjudged document must not be served cheaper
        than a judged one, or the learner would punish new knowledge, which is
        the opposite of what a knowledge base is for.
        """
        cfg = self._config
        if not document_name:
            return 1.0
        now = float(now or self._time())
        alpha, beta, exposure, _ = self._posterior(document_name, now)
        if exposure < cfg.min_observations:
            return 1.0
        try:
            draw = self._rng.betavariate(max(alpha, 1e-6), max(beta, 1e-6))
        except Exception:  # a degenerate draw is not worth a request
            draw = alpha / (alpha + beta)
        span = cfg.mult_max - cfg.mult_min
        moved = cfg.mult_min + span * draw
        return round(_clamp(moved, cfg.mult_min, cfg.mult_max), 4)

    def stats(self, document_name: str, now: float = 0.0) -> Dict[str, float]:
        """The posterior in the terms the current annotation already uses."""
        now = float(now or self._time())
        alpha, beta, exposure, _ = self._posterior(document_name, now)
        total = alpha + beta
        mean = alpha / total if total else 0.5
        return {
            "alpha": round(alpha, 4),
            "beta": round(beta, 4),
            "mean": round(mean, 4),
            "exposure": int(exposure),
        }

    def multiplier_range(self) -> Tuple[float, float]:
        """The clamp this reputation applies, for the annotation."""
        return (self._config.mult_min, self._config.mult_max)

    # ------------------------------------------------------------ internals -
    def _posterior(
        self, document_name: str, now: float
    ) -> Tuple[float, float, float, float]:
        """Read and decay one document's state; never raise."""
        cfg = self._config
        key = self._key(document_name)
        with self._lock:
            local = self._local.get(key)
        raw = local
        if raw is None and self._client is not None:
            raw = self._read(key)
            if raw is not None:
                with self._lock:
                    self._local[key] = raw
        if raw is None:
            return (cfg.prior_alpha, cfg.prior_beta, 0.0, now)
        alpha, beta, exposure, at = raw
        return self._decayed(alpha, beta, exposure, at, now)

    def _decayed(
        self,
        alpha: float,
        beta: float,
        exposure: float,
        at: float,
        now: float,
    ) -> Tuple[float, float, float, float]:
        """
        Age the evidence toward the prior.

        The evidence *beyond* the prior decays, not the prior itself, so a
        document left alone converges on "unknown" rather than on "bad".  That
        distinction is the whole reason the half-life is safe here: forgetting
        a judgement is not the same as recording a failure.
        """
        cfg = self._config
        if cfg.half_life_seconds <= 0.0 or at <= 0.0 or now <= at:
            return (alpha, beta, exposure, at)
        factor = 0.5 ** ((now - at) / cfg.half_life_seconds)
        return (
            1.0 + (alpha - 1.0) * factor if alpha > 1.0 else alpha,
            1.0 + (beta - 1.0) * factor if beta > 1.0 else beta,
            exposure * factor,
            at,
        )

    def _store(
        self,
        document_name: str,
        alpha: float,
        beta: float,
        exposure: float,
        now: float,
        previous_at: float,
    ) -> None:
        """Write through, keeping the local mirror authoritative for reads."""
        key = self._key(document_name)
        at = now if now > previous_at else previous_at or now
        with self._lock:
            self._local[key] = (alpha, beta, exposure, at)
        if self._client is None:
            return
        cfg = self._config
        try:
            client = self._client
            client.hset(key, "a", f"{alpha:.6f}")
            client.hset(key, "b", f"{beta:.6f}")
            client.hset(key, "e", f"{exposure:.6f}")
            client.hset(key, "t", f"{at:.3f}")
            if cfg.half_life_seconds > 0.0:
                client.expire(key, int(cfg.half_life_seconds * 4))
        except Exception as exc:
            self._warn("reputation write failed: %s", type(exc).__name__)

    def _read(self, key: str) -> Optional[Tuple[float, float, float, float]]:
        """One ``HGETALL``; anything unreadable is an unknown document."""
        try:
            raw = self._client.hgetall(key)  # type: ignore[union-attr]
        except Exception as exc:
            self._warn("reputation read failed: %s", type(exc).__name__)
            return None
        if not raw:
            return None
        decoded = {
            (k.decode() if isinstance(k, bytes) else str(k)): (
                v.decode() if isinstance(v, bytes) else v
            )
            for k, v in dict(raw).items()
        }
        try:
            return (
                float(decoded.get("a", 1.0)),
                float(decoded.get("b", 1.0)),
                float(decoded.get("e", 0.0)),
                float(decoded.get("t", 0.0)),
            )
        except (TypeError, ValueError):
            return None

    def _key(self, document_name: str) -> str:
        """One hash per document, next to the rest of the plugin state."""
        return f"{self._config.key_prefix}:rep:{document_name}"

    def _warn(self, message: str, *args: Any) -> None:
        """Report once, without content."""
        if self._logger is not None:
            self._logger.warning(message, *args)


class ContextRanker:
    """
    A ridge logistic regression over candidate features, with a Laplace spread.

    The vote the assembler computes today is
    ``(base * weight + bonus) * feedback_mult``, where ``weight`` comes from
    the policy JSON, ``bonus`` from three hand-set constants and ``feedback_mult``
    from a counter.  This is the same shape with the constants fitted:
    ``P(useful | features)`` per candidate, mapped onto the same clamped range,
    so a fitted model can reorder the block but can never outvote the policy
    by more than the policy allows.

    Fitting is Newton with a ridge prior, which for fifteen features and a few
    thousand rows is milliseconds and needs no dependency.  The Hessian at the
    optimum is kept as the Laplace precision; its diagonal gives a standard
    deviation per candidate, and that standard deviation is the exploration
    term — a scope with eleven labels gets a wide one and is served mostly by
    the policy, a scope with eleven thousand gets a narrow one.

    A weight is only allowed to move for a feature the corpus actually varied.
    A feature that was constant in training carries no information, its weight
    is unidentifiable, and shipping an unidentifiable number is how a model
    ends up ordering by an accident of one repository's history.
    """

    def __init__(
        self,
        weights: Optional[Sequence[float]] = None,
        precision: Optional[Sequence[Sequence[float]]] = None,
        config: Optional[LearningConfig] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        """Start at the neutral model: every multiplier ``1.0``."""
        self.weights: List[float] = list(weights or [0.0] * _N_FEATURES)
        cfg = config or LearningConfig()
        self.precision: List[List[float]] = [
            [cfg.l2 if i == j else 0.0 for j in range(_N_FEATURES)]
            for i in range(_N_FEATURES)
        ]
        if precision:
            self.precision = [[float(v) for v in row] for row in precision]
        self._config = cfg
        self._rng = rng or random.Random()

    # ------------------------------------------------------------------ api -
    def predict(self, features: CandidateFeatures) -> float:
        """``P(useful)`` for one candidate, in ``0 … 1``."""
        vector = features.vector()
        z = sum(w * x for w, x in zip(self.weights, vector))
        return _sigmoid(z)

    def std(self, features: CandidateFeatures) -> float:
        """The Laplace standard deviation of :meth:`predict` at *features*."""
        vector = features.vector()
        try:
            cov = _invert(self.precision)
        except ArithmeticError:
            return 0.0
        total = 0.0
        for i, x_i in enumerate(vector):
            row = cov[i]
            for j, x_j in enumerate(vector):
                total += x_i * row[j] * x_j
        if total <= 0.0:
            return 0.0
        p = self.predict(features)
        # Delta method on the logistic link: the spread in probability, not in
        # logits, which is the scale the multiplier is built on.
        return math.sqrt(total) * p * (1.0 - p)

    def multiplier(
        self, features: CandidateFeatures, explore: bool = False
    ) -> float:
        """
        The clamped multiplier this model asks for, ``1.0`` when untrained.

        With *explore* the prediction is drawn with its Laplace noise instead
        of taken at the mean, which is the cheap version of Thompson sampling
        for a Gaussian posterior and costs one normal deviate.
        """
        cfg = self._config
        if not self.trained:
            return 1.0
        p = self.predict(features)
        if explore:
            sigma = self.std(features)
            if sigma > 0.0:
                # The coefficient scales the Laplace deviate: ``0`` is the
                # mean with a coin flip nobody took, and it must not draw.
                p = _clamp(
                    p + self._rng.gauss(0.0, cfg.explore_coeff * sigma),
                    0.0,
                    1.0,
                )
        span = cfg.mult_max - cfg.mult_min
        return round(_clamp(cfg.mult_min + span * p, cfg.mult_min, cfg.mult_max), 4)

    @property
    def trained(self) -> bool:
        """Whether any weight moved off zero."""
        return any(abs(w) > 1e-9 for w in self.weights)

    def fit(
        self,
        samples: Sequence[FeedbackSample],
        holdout_fraction: float = 0.0,
        rng: Optional[random.Random] = None,
    ) -> Dict[str, Any]:
        """
        Fit on the labelled samples and report what the fit is worth.

        Samples with ``label == 0`` are exposures, not judgements, and are
        counted but not fitted: an unlabelled injection is exactly the case
        where the layer proved nothing, and treating it as a downvote would
        teach the model that being shown is bad.
        """
        cfg = self._config
        rng = rng or self._rng
        labelled = [s for s in samples if s.label and s.document_name]
        report: Dict[str, Any] = {
            "samples": len(samples),
            "labelled": len(labelled),
            "positive": sum(1 for s in labelled if s.label > 0),
            "holdout_auc": 0.0,
            "train_auc": 0.0,
            "collinear": (),
            "ridge": 0.0,
        }
        if not labelled:
            return report
        train, holdout = _split(labelled, holdout_fraction, rng)
        train = [_credit_kind(s) for s in train]
        holdout = [_credit_kind(s) for s in holdout]
        # How much evidence the fit rests on, weighted by kind.  The trust
        # gate below uses this: a corpus with thousands of labels that are all
        # coarse has less evidence than one with fewer counterfactual ones.
        evidence = sum(
            _KIND_CREDIT.get(s.label_kind, 1.0) for s in labelled
        )
        report["evidence"] = round(evidence, 2)
        varied, collinear = _varied_features(train)
        lam = _ridge(cfg.l2, len(train))
        report["collinear"] = collinear
        report["ridge"] = round(lam, 4)
        report["features"] = tuple(FEATURE_NAMES[i] for i in varied)
        weights = [0.0] * _N_FEATURES
        for _ in range(cfg.newton_steps):
            weights = _newton_step(
                weights, train, lam, cfg.weight_clip, varied
            )
        self.weights = weights
        self.precision = _hessian(train, lam, varied, weights)
        report["train_auc"] = round(_auc(train, self), 4)
        if holdout:
            report["holdout_auc"] = round(_auc(holdout, self), 4)
        else:
            report["holdout_auc"] = report["train_auc"]
        return report

    # ----------------------------------------------------------- serialization
    def to_dict(self) -> Dict[str, Any]:
        """The snapshot body."""
        return {
            "w": [round(w, 6) for w in self.weights],
            "diag": [round(self.precision[i][i], 6) for i in range(_N_FEATURES)],
        }

    @classmethod
    def from_dict(
        cls, raw: Dict[str, Any], config: Optional[LearningConfig] = None
    ) -> "ContextRanker":
        """Rebuild from a snapshot; a malformed one builds neutral."""
        cfg = config or LearningConfig()
        weights = raw.get("w") if isinstance(raw, dict) else None
        if not isinstance(weights, (list, tuple)) or len(weights) != _N_FEATURES:
            return cls(config=cfg)
        try:
            parsed = [float(w) for w in weights]
        except (TypeError, ValueError):
            return cls(config=cfg)
        ranker = cls(weights=parsed, config=cfg)
        diag = raw.get("diag") if isinstance(raw, dict) else None
        if isinstance(diag, (list, tuple)) and len(diag) == _N_FEATURES:
            try:
                values = [max(float(d), 1e-6) for d in diag]
            except (TypeError, ValueError):
                values = [cfg.l2] * _N_FEATURES
            ranker.precision = [
                [values[i] if i == j else 0.0 for j in range(_N_FEATURES)]
                for i in range(_N_FEATURES)
            ]
        return ranker


@dataclass(frozen=True)
class ModelSnapshot:
    """
    One published fit, with the evidence it was fitted on.

    The point of carrying ``samples``, ``holdout_auc`` and ``features`` is that
    an annotation can say *why* a document was first: not "the model said so"
    but "a fit on 1 412 labels from this repository, holding out at 0.71,
    moved it".  A snapshot that cannot say that is a snapshot nobody should
    ship, and :func:`train_ranker` refuses to.
    """

    scope: str
    model_id: str
    trained_at: float
    samples: int
    labelled: int
    positive: int
    holdout_auc: float
    features: Tuple[str, ...]
    ranker: ContextRanker
    arms: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def neutral(self) -> bool:
        """Whether reading this snapshot changes anything."""
        return not self.ranker.trained


def train_ranker(
    samples: Sequence[FeedbackSample],
    scope: str,
    config: Optional[LearningConfig] = None,
    holdout_fraction: float = 0.2,
    rng: Optional[random.Random] = None,
    now: float = 0.0,
    arms: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[ModelSnapshot, Dict[str, Any]]:
    """
    Fit one scope and decide whether the fit is worth publishing.

    The decision is the interesting part.  A scope with too few labels, or a
    fit that cannot separate held-out judgements better than a coin, publishes
    a **neutral** snapshot: weights at zero, which reads as ``1.0`` everywhere
    and behaves exactly like the layer does today.  Shipping a weak fit is
    worse than shipping none, because a weak fit still reorders the block and
    still costs the characters it reorders.
    """
    cfg = config or LearningConfig()
    rng = rng or random.Random(0)
    ranker = ContextRanker(config=cfg, rng=rng)
    report = ranker.fit(samples, holdout_fraction=holdout_fraction, rng=rng)
    enough = report["labelled"] >= cfg.min_samples
    separates = report["holdout_auc"] >= cfg.min_holdout_auc
    trusted = report.get("evidence", 0.0) >= cfg.min_evidence
    accepted = bool(enough and separates and trusted)
    if not accepted:
        if not enough:
            report["refused"] = "too_few_labels"
        elif not separates:
            report["refused"] = "holdout_auc_below_floor"
        else:
            report["refused"] = "evidence_below_threshold"
        published = ContextRanker(config=cfg, rng=rng)
    else:
        report["accepted"] = True
        published = ranker
    now = float(now or time.time())
    model_id = f"{scope}:{int(now)}:{report['labelled']}:{report['holdout_auc']:.3f}"
    snapshot = ModelSnapshot(
        scope=scope,
        model_id=model_id,
        trained_at=now,
        samples=int(report["samples"]),
        labelled=int(report["labelled"]),
        positive=int(report["positive"]),
        holdout_auc=float(report["holdout_auc"]),
        features=FEATURE_NAMES,
        ranker=published,
        # The bandit keeps its own live hash; carrying a copy into the
        # snapshot is not how the value reaches the request path, it is how
        # the value can be read back after the fact.  A snapshot that says
        # "the fit was this, and the sources were worth this" answers the
        # operator's question without a Redis read of a live counter that
        # has moved since.
        arms={str(k): dict(v) for k, v in (arms or {}).items()},
    )
    return snapshot, report


def publish_snapshot(
    client: Any,
    snapshot: ModelSnapshot,
    key_prefix: str = "agentic-kb",
) -> bool:
    """
    Write one snapshot to Redis.  Returns whether it is there.

    Fail-open in the only sense that matters for a write: a Redis that will not
    take it leaves the previous snapshot serving, and the caller keeps the
    layer static rather than partial.
    """
    payload = {
        "schema": 1,
        "scope": snapshot.scope,
        "model_id": snapshot.model_id,
        "trained_at": snapshot.trained_at,
        "samples": snapshot.samples,
        "labelled": snapshot.labelled,
        "positive": snapshot.positive,
        "holdout_auc": snapshot.holdout_auc,
        "features": list(snapshot.features),
        "arms": snapshot.arms,
        **snapshot.ranker.to_dict(),
    }
    key = f"{key_prefix}:model:{snapshot.scope}"
    try:
        client.set(key, json.dumps(payload, sort_keys=True))
        return True
    except Exception:
        return False


def load_snapshot(
    client: Any,
    scope: str,
    key_prefix: str = "agentic-kb",
    config: Optional[LearningConfig] = None,
    rng: Optional[random.Random] = None,
) -> Optional[ModelSnapshot]:
    """
    Read one scope's snapshot, refusing anything this build cannot interpret.

    A snapshot written by a later version, with features this one does not
    have, is refused rather than partially applied: half a weight vector
    reorders a block in a way nobody can explain.
    """
    cfg = config or LearningConfig()
    key = f"{key_prefix}:model:{scope}"
    try:
        raw = client.get(key)
    except Exception:
        return None
    if not raw:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    features = tuple(str(f) for f in data.get("features") or ())
    if features and features != FEATURE_NAMES:
        return None
    ranker = ContextRanker.from_dict(data, config=cfg)
    ranker._rng = rng or random.Random()  # noqa: SLF001 - same module
    return ModelSnapshot(
        scope=str(data.get("scope") or scope),
        model_id=str(data.get("model_id") or ""),
        trained_at=float(data.get("trained_at") or 0.0),
        samples=int(data.get("samples") or 0),
        labelled=int(data.get("labelled") or 0),
        positive=int(data.get("positive") or 0),
        holdout_auc=float(data.get("holdout_auc") or 0.0),
        features=features or FEATURE_NAMES,
        ranker=ranker,
        arms={
            str(k): dict(v) for k, v in (data.get("arms") or {}).items()
            if isinstance(v, dict)
        },
    )


class SourceBandit:
    """
    What a character of one source's budget is worth, per mode.

    The policy JSON says ``recipe`` in ``plan`` weighs ``0.6`` and ``doc``
    weighs ``0.6``; those numbers are guesses about a trade-off the layer
    measures every day.  The measurement is a reward per character: a document
    that earned a label brought ``label`` judgements and cost ``chars`` of the
    mode's budget, and the ratio is what a source is worth relative to the
    others competing for the same budget.

    Posterior per arm is Beta again, with the character cost averaged in, so
    an arm nobody has tried is served by the policy weight and an arm with a
    hundred judgements is served by its own history.
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        config: Optional[LearningConfig] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        """Bind the store; read nothing."""
        self._client = client
        self._config = config or LearningConfig()
        self._rng = rng or random.Random()
        self._local: Dict[str, Dict[str, float]] = {}
        self._lock = threading.Lock()

    def observe(self, mode: str, kind: str, label: float, chars: int) -> None:
        """Fold one settled document into its source arm."""
        arm = self._arm(mode, kind)
        if not arm or not label:
            return
        with self._lock:
            state = self._local.setdefault(
                arm, {"a": 1.0, "b": 1.0, "chars": 0.0, "n": 0.0}
            )
            if label > 0:
                state["a"] += float(label)
            else:
                state["b"] += float(-label)
            state["chars"] += max(float(chars), 0.0)
            state["n"] += 1.0
            snapshot = dict(state)
        if self._client is not None:
            key = self._key()
            try:
                for field_name, value in snapshot.items():
                    self._client.hset(key, f"{arm}:{field_name}", f"{value:.6f}")
            except Exception:
                pass

    def value_per_char(self, mode: str, kind: str, default: float = 1.0) -> float:
        """
        A sampled reward-per-character for one arm.

        *default* is what an untried arm returns, which is how the policy
        weight keeps serving until the arm has its own history.
        """
        arm = self._arm(mode, kind)
        state = self._read(arm) if arm else None
        cfg = self._config
        if not state or state.get("n", 0.0) < cfg.min_observations:
            return default
        try:
            draw = self._rng.betavariate(
                max(state.get("a", 1.0), 1e-6), max(state.get("b", 1.0), 1e-6)
            )
        except Exception:
            total = state.get("a", 1.0) + state.get("b", 1.0)
            draw = state.get("a", 1.0) / total if total else 0.5
        mean_chars = state.get("chars", 0.0) / state["n"] if state.get("n") else 0.0
        if mean_chars <= 0.0:
            return default
        # Normalised against a 500-character entry, the middle of what the
        # budget actually spends per item, so the number reads as a multiplier
        # and not as an inverse-character count.
        return round(_clamp(draw * 500.0 / mean_chars, 0.0, 4.0), 4)

    def arms(self) -> Dict[str, Dict[str, float]]:
        """Everything known, for a snapshot or a report."""
        with self._lock:
            return {arm: dict(state) for arm, state in self._local.items()}

    def summary(self) -> Dict[str, Dict[str, Any]]:
        """
        What each arm knows, in the two numbers an operator asks for.

        ``n`` is how many judgements the arm has folded and ``mean`` is the
        posterior mean of its reward: the value a reader can compare against
        the policy weight the JSON still ships.  The sampled value is not
        reported, because it is a draw and a draw is not a fact.
        """
        out: Dict[str, Dict[str, Any]] = {}
        for arm, state in self.arms().items():
            alpha = float(state.get("a", 1.0))
            beta = float(state.get("b", 1.0))
            total = alpha + beta
            observations = float(state.get("n", 0.0))
            mean_chars = (
                float(state.get("chars", 0.0)) / observations
                if observations
                else 0.0
            )
            out[arm] = {
                "n": int(observations),
                "mean": round(alpha / total, 4) if total else 0.5,
                "mean_chars": round(mean_chars, 1),
            }
        return out

    def _read(self, arm: str) -> Optional[Dict[str, float]]:
        """Local first; Redis fills a gap a fresh worker process has."""
        with self._lock:
            local = self._local.get(arm)
        if local is not None:
            return local
        if self._client is None:
            return None
        try:
            raw = self._client.hgetall(self._key())
        except Exception:
            return None
        if not raw:
            return None
        decoded = {
            (k.decode() if isinstance(k, bytes) else str(k)): (
                v.decode() if isinstance(v, bytes) else v
            )
            for k, v in dict(raw).items()
        }
        state: Dict[str, float] = {}
        prefix = f"{arm}:"
        for field_name, value in decoded.items():
            if not field_name.startswith(prefix):
                continue
            try:
                state[field_name[len(prefix):]] = float(value)
            except (TypeError, ValueError):
                continue
        if state:
            with self._lock:
                self._local[arm] = state
        return state or None

    @staticmethod
    def _arm(mode: str, kind: str) -> str:
        """The arm name; an unnamed source has no arm."""
        mode = (mode or "").strip()
        kind = (kind or "").strip()
        if not kind:
            return ""
        return f"{mode or '*'}|{kind}"

    def _key(self) -> str:
        """One hash for every arm of a deployment."""
        return f"{self._config.key_prefix}:arms"


class FeedbackLog:
    """
    The training stream: every injection, with its features, and its verdict.

    This is the part that cannot be recovered later.  A model can be fitted in
    an afternoon once the records exist; the records only exist if they were
    written while the traffic was happening.  So the log is written whether or
    not learning is enabled, and whether or not anything will ever read it —
    bounded by ``MAXLEN`` exactly like the event stream, and fail-open exactly
    like the publisher.

    The record carries features and names, never content: no text, no paths,
    no absolute locations.  The worker masks documents before indexing them,
    and a training log that kept the unmasked features would undo that from
    the other side.
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        stream_key: str = "agentic_kb:training",
        maxlen: int = 100000,
        local_maxlen: int = 1000,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Bind the stream; keep a local tail when Redis is absent."""
        self._client = client
        self._stream_key = stream_key
        self._maxlen = max(int(maxlen), 1)
        self._local_maxlen = max(int(local_maxlen), 1)
        self._logger = logger
        self._local: "List[FeedbackSample]" = []
        self._lock = threading.Lock()

    def append(self, sample: FeedbackSample) -> bool:
        """Write one settled sample; returns whether Redis took it."""
        payload = json.dumps(sample.to_json(), sort_keys=True)
        with self._lock:
            self._local.append(sample)
            if len(self._local) > self._local_maxlen:
                del self._local[: len(self._local) - self._local_maxlen]
        if self._client is None:
            return False
        try:
            self._client.xadd(
                self._stream_key, {"d": payload}, maxlen=self._maxlen
            )
            return True
        except Exception as exc:
            if self._logger is not None:
                self._logger.warning(
                    "training log write failed: %s", type(exc).__name__
                )
            return False

    def append_injection(
        self,
        scope: str,
        session_id: str,
        generation: str,
        model_id: str,
        features: Iterable[CandidateFeatures],
        document_names: Sequence[str],
    ) -> None:
        """
        Record what a request was shown, before it has a verdict.

        Unlabelled by design: ``label == 0`` until the next turn settles it,
        and a record that never gets settled stays at zero, which is the
        honest reading of an injection nobody proved anything about.
        """
        now = time.time()
        for item, name in zip(features, document_names):
            self.append(
                FeedbackSample(
                    scope=scope,
                    session_id=session_id,
                    generation=generation,
                    document_name=name,
                    features=item,
                    label=0.0,
                    label_kind="injection",
                    model_id=model_id,
                    at=now,
                )
            )

    def drain(
        self, count: int = 500, last_id: str = "0-0"
    ) -> Tuple[str, List[FeedbackSample]]:
        """Read a batch for the worker; returns the id to resume from."""
        if self._client is None:
            with self._lock:
                return last_id, list(self._local[-count:])
        try:
            rows = self._client.xrange(self._stream_key, min=last_id, count=count)
        except Exception as exc:
            if self._logger is not None:
                self._logger.warning(
                    "training log read failed: %s", type(exc).__name__
                )
            return last_id, []
        samples: List[FeedbackSample] = []
        seen = last_id
        for entry_id, fields in rows or ():
            seen = (
                entry_id.decode() if isinstance(entry_id, bytes) else str(entry_id)
            )
            if last_id and last_id != "0-0" and seen == last_id:
                # XRANGE includes its lower bound, so the id a previous pass
                # stopped at is handed back to it.  Skipping it is what lets a
                # caller resume by id without reading one record twice forever.
                continue
            raw = dict(fields or {})
            payload = raw.get(b"d") or raw.get("d")
            if payload is None:
                continue
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8", "replace")
            try:
                samples.append(FeedbackSample.from_json(json.loads(payload)))
            except (TypeError, ValueError):
                continue
        return seen, samples


class LabelStore:
    """
    The persistent, per-scope corpus of settled judgements.

    The training stream is bounded and lossy by design: it is a queue, and a
    queue is trimmed.  A model, though, is fitted on the *accumulation* of
    judgements a scope has made, and that accumulation outlives any one batch
    of the queue.  The store is that accumulation: one hash per scope, keyed
    by (session, generation, document), so that a re-injection of the same
    document in the same generation is a correction of one record rather than
    a duplicate of it, and a verdict always wins over the unlabelled exposure
    it settles.

    A corpus kept this way has three properties a trimmed queue cannot: it is
    deduplicated by construction, it survives the stream's ``MAXLEN``, and a
    pass of the learner that reads only the newest batch still fits on
    everything that ever settled, newest first.
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        key_prefix: str = "agentic-kb",
        ttl_seconds: float = 0.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Bind the store; keep a local copy when Redis is absent."""
        self._client = client
        self._key_prefix = key_prefix
        self._ttl = max(float(ttl_seconds), 0.0)
        self._logger = logger
        self._local: Dict[Tuple[str, str], Dict[Tuple[str, str, str], str]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ api -
    def merge(self, samples: Sequence[FeedbackSample]) -> int:
        """
        Fold settled samples into the store; return how many records changed.

        Only records that carry a verdict change anything: an unlabelled
        exposure is the state a document is in *before* it is judged, and
        overwriting a judgement with it would be forgetting.  A second verdict
        on the same (session, generation, document) replaces the first, which
        is how the store stays a corpus rather than becoming a log.
        """
        by_scope: Dict[str, Dict[str, str]] = {}
        for sample in samples:
            if not sample.scope or not sample.document_name or not sample.label:
                continue
            by_scope.setdefault(self._key(sample.scope), {})[
                self._field(sample)
            ] = json.dumps(sample.to_json(), sort_keys=True)
        if not by_scope:
            return 0

        changed = 0
        with self._lock:
            for key, fields in by_scope.items():
                local = self._local.setdefault(key, {})
                for field_name, payload in fields.items():
                    if local.get(field_name) != payload:
                        local[field_name] = payload
                        changed += 1
        # Nothing changed means this process already holds every one of these
        # records, and a fresh process always sees them as new, so an unchanged
        # batch is one that is already in Redis.
        if changed and self._client is not None:
            try:
                for key, fields in by_scope.items():
                    self._client.hset(key, mapping=fields)
                    if self._ttl > 0.0:
                        self._client.expire(key, int(self._ttl))
            except Exception as exc:
                self._warn("label store write failed: %s", type(exc).__name__)
        return changed

    def load(self, scope: str, limit: int = 10000) -> List[FeedbackSample]:
        """
        The settled corpus of one scope, newest first, bounded by *limit*.

        A record that cannot be read back is dropped, not raised on: a corpus
        that partially fails to load is still a corpus, and a fit that reads
        less than it has is safer than one that fails outright.
        """
        key = self._key(scope)
        raw: Optional[Dict[str, str]] = None
        with self._lock:
            if key in self._local:
                raw = dict(self._local[key])
        if raw is None and self._client is not None:
            try:
                stored = self._client.hgetall(key)
            except Exception as exc:
                if self._logger is not None:
                    self._logger.warning(
                        "label store read failed: %s", type(exc).__name__
                    )
                return []
            raw = {}
            for field_name, value in dict(stored or {}).items():
                if isinstance(field_name, bytes):
                    field_name = field_name.decode()
                if isinstance(value, bytes):
                    value = value.decode()
                raw[str(field_name)] = str(value)
        rows: List[FeedbackSample] = []
        for field_name, value in raw.items():
            try:
                rows.append(FeedbackSample.from_json(json.loads(value)))
            except (TypeError, ValueError):
                continue
        rows.sort(key=lambda s: s.at, reverse=True)
        return rows[: max(int(limit), 0)]

    # -------------------------------------------------------------- internals -
    def _key(self, scope: str) -> str:
        """One hash per scope, next to the snapshots the learner publishes."""
        return f"{self._key_prefix}:labels:{scope}"

    @staticmethod
    def _field(sample: FeedbackSample) -> str:
        """The deduplication key: one record per (session, generation, doc)."""
        return f"{sample.session_id}|{sample.generation}|{sample.document_name}"


class LearningPolicy:
    """
    The one object the request path talks to.

    It answers a single question — "how much should this candidate's vote move?"
    — from three places, in increasing order of how much evidence they need:
    the document's own reputation, the scope's fitted model, and the source
    arm's value per character.  All three are clamped to the same range the
    policy already allows, and all three return ``1.0`` when they do not know,
    so the layer with learning off, with Redis down, with a refused snapshot or
    with a corpus too small to fit is *the layer as it behaves today*.

    Reading a snapshot costs one ``GET`` with a local TTL cache; reading
    reputation costs nothing after the first read of a document in a process.
    There is no fitting, no matrix work and no network call per candidate.
    """

    def __init__(
        self,
        config: Optional[LearningConfig] = None,
        client: Optional[Any] = None,
        rng: Optional[random.Random] = None,
        time_fn: Any = time.monotonic,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Build the readers; open no connection."""
        self._config = config or LearningConfig()
        self._client = client
        self._rng = rng or random.Random()
        self._time = time_fn
        self._logger = logger
        self._reputation = DocReputation(
            client=client, config=self._config, rng=self._rng, logger=logger
        )
        self._bandit = SourceBandit(
            client=client, config=self._config, rng=self._rng
        )
        self._log = FeedbackLog(
            client=client,
            stream_key=self._config.stream_key,
            logger=logger,
        )
        self._snapshots: Dict[str, Tuple[float, Optional[ModelSnapshot]]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ api -
    @property
    def enabled(self) -> bool:
        """Whether learning moves anything at all."""
        return bool(self._config.enabled)

    @property
    def log(self) -> FeedbackLog:
        """The training log, which the plugin writes to on every injection."""
        return self._log

    @property
    def reputation(self) -> DocReputation:
        """The online reputation store."""
        return self._reputation

    @property
    def bandit(self) -> SourceBandit:
        """The per-source value learner."""
        return self._bandit

    def multiplier_for(
        self,
        features: CandidateFeatures,
        document_name: str = "",
        scope: str = "",
    ) -> Tuple[float, Dict[str, Any]]:
        """
        The multiplier and the reasons for it.

        Returns ``(1.0, {"reason": "disabled"})`` on a switched-off policy and
        the composition otherwise.  The second value exists for the annotation:
        "why was this recipe first" has to be answerable from the response, not
        from a log nobody kept.

        ``1.0`` here means "nothing of mine moved this", which is right for a
        caller that multiplies nothing else.  It is **not** what the read path
        relies on: the assembler does not ask this method at all while the
        switch is off, because the vote it is replacing already carries the
        static counter multiplier, and a disabled learner must leave that
        counter in place rather than overwrite it with a one.  Anything that
        consults a policy directly has to make the same difference — read
        :attr:`enabled` first, and treat a disabled policy as absent.
        """
        cfg = self._config
        if not cfg.enabled:
            return 1.0, {"reason": "disabled"}
        reputation = self._reputation.multiplier(document_name)
        snapshot = self._snapshot(scope)
        explore = snapshot is not None and self._explore_now()
        model = (
            snapshot.ranker.multiplier(features, explore=explore)
            if snapshot is not None
            else 1.0
        )
        arm = self._bandit.value_per_char(
            features.mode, features.kind, default=1.0
        )
        product = reputation * model * arm
        clamped = round(
            _clamp(product, cfg.mult_min, cfg.mult_max), 4
        )
        detail = {
            "reason": "learned" if snapshot is not None else "reputation",
            "rep": reputation,
            "model": model,
            "arm": arm,
            "explore": bool(explore),
            "model_id": snapshot.model_id if snapshot is not None else "",
            "range": [cfg.mult_min, cfg.mult_max],
        }
        return clamped, detail

    def settle(self, sample: FeedbackSample) -> None:
        """
        Fold one verdict into every learner and log it.

        Called from the settle path with the graded label of §3 of the plan;
        the coarse label the layer computes today works too, marked
        ``label_kind="coarse"``, and trains the same counters.
        """
        if not sample.document_name:
            return
        try:
            self._reputation.observe(sample.document_name, sample.label)
            self._bandit.observe(
                sample.features.mode,
                sample.features.kind,
                sample.label,
                sample.features.size_chars,
            )
            self._log.append(sample)
        except Exception as exc:  # a learner must never cost a request
            if self._logger is not None:
                self._logger.warning(
                    "learning settle failed: %s", type(exc).__name__
                )

    def annotation(self, scope: str = "") -> Dict[str, Any]:
        """
        What the request annotation should carry, without content.

        A switched-off learner reports itself and reads nothing: the request
        path is not to pay a snapshot read for a model it is not allowed to
        apply, and an operator reading the annotation of a disabled deployment
        should see the switch, not a stale model id next to ``enabled: false``.
        """
        cfg = self._config
        if not cfg.enabled:
            return {
                "enabled": False,
                "model_id": "",
                "holdout_auc": 0.0,
                "labelled": 0,
                "range": [cfg.mult_min, cfg.mult_max],
            }
        snapshot = self._snapshot(scope)
        detail = {
            "enabled": cfg.enabled,
            "model_id": snapshot.model_id if snapshot is not None else "",
            "holdout_auc": snapshot.holdout_auc if snapshot is not None else 0.0,
            "labelled": snapshot.labelled if snapshot is not None else 0,
            "range": [cfg.mult_min, cfg.mult_max],
        }
        arms = self._bandit.summary()
        if arms:
            detail["arms"] = arms
        return detail

    # ------------------------------------------------------------ internals -
    def _snapshot(self, scope: str) -> Optional[ModelSnapshot]:
        """One ``GET`` per scope per TTL; never raise, never block a request."""
        if not scope or self._client is None:
            return None
        now = self._time()
        with self._lock:
            cached = self._snapshots.get(scope)
        if cached is not None and now - cached[0] < self._config.model_ttl_seconds:
            return cached[1]
        snapshot = load_snapshot(
            self._client,
            scope,
            key_prefix=self._config.key_prefix,
            config=self._config,
            rng=self._rng,
        )
        with self._lock:
            self._snapshots[scope] = (now, snapshot)
        return snapshot

    def _explore_now(self) -> bool:
        """Whether this request samples instead of taking the mean."""
        rate = self._config.explore_rate
        if rate <= 0.0:
            return False
        if rate >= 1.0:
            return True
        return self._rng.random() < rate


# --------------------------------------------------------------- fitting ----
def _split(
    samples: Sequence[FeedbackSample], fraction: float, rng: random.Random
) -> Tuple[List[FeedbackSample], List[FeedbackSample]]:
    """
    Split by session: all injections from one session are train or holdout.

    Random sample-level splits leak information when multiple injections from
    the same session appear in the corpus (which they do, routinely).  Splitting
    by session ensures the holdout measures generalisation to *new situations*,
    not just new documents in familiar situations.
    """
    ordered = list(samples)
    if not ordered or fraction <= 0.0:
        return ordered, []

    # Group by session
    by_session: Dict[str, List[FeedbackSample]] = {}
    for sample in ordered:
        by_session.setdefault(sample.session_id, []).append(sample)

    # One session has nothing to generalise across: holding it out against
    # itself would train on nothing and score a model on the data it is
    # supposed to measure, so it all stays in the fit.
    if len(by_session) < 2:
        return ordered, []

    # Shuffle sessions (not samples)
    sessions = list(by_session.keys())
    rng.shuffle(sessions)

    # Split sessions by count, keeping at least one on each side.
    n_holdout = min(
        max(1, int(len(sessions) * fraction)), len(sessions) - 1
    )
    holdout_sessions = sessions[:n_holdout]
    train_sessions = sessions[n_holdout:]

    holdout = []
    train = []
    for session in holdout_sessions:
        holdout.extend(by_session[session])
    for session in train_sessions:
        train.extend(by_session[session])

    return train, holdout


#: How much a label of each kind is worth in the fit.
#:
#: ``counterfactual`` is a verdict with evidence behind it: the next turn
#: mined a fix, and the documents that could have caused it are the ones
#: credited.  ``coarse`` is the older rule that credits everything that was
#: on screen, which is true as far as it goes but attributes a fix to the
#: whole block rather than to the document that earned it.  The two are not
#: the same strength of evidence, and a corpus that mixes them should not
#: weigh them equally.
#:
#: The weight is applied to the magnitude of the label, which :func:`_target`
#: already reads as an evidence weight, so the direction of a verdict is
#: untouched and only how much it counts changes.
_KIND_CREDIT: Dict[str, float] = {
    "counterfactual": 1.0,
    "coarse": 0.5,
}


def _credit_kind(sample: "FeedbackSample") -> "FeedbackSample":
    """
    The sample with its label scaled by how much its kind of evidence is worth.

    An unknown kind is credited fully: a new kind of label is a reason to look
    at the weighting, not a reason to silently discard the labels.
    """
    credit = _KIND_CREDIT.get(sample.label_kind)
    if credit is None or credit == 1.0:
        return sample
    return replace(sample, label=float(sample.label) * credit)


def _target(label: float) -> Tuple[float, float]:
    """
    One graded verdict as a target direction and an evidence weight.

    The layer's labels are graded and signed — ``1.0`` for a confirmed fix,
    ``0.5`` for a plausible one, ``-1.0`` for a debunked recipe — and the
    honest reading of those is the *direction* of the verdict, observed with
    weight ``|label|``: a plausible fix is half an observation of success, a
    debunked recipe a whole observation of failure, and a document that merely
    rode along a third of an observation against it.

    The weight is what says how sure the verdict is; the target stays binary
    because the likelihood is Bernoulli's.  Folding the signed label into the
    target instead would be worse than wrong, it would be unbounded:
    ``-[y log p + (1-y) log(1-p)]`` with ``y = -1`` keeps falling as ``p``
    falls, so the objective has no optimum and the fit runs to whatever clip
    it hits.  Reading a graded label as a middling target — ``(label + 1) / 2``,
    or the label itself — fails the other way, and more quietly: a target of
    ``0.5`` is an observation that ``p = 0.5``, the one thing a corpus can
    state without teaching anything, yet the verdict still counts as labelled
    when the fit is gated.  A corpus of such verdicts then clears the holdout
    and publishes a model that learned nothing, which is the failure the gate
    exists to catch.
    """
    label = float(label)
    if label >= 0.0:
        return 1.0, label
    return 0.0, -label


def _ridge(lambda_per_100: float, count: int) -> float:
    """
    The prior precision for a corpus of *count* labels.

    Scaled with the corpus on purpose.  An absolute ridge is a belief that gets
    outvoted by data it should be tempering: ``lambda = 8`` against 800 samples
    is a rounding error, and a perfectly separating feature then drags every
    correlated weight to the clip with it.  Per-100 keeps the prior the same
    strength of belief at any sample size.
    """
    return max(lambda_per_100, 1e-6) * max(count, 1) / 100.0


def _varied_features(
    samples: Sequence[FeedbackSample],
) -> Tuple[List[int], Tuple[str, ...]]:
    """
    The identifiable features of a corpus, and the ones it cannot identify.

    Two things make a weight meaningless, and both are ordinary in this data:
    a feature the corpus never varied — every document of one repository is a
    ``recipe``, so ``kind`` never moves — and a feature another feature already
    determines.  ``score`` and ``score²`` are perfectly correlated on a corpus
    whose scores are all in one range, and the two scale dummies are collinear
    with the bias when every hit came back on the same scale.  Fitting either
    pair splits one effect between two weights whose individual values are
    arbitrary, so the pair is pruned to its first member in
    :data:`_LEARNABLE` order and the pruning is reported, not silent.
    """
    if not samples:
        return [], ()
    vectors = [sample.features.vector() for sample in samples]
    present = [
        index
        for index in _LEARNABLE
        if any(abs(vector[index] - vectors[0][index]) > 1e-12 for vector in vectors)
    ]
    kept: List[int] = []
    dropped: List[str] = []
    for index in present:
        column = [vector[index] for vector in vectors]
        collinear = any(
            abs(_correlation(column, [vector[other] for vector in vectors]))
            >= _COLLINEAR_MAX
            for other in kept
        )
        if collinear:
            dropped.append(FEATURE_NAMES[index])
            continue
        kept.append(index)
    return kept, tuple(dropped)


def _correlation(left: Sequence[float], right: Sequence[float]) -> float:
    """Pearson's r of two columns; 0 when either does not vary."""
    count = len(left)
    if not count:
        return 0.0
    mean_left = sum(left) / count
    mean_right = sum(right) / count
    numerator = 0.0
    square_left = 0.0
    square_right = 0.0
    for x, y in zip(left, right):
        dx = x - mean_left
        dy = y - mean_right
        numerator += dx * dy
        square_left += dx * dx
        square_right += dy * dy
    if square_left <= 0.0 or square_right <= 0.0:
        return 0.0
    return numerator / math.sqrt(square_left * square_right)


def _prior_precision(lam: float) -> List[List[float]]:
    """``lambda I`` over every feature, including the ones that will not move."""
    return [
        [lam if i == j else 0.0 for j in range(_N_FEATURES)]
        for i in range(_N_FEATURES)
    ]


def _objective(
    weights: Sequence[float],
    samples: Sequence[FeedbackSample],
    lam: float,
    varied: Sequence[int],
) -> float:
    """The penalized log-likelihood the Newton step ascends."""
    total = -0.5 * lam * sum(weights[i] * weights[i] for i in varied)
    for sample in samples:
        target, weight = _target(sample.label)
        vector = sample.features.vector()
        z = sum(weights[i] * vector[i] for i in varied)
        p = _clamp(_sigmoid(z), 1e-9, 1.0 - 1e-9)
        total += weight * (target * math.log(p) + (1.0 - target) * math.log1p(-p))
    return total


def _newton_step(
    weights: Sequence[float],
    samples: Sequence[FeedbackSample],
    lam: float,
    clip: float,
    varied: Sequence[int],
) -> List[float]:
    """
    One damped ridge-Newton update over the identifiable features.

    Damped, because the undamped step is exactly what breaks here: a corpus
    whose labels a single feature separates perfectly has its optimum at
    infinity, and one full Newton step from zero throws every weight to the clip
    — including the weights of features that carry no signal at all, which then
    stay clipped and reorder the block by accident.  The backtrack keeps the
    penalized log-likelihood rising, so the separating feature grows toward its
    clip while a noise feature stops where its own evidence stops.
    """
    if not varied:
        return list(weights)
    gradient = [-lam * weights[i] for i in range(_N_FEATURES)]
    hessian = _prior_precision(lam)
    for sample in samples:
        target, weight = _target(sample.label)
        vector = sample.features.vector()
        z = sum(weights[i] * vector[i] for i in varied)
        p = _sigmoid(z)
        residual = weight * (target - p)
        curvature = weight * p * (1.0 - p)
        for i in varied:
            gradient[i] += residual * vector[i]
            row = hessian[i]
            term = curvature * vector[i]
            for j in varied:
                row[j] += term * vector[j]
    for i in varied:
        hessian[i][i] += 1e-9
    try:
        delta = _solve(hessian, gradient)
    except ArithmeticError:
        return list(weights)
    current = _objective(weights, samples, lam, varied)
    step = 1.0
    best = list(weights)
    best_value = current
    for _ in range(_BACKTRACK_LIMIT):
        candidate = list(weights)
        for index in varied:
            candidate[index] = _clamp(
                weights[index] + step * delta[index], -clip, clip
            )
        value = _objective(candidate, samples, lam, varied)
        if value > best_value:
            best, best_value = candidate, value
        if value > current + 1e-12:
            return candidate
        step *= 0.5
    return best if best_value > current else list(weights)


def _hessian(
    samples: Sequence[FeedbackSample],
    lam: float,
    varied: Sequence[int],
    weights: Sequence[float],
) -> List[List[float]]:
    """
    ``X' W X + lambda I`` at the fitted weights: the Laplace precision.

    This is the number the exploration bonus is built from, so it is computed
    with the true curvature ``p (1 - p)`` at the optimum rather than its ``0.25``
    bound.  The bound would be the safe-looking choice and the wrong one: on a
    corpus a feature separates cleanly, most samples sit deep in the saturated
    tail where the real curvature is near zero, and charging ``0.25`` for them
    reports a posterior far narrower than the evidence supports — the model would
    then stop exploring a repository it has barely judged.
    """
    matrix = _prior_precision(lam)
    for sample in samples:
        _, weight = _target(sample.label)
        vector = sample.features.vector()
        z = sum(weights[i] * vector[i] for i in varied)
        p = _sigmoid(z)
        curvature = weight * p * (1.0 - p)
        for i in varied:
            row = matrix[i]
            term = curvature * vector[i]
            for j in varied:
                row[j] += term * vector[j]
    return matrix


def _solve(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> List[float]:
    """Gaussian elimination with partial pivoting; raises on a singular matrix."""
    size = len(rhs)
    rows = [
        [float(matrix[i][j]) for j in range(size)] + [float(rhs[i])]
        for i in range(size)
    ]
    for column in range(size):
        pivot = max(range(column, size), key=lambda r: abs(rows[r][column]))
        if abs(rows[pivot][column]) < 1e-12:
            raise ArithmeticError("singular system")
        rows[column], rows[pivot] = rows[pivot], rows[column]
        pivot_value = rows[column][column]
        for row in range(column + 1, size):
            factor = rows[row][column] / pivot_value
            if factor == 0.0:
                continue
            for k in range(column, size + 1):
                rows[row][k] -= factor * rows[column][k]
    solution = [0.0] * size
    for row in reversed(range(size)):
        total = rows[row][size]
        for k in range(row + 1, size):
            total -= rows[row][k] * solution[k]
        solution[row] = total / rows[row][row]
    return solution


def _invert(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    """Invert by solving against the identity; raises when singular."""
    size = len(matrix)
    columns: List[List[float]] = []
    for index in range(size):
        unit = [1.0 if i == index else 0.0 for i in range(size)]
        columns.append(_solve(matrix, unit))
    return [[columns[j][i] for j in range(size)] for i in range(size)]


def _auc(samples: Sequence[FeedbackSample], ranker: ContextRanker) -> float:
    """Rank AUC with ties at 0.5, over the positive/negative split."""
    scored = [
        (ranker.predict(s.features), 1.0 if s.label > 0 else 0.0)
        for s in samples
        if s.label
    ]
    positives = sum(1 for _, y in scored if y > 0)
    negatives = len(scored) - positives
    if not positives or not negatives:
        return 0.5
    ordered = sorted(scored, key=lambda item: item[0])
    rank_total = 0.0
    index = 0
    while index < len(ordered):
        tie_end = index
        while (
            tie_end + 1 < len(ordered)
            and abs(ordered[tie_end + 1][0] - ordered[index][0]) < 1e-12
        ):
            tie_end += 1
        average_rank = (index + tie_end) / 2.0 + 1.0
        for position in range(index, tie_end + 1):
            if ordered[position][1] > 0:
                rank_total += average_rank
        index = tie_end + 1
    numerator = rank_total - positives * (positives + 1) / 2.0
    return _clamp(numerator / (positives * negatives), 0.0, 1.0)
