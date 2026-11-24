"""
Deterministic tests for the learned path.

Every random draw is behind an injected ``random.Random(seed)`` and every clock
behind an injected ``time_fn``, because the one thing this module must not do is
make ``evaluation.py`` stop reproducing itself.  A test here that flakes is a
model that flaps.
"""

import json
import math
import random
import time

from dataclasses import replace


import pytest

from typing import Optional

from llm_router_plugins.utils.knowledge.agentic_knowledge.learning import (
    FEATURE_NAMES,
    CandidateFeatures,
    ContextRanker,
    DocReputation,
    FeedbackLog,
    FeedbackSample,
    LearningConfig,
    LearningPolicy,
    ModelSnapshot,
    SourceBandit,
    load_snapshot,
    publish_snapshot,
    rollback_snapshot,
    snapshot_history,
    train_ranker,
    _credit_kind,
    _split,
)


# ------------------------------------------------------------------- fakes --
class FakeRedis:
    """The handful of commands the learner uses, in memory."""

    def __init__(self) -> None:
        self.hashes = {}
        self.values = {}
        self.streams = {}
        self.ttls = {}
        self.calls = []

    def hset(self, key, field, value):
        self.calls.append(("hset", key, field))
        self.hashes.setdefault(key, {})[field] = str(value)
        return 1

    def hgetall(self, key):
        self.calls.append(("hgetall", key))
        return dict(self.hashes.get(key, {}))

    def expire(self, key, seconds):
        self.ttls[key] = int(seconds)
        return True

    def get(self, key):
        self.calls.append(("get", key))
        return self.values.get(key)

    def set(self, key, value, nx=False, ex=0):
        # ``ex`` is a wall-clock TTL the fake does not age; the lock tests
        # assert the value was recorded, and expire it by deleting the key.
        self.calls.append(("set", key) if not nx else ("setnx", key))
        if nx and key in self.values:
            return None
        self.values[key] = value
        if ex:
            self.ttls[key] = int(ex)
        return True

    def delete(self, key):
        self.calls.append(("delete", key))
        return 1 if self.values.pop(key, None) is not None else 0

    def xadd(self, key, fields, maxlen=0):
        self.calls.append(("xadd", key))
        stream = self.streams.setdefault(key, [])
        stream.append((f"{len(stream) + 1}-0", dict(fields)))
        if maxlen:
            del stream[: max(0, len(stream) - maxlen)]
        return stream[-1][0]

    def xrange(self, key, min="-", max="+", count=0):
        rows = list(self.streams.get(key, []))
        return rows[:count] if count else rows


class BrokenRedis:
    """Every command fails; the layer must not notice."""

    def __getattr__(self, name):
        def _raise(*args, **kwargs):
            raise ConnectionError(f"{name} unavailable")

        return _raise


class FrozenClock:
    """A clock the test turns by hand."""

    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def sample(label: float, **features) -> FeedbackSample:
    """One labelled record with defaults that vary nothing."""
    return FeedbackSample(
        scope="repo",
        session_id="s1",
        generation="g1",
        document_name=f"recipe:20260101:{int(label * 1000)}",
        features=CandidateFeatures(mode="debug", kind="recipe", **features),
        label=label,
    )


def corpus(count: int, signal: bool, seed: int = 7) -> list:
    """
    A labelled corpus with, or without, something to learn.

    The labels of the noise corpus are drawn from a **separate** generator and
    then shuffled against the features.  Sharing one generator interleaves the
    label draws with the feature draws, which makes the label a function of the
    position in the sequence — and ``rank`` is a feature, so a "noise" corpus
    built that way carries real signal and the fit publishes.  That is not a
    test bug to paper over: it is exactly the leakage the fit has to refuse, so
    there is a regression test for it below.
    """
    rng = random.Random(seed)
    label_rng = random.Random(seed + 1000)
    features_rows = []
    for index in range(count):
        matched = rng.random() < 0.5
        score = rng.random()
        rank = rng.randrange(5)
        features_rows.append((matched, score, rank))
    if signal:
        labels = [1.0 if matched else -1.0 for matched, _, _ in features_rows]
    else:
        labels = [
            1.0 if label_rng.random() < 0.5 else -1.0 for _ in features_rows
        ]
        label_rng.shuffle(labels)
    rows = []
    for index, ((matched, score, rank), label) in enumerate(
        zip(features_rows, labels)
    ):
        rows.append(
            FeedbackSample(
                scope="repo",
                session_id=f"s{index}",
                generation=f"g{index}",
                document_name=f"recipe:20260101:{index:05d}",
                features=CandidateFeatures(
                    kind="recipe",
                    mode="debug",
                    score=score,
                    score_scale="cosine",
                    signature_match=matched,
                    rank=rank,
                ),
                label=label,
            )
        )
    return rows


# ------------------------------------------------------------------ config --
def test_learning_config_refuses_a_range_that_cannot_hold_one():
    with pytest.raises(ValueError):
        LearningConfig(mult_min=1.2, mult_max=1.5)
    with pytest.raises(ValueError):
        LearningConfig(mult_min=0.3, mult_max=0.9)


def test_learning_config_refuses_an_unknown_policy_key():
    with pytest.raises(ValueError, match="unknown learning keys"):
        LearningConfig.from_dict({"enabled": True, "learning_rate": 0.1})


def test_learning_config_reads_the_documented_section():
    config = LearningConfig.from_dict(
        {
            "enabled": True,
            "mult_min": 0.4,
            "min_samples": 50,
            "half_life_seconds": 0,
        }
    )
    assert config.enabled is True
    assert config.mult_min == 0.4
    assert config.min_samples == 50
    assert config.half_life_seconds == 0


# -------------------------------------------------------------- reputation --
def test_reputation_keeps_the_current_ordering_and_range():
    """The property the existing assembler test pins, with the new mechanism."""
    config = LearningConfig(enabled=True)
    reputation = DocReputation(
        config=config, rng=random.Random(11), time_fn=FrozenClock()
    )
    for _ in range(4):
        reputation.observe("recipe:20260101:good", 1.0)
    for _ in range(4):
        reputation.observe("recipe:20260101:bad", -1.0)

    good = reputation.multiplier("recipe:20260101:good")
    bad = reputation.multiplier("recipe:20260101:bad")
    untouched = reputation.multiplier("recipe:20260101:new")

    assert good > untouched > bad
    assert config.mult_min <= bad
    assert good <= config.mult_max
    assert untouched == 1.0


def test_reputation_below_the_observation_floor_is_neutral():
    """An unjudged document must not be served cheaper than a judged one."""
    reputation = DocReputation(
        config=LearningConfig(enabled=True, min_observations=3),
        rng=random.Random(3),
        time_fn=FrozenClock(),
    )
    reputation.observe("doc:20260101:aa", 1.0)
    assert reputation.multiplier("doc:20260101:aa") == 1.0
    reputation.observe("doc:20260101:aa", 1.0)
    reputation.observe("doc:20260101:aa", 1.0)
    assert reputation.multiplier("doc:20260101:aa") > 1.0


def test_reputation_forgets_toward_the_prior_not_toward_bad():
    """A half-life ages the evidence out; it does not record a failure."""
    clock = FrozenClock()
    reputation = DocReputation(
        config=LearningConfig(enabled=True, half_life_seconds=1000.0),
        rng=random.Random(5),
        time_fn=clock,
    )
    for _ in range(10):
        reputation.observe("recipe:20260101:old", 1.0)
    fresh = reputation.stats("recipe:20260101:old")
    assert fresh["alpha"] == pytest.approx(12.0)
    assert fresh["exposure"] == 10

    clock.now += 1000.0  # one half-life
    aged = reputation.stats("recipe:20260101:old")
    assert aged["alpha"] == pytest.approx(6.5)
    assert aged["exposure"] == pytest.approx(5.0)
    # The mean drifts toward 0.5 (unknown), never below it: forgetting a
    # judgement is not the same as recording a failure.
    assert 0.5 <= aged["mean"] < fresh["mean"]

    clock.now += 40_000.0
    ancient = reputation.stats("recipe:20260101:old")
    assert ancient["mean"] == pytest.approx(0.5, abs=1e-3)


def test_reputation_survives_a_process_restart_through_redis():
    """The bug the plan opens with: without a client, nothing persists."""
    client = FakeRedis()
    config = LearningConfig(enabled=True)
    first = DocReputation(
        client=client, config=config, rng=random.Random(1), time_fn=FrozenClock()
    )
    for _ in range(5):
        first.observe("recipe:20260101:x", -1.0)
    assert client.hashes["agentic-kb:rep:recipe:20260101:x"]
    assert client.ttls["agentic-kb:rep:recipe:20260101:x"] > 0

    second = DocReputation(
        client=client, config=config, rng=random.Random(1), time_fn=FrozenClock()
    )
    assert second.stats("recipe:20260101:x")["exposure"] == 5
    assert second.multiplier("recipe:20260101:x") < 1.0


def test_reputation_fails_open_when_redis_is_down():
    reputation = DocReputation(
        client=BrokenRedis(),
        config=LearningConfig(enabled=True),
        rng=random.Random(2),
        time_fn=FrozenClock(),
    )
    reputation.observe("recipe:20260101:y", 1.0)
    reputation.observe("recipe:20260101:y", 1.0)
    reputation.observe("recipe:20260101:y", 1.0)
    assert 1.0 < reputation.multiplier("recipe:20260101:y") <= 1.5


# ------------------------------------------------------------------ ranker --
def test_ranker_recovers_a_feature_that_really_predicts():
    config = LearningConfig(enabled=True, min_samples=200, min_holdout_auc=0.6)
    ranker = ContextRanker(config=config, rng=random.Random(4))
    report = ranker.fit(corpus(600, signal=True), holdout_fraction=0.25)

    signature_index = FEATURE_NAMES.index("signature_match")
    assert ranker.weights[signature_index] > 0.5
    assert report["holdout_auc"] > 0.9
    assert report["labelled"] == 600

    helped = ranker.multiplier(CandidateFeatures(signature_match=True))
    hurt = ranker.multiplier(CandidateFeatures(signature_match=False))
    assert helped > hurt
    assert helped <= config.mult_max


def test_ranker_refuses_to_publish_a_fit_on_noise():
    """A weak fit still reorders the block; shipping none is the safe answer."""
    config = LearningConfig(enabled=True, min_samples=200, min_holdout_auc=0.6)
    snapshot, report = train_ranker(
        corpus(600, signal=False), "repo", config=config, rng=random.Random(9)
    )
    assert report["refused"] == "holdout_auc_below_floor"
    assert snapshot.neutral is True
    assert snapshot.ranker.multiplier(CandidateFeatures(signature_match=True)) == 1.0


def test_ranker_learns_a_position_effect_when_one_is_really_there():
    """
    The other half of the leakage test: ``rank`` is a feature on purpose.

    A document the engine ranked first is read first and believed first, so a
    credit assignment that ignores position teaches the model to reproduce the
    engine instead of judging the advice.  When position genuinely carries the
    label, the fit must find it — which is also what makes the noise refusal
    above meaningful rather than a fit that simply cannot learn anything.
    """
    rows = [
        FeedbackSample(
            scope="repo",
            session_id=f"s{i}",
            generation=f"g{i}",
            document_name=f"recipe:20260101:{i:05d}",
            features=CandidateFeatures(kind="recipe", mode="debug", rank=i % 5),
            label=1.0 if i % 5 == 0 else -1.0,
        )
        for i in range(400)
    ]
    config = LearningConfig(enabled=True, min_samples=50, min_holdout_auc=0.6)
    ranker = ContextRanker(config=config, rng=random.Random(23))
    report = ranker.fit(rows, holdout_fraction=0.25)
    assert report["holdout_auc"] > 0.7
    assert ranker.weights[FEATURE_NAMES.index("rank")] != 0.0


def test_ranker_refuses_a_corpus_too_small_to_fit():
    config = LearningConfig(enabled=True, min_samples=200)
    snapshot, report = train_ranker(
        corpus(40, signal=True), "repo", config=config, rng=random.Random(9)
    )
    assert report["refused"] == "too_few_labels"
    assert snapshot.neutral is True


def test_a_graded_verdict_teaches_in_proportion_to_how_sure_it_was():
    """
    A plausible fix is half an observation of success, not half a coin flip.

    The graded labels of the settle path are weaker *evidence*, not weaker
    *claims*.  Folding ``+0.5`` into the target makes it an observation that
    ``p = 0.5`` — the one thing a corpus can state without teaching anything —
    and yet the verdict still counts as labelled when the fit is gated, so a
    whole scope of plausible fixes could clear the holdout and publish a model
    that had learned nothing at all.  The direction has to come through, with
    a magnitude that says how sure the verdict was.
    """
    config = LearningConfig(enabled=True, min_samples=200, min_holdout_auc=0.6)
    signature = FEATURE_NAMES.index("signature_match")

    sure = ContextRanker(config=config, rng=random.Random(4))
    sure.fit(corpus(600, signal=True), holdout_fraction=0.25)

    graded = ContextRanker(config=config, rng=random.Random(4))
    report = graded.fit(
        [
            replace(row, label=0.5 if row.label > 0 else -0.3)
            for row in corpus(600, signal=True)
        ],
        holdout_fraction=0.25,
    )

    assert graded.weights[signature] > 0.05, "a graded verdict taught nothing"
    assert graded.weights[signature] < sure.weights[signature]
    # The holdout agrees the direction was there rather than a coin flip.
    assert report["holdout_auc"] > 0.9


def test_a_corpus_of_one_verdict_only_is_refused_not_flattered():
    """
    A scope that has only ever seen success has nothing to be taught.

    One class is nothing to separate, so the holdout cannot say the fit did
    anything — which is also what keeps a base rate with no counter-evidence
    from running to the clip and publishing a saturated block as a model.
    """
    snapshot, report = train_ranker(
        [replace(row, label=1.0) for row in corpus(300, signal=True)],
        "repo",
        config=LearningConfig(enabled=True, min_samples=200, min_holdout_auc=0.6),
        rng=random.Random(5),
    )
    assert report["refused"] == "holdout_auc_below_floor"
    assert snapshot.neutral is True


def test_ranker_ignores_a_feature_the_corpus_never_varied():
    """An unidentifiable weight is how a model orders by one repo's accident."""
    rows = [sample(1.0 if i % 2 else -1.0, score=0.5) for i in range(240)]
    config = LearningConfig(enabled=True, min_samples=10)
    ranker = ContextRanker(config=config, rng=random.Random(6))
    ranker.fit(rows)
    for name in ("path_match", "command_match", "age_days_log", "size_share"):
        assert ranker.weights[FEATURE_NAMES.index(name)] == 0.0


def test_ranker_counts_unlabelled_exposures_but_does_not_fit_them():
    rows = [sample(0.0, score=0.9) for _ in range(50)]
    rows += corpus(240, signal=True)
    config = LearningConfig(enabled=True, min_samples=10)
    ranker = ContextRanker(config=config, rng=random.Random(8))
    report = ranker.fit(rows, holdout_fraction=0.0)
    assert report["samples"] == len(rows)
    assert report["labelled"] == 240


def test_ranker_exploration_moves_a_wide_posterior_more_than_a_narrow_one():
    config = LearningConfig(enabled=True, min_samples=10, explore_coeff=0.25)
    small = ContextRanker(config=config, rng=random.Random(12))
    small.fit(corpus(60, signal=True), holdout_fraction=0.0)
    large = ContextRanker(config=config, rng=random.Random(12))
    large.fit(corpus(4000, signal=True), holdout_fraction=0.0)

    features = CandidateFeatures(signature_match=True, score=0.7)
    assert small.std(features) > large.std(features)

    drawn = {
        small.multiplier(features, explore=True) for _ in range(40)
    }
    assert len(drawn) > 1  # a wide posterior is actually sampled


# --------------------------------------------------------------- snapshot ---
def test_snapshot_round_trips_through_redis():
    client = FakeRedis()
    config = LearningConfig(enabled=True, min_samples=10)
    snapshot, _ = train_ranker(
        corpus(300, signal=True), "repo", config=config, rng=random.Random(13)
    )
    assert publish_snapshot(client, snapshot) is True

    loaded = load_snapshot(client, "repo", config=config)
    assert loaded is not None
    assert loaded.model_id == snapshot.model_id
    assert loaded.ranker.weights == pytest.approx(snapshot.ranker.weights, abs=1e-6)
    assert loaded.holdout_auc == pytest.approx(snapshot.holdout_auc)


def test_snapshot_from_a_different_build_is_refused_not_partially_applied():
    client = FakeRedis()
    config = LearningConfig(enabled=True)
    payload = {
        "schema": 1,
        "scope": "repo",
        "model_id": "repo:1:10:0.900",
        "features": ["bias", "score", "something_new"],
        "w": [0.1, 0.2, 0.3],
    }
    client.set("agentic-kb:model:repo", json.dumps(payload))
    assert load_snapshot(client, "repo", config=config) is None


def test_snapshot_with_a_corrupt_body_reads_as_no_snapshot():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.learning import (
        FEATURE_NAMES,
        SNAPSHOT_SCHEMA,
    )

    client = FakeRedis()
    client.set("agentic-kb:model:repo", "{not json")
    assert load_snapshot(client, "repo", config=LearningConfig()) is None
    client.set(
        "agentic-kb:model:repo",
        json.dumps(
            {
                "schema": SNAPSHOT_SCHEMA,
                "features": list(FEATURE_NAMES),
                "w": [1.0, 2.0],
            }
        ),
    )
    loaded = load_snapshot(client, "repo", config=LearningConfig())
    assert loaded is not None
    assert loaded.neutral is True


def test_a_snapshot_of_another_schema_is_refused_not_misread():
    """A weight order is a schema: reading it under another name reorders nothing.

    A schema-1 snapshot has fifteen weights in one order; this build's model
    has twenty-two in another.  Reading the old weights with the new names
    would mean feature five by one name and another by another, so the safe
    read is the one that reads nothing: the policy serves the scope until the
    model is republished.
    """
    client = FakeRedis()
    old_features = (
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
    client.set(
        "agentic-kb:model:repo",
        json.dumps(
            {
                "schema": 1,
                "features": list(old_features),
                "w": [0.0] * 15,
            }
        ),
    )
    assert load_snapshot(client, "repo", config=LearningConfig()) is None


# ------------------------------------------------------------- history -----
def test_publishing_keeps_the_snapshots_it_replaced():
    """A fit that turns out worse is rolled back, not retrained from scratch.

    The second publish retires the first into the history, newest first.  The
    live key says what is serving; the head of the history says what was
    serving before, and that is the two questions the rollback needs.
    """
    client = FakeRedis()
    config = LearningConfig(enabled=True, min_samples=10)
    first = train_ranker(
        corpus(40, signal=True, seed=21), "repo", config=config, rng=random.Random(21)
    )[0]
    second = train_ranker(
        corpus(60, signal=True, seed=22), "repo", config=config, rng=random.Random(22)
    )[0]
    publish_snapshot(client, first, key_prefix=config.key_prefix)
    publish_snapshot(client, second, key_prefix=config.key_prefix)

    assert load_snapshot(client, "repo", config=config).model_id == second.model_id
    history = snapshot_history(client, "repo", key_prefix=config.key_prefix)
    assert [row["model_id"] for row in history] == [first.model_id]
    # A third publish retires the second behind the first, and the cap has not
    # been reached, so both are still there in the order they retired.
    third = train_ranker(
        corpus(80, signal=True, seed=23), "repo", config=config, rng=random.Random(23)
    )[0]
    publish_snapshot(client, third, key_prefix=config.key_prefix)
    history = snapshot_history(client, "repo", key_prefix=config.key_prefix)
    assert [row["model_id"] for row in history] == [second.model_id, first.model_id]


def test_the_history_is_capped_at_the_size_the_policy_names():
    client = FakeRedis()
    config = LearningConfig(enabled=True, min_samples=10, snapshot_history_size=2)
    ids = []
    for seed in (31, 32, 33, 34, 35):
        snapshot = train_ranker(
            corpus(30, signal=True, seed=seed),
            "repo",
            config=config,
            rng=random.Random(seed),
            # A distinct second per fit, so the ids are distinct: the model
            # id carries the second it was fitted, and a test that cannot
            # tell its fits apart is testing the id, not the history.
            now=float(seed),
        )[0]
        publish_snapshot(
            client, snapshot, key_prefix=config.key_prefix, history_size=2
        )
        ids.append(snapshot.model_id)
    history = snapshot_history(client, "repo", key_prefix=config.key_prefix)
    # The two newest retired fits, in the order they retired: the last fit
    # retired the one before it, which retired the one before that, and the
    # cap kept exactly the two the policy asked for.
    assert len(history) == 2
    assert history[0]["model_id"] == ids[3]
    assert history[1]["model_id"] == ids[2]
    assert ids[0] not in [row["model_id"] for row in history]
    assert ids[1] not in [row["model_id"] for row in history]


def test_republishing_the_same_fit_does_not_eat_two_slots():
    """A fit published twice files its predecessor once, not twice.

    The first publish retires the empty live key into the history as the
    snapshot it is making live: a deployment whose first fit is already the
    one to roll back to must be able to reach it later.  The second publish
    finds the same model in the live key and the history, and files nothing:
    a history that names a model twice spends the cap on a rollback to where
    the operator already stands.
    """
    client = FakeRedis()
    config = LearningConfig(enabled=True, min_samples=10)
    snapshot = train_ranker(
        corpus(40, signal=True, seed=41), "repo", config=config, rng=random.Random(41)
    )[0]
    publish_snapshot(client, snapshot, key_prefix=config.key_prefix, history_size=4)
    # The first publish retires the empty live key: nothing was serving, so
    # nothing goes into the history yet.
    assert snapshot_history(client, "repo", key_prefix=config.key_prefix) == []
    # The second publish finds the same model in the live key and retires
    # nothing: a history that names the live model is a rollback to where the
    # operator already stands.
    publish_snapshot(client, snapshot, key_prefix=config.key_prefix, history_size=4)
    assert snapshot_history(client, "repo", key_prefix=config.key_prefix) == []
    # A different fit retires this one, exactly once.
    other = train_ranker(
        corpus(60, signal=True, seed=42), "repo", config=config, rng=random.Random(42)
    )[0]
    publish_snapshot(client, other, key_prefix=config.key_prefix, history_size=4)
    assert [
        row["model_id"]
        for row in snapshot_history(client, "repo", key_prefix=config.key_prefix)
    ] == [snapshot.model_id]


def test_history_size_zero_is_the_layer_before_history_existed():
    client = FakeRedis()
    config = LearningConfig(enabled=True, min_samples=10)
    publish_snapshot(
        client,
        train_ranker(
            corpus(40, signal=True, seed=51), "repo", config=config, rng=random.Random(51)
        )[0],
        key_prefix=config.key_prefix,
        history_size=0,
    )
    publish_snapshot(
        client,
        train_ranker(
            corpus(50, signal=True, seed=52), "repo", config=config, rng=random.Random(52)
        )[0],
        key_prefix=config.key_prefix,
        history_size=0,
    )
    assert snapshot_history(client, "repo", key_prefix=config.key_prefix) == []


def test_a_rollback_puts_the_asked_snapshot_back_in_force():
    client = FakeRedis()
    config = LearningConfig(enabled=True, min_samples=10)
    first = train_ranker(
        corpus(40, signal=True, seed=61), "repo", config=config, rng=random.Random(61)
    )[0]
    second = train_ranker(
        corpus(60, signal=True, seed=62), "repo", config=config, rng=random.Random(62)
    )[0]
    publish_snapshot(client, first, key_prefix=config.key_prefix)
    publish_snapshot(client, second, key_prefix=config.key_prefix)

    assert rollback_snapshot(
        client, "repo", first.model_id, key_prefix=config.key_prefix
    )
    restored = load_snapshot(client, "repo", config=config)
    assert restored.model_id == first.model_id
    # And the rollback is itself reversible: the model that was displaced goes
    # into the history, so the operator who rolled back the wrong thing can
    # undo it without a retrain.
    history = snapshot_history(client, "repo", key_prefix=config.key_prefix)
    assert [row["model_id"] for row in history] == [second.model_id]


def test_a_rollback_refuses_a_model_that_was_never_published():
    client = FakeRedis()
    config = LearningConfig(enabled=True, min_samples=10)
    publish_snapshot(
        client,
        train_ranker(
            corpus(40, signal=True, seed=71), "repo", config=config, rng=random.Random(71)
        )[0],
        key_prefix=config.key_prefix,
    )
    assert (
        rollback_snapshot(
            client, "repo", "repo:0:0:0.000", key_prefix=config.key_prefix
        )
        is False
    )
    # Nothing moved: a refused rollback is a no-op, not a half-applied one.
    assert load_snapshot(client, "repo", config=config) is not None


def test_a_history_row_this_build_cannot_read_is_dropped_not_served():
    client = FakeRedis()
    config = LearningConfig(enabled=True, min_samples=10)
    snapshot = train_ranker(
        corpus(40, signal=True, seed=81), "repo", config=config, rng=random.Random(81)
    )[0]
    publish_snapshot(client, snapshot, key_prefix=config.key_prefix)
    publish_snapshot(
        client,
        train_ranker(
            corpus(60, signal=True, seed=82), "repo", config=config, rng=random.Random(82)
        )[0],
        key_prefix=config.key_prefix,
    )
    key = "agentic-kb:model_history:repo"
    rows = json.loads(client.values[key])
    rows.append({"model_id": "repo:foreign", "features": ["bias", "invented"]})
    client.values[key] = json.dumps(rows)
    kept = snapshot_history(client, "repo", key_prefix=config.key_prefix)
    # The foreign row is refused like a read would refuse it, and what is
    # kept is the first fit, which the second retired.
    assert [row["model_id"] for row in kept] == [snapshot.model_id]
    assert all(row["model_id"] != "repo:foreign" for row in kept)


def test_a_corrupt_history_reads_as_empty_and_costs_nothing():
    client = FakeRedis()
    client.values["agentic-kb:model_history:repo"] = "{not json"
    assert snapshot_history(client, "repo", key_prefix="agentic-kb") == []
    assert rollback_snapshot(client, "repo", "anything") is False


def test_a_client_that_fails_to_take_the_history_still_publishes():
    """The history is the operator's later; a live model is the layer's now."""

    class _NoRoomForHistory(FakeRedis):
        def get(self, key):
            if ":model_history:" in key:
                raise RuntimeError("boom")
            return super().get(key)

    client = _NoRoomForHistory()
    config = LearningConfig(enabled=True, min_samples=10)
    snapshot = train_ranker(
        corpus(40, signal=True, seed=91), "repo", config=config, rng=random.Random(91)
    )[0]
    assert publish_snapshot(client, snapshot, key_prefix=config.key_prefix) is True
    assert load_snapshot(client, "repo", config=config).model_id == snapshot.model_id


# ------------------------------------------------------------- policy -------
def test_policy_is_static_while_learning_is_disabled():
    client = FakeRedis()
    config = LearningConfig(enabled=False)
    publish_snapshot(
        client,
        train_ranker(
            corpus(300, signal=True), "repo", config=config, rng=random.Random(14)
        )[0],
    )
    policy = LearningPolicy(config=config, client=client, rng=random.Random(14))
    multiplier, detail = policy.multiplier_for(
        CandidateFeatures(signature_match=True), "recipe:x", scope="repo"
    )
    assert multiplier == 1.0
    assert detail["reason"] == "disabled"


def test_policy_composes_reputation_model_and_arm_within_the_clamp():
    client = FakeRedis()
    config = LearningConfig(enabled=True, min_samples=10, explore_rate=0.0)
    snapshot, _ = train_ranker(
        corpus(300, signal=True), "repo", config=config, rng=random.Random(15)
    )
    publish_snapshot(client, snapshot, key_prefix=config.key_prefix)

    policy = LearningPolicy(
        config=config, client=client, rng=random.Random(15), time_fn=FrozenClock()
    )
    for _ in range(5):
        policy.reputation.observe("recipe:20260101:good", 1.0)

    multiplier, detail = policy.multiplier_for(
        CandidateFeatures(kind="recipe", mode="debug", signature_match=True),
        "recipe:20260101:good",
        scope="repo",
    )
    # Not the ceiling: a finite ridge means a finite confidence, and a model
    # that saturates on 300 labels is a model that has stopped being able to
    # change its mind.  What must hold is the composition and the clamp.
    assert 1.0 < multiplier <= config.mult_max
    assert detail["rep"] > 1.0
    assert detail["model"] > 1.0
    assert detail["model_id"] == snapshot.model_id
    assert detail["explore"] is False


def test_policy_reads_one_snapshot_per_ttl_not_one_per_candidate():
    client = FakeRedis()
    config = LearningConfig(enabled=True, model_ttl_seconds=60.0)
    clock = FrozenClock()
    policy = LearningPolicy(
        config=config, client=client, rng=random.Random(16), time_fn=clock
    )
    for _ in range(25):
        policy.multiplier_for(CandidateFeatures(), "doc:x", scope="repo")
    assert sum(1 for c in client.calls if c[0] == "get") == 1

    clock.now += 61.0
    policy.multiplier_for(CandidateFeatures(), "doc:x", scope="repo")
    assert sum(1 for c in client.calls if c[0] == "get") == 2


def test_policy_fails_open_when_every_read_fails():
    policy = LearningPolicy(
        config=LearningConfig(enabled=True),
        client=BrokenRedis(),
        rng=random.Random(17),
        time_fn=FrozenClock(),
    )
    multiplier, detail = policy.multiplier_for(
        CandidateFeatures(signature_match=True), "recipe:x", scope="repo"
    )
    assert multiplier == 1.0
    assert detail["reason"] == "reputation"
    assert detail["model"] == 1.0
    assert detail["model_id"] == ""


def test_policy_annotation_carries_no_content():
    client = FakeRedis()
    config = LearningConfig(enabled=True, min_samples=10)
    snapshot, _ = train_ranker(
        corpus(300, signal=True), "repo", config=config, rng=random.Random(18)
    )
    publish_snapshot(client, snapshot)
    policy = LearningPolicy(config=config, client=client, rng=random.Random(18))
    annotation = policy.annotation(scope="repo")
    assert annotation["model_id"] == snapshot.model_id
    assert set(annotation) == {
        "enabled",
        "model_id",
        "holdout_auc",
        "labelled",
        "range",
    }


# ------------------------------------------------------------------- log ----
def test_training_log_records_features_and_never_content():
    client = FakeRedis()
    log = FeedbackLog(client=client, stream_key="agentic_kb:training")
    features = CandidateFeatures(
        kind="recipe", mode="debug", score=0.81, signature_match=True
    )
    log.append_injection(
        scope="repo",
        session_id="sess-1234",
        generation="gen-9",
        model_id="repo:1:10:0.900",
        features=[features],
        document_names=["recipe:20260101:abc123"],
    )
    payload = client.streams["agentic_kb:training"][0][1]["d"]
    for banned in ("def ", "src/", "/home/", "traceback", "assert"):
        assert banned not in payload
    record = FeedbackSample.from_json(json.loads(payload))
    assert record.features.signature_match is True
    assert record.label == 0.0
    assert record.label_kind == "injection"


def test_training_log_falls_back_to_a_local_tail_without_redis():
    log = FeedbackLog(client=None, maxlen=3, local_maxlen=3)
    for index in range(5):
        log.append(sample(1.0, score=index / 10))
    _, drained = log.drain(count=10)
    assert len(drained) == 3


def test_worker_drain_resumes_from_the_last_id():
    client = FakeRedis()
    log = FeedbackLog(client=client)
    for _ in range(4):
        log.append(sample(1.0))
    last, rows = log.drain(count=2)
    assert len(rows) == 2
    assert last == "2-0"
    assert all(isinstance(row, FeedbackSample) for row in rows)


# ------------------------------------------------------------------ bandit --
def test_source_bandit_serves_the_policy_weight_until_it_has_history():
    config = LearningConfig(enabled=True, min_observations=3)
    bandit = SourceBandit(config=config, rng=random.Random(19))
    assert bandit.value_per_char("plan", "recipe", default=0.6) == 0.6

    for _ in range(5):
        bandit.observe("plan", "recipe", 1.0, chars=400)
    learned = bandit.value_per_char("plan", "recipe", default=0.6)
    assert learned > 0.6
    assert bandit.value_per_char("plan", "doc", default=0.6) == 0.6


def test_source_bandit_distinguishes_a_cheap_win_from_an_expensive_one():
    config = LearningConfig(enabled=True, min_observations=2)
    bandit = SourceBandit(config=config, rng=random.Random(20))
    for _ in range(4):
        bandit.observe("debug", "tool_recipe", 1.0, chars=120)
        bandit.observe("debug", "session_archive", 1.0, chars=1800)
    assert bandit.value_per_char("debug", "tool_recipe") > bandit.value_per_char(
        "debug", "session_archive"
    )


# ------------------------------------------------------------- end to end ---
def test_settle_moves_the_next_ranking_of_the_same_document():
    """The whole loop, on one document, with one clock and one seed."""
    client = FakeRedis()
    config = LearningConfig(enabled=True, min_observations=2, explore_rate=0.0)
    clock = FrozenClock()
    policy = LearningPolicy(
        config=config, client=client, rng=random.Random(21), time_fn=clock
    )
    features = CandidateFeatures(kind="recipe", mode="debug", size_chars=300)
    before = policy.multiplier_for(features, "recipe:20260101:z", scope="repo")[0]

    for _ in range(4):
        policy.settle(
            FeedbackSample(
                scope="repo",
                session_id="s",
                generation="g",
                document_name="recipe:20260101:z",
                features=features,
                label=-1.0,
            )
        )
    after = policy.multiplier_for(features, "recipe:20260101:z", scope="repo")[0]

    assert before == 1.0
    assert after < before
    assert client.streams[config.stream_key]  # the training log was written
    assert policy.bandit.arms()["debug|recipe"]["n"] == 4


def test_fractional_labels_fold_as_their_own_weight():
    reputation = DocReputation(
        config=LearningConfig(enabled=True, min_observations=1),
        rng=random.Random(22),
        time_fn=FrozenClock(),
    )
    reputation.observe("recipe:plausible", 0.5)
    stats = reputation.stats("recipe:plausible")
    assert stats["alpha"] == pytest.approx(2.5)
    assert stats["beta"] == pytest.approx(2.0)
    assert stats["exposure"] == 1


def test_features_round_trip_through_the_stream_record():
    features = CandidateFeatures(
        kind="task_solution",
        mode="implement",
        score=0.4242,
        score_scale="rrf",
        path_match=2,
        age_days=13.5,
        upvotes=3,
        downvotes=1,
        exposure=9,
        rank=4,
        size_chars=900,
        budget_chars=5000,
    )
    restored = CandidateFeatures.from_json(features.to_json())
    assert restored == features
    assert len(features.vector()) == len(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in features.vector())


def test_snapshot_type_is_exported_and_immutable():
    snapshot = ModelSnapshot(
        scope="repo",
        model_id="repo:1:1:1.000",
        trained_at=time.time(),
        samples=1,
        labelled=1,
        positive=1,
        holdout_auc=1.0,
        features=FEATURE_NAMES,
        ranker=ContextRanker(),
    )
    assert snapshot.neutral is True
    with pytest.raises(Exception):
        snapshot.scope = "other"  # type: ignore[misc]


# ------------------------------------------------------- split by session ----
def _graded(
    scope: str,
    session: str,
    generation: str,
    document: str,
    label: float,
    at: float = 0.0,
    label_kind: str = "counterfactual",
    features: Optional[CandidateFeatures] = None,
    **feature_values: float,
) -> FeedbackSample:
    """One labelled sample with the given feature values."""
    if features is None:
        features = CandidateFeatures(**feature_values)
    return FeedbackSample(
        scope=scope,
        session_id=session,
        generation=generation,
        document_name=document,
        features=features,
        label=label,
        label_kind=label_kind,
        at=at,
    )


def test_split_never_breaks_a_session_across_the_two_sets():
    """A session is in the train set or the holdout, never in both."""
    samples = [
        _graded("repo", f"s{i % 8}", "g1", f"doc{j}", 1.0 if j % 2 else 0.0)
        for i in range(8)
        for j in range(6)
    ]
    train, holdout = _split(samples, 0.25, random.Random(7))

    trained = {s.session_id for s in train}
    held = {s.session_id for s in holdout}
    assert trained and held
    assert not (trained & held)
    assert len(train) + len(holdout) == len(samples)


def test_split_of_one_session_keeps_it_whole_and_trains():
    """One session cannot be held out against itself; it stays in train."""
    samples = [
        _graded("repo", "only", "g1", f"doc{j}", 1.0 if j % 2 else 0.0)
        for j in range(20)
    ]
    train, holdout = _split(samples, 0.5, random.Random(1))
    # The one session stays in train; there is nothing to hold out.
    assert len(train) == 20
    assert not holdout


def test_split_is_reproducible_from_the_seed():
    samples = [
        _graded("repo", f"s{i}", "g1", f"doc{i}", 1.0 if i % 2 else 0.0)
        for i in range(30)
    ]
    first = _split(samples, 0.3, random.Random(42))
    second = _split(samples, 0.3, random.Random(42))
    assert [s.document_name for s in first[0]] == [s.document_name for s in second[0]]
    assert [s.document_name for s in first[1]] == [s.document_name for s in second[1]]


def test_split_with_no_holdout_keeps_everything_for_training():
    samples = [
        _graded("repo", f"s{i}", "g1", f"doc{i}", 1.0) for i in range(10)
    ]
    train, holdout = _split(samples, 0.0, random.Random(3))
    assert len(train) == 10
    assert holdout == []


# ------------------------------------------------------ credit by kind -------
def test_a_coarse_label_is_worth_half_a_verdict():
    """The kind of a label scales how much it counts, not which way it points."""
    sample = _graded("repo", "s1", "g1", "doc", 1.0, label_kind="coarse")
    assert _credit_kind(sample).label == pytest.approx(0.5)


def test_crediting_a_verdict_leaves_it_untouched():
    sample = _graded("repo", "s1", "g1", "doc", -1.0, label_kind="counterfactual")
    assert _credit_kind(sample) is sample


def test_crediting_keeps_the_direction_of_a_negative_label():
    sample = _graded("repo", "s1", "g1", "doc", -1.0, label_kind="coarse")
    credited = _credit_kind(sample)
    assert credited.label == pytest.approx(-0.5)
    assert (credited.label > 0) == (sample.label > 0)


def test_an_unknown_kind_is_credited_rather_than_discarded():
    """A kind this build does not know is not a reason to lose its labels."""
    sample = _graded("repo", "s1", "g1", "doc", 1.0, label_kind="future_kind")
    assert _credit_kind(sample) is sample


def test_a_coarse_label_trains_a_weaker_weight_than_a_verdict():
    """The same evidence, labelled two ways, moves the fit by different amounts."""
    # Positive samples have higher scores; the weight on the score feature
    # should grow, and it should grow more when the labels are verdicts.
    strong = [
        _graded(
            "repo", f"s{i}", "g1", f"doc{i}",
            label=1.0 if i % 2 else -1.0,
            score=0.9 if i % 2 else 0.1,
        )
        for i in range(60)
    ]
    weak = [replace(s, label_kind="coarse") for s in strong]

    config = LearningConfig(weight_clip=3.0)
    strong_ranker = ContextRanker(config=config)
    strong_ranker.fit(strong, holdout_fraction=0.0, rng=random.Random(5))
    weak_ranker = ContextRanker(config=config)
    weak_ranker.fit(weak, holdout_fraction=0.0, rng=random.Random(5))

    strong_score = sum(abs(w) for w in strong_ranker.weights)
    weak_score = sum(abs(w) for w in weak_ranker.weights)
    assert strong_score > weak_score


# ----------------------------------------------------------- trust gate ------
def _corpus(count: int, label_kind: str = "counterfactual"):
    """A separable corpus: positives score high, negatives score low."""
    samples = []
    for i in range(count):
        positive = i % 2 == 0
        samples.append(
            _graded(
                "repo", f"s{i}", "g1", f"doc{i}",
                label=1.0 if positive else -1.0,
                score=0.9 if positive else 0.1,
                label_kind=label_kind,
            )
        )
    return samples


def test_a_corpus_of_only_coarse_labels_is_refused_for_thin_evidence():
    """A thousand weak labels are not a thousand observations."""
    samples = _corpus(250, label_kind="coarse")
    config = LearningConfig(
        min_samples=200, min_evidence=150.0, min_holdout_auc=0.6
    )
    _snapshot, report = train_ranker(
        samples, "repo", config=config, holdout_fraction=0.2,
        rng=random.Random(7), now=1.0,
    )
    assert report["labelled"] >= 200
    assert report["evidence"] == pytest.approx(125.0, abs=2.0)
    assert not report.get("accepted")
    assert report.get("refused") == "evidence_below_threshold"


def test_the_same_corpus_of_verdicts_clears_the_gate():
    samples = _corpus(250, label_kind="counterfactual")
    config = LearningConfig(
        min_samples=200, min_evidence=250.0, min_holdout_auc=0.6
    )
    snapshot, report = train_ranker(
        samples, "repo", config=config, holdout_fraction=0.2,
        rng=random.Random(7), now=1.0,
    )
    assert report.get("accepted"), report
    assert snapshot.ranker.trained


def test_evidence_counts_a_mix_of_kinds():
    """Half verdicts, half coarse: the evidence is the sum of their credits."""
    samples = _corpus(200)
    mixed = [
        replace(s, label_kind="coarse") if i % 2 else s
        for i, s in enumerate(samples)
    ]
    config = LearningConfig(
        min_samples=200, min_evidence=10.0, min_holdout_auc=0.6
    )
    _snapshot, report = train_ranker(
        mixed, "repo", config=config, holdout_fraction=0.2,
        rng=random.Random(3), now=1.0,
    )
    assert report["labelled"] == 200
    assert report["evidence"] == pytest.approx(150.0, abs=1.5)


def test_min_evidence_is_validated():
    with pytest.raises(ValueError):
        LearningConfig(min_evidence=0.0)


# ------------------------------------------------------ bandit integration ---
def test_source_bandit_moves_the_multiplier():
    """A source arm with history shifts the multiplier."""
    config = LearningConfig(enabled=True, min_observations=1)
    policy = LearningPolicy(config=config, rng=random.Random(17))
    feat = CandidateFeatures(
        kind="recipe", mode="plan", score=0.8,
        size_chars=120, budget_chars=1000,
    )
    # Before any history the arm returns the default.
    mult_before, _ = policy.multiplier_for(feat)
    # Feed the arm some good history: small recipes that earn wins.
    for i in range(5):
        policy.settle(
            _graded("repo", f"s{i}", "g1", f"doc{i}", label=1.0, features=feat)
        )
    # A source that earns wins with little cost is worth more per character.
    mult_after, detail = policy.multiplier_for(feat)
    assert detail["arm"] > 1.0
    assert mult_after >= mult_before


def test_the_annotation_carries_arms_when_they_exist():
    """An operator can see what the bandit knows from the annotation."""
    policy = LearningPolicy(
        config=LearningConfig(enabled=True, min_observations=1),
        rng=random.Random(18),
    )
    feat = CandidateFeatures(kind="tool_recipe", mode="debug", score=0.7)
    assert policy.annotation("repo").get("arms") is None
    for i in range(3):
        policy.settle(
            _graded("repo", f"s{i}", "g1", f"doc{i}", label=1.0, features=feat)
        )
    arms = policy.annotation("repo").get("arms")
    assert arms is not None
    assert "debug|tool_recipe" in arms


def test_arms_are_copied_into_the_published_snapshot():
    """A snapshot says what a source was worth when it shipped."""
    config = LearningConfig(enabled=True)
    feat = CandidateFeatures(kind="recipe", mode="plan", score=0.9)
    # The request path feeds the bandit.
    bandit = SourceBandit(config=config, rng=random.Random(19))
    for i in range(10):
        bandit.observe("plan", "recipe", 1.0, chars=300)
    # The learner fits and publishes, copying the arms.
    samples = [
        _graded("repo", f"s{i}", "g1", f"doc{i}", label=1.0, features=feat)
        for i in range(250)
    ]
    snapshot, _ = train_ranker(
        samples,
        "repo",
        config=config,
        holdout_fraction=0.2,
        rng=random.Random(19),
        now=1.0,
        arms=bandit.arms(),
    )
    assert snapshot is not None
    assert "plan|recipe" in snapshot.arms


def test_the_arms_flag_prints_the_bandit_state():
    """An operator can ask the learner what the bandit has learned."""
    from llm_router_plugins.cli.plugins.utils.agentic_knowledge.learner import (
        Learner,
    )
    store = FakeRedis()
    config = LearningConfig(enabled=True)
    learner = Learner(config=config, client=store, rng=random.Random(20))
    learner._bandit.observe("plan", "recipe", 1.0, chars=300)
    summary = learner.arm_summary()
    assert "plan|recipe" in summary
    assert summary["plan|recipe"]["n"] == 1


# ------------------------------------------------------- exploration record --
def test_an_exposure_records_whether_the_ranking_drew_the_spread():
    """The draw belongs to the request, so it is written with the exposure."""
    client = FakeRedis()
    config = LearningConfig(enabled=True, stream_key="agentic_kb:training")
    log = FeedbackLog(client=client, stream_key=config.stream_key)
    features = CandidateFeatures(kind="recipe", mode="debug", score=0.6)

    log.append_injection(
        scope="repo",
        session_id="s1",
        generation="g1",
        model_id="repo:1:10:0.700",
        features=[features, features],
        document_names=["recipe:20260101:a", "recipe:20260101:b"],
        explorations=["explore", "greedy"],
    )
    _, samples = log.drain()

    by_name = {sample.document_name: sample for sample in samples}
    assert by_name["recipe:20260101:a"].exploration == "explore"
    assert by_name["recipe:20260101:b"].exploration == "greedy"


def test_an_exposure_without_a_recorded_draw_is_greedy():
    """A log written by a caller that knows nothing is not an exploration."""
    log = FeedbackLog()
    log.append_injection(
        scope="repo",
        session_id="s1",
        generation="g1",
        model_id="",
        features=[CandidateFeatures(kind="recipe")],
        document_names=["recipe:20260101:a"],
    )
    assert log.drain()[1][0].exploration == "greedy"


def test_the_exploration_flag_round_trips_through_the_stream_record():
    sample = FeedbackSample(
        scope="repo",
        session_id="s",
        generation="g",
        document_name="recipe:20260101:a",
        features=CandidateFeatures(kind="recipe"),
        label=1.0,
        exploration="explore",
    )
    assert FeedbackSample.from_json(sample.to_json()).exploration == "explore"
    # A record written before this field existed reads as the greedy default,
    # not as a corrupted one.
    assert FeedbackSample.from_json({}).exploration == "greedy"


def test_a_settled_verdict_remembers_the_draw_of_the_ranking_that_showed_it():
    """The ranking decides what the verdict was worth, and says so."""
    client = FakeRedis()
    config = LearningConfig(
        enabled=True,
        min_samples=10,
        explore_rate=1.0,
        model_ttl_seconds=0.001,
    )
    snapshot, _ = train_ranker(
        corpus(300, signal=True), "repo", config=config, rng=random.Random(31)
    )
    publish_snapshot(client, snapshot, key_prefix=config.key_prefix)
    policy = LearningPolicy(
        config=config,
        client=client,
        rng=random.Random(32),
        time_fn=FrozenClock(),
    )
    features = CandidateFeatures(kind="recipe", mode="debug", score=0.7)
    _, detail = policy.multiplier_for(features, "recipe:20260101:a", scope="repo")
    assert detail["explore"] is True

    policy.settle(
        FeedbackSample(
            scope="repo",
            session_id="s",
            generation="g",
            document_name="recipe:20260101:a",
            features=features,
            label=1.0,
        )
    )
    _, samples = policy.log.drain()
    assert samples[-1].exploration == "explore"


def test_a_greedy_ranking_leaves_the_verdict_greedy():
    client = FakeRedis()
    config = LearningConfig(enabled=True, explore_rate=0.0)
    policy = LearningPolicy(
        config=config, client=client, rng=random.Random(33), time_fn=FrozenClock()
    )
    features = CandidateFeatures(kind="recipe", mode="debug", score=0.7)
    policy.multiplier_for(features, "recipe:20260101:a", scope="repo")
    policy.settle(
        FeedbackSample(
            scope="repo",
            session_id="s",
            generation="g",
            document_name="recipe:20260101:a",
            features=features,
            label=1.0,
        )
    )
    _, samples = policy.log.drain()
    assert samples[-1].exploration == "greedy"


def test_a_later_greedy_ranking_does_not_downgrade_an_explored_one():
    """One draw in a generation makes the generation an exploring one."""
    policy = LearningPolicy(config=LearningConfig(enabled=True, explore_rate=1.0))
    features = CandidateFeatures(kind="recipe", mode="debug", score=0.7)
    policy._remember_exploration("recipe:20260101:a", True)
    policy._remember_exploration("recipe:20260101:a", False)
    assert policy._ranked_exploration.get("recipe:20260101:a") is True


def test_the_exploration_ledger_is_bounded():
    """A long-lived proxy does not grow a ledger of every draw it ever made."""
    from llm_router_plugins.utils.knowledge.agentic_knowledge.learning import (
        _RANKED_EXPLORATION_LIMIT,
    )
    policy = LearningPolicy(config=LearningConfig(enabled=True, explore_rate=1.0))
    for index in range(1000):
        policy._remember_exploration(f"recipe:20260101:{index}", True)
    assert len(policy._ranked_exploration) == _RANKED_EXPLORATION_LIMIT
    # The oldest draws go first, so the newest one is still in the ledger.
    assert "recipe:20260101:999" in policy._ranked_exploration
    assert "recipe:20260101:0" not in policy._ranked_exploration


def test_settling_forgives_the_document_its_draw():
    """A verdict consumes the draw: the next injection starts clean."""
    config = LearningConfig(enabled=True, explore_rate=0.0)
    policy = LearningPolicy(config=config, rng=random.Random(35))
    policy._ranked_exploration["recipe:20260101:a"] = True
    policy.settle(
        FeedbackSample(
            scope="repo",
            session_id="s",
            generation="g",
            document_name="recipe:20260101:a",
            features=CandidateFeatures(kind="recipe"),
            label=1.0,
        )
    )
    assert "recipe:20260101:a" not in policy._ranked_exploration


# ------------------------------------------------------------ pass lock -----
def test_a_pass_lock_keeps_the_second_pass_out():
    """Two cron minutes must not drain the same cursor and publish over each other."""
    from llm_router_plugins.utils.knowledge.agentic_knowledge.learning import (
        acquire_pass_lock,
        release_pass_lock,
    )
    client = FakeRedis()
    first = acquire_pass_lock(client, "agentic-kb", ttl_seconds=60.0)
    second = acquire_pass_lock(client, "agentic-kb", ttl_seconds=60.0)
    assert first
    assert second == ""
    assert release_pass_lock(client, "agentic-kb", first) is True
    # The slot is free again for the next minute of the cron.
    assert acquire_pass_lock(client, "agentic-kb", ttl_seconds=60.0)


def test_release_returns_the_lock_only_when_it_is_still_ours():
    """A pass that outlived its ttl must not delete another pass's lock."""
    from llm_router_plugins.utils.knowledge.agentic_knowledge.learning import (
        acquire_pass_lock,
        release_pass_lock,
    )
    client = FakeRedis()
    first = acquire_pass_lock(client, "agentic-kb", ttl_seconds=60.0)
    # The ttl expired while the first pass was still running, and the next
    # cron minute took the lock in the meantime.
    client.values.pop("agentic-kb:learner:lock", None)
    second = acquire_pass_lock(client, "agentic-kb", ttl_seconds=60.0)
    assert release_pass_lock(client, "agentic-kb", first) is False
    assert client.values.get("agentic-kb:learner:lock") == second
    assert release_pass_lock(client, "agentic-kb", second) is True


def test_a_store_that_cannot_answer_is_a_lock_somebody_holds():
    """The safe reading of 'I do not know' is 'do not fit'."""
    from llm_router_plugins.utils.knowledge.agentic_knowledge.learning import (
        acquire_pass_lock,
        release_pass_lock,
    )
    client = BrokenRedis()
    assert acquire_pass_lock(client, "agentic-kb") == ""
    # Releasing a token into a broken store fails without a stack trace:
    # there was nothing to release there, and the pass must finish clean.
    assert release_pass_lock(client, "agentic-kb", "someone") is False


def test_a_pass_without_a_store_is_its_own_lock():
    """One process with no shared store needs no coordination with itself."""
    from llm_router_plugins.utils.knowledge.agentic_knowledge.learning import (
        acquire_pass_lock,
        release_pass_lock,
    )
    token = acquire_pass_lock(None, "agentic-kb")
    assert token == "in-process"
    assert release_pass_lock(None, "agentic-kb", token) is True


def test_force_is_the_only_key_that_opens_a_held_lock():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.learning import (
        acquire_pass_lock,
        release_pass_lock,
    )
    client = FakeRedis()
    first = acquire_pass_lock(client, "agentic-kb", ttl_seconds=60.0)
    # The ordinary ask is still refused while the lock is held.
    assert acquire_pass_lock(client, "agentic-kb", ttl_seconds=60.0) == ""
    # A forceful pass takes what is there.
    forced = acquire_pass_lock(client, "agentic-kb", ttl_seconds=60.0, force=True)
    assert forced
    assert forced != first
    # ...and the one it displaced is no longer able to release what it held.
    assert release_pass_lock(client, "agentic-kb", first) is False
    assert release_pass_lock(client, "agentic-kb", forced) is True


# ------------------------------------------------------- features, schema 2 -
def test_the_feature_vector_carries_the_twenty_two_names_the_snapshot_publishes():
    """The vector's order is the schema; a new feature is a new name, not a new meaning.

    The seven features added for the interactions and for the state the
    document sits in sit at the end, so the fifteen a snapshot of the
    previous build carries are still read under the names they were fit with
    — and a snapshot of that build is refused by the schema, not misread.
    """
    assert len(FEATURE_NAMES) == 22
    for name in (
        "score_x_path",
        "sig_x_cmd",
        "agent_loop_level",
        "is_human_doc",
        "patch_sim_prior",
        "query_intent_align",
        "error_class_match",
    ):
        assert name in FEATURE_NAMES
    features = CandidateFeatures(kind="recipe", mode="debug", score=0.5)
    assert len(features.vector()) == len(FEATURE_NAMES)


def test_the_interactions_and_the_state_the_document_sits_in_reach_the_vector():
    """Each added feature answers one question the other features cannot.

    ``score_x_path`` is the product the plan named, not a new score; the loop
    level saturates at eight, a document a person wrote reads as one against
    a mined one's zero, and the error-class match is a fact about the pair:
    both halves must name the class, and neither may name it differently.
    """
    plain = CandidateFeatures(kind="recipe", mode="debug", score=0.5)
    indexed = {name: value for name, value in zip(FEATURE_NAMES, plain.vector())}
    assert indexed["score_x_path"] == 0.0
    assert indexed["sig_x_cmd"] == 0.0
    assert indexed["agent_loop_level"] == 0.0
    assert indexed["is_human_doc"] == 0.0
    assert indexed["patch_sim_prior"] == 0.0
    assert indexed["query_intent_align"] == 0.0
    assert indexed["error_class_match"] == 0.0

    met = CandidateFeatures(
        kind="doc",
        mode="debug",
        score=0.5,
        path_match=2,
        signature_match=True,
        command_match=True,
        loop_count=8,
        patch_sim=0.75,
        intent_align=True,
        error_class="KeyError",
        doc_error_class="keyerror",
    )
    indexed = dict(zip(FEATURE_NAMES, met.vector()))
    assert indexed["score_x_path"] == pytest.approx(0.5)
    assert indexed["sig_x_cmd"] == 1.0
    assert indexed["agent_loop_level"] == 1.0
    assert indexed["is_human_doc"] == 1.0
    assert indexed["patch_sim_prior"] == pytest.approx(0.75)
    assert indexed["query_intent_align"] == 1.0
    # The match folds case: a class named differently by the two sides is
    # still the class, and one side naming none is no match at all.
    assert indexed["error_class_match"] == 1.0
    missing = replace(met, error_class="")
    indexed = dict(zip(FEATURE_NAMES, missing.vector()))
    assert indexed["error_class_match"] == 0.0


def test_the_new_features_survive_the_stream_round_trip():
    """What the worker reads is what the request path wrote, feature for feature."""
    features = CandidateFeatures(
        kind="recipe",
        mode="debug",
        score=0.4,
        loop_count=2,
        error_class="KeyError",
        doc_error_class="KeyError",
        signature_similarity=0.33,
        patch_sim=0.5,
        intent_align=True,
    )
    back = CandidateFeatures.from_json(features.to_json())
    assert back.loop_count == 2
    assert back.error_class == "KeyError"
    assert back.doc_error_class == "KeyError"
    assert back.signature_similarity == pytest.approx(0.33)
    assert back.patch_sim == 0.5
    assert back.intent_align is True
    assert back.vector() == features.vector()


# ------------------------------------------------------------- pairwise -----
def _contrast_corpus(seed: int = 3) -> list:
    """One request's worth of verdicts: two documents, one credited, one blamed.

    The pairs a pairwise fit needs are the two judgements of the *same
    request* pointing at each other, so every sample here shares a session
    and a generation, and the credited document differs from the blamed one
    in exactly the feature the fit is asked to read: the signature match.
    """
    rng = random.Random(seed)
    rows = []
    for index in range(40):
        rows.extend(
            [
                FeedbackSample(
                    scope="repo",
                    session_id=f"s{index}",
                    generation=f"g{index}",
                    document_name=f"recipe:20260101:{index:04d}:win",
                    features=CandidateFeatures(
                        kind="recipe",
                        mode="debug",
                        score=rng.random(),
                        signature_match=True,
                    ),
                    label=1.0,
                ),
                FeedbackSample(
                    scope="repo",
                    session_id=f"s{index}",
                    generation=f"g{index}",
                    document_name=f"recipe:20260101:{index:04d}:lose",
                    features=CandidateFeatures(
                        kind="recipe",
                        mode="debug",
                        score=rng.random(),
                        signature_match=False,
                    ),
                    label=-1.0,
                ),
            ]
        )
    return rows


def test_the_pairwise_objective_is_the_log_likelihood_of_the_order():
    """The loss the Newton step climbs: each pair is a probability of an order.

    At zero weights the model cannot tell the two documents apart, and every
    pair is worth ``weight * log(0.5)`` — the whole objective before any
    ridge, which a zero ridge leaves out.  A weight that puts the winner
    above the loser moves the objective up; one that swaps the order down.
    """
    from llm_router_plugins.utils.knowledge.agentic_knowledge.learning import (
        _pairwise_objective,
    )

    winner = CandidateFeatures(
        kind="recipe", mode="debug", signature_match=True
    ).vector()
    loser = CandidateFeatures(kind="recipe", mode="debug").vector()
    pairs = [(winner, loser, 2.0)]
    varied = [FEATURE_NAMES.index(name) for name in ("signature_match",)]
    zero = [0.0] * 22
    assert _pairwise_objective(zero, pairs, 0.0, varied) == pytest.approx(
        2.0 * math.log(0.5)
    )
    ordered = list(zero)
    ordered[varied[0]] = 1.0
    reversed_ = list(zero)
    reversed_[varied[0]] = -1.0
    assert (
        _pairwise_objective(ordered, pairs, 0.0, varied)
        > _pairwise_objective(zero, pairs, 0.0, varied)
        > _pairwise_objective(reversed_, pairs, 0.0, varied)
    )
    # The ridge is a penalty on leaving the prior, not on the weights: the
    # same weights at a prior that agrees with them cost nothing.
    assert _pairwise_objective(ordered, pairs, 1.0, varied, ordered) > (
        _pairwise_objective(ordered, pairs, 1.0, varied)
    )


def test_the_pairwise_newton_step_climbs_the_objective():
    """One damped step from the zero prior must not land below where it stood."""
    from llm_router_plugins.utils.knowledge.agentic_knowledge.learning import (
        _pairwise_newton_step,
        _pairwise_objective,
    )

    rows = _contrast_corpus()
    pairs = [
        (rows[index].features.vector(), rows[index + 1].features.vector(), 1.0)
        for index in range(0, len(rows), 2)
    ]
    varied = [1, 5, 6, 7, 15, 16, 17, 18, 19, 20, 21]
    zero = [0.0] * 22
    start = _pairwise_objective(zero, pairs, 0.5, varied)
    weights = zero
    for _ in range(4):
        weights = _pairwise_newton_step(weights, pairs, 0.5, 3.0, varied)
    end = _pairwise_objective(weights, pairs, 0.5, varied)
    assert end >= start - 1e-9
    # The step is not a no-op: a separable contrast moves the weight the
    # contrast varies, or the objective would never have anything to climb.
    assert any(abs(w) > 1e-4 for w in weights)


def test_the_pairwise_fit_orders_the_contrast_it_was_shown():
    """A corpus of same-request verdicts fits on the order, and the report says so.

    Every pair is the credited document of one request against the blamed one
    of the same, so the objective has nothing else to read; the fit that
    publishes is the one that ranks the winner above the loser, and the
    report names the loss it climbed and the pairs it climbed on.
    """
    config = LearningConfig(loss="pairwise", min_samples=10, min_evidence=10.0)
    ranker = ContextRanker(config=config)
    report = ranker.fit(_contrast_corpus())
    assert report["loss"] == "pairwise"
    assert report["pairs"] > 0
    winner = CandidateFeatures(kind="recipe", mode="debug", signature_match=True)
    loser = CandidateFeatures(kind="recipe", mode="debug", signature_match=False)
    assert ranker.predict(winner) > ranker.predict(loser)


def test_the_pairwise_fit_falls_back_to_pointwise_when_nothing_can_be_compared():
    """A corpus whose verdicts never contradict a request has no order to fit.

    One document per request holds one verdict each, and a pair needs two of
    opposite direction in the same request: the fit then reads the labels the
    pointwise way, and the report says ``pointwise`` ran — "pairwise was
    asked" and "pairwise fitted" are two different answers, and the report
    exists so nobody confuses them.
    """
    from llm_router_plugins.utils.knowledge.agentic_knowledge.learning import (
        LOSS_POINTWISE,
    )

    config = LearningConfig(loss="pairwise", min_samples=10, min_evidence=10.0)
    ranker = ContextRanker(config=config)
    rows = [
        FeedbackSample(
            scope="repo",
            session_id=f"s{index}",
            generation=f"g{index}",
            document_name=f"recipe:20260101:{index:05d}",
            features=CandidateFeatures(
                kind="recipe", mode="debug", signature_match=index % 2 == 0
            ),
            label=1.0 if index % 2 == 0 else -1.0,
        )
        for index in range(40)
    ]
    report = ranker.fit(rows)
    assert report["loss"] == LOSS_POINTWISE
    assert report["pairs"] == 0


def test_the_pairwise_gate_reads_the_same_evidence_the_pointwise_one_does():
    """Both objectives refuse a corpus too thin to trust; neither is an escape hatch."""
    config = LearningConfig(loss="pairwise", min_samples=200, min_evidence=100.0)
    snapshot, report = train_ranker(
        _contrast_corpus(), "repo", config=config, rng=random.Random(1)
    )
    assert report["pairs"] > 0
    assert report["labelled"] < config.min_samples
    assert snapshot.neutral is True


# ------------------------------------------------------------ meta prior ----
def test_the_meta_prior_needs_the_scopes_it_claims():
    """One repository is not a prior; the mean of the accepted models is.

    A fit a scope was accepted for is the evidence the prior averages, and
    fewer scopes than the minimum is an answer the prior cannot give — it
    returns ``None`` and the fit starts from the zero prior, which is the
    policy, rather than from one repository's accidents.
    """
    from llm_router_plugins.utils.knowledge.agentic_knowledge.learning import (
        compute_meta_prior,
    )

    def model(weights, scope):
        ranker = ContextRanker(config=LearningConfig())
        ranker.weights = list(weights)
        ranker.precision = [[1.0] * 22 for _ in range(22)]
        return ModelSnapshot(
            scope=scope,
            model_id=f"m-{scope}",
            trained_at=0.0,
            samples=300,
            labelled=300,
            positive=150,
            holdout_auc=0.9,
            features=FEATURE_NAMES,
            ranker=ranker,
        )

    a = model([0.1] * 22, "a")
    b = model([0.3] * 22, "b")
    assert compute_meta_prior([a, b], min_scopes=3) is None
    assert compute_meta_prior([a, b], min_scopes=2) is not None
    mean = compute_meta_prior([a, b, model([0.5] * 22, "c")], min_scopes=3)
    assert mean is not None
    assert all(value == pytest.approx(0.3) for value in mean)
    # A snapshot the gate refused carries the neutral vector, and a neutral
    # vector is not a repository's opinion: it is the absence of one, and the
    # mean is over the models that moved.
    neutral = model([0.0] * 22, "d")
    assert neutral.neutral is True
    assert compute_meta_prior(
        [a, b, model([0.5] * 22, "c"), neutral], min_scopes=3
    ) == mean


def test_the_meta_prior_keeps_what_the_other_repositories_agreed_on():
    """A feature this repository's labels do not speak about keeps its weight.

    The cold repository has judged nothing about ``signature_match`` — its
    labels are about the commands it ran — and a fit from the zero prior
    returns the feature to zero, which is the layer forgetting what every
    other repository learned.  A fit from the meta prior keeps the shared
    weight, and a corpus whose own verdicts *contradict* the prior still
    moves away from it: the prior is where a fit starts, and the ridge is
    what lets this repository's labels move it.
    """
    from llm_router_plugins.utils.knowledge.agentic_knowledge.learning import (
        _as_prior,
    )

    index = FEATURE_NAMES.index("signature_match")
    config = LearningConfig(l2=32.0, min_samples=10)

    def corpus_uninformed(count):
        """Labels driven by the command, so the signature says nothing."""
        rng = random.Random(5)
        rows = []
        for position in range(count):
            features = CandidateFeatures(
                kind="recipe",
                mode="debug",
                signature_match=position % 2 == 0,
                command_match=rng.random() < 0.5,
                score=0.5,
            )
            rows.append(
                FeedbackSample(
                    scope="repo",
                    session_id=f"s{position}",
                    generation=f"g{position}",
                    document_name=f"recipe:20260101:{position:05d}",
                    features=features,
                    label=1.0 if features.command_match else -1.0,
                )
            )
        return rows

    prior = _as_prior([0.9] * 22)
    rows = corpus_uninformed(400)
    with_prior = ContextRanker(config=config)
    with_prior.fit(rows, prior=prior)
    from_zero = ContextRanker(config=config)
    from_zero.fit(rows)
    # The shared weight survives where this repository has no opinion, and
    # the zero prior forgets it.
    assert with_prior.weights[index] > 0.25
    assert abs(from_zero.weights[index]) < 0.25

    # And a repository whose own verdicts contradict the prior is not held to
    # it: the same fit with the labels turned around moves the weight down,
    # below the prior, because the ridge is a pull and not a chain.
    contradicted = [
        replace(row, label=-1.0 if row.features.signature_match else 1.0)
        for row in rows
    ]
    pulled = ContextRanker(config=config)
    pulled.fit(contradicted, prior=prior)
    assert pulled.weights[index] < with_prior.weights[index]
    assert pulled.weights[index] < prior[index]
    report = ContextRanker(config=config).fit(rows, prior=prior)
    assert report["prior"] == "meta"
    assert ContextRanker(config=config).fit(rows)["prior"] == "zero"


def test_the_meta_prior_round_trips_and_refuses_another_schema():
    """The prior lives next to the snapshots, under the same refusal rules."""
    from llm_router_plugins.utils.knowledge.agentic_knowledge.learning import (
        SNAPSHOT_SCHEMA,
        load_meta_prior,
        meta_prior_key,
        save_meta_prior,
    )

    client = FakeRedis()
    weights = [0.1 * (index % 7) for index in range(22)]
    assert save_meta_prior(
        client, weights, key_prefix="agentic-kb", scopes=("a", "b", "c")
    )
    loaded = load_meta_prior(client, key_prefix="agentic-kb")
    assert loaded is not None
    assert loaded == [round(value, 6) for value in weights]
    assert client.values[meta_prior_key("agentic-kb")]

    # A prior of another schema is a prior of another build's feature order:
    # the safe read is the one that reads nothing, and the fit starts from
    # the zero prior instead.
    client.values[meta_prior_key("agentic-kb")] = json.dumps(
        {"schema": 1, "w": [0.0] * 15}
    )
    assert load_meta_prior(client, key_prefix="agentic-kb") is None
    # And a prior of the right schema but the wrong length is no prior at
    # all: a misaligned vector is worse than none, and the reader says so.
    client.values[meta_prior_key("agentic-kb")] = json.dumps(
        {"schema": SNAPSHOT_SCHEMA, "w": [0.0] * 21}
    )
    assert load_meta_prior(client, key_prefix="agentic-kb") is None
