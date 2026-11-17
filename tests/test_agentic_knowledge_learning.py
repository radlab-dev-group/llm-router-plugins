"""
Deterministic tests for the learned path.

Every random draw is behind an injected ``random.Random(seed)`` and every clock
behind an injected ``time_fn``, because the one thing this module must not do is
make ``evaluation.py`` stop reproducing itself.  A test here that flakes is a
model that flaps.
"""

import json
import random
import time

from dataclasses import replace


import pytest

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
    train_ranker,
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

    def set(self, key, value):
        self.calls.append(("set", key))
        self.values[key] = value
        return True

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
    client = FakeRedis()
    client.set("agentic-kb:model:repo", "{not json")
    assert load_snapshot(client, "repo", config=LearningConfig()) is None
    client.set("agentic-kb:model:repo", json.dumps({"w": [1.0, 2.0]}))
    loaded = load_snapshot(client, "repo", config=LearningConfig())
    assert loaded is not None
    assert loaded.neutral is True


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
