"""
The scoreboard: what a request counts as, and what the rates mean.

A counter is worth reading only if the denominators are the ones an operator
assumes, so these tests pin the unit (a turn, not a search), the empty case
(``None``, never a ``0.0`` that reads as a measurement), and the one property
that makes the layer safe to instrument: a scoreboard that cannot be written,
or cannot be reached, costs a metric and never a request.
"""

import datetime
import json
import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from llm_router_plugins.utils.knowledge.agentic_knowledge.metrics import (
    FIELDS,
    KnowledgeMetrics,
)


class FakePipeline:
    def __init__(self, store, log):
        self._store = store
        self._log = log

    def hincrby(self, key, field, amount):
        self._log.append(("hincrby", key, field, amount))
        bucket = self._store.setdefault(key, {})
        bucket[field] = int(bucket.get(field, 0)) + int(amount)

    def expire(self, key, ttl):
        self._log.append(("expire", key, ttl))

    def execute(self):
        self._log.append(("execute",))


class FakeRedis:
    """A hash-only Redis double with the one pipeline the recorder uses."""

    def __init__(self, fail=False):
        self.data = {}
        self.calls = []
        self.fail = fail

    def pipeline(self):
        if self.fail:
            raise RuntimeError("redis is down")
        return FakePipeline(self.data, self.calls)

    def hgetall(self, key):
        if self.fail:
            raise RuntimeError("redis is down")
        return dict(self.data.get(key, {}))


# ------------------------------------------------------------------ counting -


def test_counters_accumulate_on_one_scope_day():
    metrics = KnowledgeMetrics()
    metrics.record("webapp", turns=1, injected_turns=1, injected_docs=3)
    metrics.record("webapp", turns=1, injected_turns=0, injected_docs=0, loops=1)
    snap = metrics.snapshot("webapp")
    assert snap["turns"] == 2
    assert snap["injected_turns"] == 1
    assert snap["injected_docs"] == 3
    assert snap["loops"] == 1


def test_a_scope_counts_on_its_own_scoreboard():
    metrics = KnowledgeMetrics()
    metrics.record("webapp", turns=3)
    metrics.record("other", turns=5)
    assert metrics.snapshot("webapp")["turns"] == 3
    assert metrics.snapshot("other")["turns"] == 5


def test_a_field_the_scoreboard_does_not_own_is_dropped():
    # The caller is the request path: a renamed counter must not become a
    # failed request, and must not smuggle an unknown field into the hash.
    metrics = KnowledgeMetrics()
    metrics.record("webapp", turns=1, turns_typo=99)
    snap = metrics.snapshot("webapp")
    assert snap["turns"] == 1
    assert set(snap) == set(FIELDS)


def test_an_empty_scope_records_nothing():
    metrics = KnowledgeMetrics()
    metrics.record("", turns=1)
    assert metrics.snapshot("") == {}


# -------------------------------------------------------------------- rates --


def test_a_row_names_the_scope_it_measured():
    # A report over several repositories is only readable when every row says
    # which repository it describes.
    metrics = KnowledgeMetrics()
    metrics.record("webapp", turns=2)
    assert metrics.derived("webapp")["scope"] == "webapp"


def test_derived_reports_the_rates_a_deployment_reads():
    metrics = KnowledgeMetrics()
    metrics.record(
        "webapp",
        turns=10,
        injected_turns=4,
        failure_turns=5,
        loops=2,
        escalations=1,
        green_after_injection=3,
        feedback_up=3,
        feedback_down=1,
    )
    derived = metrics.derived("webapp")
    assert derived["adoption_rate"] == 0.75  # 3 of the 4 turns shown knowledge
    assert derived["knowledge_help_rate"] == 0.75  # 3 of 4 settled documents
    # a stumble and a loop are reported apart: 5 turns met a failure, 2 of
    # them met it as a loop
    assert derived["failure_turn_share"] == 0.5
    assert derived["loop_turn_share"] == 0.2
    assert derived["escalation_rate"] == 0.1


def test_a_rate_with_nothing_to_measure_is_none_not_zero():
    # A day with no injected turn did not measure "0% adoption", it measured
    # nothing; reporting 0.0 would let a quiet day read as a bad one.
    derived = KnowledgeMetrics().derived("quiet-repo")
    assert derived["adoption_rate"] is None
    assert derived["knowledge_help_rate"] is None
    assert derived["loop_turn_share"] is None
    assert derived["escalation_rate"] is None
    # the counts themselves are a real zero, though
    assert derived["turns"] == 0


# ------------------------------------------------------------------- redis ---


def test_the_counters_reach_redis_with_a_lifetime():
    redis = FakeRedis()
    metrics = KnowledgeMetrics(client=redis, ttl_seconds=99)
    metrics.record("webapp", turns=1, injected_turns=1)
    written = [call for call in redis.calls if call[0] == "hincrby"]
    assert len(written) == 2
    assert ("expire", written[0][1], 99) in redis.calls


def test_a_store_that_cannot_be_written_costs_a_metric_not_a_request():
    metrics = KnowledgeMetrics(client=FakeRedis(fail=True))
    metrics.record("webapp", turns=1)  # must not raise
    # the in-process half still counted, so the run is measurable
    assert metrics.snapshot("webapp")["turns"] == 1


def test_an_unreachable_store_falls_back_to_the_in_process_half():
    redis = FakeRedis()
    metrics = KnowledgeMetrics(client=redis)
    metrics.record("webapp", turns=2)
    redis.fail = True  # the write succeeded, the read does not answer
    assert metrics.snapshot("webapp")["turns"] == 2


def test_a_stored_count_and_its_local_twin_are_not_summed():
    # Redis and the local dict hold the same writes seen from two sides; the
    # read is the larger half, and summing would double the request path.
    redis = FakeRedis()
    metrics = KnowledgeMetrics(client=redis)
    metrics.record("webapp", turns=4)
    key = metrics._key("webapp")
    redis.data[key]["turns"] = 6  # another process wrote more since
    assert metrics.snapshot("webapp")["turns"] == 6


# ---------------------------------------------------------------- reader CLI -


class RecordingReader:
    """A scoreboard the command can be pointed at instead of a live Redis."""

    def __init__(self, rows):
        self.rows = dict(rows)
        self.reads = []

    def derived(self, scope, day=""):
        self.reads.append((scope, day))
        return self.rows.get((scope, day), _empty_row(scope, day))


def _empty_row(scope, day):
    return {
        "scope": scope,
        "day": day,
        "turns": 0,
        "injected_turns": 0,
        "injected_docs": 0,
        "loops": 0,
        "escalations": 0,
        "read_failures": 0,
        "failure_turns": 0,
        "adoption_rate": None,
        "knowledge_help_rate": None,
        "failure_turn_share": None,
        "loop_turn_share": None,
        "escalation_rate": None,
    }


def test_the_reader_reads_the_scope_it_was_given(capsys):
    from llm_router_plugins.cli.plugins.utils.agentic_knowledge.stats import main

    row = _empty_row("webapp", "20261116")
    row.update({"turns": 10, "injected_turns": 4, "adoption_rate": 0.75})
    reader = RecordingReader({("webapp", "20261116"): row})
    code = main(
        ["--scope", "webapp", "--day", "20261116", "--config", _policy()],
        metrics=reader,
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "webapp" in out
    assert "75.0%" in out
    assert reader.reads == [("webapp", "20261116")]


def test_the_reader_emits_json_for_a_dashboard(capsys):
    import json as json_module

    from llm_router_plugins.cli.plugins.utils.agentic_knowledge.stats import main

    row = _empty_row("webapp", "20261116")
    row["turns"] = 3
    reader = RecordingReader({("webapp", "20261116"): row})
    code = main(
        [
            "--scope",
            "webapp",
            "--day",
            "20261116",
            "--json",
            "--config",
            _policy(),
        ],
        metrics=reader,
    )
    assert code == 0
    emitted = json_module.loads(capsys.readouterr().out)
    row = emitted["scoreboard"][0]
    assert row["turns"] == 3
    # a rate with nothing behind it stays null, not 0
    assert row["adoption_rate"] is None
    # without the flag there is no learned state, and the scoreboard stays a
    # list of rows rather than a wrapper: a dashboard that predates the
    # flag reads the same shape it always read.
    assert "learning" not in emitted


def test_a_window_of_days_is_read_oldest_first():
    from llm_router_plugins.cli.plugins.utils.agentic_knowledge.stats import _days

    assert _days("20261116", 3) == ["20261114", "20261115", "20261116"]
    assert _days("", 1)[0] == datetime.date.today().strftime("%Y%m%d")


def test_several_scopes_are_read_in_one_pass(capsys):
    from llm_router_plugins.cli.plugins.utils.agentic_knowledge.stats import main

    rows = {
        ("webapp", "20261116"): _empty_row("webapp", "20261116"),
        ("other", "20261116"): _empty_row("other", "20261116"),
    }
    reader = RecordingReader(rows)
    code = main(
        [
            "--scope",
            "webapp",
            "--scope",
            "other",
            "--day",
            "20261116",
            "--config",
            _policy(),
        ],
        metrics=reader,
    )
    assert code == 0
    assert [scope for scope, _ in reader.reads] == ["webapp", "other"]


def _policy() -> str:
    """A policy file on disk, so the command reads one exactly as the proxy does."""
    import tempfile

    directory = tempfile.mkdtemp()
    path = pathlib.Path(directory) / "policy.json"
    path.write_text(json.dumps({"collection_prefix": "agentic_kb"}))
    return str(path)


# ------------------------------------------------------------- learning ----
class _LearnedStore:
    """A client carrying the learned state of one scope, for the view."""

    def __init__(self, prefix="agentic-kb", stream_rows=()):
        self.kv = {}
        self.hashes = {}
        self.stream = list(stream_rows)

    def get(self, key):
        return self.kv.get(key)

    def set(self, key, value, **_):
        self.kv[key] = value
        return True

    def hset(self, key, *args, mapping=None, **_):
        bucket = self.hashes.setdefault(key, {})
        if mapping:
            for name, payload in mapping.items():
                bucket[name] = payload
        return 1

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))

    def hincrby(self, key, field, amount=1):
        bucket = self.hashes.setdefault(key, {})
        current = int(bucket.get(field, "0")) + int(amount)
        bucket[field] = str(current)
        return current

    def expire(self, key, seconds):
        return True

    def xrange(self, key, min="-", max="+", count=0):
        rows = list(self.stream)
        return rows[:count] if count else rows


def _policy_with_learning(**overrides) -> str:
    """A policy file that carries a learning section, as the proxy ships it."""
    import tempfile

    learning = {"enabled": True, "min_samples": 4, "snapshot_history_size": 3}
    learning.update(overrides)
    directory = tempfile.mkdtemp()
    path = pathlib.Path(directory) / "policy.json"
    path.write_text(
        json.dumps({"collection_prefix": "agentic_kb", "learning": learning})
    )
    return str(path)


def test_the_learning_view_reports_what_is_serving_and_what_it_rests_on():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.learning import (
        publish_snapshot,
        train_ranker,
    )
    from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
        AgenticKnowledgeConfig,
    )
    from llm_router_plugins.cli.plugins.utils.agentic_knowledge.stats import (
        learning_view,
    )

    store = _LearnedStore()
    config = AgenticKnowledgeConfig.from_file(pathlib.Path(_policy_with_learning()))
    config.validate()
    learning = config.learning

    # Two fits of the scope, the second replacing the first, and a corpus the
    # next fit would stand on.
    from llm_router_plugins.utils.knowledge.agentic_knowledge.learning import (
        FeedbackSample,
        CandidateFeatures,
        LabelStore,
    )

    samples = []
    for index in range(12):
        matched = index % 2 == 0
        samples.append(
            FeedbackSample(
                scope="webapp",
                session_id=f"s{index % 3}",
                generation=f"g{index}",
                document_name=f"recipe:20260101:{index:05d}",
                features=CandidateFeatures(
                    kind="recipe",
                    mode="debug",
                    score=0.5,
                    score_scale="cosine",
                    signature_match=matched,
                    rank=index % 5,
                ),
                label=1.0 if matched else -1.0,
            )
        )
    first, _ = train_ranker(
        samples, "webapp", config=learning, rng=__import__("random").Random(1), now=100.0
    )
    publish_snapshot(store, first, key_prefix="agentic-kb")
    second, _ = train_ranker(
        samples, "webapp", config=learning, rng=__import__("random").Random(2), now=200.0
    )
    publish_snapshot(store, second, key_prefix="agentic-kb")
    store_labels = LabelStore(client=store, key_prefix="agentic-kb")
    store_labels.merge(samples)

    view = learning_view("webapp", config, store)
    assert view["enabled"] is True
    # The model in force is the second fit, and it says what it rests on.
    assert view["model"]["model_id"] == second.model_id
    assert view["model"]["labelled"] == second.labelled
    # The first fit is retired, in the history a rollback could reach.
    assert view["history"]["count"] == 1
    assert view["history"]["kept"] == 3
    assert view["history"]["newest"] == [first.model_id]
    # The corpus the next fit would stand on.
    assert view["corpus"]["records"] == 12
    assert view["corpus"]["labelled"] == 12


def test_the_learning_view_reads_without_a_redis_as_saying_so():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
        AgenticKnowledgeConfig,
    )
    from llm_router_plugins.cli.plugins.utils.agentic_knowledge.stats import (
        learning_view,
    )

    config = AgenticKnowledgeConfig.from_file(
        pathlib.Path(_policy_with_learning())
    )
    config.validate()
    view = learning_view("webapp", config, None)
    assert view["enabled"] is True
    assert "no Redis" in view.get("note", "")
    assert "model" not in view


def test_the_learning_monitors_read_the_shape_of_the_log():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
        AgenticKnowledgeConfig,
    )
    from llm_router_plugins.cli.plugins.utils.agentic_knowledge.stats import (
        learning_view,
    )

    rows = []
    # A document the layer has shown 60 times, and one it has shown twice:
    # a self-reinforcing layer names the first one, and the monitor says so.
    for index in range(60):
        rows.append(
            (
                f"{index + 1}-0",
                {
                    "d": json.dumps(
                        {
                            "scope": "webapp",
                            "session": f"s{index % 2}",
                            "gen": f"g{index}",
                            "doc": "recipe:20260101:heavy",
                            "kind": "injection",
                            "label": 0,
                            "f": {"kind": "recipe", "mode": "debug"},
                        }
                    )
                },
            )
        )
    for index in range(2):
        rows.append(
            (
                f"{100 + index}-0",
                {
                    "d": json.dumps(
                        {
                            "scope": "webapp",
                            "session": f"s{index}",
                            "gen": f"g{index}",
                            "doc": "recipe:20260101:light",
                            "kind": "injection",
                            "label": 0,
                            "f": {"kind": "recipe", "mode": "debug"},
                        }
                    )
                },
            )
        )
    # A row of another scope: the view is about this layer, not the log.
    rows.append(
        (
            "200-0",
            {
                "d": json.dumps(
                    {
                        "scope": "other",
                        "session": "s0",
                        "gen": "g0",
                        "doc": "recipe:20260101:elsewhere",
                        "kind": "injection",
                        "label": 0,
                        "f": {"kind": "recipe", "mode": "debug"},
                    }
                )
            },
        )
    )
    store = _LearnedStore(stream_rows=rows)
    config = AgenticKnowledgeConfig.from_file(
        pathlib.Path(_policy_with_learning())
    )
    config.validate()
    view = learning_view("webapp", config, store)
    monitors = view["monitors"]
    assert monitors["window"] == 62
    assert monitors["docs"] == 2
    # Half of the documents the log names are the heavy one.
    assert monitors["docs_over_50_exposures"] == 0.5
    assert monitors["heaviest"] == 60


def test_the_log_reports_the_share_of_it_the_layer_spent_exploring():
    """
    The explored share is read off the records, not guessed from a config.

    A fit reads what the log holds, and how much of that was a draw rather
    than a decision is the difference between evidence and an echo: a greedy
    vote is one the serving policy would have cast anyway.  The number is
    therefore counted from the ``exploration`` field of the records
    themselves — a deployment whose ``explore_rate`` says 0.1 and whose log
    says 0.0 has an exploration term that never fires, and only the log says
    so.
    """
    from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
        AgenticKnowledgeConfig,
    )
    from llm_router_plugins.cli.plugins.utils.agentic_knowledge.stats import (
        learning_view,
    )

    def row(entry_id, explored):
        return (
            f"{entry_id}-0",
            {
                "d": json.dumps(
                    {
                        "scope": "webapp",
                        "session": "s0",
                        "generation": f"g{entry_id}",
                        "doc": f"recipe:20260101:{entry_id:03d}",
                        "kind": "injection",
                        "label": 0,
                        "f": {"kind": "recipe", "mode": "debug"},
                        "exploration": "explore" if explored else "greedy",
                    }
                )
            },
        )

    rows = [row(index, index % 4 == 0) for index in range(20)]
    store = _LearnedStore(stream_rows=rows)
    config = AgenticKnowledgeConfig.from_file(
        pathlib.Path(_policy_with_learning())
    )
    config.validate()
    monitors = learning_view("webapp", config, store)["monitors"]
    # Five of the twenty drew the spread.
    assert monitors["window"] == 20
    assert monitors["explored_share"] == 0.25


def test_an_old_record_counts_as_greedy_rather_than_as_unknown():
    """
    A record written before the exploration field existed is a greedy one.

    The field arrives with a deployment, not with a fit, and a stream keeps
    what it kept: a reader that treated a missing field as "unknown" would
    either inflate the explored share with rows that explored nothing or
    divide by a window it had just silently shrunk.  The default is the
    behaviour those records were written under.
    """
    from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
        AgenticKnowledgeConfig,
    )
    from llm_router_plugins.cli.plugins.utils.agentic_knowledge.stats import (
        learning_view,
    )

    rows = [
        (
            f"{index}-0",
            {
                "d": json.dumps(
                    {
                        "scope": "webapp",
                        "session": "s0",
                        "generation": f"g{index}",
                        "doc": f"recipe:20260101:{index:03d}",
                        "kind": "injection",
                        "label": 0,
                        "f": {"kind": "recipe", "mode": "debug"},
                    }
                )
            },
        )
        for index in range(5)
    ]
    store = _LearnedStore(stream_rows=rows)
    config = AgenticKnowledgeConfig.from_file(
        pathlib.Path(_policy_with_learning())
    )
    config.validate()
    monitors = learning_view("webapp", config, store)["monitors"]
    assert monitors["window"] == 5
    assert monitors["explored_share"] == 0.0


def test_the_learning_view_keeps_the_shape_of_the_label_corpus():
    """
    The view reports the feature distribution the next fit would read.

    A fit that used to clear its holdout and now refuses it is usually a fit
    whose input moved, and "the input moved" is a claim about a distribution
    that has to have been written down before it is needed.  Two features of
    the same corpus are pinned here for that reason: one the rows vary, which
    carries a non-zero spread, and one they never vary, whose spread is
    exactly zero — the number that tells an operator the fit will prune it.
    """
    from llm_router_plugins.utils.knowledge.agentic_knowledge.learning import (
        FEATURE_NAMES,
        CandidateFeatures,
        FeedbackSample,
        LabelStore,
    )
    from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
        AgenticKnowledgeConfig,
    )
    from llm_router_plugins.cli.plugins.utils.agentic_knowledge.stats import (
        learning_view,
    )

    store = _LearnedStore()
    config = AgenticKnowledgeConfig.from_file(
        pathlib.Path(_policy_with_learning())
    )
    config.validate()

    samples = []
    for index in range(4):
        samples.append(
            FeedbackSample(
                scope="webapp",
                session_id=f"s{index}",
                generation=f"g{index}",
                document_name=f"recipe:20260101:{index:05d}",
                features=CandidateFeatures(
                    kind="recipe",
                    mode="debug",
                    # Varies with the row: the score is clamped straight into
                    # the vector, so what the row says is what the fit reads.
                    score=index / 4.0,
                    # Never varies: every row came back on the same scale.
                    score_scale="cosine",
                ),
                label=1.0 if index % 2 == 0 else -1.0,
            )
        )
    LabelStore(client=store, key_prefix="agentic-kb").merge(samples)

    corpus = learning_view("webapp", config, store)["corpus"]
    features = corpus["features"]
    assert set(features) == set(FEATURE_NAMES)
    # score is 0.0, 0.25, 0.5, 0.75: mean 0.375, population standard
    # deviation 0.27950849...
    assert features["score"]["mean"] == pytest.approx(0.375)
    assert features["score"]["std"] == pytest.approx(0.27950849, abs=1e-5)
    # score_sq moves with it, so its spread is non-zero too.
    assert features["score_sq"]["std"] > 0.0
    # The scale never moves, and the view says so with a zero rather than a
    # gap: zero spread is the number that tells an operator the fit prunes it.
    assert features["scale_rrf"]["mean"] == 0.0
    assert features["scale_rrf"]["std"] == 0.0
    assert features["scale_reranker"]["mean"] == 0.0
    assert features["scale_reranker"]["std"] == 0.0


def test_a_corpus_with_nothing_labelled_reports_no_distribution():
    """
    No labelled rows is "nobody measured", not "every feature sits at zero".

    Unlabelled exposures are the majority of a young scope's log, and a
    distribution over them describes the traffic the layer was shown, not the
    corpus a fit reads — so the view leaves the field empty rather than
    printing the shape of the wrong thing.
    """
    from llm_router_plugins.utils.knowledge.agentic_knowledge.learning import (
        CandidateFeatures,
        FeedbackSample,
        LabelStore,
    )
    from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
        AgenticKnowledgeConfig,
    )
    from llm_router_plugins.cli.plugins.utils.agentic_knowledge.stats import (
        learning_view,
    )

    store = _LearnedStore()
    config = AgenticKnowledgeConfig.from_file(
        pathlib.Path(_policy_with_learning())
    )
    config.validate()
    LabelStore(client=store, key_prefix="agentic-kb").merge(
        [
            FeedbackSample(
                scope="webapp",
                session_id="s0",
                generation="g0",
                document_name="recipe:20260101:00001",
                features=CandidateFeatures(kind="recipe", mode="debug"),
                label=0.0,
            )
        ]
    )

    corpus = learning_view("webapp", config, store)["corpus"]
    # The store holds no verdicts at all: records, labelled and the
    # distribution are all zero, and the distribution says "nobody measured"
    # with a None rather than a shape over an empty thing.
    assert corpus["records"] == 0
    assert corpus["labelled"] == 0
    assert corpus["features"] is None


def test_the_reader_emits_the_learning_view_with_a_flag(capsys, monkeypatch):
    from llm_router_plugins.cli.plugins.utils.agentic_knowledge.stats import (
        main,
    )

    store = _LearnedStore()
    monkeypatch.setattr(
        "llm_router_plugins.cli.plugins.utils.agentic_knowledge.stats._redis_client",
        lambda config: store,
    )
    row = _empty_row("webapp", "20261116")
    reader = RecordingReader({("webapp", "20261116"): row})
    code = main(
        [
            "--scope",
            "webapp",
            "--day",
            "20261116",
            "--json",
            "--learning",
            "--config",
            _policy_with_learning(),
        ],
        metrics=reader,
    )
    assert code == 0
    emitted = json.loads(capsys.readouterr().out)
    assert "scoreboard" in emitted
    learning = emitted["learning"][0]
    assert learning["scope"] == "webapp"
    assert learning["enabled"] is True
