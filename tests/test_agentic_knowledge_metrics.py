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
    assert emitted[0]["turns"] == 3
    # a rate with nothing behind it stays null, not 0
    assert emitted[0]["adoption_rate"] is None


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
