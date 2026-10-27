"""Forced documentation publication replaces only its scope in the stream."""

import json
import logging
import pathlib
import sys

import pytest
from redis.exceptions import ResponseError

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agentic_knowledge_support import FakeRedis, knowledge_config, knowledge_event
from llm_router_plugins.cli.plugins.utils.agentic_knowledge import docs
from llm_router_plugins.utils.knowledge.agentic_knowledge.docs_ingestor import (
    DocsIndexer,
    DocsScanState,
    prune_event,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.events import (
    KnowledgeEventPublisher,
)


class StreamRedis(FakeRedis):
    def __init__(self):
        super().__init__()
        self.entries = {}
        self.sequence = 0
        self.deleted = []

    def xadd(self, key, fields, **kwargs):
        super().xadd(key, fields, **kwargs)
        self.sequence += 1
        entry_id = f"{self.sequence}-0"
        self.entries[entry_id] = fields
        return entry_id

    def xrevrange(self, key, max="+", min="-", count=None):
        self._maybe_fail("xrevrange")
        return list(reversed(list(self.entries.items())))[:count]

    def xrange(self, key, min="-", max="+", count=None):
        self._maybe_fail("xrange")
        if min.startswith("("):
            raise ResponseError(
                "Invalid stream ID specified as stream command argument"
            )
        lower = tuple(map(int, min.split("-"))) if min != "-" else (0, 0)
        upper = (
            tuple(map(int, max.split("-"))) if max != "+" else (2**64 - 1, 2**64 - 1)
        )
        return [
            (entry_id, fields)
            for entry_id, fields in self.entries.items()
            if lower <= tuple(map(int, entry_id.split("-"))) <= upper
        ][:count]

    def xdel(self, key, *ids):
        self._maybe_fail("xdel")
        for entry_id in ids:
            self.entries.pop(entry_id, None)
            self.deleted.append(entry_id)
        return len(ids)


def setup_indexer(tmp_path):
    (tmp_path / "README.md").write_text("# Żółć\n\nFull documentation.\n")
    config = knowledge_config()
    redis = StreamRedis()
    publisher = KnowledgeEventPublisher(
        connection=config.redis,
        stream_key=config.stream_key,
        key_prefix=config.key_prefix,
        client=redis,
    )
    state = DocsScanState(key_prefix=config.key_prefix, client=redis)
    service = DocsIndexer(config=config, publisher=publisher, state=state)
    return service, publisher, redis


def test_cli_force_republishes_unchanged_files_and_replaces_old_stream_entries(
    tmp_path,
    monkeypatch,
    capsys,
):
    service, publisher, redis = setup_indexer(tmp_path)
    monkeypatch.setattr(docs, "_load_config", lambda *args: service.config)
    monkeypatch.setattr(docs, "_publisher", lambda *args: publisher)
    monkeypatch.setattr(docs, "DocsIndexer", lambda **kwargs: service)
    argv = ["--path", str(tmp_path), "--scope", "demo", "--json"]
    assert docs.main(argv) == 0
    assert json.loads(capsys.readouterr().out)["published"] == 1
    original = dict(redis.entries)
    assert docs.main(argv) == 0
    assert json.loads(capsys.readouterr().out)["unchanged"] == 1
    assert redis.entries == original

    assert docs.main(argv + ["--force"]) == 0
    outcome = json.loads(capsys.readouterr().out)
    assert outcome["published"] == 1
    assert outcome["unchanged"] == outcome["pruned"] == 0
    assert redis.deleted == list(original)
    assert len(redis.entries) == 1
    event = json.loads(next(iter(redis.entries.values()))["event"])
    assert event["scope"] == "demo" and event["kind"] == "doc"
    assert "Full documentation." in event["text"]
    assert docs.main(argv) == 0
    assert json.loads(capsys.readouterr().out)["published"] == 0


def test_force_replaces_docs_and_prunes_but_preserves_other_events(tmp_path):
    service, publisher, redis = setup_indexer(tmp_path)
    old = knowledge_event(kind="doc", scope="demo", fingerprint="old")
    prune = prune_event("demo", ["doc:20260101:old"])
    others = [
        knowledge_event(kind="doc", scope="other", fingerprint="other"),
        knowledge_event(kind="recipe", scope="demo", fingerprint="recipe"),
        knowledge_event(kind="session_archive", scope="demo", fingerprint="session"),
    ]
    publisher.publish([old, prune, *others])
    # More than one page, and undecodable entries must not break cleanup.
    for _ in range(205):
        redis.xadd(service.config.stream_key, {b"event": b"not json"})
    last = redis.xadd(
        service.config.stream_key, {b"event": json.dumps(old.to_dict()).encode()}
    )
    retained = {
        key: value
        for key, value in redis.entries.items()
        if key not in ("1-0", "2-0", last)
    }
    outcome = service.index("demo", str(tmp_path), force=True, republish=True)
    assert outcome.status == "ok" and outcome.published == 1
    assert set(redis.deleted) == {"1-0", "2-0", last}
    assert all(redis.entries[key] == value for key, value in retained.items())
    assert len(redis.entries) == len(retained) + 1
    for event in others:
        assert redis.get(publisher._seen_key(event)) == "1"


def test_force_clears_dedup_even_when_old_stream_was_trimmed(tmp_path):
    service, publisher, redis = setup_indexer(tmp_path)
    assert service.index("demo", str(tmp_path), force=True).published == 1
    redis.entries.clear()
    outcome = service.index("demo", str(tmp_path), force=True, republish=True)
    assert outcome.status == "ok" and outcome.published == 1
    assert len(redis.entries) == 1


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("overflow", [False, True])
def test_force_legacy_redis_pagination_handles_sequence_and_overflow(
    tmp_path, binary, overflow
):
    service, publisher, redis = setup_indexer(tmp_path)
    old = knowledge_event(kind="doc", scope="demo", fingerprint="old")
    other = knowledge_event(kind="doc", scope="other", fingerprint="other")
    # The retained page boundary must not be reread; the next page crosses
    # a sequence overflow, and the stream ends at the largest possible ID.
    ids = [f"10-{i}" for i in range(200)]
    if overflow:
        ids[-1] = f"10-{2**64 - 1}"
    else:
        ids.append("10-200")
    ids.extend(["11-0", f"{2**64 - 1}-{2**64 - 1}"])
    redis.entries = {
        entry_id: {"event": json.dumps((other if i == 199 else old).to_dict())}
        for i, entry_id in enumerate(ids)
    }
    if binary:
        reverse, forward, delete = redis.xrevrange, redis.xrange, redis.xdel
        redis.xrevrange = lambda *a, **kw: [
            (k.encode(), v) for k, v in reverse(*a, **kw)
        ]
        redis.xrange = lambda key, min, max, count: [
            (k.encode(), v)
            for k, v in forward(key, min=min, max=max.decode(), count=count)
        ]
        redis.xdel = lambda key, entry_id: delete(key, entry_id.decode())
    outcome = service.index("demo", str(tmp_path), republish=True)
    assert outcome.status == "ok" and outcome.published == 1
    assert redis.deleted == [entry_id for entry_id in ids if entry_id != ids[199]]
    assert set(redis.entries) == {ids[199], "1-0"}


def test_force_logs_redis_error_details(tmp_path, caplog):
    service, publisher, redis = setup_indexer(tmp_path)
    publisher._logger = logging.getLogger("force-test")

    def failed_range(*args, **kwargs):
        raise ResponseError(
            "WRONGTYPE Operation against a key holding the wrong kind of value"
        )

    redis.xrevrange = failed_range
    with caplog.at_level(logging.WARNING, logger="force-test"):
        outcome = service.index("demo", str(tmp_path), republish=True)
    assert outcome.status == "unavailable"
    assert "ResponseError: WRONGTYPE" in caplog.text
    assert redis.deleted == []


@pytest.mark.parametrize(
    "failure", ["xrevrange", "xrange", "xdel", "delete", "xadd"]
)
def test_force_failure_is_reported_and_can_be_retried(tmp_path, failure):
    service, publisher, redis = setup_indexer(tmp_path)
    service.index("demo", str(tmp_path), force=True)
    previous = redis.get(service.state.key("demo"))
    redis.fail_on.add(failure)
    outcome = service.index("demo", str(tmp_path), force=True, republish=True)
    assert outcome.status == "unavailable"
    assert redis.get(service.state.key("demo")) == previous
    redis.fail_on.clear()
    assert (
        service.index("demo", str(tmp_path), force=True, republish=True).published
        == 1
    )


@pytest.mark.parametrize("incomplete", ["truncated", "skipped"])
def test_force_does_not_remove_stream_after_incomplete_scan(
    tmp_path, monkeypatch, incomplete
):
    from llm_router_plugins.utils.knowledge.agentic_knowledge import docs_ingestor

    service, publisher, redis = setup_indexer(tmp_path)
    service.index("demo", str(tmp_path), force=True)
    original = dict(redis.entries)
    report = docs_ingestor.ScanReport(**{incomplete: True})
    monkeypatch.setattr(
        docs_ingestor, "scan_repository", lambda *args, **kwargs: report
    )
    outcome = service.index("demo", str(tmp_path), force=True, republish=True)
    assert outcome.status == "unavailable"
    assert redis.entries == original
    assert redis.deleted == []


@pytest.mark.parametrize(
    "ceiling", ["max_files", "max_file_bytes", "max_total_bytes"]
)
def test_cli_force_reads_full_files_beyond_automatic_scan_limits(
    tmp_path,
    monkeypatch,
    capsys,
    ceiling,
):
    service, publisher, redis = setup_indexer(tmp_path)
    service.index("demo", str(tmp_path), force=True)
    original = dict(redis.entries)
    text = "# Changelog\n\n" + "x" * 435197 + "\nEND OF FILE\n"
    (tmp_path / "CHANGELOG.md").write_text(text)
    (tmp_path / "guide.md").write_text("# Guide\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "excluded.md").write_text("Not documentation")
    setattr(service.config, "docs_" + ceiling, 1)
    monkeypatch.setattr(docs, "_load_config", lambda *args: service.config)
    monkeypatch.setattr(docs, "_publisher", lambda *args: publisher)
    monkeypatch.setattr(docs, "DocsIndexer", lambda **kwargs: service)
    argv = ["--path", str(tmp_path), "--scope", "demo", "--force"]

    assert docs.main(argv + ["--dry-run"]) == 0
    preview = capsys.readouterr().out
    assert "3 file(s), 3 document(s)" in preview
    assert "CHANGELOG.md" in preview and "excluded.md" not in preview
    assert redis.entries == original
    assert redis.deleted == []

    assert docs.main(argv + ["--json"]) == 0
    outcome = json.loads(capsys.readouterr().out)
    assert outcome["status"] == "ok"
    assert outcome["files"] == outcome["published"] == 3
    assert redis.deleted == list(original)
    events = [json.loads(fields["event"]) for fields in redis.entries.values()]
    changelog = next(
        event
        for event in events
        if event["metadata"]["source_file"] == "CHANGELOG.md"
    )
    assert text in changelog["text"]
    assert service.config.docs_max_file_bytes == (
        1 if ceiling == "max_file_bytes" else 200000
    )


@pytest.mark.parametrize("failure", ["read", "stat", "walk"])
def test_force_read_errors_leave_stream_and_state_untouched(
    tmp_path, monkeypatch, failure
):
    from llm_router_plugins.utils.knowledge.agentic_knowledge import docs_ingestor

    service, publisher, redis = setup_indexer(tmp_path)
    service.index("demo", str(tmp_path), force=True)
    original = dict(redis.entries)
    previous = redis.get(service.state.key("demo"))
    if failure == "read":
        monkeypatch.setattr(docs_ingestor, "_read_text", lambda path: None)
    elif failure == "stat":
        real_stat = docs_ingestor.os.stat

        def failed_stat(path, *args, **kwargs):
            if str(path).endswith("README.md"):
                raise PermissionError("unreadable file")
            return real_stat(path, *args, **kwargs)

        monkeypatch.setattr(docs_ingestor.os, "stat", failed_stat)
    else:
        real_walk = docs_ingestor.os.walk

        def failed_walk(*args, **kwargs):
            yield from real_walk(*args, **kwargs)
            if kwargs.get("onerror"):
                kwargs["onerror"](PermissionError("unreadable directory"))

        monkeypatch.setattr(docs_ingestor.os, "walk", failed_walk)
    outcome = service.index("demo", str(tmp_path), republish=True)
    assert outcome.status == "unavailable"
    assert "unreadable" in outcome.reason
    assert redis.entries == original
    assert redis.deleted == []
    assert redis.get(service.state.key("demo")) == previous


def test_force_dry_run_does_not_connect_to_redis(tmp_path, monkeypatch):
    setup_indexer(tmp_path)

    def forbidden(*args):
        raise AssertionError("dry-run accessed Redis")

    monkeypatch.setattr(docs, "_publisher", forbidden)
    assert docs.main(["--path", str(tmp_path), "--force", "--dry-run"]) == 0


def test_force_partial_publish_does_not_save_complete_scan(tmp_path):
    service, publisher, redis = setup_indexer(tmp_path)
    (tmp_path / "guide.md").write_text("# More documentation\n")
    append = redis.xadd

    def fail_second(*args, **kwargs):
        if redis.sequence:
            raise RuntimeError("connection lost")
        return append(*args, **kwargs)

    redis.xadd = fail_second
    outcome = service.index("demo", str(tmp_path), force=True, republish=True)
    assert outcome.status == "unavailable"
    assert service.state.load("demo") == {}
    redis.xadd = append
    assert (
        service.index("demo", str(tmp_path), force=True, republish=True).published
        == 2
    )


def test_force_empty_repository_removes_only_previous_docs(tmp_path):
    service, publisher, redis = setup_indexer(tmp_path)
    service.index("demo", str(tmp_path), force=True)
    (tmp_path / "README.md").unlink()
    outcome = service.index(
        "demo", str(tmp_path), force=True, republish=True, allow_prune=False
    )
    assert outcome.status == "ok" and outcome.published == 0
    assert redis.entries == {}
    assert service.state.load("demo") == {}


def test_force_still_requires_redis_configuration(tmp_path, monkeypatch):
    setup_indexer(tmp_path)
    monkeypatch.delenv("LLM_ROUTER_KNOWLEDGE_AGENTIC_REDIS_HOST", raising=False)
    with pytest.raises(SystemExit) as exc:
        docs.main(["--path", str(tmp_path), "--force"])
    assert "REDIS_HOST is unset" in str(exc.value)
    assert "does not load .env files automatically" in str(exc.value)


def test_force_rebuilds_deleted_database_through_worker(tmp_path, monkeypatch):
    import time

    from test_agentic_knowledge_upload import UploadSSE, make_worker

    # Compare complete upload payloads with a deterministic publication time.
    stamp = time.time()
    monkeypatch.setattr(time, "time", lambda: stamp)
    service, publisher, redis = setup_indexer(tmp_path)
    service.index("demo", str(tmp_path), force=True)
    original = UploadSSE()
    assert make_worker(original)._process_batch(list(redis.entries.items())) == (
        1,
        1,
    )
    assert len(original.uploads) == 1

    # SSE was rebuilt while Redis still remembers the successful first scan.
    empty_database = UploadSSE()
    outcome = service.index("demo", str(tmp_path), republish=True)
    assert outcome.published == 1
    assert make_worker(empty_database)._process_batch(
        list(redis.entries.items())
    ) == (1, 1)
    # The worker creates the collection the layout it reads names: one
    # collection per repository, not one per document kind.
    assert empty_database.created == [("agentic_kb_demo", "embed", "rerank")]
    assert empty_database.uploads == original.uploads


def test_replacement_rejects_events_outside_selected_docs_scope(tmp_path):
    service, publisher, redis = setup_indexer(tmp_path)
    service.index("demo", str(tmp_path), force=True)
    original = dict(redis.entries)
    for event in (
        knowledge_event(kind="doc", scope="other"),
        knowledge_event(kind="recipe", scope="demo"),
    ):
        assert publisher.replace_docs("demo", [event]).status == "unavailable"
        assert redis.entries == original
    assert publisher.replace_docs("", []).status == "unavailable"
    assert redis.entries == original


def test_replacement_supports_binary_stream_ids(tmp_path):
    service, publisher, redis = setup_indexer(tmp_path)
    service.index("demo", str(tmp_path), force=True)
    reverse_range, forward_range, delete = redis.xrevrange, redis.xrange, redis.xdel

    def binary_reverse(*args, **kwargs):
        return [
            (key.encode(), fields) for key, fields in reverse_range(*args, **kwargs)
        ]

    def binary_forward(key, min, max, count):
        return [
            (entry_id.encode(), fields)
            for entry_id, fields in forward_range(
                key, min=min, max=max.decode(), count=count
            )
        ]

    redis.xrevrange = binary_reverse
    redis.xrange = binary_forward
    redis.xdel = lambda key, entry_id: delete(key, entry_id.decode())
    assert service.index("demo", str(tmp_path), republish=True).published == 1
    assert redis.deleted == ["1-0"]
    assert list(redis.entries) == ["2-0"]
