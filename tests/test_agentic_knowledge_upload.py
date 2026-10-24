"""Whole documentation files travel through SSE's multipart upload contract."""

import dataclasses
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agentic_knowledge_support import (
    FakeRedis,
    FakeSSE,
    knowledge_config,
    knowledge_event,
    stream_message,
)
from llm_router_plugins.cli.plugins.utils.agentic_knowledge.worker import (
    KnowledgeIndexWorker,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.docs_ingestor import (
    doc_events,
    prune_event,
    scan_repository,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.documents import (
    parse_document_text,
)


class UploadSSE(FakeSSE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.uploads = []
        self.creation_options = []

    def create_collection(self, name, **kwargs):
        self.creation_options.append(kwargs)
        return super().create_collection(name, **kwargs)

    def upload_files(
        self, collection, files, *, indexing_options=None, file_metadata=None
    ):
        self._maybe_fail("upload_files")
        self.uploads.append((collection, files, indexing_options, file_metadata))


def make_worker(sse, redis=None, chunk_tokens=400, chunk_overlap=60):
    config = knowledge_config()
    config.worker = dataclasses.replace(
        config.worker,
        embedder="embed",
        reranker="rerank",
        chunk_tokens=chunk_tokens,
        chunk_overlap=chunk_overlap,
    )
    return KnowledgeIndexWorker(
        config=config,
        redis_client=redis or FakeRedis(),
        sse_client=sse,
    )


@pytest.mark.parametrize("entrypoint", ["batch", "single"])
@pytest.mark.parametrize(
    "chunk_tokens,chunk_overlap", [(400, 60), (900, 120), (0, 0)]
)
def test_full_file_upload_uses_collection_chunking(
    tmp_path, entrypoint, chunk_tokens, chunk_overlap
):
    text = "# Guide\n\n" + "## Detail\n\nZażółć gęślą jaźń.\n\n" * 500
    (tmp_path / "guide.md").write_text(text, encoding="utf-8")
    events = doc_events(
        scan_repository(str(tmp_path), scope="demo").records, scope="demo"
    )
    assert len(events) == 1
    sse = UploadSSE()
    redis = FakeRedis()
    worker = make_worker(sse, redis, chunk_tokens, chunk_overlap)
    if entrypoint == "single":
        worker.index_event(events[0])
    else:
        assert worker._process_batch([("1-0", stream_message(events[0]))]) == (1, 1)
        assert redis.acked == ["1-0"]
    assert sse.indexed == []
    assert len(sse.uploads) == 1
    collection, files, options, metadata = sse.uploads[0]
    assert collection == "agentic_kb_demo_docs"
    assert len(files) == 1
    name, content, mime = files[0]
    assert name.startswith("doc:") and name.endswith(".txt")
    assert mime == "text/plain"
    header, body = parse_document_text(content.decode("utf-8"))
    # The document envelope normalizes outer whitespace and ends with a newline.
    assert body == text.strip() + "\n"
    assert header.source == "guide.md"
    assert metadata[name]["category"] == "doc"
    assert metadata[name]["options"]["source_file"] == "guide.md"
    assert metadata[name]["options"]["scope"] == "demo"
    assert options == sse.creation_options[0]["chunking"]
    assert options == (
        {
            "max_tokens_in_chunk": chunk_tokens,
            "number_of_overlap_tokens": chunk_overlap,
        }
        if chunk_tokens
        else None
    )


def test_multiple_files_share_one_upload_per_collection():
    events = [
        knowledge_event(kind="doc", fingerprint=str(i), scope=scope)
        for i, scope in enumerate(("alpha", "alpha", "beta"))
    ]
    sse = UploadSSE()
    redis = FakeRedis()
    worker = make_worker(sse, redis)
    assert worker._process_batch(
        [(str(i), stream_message(e)) for i, e in enumerate(events)]
    ) == (3, 3)
    assert [(call[0], len(call[1])) for call in sse.uploads] == [
        ("agentic_kb_alpha_docs", 2),
        ("agentic_kb_beta_docs", 1),
    ]
    assert redis.acked == ["0", "1", "2"]
    assert sse.indexed == []


def test_documents_and_mined_events_use_their_own_endpoints():
    sse = UploadSSE()
    worker = make_worker(sse)
    events = [
        knowledge_event(kind=kind) for kind in ("doc", "recipe", "session_archive")
    ]
    assert worker._process_batch(
        [(str(i), stream_message(e)) for i, e in enumerate(events)]
    ) == (3, 3)
    assert len(sse.uploads) == 1
    assert [call[0] for call in sse.indexed] == [
        "agentic_kb_webapp",
        "agentic_kb_webapp_sessions",
    ]


def test_duplicate_deliveries_do_not_duplicate_attachments():
    sse = UploadSSE()
    redis = FakeRedis()
    worker = make_worker(sse, redis)
    message = stream_message(knowledge_event(kind="doc"))
    assert worker._process_batch([("1", message), ("2", message)]) == (2, 2)
    assert len(sse.uploads) == 1
    assert len(sse.uploads[0][1]) == 1
    assert redis.acked == ["1", "2"]


@pytest.mark.parametrize("target", ["text", "metadata"])
def test_conflicting_attachments_are_not_uploaded_or_acknowledged(target):
    event = knowledge_event(kind="doc")
    replacement = (
        "different content" if target == "text" else {"source_file": "other.md"}
    )
    conflicting = dataclasses.replace(event, **{target: replacement})
    sse = UploadSSE()
    redis = FakeRedis()
    worker = make_worker(sse, redis)
    entries = [("1", stream_message(event)), ("2", stream_message(conflicting))]
    assert worker._process_batch(entries) == (2, 0)
    assert sse.uploads == sse.indexed == []
    assert redis.acked == []
    assert all(redis.kv[worker._tries_key(entry_id)] == 1 for entry_id, _ in entries)


def test_existing_collection_still_receives_configured_chunking():
    sse = UploadSSE(exists=("agentic_kb_webapp_docs",))
    worker = make_worker(sse, chunk_tokens=900, chunk_overlap=120)
    worker.index_event(knowledge_event(kind="doc"))
    assert sse.creation_options == []
    assert len(sse.uploads) == 1
    assert sse.uploads[0][2] == {
        "max_tokens_in_chunk": 900,
        "number_of_overlap_tokens": 120,
    }


def test_upload_failure_retries_all_files():
    sse = UploadSSE(fail_methods=("upload_files",))
    redis = FakeRedis()
    worker = make_worker(sse, redis)
    entries = [
        (str(i), stream_message(knowledge_event(kind="doc", fingerprint=str(i))))
        for i in range(2)
    ]
    assert worker._process_batch(entries) == (2, 0)
    assert redis.acked == []
    assert all(redis.kv[worker._tries_key(entry_id)] == 1 for entry_id, _ in entries)
    sse.fail_methods.clear()
    redis.pending.extend(entries)
    assert worker._reclaim() == 2
    assert sorted(redis.acked) == ["0", "1"]
    assert len(sse.uploads) == 1
    assert len(sse.uploads[0][1]) == 2


@pytest.mark.parametrize("target", ["body", "metadata"])
def test_masking_failure_never_uploads_the_file(target):
    sse = UploadSSE()
    redis = FakeRedis()
    worker = make_worker(sse, redis)
    masker = worker._get_masker()

    class BrokenMasker:
        def mask_payload(self, payload):
            if isinstance(payload, dict) == (target == "metadata"):
                raise RuntimeError("secret")
            return masker.mask_payload(payload)

    worker._masker = BrokenMasker()
    worker._process_entry("1", stream_message(knowledge_event(kind="doc")))
    assert sse.uploads == sse.indexed == sse.created == []
    assert redis.acked == []


def test_upload_masks_body_and_metadata():
    event = dataclasses.replace(
        knowledge_event(kind="doc", text="Contact jane@example.com"),
        metadata={"contact": "jane@example.com", "thread_id": "thread-1"},
    )
    sse = UploadSSE()
    make_worker(sse).index_event(event)
    assert "jane@example.com" not in repr(sse.uploads)
    assert (
        sse.uploads[0][3][sse.uploads[0][1][0][0]]["options"]["thread_id"]
        == "thread-1"
    )


def test_prune_recognizes_uploaded_and_legacy_document_names():
    names = [
        "doc:20261010:old.txt",
        "doc:20261010:old.input_text",
        "doc:20261010:keep.txt",
    ]
    sse = UploadSSE(names={"agentic_kb_webapp_docs": names})
    make_worker(sse).index_event(prune_event("webapp", ["doc:20261010:old"]))
    assert sse.deleted == [("agentic_kb_webapp_docs", names[:2])]
    assert sse.uploads == sse.indexed == []
