"""
The documentation ingestor: reading a repository's markdown into the KB.

Documentation is the one knowledge kind the miner does not produce, so its
path is its own: scan the repository the request declared, split each file at
its headings, and publish only what changed.  The tests pin what that has to
guarantee — an unchanged file publishes nothing, a changed file replaces the
document it superseded, a scan that did not see the whole repository retires
nothing, and a request never pays for a scan twice in an hour.
"""

import dataclasses
import os
import pathlib
import sys

import pytest

sys.path.insert(1, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from agentic_knowledge_support import (  # noqa: E402
    FakePublisher,
    FakeRedis,
    FakeSSE,
    FakeStore,
    base_items,
    codex_payload,
    knowledge_config,
)

from llm_router_plugins.cli.plugins.utils.agentic_knowledge.worker import (  # noqa: E402,E501
    KnowledgeIndexWorker,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge import (  # noqa: E402
    AgenticKnowledgePlugin,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.cache import (  # noqa: E402
    InMemoryContextCache,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.docs_ingestor import (  # noqa: E402,E501
    CONTROL_DOCS_PRUNE,
)

# --------------------------------------------------------------------- repo -
README = "# Demo\n\nA service for exports.\n"
GUIDE = "# Guide\n\n## Deploy\n\nRun `make run`.\n"


def write(root, relative, text):
    path = os.path.join(str(root), relative)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def demo_repo(tmp_path):
    """A repository with documentation, code and a dependency tree."""
    root = str(tmp_path)
    write(root, "README.md", README)
    write(root, "docs/guide.md", GUIDE)
    write(root, "docs/api/reference.md", "# API\n\nEndpoints.\n")
    write(root, "node_modules/pkg/README.md", "# dependency\n")
    write(root, "src/app.py", "x = 1\n")
    return root


def indexer(config=None, publisher=None, state=None, clock=None):
    from llm_router_plugins.utils.knowledge.agentic_knowledge.docs_ingestor import (
        DocsIndexer,
    )

    kwargs = {"config": config or knowledge_config()}
    if publisher is not None:
        kwargs["publisher"] = publisher
    if state is not None:
        kwargs["state"] = state
    if clock is not None:
        kwargs["clock"] = clock
    return DocsIndexer(**kwargs)


def scanning_state():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.docs_ingestor import (
        DocsScanState,
    )

    return DocsScanState(key_prefix="agentic-kb", client=FakeRedis())


# ------------------------------------------------------------- discovery --
def test_a_scan_reads_the_documentation_and_leaves_the_rest(tmp_path):
    from llm_router_plugins.utils.knowledge.agentic_knowledge.docs_ingestor import (
        scan_repository,
    )

    report = scan_repository(demo_repo(tmp_path), scope="demo")
    found = sorted({record.source_file for record in report.records})
    assert found == ["README.md", "docs/api/reference.md", "docs/guide.md"]
    assert report.files == 3
    assert not report.truncated


def test_a_dependency_tree_is_never_descended_into(tmp_path):
    from llm_router_plugins.utils.knowledge.agentic_knowledge.docs_ingestor import (
        scan_repository,
    )

    report = scan_repository(demo_repo(tmp_path), scope="demo")
    assert not any(
        "node_modules" in path
        for path in {record.source_file for record in report.records}
    )


def test_a_file_above_the_ceiling_is_skipped_not_read(tmp_path):
    from llm_router_plugins.utils.knowledge.agentic_knowledge.docs_ingestor import (
        scan_repository,
    )

    root = demo_repo(tmp_path)
    write(root, "docs/huge.md", "# Huge\n\n" + "x" * 5000)
    report = scan_repository(root, scope="demo", max_file_bytes=1024)
    assert report.skipped == 1
    assert not any(r.source_file == "docs/huge.md" for r in report.records)


def test_a_scan_stopped_by_a_ceiling_says_it_is_truncated(tmp_path):
    from llm_router_plugins.utils.knowledge.agentic_knowledge.docs_ingestor import (
        scan_repository,
    )

    report = scan_repository(demo_repo(tmp_path), scope="demo", max_files=2)
    assert report.truncated is True
    assert report.files == 2


def test_a_directory_that_is_not_there_is_reported_not_raised():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.docs_ingestor import (
        scan_repository,
    )

    assert scan_repository("/no/such/repository").reason == "no-such-directory"


def test_a_symlinked_documentation_file_is_not_read(tmp_path):
    from llm_router_plugins.utils.knowledge.agentic_knowledge.docs_ingestor import (
        scan_repository,
    )

    root = demo_repo(tmp_path)
    os.symlink("/etc/passwd", os.path.join(root, "docs", "elsewhere.md"))
    report = scan_repository(root, scope="demo")
    assert not any("elsewhere" in r.source_file for r in report.records)


@pytest.mark.parametrize(
    "path, expected",
    [
        ("README.md", "readme"),
        ("CHANGELOG.md", "changelog"),
        ("CONTRIBUTING.md", "contributing"),
        ("LICENSE", "license"),
        ("SECURITY.md", "security"),
        ("AGENTS.md", "agent"),
        ("docs/deploy.md", "guide"),
        ("notes/edge-cases.md", "note"),
    ],
)
def test_a_documentation_file_is_classified_by_its_name(path, expected):
    from llm_router_plugins.utils.knowledge.agentic_knowledge.docs_ingestor import (
        classify_document,
    )

    assert classify_document(path) == expected


# ------------------------------------------------------------- sectioning --
def test_a_section_carries_the_heading_path_it_sits_under():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.docs_ingestor import (
        split_sections,
    )

    sections = split_sections(
        "# Guide\n\n"
        + "Intro that is long enough to be a section of its own. " * 4
        + "\n\n## Deploy\n\n"
        + "Deploy instructions long enough to stand on their own. " * 4
        + "\n\n### Docker\n\n"
        + "Docker notes long enough to stand on their own as well. " * 4,
        400,
    )
    headings = [heading for heading, _ in sections]
    assert headings == ["Guide", "Guide > Deploy", "Guide > Deploy > Docker"]


def test_a_hash_inside_a_code_block_is_not_a_heading():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.docs_ingestor import (
        split_sections,
    )

    sections = split_sections(
        "# Guide\n\n```bash\n# not a heading\nmake run\n```\n", 1200
    )
    assert len(sections) == 1
    assert "# not a heading" in sections[0][1]


def test_a_long_section_is_cut_and_each_piece_keeps_its_heading():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.docs_ingestor import (
        split_sections,
    )

    body = "\n\n".join(
        f"Paragraph {index} says something worth retrieving on its own."
        for index in range(30)
    )
    sections = split_sections(f"# Guide\n\n## Deploy\n\n{body}\n", 700)
    assert len(sections) > 1
    for heading, piece in sections:
        assert heading == "Guide > Deploy"
        assert len(piece) <= 700
        if piece is not sections[0][1]:
            assert piece.startswith("Guide > Deploy")


def test_a_stub_section_is_folded_into_its_predecessor():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.docs_ingestor import (
        split_sections,
    )

    sections = split_sections(
        "# Guide\n\n" + "Long enough to stand alone. " * 20 + "\n\n## See also\n",
        1200,
    )
    assert len(sections) == 1
    assert "## See also" in sections[0][1]


def test_a_document_without_headings_is_one_section():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.docs_ingestor import (
        split_sections,
    )

    sections = split_sections("Just prose, all the way down.\n", 1200)
    assert sections == [("", "Just prose, all the way down.")]


# ----------------------------------------------------------- records/events -
def test_a_section_is_fingerprinted_on_the_whole_files_content(tmp_path):
    from llm_router_plugins.utils.knowledge.agentic_knowledge.docs_ingestor import (
        scan_repository,
    )

    root = demo_repo(tmp_path)
    first = scan_repository(root, scope="demo")
    second = scan_repository(root, scope="demo")
    assert [r.fingerprint for r in first.records] == [
        r.fingerprint for r in second.records
    ]

    write(root, "docs/guide.md", GUIDE + "\nChanged.\n")
    third = scan_repository(root, scope="demo")
    changed = {r.source_file: r.fingerprint for r in third.records}
    before = {r.source_file: r.fingerprint for r in first.records}
    assert changed["docs/guide.md"] != before["docs/guide.md"]
    assert changed["README.md"] == before["README.md"]


def test_a_doc_event_names_its_file_type_and_section_in_its_header(tmp_path):
    from llm_router_plugins.utils.knowledge.agentic_knowledge.docs_ingestor import (
        doc_events,
        scan_repository,
    )
    from llm_router_plugins.utils.knowledge.agentic_knowledge.documents import (
        parse_document_text,
    )

    report = scan_repository(demo_repo(tmp_path), scope="demo")
    record = next(r for r in report.records if r.source_file == "docs/guide.md")
    event = doc_events([record], scope="demo", agent="codex")[0]
    meta, body = parse_document_text(event.text)

    assert event.kind == "doc"
    assert event.scope == "demo"
    assert event.mode == ""  # documentation is not an episode
    assert (meta.kind, meta.source, meta.doc_type) == (
        "doc",
        "docs/guide.md",
        "guide",
    )
    assert meta.heading in ("Guide", "Guide > Deploy")
    assert "Run `make run`." in body


def test_a_prune_event_carries_the_names_it_retires():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.docs_ingestor import (
        prune_event,
    )

    event = prune_event("demo", ["doc:20261021:bbb", "doc:20261021:aaa"])
    assert event is not None
    assert event.kind == CONTROL_DOCS_PRUNE
    assert event.metadata["names"] == ["doc:20261021:aaa", "doc:20261021:bbb"]
    assert event.text == ""
    assert prune_event("demo", []) is None


# ------------------------------------------------------------- the indexer -
def test_the_first_scan_publishes_the_whole_documentation(tmp_path):
    publisher = FakePublisher()
    outcome = indexer(publisher=publisher, state=scanning_state()).index(
        "demo", demo_repo(tmp_path)
    )
    assert outcome.status == "ok"
    assert outcome.published == 3
    assert {event.kind for event in publisher.published} == {"doc"}


def test_an_unchanged_file_is_published_once(tmp_path):
    root = demo_repo(tmp_path)
    publisher = FakePublisher()
    service = indexer(publisher=publisher, state=scanning_state())

    service.index("demo", root)
    publisher.published.clear()
    outcome = service.index("demo", root, force=True)

    assert outcome.published == 0
    assert outcome.unchanged == 3


def test_touching_a_file_without_changing_it_publishes_nothing(tmp_path):
    root = demo_repo(tmp_path)
    publisher = FakePublisher()
    service = indexer(publisher=publisher, state=scanning_state())
    service.index("demo", root)
    publisher.published.clear()

    os.utime(os.path.join(root, "README.md"), (0, 0))
    outcome = service.index("demo", root, force=True)

    assert outcome.published == 0
    assert outcome.pruned == 0


def test_a_changed_file_replaces_the_document_it_superseded(tmp_path):
    root = demo_repo(tmp_path)
    publisher = FakePublisher()
    service = indexer(publisher=publisher, state=scanning_state())
    service.index("demo", root)
    old_names = [
        f"doc:{_today()}:{event.fingerprint}" for event in publisher.published
    ]
    publisher.published.clear()

    write(root, "README.md", README + "\n## New\n\nSomething new here.\n")
    service.index("demo", root, force=True)

    prune = [e for e in publisher.published if e.kind == CONTROL_DOCS_PRUNE]
    assert len(prune) == 1
    assert prune[0].metadata["names"] == [
        name for name in old_names if name in prune[0].metadata["names"]
    ]
    assert any(event.kind == "doc" for event in publisher.published)


def test_a_deleted_file_has_its_documents_retired(tmp_path):
    root = demo_repo(tmp_path)
    publisher = FakePublisher()
    service = indexer(publisher=publisher, state=scanning_state())
    service.index("demo", root)
    guide = next(
        event
        for event in publisher.published
        if event.metadata["source_file"] == "docs/guide.md"
    )
    publisher.published.clear()

    os.remove(os.path.join(root, "docs", "guide.md"))
    outcome = service.index("demo", root, force=True)

    assert outcome.files == 2
    assert outcome.published == 0
    assert outcome.pruned == 1
    prune = publisher.published[0]
    assert prune.kind == CONTROL_DOCS_PRUNE
    assert prune.metadata["names"] == [f"doc:{_today()}:{guide.fingerprint}"]


def test_a_truncated_scan_retires_nothing(tmp_path):
    root = demo_repo(tmp_path)
    state = scanning_state()
    indexer(publisher=FakePublisher(), state=state).index("demo", root)

    publisher = FakePublisher()
    config = knowledge_config(docs_max_files=1)
    outcome = indexer(config=config, publisher=publisher, state=state).index(
        "demo", root, force=True
    )
    assert outcome.truncated is True
    assert outcome.pruned == 0
    assert not any(e.kind == CONTROL_DOCS_PRUNE for e in publisher.published)


def test_the_scans_of_one_process_are_bounded_by_a_cooldown(tmp_path):
    root = demo_repo(tmp_path)
    now = [1_000.0]
    publisher = FakePublisher()
    service = indexer(
        publisher=publisher, state=scanning_state(), clock=lambda: now[0]
    )

    service.index("demo", root)
    publisher.published.clear()
    cooled = service.index("demo", root)
    assert cooled.status == "skipped"
    assert cooled.reason == "cooldown"

    now[0] += 21_601
    write(root, "docs/late.md", "# Late\n\nAdded while the scan was cooling down.\n")
    assert service.index("demo", root).status == "ok"
    assert [event.metadata["source_file"] for event in publisher.published] == [
        "docs/late.md"
    ]


def test_manual_mode_publishes_nothing_from_a_request(tmp_path):
    publisher = FakePublisher()
    config = knowledge_config(docs_auto_index=False)
    outcome = indexer(config=config, publisher=publisher, state=scanning_state())
    assert outcome.index("demo", demo_repo(tmp_path)).reason == "manual"


def test_the_operators_command_overrides_manual_mode(tmp_path):
    publisher = FakePublisher()
    config = knowledge_config(docs_auto_index=False)
    service = indexer(config=config, publisher=publisher, state=scanning_state())
    assert service.index("demo", demo_repo(tmp_path), force=True).status == "ok"


def test_a_repository_this_host_cannot_see_is_skipped(tmp_path):
    outcome = indexer(publisher=FakePublisher(), state=scanning_state()).index(
        "demo", str(pathlib.Path(tmp_path) / "not-checked-out"), force=True
    )
    assert outcome.status == "skipped"
    assert outcome.reason == "no-such-directory"


def test_a_request_without_a_scope_or_a_root_publishes_nothing(tmp_path):
    service = indexer(publisher=FakePublisher(), state=scanning_state())
    assert service.index("", str(tmp_path), force=True).reason == "no scope"
    assert service.index("demo", "", force=True).reason == "no root"


def test_a_disabled_policy_never_reads_the_filesystem(tmp_path):
    publisher = FakePublisher()
    config = knowledge_config(docs_enabled=False)
    outcome = indexer(config=config, publisher=publisher).index(
        "demo", demo_repo(tmp_path)
    )
    assert outcome.status == "disabled"
    assert publisher.published == []


def test_a_scan_without_a_place_to_remember_never_published(tmp_path):
    from llm_router_plugins.utils.knowledge.agentic_knowledge.docs_ingestor import (
        DocsScanState,
    )

    publisher = FakePublisher()
    no_state = DocsScanState(key_prefix="agentic-kb", client=None)
    outcome = indexer(publisher=publisher, state=no_state).index(
        "demo", demo_repo(tmp_path), force=True
    )
    assert outcome.status == "unavailable"
    assert outcome.reason == "state"
    assert publisher.published == []


def test_a_failed_publish_does_not_record_the_scan(tmp_path):
    root = demo_repo(tmp_path)
    state = scanning_state()

    class BrokenPublisher:
        def publish(self, events):
            raise RuntimeError("redis is down")

    outcome = indexer(publisher=BrokenPublisher(), state=state).index(
        "demo", root, force=True
    )
    assert outcome.status == "unavailable"
    assert outcome.reason == "publish"

    publisher = FakePublisher()
    retry = indexer(publisher=publisher, state=state).index("demo", root, force=True)
    assert retry.published == 3


def test_the_indexer_does_not_raise(tmp_path, monkeypatch):
    from llm_router_plugins.utils.knowledge.agentic_knowledge import docs_ingestor

    def explode(*args, **kwargs):
        raise RuntimeError("filesystem on fire")

    monkeypatch.setattr(docs_ingestor, "scan_repository", explode)
    service = indexer(publisher=FakePublisher(), state=scanning_state())
    with pytest.raises(RuntimeError):
        service.index("demo", demo_repo(tmp_path), force=True)


# ------------------------------------------------------------------ state --
def test_the_scan_state_round_trips_and_ignores_rubbish():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.docs_ingestor import (
        DocsScanState,
    )

    redis = FakeRedis()
    state = DocsScanState(key_prefix="agentic-kb", client=redis)
    assert state.load("demo") == {}
    assert state.save("demo", {"README.md": {"h": "abc", "n": ["doc:1:a"]}})
    assert state.load("demo") == {"README.md": {"h": "abc", "n": ["doc:1:a"]}}

    redis.kv[state.key("demo")] = "{not json"
    assert state.load("demo") == {}


# ----------------------------------------------------------------- worker --
def test_a_doc_event_is_indexed_into_the_scopes_collection():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.docs_ingestor import (
        doc_events,
    )

    sse = FakeSSE()
    event = doc_events(
        [_record()],
        scope="webapp",
        agent="codex",
    )[0]
    worker = _worker(sse=sse)
    worker.index_event(event)

    collection, documents = sse.indexed[0]
    # Documentation is indexed apart from mined knowledge: it is written on a
    # human's clock, re-indexed as a set, and read with its own metadata
    # restriction, so it lives in a collection of its own.
    assert collection == "agentic_kb_webapp_docs"
    assert documents[0]["category"] == "doc"
    assert documents[0]["filepath"].startswith("doc:")
    assert 'src="docs/guide.md"' in documents[0]["pages"][0]["page_content"]


def test_a_prune_event_deletes_the_documents_the_engine_holds():
    sse = FakeSSE(
        names={
            "agentic_kb_webapp_docs": [
                "doc:20261010:old1.input_text",
                "doc:20261010:new.input_text",
            ]
        }
    )
    worker = KnowledgeIndexWorker(
        config=knowledge_config(), redis_client=FakeRedis(), sse_client=sse
    )
    worker.index_event(_prune("doc:20261010:old1"))

    assert sse.deleted == [
        ("agentic_kb_webapp_docs", ["doc:20261010:old1.input_text"])
    ]
    assert not sse.indexed


def test_a_prune_event_deletes_nothing_the_engine_does_not_have():
    sse = FakeSSE(names={"agentic_kb_webapp_docs": ["doc:20261010:keep.input_text"]})
    worker = KnowledgeIndexWorker(
        config=knowledge_config(), redis_client=FakeRedis(), sse_client=sse
    )
    worker.index_event(_prune("doc:20261010:gone"))
    assert sse.deleted == []


# ----------------------------------------------------------------- plugin --
def test_the_annotation_reports_the_documentation_scan(tmp_path):
    plugin, publisher = _plugin(tmp_path)
    payload = plugin.apply(request_at(demo_repo(tmp_path)))

    docs = payload["routing"]["knowledge"]["docs"]
    assert docs["status"] == "ok"
    assert docs["files"] == 3
    assert docs["published"] == 3
    assert {event.kind for event in publisher.published} == {"doc"}


def test_documentation_is_not_offered_without_a_write_path(tmp_path):
    root = str(demo_repo(tmp_path))
    plugin = AgenticKnowledgePlugin(
        config=knowledge_config(write_enabled=False),
        store=FakeStore(),
        publisher=FakePublisher(),
        cache=InMemoryContextCache(),
    )
    payload = plugin.apply(request_at(root))
    assert "docs" not in payload["routing"]["knowledge"]


def test_a_failing_documentation_scan_still_answers_the_request(
    tmp_path, monkeypatch
):
    from llm_router_plugins.utils.knowledge.agentic_knowledge import plugin as module

    monkeypatch.setattr(
        module.DocsIndexer,
        "index",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    plugin, _publisher = _plugin(tmp_path)
    payload = plugin.apply(request_at(demo_repo(tmp_path)))
    assert payload["routing"]["knowledge"]["docs"]["reason"] == "scan"


# ------------------------------------------------------------------ cli ----
def test_the_documentation_command_lists_a_scan_without_publishing(tmp_path):
    from llm_router_plugins.cli.plugins.utils.agentic_knowledge.docs import main

    code = main(["--path", str(demo_repo(tmp_path)), "--dry-run", "--json"])
    assert code == 0


def test_the_documentation_command_needs_the_stream(tmp_path, monkeypatch):
    from llm_router_plugins.cli.plugins.utils.agentic_knowledge.docs import main

    monkeypatch.delenv("LLM_ROUTER_KNOWLEDGE_AGENTIC_REDIS_HOST", raising=False)
    with pytest.raises(SystemExit) as exit_info:
        main(["--path", str(demo_repo(tmp_path))])
    assert "stream" in str(exit_info.value).lower()


# --------------------------------------------------------------- fixtures --
def request_at(root):
    """A main request made in *root*, so its scope and repository resolve."""
    return codex_payload(base_items(cwd=str(root)))


def _worker(sse):
    config = knowledge_config()
    config.worker = dataclasses.replace(
        config.worker, embedder="bge-m3", reranker="bge-reranker-v2-m3"
    )
    return KnowledgeIndexWorker(
        config=config, redis_client=FakeRedis(), sse_client=sse
    )


def _plugin(tmp_path):
    publisher = FakePublisher()
    plugin = AgenticKnowledgePlugin(
        config=knowledge_config(docs_rescan_interval_seconds=0),
        store=FakeStore(),
        publisher=publisher,
        cache=InMemoryContextCache(),
    )
    return plugin, publisher


def _record():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.docs_ingestor import (
        DocRecord,
    )

    return DocRecord(
        source_file="docs/guide.md",
        doc_type="guide",
        heading="Guide",
        position=0,
        text="## Deploy\n\nRun `make run`.",
        content_hash="abc",
        fingerprint="abc123",
        size=40,
        mtime=0.0,
    )


def _prune(*names):
    from llm_router_plugins.utils.knowledge.agentic_knowledge.docs_ingestor import (
        prune_event,
    )

    return prune_event("webapp", list(names))


def _today():
    import datetime

    return datetime.date.today().strftime("%Y%m%d")


def test_a_version_control_directory_is_skipped_even_when_configured_in(tmp_path):
    from llm_router_plugins.utils.knowledge.agentic_knowledge.docs_ingestor import (
        scan_repository,
    )

    root = demo_repo(tmp_path)
    write(root, ".git/COMMIT_EDITMSG", "# a commit message\n")
    report = scan_repository(root, scope="demo", excluded_dirs=["node_modules"])
    assert not any(
        record.source_file.startswith(".git") for record in report.records
    )


# ------------------------------------------------ collection-bound state --
def test_a_listing_written_for_another_collection_is_not_a_listing():
    """
    Moving the documentation to another collection republishes it.

    The scan state says which files are already indexed — and a listing of the
    collection nobody reads would tell the next scan that a repository has its
    documentation, in a place where no search looks.  So a listing names its
    collection, and one that names another is no listing at all.
    """
    from llm_router_plugins.utils.knowledge.agentic_knowledge.docs_ingestor import (
        DocsScanState,
    )

    state = DocsScanState(key_prefix="agentic-kb", client=FakeRedis())
    entries = {"README.md": {"h": "abc", "n": ["doc:1:a"]}}
    assert state.save("demo", entries, "agentic_kb_demo")
    assert state.load("demo", "agentic_kb_demo") == entries
    assert state.load("demo", "agentic_kb_demo_docs") == {}


def test_a_listing_without_a_collection_is_read_by_the_first_one():
    """A listing predating the field is trusted once, then rewritten."""
    from llm_router_plugins.utils.knowledge.agentic_knowledge.docs_ingestor import (
        DocsScanState,
    )

    redis = FakeRedis()
    state = DocsScanState(key_prefix="agentic-kb", client=redis)
    redis.kv[state.key("demo")] = '{"v": 1, "files": {"README.md": {"h": "abc"}}}'
    assert state.load("demo", "agentic_kb_demo_docs") == {}


def test_a_collection_move_republishes_the_documentation(tmp_path):
    root = demo_repo(tmp_path)
    state = scanning_state()
    before = knowledge_config(collection_prefix="kbfirst")
    outcome = indexer(config=before, publisher=FakePublisher(), state=state).index(
        "demo", root, force=True
    )
    assert outcome.published == 3

    publisher = FakePublisher()
    after = knowledge_config(collection_prefix="kbsecond")
    moved = indexer(config=after, publisher=publisher, state=state).index(
        "demo", root, force=True
    )
    assert moved.published == 3
    assert moved.unchanged == 0

    same = indexer(config=after, publisher=FakePublisher(), state=state).index(
        "demo", root, force=True
    )
    assert same.published == 0
    assert same.unchanged == 3


# --------------------------------------------------------- read per mode --
def test_documentation_is_searched_in_its_own_collection():
    """One request for the repository's prose, and it names its kinds."""
    store = FakeStore()
    plugin = AgenticKnowledgePlugin(
        config=knowledge_config(),
        store=store,
        publisher=FakePublisher(),
        cache=InMemoryContextCache(),
    )
    plugin.apply(codex_payload(base_items(cwd="/repo/webapp"), mode="implement"))

    documented = [
        request for request in store.requests if request.categories == ("doc",)
    ]
    assert len(documented) == 1
    assert documented[0].collection == "agentic_kb_webapp_docs"
    assert documented[0].doc_types == ("readme", "guide", "contributing")
    mined = [request for request in store.requests if request.categories != ("doc",)]
    assert all(request.collection != "agentic_kb_webapp_docs" for request in mined)


def test_a_mode_without_documentation_asks_the_docs_collection_for_nothing():
    store = FakeStore()
    plugin = AgenticKnowledgePlugin(
        config=knowledge_config(),
        store=store,
        publisher=FakePublisher(),
        cache=InMemoryContextCache(),
    )
    plugin.apply(codex_payload(base_items(cwd="/repo/webapp"), mode="debug"))
    assert all(
        request.collection != "agentic_kb_webapp_docs" for request in store.requests
    )


def test_the_documentation_search_is_not_the_failure_query():
    """The prose is asked about the work, not about the stack trace."""
    from agentic_knowledge_support import (
        _command,
        _command_output,
        _heredoc_patch,
        _user,
    )

    store = FakeStore()
    plugin = AgenticKnowledgePlugin(
        config=knowledge_config(),
        store=store,
        publisher=FakePublisher(),
        cache=InMemoryContextCache(),
    )
    items = base_items(cwd="/repo/webapp") + [
        _user("rename the export helper"),
        _heredoc_patch("c1", "e1", paths=("src/export.py",)),
        _command("c2", "e2", "pytest tests/test_export.py"),
        _command_output("c2", "e2", "AssertionError: boom"),
    ]
    plugin.apply(codex_payload(items, mode="implement"))

    documented = [
        request for request in store.requests if request.categories == ("doc",)
    ]
    assert documented
    assert "rename the export helper" in documented[0].query
    assert "AssertionError" not in documented[0].query
