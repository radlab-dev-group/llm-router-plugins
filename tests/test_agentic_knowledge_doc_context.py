"""
The standing documentation digest: what a repository has written down.

The layer retrieves a paragraph when a query matches one; the digest is what
is always there instead — which documentation files exist and what they cover.
The properties that matter are the ones a retrieved chunk does not need: the
digest is **structural** (no prose, so nothing unmasked reaches a request),
**stable** (it moves when the documentation's outline moves, not when its text
is reworded), and **budgeted first** (the always-present map is what a mode
pays for before the chunks a query happens to find).
"""

import datetime
import os
import sys
import pathlib

sys.path.insert(1, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from agentic_knowledge_support import (  # noqa: E402
    FakePublisher,
    FakeRedis,
    FakeStore,
    base_items,
    codex_payload,
    knowledge_config,
    store_hit,
)

from llm_router_plugins.utils.knowledge.agentic_knowledge.assembler import (  # noqa: E402,E501
    ContextAssembler,
    DOC_CONTEXT_TITLE,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.doc_context import (  # noqa: E402,E501
    DocContext,
    DocContextStore,
    DocFile,
    build_doc_context,
    render_doc_context,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.docs_ingestor import (  # noqa: E402,E501
    DocsIndexer,
    DocsScanState,
    document_headings,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.documents import (  # noqa: E402,E501
    build_document_text,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.sse import (  # noqa: E402
    STATUS_OK,
    SearchResult,
)


def doc_file(path, doc_type, *headings, content_hash="h1"):
    return DocFile(
        path=path,
        doc_type=doc_type,
        headings=tuple(headings),
        content_hash=content_hash,
    )


# ------------------------------------------------------------- the digest --
def test_the_digest_names_the_files_and_their_outlines():
    context = build_doc_context(
        "demo",
        [
            doc_file("docs/guide.md", "guide", "Guide", "Deploy"),
            doc_file("README.md", "readme", "Demo", "Install"),
        ],
    )
    assert "- README.md (readme): Demo; Install" in context.text
    assert "- docs/guide.md (guide): Guide; Deploy" in context.text


def test_the_introduction_is_listed_before_the_pages():
    """A reader meets the project before its documentation tree."""
    context = build_doc_context(
        "demo",
        [
            doc_file("CHANGELOG.md", "changelog", "v2"),
            doc_file("README.md", "readme", "Demo"),
            doc_file("docs/a.md", "guide", "A"),
        ],
    )
    lines = [line for line in context.text.splitlines() if line.startswith("- ")]
    assert lines[0].startswith("- README.md")


def test_an_empty_repository_has_no_digest():
    assert build_doc_context("demo", []).empty
    assert render_doc_context(DocContext()) == ""


def test_the_digest_is_structural_and_carries_no_prose():
    """
    The digest is a map, not an excerpt.

    The write path masks a document before indexing it; a digest built on the
    request path would carry the README's sentences unmasked into a model.
    Paths, kinds and headings are what the layer already files unmasked, and
    they are enough for an agent to aim a question.
    """
    context = build_doc_context(
        "demo",
        [
            doc_file("README.md", "readme", "Secrets", "Tokens"),
            doc_file("docs/guide.md", "guide", "Guide"),
        ],
    )
    assert "Secrets" in context.text
    assert "api_key" not in context.text
    assert context.text.count("\n") == 1  # one line per file, no prose


def test_a_digest_of_a_repository_of_pages_stays_structural():
    context = build_doc_context(
        "demo",
        [doc_file(f"docs/page-{index}.md", "guide", "Page") for index in range(5)],
    )
    for line in context.text.splitlines():
        assert line.startswith("- docs/page-")


def test_the_digest_stops_at_its_budget():
    files = [
        doc_file(f"docs/page-{index:02d}.md", "guide", "Title")
        for index in range(40)
    ]
    context = build_doc_context("demo", files, max_chars=200)
    assert len(context.text) <= 200
    assert context.truncated


def test_a_partial_map_says_it_is_partial():
    context = build_doc_context(
        "demo",
        [doc_file(f"docs/page-{index}.md", "guide", "T") for index in range(20)],
        max_chars=160,
    )
    assert "not listed" in context.text


def test_the_ordering_does_not_move_when_a_file_is_edited():
    """
    A map rearranges itself when files arrive, not when prose is reworded.

    The digest is injected into a prompt the provider caches; a request that
    reorders the same files because one file's content hash changed spends the
    cache and tells the agent the world moved.
    """
    before = build_doc_context(
        "demo",
        [
            doc_file("docs/b.md", "guide", "B", content_hash="old"),
            doc_file("docs/a.md", "guide", "A"),
        ],
    )
    after = build_doc_context(
        "demo",
        [
            doc_file("docs/b.md", "guide", "B", content_hash="new"),
            doc_file("docs/a.md", "guide", "A"),
        ],
    )
    assert before.text == after.text
    assert before.fingerprint != after.fingerprint


def test_the_fingerprint_moves_when_the_outline_moves():
    before = build_doc_context("demo", [doc_file("README.md", "readme", "Demo")])
    after = build_doc_context(
        "demo", [doc_file("README.md", "readme", "Demo", "Install")]
    )
    assert before.fingerprint != after.fingerprint


def test_the_rendered_digest_tells_the_agent_it_can_ask():
    rendered = render_doc_context(
        build_doc_context("demo", [doc_file("docs/guide.md", "guide", "Deploy")])
    )
    assert "docs/guide.md" in rendered
    assert "asked for" in rendered


def test_long_headings_are_cut_and_empty_ones_dropped():
    context = build_doc_context(
        "demo",
        [doc_file("README.md", "readme", "A heading " * 30, "", "Install")],
    )
    assert len(context.text) < 200
    assert "Install" in context.text


# ---------------------------------------------------------------- outline --
def test_a_heading_inside_a_code_block_is_not_part_of_the_outline():
    text = "# Title\n\n```bash\n# not a heading\necho hi\n```\n\n## Real\n"
    assert document_headings(text) == ("Title", "Real")


def test_the_outline_is_the_documents_own_order():
    text = "# Top\n\n## Second\n\n### Third\n\n## Fourth\n"
    assert document_headings(text) == ("Top", "Second", "Third", "Fourth")


# ------------------------------------------------------------------ store --
def test_the_digest_round_trips_through_the_store():
    store = DocContextStore(key_prefix="agentic-kb", client=FakeRedis())
    context = build_doc_context("demo", [doc_file("README.md", "readme", "Demo")])
    assert store.put(context)
    read = store.get("demo")
    assert read is not None
    assert read.text == context.text
    assert read.source_files == ("README.md",)
    assert read.fingerprint == context.fingerprint


def test_an_empty_digest_is_not_written():
    store = DocContextStore(key_prefix="agentic-kb", client=FakeRedis())
    assert not store.put(DocContext(scope="demo"))
    assert store.get("demo") is None


def test_a_rubbish_value_reads_as_no_digest():
    redis = FakeRedis()
    store = DocContextStore(key_prefix="agentic-kb", client=redis)
    redis.kv[store.key("demo")] = "{not json"
    assert store.get("demo") is None


def test_a_digest_without_text_is_not_a_digest():
    redis = FakeRedis()
    store = DocContextStore(key_prefix="agentic-kb", client=redis)
    redis.kv[store.key("demo")] = '{"v": 1, "text": ""}'
    assert store.get("demo") is None


def test_a_store_with_no_redis_is_unavailable():
    store = DocContextStore(key_prefix="agentic-kb", client=None)
    assert not store.available
    assert store.get("demo") is None


# --------------------------------------------------------- the scan owns it --
def write(root, relative, text):
    path = os.path.join(str(root), relative)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def demo_repo(tmp_path):
    root = str(tmp_path)
    write(
        root,
        "README.md",
        "# Demo\n\nAn export service.\n\n## Install\n\npip install -e .\n",
    )
    write(
        root,
        "docs/guide.md",
        "# Guide\n\n## Deploy\n\nRun `make run`, then check.\n",
    )
    write(root, "src/app.py", "x = 1\n")
    return root


def scanning_indexer(config=None, publisher=None):
    redis = FakeRedis()
    return (
        DocsIndexer(
            config=config or knowledge_config(),
            publisher=publisher or FakePublisher(),
            state=DocsScanState(key_prefix="agentic-kb", client=redis),
            context_store=DocContextStore(key_prefix="agentic-kb", client=redis),
        ),
        redis,
    )


def test_a_scan_produces_the_standing_digest(tmp_path):
    indexer, _redis = scanning_indexer()
    outcome = indexer.index("demo", demo_repo(tmp_path), force=True)
    assert outcome.context is not None
    assert "README.md" in outcome.context.text
    assert "Install" in outcome.context.text


def test_the_digest_outlives_the_scan_that_made_it(tmp_path):
    """A request on cooldown reads the map the last scanning request wrote."""
    indexer, _redis = scanning_indexer()
    made = indexer.index("demo", demo_repo(tmp_path), force=True).context
    assert indexer.context("demo").text == made.text


def test_a_digest_is_not_read_when_the_policy_switches_it_off(tmp_path):
    config = knowledge_config(doc_context_enabled=False)
    indexer, _redis = scanning_indexer(config=config)
    outcome = indexer.index("demo", demo_repo(tmp_path), force=True)
    assert outcome.context is None
    assert indexer.context("demo") is None


def test_a_repository_with_no_documentation_produces_no_digest(tmp_path):
    root = str(tmp_path)
    write(root, "src/app.py", "x = 1\n")
    indexer, _redis = scanning_indexer()
    outcome = indexer.index("demo", root, force=True)
    assert outcome.context is None


# ------------------------------------------------------------ assembled it --
def assemble(results, mode="implement", doc_context=None, **kwargs):
    config = knowledge_config()
    return ContextAssembler(**kwargs).assemble(
        results, config.modes[mode], doc_context=doc_context
    )


def _doc_hit(source="docs/guide.md", doc_type="guide", heading="Deploy"):
    body = build_document_text(
        "doc",
        "Run `make run` to deploy the service.",
        date=datetime.date.today(),
        paths=(source,),
        source=source,
        doc_type=doc_type,
        heading=heading,
    )
    return SearchResult(
        STATUS_OK,
        hits=(store_hit("doc", name="doc:20261021:aa", text=body, score=0.8),),
    )


def _digest(*files, max_chars=700):
    return build_doc_context("webapp", list(files), max_chars=max_chars)


def test_the_digest_is_rendered_as_its_own_section_first():
    context = assemble(
        ((_doc_hit(), ("doc",)),),
        doc_context=_digest(
            doc_file("README.md", "readme", "Demo", "Install"),
            doc_file("docs/guide.md", "guide", "Guide", "Deploy"),
        ),
    )
    assert DOC_CONTEXT_TITLE in context.rendered
    assert context.rendered.index(DOC_CONTEXT_TITLE) < context.rendered.index(
        "## Conventions"
    )


def test_the_digest_is_injected_when_nothing_was_retrieved():
    """Level one is always present; a query that finds nothing changes that."""
    context = assemble(
        ((SearchResult(STATUS_OK), ("doc",)),),
        doc_context=_digest(doc_file("README.md", "readme", "Demo")),
    )
    assert context.hits == ()
    assert "README.md" in context.rendered


def test_the_context_reports_the_digest_it_injected():
    digest = _digest(doc_file("README.md", "readme", "Demo"))
    context = assemble(((_doc_hit(), ("doc",)),), doc_context=digest)
    assert context.doc_context is digest


def test_a_digest_too_big_for_the_mode_is_not_injected():
    """A map that does not fit leaves the block alone rather than squeezing it."""
    # A digest longer than the mode's whole budget: the block is not built
    # around something that cannot fit in it.
    oversized = DocContext(
        scope="webapp",
        text="- " + ("docs/page.md" * 400),
        source_files=("docs/page.md",),
        fingerprint="deadbeef",
    )
    assert oversized.chars > knowledge_config().modes["review"].max_chars
    context = assemble(
        ((_doc_hit(), ("doc",)),),
        mode="review",
        doc_context=oversized,
    )
    assert DOC_CONTEXT_TITLE not in context.rendered
    assert context.doc_context is None
    assert "## Conventions" in context.rendered


def test_the_budget_still_holds_with_a_digest():
    policy = knowledge_config().modes["implement"]
    context = assemble(
        ((_doc_hit(), ("doc",)),),
        doc_context=_digest(
            *[
                doc_file(f"docs/page-{index:02d}.md", "guide", "T")
                for index in range(6)
            ],
            max_chars=400,
        ),
    )
    assert context.chars <= policy.max_chars


def test_no_digest_renders_the_block_as_before():
    context = assemble(((_doc_hit(), ("doc",)),))
    assert DOC_CONTEXT_TITLE not in context.rendered
    assert "## Conventions" in context.rendered


# ------------------------------------------------------------ plugin level --
def test_the_annotation_reports_the_documentation_context(tmp_path):
    from llm_router_plugins.utils.knowledge.agentic_knowledge import (
        AgenticKnowledgePlugin,
    )
    from llm_router_plugins.utils.knowledge.agentic_knowledge.cache import (
        InMemoryContextCache,
    )
    from agentic_knowledge_support import _user

    root = str(tmp_path)
    write(
        root,
        "README.md",
        "# Demo\n\nAn export service.\n\n## Install\n\npip install -e .\n",
    )
    write(
        root, "docs/guide.md", "# Guide\n\n## Deploy\n\nRun `make run` to deploy.\n"
    )
    redis = FakeRedis()
    indexer = DocsIndexer(
        config=knowledge_config(),
        publisher=FakePublisher(),
        state=DocsScanState(key_prefix="agentic-kb", client=redis),
        context_store=DocContextStore(key_prefix="agentic-kb", client=redis),
    )
    # The repository was indexed by a machine that can see it (the CLI), and
    # this request does not scan again: it reads the standing digest.
    indexer.index("webapp", root, force=True)

    plugin = AgenticKnowledgePlugin(
        config=knowledge_config(docs_auto_index=False),
        store=FakeStore(),
        publisher=FakePublisher(),
        cache=InMemoryContextCache(),
    )
    plugin._docs = indexer
    payload = plugin.apply(
        codex_payload(
            base_items(cwd="/repo/webapp") + [_user("rename the export helper")],
            mode="implement",
        )
    )
    knowledge = payload["routing"]["knowledge"]
    assert knowledge["doc_context"]["level"] == "digest"
    assert knowledge["doc_context"]["chars"] > 0
    assert "README.md" in knowledge["doc_context"]["source_files"]
    assert "## Documentation in this repository" in _injected(payload)


def test_a_second_request_reads_the_same_digest(tmp_path):
    """
    The digest is per repository, not per request.

    A long session must see one stable map: the second command reads what the
    first command's scan wrote, and gets the same text back.
    """
    from llm_router_plugins.utils.knowledge.agentic_knowledge import (
        AgenticKnowledgePlugin,
    )
    from llm_router_plugins.utils.knowledge.agentic_knowledge.cache import (
        InMemoryContextCache,
    )

    root = str(tmp_path)
    write(root, "README.md", "# Demo\n\nAn export service.\n")
    redis = FakeRedis()
    indexer = DocsIndexer(
        config=knowledge_config(),
        publisher=FakePublisher(),
        state=DocsScanState(key_prefix="agentic-kb", client=redis),
        context_store=DocContextStore(key_prefix="agentic-kb", client=redis),
    )
    indexer.index("webapp", root, force=True)

    plugin = AgenticKnowledgePlugin(
        config=knowledge_config(docs_auto_index=False),
        store=FakeStore(),
        publisher=FakePublisher(),
        cache=InMemoryContextCache(),
    )
    plugin._docs = indexer
    first = plugin.apply(
        codex_payload(base_items(cwd="/repo/webapp"), mode="implement", turn_id="t1")
    )
    second = plugin.apply(
        codex_payload(base_items(cwd="/repo/webapp"), mode="implement", turn_id="t2")
    )
    a = first["routing"]["knowledge"]["doc_context"]
    b = second["routing"]["knowledge"]["doc_context"]
    assert a["chars"] == b["chars"]
    assert a["source_files"] == b["source_files"]


def test_documentation_context_is_off_switched_off():
    from llm_router_plugins.utils.knowledge.agentic_knowledge import (
        AgenticKnowledgePlugin,
    )
    from llm_router_plugins.utils.knowledge.agentic_knowledge.cache import (
        InMemoryContextCache,
    )

    plugin = AgenticKnowledgePlugin(
        config=knowledge_config(doc_context_enabled=False),
        store=FakeStore(),
        publisher=FakePublisher(),
        cache=InMemoryContextCache(),
    )
    payload = plugin.apply(
        codex_payload(base_items(cwd="/repo/webapp"), mode="implement")
    )
    assert "doc_context" not in payload["routing"]["knowledge"]


def _injected(payload):
    import json

    parts = []
    for item in payload.get("input", ()):
        if isinstance(item, dict) and item.get("role") == "developer":
            for part in item.get("content", ()):
                parts.append(part.get("text", ""))
    return "\n".join(parts) or json.dumps(payload.get("input", ""))
