"""
The context assembler: weighted voting, per-source caps, budget and rendering.

The assembler turns several search results into one block: it normalizes each
hit's score, multiplies by the source's weight and a bonus for matching the
failure or the files, deduplicates by document, cuts to the mode's character
budget, and renders sections in a fixed order.  An empty answer renders to
nothing, so there is nothing to inject.
"""

import sys
import pathlib
import datetime

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agentic_knowledge_support import knowledge_config, store_hit

from llm_router_plugins.utils.knowledge.agentic_knowledge.assembler import (
    ContextAssembler,
    KnowledgeHit,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.documents import (
    build_document_text,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.sse import (
    STATUS_OK,
    SearchResult,
)


def hit(
    category="recipe", name="recipe:20261016:deadbeef", text="fix body", score=0.9
):
    return store_hit(category, name=name, text=text, score=score)


def assemble(results, mode="debug", doc_context=None, **assembler_kwargs):
    config = knowledge_config()
    policy = config.modes[mode]
    assembler = ContextAssembler(**assembler_kwargs)
    return assembler.assemble(results, policy, doc_context=doc_context)


def results_for(*hits):
    day = datetime.date.today().strftime("%Y%m%d")
    from agentic_knowledge_support import StoreHit

    def with_header(h):
        text = build_document_text(
            h.collection and "recipe" or "recipe", h.text, date=datetime.date.today()
        )
        return StoreHit(
            collection="agentic_kb_webapp",
            document_name=f"recipe:{day}:deadbeef",
            text=text,
            score=h.score,
            score_normalized=h.score_normalized,
        )

    return tuple(
        (SearchResult(STATUS_OK, hits=(with_header(h),)), ("recipe",)) for h in hits
    )


def test_empty_results_render_to_nothing():
    context = assemble(())
    assert context.empty
    assert context.rendered == ""
    assert context.best_recipe_score == 0.0


def test_a_recipe_is_rendered_in_the_verified_fixes_section():
    context = assemble(results_for(hit(score=0.9)))
    assert not context.empty
    assert "## Verified fixes in this repo" in context.rendered
    assert context.hits
    assert context.best_recipe_score > 0


def test_deadends_are_separated_and_collected():
    # A dead end has its own section and is surfaced for the dead-ends list.
    from agentic_knowledge_support import StoreHit

    day = datetime.date.today().strftime("%Y%m%d")

    def deadend_hit(text="tried and it still failed"):
        return StoreHit(
            collection="agentic_kb_webapp",
            document_name=f"deadend:{day}:aa",
            text=build_document_text("deadend", text, date=datetime.date.today()),
            score=0.8,
            score_normalized=0.8,
        )

    context = assemble(
        ((SearchResult(STATUS_OK, hits=(deadend_hit(),)), ("deadend",)),)
    )
    assert "## Known dead ends (do not repeat)" in context.rendered
    assert context.deadends


def test_budget_trims_to_the_mode_character_limit():
    config = knowledge_config()
    policy = config.modes["implement"]
    many = [hit(name=f"recipe:1:{i}", text="x" * 40, score=0.9) for i in range(20)]
    context = assemble(results_for(*many), mode="implement")
    assert context.chars <= policy.max_chars


def test_a_document_named_for_one_kind_is_not_misclassified():
    # A recipe that is genuinely a recipe stays in the recipe section.
    context = assemble(results_for(hit(category="recipe", score=0.9)))
    assert context.best_recipe_score > 0
    assert all(h.category == "recipe" for h in context.hits)


def test_a_headerless_document_from_a_multi_kind_search_is_a_document():
    """
    Conventions are not verified fixes, whatever the search happened to ask.

    One request covers a collection's several kinds, so the request's categories
    cannot label a chunk that carries neither a header nor a plugin name.
    Guessing one would file a conventions page under “verified fixes”, at the
    recipes' weight, and hand the escalation policy a “strong recipe” that is
    only a style guide.
    """
    from agentic_knowledge_support import StoreHit

    page = StoreHit(
        collection="agentic_kb_webapp",
        document_name="conventions.md",
        text="Use four spaces and keyword arguments.",
        score=0.95,
        score_normalized=0.95,
    )
    context = assemble(
        ((SearchResult(STATUS_OK, hits=(page,)), ("recipe", "doc")),),
        mode="implement",
    )
    assert context.hits[0].category == "doc"
    assert "## Verified fixes in this repo" not in context.rendered
    assert context.best_recipe_score == 0.0


def test_a_path_bonus_cannot_outvote_a_tenfold_better_base():
    """A bonus is a shift, not a scale change.

    Under a fused rank the engine's best hit sits at ~0.033.  Before the
    rank was remapped onto its own scale, a documentation chunk that merely
    named the file being edited (0.12) outranked a chunk ten times more
    relevant, because the bonus was an order of magnitude larger than the
    whole base.  After the remap the base is 1.0 against the other's 0.1,
    and the bonus is what a bonus is allowed to do: move a close race.
    """
    import datetime

    day = datetime.date.today().strftime("%Y%m%d")
    from agentic_knowledge_support import StoreHit

    def hit(name, base, kind="recipe", paths=()):
        text = build_document_text(
            kind, "body", date=datetime.date.today(), paths=paths
        )
        return StoreHit(
            collection="agentic_kb_webapp",
            document_name=f"{kind}:{day}:{name}",
            text=text,
            score=base,
            score_normalized=base,
        )

    better = hit("aaa111", 0.033, kind="recipe")
    pathy = hit("bbb222", 0.0033, kind="doc", paths=("src/export.py",))
    result = SearchResult(STATUS_OK, hits=(better, pathy), score_scale="rrf")
    context = assemble(
        [(result, ("recipe", "doc"))], mode="debug", touched_paths=("src/export.py",)
    )
    assert context.hits[0].document_name.endswith("aaa111")
    # The remap: the best fused rank of the set is the ceiling of its scale.
    assert context.hits[0].score == pytest.approx(1.0, abs=1e-9)


def test_a_cosine_scale_is_not_rescaled():
    """Cosine already lives on [0, 1]; the engine's number is the number."""
    day = datetime.date.today().strftime("%Y%m%d")
    from agentic_knowledge_support import StoreHit

    text = build_document_text("recipe", "body", date=datetime.date.today())
    hit = StoreHit(
        collection="agentic_kb_webapp",
        document_name=f"recipe:{day}:ccc333",
        text=text,
        score=0.61,
        score_normalized=0.61,
    )
    result = SearchResult(STATUS_OK, hits=(hit,), score_scale="cosine")
    context = assemble([(result, ("recipe",))])
    assert context.hits[0].score == pytest.approx(0.61, abs=1e-9)


def test_a_forced_recipe_never_breaks_the_character_budget():
    # When nothing else fit, the best recipe is shortened to the room left —
    # not rendered at full length.  A limit that collapsed to 0 used to be read
    # by the entry renderer as "no ceiling", so a near-the-floor budget produced
    # an injection several times over max_chars.
    from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
        ModePolicy,
        SourcePolicy,
    )

    long_text = build_document_text(
        "recipe", "a long verified fix " * 40, date=datetime.date.today()
    )
    policy = ModePolicy(
        name="debug",
        max_chars=200,
        sources=(SourcePolicy(category="recipe", weight=1.0, max_items=3),),
    )
    results = (
        (SearchResult(STATUS_OK, hits=(store_hit(text=long_text),)), ("recipe",)),
    )
    context = ContextAssembler().assemble(results, policy)
    assert context.chars <= policy.max_chars


def test_a_citation_names_the_phase_the_episode_was_mined_in():
    text = build_document_text("recipe", "the fix body", phase="test")
    result = SearchResult(STATUS_OK, hits=(store_hit(text=text),))
    context = assemble([(result, ("recipe",))])
    assert context.rendered
    assert "recipe · in test" in context.rendered
    assert context.hits[0].phase == "test"


def test_a_document_without_a_phase_cites_without_one():
    text = build_document_text("recipe", "the fix body")
    result = SearchResult(STATUS_OK, hits=(store_hit(text=text),))
    context = assemble([(result, ("recipe",))])
    assert "in test" not in context.rendered
    assert context.hits[0].phase == ""


# --------------------------------------------------------- documentation --
def _doc(
    source="docs/guide.md",
    doc_type="guide",
    heading="Deploy",
    text="Run `make run`.",
    score=0.7,
    declared=True,
):
    """One documentation hit, with the header the ingestor writes."""
    kwargs = {}
    if declared:
        kwargs = {"source": source, "doc_type": doc_type, "heading": heading}
    body = build_document_text(
        "doc",
        text,
        date=datetime.date.today(),
        paths=(source,),
        **kwargs,
    )
    from agentic_knowledge_support import StoreHit

    return StoreHit(
        collection="agentic_kb_webapp_docs",
        document_name=f"doc:20261016:{abs(hash(source)) % 9999:04d}",
        text=body,
        score=score,
        score_normalized=score,
    )


def _doc_results(*hits):
    return ((SearchResult(STATUS_OK, hits=tuple(hits)), ("doc",)),)


def test_a_document_of_a_kind_the_mode_does_not_read_is_dropped():
    """`implement` reads guides and rules; the changelog is not one of them."""
    context = assemble(
        _doc_results(
            _doc(source="CHANGELOG.md", doc_type="changelog", heading=""),
        ),
        mode="implement",
    )
    assert context.hits == ()
    assert context.rendered == ""


def test_a_document_of_a_kind_the_mode_reads_is_kept():
    context = assemble(_doc_results(_doc()), mode="implement")
    assert [hit.category for hit in context.hits] == ["doc"]


def test_a_document_that_declares_no_kind_is_not_dropped():
    """
    A restriction excludes other kinds, not a missing one.

    A page indexed before the attribute existed is not a changelog; dropping
    documentation over a header format would lose a guide to a field it never
    wrote, which is the silent direction nobody debugs.
    """
    context = assemble(
        _doc_results(_doc(source="docs/guide.md", declared=False)),
        mode="implement",
    )
    assert len(context.hits) == 1


def test_a_document_about_a_file_the_turn_touches_outranks_one_that_does_not():
    """`docs/export.md` describes `src/export.py`; the shared stem is the link."""
    context = assemble(
        _doc_results(
            _doc(source="docs/other.md", heading="Other", score=0.71),
            _doc(source="docs/export.md", heading="Export", score=0.70),
        ),
        mode="implement",
        touched_paths=("src/export.py",),
    )
    assert [hit.source for hit in context.hits] == [
        "docs/export.md",
        "docs/other.md",
    ]


def test_the_documentation_bonus_does_not_move_mined_knowledge():
    """The stem bonus is about documentation; a recipe cites files directly."""
    recipe = hit(text="the fix")
    context = assemble(
        ((SearchResult(STATUS_OK, hits=(recipe,)), ("recipe",)),),
        mode="debug",
        touched_paths=("src/a.py",),
    )
    assert context.hits[0].category == "recipe"
    assert context.hits[0].doc_type == ""


def test_a_documentation_entry_is_cited_by_file_and_heading():
    context = assemble(_doc_results(_doc()), mode="implement")
    assert "docs/guide.md › Deploy" in context.rendered
    assert "[doc · guide ·" in context.rendered


def test_a_documentation_entry_without_a_heading_cites_only_the_file():
    context = assemble(
        _doc_results(_doc(source="README.md", doc_type="readme", heading="")),
        mode="plan",
    )
    assert "[doc · readme · README.md]" in context.rendered
    assert "›" not in context.rendered


def test_a_documentation_hit_carries_what_the_citation_needs():
    context = assemble(_doc_results(_doc()), mode="implement")
    cited = context.hits[0]
    assert cited.source == "docs/guide.md"
    assert cited.doc_type == "guide"
    assert cited.heading == "Deploy"


def test_the_kind_is_inferred_from_the_file_when_the_header_omits_it():
    """`CHANGELOG.md` is a changelog whatever its header forgot."""
    context = assemble(
        _doc_results(
            _doc(source="CHANGELOG.md", doc_type="", heading="", declared=False)
        ),
        mode="plan",
    )
    assert context.hits[0].doc_type == "changelog"


# --------------------------------------------------- documentation links --
def _mined(category, paths, text="the verified fix", name=None):
    """One mined document, headered with the files it speaks about."""
    body = build_document_text(
        category,
        text,
        date=datetime.date.today(),
        paths=paths,
    )
    return store_hit(category, name=name or f"{category}:20261016:beef", text=body)


def _map(*paths):
    from llm_router_plugins.utils.knowledge.agentic_knowledge.doc_context import (
        DocContext,
    )

    return DocContext(scope="webapp", text="", source_files=tuple(paths))


def test_a_recipe_points_at_the_page_documenting_the_module_it_fixed():
    """The recipe names `src/export.py`; the map holds `docs/export.md`."""
    context = assemble(
        (
            (
                SearchResult(
                    STATUS_OK, hits=(_mined("recipe", ("src/export.py",)),)
                ),
                ("recipe",),
            ),
        ),
        doc_context=_map("README.md", "docs/export.md"),
    )
    assert "the verified fix → docs/export.md" in context.rendered
    assert "→ README.md" not in context.rendered


def test_a_deadend_points_at_its_module_page_too():
    context = assemble(
        (
            (
                SearchResult(
                    STATUS_OK,
                    hits=(
                        _mined(
                            "deadend",
                            ("src/queue.py",),
                            text="the queue never drains",
                        ),
                    ),
                ),
                ("deadend",),
            ),
        ),
        doc_context=_map("docs/queue.md"),
    )
    assert "the queue never drains → docs/queue.md" in context.rendered


def test_a_documentation_entry_points_at_nothing():
    """A page is the pointed-at, not the pointer."""
    context = assemble(
        _doc_results(_doc()),
        doc_context=_map("docs/guide.md"),
    )
    assert "→" not in context.rendered


def test_a_mined_entry_without_a_documented_path_gets_no_pointer():
    """A citation is worth its characters only when it leads somewhere."""
    context = assemble(
        (
            (
                SearchResult(STATUS_OK, hits=(_mined("recipe", ("src/queue.py",)),)),
                ("recipe",),
            ),
        ),
        doc_context=_map("README.md"),
    )
    assert "→" not in context.rendered


def test_a_pointer_needs_no_retrieval_of_the_page():
    """The map is enough; the query that found the recipe never asked for docs."""
    context = assemble(
        (
            (
                SearchResult(
                    STATUS_OK, hits=(_mined("recipe", ("src/export.py",)),)
                ),
                ("recipe",),
            ),
        ),
        doc_context=_map("docs/export.md"),
    )
    assert "→ docs/export.md" in context.rendered


def test_a_pointer_is_not_a_second_citation_of_the_same_file():
    """An entry that already names the page does not point at it again."""
    context = assemble(
        (
            (
                SearchResult(
                    STATUS_OK,
                    hits=(
                        _mined(
                            "recipe",
                            ("README.md",),
                            text="the readme lied about install",
                        ),
                    ),
                ),
                ("recipe",),
            ),
        ),
        doc_context=_map("README.md"),
    )
    assert "README.md]" in context.rendered
    assert "→" not in context.rendered


def test_a_readme_points_at_the_directory_it_documents():
    """`pkg/README.md` is the page about the code in `pkg/`."""
    context = assemble(
        (
            (
                SearchResult(
                    STATUS_OK,
                    hits=(
                        _mined(
                            "recipe",
                            ("llm_router_plugins/maskers/fast_masker/masker.py",),
                            text="mask the nrb before logging",
                        ),
                    ),
                ),
                ("recipe",),
            ),
        ),
        doc_context=_map(
            "README.md", "llm_router_plugins/maskers/fast_masker/README.md"
        ),
    )
    assert (
        "mask the nrb before logging → llm_router_plugins/maskers/fast_masker/README.md"
        in context.rendered
    )


def test_the_root_readme_is_not_documentation_of_everything():
    """A shared stem with a generic name is not a link; the directory is."""
    context = assemble(
        (
            (
                SearchResult(
                    STATUS_OK,
                    hits=(_mined("recipe", ("src/export.py",), text="the fix"),),
                ),
                ("recipe",),
            ),
        ),
        doc_context=_map("README.md", "other/README.md"),
    )
    assert "→" not in context.rendered


def test_two_readmes_of_the_same_stem_do_not_cross_link():
    """A fix about the root README is not a fix about a component's README."""
    context = assemble(
        (
            (
                SearchResult(
                    STATUS_OK,
                    hits=(
                        _mined(
                            "deadend",
                            ("docs/README.md",),
                            text="the install section lied",
                        ),
                    ),
                ),
                ("deadend",),
            ),
        ),
        doc_context=_map("llm_router_plugins/maskers/fast_masker/README.md"),
    )
    assert "→" not in context.rendered


def test_a_ceiling_that_covers_the_pointer_keeps_the_line_under_it():
    """
    The tail is part of the line, so the ceiling has to leave room for it.

    When nothing else fits, the best recipe is cut to the room left.  Cutting
    the quote to a ceiling that ignored the pointer would render a line longer
    than the mode's budget, the one number the operator reasons about.
    """
    from llm_router_plugins.utils.knowledge.agentic_knowledge.assembler import (
        _Candidate,
        _entry,
    )
    from llm_router_plugins.utils.knowledge.agentic_knowledge.documents import (
        DocumentMeta,
    )

    body = "a long verified fix " * 40
    hit = KnowledgeHit(
        category="recipe",
        document_name="recipe:20261016:beef",
        text=body,
        score=0.9,
        weight=0.9,
        paths=("src/export.py",),
    )
    candidate = _Candidate(hit=hit, meta=DocumentMeta())
    pointer = " → docs/export.md"
    head_len = len(_entry(candidate)) - len(body)
    ceiling = head_len + len(body) - 10
    line = _entry(candidate, limit=ceiling, doc_files=("docs/export.md",))
    assert len(line) <= ceiling
    assert line.endswith(pointer)


def test_a_pointer_never_pushes_the_block_over_the_budget():
    context = assemble(
        (
            (
                SearchResult(
                    STATUS_OK,
                    hits=(_mined("recipe", ("src/export.py",), text="x" * 400),),
                ),
                ("recipe",),
            ),
        ),
        doc_context=_map("docs/export.md"),
        signature_hash="h",
    )
    policy = knowledge_config().modes["debug"]
    assert context.chars <= policy.max_chars
