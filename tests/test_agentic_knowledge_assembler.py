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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agentic_knowledge_support import knowledge_config, store_hit

from llm_router_plugins.utils.knowledge.agentic_knowledge.assembler import (
    ContextAssembler,
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


def assemble(results, mode="debug", **assembler_kwargs):
    config = knowledge_config()
    policy = config.modes[mode]
    assembler = ContextAssembler(**assembler_kwargs)
    return assembler.assemble(results, policy)


def results_for(*hits):
    day = datetime.date.today().strftime("%Y%m%d")
    from agentic_knowledge_support import StoreHit

    def with_header(h):
        text = build_document_text(
            h.collection and "recipe" or "recipe", h.text, date=datetime.date.today()
        )
        return StoreHit(
            collection="agentic-kb-webapp",
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
            collection="agentic-kb-webapp",
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
