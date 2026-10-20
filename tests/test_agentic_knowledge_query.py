"""
The query builder: a mode-shaped question and the collections to ask it.

The query differs by mode — a failure signature in debug, files in review — and
the builder also does the bookkeeping the engine cannot: the archive lives in
its own collection and is restricted to the current thread.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agentic_knowledge_support import (
    knowledge_config,
    loop_items,
    codex_payload,
)

from llm_router_plugins.utils.knowledge.agentic_knowledge.adapters.codex import (
    CodexAgentAdapter,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.query import QueryBuilder


def turn_for(mode="implement", failures=1, cwd="/repo/webapp"):
    return CodexAgentAdapter().parse(codex_payload(loop_items(failures), mode=mode))


def builder(config=None):
    config = config or knowledge_config()
    return QueryBuilder(
        collection_namer=config.collection, min_similarity=config.min_similarity
    )


def test_no_scope_means_no_requests():
    turn = turn_for()
    turn = _rebuild(turn, scope="")
    assert builder().build(turn, knowledge_config().modes["implement"]) == ()


def test_debug_query_leads_with_the_failure_signature():
    turn = turn_for(mode="debug", failures=2)
    query = builder().query_for(turn)
    assert "AssertionError" in query


def test_implement_query_includes_the_command_and_paths():
    turn = turn_for(mode="implement")
    query = builder().query_for(turn)
    assert "pytest" in query.lower() or "test" in query.lower()


def test_archive_requests_are_restricted_to_the_thread():
    turn = turn_for(mode="debug")
    requests = builder().build(turn, knowledge_config().modes["debug"])
    archive = [r for r in requests if "session_archive" in r.categories]
    assert archive and all(r.thread_id == "thread-1" for r in archive)
    assert any(r.collection.endswith("_sessions") for r in archive)


def test_non_archive_requests_are_not_thread_restricted():
    turn = turn_for(mode="debug")
    requests = builder().build(turn, knowledge_config().modes["debug"])
    non_archive = [r for r in requests if "session_archive" not in r.categories]
    assert all(r.thread_id == "" for r in non_archive)


def _rebuild(turn, **overrides):
    from dataclasses import replace

    return replace(turn, **overrides)


def test_a_category_listed_twice_still_restricts_the_archive_to_the_thread():
    # A policy that names `session_archive` twice must not turn the thread
    # filter off.  The old code decided "archive-only" from the *count* of
    # categories, so a duplicated source dropped the filter and made every
    # thread's archived window in the repository searchable.
    from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
        ModePolicy,
        SourcePolicy,
    )

    policy = ModePolicy(
        name="debug",
        sources=(
            SourcePolicy(category="session_archive", weight=0.6),
            SourcePolicy(category="session_archive", weight=0.4),
        ),
    )
    turn = turn_for(mode="debug")
    requests = builder().build(turn, policy)
    assert len(requests) == 1
    assert requests[0].categories == ("session_archive",)
    assert requests[0].thread_id == "thread-1"

def test_a_source_phase_restriction_reaches_the_request():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
        ModePolicy,
        SourcePolicy,
    )

    policy = ModePolicy(
        name="implement",
        sources=(SourcePolicy(category="recipe", weight=1.0, phases=("test",)),),
    )
    requests = builder().build(turn_for(), policy)
    assert requests
    assert requests[0].phases == ("test",)


def test_a_source_that_reads_every_phase_widens_the_request():
    """
    Two sources share one search; if either wants every phase, the other's
    restriction would drop half of what the mode asked to read.
    """
    from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
        ModePolicy,
        SourcePolicy,
    )

    policy = ModePolicy(
        name="implement",
        sources=(
            SourcePolicy(category="recipe", weight=1.0, phases=("test",)),
            SourcePolicy(category="deadend", weight=0.9),
        ),
    )
    requests = builder().build(turn_for(), policy)
    assert len(requests) == 1
    assert requests[0].categories == ("recipe", "deadend")
    assert requests[0].phases == ()


def test_restricting_sources_of_one_collection_union_their_phases():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
        ModePolicy,
        SourcePolicy,
    )

    policy = ModePolicy(
        name="implement",
        sources=(
            SourcePolicy(category="recipe", weight=1.0, phases=("test",)),
            SourcePolicy(
                category="deadend", weight=0.9, phases=("debug", "test")
            ),
        ),
    )
    requests = builder().build(turn_for(), policy)
    assert requests[0].phases == ("test", "debug")


def test_an_archive_request_never_carries_phases():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
        ModePolicy,
        SourcePolicy,
    )

    policy = ModePolicy(
        name="debug",
        sources=(
            SourcePolicy(category="recipe", weight=1.0, phases=("test",)),
            SourcePolicy(
                category="session_archive", weight=0.6, phases=("test",)
            ),
        ),
    )
    requests = builder().build(turn_for(mode="debug"), policy)
    by_kind = {r.categories[0]: r for r in requests if r.categories}
    assert by_kind["recipe"].phases == ("test",)
    assert by_kind["session_archive"].phases == ()
    assert by_kind["session_archive"].thread_id == "thread-1"
