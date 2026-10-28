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


def test_a_debug_query_carries_the_failure_the_miner_read():
    """The query asks with the readable signature, not only the digest.

    The miner normalizes the failure on the writer's side and returns the
    readable text; the reader must put that text into the query, because a
    digest is not a question.  The test command travels with it for a mode
    that verifies.
    """
    base = builder()
    turn = turn_for(mode="debug", failures=2)
    b = QueryBuilder(
        collection_namer=base.collection_namer,
        min_similarity=base.min_similarity,
        signature="FAILED tests/test_a.py::test_x - KeyError: 'missing'",
        test_command="pytest tests/test_a.py",
    )
    query = b.query_for(turn)
    assert "KeyError: 'missing'" in query


def test_test_mode_uses_the_test_command_it_was_given():
    base = builder()
    turn = turn_for(mode="test")
    given = QueryBuilder(
        collection_namer=base.collection_namer,
        min_similarity=base.min_similarity,
        test_command="go test ./...",
    )
    query = given.query_for(turn)
    assert "go test ./..." in query


def test_a_query_never_grows_past_the_ceiling():
    base = builder()
    turn = turn_for(mode="debug", failures=2)
    b = QueryBuilder(
        collection_namer=base.collection_namer,
        min_similarity=base.min_similarity,
        signature="x" * 2000,
    )
    assert len(b.query_for(turn)) <= 900


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



def test_non_archive_requests_are_not_thread_restricted():
    turn = turn_for(mode="debug")
    requests = builder().build(turn, knowledge_config().modes["debug"])
    non_archive = [r for r in requests if "session_archive" not in r.categories]
    assert all(r.thread_id == "" for r in non_archive)


def test_a_thread_restriction_never_mixes_the_archive_with_knowledge():
    """
    The engine intersects filter families, so a request that carries a
    ``thread_id`` and asks for more than the archive would cut the mined
    knowledge to the documents sharing this thread — silently dropping every
    recipe mined in an earlier session.  The builder must not produce one:
    a request that names a thread names only the archive's category.
    """
    turn = turn_for(mode="debug")
    for policy in (
        knowledge_config().modes["debug"],
        knowledge_config(collection_layout="split").modes["debug"],
    ):
        for request in builder().build(turn, policy):
            if request.thread_id:
                assert set(request.categories) == {"session_archive"}, request


def test_one_repository_collection_still_answers_three_requests():
    """Merged collections, not merged requests: the filters do the separating."""
    requests = builder().build(turn_for(mode="debug"), knowledge_config().modes["debug"])
    assert len(requests) == 3
    assert {request.collection for request in requests} == {"agentic_kb_webapp"}
    assert {tuple(sorted(request.categories)) for request in requests} == {
        ("deadend", "playbook", "recipe"),
        ("doc",),
        ("session_archive",),
    }
    # Each request keeps its own metadata dimension: the thread on the archive,
    # the doc types on the documentation, nothing on the mined knowledge.
    by_categories = {request.categories: request for request in requests}
    assert by_categories[("session_archive",)].thread_id == "thread-1"
    assert by_categories[("doc",)].doc_types == ("guide", "note")
    knowledge = by_categories[("recipe", "deadend", "playbook")]
    assert knowledge.thread_id == ""
    assert knowledge.doc_types == ()


def _rebuild(turn, **overrides):
    from dataclasses import replace

    return replace(turn, **overrides)


def test_a_review_turn_that_named_nothing_still_asks_the_failure():
    """A mode whose own shape is empty is not a mode that asks nothing.

    A review whose patches named no path and whose user said no word would
    build an empty question from its own shape — and an empty question is a
    request that never asked the knowledge base.  The fallback asks with the
    one fact the turn does carry: the failure it is stuck on.
    """
    from agentic_knowledge_support import base_items

    # The environment context opens the turn, but the user says no word.
    items = [base_items()[0]] + [
        {
            "type": "function_call",
            "name": "exec_command",
            "call_id": "c",
            "id": "e",
            "arguments": '{"cmd": "pytest tests/test_a.py"}',
        },
        {
            "type": "function_call_output",
            "call_id": "c",
            "id": "o",
            "output": (
                "Process exited with code 1\n"
                "Output:\nFAILED tests/test_a.py::test_x - "
                "AssertionError: expected 1 got 2"
            ),
        },
    ]
    turn = CodexAgentAdapter().parse(codex_payload(items, mode="review"))
    assert not turn.latest_user_text
    query = builder().query_for(turn)
    assert "AssertionError" in query
    # And the request is actually built, not just the query.
    assert builder().build(turn, knowledge_config().modes["review"])


def test_a_turn_that_says_nothing_at_all_builds_no_query():
    """The fallback invents nothing: an empty turn asks nothing."""
    from agentic_knowledge_support import base_items

    # Only the environment context; no user command, no tool call.
    turn = CodexAgentAdapter().parse(codex_payload(base_items(), mode="review"))
    # The environment context is not the user's words.
    assert builder().query_for(_rebuild(turn, latest_user_text="")) == ""


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
            SourcePolicy(category="deadend", weight=0.9, phases=("debug", "test")),
        ),
    )
    requests = builder().build(turn_for(), policy)
    assert requests[0].phases == ("test", "debug")


def test_a_source_mode_restriction_reaches_the_request():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
        ModePolicy,
        SourcePolicy,
    )

    policy = ModePolicy(
        name="implement",
        sources=(SourcePolicy(category="recipe", weight=1.0, modes=("debug",)),),
    )
    requests = builder().build(turn_for(), policy)
    assert requests
    assert requests[0].modes == ("debug",)
    assert requests[0].phases == ()


def test_a_source_that_reads_every_mode_widens_the_request():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
        ModePolicy,
        SourcePolicy,
    )

    policy = ModePolicy(
        name="implement",
        sources=(
            SourcePolicy(category="recipe", weight=1.0, modes=("debug",)),
            SourcePolicy(category="deadend", weight=0.9),
        ),
    )
    requests = builder().build(turn_for(), policy)
    assert len(requests) == 1
    assert requests[0].modes == ()


def test_an_archive_request_never_carries_phases():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
        ModePolicy,
        SourcePolicy,
    )

    policy = ModePolicy(
        name="debug",
        sources=(
            SourcePolicy(category="recipe", weight=1.0, phases=("test",)),
            SourcePolicy(category="session_archive", weight=0.6, phases=("test",)),
        ),
    )
    requests = builder().build(turn_for(mode="debug"), policy)
    by_kind = {r.categories[0]: r for r in requests if r.categories}
    assert by_kind["recipe"].phases == ("test",)
    assert by_kind["session_archive"].phases == ()
    assert by_kind["session_archive"].thread_id == "thread-1"


# --------------------------------------------------------- documentation --
def _policy(*sources):
    from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
        ModePolicy,
    )

    return ModePolicy(name="implement", sources=tuple(sources))


def test_documentation_is_its_own_request():
    """The request, not the collection, is what separates prose from knowledge."""
    from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
        SourcePolicy,
    )

    requests = builder().build(
        turn_for(), _policy(SourcePolicy(category="doc", weight=0.5))
    )
    assert len(requests) == 1
    assert requests[0].collection == "agentic_kb_webapp"
    assert requests[0].categories == ("doc",)


def test_documentation_and_knowledge_are_two_requests():
    """
    One request carries one query and one metadata dimension, so the
    documentation's doc_type and the mined knowledge's failure signature never
    share one — whatever collection the layout gives them.
    """
    from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
        SourcePolicy,
    )

    requests = builder().build(
        turn_for(),
        _policy(
            SourcePolicy(category="recipe", weight=1.0),
            SourcePolicy(category="doc", weight=0.5, doc_types=("guide",)),
        ),
    )
    assert len(requests) == 2
    by_categories = {request.categories: request for request in requests}
    assert set(by_categories) == {("recipe",), ("doc",)}
    assert by_categories[("recipe",)].doc_types == ()
    assert by_categories[("doc",)].doc_types == ("guide",)


def test_the_documentation_query_carries_no_failure_signature():
    """
    A conventions page is not written about a stack trace.

    The mined query leads with the normalized failure in debug; asking the
    documentation collection with the same text trades the one thing it is
    filed under — the module and the concept — for noise.
    """
    turn = turn_for(mode="debug", failures=2)
    mined = builder().query_for(turn)
    assert "AssertionError" in mined
    documented = builder().docs_query_for(turn)
    assert "AssertionError" not in documented


def test_the_documentation_query_names_the_files_and_their_stems():
    """`docs/export.md` documents `src/export.py`; the stem is the bridge."""
    from llm_router_plugins.utils.knowledge.agentic_knowledge.adapters.codex import (  # noqa: E501
        CodexAgentAdapter,
    )
    from agentic_knowledge_support import (
        _heredoc_patch,
        _user,
        base_items,
        codex_payload,
    )

    items = base_items(cwd="/repo/webapp") + [
        _user("rename the export helper"),
        _heredoc_patch("c1", "e1", paths=("src/export.py",)),
    ]
    turn = CodexAgentAdapter().parse(codex_payload(items, mode="implement"))
    query = builder().docs_query_for(turn)
    assert "rename the export helper" in query
    assert "src/export.py" in query
    assert "export" in query


def test_a_doc_source_without_doc_types_restricts_nothing():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
        SourcePolicy,
    )

    requests = builder().build(
        turn_for(), _policy(SourcePolicy(category="doc", weight=0.5))
    )
    assert requests[0].doc_types == ()


def test_doc_types_are_unioned_across_the_sources_of_a_collection():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
        SourcePolicy,
    )

    requests = builder().build(
        turn_for(),
        _policy(
            SourcePolicy(category="doc", weight=0.5, doc_types=("guide",)),
            SourcePolicy(category="doc", weight=0.4, doc_types=("readme",)),
        ),
    )
    assert len(requests) == 1
    assert requests[0].doc_types == ("guide", "readme")


def test_one_unrestricted_doc_source_reads_the_whole_collection():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
        SourcePolicy,
    )

    requests = builder().build(
        turn_for(),
        _policy(
            SourcePolicy(category="doc", weight=0.5, doc_types=("guide",)),
            SourcePolicy(category="doc", weight=0.4),
        ),
    )
    assert requests[0].doc_types == ()


def test_the_documentation_request_carries_no_thread_or_phase():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
        SourcePolicy,
    )

    requests = builder().build(
        turn_for(mode="debug"), _policy(SourcePolicy(category="doc"))
    )
    request = requests[0]
    assert request.thread_id == ""
    assert request.phases == ()
    assert request.modes == ()
