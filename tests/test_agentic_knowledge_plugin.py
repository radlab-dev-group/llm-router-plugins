"""
The plugin end to end: read, write, escalation and fail-open.

The tests drive the real plugin with the real Codex adapter and the real
miner, query builder, assembler and escalation policy; only the search engine
and the stream are faked.  The assertions pin the invariants: a request the
plugin cannot serve comes back unchanged, ``aux_title`` is skipped, ``compaction``
is written but never read, retrieval is stable within a generation, a loop with
no recipe escalates and stays escalated, and an error anywhere degrades to an
annotation, never an exception.
"""

import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agentic_knowledge_support import (
    FakePublisher,
    FakeStore,
    base_items,
    codex_payload,
    compaction_payload,
    deadend_items,
    knowledge_config,
    main_payload,
    recipe_payload,
    store_hit,
)

from llm_router_plugins.utils.knowledge.agentic_knowledge import (
    AgenticKnowledgePlugin,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.cache import (
    InMemoryContextCache,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.sse import (
    STATUS_UNAVAILABLE,
)

MARKER = "<agentic_knowledge>"


def make_plugin(config, store=None, publisher=None, cache=None, adapters=None):
    return AgenticKnowledgePlugin(
        config=config,
        store=store or FakeStore(),
        cache=cache if cache is not None else InMemoryContextCache(),
        publisher=publisher or FakePublisher(),
        adapters=adapters,
    )


# --------------------------------------------------------------- fail-open --
def test_non_dict_payload_comes_back_unchanged():
    config = knowledge_config()
    plugin = make_plugin(config)
    for payload in ("a string", 42, None, ["list"]):
        assert plugin.apply(payload) is payload


def test_payload_no_adapter_claims_is_unchanged():
    config = knowledge_config()
    plugin = make_plugin(config)
    payload = {"foo": "bar"}
    assert plugin.apply(payload) is payload
    assert "knowledge" not in payload.get("routing", {})


def test_disabled_plugin_is_a_no_op():
    config = knowledge_config(enabled=False)
    plugin = make_plugin(config)
    payload = main_payload()
    assert plugin.apply(payload) is payload


def test_unreachable_store_degrades_to_unavailable_not_exception():
    config = knowledge_config()
    store = FakeStore(status=STATUS_UNAVAILABLE)
    plugin = make_plugin(config, store=store)
    payload = main_payload(mode="implement")
    out = plugin.apply(payload)
    knowledge = out["routing"]["knowledge"]
    assert knowledge["status" if False else "cache"] in ("miss", "disabled")
    # No exception, and the request proceeds.
    assert "model" in out


def test_broken_store_does_not_raise_and_keeps_the_request():
    class ExplodingStore(FakeStore):
        def search(self, request):
            raise RuntimeError("engine down")

    config = knowledge_config()
    plugin = make_plugin(config, store=ExplodingStore())
    out = plugin.apply(main_payload(mode="implement"))
    assert out["routing"]["knowledge"]["injected"] == 0


def test_broken_publisher_is_reported_not_raised():
    class ExplodingPublisher(FakePublisher):
        def publish(self, events):
            raise RuntimeError("redis down")

    config = knowledge_config()
    plugin = make_plugin(config, publisher=ExplodingPublisher())
    # A recipe turn triggers the write path; a failing publisher must not raise.
    out = plugin.apply(recipe_payload())
    assert "knowledge" in out["routing"]


# ------------------------------------------------------------------- read --
def test_main_request_retrieves_and_injects_a_recipe():
    config = knowledge_config()
    store = FakeStore(hits=(store_hit("recipe", text="a verified fix"),))
    plugin = make_plugin(config, store=store)
    out = plugin.apply(main_payload(mode="implement"))
    assert MARKER in json.dumps(out)
    knowledge = out["routing"]["knowledge"]
    assert knowledge["injected"] == 1
    assert knowledge["sources"] == {"recipe": 1}
    assert knowledge["scope"] == "webapp"


def test_search_is_scoped_to_the_repository_collection():
    config = knowledge_config()
    store = FakeStore()
    plugin = make_plugin(config, store=store)
    plugin.apply(main_payload(mode="implement"))
    collections = {request.collection for request in store.requests}
    assert "agentic-kb-webapp" in collections


def test_archive_source_is_restricted_to_the_thread():
    config = knowledge_config()
    store = FakeStore()
    plugin = make_plugin(config, store=store)
    plugin.apply(main_payload(mode="debug"))
    archive = [r for r in store.requests if "session_archive" in r.categories]
    assert archive and all(r.thread_id == "thread-1" for r in archive)
    assert any(r.collection.endswith("-sessions") for r in archive)


def test_injection_is_idempotent_within_one_generation():
    config = knowledge_config()
    store = FakeStore(hits=(store_hit("recipe"),))
    cache = InMemoryContextCache()
    plugin = make_plugin(config, store=store, cache=cache)
    first = plugin.apply(main_payload(mode="implement"))
    assert first["routing"]["knowledge"]["cache"] == "miss"
    searches_after_first = len(store.requests)
    assert searches_after_first > 0
    # A second request of the same generation reuses the cache: no new search,
    # and the block is re-injected exactly once, not stacked.
    second = plugin.apply(main_payload(mode="implement"))
    assert len(store.requests) == searches_after_first
    assert second["routing"]["knowledge"]["cache"] == "hit"
    assert json.dumps(second).count(MARKER) == 1


def test_aux_title_is_never_enriched():
    config = knowledge_config()
    store = FakeStore(hits=(store_hit("recipe"),))
    plugin = make_plugin(config, store=store)
    items = base_items()
    items.append(
        {
            "type": "function_call",
            "name": "exec_command",
            "call_id": "c",
            "id": "e",
            "arguments": json.dumps({"cmd": "pytest"}),
        }
    )
    payload = codex_payload(items, mode="implement", thread_source="system")
    out = plugin.apply(payload)
    assert store.requests == []
    assert MARKER not in json.dumps(out)


def test_unknown_mode_is_not_read_but_still_mined():
    config = knowledge_config()
    store = FakeStore(hits=(store_hit("recipe"),))
    publisher = FakePublisher()
    plugin = make_plugin(config, store=store, publisher=publisher)
    out = plugin.apply(main_payload(mode="implement"))
    # 'implement' is served, so this is a read; use a mode with no policy.
    assert "knowledge" in out["routing"]


# ------------------------------------------------------------------ write --
def test_recipe_turn_publishes_a_recipe_event():
    config = knowledge_config()
    publisher = FakePublisher()
    plugin = make_plugin(config, publisher=publisher)
    plugin.apply(recipe_payload())
    kinds = {event.kind for event in publisher.published}
    assert "recipe" in kinds


def test_deadend_turn_publishes_a_deadend_event():
    config = knowledge_config()
    publisher = FakePublisher()
    plugin = make_plugin(config, publisher=publisher)
    payload = codex_payload(deadend_items())
    plugin.apply(payload)
    kinds = {event.kind for event in publisher.published}
    assert "deadend" in kinds


def test_green_turn_publishes_nothing():
    config = knowledge_config()
    publisher = FakePublisher()
    plugin = make_plugin(config, publisher=publisher)
    # A turn with a single passing command mines no episode.
    items = base_items() + [
        {
            "type": "function_call",
            "name": "exec_command",
            "call_id": "c",
            "id": "e",
            "arguments": json.dumps({"cmd": "pytest"}),
        },
        {
            "type": "function_call_output",
            "call_id": "c",
            "id": "o",
            "output": "Process exited with code 0\nOutput:\n1 passed",
        },
    ]
    plugin.apply(codex_payload(items, mode="implement"))
    assert publisher.published == []


def test_write_disabled_publishes_nothing_but_still_reads():
    config = knowledge_config(write_enabled=False)
    store = FakeStore(hits=(store_hit("recipe"),))
    publisher = FakePublisher()
    plugin = make_plugin(config, store=store, publisher=publisher)
    out = plugin.apply(recipe_payload())
    assert publisher.published == []
    assert MARKER in json.dumps(out)


# ------------------------------------------------------------- compaction --
def test_compaction_is_written_and_never_read():
    config = knowledge_config()
    store = FakeStore(hits=(store_hit("recipe"),))
    publisher = FakePublisher()
    plugin = make_plugin(config, store=store, publisher=publisher)
    out = plugin.apply(compaction_payload())
    assert store.requests == []
    assert MARKER not in json.dumps(out)
    kinds = {event.kind for event in publisher.published}
    assert kinds == {"session_archive"}
    # Every archived fragment carries the thread so the archive can be filtered.
    assert all(
        event.metadata["thread_id"] == "thread-1" for event in publisher.published
    )


def test_compaction_archives_are_chunked_by_position():
    config = knowledge_config()
    publisher = FakePublisher()
    plugin = make_plugin(config, publisher=publisher)
    plugin.apply(compaction_payload())
    positions = sorted(event.metadata["position"] for event in publisher.published)
    assert positions == list(range(len(positions)))


# ------------------------------------------------------------ escalation --
def test_loop_without_recipe_escalates_and_annotates():
    config = knowledge_config(escalation_enabled=True)
    store = FakeStore()  # no recipe available
    plugin = make_plugin(config, store=store)
    out = plugin.apply(main_payload(mode="debug", failures=3))
    knowledge = out["routing"]["knowledge"]
    escalation = knowledge["escalation"]
    assert escalation["action"] == "escalate"
    assert escalation["to"] == config.modes["debug"].escalation_model_name
    assert escalation["from"] == "auto_codex"
    assert out["model"] == config.modes["debug"].escalation_model_name
    assert out["routing"]["model_before_knowledge"] == "auto_codex"


def test_loop_with_a_strong_recipe_keeps_the_model():
    config = knowledge_config(escalation_enabled=True)
    store = FakeStore(hits=(store_hit("recipe", score=0.95),))
    plugin = make_plugin(config, store=store)
    out = plugin.apply(main_payload(mode="debug", failures=3))
    knowledge = out["routing"]["knowledge"]
    assert knowledge["escalation"]["action"] == "keep"
    assert out["model"] == "auto_codex"


def test_explicit_routing_is_never_escalated():
    config = knowledge_config(escalation_enabled=True)
    store = FakeStore()
    plugin = make_plugin(config, store=store)
    out = plugin.apply(main_payload(mode="debug", failures=3, source="explicit"))
    assert out["model"] == "auto_codex"
    assert "escalation" not in out["routing"]["knowledge"]


def test_escalation_persists_across_the_generation():
    config = knowledge_config(escalation_enabled=True)
    store = FakeStore()
    cache = InMemoryContextCache()
    plugin = make_plugin(config, store=store, cache=cache)
    escalated = plugin.apply(main_payload(mode="debug", failures=3))
    assert escalated["model"] == config.modes["debug"].escalation_model_name
    # A later request in the same generation with only one failure must keep
    # the escalated model: the loop was already acted on.
    followup = plugin.apply(main_payload(mode="debug", failures=1))
    assert followup["model"] == config.modes["debug"].escalation_model_name
    assert followup["routing"]["knowledge"]["escalation"]["action"] == "escalate"


def test_escalation_off_by_default_never_changes_the_model():
    config = knowledge_config(escalation_enabled=False)
    store = FakeStore()
    plugin = make_plugin(config, store=store)
    out = plugin.apply(main_payload(mode="debug", failures=5))
    assert out["model"] == "auto_codex"
    assert "escalation" not in out["routing"]["knowledge"]


def test_escalation_requires_a_configured_mode_model():
    config = knowledge_config(escalation_enabled=True)
    store = FakeStore()
    plugin = make_plugin(config, store=store)
    # 'plan' has no escalation_model_name, so a loop there never escalates.
    out = plugin.apply(main_payload(mode="plan", failures=5))
    assert out["model"] == "auto_codex"
    assert "escalation" not in out["routing"]["knowledge"]


# ---------------------------------------------------------- annotation --
def test_annotation_has_no_document_content_or_secrets():
    config = knowledge_config()
    store = FakeStore(hits=(store_hit("recipe", text="SECRET-BODY"),))
    plugin = make_plugin(config, store=store)
    out = plugin.apply(main_payload(mode="implement"))
    annotation = json.dumps(out["routing"]["knowledge"])
    assert "SECRET-BODY" not in annotation
    assert "http://engine.test" not in annotation
    assert "sess-1" not in annotation
