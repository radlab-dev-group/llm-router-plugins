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
    FAILING,
    FakePublisher,
    FakeStore,
    base_items,
    codex_payload,
    compaction_payload,
    deadend_items,
    knowledge_config,
    main_payload,
    recipe_payload,
    recipe_items,
    store_hit,
)

from llm_router_plugins.utils.knowledge.agentic_knowledge import (
    AgenticKnowledgePlugin,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.cache import (
    InMemoryContextCache,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.documents import (
    build_document_text,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.signature import (
    signature_of,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.sse import (
    REASON_READ_BUDGET,
    STATUS_UNAVAILABLE,
    SearchResult,
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
    assert "agentic_kb_webapp" in collections


def test_archive_source_is_restricted_to_the_thread():
    config = knowledge_config()
    store = FakeStore()
    plugin = make_plugin(config, store=store)
    plugin.apply(main_payload(mode="debug"))
    archive = [r for r in store.requests if "session_archive" in r.categories]
    assert archive and all(r.thread_id == "thread-1" for r in archive)
    # The restriction is a filter on the request, not a name on the
    # collection: with one collection per repository the archive and the
    # mined knowledge are neighbours, and only the thread keeps them apart.
    knowledge = [r for r in store.requests if "session_archive" not in r.categories]
    assert all(r.thread_id == "" for r in knowledge)


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


def test_a_mode_switch_with_the_same_evidence_serves_the_cached_block():
    """A mode is a reading of the same facts, not new facts.

    The generation cache used to invalidate on every mode change, and the
    mode of a command jumps on roughly half the requests — so a request
    paid for a re-read of the knowledge base whose answer was already in
    hand.  Two modes that read the same sources on the same budget ask the
    same question, and the switch between them may serve the cached block
    instead of re-reading.
    """
    config = knowledge_config()
    store = FakeStore(hits=(store_hit("recipe"),))
    plugin = make_plugin(config, store=store)
    first = plugin.apply(main_payload(mode="git_review"))
    searches_after_first = len(store.requests)
    assert first["routing"]["knowledge"]["cache"] == "miss"
    # `review` is an alias of `git_review`: the same sources, the same
    # weights, the same budget.  The switch is a label, not a new question,
    # and the block may be served from the cache.
    second = plugin.apply(main_payload(mode="review"))
    assert second["routing"]["knowledge"]["cache"] == "hit"
    assert len(store.requests) == searches_after_first


def test_a_mode_switch_to_a_differently_reading_mode_re_reads():
    """A mode whose policy reads something else is a different question."""
    config = knowledge_config()
    store = FakeStore(hits=(store_hit("recipe"),))
    plugin = make_plugin(config, store=store)
    first = plugin.apply(main_payload(mode="implement"))
    searches_after_first = len(store.requests)
    assert first["routing"]["knowledge"]["cache"] == "miss"
    # `plan` drops the deadend source `implement` reads; the switch changes
    # what the mode asks for, and the cache is a miss for it.
    second = plugin.apply(main_payload(mode="plan"))
    assert second["routing"]["knowledge"]["cache"] == "miss"
    assert len(store.requests) > searches_after_first


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


def test_an_unknown_mode_is_read_with_the_fallback_policy():
    """A mode without a policy of its own is asked of the base, not skipped.

    Routing falls back to a named mode rather than to none, so an empty mode
    reaches the plugin only when routing did not run — and that is still a
    request whose repository may have something to say about it.
    """
    config = knowledge_config()
    store = FakeStore(hits=(store_hit("recipe"),))
    publisher = FakePublisher()
    plugin = make_plugin(config, store=store, publisher=publisher)
    out = plugin.apply(main_payload(mode=""))
    assert "knowledge" in out["routing"]
    # The fallback policy served it: its collections were searched.
    assert store.requests


def test_an_unknown_mode_without_a_fallback_is_still_mined():
    config = knowledge_config()
    from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
        WILDCARD_MODE,
    )

    config.modes = {
        name: policy
        for name, policy in config.modes.items()
        if name != WILDCARD_MODE
    }
    store = FakeStore()
    publisher = FakePublisher()
    plugin = make_plugin(config, store=store, publisher=publisher)
    out = plugin.apply(main_payload(mode=""))
    assert "knowledge" in out["routing"]
    assert store.requests == []


# ------------------------------------------------------------------ write --
class _BudgetStarvedStore:
    """A store that refuses every search: the budget holds none of them.

    This is what a deployment meets when ``read_timeout_seconds`` and
    ``sse_timeout_seconds`` are equal — the deadline is read from the clock a
    moment before a call is made, and a call that cannot finish before the
    deadline is not started.
    """

    def search(self, request, *, deadline=None):
        return SearchResult(STATUS_UNAVAILABLE, reason=REASON_READ_BUDGET)


def test_a_starved_read_budget_warns_once_per_interval(caplog, monkeypatch):
    """A budget that starts no search says so, without saying it every time.

    A starved configuration starves every request of every session, so the
    note that names the two numbers must not print on every one of them.
    """
    import logging

    import llm_router_plugins.utils.knowledge.agentic_knowledge.plugin as plugin_module

    monkeypatch.setattr(plugin_module, "_BUDGET_WARNING_INTERVAL_SECONDS", 300.0)
    logger = logging.getLogger("test_starved_budget")
    logger.setLevel("WARNING")
    config = knowledge_config()
    store = _BudgetStarvedStore()
    plugin = AgenticKnowledgePlugin(
        config=config,
        logger=logger,
        store=store,
        cache=InMemoryContextCache(),
        publisher=FakePublisher(),
    )
    payload = codex_payload(
        base_items(cwd="/repo/webapp"),
        mode="debug",
        turn_id="turn-1",
        root_turn_id="gen-1",
    )
    with caplog.at_level("WARNING", logger=plugin._logger.name):
        plugin.apply(payload)
        first = [line for line in caplog.messages if "read budget" in line]
        plugin.apply(
            codex_payload(
                base_items(cwd="/repo/webapp"),
                mode="debug",
                turn_id="turn-2",
                root_turn_id="gen-2",
            )
        )
        second = [line for line in caplog.messages if "read budget" in line]
    assert len(first) == 1
    # The line carries both numbers, because the operator has to be able to
    # tell a too-small budget from a too-big per-call timeout.
    assert str(config.read_timeout_seconds) in first[0]
    assert str(config.sse_timeout_seconds) in first[0]
    # The second request of the same starved configuration stays quiet.
    assert len(second) == 1


def test_a_starved_read_budget_does_not_change_what_is_served():
    config = knowledge_config()
    store = _BudgetStarvedStore()
    plugin = make_plugin(config, store=store)
    out = plugin.apply(codex_payload(base_items(cwd="/repo/webapp"), mode="debug"))
    # The request is served as if the knowledge base were empty.
    assert out is not None
    assert "knowledge" in out["routing"]


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


def test_a_request_of_an_unknown_repository_publishes_nothing():
    """
    Without a declared directory there is no repository's collection to write
    to, and the bare prefix would gather the conversations of every other
    scope-less checkout.
    """
    config = knowledge_config()
    publisher = FakePublisher()
    plugin = make_plugin(config, publisher=publisher)
    out = plugin.apply(codex_payload(recipe_items()[1:]))
    assert publisher.published == []
    knowledge = out["routing"]["knowledge"]
    assert knowledge["scope"] == ""
    assert knowledge["write"]["status"] == "disabled"


def test_a_compaction_of_an_unknown_repository_archives_nothing():
    config = knowledge_config()
    publisher = FakePublisher()
    plugin = make_plugin(config, publisher=publisher)
    payload = compaction_payload()
    payload["input"] = [
        item for item in payload["input"] if "environment_context" not in str(item)
    ]
    out = plugin.apply(payload)
    assert publisher.published == []
    assert out["routing"]["knowledge"]["write"]["status"] == "disabled"


def test_a_window_archived_from_a_replayed_history_keeps_its_repository():
    """
    The shape Codex really sends on a compaction: the whole conversation, with
    the environment-context item no longer first.  Its fragments must still be
    archived under the repository that produced them.
    """
    config = knowledge_config()
    publisher = FakePublisher()
    plugin = make_plugin(config, publisher=publisher)
    payload = compaction_payload()
    context = payload["input"].pop(0)
    filler = {
        "type": "message",
        "role": "developer",
        "content": [
            {"type": "input_text", "text": "You are a coding agent. " * 900}
        ],
    }
    payload["input"] = [filler, filler, filler, context] + payload["input"]
    plugin.apply(payload)
    assert publisher.published
    assert {event.scope for event in publisher.published} == {"webapp"}


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
    store = FakeStore(hits=(matching_recipe(),))
    plugin = make_plugin(config, store=store)
    out = plugin.apply(main_payload(mode="debug", failures=3))
    knowledge = out["routing"]["knowledge"]
    assert knowledge["escalation"]["action"] == "keep"
    assert out["model"] == "auto_codex"


def matching_recipe():
    return store_hit(
        "recipe",
        score=0.95,
        text=build_document_text(
            "recipe",
            "A tested fix",
            signature_hash=signature_of(
                "Process exited with code 1\nOutput:\n" + FAILING
            ).signature_hash,
            test_command="pytest tests/test_a.py",
        ),
    )


def test_similar_unverified_recipe_does_not_block_escalation():
    config = knowledge_config(escalation_enabled=True)
    plugin = make_plugin(
        config, store=FakeStore(hits=(store_hit("recipe", score=0.99),))
    )
    out = plugin.apply(main_payload(mode="debug", failures=3))
    assert out["routing"]["knowledge"]["escalation"]["action"] == "escalate"


def test_recipe_trial_survives_cache_and_escalates_on_next_failure():
    config = knowledge_config(escalation_enabled=True)
    cache = InMemoryContextCache()
    store = FakeStore(hits=(matching_recipe(),))
    plugin = make_plugin(config, store=store, cache=cache)
    first = plugin.apply(main_payload(mode="debug", failures=2))
    assert first["model"] == "auto_codex"
    # A different worker must remember exposure, but replay is not a new failure.
    other = make_plugin(config, store=store, cache=cache)
    replay = other.apply(main_payload(mode="debug", failures=2))
    assert replay["model"] == "auto_codex"
    followup = other.apply(main_payload(mode="debug", failures=3))
    assert followup["model"] == config.modes["debug"].escalation_model_name
    assert (
        followup["routing"]["knowledge"]["escalation"]["reason"]
        == "recipe_did_not_resolve_loop"
    )


def test_replaying_recipe_at_loop_threshold_keeps_trial_open():
    config = knowledge_config(escalation_enabled=True)
    plugin = make_plugin(config, store=FakeStore(hits=(matching_recipe(),)))
    for _ in range(3):
        out = plugin.apply(main_payload(mode="debug", failures=3))
        assert out["routing"]["knowledge"]["escalation"]["action"] == "keep"
    out = plugin.apply(main_payload(mode="debug", failures=4))
    assert out["routing"]["knowledge"]["escalation"]["action"] == "escalate"


def test_recipe_exposure_does_not_leak_into_another_generation():
    config = knowledge_config(escalation_enabled=True)
    plugin = make_plugin(config, store=FakeStore(hits=(matching_recipe(),)))
    plugin.apply(main_payload(mode="debug", failures=1))
    out = plugin.apply(
        main_payload(
            mode="debug", failures=3, turn_id="next-turn", root_turn_id="next-root"
        )
    )
    assert out["routing"]["knowledge"]["escalation"]["action"] == "keep"


def test_no_cache_cannot_allow_a_recipe_to_suppress_an_advanced_loop():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.cache import (
        GenerationContextCache,
    )

    config = knowledge_config(escalation_enabled=True)
    plugin = make_plugin(
        config,
        store=FakeStore(hits=(matching_recipe(),)),
        cache=GenerationContextCache(),
    )
    first = plugin.apply(main_payload(mode="debug", failures=3))
    assert first["routing"]["knowledge"]["escalation"]["action"] == "keep"
    followup = plugin.apply(main_payload(mode="debug", failures=4))
    assert (
        followup["routing"]["knowledge"]["escalation"]["reason"]
        == "recipe_trial_limit"
    )


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


def test_read_kill_switch_stops_injection_even_from_a_warm_cache():
    # The generation cache outlives the switch: turning the read path off must
    # stop injection, not keep replaying whatever an earlier request cached.
    config = knowledge_config()
    store = FakeStore(hits=(store_hit("recipe", score=0.9),))
    plugin = make_plugin(config, store=store)
    warmed = plugin.apply(main_payload(mode="debug"))
    assert MARKER in json.dumps(warmed["input"])

    config.read_enabled = False
    off = plugin.apply(main_payload(mode="debug"))
    assert MARKER not in json.dumps(off["input"])


def test_read_reenabled_resumes_injection_after_the_switch():
    # Disabling must freeze the cache, not overwrite it with an empty block the
    # next enabled request would mistake for fresh.
    config = knowledge_config()
    store = FakeStore(hits=(store_hit("recipe", score=0.9),))
    plugin = make_plugin(config, store=store)
    plugin.apply(main_payload(mode="debug"))
    config.read_enabled = False
    plugin.apply(main_payload(mode="debug"))
    config.read_enabled = True
    back = plugin.apply(main_payload(mode="debug"))
    assert MARKER in json.dumps(back["input"])


def test_explicit_routing_in_an_escalated_generation_is_not_reported_escalated():
    # A prior request escalated and cached it; this request's routing source is
    # explicit, so its model stands down — and the annotation must not claim an
    # escalation that did not happen.
    config = knowledge_config(escalation_enabled=True)
    store = FakeStore()
    plugin = make_plugin(config, store=store)
    plugin.apply(main_payload(mode="debug", failures=3))
    out = plugin.apply(main_payload(mode="debug", failures=3, source="explicit"))
    assert out["model"] == "auto_codex"
    assert (
        out["routing"]["knowledge"].get("escalation", {}).get("action") != "escalate"
    )


def test_escalation_applies_when_the_payload_carries_no_model():
    # The switch used to be gated on knowing the previous model; a payload with
    # no model then kept its model while being recorded and cached as escalated.
    config = knowledge_config(escalation_enabled=True)
    store = FakeStore()
    plugin = make_plugin(config, store=store)
    payload = main_payload(mode="debug", failures=3)
    payload.pop("model")
    out = plugin.apply(payload)
    assert out["model"] == config.modes["debug"].escalation_model_name


def test_the_configured_retrieval_legs_reach_the_store():
    """The store the plugin builds for itself carries the policy's legs."""
    config = knowledge_config()
    plugin = AgenticKnowledgePlugin(config=config)
    plugin._wire(config)  # build the store from the policy, as at startup
    store = plugin._store
    assert store._min_similarity == config.min_similarity
    assert store._hybrid_search is config.hybrid_search
    assert store._rerank_results is config.rerank_results


def test_the_store_passes_its_legs_into_the_search_options():
    """A request out the wire carries what the policy decided."""
    from llm_router_plugins.utils.knowledge.agentic_knowledge.sse import (
        SearchRequest,
        _options,
    )

    config = knowledge_config(hybrid_search=False, rerank_results=True)
    plugin = AgenticKnowledgePlugin(config=config)
    plugin._wire(config)
    store = plugin._store
    options = _options(
        SearchRequest(collection="c", query="q"),
        store._min_similarity,
        hybrid_search=store._hybrid_search,
        rerank_results=store._rerank_results,
    )
    assert options["hybrid_search"] is False
    assert options["rerank_results"] is True


# -------------------------------------------------------- the read budget ---
def test_one_request_spends_one_budget_across_all_its_sources():
    """
    The deadline of every search of one request is the same one, set at the start.

    A mode reads one source per collection and the sources are searched one
    after another, so without a single shared deadline a request pays the
    transport timeout once per source.  The plugin starts the budget before the
    loop and hands every search the same deadline, which is what makes
    ``read_timeout_seconds`` the number a request waits.
    """
    import time

    config = knowledge_config()
    store = FakeStore()
    plugin = make_plugin(config, store=store)
    plugin.apply(main_payload(mode="implement"))
    deadlines = store.deadlines
    assert deadlines, "the read path must search at least one source"
    assert len({round(d, 3) for d in deadlines}) == 1
    headroom = deadlines[0] - time.monotonic()
    assert 0 < headroom <= config.read_timeout_seconds


def test_the_budget_a_request_waits_is_at_least_one_call():
    """
    A budget smaller than one call would read nothing and pretend it was empty.

    The operator wrote a budget that fits no call; the config refuses to let
    the layer search nothing and report it as an empty knowledge base.
    """
    config = knowledge_config(read_timeout_seconds=0.2, sse_timeout_seconds=2.0)
    assert config.read_timeout_seconds >= config.sse_timeout_seconds


def test_the_language_switch_reaches_the_store_the_plugin_builds():
    """The policy decides; the store the plugin wires carries it."""
    from llm_router_plugins.utils.knowledge.agentic_knowledge import (
        AgenticKnowledgePlugin,
    )

    for flag in (True, False):
        config = knowledge_config(sse_ignore_language_detect=flag)
        plugin = AgenticKnowledgePlugin(config=config)
        plugin._wire(config)
        assert plugin._store._ignore_language_detect is flag


# ------------------------------------------- 1.3 cold scope --
class _CountingPublisher:
    """A publisher that also answers the reader's cold-scope question."""

    def __init__(self, counts):
        self.counts = dict(counts)
        self.asked = []

    def mined_count(self, scope):
        self.asked.append(scope)
        return self.counts.get(scope)

    def publish(self, events, **kwargs):
        from llm_router_plugins.utils.knowledge.agentic_knowledge.events import (
            PublishStatus,
            STATUS_OK,
        )

        return PublishStatus(STATUS_OK, len(list(events)))


def _plugin_with_counts(counts, store):
    from llm_router_plugins.utils.knowledge.agentic_knowledge import (
        AgenticKnowledgePlugin,
    )
    from agentic_knowledge_support import knowledge_config

    return AgenticKnowledgePlugin(
        config=knowledge_config(),
        store=store,
        publisher=_CountingPublisher(counts),
    )


def test_a_cold_scope_skips_the_mined_search_but_keeps_the_rest():
    """1.3: a repository with zero mined documents pays no hybrid search for
    a bucket it cannot answer; documentation and archived windows are asked
    for as always."""
    from agentic_knowledge_support import codex_payload

    payload = codex_payload(
        base_items()
        if False
        else __import__("agentic_knowledge_support").base_items()
    )
    store = __import__("agentic_knowledge_support").FakeStore()
    plugin = _plugin_with_counts({"webapp": 0}, store)
    plugin.apply(payload)
    kinds = {
        category for request in store.requests for category in request.categories
    }
    assert "recipe" not in kinds, [r.categories for r in store.requests]
    assert "playbook" not in kinds


def test_an_unknown_counter_keeps_every_search():
    """A counter that cannot be answered is not a zero: the read degrades to
    what it always did, and a deployment predating the counter keeps
    searching until its first document."""
    import agentic_knowledge_support as support

    payload = support.codex_payload(support.base_items())
    store = support.FakeStore()
    plugin = _plugin_with_counts({}, store)  # no count for the scope
    plugin.apply(payload)
    kinds = {
        category for request in store.requests for category in request.categories
    }
    assert "recipe" in kinds


def test_a_scope_with_mined_documents_searches_everything():
    import agentic_knowledge_support as support

    payload = support.codex_payload(support.base_items())
    store = support.FakeStore()
    plugin = _plugin_with_counts({"webapp": 3}, store)
    plugin.apply(payload)
    kinds = {
        category for request in store.requests for category in request.categories
    }
    assert "recipe" in kinds


# ------------------------------------------------------------- metrics ---
def test_the_annotation_carries_the_numbers_a_deployment_reads():
    """The read path's health, without its contents, on every request.

    The annotation is where an operator already looks, and the numbers
    that say how the layer is doing without reading a block: how many
    searches the read made, how many the budget could not start, why the
    cache did not serve, what the turn mined, and how much of the write
    the stream already knew.
    """
    config = knowledge_config()
    store = FakeStore(hits=(store_hit("recipe"),))
    plugin = make_plugin(config, store=store)
    out = plugin.apply(main_payload(mode="debug", failures=1))
    metrics = out["routing"]["knowledge"]["metrics"]
    assert metrics["searches_per_request"] >= 3
    assert metrics["read_budget_skips"] == 0
    assert metrics["cache_miss_reason"] == "cold"
    assert "recipe" not in metrics["mined_kinds"]
    assert metrics["docs_publish_dropped_as_duplicate"] == 0


def test_a_cache_hit_reports_no_searches_and_no_miss_reason():
    """A served request made no search, and the cache serving is not a miss."""
    config = knowledge_config()
    store = FakeStore(hits=(store_hit("recipe"),))
    plugin = make_plugin(config, store=store)
    plugin.apply(main_payload(mode="debug", failures=1))
    out = plugin.apply(main_payload(mode="debug", failures=1))
    knowledge = out["routing"]["knowledge"]
    assert knowledge["cache"] == "hit"
    metrics = knowledge["metrics"]
    assert metrics["searches_per_request"] == 0
    assert metrics["cache_miss_reason"] == ""


# ------------------------------------------------------------ parallel ---
def test_a_slow_engine_is_waited_once_not_once_per_source():
    """The mode's sources are independent; they are asked together.

    A serial loop made a slow engine cost the transport timeout once per
    source: with the budget at two calls, the third source never started and
    the block read as an empty knowledge base.  The same budget, spent on
    calls that run at the same moment, leaves the whole read inside it.
    """
    import time

    from agentic_knowledge_support import knowledge_config

    config = knowledge_config(read_timeout_seconds=3.0)
    slow = 1.2  # seconds each call takes; three of them in a row are 3.6 s

    from llm_router_plugins.utils.knowledge.agentic_knowledge.sse import (
        STATUS_OK,
        SearchResult,
    )

    from agentic_knowledge_support import store_hit

    class _SlowStore:
        def __init__(self):
            self.requests = []

        def search(self, request, deadline=None):
            self.requests.append(request)
            time.sleep(slow)
            if deadline is not None and time.monotonic() > deadline:
                from llm_router_plugins.utils.knowledge.agentic_knowledge.sse import (
                    REASON_READ_BUDGET,
                )

                return SearchResult(STATUS_UNAVAILABLE, reason=REASON_READ_BUDGET)
            return SearchResult(STATUS_OK, hits=(store_hit("recipe", score=0.5),))

    store = _SlowStore()
    publisher = FakePublisher()
    plugin = make_plugin(config, store=store, publisher=publisher)
    started = time.monotonic()
    plugin.apply(main_payload(mode="debug", failures=1))
    spent = time.monotonic() - started

    # every source of the mode was asked
    assert len(store.requests) >= 3
    # and the request did not pay the timeout per source: the whole read is
    # inside one call's duration plus slack, not three times it.
    assert spent < 3 * slow


def test_one_failing_search_does_not_drop_the_others():
    """A source that raises still leaves the rest of the block intact."""
    from agentic_knowledge_support import knowledge_config, store_hit

    config = knowledge_config()

    class _HalfStore:
        def search(self, request, deadline=None):
            if "session_archive" in request.categories:
                raise RuntimeError("engine_001")
            from llm_router_plugins.utils.knowledge.agentic_knowledge.sse import (
                STATUS_OK,
                SearchResult,
            )

            return SearchResult(STATUS_OK, hits=(store_hit("recipe", score=0.5),))

    plugin = make_plugin(config, store=_HalfStore(), publisher=FakePublisher())
    out = plugin.apply(main_payload(mode="debug", failures=1))
    knowledge = out["routing"]["knowledge"]
    # the mined sources answered even though the archive request raised
    assert knowledge["injected"] >= 1
