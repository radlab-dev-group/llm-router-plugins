"""Shared session memory: key isolation, contract, wiring and ENV configuration."""

import json
import logging
import os
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

from llm_router_plugins.utils.routing.agentic_routing.codex.classifier import (
    SOURCE_FALLBACK,
    SOURCE_PHASE,
    SOURCE_MEMORY,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.config import (
    CodexRoutingConfig,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.payload import (
    CodexActivity,
    CodexPayloadParser,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.plugin import (
    CodexRoutingPlugin,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.state import (
    CodexMemoryConfig,
    InMemoryRoutingStateStore,
    RedisConnectionSettings,
    RedisRoutingStateStore,
    SessionRoutingState,
    build_state_store,
    connection_from_env,
    memory_config_from_raw,
    merge_state,
    record_state,
    remember_decision,
    resolve_memory,
    session_key,
    validate_connection,
)

PREFIX = "LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_"
_ROOT = pathlib.Path(__file__).resolve().parent.parent
_MEMORY_ENV = (
    "MEMORY_ENABLED",
    "MEMORY_BACKEND",
    "MEMORY_TTL_SECONDS",
    "MEMORY_MAX_SESSIONS",
    "MEMORY_MAX_EVENTS",
    "MEMORY_MAX_CALLS",
    "MEMORY_KEY_PREFIX",
    "MEMORY_MAX_RETRIES",
    "REDIS_HOST",
    "REDIS_PORT",
    "REDIS_DB",
    "REDIS_PASSWORD",
    "REDIS_PROTOCOL",
    "REDIS_USERNAME",
    "REDIS_SSL",
    "REDIS_SSL_CA_CERTS",
    "REDIS_SSL_CERTFILE",
    "REDIS_SSL_KEYFILE",
    "REDIS_SSL_CERT_REQS",
    "REDIS_SOCKET_CONNECT_TIMEOUT",
    "REDIS_SOCKET_TIMEOUT",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """No memory variable leaks between tests."""
    for name in _MEMORY_ENV:
        monkeypatch.delenv(f"{PREFIX}{name}", raising=False)


@pytest.fixture
def policy():
    return CodexMemoryConfig(enabled=True, backend="memory", ttl_seconds=900)


@pytest.fixture
def config():
    path = _ROOT / "llm_router_plugins/resources/routing/agentic_routing_codex.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["settings"]["semantic"]["enabled"] = False
    result = CodexRoutingConfig._from_raw(raw)
    result.validate_args()
    return result


def command(text, call_id="c1", name="exec_command"):
    return {
        "type": "function_call",
        "name": name,
        "call_id": call_id,
        "arguments": json.dumps({"cmd": text}),
    }


def call_output(
    call_id="c1", text="Process exited with code 0", name="exec_command"
):
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": text,
        "name": name,
    }


def payload(
    *items,
    turn="turn-1",
    session="s1",
    thread="t1",
    agent="/root",
    user="Dodaj funkcję.",
):
    body = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": user}],
        }
    ]
    body.extend(items)
    return {
        "model": "auto_codex",
        "input": body,
        "client_metadata": {
            "session_id": session,
            "thread_id": thread,
            "turn_id": turn,
            "x-codex-turn-metadata": json.dumps({"agent_name": agent}),
        },
    }


# --- identity ----------------------------------------------------------------


def test_key_isolates_session_thread_and_agent():
    base = session_key("ns", "session", "thread", "/root")
    assert base
    assert session_key("ns", "session", "thread", "worker") != base
    assert session_key("ns", "session", "other", "/root") != base
    assert session_key("ns", "other", "thread", "/root") != base
    assert session_key("other-ns", "session", "thread", "/root") != base
    assert session_key("ns", "session", "thread", "/root") == base


@pytest.mark.parametrize(
    "session, thread",
    [
        ("", "thread"),
        ("session", ""),
        ("", ""),
    ],
)
def test_insufficient_identity_stays_stateless(session, thread):
    assert session_key("ns", session, thread, "/root") is None


def test_hostile_identifier_cannot_escape_the_namespace():
    hostile = "a:../../etc:b"
    key = session_key("ns", hostile, "thread", "/root")
    assert key.count(":") == 4
    assert key.startswith("ns:v1:")
    assert key != session_key("ns", "a", "thread", "/root")
    assert key == session_key("ns", hostile, "thread", "/root")


# --- store contract ----------------------------------------------------------


def test_write_then_read_roundtrip(policy):
    store = InMemoryRoutingStateStore(policy)
    key = session_key(policy.key_prefix, "s", "t", "a")
    assert (
        store.write(key, SessionRoutingState(mode="test", generation="g1"), 0).state
        == "written"
    )
    state, status = store.read(key)
    assert status.state == "hit"
    assert (state.mode, state.generation, state.version) == ("test", "g1", 1)


def test_stale_version_never_overwrites(policy):
    store = InMemoryRoutingStateStore(policy)
    key = session_key(policy.key_prefix, "s", "t", "a")
    store.write(key, SessionRoutingState(mode="test"), 0)
    status = store.write(key, SessionRoutingState(mode="debug"), 0)
    assert status.state == "conflict"
    assert store.read(key)[0].mode == "test"
    assert store.write(key, SessionRoutingState(mode="debug"), 1).state == "written"
    assert store.read(key)[0].mode == "debug"


def test_write_over_a_vanished_record_is_a_conflict(policy):
    store = InMemoryRoutingStateStore(policy)
    key = session_key(policy.key_prefix, "s", "t", "a")
    assert store.write(key, SessionRoutingState(mode="test"), 7).state == "conflict"


def test_ttl_expires_a_session(policy):
    clock = SimpleNamespace(now=1000.0)
    store = InMemoryRoutingStateStore(
        CodexMemoryConfig(enabled=True, backend="memory", ttl_seconds=30),
        clock=lambda: clock.now,
    )
    key = session_key("ns", "s", "t", "a")
    store.write(key, SessionRoutingState(mode="test"), 0)
    clock.now += 29
    assert store.read(key)[1].state == "hit"
    clock.now += 2
    assert store.read(key)[1].state == "expired"
    assert store.read(key)[1].state == "miss"


def test_session_cap_evicts_the_oldest(policy):
    capped = CodexMemoryConfig(
        enabled=True, backend="memory", ttl_seconds=900, max_sessions=2
    )
    store = InMemoryRoutingStateStore(capped)
    for index in range(3):
        store.write(
            session_key("ns", f"s{index}", "t", "a"),
            SessionRoutingState(mode="test"),
            0,
        )
        time.sleep(0.001)
    assert store.read(session_key("ns", "s0", "t", "a"))[1].state == "miss"
    assert store.read(session_key("ns", "s2", "t", "a"))[1].state == "hit"


@pytest.mark.parametrize(
    "raw",
    ["", "[]", '{"v": 99, "mode": "test"}', '{"v": 1}', '{"v": 1, "mode": ""}'],
)
def test_unusable_serialization_parses_to_nothing(raw):
    assert SessionRoutingState.from_json(raw) is None


def test_serialization_roundtrip_carries_every_field():
    state = SessionRoutingState(
        mode="debug",
        kind="test_failure",
        reason="test command failed",
        generation="g",
        event_id="e",
        fingerprint="f",
        pending_calls=("c1",),
        seen_events=("e1", "e2"),
        version=3,
    )
    assert SessionRoutingState.from_json(state.to_json()) == state


@pytest.mark.parametrize("fail", ["read", "write"])
def test_injected_store_failure_reports_unavailable_without_raising(policy, fail):
    store = InMemoryRoutingStateStore(policy)
    key = session_key("ns", "s", "t", "a")
    if fail == "read":
        store.fail_reads()
        assert store.read(key)[1].state == "unavailable"
    else:
        store.fail_writes()
        assert (
            store.write(key, SessionRoutingState(mode="test"), 0).state
            == "unavailable"
        )


# --- merge semantics ---------------------------------------------------------


def test_new_command_generation_drops_the_carried_phase():
    stored = SessionRoutingState(mode="test", generation="turn-1")
    assert merge_state(stored, "turn-2", True) is None
    assert merge_state(stored, "turn-1", True).mode == "test"
    assert merge_state(stored, "", True) is None
    assert merge_state(stored, "turn-1", False) is None
    assert merge_state(None, "turn-1", True) is None


def test_record_state_bounds_the_identifier_lists():
    config = CodexMemoryConfig(
        enabled=True, backend="memory", max_events=2, max_calls=1
    )
    previous = SessionRoutingState(
        seen_events=("e1", "e2"), pending_calls=("c1", "c2")
    )
    record = record_state(
        "test", "command", "python -m pytest", "g", "e3", "fp", previous, config
    )
    assert record.seen_events == ("e2", "e3")
    assert record.pending_calls == ("c2",)


def _phase_request(config, *items, turn="turn-1"):
    from llm_router_plugins.utils.routing.agentic_routing.codex.phase import (
        detect_phase_evidence,
    )

    request = CodexPayloadParser().parse(payload(*items, turn=turn))
    evidence = detect_phase_evidence(request.activity, config.phase)
    return request, SimpleNamespace(
        source="phase",
        mode=evidence.mode,
        reason=evidence.reason,
        evidence=evidence,
    )


def test_remember_accumulates_bounded_events_and_pending_calls(config):
    policy = CodexMemoryConfig(
        enabled=True, backend="memory", max_events=2, max_calls=1
    )
    store = InMemoryRoutingStateStore(policy)
    first = dict(command("pytest tests/test_a.py", "c1"), id="e1")
    second = dict(command("pytest tests/test_a.py", "c2"), id="e2")
    third = dict(command("pytest tests/test_a.py", "c3"), id="e3")
    for items in ((first,), (first, second), (first, second, third)):
        request, decision = _phase_request(config, *items)
        assert remember_decision(store, policy, request, decision) == "written"
    state = store.read(session_key(policy.key_prefix, "s1", "t1", "/root"))[0]
    assert state.seen_events == ("e2", "e3")
    assert state.pending_calls == ("c3",)


def test_remember_rejects_older_prefix_and_retired_generation(config, policy):
    store = InMemoryRoutingStateStore(policy)
    first = dict(command("pytest tests/test_a.py", "c1"), id="e1")
    second = dict(command("git log", "c2"), id="e2")
    for items, turn in (
        ((first,), "turn-1"),
        ((first, second), "turn-1"),
        ((first,), "turn-2"),
    ):
        request, decision = _phase_request(config, *items, turn=turn)
        assert remember_decision(store, policy, request, decision) == "written"
    request, decision = _phase_request(config, first, second)
    assert remember_decision(store, policy, request, decision) == "conflict"


def test_memory_backend_warns_that_it_is_not_shared(policy, caplog):
    with caplog.at_level(logging.WARNING):
        build_state_store(policy)
    assert "isolated replay" in caplog.text


def test_startup_probe_is_not_reused_by_worker(monkeypatch):
    probe, worker = Mock(), Mock()
    factory = Mock(side_effect=[probe, worker])
    monkeypatch.setitem(sys.modules, "redis", SimpleNamespace(Redis=factory))
    settings = RedisConnectionSettings(host="cache")
    store, status = build_state_store(
        CodexMemoryConfig(enabled=True, connection=settings)
    )
    assert status.state == "written"
    probe.close.assert_called_once_with()
    worker.get.return_value = None
    assert store.read("ns:key")[1].state == "miss"
    assert factory.call_count == 2


def _preview(config, policy, state, *items):
    request = CodexPayloadParser().parse(payload(*items))
    return request, resolve_memory(state, request, config.phase, policy)


def test_incremental_failure_links_pending_call_without_tool_name(config, policy):
    request, initial = _preview(
        config, policy, None, dict(command("pytest tests/test_a.py"), id="e1")
    )
    assert initial.state.pending_calls == ("c1",)
    partial = replace(
        request,
        activity=(
            CodexActivity(
                kind="function_call_output",
                text="Process exited with code 1",
                call_id="c1",
                event_id="result-1",
            ),
        ),
    )
    resolved = resolve_memory(initial.state, partial, config.phase, policy)
    assert resolved.evidence.mode == "debug"
    assert resolved.evidence.kind == "test_failure"
    assert resolved.state.pending_calls == ()
    assert resolved.state.seen_events == ("e1", "result-1")
    replay = resolve_memory(resolved.state, partial, config.phase, policy)
    assert replay.state == resolved.state
    assert replay.evidence is None
    assert replay.carried.mode == "debug"


def test_late_incremental_failure_after_patch_does_not_roll_back_mode(
    config, policy
):
    patch = {
        "type": "function_call",
        "name": "apply_patch",
        "call_id": "patch-1",
        "id": "e2",
        "arguments": "*** Begin Patch\n*** Update File: src/main.py\n@@\n-old\n+new\n*** End Patch",
    }
    request, initial = _preview(
        config, policy, None, dict(command("pytest tests/test_a.py"), id="e1"), patch
    )
    assert initial.state.mode == "implement"
    partial = replace(
        request,
        activity=(
            CodexActivity(
                kind="function_call_output",
                text="Process exited with code 1",
                call_id="c1",
                event_id="late-result",
            ),
        ),
    )
    resolved = resolve_memory(initial.state, partial, config.phase, policy)
    assert resolved.evidence is None
    assert resolved.carried.mode == "implement"
    assert resolved.state.mode == "implement"
    assert resolved.state.pending_calls == ("patch-1",)


@pytest.mark.parametrize("text", ["Process exited with code 0", "unknown status"])
def test_success_or_unknown_incremental_status_is_not_new_phase(
    config, policy, text
):
    request, initial = _preview(
        config, policy, None, command("pytest tests/test_a.py")
    )
    partial = replace(
        request,
        activity=(CodexActivity("function_call_output", text, call_id="c1"),),
    )
    resolved = resolve_memory(initial.state, partial, config.phase, policy)
    assert resolved.evidence is None
    assert resolved.carried.mode == "test"
    assert resolved.state.pending_calls == (
        () if text != "unknown status" else ("c1",)
    )


def test_stale_and_disjoint_histories_cannot_carry_or_write(config, policy):
    first = dict(command("pytest tests/test_a.py", "c1"), id="e1")
    second = dict(command("git log", "c2"), id="e2")
    _, initial = _preview(config, policy, None, first, second)
    for items in (
        (first,),
        (dict(command("git log", "other"), id="foreign"),),
        (second, first),
    ):
        _, resolved = _preview(config, policy, initial.state, *items)
        assert resolved.status.state == "conflict"
        assert resolved.carried is None
        assert resolved.state is None


def test_changed_context_requires_causal_overlap(config, policy):
    first = dict(command("pytest tests/test_a.py"), id="e1")
    request, initial = _preview(config, policy, None, first)
    changed = replace(request, window_id="new-window", activity=())
    assert (
        resolve_memory(initial.state, changed, config.phase, policy).carried is None
    )
    changed = replace(changed, activity=request.activity)
    replay = resolve_memory(initial.state, changed, config.phase, policy)
    assert replay.carried.mode == "test"
    assert replay.state == initial.state


def test_neutral_assistant_resume_preserves_entire_record(config, policy):
    request, initial = _preview(
        config, policy, None, dict(command("pytest tests/test_a.py"), id="e1")
    )
    stored = replace(initial.state, version=7, updated_at=123.0)
    resumed = replace(
        request,
        activity=(
            CodexActivity(
                "assistant",
                "Kontynuuję",
                event_id="resume",
            ),
        ),
    )
    resolved = resolve_memory(stored, resumed, config.phase, policy)
    assert resolved.status.state == "hit"
    assert resolved.evidence is None
    assert resolved.carried == stored
    assert resolved.state == stored


@pytest.mark.parametrize(
    "activity, changed_context",
    [
        ((CodexActivity("function_call", "{}", name="unknown_tool"),), False),
        ((CodexActivity("assistant", "Kontynuuję"),), True),
        ((CodexActivity("assistant", "Teraz uruchomię pytest."),), False),
        (
            (CodexActivity("function_call_output", "unknown", call_id="foreign"),),
            False,
        ),
    ],
)
def test_disjoint_resume_requires_structural_neutrality_and_same_context(
    config,
    policy,
    activity,
    changed_context,
):
    request, initial = _preview(
        config, policy, None, dict(command("pytest tests/test_a.py"), id="e1")
    )
    resumed = replace(
        request,
        activity=activity,
        window_id="changed" if changed_context else request.window_id,
    )
    resolved = resolve_memory(initial.state, resumed, config.phase, policy)
    assert resolved.status.state == "conflict"
    assert resolved.carried is None
    assert resolved.state is None


def test_unknown_incremental_result_keeps_call_for_later_failure(config, policy):
    request, initial = _preview(
        config, policy, None, dict(command("pytest tests/test_a.py"), id="e1")
    )
    partial = replace(
        request,
        activity=(
            CodexActivity(
                "function_call_output",
                "Still running",
                call_id="c1",
                event_id="partial",
            ),
        ),
    )
    unresolved = resolve_memory(initial.state, partial, config.phase, policy)
    assert unresolved.state.pending_evidence == initial.state.pending_evidence
    assert unresolved.evidence is None
    complete = replace(
        request,
        activity=(
            CodexActivity(
                "function_call_output",
                "Process exited with code 1",
                call_id="c1",
                event_id="complete",
            ),
        ),
    )
    resolved = resolve_memory(unresolved.state, complete, config.phase, policy)
    assert resolved.evidence.mode == "debug"
    assert resolved.state.pending_calls == ()


def test_pending_bound_prevents_resolving_evicted_call(config):
    policy = CodexMemoryConfig(
        enabled=True, backend="memory", max_events=2, max_calls=1
    )
    request, initial = _preview(
        config,
        policy,
        None,
        dict(command("pytest tests/test_a.py", "c1"), id="e1"),
        dict(command("git log", "c2"), id="e2"),
    )
    partial = replace(
        request,
        activity=(
            CodexActivity(
                "function_call_output", "Process exited with code 1", call_id="c1"
            ),
        ),
    )
    resolved = resolve_memory(initial.state, partial, config.phase, policy)
    assert resolved.status.state == "conflict"
    assert resolved.evidence is None


def test_fork_rebuilds_client_and_script_without_inherited_lock(monkeypatch):
    first, second = Mock(), Mock()
    first.get.return_value = second.get.return_value = None
    factory = Mock(side_effect=[first, second])
    monkeypatch.setitem(sys.modules, "redis", SimpleNamespace(Redis=factory))
    store = RedisRoutingStateStore(CodexMemoryConfig(enabled=True))
    assert store.read("key")[1].state == "miss"
    original_pid = os.getpid()
    store._lock.acquire()
    store._script = Mock()
    monkeypatch.setattr(os, "getpid", lambda: original_pid + 1)
    assert store.read("key")[1].state == "miss"
    assert factory.call_count == 2
    assert store._script is None


def test_expired_memory_record_cannot_be_updated_using_old_version():
    clock = SimpleNamespace(now=0.0)
    store = InMemoryRoutingStateStore(
        CodexMemoryConfig(ttl_seconds=1), clock=lambda: clock.now
    )
    store.write("key", SessionRoutingState(mode="test"), 0)
    clock.now = 2.0
    assert (
        store.write("key", SessionRoutingState(mode="debug"), 1).state == "conflict"
    )
    assert (
        store.write("key", SessionRoutingState(mode="implement"), 0).state
        == "written"
    )


@pytest.mark.parametrize(
    "field, value",
    [
        ("ver", -1),
        ("ver", True),
        ("ver", 1.5),
        ("ts", float("nan")),
        ("ts", float("inf")),
        ("seen", [1]),
        ("pending", "call"),
        ("gen", 123),
        ("calls", [{"call_id": "c1"}]),
    ],
)
def test_malformed_record_fields_are_a_miss(field, value):
    data = json.loads(SessionRoutingState(mode="test").to_json())
    data[field] = value
    assert SessionRoutingState.from_json(json.dumps(data)) is None


def test_redis_invalid_script_reply_reports_unavailable():
    client = Mock()
    client.register_script.return_value = Mock(return_value=["unexpected", "1"])
    store = RedisRoutingStateStore(CodexMemoryConfig(), client)
    assert (
        store.write("key", SessionRoutingState(mode="test"), 0).state
        == "unavailable"
    )


def test_new_neutral_generation_retires_old_phase_without_storing_fallback(
    config, policy
):
    store = InMemoryRoutingStateStore(policy)
    request, decision = _phase_request(
        config, dict(command("pytest tests/test_a.py"), id="e1")
    )
    assert (
        remember_decision(store, policy, request, decision, rules=config.phase)
        == "written"
    )
    neutral = replace(request, turn_id="turn-2", activity=())
    fallback = SimpleNamespace(mode="implement", source="fallback", evidence=None)
    assert (
        remember_decision(store, policy, neutral, fallback, rules=config.phase)
        == "written"
    )
    state = store.read(session_key(policy.key_prefix, "s1", "t1", "/root"))[0]
    assert state.reset is True
    assert state.mode == ""
    assert resolve_memory(state, neutral, config.phase, policy).carried is None
    assert (
        remember_decision(store, policy, request, decision, rules=config.phase)
        == "conflict"
    )


def test_repeated_results_with_new_event_ids_do_not_refresh_state(config, policy):
    call = dict(command("pytest tests/test_a.py"), id="e1")
    result = dict(call_output(), id="output-1")
    _, initial = _preview(config, policy, None, call, result)
    _, replay = _preview(
        config,
        policy,
        initial.state,
        call,
        result,
        dict(call_output(), id="output-2"),
    )
    assert replay.state == initial.state
    assert replay.evidence is None


def test_repeated_call_id_with_new_event_id_does_not_roll_back_patch(config, policy):
    call = dict(command("pytest tests/test_a.py"), id="e1")
    patch = {
        "type": "function_call",
        "name": "apply_patch",
        "call_id": "p1",
        "id": "e2",
        "arguments": "*** Begin Patch\n*** Update File: src/main.py\n@@\n-old\n+new\n*** End Patch",
    }
    _, initial = _preview(config, policy, None, call, patch)
    _, replay = _preview(
        config, policy, initial.state, call, patch, dict(call, id="new-id")
    )
    assert replay.state == initial.state
    assert replay.evidence is None


def test_store_recovers_after_transient_connection_failure():
    client = Mock()
    client.get.side_effect = [
        ConnectionError("failure"),
        SessionRoutingState(mode="test").to_json(),
    ]
    store = RedisRoutingStateStore(CodexMemoryConfig(), client)
    assert store.read("key")[1].state == "unavailable"
    assert store.read("key")[1].state == "hit"


def test_cas_retry_cannot_replace_a_concurrent_generation(config, policy):
    class ConcurrentGenerationStore(InMemoryRoutingStateStore):
        raced = False

        def write(self, key, state, expected_version):
            if not self.raced:
                self.raced = True
                super().write(
                    key,
                    SessionRoutingState(mode="implement", generation="turn-2"),
                    expected_version,
                )
                return SimpleNamespace(state="conflict")
            return super().write(key, state, expected_version)

    store = ConcurrentGenerationStore(policy)
    request, decision = _phase_request(config, command("pytest tests/test_a.py"))
    assert (
        remember_decision(store, policy, request, decision, rules=config.phase)
        == "conflict"
    )
    assert (
        store.read(session_key(policy.key_prefix, "s1", "t1", "/root"))[0].generation
        == "turn-2"
    )


def test_expected_version_anchors_classification_before_concurrent_write(
    config, policy
):
    store = InMemoryRoutingStateStore(policy)
    request, decision = _phase_request(config, command("pytest tests/test_a.py"))
    key = session_key(policy.key_prefix, "s1", "t1", "/root")
    store.write(key, SessionRoutingState(mode="implement", generation="turn-2"), 0)
    assert (
        remember_decision(
            store,
            policy,
            request,
            decision,
            rules=config.phase,
            expected_version=0,
        )
        == "conflict"
    )
    assert store.read(key)[0].generation == "turn-2"


def test_incremental_failure_uses_configured_phase_modes(config, policy):
    patterns = tuple(
        replace(pattern, mode="custom-test") if pattern.mode == "test" else pattern
        for pattern in config.phase.commands
    )
    rules = replace(
        config.phase,
        test_mode="custom-test",
        failure_mode="custom-debug",
        commands=patterns,
    )
    request = CodexPayloadParser().parse(payload(command("pytest tests/test_a.py")))
    initial = resolve_memory(None, request, rules, policy)
    assert initial.state.mode == "custom-test"
    partial = replace(
        request,
        activity=(
            CodexActivity(
                "function_call_output",
                "Process exited with code 1",
                call_id="c1",
            ),
        ),
    )
    assert (
        resolve_memory(initial.state, partial, rules, policy).evidence.mode
        == "custom-debug"
    )


def test_replay_with_duplicated_wire_events_is_unchanged(config, policy):
    call = dict(command("pytest tests/test_a.py"), id="e1")
    _, initial = _preview(config, policy, None, call, call)
    _, replay = _preview(config, policy, initial.state, call, call)
    assert replay.state == initial.state


@pytest.mark.parametrize("fresh_phase", [False, True])
def test_neutral_extensions_preserve_original_evidence_deadline(config, fresh_phase):
    clock = [100.0]
    policy = CodexMemoryConfig(enabled=True, backend="memory", ttl_seconds=10)
    store = InMemoryRoutingStateStore(policy, clock=lambda: clock[0])
    call = dict(command("pytest tests/test_a.py"), id="e1")
    request, decision = _phase_request(config, call)
    key = session_key(policy.key_prefix, "s1", "t1", "/root")
    assert (
        remember_decision(store, policy, request, decision, rules=config.phase)
        == "written"
    )
    initial = store.read(key)[0]
    items = [call]
    for age in (3, 6, 9):
        clock[0] = 100.0 + age
        neutral = {
            "type": "message",
            "role": "assistant",
            "id": f"n{age}",
            "content": [{"type": "output_text", "text": "Kontynuuję"}],
        }
        items.append(neutral)
        request = CodexPayloadParser().parse(payload(*items))
        carried = SimpleNamespace(mode="test", source="memory", evidence=None)
        assert (
            remember_decision(store, policy, request, carried, rules=config.phase)
            == "written"
        )
        stored = store.read(key)[0]
        assert stored.history != initial.history
        assert stored.updated_at == initial.updated_at
        assert stored.version > initial.version
    if fresh_phase:
        clock[0] = 109.5
        request, decision = _phase_request(
            config,
            *items,
            dict(command("pytest tests/test_b.py", "c2"), id="e2"),
        )
        assert (
            remember_decision(store, policy, request, decision, rules=config.phase)
            == "written"
        )
        assert store.read(key)[0].updated_at == 109.5
    clock[0] = 110.0
    assert store.read(key)[1].state == ("hit" if fresh_phase else "expired")
    if fresh_phase:
        clock[0] = 119.5
        assert store.read(key)[1].state == "expired"


def test_state_diagnostics_never_include_exception_credentials(caplog):
    client = Mock()
    client.get.side_effect = ConnectionError(
        "redis://secret-user:secret-password@cache failed"
    )
    store = RedisRoutingStateStore(
        CodexMemoryConfig(), client, logging.getLogger("memory-secrets")
    )
    with caplog.at_level(logging.WARNING):
        _, status = store.read("key")
    assert "ConnectionError" in status.detail
    assert "secret-user" not in status.detail + caplog.text
    assert "secret-password" not in status.detail + caplog.text


@pytest.mark.parametrize("startup", [True, False])
def test_redis_tracebacks_mask_credentials_in_exception_chains(
    startup, monkeypatch, caplog
):
    policy = CodexMemoryConfig(
        enabled=True,
        backend="redis",
        connection=RedisConnectionSettings(
            host="cache",
            username="configured-user",
            password="configured-password",
        ),
    )

    def fail(*args, **kwargs):
        try:
            raise ValueError("rediss://url-user:url-password@cache invalid reply")
        except ValueError as exc:
            raise ConnectionError(
                "configured-user configured-password connection failed"
            ) from exc

    client = Mock()
    client.ping.side_effect = fail
    client.get.side_effect = fail
    monkeypatch.setitem(
        sys.modules, "redis", SimpleNamespace(Redis=Mock(return_value=client))
    )
    with caplog.at_level(logging.WARNING):
        if startup:
            store, status = build_state_store(policy, client=client)
            assert store is None
        else:
            _, status = RedisRoutingStateStore(policy, client).read("key")

    assert status.state == "unavailable"
    for secret in (
        "url-user",
        "url-password",
        "configured-user",
        "configured-password",
    ):
        assert secret not in caplog.text
    assert "[REDACTED]" in caplog.text
    assert "ValueError" in caplog.text
    assert "ConnectionError" in caplog.text
    assert "invalid reply" in caplog.text
    assert "connection failed" in caplog.text
    assert "Traceback (most recent call last)" in caplog.text


def test_large_wire_identifiers_are_bounded_but_still_link_outputs(config, policy):
    call_id, event_id = "c" * 10000, "e" * 10000
    request, initial = _preview(
        config,
        policy,
        None,
        dict(command("pytest tests/test_a.py", call_id), id=event_id),
    )
    assert len(initial.state.pending_calls[0]) <= 128
    assert len(initial.state.seen_events[0]) <= 128
    output = replace(
        request,
        activity=(
            CodexActivity(
                "function_call_output", "Process exited with code 1", call_id=call_id
            ),
        ),
    )
    assert (
        resolve_memory(initial.state, output, config.phase, policy).evidence.mode
        == "debug"
    )


# --- cascade wiring ----------------------------------------------------------


def apply(config, store, body):
    """Run one payload through the plugin and return its resolved decision."""
    plugin = CodexRoutingPlugin(
        logger=None, config=config, semantic=None, memory=store
    )
    result = plugin.apply(body)
    return SimpleNamespace(
        mode=result["agent_mode"],
        source=result["routing"]["source"],
        memory=result["routing"].get("memory", "disabled"),
    )


def test_memory_carries_a_phase_through_neutral_activity(config):
    store = InMemoryRoutingStateStore(config.memory)
    assert (
        apply(
            config, store, payload(command("python -m pytest tests/test_a.py"))
        ).source
        == SOURCE_PHASE
    )

    # A payload that carries only the result of a call it never showed is not
    # evidence of anything by itself: the memory is what keeps the phase.
    partial = payload(
        {
            "type": "function_call_output",
            "call_id": "c1",
            "output": "Process exited with code 0",
        },
    )
    decision = apply(config, store, partial)
    assert decision.source == SOURCE_MEMORY
    assert decision.mode == "test"
    assert decision.memory == "hit"


def test_a_complete_history_is_still_decided_from_its_own_evidence(config):
    store = InMemoryRoutingStateStore(config.memory)
    apply(config, store, payload(command("python -m pytest tests/test_a.py")))
    complete = payload(command("python -m pytest tests/test_a.py"), call_output())
    decision = apply(config, store, complete)
    assert decision.source == SOURCE_PHASE


def test_a_new_command_resets_the_carried_phase(config):
    store = InMemoryRoutingStateStore(config.memory)
    apply(config, store, payload(command("python -m pytest tests/test_a.py")))
    assert (
        apply(config, store, payload(user="Zmień opis opcji.", turn="turn-2")).source
        != SOURCE_MEMORY
    )


def test_fresh_evidence_beats_the_memory(config):
    store = InMemoryRoutingStateStore(config.memory)
    apply(config, store, payload(command("python -m pytest tests/test_a.py")))
    decision = apply(config, store, payload(command("git log --oneline")))
    assert decision.source == SOURCE_PHASE
    assert decision.mode == "git_review"


def test_a_fallback_decision_is_never_remembered(config):
    store = InMemoryRoutingStateStore(config.memory)
    assert (
        apply(config, store, payload(user="Opisz krótko ten moduł.")).source
        == SOURCE_FALLBACK
    )
    key = session_key(config.memory.key_prefix, "s1", "t1", "/root")
    assert store.read(key)[1].state == "miss"


def test_threads_and_agents_do_not_share_a_phase(config):
    store = InMemoryRoutingStateStore(config.memory)
    apply(config, store, payload(command("python -m pytest tests/test_a.py")))
    other = apply(
        config,
        store,
        payload(command("python -m pytest tests/test_a.py"), thread="other-thread"),
    )
    assert other.source == SOURCE_PHASE
    again = apply(config, store, payload(user="Dokończ."))
    assert again.source == SOURCE_MEMORY
    assert again.mode == "test"


def test_a_request_without_session_identity_stays_stateless(config):
    store = InMemoryRoutingStateStore(config.memory)
    body = payload(command("python -m pytest tests/test_a.py"))
    body["client_metadata"] = {"turn_id": "turn-1"}
    assert apply(config, store, body).source == SOURCE_PHASE
    follow_up = payload(user="Dokończ.")
    follow_up["client_metadata"] = {"turn_id": "turn-1"}
    assert apply(config, store, follow_up).source != SOURCE_MEMORY


def test_a_broken_store_cannot_break_the_cascade(config):
    store = InMemoryRoutingStateStore(config.memory)
    apply(config, store, payload(command("python -m pytest tests/test_a.py")))
    store.fail_reads()
    store.fail_writes()
    decision = apply(config, store, payload(user="Dokończ."))
    assert decision.mode in config.mode_by_name
    assert decision.memory == "unavailable"


def test_the_cascade_runs_statelessly_without_a_store(config):
    assert apply(config, None, payload(user="Dokończ.")).source != SOURCE_MEMORY


# --- plugin -------------------------------------------------------------------


def build_plugin(config, store):
    return CodexRoutingPlugin(
        logger=None, config=config, semantic=None, memory=store
    )


def test_plugin_writes_only_a_phase_decision(config):
    store = InMemoryRoutingStateStore(config.memory)
    plugin = build_plugin(config, store)
    key = session_key(config.memory.key_prefix, "s1", "t1", "/root")
    plugin.apply(payload(user="Opisz ten moduł."))
    assert store.read(key)[1].state == "miss"
    plugin.apply(payload(command("python -m pytest tests/test_a.py")))
    state, status = store.read(key)
    assert status.state == "hit"
    assert (state.mode, state.generation) == ("test", "turn-1")
    assert state.kind == "command"


def test_plugin_replays_the_same_history_without_changing_the_version(config):
    store = InMemoryRoutingStateStore(config.memory)
    plugin = build_plugin(config, store)
    key = session_key(config.memory.key_prefix, "s1", "t1", "/root")
    body = payload(command("python -m pytest tests/test_a.py"))
    plugin.apply(body)
    first = store.read(key)[0].version
    plugin.apply(body)
    plugin.apply(body)
    assert store.read(key)[0].version == first


def test_plugin_routes_the_memory_decision_to_the_configured_model(config):
    store = InMemoryRoutingStateStore(config.memory)
    plugin = build_plugin(config, store)
    plugin.apply(payload(command("python -m pytest tests/test_a.py")))
    result = plugin.apply(payload(user="Dokończ."))
    assert result["agent_mode"] == "test"
    assert result["routing"]["source"] == SOURCE_MEMORY
    assert result["model"] == config.mode_by_name["test"].model_name


def test_plugin_still_routes_when_the_memory_is_down(config):
    store = InMemoryRoutingStateStore(config.memory)
    plugin = build_plugin(config, store)
    plugin.apply(payload(command("python -m pytest tests/test_a.py")))
    store.fail_reads()
    store.fail_writes()
    result = plugin.apply(payload(user="Dokończ."))
    assert result["agent_mode"] in config.mode_by_name


# --- configuration ------------------------------------------------------------


def test_connection_defaults_come_from_the_env_contract():
    connection = connection_from_env(PREFIX)
    assert connection.host == ""
    assert connection.configured is False
    assert (connection.port, connection.db, connection.protocol) == (6379, 0, 3)
    assert connection.password is None
    assert connection.username is None
    assert connection.ssl is False
    assert connection.ssl_cert_reqs == "required"
    assert connection.socket_connect_timeout == 1.0
    assert connection.socket_timeout == 1.0


def test_connection_reads_types_and_normalizes_an_empty_password(monkeypatch):
    monkeypatch.setenv(f"{PREFIX}REDIS_HOST", " cache.internal ")
    monkeypatch.setenv(f"{PREFIX}REDIS_PORT", "6380")
    monkeypatch.setenv(f"{PREFIX}REDIS_DB", "3")
    monkeypatch.setenv(f"{PREFIX}REDIS_PASSWORD", "   ")
    monkeypatch.setenv(f"{PREFIX}REDIS_PROTOCOL", "2")
    connection = connection_from_env(PREFIX)
    assert connection.host == "cache.internal"
    assert connection.configured is True
    assert (connection.port, connection.db, connection.protocol) == (6380, 3, 2)
    assert connection.password is None


def test_acl_and_tls_are_opt_in(monkeypatch):
    monkeypatch.setenv(f"{PREFIX}REDIS_HOST", "cache")
    monkeypatch.setenv(f"{PREFIX}REDIS_USERNAME", "router")
    monkeypatch.setenv(f"{PREFIX}REDIS_PASSWORD", "secret")
    monkeypatch.setenv(f"{PREFIX}REDIS_SSL", "on")
    monkeypatch.setenv(f"{PREFIX}REDIS_SSL_CA_CERTS", "/etc/ssl/ca.pem")
    connection = connection_from_env(PREFIX)
    arguments = connection.client_kwargs()
    assert arguments["username"] == "router"
    assert arguments["password"] == "secret"
    assert arguments["ssl"] is True
    assert arguments["ssl_cert_reqs"] == "required"
    assert arguments["ssl_ca_certs"] == "/etc/ssl/ca.pem"


def test_tls_without_a_host_and_plaintext_kwargs():
    arguments = RedisConnectionSettings(host="h").client_kwargs()
    assert "ssl" not in arguments
    assert "username" not in arguments
    assert arguments["socket_timeout"] == 1.0


@pytest.mark.parametrize(
    "name, value",
    [
        ("REDIS_PORT", "not-a-port"),
        ("REDIS_DB", "x"),
        ("REDIS_PROTOCOL", "wat"),
        ("REDIS_SSL", "maybe"),
        ("REDIS_SSL_CERT_REQS", "sometimes"),
        ("REDIS_SOCKET_TIMEOUT", "soon"),
    ],
)
def test_malformed_connection_variables_are_rejected(monkeypatch, name, value):
    monkeypatch.setenv(f"{PREFIX}{name}", value)
    with pytest.raises(ValueError):
        connection_from_env(PREFIX)


@pytest.mark.parametrize(
    "settings, problem",
    [
        ({"port": 0}, "port"),
        ({"port": 70000}, "port"),
        ({"db": 16}, "db"),
        ({"protocol": 4}, "protocol"),
        ({"ssl_cert_reqs": "?"}, "cert_reqs"),
        ({"socket_timeout": 0}, "socket_timeout"),
        ({"socket_connect_timeout": -1}, "socket_connect_timeout"),
    ],
)
def test_out_of_range_connection_values_are_rejected(settings, problem):
    with pytest.raises(ValueError, match=problem):
        validate_connection(RedisConnectionSettings(**settings))


def test_memory_defaults_to_off_without_any_configuration():
    resolved = memory_config_from_raw(None, PREFIX)
    assert resolved.enabled is False
    assert resolved.usable is False
    assert resolved.connection.host == ""


def test_memory_json_section_is_used_when_env_is_silent():
    resolved = memory_config_from_raw(
        {
            "enabled": True,
            "backend": "memory",
            "ttl_seconds": 60,
            "max_sessions": 5,
            "max_events": 3,
            "max_calls": 2,
            "key_prefix": "custom",
            "max_retries": 0,
        },
        PREFIX,
    )
    assert (resolved.enabled, resolved.backend, resolved.ttl_seconds) == (
        True,
        "memory",
        60,
    )
    assert (resolved.max_sessions, resolved.max_events) == (5, 3)
    assert (resolved.max_calls, resolved.key_prefix, resolved.max_retries) == (
        2,
        "custom",
        0,
    )


def test_environment_overrides_the_json_for_non_secret_settings(monkeypatch):
    monkeypatch.setenv(f"{PREFIX}MEMORY_TTL_SECONDS", "45")
    monkeypatch.setenv(f"{PREFIX}MEMORY_BACKEND", "memory")
    monkeypatch.setenv(f"{PREFIX}MEMORY_MAX_SESSIONS", "7")
    monkeypatch.setenv(f"{PREFIX}MEMORY_ENABLED", "true")
    resolved = memory_config_from_raw(
        {
            "enabled": False,
            "backend": "redis",
            "ttl_seconds": 900,
            "max_sessions": 5000,
        },
        PREFIX,
    )
    assert resolved.enabled is True
    assert resolved.backend == "memory"
    assert resolved.ttl_seconds == 45
    assert resolved.max_sessions == 7


def test_connection_settings_have_no_json_counterpart():
    """A config file can never point the plugin at a Redis or carry a secret."""
    with pytest.raises(ValueError, match="Unknown settings.memory"):
        memory_config_from_raw({"host": "cache"}, PREFIX)
    with pytest.raises(ValueError, match="Unknown settings.memory"):
        memory_config_from_raw({"url": "redis://cache:6379/0"}, PREFIX)
    with pytest.raises(ValueError, match="Unknown settings.memory"):
        memory_config_from_raw({"password": "hunter2"}, PREFIX)


@pytest.mark.parametrize(
    "section",
    [
        {"backend": "memcached"},
        {"ttl_seconds": 0},
        {"max_sessions": -1},
        {"max_events": "many"},
        {"max_retries": -2},
        {"key_prefix": "has space"},
        {"key_prefix": ""},
        {"enabled": "perhaps"},
    ],
)
def test_invalid_memory_sections_are_rejected(section):
    with pytest.raises(ValueError, match="memory|settings.memory"):
        memory_config_from_raw(section, PREFIX)


def test_enabling_the_redis_backend_without_a_host_is_not_usable():
    resolved = memory_config_from_raw({"enabled": True, "backend": "redis"}, PREFIX)
    assert resolved.enabled is True
    assert resolved.usable is False


def test_enabling_redis_without_a_host_fails_validation(monkeypatch):
    path = _ROOT / "llm_router_plugins/resources/routing/agentic_routing_codex.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["settings"]["memory"] = {"enabled": True, "backend": "redis"}
    raw["settings"]["semantic"]["enabled"] = False
    config = CodexRoutingConfig._from_raw(raw)
    config.validate_args()
    assert config.memory_status.state == "unconfigured"
    plugin = CodexRoutingPlugin(config=config)
    assert plugin._memory is None
    assert plugin.apply(payload())["agent_mode"] in config.mode_by_name


def test_shipped_configuration_stays_stateless():
    path = _ROOT / "llm_router_plugins/resources/routing/agentic_routing_codex.json"
    shipped = CodexRoutingConfig._from_raw(
        json.loads(path.read_text(encoding="utf-8"))
    )
    shipped.validate_args()
    assert shipped.memory.enabled is False


def test_build_state_store_explains_why_there_is_none():
    store, status = build_state_store(CodexMemoryConfig(enabled=False))
    assert store is None and status.state == "disabled"
    store, status = build_state_store(
        CodexMemoryConfig(enabled=True, backend="redis")
    )
    assert store is None and status.state == "unconfigured"


def test_build_state_store_honours_the_backend(monkeypatch):
    client = Mock()
    monkeypatch.setitem(
        sys.modules, "redis", SimpleNamespace(Redis=Mock(return_value=client))
    )
    monkeypatch.setenv(f"{PREFIX}REDIS_HOST", "cache")
    store, status = build_state_store(
        memory_config_from_raw({"enabled": True, "backend": "memory"}, PREFIX)
    )
    assert isinstance(store, InMemoryRoutingStateStore)
    built, _ = build_state_store(
        memory_config_from_raw({"enabled": True, "backend": "redis"}, PREFIX)
    )
    assert isinstance(built, RedisRoutingStateStore)


def test_plugin_checks_redis_connection_at_startup(config, monkeypatch):
    client = Mock()
    factory = Mock(return_value=client)
    monkeypatch.setitem(sys.modules, "redis", SimpleNamespace(Redis=factory))
    monkeypatch.setenv(f"{PREFIX}MEMORY_ENABLED", "1")
    monkeypatch.setenv(f"{PREFIX}REDIS_HOST", "cache")
    monkeypatch.setenv(f"{PREFIX}REDIS_PROTOCOL", "2")
    config.memory = memory_config_from_raw(None, PREFIX)

    plugin = CodexRoutingPlugin(config=config)

    client.ping.assert_called_once_with()
    assert isinstance(plugin._memory, RedisRoutingStateStore)
    assert factory.call_args.kwargs["host"] == "cache"
    assert factory.call_args.kwargs["protocol"] == 2
    assert factory.call_args.kwargs["socket_connect_timeout"] == 1.0
    assert factory.call_args.kwargs["socket_timeout"] == 1.0


@pytest.mark.parametrize(
    "failure",
    [
        ConnectionError("Connection refused"),
        TimeoutError("Connection timed out"),
    ],
)
@pytest.mark.parametrize("with_logger", [True, False])
def test_plugin_warns_at_startup_when_redis_is_unreachable(
    config,
    monkeypatch,
    caplog,
    failure,
    with_logger,
):
    client = Mock()
    client.ping.side_effect = failure
    monkeypatch.setitem(
        sys.modules, "redis", SimpleNamespace(Redis=Mock(return_value=client))
    )
    monkeypatch.setenv(f"{PREFIX}MEMORY_ENABLED", "1")
    monkeypatch.setenv(f"{PREFIX}REDIS_HOST", "wrong-address")
    config.memory = memory_config_from_raw(None, PREFIX)
    logger = logging.getLogger("codex-startup-test") if with_logger else None

    with caplog.at_level(logging.WARNING):
        plugin = CodexRoutingPlugin(logger=logger, config=config)

    client.ping.assert_called_once_with()
    assert "Codex routing memory disabled" in caplog.text
    assert type(failure).__name__ in caplog.text
    record = next(
        record for record in caplog.records if "memory disabled" in record.message
    )
    assert "state.py" in record.message
    assert "redis_client.ping()" in record.message
    assert str(failure) in caplog.text
    assert "Traceback (most recent call last)" in caplog.text
    assert plugin._memory is None
    assert plugin.apply(payload())["agent_mode"] in config.mode_by_name


@pytest.mark.parametrize("enabled, backend", [(False, "redis"), (True, "memory")])
def test_plugin_does_not_connect_when_redis_memory_is_not_enabled(
    config,
    monkeypatch,
    enabled,
    backend,
):
    factory = Mock()
    monkeypatch.setitem(sys.modules, "redis", SimpleNamespace(Redis=factory))
    monkeypatch.setenv(f"{PREFIX}MEMORY_ENABLED", "1" if enabled else "0")
    monkeypatch.setenv(f"{PREFIX}MEMORY_BACKEND", backend)
    monkeypatch.setenv(f"{PREFIX}REDIS_HOST", "wrong-address")
    config.memory = memory_config_from_raw(None, PREFIX)

    CodexRoutingPlugin(config=config)

    factory.assert_not_called()


def test_build_state_store_reports_a_missing_client_loudly(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "redis", None)
    monkeypatch.setenv(f"{PREFIX}REDIS_HOST", "cache")
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "redis":
            raise ImportError("no redis")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    store, status = build_state_store(
        memory_config_from_raw({"enabled": True, "backend": "redis"}, PREFIX)
    )
    assert store is None
    assert status.state == "unconfigured"


@pytest.mark.parametrize(
    "reply, expected",
    [
        ([b"written", b"1"], "written"),
        ([b"conflict", b"1"], "conflict"),
    ],
)
def test_redis_write_registers_and_reuses_atomic_script(reply, expected):
    policy = CodexMemoryConfig(enabled=True, backend="redis")
    script = Mock(return_value=reply)
    client = Mock()
    client.register_script.return_value = script
    store = RedisRoutingStateStore(policy, client)
    key = session_key(policy.key_prefix, "s", "t", "a")
    state = SessionRoutingState(mode="git_review", generation="g")

    assert store.write(key, state, 0).state == expected
    assert store.write(key, state, 1).state == expected

    client.register_script.assert_called_once()
    assert "redis.call" in client.register_script.call_args.args[0]
    assert script.call_count == 2
    for expected_version, call in enumerate(script.call_args_list):
        assert call.kwargs["keys"] == [key, f"{policy.key_prefix}:v1:sessions"]
        args = call.kwargs["args"]
        saved = SessionRoutingState.from_json(args[0])
        assert saved.mode == "git_review"
        assert saved.version == expected_version + 1
        assert args[1] == expected_version
        assert args[2] == policy.ttl_seconds
        assert args[4] == policy.max_sessions


@pytest.mark.parametrize("operation", ["read", "write", "clear", "cleanup"])
@pytest.mark.parametrize("with_logger", [True, False])
def test_redis_failures_log_traceback(operation, with_logger, caplog):
    class ResponseError(Exception):
        pass

    failure = ResponseError("ERR syntax error in atomic memory script")
    client = Mock()
    client.get.side_effect = failure
    client.eval.side_effect = failure
    client.register_script.return_value = Mock(side_effect=failure)
    policy = CodexMemoryConfig(enabled=True, backend="redis")
    logger = logging.getLogger("codex-memory-test") if with_logger else None
    store = RedisRoutingStateStore(policy, client, logger=logger)
    key = session_key(policy.key_prefix, "s", "t", "a")

    with caplog.at_level(logging.WARNING):
        if operation == "read":
            state, status = store.read(key)
            assert state is None
        elif operation == "write":
            status = store.write(key, SessionRoutingState(mode="git_review"), 0)
        elif operation == "clear":
            status = store.clear(key)
        else:
            client.get.side_effect = None
            client.get.return_value = "invalid record"
            state, status = store.read(key)
            assert state is None

    assert status.state == ("miss" if operation == "cleanup" else "unavailable")
    record = next(
        record
        for record in caplog.records
        if "Codex routing memory" in record.message
    )
    assert operation in record.message
    assert "state.py" in record.message
    assert "ResponseError" in record.message
    assert str(failure) in caplog.text
    assert "Traceback (most recent call last)" in caplog.text


# --- Redis integration (skipped unless a server is reachable) -----------------

REDIS_URL = os.environ.get("CODEX_ROUTING_REDIS_URL")


def _redis_client():
    import redis

    if REDIS_URL:
        return redis.Redis.from_url(REDIS_URL, decode_responses=True)
    return redis.Redis(host="127.0.0.1", port=6379, decode_responses=True)


def _namespace():
    return f"codex-routing-test-{os.getpid()}-{int(time.time() * 1000)}"


redis_store = pytest.mark.skipif(
    REDIS_URL is None and os.environ.get("CODEX_ROUTING_REDIS_ENABLED") != "1",
    reason="set CODEX_ROUTING_REDIS_URL (or CODEX_ROUTING_REDIS_ENABLED=1) "
    "to run the Redis integration tests",
)


@redis_store
def test_two_independent_clients_share_one_session():
    namespace = _namespace()
    policy = CodexMemoryConfig(enabled=True, backend="redis", key_prefix=namespace)
    first = RedisRoutingStateStore(policy, _redis_client())
    second = RedisRoutingStateStore(policy, _redis_client())
    key = session_key(namespace, "s", "t", "a")
    first.clear(key)
    try:
        assert (
            first.write(
                key, SessionRoutingState(mode="test", generation="g"), 0
            ).state
            == "written"
        )
        state, status = second.read(key)
        assert status.state == "hit"
        assert state.mode == "test"
        assert state.version == 1
    finally:
        first.clear(key)


@redis_store
def test_concurrent_workers_cannot_both_win():
    namespace = _namespace()
    policy = CodexMemoryConfig(enabled=True, backend="redis", key_prefix=namespace)
    first = RedisRoutingStateStore(policy, _redis_client())
    second = RedisRoutingStateStore(policy, _redis_client())
    key = session_key(namespace, "s", "t", "a")
    first.clear(key)
    try:
        assert (
            first.write(key, SessionRoutingState(mode="test"), 0).state == "written"
        )
        assert (
            second.write(key, SessionRoutingState(mode="debug"), 0).state
            == "conflict"
        )
        assert second.read(key)[0].mode == "test"
        assert (
            second.write(key, SessionRoutingState(mode="debug"), 1).state
            == "written"
        )
        assert second.read(key)[0].mode == "debug"
    finally:
        first.clear(key)


@redis_store
def test_ttl_actually_expires_the_record():
    namespace = _namespace()
    policy = CodexMemoryConfig(
        enabled=True, backend="redis", key_prefix=namespace, ttl_seconds=1
    )
    store = RedisRoutingStateStore(policy, _redis_client())
    key = session_key(namespace, "s", "t", "a")
    store.clear(key)
    try:
        store.write(key, SessionRoutingState(mode="test"), 0)
        client = _redis_client()
        assert client.ttl(key) > 0
        deadline = time.time() + 5
        while time.time() < deadline and client.get(key) is not None:
            time.sleep(0.2)
        assert client.get(key) is None
    finally:
        store.clear(key)


@redis_store
@pytest.mark.parametrize("ttl_ms", [500, 5000])
@pytest.mark.parametrize("changed_phase", [False, True])
def test_redis_updates_preserve_deadline_only_for_unchanged_evidence(
    ttl_ms, changed_phase
):
    namespace = _namespace()
    policy = CodexMemoryConfig(enabled=True, backend="redis", key_prefix=namespace)
    client = _redis_client()
    store = RedisRoutingStateStore(policy, client)
    key = session_key(namespace, "s", "t", "a")
    index = f"{namespace}:v1:sessions"
    state = SessionRoutingState(mode="git_review", generation="g")
    try:
        assert store.write(key, state, 0).state == "written"
        before = store.read(key)[0]
        client.pexpire(key, ttl_ms)
        deadline = time.time() + client.pttl(key) / 1000
        time.sleep(0.02)
        incoming = replace(before, mode="test") if changed_phase else before

        assert store.write(key, incoming, before.version).state == "written"

        after, status = store.read(key)
        assert status.state == "hit"
        assert after.version == before.version + 1
        assert after.mode == incoming.mode
        if changed_phase:
            assert after.updated_at > before.updated_at
            assert client.pttl(key) > ttl_ms
            assert client.zscore(index, key) > deadline
        else:
            assert after.updated_at == before.updated_at
            assert 0 < client.pttl(key) <= ttl_ms
            assert client.zscore(index, key) == pytest.approx(deadline, abs=0.05)
    finally:
        store.clear(key)
        client.delete(index)


@redis_store
def test_session_cap_prunes_only_this_namespaces_keys():
    namespace = _namespace()
    policy = CodexMemoryConfig(
        enabled=True, backend="redis", key_prefix=namespace, max_sessions=2
    )
    store = RedisRoutingStateStore(policy, _redis_client())
    foreign = f"{namespace}-foreign"
    client = _redis_client()
    client.set(foreign, "keep me")
    keys = [session_key(namespace, f"s{index}", "t", "a") for index in range(4)]
    try:
        for key in keys:
            store.clear(key)
        for key in keys:
            store.write(key, SessionRoutingState(mode="test"), 0)
        assert client.get(foreign) == b"keep me" or client.get(foreign) == "keep me"
        assert store.read(keys[-1])[1].state == "hit"
        assert store.read(keys[0])[1].state == "miss"
    finally:
        for key in keys:
            store.clear(key)
        client.delete(foreign)
        client.delete(f"{namespace}:v1:sessions")


@redis_store
def test_a_corrupt_record_degrades_to_a_miss_and_is_dropped():
    namespace = _namespace()
    policy = CodexMemoryConfig(enabled=True, backend="redis", key_prefix=namespace)
    store = RedisRoutingStateStore(policy, _redis_client())
    key = session_key(namespace, "s", "t", "a")
    client = _redis_client()
    client.set(key, "{ this is not the record format")
    try:
        state, status = store.read(key)
        assert state is None
        assert status.state == "miss"
        assert client.get(key) is None
    finally:
        client.delete(key)
        client.delete(f"{namespace}:v1:sessions")


@redis_store
def test_corrupt_record_cleanup_cannot_delete_concurrent_valid_write():
    namespace = _namespace()
    policy = CodexMemoryConfig(enabled=True, backend="redis", key_prefix=namespace)
    client = _redis_client()
    key = session_key(namespace, "s", "t", "a")
    client.set(key, "corrupt")
    replacement = SessionRoutingState(mode="test", version=1).to_json()

    class ConcurrentRepair:
        def get(self, requested_key):
            raw = client.get(requested_key)
            client.set(requested_key, replacement)
            return raw

        def __getattr__(self, name):
            return getattr(client, name)

    store = RedisRoutingStateStore(policy, ConcurrentRepair())
    try:
        assert store.read(key)[1].state == "miss"
        assert client.get(key) == replacement
    finally:
        client.delete(key)


@redis_store
@pytest.mark.parametrize("fresh_phase", [False, True])
def test_neutral_extension_preserves_redis_evidence_deadline(config, fresh_phase):
    namespace = _namespace()
    policy = CodexMemoryConfig(enabled=True, backend="redis", key_prefix=namespace)
    client = _redis_client()
    store = RedisRoutingStateStore(policy, client)
    call = dict(command("pytest tests/test_a.py"), id="e1")
    request, decision = _phase_request(config, call)
    key = session_key(namespace, "s1", "t1", "/root")
    try:
        assert (
            remember_decision(store, policy, request, decision, rules=config.phase)
            == "written"
        )
        before = store.read(key)[0]
        client.pexpire(key, 500)
        ttl = client.pttl(key)
        request = CodexPayloadParser().parse(
            payload(
                call,
                {
                    "type": "message",
                    "role": "assistant",
                    "id": "neutral",
                    "content": [{"type": "output_text", "text": "Kontynuuję"}],
                },
            )
        )
        carried = SimpleNamespace(mode="test", source="memory", evidence=None)
        assert (
            remember_decision(store, policy, request, carried, rules=config.phase)
            == "written"
        )
        after = store.read(key)[0]
        assert after.history != before.history
        assert after.version == before.version + 1
        assert after.updated_at == before.updated_at
        assert 0 < client.pttl(key) <= ttl
        if fresh_phase:
            request, decision = _phase_request(
                config,
                call,
                {
                    "type": "message",
                    "role": "assistant",
                    "id": "neutral",
                    "content": [{"type": "output_text", "text": "Kontynuuję"}],
                },
                dict(command("pytest tests/test_b.py", "c2"), id="e2"),
            )
            assert (
                remember_decision(
                    store, policy, request, decision, rules=config.phase
                )
                == "written"
            )
            assert store.read(key)[0].updated_at > before.updated_at
            assert client.pttl(key) > (policy.ttl_seconds - 1) * 1000
        time.sleep(0.55)
        assert store.read(key)[1].state == ("hit" if fresh_phase else "miss")
    finally:
        store.clear(key)
        client.delete(f"{namespace}:v1:sessions")


@redis_store
def test_neutral_assistant_resume_does_not_refresh_redis_ttl(config):
    namespace = _namespace()
    policy = CodexMemoryConfig(enabled=True, backend="redis", key_prefix=namespace)
    client = _redis_client()
    store = RedisRoutingStateStore(policy, client)
    request, initial = _preview(
        config, policy, None, dict(command("pytest tests/test_a.py"), id="e1")
    )
    key = session_key(namespace, "s1", "t1", "/root")
    try:
        assert store.write(key, initial.state, 0).state == "written"
        before = store.read(key)[0]
        client.pexpire(key, 5000)
        ttl = client.pttl(key)
        resumed = replace(
            request,
            activity=(
                CodexActivity(
                    "assistant",
                    "Kontynuuję",
                    event_id="resume",
                ),
            ),
        )
        decision = SimpleNamespace(mode="test", source="memory", evidence=None)
        assert (
            remember_decision(
                store,
                policy,
                resumed,
                decision,
                rules=config.phase,
                expected_version=before.version,
            )
            == "unchanged"
        )
        assert 0 < client.pttl(key) <= ttl
        assert store.read(key)[0] == before
    finally:
        store.clear(key)
        client.delete(f"{namespace}:v1:sessions")


@redis_store
def test_incremental_failure_crosses_clients_and_replay_does_not_extend_ttl(config):
    namespace = _namespace()
    policy = CodexMemoryConfig(enabled=True, backend="redis", key_prefix=namespace)
    first = RedisRoutingStateStore(policy, _redis_client())
    second = RedisRoutingStateStore(policy, _redis_client())
    request, initial = _preview(
        config, policy, None, dict(command("pytest tests/test_a.py"), id="e1")
    )
    decision = SimpleNamespace(
        mode="test", source="phase", evidence=initial.evidence
    )
    key = session_key(namespace, "s1", "t1", "/root")
    try:
        assert (
            remember_decision(first, policy, request, decision, rules=config.phase)
            == "written"
        )
        partial = replace(
            request,
            activity=(
                CodexActivity(
                    "function_call_output",
                    "Process exited with code 1",
                    call_id="c1",
                    event_id="result-1",
                ),
            ),
        )
        stored = second.read(key)[0]
        resolved = resolve_memory(stored, partial, config.phase, policy)
        assert resolved.evidence.mode == "debug"
        decision = SimpleNamespace(
            mode="debug", source="phase", evidence=resolved.evidence
        )
        assert (
            remember_decision(second, policy, partial, decision, rules=config.phase)
            == "written"
        )
        assert first.read(key)[0].pending_calls == ()
        client = _redis_client()
        client.pexpire(key, 5000)
        version = first.read(key)[0].version
        decision = SimpleNamespace(mode="debug", source="memory", evidence=None)
        assert (
            remember_decision(first, policy, partial, decision, rules=config.phase)
            == "unchanged"
        )
        assert first.read(key)[0].version == version
        assert 0 < client.pttl(key) <= 5000
    finally:
        first.clear(key)
        _redis_client().delete(f"{namespace}:v1:sessions")


@redis_store
def test_real_concurrent_cas_has_exactly_one_winner():
    namespace = _namespace()
    policy = CodexMemoryConfig(enabled=True, backend="redis", key_prefix=namespace)
    stores = [RedisRoutingStateStore(policy, _redis_client()) for _ in range(2)]
    key = session_key(namespace, "s", "t", "a")
    barrier = Barrier(2)

    def write(index):
        barrier.wait(timeout=5)
        return stores[index].write(key, SessionRoutingState(mode="test"), 0).state

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            statuses = list(pool.map(write, range(2)))
        assert sorted(statuses) == ["conflict", "written"]
        assert stores[0].read(key)[0].version == 1
    finally:
        stores[0].clear(key)
        _redis_client().delete(f"{namespace}:v1:sessions")


@redis_store
def test_pruning_never_deletes_foreign_key_even_if_index_contains_it():
    namespace = _namespace()
    policy = CodexMemoryConfig(
        enabled=True, backend="redis", key_prefix=namespace, max_sessions=1
    )
    client = _redis_client()
    store = RedisRoutingStateStore(policy, client)
    foreign = f"{namespace}-foreign"
    index = f"{namespace}:v1:sessions"
    key = session_key(namespace, "s", "t", "a")
    try:
        client.set(foreign, "keep")
        client.zadd(index, {foreign: time.time() + 100})
        assert (
            store.write(key, SessionRoutingState(mode="test"), 0).state == "written"
        )
        assert client.get(foreign) == "keep"
        assert store.read(key)[1].state == "hit"
    finally:
        store.clear(key)
        client.delete(foreign, index)


@redis_store
@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
def test_real_fork_reopens_connection_and_replaces_inherited_locked_mutex():
    namespace = _namespace()
    client = _redis_client()
    kwargs = client.connection_pool.connection_kwargs
    settings = RedisConnectionSettings(
        host=kwargs["host"],
        port=kwargs["port"],
        db=kwargs.get("db", 0),
        username=kwargs.get("username"),
        password=kwargs.get("password"),
        protocol=kwargs.get("protocol"),
    )
    policy = CodexMemoryConfig(
        enabled=True, key_prefix=namespace, connection=settings
    )
    store = RedisRoutingStateStore(policy)
    key = session_key(namespace, "s", "t", "a")
    try:
        assert (
            store.write(key, SessionRoutingState(mode="test"), 0).state == "written"
        )
        store._lock.acquire()
        pid = os.fork()
        if pid == 0:
            try:
                state, status = store.read(key)
                success = (
                    status.state == "hit"
                    and state.mode == "test"
                    and store._pid == os.getpid()
                )
                os._exit(0 if success else 1)
            except BaseException:
                os._exit(1)
        store._lock.release()
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        assert store.read(key)[0].mode == "test"
    finally:
        store.clear(key)
        client.delete(f"{namespace}:v1:sessions")


@redis_store
def test_a_dead_redis_never_breaks_routing(config):
    """A client that raises on every command leaves the plugin routing statelessly."""

    class Dead:
        def get(self, *args, **kwargs):
            raise ConnectionError("Connection refused")

        def set(self, *args, **kwargs):
            raise ConnectionError("Connection refused")

        def delete(self, *args, **kwargs):
            raise ConnectionError("Connection refused")

        def zrem(self, *args, **kwargs):
            raise ConnectionError("Connection refused")

        def register_script(self, script):
            def run(**kwargs):
                raise ConnectionError("Connection refused")

            return run

    namespace = _namespace()
    policy = CodexMemoryConfig(enabled=True, backend="redis", key_prefix=namespace)
    store = RedisRoutingStateStore(policy, Dead())
    routed_config = config
    routed_config.memory = policy
    plugin = CodexRoutingPlugin(
        logger=None, config=routed_config, semantic=None, memory=store
    )
    result = plugin.apply(payload(command("python -m pytest tests/test_a.py")))
    assert result["agent_mode"] == "test"
    follow_up = plugin.apply(payload(user="Dokończ."))
    assert follow_up["agent_mode"] in config.mode_by_name
