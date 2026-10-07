"""Shared session memory: key isolation, contract, wiring and ENV configuration."""

import json
import logging
import os
import pathlib
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

from llm_router_plugins.utils.routing.agentic_routing.codex.classifier import (
    SOURCE_FALLBACK,
    SOURCE_PHASE,
    SOURCE_MEMORY,
    CodexModeClassifier,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.config import CodexRoutingConfig
from llm_router_plugins.utils.routing.agentic_routing.codex.payload import CodexPayloadParser
from llm_router_plugins.utils.routing.agentic_routing.codex.plugin import CodexRoutingPlugin
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
    session_key,
    validate_connection,
)


PREFIX = "LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_"
_ROOT = pathlib.Path(__file__).resolve().parent.parent
_MEMORY_ENV = (
    "MEMORY_ENABLED", "MEMORY_BACKEND", "MEMORY_TTL_SECONDS", "MEMORY_MAX_SESSIONS",
    "MEMORY_MAX_EVENTS", "MEMORY_MAX_CALLS", "MEMORY_KEY_PREFIX", "MEMORY_MAX_RETRIES",
    "REDIS_HOST", "REDIS_PORT", "REDIS_DB", "REDIS_PASSWORD", "REDIS_PROTOCOL",
    "REDIS_USERNAME", "REDIS_SSL", "REDIS_SSL_CA_CERTS", "REDIS_SSL_CERTFILE",
    "REDIS_SSL_KEYFILE", "REDIS_SSL_CERT_REQS", "REDIS_SOCKET_CONNECT_TIMEOUT",
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
    return {"type": "function_call", "name": name, "call_id": call_id,
            "arguments": json.dumps({"cmd": text})}


def call_output(call_id="c1", text="Process exited with code 0", name="exec_command"):
    return {"type": "function_call_output", "call_id": call_id, "output": text,
            "name": name}


def payload(*items, turn="turn-1", session="s1", thread="t1", agent="/root", user="Dodaj funkcję."):
    body = [{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": user}]}]
    body.extend(items)
    return {
        "model": "auto_codex", "input": body,
        "client_metadata": {
            "session_id": session, "thread_id": thread, "turn_id": turn,
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


@pytest.mark.parametrize("session, thread", [
    ("", "thread"), ("session", ""), ("", ""),
])
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
    assert store.write(key, SessionRoutingState(mode="test", generation="g1"), 0).state == "written"
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
        store.write(session_key("ns", f"s{index}", "t", "a"),
                    SessionRoutingState(mode="test"), 0)
        time.sleep(0.001)
    assert store.read(session_key("ns", "s0", "t", "a"))[1].state == "miss"
    assert store.read(session_key("ns", "s2", "t", "a"))[1].state == "hit"


@pytest.mark.parametrize("raw", ['', '[]', '{"v": 99, "mode": "test"}',
                                 '{"v": 1}', '{"v": 1, "mode": ""}'])
def test_unusable_serialization_parses_to_nothing(raw):
    assert SessionRoutingState.from_json(raw) is None


def test_serialization_roundtrip_carries_every_field():
    state = SessionRoutingState(
        mode="debug", kind="test_failure", reason="test command failed",
        generation="g", event_id="e", fingerprint="f",
        pending_calls=("c1",), seen_events=("e1", "e2"), version=3,
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
        assert store.write(key, SessionRoutingState(mode="test"), 0).state == "unavailable"


# --- merge semantics ---------------------------------------------------------


def test_new_command_generation_drops_the_carried_phase():
    stored = SessionRoutingState(mode="test", generation="turn-1")
    assert merge_state(stored, "turn-2", True) is None
    assert merge_state(stored, "turn-1", True).mode == "test"
    assert merge_state(stored, "", True) is None
    assert merge_state(stored, "turn-1", False) is None
    assert merge_state(None, "turn-1", True) is None


def test_record_state_bounds_the_identifier_lists():
    config = CodexMemoryConfig(enabled=True, backend="memory", max_events=2, max_calls=1)
    previous = SessionRoutingState(seen_events=("e1", "e2"), pending_calls=("c1", "c2"))
    record = record_state(
        "test", "command", "python -m pytest", "g", "e3", "fp", previous, config
    )
    assert record.seen_events == ("e2", "e3")
    assert record.pending_calls == ("c2",)


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
    assert apply(config, store, payload(command("python -m pytest tests/test_a.py"))).source == SOURCE_PHASE

    # A payload that carries only the result of a call it never showed is not
    # evidence of anything by itself: the memory is what keeps the phase.
    partial = payload(
        {"type": "function_call_output", "call_id": "c1",
         "output": "Process exited with code 0"},
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
    assert apply(config, store, payload(user="Zmień opis opcji.", turn="turn-2")).source != SOURCE_MEMORY


def test_fresh_evidence_beats_the_memory(config):
    store = InMemoryRoutingStateStore(config.memory)
    apply(config, store, payload(command("python -m pytest tests/test_a.py")))
    decision = apply(config, store, payload(command("git log --oneline")))
    assert decision.source == SOURCE_PHASE
    assert decision.mode == "git_review"


def test_a_fallback_decision_is_never_remembered(config):
    store = InMemoryRoutingStateStore(config.memory)
    assert apply(config, store, payload(user="Opisz krótko ten moduł.")).source == SOURCE_FALLBACK
    key = session_key(config.memory.key_prefix, "s1", "t1", "/root")
    assert store.read(key)[1].state == "miss"


def test_threads_and_agents_do_not_share_a_phase(config):
    store = InMemoryRoutingStateStore(config.memory)
    apply(config, store, payload(command("python -m pytest tests/test_a.py")))
    other = apply(config, store, payload(command("python -m pytest tests/test_a.py"),
                                         thread="other-thread"))
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


@pytest.mark.parametrize("name, value", [
    ("REDIS_PORT", "not-a-port"),
    ("REDIS_DB", "x"),
    ("REDIS_PROTOCOL", "wat"),
    ("REDIS_SSL", "maybe"),
    ("REDIS_SSL_CERT_REQS", "sometimes"),
    ("REDIS_SOCKET_TIMEOUT", "soon"),
])
def test_malformed_connection_variables_are_rejected(monkeypatch, name, value):
    monkeypatch.setenv(f"{PREFIX}{name}", value)
    with pytest.raises(ValueError):
        connection_from_env(PREFIX)


@pytest.mark.parametrize("settings, problem", [
    ({"port": 0}, "port"), ({"port": 70000}, "port"),
    ({"db": 16}, "db"), ({"protocol": 4}, "protocol"),
    ({"ssl_cert_reqs": "?"}, "cert_reqs"),
    ({"socket_timeout": 0}, "socket_timeout"),
    ({"socket_connect_timeout": -1}, "socket_connect_timeout"),
])
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
        {"enabled": True, "backend": "memory", "ttl_seconds": 60,
         "max_sessions": 5, "max_events": 3, "max_calls": 2,
         "key_prefix": "custom", "max_retries": 0},
        PREFIX,
    )
    assert (resolved.enabled, resolved.backend, resolved.ttl_seconds) == (True, "memory", 60)
    assert (resolved.max_sessions, resolved.max_events) == (5, 3)
    assert (resolved.max_calls, resolved.key_prefix, resolved.max_retries) == (2, "custom", 0)


def test_environment_overrides_the_json_for_non_secret_settings(monkeypatch):
    monkeypatch.setenv(f"{PREFIX}MEMORY_TTL_SECONDS", "45")
    monkeypatch.setenv(f"{PREFIX}MEMORY_BACKEND", "memory")
    monkeypatch.setenv(f"{PREFIX}MEMORY_MAX_SESSIONS", "7")
    monkeypatch.setenv(f"{PREFIX}MEMORY_ENABLED", "true")
    resolved = memory_config_from_raw(
        {"enabled": False, "backend": "redis", "ttl_seconds": 900,
         "max_sessions": 5000},
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


@pytest.mark.parametrize("section", [
    {"backend": "memcached"},
    {"ttl_seconds": 0},
    {"max_sessions": -1},
    {"max_events": "many"},
    {"max_retries": -2},
    {"key_prefix": "has space"},
    {"key_prefix": ""},
    {"enabled": "perhaps"},
])
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
    config = CodexRoutingConfig._from_raw(raw)
    with pytest.raises(ValueError, match="REDIS_HOST"):
        config.validate_args()


def test_shipped_configuration_stays_stateless():
    path = _ROOT / "llm_router_plugins/resources/routing/agentic_routing_codex.json"
    shipped = CodexRoutingConfig._from_raw(json.loads(path.read_text(encoding="utf-8")))
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
    monkeypatch.setitem(sys.modules, "redis", SimpleNamespace(Redis=Mock(return_value=client)))
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

    plugin = CodexRoutingPlugin(config=config)

    client.ping.assert_called_once_with()
    assert isinstance(plugin._memory, RedisRoutingStateStore)
    assert factory.call_args.kwargs["host"] == "cache"
    assert factory.call_args.kwargs["protocol"] == 2
    assert factory.call_args.kwargs["socket_connect_timeout"] == 1.0
    assert factory.call_args.kwargs["socket_timeout"] == 1.0


@pytest.mark.parametrize("failure", [
    ConnectionError("Connection refused"), TimeoutError("Connection timed out"),
])
@pytest.mark.parametrize("with_logger", [True, False])
def test_plugin_warns_at_startup_when_redis_is_unreachable(
    config, monkeypatch, caplog, failure, with_logger,
):
    client = Mock()
    client.ping.side_effect = failure
    monkeypatch.setitem(sys.modules, "redis", SimpleNamespace(Redis=Mock(return_value=client)))
    monkeypatch.setenv(f"{PREFIX}MEMORY_ENABLED", "1")
    monkeypatch.setenv(f"{PREFIX}REDIS_HOST", "wrong-address")
    logger = logging.getLogger("codex-startup-test") if with_logger else None

    with caplog.at_level(logging.WARNING):
        plugin = CodexRoutingPlugin(logger=logger, config=config)

    client.ping.assert_called_once_with()
    assert "Codex routing memory disabled" in caplog.text
    assert str(failure) in caplog.text
    assert plugin._memory is None
    assert plugin.apply(payload())["agent_mode"] in config.mode_by_name


@pytest.mark.parametrize("enabled, backend", [(False, "redis"), (True, "memory")])
def test_plugin_does_not_connect_when_redis_memory_is_not_enabled(
    config, monkeypatch, enabled, backend,
):
    factory = Mock()
    monkeypatch.setitem(sys.modules, "redis", SimpleNamespace(Redis=factory))
    monkeypatch.setenv(f"{PREFIX}MEMORY_ENABLED", "1" if enabled else "0")
    monkeypatch.setenv(f"{PREFIX}MEMORY_BACKEND", backend)
    monkeypatch.setenv(f"{PREFIX}REDIS_HOST", "wrong-address")

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
        assert first.write(key, SessionRoutingState(mode="test", generation="g"), 0).state == "written"
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
        assert first.write(key, SessionRoutingState(mode="test"), 0).state == "written"
        assert second.write(key, SessionRoutingState(mode="debug"), 0).state == "conflict"
        assert second.read(key)[0].mode == "test"
        assert second.write(key, SessionRoutingState(mode="debug"), 1).state == "written"
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
