"""Memory configuration is optional, independent and credential-safe."""

import json
import logging
import os
import pathlib
from dataclasses import replace

import pytest

from llm_router_plugins.utils.routing.agentic_routing.codex import plugin as module
from llm_router_plugins.utils.routing.agentic_routing.codex.config import (
    CodexRoutingConfig,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.classifier import (
    RoutingDecision,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.phase import (
    PhaseEvidence,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.state import (
    InMemoryRoutingStateStore,
    MemoryStatus,
)
from llm_router_plugins.utils.routing.constants import AGENTIC_CODEX_ROUTING_PREFIX


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for key in list(os.environ):
        if key.startswith(AGENTIC_CODEX_ROUTING_PREFIX):
            monkeypatch.delenv(key)


def raw_config():
    path = pathlib.Path(__file__).resolve().parents[1] / (
        "llm_router_plugins/resources/routing/agentic_routing_codex.json"
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["settings"]["semantic"]["enabled"] = False
    raw["settings"].pop("memory", None)
    return raw


def route(plugin):
    return plugin.apply({"model": "auto_codex", "input": "Continue."})


@pytest.mark.parametrize(
    "name,value",
    [
        ("MEMORY_ENABLED", "secret-invalid-boolean"),
        ("MEMORY_BACKEND", "secret-invalid-backend"),
        ("MEMORY_TTL_SECONDS", "secret-invalid-integer"),
        ("MEMORY_TTL_SECONDS", "0"),
        ("MEMORY_MAX_SESSIONS", "-1"),
        ("MEMORY_MAX_EVENTS", "0"),
        ("MEMORY_MAX_CALLS", "0"),
        ("MEMORY_MAX_RETRIES", "-1"),
        ("MEMORY_KEY_PREFIX", "unsafe prefix"),
        ("REDIS_PORT", "secret-invalid-port"),
        ("REDIS_PORT", "65536"),
        ("REDIS_DB", "-1"),
        ("REDIS_PROTOCOL", "4"),
        ("REDIS_SSL", "secret-invalid-tls"),
        ("REDIS_SSL_CERT_REQS", "secret-invalid-cert"),
        ("REDIS_SOCKET_TIMEOUT", "nan"),
        ("REDIS_SOCKET_CONNECT_TIMEOUT", "0"),
    ],
)
def test_invalid_memory_env_is_stateless_and_redacted(
    monkeypatch,
    caplog,
    name,
    value,
):
    monkeypatch.setenv(AGENTIC_CODEX_ROUTING_PREFIX + "MEMORY_ENABLED", "true")
    monkeypatch.setenv(AGENTIC_CODEX_ROUTING_PREFIX + "REDIS_HOST", "private-host")
    monkeypatch.setenv(AGENTIC_CODEX_ROUTING_PREFIX + name, value)
    with caplog.at_level(logging.WARNING):
        config = CodexRoutingConfig._from_raw(raw_config())
        plugin = module.CodexRoutingPlugin(logging.getLogger(__name__), config)
        result = route(plugin)
    assert plugin._memory is None
    assert not config.memory.enabled
    assert result["agent_mode"] == "implement"
    assert result["routing"]["memory"] == "unconfigured"
    assert "invalid configuration" in caplog.text
    assert "secret-invalid" not in caplog.text
    assert "private-host" not in caplog.text


def test_missing_host_does_not_inherit_host_application_env(monkeypatch):
    monkeypatch.setenv(AGENTIC_CODEX_ROUTING_PREFIX + "MEMORY_ENABLED", "true")
    for prefix in ("REDIS_", "LLM_ROUTER_REDIS_", "LLM_ROUTER_AUTH_REDIS_"):
        monkeypatch.setenv(prefix + "HOST", "host-application")
        monkeypatch.setenv(prefix + "PASSWORD", "host-secret")
    plugin = module.CodexRoutingPlugin(
        config=CodexRoutingConfig._from_raw(raw_config())
    )
    assert plugin._memory is None
    assert plugin._config.memory.connection.host == ""
    assert plugin._config.memory.connection.password is None
    assert route(plugin)["routing"]["memory"] == "unconfigured"


def test_env_applies_to_preloaded_config(monkeypatch):
    config = CodexRoutingConfig._from_raw(raw_config())
    monkeypatch.setenv(AGENTIC_CODEX_ROUTING_PREFIX + "MEMORY_TTL_SECONDS", "bad")
    plugin = module.CodexRoutingPlugin(config=config)
    assert route(plugin)["routing"]["memory"] == "unconfigured"


def test_bad_json_memory_is_fail_open_but_routing_config_is_not():
    raw = raw_config()
    raw["settings"]["memory"] = {"enabled": True, "ttl_seconds": "private-value"}
    config = CodexRoutingConfig._from_raw(raw)
    assert route(module.CodexRoutingPlugin(config=config))["routing"]["memory"] == (
        "unconfigured"
    )
    config.trigger_model = ""
    with pytest.raises(ValueError, match="trigger"):
        module.CodexRoutingPlugin(config=config)


def test_store_factory_error_is_fail_open_and_redacted(monkeypatch, caplog):
    def broken_factory(*args, **kwargs):
        raise ValueError("password=private-secret session=private-session")

    monkeypatch.setattr(module, "build_state_store", broken_factory)
    with caplog.at_level(logging.WARNING):
        plugin = module.CodexRoutingPlugin(
            logging.getLogger(__name__),
            CodexRoutingConfig._from_raw(raw_config()),
        )
        result = route(plugin)
    assert result["routing"]["memory"] == "unavailable"
    assert "private-secret" not in caplog.text
    assert "private-session" not in caplog.text


def memory_payload(*items, turn="turn-1"):
    return {
        "model": "auto_codex",
        "input": list(items),
        "client_metadata": {
            "session_id": "session",
            "thread_id": "thread",
            "turn_id": turn,
            "x-codex-turn-metadata": json.dumps({"agent_name": "/root"}),
        },
    }


def memory_plugin():
    config = CodexRoutingConfig._from_raw(raw_config())
    store = InMemoryRoutingStateStore(config.memory)
    return module.CodexRoutingPlugin(config=config, memory=store), store


def test_plugin_resolves_incremental_failure_without_user_message():
    plugin, store = memory_plugin()
    first = memory_payload(
        {
            "type": "function_call",
            "name": "exec_command",
            "call_id": "tests",
            "arguments": '{"cmd":"pytest"}',
        }
    )
    assert plugin.apply(first)["agent_mode"] == "test"
    result = plugin.apply(
        memory_payload(
            {
                "type": "function_call_output",
                "call_id": "tests",
                "id": "failure",
                "output": "Process exited with code 1\nOutput:\nfailure",
            }
        )
    )
    assert result["agent_mode"] == "debug"
    assert result["routing"]["evidence"] == "test_failure"
    assert plugin.apply(memory_payload())["agent_mode"] == "debug"


def test_plugin_memory_is_independent_of_keyword_heuristics():
    config = replace(
        CodexRoutingConfig._from_raw(raw_config()), heuristic_enabled=False
    )
    store = InMemoryRoutingStateStore(config.memory)
    plugin = module.CodexRoutingPlugin(config=config, memory=store)
    plugin.apply(
        memory_payload(
            {
                "type": "function_call",
                "name": "exec_command",
                "call_id": "tests",
                "arguments": '{"cmd":"pytest"}',
            }
        )
    )
    result = plugin.apply(memory_payload())
    assert result["agent_mode"] == "test"
    assert result["routing"]["source"] == "memory"


def test_disabling_phase_also_prevents_carrying_an_existing_phase():
    plugin, store = memory_plugin()
    plugin.apply(
        memory_payload(
            {
                "type": "function_call",
                "name": "exec_command",
                "call_id": "tests",
                "arguments": '{"cmd":"pytest"}',
            }
        )
    )
    plugin._config.phase = replace(plugin._config.phase, enabled=False)
    result = plugin.apply(memory_payload())
    assert result["agent_mode"] == "implement"
    assert result["routing"]["source"] == "fallback"


def test_plugin_passes_the_read_version_and_phase_rules_to_persistence(monkeypatch):
    plugin, store = memory_plugin()
    calls = []
    real_remember = module.remember_decision

    def capture(*args, **kwargs):
        calls.append(kwargs)
        return real_remember(*args, **kwargs)

    monkeypatch.setattr(module, "remember_decision", capture)
    plugin.apply(
        memory_payload(
            {
                "type": "function_call",
                "name": "exec_command",
                "call_id": "tests",
                "arguments": '{"cmd":"pytest"}',
            }
        )
    )
    plugin.apply(memory_payload())
    assert [call["expected_version"] for call in calls] == [0, 1]
    assert all(call["rules"] is plugin._config.phase for call in calls)


def test_factory_status_is_preserved(monkeypatch):
    monkeypatch.setattr(
        module,
        "build_state_store",
        lambda *args, **kwargs: (
            None,
            MemoryStatus("unavailable", "private-secret"),
        ),
    )
    plugin = module.CodexRoutingPlugin(
        config=CodexRoutingConfig._from_raw(raw_config())
    )
    assert route(plugin)["routing"]["memory"] == "unavailable"


def test_memory_write_error_keeps_successful_routing(monkeypatch, caplog):
    def broken_write(*args, **kwargs):
        raise ValueError("private-secret")

    monkeypatch.setattr(module, "remember_decision", broken_write)
    with caplog.at_level(logging.WARNING):
        plugin = module.CodexRoutingPlugin(
            logging.getLogger(__name__),
            CodexRoutingConfig._from_raw(raw_config()),
        )
        result = route(plugin)
    assert result["agent_mode"] == "implement"
    assert result["routing"]["memory"] == "unavailable"
    assert "private-secret" not in caplog.text


def test_preloaded_bad_connection_validation_is_fail_open():
    config = CodexRoutingConfig._from_raw(raw_config())
    config.memory = replace(
        config.memory,
        enabled=True,
        connection=replace(config.memory.connection, host="private-host", port=-1),
    )
    config.validate_args()
    assert not config.memory.enabled
    assert route(module.CodexRoutingPlugin(config=config))["routing"]["memory"] == (
        "unconfigured"
    )


def test_old_configuration_stays_stateless():
    config = CodexRoutingConfig._from_raw(raw_config())
    plugin = module.CodexRoutingPlugin(config=config)
    assert not config.memory.enabled
    assert config.memory_status is None
    assert plugin._memory is None
    assert "memory" not in route(plugin)["routing"]


def test_memory_env_precedence_and_redis_connection_contract(monkeypatch):
    raw = raw_config()
    raw["settings"]["memory"] = {
        "enabled": False,
        "backend": "memory",
        "ttl_seconds": 30,
        "max_sessions": 4,
        "max_events": 5,
        "max_calls": 6,
        "max_retries": 0,
        "key_prefix": "json-prefix",
    }
    overrides = {
        "MEMORY_ENABLED": "true",
        "MEMORY_BACKEND": "redis",
        "MEMORY_TTL_SECONDS": "120",
        "MEMORY_MAX_SESSIONS": "10",
        "MEMORY_MAX_EVENTS": "8",
        "MEMORY_MAX_CALLS": "3",
        "MEMORY_MAX_RETRIES": "2",
        "MEMORY_KEY_PREFIX": "installation-a:codex",
        "REDIS_HOST": "cache.internal",
        "REDIS_PORT": "6380",
        "REDIS_DB": "2",
        "REDIS_PROTOCOL": "2",
        "REDIS_PASSWORD": "",
        "REDIS_USERNAME": "acl",
        "REDIS_SSL": "yes",
        "REDIS_SSL_CA_CERTS": "/ca.pem",
        "REDIS_SSL_CERTFILE": "/cert.pem",
        "REDIS_SSL_KEYFILE": "/key.pem",
        "REDIS_SOCKET_CONNECT_TIMEOUT": "0.2",
        "REDIS_SOCKET_TIMEOUT": "0.3",
    }
    for name, value in overrides.items():
        monkeypatch.setenv(AGENTIC_CODEX_ROUTING_PREFIX + name, value)
    config = CodexRoutingConfig._from_raw(raw)
    config.validate_args()
    assert config.memory_status is None
    policy = config.memory
    assert policy.enabled and policy.backend == "redis"
    assert (policy.ttl_seconds, policy.max_sessions, policy.max_events) == (
        120,
        10,
        8,
    )
    assert (policy.max_calls, policy.max_retries) == (3, 2)
    assert policy.key_prefix == "installation-a:codex"
    kwargs = policy.connection.client_kwargs()
    assert kwargs == {
        "host": "cache.internal",
        "port": 6380,
        "db": 2,
        "protocol": 2,
        "password": None,
        "username": "acl",
        "decode_responses": True,
        "ssl": True,
        "ssl_cert_reqs": "required",
        "ssl_ca_certs": "/ca.pem",
        "ssl_certfile": "/cert.pem",
        "ssl_keyfile": "/key.pem",
        "socket_connect_timeout": 0.2,
        "socket_timeout": 0.3,
    }


@pytest.mark.parametrize(
    "raw_memory", [[], "private-secret", {"password": "secret"}]
)
def test_bad_memory_section_cannot_carry_connection_settings(raw_memory):
    raw = raw_config()
    raw["settings"]["memory"] = raw_memory
    config = CodexRoutingConfig._from_raw(raw)
    assert config.memory_status.state == "unconfigured"
    assert config.memory.connection.password is None


def test_memory_backend_warns_that_it_is_not_production(monkeypatch, caplog):
    monkeypatch.setenv(AGENTIC_CODEX_ROUTING_PREFIX + "MEMORY_ENABLED", "true")
    monkeypatch.setenv(AGENTIC_CODEX_ROUTING_PREFIX + "MEMORY_BACKEND", "memory")
    with caplog.at_level(logging.WARNING):
        plugin = module.CodexRoutingPlugin(
            logging.getLogger(__name__),
            CodexRoutingConfig._from_raw(raw_config()),
        )
    assert isinstance(plugin._memory, InMemoryRoutingStateStore)
    assert "isolated tests/replay only" in caplog.text


def test_evidence_diagnostics_do_not_include_event_or_call_identifiers():
    decision = RoutingDecision(
        "test",
        "phase",
        1.0,
        1.0,
        evidence=PhaseEvidence(
            mode="test",
            kind="command",
            event_id="private-event",
            call_id="private-call",
            reason="test command",
        ),
    )
    assert module.CodexRoutingPlugin._diagnostics(decision) == {
        "evidence": "command"
    }


@pytest.mark.parametrize("outcome", ["conflict", "unavailable"])
def test_write_outcome_is_visible_in_diagnostics(monkeypatch, outcome):
    monkeypatch.setattr(module, "remember_decision", lambda *args, **kwargs: outcome)
    plugin = module.CodexRoutingPlugin(
        config=CodexRoutingConfig._from_raw(raw_config())
    )
    assert route(plugin)["routing"]["memory"] == outcome


def test_raised_memory_read_retries_statelessly_without_raw_logging(caplog):
    class BrokenStore:
        def read(self, key):
            raise ValueError("private-secret private-session")

    payload = {
        "model": "auto_codex",
        "input": "Continue.",
        "client_metadata": {
            "session_id": "private-session",
            "thread_id": "private-thread",
            "turn_id": "private-turn",
        },
    }
    with caplog.at_level(logging.WARNING):
        plugin = module.CodexRoutingPlugin(
            logging.getLogger(__name__),
            CodexRoutingConfig._from_raw(raw_config()),
            memory=BrokenStore(),
        )
        result = plugin.apply(payload)
    assert result["agent_mode"] == "implement"
    assert result["routing"]["memory"] == "unavailable"
    assert "private-secret" not in caplog.text
    assert "private-session" not in caplog.text
