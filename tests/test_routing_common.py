"""
Tests for the shared routing layer (``llm_router_plugins.utils.routing``):
``common.py`` helpers, ``RoutingConfigBase``, the ``RoutingTarget`` contract
and the backward-compatibility shims.

Run with:
    pytest tests/test_routing_common.py -v
"""

import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

from llm_router_plugins.utils.text_extractor import extract_user_text
from llm_router_plugins.utils.routing.common import (
    RoutingConfigBase,
    annotate_routing,
    build_embedding_router,
    check_router_has_vectors,
    env_bool,
    env_float,
    env_int,
    resolve_persist_dir,
    should_route,
)
from llm_router_plugins.utils.routing.target import RoutingTarget

_PREFIX = "LLM_ROUTER_ROUTING_TEST_"


@pytest.fixture(autouse=True)
def clean_routing_test_env(monkeypatch):
    """Clear all ``LLM_ROUTER_ROUTING_TEST_*`` env vars for each test."""
    for key in list(os.environ.keys()):
        if key.startswith(_PREFIX):
            monkeypatch.delenv(key)


# --------------- extract_user_text


class TestExtractUserText:
    def test_last_message_wins(self):
        payload = {
            "messages": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "user text"},
            ],
            "query": "should not be used",
        }
        assert extract_user_text(payload) == "user text"

    def test_user_last_statement(self):
        assert extract_user_text({"user_last_statement": "hello"}) == "hello"

    def test_query_fallback(self):
        assert extract_user_text({"query": "q"}) == "q"

    def test_prompt_fallback(self):
        assert extract_user_text({"prompt": "p"}) == "p"

    def test_input_fallback(self):
        assert extract_user_text({"input": "i"}) == "i"

    def test_priority_order(self):
        payload = {
            "user_last_statement": "a",
            "query": "b",
            "prompt": "c",
            "input": "d",
        }
        assert extract_user_text(payload) == "a"

    def test_empty_payload(self):
        assert extract_user_text({}) == ""

    def test_empty_messages_list(self):
        assert extract_user_text({"messages": [], "query": "q"}) == "q"


# --------------- should_route


class TestShouldRoute:
    def test_matching_trigger(self):
        assert should_route({"model": "auto"}, ("auto",)) is True

    def test_trigger_is_trimmed(self):
        assert should_route({"model": "auto  "}, ("auto",)) is True

    def test_non_matching_trigger(self):
        assert should_route({"model": "gpt-4"}, ("auto",)) is False

    def test_non_string_model(self):
        assert should_route({"model": 123}, ("auto",)) is False
        assert should_route({}, ("auto",)) is False


# --------------- annotate_routing


class TestAnnotateRouting:
    def test_base_fields(self):
        payload = {}
        annotate_routing(payload, "plugin_x", "model_a", 0.5)
        assert payload["model"] == "model_a"
        assert payload["routing"]["plugin"] == "plugin_x"
        assert payload["routing"]["similarity"] == pytest.approx(0.5)

    def test_extra_fields(self):
        payload = {}
        annotate_routing(payload, "plugin_x", "model_a", 0.7, target_name="t1")
        assert payload["routing"]["target_name"] == "t1"

    def test_similarity_coerced_to_float(self):
        payload = {}
        annotate_routing(payload, "plugin_x", "model_a", "0.4")
        assert payload["routing"]["similarity"] == pytest.approx(0.4)


# --------------- env helpers


class TestEnvHelpers:
    def test_env_int_valid(self, monkeypatch):
        monkeypatch.setenv(f"{_PREFIX}TOP_K", "5")
        assert env_int(_PREFIX, "TOP_K") == 5

    def test_env_int_invalid_ignored(self, monkeypatch):
        monkeypatch.setenv(f"{_PREFIX}TOP_K", "not-a-number")
        assert env_int(_PREFIX, "TOP_K") is None

    def test_env_int_unset(self):
        assert env_int(_PREFIX, "TOP_K") is None

    def test_env_float_valid(self, monkeypatch):
        monkeypatch.setenv(f"{_PREFIX}THRESHOLD", "0.55")
        assert env_float(_PREFIX, "THRESHOLD") == pytest.approx(0.55)

    def test_env_float_invalid_ignored(self, monkeypatch):
        monkeypatch.setenv(f"{_PREFIX}THRESHOLD", "abc")
        assert env_float(_PREFIX, "THRESHOLD") is None

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("1", True),
            ("true", True),
            ("YES", True),
            ("on", True),
            ("0", False),
            ("false", False),
            ("no", False),
            ("OFF", False),
        ],
    )
    def test_env_bool_values(self, monkeypatch, value, expected):
        monkeypatch.setenv(f"{_PREFIX}FLAG", value)
        assert env_bool(_PREFIX, "FLAG") is expected

    def test_env_bool_unrecognized(self, monkeypatch):
        monkeypatch.setenv(f"{_PREFIX}FLAG", "maybe")
        assert env_bool(_PREFIX, "FLAG") is None

    def test_env_bool_unset(self):
        assert env_bool(_PREFIX, "FLAG") is None

    def test_resolve_persist_dir_prefers_env(self, monkeypatch):
        monkeypatch.setenv(f"{_PREFIX}PERSIST_DIR", "/from/env")
        assert resolve_persist_dir(_PREFIX, "/from/config") == "/from/env"

    def test_resolve_persist_dir_falls_back_to_configured(self):
        assert resolve_persist_dir(_PREFIX, "/from/config") == "/from/config"

    def test_resolve_persist_dir_none(self):
        assert resolve_persist_dir(_PREFIX, None) is None
        assert resolve_persist_dir(_PREFIX, "") is None


# --------------- RoutingConfigBase


class _TestConfig(RoutingConfigBase):
    """Minimal concrete config used to exercise the base class protocol."""

    _ENV_PREFIX = _PREFIX
    _DEFAULT_CONFIG_PATH = None

    @staticmethod
    def _from_raw(raw: dict) -> "_TestConfig":
        if "value" not in raw:
            raise KeyError("value")
        return _TestConfig(value=raw["value"])

    def __init__(self, value: str) -> None:
        self.value = value


def _write_config(tmp_path: pathlib.Path, name: str, raw: dict) -> pathlib.Path:
    path = tmp_path / name
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


class TestRoutingConfigBase:
    def test_from_file_explicit_path(self, tmp_path):
        path = _write_config(tmp_path, "cfg.json", {"value": "from-file"})
        assert _TestConfig.from_file(path).value == "from-file"

    def test_from_file_env_raw_json_wins(self, monkeypatch, tmp_path):
        path = _write_config(tmp_path, "cfg.json", {"value": "from-file"})
        monkeypatch.setenv(f"{_PREFIX}CONFIG", json.dumps({"value": "from-env"}))
        assert _TestConfig.from_file(path).value == "from-env"

    def test_from_file_env_file_path(self, monkeypatch, tmp_path):
        path = _write_config(tmp_path, "cfg.json", {"value": "from-env-file"})
        monkeypatch.setenv(f"{_PREFIX}CONFIG", str(path))
        assert _TestConfig.from_file().value == "from-env-file"

    def test_from_file_missing_required_key(self, monkeypatch):
        monkeypatch.setenv(f"{_PREFIX}CONFIG", json.dumps({}))
        with pytest.raises(KeyError, match="value"):
            _TestConfig.from_file()

    def test_from_json_empty_string(self):
        with pytest.raises(ValueError, match="empty config string"):
            _TestConfig.from_json("")

    def test_from_json_invalid_json(self):
        with pytest.raises(json.JSONDecodeError):
            _TestConfig.from_json("{not json")

    def test_validate_semantic_params_ok(self):
        RoutingConfigBase.validate_semantic_params(256, 64, 3)

    @pytest.mark.parametrize(
        ("chunk_size", "chunk_overlap", "top_k"),
        [
            (0, 0, 1),
            (256, -1, 1),
            (256, 0, 0),
        ],
    )
    def test_validate_semantic_params_rejects(
        self, chunk_size, chunk_overlap, top_k
    ):
        with pytest.raises(ValueError):
            RoutingConfigBase.validate_semantic_params(
                chunk_size, chunk_overlap, top_k
            )


# --------------- router contract


class TestRouterContract:
    def test_check_router_has_vectors_raises(self):
        class _EmptyRouter:
            has_vectors = False

        with pytest.raises(ValueError, match="has no vectors"):
            check_router_has_vectors(_EmptyRouter(), label="TestPlugin")

    def test_check_router_has_vectors_ok(self):
        class _FullRouter:
            has_vectors = True

        check_router_has_vectors(_FullRouter())

    def test_build_router_with_mock_model(self, mock_sentence_transformer):
        targets = (
            RoutingTarget(
                name="alpha",
                model_name="model_a",
                description="First target",
                examples=["example one"],
            ),
            RoutingTarget(
                name="beta",
                model_name="model_b",
                description="Second target",
                examples=["example two"],
            ),
        )

        router = build_embedding_router(
            embedding_model="mock/model",
            chunk_size=256,
            chunk_overlap=64,
            top_k=2,
            routing_targets=targets,
        )

        assert router.has_vectors
        result = router.route("hello world")
        assert result["target_name"] in {"alpha", "beta"}
        assert result["model_name"] in {"model_a", "model_b"}
        # FAISS float32 inner products can exceed 1.0 by a tiny epsilon.
        assert result["similarity"] > 0.0
        assert abs(result["similarity"] - 1.0) <= 1e-3


# --------------- target contract & shims


class TestTargetContractAndShims:
    def test_agent_mode_is_routing_target(self):
        from llm_router_plugins.utils.routing.agentic_routing.general.config import (
            AgentMode,
            AgenticRoutingConfig,
        )

        assert issubclass(AgentMode, RoutingTarget)
        assert issubclass(AgenticRoutingConfig, RoutingConfigBase)

        mode = AgentMode(
            name="plan",
            model_name="model_a",
            description="Planning mode",
            examples=("e1",),
            keywords=["plan"],
            phrases=[],
            patterns=[],
            weights={},
        )
        # Base fields are accessible through the subclass contract
        assert mode.name == "plan"
        assert mode.examples == ("e1",)

    def test_semantic_biencoder_config_is_routing_config_base(self):
        from llm_router_plugins.utils.routing.semantic_biencoder.config import (
            SemanticBiEncoderConfig,
        )

        assert issubclass(SemanticBiEncoderConfig, RoutingConfigBase)
        # The bundled default config resolves and loads.
        cfg = SemanticBiEncoderConfig.from_file()
        assert len(cfg.routing_targets) >= 5

    def test_embedder_shim_keeps_old_import_path(self):
        from llm_router_plugins.utils.routing.embedder import (
            EmbeddingRouter as SharedEmbeddingRouter,
        )
        from llm_router_plugins.utils.routing.semantic_biencoder.embedder import (
            EmbeddingRouter as ShimEmbeddingRouter,
            EmbeddingRouterConfig as ShimEmbeddingRouterConfig,
        )

        assert ShimEmbeddingRouter is SharedEmbeddingRouter
        assert ShimEmbeddingRouterConfig is not None

    def test_routing_target_reexported_from_plugin_config(self):
        from llm_router_plugins.utils.routing.semantic_biencoder.config import (
            RoutingTarget as ShimRoutingTarget,
        )

        assert ShimRoutingTarget is RoutingTarget
