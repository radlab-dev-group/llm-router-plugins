"""
Tests for SemanticBiEncoderRoutingPlugin.

Run with:
    pytest tests/test_semantic_biencoder_routing.py -v
"""

import json
import os
import pathlib
import sys

sys.path.insert(
    0, str(pathlib.Path(__file__).resolve().parent.parent)
)

import pytest

from llm_router_plugins.utils.routing.semantic_biencoder.config import (
    SemanticBiEncoderConfig,
)
from llm_router_plugins.utils.routing.semantic_biencoder.embedder import (
    EmbeddingRouter,
)


# ---------- helpers ----------
_CONFIG_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "llm_router_plugins"
    / "resources"
    / "routing"
    / "semantic_biencoder.json"
)


def _load_config() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _clean_env():
    kept: dict = {}
    for key in list(os.environ.keys()):
        if key.startswith("LLM_ROUTER_ROUTING_SEMANTIC_EURO"):
            kept[key] = os.environ.pop(key)
    return kept


def _restore_env(kept: dict) -> None:
    for k, v in kept.items():
        os.environ[k] = v


# -------------------------- fixtures

@pytest.fixture(autouse=True)
def clean_semantic_biencoder_env():
    kept = _clean_env()
    yield
    _restore_env(kept)


# --------------- config tests

def test_config_loads_from_file():
    cfg = SemanticBiEncoderConfig.from_file(_CONFIG_PATH)
    assert cfg.embedding_model == "radlab/semantic-euro-bert-encoder-v1"
    assert cfg.chunk_size == 256
    assert cfg.chunk_overlap == 64
    assert cfg.similarity_threshold == 0.0
    assert cfg.top_k == 1
    assert len(cfg.routing_targets) > 0
    assert cfg.target_names == [t.name for t in cfg.routing_targets]


def test_config_target_models():
    cfg = SemanticBiEncoderConfig.from_file(_CONFIG_PATH)
    models = cfg.target_models
    assert "code-generation" in models
    assert models["code-generation"] == "qwen3.6:35b"
    assert "creative-writing" in models
    assert models["creative-writing"] == "gpt-oss:120b"


def test_config_default_path():
    """Config should load from resources when path is None."""
    cfg = SemanticBiEncoderConfig.from_file()
    assert len(cfg.routing_targets) >= 5


# --------------- embedder tests

class TestEmbedder:
    """Tests for the EmbeddingRouter (without actually loading the model)."""

    def test_split_into_chunks_single_chunk(self):
        text = "Short text."
        chunks = EmbeddingRouter._split_into_chunks(text, 256, 64)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_into_chunks_multiple_chunks(self):
        text = " ".join(["word"] * 500)
        chunks = EmbeddingRouter._split_into_chunks(text, 100, 20)
        assert len(chunks) > 1
        # First chunk
        assert len(chunks[0].split()) == 100
        # Last chunk
        assert len(chunks[-1].split()) <= 100

    def test_split_into_chunks_exact_fit(self):
        text = " ".join(["word"] * 100)
        chunks = EmbeddingRouter._split_into_chunks(text, 100, 20)
        assert len(chunks) == 1

    def test_split_into_chunks_overlap(self):
        text = " ".join(["word"] * 200)
        chunks = EmbeddingRouter._split_into_chunks(text, 100, 50)
        # Check overlap: last 50 tokens of chunk 0 should overlap with first 50 of chunk 1
        chunk0_tokens = chunks[0].split()
        chunk1_tokens = chunks[1].split()
        overlap_actual = len(
            set(chunk0_tokens[-50:]) & set(chunk1_tokens[:50])
        )
        assert overlap_actual > 0

    @staticmethod
    def test_cosine_similarity_identical():
        a = [1.0, 0.0, 0.0]
        b = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        import numpy as np
        sims = EmbeddingRouter._cosine_similarity(np.array(a), np.array(b))
        assert sims[0] == pytest.approx(1.0, abs=1e-6)
        assert sims[1] == pytest.approx(0.0, abs=1e-6)

    @staticmethod
    def test_cosine_similarity_zero_vector():
        a = [0.0, 0.0, 0.0]
        b = [[1.0, 0.0, 0.0]]
        import numpy as np
        sims = EmbeddingRouter._cosine_similarity(np.array(a), np.array(b))
        assert all(s == 0 for s in sims)

    @staticmethod
    def test_cosine_similarity_opposite():
        a = [1.0, 0.0]
        b = [[-1.0, 0.0]]
        import numpy as np
        sims = EmbeddingRouter._cosine_similarity(np.array(a), np.array(b))
        assert sims[0] == pytest.approx(-1.0, abs=1e-6)


# --------------- routing integration (mocked model)

class TestRoutingIntegration:
    """Integration tests that patch SentenceTransformer to avoid downloading."""

    @pytest.fixture(autouse=True)
    def patch_sentence_transformer(self, monkeypatch):
        """Mock SentenceTransformer to return fake embeddings."""
        import numpy as np

        class MockModel:
            def __init__(self, *args, **kwargs):
                self.embed_dim = 768

            def encode(self, texts, show_progress_bar=False, convert_to_numpy=True):
                if isinstance(texts, str):
                    texts = [texts]
                result = np.random.RandomState(42).rand(len(texts), self.embed_dim)
                # Normalize so cosine similarity is deterministic
                norms = np.linalg.norm(result, axis=1, keepdims=True)
                norms[norms == 0] = 1e-10
                result = result / norms
                if len(texts) == 1:
                    return result[0]
                return result

        monkeypatch.setattr(
            "llm_router_plugins.utils.routing.semantic_biencoder.embedder.SentenceTransformer",
            MockModel,
        )

    def test_route_code_query(self):
        from llm_router_plugins.utils.routing.semantic_biencoder.semantic_biencoder_routing import (
            SemanticBiEncoderRoutingPlugin,
        )

        plugin = SemanticBiEncoderRoutingPlugin()
        payload = {
            "model": "auto",
            "messages": [{"role": "user", "content": "Write a Python function to sort a list"}],
        }
        result = plugin.apply(payload)
        assert result["model"] != "auto"
        assert "routing" in result
        assert "target_name" in result["routing"]
        assert "similarity" in result["routing"]

    def test_route_creative_query(self):
        from llm_router_plugins.utils.routing.semantic_biencoder.semantic_biencoder_routing import (
            SemanticBiEncoderRoutingPlugin,
        )

        plugin = SemanticBiEncoderRoutingPlugin()
        payload = {
            "model": "auto",
            "messages": [{"role": "user", "content": "Write a creative story about a dragon"}],
        }
        result = plugin.apply(payload)
        assert result["model"] != "auto"
        assert "routing" in result

    def test_route_math_query(self):
        from llm_router_plugins.utils.routing.semantic_biencoder.semantic_biencoder_routing import (
            SemanticBiEncoderRoutingPlugin,
        )

        plugin = SemanticBiEncoderRoutingPlugin()
        payload = {
            "model": "auto",
            "messages": [{"role": "user", "content": "Calculate the probability of getting two sixes"}],
        }
        result = plugin.apply(payload)
        assert result["model"] != "auto"
        assert "routing" in result

    def test_non_auto_model_passthrough(self):
        from llm_router_plugins.utils.routing.semantic_biencoder.semantic_biencoder_routing import (
            SemanticBiEncoderRoutingPlugin,
        )

        plugin = SemanticBiEncoderRoutingPlugin()
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hello"}],
        }
        result = plugin.apply(payload)
        assert result["model"] == "gpt-4"
        assert "routing" not in result

    def test_no_text_content_returns_unchanged(self):
        from llm_router_plugins.utils.routing.semantic_biencoder.semantic_biencoder_routing import (
            SemanticBiEncoderRoutingPlugin,
        )

        plugin = SemanticBiEncoderRoutingPlugin()
        payload = {
            "model": "auto",
            "messages": [],
        }
        result = plugin.apply(payload)
        # Should not crash, payload returned as-is or with minimal changes
        assert result is not None

    def test_query_field_fallback(self):
        from llm_router_plugins.utils.routing.semantic_biencoder.semantic_biencoder_routing import (
            SemanticBiEncoderRoutingPlugin,
        )

        plugin = SemanticBiEncoderRoutingPlugin()
        payload = {
            "model": "auto",
            "query": "debug this Python function",
        }
        result = plugin.apply(payload)
        assert result["model"] != "auto"

    def test_prompt_field_fallback(self):
        from llm_router_plugins.utils.routing.semantic_biencoder.semantic_biencoder_routing import (
            SemanticBiEncoderRoutingPlugin,
        )

        plugin = SemanticBiEncoderRoutingPlugin()
        payload = {
            "model": "auto",
            "prompt": "write a poem about mountains",
        }
        result = plugin.apply(payload)
        assert result["model"] != "auto"


# --------------- env override tests

class TestEnvOverrides:
    def test_env_override_embedding_model(self, monkeypatch):
        from llm_router_plugins.utils.routing.semantic_biencoder.semantic_biencoder_routing import (
            SemanticBiEncoderRoutingPlugin,
        )

        monkeypatch.setenv(
            "LLM_ROUTER_ROUTING_SEMANTIC_EURO_MODEL",
            "custom/embedding-model",
        )

        class MockModel:
            embed_dim = 768
            def __init__(self, *args, **kwargs):
                pass
            def encode(self, texts, **kwargs):
                return [[0.5] * 768] if isinstance(texts, list) else [0.5] * 768

        monkeypatch.setattr(
            "llm_router_plugins.utils.routing.semantic_biencoder.embedder.SentenceTransformer",
            MockModel,
        )

        plugin = SemanticBiEncoderRoutingPlugin()
        assert plugin._config.embedding_model == "custom/embedding-model"

    def test_env_override_targets(self, monkeypatch):
        from llm_router_plugins.utils.routing.semantic_biencoder.semantic_biencoder_routing import (
            SemanticBiEncoderRoutingPlugin,
        )

        monkeypatch.setenv(
            "LLM_ROUTER_ROUTING_SEMANTIC_EURO_TARGETS",
            "code-generation|creative-writing",
        )

        class MockModel:
            embed_dim = 768
            def __init__(self, *args, **kwargs):
                pass
            def encode(self, texts, **kwargs):
                return [[0.5] * 768] if isinstance(texts, list) else [0.5] * 768

        monkeypatch.setattr(
            "llm_router_plugins.utils.routing.semantic_biencoder.embedder.SentenceTransformer",
            MockModel,
        )

        plugin = SemanticBiEncoderRoutingPlugin()
        target_names = [t.name for t in plugin._config.routing_targets]
        assert set(target_names) == {"code-generation", "creative-writing"}

    def test_env_override_chunk_size(self, monkeypatch):
        from llm_router_plugins.utils.routing.semantic_biencoder.semantic_biencoder_routing import (
            SemanticBiEncoderRoutingPlugin,
        )

        monkeypatch.setenv(
            "LLM_ROUTER_ROUTING_SEMANTIC_EURO_CHUNK_SIZE",
            "128",
        )

        class MockModel:
            embed_dim = 768
            def __init__(self, *args, **kwargs):
                pass
            def encode(self, texts, **kwargs):
                return [[0.5] * 768] if isinstance(texts, list) else [0.5] * 768

        monkeypatch.setattr(
            "llm_router_plugins.utils.routing.semantic_biencoder.embedder.SentenceTransformer",
            MockModel,
        )

        plugin = SemanticBiEncoderRoutingPlugin()
        assert plugin._config.chunk_size == 128

    def test_invalid_target_name_ignored(self, monkeypatch):
        from llm_router_plugins.utils.routing.semantic_biencoder.semantic_biencoder_routing import (
            SemanticBiEncoderRoutingPlugin,
        )

        monkeypatch.setenv(
            "LLM_ROUTER_ROUTING_SEMANTIC_EURO_TARGETS",
            "code-generation|nonexistent-target",
        )

        class MockModel:
            embed_dim = 768
            def __init__(self, *args, **kwargs):
                pass
            def encode(self, texts, **kwargs):
                return [[0.5] * 768] if isinstance(texts, list) else [0.5] * 768

        monkeypatch.setattr(
            "llm_router_plugins.utils.routing.semantic_biencoder.embedder.SentenceTransformer",
            MockModel,
        )

        plugin = SemanticBiEncoderRoutingPlugin()
        target_names = [t.name for t in plugin._config.routing_targets]
        assert "nonexistent-target" not in target_names
        assert "code-generation" in target_names
