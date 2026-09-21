"""
Tests for SemanticBiEncoderRoutingPlugin.

Run with:
    pytest tests/test_semantic_biencoder_routing.py -v
"""

import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
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
        if key.startswith("LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER"):
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
    assert cfg.embedding_model == "google/embeddinggemma-300m"
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


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vector *a* (dim,) and matrix *b* (n, dim)."""
    a_norm = np.linalg.norm(a)
    if a_norm == 0:
        return np.zeros(b.shape[0])
    b_norm = np.linalg.norm(b, axis=1)
    b_norm[b_norm == 0] = 1e-10
    return np.dot(b, a) / (b_norm * a_norm)


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
        overlap_actual = len(set(chunk0_tokens[-50:]) & set(chunk1_tokens[:50]))
        assert overlap_actual > 0

    @staticmethod
    def test_cosine_similarity_identical():
        a = [1.0, 0.0, 0.0]
        b = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        import numpy as np

        sims = _cosine_similarity(np.array(a), np.array(b))
        assert sims[0] == pytest.approx(1.0, abs=1e-6)
        assert sims[1] == pytest.approx(0.0, abs=1e-6)

    @staticmethod
    def test_cosine_similarity_zero_vector():
        a = [0.0, 0.0, 0.0]
        b = [[1.0, 0.0, 0.0]]
        import numpy as np

        sims = _cosine_similarity(np.array(a), np.array(b))
        assert all(s == 0 for s in sims)

    @staticmethod
    def test_cosine_similarity_opposite():
        a = [1.0, 0.0]
        b = [[-1.0, 0.0]]
        import numpy as np

        sims = _cosine_similarity(np.array(a), np.array(b))
        assert sims[0] == pytest.approx(-1.0, abs=1e-6)


# --------------- query sliding-window encoding (mocked model)


class _FakeTokenizer:
    """A minimal stand-in for a HuggingFace tokenizer used by the mock model."""

    def __init__(self, vocab: int = 512):
        self.vocab = vocab

    def _to_ids(self, text: str) -> list:
        words = text.split()
        return [i % self.vocab for i, _ in enumerate(words)]

    def _to_words(self, ids: list) -> list:
        return [str(i) for i in ids]

    def encode(self, text, return_tensors=None):
        ids = self._to_ids(text)
        if return_tensors == "pt":
            import torch

            return torch.tensor([ids])
        return ids

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(self._to_words(ids))


def _install_tokenizer(mock_model, max_seq_length, vocab=512):
    """Attach a fake tokenizer and a ``max_seq_length`` to the shared mock model."""
    tokenizer = _FakeTokenizer(vocab)
    mock_model.tokenizer = tokenizer
    mock_model.max_seq_length = max_seq_length
    return tokenizer


def _counting_encode(mock_model, calls):
    """Replace ``encode`` on *mock_model* with a spy that records every batch."""
    original_encode = mock_model.encode

    def counting_encode(self, texts, **kwargs):
        calls.append(list(texts))
        return original_encode(self, texts, **kwargs)

    mock_model.encode = counting_encode
    return counting_encode


class TestEncodeQuery:
    """Tests for :meth:`EmbeddingRouter._encode_query` sliding-window behaviour."""

    @pytest.fixture(autouse=True)
    def patch_sentence_transformer(self, mock_sentence_transformer):
        """Activate the shared deterministic SentenceTransformer mock (conftest)."""

    @staticmethod
    def _make_router(cfg_overrides=None):
        cfg = SemanticBiEncoderConfig.from_file(_CONFIG_PATH)
        for key, value in (cfg_overrides or {}).items():
            setattr(cfg, key, value)
        router = EmbeddingRouter(cfg)
        router.initialize()
        return router

    def test_fast_path_without_tokenizer(self, mock_sentence_transformer):
        """Default mock (no tokenizer/max_seq_length) keeps the single-encode path."""
        mock_sentence_transformer.tokenizer = None
        mock_sentence_transformer.max_seq_length = None
        router = self._make_router()
        result = router.route("Write a Python function to sort a list")
        assert result["target_name"]
        assert -1e-6 <= result["similarity"] <= 1.0 + 1e-6

    def test_short_text_single_encode(self, mock_sentence_transformer):
        """Text within max_seq_length issues exactly one encode call."""
        _install_tokenizer(mock_sentence_transformer, max_seq_length=128)
        router = self._make_router()

        calls = []
        _counting_encode(mock_sentence_transformer, calls)

        text = " ".join(["alpha"] * 10)
        vector = router._encode_query(text)
        assert vector.shape == (1, mock_sentence_transformer.embed_dim)
        assert len(calls) == 1
        assert calls[0] == [text]
        assert float(np.linalg.norm(vector[0])) == pytest.approx(1.0, abs=1e-6)

    def test_long_text_windows_capped_at_head(self, mock_sentence_transformer):
        """5000-token text with max_seq=128/overlap=32 uses exactly 4 head windows."""
        _install_tokenizer(mock_sentence_transformer, max_seq_length=128)
        router = self._make_router({"chunk_overlap": 32})

        calls = []
        _counting_encode(mock_sentence_transformer, calls)

        text = " ".join(["beta"] * 5000)
        router._encode_query(text)
        assert len(calls) == 1
        batch = calls[0]
        assert len(batch) == 4
        assert batch[0].split() == " ".join(str(i) for i in range(128)).split()
        for window in batch:
            assert len(window.split()) == 128

    def test_zero_overlap_full_stride(self, mock_sentence_transformer):
        """chunk_overlap=0 strides by the full window and stays within the cap."""
        _install_tokenizer(mock_sentence_transformer, max_seq_length=64)
        router = self._make_router({"chunk_overlap": 0})

        calls = []
        _counting_encode(mock_sentence_transformer, calls)

        text = " ".join(["gamma"] * 1000)
        vector = router._encode_query(text)
        assert len(calls[0]) == 4
        assert vector.shape == (1, mock_sentence_transformer.embed_dim)
        assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-6)

    def test_large_overlap_clamped(self, mock_sentence_transformer):
        """chunk_overlap > max_seq // 4 is clamped instead of raising ValueError."""
        _install_tokenizer(mock_sentence_transformer, max_seq_length=128)
        cfg = SemanticBiEncoderConfig.from_file(_CONFIG_PATH)
        cfg.chunk_overlap = 1000
        router = EmbeddingRouter(cfg)
        router._model = mock_sentence_transformer()

        calls = []
        _counting_encode(mock_sentence_transformer, calls)

        text = " ".join(["delta"] * 1000)
        vector = router._encode_query(text)
        assert len(calls[0]) == 4
        for window in calls[0]:
            assert len(window.split()) == 128
        assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-6)

    def test_aggregation_unit_mean_of_windows(self, mock_sentence_transformer):
        """Aggregated vector equals the re-normalised mean of the window vectors."""
        _install_tokenizer(mock_sentence_transformer, max_seq_length=128)
        router = self._make_router({"chunk_overlap": 32})
        text = " ".join(["epsilon"] * 2000)

        vector = router._encode_query(text)

        windows = EmbeddingRouter._split_into_chunks(
            text, 128, 32, router._model.tokenizer
        )[:4]
        raw = np.asarray(
            router._model.encode(
                windows, show_progress_bar=False, convert_to_numpy=True
            )
        )
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        expected = (raw / norms).mean(axis=0)
        expected = expected / float(np.linalg.norm(expected))
        assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-6)
        assert np.allclose(vector[0], expected)

    def test_route_similarity_in_range(self, mock_sentence_transformer):
        """route() on an over-length query returns a similarity within [0, 1]."""
        _install_tokenizer(mock_sentence_transformer, max_seq_length=128)
        router = self._make_router({"chunk_overlap": 32})
        text = " ".join(["zeta"] * 3000)
        result = router.route(text)
        assert result["target_name"]
        assert -1e-6 <= result["similarity"] <= 1.0 + 1e-6
        assert all(
            -1e-6 <= entry["similarity"] <= 1.0 + 1e-6
            for entry in result["all_scores"]
        )


# --------------- routing integration (mocked model)


class TestRoutingIntegration:
    """Integration tests that patch SentenceTransformer to avoid downloading."""

    @pytest.fixture(autouse=True)
    def patch_sentence_transformer(self, mock_sentence_transformer):
        """Activate the shared deterministic SentenceTransformer mock (conftest)."""

    def test_route_code_query(self):
        from llm_router_plugins.utils.routing.semantic_biencoder.semantic_biencoder_routing import (
            SemanticBiEncoderRoutingPlugin,
        )

        plugin = SemanticBiEncoderRoutingPlugin()
        payload = {
            "model": "auto",
            "messages": [
                {"role": "user", "content": "Write a Python function to sort a list"}
            ],
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
            "messages": [
                {"role": "user", "content": "Write a creative story about a dragon"}
            ],
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
            "messages": [
                {
                    "role": "user",
                    "content": "Calculate the probability of getting two sixes",
                }
            ],
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

    def test_below_threshold_passthrough(self, monkeypatch):
        """A similarity below ``similarity_threshold`` leaves the payload alone."""
        from llm_router_plugins.utils.routing.semantic_biencoder.semantic_biencoder_routing import (
            SemanticBiEncoderRoutingPlugin,
        )

        raw = _load_config()
        raw["settings"][
            "similarity_threshold"
        ] = 2.0  # unreachable by any cosine score
        monkeypatch.setenv(
            "LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_CONFIG", json.dumps(raw)
        )

        plugin = SemanticBiEncoderRoutingPlugin()
        payload = {
            "model": "auto",
            "messages": [
                {"role": "user", "content": "Write a Python function to sort a list"}
            ],
        }
        result = plugin.apply(payload)
        assert result["model"] == "auto"
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
    def test_env_override_embedding_model(
        self, monkeypatch, mock_sentence_transformer
    ):
        from llm_router_plugins.utils.routing.semantic_biencoder.semantic_biencoder_routing import (
            SemanticBiEncoderRoutingPlugin,
        )

        monkeypatch.setenv(
            "LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_MODEL",
            "custom/embedding-model",
        )

        plugin = SemanticBiEncoderRoutingPlugin()
        assert plugin._config.embedding_model == "custom/embedding-model"

    def test_env_override_targets(self, monkeypatch, mock_sentence_transformer):
        from llm_router_plugins.utils.routing.semantic_biencoder.semantic_biencoder_routing import (
            SemanticBiEncoderRoutingPlugin,
        )

        monkeypatch.setenv(
            "LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_TARGETS",
            "code-generation|creative-writing",
        )

        plugin = SemanticBiEncoderRoutingPlugin()
        target_names = [t.name for t in plugin._config.routing_targets]
        assert set(target_names) == {"code-generation", "creative-writing"}

    def test_env_override_chunk_size(self, monkeypatch, mock_sentence_transformer):
        from llm_router_plugins.utils.routing.semantic_biencoder.semantic_biencoder_routing import (
            SemanticBiEncoderRoutingPlugin,
        )

        monkeypatch.setenv(
            "LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_CHUNK_SIZE",
            "128",
        )

        plugin = SemanticBiEncoderRoutingPlugin()
        assert plugin._config.chunk_size == 128

    def test_invalid_target_name_ignored(
        self, monkeypatch, mock_sentence_transformer
    ):
        from llm_router_plugins.utils.routing.semantic_biencoder.semantic_biencoder_routing import (
            SemanticBiEncoderRoutingPlugin,
        )

        monkeypatch.setenv(
            "LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_TARGETS",
            "code-generation|nonexistent-target",
        )

        plugin = SemanticBiEncoderRoutingPlugin()
        target_names = [t.name for t in plugin._config.routing_targets]
        assert "nonexistent-target" not in target_names
        assert "code-generation" in target_names


# ------ FAISS persistence tests


class TestFAISSPersistence:
    """Tests for FAISS index save/load disk persistence."""

    @pytest.fixture(autouse=True)
    def clean_semantic_biencoder_env(self):
        kept = _clean_env()
        yield
        _restore_env(kept)

    @pytest.fixture(autouse=True)
    def patch_sentence_transformer(self, mock_sentence_transformer):
        """Activate the shared deterministic SentenceTransformer mock (conftest)."""

    def test_persist_and_reload_produces_same_results(self, tmp_path):
        """Re-loading a saved FAISS index should give identical routing results."""
        persist_dir = str(tmp_path / "persist")
        cfg = SemanticBiEncoderConfig.from_file(_CONFIG_PATH)
        cfg.vector_store_path = persist_dir

        # Build first router
        router1 = EmbeddingRouter(cfg, persist_dir=persist_dir)
        router1.initialize()
        result1 = router1.route("Write a Python function to sort a list")

        # Create a second router that loads from disk
        router2 = EmbeddingRouter(cfg, persist_dir=persist_dir)
        router2.initialize()  # should load from disk, not rebuild
        result2 = router2.route("Write a Python function to sort a list")

        assert result1["model_name"] == result2["model_name"]
        assert result1["target_name"] == result2["target_name"]
        assert result1["similarity"] == result2["similarity"]

    def test_persist_files_exist(self, tmp_path):
        """Verify that build_index creates index.faiss and docstore.pkl."""
        persist_dir = str(tmp_path / "persist")
        cfg = SemanticBiEncoderConfig.from_file(_CONFIG_PATH)
        cfg.vector_store_path = persist_dir

        router = EmbeddingRouter(cfg, persist_dir=persist_dir)
        router.initialize()

        assert os.path.isfile(os.path.join(persist_dir, "index.faiss"))
        assert os.path.isfile(os.path.join(persist_dir, "docstore.pkl"))

    def test_no_persist_when_dir_is_none(self):
        """Verify that persist_dir=None does not create any files."""
        cfg = SemanticBiEncoderConfig.from_file(_CONFIG_PATH)

        router = EmbeddingRouter(cfg, persist_dir=None)
        router.initialize()

        assert router._faiss_index is not None  # FAISS is still used in-memory
        assert router._persist_dir is None
