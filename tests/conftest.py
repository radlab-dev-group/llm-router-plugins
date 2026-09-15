"""
Shared pytest fixtures for the routing plugin test suites.
"""

import numpy as np
import pytest


def _make_mock_model_class():
    """
    Build a deterministic stand-in for ``sentence_transformers.SentenceTransformer``.

    The mock produces normalized, seeded random vectors so cosine similarities
    are stable across runs and no model download is ever triggered.
    """

    class MockModel:
        embed_dim = 768

        def __init__(self, *args, **kwargs):
            pass

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

    return MockModel


@pytest.fixture
def mock_sentence_transformer(monkeypatch):
    """
    Patch ``sentence_transformers.SentenceTransformer`` with a deterministic
    mock so the routing tests never download a real model.

    The patch is applied to the ``sentence_transformers`` module itself, which
    keeps working with the lazy ``from sentence_transformers import
    SentenceTransformer`` imports inside the embedder.
    """
    import sentence_transformers

    MockModel = _make_mock_model_class()
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", MockModel)
    return MockModel
