"""
Embedding-based router using BiEncoder.

For each routing target, the embedder pre-computes a set of embeddings from the
target's description and examples using a sliding-window context.  At query time
the user message is embedded and matched against all stored embeddings via cosine
similarity, returning the best-matching target.
"""

import logging
import numpy as np

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer

from llm_router_plugins.utils.routing.semantic_biencoder.config import (
    SemanticBiEncoderConfig,
)


@dataclass
class _TargetEmbeddings:
    """Store pre-computed embeddings for a single routing target."""

    name: str
    model_name: str
    embeddings: np.ndarray  # shape: (n_chunks, embed_dim)
    labels: List[str]  # name of each chunk (for debugging)


class EmbeddingRouter:
    """
    Router that uses a BiEncoder model to compute embeddings and selects the
    nearest semantic target for each incoming user message.
    """

    def __init__(
        self,
        config: SemanticBiEncoderConfig,
        logger: Optional[logging.Logger] = None,
    ) -> None:

        self._config = config
        self._logger = logger
        self._model: Optional[SentenceTransformer] = None
        self._targets: List[_TargetEmbeddings] = []
        self._initialized = False

    # ------------------------------------------------------------------ init
    def initialize(self) -> None:
        """Load the model and pre-compute embeddings for all targets."""
        if self._initialized:
            return

        self._load_model()
        self._build_index()
        self._initialized = True

    def _load_model(self) -> None:
        model_name = self._config.embedding_model
        if self._logger:
            self._logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name, device="cpu", trust_remote_code=True)
        if self._logger:
            self._logger.info("Embedding model loaded successfully.")

    def _build_index(self) -> None:
        """Pre-compute embeddings for every routing target."""
        chunk_size = self._config.chunk_size
        overlap = self._config.chunk_overlap

        for target in self._config.routing_targets:
            texts: List[str] = []
            texts.append(f"Target: {target.name}. {target.description}")
            texts.extend(target.examples)

            chunks: List[str] = []
            for text in texts:
                chunks.extend(
                    self._split_into_chunks(text, chunk_size, overlap)
                )

            if not chunks:
                continue

            embeddings = self._model.encode(
                chunks,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings)

            self._targets.append(
                _TargetEmbeddings(
                    name=target.name,
                    model_name=target.model_name,
                    embeddings=embeddings,
                    labels=[f"{target.name}" for _ in chunks],
                )
            )

        if self._logger:
            self._logger.info(
                "Index built: %d targets, %d total embeddings.",
                len(self._targets),
                sum(t.embeddings.shape[0] for t in self._targets),
            )

    # --------------------------------------------------------- query
    def route(self, user_message: str) -> Dict[str, Any]:
        """
        Embed *user_message* and return the best-matching routing target.

        Returns a dict with keys:
            - ``model_name`` (str): the model to use
            - ``target_name`` (str): the matched target name
            - ``similarity`` (float): cosine similarity score (0–1)
            - ``all_scores`` (List[Tuple[str, float]]): full ranking
        """
        self.initialize()
        user_embedding = self._model.encode(
            [user_message], show_progress_bar=False, convert_to_numpy=True
        )
        if isinstance(user_embedding, list):
            user_embedding = np.array(user_embedding)
        user_embedding = user_embedding.squeeze()  # (embed_dim,)

        all_scores: List[Tuple[str, float, str]] = []

        for target in self._targets:
            # cosine similarity: (1 x N) dot (N x M) → (1 x M)
            sims = self._cosine_similarity(user_embedding, target.embeddings)
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])
            all_scores.append((target.name, best_score, target.model_name))

        # Sort by similarity descending
        all_scores.sort(key=lambda x: x[1], reverse=True)

        best_name, best_sim, best_model = all_scores[0]

        return {
            "model_name": best_model,
            "target_name": best_name,
            "similarity": best_sim,
            "all_scores": [
                {"target": n, "similarity": s} for n, s, _ in all_scores
            ],
        }

    # -------------------------------------------------------- utilities
    @staticmethod
    def _split_into_chunks(
        text: str, chunk_size: int, overlap: int
    ) -> List[str]:
        """Split *text* into overlapping chunks of *chunk_size* tokens."""
        tokens = text.split()
        if len(tokens) <= chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        stride = chunk_size - overlap
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunks.append(" ".join(tokens[start:end]))
            if end >= len(tokens):
                break
            start += stride
        return chunks

    @staticmethod
    def _cosine_similarity(
        a: np.ndarray, b: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between vector *a* (dim,) and matrix *b*
        (n, dim). Returns array of shape (n,).
        """
        a_norm = np.linalg.norm(a)
        if a_norm == 0:
            return np.zeros(b.shape[0])
        b_norm = np.linalg.norm(b, axis=1)
        b_norm[b_norm == 0] = 1e-10
        return np.dot(b, a) / (b_norm * a_norm)
