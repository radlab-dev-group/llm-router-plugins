"""
Embedding-based router using BiEncoder with FAISS vector store.

For each routing target, the embedder pre-computes a set of embeddings from the
target's description and examples using a sliding-window context.  At query time
the user message is embedded and matched against all stored embeddings via FAISS
(inner product on L2-normalized vectors = cosine similarity), returning the
best-matching target.

When *persist_dir* is provided the FAISS index and docstore are saved to disk
(on ``{persist_dir}/index.faiss`` and ``{persist_dir}/docstore.pkl``) and
re-loaded on the next initialization.
"""

import functools
import os
import pickle
import logging

import numpy as np

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer

from llm_router_plugins.utils.routing.semantic_biencoder.config import (
    SemanticBiEncoderConfig,
)

@functools.lru_cache(maxsize=1)
def _import_faiss() -> Any:
    """
    Lazy-import FAISS on first call, then cache the result.

    Returns
    -------
    Any
        The faiss module object.
    """
    import faiss  # type: ignore[import-untyped]

    return faiss


@dataclass
class _TargetEmbeddings:
    """
    Legacy dataclass kept for backward compat — no longer used internally.

    Attributes
    ----------
    name : str
    model_name : str
    embeddings : np.ndarray
    labels : List[str]
    """

    name: str
    model_name: str
    embeddings: np.ndarray  # shape: (n_chunks, embed_dim)
    labels: List[str]  # name of each chunk (for debugging)


FAISS = _import_faiss()


class EmbeddingRouter:
    """
    Router that uses a BiEncoder model to compute embeddings and selects the
    nearest semantic target for each incoming user message.

    Parameters
    ----------
    config : SemanticBiEncoderConfig
        Routing configuration (targets, chunking params, embedding model).
    logger : logging.Logger, optional
        Logger instance.
    persist_dir : str, optional
        Directory where the FAISS index and docstore are saved.  If
        ``None`` the index is kept in memory only.
    """

    def __init__(
        self,
        config: SemanticBiEncoderConfig,
        logger: Optional[logging.Logger] = None,
        persist_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize the router with configuration and optional persistence.

        Parameters
        ----------
        config : SemanticBiEncoderConfig
            Routing configuration (targets, chunking params, embedding model).
        logger : logging.Logger, optional
            Logger instance.
        persist_dir : str, optional
            Directory where the FAISS index and docstore are saved.

        Raises
        ------
        ValueError
            If *config* contains no routing targets.
        """
        self._config = config
        self._logger = logger
        self._persist_dir: Optional[str] = persist_dir
        self._model: Optional[SentenceTransformer] = None
        self._faiss_index: Any = None

        # doc_id -> target_name
        self._doc_store: Dict[int, str] = {}

        self._id_counter: int = 0
        self._initialized = False

    @property
    def has_vectors(self) -> bool:
        """True when the FAISS index contains at least one vector."""
        return (
            self._faiss_index is not None
            and getattr(self._faiss_index, "ntotal", 0) > 0
        )

    def initialize(self) -> None:
        """
        Load the model and pre-compute / load the FAISS index.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the embedding model cannot be loaded (e.g. network failure).
        ValueError
            If no routing targets are defined.
        """
        if self._initialized:
            return

        self._load_model()

        if self._persist_dir and self._load_index():
            if self._logger:
                self._logger.info(
                    "FAISS index loaded from %s (%d vectors)",
                    self._persist_dir,
                    self._faiss_index.ntotal if self._faiss_index else 0,
                )
            self._initialized = True
            return

        self._build_index()
        self._save_index()
        self._initialized = True

    def route(self, user_message: str) -> Dict[str, Any]:
        """
        Embed *user_message* and return the best-matching routing target.

        Parameters
        ----------
        user_message : str
            The user's input text to route.

        Returns
        -------
        dict
            Dictionary with keys:
            - ``model_name`` (str): the model to use
            - ``target_name`` (str): the matched target name
            - ``similarity`` (float): cosine similarity score (0–1)
            - ``all_scores`` (List[dict]): full ranking

        Raises
        ------
        RuntimeError
            If the router has not been initialised.
        """
        self._ensure_initialized()
        assert self._model is not None
        assert self._faiss_index is not None

        user_embedding = self._model.encode(
            [user_message], show_progress_bar=False, convert_to_numpy=True
        )
        user_embedding = self._to_numpy(user_embedding)
        user_embedding = user_embedding.squeeze()  # (embed_dim,)

        # L2-normalise the query
        norm = float(np.linalg.norm(user_embedding))
        if norm > 0:
            user_embedding = user_embedding / norm
        user_embedding = user_embedding.reshape(1, -1)  # (1, embed_dim)

        # FAISS query
        k = min(self._config.top_k, self._faiss_index.ntotal)
        scores, doc_ids = self._faiss_index.search(user_embedding, k)

        # Aggregate scores per target
        target_scores: Dict[str, List[float]] = {}
        for s, doc_id in zip(scores[0], doc_ids[0]):
            if doc_id < 0:
                continue
            tname = self._doc_store.get(doc_id, "unknown")
            target_scores.setdefault(tname, []).append(float(s))

        # Build ranked list
        target_models: Dict[str, str] = {
            t.name: t.model_name for t in self._config.routing_targets
        }
        all_scores: List[Tuple[str, float, str]] = []
        for tname, sims in target_scores.items():
            avg_sim = float(np.mean(sims))
            all_scores.append((tname, avg_sim, target_models.get(tname, "unknown")))
        all_scores.sort(key=lambda x: x[1], reverse=True)

        if all_scores:
            best_name, best_sim, best_model = all_scores[0]
        else:
            best_name, best_sim, best_model = "unknown", 0.0, "unknown"

        return {
            "model_name": best_model,
            "target_name": best_name,
            "similarity": best_sim,
            "all_scores": [{"target": n, "similarity": s} for n, s, _ in all_scores],
        }

    def _load_model(self) -> None:
        """
        Load the SentenceTransformer embedding model.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the model cannot be loaded.
        """
        model_name = self._config.embedding_model
        if self._logger:
            self._logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(
            model_name, device="cpu", trust_remote_code=True
        )
        if self._logger:
            self._logger.info("Embedding model loaded successfully.")

    def _build_index(self) -> None:
        """
        Encode all target chunks and build the FAISS index.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If no embeddings can be produced.
        RuntimeError
            If FAISS fails to create the index.
        """
        tokenizer = getattr(self._model, "tokenizer", None)  # type: ignore[union-attr]
        chunk_size = self._config.chunk_size
        overlap = self._config.chunk_overlap
        total_chunks = 0

        for target in self._config.routing_targets:
            texts: List[str] = [f"Target: {target.name}. {target.description}"]
            texts.extend(target.examples)

            chunks: List[str] = []
            for text in texts:
                chunks.extend(
                    self._split_into_chunks(text, chunk_size, overlap, tokenizer)
                )

            if not chunks:
                continue

            embeddings = self._model.encode(
                chunks,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            embeddings = self._to_numpy(embeddings)

            # L2-normalise so inner-product = cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            normalized = embeddings / norms

            # Create the FAISS index on first batch (needs the dimension)
            if self._faiss_index is None:
                dim = normalized.shape[1]
                self._faiss_index = FAISS.IndexFlatIP(dim)

            # Batch-add all chunks for this target in one call
            n_before = len(self._doc_store)
            self._faiss_index.add(normalized)
            for i in range(len(normalized)):
                self._doc_store[n_before + i] = target.name
                self._id_counter += 1
                total_chunks += 1

        if self._logger:
            self._logger.info(
                "Index built: %d targets, %d total embeddings.",
                len(self._config.routing_targets),
                total_chunks,
            )

    def _load_index(self) -> bool:
        """
        Load a previously-saved FAISS index and docstore. Returns False on failure.

        Returns
        -------
        bool
            True if the index was loaded successfully, False otherwise.

        Raises
        ------
        RuntimeError
            If the FAISS index is corrupted.
        pickle.UnpicklingError
            If the docstore pickle file is corrupted.
        """
        index_path = os.path.join(self._persist_dir, "index.faiss")
        docstore_path = os.path.join(self._persist_dir, "docstore.pkl")
        if not (
            os.path.isdir(self._persist_dir)
            and os.path.isfile(index_path)
            and os.path.isfile(docstore_path)
        ):
            return False

        faiss = _import_faiss()
        faiss_index = faiss.read_index(index_path)
        with open(docstore_path, "rb") as fh:
            docstore = pickle.load(fh)

        # Verify dimensionality matches the current model
        if self._model is not None:
            dummy = self._model.encode(
                ["."], show_progress_bar=False, convert_to_numpy=True
            )
            dim = len(self._to_numpy(dummy))
            try:
                if faiss_index.d != dim:
                    if self._logger:
                        self._logger.warning(
                            "Dimension mismatch (%d vs %d) — rebuilding index",
                            faiss_index.d,
                            dim,
                        )
                    return False
            except AttributeError:
                if self._logger:
                    self._logger.warning("Corrupted FAISS index (no .d) — rebuilding")
                return False

        self._faiss_index = faiss_index
        self._doc_store = docstore
        self._id_counter = faiss_index.ntotal
        return True

    def save_index(self) -> None:
        """
        Persist the current FAISS index and docstore to disk.

        Returns
        -------
        None

        Raises
        ------
        IOError
            If the directory cannot be created or files cannot be written.
        """
        if not self._persist_dir or self._faiss_index is None:
            return
        os.makedirs(self._persist_dir, exist_ok=True)
        FAISS.write_index(
            self._faiss_index, os.path.join(self._persist_dir, "index.faiss")
        )
        with open(os.path.join(self._persist_dir, "docstore.pkl"), "wb") as fh:
            pickle.dump(self._doc_store, fh)
        if self._logger:
            self._logger.info("FAISS index saved to %s", self._persist_dir)

    def _save_index(self) -> None:
        """
        Internal save — always run after building the index.

        Returns
        -------
        None

        Raises
        ------
        IOError
            If disk write fails.
        """
        self.save_index()

    def _ensure_initialized(self) -> None:
        """
        Ensure the router is initialised.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If :meth:`initialize` fails.
        """
        if not self._initialized:
            self.initialize()

    @staticmethod
    def _to_numpy(embedding: Any) -> np.ndarray:
        """Convert an embedding (possibly a Python ``list``) to a NumPy array."""
        return np.asarray(embedding) if isinstance(embedding, list) else embedding

    @staticmethod
    def _split_into_chunks(
        text: str, chunk_size: int, overlap: int, tokenizer: Optional[Any] = None
    ) -> List[str]:
        """
        Split *text* into overlapping chunks of *chunk_size* tokens.

        Parameters
        ----------
        text : str
            The input text to split.
        chunk_size : int
            Maximum number of **tokens** per chunk (not words).
        overlap : int
            Number of overlapping tokens between consecutive chunks.
        tokenizer : SentenceTransformerTokenizer, optional
            When provided, *chunk_size* is measured in real model tokens
            (via ``tokenizer.encode`` / ``decode``).  Without a tokenizer the
            method falls back to simple word splitting (``text.split()``), so
            *chunk_size* becomes an approximation based on whitespace-delimited
            words.

        Returns
        -------
        List[str]
            A list of overlapping text chunks.

        Raises
        ------
        ValueError
            If *chunk_size* <= 0 or *overlap* >= *chunk_size*.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        if overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({overlap}) must be < chunk_size ({chunk_size})"
            )

        # Tokenizer-based path (produces correct token counts)
        if tokenizer is not None:
            ids = tokenizer.encode(text, return_tensors="pt")[0].tolist()
            n_tokens = len(ids)
            if n_tokens <= chunk_size:
                return [text]

            stride = max(chunk_size - overlap, 1)
            chunks: List[str] = []
            start = 0
            while start < n_tokens:
                end = min(start + chunk_size, n_tokens)
                chunks.append(
                    tokenizer.decode(ids[start:end], skip_special_tokens=True)
                )
                if end >= n_tokens:
                    break
                start += stride
            return chunks

        # Word-splitting fallback (chunk_size ≈ word count)
        tokens = text.split()
        if len(tokens) <= chunk_size:
            return [text]

        stride = max(chunk_size - overlap, 1)
        chunks: List[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunks.append(" ".join(tokens[start:end]))
            if end >= len(tokens):
                break
            start += stride
        return chunks

