"""
Embedding-based router using BiEncoder with FAISS vector store.

For each routing target, the embedder pre-computes a set of embeddings from the
target's description and examples using a sliding-window context.  At query time
the user message is embedded and matched against all stored embeddings via FAISS
(inner product on L2-normalised vectors = cosine similarity), returning the
best-matching target.

When *persist_dir* is provided the FAISS index and docstore are saved to disk
(on ``{persist_dir}/index.faiss`` and ``{persist_dir}/docstore.pkl``) and
re-loaded on the next initialisation.
"""

import logging
from dataclasses import dataclass
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from llm_router_plugins.utils.routing.semantic_biencoder.config import (
    SemanticBiEncoderConfig,
)

_faiss: Any = None


def _import_faiss() -> Any:
    """
    Lazy import of faiss to allow module loading without it installed.

    Returns
    -------
    Any
        The faiss module object.

    Raises
    ------
    None
    """
    global _faiss
    if _faiss is None:
        import faiss as _faiss_module

        _faiss = _faiss_module
    return _faiss


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

        Returns
        -------
        None

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
        self._docstore: Dict[int, str] = {}  # doc_id -> target_name
        self._id_counter: int = 0
        self._initialized = False

    # -------------- init

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
        chunk_size = self._config.chunk_size
        overlap = self._config.chunk_overlap
        total_chunks = 0

        for target in self._config.routing_targets:
            texts: List[str] = [f"Target: {target.name}. {target.description}"]
            texts.extend(target.examples)

            chunks: List[str] = []
            for text in texts:
                chunks.extend(self._split_into_chunks(text, chunk_size, overlap))

            if not chunks:
                continue

            embeddings = self._model.encode(
                chunks,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings)

            # L2-normalise so inner-product = cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            normalized = embeddings / norms

            # Create the FAISS index on first batch (needs the dimension)
            if self._faiss_index is None:
                dim = normalized.shape[1]
                self._faiss_index = _import_faiss().IndexFlatIP(dim)

            # Add each chunk individually with a monotonically increasing doc ID
            for i in range(len(normalized)):
                self._faiss_index.add(normalized[i : i + 1])
                self._docstore[self._id_counter] = target.name
                self._id_counter += 1
                total_chunks += 1

        if self._logger:
            self._logger.info(
                "Index built: %d targets, %d total embeddings.",
                len(self._config.routing_targets),
                total_chunks,
            )

    # ------ query

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
        if isinstance(user_embedding, list):
            user_embedding = np.array(user_embedding)
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
            tname = self._docstore.get(doc_id, "unknown")
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

    # ------ I/O

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
        faiss_module = _import_faiss()
        faiss_module.write_index(
            self._faiss_index, os.path.join(self._persist_dir, "index.faiss")
        )
        with open(os.path.join(self._persist_dir, "docstore.pkl"), "wb") as fh:
            pickle.dump(self._docstore, fh)
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

        _faiss = _import_faiss()
        faiss_index = _faiss.read_index(index_path)
        with open(docstore_path, "rb") as fh:
            docstore = pickle.load(fh)

        # Verify dimensionality matches the current model
        if self._model is not None:
            dummy = self._model.encode(
                ["."], show_progress_bar=False, convert_to_numpy=True
            )
            if isinstance(dummy, list):
                dummy = np.array(dummy)
            dim = len(dummy)
            if faiss_index.d != dim:
                if self._logger:
                    self._logger.warning(
                        "Dimension mismatch (%d vs %d) — rebuilding index",
                        faiss_index.d,
                        dim,
                    )
                return False

        self._faiss_index = faiss_index
        self._docstore = docstore
        self._id_counter = faiss_index.ntotal
        return True

    # ------ helpers
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

    # ------ split / similarity utilities
    @staticmethod
    def _split_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Split *text* into overlapping chunks of *chunk_size* tokens.

        Parameters
        ----------
        text : str
            The input text to split.
        chunk_size : int
            Maximum number of tokens per chunk.
        overlap : int
            Number of overlapping tokens between consecutive chunks.

        Returns
        -------
        List[str]
            A list of overlapping text chunks.

        Raises
        ------
        ValueError
            If *chunk_size* <= 0 or *overlap* >= *chunk_size*.
        """
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
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between vector *a* (dim,) and matrix *b*
        (n, dim). Returns array of shape (n,).

        Parameters
        ----------
        a : np.ndarray
            Query vector of shape (embed_dim,).
        b : np.ndarray
            Reference matrix of shape (n, embed_dim).

        Returns
        -------
        np.ndarray
            Array of cosine similarities of shape (n,).

        Raises
        ------
        ValueError
            If the dimensions of *a* and *b* do not match.
        """
        a_norm = np.linalg.norm(a)
        if a_norm == 0:
            return np.zeros(b.shape[0])
        b_norm = np.linalg.norm(b, axis=1)
        b_norm[b_norm == 0] = 1e-10
        return np.dot(b, a) / (b_norm * a_norm)
