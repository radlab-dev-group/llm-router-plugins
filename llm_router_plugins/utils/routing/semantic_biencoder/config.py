"""
Configuration dataclass for the Semantic BiEncoder routing plugin.

JSON structure::

    {
      "embedding_model": "radlab/semantic-euro-bert-encoder-v1",
      "settings": {
        "chunk_size": 256,
        "chunk_overlap": 64,
        "similarity_threshold": 0.0,
        "top_k": 1
      },
      "routing_targets": [
        {
          "name": "code-generation",
          "model_name": "qwen3.6:35b",
          "description": "Model specialized for code-related tasks.",
          "examples": ["Write a Python function to sort a list", ...]
        }
      ],
      "vector_store_path": null
    }
"""

import json
import os
import pathlib

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from llm_router_plugins.utils.routing.constants import (
    SEMANTIC_BIENCODER_ROUTING_PREFIX,
)


# The env var that can hold the entire config as a raw JSON string.
_CONFIG_JSON_ENV = f"{SEMANTIC_BIENCODER_ROUTING_PREFIX}CONFIG"


@dataclass
class SemanticBiEncoderConfig:
    """
    Immutable snapshot of SemanticBiEncoder routing configuration.

    This class is loaded from the JSON config file and provides read-only
    access to the routing targets, embedding model, and chunking settings.

    Parameters
    ----------
    embedding_model : str
        The HuggingFace model identifier used to compute embeddings
        (e.g. ``"radlab/semantic-euro-bert-encoder-v1"``).
    chunk_size : int
        Number of tokens per chunk when splitting target text.
    chunk_overlap : int
        Number of tokens overlapping between adjacent chunks.
    similarity_threshold : float
        Minimum cosine similarity score for a target to be considered valid.
    top_k : int
        Number of nearest neighbors to retrieve during routing queries.
    routing_targets : list
        List of :class:`RoutingTarget` dataclasses describing each target.
    vector_store_path : str or None
        Directory path for persisting the FAISS index and doc_store.
        If ``None``, the index is kept in memory only.
    """

    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    similarity_threshold: float
    top_k: int
    routing_targets: List["RoutingTarget"]
    vector_store_path: Optional[str]

    @property
    def target_names(self) -> List[str]:
        """
        Return the names of all routing targets.

        Returns
        -------
        List[str]
            A list of target name strings, in the order defined in config.
        """
        return [t.name for t in self.routing_targets]

    @property
    def target_models(self) -> Dict[str, str]:
        """
        Return a mapping from target name to model name.

        Returns
        -------
        Dict[str, str]
            A dictionary mapping each target name to its associated model name.
        """
        return {t.name: t.model_name for t in self.routing_targets}

    @classmethod
    def from_file(
        cls, path: Optional[pathlib.Path] = None
    ) -> "SemanticBiEncoderConfig":
        """
        Load configuration from a JSON file or from the ``LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_CONFIG``
        environment variable.

        The env var supports **two forms**:

        1. **Raw JSON string** — value starts with ``{`` or ``[`` → parsed directly.
        2. **File path** — anything else → opened as a JSON config file.

        When the env var is set (non-empty), it takes priority over *path* and
        the default location.  Unlike previous versions, there is no silent
        fall-through: if the specified file does not exist or contains invalid
        JSON the error propagates so the user sees exactly what went wrong.

        When *no* env var is present (or it is empty), the file is loaded from
        *path* if given, or from the default location
        ``llm_router_plugins/resources/routing/semantic_biencoder.json``.

        Parameters
        ----------
        path : pathlib.Path or None, optional
            Path to the JSON config file. Used only when the env var is unset
            or empty.

        Returns
        -------
        SemanticBiEncoderConfig
            An immutable config dataclass populated from the JSON source.

        Raises
        ------
        FileNotFoundError
            If the env var points to a file that does not exist, or if no env
            var is set and the default config is missing.
        KeyError
            If the JSON (from env or file) is missing required fields.
        json.JSONDecodeError
            If the env var value or config file contains invalid JSON.
        ValueError
            If ``chunk_size`` <= 0, ``chunk_overlap`` < 0, or ``top_k`` < 1.
        """
        # ---- env-var shortcut (raw JSON string or file path) -------------------
        raw_json = os.environ.get(_CONFIG_JSON_ENV)
        if raw_json is not None:
            stripped = raw_json.strip()
            # --- Case A: raw JSON string (starts with { or [) ------------------
            if stripped and stripped[0] in ("{", "["):
                return cls.from_json(stripped)
            # --- Case B: file path supplied via env var -------------------------
            if stripped:
                # No fall-through — raise immediately if the file can't be read
                with open(stripped, "r", encoding="utf-8") as fh:
                    raw = json.load(fh)
                return cls._from_raw(raw)
            # env var is set but empty — fall through to path / default

        # ---- file from argument or default location -----------------------------
        if path is None:
            path = (
                pathlib.Path(__file__).resolve().parent.parent.parent.parent
                / "resources"
                / "routing"
                / "semantic_biencoder.json"
            )

        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)

        return cls._from_raw(raw)

    @classmethod
    def from_json(cls, raw: str) -> "SemanticBiEncoderConfig":
        """
        Parse configuration from a raw JSON string.

        The JSON structure mirrors the ``semantic_biencoder.json`` file format::

            {
              "embedding_model": "radlab/semantic-euro-bert-encoder-v1",
              "settings": { "chunk_size": 256, ... },
              "routing_targets": [ ... ],
              "vector_store_path": null
            }

        Parameters
        ----------
        raw : str
            A valid JSON string.

        Returns
        -------
        SemanticBiEncoderConfig
            An immutable config dataclass populated from the parsed JSON.

        Raises
        ------
        KeyError
            If required fields are missing.
        json.JSONDecodeError
            If *raw* is not valid JSON.
        ValueError
            If ``chunk_size`` <= 0, ``chunk_overlap`` < 0, or ``top_k`` < 1.
        """
        if not raw:
            raise ValueError(
                f"SemanticBiEncoderConfig.from_json: empty config string — "
                f"check that {SEMANTIC_BIENCODER_ROUTING_PREFIX}CONFIG env var "
                "is set to a valid JSON object or a file path (not an empty string)"
            )
        parsed = json.loads(raw)
        return cls._from_raw(parsed)

    @staticmethod
    def _from_raw(raw: Dict[str, Any]) -> "SemanticBiEncoderConfig":
        """Internal helper shared by ``from_file`` and ``from_json``."""
        settings = raw["settings"]
        targets: List["RoutingTarget"] = []
        for t in raw["routing_targets"]:
            targets.append(
                RoutingTarget(
                    name=t["name"],
                    model_name=t["model_name"],
                    description=t["description"],
                    examples=t.get("examples", []),
                )
            )

        return SemanticBiEncoderConfig(
            embedding_model=raw["embedding_model"],
            chunk_size=settings["chunk_size"],
            chunk_overlap=settings["chunk_overlap"],
            similarity_threshold=settings["similarity_threshold"],
            top_k=settings["top_k"],
            routing_targets=targets,
            vector_store_path=raw.get("vector_store_path")
            or settings.get("vector_store_path"),
        )


@dataclass(frozen=True)
class RoutingTarget:
    """
    Definition of a single routing target.

    Each target describes a semantic domain (e.g. ``code-generation``,
    ``creative-writing``) along with the model to route to when that
    domain is detected.

    Parameters
    ----------
    name : str
        Unique identifier for this target (used in ``target_name`` in results).
    model_name : str
        The model name to select when this target is the best match.
    description : str
        Human-readable description used for embedding.
    examples : List[str]
        Example user queries used for embedding.  These should be representative
        of the queries that should route to this target.
    """

    name: str
    model_name: str
    description: str
    examples: List[str]
