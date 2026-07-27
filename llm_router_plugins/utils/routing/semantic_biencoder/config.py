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
from typing import Any, Dict, List, Optional, Tuple

from llm_router_plugins.utils.routing.constants import (
    SEMANTIC_BIENCODER_ROUTING_PREFIX,
)


"""
The env var that can hold the entire config as a raw JSON string.
"""
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
    routing_targets : tuple
        Immutable sequence of :class:`RoutingTarget` dataclasses describing each target.
    vector_store_path : str or None
        Directory path for persisting the FAISS index and doc_store.
        If ``None``, the index is kept in memory only.
    """

    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    similarity_threshold: float
    top_k: int
    routing_targets: tuple["RoutingTarget", ...]
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

        if path is None:
            raise ValueError(
                f"SemanticBiEncoderConfig.from_file: empty config path — "
                f"check that {SEMANTIC_BIENCODER_ROUTING_PREFIX}CONFIG env var "
                "is set to a valid file path (not an empty string)"
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
        for required_key in ("embedding_model", "settings", "routing_targets"):
            if required_key not in raw:
                raise KeyError(
                    f"Missing required top-level key '{required_key}' in config. "
                    f"Available keys: {list(raw.keys())}"
                )

        settings = raw["settings"]
        for setting_key in (
            "chunk_size",
            "chunk_overlap",
            "similarity_threshold",
            "top_k",
        ):
            if setting_key not in settings:
                raise KeyError(
                    f"Missing required field '{setting_key}' in settings. "
                    f"Available fields: {list(settings.keys())}"
                )

        for idx, target in enumerate(raw["routing_targets"]):
            for key in ("name", "model_name", "description"):
                if key not in target:
                    raise KeyError(
                        f"Missing required field '{key}' in routing_targets[{idx}]. "
                        f"Available fields: {list(target.keys())}"
                    )

        targets = tuple(
            RoutingTarget(
                name=t["name"],
                model_name=t["model_name"],
                description=t["description"],
                examples=tuple(t.get("examples", [])),
            )
            for t in raw["routing_targets"]
        )

        chunk_size = settings["chunk_size"]
        chunk_overlap = settings["chunk_overlap"]
        top_k = settings["top_k"]

        if chunk_size <= 0:
            raise ValueError(f"Expected 'chunk_size' > 0, got {chunk_size}.")
        if chunk_overlap < 0:
            raise ValueError(f"Expected 'chunk_overlap' >= 0, got {chunk_overlap}.")
        if top_k < 1:
            raise ValueError(f"Expected 'top_k' >= 1, got {top_k}.")

        return SemanticBiEncoderConfig(
            embedding_model=raw["embedding_model"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            similarity_threshold=settings["similarity_threshold"],
            top_k=top_k,
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
    examples : Tuple[str, ...]
        Example user queries used for embedding.  These should be representative
        of the queries that should route to this target.
    """

    name: str
    model_name: str
    description: str
    examples: Tuple[str, ...]
