"""
Configuration dataclass for the Semantic BiEncoder routing plugin.

JSON structure::

    {
      "embedding_model": "google/embeddinggemma-300m",
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

import pathlib

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional

from llm_router_plugins.utils.routing.constants import (
    SEMANTIC_BIENCODER_ROUTING_PREFIX,
)
from llm_router_plugins.utils.routing.common import RoutingConfigBase
from llm_router_plugins.utils.routing.target import RoutingTarget

# Re-exported for backward compatibility — ``RoutingTarget`` is now a shared
# routing concept (see ``llm_router_plugins.utils.routing.target``).
__all__ = ["RoutingTarget", "SemanticBiEncoderConfig"]


@dataclass
class SemanticBiEncoderConfig(RoutingConfigBase):
    """
    Snapshot of SemanticBiEncoder routing configuration.

    This class is loaded from the JSON config file and provides read-only
    access to the routing targets, embedding model, and chunking settings.

    Parameters
    ----------
    embedding_model : str
        The HuggingFace model identifier used to compute embeddings
        (e.g. ``"google/embeddinggemma-300m"``).
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

    # RoutingConfigBase hooks (ClassVar — not dataclass fields)
    _ENV_PREFIX: ClassVar[str] = SEMANTIC_BIENCODER_ROUTING_PREFIX
    _DEFAULT_CONFIG_PATH: ClassVar[Optional[pathlib.Path]] = (
        pathlib.Path(__file__).resolve().parent.parent.parent.parent
        / "resources"
        / "routing"
        / "semantic_biencoder.json"
    )

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
    def _from_raw(cls, raw: Dict[str, Any]) -> "SemanticBiEncoderConfig":
        """Parse and validate the decoded JSON dict (``RoutingConfigBase`` hook)."""
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
        RoutingConfigBase.validate_semantic_params(chunk_size, chunk_overlap, top_k)

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
