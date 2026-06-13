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
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


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
        Number of nearest neighbours to retrieve during routing queries.
    routing_targets : list
        List of :class:`RoutingTarget` dataclasses describing each target.
    vector_store_path : str or None
        Directory path for persisting the FAISS index and docstore.
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
        Load configuration from a JSON file.

        If *path* is ``None``, the file is loaded from the default location:
        ``llm_router_plugins/resources/routing/semantic_biencoder.json``.

        Parameters
        ----------
        path : pathlib.Path or None, optional
            Path to the JSON config file. If ``None``, the bundled default
            config is used.

        Returns
        -------
        SemanticBiEncoderConfig
            An immutable config dataclass populated from the JSON file.

        Raises
        ------
        FileNotFoundError
            If the JSON file does not exist.
        KeyError
            If the JSON file is missing required fields (``embedding_model``,
            ``settings``, ``routing_targets``).
        json.JSONDecodeError
            If the file is not valid JSON.
        ValueError
            If ``chunk_size`` <= 0, ``chunk_overlap`` < 0, or ``top_k`` < 1.
        """
        if path is None:
            path = (
                pathlib.Path(__file__).resolve().parent.parent.parent.parent
                / "resources"
                / "routing"
                / "semantic_biencoder.json"
            )

        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)

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

        return cls(
            embedding_model=raw["embedding_model"],
            chunk_size=settings["chunk_size"],
            chunk_overlap=settings["chunk_overlap"],
            similarity_threshold=settings["similarity_threshold"],
            top_k=settings["top_k"],
            routing_targets=targets,
            vector_store_path=raw.get("vector_store_path"),
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
