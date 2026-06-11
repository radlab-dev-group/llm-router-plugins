"""
Configuration dataclass for the SemanticBiEncoder routing plugin.

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
          "description": "Model specialized for code-related tasks...",
          "examples": ["Write a Python function...", ...]
        },
        ...
      ]
    }
"""

import json
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class _TargetDef:
    """A single routing target definition."""

    name: str
    model_name: str
    description: str
    examples: List[str]


@dataclass
class SemanticBiEncoderConfig:
    """Immutable snapshot of the semantic_biencoder JSON config."""

    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    similarity_threshold: float
    top_k: int
    routing_targets: List[_TargetDef]

    @classmethod
    def from_file(cls, path: Optional[pathlib.Path] = None) -> "SemanticBiEncoderConfig":
        """Load configuration from the bundled JSON file."""
        if path is None:
            _DEFAULT_PATH = (
                pathlib.Path(__file__).resolve().parent.parent.parent.parent
                / "resources"
                / "routing"
                / "semantic_biencoder.json"
            )
        else:
            _DEFAULT_PATH = path

        raw = cls._load_json(_DEFAULT_PATH)
        return cls(
            embedding_model=raw.get("embedding_model", "radlab/semantic-euro-bert-encoder-v1"),
            chunk_size=raw.get("settings", {}).get("chunk_size", 256),
            chunk_overlap=raw.get("settings", {}).get("chunk_overlap", 64),
            similarity_threshold=raw.get("settings", {}).get("similarity_threshold", 0.0),
            top_k=raw.get("settings", {}).get("top_k", 1),
            routing_targets=[
                _TargetDef(
                    name=t["name"],
                    model_name=t["model_name"],
                    description=t["description"],
                    examples=t.get("examples", []),
                )
                for t in raw.get("routing_targets", [])
            ],
        )

    @property
    def target_names(self) -> List[str]:
        return [t.name for t in self.routing_targets]

    @property
    def target_models(self) -> Dict[str, str]:
        return {t.name: t.model_name for t in self.routing_targets}

    @staticmethod
    def _load_json(path: pathlib.Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
