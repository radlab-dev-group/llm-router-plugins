"""
Configuration dataclass for the Simple Semantic routing plugin.

JSON structure::

    {
      "settings": {
        "len_thresholds_max": { "simple": 25, "medium": 150 },
        "default_models":     { "simple": "gpt-oss:120b", "medium": "qwen3.6:35b" },
        "intent_adjustment":  { "code": "medium", "math": "medium",
                               "creative": "simple", "general": "simple",
                               "none": "" }
      },
      "intents": {
        "code": {
          "keywords":   ["code", "program", ...],
          "phrases":    ["write code", "fix bug", ...],
          "patterns":   ["function\\\\s+\\\\w+", "class\\\\s+\\\\w+", ...],
          "weights":    { "debug": 3, "implement": 2, ... }
        }
      },
      "none": {
          "keywords":   ["hello", ...],
          "phrases":    ["hi there", ...],
          "patterns":   []
      }
    }
"""

import json
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class RoutingConfig:
    """
    Immutable snapshot of routing configuration from *simple_semantic.json*.
    """

    thresholds: Dict[str, int]
    default_models: Dict[str, str]
    intent_adjustment: Dict[str, str]
    intents: Dict[str, Dict[str, List[str]]]
    none_keywords: List[str]

    @classmethod
    def from_file(cls, path: Optional[pathlib.Path] = None) -> "RoutingConfig":
        """Load configuration from the bundled JSON file or *path*."""
        if path is None:
            _DEFAULT_PATH = (
                pathlib.Path(__file__).resolve().parent.parent.parent.parent
                / "resources"
                / "routing"
                / "simple_semantic.json"
            )
        else:
            _DEFAULT_PATH = path

        raw = cls._load_json(_DEFAULT_PATH)
        settings = raw["settings"]
        return cls(
            thresholds=settings["len_thresholds_max"],
            default_models=settings["default_models"],
            intent_adjustment=settings["intent_adjustment"],
            intents=raw["intents"],
            none_keywords=raw.get("none", []) or [],
        )

    @property
    def models_list(self) -> List[str]:
        return list(self.default_models.values())

    @property
    def intent_categories(self) -> Dict[str, Dict[str, List[str]]]:
        return self.intents

    @property
    def complexity_thresholds(self) -> Tuple[int, int]:
        return self.thresholds["simple"], self.thresholds["medium"]

    @staticmethod
    def _load_json(path: pathlib.Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
