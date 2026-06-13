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
          "phrases":    ["write code:5", "fix bug:4", ...],
          "patterns":   ["function\\\\s+\\\\w+", ...],
          "weights":    { "debug": 5, "implement": 4, ... }
        }
      },
      "none": {
          "keywords":   ["hello", ...],
          "phrases":    ["hello:1", ...],
          "patterns":   ["^\\\\b(hello|hi|hey)\\\\b", ...]
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

    This class holds a read-only view of the intent definitions, complexity
    thresholds, default models, and intent-adjustment rules loaded from the
    JSON config file.

    Parameters
    ----------
    thresholds : dict
        Maximum token counts for complexity levels: ``{"simple": 25, "medium": 150}``.
    default_models : dict
        Default model for each complexity level: ``{"simple": "gpt-oss:120b", "medium": "qwen3.6:35b"}``.
    intent_adjustment : dict
        Mapping from intent name to complexity override: ``{"code": "medium", "creative": "simple"}``.
    intents : dict
        Intent category definitions — each key maps to ``{"keywords", "phrases", "patterns", "weights"}``.
    none_keywords : list
        Keywords that indicate "none" intent (greetings, thanks, etc.).
    """

    thresholds: Dict[str, int]
    default_models: Dict[str, str]
    intent_adjustment: Dict[str, str]
    intents: Dict[str, Dict[str, List[str]]]
    none_keywords: List[str]

    @classmethod
    def from_file(cls, path: Optional[pathlib.Path] = None) -> "RoutingConfig":
        """
        Load configuration from the bundled JSON file or *path*.

        If *path* is ``None``, the file is loaded from the default location:
        ``llm_router_plugins/resources/routing/simple_semantic.json``.

        Parameters
        ----------
        path : pathlib.Path or None, optional
            Path to the JSON config file. If ``None``, the bundled default
            config is used.

        Returns
        -------
        RoutingConfig
            An immutable config dataclass populated from the JSON file.

        Raises
        ------
        FileNotFoundError
            If the JSON file does not exist.
        KeyError
            If the JSON file is missing required fields (``settings``, ``intents``).
        json.JSONDecodeError
            If the file is not valid JSON.
        ValueError
            If threshold values are not positive integers.
        """
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
        """
        Return the list of model names from ``default_models``.

        Returns
        -------
        List[str]
            A list of unique model names, preserving order.
        """
        return list(self.default_models.values())

    @property
    def intent_categories(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Return the intent definitions dict.

        Returns
        -------
        Dict[str, Dict[str, List[str]]]
            The full intents dictionary from the config file.
        """
        return self.intents

    @property
    def complexity_thresholds(self) -> Tuple[int, int]:
        """
        Return (simple_threshold, medium_threshold) tuple.

        Returns
        -------
        Tuple[int, int]
            A two-element tuple: (simple_threshold, medium_threshold).

        Raises
        ------
        KeyError
            If the config does not contain both ``"simple"`` and ``"medium"`` keys.
        """
        return self.thresholds["simple"], self.thresholds["medium"]

    @staticmethod
    def _load_json(path: pathlib.Path) -> Dict[str, Any]:
        """
        Load and parse a JSON file.

        Parameters
        ----------
        path : pathlib.Path
            Path to the JSON file to load.

        Returns
        -------
        Dict[str, Any]
            The parsed JSON content as a dictionary.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        json.JSONDecodeError
            If the file is not valid JSON.
        """
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
