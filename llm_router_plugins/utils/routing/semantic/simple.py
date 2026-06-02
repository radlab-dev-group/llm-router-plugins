"""
Default semantic routing plugin.

Performs two-stage heuristic routing: intent classification + complexity
analysis to select the best model when ``payload["model"] == "auto"``.

All configuration is loaded from ``resources/routing/semantic/simple.json``.
"""

import os
import json
import logging
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from llm_router_plugins.plugin_interface import PluginInterface

_CONFIG_PATH = (
    pathlib.Path(__file__).resolve().parent.parent.parent.parent
    / "resources"
    / "routing"
    / "semantic"
    / "simple.json"
)


@dataclass(frozen=True)
class RoutingConfig:
    """Immutable snapshot of routing configuration from *simple.json*.

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
              "patterns":   ["function\\s+\\w+", "class\\s+\\w+", ...],
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

    thresholds: Dict[str, int]
    default_models: Dict[str, str]
    intent_adjustment: Dict[str, str]
    intents: Dict[str, Dict[str, List[str]]]
    none_keywords: List[str]

    @classmethod
    def from_file(cls, path: pathlib.Path = _CONFIG_PATH) -> "RoutingConfig":
        """
        Load configuration from *path* and return a new :class:`RoutingConfig`.
        """
        raw = cls._load_config_json(path)
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
    def _load_config_json(path: pathlib.Path) -> Dict[str, Any]:
        """
        Load and return the JSON config file.

        Validates early at import time by calling this function at module level.
        """
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)


class DefaultSemanticRoutingPlugin(PluginInterface):
    """
    Semantic routing plugin that selects a model from a configured pool
    based on the intent and complexity of the user's input.

    All configuration lives in the :class:`RoutingConfig` dataclass --
    everything is loaded from ``resources/routing/semantic/simple.json``.
    Environment variables prefixed with ``LLM_ROUTER_ROUTING_`` override
    the JSON config when explicitly set.
    """

    name = "simple_semantic_routing"

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        super().__init__(logger=logger)

        self._config = RoutingConfig.from_file()
        self._models: List[str] = []
        self._intent_categories: Dict[str, List[str]] = {}
        self._complexity_thresholds: List[int] = []
        self._default_model: str = ""

        self._load_config()

    def _load_config(self) -> None:
        cfg = self._config

        thresh_str = os.getenv("LLM_ROUTER_ROUTING_COMPLEXITY_THRESHOLDS", "")
        if thresh_str:
            parts = thresh_str.split("|")
            if len(parts) == 2:
                try:
                    self._complexity_thresholds = [int(parts[0]), int(parts[1])]
                except ValueError:
                    if self._logger:
                        self._logger.warning(
                            "Malformed LLM_ROUTER_ROUTING_COMPLEXITY_THRESHOLDS, "
                            "using defaults"
                        )
                    self._complexity_thresholds = list(cfg.complexity_thresholds)
            else:
                if self._logger:
                    self._logger.warning(
                        "LLM_ROUTER_ROUTING_COMPLEXITY_THRESHOLDS must have 2 values, "
                        "using defaults"
                    )
                self._complexity_thresholds = list(cfg.complexity_thresholds)
        else:
            self._complexity_thresholds = [
                cfg.thresholds["simple"],
                cfg.thresholds["medium"],
            ]

        # --- models ---
        models_str = os.getenv("LLM_ROUTER_ROUTING_MODELS", "")
        if models_str:
            self._models = [m.strip() for m in models_str.split("|") if m.strip()]
        else:
            self._models = cfg.models_list

        # --- intent categories ---
        intents: Dict[str, Dict[str, List[str]]] = cfg.intent_categories

        # Environment variables override JSON
        for key, value in os.environ.items():
            if key.startswith("LLM_ROUTER_ROUTING_INTENT_"):
                category = key[len("LLM_ROUTER_ROUTING_INTENT_") :]
                entries = [e.strip() for e in value.split("|") if e.strip()]
                # Parse key:value pairs for weights
                kw_list = []
                ph_list = []
                for entry in entries:
                    if ":" in entry:
                        ph_list.append(entry)  # treated as phrase:"weight"
                    else:
                        kw_list.append(entry)
                merged: Dict[str, List[str]] = {
                    "keywords": kw_list,
                    "phrases": ph_list,
                    "patterns": [],
                    "weights": {},
                }
                intents[category.lower()] = merged

        self._intent_categories = intents

        # --- default model ---
        dm_env = os.getenv("LLM_ROUTER_ROUTING_DEFAULT_MODEL")
        if dm_env:
            self._default_model = dm_env
        else:
            self._default_model = cfg.default_models.get(
                "simple", cfg.models_list[0]
            )

    # ---- routing -------------------------------------------------------

    def apply(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if payload.get("model") != "auto":
            return payload

        text = self._get_text_from_payload(payload)
        if not text:
            payload["model"] = self._default_model
            if self._logger:
                self._logger.warning(
                    "No text content found, using default model %s",
                    self._default_model,
                )
            return payload

        intent = self._classify_intent(text)
        token_estimate = self._estimate_tokens(text)
        complexity = self._complexity_level(token_estimate)
        payload["model"] = self._select_model(intent, complexity)

        if self._logger:
            self._logger.info(
                "Semantic routing: intent=%s, complexity=%s (%d tokens) -> %s",
                intent,
                complexity,
                token_estimate,
                payload["model"],
            )
        return payload

    def _get_text_from_payload(self, payload: Dict[str, Any]) -> str:
        messages = payload.get("messages")
        if isinstance(messages, list) and messages:
            last_msg = messages[-1]
            content = last_msg.get("content", "")
            if content:
                return str(content)
        for key in ("user_last_statement", "query", "prompt", "input"):
            val = payload.get(key)
            if val:
                return str(val)
        return ""

    def _estimate_tokens(self, text: str) -> int:
        """Rough token count estimate: whitespace word count * 1.25.

        This is not a real tokenizer -- it uses a simple heuristic.
        """
        word_count = len(text.split()) if text else 0
        return int(word_count * 1.25)

    def _complexity_level(self, token_estimate: int) -> str:
        if token_estimate <= self._complexity_thresholds[0]:
            return "simple"
        if token_estimate <= self._complexity_thresholds[1]:
            return "medium"
        return "complex"

    def _classify_intent(self, text: str) -> str:
        text_lower = text.lower()
        best_category = "none"
        best_score = 0.0

        for category, intent_def in self._intent_categories.items():
            if category == "none":
                continue

            keywords = intent_def.get("keywords", [])
            phrases = intent_def.get("phrases", [])
            patterns = intent_def.get("patterns", [])
            weights = intent_def.get("weights", {})

            score = 0.0

            # --- weighted keyword scoring ---
            for kw in keywords:
                w = weights.get(kw, 1)
                if kw in text_lower:
                    score += w

            # --- phrase scoring (multi-word expressions) ---
            for phrase in phrases:
                if isinstance(phrase, str) and ":" in phrase:
                    # phrase:"weight" format
                    parts = phrase.rsplit(":", 1)
                    p_text, w = parts[0].strip().lower(), float(parts[1].strip())
                else:
                    p_text, w = phrase.lower(), 2.0
                if p_text in text_lower:
                    score += w

            # --- regex pattern scoring ---
            for pat in patterns:
                try:
                    if re.search(pat, text_lower):
                        score += 3.0
                except re.error:
                    pass

            if score > best_score:
                best_score = score
                best_category = category

        return best_category

    def _select_model(self, intent: str, complexity: str) -> str:
        n = len(self._models)

        # 1. Base index from complexity
        complexity_map: Dict[str, int] = {
            "simple": 0,
            "medium": n // 2,
            "complex": n - 1,
        }
        idx = complexity_map.get(complexity, 0)

        # 2. Index from intent target
        intent_idx = self._intent_index(intent)

        # 3. If intent matched, use the stronger signal (boost toward intent).
        #    If no intent matched, demote one step from complexity baseline.
        if intent_idx >= 0:
            idx = max(idx, intent_idx)
        else:
            idx -= 1

        # 4. Clamp to valid range
        idx = max(0, min(idx, n - 1))
        return self._models[idx]

    # ---- internal helpers ------------------------------------------------

    def _complexity_to_model_index(self, target: str) -> int:
        """Map a complexity level name to a model index via the config."""
        if target in self._config.default_models:
            name = self._config.default_models[target]
            for i in range(len(self._models) - 1, -1, -1):
                if self._models[i] == name:
                    return i
        fallback: Dict[str, int] = {
            "simple": 0,
            "medium": len(self._models) // 2,
        }
        return fallback.get(target, len(self._models) - 1)

    def _intent_index(self, intent: str) -> int:
        """Return the model index for *intent* via intent_adjustment, or -1."""
        target = self._config.intent_adjustment.get(intent, "")
        if not target:
            return -1
        return self._complexity_to_model_index(target)
