"""
SimpleSemanticRoutingPlugin — heuristic two‑stage routing (intent + complexity).

Performs intent classification via keyword/phrase/pattern scoring, then selects
a model from a configured pool based on the detected intent and estimated text
complexity.

All configuration is loaded from
``llm_router_plugins/resources/routing/simple_semantic.json`` and can be
overridden by environment variables:

    LLM_ROUTER_ROUTING_COMPLEXITY_THRESHOLDS  - pipe-separated simple|medium
    LLM_ROUTER_ROUTING_MODELS                 - pipe-separated model names
    LLM_ROUTER_ROUTING_DEFAULT_MODEL          - fallback model name
    LLM_ROUTER_ROUTING_INTENT_<name>          - override intent keywords

Example JSON configuration::

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
          "keywords":   ["code", "program", "debug", ...],
          "phrases":    ["write code:5", "fix bug:4", ...],
          "patterns":   ["function\\\\s+\\\\w+", ...],
          "weights":    { "debug": 5, "implement": 4, ... }
        }
      },
      "none": {
          "keywords":   ["hello", "hi", ...],
          "phrases":    ["hello:1", "thanks:1", ...],
          "patterns":   ["^\\\\b(hello|hi|hey)\\\\b", ...],
          "weights":    { "hello": 1, "hi": 1 }
      }
    }
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional

from llm_router_plugins.plugin_interface import PluginInterface
from llm_router_plugins.utils.routing.simple_semantic.config import (
    RoutingConfig,
)


class SimpleSemanticRoutingPlugin(PluginInterface):
    """
    Heuristic semantic routing plugin.

    When ``payload["model"] == "auto"`` the plugin classifies the last user
    message intent (code / math / creative / general / none) and estimates
    text complexity, then selects the best-matching model from a configured
    pool.
    """

    name = "simple_semantic_routing"

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        super().__init__(logger=logger)

        self._config = RoutingConfig.from_file()
        self._models: List[str] = []
        self._intent_categories: Dict[str, Dict[str, Any]] = {}
        self._complexity_thresholds: List[int] = []
        self._default_model: str = ""

        self._load_config()

    def _load_config(self) -> None:
        cfg = self._config

        # --- complexity thresholds ---
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

        # --- intents ---
        intents: Dict[str, Dict[str, Any]] = dict(cfg.intent_categories)
        for key, value in os.environ.items():
            if key.startswith("LLM_ROUTER_ROUTING_INTENT_"):
                category = key[len("LLM_ROUTER_ROUTING_INTENT_") :].lower()
                entries = [e.strip() for e in value.split("|") if e.strip()]
                kw_list = [e for e in entries if ":" not in e]
                ph_list = [e for e in entries if ":" in e]
                intents[category] = {
                    "keywords": kw_list,
                    "phrases": ph_list,
                    "patterns": [],
                    "weights": {},
                }
        self._intent_categories = intents

        # --- default model ---
        dm_env = os.getenv("LLM_ROUTER_ROUTING_DEFAULT_MODEL")
        self._default_model = dm_env or cfg.default_models.get(
            "simple", cfg.models_list[0]
        )

    def apply(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process *payload*.  If ``payload["model"] == "auto"`` route to the
        best-matching model based on intent + complexity.
        """
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
                "SimpleSemanticRouting: intent=%s, complexity=%s (%d tokens) -> %s",
                intent,
                complexity,
                token_estimate,
                payload["model"],
            )
        return payload

    # ------------------------ text extraction

    @staticmethod
    def _get_text_from_payload(payload: Dict[str, Any]) -> str:
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

    # ------------------------ token estimation

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token count estimate: word count * 1.25."""
        word_count = len(text.split()) if text else 0
        return int(word_count * 1.25)

    # ------------------------ complexity

    def _complexity_level(self, token_estimate: int) -> str:
        if token_estimate <= self._complexity_thresholds[0]:
            return "simple"
        if token_estimate <= self._complexity_thresholds[1]:
            return "medium"
        return "complex"

    # ------------------------ intent classification

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

            # weighted keyword scoring
            for kw in keywords:
                w = weights.get(kw, 1)
                if kw in text_lower:
                    score += w

            # phrase scoring (multi-word expressions)
            for phrase in phrases:
                if isinstance(phrase, str) and ":" in phrase:
                    parts = phrase.rsplit(":", 1)
                    p_text, w = parts[0].strip().lower(), float(parts[1].strip())
                else:
                    p_text, w = phrase.lower(), 2.0
                if p_text in text_lower:
                    score += w

            # regex pattern scoring
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

    # ------------------------ model selection

    def _select_model(self, intent: str, complexity: str) -> str:
        n = len(self._models)

        complexity_map: Dict[str, int] = {
            "simple": 0,
            "medium": n // 2,
            "complex": n - 1,
        }
        idx = complexity_map.get(complexity, 0)
        intent_idx = self._intent_index(intent)

        if intent_idx >= 0:
            idx = max(idx, intent_idx)
        else:
            idx -= 1

        idx = max(0, min(idx, n - 1))
        return self._models[idx]

    def _complexity_to_model_index(self, target: str) -> int:
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
        target = self._config.intent_adjustment.get(intent, "")
        if not target:
            return -1
        return self._complexity_to_model_index(target)
