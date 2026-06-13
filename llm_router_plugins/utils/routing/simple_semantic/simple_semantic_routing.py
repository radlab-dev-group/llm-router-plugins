"""
SimpleSemanticRoutingPlugin — heuristic two‑stage routing (intent + complexity).

Performs intent classification via keyword/phrase/pattern scoring, then selects
a model from a configured pool based on the detected intent and estimated text
complexity.

No embedding model is required — routing is a fast, pure‑text classification.

===== Scoring stages =====

**Stage 1 — Intent classification**

Each intent category (e.g. ``code``, ``math``, ``creative``, ``general``) defines
three scoring sources. All three are summed; the intent with the highest total
score wins. If no intent score exceeds zero the intent is classified as ``none``.

- **Keywords** — each keyword in the JSON carries an optional weight.
  If the keyword is found (case‑insensitive) in the input text the score
  increases by that weight (default 1).

- **Phrases** — multi‑word expressions like ``"write code:5"`` where the number
  after ``:`` is the weight (default 2.0). If the phrase is found in the
  lower‑cased input the score increases by that weight.

- **Patterns** — regex patterns; each match adds **3.0** to the score.

**Stage 2 — Complexity estimation**

Token count is estimated with a simple word‑count heuristic::

    token_estimate = len(input_text.split()) * 1.25

This is compared against two thresholds from the config::

    | Tokens              | Complexity |
    |-----      --------|------------|
    | <= simple threshold | simple     |
    | <= medium threshold | medium     |
    | > medium threshold  | complex    |

**Stage 3 — Model selection**

The pool of models is a list defined in ``default_models`` (e.g.
``["gpt-oss:120b", "qwen3.6:35b"]``). The complexity and intent together
determine the index into this pool::

    complexity -> base index:  simple -> 0,  medium -> n//2,  complex -> n-1
    intent_adjustment (if present) -> can INCREASE the base index but never
                                      decrease it
    clamp to [0, n-1] -> pick model at that index

If no text is found or the intent is ``none`` the fallback model
``default_models["simple"]`` is used.

===== Configuration =====

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
    Heuristic intent + complexity based model selection.

    Activates only when ``payload["model"] == "auto"``.  The plugin:

    1. Extracts the last user message text from the payload.
    2. Classifies intent (code / math / creative / general) by scoring
       keywords, phrases, and regex patterns for each intent category.
    3. Estimates complexity (simple / medium / complex) from token count.
    4. Selects the model from a configured pool based on intent and complexity.

    Attributes
    ----------
    name : str
        Plugin identifier used in the registry (``"simple_semantic_routing"``).
    """

    name = "simple_semantic_routing"

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize the plugin: load config and resolve environment overrides.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger instance.

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist.
        KeyError
            If the config file is missing required fields.
        ValueError
            If threshold values are not valid integers.
        """
        super().__init__(logger=logger)

        self._config = RoutingConfig.from_file()
        self._models: List[str] = []
        self._intent_categories: Dict[str, Dict[str, Any]] = {}
        self._complexity_thresholds: List[int] = []
        self._default_model: str = ""

        self._load_config()

    def _load_config(self) -> None:
        """
        Load configuration from JSON and apply environment variable overrides.

        Returns
        -------
        None

        Raises
        ------
        None
        """
        cfg = self._config

        # --- complexity thresholds ---
        thresh_str = os.getenv("LLM_ROUTER_ROUTING_COMPLEXITY_THRESHOLDS", "")
        if thresh_str:
            parts = thresh_str.split("|")
            if len(parts) == 2:
                try:
                    self._complexity_thresholds = [
                        int(parts[0]),
                        int(parts[1]),
                    ]
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
                category = key[len("LLM_ROUTER_ROUTING_INTENT_") :]
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

        Parameters
        ----------
        payload : dict
            The incoming message payload.

        Returns
        -------
        dict
            The modified payload with ``"model"`` set to the selected model
            name.  If the input model is not ``"auto"``, the payload is
            returned unchanged.

        Raises
        ------
        None
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

    # ------ text extraction

    @staticmethod
    def _get_text_from_payload(payload: Dict[str, Any]) -> str:
        """
        Extract user text from the payload.

        Order of preference: ``messages[-1].content`` > ``user_last_statement``
        > ``query`` > ``prompt`` > ``input``.

        Parameters
        ----------
        payload : dict
            The message payload to extract text from.

        Returns
        -------
        str
            The extracted text, or an empty string if no text is found.

        Raises
        ------
        None
        """
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

    # ------ token estimation

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        Rough token count estimate: word count * 1.25.

        Parameters
        ----------
        text : str
            The input text to estimate token count for.

        Returns
        -------
        int
            An approximate token count.

        Raises
        ------
        None
        """
        word_count = len(text.split()) if text else 0
        return int(word_count * 1.25)

    # ------ complexity

    def _complexity_level(self, token_estimate: int) -> str:
        """
        Classify text complexity from token estimate.

        Parameters
        ----------
        token_estimate : int
            The estimated token count (from :meth:`_estimate_tokens`).

        Returns
        -------
        str
            One of ``"simple"``, ``"medium"``, or ``"complex"``.

        Raises
        ------
        None
        """
        if token_estimate <= self._complexity_thresholds[0]:
            return "simple"
        if token_estimate <= self._complexity_thresholds[1]:
            return "medium"
        return "complex"

    # ------ intent classification

    def _classify_intent(self, text: str) -> str:
        """
        Classify user intent by scoring keywords, phrases, and patterns.

        Each intent category accumulates a score from three sources:

        1. **Keywords** — case-insensitive substring match with optional weight
           (default 1).
        2. **Phrases** — multi-word expression with optional weight (``"phrase:5"``);
           default weight 2.0.
        3. **Patterns** — regex match; each match adds **3.0**.

        Parameters
        ----------
        text : str
            The user input text to classify.

        Returns
        -------
        str
            The name of the highest-scoring intent category, or ``"none"``.

        Raises
        ------
        None
        """
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

    # ------ model selection

    def _select_model(self, intent: str, complexity: str) -> str:
        """
        Select the model from the configured pool.

        Parameters
        ----------
        intent : str
            Classified intent (e.g. ``"code"``, ``"math"``, ``"none"``).
        complexity : str
            Complexity level (``"simple"``, ``"medium"``, ``"complex"``).

        Returns
        -------
        str
            Model name from the pool.

        Raises
        ------
        None
        """
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
        """
        Map a complexity string to a model pool index via ``default_models``.

        Parameters
        ----------
        target : str
            Complexity string to look up (e.g. ``"medium"``).

        Returns
        -------
        int
            The model pool index corresponding to the target complexity.

        Raises
        ------
        None
        """
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
        """
        Look up the target complexity for an intent from ``intent_adjustment``.

        Parameters
        ----------
        intent : str
            The classified intent name (e.g. ``"code"``).

        Returns
        -------
        int
            The target model pool index, or ``-1`` if no adjustment is defined.

        Raises
        ------
        None
        """
        target = self._config.intent_adjustment.get(intent, "")
        if not target:
            return -1
        return self._complexity_to_model_index(target)
