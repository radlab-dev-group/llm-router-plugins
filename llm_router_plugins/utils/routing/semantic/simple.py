"""
Default semantic routing plugin.

Performs two-stage heuristic routing: intent classification + complexity
analysis to select the best model when ``payload["model"] == "auto"``.

All configuration comes from environment variables prefixed with
``LLM_ROUTER_ROUTING_`` (see ``_load_config`` for the full list).
"""

import logging
import os
from typing import Dict, List, Optional

from llm_router_plugins.plugin_interface import PluginInterface


_DEFAULT_MODELS = ("gpt-oss:120b", "qwen3.6:35b")

_DEFAULT_INTENT_CATEGORIES = {
    "code": [
        "code",
        "program",
        "function",
        "class",
        "algorithm",
        "debug",
        "kod",
        "kodu",
        "kodem",
        "kodzie",
        "program",
        "programu",
        "programem",
        "programie",
        "funkcja",
        "funkcji",
        "funkcję",
        "funkcją",
        "funkcje",
        "klasa",
        "klasy",
        "klasę",
        "klasą",
        "klasie",
        "algorytm",
        "algorytmu",
        "algorytmem",
        "algorytmie",
        "algorytmy",
        "debug",
        "debuguj",
        "debugowanie",
        "debugowania",
        "debugowaniem",
        "błąd",
        "błędu",
        "błędem",
        "błędzie",
        "błędy",
        "napraw",
        "naprawić",
        "popraw",
        "poprawić",
        "zaimplementuj",
        "implementacja",
        "implementacji",
        "python",
        "pythonie",
        "css",
        "js",
        "c++",
        "java",
        "ts",
    ],
    "math": [
        "math",
        "calculate",
        "equation",
        "formula",
        "probability",
        "statistics",
        "matematyka",
        "matematyki",
        "matematyką",
        "matematyce",
        "oblicz",
        "obliczyć",
        "obliczenia",
        "obliczeń",
        "policz",
        "policzyć",
        "równanie",
        "równania",
        "równaniu",
        "równaniem",
        "wzór",
        "wzoru",
        "wzorem",
        "wzorze",
        "prawdopodobieństwo",
        "prawdopodobieństwa",
        "prawdopodobieństwem",
        "prawdopodobieństwie",
        "statystyka",
        "statystyki",
        "statystyką",
        "statystyce",
        "średnia",
        "średniej",
        "medianę",
        "mediana",
        "wariancja",
        "wariancji",
    ],
    "creative": [
        "write",
        "creative",
        "story",
        "poem",
        "email",
        "draft",
        "napisz",
        "napisać",
        "pisz",
        "kreatywny",
        "kreatywna",
        "kreatywne",
        "kreatywnie",
        "historia",
        "historii",
        "historię",
        "historią",
        "opowiadanie",
        "opowiadania",
        "opowiadaniem",
        "wiersz",
        "wiersza",
        "wierszem",
        "wierszu",
        "poemat",
        "poematu",
        "email",
        "e-mail",
        "mail",
        "maila",
        "mailem",
        "szkic",
        "szkicu",
        "szkicem",
        "projekt",
        "projektu",
        "projektem",
        "zredaguj",
        "redakcja",
        "redakcji",
    ],
    "general": [
        "help",
        "what",
        "how",
        "why",
        "explain",
        "pomoc",
        "pomocy",
        "pomóż",
        "pomocą",
        "co",
        "czym",
        "czego",
        "jak",
        "jaki",
        "jaka",
        "jakie",
        "jakiego",
        "jakiej",
        "dlaczego",
        "czemu",
        "wyjaśnij",
        "wyjaśnić",
        "wytłumacz",
        "wytłumaczyć",
        "opisz",
        "opisać",
        "powiedz",
        "powiedzieć",
        "przedstaw",
        "przedstawić",
    ],
    "none": [],
}

# (simple_max, medium_max) in tokens which are calculated as
# number_of_words * 1.25
_DEFAULT_COMPLEXITY_THRESHOLDS = (25, 150)


class DefaultSemanticRoutingPlugin(PluginInterface):
    """
    Semantic routing plugin that selects a model from a configured pool
    based on the intent and complexity of the user's input.

    When ``payload["model"] == "auto"``, the plugin:

    1. Classifies the input intent via keyword scoring.
    2. Estimates complexity via token count.
    3. Combines both signals to pick a model index.

    All other cases pass the payload through unchanged.

    Environment variables (all optional):

    * ``LLM_ROUTER_ROUTING_MODELS`` – pipe-separated model names
      (first = cheapest, last = strongest).
    * ``LLM_ROUTER_ROUTING_INTENT_<CATEGORY>`` – pipe-separated trigger
      keywords per intent category.
    * ``LLM_ROUTER_ROUTING_COMPLEXITY_THRESHOLDS`` – pipe-separated
      ``[simple_max|medium_max]`` in token counts.
    * ``LLM_ROUTER_ROUTING_DEFAULT_MODEL`` – fallback model name.
    """

    name = "simple_semantic_routing"

    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger=logger)
        self._MODELS: List[str] = []
        self._INTENT_CATEGORIES: dict = {}
        self._COMPLEXITY_THRESHOLDS: List[int] = []
        self._DEFAULT_MODEL: str = ""
        self._load_config()

    # ------------------------------------------------------------------ config
    def _load_config(self) -> None:
        # --- models ---
        models_str = os.getenv("LLM_ROUTER_ROUTING_MODELS", "")
        if models_str:
            self._MODELS = [m.strip() for m in models_str.split("|") if m.strip()]
        if not self._MODELS:
            self._MODELS = list(_DEFAULT_MODELS)

        # --- intent categories: pick up any LLM_ROUTER_ROUTING_INTENT_* env vars ---
        intents: dict = {}
        for key, value in os.environ.items():
            if key.startswith("LLM_ROUTER_ROUTING_INTENT_"):
                category = key[len("LLM_ROUTER_ROUTING_INTENT_") :]
                keywords = [kw.strip() for kw in value.split("|") if kw.strip()]
                if keywords:
                    intents[category.lower()] = keywords
        # Fallback to hard-coded defaults only when nothing was provided via env.
        self._INTENT_CATEGORIES = (
            intents if intents else dict(_DEFAULT_INTENT_CATEGORIES)
        )

        # --- complexity thresholds ---
        thresh_str = os.getenv("LLM_ROUTER_ROUTING_COMPLEXITY_THRESHOLDS", "")
        if thresh_str:
            parts = thresh_str.split("|")
            if len(parts) == 2:
                try:
                    self._COMPLEXITY_THRESHOLDS = [int(parts[0]), int(parts[1])]
                except ValueError:
                    if self._logger:
                        self._logger.warning(
                            "Malformed LLM_ROUTER_ROUTING_COMPLEXITY_THRESHOLDS, "
                            "using defaults"
                        )
                    self._COMPLEXITY_THRESHOLDS = list(
                        _DEFAULT_COMPLEXITY_THRESHOLDS
                    )
            else:
                if self._logger:
                    self._logger.warning(
                        "LLM_ROUTER_ROUTING_COMPLEXITY_THRESHOLDS must have 2 values, "
                        "using defaults"
                    )
                self._COMPLEXITY_THRESHOLDS = list(_DEFAULT_COMPLEXITY_THRESHOLDS)
        else:
            self._COMPLEXITY_THRESHOLDS = list(_DEFAULT_COMPLEXITY_THRESHOLDS)

        # --- default model ---
        self._DEFAULT_MODEL = os.getenv(
            "LLM_ROUTER_ROUTING_DEFAULT_MODEL", _DEFAULT_MODELS[0]
        )

    # ---------------------------------------------------------------- routing
    def apply(self, payload: Dict) -> Dict:
        if payload.get("model") != "auto":
            return payload

        text = self._get_text_from_payload(payload)
        if not text:
            payload["model"] = self._DEFAULT_MODEL
            if self._logger:
                self._logger.warning(
                    "No text content found, using default model %s",
                    self._DEFAULT_MODEL,
                )
            return payload

        intent = self._classify_intent(text)
        token_count = self._count_tokens(text)
        complexity = self._complexity_level(token_count)
        payload["model"] = self._select_model(intent, complexity)

        if self._logger:
            self._logger.info(
                "Semantic routing: intent=%s, complexity=%s (%d tokens) -> %s",
                intent,
                complexity,
                token_count,
                payload["model"],
            )
        return payload

    def _get_text_from_payload(self, payload: Dict) -> str:
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

    def _count_tokens(self, text: str) -> int:
        word_count = len(text.split()) if text else 0
        return int(word_count * 1.25)

    def _complexity_level(self, token_count: int) -> str:
        if token_count <= self._COMPLEXITY_THRESHOLDS[0]:
            return "simple"
        if token_count <= self._COMPLEXITY_THRESHOLDS[1]:
            return "medium"
        return "complex"

    def _classify_intent(self, text: str) -> str:
        text_lower = text.lower()
        best_category = "none"
        best_score = 0

        for category, keywords in self._INTENT_CATEGORIES.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            if score > best_score:
                best_score = score
                best_category = category

        return best_category

    def _select_model(self, intent: str, complexity: str) -> str:
        n = len(self._MODELS)

        complexity_map = {"simple": 0, "medium": n // 2, "complex": n - 1}
        idx = complexity_map.get(complexity, 0)

        intent_adjustment = {
            "code": 1,
            "math": 1,
            "creative": 0,
            "general": 0,
            "none": -1,
        }
        idx += intent_adjustment.get(intent, 0)

        idx = max(0, min(idx, n - 1))
        return self._MODELS[idx]
