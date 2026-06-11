"""
SemanticBiEncoderRoutingPlugin — embedding-based model routing.

Uses the **radlab/semantic-euro-bert-encoder-v1** BiEncoder from HuggingFace
to compute semantic embeddings for a set of pre-configured routing targets.
For each incoming user message the plugin finds the best-matching target via
cosine similarity and selects the associated model.

Configuration is loaded from
``llm_router_plugins/resources/routing/semantic/semantic_biencoder.json``
and can be overridden by environment variables:

    LLM_ROUTER_ROUTING_SEMANTIC_EURO_MODEL   - override the embedding model name
    LLM_ROUTER_ROUTING_SEMANTIC_EURO_TARGETS - pipe-separated list of target names
    LLM_ROUTER_ROUTING_SEMANTIC_EURO_CHUNK_SIZE    - override chunk size
    LLM_ROUTER_ROUTING_SEMANTIC_EURO_CHUNK_OVERLAP - override chunk overlap

Example JSON configuration::

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
          "examples": ["Write a Python function...", ...]
        }
      ]
    }
"""

import logging
import os
from typing import Any, Dict, Optional

from llm_router_plugins.plugin_interface import PluginInterface
from llm_router_plugins.utils.routing.semantic_biencoder.config import (
    SemanticBiEncoderConfig,
)
from llm_router_plugins.utils.routing.semantic_biencoder.embedder import (
    EmbeddingRouter,
)


class SemanticBiEncoderRoutingPlugin(PluginInterface):
    """
    Embedding-based semantic routing plugin.

    When ``payload["model"] == "auto"`` the plugin embeds the last user
    message and selects the nearest semantic target (and its associated model)
    using cosine similarity.
    """

    name = "semantic_biencoder_routing"

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        super().__init__(logger=logger)

        self._config = SemanticBiEncoderConfig.from_file()
        self._router = EmbeddingRouter(self._config, logger=self._logger)

        # Environment overrides
        self._override_from_env()
        self._router.initialize()

    def _override_from_env(self) -> None:
        """Apply environment variable overrides to config."""
        model_env = os.getenv("LLM_ROUTER_ROUTING_SEMANTIC_EURO_MODEL")
        if model_env:
            self._config.embedding_model = model_env
            if self._logger:
                self._logger.info(
                    "Overriding embedding model: %s", model_env
                )

        targets_env = os.getenv("LLM_ROUTER_ROUTING_SEMANTIC_EURO_TARGETS")
        if targets_env:
            allowed = set(self._config.target_names)
            selected = [
                t.strip()
                for t in targets_env.split("|")
                if t.strip() and t.strip() in allowed
            ]
            if selected and selected != list(self._config.target_names):
                filtered = [
                    t for t in self._config.routing_targets if t.name in selected
                ]
                self._config.routing_targets = filtered
                if self._logger:
                    self._logger.info(
                        "Overriding routing targets: %s",
                        "|".join(selected),
                    )

        chunk_size_env = os.getenv("LLM_ROUTER_ROUTING_SEMANTIC_EURO_CHUNK_SIZE")
        if chunk_size_env:
            try:
                self._config.chunk_size = int(chunk_size_env)
            except ValueError:
                pass

        chunk_overlap_env = os.getenv(
            "LLM_ROUTER_ROUTING_SEMANTIC_EURO_CHUNK_OVERLAP"
        )
        if chunk_overlap_env:
            try:
                self._config.chunk_overlap = int(chunk_overlap_env)
            except ValueError:
                pass

    def apply(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process *payload*.  If ``payload["model"] == "auto"`` route to the
        best-matching model via semantic similarity.
        """
        if payload.get("model") != "auto":
            return payload

        text = self._get_text_from_payload(payload)
        if not text:
            if self._logger:
                self._logger.warning(
                    "SemanticBiEncoderRouting: no text content found, returning payload unchanged."
                )
            return payload

        result = self._router.route(text)

        print("Semantic bi-encoder routing b " * 3)
        print(result)
        print("Semantic bi-encoder routing e " * 3)

        payload["model"] = result["model_name"]
        payload["routing"] = {
            "plugin": self.name,
            "target_name": result["target_name"],
            "similarity": result["similarity"],
        }

        if self._logger:
            self._logger.info(
                "SemanticBiEncoderRouting: text='%s' target='%s' similarity=%.4f -> model=%s",
                text[:80],
                result["target_name"],
                result["similarity"],
                result["model_name"],
            )

        return payload

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
