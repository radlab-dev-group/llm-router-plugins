"""
SemanticBiEncoderRoutingPlugin — embedding-based model routing.

Uses the **radlab/semantic-euro-bert-encoder-v1** BiEncoder from HuggingFace
to compute semantic embeddings for a set of pre-configured routing targets.
For each incoming user message the plugin finds the best-matching target via
cosine similarity and selects the associated model.

Configuration is loaded from
``llm_router_plugins/resources/routing/semantic/semantic_biencoder.json``
and can be overridden by environment variables:

    LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_MODEL   - override the embedding model name
    LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_TARGETS - pipe-separated list of target names
    LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_CHUNK_SIZE    - override chunk size
    LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_CHUNK_OVERLAP - override chunk overlap
    LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_PERSIST_DIR   - directory for FAISS index persistence

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

    Attributes
    ----------
    name : str
        Plugin identifier (``"semantic_biencoder_routing"``).
    """

    name = "semantic_biencoder_routing"

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize the plugin: load config, resolve persist dir, build FAISS index.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger instance. If ``None``, a default logger is used internally.

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist at the expected path.
        KeyError
            If the configuration file is missing required fields.
        ValueError
            If no routing targets are defined in the configuration.
        """
        super().__init__(logger=logger)

        self._config = SemanticBiEncoderConfig.from_file()
        persist_dir = self._config.vector_store_path
        env_persist = os.getenv("LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_PERSIST_DIR")
        if env_persist:
            persist_dir = env_persist
            if self._logger:
                self._logger.info(
                    "Overriding vector store path: %s",
                    persist_dir,
                )

        self._router = EmbeddingRouter(
            self._config,
            logger=self._logger,
            persist_dir=persist_dir,
        )

        # Environment overrides
        self._override_from_env()
        self._router.initialize()

    def _override_from_env(self) -> None:
        """
        Apply environment variable overrides to config.

        Supported environment variables (prefix
        ``LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_``):

        - ``MODEL`` — override the embedding model name
        - ``TARGETS`` — pipe-separated target name whitelist (targets not
          in the config file are silently ignored)
        - ``CHUNK_SIZE`` — override the chunk size used for embedding
        - ``CHUNK_OVERLAP`` — override the chunk overlap used for embedding

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        None
        """
        model_env = os.getenv("LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_MODEL")
        if model_env:
            self._config.embedding_model = model_env
            if self._logger:
                self._logger.info("Overriding embedding model: %s", model_env)

        targets_env = os.getenv("LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_TARGETS")
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

        chunk_size_env = os.getenv(
            "LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_CHUNK_SIZE"
        )
        if chunk_size_env:
            try:
                self._config.chunk_size = int(chunk_size_env)
            except ValueError:
                pass

        chunk_overlap_env = os.getenv(
            "LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_CHUNK_OVERLAP"
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

        The last user message is extracted from the payload (via ``messages``,
        ``user_last_statement``, ``query``, or ``prompt``), embedded, and
        matched against the FAISS index.  The result replaces ``payload["model"]``
        with the selected model name and adds a ``"routing"`` metadata dict.

        Parameters
        ----------
        payload : dict
            The incoming message payload containing at least the ``"model"``
            key (set to ``"auto"`` for routing to activate) and either
            ``"messages"`` or ``"query"``.

        Returns
        -------
        dict
            The modified payload. If ``payload["model"] != "auto"`` the
            payload is returned unchanged.  If routing occurs,
            ``payload["model"]`` is set to the selected model name and
            ``payload["routing"]`` is added with:

            - ``"plugin"`` (str): ``"semantic_biencoder_routing"``
            - ``"target_name"`` (str): name of the matched target
            - ``"similarity"`` (float): mean cosine similarity score

        Raises
        ------
        None
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
        """
        Extract the user message text from *payload*.

        The text is extracted using the following priority:

        1. ``payload["messages"][-1]["content"]`` (last message in a chat history)
        2. ``payload["user_last_statement"]``
        3. ``payload["query"]``
        4. ``payload["prompt"]``
        5. ``payload["input"]``

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
