"""
SemanticBiEncoderRoutingPlugin — embedding-based model routing.

Uses a BiEncoder embedding model (e.g. ``google/embeddinggemma-300m``) from
HuggingFace to compute semantic embeddings for a set of pre-configured routing
targets.  For each incoming user message the plugin finds the best-matching
target via cosine similarity and selects the associated model.

Configuration is loaded from
``llm_router_plugins/resources/routing/semantic_biencoder.json``
and can be overridden by environment variables:

    LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_CONFIG
        - full configuration as a raw JSON string or a path to a config file
    LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_MODEL
        - override the embedding model name
    LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_TARGETS
        - pipe-separated list of target names
    LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_CHUNK_SIZE
        - override chunk size
    LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_CHUNK_OVERLAP
        - override chunk overlap
    LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_PERSIST_DIR
        - directory for FAISS index persistence

Example JSON configuration::

    {
      "embedding_model": "google/embeddinggemma-300m",
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

import os
import logging

from dataclasses import replace
from typing import Any, Dict, Optional

from llm_router_plugins.plugin_interface import PluginInterface
from llm_router_plugins.utils.text_extractor import extract_user_text
from llm_router_plugins.utils.routing.common import (
    annotate_routing,
    build_embedding_router,
    env_int,
    resolve_persist_dir,
    should_route,
)
from llm_router_plugins.utils.routing.semantic_biencoder.config import (
    SemanticBiEncoderConfig,
)
from llm_router_plugins.utils.routing.constants import (
    SEMANTIC_BIENCODER_ROUTING_PREFIX,
)

_MISSING_DEPENDENCIES_MESSAGE = (
    "SemanticBiEncoderRouting: the sentence-transformers / FAISS dependencies "
    "are not installed — install them to enable semantic routing"
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
        Initialize the plugin: load config, apply env overrides, build the
        FAISS index.

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
            If no routing targets are defined in the configuration, the
            parameters are invalid, the ML dependencies are missing, or the
            router produced an empty index.
        """
        super().__init__(logger=logger)

        self._config = SemanticBiEncoderConfig.from_file()
        self._override_from_env()
        self._validate_args()

        persist_dir = resolve_persist_dir(
            SEMANTIC_BIENCODER_ROUTING_PREFIX,
            self._config.vector_store_path,
            logger=self._logger,
        )
        self._router = build_embedding_router(
            embedding_model=self._config.embedding_model,
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
            top_k=self._config.top_k,
            routing_targets=self._config.routing_targets,
            logger=self._logger,
            persist_dir=persist_dir,
            missing_deps_hint=_MISSING_DEPENDENCIES_MESSAGE,
        )

    def _validate_args(self) -> None:
        """
        Validate that all required routing configuration is present
        after env overrides and router initialization.

        Raises
        ------
        ValueError
            If required config is missing, empty, or router failed to load vectors.
        """
        if self._config.embedding_model is None or not self._config.embedding_model:
            raise ValueError(
                "SemanticBiEncoderRouting: no embedding_model configured — "
                "check 'embedding_model' in the JSON config or the "
                f"{SEMANTIC_BIENCODER_ROUTING_PREFIX}MODEL environment variable"
            )

        if not self._config.routing_targets:
            raise ValueError(
                "SemanticBiEncoderRouting: no routing targets defined — "
                "check 'routing_targets' in the JSON config or the "
                f"{SEMANTIC_BIENCODER_ROUTING_PREFIX}TARGETS environment variable"
            )

        if self._config.chunk_size <= 0:
            raise ValueError(
                "SemanticBiEncoderRouting: chunk_size must be > 0, "
                f"got {self._config.chunk_size}"
            )

        if self._config.chunk_overlap < 0:
            raise ValueError(
                "SemanticBiEncoderRouting: chunk_overlap must be >= 0, "
                f"got {self._config.chunk_overlap}"
            )

    def apply(self, payload: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """
        Process *payload*.  If ``payload["model"] == "auto"`` route to the
        best-matching model via semantic similarity.

        The last user message is extracted from the payload (via ``messages``,
        ``user_last_statement``, ``query``, or ``prompt``), embedded, and
        matched against the FAISS index.  A match is only accepted when its
        similarity is greater than or equal to the configured
        ``similarity_threshold``.  The result replaces ``payload["model"]``
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
        if not should_route(payload, ("auto",)):
            return payload

        text = extract_user_text(payload)
        if not text:
            if self._logger:
                self._logger.warning(
                    "SemanticBiEncoderRouting: no text content "
                    "found, returning payload unchanged."
                )
            return payload

        result = self._router.route(text)
        similarity = float(result["similarity"])

        if similarity < self._config.similarity_threshold:
            if self._logger:
                self._logger.info(
                    "SemanticBiEncoderRouting: text='%s' target='%s' "
                    "similarity=%.4f is below threshold %.4f — "
                    "leaving model unchanged",
                    text[:80],
                    result["target_name"],
                    similarity,
                    self._config.similarity_threshold,
                )
            return payload

        annotate_routing(
            payload,
            self.name,
            result["model_name"],
            similarity,
            target_name=result["target_name"],
        )

        if self._logger:
            self._logger.info(
                "SemanticBiEncoderRouting: text='%s' "
                "target='%s' similarity=%.4f -> model=%s",
                text[:80],
                result["target_name"],
                similarity,
                result["model_name"],
            )

        return payload

    def _override_from_env(self) -> None:
        """
        Apply environment variable overrides to config, creating a **new**
        ``SemanticBiEncoderConfig`` so the immutable-snapshot contract is
        preserved.

        Supported environment variables (prefix
        ``LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_``):

        - ``MODEL`` — override the embedding model name
        - ``TARGETS`` — pipe-separated target name whitelist (targets not
          in the config file are silently ignored)
        - ``CHUNK_SIZE`` — override the chunk size used for embedding
        - ``CHUNK_OVERLAP`` — override the chunk overlap used for embedding
        - ``PERSIST_DIR`` — directory for FAISS index persistence

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
        original = self._config
        overrides: Dict[str, Any] = {}

        model_env = os.getenv(f"{SEMANTIC_BIENCODER_ROUTING_PREFIX}MODEL")
        if model_env:
            overrides["embedding_model"] = model_env
            if self._logger:
                self._logger.info("Overriding embedding model: %s", model_env)

        targets_env = os.getenv(f"{SEMANTIC_BIENCODER_ROUTING_PREFIX}TARGETS")
        if targets_env:
            allowed = set(original.target_names)
            selected = [
                t.strip()
                for t in targets_env.split("|")
                if t.strip() and t.strip() in allowed
            ]
            if selected and selected != list(original.target_names):
                overrides["routing_targets"] = tuple(
                    t for t in original.routing_targets if t.name in selected
                )
                if self._logger:
                    self._logger.info(
                        "Overriding routing targets: %s",
                        "|".join(selected),
                    )

        chunk_size = env_int(SEMANTIC_BIENCODER_ROUTING_PREFIX, "CHUNK_SIZE")
        if chunk_size is not None:
            overrides["chunk_size"] = chunk_size

        chunk_overlap = env_int(SEMANTIC_BIENCODER_ROUTING_PREFIX, "CHUNK_OVERLAP")
        if chunk_overlap is not None:
            overrides["chunk_overlap"] = chunk_overlap

        if overrides:
            self._config = replace(original, **overrides)
