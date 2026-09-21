"""
Backward-compatibility shim.

The shared :class:`EmbeddingRouter` now lives in
:mod:`llm_router_plugins.utils.routing.embedder` (used by both the
``semantic_biencoder`` and the ``agentic_routing`` plugins).  This module
re-exports it so that existing imports of
``llm_router_plugins.utils.routing.semantic_biencoder.embedder`` keep working.
"""

from llm_router_plugins.utils.routing.embedder import (  # noqa: F401
    EmbeddingRouter,
    EmbeddingRouterConfig,
)

__all__ = ["EmbeddingRouter", "EmbeddingRouterConfig"]
