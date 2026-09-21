"""
Semantic (embedding + vector store) layer of the Agentic routing cascade.

This module owns the *probabilistic* half of mode detection.  The deterministic
layers (explicit mode, declarative rules, session affinity, keyword scoring)
run first; only when none of them answers does the plugin ask the vector
store, which requires an embedding model and a FAISS index.

The layer is deliberately thin: it wraps a router produced by
:func:`llm_router_plugins.utils.routing.common.build_embedding_router`, turns
its raw result into an :class:`AgentMode`, and decides whether the similarity
is good enough to be trusted.  Keeping it isolated means the deterministic
part of the plugin can be imported, configured and tested without any ML
dependency installed.
"""

import logging

from typing import Any, Mapping, Optional, Tuple

from llm_router_plugins.utils.routing.agentic_routing.general.config import AgentMode

__all__ = ["SemanticLayer"]


class SemanticLayer:
    """
    Similarity-based mode lookup over an already built router.

    A layer without a router is *unavailable* rather than broken: every
    resolution attempt simply returns ``None`` so that the cascade can continue
    unchanged.  This keeps the plugin usable when ``semantic.enabled`` is
    ``false`` or when the optional dependencies are missing.

    Parameters
    ----------
    router : Any
        Initialized router exposing ``route(text)``, or ``None`` when semantic
        routing is not available.
    threshold : float
        Minimum cosine similarity accepted for a match.
    mode_by_name : Mapping[str, AgentMode]
        Lookup table used to translate the router target name into a mode.
    logger : logging.Logger, optional
        Logger instance.  If ``None``, logging is skipped.

    Raises
    ------
    None
    """

    def __init__(
        self,
        router: Optional[Any],
        threshold: float,
        mode_by_name: Mapping[str, AgentMode],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Store the router, acceptance threshold and mode lookup table.

        Parameters
        ----------
        router : Any
            Router exposing ``route(text)``, or ``None``.
        threshold : float
            Minimum similarity accepted for a match.
        mode_by_name : Mapping[str, AgentMode]
            Known modes indexed by name.
        logger : logging.Logger, optional
            Logger instance.  If ``None``, logging is skipped.

        Returns
        -------
        None

        Raises
        ------
        None
        """
        self._router = router
        self._threshold = float(threshold)
        self._mode_by_name = mode_by_name
        self._logger = logger

    @property
    def available(self) -> bool:
        """
        Report whether a semantic lookup is possible at all.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            ``True`` when a router was injected, ``False`` otherwise.

        Raises
        ------
        None
        """
        return self._router is not None

    def resolve(self, text: str) -> Tuple[Optional[AgentMode], float]:
        """
        Resolve *text* to a configured mode through the vector store.

        Parameters
        ----------
        text : str
            The request text.  Empty text never reaches the router.

        Returns
        -------
        Tuple[Optional[AgentMode], float]
            The accepted mode and its similarity, or ``(None, similarity)``
            when the router produced nothing usable — no target, an unknown
            target, or a similarity below the configured threshold.

        Raises
        ------
        None
        """
        if self._router is None or not text:
            return None, 0.0

        result = self._router.route(text)
        similarity = float(result.get("similarity", 0.0) or 0.0)
        target = str(result.get("target_name", "") or "")
        mode = self._mode_by_name.get(target)

        if mode is not None and similarity >= self._threshold:
            return mode, similarity

        if self._logger:
            self._logger.info(
                "AgenticRouting: semantic match '%s' similarity=%.4f is "
                "below threshold %.4f, ignoring it",
                target or "unknown",
                similarity,
                self._threshold,
            )
        return None, similarity
