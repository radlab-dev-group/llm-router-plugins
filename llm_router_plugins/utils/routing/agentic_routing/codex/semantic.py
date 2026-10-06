"""
Semantic (embedding + vector store) layer of the Codex routing cascade.

This module owns the *probabilistic* half of work-mode detection.  The
deterministic layers (explicit mode, request class, collaboration block,
keyword scoring) run first; only when none of them answers does the plugin ask
the vector store, which requires an embedding model and a FAISS index.

The layer is deliberately thin: it wraps a router produced by
:func:`llm_router_plugins.utils.routing.common.build_embedding_router`, turns
its raw result into a :class:`CodexMode`, and decides whether the cosine
similarity is good enough to be trusted.  Keeping it isolated means the
deterministic part of the plugin can be imported, configured and tested
without any ML dependency installed.

A router is queried only after the deterministic layers have stayed silent, and
then at most once per request: the same lookup accepts a semantic match and
reports the cosine of the fallback mode, so a single request never embeds the
same text twice.
"""

import logging

from typing import Any, Dict, Mapping, Optional, Tuple

from llm_router_plugins.utils.routing.agentic_routing.codex.payload import (
    CodexRequest,
)

from llm_router_plugins.utils.routing.agentic_routing.codex.config import CodexMode

__all__ = ["CodexSemanticLayer"]


class CodexSemanticLayer:
    """
    Cosine-similarity mode lookup over an already built embedding router.

    A layer without a router is *unavailable* rather than broken: every
    resolution attempt simply returns nothing so that the cascade can continue
    unchanged.  This keeps the plugin usable when ``semantic.enabled`` is
    ``false``, when the optional dependencies are missing or when the embedding
    model cannot be loaded — routing then stays purely deterministic.

    Parameters
    ----------
    router : Any
        Initialized router exposing ``route(text)``, or ``None`` when semantic
        routing is not available.
    threshold : float
        Minimum cosine similarity accepted for a semantic match.
    mode_by_name : Mapping[str, CodexMode]
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
        mode_by_name: Mapping[str, CodexMode],
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        Store the router, acceptance threshold and mode lookup table.

        Parameters
        ----------
        router : Any
            Router exposing ``route(text)``, or ``None``.
        threshold : float
            Minimum cosine similarity accepted for a match.
        mode_by_name : Mapping[str, CodexMode]
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

    def route(self, request: CodexRequest) -> Optional[Dict[str, Any]]:
        """
        Query the vector store once, translating failures into ``None``.

        The query is the request's semantic context
        (:meth:`_build_semantic_context`): ``latest_user_text`` followed by the
        agent's last utterance, so the embedding sees what the agent is about
        to work on, not the whole history of the thread.  When that context is
        empty the router is never called.  Anything that raises while building
        the context or querying the router is translated into ``None``, so a
        broken request only loses its semantic answer, never the deterministic
        cascade.

        Parameters
        ----------
        request : CodexRequest
            The parsed request whose context is embedded.

        Returns
        -------
        Optional[Dict[str, Any]]
            The raw router result, or ``None`` when no lookup was possible or
            the lookup failed.

        Raises
        ------
        None
        """
        if self._router is None or not request:
            return None

        text = None
        try:
            text = self._build_semantic_context(request)
        except Exception as exc:  # never let context building break routing
            self._warn("CodexRouting: context building failed, ignoring it: %s", exc)
            return None

        if not text or not text.strip():
            return None

        self._info("CodexRouting: text used to route: %s", text)

        try:
            result = self._router.route(text)
        except Exception as exc:
            self._warn("CodexRouting: semantic lookup failed, ignoring it: %s", exc)
            return None

        return result if isinstance(result, dict) else None

    def accept(
        self, result: Optional[Mapping[str, Any]]
    ) -> Tuple[Optional[CodexMode], float]:
        """
        Decide whether a router result names a mode confidently enough.

        Parameters
        ----------
        result : Optional[Mapping[str, Any]]
            A router result, or ``None`` when no lookup happened.

        Returns
        -------
        Tuple[Optional[CodexMode], float]
            The accepted mode and its similarity, or ``(None, similarity)``
            when the router produced nothing usable — no target, an unknown
            target, or a similarity below the configured threshold.

        Raises
        ------
        None
        """
        if not result:
            return None, 0.0

        similarity = float(result.get("similarity", 0.0) or 0.0)
        target = str(result.get("target_name", "") or "")
        mode = self._mode_by_name.get(target)

        if mode is not None and similarity >= self._threshold:
            return mode, similarity

        if self._logger is not None:
            if mode is None:
                self._logger.info(
                    "CodexRouting: semantic target '%s' is not a configured "
                    "mode, ignoring it",
                    target or "unknown",
                )
            else:
                self._logger.info(
                    "CodexRouting: semantic match '%s' similarity=%.4f is "
                    "below threshold %.4f, ignoring it",
                    target,
                    similarity,
                    self._threshold,
                )
        return None, similarity

    def resolve(self, request: CodexRequest) -> Tuple[Optional[CodexMode], float]:
        """
        Resolve *request* to a configured mode through the vector store.

        Parameters
        ----------
        request : CodexRequest
            The parsed request.  An empty context never reaches the router.

        Returns
        -------
        Tuple[Optional[CodexMode], float]
            The accepted mode and its similarity, or ``(None, similarity)``
            when the router produced nothing usable.

        Raises
        ------
        None
        """
        return self.accept(self.route(request))

    def similarity_for(
        self,
        mode_name: str,
        result: Optional[Mapping[str, Any]],
    ) -> Optional[float]:
        """
        Return the cosine similarity reported for *mode_name* in *result*.

        Used to report an embedding-based confidence for a mode the semantic
        layer did not win: the router scores every indexed mode in
        ``all_scores``, so the similarity of the fallback mode is read from
        there instead of reporting a bare ``0.0``.

        Parameters
        ----------
        mode_name : str
            Name of the mode whose similarity is requested.
        result : Optional[Mapping[str, Any]]
            A router result, or ``None`` when no lookup happened.

        Returns
        -------
        Optional[float]
            The cosine similarity in ``[0.0, 1.0]``, or ``None`` when the
            result does not mention the mode at all.

        Raises
        ------
        None
        """
        if self._router is None or not result or not mode_name:
            return None

        for entry in result.get("all_scores", []) or []:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("target", "")) == mode_name:
                return float(entry.get("similarity", 0.0) or 0.0)

        if str(result.get("target_name", "") or "") == mode_name:
            return float(result.get("similarity", 0.0) or 0.0)

        return None

    @staticmethod
    def _build_semantic_context(
        request: CodexRequest, last_agent_messages: int = 1
    ) -> Optional[str]:
        """
        Assemble the text embedded for the semantic lookup.

        The newest user text is followed by the text of the last
        *last_agent_messages* assistant ``output_text`` parts (oldest first),
        joined with newlines.  Assistant turns give the embedding model the
        thread context that the one-line user message alone lacks, which is
        what separates a "review this" turn from a "fix this" turn.

        Parameters
        ----------
        request : CodexRequest
            The parsed request.
        last_agent_messages : int
            How many of the newest assistant message parts are appended.

        Returns
        -------
        Optional[str]
            The assembled query, or ``None`` when the request carries no text
            at all (neither user nor assistant).

        Raises
        ------
        None
        """
        _text = request.latest_user_text or ""

        _messages = request.assistant_messages or []
        _assistant_messages = []
        for _msg in _messages:
            for _item in _msg["content"]:
                _assistant_messages.append(_item["text"])

        _text = [_text] + _assistant_messages[-last_agent_messages:]
        if not len(_text):
            return None
        return "\n".join(_text)

    def _warn(self, message: str, *args: Any) -> None:
        """
        Log a warning when a logger is available.

        Parameters
        ----------
        message : str
            The log message, optionally with ``%`` placeholders.
        *args : Any
            Arguments for the ``%`` placeholders.

        Returns
        -------
        None
        """
        if self._logger is not None:
            self._logger.warning(message, *args)

    def _info(self, message: str, *args: Any) -> None:
        """
        Log a info when a logger is available.

        Parameters
        ----------
        message : str
            The log message, optionally with ``%`` placeholders.
        *args : Any
            Arguments for the ``%`` placeholders.

        Returns
        -------
        None
        """
        if self._logger is not None:
            self._logger.info(message, *args)
