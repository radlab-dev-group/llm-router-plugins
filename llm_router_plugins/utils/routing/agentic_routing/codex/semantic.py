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
import math
from numbers import Real

from typing import Any, Dict, Mapping, Optional, Tuple

from llm_router_plugins.utils.routing.agentic_routing.codex.payload import (
    CodexRequest,
    REQUEST_CLASS_AUX_TITLE,
    REQUEST_CLASS_COMPACTION,
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
    min_margin : float
        Required cosine lead over the runner-up; ties always abstain.
    intent_max_chars, phase_max_chars : int
        Independent context-section budgets from the supplied configuration.

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
        *,
        min_margin: float,
        intent_max_chars: int,
        phase_max_chars: int,
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
        self._min_margin = min_margin
        self._intent_max_chars = intent_max_chars
        self._phase_max_chars = phase_max_chars
        self._semantic_modes = {
            name for name in mode_by_name
            if name not in (REQUEST_CLASS_AUX_TITLE, REQUEST_CLASS_COMPACTION)
        }

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

        The query comprises independently budgeted intent and phase sections
        (:meth:`_build_semantic_parts`). The shared router encodes each section
        separately before a single vector-store lookup. When that context is
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

        try:
            parts = self._build_semantic_parts(
                request, self._intent_max_chars, self._phase_max_chars
            )
        except Exception as exc:  # never let context building break routing
            self._warn("CodexRouting: context building failed, ignoring it: %s", exc)
            return None

        if not parts:
            return None

        try:
            route_context = getattr(self._router, "route_context", None)
            result = (
                route_context(parts) if callable(route_context)
                else self._router.route("\n".join(parts))
            )
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
            target, an incomplete ranking, or insufficient similarity/margin.

        Raises
        ------
        None
        """
        if not isinstance(result, Mapping) or not result:
            return None, 0.0

        similarity = self._cosine(result.get("similarity"))
        if similarity is None:
            return None, 0.0
        target = str(result.get("target_name", "") or "")
        mode = self._mode_by_name.get(target)
        entries = result.get("all_scores")
        scores = {}
        if not isinstance(entries, (list, tuple)):
            return None, similarity
        for entry in entries:
            if not isinstance(entry, Mapping):
                return None, similarity
            name = entry.get("target")
            score = self._cosine(entry.get("similarity"))
            if (
                not isinstance(name, str) or name not in self._semantic_modes
                or name in scores or score is None
            ):
                return None, similarity
            scores[name] = score
        if set(scores) != self._semantic_modes or target not in scores:
            self._info("CodexRouting: incomplete semantic ranking, ignoring it")
            return None, similarity
        if not math.isclose(similarity, scores[target], rel_tol=1e-6, abs_tol=1e-7):
            return None, similarity
        runner_up = max(
            (score for name, score in scores.items() if name != target), default=None
        )
        margin = scores[target] - runner_up if runner_up is not None else None
        self._info(
            "CodexRouting: semantic target=%s similarity=%.4f runner_up=%s margin=%s",
            target, similarity, runner_up, margin,
        )
        if similarity < self._threshold:
            self._info(
                "CodexRouting: semantic match '%s' similarity=%.4f is below threshold %.4f",
                target, similarity, self._threshold,
            )
        if mode is not None and similarity >= self._threshold and (
            margin is None or (margin > 0 and (
                margin >= self._min_margin
                or math.isclose(margin, self._min_margin, rel_tol=0, abs_tol=1e-12)
            ))
        ):
            return mode, similarity
        return None, similarity

    @staticmethod
    def _cosine(value: Any) -> Optional[float]:
        """Reject malformed and non-finite scores instead of inventing confidence."""
        if isinstance(value, bool) or not isinstance(value, Real):
            return None
        number = float(value)
        return number if math.isfinite(number) and -1 <= number <= 1 else None

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
            The cosine similarity in ``[-1.0, 1.0]``, or ``None`` when the
            result does not mention the mode at all.

        Raises
        ------
        None
        """
        if self._router is None or not result or not mode_name:
            return None

        entries = result.get("all_scores", [])
        for entry in entries if isinstance(entries, (list, tuple)) else []:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("target", "")) == mode_name:
                return self._cosine(entry.get("similarity"))

        if str(result.get("target_name", "") or "") == mode_name:
            return self._cosine(result.get("similarity"))

        return None

    @staticmethod
    def _build_semantic_context(
        request: CodexRequest, last_agent_messages: int = 1
    ) -> Optional[str]:
        """Compatibility view of the request's budgeted semantic sections."""
        budget = request.classify_max_chars
        parts = CodexSemanticLayer._build_semantic_parts(
            request, budget, budget // 2 if request.intent_text else budget,
            last_agent_messages,
        )
        if not parts:
            return None
        if budget > 0 and len(parts) == 2:
            intent, phase = parts
            return "\n".join((intent[:max(0, budget - len(phase) - 1)], phase))
        return "\n".join(parts)

    @staticmethod
    def _build_semantic_parts(
        request: CodexRequest, intent_max_chars: int, phase_max_chars: int,
        last_agent_messages: int = 1,
    ) -> Tuple[str, ...]:
        """
        Assemble the text embedded for the semantic lookup.

        The current user intent is followed by the text of the last
        *last_agent_messages* assistant ``output_text`` parts (oldest first),
        plus the latest linked tool activity. Only the active user turn is
        eligible; a separate phase budget protects it from a long user prompt.

        Parameters
        ----------
        request : CodexRequest
            The parsed request.
        intent_max_chars, phase_max_chars : int
            Independent character budgets, applied before section encoding.
        last_agent_messages : int
            How many of the newest assistant message parts are appended.

        Returns
        -------
        Tuple[str, ...]
            Independently budgeted sections, empty when the request carries no text
            at all (neither user nor assistant).

        Raises
        ------
        None
        """
        utterances = [
            "\n".join(part["text"] for part in message["content"])
            for message in (request.assistant_messages or [])
        ]
        phase_parts = (
            utterances[-last_agent_messages:] if last_agent_messages > 0 else []
        )
        tool_events = [
            event for event in request.activity
            if event.kind in ("function_call", "function_call_output") and event.name
        ]
        if tool_events:
            latest = tool_events[-1]
            call = next(
                (event for event in reversed(tool_events)
                 if event.kind == "function_call" and event.call_id == latest.call_id),
                None,
            )
            if call is not None:
                phase_parts.append(call.name + "\n" + call.text)
                if latest.kind == "function_call_output":
                    phase_parts.append(latest.text)

        intent = request.intent_text.strip()[:intent_max_chars]
        phase_parts = [part for part in phase_parts if part.strip()]
        if phase_parts:
            part_budget = max(
                1, (phase_max_chars - len(phase_parts) + 1) // len(phase_parts)
            )
            phase_parts = [part[:part_budget] for part in phase_parts]
        phase = "\n".join(phase_parts)[:phase_max_chars]
        return tuple(part for part in (intent, phase) if part)

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
