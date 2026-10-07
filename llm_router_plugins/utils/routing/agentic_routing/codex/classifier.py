"""
Work-mode resolution cascade for the Codex routing plugin.

:class:`CodexModeClassifier` turns a parsed
:class:`~llm_router_plugins.utils.routing.agentic_routing.codex.payload.CodexRequest`
into a :class:`RoutingDecision`.  It is fully deterministic whenever no
semantic layer is injected: layers are tried in a fixed order, the first
layer that can answer wins, and every cheap layer runs before the optional
embedding lookup.

Cascade
-------
1. **explicit** — a mode named directly by the caller in the payload
   (``agent_mode`` / ``codex_mode`` / ``metadata.agent_mode``).
2. **class** — request classes that are not real conversations: context
   compaction and auxiliary title generation.
3. **collaboration_mode** — the ``<collaboration_mode>`` block injected by the
   Codex CLI declares Plan Mode.
4. **phase** — a clear current action in assistant/tool activity after the
   latest user command.
5. **heuristic** — keyword scoring of the current user intent against the
   keyword-scored modes (``test``, ``git_review``, ``review``, ``debug``).
6. **semantic** — embedding cosine similarity over the mode descriptions and
   examples, delegated to
   :class:`~llm_router_plugins.utils.routing.agentic_routing.codex.semantic.
   CodexSemanticLayer`.  Optional: a cascade without a layer skips it.
7. **fallback** — the configured fallback mode (``implement`` by default).

A main turn therefore never falls below the fallback mode: the keyword and
semantic layers can specialise the decision but can never make it weaker.
The deterministic layers answer without ever touching the embedding stack: a
request queries the vector store at most once, and only when they stay silent.
A decision taken by a deterministic layer reports the confidence derived from
its own signal — ``score / (score + 1)`` for a keyword hit, ``1.0`` above it.

Example
-------
::

    request = CodexPayloadParser().parse(payload)
    decision = CodexModeClassifier(config).classify(payload, request)
    decision.mode        # "plan"
    decision.source      # "collaboration_mode"
    decision.similarity  # 1.0
"""

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple

from llm_router_plugins.utils.routing.agentic_routing.codex.config import (
    CodexRoutingConfig,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.payload import (
    COLLABORATION_MODE_PLAN,
    REQUEST_CLASS_AUX_TITLE,
    REQUEST_CLASS_COMPACTION,
    CodexRequest,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.scoring import (
    CodexModeScorer,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.phase import (
    PhaseEvidence,
    detect_phase_evidence,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.semantic import (
    CodexSemanticLayer,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.state import (
    MemoryStatus,
    RoutingStateStore,
    merge_state,
    session_key,
)

__all__ = [
    "SOURCE_EXPLICIT",
    "SOURCE_CLASS",
    "SOURCE_COLLABORATION_MODE",
    "SOURCE_PHASE",
    "SOURCE_HEURISTIC",
    "SOURCE_SEMANTIC",
    "SOURCE_MEMORY",
    "SOURCE_FALLBACK",
    "HEURISTIC_MODES",
    "CLASS_ROUTED_MODES",
    "CodexModeClassifier",
    "RoutingDecision",
]

#: The caller named the mode in the payload itself.
SOURCE_EXPLICIT = "explicit"

#: The request class (compaction / title generation) decided the mode.
SOURCE_CLASS = "class"

#: The ``<collaboration_mode>`` block declared the mode.
SOURCE_COLLABORATION_MODE = "collaboration_mode"

#: A clear current assistant action or executed tool decided the work phase.
SOURCE_PHASE = "phase"

#: Keyword scoring of the latest user message decided the mode.
SOURCE_HEURISTIC = "heuristic"

#: Embedding cosine similarity over the mode examples decided the mode.
SOURCE_SEMANTIC = "semantic"

#: No layer could decide, the configured fallback mode was used.
SOURCE_FALLBACK = "fallback"

#: A still-fresh phase evidence from the shared session memory decided the mode.
SOURCE_MEMORY = "memory"

#: Modes eligible for keyword scoring.  ``plan`` is decided by the
#: collaboration block and ``implement`` is the fallback, so neither needs
#: keywords; ``aux_title``/``compaction`` are class-routed. Candidate order
#: never resolves conflicts: tied or insufficiently separated scores abstain.
HEURISTIC_MODES: Tuple[str, ...] = ("test", "git_review", "review", "debug")

#: Modes the request-class layer decides on its own.  The names match the
#: ``REQUEST_CLASS_*`` values of :mod:`~codex.payload`, and the plugin leaves
#: them out of the semantic index.
CLASS_ROUTED_MODES: Tuple[str, ...] = (
    REQUEST_CLASS_COMPACTION,
    REQUEST_CLASS_AUX_TITLE,
)

#: Name of the payload key holding an explicit mode override.
_AGENT_MODE_KEY = "agent_mode"

#: Name of the payload key holding an alternative explicit mode override.
_CODEX_MODE_KEY = "codex_mode"


@dataclass(frozen=True)
class RoutingDecision:
    """
    Result of the work-mode resolution cascade.

    Parameters
    ----------
    mode : str
        Name of the selected Codex work mode.
    source : str
        Name of the cascade layer that produced the decision (one of the
        ``SOURCE_*`` constants).
    score : float
        Raw heuristic score.  ``1.0`` for the deterministic layers that are
        not scored, ``0.0`` for the fallback layer.
    similarity : float
        Value in ``[0.0, 1.0]`` reported in ``payload["routing"]``. For heuristics
        this is score strength, not a calibrated probability of correctness.
    reason : str
        Why this layer answered, as a short code: the phase evidence reason, or
        a marker such as ``"below threshold"``.  Diagnostic only.
    evidence : Optional[PhaseEvidence]
        The phase evidence behind a phase or memory decision, ``None`` otherwise.
    memory : MemoryStatus or None
        What the session memory did — hit, miss, expired, conflict, disabled,
        unavailable — ``None`` when the request never consulted it.
    semantic : str
        Outcome of the semantic layer: ``"disabled"``, ``"unavailable"``,
        ``"accepted"``, ``"below threshold"``, ``"ambiguous"`` or ``""`` when
        the layer was never reached.  A refusal is reported as a reason, never
        as a probability.
    """

    mode: str
    source: str
    score: float
    similarity: float
    reason: str = ""
    evidence: Optional[PhaseEvidence] = None
    memory: Optional[MemoryStatus] = None
    semantic: str = ""


class CodexModeClassifier:
    """
    Resolve the Codex work mode of a request through the cascade.

    The classifier owns the *order* of the layers, not their data: the request
    comes from :class:`~codex.payload.CodexPayloadParser`, the mode table and
    the thresholds from :class:`~codex.config.CodexRoutingConfig`, and the
    optional embedding lookup from
    :class:`~codex.semantic.CodexSemanticLayer`.  Layers are tried in a fixed
    order and the first one that can answer wins, so an identical request is
    always decided identically.

    The instance holds no per-request state, so one classifier is safely shared
    by every request a plugin handles.

    Parameters
    ----------
    config : CodexRoutingConfig
        Routing configuration providing the modes, the fallback and the
        heuristic settings.
    semantic : CodexSemanticLayer, optional
        Embedding similarity layer, consulted only after every deterministic
        layer has stayed silent.  When omitted or unavailable the cascade stays
        fully deterministic.
    scorer : CodexModeScorer, optional
        Keyword scorer shared by every request.  Built on demand, so injecting
        one is only useful to share a pre-warmed plan cache or to stub scoring
        out.
    """

    def __init__(
        self,
        config: CodexRoutingConfig,
        semantic: Optional[CodexSemanticLayer] = None,
        scorer: Optional[CodexModeScorer] = None,
        memory: Optional[RoutingStateStore] = None,
    ) -> None:
        """
        Store the configuration, the layers and the optional session memory.

        Parameters
        ----------
        config : CodexRoutingConfig
            Routing configuration used by every layer of the cascade.
        semantic : CodexSemanticLayer, optional
            Embedding similarity layer, or ``None`` for a purely
            deterministic cascade.
        scorer : CodexModeScorer, optional
            Keyword scorer, built when omitted.
        memory : RoutingStateStore, optional
            Shared session memory, consulted after the current turn's evidence
            and before the user text is scored again.

        Returns
        -------
        None

        Raises
        ------
        None
        """
        self._config = config
        self._semantic = semantic
        self._memory = memory
        self._scorer = scorer if scorer is not None else CodexModeScorer(
            negation_pattern=config.heuristic_negation_pattern,
            weights=config.heuristic_weights,
        )

    @property
    def memory(self) -> Optional[RoutingStateStore]:
        """
        Return the shared session memory, if one was injected.

        Returns
        -------
        Optional[RoutingStateStore]
            The store, or ``None`` when the cascade runs statelessly.
        """
        return self._memory

    def classify(
        self, payload: Dict[str, Any], request: CodexRequest
    ) -> RoutingDecision:
        """
        Resolve the Codex work mode for *request*.

        Parameters
        ----------
        payload : Dict[str, Any]
            The raw request payload, source of the explicit override keys.
        request : CodexRequest
            The parsed payload, source of the request class, collaboration mode
            and latest user text.

        Returns
        -------
        RoutingDecision
            The selected mode and the layer that selected it.  A mode that is
            not configured is skipped, so the cascade always returns a
            decision.

        Raises
        ------
        None
        """
        body: Dict[str, Any] = payload if isinstance(payload, dict) else {}
        config = self._config
        semantic = self._semantic
        modes = config.mode_by_name

        explicit = self._explicit_mode(body, modes)
        if explicit is not None:
            return RoutingDecision(explicit, SOURCE_EXPLICIT, 1.0, 1.0)

        if (
            request.request_class in CLASS_ROUTED_MODES
            and request.request_class in modes
        ):
            return RoutingDecision(request.request_class, SOURCE_CLASS, 1.0, 1.0)

        if request.collaboration_mode == COLLABORATION_MODE_PLAN and "plan" in modes:
            return RoutingDecision("plan", SOURCE_COLLABORATION_MODE, 1.0, 1.0)

        memory_status: Optional[MemoryStatus] = None
        if config.heuristic_enabled:
            evidence = detect_phase_evidence(request.activity, config.phase)
            # remembered, memory_status = (None, None)
            if evidence is not None and evidence.mode in modes:
                return RoutingDecision(
                    evidence.mode, SOURCE_PHASE, 1.0, 1.0,
                    reason=evidence.reason, evidence=evidence,
                    memory=MemoryStatus("not consulted", "fresh evidence")
                    if self._memory is not None else None,
                )
            remembered, memory_status = self._remembered(request, modes)
            if remembered is not None:
                return RoutingDecision(
                    remembered, SOURCE_MEMORY, 1.0, 1.0,
                    reason="carried phase", memory=memory_status,
                )
            decision = self._heuristic_mode(request.intent_text, modes)
            if decision is not None:
                return replace(decision, memory=memory_status)

        routed = (
            semantic.route(request)
            if semantic is not None and semantic.available
            else None
        )
        mode, similarity = (
            semantic.accept(routed) if semantic is not None else (None, 0.0)
        )
        if mode is not None:
            return RoutingDecision(
                mode.name, SOURCE_SEMANTIC, similarity, similarity,
                memory=memory_status,
            )
        semantic_outcome = self._semantic_outcome(semantic, routed, similarity)

        fallback_similarity = 0.0
        if semantic is not None:
            cosine = semantic.similarity_for(config.fallback_mode, routed)
            if cosine is not None:
                fallback_similarity = cosine
        return RoutingDecision(
            config.fallback_mode,
            SOURCE_FALLBACK,
            0.0,
            fallback_similarity,
            reason="no layer answered",
            memory=memory_status,
            semantic=semantic_outcome,
        )

    def _remembered(
        self, request: CodexRequest, modes: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[MemoryStatus]]:
        """
        Return the phase the session memory still vouches for, if any.

        The memory only carries a phase inside the same command generation: a
        new user instruction resets what the agent was doing, and a request
        without enough identifiers to know which generation it belongs to gets
        nothing rather than a guess.

        Parameters
        ----------
        request : CodexRequest
            The request being routed.
        modes : Dict[str, Any]
            Configured modes by name; an unknown remembered mode is ignored.

        Returns
        -------
        Tuple[Optional[str], Optional[MemoryStatus]]
            The carried mode and what the store reported, the status being
            ``None`` when a disabled memory was never consulted at all.
        """
        store = self._memory
        if store is None:
            return None, None
        key = session_key(
            self._config.memory.key_prefix,
            request.session_id,
            request.thread_id,
            request.agent_name,
        )
        if key is None or not request.turn_id:
            return None, MemoryStatus("miss", "no session identity")
        stored, status = store.read(key)
        carried = merge_state(stored, request.turn_id, stored is not None)
        if carried is None or carried.mode not in modes:
            return None, status
        return carried.mode, status

    @staticmethod
    def _semantic_outcome(
        semantic: Optional[CodexSemanticLayer],
        routed: Any,
        similarity: float,
    ) -> str:
        """
        Name the reason the semantic layer did not decide, without a probability.

        Distinguishing "off", "broken" and "genuinely ambiguous" is the point:
        an outage and an unclear ranking call for different fixes, and neither
        is a statement about how likely the fallback is to be right.
        """
        if semantic is None:
            return "disabled"
        if not semantic.available:
            return "unavailable"
        if not routed:
            return "unavailable"
        if similarity <= 0:
            return "no ranking"
        if similarity < semantic.threshold:
            return "below threshold"
        return "ambiguous"

    @staticmethod
    def _explicit_mode(
        body: Dict[str, Any],
        modes: Dict[str, Any],
    ) -> Optional[str]:
        """
        Return the mode explicitly requested by the caller, if any.

        Parameters
        ----------
        body : Dict[str, Any]
            The raw payload.
        modes : Dict[str, Any]
            Mapping of configured mode names.

        Returns
        -------
        Optional[str]
            The named mode when it is configured, otherwise ``None``.

        Raises
        ------
        None
        """
        for value in (
            body.get(_AGENT_MODE_KEY),
            body.get(_CODEX_MODE_KEY),
            CodexModeClassifier._metadata_mode(body),
        ):
            if isinstance(value, str):
                name = value.strip()
                if name in modes:
                    return name
        return None

    @staticmethod
    def _metadata_mode(body: Dict[str, Any]) -> Any:
        """
        Return ``payload["metadata"]["agent_mode"]`` when present.

        Parameters
        ----------
        body : Dict[str, Any]
            The raw payload.

        Returns
        -------
        Any
            The metadata override value, or ``None`` when the metadata block is
            missing or is not a mapping.

        Raises
        ------
        None
        """
        metadata = body.get("metadata")
        if not isinstance(metadata, dict):
            return None
        return metadata.get(_AGENT_MODE_KEY)

    def _heuristic_mode(
        self,
        text: str,
        modes: Dict[str, Any],
    ) -> Optional[RoutingDecision]:
        """
        Score the latest user text against the heuristic candidate modes.

        Parameters
        ----------
        text : str
            The latest user message of the request.
        modes : Dict[str, Any]
            Configured modes by name, normally ``config.mode_by_name``, reused
            from the cascade so the lookup is built once per request.

        Returns
        -------
        Optional[RoutingDecision]
            A :data:`SOURCE_HEURISTIC` decision when a candidate scores at
            least ``config.heuristic_min_score`` and leads the runner-up by
            ``config.heuristic_min_margin``. Ties always abstain. Reported
            similarity is heuristic strength, not calibrated probability.

        Raises
        ------
        None
        """
        if not text:
            return None

        candidates = [modes[name] for name in HEURISTIC_MODES if name in modes]
        if not candidates:
            return None

        ranking = self._scorer.rank_modes(text, candidates)
        best = ranking[0]
        runner_up = ranking[1].score if len(ranking) > 1 else 0.0
        score = best.score
        if (
            score <= 0
            or score < self._config.heuristic_min_score
            or score <= runner_up
            or score - runner_up < self._config.heuristic_min_margin
        ):
            return None

        return RoutingDecision(
            mode=best.mode.name,
            source=SOURCE_HEURISTIC,
            score=score,
            similarity=self._scorer.score_to_similarity(score),
        )
