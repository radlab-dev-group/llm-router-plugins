"""
Work-mode resolution cascade for the Codex routing plugin.

The classifier turns a parsed
:class:`~llm_router_plugins.utils.routing.agentic_routing.codex.
payload.CodexRequest` into a :class:`RoutingDecision`.  It is fully
deterministic whenever no semantic layer is injected: layers are tried in a
fixed order, the first layer that can answer wins, and every cheap layer runs
before the optional embedding lookup.

Cascade
-------
1. **explicit** — a mode named directly by the caller in the payload
   (``agent_mode`` / ``codex_mode`` / ``metadata.agent_mode``).
2. **class** — request classes that are not real conversations: context
   compaction and auxiliary title generation.
3. **collaboration_mode** — the ``<collaboration_mode>`` block injected by the
   Codex CLI declares Plan Mode.
4. **heuristic** — keyword scoring of the latest user message against the
   ``test`` / ``review`` / ``debug`` modes.
5. **semantic** — embedding cosine similarity over the mode descriptions and
   examples, delegated to
   :class:`~llm_router_plugins.utils.routing.agentic_routing.codex.semantic.
   CodexSemanticLayer`.  Optional: a cascade without a layer skips it.
6. **fallback** — the configured fallback mode (``implement`` by default).

A main turn therefore never falls below the fallback mode: the keyword and
semantic layers can specialise the decision but can never make it weaker.
A single request queries the vector store at most once; the same lookup also
supplies the cosine similarity reported for keyword and fallback decisions.

Example
-------
::

    request = parse_codex_payload(payload)
    decision = classify(payload, request, config)
    decision.mode        # "plan"
    decision.source      # "collaboration_mode"
    decision.similarity  # 1.0
"""

from dataclasses import dataclass
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
    detect_mode,
    score_to_similarity,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.semantic import (
    CodexSemanticLayer,
)

__all__ = [
    "SOURCE_EXPLICIT",
    "SOURCE_CLASS",
    "SOURCE_COLLABORATION_MODE",
    "SOURCE_HEURISTIC",
    "SOURCE_SEMANTIC",
    "SOURCE_FALLBACK",
    "HEURISTIC_MODES",
    "RoutingDecision",
    "classify",
]

#: The caller named the mode in the payload itself.
SOURCE_EXPLICIT = "explicit"

#: The request class (compaction / title generation) decided the mode.
SOURCE_CLASS = "class"

#: The ``<collaboration_mode>`` block declared the mode.
SOURCE_COLLABORATION_MODE = "collaboration_mode"

#: Keyword scoring of the latest user message decided the mode.
SOURCE_HEURISTIC = "heuristic"

#: Embedding cosine similarity over the mode examples decided the mode.
SOURCE_SEMANTIC = "semantic"

#: No layer could decide, the configured fallback mode was used.
SOURCE_FALLBACK = "fallback"

#: Modes eligible for keyword scoring.  ``plan`` is decided by the
#: collaboration block and ``implement`` is the fallback, so neither needs
#: keywords; ``aux_title``/``compaction`` are class-routed.
HEURISTIC_MODES: Tuple[str, ...] = ("test", "review", "debug")

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
        Confidence in ``[0.0, 1.0]`` reported in ``payload["routing"]``.

    Raises
    ------
    None
    """

    mode: str
    source: str
    score: float
    similarity: float


def classify(
    payload: Dict[str, Any],
    request: CodexRequest,
    config: CodexRoutingConfig,
    semantic: Optional[CodexSemanticLayer] = None,
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
    config : CodexRoutingConfig
        Routing configuration providing the modes, the fallback and the
        heuristic settings.
    semantic : CodexSemanticLayer, optional
        Embedding similarity layer consulted after the deterministic layers.
        When omitted or unavailable the cascade stays fully deterministic.

    Returns
    -------
    RoutingDecision
        The selected mode and the layer that selected it.  A mode that is not
        configured is skipped, so the cascade always returns a decision.

    Raises
    ------
    None
    """
    body: Dict[str, Any] = payload if isinstance(payload, dict) else {}
    modes = config.mode_by_name

    explicit = _explicit_mode(body, modes)
    if explicit is not None:
        return RoutingDecision(explicit, SOURCE_EXPLICIT, 1.0, 1.0)

    if request.request_class == REQUEST_CLASS_COMPACTION and "compaction" in modes:
        return RoutingDecision("compaction", SOURCE_CLASS, 1.0, 1.0)

    if request.request_class == REQUEST_CLASS_AUX_TITLE and "aux_title" in modes:
        return RoutingDecision("aux_title", SOURCE_CLASS, 1.0, 1.0)

    if request.collaboration_mode == COLLABORATION_MODE_PLAN and "plan" in modes:
        return RoutingDecision("plan", SOURCE_COLLABORATION_MODE, 1.0, 1.0)

    routed = None
    if semantic is not None and semantic.available:
        routed = semantic.route(request.latest_user_text)

    if config.heuristic_enabled:
        decision = _heuristic_mode(
            request.latest_user_text, config, semantic=semantic, routed=routed
        )
        if decision is not None:
            return decision

    mode, similarity = semantic.accept(routed) if semantic is not None else (None, 0.0)
    if mode is not None:
        return RoutingDecision(mode.name, SOURCE_SEMANTIC, similarity, similarity)

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
    )


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
        _metadata_mode(body),
    ):
        if isinstance(value, str):
            name = value.strip()
            if name in modes:
                return name
    return None


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
    text: str,
    config: CodexRoutingConfig,
    semantic: Optional[CodexSemanticLayer] = None,
    routed: Optional[Dict[str, Any]] = None,
) -> Optional[RoutingDecision]:
    """
    Score the latest user text against the heuristic candidate modes.

    Parameters
    ----------
    text : str
        The latest user message of the request.
    config : CodexRoutingConfig
        Routing configuration providing the candidate modes and the minimum
        score required to accept a match.
    semantic : CodexSemanticLayer, optional
        Embedding layer used to replace the derived keyword confidence with the
        cosine similarity of the winning mode.
    routed : Optional[Dict[str, Any]]
        The router lookup already performed for this request, reused here so
        the text is never embedded twice.

    Returns
    -------
    Optional[RoutingDecision]
        A :data:`SOURCE_HEURISTIC` decision when a candidate scores at least
        ``config.heuristic_min_score``, otherwise ``None``.

    Raises
    ------
    None
    """
    if not text:
        return None

    candidates = [mode for mode in config.codex_modes if mode.name in HEURISTIC_MODES]
    if not candidates:
        return None

    best_mode, score = detect_mode(text, candidates)
    if best_mode is None or score < config.heuristic_min_score:
        return None

    similarity = score_to_similarity(score)
    if semantic is not None:
        cosine = semantic.similarity_for(best_mode.name, routed)
        if cosine is not None:
            similarity = cosine

    return RoutingDecision(
        mode=best_mode.name,
        source=SOURCE_HEURISTIC,
        score=score,
        similarity=similarity,
    )
