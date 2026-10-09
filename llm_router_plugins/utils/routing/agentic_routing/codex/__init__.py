"""
Codex CLI routing plugin.

Self-contained routing of the OpenAI-Responses-style requests emitted by the
Codex CLI coding agent.  Requests carrying the trigger model (default
``"auto_codex"``) are classified into a work mode and rewritten to the model
configured for that mode:

======================  ==========================================  ==============
Mode                    Routed by                                   Model
======================  ==========================================  ==============
``plan``                ``<collaboration_mode>`` Plan Mode block    Flash-Next
``implement``           fallback for plain main turns               Flash-Next
``test``                keyword scoring of the latest message       27B
``git_review``          keyword scoring of the latest message       27B
``review``              keyword scoring of the latest message       Flash-Next
``debug``               keyword scoring of the latest message       Flash-Next
``aux_title``           request class (system title generation)     27B
``compaction``          request class (context compaction)          27B
======================  ==========================================  ==============

Modules
-------
- :mod:`~codex.payload` — payload → normalized :class:`CodexRequest` via
  :class:`CodexPayloadParser`
- :mod:`~codex.scoring` — deduplicated keyword, phrase and regex scoring
  (:class:`CodexModeScorer`)
- :mod:`~codex.classifier` — the deterministic mode cascade
  (:class:`CodexModeClassifier`)
- :mod:`~codex.semantic` — embedding cosine similarity (optional, fail-open)
- :mod:`~codex.config` — configuration loading, validation and env overrides
- :mod:`~codex.plugin` — the plugin rewriting ``payload["model"]``

Example
-------
::

    from llm_router_plugins.utils.routing.agentic_routing.codex import (
        CodexRoutingPlugin,
    )

    plugin = CodexRoutingPlugin(logger)
    result = plugin.apply({"model": "auto_codex", "input": [...]})
    result["model"]
    # "qwen/Qwen3.8-Flash-Next"

The cascade is deterministic first, so a request is routed identically, offline
and for free whenever a cheap layer can answer.  The optional semantic layer
contributes embedding cosine similarity over the mode descriptions and
examples through the shared BiEncoder + FAISS router
(``llm_router_plugins.utils.routing.embedder``); without it — or without
``faiss`` installed — the layer steps aside and nothing else changes.
Codex opts into ``settings.semantic.aggregation = per_target_top_k``: complete,
balanced per-mode scores must meet both the threshold and ``min_margin``.
``intent_max_chars`` and ``phase_max_chars`` budget independently encoded
context sections. These four settings are required in the supplied JSON.

Within each mode, scoring keeps the strongest non-overlapping matches, preferring
longer spans at equal weights; each declared rule contributes at most once.
Each rule supplies only its first non-negated match, without retrying a later
occurrence after losing deduplication.
``rank_modes(text, modes)`` returns ``ModeScore(mode, score, matches)`` values
whose matches are ``SignalMatch(signal, start, end, weight)`` values with offsets
in lower-cased text (exclusive end). These types live in ``codex.scoring``.
``detect_mode()`` returns ``(None, top_score)`` for a tied top score.
The classifier requires both ``settings.heuristic_min_score`` and a lead over the
runner-up of at least ``settings.heuristic_min_margin`` (default ``1.0``); ties
are rejected even at margin zero and continue to semantic routing or fallback.
Heuristic routing similarity remains ``score / (score + 1)`` for API compatibility:
it is heuristic strength, not a calibrated probability.

``settings.heuristic_negation_pattern`` configures local PL/EN action prohibitions
such as ``nie uruchamiaj`` / ``do not run``, ``nie pisz`` / ``do not write`` and
``bez uruchamiania`` / ``without running``, not every ``nie`` (``nie działa`` is
debugging evidence). The regex matches the entire prohibited fragment and defines
its own boundaries, without a separate parser. The default scope ends at sentence/semicolon/comma boundaries,
contrastive ``ale`` / ``but``, or a new positive action after ``i`` / ``and``:
prohibiting test execution does not suppress a separate request to write tests.
An empty regex disables filtering; zero-width spans are ignored.
This is not full NLP or a semantic veto. Negation, margin, weights and complete phase
rules must be supplied in the loaded config; missing fields are reported, not
filled from another JSON file.
``settings.heuristic_weights`` supplies default keyword, phrase and pattern weights;
per-keyword weights and phrase suffixes retain precedence.

The plugin is registered as ``"agentic_routing_codex"`` and answers only its own
trigger; ``"auto"`` traffic stays with the semantic routing plugins.
"""

from llm_router_plugins.utils.routing.agentic_routing.codex.classifier import (
    HEURISTIC_MODES,
    CLASS_ROUTED_MODES,
    SOURCE_CLASS,
    SOURCE_COLLABORATION_MODE,
    SOURCE_EXPLICIT,
    SOURCE_FALLBACK,
    SOURCE_HEURISTIC,
    SOURCE_PHASE,
    SOURCE_SEMANTIC,
    RoutingDecision,
    CodexModeClassifier,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.config import (
    CodexMode,
    CodexRoutingConfig,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.payload import (
    COLLABORATION_MODE_DEFAULT,
    COLLABORATION_MODE_PLAN,
    DEFAULT_CLASSIFY_MAX_CHARS,
    REQUEST_CLASS_AUX_TITLE,
    REQUEST_CLASS_COMPACTION,
    REQUEST_CLASS_MAIN,
    CodexActivity,
    CodexRequest,
    CodexPayloadParser,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.plugin import (
    CodexRoutingPlugin,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.scoring import (
    CodexModeScorer,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.semantic import (
    CodexSemanticLayer,
)

__all__ = [
    "COLLABORATION_MODE_DEFAULT",
    "COLLABORATION_MODE_PLAN",
    "REQUEST_CLASS_AUX_TITLE",
    "REQUEST_CLASS_COMPACTION",
    "REQUEST_CLASS_MAIN",
    "SOURCE_CLASS",
    "SOURCE_COLLABORATION_MODE",
    "SOURCE_EXPLICIT",
    "SOURCE_FALLBACK",
    "SOURCE_HEURISTIC",
    "SOURCE_PHASE",
    "SOURCE_SEMANTIC",
    "HEURISTIC_MODES",
    "CLASS_ROUTED_MODES",
    "DEFAULT_CLASSIFY_MAX_CHARS",
    "CodexMode",
    "CodexModeClassifier",
    "CodexModeScorer",
    "CodexPayloadParser",
    "CodexActivity",
    "CodexRequest",
    "CodexRoutingConfig",
    "CodexRoutingPlugin",
    "CodexSemanticLayer",
    "RoutingDecision",
]
