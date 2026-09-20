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
``implement``           fallback for plain main turns               27B
``test``                keyword scoring of the latest message       27B
``git_review``          keyword scoring of the latest message       27B
``review``              keyword scoring of the latest message       27B
``debug``               keyword scoring of the latest message       27B
``aux_title``           request class (system title generation)     Flash-Next
``compaction``          request class (context compaction)          Flash-Next
======================  ==========================================  ==============

Modules
-------
- :mod:`~codex.payload` — payload → normalized :class:`CodexRequest`
- :mod:`~codex.scoring` — keyword, phrase and regex scoring
- :mod:`~codex.classifier` — the deterministic mode cascade
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
examples through the same BiEncoder + FAISS router used by the
``agentic_routing`` plugin; without it — or without ``faiss`` installed — the
layer steps aside and nothing else changes.

The plugin is registered as ``"agentic_routing_codex"`` and coexists with the
``"agentic_routing"`` plugin, which handles the ``"auto_agentic"`` trigger.
"""

from llm_router_plugins.utils.routing.agentic_routing.codex.classifier import (
    HEURISTIC_MODES,
    SOURCE_CLASS,
    SOURCE_COLLABORATION_MODE,
    SOURCE_EXPLICIT,
    SOURCE_FALLBACK,
    SOURCE_HEURISTIC,
    SOURCE_SEMANTIC,
    RoutingDecision,
    classify,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.config import (
    CodexMode,
    CodexRoutingConfig,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.payload import (
    COLLABORATION_MODE_DEFAULT,
    COLLABORATION_MODE_PLAN,
    REQUEST_CLASS_AUX_TITLE,
    REQUEST_CLASS_COMPACTION,
    REQUEST_CLASS_MAIN,
    CodexRequest,
    parse_codex_payload,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.plugin import (
    CodexRoutingPlugin,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.scoring import (
    detect_mode,
    score_mode,
    score_to_similarity,
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
    "SOURCE_SEMANTIC",
    "HEURISTIC_MODES",
    "CodexMode",
    "CodexRequest",
    "CodexRoutingConfig",
    "CodexRoutingPlugin",
    "CodexSemanticLayer",
    "RoutingDecision",
    "classify",
    "detect_mode",
    "parse_codex_payload",
    "score_mode",
    "score_to_similarity",
]
