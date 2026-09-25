"""
Claude Code model swap plugin.

Self-contained rewriting of the Anthropic-Messages requests emitted by the
**Claude Code** CLI.  A request whose model name belongs to a configured tier
is rewritten to the model that tier serves, so the tier-to-model mapping lives
in one gateway configuration file instead of one environment variable per tier
on every developer machine:

===============  ============================================  =========
Tier             Model IDs Claude Code sends                   Background
===============  ============================================  =========
``fable``        ``claude-fable-5-1``, ``claude-fable-5``      hardest tasks
``opus``         ``claude-opus-5-5``, ``claude-opus-4-8``      deep reasoning
``plan``         ``opusplan``, unresolved                      Plan Mode phase
``sonnet``       ``claude-sonnet-5``, ``claude-sonnet-4-6``    daily coding
``haiku``        ``claude-haiku-4-5``, ``claude-3-5-haiku-*``  background calls
===============  ============================================  =========

Those are the IDs Claude Code sends today, resolved from its own ``fable`` /
``opus`` / ``sonnet`` / ``haiku`` aliases and from
``ANTHROPIC_DEFAULT_<TIER>_MODEL``; naming them here instead lets one file
replace that whole set of variables.

Matching is done on the model name alone — no request content is classified, no
embeddings are computed and no dependencies beyond the standard library are
used.  A name that no tier claims costs one dictionary lookup: the payload is
returned as the same object, without a log line.


Modules
-------
- :mod:`~claude_code.mapping` — model-name normalization and the match cascade
- :mod:`~claude_code.config` — configuration loading, validation, env overrides
- :mod:`~claude_code.plugin` — the plugin rewriting ``payload["model"]``

Example
-------
::

    from llm_router_plugins.utils.routing.agentic_routing.claude_code import (
        ClaudeCodeRoutingPlugin,
    )

    plugin = ClaudeCodeRoutingPlugin(logger)
    result = plugin.apply({"model": "claude-sonnet-5", "messages": [...]})
    result["model"]
    # "qwen/Qwen3.8-Flash-Next"

The plugin fails open: a payload it cannot process is returned untouched rather
than turning a configuration mistake into a failed request.  It coexists with
``"agentic_routing_codex"`` (Codex CLI) and the two semantic plugins that answer
``"auto"``, because it activates on the model name it is configured for rather
than on a reserved trigger.
"""

from llm_router_plugins.utils.routing.agentic_routing.claude_code.config import (
    ClaudeCodeMode,
    ClaudeCodeRoutingConfig,
)
from llm_router_plugins.utils.routing.agentic_routing.claude_code.mapping import (
    DEFAULT_MODEL_FIELDS,
    DEFAULT_PROVIDER_PREFIXES,
    KNOWN_MODEL_FAMILIES,
    MATCH_EXACT,
    MATCH_FAMILY,
    MATCH_LITERAL,
    MATCH_WILDCARD,
    ModelMatcher,
    ModelMatch,
    find_ambiguous_wildcards,
    find_duplicate_literals,
    is_wildcard,
    model_family,
    normalize_model_name,
    validate_pattern,
)
from llm_router_plugins.utils.routing.agentic_routing.claude_code.plugin import (
    ClaudeCodeRoutingPlugin,
)

__all__ = [
    "DEFAULT_MODEL_FIELDS",
    "DEFAULT_PROVIDER_PREFIXES",
    "KNOWN_MODEL_FAMILIES",
    "MATCH_EXACT",
    "MATCH_FAMILY",
    "MATCH_LITERAL",
    "MATCH_WILDCARD",
    "ClaudeCodeMode",
    "ClaudeCodeRoutingConfig",
    "ClaudeCodeRoutingPlugin",
    "ModelMatch",
    "ModelMatcher",
    "find_ambiguous_wildcards",
    "find_duplicate_literals",
    "is_wildcard",
    "model_family",
    "normalize_model_name",
    "validate_pattern",
]
