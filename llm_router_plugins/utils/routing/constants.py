"""
Constants shared across routing plugins.
"""

# Prefix used for all semantic-routing environment variable names.
# Individual env vars are constructed via f-string at the call site:
#   f"{SEMANTIC_ROUTING_PREFIX}COMPLEXITY_THRESHOLDS"
SEMANTIC_ROUTING_PREFIX = "LLM_ROUTER_ROUTING_SEMANTIC_"

# Prefix for BiEncoder-specific semantic routing environment variables.
SEMANTIC_BIENCODER_ROUTING_PREFIX = f"{SEMANTIC_ROUTING_PREFIX}BIENCODER_"

# Base prefix for agentic-routing environment variable names. It is kept as the
# base of the Codex prefix below, which is part of the deployed env-var contract.
AGENTIC_ROUTING_PREFIX = f"{SEMANTIC_ROUTING_PREFIX}AGENTIC_"

# Prefix for Codex-agentic routing environment variable names.
AGENTIC_CODEX_ROUTING_PREFIX = f"{AGENTIC_ROUTING_PREFIX}CODEX_"

# Prefix for Claude Code model-swap environment variable names.
AGENTIC_CLAUDE_CODE_ROUTING_PREFIX = f"{AGENTIC_ROUTING_PREFIX}CLAUDE_CODE_"
