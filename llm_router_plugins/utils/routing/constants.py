"""
Constants shared across routing plugins.
"""

# Prefix used for all semantic-routing environment variable names.
# Individual env vars are constructed via f-string at the call site:
#   f"{SEMANTIC_ROUTING_PREFIX}COMPLEXITY_THRESHOLDS"
SEMANTIC_ROUTING_PREFIX = "LLM_ROUTER_ROUTING_SEMANTIC_"

# Prefix for BiEncoder-specific semantic routing environment variables.
SEMANTIC_BIENCODER_ROUTING_PREFIX = f"{SEMANTIC_ROUTING_PREFIX}BIENCODER_"

# Prefix for agentic-routing environment variable names.
AGENTIC_ROUTING_PREFIX = f"{SEMANTIC_ROUTING_PREFIX}AGENTIC_"
