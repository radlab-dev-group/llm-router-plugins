"""
Constants shared across routing plugins.
"""

# Prefix used for all semantic-routing environment variable names.
# Individual env vars are constructed via f-string at the call site:
#   f"{SEMANTIC_ROUTING_PREFIX}COMPLEXITY_THRESHOLDS"
SEMANTIC_ROUTING_PREFIX = "LLM_ROUTER_ROUTING_SEMANTIC_"
