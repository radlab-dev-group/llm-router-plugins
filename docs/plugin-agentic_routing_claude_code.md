# Agentic Claude Code routing guide

`agentic_routing_claude_code` is the Claude Code counterpart of the Codex
agentic router. It performs a tier-to-model mapping for Claude Code requests.

## What it does

`ClaudeCodeRoutingPlugin` maps the tier of a Claude Code request to a model.
It is a lighter router than the Codex one: it does not classify work modes, it
maps the request to a model by tier.

## Configuration

Environment variables under `LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CLAUDE_CODE_`:
the tier-to-model mapping. See
`llm_router_plugins/utils/routing/agentic_routing/claude_code/README.md`.

## When to use it

Use `agentic_routing_claude_code` when the traffic is Claude Code and you want
a tier-based model choice.
