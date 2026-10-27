# Agentic Codex routing guide

`agentic_routing_codex` routes a **Codex CLI** request, not a generic model
choice. It classifies the request's class and work mode and picks a model for
each.

## What it does

`CodexRoutingPlugin` activates when `payload["model"]` equals its trigger value
(`"auto_codex"` by default). It reads the request's class (main turn, title
generation, compaction) and the work mode, and routes each to the model the
policy names for it. The decision is annotated under `payload["routing"]`.

It is the plugin the `agentic_knowledge` layer reads: the knowledge layer asks
the base of the repository for the mode this plugin resolved, so the two agree
on what the turn is.

## Configuration

Environment variables under `LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_`: the
policy, the trigger value, and the per-class and per-mode model mapping. See
`llm_router_plugins/utils/routing/agentic_routing/codex/README.md`.

## When to use it

Use `agentic_routing_codex` when the traffic is Codex CLI and you want the
model to follow the class and mode of the turn. It is the only routing plugin
that is aware of the Codex request shapes.
