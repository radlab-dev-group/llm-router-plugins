# Simple semantic routing guide

Routing plugins run on the way **in** and pick a model by rewriting
`payload["model"]`. `simple_semantic_routing` is the heuristic one: no
embeddings, no external index, just pattern matching on the user's text.

## What it does

`SimpleSemanticRoutingPlugin` activates when `payload["model"] == "auto"`.
It reads the last user statement, matches it against configured intent
patterns, and picks the model for the matched intent. The decision is
annotated under `payload["routing"]`.

## Configuration

Environment variables under `LLM_ROUTER_ROUTING_SEMANTIC_`:

- `...COMPLEXITY_THRESHOLDS` — the thresholds that separate work states,
- `...MODELS` — the set of models it may route to,
- `...DEFAULT_MODEL` — the model chosen when nothing matches,
- `...INTENT_<name>` — one variable per intent, carrying its patterns.

## When to use it

Use `simple_semantic_routing` when routing must stay offline and fast, and the
work is describable with word patterns. It is the lightest of the three
routing plugins: no FAISS, no embedding model, no persisted index.
