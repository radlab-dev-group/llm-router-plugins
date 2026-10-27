# Bi-encoder semantic routing guide

`semantic_biencoder_routing` is the embedding-based routing plugin. It encodes
the user's last statement with a bi-encoder, searches a persisted FAISS index
of the targets, and routes to the closest one.

## What it does

`SemanticBiEncoderRoutingPlugin` activates when `payload["model"] == "auto"`.
It builds or loads a FAISS index over the configured targets, embeds the query,
and picks the nearest target's model. The index is persisted so it is built
once and loaded afterwards. The decision is annotated under
`payload["routing"]`.

It shares the common routing layer with the other embedding router: the
`EmbeddingRouter` in `llm_router_plugins/utils/routing/embedder.py`, the
`RoutingTarget` in `target.py`, and the config helpers in `common.py`.

## Configuration

Environment variables under `LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_`: the
targets, the embedder model, the similarity threshold, and the index
persistence directory. See
`llm_router_plugins/utils/routing/semantic_biencoder/README.md`.

## When to use it

Use `semantic_biencoder_routing` when the routing decision needs semantic
similarity rather than word patterns, and the set of targets is stable enough
to index once.
