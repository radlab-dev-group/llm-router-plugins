# Semantic BiEncoder Routing (`semantic_biencoder_routing`)

Embedding-based model selection. A set of **routing targets** — each one a name, a model and a pile of natural-language
examples — is embedded once at startup into a FAISS index. Every incoming request that asks for `model: "auto"` has
its last user message embedded and matched against that index by cosine similarity, and `payload["model"]` is replaced
with the model of the closest target.

This is the "teach routing by writing examples" plugin: to change a decision, add or reword examples rather than
maintain keyword lists. It needs `sentence-transformers` and `faiss` and it loads an embedding model into the router
process.

- **Plugin name (registry key):** `semantic_biencoder_routing`
- **Class:** `llm_router_plugins.utils.routing.semantic_biencoder.semantic_biencoder_routing.SemanticBiEncoderRoutingPlugin`
- **Default config:** `llm_router_plugins/resources/routing/semantic_biencoder.json`
- **Trigger:** `payload["model"]` equals `auto` after trimming (`should_route`)
- **Env prefix:** `LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_`
- **Dependencies:** `pip install -e ".[ml]"` (or `[ml-gpu]`)

---

## What the plugin changes

| Key       | Value                                                            |
|-----------|---------------------------------------------------------------------|
| `model`   | `model_name` of the winning routing target                           |
| `routing` | `plugin`, `target_name`, `similarity` (mean cosine of the match)     |

Everything else in the payload is untouched. A rejected match (similarity below threshold) and a request without
extractable text leave the payload **as it is**, so `model` stays `"auto"` in both cases.

```json
{
  "model": "qwen3.6:35b",
  "routing": {
    "plugin": "semantic_biencoder_routing",
    "similarity": 0.8224,
    "target_name": "code-generation"
  }
}
```

The log line that explains every decision:

```text
SemanticBiEncoderRouting: text='Napisz funkcję w Pythonie, która parsuje plik CSV' target='code-generation' similarity=0.8224 -> model=qwen3.6:35b
```

---

## Quick start

### 1. Install with the ML extras

```bash
# from a checkout of this repository, into the environment the router runs in
pip install -e ".[ml]"        # CPU FAISS + sentence-transformers
pip install -e ".[ml-gpu]"    # CUDA FAISS instead — never both FAISS builds at once
```

Without these imports this plugin **does not start**: the constructor raises and the router refuses to boot.

### 2. Load the plugin into the router

```bash
export LLM_ROUTER_UTILS_PLUGINS_PIPELINE="semantic_biencoder_routing"
```

Resolved through `llm_router_plugins.utils.registry.MAIN_UTILS_REGISTRY`, instantiated once per process, then run on
every prepared payload before guardrails and provider selection.

### 3. Make the embedding weights available

`embedding_model` is a Hugging Face id **or a local directory**. `google/embeddinggemma-300m` is gated behind the Gemma
license, so either accept it and `huggingface-cli login`, or point at a local copy:

```json
{ "embedding_model": "/models/google/embeddinggemma-300m" }
```

or without touching the file:

```bash
export LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_MODEL="/models/google/embeddinggemma-300m"
```

A local path also removes the startup download, which matters for air-gapped or restart-heavy deployments.

### 4. Declare the models in the router model config

- the **`auto`** alias, backed by a `builtin` provider (the client asks for it, this plugin replaces it);
- **every `model_name` used by a routing target** (`qwen3.6:35b`, `gpt-oss:120b`, …) as a normal reachable model.

```jsonc
// fragment of the llm-router models config
"semantic_routing": {
  "auto": {
    "providers": [
      { "id": "semantic-routing__auto", "api_type": "builtin", "api_host": null,
        "api_token": "", "model_path": "", "input_size": 0, "weight": 1.0,
        "tool_calling": false }
    ]
  }
}
```

### 5. Verify

```text
Embedding model loaded successfully.
Index built: 6 targets, 79 total embeddings.
SemanticBiEncoderRouting: text='…' target='code-generation' similarity=0.8224 -> model=qwen3.6:35b
```

Startup of the shipped config (6 targets, 79 vectors) takes about 9 s on CPU including the model load. See
[Verifying the plugin](#verifying-the-plugin) for a runnable check.

---

## How it works

### Startup (constructor)

1. `SemanticBiEncoderConfig.from_file()` — bundled JSON, or the `…_CONFIG` env var (raw JSON string **or** a path).
2. `_override_from_env()` — `MODEL`, `TARGETS`, `CHUNK_SIZE`, `CHUNK_OVERLAP` (a new frozen config instance).
3. `_validate_args()` — embedding model present, at least one target, sane chunking.
4. `build_embedding_router()` — loads the sentence-transformers model with `device="cpu"`, `trust_remote_code=True`,
   then either loads a persisted index or builds one:
   - each target is indexed as its own documents: `"Target: {name}. {description}"` plus every `examples` entry, each
     split into overlapping chunks (`chunk_size` 256, `chunk_overlap` 64) — so the target **name** is embedded too;
   - every chunk is embedded, L2-normalized, and added to a FAISS index (inner product on unit vectors = cosine);
   - `doc_id → target_name` goes to a pickled docstore.
5. With `PERSIST_DIR` / `vector_store_path` set, the index is written to `index.faiss` + `docstore.pkl` (or reloaded).

### Request path

```text
should_route(model == "auto") ─▶ extract_user_text ─▶ embed query ─▶ FAISS top_k ─▶ mean per target ─▶ threshold ─▶ annotate
```

- **Text** comes from `llm_router_plugins.utils.text_extractor.extract_user_text`, in priority order:
  `messages[-1].content`, `user_last_statement`, `query`, `prompt`, `input`. No text → warning + untouched payload.
- **Long queries** are handled inside the shared router: if the text exceeds the embedding model's `max_seq_length`
  (2 048 tokens for EmbeddingGemma), it is split into overlapping token windows, at most the first
  `MAX_QUERY_WINDOWS = 4` are encoded, and the window vectors are averaged and renormalized — so a huge prompt costs a
  bounded amount of compute and stays on the same cosine scale.
- **Aggregation**: the `top_k` hits are grouped per target and their scores **averaged**; the highest average wins.
- **Acceptance**: the match is used only when `similarity >= similarity_threshold`, otherwise the payload is left
  untouched (logged at INFO). With the shipped `similarity_threshold: 0.0` essentially every request is rerouted.

### Measured behaviour

Shipped config (`google/embeddinggemma-300m`, `top_k: 1`, `threshold: 0.0`), real numbers:

| Input                                              | Target             | Similarity | Model           |
|----------------------------------------------------|--------------------|------------|-----------------|
| `Napisz funkcję w Pythonie, która parsuje plik CSV` | `code-generation`  | 0.8224     | `qwen3.6:35b`   |
| `Napisz wiersz o jesieni w górach`                  | `creative-writing` | 0.7612     | `gpt-oss:120b`  |
| `Oblicz odchylenie standardowe dla tego zbioru`     | `math-analysis`    | 0.7096     | `qwen3.6:35b`   |
| `Skonfiguruj nginx jako reverse proxy z SSL`        | `system-admin`     | 0.7056     | `gpt-oss:120b`  |
| `Ile jest stolic w Europie?`                        | `data-science`     | **0.3518** | `qwen3.6:35b`   |
| `aaaa bbbb cccc dddd`                              | `code-generation`  | **0.3713** | `qwen3.6:35b`   |

The last two rows are the whole argument for setting a threshold: unrelated and nonsense inputs still get a target —
at roughly half the similarity of a real match. With `similarity_threshold` at `0.5`, both would keep asking `auto`.

---

## Configuration

### Config file layout

```jsonc
{
  "description": "what this config is for (documentation only)",
  "embedding_model": "google/embeddinggemma-300m",   // top level, Hub id or local path
  "settings": {
    "chunk_size": 256,
    "chunk_overlap": 64,
    "similarity_threshold": 0.0,                     // accept gate; NO env override
    "top_k": 1,                                      // neighbours averaged; NO env override
    "vector_store_path": ""                          // "" = in-memory index only
  },
  "routing_targets": [
    {
      "name": "code-generation",
      "model_name": "qwen3.6:35b",
      "description": "Model specialized for code-related tasks: writing, debugging, refactoring.",
      "examples": ["Write a Python function to sort…", "Popraw błąd w tym kodzie C++…"]
    }
  ]
}
```

Required keys: top-level `embedding_model`, `settings`, `routing_targets`; inside `settings` all four of
`chunk_size`, `chunk_overlap`, `similarity_threshold`, `top_k`; inside each target `name`, `model_name`,
`description` (`examples` is optional but effectively decides quality). `vector_store_path` is accepted at top level or
under `settings`.

Two configs ship in the package: `semantic_biencoder.json` (Hub id, in-memory index) and `semantic_biencoder-pk.json`
(local weights, persisted index) — the latter is a machine-specific example, not the default.

### Environment variables

Prefix `LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_`, applied at construction (restart to apply):

| Variable              | Effect                                                                        |
|-----------------------|--------------------------------------------------------------------------------|
| `…_CONFIG`            | full config: raw JSON string **or** path to a file (wins over the bundled file) |
| `…_MODEL`             | embedding model (Hub id or local directory)                                     |
| `…_TARGETS`           | pipe-separated whitelist of target names; unknown names are dropped silently     |
| `…_CHUNK_SIZE`        | token chunk size used when indexing targets                                     |
| `…_CHUNK_OVERLAP`     | token overlap between chunks                                                    |
| `…_PERSIST_DIR`       | directory for `index.faiss` + `docstore.pkl`                                     |

`similarity_threshold` and `top_k` are **config-only** — there is no env override for them. Tune them by editing the
JSON, or hand the whole config to the process via `…_CONFIG` (which accepts an inline JSON string, so it also works in
a container without a volume).

### Startup failures

Every one of these raises at plugin construction, i.e. the router does not come up:

- `…_CONFIG` points at a missing file or invalid JSON (`FileNotFoundError`, `json.JSONDecodeError`);
- a required key is missing (`KeyError` names the key and the available ones);
- no `embedding_model`, no targets, `chunk_size <= 0`, `chunk_overlap < 0`, `top_k < 1` (`ValueError`);
- `faiss` / `sentence-transformers` not importable → `SemanticBiEncoderRouting: the sentence-transformers / FAISS
  dependencies are not installed — install them to enable semantic routing`;
- the embedding model cannot be loaded (missing weights, no network, license not accepted, broken `transformers`
  install);
- the built index contains no vectors — every target is empty (no `description`, no `examples`).

---

## Tuning and gotchas

- **`similarity_threshold: 0.0` means "always reroute".** Measured: garbage scores 0.37 and a wrong target 0.35, while
  real matches sit at 0.70–0.82. A threshold of `0.5`–`0.6` turns the plugin into "route when we are confident,
  otherwise leave `auto` alone" — usually what you want in production. Note the rejection leaves `model` as `auto`, so
  the router's own default for `auto` must be a model you are happy with.
- **`top_k: 1` is a single-vote decision.** With `top_k: 3` the scores of three chunks are averaged per target, which
  damps one unlucky example; it costs nothing at query time. Consider `top_k: 3`–`5` once you have many examples per
  target.
- **`description` + `examples` *are* the classifier.** A target with vague text wins or loses randomly. Write examples
  the way users actually phrase requests (the shipped config mixes Polish and English for exactly this reason), and
  keep targets distinguishable: `data-science` vs `math-analysis` overlap in the shipped set, which is why
  "how many capitals are in Europe?" landed on `data-science`.
- **Name targets descriptively.** The indexed text starts with `Target: {name}. `, so `code-generation` contributes
  signal while `t1` or `v2` contributes noise.
- **Renaming or deleting a target with a persisted index can produce `model: "unknown"`.** The loader detects a
  *dimension* mismatch and rebuilds, but it cannot detect that `docstore.pkl` belongs to an older target list: an old
  `doc_id` whose target name no longer exists maps to `model_name = "unknown"`, which is then written into
  `payload["model"]`. After changing targets, delete `index.faiss` + `docstore.pkl` (or use a fresh
  `…_PERSIST_DIR`).
- **The embedding model sees 2 048 tokens.** `chunk_size` above that buys nothing; over-length queries are windowed
  (first 4 windows), so very long first messages are represented by their head.
- **CPU inference.** The model loads with `device="cpu"` regardless of the FAISS build; each routed request costs one
  embedding pass (up to 4 windows). The router process pays a one-off ~9 s startup and a few hundred MB of RSS.
- **EmbeddingGemma does not support `float16` activations**, and its Hub weights are gated. Use `float32`/`bfloat16`
  and a local copy or a logged-in HF cache.
- **`…_TARGETS` typos are silent.** An unknown name in the whitelist is dropped; if every name is wrong, the whitelist
  is ignored and the full target list is used.
- **This plugin and `simple_semantic_routing` both trigger on `auto`.** In one pipeline the first rewrites `model` and
  the second is a no-op. Also note the trigger check differs slightly: this plugin trims (`" auto "` matches),
  `simple_semantic_routing` compares strictly.
- **Unlike `agentic_routing_codex`, nothing here is fail-open.** A FAISS or embedding failure at request time
  propagates instead of passing the request through, and a broken config stops the router. That is deliberate for a
  routing-critical plugin, but it means the embedding stack is on your request critical path — keep the weights local.
- **Target names are the routing vocabulary.** They appear in `routing.target_name` and in the logs, so name them for
  the person who will be grepping them.

---

## Verifying the plugin

### Live check (loads the embedding model)

```python
"""Public-API check of the Semantic BiEncoder routing plugin."""
import logging
import os

# point at a local copy of the weights, or drop it and let sentence-transformers
# download google/embeddinggemma-300m from the Hub (gated — needs a login)
os.environ["LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_MODEL"] = (
    "/models/google/embeddinggemma-300m"
)

logging.basicConfig(level=logging.INFO, format="%(message)s")

from llm_router_plugins.utils.routing.semantic_biencoder.semantic_biencoder_routing import (
    SemanticBiEncoderRoutingPlugin,
)

plugin = SemanticBiEncoderRoutingPlugin(logger=logging.getLogger("routing"))

TEXTS = [
    "Napisz funkcję w Pythonie, która parsuje plik CSV",
    "Oblicz odchylenie standardowe dla tego zbioru danych",
    "Napisz wiersz o jesieni w górach",
    "Skonfiguruj nginx jako reverse proxy z SSL",
    "Ile jest stolic w Europie?",
]
for text in TEXTS:
    out = plugin.apply({"model": "auto", "messages": [{"role": "user", "content": text}]})
    routing = out.get("routing", {})
    print(f"{text[:42]!r:46} -> {out['model']:14} {routing.get('target_name'):18} "
          f"similarity={routing.get('similarity', 0):.4f}")
```

Expected output (shipped config, `google/embeddinggemma-300m`):

```text
Embedding model loaded successfully.
Index built: 6 targets, 79 total embeddings.
SemanticBiEncoderRouting: text='Napisz funkcję w Pythonie, która parsuje plik CSV' target='code-generation' similarity=0.8224 -> model=qwen3.6:35b
SemanticBiEncoderRouting: text='Oblicz odchylenie standardowe dla tego zbioru danych' target='math-analysis' similarity=0.7096 -> model=qwen3.6:35b
SemanticBiEncoderRouting: text='Napisz wiersz o jesieni w górach' target='creative-writing' similarity=0.7612 -> model=gpt-oss:120b
SemanticBiEncoderRouting: text='Skonfiguruj nginx jako reverse proxy z SSL' target='system-admin' similarity=0.7056 -> model=gpt-oss:120b
SemanticBiEncoderRouting: text='Ile jest stolic w Europie?' target='data-science' similarity=0.3518 -> model=qwen3.6:35b
'Napisz funkcję w Pythonie, która parsuje p'   -> qwen3.6:35b    code-generation    similarity=0.8224
'Oblicz odchylenie standardowe dla tego zbi'   -> qwen3.6:35b    math-analysis      similarity=0.7096
'Napisz wiersz o jesieni w górach'             -> gpt-oss:120b   creative-writing   similarity=0.7612
'Skonfiguruj nginx jako reverse proxy z SSL'   -> gpt-oss:120b   system-admin       similarity=0.7056
'Ile jest stolic w Europie?'                   -> qwen3.6:35b    data-science       similarity=0.3518
```

Re-run it with a `PERSIST_DIR` to confirm the second startup logs `FAISS index loaded from … (79 vectors)`.

### Router smoke test

```bash
curl -sS http://<router-host>:8081/v1/chat/completions \
  -H "Authorization: Bearer $LLM_ROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"Napisz testy jednostkowe dla modułu płatności"}]}' \
  | head -c 300
```

### Automated tests

```bash
python -m pytest tests/test_semantic_biencoder_routing.py -q   # config, index, persistence
python -m pytest tests/test_routing_common.py -q               # shared embedding router
```

---

## Troubleshooting

| Symptom                                                     | Cause                                                                | Fix                                                     |
|--------------------------------------------------------------|------------------------------------------------------------------------|-----------------------------------------------------------|
| Router refuses to start with a `…_CONFIG` error               | bad path or invalid JSON in the env var                                | the value may be a path **or** an inline JSON string       |
| `sentence-transformers / FAISS dependencies are not installed` | ML extras missing in the router environment                           | `pip install -e ".[ml]"`                                    |
| Model load fails at startup                                    | gated weights, no HF login, no network, unreadable path                | accept the license + `huggingface-cli login`, or use a local dir |
| Nothing is rerouted                                             | plugin not in `LLM_ROUTER_UTILS_PLUGINS_PIPELINE`, or `model` ≠ `auto` | check the pipeline and the incoming model string           |
| Every request is rerouted, even nonsense                       | `similarity_threshold: 0.0` (the shipped default)                       | set `0.5`–`0.6` in the config                               |
| Correct target, low similarity (0.4–0.6)                       | examples do not resemble real traffic                                  | add real user phrasings to that target's `examples`        |
| Wrong target with high similarity                              | overlapping target definitions                                         | rewrite descriptions so targets are disjoint, move examples |
| `payload["model"]` becomes `"unknown"`                          | stale persisted index whose docstore names removed targets             | delete `index.faiss` + `docstore.pkl`, restart              |
| `no text content found` warning                                  | payload uses a field outside the extraction priority list              | send `messages[-1].content`, `query`, `prompt` or `input`    |
| Requests fail with provider 4xx after routing                     | a target `model_name` is not declared in `LLM_ROUTER_MODELS_CONFIG`     | declare every target model                                  |
| Second startup is as slow as the first                          | no `PERSIST_DIR` / `vector_store_path`, so the index is rebuilt          | set `…_PERSIST_DIR` (and clear it after config changes)       |
| Slow first routed request                                     | embedding pass on CPU                                                    | warm the plugin at startup, keep queries short                |

---

## Reference

### Module map

| Module                                | Responsibility                                                          |
|---------------------------------------|---------------------------------------------------------------------------|
| `semantic_biencoder_routing.py`       | plugin: trigger gate, text extraction, threshold, annotation, env overrides |
| `config.py`                           | `SemanticBiEncoderConfig` (JSON schema, required keys, validation)         |
| `embedder.py`                         | backward-compatibility shim re-exporting the shared `EmbeddingRouter`      |
| `../embedder.py`                       | the shared BiEncoder + FAISS router (index build, windowed queries, persistence) |
| `../common.py`                         | `should_route`, `annotate_routing`, `build_embedding_router`, config base  |
| `../target.py`                         | `RoutingTarget` (`name`, `model_name`, `description`, `examples`)          |

### Defaults at a glance

| Key                            | Shipped value                     |
|--------------------------------|------------------------------------|
| trigger                        | `auto` (after `strip()`)            |
| embedding model                | `google/embeddinggemma-300m`, CPU |
| `similarity_threshold`         | `0.0` (config-only)                 |
| `top_k`                        | `1` (config-only)                   |
| `chunk_size` / `chunk_overlap` | `256` / `64`                        |
| targets                        | 6 → 79 index vectors                |
| query window cap               | `MAX_QUERY_WINDOWS = 4`             |
| persistence files              | `index.faiss`, `docstore.pkl`       |
| payload keys written           | `model`, `routing`                  |

### See also

- [Semantic Routing Plugins reference](../README.md#2-bi-encoder-semantic-routing-embedding-based) — §2 of the shared
  routing README: index-building walkthrough, per-target score aggregation example and persistence details.
- [`simple_semantic`](../simple_semantic/README.md) — dependency-free keyword + complexity routing on the same trigger.
- [`agentic_routing/codex`](../agentic_routing/codex/README.md) — Codex CLI work-mode routing, deterministic-first and
  fail-open, reusing this plugin's embedding router.
- [Root README §2.7](../../../../README.md#27-semantic-routing-model-selection) — plugin overview and triggers.
