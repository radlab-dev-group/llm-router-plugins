# Learning in Agentic Routing — how the router learns, where the data comes from, and what gets saved

This document describes, end to end, every mechanism in `agentic_routing` that
adapts behaviour over time or that consumes labelled data — i.e. everything
that sits between "static configuration" and "a model that changes on its
own". It covers the Codex routing plugin (`agentic_routing_codex`), the only
agentic plugin with a learned component, and — because the two are designed to
be read together — the sibling `agentic_knowledge` learned path, which is the
only place in the repository where labels from live traffic are fitted into
weights.

The short version, before the details:

| Mechanism | Where it lives | Speed | What is "learned" | What is saved |
|---|---|---|---|---|
| Embedding index build | `utils/routing/embedder.py` | offline, at plugin startup (or manually) | nothing — a frozen set of vectors over hand-written mode descriptions/examples | `index.faiss` + `docstore.pkl` |
| Offline calibration | `codex/evaluation.py` + `scripts/codex-tune-eval.sh` | offline, human-driven | threshold/margin (and, by human decision, descriptions/examples) | JSON reports + a new config file |
| Session memory | `codex/state.py` | online, per request | a *reliable phase* per session and a *carried decision* per command generation | Redis hash (versioned, TTL-bounded) |
| Knowledge learned path | `knowledge/agentic_knowledge/learning.py` | online counting + offline fitting | per-document Beta posterior, per-scope ridge logistic regression, per-source bandit | Redis: training stream, label corpus, model snapshots + a per-scope rollback history, reputation hashes, arm hash |

`agentic_routing_claude_code` has **no** learning at all: it rewrites
`payload["model"]` by model-name match and exits. It is mentioned only as a
boundary.

---

## 1. What "learning" means in this codebase

There are exactly four places where behaviour changes as a function of data
that was *observed* rather than *written*, and they run at deliberately
different speeds. Keeping them separate is a design decision, not an accident:

1. **The embedding index is not a trained model.** It is a biencoder (a frozen
   sentence-transformer checkpoint) applied to a small, human-maintained text
   corpus that lives in the plugin JSON config. Building the index is a
   *compilation* step: same inputs, same vectors, every time. The "model"
   part (the embedding weights) is never touched; only the index of
   *exemplars* is rebuilt. This is documented in §4–§5.

2. **Calibration is the only loop that changes decision parameters.**
   `threshold` and `min_margin` are the numbers the semantic layer compares
   its cosine similarities against. They are not fitted by gradient descent;
   they are searched on a labelled corpus with a deterministic grid, selected
   on a calibration split and verified on a holdout split. The loop is
   human-in-the-loop: the script proposes, the operator disposes. This is
   documented in §6–§7.

3. **Session memory is online but not statistical.** It remembers, per
   session, the last reliable phase and the last accepted semantic decision,
   inside the *generation* (the current user command) that produced them. It
   is a bounded cache with a compare-and-set lock, not a model: it has no
   weights, no distribution, no update rule beyond "a newer reliable signal
   replaces the old one". Its learning is the learning of *continuity*, not
   of *prediction*. This is documented in §8.

4. **The knowledge learned path is the only true statistical learner in the
   repository.** It consumes graded labels produced from live traffic
   ("the fix was verified", "the same failure came back"), maintains a Beta
   posterior per document, fits a ridge logistic regression per repository
   scope, and runs a bandit over source arms. Nothing trains on the request
   path; the request path only reads a snapshot and appends to a log. This is
   documented in §9–§12.

A request that reaches the router is never failed because of any of these
mechanisms: every one of them fails open to the static configuration.

---

## 2. Where the data comes from

### 2.1 The mode table (the "training corpus" of the semantic layer)

`llm_router_plugins/resources/routing/agentic_routing_codex.json`, section
`codex_modes`. Eight modes are defined; six of them participate in semantic
routing (`aux_title` and `compaction` are resolved by request class and never
indexed — `CLASS_ROUTED_MODES` is excluded in
`CodexRoutingPlugin._build_router`).

For every mode the JSON carries the text that becomes the index:

| mode | model | description (chars) | examples | example chars | keywords | phrases | patterns |
|---|---|---|---|---|---|---|---|
| `plan` | `qwen/Qwen3.8-Flash-Next` | 356 | 25 | 1546 | 15 | 25 | 4 |
| `implement` | `qwen/Qwen3.8-Flash-Next` | 515 | 25 | 1281 | 23 | 31 | 5 |
| `test` | `qwen/Qwen3.8-27B` | 443 | 25 | 1218 | 11 | 30 | 5 |
| `git_review` | `qwen/Qwen3.8-27B` | 521 | 29 | 1377 | 24 | 69 | 9 |
| `review` | `qwen/Qwen3.8-Flash-Next` | 506 | 36 | 1880 | 16 | 40 | 4 |
| `debug` | `qwen/Qwen3.8-Flash-Next` | 408 | 29 | 1375 | 27 | 30 | 5 |
| `aux_title` | `qwen/Qwen3.8-27B` | 152 | 10 | 432 | 0 | 0 | 0 |
| `compaction` | `qwen/Qwen3.8-27B` | 206 | 10 | 651 | 0 | 0 | 0 |

That is the entire semantic "training set": 6 descriptions plus 169 examples
for the indexed modes, in Polish and English, written by hand. Each mode's
`examples` list is the per-class exemplar set; each `description` is the
per-class label statement. The keywords/phrases/patterns of the same block
feed the *heuristic* layer, not the embedding layer.

The JSON also carries the decision parameters themselves (`settings.semantic.*`
and `settings.heuristic_*`) — see §3 — so the config file is both the data and
the parameters of the model it serves.

### 2.2 The labelled evaluation corpus

`tests/data/codex_routing_quality.json` — 73 cases, schema version 1:

- splits: **10 calibration, 63 holdout** (a session or a `sequence` may never
  span both splits; the loader rejects a corpus that does);
- expected modes: `implement` 28, `test` 22, `git_review` 8, `review` 5,
  `plan` 4, `debug` 3, `aux_title` 2, `compaction` 1;
- 2 cases are `ambiguous` (no prefix before the decision determines a mode;
  they are excluded from accuracy and break labelled transition runs);
- 49 cases carry a `sequence` (9 distinct sequences), 50 carry `metadata`
  (session/thread/turn identifiers for replay isolation), 34 a
  `source_session`, 5 a `payload`.

A case is one **request prefix** — the payload the router saw at the moment a
decision had to be made — paired with the work mode a human reading only what
came *before* that decision would label. The file's own `description` states
the rules: labels describe the current action, not the eventual deliverable of
a compound task; a session belongs to one split; ambiguous prefixes are
excluded from accuracy and break transition runs; special requests are scored
separately; the historical baseline covers only its original case IDs; session
IDs are anonymized.

45 cases carry a free-text `provenance`: 6 name the exact log file and line
they came from (e.g. `"User prompt from
agents-conversation/codex/conv-01/default-session-prepare-tests.log:765;
minimal input reconstructed, not a full captured payload."`), 11 say plainly
that the input is reconstructed or a synthetic continuation (e.g. `"Synthetic
continuation of log-git-first-phase."`), and the rest explain why a case is
shaped as it is — including the ambiguous ones (`"A 401 from the embedding
backend, not a failing test: nothing before the decision says what the agent
does next, so the case is reported as ambiguous."`).

Two hard rules keep this corpus honest:

1. **Holdout must never leak into the index.** The dataset description says so
   out loud: "Holdout must not be copied into semantic examples." Calibrating
   against the holdout retires it and requires a new independent set.
2. **Historical routing decisions are not labels.** The replay always re-runs
   the *current* implementation; the report is a measurement of the code in
   the checkout, not of what the production router decided at record time.

### 2.3 Recorded traffic (unlabelled, for regression and case mining)

`agents-conversation/codex/` holds router debug logs with full request bodies
indented (the format `scripts/plugins/auto_codex/eval/extract_payloads.py`
requires — a one-line body is not recovered). The pipeline:

```
agents-conversation/codex/*.log
   │  extract_payloads.py      (one JSON line per request: payload + recorded decision)
   ▼
corpus.jsonl  (~793 requests ≈ 180 MB — scratch only, never committed)
   │  replay.py [--determinism|--semantic]   (re-run the current code over the corpus)
   ▼
decisions.jsonl
   │  compare_history.py / inspect_case.py   (diagnose what moved)
   ▼
per-source agreement + recorded→current confusion table
```

`replay.py` checks three invariants on every run: `apply()` raises nothing
(fail-open); an unrouted payload comes back byte-for-byte the same object; a
routed payload gains exactly `agent_mode` and `routing`. `--determinism`
replays the corpus backwards through a second plugin instance and asserts the
decisions are identical — a hidden dependence on request order would show up
here.

Recorded decisions are paired with payloads **in stream order**, so a payload
whose decision line never arrived keeps `recorded: null`, and an interleaved
decision can be mispaired; both cases are reported by `--list-unrouted`.
This directory measures *what moved*, not correctness — correctness belongs
to the labelled corpus in §2.2 and the metrics in §6.

### 2.4 Live traffic (the knowledge learned path only)

The `agentic_knowledge` plugin reads the work state the router has already
derived (`payload["routing"]`, `payload["agent_mode"]`, `payload["model"]`),
retrieves from the Semantic Search Engine, and injects a context block. From
that loop it produces the only labels the repository trains on:

- **Exposures** — every injected document, with its features, logged to the
  training stream with `label = 0.0`, *before* any verdict exists. Written
  whether or not learning is enabled, because a label can only attach to an
  exposure that was written down while it happened.
- **Verdicts** — on the *next turn* of the same session, the settle path
  (`plugin.py::_settle_feedback` / `_learn_from_verdict`) grades each injected
  document against the evidence of that turn (mined fix, repeated failure
  signature, touched paths, commands that went green). See §9.3 for the full
  label table.

The training record carries features and names, never content: no text, no
paths, no command, no absolute location. A record that carried a path would
undo the worker's masking "from the other side".

---

## 3. The configuration that is the model

`llm_router_plugins/resources/routing/agentic_routing_codex.json`. Precedence:
`…_CONFIG` env var (raw JSON or a file path, no silent fall-through) → the
supplied file → the bundled default. Environment variables then override the
file. Every override is logged.

### 3.1 The decision parameters (what calibration can move)

| Parameter | Shipped value | Role |
|---|---|---|
| `settings.trigger_model` | `auto_codex` | the `payload["model"]` value that activates the plugin |
| `settings.fallback_mode` | `implement` | the floor: a main turn never resolves to anything weaker than this |
| `settings.heuristic_enabled` | `true` | the keyword layer participates |
| `settings.heuristic_min_score` | `3.0` | minimum keyword score to accept a heuristic match |
| `settings.heuristic_min_margin` | `1.0` | required lead over the runner-up; ties always abstain |
| `settings.heuristic_weights` | keyword 1.0, phrase 2.0, pattern 3.0 | default signal weights; per-keyword weights and `":w"` phrase suffixes take precedence |
| `settings.heuristic_negation_pattern` | (PL/EN prohibition regex) | local action prohibitions (`nie uruchamiaj`, `do not run`, …); scope ends at sentence boundaries, contrastive `ale`/`but`, or a new positive action; an empty regex disables filtering |
| `settings.classify_max_chars` | `4000` | parser history budget |
| `settings.semantic.enabled` | `true` | master switch of the embedding layer |
| `settings.semantic.threshold` | `0.44` | minimum cosine similarity for a semantic acceptance |
| `settings.semantic.aggregation` | `per_target_top_k` | required for Codex: a complete, balanced per-mode ranking |
| `settings.semantic.min_margin` | `0.002` | absolute floor on the lead over the runner-up |
| `settings.semantic.min_margin_relative` | `0.004` | the lead must also reach this fraction of the runner-up's own similarity — the term that travels with the embedding model |
| `settings.semantic.intent_max_chars` | `2000` | independent budget of the intent section |
| `settings.semantic.phase_max_chars` | `2000` | independent budget of the phase section |
| `settings.semantic.top_k` | `4` | best fragments per mode in the balanced ranking |
| `settings.semantic.chunk_size` | `256` | tokens per index chunk |
| `settings.semantic.chunk_overlap` | `64` | overlap between adjacent index chunks |
| `embedding_model` | `/mnt/data2/llms/models/community/google/embeddinggemma-300m` | the frozen biencoder checkpoint (local path in the shipped config) |
| `settings.vector_store_path` | `""` (empty) | FAISS persistence directory; empty means in-memory |

`settings.semantic` is validated on load. Five fields are **required** —
`aggregation`, `min_margin` (finite, in `[0, 2]`), `min_margin_relative`
(finite, in `[0, 1)`), `intent_max_chars` and `phase_max_chars` (positive
integers) — and a missing one is a `KeyError`, not a default: required values
are reported, not filled from another JSON file. The rest have defaults:
`enabled` true, `threshold` 0.44, `chunk_size` 256, `chunk_overlap` 64,
`top_k` 4.

### 3.2 Environment overrides

Prefix `LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_`. The full set:

- config: `…_CONFIG`
- routing: `…_TRIGGER`, `…_FALLBACK_MODE`
- heuristic: `…_HEURISTIC_ENABLED`, `…_HEURISTIC_MIN_SCORE`, `…_HEURISTIC_MIN_MARGIN`, `…_CLASSIFY_MAX_CHARS`
- semantic: `…_MODEL` (embedding model), `…_SEMANTIC_ENABLED`, `…_SIMILARITY_THRESHOLD`, `…_TOP_K`, `…_CHUNK_SIZE`, `…_CHUNK_OVERLAP`, `…_PERSIST_DIR` (wins over `vector_store_path`)
- per-mode signals: `…_MODE_<mode>_KEYWORDS` (pipe-separated)
- memory: `…_MEMORY_ENABLED`, `…_MEMORY_BACKEND`, `…_MEMORY_TTL_SECONDS`, `…_MEMORY_MAX_SESSIONS`, `…_MEMORY_MAX_EVENTS`, `…_MEMORY_MAX_CALLS`, `…_MEMORY_KEY_PREFIX`, `…_MEMORY_MAX_RETRIES`, `…_MEMORY_CARRY_DECISIONS`, `…_MEMORY_DECISION_SWITCH_MARGIN`, `…_MEMORY_DECISION_MAX_REQUESTS`, `…_MEMORY_DECISION_MAX_AGE_SECONDS`
- Redis connection (never in JSON, by design): `…_REDIS_HOST`, `…_REDIS_PORT`, `…_REDIS_DB`, `…_REDIS_PASSWORD`, `…_REDIS_PROTOCOL`, `…_REDIS_USERNAME`, `…_REDIS_SSL`, `…_REDIS_SSL_CA_CERTS`, `…_REDIS_SSL_CERTFILE`, `…_REDIS_SSL_KEYFILE`, `…_REDIS_SSL_CERT_REQS`, `…_REDIS_SOCKET_CONNECT_TIMEOUT`, `…_REDIS_SOCKET_TIMEOUT`

At startup the plugin also **lints** the signals and warns about settings that
can never score (an uncompileable or upper-case pattern, a mode without
`model_name`, `chunk_overlap` not below `chunk_size`) without changing
anything.

### 3.3 The session-memory policy

`settings.memory` (shipped values in parentheses): `enabled` (false — the
shipped config is stateless), `backend` (redis; `memory` is for isolated
replay and tests only), `ttl_seconds` (900), `max_sessions` (10000),
`max_events` (64), `max_calls` (32), `key_prefix`
(`llm-router:codex-routing`), `max_retries` (1), `carry_decisions` (true —
within the disabled section, so enabling memory also enables decision carry),
`decision_switch_margin` (0.008), `decision_max_requests` (12),
`decision_max_age_seconds` (180).

`decision_switch_margin` is deliberately larger than `semantic.min_margin`:
staying on the remembered mode is cheap, moving off it is not.

---

## 4. The "training" step: building the FAISS index

`llm_router_plugins/utils/routing/embedder.py::EmbeddingRouter`.

Startup sequence (`initialize()`, once per process):

1. `_load_model()` — `SentenceTransformer(embedding_model, device="cpu",
   trust_remote_code=True)`. The weights are the frozen checkpoint from §3.1;
   nothing is fine-tuned.
2. If `persist_dir` is set and `_load_index()` succeeds, the saved index is
   used and the build is skipped.
3. Otherwise `_build_index()` runs, then `_save_index()`.

`_build_index()`, per routing target (the six non-class modes):

1. texts = `["Target: {name}. {description}"] + examples` — the label
   statement first, then the exemplars, in order;
2. each text is split with `_split_into_chunks(text, 256, 64, tokenizer)` — a
   token-based sliding window (stride `chunk_size - overlap`); a text that
   fits in one window stays whole;
3. the chunks are encoded in one batch and **L2-normalised**, so the FAISS
   inner product is exactly cosine similarity;
4. the first batch creates `faiss.IndexFlatIP(dim)`; all chunks are added;
5. `_doc_store[doc_id] = target.name` records which mode owns each vector.

So the index is a bag of 175 short texts (one description plus 25–36
examples per indexed mode) turned into unit vectors. The model is the frozen
checkpoint; the *data* is the JSON; the build is deterministic.

### 4.1 Saving the model (the index)

`save_index()` writes two files into `persist_dir`:

- `{persist_dir}/index.faiss` — the FAISS flat inner-product index;
- `{persist_dir}/docstore.pkl` — the `doc_id → target_name` pickle.

On the next `initialize()`, `_load_index()`:

- loads both files (a missing directory or file falls through to a fresh
  build);
- encodes `"."` and compares the resulting dimension with the index's `d`;
- on a dimension mismatch (i.e. a different embedding model) it logs a
  warning and **rebuilds** instead of serving a wrong-dimensionality index;
- a corrupted index (no `.d`) or an un-pickleable docstore also rebuilds.

What forces a rebuild (per the shipped guidance): a change of the embedding
model, of the set of indexed modes, of any description or example, or of the
chunking parameters. `threshold`, `min_margin`, `top_k` and the query
budgets do **not** require re-embedding. The engine does not auto-invalidate
after a config change — the safe move is a new, versioned directory. The FAISS
index and the Redis session memory are independent; calibrating embeddings
must not flush Redis.

The evaluator deliberately ignores any persisted index: it rebuilds in
memory, so a stale saved index cannot hide a description/example change
(`metadata.persistent_index` is always `false` in a report).

---

## 5. Inference: how one request is decided

### 5.1 The cascade

`codex/classifier.py::CodexModeClassifier.classify`. Layers in fixed order;
the first one that can answer wins; a request queries the vector store at
most once, and only when every cheaper layer stayed silent.

| # | Layer | Source code | What it reads | Confidence reported |
|---|---|---|---|---|
| 1 | explicit | `SOURCE_EXPLICIT` | `agent_mode` / `codex_mode` / `metadata.agent_mode` in the payload | 1.0 |
| 2 | class | `SOURCE_CLASS` | request class: `compaction` > `aux_title` > `main` (strict priority) | 1.0 |
| 3 | collaboration | `SOURCE_COLLABORATION_MODE` | the `<collaboration_mode>` block: `# Plan Mode` | 1.0 |
| 4 | phase | `SOURCE_PHASE` | evidence from the active turn's assistant/tool activity (`detect_phase_evidence`), or the remembered evidence of the same generation | 1.0 |
| 4b | carried phase | `SOURCE_MEMORY` | the phase the memory carried forward when no fresh evidence exists | 1.0 |
| 5 | heuristic | `SOURCE_HEURISTIC` | keyword scoring of the latest user intent (`test`, `git_review`, `review`, `debug`) | `score / (score + 1)` — strength, not a calibrated probability |
| 6 | semantic | `SOURCE_SEMANTIC` | embedding cosine over the mode descriptions and examples | the cosine itself |
| 6b | carried decision | `SOURCE_MEMORY` | the last accepted semantic decision of this generation, when the current ranking is too weak to unsettle it | 1.0 when the ranking also accepted (the hint renews); the fallback mode's own cosine, read from the ranking, when the ranking refused |
| 7 | fallback | `SOURCE_FALLBACK` | nothing | 0.0 (with the fallback mode's own cosine where available) |

Two invariants protect the tail: a main turn never falls below
`fallback_mode`, and a deterministic layer's confidence is derived from its
own signal — the cascade never invents a probability it does not have.

### 5.2 What goes into the embedding

`codex/semantic.py::_build_semantic_parts`. Two independently budgeted
sections, each encoded separately and then combined:

- **intent** — the latest user text, capped at `intent_max_chars`;
- **phase** — the newest assistant `output_text` parts, plus a *structural*
  description of the current tool activity (`describe_activity`: which action
  ran, which recognised phase, success/failure, how many files a patch
  touched). The raw tool output is **never** embedded — a large read would
  otherwise let the topic of a file stand in for what the agent is doing.
  Repeated actions collapse into one clause; distinct actions are capped by
  `activity_description_limit`.

`route_context(parts)` encodes each section in its own query-window budget
(a section longer than the model context is split into overlapping windows,
capped at `MAX_QUERY_WINDOWS = 4` from the head), L2-normalises the mean of
the per-section unit vectors, and performs **one** FAISS lookup. A request
with no text at all never reaches the router.

### 5.3 The balanced ranking

With `aggregation = per_target_top_k` the router searches the **whole** index
(`k = ntotal`), groups the similarities by target, keeps the best
`min(top_k, min available fragments)` per mode, and averages them. Every
configured mode must be present in the ranking — a ranking that does not
cover all modes is rejected as incomplete (and the production layer fails
open). This is what makes a *lead over the runner-up* readable at all:
the scores are balanced, not a global top-k that can leave a mode unranked.

### 5.4 The acceptance rule

`codex/semantic.py::assess`, in this exact order:

1. no usable mapping, or a non-finite / out-of-`[-1,1]` similarity → refuse
   (`no ranking`);
2. the ranking is not a complete, duplicate-free, finite coverage of the
   indexed modes, or the reported top similarity does not match the ranking's
   own score for that target → refuse (`incomplete ranking`);
3. similarity below `threshold` → refuse (`below threshold`);
4. the named target is not a configured mode → refuse (`incomplete ranking`);
5. there is no runner-up at all → **accept**;
6. `margin <= 0` (a tie or an inversion) → refuse (`tie`);
7. accept when `margin >= max(min_margin, min_margin_relative ×
   runner_up_similarity)` — an absolute floor plus a term that scales with
   the scores, because cosines of one request live in a narrow band
   (0.55 and 0.9 do not mean the same thing);
8. otherwise, if the runner-up **is** the caller's fallback mode, accept with
   reason `runner-up is fallback`: refusing and accepting land on the same
   mode, so refusing would throw away only the ranking's own signal;
9. anything left → refuse (`ambiguous`).

Every refusal is reported as a *reason*, never as a probability. The decision
is written back by `annotate_routing`: `payload["model"]` is rewritten,
`payload["agent_mode"]` set, and `payload["routing"]` carries
`plugin`, `similarity`, `source`, `codex_class`, `collaboration_mode`,
`request_kind`, `thread_id`, `turn_id` plus the diagnostic fields
`reason`, `evidence`, `memory`, `semantic` only when they say something.

The plugin itself fails open at every stage: a payload it cannot parse or
classify is returned as the same object; a memory read failure re-runs the
classifier statelessly; a broken router disables the semantic layer and
nothing else.

---

## 6. The calibration loop: the only place parameters are learned from data

`codex/evaluation.py` — an offline replay of the **current** implementation
over the labelled corpus of §2.2. No generation model is called; the report
is always a replay of the code in the checkout, so a change in the cascade
shows up as a change in the numbers instead of being blended with recorded
history.

### 6.1 Variants

| Variant | What ran |
|---|---|
| `deterministic` | the cascade without the semantic layer — what a `semantic.enabled=false` deployment runs; fully reproducible |
| `stateful` | the deterministic cascade plus session memory, one fresh in-memory store per session sequence (a replay never touches real Redis) |
| `cascade` | the full cascade with the semantic layer — what a full deployment does |
| `semantic_only` | the semantic ranking alone for main turns (request-class routing kept for the helpers); a diagnostic, not an accuracy upper bound |

A router failure during evaluation is a `RuntimeError`, not a fallback: an
evaluation that silently degraded to the fallback would report a lookup
outage as a routing choice.

### 6.2 Metrics

Everything is computed on the **mode**, not the model — reassigning
`model_name` in the config moves the auxiliary model metrics and leaves the
mode metrics byte-for-byte identical.

- `mode_accuracy`, `main_mode_accuracy`, `model_accuracy`;
- per-mode `total` / `correct` / `predicted` / `recall` / `precision`;
- `mode_confusion` (expected → predicted) and `model_confusion`;
- `sources` (how often each cascade layer decided) and `fallback_reasons`;
- `semantic_acceptance_rate`, `fallback_rate`, `ambiguous_rate`;
- **transitions**, per session sequence: `pairs`, `expected_switches`,
  `actual_switches`, `missed_switches`, `unnecessary_switches`,
  `mean_switch_delay` (steps needed to reach the newly expected value; a
  prediction already right has delay 0), `censored_switches` (never caught up
  within the run). Special requests do not advance main-work time; an
  ambiguous prefix ends a labelled run.
- `mean_routing_ms` (diagnostic — two diagnostic passes and cache warm-up
  mean this is not a production latency benchmark).

Ambiguous cases are tallied separately and excluded from accuracy, so an
unlabeled prefix can neither inflate nor pollute a number.

### 6.3 The command

```
python -m llm_router_plugins.utils.routing.agentic_routing.codex.evaluation \
  --config llm_router_plugins/resources/routing/agentic_routing_codex.json \
  --dataset tests/data/codex_routing_quality.json \
  --split calibration|holdout|all \
  [--no-semantic] \
  [--baseline tests/data/codex_routing_baseline.json]
```

- `--no-semantic` replays the deterministic and stateful variants without
  loading any embedding model (milliseconds); semantic evaluation requires
  `semantic.enabled` and `per_target_top_k`.
- `--baseline` compares all variants against a frozen replay snapshot
  **only on shared case IDs and unchanged labels**; new cases never receive
  invented historical predictions, and a changed label aborts the run. The
  baseline's model names are remapped with today's config so mode metrics
  stay mapping-independent; the per-case regression test keeps an aggregate
  gain from hiding a loss on a formerly correct case.
- the report carries `config_sha256`, `dataset_sha256`, `baseline_sha256`,
  `metrics_version: 2`, the embedding model, the threshold and margin, the
  split, and `initialization_ms`.

`tests/data/codex_routing_baseline.json` is a frozen replay snapshot with
unverified provenance — a comparison point, not proof of the implementation
before the phase/memory changes.

### 6.4 The automated grid: `scripts/codex-tune-eval.sh`

This is the machine half of the calibration loop. It:

1. copies the source config and dataset into a fresh output directory and
   evaluates the **source config on calibration** first (that is the
   tie-breaker's anchor: an exact tie keeps the source config);
2. for every pair of `--thresholds` (default `0.40 0.44 0.48 0.52 0.56`) and
   `--margins` (default `0.001 0.005 0.02 0.05`), writes a candidate config,
   runs a **real replay on calibration only** (each point reloads the
   embedding model, so the default grid is expensive), and appends the
   result to `candidates.jsonl`;
3. ranks candidates by `cascade.main_mode_accuracy` descending, then fewer
   `missed_switches`, then fewer `unnecessary_switches`, then overall
   `mode_accuracy`, then the candidate index (the source config first), and
   writes `selection.json` + `selected-config.json`;
4. **only after freezing the selection** runs holdout for both the source and
   the selected config and writes `holdout-comparison.json`.

The holdout is never used to select thresholds — that is the whole point of
the two splits. The script does not train embeddings, does not improve
examples, does not tune heuristics, does not install dependencies, does not
touch production Redis, and does not deploy the winner: it produces configs,
reports and per-run logs, and the operator decides.

---

## 7. The human calibration procedure

The shipped guide is `CODEX_EVAL_HOWTO_PL.md` / `CODEX_EVAL_HOWTO_EN.md`; the
summary of what a calibration run actually changes:

1. **Freeze a reference point** — run the source config on both splits and
   keep the report.
2. **Find where errors occur** — the mode confusion matrix and the transition
   counters name the pairs and the sequences worth a look; `compare_history`
   and `inspect_case` mine individual disagreements from recorded traffic.
3. **Improve descriptions and examples** — the *data* of §2.1. This is the
   only change that moves the embeddings themselves; it requires a new index
   (new versioned `PERSIST_DIR`) and is calibrated on the calibration split
   only. Do not copy evaluation prompts into indexed examples.
4. **Select threshold and margin** — the grid of §6.4, on calibration only.
5. **Tune other parameters only afterward** — heuristic score/margin, the
   section budgets; these do not require re-embedding.
6. **Select by quality, not by acceptance count** — a configuration that
   accepts more but confuses more is worse; the selection order of §6.4 is
   written down precisely so this cannot be gamed by a single number.
7. **Verify on the untouched holdout** — the same embedding weights and the
   same dataset; if the holdout regresses, the calibration overfit.

What a calibration run may change: `threshold`, `min_margin`, `top_k`, the
section budgets, the heuristic score/margin, and (by human decision) the
mode descriptions and examples. What it may never do: touch the embedding
weights, mix splits, or write to production state.

---

## 8. Online learning: the session memory

`codex/state.py`. This is the only routing state that survives a request, and
it is a **continuity** memory, not a statistical one: no weights, no counts,
no distribution — the last *reliable* phase of the current command generation,
plus the last *accepted* decision of that generation.

### 8.1 Why it exists

Codex runs one action per model call. Between two calls the user's command
does not change, but the evidence does: the test suite has now run, the patch
has landed, the failure arrived. A stateless cascade re-reads whatever the
client happened to send and can lose the thread between two requests of the
same action — especially when the history arrives **incrementally** (the
result of a call it already showed, without the call). The memory carries a
reliable phase across that gap.

It lives in **Redis**, shared by every worker pointed at the same instance and
namespace; there is deliberately **no local-process cache**, because two
workers holding two ideas of the same session's phase is worse than no memory
at all. The `memory` backend exists only for isolated replay and contract
tests, and warns about exactly that.

### 8.2 Identity: what one memory belongs to

```
{key_prefix}:v{STATE_SCHEMA_VERSION}:{session_id}:{thread_id}:{agent_name}
```

`session_id`, `thread_id` and `agent_name` come from `client_metadata` (and
`x-codex-turn-metadata`); a component that is not `[A-Za-z0-9._-]{1,128}` is
replaced by `~` plus a 16-hex-character SHA-256 prefix, so a client-supplied
identifier can neither forge a key nor escape the namespace. Without both a
session and a thread identifier there is nothing to isolate and routing stays
stateless rather than guessing. `turn_id` is the **generation**: it separates
the current command from the one before it inside one thread, and it is what
turns an older or causally incomparable update into a *conflict* instead of an
overwrite. A new command resets the phase.

There is no automatic installation identifier in the key: independent
installations sharing a Redis instance must choose distinct
`MEMORY_KEY_PREFIX` values.

### 8.3 What a record holds (and what it never holds)

Serialized as compact JSON tagged with `STATE_SCHEMA_VERSION = 1`:

| field | meaning |
|---|---|
| `mode`, `kind`, `reason` | the remembered mode and the kind/reason of the evidence that produced it (`announcement`, `command`, `patch`, `test_failure`) |
| `gen` | `turn_id` of the command generation |
| `ev`, `fp` | identifier and 16-hex digest of the evidence (a digest of *structure*: tool name, command shape, status — never its content) |
| `pending`, `calls` | call identifiers opened and not yet settled, bounded by `max_calls` |
| `seen`, `seen_calls`, `history` | recently accounted event/call identifiers and the causal frontier of the transcript, bounded by `max_events` |
| `retired` | generations that have been superseded |
| `context` | digest of the window/context identity, so a changed context is visible |
| `ver`, `ts` | the compare-and-set version and the wall-clock second of the last evidence change |
| `dec`, `dec_ts`, `dec_seq`, `dec_n` | the remembered *decision* of this generation: its mode, when it was written, the request ordinal it was written at, and how many requests have leaned on it |
| `reset` | a bare "this generation started fresh" marker |

Never: model names, conversation text, tool output, file contents, credentials
or full session identifiers. A damaged, unknown-version or non-object record
is dropped and reported as a `miss` — the memory never evaluates or imports
what it read.

### 8.4 How a write is made safe

Every write runs the Lua script `_ATOMIC_WRITE_SCRIPT` server-side:

1. read the current record and its `ver`; if `ver != expected_version`,
   return `conflict` (optimistic lock — two workers cannot interleave a read,
   a merge and a write);
2. decide whether the write is **metadata-only**: if every evidence-bearing
   field (`gen`, `mode`, `kind`, `reason`, `ev`, `fp`, `action`, `reset`)
   equals the stored one, the record's original timestamp and remaining TTL
   are preserved (a decision-hint refresh must not make a session look
   fresher than it is), and an already-expired record becomes a `conflict`;
3. otherwise write with the full `ttl_seconds`;
4. maintain `{key_prefix}:v{N}:sessions`, a sorted set of session keys scored
   by expiry: prune entries already expired, and on overflow above
   `max_sessions` delete the oldest keys **whose name starts with this
   namespace's prefix** — pruning never touches anything outside it, and the
   keyspace is never scanned.

`read` returns `hit`, `miss`, `expired` (it lived here, the TTL won) or
`unavailable` (Redis could not be reached). `remember_decision` returns
`written`, `unchanged`, `skipped`, `conflict` or `unavailable` with a reason —
diagnostics, never exceptions.

The client is built lazily per process (`_connection()` re-arms on a pid
change), so a worker forked from a preloaded application never inherits its
parent's sockets; when memory is enabled, construction checks connectivity
with a `PING` on a temporary client and closes it. Redis errors also emit a
`WARNING` with a traceback in which `redis://…@` credentials and the
configured username/password are masked — treat those logs as sensitive for
anything else they may contain.

### 8.5 What the memory may and may not do

- It is consulted only **after** the current turn's own evidence
  (`SOURCE_PHASE` outranks it), and it never re-decides a phase on its own —
  it stores the phase the cascade actually routed with.
- It never stores a fallback, a weak/refused semantic match, a title request
  or a compaction: those are not facts about what the agent is doing.
- Replaying the same history is a **no-op**: identical evidence in the same
  generation does not bump the version, so a retry cannot age a session
  artificially.
- Causal continuity is verified before anything is carried. `resolve_memory`
  walks the activity of the active turn against `history` and classifies the
  relation as `fresh`, `extension`, `replay`, `incremental output` or
  `new generation`; a reordered, stale or incomparable history, a changed
  context, or a missing causal frontier is a **conflict**, and a conflict
  permits stateless routing from the payload but contributes no memory and
  overwrites nothing.
- A failed linked test command settles into `failure_mode` (`debug`): the
  pending call must match the stored action token and the configured test
  mode, and the execution status must parse as a failure.

### 8.6 The carried decision (the second, faster loop)

One user command becomes hundreds of model calls. If the ranking wavers by a
few thousandths of a cosine between two modes, those calls land on different
models. `decision_hint` answers inside the generation that produced it, and
only while `decision_requests < decision_max_requests` (12) and
`now - decision_at <= decision_max_age_seconds` (180). Fresh phase evidence
always outranks it.

`_carried_decision` then weighs the current ranking against the hint: the hint
survives when the ranking's target is the hint, when there is no margin at
all, or when the margin is below `decision_switch_margin` (0.008) — a
different mode must lead clearly to take over. A carried decision is written
by `_remember_hint`, which never resets a phase, never replaces a newer hint
(`decision_seq`), and keeps the record's evidence time and TTL. It is stored
*beside* the evidence, never instead of it, and reports itself in the
annotation as `source=memory`, `reason="carried decision"`,
`semantic=<outcome of the ranking that did not win>`.

### 8.7 Failure behaviour

`routing.memory` distinguishes `disabled`, `unconfigured` (invalid policy or
ENV, missing host, missing Redis client library), `unavailable` (connection or
operation failure), `hit`, `miss`, `expired`, `conflict`. `unavailable` is a
broken deployment; `miss` is an ordinary cold start. A memory error never
fails a request and never activates a process-local fallback; if the *read*
itself raises, the plugin re-runs the classifier without memory and marks the
decision `unavailable`. Connection and socket timeouts default to 1 s per
operation. Restarting a worker keeps the state; restarting Redis without
persistence loses it, which costs only the carried phase. To reset memory,
remove that installation's exact versioned keys and its session index — never
`FLUSHDB`, never a broad prefix.

---

## 9. The learned path next door: where labels come from and what fits them

`agentic_routing` produces the work state; `agentic_knowledge` (which sits
behind it in the pipeline and reads `payload["routing"]`, `payload["agent_mode"]`
and `payload["model"]` exactly as the router left them) is what turns traffic
into labels and weights. It is documented here because it is the only place in
this repository where a model is fitted, and because its features are the
router's own outputs.

One rule explains the whole shape: **nothing trains on the request path.** A
proxied request has a small read budget and must fail open, so the request
path only *reads* (one reputation hash, one snapshot `GET`) and *appends* (the
features it saw, to a training stream). Fitting happens in a separate process,
and what it publishes is a versioned, clamped, auditable snapshot. A model
that cannot beat the policy publishes the policy.

### 9.1 The three learners

| Level | Question | Model | Where it runs | How it is used |
|---|---|---|---|---|
| document | does this advice work? | Beta posterior with a half-life, sampled by Thompson's rule | online, as counting | a multiplier in `mult_min … mult_max` (0.3 … 1.5) |
| candidate features | which features predict that advice works? | ridge logistic regression, damped Newton, corpus-scaled ridge, Laplace spread per weight | offline, `llm-router-agentic-kb-learner` | a versioned snapshot read with one `GET` + 60 s local cache; the spread is the exploration term |
| source per mode | what is a character of this source's budget worth here? | Beta posterior per `(mode, kind)` with the character cost averaged in | online, as counting | a value-per-character; the hand-written policy `weight` is the prior for an untried arm |

All three return `1.0` when they do not know, and all three are clamped to the
same range the static counter already used — so Redis down, a refused
snapshot, or a corpus too small to fit all degrade to "the layer as it
behaves today". One asymmetry on purpose: with `learning.enabled: false` the
policy is **absent, not neutral**. A `1.0` from a disabled learner would still
steer, because the static vote it replaces already carries the counter
multiplier of `0.3 … 1.5`; so the read path checks the switch first, and the
annotation reports `enabled: false` with no `model_id` next to it. The
collection is not off with it: exposures keep landing in the training stream,
because a label can only attach to an exposure that was written down while it
happened.

### 9.2 The features (the training data, field by field)

`FEATURE_NAMES` — 15 numbers, in order: `bias`, `score`, `score_sq`,
`scale_rrf`, `scale_reranker`, `signature_match`, `path_match`,
`path_stem_match`, `command_match`, `age_days_log`, `upvotes_log`,
`downvotes_log`, `exposure_log`, `rank`, `size_share`.

`CandidateFeatures.vector()` builds them from what the assembler already holds:
the normalized score clamped to `[0,1]` and its square; one-hot of the scale
the engine reported the score on (`rrf`, `reranker`); whether the candidate's
signature matched; whether it names a touched path or a path stem
(`/4`); whether it declares the verification command; `log1p` of the document's
age in days, its upvotes, downvotes and exposures; the engine rank
(`/20`, capped); and the share of the mode's character budget the document
consumes. No text, no path, no command — features and short names only,
because the worker masks documents before indexing them and a training record
carrying a path would undo that from the other side.

Only a subset may ever move: `_LEARNABLE = (1, 5, 6, 7, 8, 9, 13)` —
`score`, `signature_match`, `path_match`, `path_stem_match`, `command_match`,
`age_days_log`, `rank`. A feature the corpus never varied (every document of
one repository is the same `kind`), or one another feature determines
(`score` vs `score_sq` when all scores sit in one range; the two scale
dummies when every hit came on the same scale), is **pruned** rather than
fitted, and the pruning is reported in the fit report (`collinear`) instead of
silent: splitting one effect across two arbitrary weights is not learning.
Correlation at or above `0.995` counts as one feature wearing two names.

### 9.3 The labels (how traffic becomes a target)

Written on the settle path, one turn after the injection, graded per document
against the evidence of that turn (its signature, its touched files, the
command that went green):

| label | when | weight in the fit |
|---|---|---|
| `+1.0` confirmed | the document names the very failure that was verified fixed this turn | 1.0 |
| `+0.5` plausible | the turn was resolved and the document shares a touched file or declares the command that just went green | 0.5 |
| `0.0` unattributed | the turn was resolved and this document has nothing in common with it — logged as an exposure, fitted as nothing | 0 |
| `-1.0` debunked | the same failure came back and the document was about it | 1.0 |
| `-0.3` rejected | the failure came back and the document is unrelated to it | 1/3 |
| `±1.0` coarse | the document names neither a signature nor a file (a plain text hit): the whole-block verdict, marked so a fit can tell the two kinds of credit apart | 0.5 |

Every record also carries `exploration`: whether the ranking that showed the
document drew the model's Laplace spread or took its mean. It is written with
the exposure rather than reconstructed afterwards, because the draw belongs to
the request — an off-policy reading of this corpus (IPS, a doubly-robust
estimate, or simply "how much of what we learned was paid for by exploring")
needs the policy that produced the vote, and that policy is gone by the time a
verdict does. A document ranked twice in one generation and explored once
counts as explored; a draw older than a generation's window of documents is
dropped, because it is a fact about a ranking nobody is holding.

`0.0` is not a downvote: it means "this turn proves nothing about this
document", and training it as a failure would teach the model that being shown
is bad. A coarse credit is worth half a counterfactual one (`_KIND_CREDIT`:
counterfactual 1.0, coarse 0.5); an unknown kind is credited fully, because a
new kind of label is a reason to look at the weighting, not to silently drop
labels.

The graded number is read as **direction plus weight**, not as a target: the
likelihood is Bernoulli, so `label >= 0` is an observation of success with
weight `label`, and `label < 0` an observation of failure with weight
`|label|`. Folding a signed label into the target makes the objective
unbounded (it keeps falling as `p` falls and the fit runs to the clip);
folding a graded label into the target as `0.5` states "p = 0.5" — the one
thing a corpus can state without teaching anything — while still counting as
labelled when the fit is gated. Both failures are documented in the code
because both are easy to ship.

### 9.4 Where the corpus lives, and what it takes to publish

- **Training stream** (`learning.stream_key`, default `agentic_kb:training`):
  a Redis Stream, `XADD … MAXLEN ~100000`, one record per exposure and per
  settled verdict. Bounded and lossy by design; written whether or not
  learning is enabled, because a label can only attach to an exposure that was
  written down while it happened.
- **Label corpus** (`{key_prefix}:labels:{scope}`, one Redis hash per scope,
  field = `(session, generation, document)`) the *accumulation* of settled
  judgements, deduplicated by construction, verdict-wins-over-exposure, TTL =
  4 × `half_life_seconds` (default 30 days ⇒ 120). The learner fits on the
  corpus, never on the batch it drained: the batch decides *which* scopes are
  worth refitting, the corpus decides what the fit says.
- **Cursor** (`{key_prefix}:learner:cursor`): the stream id the last pass
  stopped at; a pass resumes, folds what is new, fits, publishes, saves.
- **Snapshot** (`{key_prefix}:model:{scope}`): `schema`, `scope`, `model_id`
  (`{scope}:{unix}:{labelled}:{auc}`), `trained_at`, `samples`, `labelled`,
  `positive`, `holdout_auc`, `features`, `arms`, the weight vector `w` and the
  Laplace diagonal `diag`. A snapshot whose `features` differ from this
  build's `FEATURE_NAMES` is **refused**, not partially applied.
- **History** (`{key_prefix}:model_history:{scope}`): the snapshots a publish
  retired, newest first, deduplicated by `model_id`, capped at
  `snapshot_history_size` (8; 0 keeps only the live key). A rollback copies a
  row back to the live key and retires whatever it displaces, so a rollback
  is itself reversible; an id that is not in the history is refused.
- **Reputation** (`DocReputation`) per document: `a`, `b`, `e`, `t` in one
  hash at `{key_prefix}:rep:{document_name}`, TTL 4 × half-life, with a local mirror; **arms**
  (`{key_prefix}:arms`): `a`, `b`, `chars`, `n` per `mode|kind`.

Publishing requires all three gates: `labelled >= min_samples` (200),
`holdout_auc >= min_holdout_auc` (0.6), and weighted
`evidence >= min_evidence` (default 100). Any miss publishes a **neutral** snapshot —
weights at zero, reads as `1.0` everywhere — and the log line says which gate
refused (`too_few_labels`, `holdout_auc_below_floor`,
`evidence_below_threshold`). "The learner did not run" and "the learner
declined to publish" are two different sentences, and the report keeps them
apart.

The holdout split is **by session**, not by sample (sample-level splits leak:
the same session's injections share context), keeping at least one session on
each side; a scope with a single session trains on all of it rather than
scoring a model against itself. The ridge is corpus-scaled —
`l2 × n / 100` (default `l2 = 32`) — so the prior keeps the same strength of
belief at any sample size, and Newton steps are damped by backtracking on the
penalized log-likelihood, because an undamped step from zero throws every
weight (including the noise features') to `weight_clip` on a separable corpus.
The exploration term is the Laplace standard deviation of the probability
(delta method on the logistic link) computed with the true curvature
`p (1 - p)` at the optimum, scaled by `explore_coeff` (0.25), and applied on
`explore_rate` (0.1) of requests.

### 9.5 Running the learner

```
llm-router-agentic-kb-learner                       # one pass: drain, merge, fit, publish
llm-router-agentic-kb-learner --scope webapp        # one scope
llm-router-agentic-kb-learner --holdout 0.2 --seed 0
llm-router-agentic-kb-learner --max-records 50000   # cap one pass's stream reads
llm-router-agentic-kb-learner --window 20000        # corpus ceiling per scope
llm-router-agentic-kb-learner --reset-cursor        # ignore the saved cursor
llm-router-agentic-kb-learner --show                # print each touched scope's weights
llm-router-agentic-kb-learner --arms                # print the bandit's view per (mode, kind)
```

It needs Redis and nothing else — no engine, no embedder, no reranker — and it
is not a daemon: a pass is a pass, which makes `cron` the natural supervisor.
A pass takes a lock (`{prefix}:learner:lock`, `SET NX EX
learning.lock_ttl_seconds`, default four hours) before it drains anything, and
a pass that finds it held says so and exits without fitting: two overlapping
passes would read the same cursor, publish over each other, and leave two
histories with no record of which fit won. Release is compare-and-delete by
the token the pass was given, so a pass that ran past its own TTL cannot
delete the lock a later pass has since taken. `--force` writes past a held
lock when an operator has decided the pass will not finish; `--lock-ttl`
moves the crash window.
It writes nothing to the engine and touches no policy: the policy's thresholds
and source weights are never overwritten by a model; the model multiplies,
the policy clamps. That clamp is the brake an operator can read in a diff.

Learning is off by default (`learning.enabled: false`), and "off" is not
"empty": the learner has **no vote at all** — it is not consulted, its
snapshot is not read, and the static feedback counters keep their own
multiplier — while **the training log still runs**. Collecting with learning
off is the recommended first step: it costs one bounded stream write per
settled turn, changes no ranking (not the counters, not the order), and is the
corpus a later phase trains on.

### 9.6 Measuring the learned path: the `read_learned` replay

The same offline replay that measures the static layer
(`knowledge/agentic_knowledge/evaluation.py`, the command
`python -m llm_router_plugins.utils.knowledge.agentic_knowledge.evaluation
--dataset …`) carries two learned variants, next to `no_kb`, `read` and
`read_escalate`. `read_learned` walks the sessions in order with a real
`LearningPolicy` in memory — one per **scope**, because the holdout split is
by session and a corpus of one session has nothing to hold out — and the
anti-leak rule stays a rule about time: the model a request may see was
fitted on labels already earned when that request was scored, and on nothing
earned later. The replay's gates are tighter than the deployment's
(`min_samples: 12`, `min_evidence: 8.0`, `min_holdout_auc: 0.6`) because the
question is "would the learned path have ranked better", not "is this dataset
a statistically legal corpus"; determinism is kept with one seeded generator
per scope and a clock that is a counter, so a model id built from the second
a fit ran cannot move between two runs of the same dataset. A scope without
labels yet reads the neutral snapshot, which reorders nothing, so the variant
degrades to `read` rather than to a guess. `read_reputation` is the same
walker with the learner's other two loops held back: the document's Beta
posterior on its own, no fitted weights and no source arm, in a namespace of
its own. It exists because a learned variant that loses ground says nothing
about **which** of the three loops lost it, and the plan's phase-2 criterion
is about the reputation loop alone: `reputation_uplift_vs_read` is that
loop's number, and the difference between `model_uplift_vs_read` and it is
what the fitted model and the arms add on top.

What the report adds on top of the per-variant counters:

- `model_uplift_vs_read` — `recipe_hit_rate(read_learned) −
  recipe_hit_rate(read)`, with a bootstrap confidence interval resampled
  **by session** (200 draws, fixed seed), because the turns of one session
  are not independent evidence; one session reports a point and no interval,
  and says so instead of inventing one.
- `reputation_uplift_vs_read` — the same measurement for the reputation
  variant: the phase-2 gate, read before a fitted model is worth its cost.
- `char_efficiency` — `turns_saved / injected_chars_per_turn`: the number the
  layer is actually being paid for, because knowledge costs tokens.
- a per-scope appendix: how many times the fit ran, how many times it was
  accepted, which gates refused it, the last holdout AUC, and the model ids
  it published along the way.

The `no_kb` baseline is counted once for the whole report, not per variant,
because it is the counterfactual the other variants are measured against —
and `turns_saved` for any variant is the difference to that one baseline.

### 9.7 How much evidence each loop needs

| what | roughly when it starts to matter | what it does before that |
|---|---|---|
| document reputation | ~20–30 labels per scope | `1.0` under `min_observations`; the posterior is wide and the clamp holds it |
| per-scope model | ~1 000 labels per scope | a neutral snapshot is published; the layer is the static one |
| source bandit | ~100 labels per arm | the arm returns the policy weight from the JSON |

A scope with a few dozen labels is served mostly by the policy and the
reputation; the model's share grows with the evidence, through the Laplace
spread, rather than with a switch.

### 9.8 The response annotation: why this recipe was first

The read path answers from the response itself, not from a log nobody kept.
`payload` carries a `learning` block whenever the layer ran: `enabled`, the
serving `model_id`, the `holdout_auc` and `labelled` the snapshot was fitted
with, the `range` the product is clamped to, the bandit's arm values, and —
per candidate document — the multiplier it got and the three factors that
made it:

```jsonc
"learning": {
  "enabled": true,
  "model_id": "webapp:1763347200:1412:0.710",
  "holdout_auc": 0.71,
  "labelled": 1412,
  "range": [0.3, 1.5],
  "candidates": {
    "fixes/api-500.md": {"mult": 1.31, "rep": 1.2, "model": 1.08, "arm": 1.03, "explore": false}
  }
}
```

`mult = clamp(rep × model × arm)`: the document reputation (Beta posterior),
the per-scope ranker (with the Laplace spread when the request drew
exploration), and the source bandit's value-per-character for the candidate's
`mode|kind`. A disabled deployment reports `enabled: false` and no `model_id`
— the switch is visible next to the absence of a model, which is the point of
treating a disabled learner as absent.

### 9.9 The threshold grid: `llm-router-agentic-kb-tune`

The policy's own numbers — `min_similarity` (the retrieval floor),
`strong_recipe_score`, `loop_threshold`, and a factor on each source's
`weight` — are the operator's, and they stay the operator's: the command
measures a grid, reports, and writes nothing to the policy file. It replays
every point of the Cartesian product (the order is the order the operator
named, so point 17 of the report is reproducible as a tuple) and reports the
point that maximizes

```
recipe_hit_rate − λ · injected_chars_per_turn
```

under the constraint `escalation_precision ≥ floor` (the floor the operator
sets; a missing floor is `1.0`, i.e. "only escalate when it is the loop").
The selection is made on the calibration half and the holdout half is only
measured: `--holdout-session` names the sessions the operator already keeps,
`--holdout-fraction` splits them by a hash of `--seed` and the session id so
two runs split the same way, and a fraction never eats the calibration. Every
point carries what it earned on each half, and the report holds two delta
tables — the one the grid was chosen on, and the one it was judged on —
because a point selected on the sessions it reports has been fitted to them.
With no holdout named the report says so rather than printing a column of
zeros. The report is JSON + markdown with the winner and the deltas against
the shipped policy on the same sessions, so a pull request can quote it:

```
llm-router-agentic-kb-tune --dataset tests/data/eval.json \
  --min-similarity 0.25 0.35 0.45 --strong-recipe-score 0.5 0.6 0.7 \
  --loop-threshold 2 3 4 --weight-factor 0.8 1.0 1.2 \
  --lambda 0.001 --precision-floor 0.8 \
  [--holdout-fraction 0.3] [--seed 11]   # or --holdout-session ID, repeatable
  [--per-category-factors]   # one exponential grid per source category
  [--output tune-report/]
```

A point that misses the precision floor is reported with the reason
(`escalation precision below floor`) and is not selected; if no point is
eligible the report says so and names the best of the ineligible rather than
silently promoting one. The grid is deliberately small and explicit: a
search that can find a better policy on the dataset it was searched on has
found a policy that fits that dataset, and the holdout the operator keeps is
the one thing it cannot see.

### 9.10 The scoreboard's view of the learned state

`llm-router-agentic-kb-stats --learning` reads, per scope: the model in
force (`model_id`, holdout AUC, labelled, when it was trained), the history a
rollback could reach (with each row's own AUC and sample count), the corpus
it rests on (the label store's counts), the bandit's arms, and the
self-reinforcement monitors of the training log — the documents with the
most exposures and the heaviest of them — so the operator can see a
document's reputation and the model that serves it from one command. The
scoreboard rows themselves are unchanged; `--learning` adds a `learning`
block to `--json` for a dashboard to consume.

---

## 10. What is never learned

- **No model weights are updated anywhere in the routing path.** The
  embedding checkpoint is loaded, never fine-tuned; the heuristic weights, the
  mode table and every threshold come from configuration.
- **No online learning from outcomes in routing.** A routing decision is never
  scored by what happened next; there is no reward, no bandit and no gradient
  on the Codex cascade. The only online state is the session memory of §8,
  and it decays by TTL rather than by evidence.
- **No feedback from the answer's quality.** Evaluating answer quality, token
  cost or provider latency across the mapped models requires a separate
  generation experiment; nothing in the router measures it.
- **No automatic reconfiguration.** No loop writes a config file, restarts a
  worker, or rebuilds an index. Calibration produces artefacts; a human ships
  them. `llm-router-agentic-kb-tune` (§9.9) is the same discipline for the
  knowledge policy: it replays, ranks and reports, and never edits the file it
  measured against — a tune that rewrote the policy it scored has already
  answered its own question.
- **No unbounded authority for a learned number.** Every learned multiplier is
  clamped to a configured range, and the policy's own thresholds stay the
  operator's. A model's snapshot is refused outright when its `features` do
  not match this build's `FEATURE_NAMES`, and a rollback refuses an id that
  is not in the history it could honestly restore.
- **No cross-installation learning.** Session keys are namespaced per
  installation; knowledge scopes are per repository; nothing aggregates across
  deployments.
- **`agentic_routing_claude_code` learns nothing.** Its mapping from Claude
  Code tier aliases to gateway models is static configuration, matched on the
  model name alone.

---

## 11. Operating it

### 11.1 Which change requires which action

| Change | Rebuild the index? | Re-run calibration? |
|---|---|---|
| `threshold`, `min_margin`, `min_margin_relative` | no | selection is exactly this — calibration grid, then holdout |
| `top_k`, `chunk_size`, `chunk_overlap` | chunking: yes; `top_k`: no | yes, the ranking shape moves |
| a mode `description` or `examples` | **yes** (new versioned `PERSIST_DIR`) | yes, and never copy holdout prompts into examples |
| the set of indexed modes | **yes** | yes; the evaluator refuses an index missing configured targets |
| the embedding model | **yes** (dimension check rebuilds, but a same-dimension swap will not be caught — use a new directory) | yes, absolutely: thresholds are model-specific |
| heuristic keywords/phrases/patterns, `heuristic_min_score`, `heuristic_min_margin` | no | run `--no-semantic` first, then the full report |
| memory policy (`MEMORY_*`) | no | `stateful` variant in the report is the measurement |
| mode → `model_name` mapping | no | mode metrics are unchanged by construction; only the model appendix moves |

### 11.2 Reading a decision

One log line per routed request carries the whole story:

```
Codex routing: mode=test source=heuristic model=qwen/Qwen3.8-27B similarity=0.800 class=main reason=… evidence=command memory=hit semantic=accepted
```

and `payload["routing"]` carries the same fields plus the diagnostics of
§5.4. `source` says which layer learned enough to answer; `semantic` says what
the ranking supported even when it did not win; `memory` says whether the
session's continuity contributed. The cheapest health check on a deployment is
the distribution of `source` and `fallback_reasons` — the same two counters the
evaluator reports.

### 11.3 Costs and failure modes to expect

- **Semantic startup is expensive, per process**: the embedding model loads on
  CPU, roughly 25 s of startup in the recorded runs (`initialization_ms` in
  every report). A persisted index skips the encoding of the corpus, not the
  model load.
- **Query cost is bounded**: at most `MAX_QUERY_WINDOWS = 4` embedding passes
  for an over-length section, one FAISS lookup per request, and the lookup is
  skipped entirely whenever a deterministic layer answered.
- **A missing `faiss` / `sentence_transformers` install disables semantics and
  nothing else** (`_MISSING_DEPENDENCIES_MESSAGE` names the env var to set
  explicitly instead); the deterministic cascade is a supported production
  configuration, not a degraded one.
- **Redis down is a `miss`**, never an error: routing continues statelessly.
- **The 73-case corpus is a regression starting point, not statistical proof.**
  It is small, and the holdout has been seen before; read the per-mode
  confusion and transition counters, not just the accuracy headline.

### 11.4 Where to look for each piece

| Topic | File |
|---|---|
| index build/save/load, chunking, query windows, aggregation | `../embedder.py` |
| cascade order, carried phase/decision, confidence semantics | `codex/classifier.py` |
| cosine acceptance rule, section budgets, ranking validation | `codex/semantic.py` |
| keyword scoring, negation, deduplication, ties | `codex/scoring.py` |
| phase evidence taxonomy and activity description | `codex/phase.py`, `codex/phase_config.py` |
| payload normalization, request classes, metadata locations | `codex/payload.py` |
| config loading, env overrides, validation, linting | `codex/config.py` |
| session memory, keys, CAS script, decision hints, statuses | `codex/state.py` |
| replay, metrics, splits, baseline comparison, report metadata | `codex/evaluation.py` |
| the grid and its selection rule | `../../../../scripts/codex-tune-eval.sh` |
| corpus, provenance and labelling rules | `../../../../tests/data/codex_routing_quality.json` |
| recorded-traffic pipeline | `../../../../scripts/plugins/auto_codex/eval/` |
| the full calibration guide | `codex/CODEX_EVAL_HOWTO_PL.md`, `codex/CODEX_EVAL_HOWTO_EN.md` |
| the learned path (reputation, ranker, bandit, stream, corpus, snapshot, history, rollback) | `../../knowledge/agentic_knowledge/learning.py` |
| the learner process (`--report`, `--show`, `--arms`, `--history`, `--rollback`) | `../../../../llm_router_plugins/cli/plugins/utils/agentic_knowledge/learner.py` |
| label grading on the settle path | `../../knowledge/agentic_knowledge/plugin.py` |
| the counterfactual replay and its five variants (`no_kb`, `read`, `read_escalate`, `read_learned`, `read_reputation`), the uplift bootstrap, `char_efficiency` | `../../knowledge/agentic_knowledge/evaluation.py` |
| the threshold grid and its objective/constraint | `../../../../llm_router_plugins/cli/plugins/utils/agentic_knowledge/tune.py` |
| the scoreboard and `--learning` | `../../../../llm_router_plugins/cli/plugins/utils/agentic_knowledge/stats.py` |
| the learning policy in the JSON (`enabled`, gates, bounds) | `../../../resources/knowledge/agentic_knowledge.json` |

Tests that pin the behaviour described here:
`tests/test_codex_routing_quality.py` (evaluation plumbing and the per-case
regression rule, without loading a model),
`tests/test_codex_semantic_ranking.py` (the acceptance rule),
`tests/test_codex_routing_state.py` and `tests/test_codex_memory_configuration.py`
(memory semantics, CAS, conflicts, statuses),
`tests/test_codex_routing_phase.py` (evidence taxonomy),
`tests/test_agentic_routing_codex.py` (plugin contract, fail-open, annotation),
`tests/test_codex_tune_eval_script.py` (the grid's selection discipline),
and for the learned path next door:
`tests/test_agentic_knowledge_learning.py` (the learners, the gates, fail-open,
no content in a training record),
`tests/test_agentic_knowledge_learning_plugin.py` and
`tests/test_agentic_knowledge_learning_integration.py` (grading on the settle
path, the annotation, Redis-backed feedback),
`tests/test_agentic_knowledge_assembler.py` (the per-candidate multiplier and
its detail),
`tests/test_agentic_knowledge_evaluation.py` (the `read_learned` and
`read_reputation` replays, the uplift intervals, `char_efficiency`),
`tests/test_agentic_knowledge_metrics.py` (the `--learning` view),
`tests/test_agentic_knowledge_tune.py` (the objective, the constraint, and
that nothing writes the policy).
