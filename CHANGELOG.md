# Changelog

All notable changes to the `llm-router-plugins` package are documented here.
Versions follow the `MAJOR.MINOR.PATCH` scheme; a new entry is added for every
release tagged as `v<version>`.

## Summary

| Version | Changelog |
| ------- | --------- |
| Unreleased | **Added:** `agentic_knowledge` indexes the repository's own documentation — a heading-aware scan of its `*.md` files published as `doc` documents, republished on content change, superseded documents retired through a `docs_prune` control message (never on a truncated scan), plus the `llm-router-agentic-kb-docs` command and the `docs` policy section. Documentation now lives in `<prefix>_<scope>_docs`, is read per work mode through `doc_types`, is searched on the turn's subject rather than its failure signature, and a guide about a file the turn edits outranks one that is not. |
| 0.1.5 | **Added:** `agentic_knowledge` — a knowledge layer that runs behind the Codex router: per-work-mode retrieval of verified fixes, dead ends and the pre-compaction archive from the Semantic Search Engine, a masked learn-from-traffic write path through a Redis Stream and an indexing worker, and optional knowledge-or-escalate model selection. Adapters for Codex (full) and Claude Code (read-only), plus offline counterfactual evaluation. **Fix:** the `agentic_knowledge` collection names are reduced to the engine's character set at configuration time, so a repository scope that would produce a hyphen can no longer make every indexing attempt fail with a 500 at collection creation; the retention pass still sweeps the legacy `prefix-<scope>-sessions` archives; the repository of a request is read from the leading messages, so a compaction (which re-sends the whole window) is filed under its own project instead of into a scope-less prefix, and a request with no known repository publishes nothing; a search of a repository that has not been indexed yet is an empty answer, not a failed search **Added:** engine login for a Semantic Search Engine that authenticates by username and password rather than by a pre-issued token (`SSE_API_USERNAME`/`SSE_API_PASSWORD`); every mined episode now carries the work **phase** its evidence names (`test`, `implement`, …) in its header and in `metadata_json.phase`, and a retrieval source can be restricted in its policy to phases (`"phases": ["test"]`) or to routing modes (`"modes": ["debug"]`) — the filter is engine-side, unions across names, intersects the source's categories, never touches the archived window whose `thread_id` owns the single metadata slot, and a policy cannot mix the two dimensions in one request. |
| 0.1.4 | **Fix:** the Codex routing record follows the phase that was actually routed with instead of the last signal it could rebuild, so an interleaved command no longer freezes the record and repeats a conflict on every request of the turn (captured-session replay: 11 conflicts -> 0). **Added:** `RememberResult` carries why a memory write was skipped; the log names the real cause instead of an undifferentiated `version conflict`. |
| 0.1.3 | **Added:** an accepted semantic decision can be carried across one command generation (`MEMORY_CARRY_DECISIONS`). **Changed:** semantic rankings are read against their own scale; the phase layer separates work states from shell noise; CLI-injected context is kept out of the intent text. |
| 0.1.2 | **Added:** optional Redis-backed shared session memory with `MEMORY_MAX_RETRIES`; offline routing evaluation with calibration replay; a configurable `phase` layer reading assistant actions and tool activity; heuristic `negation_pattern` / `min_margin` / `weights`; `per_target_top_k` aggregation. **Changed:** mode definitions refined; semantic context built from the latest assistant utterance only. |
| 0.1.1 | **Added:** `ClaudeCodeRoutingPlugin` (tier-to-model mapping); `lint_signals` start-up validation; `classify_max_chars` budget. **Changed:** `agentic_routing` consolidated into `CodexRoutingPlugin`; word-boundary heuristic matching; similarity threshold lowered to `0.51`; sliding-window encoding for over-length queries. |
| 0.1.0 | **Deps:** moved to `radlab-pii-classification`; additional files shipped in the distribution. |
| 0.0.6 | **Added:** `SemanticBiEncoder` routing with a persisted FAISS store (`vector_store_path`) and persistence tests. **Changed:** `simple_semantic_routing` removed in favour of the BiEncoder; embedding model referenced by package identifier. |
| 0.0.5 | **Added:** `TimeRule`, an IBAN masking test suite, spaced car-plate formats. **Changed:** multi-format IBAN anchors, `_iban_mod97` moved to `validators`, PEP 8 import cleanup and no `__init__.py` re-exports. |
| 0.0.2 | **Added:** PII masking plugin, dynamic pseudonym generation with de-anonymization in `FastMasker`, LangChain RAG plugin with CLI, `SojkaGuardPlugin`, lint/format/type/security configuration. **Changed:** `mask()` returns `(masked_text, mappings)`; `numpy` loaded only when LangChain RAG is enabled. |
| 0.0.1 | **Added:** `fast_masker` plugin (IP, KRS, JWT, NRB, NIP, URL, SSN and other rules), packaging (`setup.py`) with the initial `.version`, License section in the README. |

## [Unreleased]

### Added
- `agentic_knowledge` indexes **the repository's own documentation**. `docs_ingestor.py` scans the directory a request
  declared for its `*.md` files, splits each one at its headings (a section keeps its heading path, cut at
  `docs.max_section_chars`), and publishes one `doc` event per section to the stream the miner already uses — so the
  worker masks and indexes documentation exactly as it does a mined recipe, into `category=doc` of
  `<prefix>_<scope>`, which `plan`, `implement` and `review` already ask for and which rendered nothing before.
- A file is republished when its **content** changes, not when it is looked at: the digest of the whole file feeds every
  section's fingerprint, and a scan state in Redis (`<key_prefix>:docs:<scope>`, one key per scope) compares digests, so
  a `touch` or a branch checkout over unchanged files publishes nothing. A changed file gets new document names and the
  scan publishes a `docs_prune` control message retiring the names it superseded, which is also what removes the
  documents of a deleted file. **A scan that ran out of `docs.max_files` or `docs.max_total_bytes` publishes no
  retirement at all** — pruning against a partial listing would delete documentation the scan never reached.
- The scan runs on the request path, bounded by `docs.rescan_interval_seconds` per process and by its byte and file
  ceilings, and is **fail-open** like the rest of the layer: a repository the router cannot see is
  `skipped: no-such-directory`, never an error. Where the router does not run next to the agents, the new
  `llm-router-agentic-kb-docs` command (`--path`, `--scope`, `--dry-run`, `--no-prune`, `--json`) does the same scan
  against the same policy from a machine that can; it publishes to the stream and never talks to the engine.
- Policy: a `docs` section (`enabled`, `auto_index`, `patterns`, `excluded_dirs`, `max_files`, `max_file_bytes`,
  `max_total_bytes`, `max_section_chars`, `rescan_interval_seconds`, `state_ttl_seconds`) and the matching
  `LLM_ROUTER_KNOWLEDGE_AGENTIC_DOCS_*` environment overrides. `payload["routing"]["knowledge"]` carries a `docs`
  outcome (`status`, `files`, `published`, `pruned`, `reason`, `truncated`) — no paths and no document names.
- Documentation is **its own collection**, `<prefix>_<scope>_docs`, apart from mined knowledge. One search request
  carries one metadata dimension, so a mode can narrow a search by `doc_type` only when nothing else shares the request;
  giving the documentation a collection of its own is what buys that, and it lets a repository's prose be re-indexed and
  pruned as a set. The scan state records the collection a listing was written for, so a policy that moves the
  documentation republishes it where it now belongs instead of believing a listing of files indexed where nobody reads.
- **`SourcePolicy.doc_types`** — which documentation kinds a work mode reads (`readme`, `guide`, `changelog`,
  `contributing`, `security`, `license`, `agent`, `note`), sent as engine-side `metadata_filters` on `doc_type`, one
  `eq` per kind, unioned. It is the `doc` source's own dimension: a `doc` source restricting phases or modes, or
  `doc_types` on any other source, is a configuration error. The shipped policy reads `readme`/`guide`/`contributing`
  when implementing, adds `changelog` and `note` when planning, and `guide`/`contributing`/`security` when reviewing;
  an `AGENTS.md` is in no list, because the agent already has that file in its context.
- Documentation is **searched with a query of its own**: the user's words plus the touched paths and their stems, never
  the failure signature. A recipe is filed under the error it fixed and a guide is filed under the module it explains;
  asking the second with the first was retrieval noise. `ContextAssembler` adds a bonus when a retrieved document
  **documents a file the turn is touching** (`docs/export.md` for an edited `src/export.py`) — a link no retrieval leg
  sees — drops a document whose declared kind the mode does not read, and cites what it quotes by file and heading
  (`[doc · guide · docs/export.md › Export helpers]`).

## [0.1.5] - 2026-10-17

### Added
- `agentic_knowledge` — an **agentic knowledge layer** for coding agents
  (`llm_router_plugins/utils/knowledge/agentic_knowledge/`), registered in `MAIN_UTILS_REGISTRY` and placed in the
  utils pipeline **behind** `agentic_routing_codex`. It reads the routing decision the router has just made, retrieves
  what this repository has already learned from the Semantic Search Engine (SSE), and injects it as a separate,
  removable context block — with no cooperation from the agent. Fully fail-open: no SSE host means no reads, no Redis
  means no writes, and any failure passes the request through untouched.
- **Engine login.** A Semantic Search Engine that authenticates by username and
  password rather than by a pre-issued token now works: `…_SSE_API_USERNAME` and
  `…_SSE_API_PASSWORD` flow from the environment through the store and the
  indexing worker, and the client logs in on its first call.  A token, when set,
  wins, and a name without a secret is no credential at all, so nothing is ever
  attempted that would log a 401 as an indexing failure.
- **Phases and modes.** Every mined episode carries the work phase its own
  evidence names — the mode most of its items are in, the newest one breaking a
  tie, the turn's mode when it names none — written into the document header and
  into `metadata_json.phase`, verbatim, because the reader filters on it; the
  routing `mode` of the producing request is stored the same way.  A retrieval
  source can be restricted in the policy to phases
  (`{"category": "recipe", "phases": ["test", "debug"]}`) or to modes
  (`{"category": "deadend", "modes": ["debug"]}`): one engine-side
  `metadata_filters` entry per name (the engine unions them, which is exactly
  "one of these"), intersecting the source's categories.  The two share the
  single metadata slot the engine allows, so a policy may restrict one
  dimension but not both in one request; a source without a restriction leaves
  the whole request unrestricted, and an archived source never carries either —
  its `thread_id` filter owns the slot, and a second would widen the archived
  window to every thread of the repository.  The phase is cited on every
  injected entry (`[recipe · in test · 2026-10-12 · src/a.py]`).
- **Fix:** the daemon worker no longer reports a ``TimeoutError`` on every idle
  read.  Its blocking read waits ``worker.block_ms`` (5 s) server-side, but its
  Redis socket timeout came from the request-path settings, where 1 s is a
  feature — the socket closed first and the loop never actually blocked.  The
  worker now raises only its own socket timeout over its own block (plus a
  margin); an operator-set socket timeout that already covers it is kept.
- **Added:** `scripts/run-agentic-kb-worker.sh` starts the indexing worker with
  the environment of the router's own run script — it reads that script's
  assignments rather than executing it, so no second server starts — and
  `scripts/run-agentic-kb-worker-pk.sh` the same over the personal
  `run-rest-api-gunicorn-pk.sh`.  A `--once` run now also logs how many events
  the batch drained, so a cron worker is not indistinguishable from one that
  never ran.
- **Fix:** a search for a repository whose knowledge has not been indexed yet
  stops being a per-request failure.  The engine answers a missing collection
  with an error code in a 200 body rather than with an empty list, so every
  request of a project before its first episode was logged as a failed search;
  the read path now recognizes that code and answers it as an empty one
  (nothing known yet) until the worker creates the collection on the first
  event.
- **Fix:** the repository of a request is found in the leading messages rather
  than only in the first `input` item.  A Codex compaction re-sends the whole
  conversation, so its `<environment_context>` sits behind the developer
  instructions — every event it produced carried an empty scope and was indexed
  into the bare collection prefix instead of the project's own, so a real
  repository never got its collections.  The search is bounded (a message and a
  request each have a ceiling, a fraction of a millisecond) and reads only what
  the agent sends as a message: a `<cwd>` quoted by a tool output is not a
  declaration, and the first declaration wins over one quoted later.  A request
  whose repository stays unknown now publishes nothing — the annotation says
  `write.status=disabled` — instead of archiving the conversation of one
  checkout into a prefix shared by every other scope-less one.
- **Fix:** the *scope* half of a collection name is reduced to the engine's
  character set like the prefix — a hyphenated checkout (`llm-router`) could
  never get a collection created (a 500, so nothing indexed), and the indexing
  consumer group now starts at the start of the stream, so episodes the proxy
  published while no worker was running are drained instead of skipped
  silently.
- **Read path.** Per-work-mode retrieval policies (`debug`, `test`, `implement`, `plan`, `git_review`, `review`) over
  weighted sources (`{category, weight, max_items}`) with a character budget per mode; `ContextAssembler` normalizes
  SSE scores, adds a bonus for a matching error signature or path, deduplicates by document name and cuts to the
  budget. The block is appended as the **last element of `input`** (a `developer` message), never to `instructions`, so
  the CLI's prompt-cache prefix survives, and an already-present block is never duplicated. `GenerationContextCache`
  keeps the injected set per command generation in Redis, so repeated requests of one turn issue no search and get
  byte-identical text.
- **Write path.** `EpisodeMiner` derives a `recipe` from a `test_failure → patch → test pass` episode and a `deadend`
  from a failure repeating with an identical signature; `KnowledgeEventPublisher` deduplicates with `SET NX` before
  `XADD … MAXLEN ~`, releasing the key when `XADD` fails so an episode is retried rather than lost. The new
  `llm-router-agentic-kb-worker` console script consumes the stream with a consumer group, masks every document with
  `FastMasker` and discards the mappings, creates a scope's collection on first use, retries indexing a bounded number
  of times, and applies retention to this plugin's own documents only.
- **Compaction archive.** A request classified `compaction` is written and never read from: its window is chunked and
  indexed into `<prefix>-<scope>-sessions`, so later requests of the same thread get back the exact fragments the
  summary dropped (`## Exact context from before compaction`).
- **Knowledge-or-escalate.** `EscalationPolicy` switches a looping agent to the mode's `escalation_model_name` only
  when routing was not explicit, the same error signature has failed `loop_threshold` times in the generation, and no
  recipe scores at least `strong_recipe_score`; a strong recipe keeps the routed model. The escalated model persists
  for the generation. **Off by default** (`escalation.enabled=false`) until the replay shows the detection is right.
- **Adapters.** Codex reuses `codex.payload` and `codex.phase` instead of parsing again and serves the full read and
  write path; Claude Code is served in a basic form — scope from the `Working directory` line, latest user text,
  injection into `system` — and, having no phase evidence, reads the shared knowledge base without mining it or
  escalating.
- **Scope isolation.** The repository is identified by the working directory the agent itself declares (`<cwd>` for
  Codex, `Working directory:` for Claude Code) and reduced to a path-free key; one collection per scope, so a search of
  one repository cannot return a document of another and no absolute path is ever indexed.
- **Annotation.** `payload["routing"]["knowledge"]` reports the adapter, scope, injected hits, characters, per-source
  counts, cache state, loop count, escalation and write status — never document content, tokens, hosts or full session
  ids. When this plugin replaces a model, the previous one is kept in `routing.model_before_knowledge`.
- **Counterfactual evaluation.** `agentic_knowledge/evaluation.py` replays recorded sessions offline, feeding the store
  only events mined *before* the request under evaluation, and reports `recipe_hit_rate`, `turns_potentially_saved`,
  `missed_saves`, loop-detection precision/recall, `escalations` and `injected_chars_per_turn` for the `no_kb`, `read`
  and `read_escalate` variants; `scripts/agentic-kb-eval.sh` runs it.
- `knowledge` pip extra (`sse-lib`, `redis>=5`) and `llm_router_plugins/resources/knowledge/agentic_knowledge.json`,
  the reviewable policy — connection settings come exclusively from `LLM_ROUTER_KNOWLEDGE_AGENTIC_*`.

### Fixed
- `agentic_knowledge`: collection names are now reduced to the engine's
  character set when the policy is loaded.  The engine refuses to create a
  collection whose name is not identifier-like — a hyphen, or a name starting
  with a digit, is a `500 Internal Server Error` at creation time, and the
  default `agentic-kb-<scope>` name carries a hyphen, so no repository could
  ever be indexed: the worker kept retrying an event that could never land.
  The prefix and the collection names are now one engine-safe spelling
  (`agentic_kb_<scope>` / `agentic_kb_<scope>_sessions`), and the retention
  pass still sweeps the legacy `agentic-kb-<scope>-sessions` archives.

## [0.1.4] - 2026-10-13

### Added
- `RememberResult` — the outcome of `remember_decision` now carries a log-safe
  reason for a conflict (`state` still compares equal to the outcome string).

### Changed
- A skipped Codex routing memory write is logged with the actual cause instead
  of one undifferentiated `version conflict`:
  - `expected version N, record is M` — a concurrent writer moved first;
  - `incomparable history` / `stale history` / `reordered history` /
    `context changed` — the continuity proof failed;
  - `mode disagreement`, `generation changed`, `write` — the other refusal
    paths.

### Fixed
- The Codex routing record now follows the phase the cascade actually routed
  with instead of the last signal it could rebuild from the history. The
  cascade names the phase dominating the recent window, so an edit interleaved
  between test runs is still an implementation turn. Previously the record
  froze on the last signal, the write was refused, and every later request of
  the same command generation reported a `mode disagreement` conflict.
  Verified by replaying a captured Codex CLI session (149 requests): conflicts
  drop from 11 to 0 and the stored record tracks the decided mode.

## [0.1.3] - 2026-10-11

### Added
- An accepted semantic decision can be carried across one command generation,
  so repeated requests of a long agent turn do not re-decide it from scratch
  (opt-in via `MEMORY_CARRY_DECISIONS`).

### Changed
- Semantic rankings are read against their own scale: the similarity
  threshold, the minimum margin and the top-k are compared on the score
  distribution of the request, not on an absolute cosine.
- The phase layer distinguishes work states from shell noise, so a turn is
  not re-labelled by an unrelated command interleaved between real work.
- CLI-injected context is kept out of the intent text used by the heuristic
  and semantic layers.

### Fixed
- Held out the conv-03 goal turn and reported refusal rates in the routing
  quality suite.

## [0.1.2] - 2026-10-08

### Added
- Shared session memory for the Codex routing cascade, Redis-backed and
  optional, so a neutral request in the middle of an action keeps the last
  reliable phase; `MEMORY_MAX_RETRIES` bounds the attempts after a version
  conflict before the request continues statelessly.
- Offline Codex routing evaluation with semantic quality calibration and a
  replay harness.
- A configurable `phase` layer that routes from assistant actions and tool
  activity (commands, patches, test outcomes).
- A configurable heuristic `negation_pattern`, `min_margin` and `weights`.
- `per_target_top_k` aggregation, `min_margin` and independent segment budgets
  in the semantic layer.

### Changed
- Routing mode definitions for `plan`, `implement`, `test`, `git_review`,
  `review` and `debug` were refined.
- Semantic context is built from the latest assistant utterance only.
- `aux_title` routing recognizes title requests by shape in addition to
  metadata.

## [0.1.1] - 2026-09-25

### Added
- `ClaudeCodeRoutingPlugin` — tier-to-model mapping for Claude Code, with
  configuration, registry entry, docs and tests.
- Startup linting for unusable signals and `lint_signals` configuration
  validation.
- A `classify_max_chars` budget for the user text the deterministic layers
  may read.

### Changed
- `agentic_routing` was consolidated into `CodexRoutingPlugin`; the generic
  `agentic_routing` plugin and its test suite were removed.
- Heuristic scoring now uses word-boundary matching and a stricter
  `heuristic_min_score`.
- The semantic similarity threshold was lowered to `0.51`.
- Sliding-window encoding for over-length queries with capped window
  aggregation.

## [0.1.0] - 2026-09-06

### Changed
- Dependency moved to `radlab-pii-classification` and additional files added
  to the package distribution.

## [0.0.6] - 2026-08-31

### Added
- `SemanticBiEncoder` routing plugin with a FAISS vector store and persistence,
  plus a `vector_store_path` configuration.
- FAISS persistence tests and environment-variable name updates.

### Changed
- `simple_semantic_routing` was removed; the BiEncoder plugin replaces it.
- Embedding model identified by package name
  (`radlab/semantic-euro-bert-encoder-v1`).
- Extensive docstrings and documentation for the semantic routing plugins.

## [0.0.5] - 2026-07-16

### Added
- `TimeRule` in `fast_masker`.
- Comprehensive IBAN masking tests for `BankAccountRule` across formats and
  countries.
- Spaced car-plate format support in `CarPlateRule`.

### Changed
- IBAN anchor handling in `BankAccountRule` with enhanced multi-format regex.
- `_iban_mod97` logic moved to `validators`.
- PEP 8 import cleanup and removal of `__init__.py` re-exports.

## [0.0.2] - 2026-05-10

### Added
- PII masking plugin support.
- Dynamic pseudonym generation, mapping storage and de-anonymization in
  `FastMasker`.
- `LangChain RAG` utility plugin with engine, registry, pipeline and a CLI
  wrapper for indexing and search.
- `SojkaGuardPlugin` guardrail.
- Linting, formatting, type-checking and security configuration
  (Black, Flake8, MyPy, Pylint, Bandit).
- Medusa data preparation script for LangChain RAG indexing.

### Changed
- `PayloadInterface.mask` returns `(masked_text, mappings)` to support
  de-anonymization.
- `maskers_http_base` returns the anonymized payload instead of plain text.
- FastMasker rules standardized to return `(text, mappings)` and to accept an
  `anonymizer_fn`.
- `numpy` is loaded only when LangChain RAG is enabled.
- Guardrails inherit from `GuardrailsBase`.

## [0.0.1] - 2025-11-28

### Added
- `fast_masker` plugin with rules for IP, KRS, JWT, NRB, NIP, URL, SSN and
  other personal-data patterns.
- Packaging configuration (`setup.py`) and the initial `.version` file.
- License section in the README.
