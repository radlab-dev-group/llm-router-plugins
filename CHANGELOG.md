# Changelog

All notable changes to the `llm-router-plugins` package are documented here.
Versions follow the `MAJOR.MINOR.PATCH` scheme; a new entry is added for every
release tagged as `v<version>`.

## Summary

| Version | Changelog |
| ------- | --------- |
| 0.1.4 | **Fix:** the Codex routing record follows the phase that was actually routed with instead of the last signal it could rebuild, so an interleaved command no longer freezes the record and repeats a conflict on every request of the turn (captured-session replay: 11 conflicts -> 0). **Added:** `RememberResult` carries why a memory write was skipped; the log names the real cause instead of an undifferentiated `version conflict`. |
| 0.1.3 | **Added:** an accepted semantic decision can be carried across one command generation (`MEMORY_CARRY_DECISIONS`). **Changed:** semantic rankings are read against their own scale; the phase layer separates work states from shell noise; CLI-injected context is kept out of the intent text. |
| 0.1.2 | **Added:** optional Redis-backed shared session memory with `MEMORY_MAX_RETRIES`; offline routing evaluation with calibration replay; a configurable `phase` layer reading assistant actions and tool activity; heuristic `negation_pattern` / `min_margin` / `weights`; `per_target_top_k` aggregation. **Changed:** mode definitions refined; semantic context built from the latest assistant utterance only. |
| 0.1.1 | **Added:** `ClaudeCodeRoutingPlugin` (tier-to-model mapping); `lint_signals` start-up validation; `classify_max_chars` budget. **Changed:** `agentic_routing` consolidated into `CodexRoutingPlugin`; word-boundary heuristic matching; similarity threshold lowered to `0.51`; sliding-window encoding for over-length queries. |
| 0.1.0 | **Deps:** moved to `radlab-pii-classification`; additional files shipped in the distribution. |
| 0.0.6 | **Added:** `SemanticBiEncoder` routing with a persisted FAISS store (`vector_store_path`) and persistence tests. **Changed:** `simple_semantic_routing` removed in favour of the BiEncoder; embedding model referenced by package identifier. |
| 0.0.5 | **Added:** `TimeRule`, an IBAN masking test suite, spaced car-plate formats. **Changed:** multi-format IBAN anchors, `_iban_mod97` moved to `validators`, PEP 8 import cleanup and no `__init__.py` re-exports. |
| 0.0.2 | **Added:** PII masking plugin, dynamic pseudonym generation with de-anonymization in `FastMasker`, LangChain RAG plugin with CLI, `SojkaGuardPlugin`, lint/format/type/security configuration. **Changed:** `mask()` returns `(masked_text, mappings)`; `numpy` loaded only when LangChain RAG is enabled. |
| 0.0.1 | **Added:** `fast_masker` plugin (IP, KRS, JWT, NRB, NIP, URL, SSN and other rules), packaging (`setup.py`) with the initial `.version`, License section in the README. |

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
