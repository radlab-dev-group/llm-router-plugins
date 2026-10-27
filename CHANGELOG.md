# Changelog

All notable changes to the `llm-router-plugins` package are documented here.
Versions follow the `MAJOR.MINOR.PATCH` scheme; a new entry is added for every
release tagged as `v<version>`.

## Summary

| Version | Changelog |
| ------- | --------- |
| 0.1.5 | **Changed:** `agentic_knowledge` stores one repository in one collection. Every document kind — mined knowledge, the pre-compaction archive, the repository's own documentation — is now written into `<prefix>_<scope>` and told apart by its engine `category`, which the engine resolves to a document set before scoring, so a search that did not ask for a category never sees it. `collection_layout` (`single`, the shipped default, or `split`, the older three-collections spelling for a deployment indexed with it) decides the collection a reader asks and a worker writes; the split layout keeps working for legacy data, and retention still sweeps the `..._sessions` collections it left behind. A mode still asks one question per source group — mined knowledge, the archive restricted to its `thread_id`, the documentation restricted by `doc_type` — whatever collection the layout gives them. **Fixed:** a search of one repository now returns a document of only that repository under either layout, and the retention pass deletes only the document kinds it names, so a shared collection can no longer age a README on the archive's clock. |
| 0.1.4 | **Added:** `agentic_knowledge` — a knowledge layer that runs behind the Codex router: per-work-mode retrieval of verified fixes, dead ends and the pre-compaction archive from the Semantic Search Engine, a masked learn-from-traffic write path through a Redis Stream and an indexing worker, and optional knowledge-or-escalate model selection. Adapters for Codex (full) and Claude Code (read-only), plus offline counterfactual evaluation. **Fix:** the `agentic_knowledge` collection names are reduced to the engine's character set at configuration time, so a repository scope that would produce a hyphen can no longer make every indexing attempt fail with a 500 at collection creation; the retention pass still sweeps the legacy `prefix-<scope>-sessions` archives; the repository of a request is read from the leading messages, so a compaction (which re-sends the whole window) is filed under its own project instead of into a scope-less prefix, and a request with no known repository publishes nothing; a search of a repository that has not been indexed yet is an empty answer, not a failed search **Added:** engine login for a Semantic Search Engine that authenticates by username and password rather than by a pre-issued token (`SSE_API_USERNAME`/`SSE_API_PASSWORD`); every mined episode now carries the work **phase** its evidence names (`test`, `implement`, …) in its header and in `metadata_json.phase`, and a retrieval source can be restricted in its policy to phases (`"phases": ["test"]`) or to routing modes (`"modes": ["debug"]`) — the filter is engine-side, unions across names, intersects the source's categories, never touches the archived window whose `thread_id` owns the single metadata slot, and a policy cannot mix the two dimensions in one request. **Added:** `agentic_knowledge` indexes the repository's own documentation — a heading-aware scan of its `*.md` files published as `doc` documents, republished on content change, superseded documents retired through a `docs_prune` control message (never on a truncated scan), plus the `llm-router-agentic-kb-docs` command and the `docs` policy section. Documentation now lives in `<prefix>_<scope>_docs`, is read per work mode through `doc_types`, is searched on the turn's subject rather than its failure signature, and a guide about a file the turn edits outranks one that is not. On top of retrieval a session carries a **standing documentation map** — structural, stable per repository, budgeted first — and every mined fix or dead end points at the page that documents the module it touched. **Fix:** the Codex routing record follows the phase that was actually routed with instead of the last signal it could rebuild, so an interleaved command no longer freezes the record and repeats a conflict on every request of the turn (captured-session replay: 11 conflicts -> 0). **Added:** `RememberResult` carries why a memory write was skipped; the log names the real cause instead of an undifferentiated `version conflict`. **Added:** `agentic_knowledge` mines `playbook` — a change a test run accepted with no failure in between — so the correct steps of a task are knowledge too, not only the ones that went wrong; a mode routing did not resolve is served by a `*` fallback policy instead of reading nothing; every mode now reads the kinds it ignored (`doc` in `debug`/`test`, `deadend` in `implement`, `playbook`/`recipe` where a plan or a review could use them). **Fix:** a read budget equal to one engine call started no search of any request — the deadline left no room for the call it budgeted — which looked like an empty knowledge base; the budget is now raised above one call at validation, defaults to three calls, and the "searches not started" warning names both numbers and repeats at most once per repository per interval. |
| 0.1.3 | **Added:** an accepted semantic decision can be carried across one command generation (`MEMORY_CARRY_DECISIONS`). **Changed:** semantic rankings are read against their own scale; the phase layer separates work states from shell noise; CLI-injected context is kept out of the intent text. |
| 0.1.2 | **Added:** optional Redis-backed shared session memory with `MEMORY_MAX_RETRIES`; offline routing evaluation with calibration replay; a configurable `phase` layer reading assistant actions and tool activity; heuristic `negation_pattern` / `min_margin` / `weights`; `per_target_top_k` aggregation. **Changed:** mode definitions refined; semantic context built from the latest assistant utterance only. |
| 0.1.1 | **Added:** `ClaudeCodeRoutingPlugin` (tier-to-model mapping); `lint_signals` start-up validation; `classify_max_chars` budget. **Changed:** `agentic_routing` consolidated into `CodexRoutingPlugin`; word-boundary heuristic matching; similarity threshold lowered to `0.51`; sliding-window encoding for over-length queries. |
| 0.1.0 | **Deps:** moved to `radlab-pii-classification`; additional files shipped in the distribution. |
| 0.0.6 | **Added:** `SemanticBiEncoder` routing with a persisted FAISS store (`vector_store_path`) and persistence tests. **Changed:** `simple_semantic_routing` removed in favour of the BiEncoder; embedding model referenced by package identifier. |
| 0.0.5 | **Added:** `TimeRule`, an IBAN masking test suite, spaced car-plate formats. **Changed:** multi-format IBAN anchors, `_iban_mod97` moved to `validators`, PEP 8 import cleanup and no `__init__.py` re-exports. |
| 0.0.2 | **Added:** PII masking plugin, dynamic pseudonym generation with de-anonymization in `FastMasker`, LangChain RAG plugin with CLI, `SojkaGuardPlugin`, lint/format/type/security configuration. **Changed:** `mask()` returns `(masked_text, mappings)`; `numpy` loaded only when LangChain RAG is enabled. |
| 0.0.1 | **Added:** `fast_masker` plugin (IP, KRS, JWT, NRB, NIP, URL, SSN and other rules), packaging (`setup.py`) with the initial `.version`, License section in the README. |

## [0.1.5] - 2026-10-27

### Changed
- **One collection per repository.** Every document kind of a repository now
  lives in `<prefix>_<scope>`: mined knowledge, the pre-compaction session
  archive and the repository's own documentation are written into one
  collection and told apart by their engine `category`, not by the name of a
  collection. A category is a boundary of the same hardness a collection was —
  the engine turns it into a set of document names *before* the vector leg
  scores anything, so a request that did not ask for the documentation never
  sees it, and its `max_results` is spent within what survived the cut.
- **`collection_layout`** — `single` (the default) or `split`, the older
  three-collections spelling: `<prefix>_<scope>` for mined knowledge,
  `<prefix>_<scope>_sessions` for the archive, `<prefix>_<scope>_docs` for the
  documentation. The layout is a property of the data at rest, so a deployment
  switches the reader and the writer at once; `split` stays, so a repository
  indexed with it keeps being read and stays reversible without re-indexing.
  `LLM_ROUTER_KNOWLEDGE_AGENTIC_COLLECTION_LAYOUT` overrides the policy.
- **A mode still asks one question per source group.** The request that
  retrieves the archive is restricted by its `thread_id`, the one that
  retrieves the documentation by `doc_type`, and the one that retrieves mined
  knowledge by neither — whatever collection the layout gives them. A request
  never mixes the archive with mined knowledge: the engine intersects filter
  families, so a `thread_id` on such a request would cut the mined knowledge to
  the documents of this one thread, silently dropping every recipe mined in an
  earlier session.

### Fixed
- **The retention pass deletes only the document kinds it names.** One
  collection now holds several kinds on different clocks: an archived window
  ages after `write.archive.ttl_seconds`, a README ages when a human deletes
  the file, a mined recipe retires by being superseded. The pass sweeps the
  repository's collection (and still the legacy `..._sessions` collections it
  left behind), but deletes only the archive's kind, so a shared collection
  cannot age a README on a fourteen-day clock.

## [Unreleased]

### Added
- **`--clean-redis-memory-cache` on the indexing worker** drops the layer's
  cached state in Redis before it starts: an assembled generation context, a
  repository's documentation scan state and its standing digest, and the
  deduplication claims that remember which events were already published. The
  case it is for is a deleted collection — the scan state goes on saying the
  documentation is indexed and the claims go on saying its events were seen, so
  nothing is written again until every TTL expires by itself, and an operator
  watching an empty collection has no lever but time. The keys are matched by
  their own namespace segments under `SCAN` cursors and deleted in batches, so
  neither the event stream and its consumer group — a queue, not a cache, whose
  unpublished episodes would be lost with it — nor another plugin's keys in the
  same database are touched. A clean that cannot reach Redis stops the run with
  the engine's own message rather than indexing on a cache that was not cleared.
- `agentic_knowledge` mines a fourth kind of episode: **`playbook`**, a change
  whose test run passed with **no failure in between**. Until now a turn that
  went right from the first try taught the repository nothing — a recipe needs a
  failure to be filed under, and most work has none — so the base filled up with
  fixes and dead ends while the procedures that work were never written down.
  A procedure is one stretch of a turn between two green runs, so a turn that
  implements, tests, extends and tests again writes two of them; the document
  leads with what was asked for, then what was changed, then the command that
  proved it. It is never written for a stretch a verified recipe already covers,
  and never for a change no run accepted: an intention indexed as a verified
  procedure is a guess the next session would trust. It carries no failure
  signature, so it cannot be the "strong recipe" that holds a looping model in
  place, and it renders under its own heading,
  `## Verified procedures in this repo`.
- **A fallback policy (`modes."*"`)** serves a mode routing did not resolve —
  an empty mode, or a name the policy does not know — so the request is asked of
  the base instead of being skipped. A mode's own policy always wins, and a
  configuration without `*` keeps the previous behaviour exactly.
- **Every mode reads what it ignored.** `debug` and `test` now read the
  repository's documentation (a guide about the module that failed; the
  contributing rules a change has to keep), `implement` reads dead ends, and
  `plan`/`review`/`git_review` read mined knowledge and procedures. Every mode
  reads `playbook` in some measure, weighted for what the mode is doing.

### Changed
- The shipped `read.timeout_seconds` is 6.0 — three calls of the two-second
  engine default, which is how many collections a widened mode can ask.

### Fixed
- **A read budget equal to one engine call started no search at all.** A search
  begins only while a whole call fits before the request's deadline, and the
  deadline was read from the clock a moment *before* the call was made — so a
  budget of exactly one call was spent by the microseconds in between, and every
  request of a deployment that set `READ_TIMEOUT_SECONDS` to
  `SSE_API_TIMEOUT` reported "N search(es) not started" and injected nothing.
  Validation now keeps a margin over one call, so the invariant
  "a request can always ask once" holds whatever the environment says.
- That warning now names the per-call timeout next to the budget (the usual
  cause is the second number, not the first) and repeats at most once per
  repository per interval: a starved configuration starves every request of
  every session, and a warning printed on every request is a warning nobody
  reads.

## [0.1.4] - 2026-10-21

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
- **Added:** `scripts/plugins/agentic_knowledge/run-agentic-kb-worker.sh` in the `llm-router` checkout
  starts the indexing worker with the environment of the router's own run script — it reads that
  script's assignments rather than executing it, so no second server starts.
  A `--once` run now also logs how many events the batch drained, so a cron
  worker is not indistinguishable from one that never ran.
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
- `agentic-knowledge` pip extra (`sse-lib`, `redis>=5`) and `llm_router_plugins/resources/knowledge/agentic_knowledge.json`,
  the reviewable policy — connection settings come exclusively from `LLM_ROUTER_KNOWLEDGE_AGENTIC_*`.
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
- A session carries a **standing documentation context** (`doc_context.py`): a short map of what the repository has
  written down — its documentation files, their kinds, and their heading outlines — injected on every request a mode
  reads documentation in, whether or not this turn's query matched anything. It answers what the project is and where
  its documentation lives, which is what an agent otherwise rediscovers with `ls` and `find` once per command. The map
  is **structural and carries no prose**: the write path masks a document before indexing it, and a digest of README
  sentences built on the request path would carry them unmasked to a model. `docs.context_enabled`,
  `docs.context_max_chars` (default `700`) and `docs.context_ttl_seconds`, with the matching `…_DOCS_CONTEXT_*`
  environment overrides.
- The digest is **stable per repository, not per request**: it is refreshed by the scan that already runs and by
  `llm-router-agentic-kb-docs`, kept in Redis (`<key_prefix>:docs:context:<scope>`), and read with one `GET` by every
  other request of the session. Its text moves when the documentation's outline moves and not when a paragraph is
  reworded, so the prompt under the provider's cache does not move either; its fingerprint records the revisions it was
  drawn from. It is charged against `modes.<mode>.max_chars` **before** anything is retrieved, and a digest too big for
  the mode is not injected at all. `payload["routing"]["knowledge"]["doc_context"]` reports `level`, `chars` and
  `source_files`.
- Documentation is **searched with a query of its own**: the user's words plus the touched paths and their stems, never
  the failure signature. A recipe is filed under the error it fixed and a guide is filed under the module it explains;
  asking the second with the first was retrieval noise. `ContextAssembler` adds a bonus when a retrieved document
  **documents a file the turn is touching** (`docs/export.md` for an edited `src/export.py`) — a link no retrieval leg
  sees — drops a document whose declared kind the mode does not read, and cites what it quotes by file and heading
  (`[doc · guide · docs/export.md › Export helpers]`).
- **A mined entry points at the page that documents the module it changed.** A
  verified fix says what worked; the repository's own page about that module
  says what it has to keep doing, and neither search connects the two: a recipe
  is filed under a failure, a page under a module. The link is resolved on the
  reader side against the documentation map the request already carries: a named
  page matches by file stem (`src/export.py` → `docs/export.md`), a generic page
  (`README.md`, `AGENTS.md`) by the directory it documents, and never a second
  time for a path the entry already cites — so a recipe or dead end renders with
  a ` → docs/export.md` tail whether or not the query retrieved the page. The
  same rule now drives the documentation ranking bonus, which previously matched
  generic stems too: a root `README.md` is not documentation of everything. Two deliberate limits: the link
  is not written into the document's metadata, because the engine never returns
  document metadata on a search hit — a field nothing could read would be
  write-only — and a documentation page never counts toward
  `best_recipe_score`, because a README is not a verified fix and must not
  push a mode past its escalation threshold.
- `RememberResult` — the outcome of `remember_decision` now carries a log-safe
  reason for a conflict (`state` still compares equal to the outcome string).

### Changed
- **The worker says on the console what it is doing, and follows the router's
  debug switch.** On start it writes its configuration block (engine, stream,
  group, consumer, chunking, retention, and which switch chose the log level);
  during a run it reports each batch's read/indexed/dropped counts and elapsed
  time, names a created collection, reports what retention and a docs prune
  removed, and beats an idle heartbeat every five minutes. `DEBUG` adds the
  per-event line — kind, scope, thread, fingerprint. The level resolves from
  `--log-level`, else `LLM_ROUTER_IN_DEBUG` (the router's own debug switch,
  which the operator's wrapper already exports), else `LLM_ROUTER_LOG_LEVEL`,
  else `INFO`, so a deployment the operator turned debug on gets a debug worker
  without repeating the switch. A failure is logged with the engine's own
  message, not only the exception's class, and credentials are reported by kind
  (`auth=token`), never by value.
- **The knowledge extra and its CLI package are named after the plugin.** `pip install
  radlab-llm-router-plugins[knowledge]` is now `…[agentic-knowledge]`, so an operator installing
  the extra sees the same name the pipeline does, and the console commands are built from
  `llm_router_plugins.cli.plugins.utils.agentic_knowledge` — `utils.knowledge` named a directory
  of two commands and said nothing about which layer they belonged to. The commands themselves,
  `llm-router-agentic-kb-worker` and `llm-router-agentic-kb-docs`, are unchanged.
- **The operator's knowledge scripts moved to the `llm-router` checkout, and the worker stopped
  fishing for `sse_lib` in a sibling checkout.** The `run-agentic-kb-{worker,docs}.sh` wrappers now
  live in `scripts/plugins/agentic_knowledge/` of that checkout, next to the plugin they operate,
  with their configuration still in the `run-rest-api-gunicorn.sh` script they read.
  The `sse-lib` dependency of the `knowledge` extra is pinned to `>=0.1.0` — the first
  published release with `ignore_question_lang_detect` on `SSEClient.search` — and the
  wrappers no longer prepend a sibling `semantic-search-engine/sse_lib/src` to
  `PYTHONPATH`: the interpreter the script runs with (default `python3`, override with
  `PYTHON=`) must have the package installed from the extra.
- **The read path stopped losing every search to its own timeout, and stopped asking the engine to
  guess the query's language.** Two independent faults made the knowledge base invisible in
  production, measured against a real engine on another host:
  `sse_api.engine.controllers.search.semantic` detects the language of the *question* and applies it
  twice — as a metadata filter on the vector leg (`text_language == <lang>`), and as the name of the
  PostgreSQL full-text configuration (`pl` → `polish`).  A knowledge base is indexed in the language
  of the code and asked in the language of the operator, so the first filter throws the answers away
  and, on a cluster without a `polish` text search configuration, the second answers a Polish
  question with `ProgrammingError: text search configuration "polish" does not exist` — an HTTP 500,
  which this layer can only read as an unavailable engine.  The plugin now sends
  `ignore_question_lang_detect` (`…_SSE_IGNORE_LANGUAGE_DETECT`, on by default; a client too old to
  take the keyword is detected and asked without it), so the engine drops the filter and falls back
  to the `simple` configuration.
  Separately: one number used to be both the per-call transport timeout and the read budget, so a
  mode reading three sources waited three times what the operator wrote, while the 1.5 s default was
  below the 1.7 s median a hybrid search actually takes — five of five searches timed out.  They are
  two numbers now: `…_SSE_API_TIMEOUT` (default `2.0`) ceilings one call, `…_READ_TIMEOUT_SECONDS`
  (default `4.0`) ceilings the request, a search starts only while a whole call fits before that
  deadline, and the budget is never allowed below one call.  A budget spent mid-request skips the
  remaining sources and warns once, naming the scope and the number skipped.

- The knowledge generation cache entry grew a schema version (`2`): a hit now keeps `source`, `doc_type` and `heading`,
  and an entry keeps the standing documentation digest it was built with, so a cache hit reports the same context the
  first request of the generation reported. Entries written under schema `1` are ignored rather than misread — one
  rebuilt block per generation, then normal service.
- A skipped Codex routing memory write is logged with the actual cause instead
  of one undifferentiated `version conflict`:
  - `expected version N, record is M` — a concurrent writer moved first;
  - `incomparable history` / `stale history` / `reordered history` /
    `context changed` — the continuity proof failed;
  - `mode disagreement`, `generation changed`, `write` — the other refusal
    paths.

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
