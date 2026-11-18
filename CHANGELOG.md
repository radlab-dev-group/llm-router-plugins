# Changelog

All notable changes to the `llm-router-plugins` package are documented here.
Versions follow the `MAJOR.MINOR.PATCH` scheme; a new entry is added for every
release tagged as `v<version>`.

## Summary

| Version | Changelog |
| ------- | --------- |
| 0.1.9 | **Added:** the knowledge layer accepts a green compiler, linter, type-checker or shell run as a verified fix (`verification_type`), mines two new kinds (`task_solution`, `tool_recipe`), scores injected knowledge so a debunked recipe sinks and a proven one rises, warns before a known dead end, and retries a path-scoped search that came back empty. **Changed:** injected diffs keep their line structure in fenced ```diff blocks (quote ceiling 1800, truncation never opens a fence), the pre-compaction archive reads as a `Previous Conversation Milestones` timeline, the cross-encoder reranker (`rerank_results`) and adaptive model escalation ship enabled by default (`strong_recipe_score` `0.6`), `.plans` is excluded from the documentation scan. **Fixed:** stores archived compaction windows as complete documents to preserve full knowledge state during compaction. |
| 0.1.8 | **Fixed:** NIP number detection in `NipRule` to support spaces and 3-2-2-3 grouping format (`118 193 42 72`, `118-19-34-272`); restricted `KrsRule` regex to plain 10-digit strings (`\d{10}`), preventing space-separated or malformed NIP numbers from being erroneously classified as KRS in the masking pipeline. **Added:** dedicated test suite in `tests/test_nip_rule.py` for NIP formatting, boundary conditions, markdown markers, and pipeline precedence over KRS. |
| 0.1.7 | **Added:** exported all built-in rule classes (`KrsRule`, `RegonRule`, `PostalCodeRule`, `NrbRule`, `UrlRule`, `EuVatRule`, etc.) in `fast_masker.rules.__init__`. **Changed:** unified pseudonym formatting across all `FastMasker` rules via `_format_pseudo` helper in `BaseRule`, preventing duplicate outer brackets when custom bracketed formats (e.g. `[TAG_1]`, `{TAG}`) are passed from `anonymizer_fn`. **Fixed:** phone number boundary matching in `PhoneRule` (`(?<!\d)` / `(?!\d)`) allowing detection adjacent to punctuation, quotes, and brackets; expanded Polish phone prefix support (infolines `80x`, mobile `88`, landline area codes `12`-`95`) and support for parenthesized area/country codes in domestic and international rules; corrected EU VAT validation (`is_valid_eu_vat`) to enforce valid EU member state prefixes and structure (including Polish NIP checksum validation for `PL`), eliminating false-positive VAT/NIP matches on concatenated text (e.g. `waluty2026`). |
| 0.1.6 | **Changed:** a hybrid search is read on the scale its scores were produced on — a fused (RRF) rank is remapped onto 0-1 inside the result set that produced it before a bonus or a threshold is applied to it, and a hit carries the name of the scale its score lives on (`score_scale`), so `strong_recipe_score` and `min_similarity` are no longer read against a number of another kind. The sources a mode reads are read in parallel under one shared budget instead of one after another, and the shipped `read.timeout_seconds` falls to 3.0. The injected annotation gains a `metrics` block: how many searches the read made and how many the budget could not start. **Fixed:** a scope that arrives padded or with non-ASCII letters is normalized, transliteration included, before it becomes a collection name; an edit made by the shell itself — a redirect, `tee`, `sed -i` — is evidence of changed code again in a CLI without a patch tool; the read cache is keyed by the policy the mode reads with, so switching to a mode that asks a different question no longer serves the previous mode's answer; a document is named for the day its event was created, so a re-index does not collide with the documents of another day; a documentation scan that was truncated holds its prune instead of retiring files it never saw; the escalation threshold is compared on the scale its score was produced on, so a cosine-tuned `strong_recipe_score` does not silently stop firing under `rerank_results` — where a squashed cross-encoder logit tops out below the threshold — and a recipe that matches the failure's own signature and carries its verification command now counts on identity (reason `strong_recipe_identity`) instead of being reported as no known fix, with the mismatch named by a warning at start. **Fixed:** the package declares the dependencies its modules import at module level — `pandas`, `requests` and `numpy` — a fresh install of a masker no longer fails on `import pandas`; a test now reads the setup.py metadata and fails the suite if a module reaches for a package no requirement or extra declares, so the gap is caught at the first commit instead of at the first clean install. **Added:** `git_review` reads the repository's documentation the way `review` does, and `review` is its alias; a repository without mined documents skips the mined search and is still answered from the archive and the documentation; the indexing worker reports `pending` and `oldest_age` in its heartbeat; the miner names, in its diagnostics, the turn it read as no failure and the failure nothing changed for; a shell command an operator edits is visible in the mined document that quotes it. |
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

## [Unreleased]

### Added
- **The knowledge layer can learn from its own traffic (`learning`, opt-in).** The layer has
  always voted on what it injected — a verified fix credits the block, a returning failure
  blames it — but every number that turns those votes into a ranking was chosen by hand.  They
  can be estimated instead, and `learning.py` is the estimation: a document's reputation is a
  Beta posterior with a half-life and Thompson sampling (so a recipe nobody has needed for two
  months drifts back to *unknown* instead of staying famous), a per-scope ridge logistic
  regression over the features a candidate already carries is fitted offline, and a per
  `(mode, kind)` bandit learns what a character of one source's budget is worth.  Nothing
  trains on the request path: the proxy reads a reputation from one hash and a versioned
  snapshot from one key, writes the features it showed to a bounded training stream, and every
  multiplier is clamped to the `0.3 … 1.5` range the static counter already used.  A fit the
  holdout cannot separate, or a scope with too few labels, publishes a *neutral* snapshot —
  which reads as `1.0` and is the layer as it behaves today.  The policy JSON is never
  overwritten by a model; the model multiplies and the policy clamps.
  **The verdict is now graded per document** rather than credited to the whole block:
  `+1.0` names the failure that was fixed, `+0.5` shares a touched file or the command that
  went green, `0.0` proves nothing about that document and trains nothing, `-1.0` was about the
  failure that came back, `-0.3` rode along in a block that did not help; a document whose own
  header names nothing falls back to the block's verdict, marked `coarse` so a fit can tell the
  two kinds of credit apart.  A graded verdict enters the fit as its **direction** with weight
  `|label|` — half an observation of success, a third of an observation against — because the
  likelihood is Bernoulli's; putting the graded number in the target instead makes `+0.5` an
  observation that `p = 0.5`, which teaches nothing while still counting as labelled when the
  fit is gated, so a scope of plausible verdicts would clear the holdout and publish a model
  that had learned nothing.  Off by default (`learning.enabled`,
  `LLM_ROUTER_KNOWLEDGE_AGENTIC_LEARNING_ENABLED`); **the training log is written either way**,
  because a label can only be attached to an exposure that was recorded while it happened, and
  collecting with learning off is the recommended first step.
  `llm-router-agentic-kb-learner` drains the stream, fits every scope and publishes; the
  annotation carries `model_id`, `holdout_auc` and `labelled`, so "why was this recipe first"
  is answered from the response.
- **The injection votes live in Redis, not in one worker.** `FeedbackTracker` was built without
  a client, so a verdict counted in one process was invisible to the others and gone on
  restart — the shared reputation the changelog had promised never existed.  The plugin now
  wires one fork-aware handle through a lazy client to the tracker, the reputation store, the
  bandit, the training stream and the model snapshots; with no Redis each keeps the in-process
  state it had before, and `metrics.feedback.source` says which of the two is answering.
- **The learner keeps a labelled corpus, not just a batch.** The stream is a queue and a queue
  is trimmed, so a model fitted on the newest batch models only what survived ``MAXLEN`` this
  hour.  One hash per scope accumulates every settled judgement, keyed by (session, generation,
  document) so a re-injection of the same document in the same generation corrects one record
  instead of duplicating it, and an unlabelled exposure never overwrites a judgement.  The
  learner resumes from a persisted cursor: a restart reads only what has not been read, and the
  batch decides *which* scopes are worth refitting while the corpus decides what the fit says.
- **The holdout is split by session, not by sample.** A sample-level split puts two injections
  of one session on opposite sides, so the holdout measures how well the model remembers that
  session rather than how well it generalises.  A session now stays whole, and a corpus of a
  single session keeps it in the fit instead of training on nothing and scoring on the data it
  was meant to measure.
- **A label counts for what its evidence is worth.** A counterfactual verdict — the next turn
  mined a fix, and this document could have caused it — is stronger evidence than the coarse
  rule that credits everything that was on screen, and the two are no longer weighed equally:
  a coarse label moves the fit half as far, the direction of a verdict untouched.  A fit is
  gated on the weighted evidence it rests on (`learning.min_evidence`), reported as
  ``evidence``, and refused with `evidence_below_threshold` when a corpus of many weak labels
  has not actually observed much.
- **The learner command bounds its own work.** `--max-records` caps what one pass reads,
  `--window` bounds the corpus per scope, `--reset-cursor` starts at the head of the stream,
  and `--show` prints the weight vector of every touched scope.
### Fixed
- **A timeout the operator set is the timeout the layer uses.** Both `read.sse_timeout_seconds`
  (`…_SSE_API_TIMEOUT`) and `read.timeout_seconds` (`…_READ_TIMEOUT_SECONDS`) were clamped to a
  hard 10 s, silently: a deployment that set `40` waited 10 and read the failure as a broken
  timeout setting rather than as a ceiling nobody had set. The ceiling is now 60 s — a limit on a
  typo, not a policy — and a clamp that does bite logs a warning naming the setting it clamped
  and the number it settled on.  What bounds a request remains `read.timeout_seconds`, the
  budget.

### Added
- **A literal question is asked of the engine's text index alone.** When the tokens of a query
  are mostly identifiers — an exception class, a dotted module path, a `::`-qualified name, an
  underscored function — the request drops the vector leg: no embedding of the query, no round
  trip to the vector store, and no cosine drifting to whatever the code merely resembles. The
  share that decides is `read.fts_only_min_identifier_share` (`…_FTS_ONLY_MIN_IDENTIFIER_SHARE`,
  default `0.5`; `0` keeps every search on both legs), and it counts only tokens that look like
  code — the prose words of a question never count, so a query that merely *mentions* one
  identifier among its words still asks both legs. The engine answers such a search on its own
  scale, `score_scale: "fts"`: a `Postgres` text rank, which `escalation.strong_recipe_scores`
  leaves unnamed and so measures against the fallback `strong_recipe_score`. The fast path asks
  the engine that already supports it (`lexical_only` in the search options, added to the engine
  beside this change) and says so in the read's own log line.
- **The read path stops asking the engine a question a turn cannot have.** A turn
  that named no file, met no failure and ran no command builds its query from its
  own sentence alone, and below a dozen characters that sentence is an
  acknowledgement, not a question; such a turn now makes no search at all instead
  of three. A short turn that carries any of the three structured signals is still
  asked, so the guard is about what a query is built from, not about how briefly
  the operator wrote.
- **An identical question, asked twice in one generation, reaches the engine once.**
  `KnowledgeStore` keeps a short-lived, per-process memo of the answers it has
  already given to a frozen `SearchRequest`, so a read that asks the same question
  of the same collection more than once pays the transport only the first time.
  Only a real answer is remembered — a down engine and a collection not indexed
  yet are states that change, and both are asked again. A memo hit carries
  `reason: "memo"`, and the scoreboard counts it as `memo_hits`, the saving read
  against `searches_per_request`; the `llm-router-agentic-kb-stats` table gains a
  `memo` column.
- **`agentic_knowledge` measures itself, and says so on one line.** `KnowledgeMetrics` counts every read-path
  request onto a per-scope, per-day Redis hash and derives the rates the knowledge layer is judged by:
  `adoption_rate` (of the turns shown knowledge, how many went on to mine a verified fix),
  `knowledge_help_rate` (the same verdict per document), `failure_turn_share` and `loop_turn_share` (meeting a
  failure at all, and meeting it often enough to be stuck — the layer's own threshold decides which is which),
  and `escalation_rate`. Written once per request after the payload is final, in one pipelined `HINCRBY`,
  fire-and-forget: a Redis that cannot be written costs a metric and never a request. Without Redis the
  counters still count, in process only. The new `llm-router-agentic-kb-stats` command reads them back as a
  table or as JSON, for one repository or several, over a window of days; a rate with no measurement behind
  it is `null`, never `0.0`, so a quiet day cannot be read as a bad one.
- **The counterfactual replay scores retrieval, not just recall.** Each reading variant's report gains
  `precision_at_3` and `mrr` against the document that eventually closed the failure, ranked where the search
  returned it and at a document's first appearance. These are measured offline because a live request has no
  ground truth — only the replay knows which document the agent went on to need.
- **The knowledge the layer shows is scored by the engine, not only by one process.** The feedback the read
  path settles — a turn that verified a fix credits the document it was shown, a turn that repeated the same
  failure blames it — now increments a shared Redis hash and, once a policy interval has passed, is handed to
  the engine's `update_knowledge_quality` endpoint as absolute totals: a name the engine does not accept stays
  queued for the next attempt, and a verdict that outlives a restart is the deployment's verdict, not one
  worker's. A document is paired with the collection it was retrieved from while the block is still in hand,
  so the push asks the engine for the right one.
- **A source chooses how its full-text keywords are combined.** The keywords of a turn are a disjunction of
  candidate signals — an exception class, a file stem, an identifier — and the engine's default conjunction
  asked the full-text leg to hold all of them in one chunk, which no recipe does: with fifteen keywords the
  lexical half of a hybrid search retrieved nothing. A source may now say `keywords_match: "any"` or `"and"`,
  the mode sets the default for its sources, and a bucket that mixes sources keeps the strictest, because one
  request cannot hold two match modes at once. The shipped policy asks `any`, where the failure shapes the
  query, so a stack trace retrieves the recipe mined from the same stack trace by literal token again.

### Changed
- **`agentic_knowledge` reads the escalation threshold on the scale the recipe was scored on.**
  `escalation.strong_recipe_scores` states a bound per retrieval scale (`cosine`, `rrf`,
  `reranker`, or `""` for a hit whose scale the engine did not state); `escalation.strong_recipe_score`
  stays as the fallback for a scale the table leaves unnamed, so a deployment that set only the float
  keeps exactly the behaviour it set. A row may be `null` — "decide by identity, not by score" — which
  is what the shipped policy now says for `reranker`, replacing the special case that was hard-coded
  there: a squashed cross-encoder logit is not a relevance, and one number read on three scales was
  three silent rules. `EscalationPolicy.strong_recipe_reason` asks
  `AgenticKnowledgeConfig.threshold_for_scale(hit.score_scale)` per hit, and the start-up check names
  the label a bound came from instead of assuming one scale for the whole deployment.
  Overridable as JSON with `LLM_ROUTER_KNOWLEDGE_AGENTIC_STRONG_RECIPE_SCORES`.
- **A new recipe retires the older recipe of the same failure.** A recipe is keyed by the failure it
  solves, not by the patch that solved it, so a second recipe for the same signature is a correction:
  the first fix met a wall that is no longer there, and a reader who is shown it walks into the same
  moved wall. The worker keeps the pairing in Redis under the failure's digest, and a recipe older than
  `write.supersede_after_days` — one day by default, zero for off — is deactivated on the indexing of the
  newer one: taken out of the search, not deleted, so the quality the verdicts put on the document
  survives the retirement, and a reader who asks by name still finds it. A same-day refinement retires
  nothing.

