# Changelog

All notable changes to the `llm-router-plugins` package are documented here.
Versions follow the `MAJOR.MINOR.PATCH` scheme; a new entry is added for every
release tagged as `v<version>`.

## Summary

| Version | Changelog |
| ------- | --------- |
| 0.1.10 | **Added:** the read path asks a better question and answers it with less — keywords mined from the user's sentence, `path_scope_for` scoped by shared module instead of common prefix, `max_results` derived from the mode's source limits, and the debug query built from the line that names the failure; a contested top-5 is re-asked through the cross-encoder before the block is built (`rerank_when_contested`, off by default); the signature bonus is continuous (Jaccard of the failure tokens) and a superseded same-file recipe is damped; lockfile-only patches stop quoting themselves into recipes and a reverted change is mined as `reverted_paths`; `--bootstrap-git` seeds a scope from the fix commits of its own history; the attribution window spans 3 requests with a 0.85-per-turn discount and a verdict is checked against the change itself — `adopted` at ≥40% diff overlap, unattributed below 5%, `reverted` at −1.0; the ranker gains a `pairwise` (RankNet) objective beside `pointwise`, a cross-repository meta-prior for cold scopes (`meta_prior_enabled`, off by default), and a 22-feature candidate (interactions, loop level, `is_human_doc`, `patch_sim_prior`, `query_intent_align`, `error_class_match`) under `SNAPSHOT_SCHEMA` 2; the tune gains `--search bayesian` (seeded TPE on the multi-criteria objective) and `--write-policy`, which writes the chosen point as a reviewable `agentic_knowledge.tuned.json` proposal the layer never loads. **Changed:** snapshots of schema 1 are refused, not read under the wrong names. **Fixed:** the attribution window no longer drops an entry a no-verdict turn was not meant to judge; `_commit_patch` shows the patch in the repository it mines from; `LabelStore` logs a failed write instead of raising; the tune's `--search`/`--evaluations`/`--write-policy` flags exist. |
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
- **The miner quotes fairly, and says when it cuts.** The diff a document
  carries is spent file by file under a budget: an envelope is split into its
  per-file sections, source files are quoted before generated bookkeeping
  (`dist/`, `build/`, lockfiles, snapshots, the miner's own list), a section
  that fits only partly is cut in front of the next `*** Update File:` header
  rather than in the middle of a hunk, and every file the budget could not
  reach is named in a marker — a partial document that reads as complete
  sends the next session to apply half of a fix. The kinds that carry a whole
  task (`playbook`, `task_solution`) quote under a wider ceiling (8000,
  against the 4000 of a single failure's fix; the read path already holds the
  real ceiling, the assembler quotes 1800–3000 of whatever is retrieved), and
  a ceiling an operator lowered below a kind's own wins for every kind. The
  cut is counted: each mining pass reports a `truncated` diagnostic, the layer
  records `mined_docs` / `mined_truncated` per scope, and `stats` prints them
  as `mined` / `trunc` columns beside the new `truncated_share` rate.
- **The read path asks a better question, and answers it with less.** `_extract_keywords`
  mines the user's own sentence as well as the failure and the command — backticked
  identifiers, `snake_case` and `CamelCase` tokens included — so a `plan` or `implement`
  turn whose query was empty before now asks the text index something, and agent verbs
  stop counting as keywords. `path_scope_for` reads the *module* the paths share instead
  of their common prefix, so `src/auth/jwt.py` and `tests/auth/test_jwt.py` scope the
  search to `auth` instead of zeroing the filter; a scope that would filter everything
  back out is dropped, not applied. `max_results` is derived from the mode's source
  limits as `max(6, min(limits × 2, 12))` — the old floor of 15 made the engine fetch
  150 vectors per query and queue for seconds — and the debug query is built from the
  line that names the failure, not the `Process exited` wrapper or the traceback frames
  around it.
- **The store re-asks a contested result before the block is built.** A `debug` request
  whose top hits sit within `rerank_contested_gap` of each other (default `0.05`) is
  answered by a second pass over the top five with the cross-encoder, under the same
  deadline the first search ran under: the cost is paid only when the engine was not
  confident, and a rerank that cannot finish in time is the first pass's answer, not a
  timeout. `rerank_when_contested`, `rerank_contested_gap` and `rerank_top_k`
  (`…_RERANK_WHEN_CONTESTED`, `…_RERANK_CONTESTED_GAP`, `…_RERANK_TOP_K`) keep it off by
  default until a deployment's CPU can spare it.
- **The assembler ranks by resemblance, not by coincidence.** The signature bonus is
  continuous: a failure apart by one variable name from the document's earns a share of
  the full bonus in proportion to the Jaccard of their tokens instead of nothing. A
  recipe that is superseded — an older document for the same file whose failure the
  newer, better-attested document also names — is damped by half a vote, so a block
  does not carry two fixes for one file when one of them is what the agent stopped
  doing. A single, highly relevant recipe may spend up to 3000 characters instead of the
  shared quote ceiling, because a fix that is cut off is a fix that is not applied. The
  block opens with `> [!IMPORTANT]` and its dead ends under
  `## Known dead ends (DO NOT ATTEMPT)`, which the models that repeat abandoned
  approaches are least likely to ignore.
- **The miner writes less noise, and reads an undo as a verdict.** Patches that change
  nothing but lockfiles or build output no longer quote themselves into a recipe; a
  `git checkout --` / `git restore` / `git revert` of a file the turn just changed is
  mined as `reverted_paths`, the explicit negative signal that the advice was applied
  and rejected; `turn_added_lines` hands the credit assignment the side of the adoption
  comparison that cannot be recomputed later.
- **A repository can bootstrap its knowledge base from its own history.**
  `llm-router-agentic-kb-learner --bootstrap-git REPO_ROOT` (with `--bootstrap-limit`
  and `--bootstrap-since`) reads the commit subjects that say a change repaired
  something — `fix`, `bugfix`, `hotfix`, `repair`, `correct`, `revert` — and publishes
  one recipe per file each touched, with the commit's patch quoted and the subject as
  the signature, through the same publisher the proxy uses, so re-running a bootstrap
  is a no-op. A cold deployment on its first project has no meta-prior to start from and
  no labels of its own yet; the fix commits it already made are the closest thing to
  either, and a commit that only bumped a lockfile is a dependency, not a fix, and is
  skipped. A repository without a git history is nothing, not an error.
- **The verdict is attributed across the turns it actually spans, and checked against
  what the agent did.** A request's documents are kept in a window of
  `_ATTRIBUTION_HORIZON` (3) requests instead of the one turn the layer used to claim,
  so an agent that is shown a fix, edits on the next turn and runs the test that proves
  it on the one after is credited: the verdict walks the window from the newest request
  to the oldest, each injection judged at its own delay, discounted by
  `_ATTRIBUTION_DISCOUNT` (0.85) per turn of it, and a turn that proves nothing keeps
  the window it was given — a no-verdict is not a verdict of zero, and the entries wait
  for one. A document is credited once per verdict, at the nearest injection that
  showed it: three consecutive showings and one green test are one piece of evidence,
  not three, because counting the same evidence three times is the self-reinforcement
  the monitors exist to catch. **The label is now checked against the change itself.**
  A document that shares a file with a resolved turn is credited `1.0` as `adopted` when
  the turn repeated at least 40% of the lines the document's diff adds — shown, adopted,
  green is the only causal chain the layer can observe; it is logged as an exposure with
  no label when it repeated less than 5% of them — same file, none of the change, the
  turn had its own reason to be there, and a credit it did not cause is not a credit.
  A document about a file the agent reverted is labelled `-1.0` as `reverted`, whatever
  else the turn shows: applied and then undone is the strongest negative available
  without waiting for the failure to come back.
- **The ranker fits the ranking, not just the labels.** `learning.loss` gains
  `pairwise` beside the original `pointwise`: the objective is then the
  log-likelihood of the order, one comparison at a time — the probability that the
  document a request credited should have stood above the one it blamed — built only
  from pairs of opposite judgements of the *same* request, capped at 20000 and taken in
  a deterministic order so the same corpus fits to the same weights. The Newton step is
  RankNet's, over the difference of the two feature vectors, with the same ridge prior
  and the same backtracking as the pointwise fit. A corpus whose verdicts never
  contradict a request has an order to learn and nothing to learn it from, and the fit
  falls back to pointwise — and the report names which of the two actually ran, because
  "pairwise was asked" and "pairwise fitted" are two different answers.
- **A cold repository starts from what the others learned.** `compute_meta_prior`
  takes the mean of the weight vectors the scopes this pass *accepted* — a fit the gate
  refused carries the neutral vector, and a neutral vector is not a repository's
  opinion, it is the absence of one — and refuses to write it at all below
  `meta_prior_min_scopes` (3) accepted scopes, because one repository's weights are
  that repository's accidents and a prior of accidents is worse than the policy. The
  learner reads the prior once per pass (`meta_prior_enabled`, off by default), fits
  every touched scope from it as a MAP prior, and rewrites it from what this pass
  accepted; a pass that accepted nothing leaves the prior alone, because a pass that
  saw no evidence is not a reason to forget the evidence an earlier pass saw. The prior
  lives at `{prefix}:meta_prior` under the same schema the snapshots use, and a prior of
  another schema or the wrong length is read as no prior.
- **The candidate carries the state the document sits in.** `CandidateFeatures` grows
  from 15 to 22 features, at the end of the vector so the order of the first fifteen
  is unchanged: `score_x_path` and `sig_x_cmd` (the interactions the plan named),
  `agent_loop_level` (saturating at eight — the first try wants the general and the
  third wants the specific, the verified, the about-this-failure), `is_human_doc`
  (documentation is prose about how the repository does things; a recipe is a record of
  what happened, and a corpus that varies only one of the two cannot identify the
  weight of the other), `patch_sim_prior` (how much of the document's diff the turn's
  own change repeats — the label the credit assignment reads, as the feature the model
  reads it with), `query_intent_align` (a fix in debug, a procedure in implement, a
  page in plan — whether the match is the kind of document the turn is looking for,
  which text matching cannot see) and `error_class_match` (the class of the exception,
  `KeyError` not the message, matched case-folded; both halves of the pair must name
  the class). `SNAPSHOT_SCHEMA` is now 2, and a snapshot of another schema is refused,
  not read under the wrong names.
- **The tune searches the space instead of walking it.** `--search bayesian` replaces
  the grid with a TPE walk on a `--evaluations` budget (default 40): the first eight
  measurements are uniform, each one after that is proposed by the ratio of the
  Parzen-window density of the measurements that earned most to the density of the
  rest, half the candidates drawn from each, and a proposal that repeats a measured
  point is resampled and then given up on in favour of a uniform draw. The search
  ranks on the multi-criteria objective the plan named —
  `hit + 0.3·recall − λ·chars − 1.5·max(0, floor − precision)` — because a point that
  cannot keep the precision floor pays a penalty that grows without bound as it misses
  it, and a very high hit rate cannot buy its way under a constraint the number
  contains. The walk is deterministic under `--seed`: the same dataset, budget and seed
  propose the same points in the same order, which is the only reason a report saying
  "the search found this" can be checked by anyone but the machine that found it.
  `--write-policy` writes the chosen point next to the report as
  `agentic_knowledge.tuned.json`, a proposal with the numbers it claims so the merge
  into the real policy is a diff against the evidence: the layer never reads the file,
  and a tool that edits the policy it measured against has answered its own question.
- **The feature table the fit can move is named in the report.** A corpus that varies
  none of a feature's values cannot identify its weight, and the report of a fit now
  says which features it could move — the varied ones inside the learnable set — so a
  scope that "learned nothing" can be read as "the corpus held no variation to learn
  from" rather than as a failure of the fit.

### Fixed
- **The attribution window no longer loses a verdict a turn early.** `_settle_feedback`
  popped the whole window on every request, so a turn that neither proved nor
  disproved what the previous one showed dropped the previous entry before the next
  verdict could read it: the multi-turn credit the window exists for never arrived.
  The entries are now walked, and only the ones the verdict of this turn read leave;
  the rest stay, capped at the horizon by the arrival that keeps writing them.
- **`_commit_patch` runs in the repository it shows the patch of.** The `git show`
  that quotes a bootstrap commit's change to a file was run from the process's working
  directory, so on a machine where the layer's own checkout is a different repository
  it resolved the commit — or the path — against the wrong history. The repository the
  bootstrap was asked on is the repository the patch is shown from.
- **`LabelStore` reports a failed write instead of raising.** The store's `merge`
  caught a Redis failure and called a `_warn` helper it did not define, so the first
  write failure of a pass turned the pass into an exception rather than into a logged
  warning and a local copy of the corpus.
- **The tune's `--search`, `--evaluations` and `--write-policy` flags exist.** `main`
  read all three from the parser, and `build_parser` defined none of them, so the
  Bayesian search could only be reached by calling `tune()` directly; a run with
  `--search bayesian` on the command line failed with an unrecognized argument.

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
- **A disabled learner is absent, not a vote of one.** With `learning.enabled: false` the read
  path no longer asks the learner for a multiplier at all: a `1.0` from a disabled learner
  would have silently overwritten the static feedback counter's `0.3 … 1.5`, so "learning off"
  would still have been steering. The switch is checked first, and a disabled policy is
  treated as absent — no ranking vote, no snapshot read, no `model_id` in the annotation next
  to `enabled: false`. The collection is not off with it: exposures keep landing in the
  training stream, because a label can only attach to an exposure that was written down while
  it happened.
- **The replay measures what the learned path adds.** The counterfactual replay of
  `knowledge/agentic_knowledge/evaluation.py` gains a fourth variant, `read_learned`, beside
  `no_kb`, `read` and `read_escalate`: it walks the sessions in order with a real
  `LearningPolicy` in memory — one per scope, because the holdout split is by session — and
  the anti-leak rule stays a rule about time, the model a request may see having been fitted
  on labels already earned when that request was scored. The report gains
  `model_uplift_vs_read` — the difference of the two hit rates with a bootstrap interval
  resampled by session (200 draws, fixed seed), because the turns of one session are not
  independent evidence; one session reports a point and no interval, and says so — and
  `char_efficiency`, `turns_saved` per `injected_chars_per_turn`, the number the layer is
  actually paid for, because knowledge costs tokens. A per-scope appendix records how often
  the fit ran and which gate refused it, so "the model never fitted" and "the model fitted
  and was refused" stay two different answers.
- **A published fit keeps what it replaced, and can go back.** A publish now retires the
  snapshot it displaces into `{key_prefix}:model_history:{scope}`, newest first, deduplicated
  by `model_id`, capped at `learning.snapshot_history_size` (8; 0 keeps only the live key),
  and `llm-router-agentic-kb-learner --history` prints the model in force and what a rollback
  could reach, with each row's own holdout AUC and sample count. `--rollback <model_id>`
  copies one of them back to the live key and retires what it displaces, so a rollback is
  itself reversible; an id that is not in the history is refused, because a rollback to a
  model nobody fitted is a fresh publish wearing a disguise.
- **The scoreboard reads the learned state of a scope.** `llm-router-agentic-kb-stats
  --learning` prints, per scope: the model in force and the evidence it was fitted on, the
  history a rollback could reach, the label corpus it rests on, the bandit's value per
  `(mode, kind)`, and the self-reinforcement monitors of the training log — the share of
  documents with more than 50 exposures and the heaviest of them, the trend the plan's
  "does it grow monotonically" is about. The scoreboard rows are unchanged; `--json` gains a
  `learning` block for a dashboard to consume.
- **The policy's thresholds can be searched on a replayed dataset.** `llm-router-agentic-kb-tune`
  walks the Cartesian product of the values the operator names — `min_similarity`,
  `strong_recipe_score`, `loop_threshold`, and a factor on each source's shipped weight —
  replays every point, and reports the one maximizing `recipe_hit_rate − λ ·
  injected_chars_per_turn` under the constraint `escalation_precision ≥ --precision-floor`
  (a missing floor is 1.0: only escalate when it is the loop). The report is JSON and
  markdown with the winner and the deltas against the shipped policy on the same sessions, so
  a pull request can quote it; a point below the floor is reported with the reason and not
  selected. It writes nothing to the policy: the file is the operator's to change, and a
  tune that edits the policy it measured against has already answered its own question.
- **The annotation names the multiplier of each candidate.** The read's `learning` block gains
  a `candidates` map, one entry per injected document: the multiplier it got and the three
  factors that made it — the document reputation, the per-scope model (with its exploration
  flag) and the source bandit's value-per-character — so "why was this recipe first" is
  answerable from the response itself, not from a log nobody kept.
- **How the router learns, end to end.** `agentic_routing/README_LEARNING.md` writes down
  every mechanism that adapts behaviour over time or consumes labelled data: the frozen
  index, the calibration loop and its grid, the session memory, and the knowledge learned
  path — features field by field, labels and how traffic becomes a target, the Redis layout,
  the publishing gates, the `read_learned` replay and its uplift interval, the response
  annotation, the threshold grid and its objective, and what is never learned.
- **An exposure remembers whether the ranking explored.** Every training-stream record
  carries `exploration`: `explore` when the model's Laplace spread was drawn for that
  document on that request, `greedy` when its mean was taken.  It is written with the
  exposure and not reconstructed, because the draw belongs to the request — an off-policy
  reading of the corpus (IPS, a doubly-robust estimate, or the plain question of how much
  of what the layer learned was paid for by exploring) needs the policy that produced the
  vote, and that policy is gone by the time a verdict does.  A document ranked twice in a
  generation and explored once counts as explored, and the ledger of draws is bounded to a
  generation's window: a draw older than that is a fact about a ranking nobody is holding.
- **The replay measures the reputation loop on its own.** The counterfactual replay gains
  `read_reputation` beside `read_learned`: the same walker with the learner's other two
  loops held back — the document's Beta posterior, no fitted weights, no source arm, in a
  namespace of its own — and the report gains `reputation_uplift_vs_read`, the same
  session-bootstrapped interval as the model's.  It exists because a learned variant that
  lost ground said nothing about **which** of the three loops lost it: the reputation
  uplift is the phase-2 criterion read on its own, and the difference between the two
  uplifts is what the fitted model and the arms add on top of it.
- **A learner pass holds a lock, and only one pass at a time.** A pass now takes
  `{prefix}:learner:lock` (`SET NX EX learning.lock_ttl_seconds`, four hours by default)
  before it drains anything, and a pass that finds it held logs one line and exits without
  fitting: two overlapping passes read the same cursor, publish over each other, and leave
  two histories with no record of which fit won — which is how a cron whose previous run is
  still going, or a second impatient operator, used to end up deciding what a scope serves.
  Release is a compare-and-delete on the token the pass was given, so a pass that outlived
  its own TTL cannot delete the lock a later pass has since taken.  `--force` writes past a
  held lock when an operator has decided that pass will not finish; `--lock-ttl` moves how
  long a crash is overridable in.
- **The scoreboard reads the shape of the corpus, and how much of the log was paid for
  exploring.** The `--learning` view's corpus block gains a **per-feature mean and standard
  deviation** over the labelled rows: the shape of the input the next fit will read, which is the
  only way "the fit refused because the input moved" becomes a claim anyone can check, and a
  feature whose standard deviation is `0` is a feature the fit prunes before it starts (the
  hand-printed view names how many of the fifteen that is).  A scope with nothing labelled gets
  no distribution at all, because a distribution over unlabelled exposures describes the traffic
  rather than the corpus.  The log's monitors gain the window's **explored share** — the records
  whose ranking drew the model's Laplace spread instead of taking its mean — counted from the
  record's own `exploration` field, because a greedy vote is one the policy already serving would
  have cast anyway, and a corpus of nothing else tells a fit what the current model believes, not
  what is true.  A record written before the field existed counts as greedy, the behaviour it was
  written under.
- **The tune can no longer meet its escalation floor by escalating on nothing.** A precision
  floor is satisfiable by a point whose loop threshold is high enough never to fire: it never
  escalates, it never escalates wrongly, and it reads as flawless on the one number the search
  was constrained by, while catching none of the loops the dataset recorded.  The replay now
  counts the ground-truth loops each variant was shown and reports the variant's
  `escalation_recall` — the share of them it escalated on — and `llm-router-agentic-kb-tune`
  gates a point on `--recall-floor` beside the precision floor, refuses a point that breaks both
  with both reasons, and prints the number in both delta tables and in the objective line, where
  the absence of a floor is said out loud rather than left for a reader to notice.  A dataset
  with no loops meets the floor by definition: nothing to catch is not a recall of zero.
- **The tune selects on calibration and is judged on a holdout.** `--holdout-session`
  names sessions the grid may not select on (the operator's own holdout wins over any
  fraction), `--holdout-fraction` splits the dataset by a hash of `--seed` and the session
  id so two runs split the same way, and a fraction never eats the calibration.  Every grid
  point now carries what it earned on each half and the report holds two delta tables — the
  one the winner was chosen on and the one it was judged on — because a point selected on
  the sessions it reports has been fitted to them, which is the criterion the plan set for
  this command and the tool did not enforce until now.  A run with no holdout named says so
  in the report instead of printing a column of zeros.

- **A learner's reputation now keeps the clock its owner keeps.** `LearningPolicy` built
  its `DocReputation` without the injected clock, so a replay that counts its turns and a
  test that freezes time both decayed reputation against the wall clock instead: a
  posterior half-decayed against its own timestamps, an exposure count that truncated a
  judgement away, and a report that could not be reproduced from its seed.  The policy's
  clock is passed through now, and the replay's counter is the only clock the learned path
  reads.
- **The ranker's feedback is the deployment's, not one process'.** `all_feedback()` answered
  from process memory, so every worker ranked the same recipe by the votes *its* requests had
  cast, and a restart zeroed the whole reputation the static counter had been built to keep
  shared. The shared hashes are now read in one `SCAN` pass over the feedback namespace with a
  pipeline of `HGETALL`s, trusted for 5 s so a keyspace walk is paid at most once per worker
  per generation, and a shared read that fails falls back to the local verdicts rather than to
  an empty answer — the local ones are re-read on every call, because without a shared store
  they are all there is. `metrics.feedback.source` says which of the two answered.
- **The first verdict of a freshly booted process is no longer held back.** The rate-limit
  of the feedback push compared `time.monotonic()` against a "never pushed" sentinel of
  `0.0`, but the clock has no fixed origin: on a machine whose uptime is below the interval
  — a runner, a container, a proxy restarted in the first hour — the sentinel reads as
  "pushed a moment ago", and every settled verdict of that boot was gated away silently.
  "Never" is `None` now, and the interval counts from the push that actually reached the
  engine.
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
- **The snapshot schema is 2.** Snapshots carry the 22-feature order, and
  `load_snapshot` refuses a snapshot of schema 1 instead of reading fifteen weights
  under the first fifteen of twenty-two names: the order of the weights is the schema,
  and a misread is worse than a refusal, because a refusal is served by the policy
  until the model is republished and a misread is served by a model nobody can
  explain. The learner republishes on the first pass after an upgrade.

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

