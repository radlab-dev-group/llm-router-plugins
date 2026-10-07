# Codex routing calibration and evaluation

Wersja polska: [CODEX_EVAL_HOWTO_PL.md](CODEX_EVAL_HOWTO_PL.md).

## 1. What you are actually calibrating

This guide primarily covers `agentic_routing_codex` and the implementation in this repository described during the October 6–7, 2026 verification. The values and results below are reference points from that review, not a guarantee that they remain current after subsequent configuration or code changes.

**The goal is to identify the current work mode correctly, and only then select the model assigned to it.** You are not tuning routing around the name of a particular response-generating model.

There are three independent mechanisms:

1. **Structural rules and work phase** — an explicit mode, Plan Mode, titles, compaction, tool calls, and their results.
2. **Heuristics** — scoring keywords, phrases, and regular expressions.
3. **Semantics** — embedding similarity between the current activity and mode descriptions and examples.

Optional phase memory in Redis complements these mechanisms.

Resolution order:

```text
explicit mode
→ request class: title / compaction
→ Plan Mode declaration
→ current phase / trustworthy phase memory
→ heuristics
→ embeddings
→ fallback
→ mode-to-model mapping
```

**Important:** if heuristics already make an incorrect decision, lowering the embedding threshold will not fix it. The semantic layer is not consulted in that case.

Calibration in the current system **does not mean training or fine-tuning the embedding model**. It means selecting an embedding model, descriptions, examples, indexing parameters, and acceptance thresholds.

## 2. What the configuration files and settings are for

| File / setting | Purpose |
|---|---|
| `llm_router_plugins/resources/routing/agentic_routing_codex.json` | Codex modes, rules, semantic examples, thresholds, and mode-to-model mappings. |
| `tests/data/codex_routing_quality.json` | Requests with expected modes. Used for evaluation; not automatically added to the embedding index. |
| `tests/data/codex_routing_baseline.json` | Frozen earlier predictions for comparisons on common cases. |
| `codex-routing-holdout-verification.json` | Saved output from a particular replay. A report, not configuration. |
| `LLM_ROUTER_MODELS_CONFIG` | Model configuration in the router application: availability, provider, serving parameters, etc. Does not define examples for work-mode classification. |
| `simple_semantic.json` | Configuration for a separate heuristic plugin. Despite its name, it does not require embeddings. |
| `semantic_biencoder.json` | Configuration for a separate embedding-based general routing plugin. Does not replace the Codex configuration. |
| `agentic_routing_claude_code.json` | Configuration for a different plugin, for Claude Code. |

Changing `semantic_biencoder.json` **does not automatically calibrate Codex**. The embedding engine is shared, but the configurations and decision layers are separate.

### Key Codex configuration fields

| Field | Meaning |
|---|---|
| `embedding_model` | The model that creates vectors; a local path or model identifier. Not the model answering the user. |
| `settings.trigger_model` | The alias that activates the plugin, `auto_codex` by default. |
| `settings.fallback_mode` | Mode selected when no mechanism resolves the request, currently `implement`. |
| `codex_modes[].name` | Mode name, such as `review`. |
| `codex_modes[].model_name` | Model used after this mode is selected. |
| `codex_modes[].description` | Description of the mode's meaning, indexed semantically. |
| `codex_modes[].examples` | Examples of activities belonging to the mode, indexed semantically. |
| `keywords`, `phrases`, `patterns`, `weights` | Heuristic signals, not embedding examples. |
| `settings.phase` | Activity interpretation rules: commands, tools, test paths, announcements, etc. |
| `settings.semantic` | Indexing and semantic acceptance parameters. |
| `settings.vector_store_path` | Optional directory for a persistent semantic index. Not session memory. |

**Redis is a separate mechanism.** Configure its connection using independent environment variables, not by adding connection details to the embedding configuration or automatically reusing Redis settings for auth.

## 3. Understanding scores and similarity

### Heuristic scoring

Current default weights:

```text
keyword = 1
phrase  = 2
pattern = 3
```

You can change them through `settings.heuristic_weights`, and override the weight of an individual keyword or phrase.

Example syntax for fields within one mode:

```json
{
  "keywords": ["pytest"],
  "weights": {"pytest": 3},
  "phrases": ["run unit tests:4"],
  "patterns": ["\\brun\\s+tests\\b"]
}
```

Overlapping matches are deduplicated: **do not assume every matching rule always adds to the total**. Repeating a single word does not increase the score indefinitely either.

Acceptance requires:

```text
best score > 0
best score >= heuristic_min_score
best score > second score
best score - second score >= heuristic_min_margin
```

Current configuration:

```text
heuristic_min_score  = 3.0
heuristic_min_margin = 1.0
```

For heuristics, `routing.similarity` is a transformation:

```text
similarity = score / (score + 1)

score = 3 → similarity = 0.75
```

**This does not mean a “75% chance that the mode is correct.”**

### Embedding scores

Descriptions and examples are converted into normalized vectors. The engine compares them with the request vector using cosine similarity.

With `aggregation: "per_target_top_k"`:

1. Similarities are retrieved for all indexed chunks.
2. The same number of best chunks is selected for each mode.
3. Their similarities are averaged.
4. This produces the `all_scores` mode ranking.

The effective number of chunks per mode is:

```text
min(top_k, number of chunks in the smallest class)
```

Therefore, `top_k: 3` **does not mean three modes in the result**. It means up to three best chunks for each mode.

Acceptance requires:

```text
s1 >= threshold
s1 > s2
s1 - s2 >= min_margin
```

Here, `s1` is the best mode's score and `s2` is the second-best score.

Current settings:

```json
{
  "threshold": 0.51,
  "min_margin": 0.05,
  "aggregation": "per_target_top_k"
}
```

Examples:

| Best score | Second score | Decision with the current thresholds |
|---:|---:|---|
| `0.62` | `0.54` | Accept: sufficient score and margin. |
| `0.62` | `0.60` | Reject: insufficient margin. |
| `0.49` | `0.35` | Reject: score too low. |
| `0.62` | `0.62` | Reject: tie. |

**A cosine similarity of `0.62` does not mean a 62% probability of correctness either.** Do not compare it directly with heuristic `0.75` or structural `1.0`.

## 4. What actually goes into embeddings

The mode index contains:

- the mode name and its `description`;
- texts from `examples`.

Names of assigned response-generating models are not semantic evidence. `aux_title` and `compaction` are resolved structurally and do not participate in the main-mode semantic ranking.

The semantic query includes separate sections for:

- the intent of the current user instruction;
- current phase context: the agent's latest message and a structured description of tool activity.

**Raw contents of a read file or tool result are not added as the phase description.** This prevents the file's subject matter from replacing information about what the agent is doing.

Intent and phase sections are embedded separately, then their vectors are averaged and normalized.

| Parameter | What it controls |
|---|---|
| `intent_max_chars` | Character limit for the intent section; currently `2000`. |
| `phase_max_chars` | Character limit for the phase section; currently `2000`. |
| `classify_max_chars` | Separate history-parser budget; not an intent-versus-phase weight. |
| `chunk_size` | Chunk size for indexed descriptions/examples; currently `256`. |
| `chunk_overlap` | Chunk overlap; currently `64`. |
| `top_k` | Number of best chunks considered per mode; currently `3`. |

Increasing `phase_max_chars` does not directly mean “give the phase twice the weight.” It changes how much text is available.

## 5. Preparing a good calibration dataset

You already have `tests/data/codex_routing_quality.json`. Use it as a starting point, but stronger evaluation requires more independent sessions.

Minimal example of a separate dataset:

```json
{
  "schema_version": 1,
  "cases": [
    {
      "id": "cal-review-001",
      "split": "calibration",
      "source_session": "session-cal-001",
      "expected_mode": "review",
      "ambiguous": false,
      "input": [
        {
          "type": "message",
          "role": "user",
          "content": [
            {
              "type": "input_text",
              "text": "Review this module's error handling and describe the issues. Do not change files."
            }
          ]
        }
      ]
    },
    {
      "id": "holdout-review-001",
      "split": "holdout",
      "source_session": "session-holdout-001",
      "expected_mode": "review",
      "ambiguous": false,
      "input": [
        {
          "type": "message",
          "role": "user",
          "content": [
            {
              "type": "input_text",
              "text": "Assess the validator's correctness and identify risks without preparing a fix."
            }
          ]
        }
      ]
    }
  ]
}
```

For phase sequences, add:

- a shared `sequence` value;
- `session_id`, `thread_id`, `agent_name`, and `turn_id` in the case's `metadata` field;
- actual history prefixes: tool calls, identifiers, and results;
- optional `payload` for other request fields, such as `tools`, response format, and Codex metadata.

**Note:** during replay, the case's `metadata` is added to the request's `client_metadata`. An explicit mode override can be retained in `payload`.

Labeling rules:

1. The expected mode must follow solely from information available **before the decision**.
2. A label describes the current activity, not the overall task subject.
3. A historical router decision is not automatically a correct label.
4. A complete session belongs to either `calibration` or `holdout` — do not split nearly identical prefixes across the two sets.
5. Mark genuinely unresolved cases as `ambiguous: true`; do not use this to hide errors.
6. Remove secrets, but preserve the structure needed to recognize the phase.

Include difficult contrasts, such as:

- adding a mock in production code versus writing tests;
- GitHub integration versus Git history analysis;
- developing a testing strategy versus running tests;
- assessing code versus preparing a fix;
- a neutral file read after running tests;
- the phase sequence `implement → test → debug → implement`.

## 6. Running evaluate

### Automated tuning and evaluation

The Bash script runs the complete workflow (requires `jq`, Python with the project and `[ml]` dependencies):

```bash
bash scripts/codex-tune-eval.sh \
  --python /path/to/venv/bin/python \
  --thresholds "0.45 0.50 0.51 0.55 0.60" \
  --margins "0.02 0.05 0.08 0.10" \
  --output-dir ./workdir/codex-tuning-01
```

Use `--config`, `--dataset` and `--baseline` for custom files; `--help` lists all options. The output directory must be new. Without `--output-dir`, a new `codex-eval-*` directory is created in the repository. User-supplied paths are relative to the invocation directory; the script can run from any directory.

The script evaluates the source configuration, then every `threshold`/`min_margin` pair **on calibration only**. It selects the highest `cascade.main_mode_accuracy`, then fewer missed and unnecessary mode switches, then higher `mode_accuracy`. An exact tie retains the source configuration. Only after freezing selection does it evaluate source and selected configurations on holdout. It does not train embeddings, improve examples or tune heuristics.

Outputs include `selection.json` (candidate ranking), `selected-config.json`, `holdout-comparison.json`, full reports and separate logs for every replay. Source configuration and dataset snapshots are retained. The production configuration is never overwritten and the winner is not deployed automatically — inspect precision/recall, special cases and holdout regressions before deployment.

Each candidate performs a real replay and reloads the embedding model, so the default grid can be expensive. Errors stop the workflow; logs and partial reports remain for diagnosis. Use `--no-semantic` for evaluation without embeddings: it skips tuning and reports deterministic variants on both splits. The script neither installs dependencies nor uses production Redis.

Run the following commands from the repository root. `python` should point to a stable interpreter in the project's PyCharm environment.

Semantic evaluation requires optional dependencies:

```bash
python -m pip install -e '.[ml]'
```

This is an installation instruction for you, not a record of an installation performed. Earlier semantic verification was blocked by an environment using Python `3.11.0rc1`; use a compatible, stable interpreter for these measurements.

### A. Reference run without embeddings

```bash
python -m llm_router_plugins.utils.routing.agentic_routing.codex.evaluation \
  --config llm_router_plugins/resources/routing/agentic_routing_codex.json \
  --dataset tests/data/codex_routing_quality.json \
  --split calibration \
  --no-semantic \
  > codex-calibration-deterministic.json
```

This produces the `deterministic` and `stateful` variants.

### B. Embedding evaluation on calibration

```bash
python -m llm_router_plugins.utils.routing.agentic_routing.codex.evaluation \
  --config llm_router_plugins/resources/routing/agentic_routing_codex.json \
  --dataset tests/data/codex_routing_quality.json \
  --split calibration \
  > codex-calibration-semantic.json
```

This evaluation adds `cascade` and `semantic_only`.

### C. Final holdout evaluation

After tuning is complete:

```bash
python -m llm_router_plugins.utils.routing.agentic_routing.codex.evaluation \
  --config llm_router_plugins/resources/routing/agentic_routing_codex.json \
  --dataset tests/data/codex_routing_quality.json \
  --split holdout \
  --baseline tests/data/codex_routing_baseline.json \
  > codex-holdout-semantic.json
```

`--split holdout` is the default, but specifying it explicitly is helpful. Use `--split all` for overall diagnostics, not for choosing thresholds using the control set.

### Important evaluator behavior

- It reads the specified `JSON` file; it does not apply the normal routing configuration overrides from environment variables.
- It builds a fresh in-memory index; it does not use the production persistent index.
- It does not call response-generating models.
- It does not use production Redis. Memory is replayed in an isolated local store per sequence.
- Without `--no-semantic`, it requires enabled semantic routing and `aggregation: "per_target_top_k"`.
- An embedding loading or lookup failure should abort evaluation rather than masquerade as a successful fallback evaluation.

**Therefore, setting a threshold through an environment variable does not change this CLI's result. For an experiment, write the threshold into the file passed through `--config`.**

## 7. Reading report variants

| Variant | What it measures |
|---|---|
| `deterministic` | The cascade without embeddings or session memory. |
| `stateful` | The deterministic cascade with phase-memory replay. |
| `cascade` | The cascade with embeddings but without session memory. |
| `semantic_only` | Embeddings for main requests; titles and compaction still use structural routing. |

**The current evaluator does not report a separate “memory + semantics” variant.** Do not treat `cascade` as a complete production simulation with Redis enabled.

`semantic_only` helps reveal ranking quality itself. It is not the “maximum possible result,” because it deliberately bypasses some stronger structural signals.

### Key metrics

| Metric | Interpretation |
|---|---|
| `mode_accuracy` | Fraction of correct modes among labeled cases. |
| `main_mode_accuracy` | Accuracy for main work, excluding titles and compaction. |
| `per_mode.precision` | Of the predictions of a particular mode, how many were correct? |
| `per_mode.recall` | Of the cases requiring a particular mode, how many were detected? |
| `mode_confusion` | Row: expected mode; column: predicted mode. |
| `sources` | Which layer made the decisions. |
| `fallback_reasons` | Reasons for fallback decisions recorded by the variant. |
| `semantic_acceptance_rate` | Fraction of labeled cases resolved by the `semantic` source; not semantic precision or the acceptance rate restricted to lookups. |
| `special_cases` | Separate results for titles and compaction. |
| `ambiguous_count` | Cases excluded from accuracy because of ambiguity. |
| `mean_routing_ms` | Mean replay time for a variant, not total production latency. |

For example, high `test` recall and low `test` precision mean tests are detected, but this mode also captures unrelated tasks.

For sequences, pay particular attention to:

- `unnecessary_switches` — a mode change when the expected phase did not change;
- `missed_switches` — no change when a transition was expected;
- `mean_switch_delay` — delay in detecting a new phase, measured in subsequent requests, not milliseconds;
- `censored_switches` — transitions for which the correct phase was not reached before the available fragment ended.

Correct switch counts alone do not prove correct mode selection. Always read them together with accuracy and the confusion matrix.

**Do not select a configuration based on `model_accuracy`.** If multiple modes share one model, an incorrect mode may look like a correct model selection.

## 8. A practical embedding calibration procedure

### Step 1: freeze a reference point

Save a report for the current configuration on `calibration`. Preserve the configuration and dataset used for the measurement.

The report includes their hashes in `metadata`. Also record the code version, embedding weights revision, and dependency versions — a configuration hash cannot detect replacement of model files at the same path.

### Step 2: identify where errors occur

Start by comparing `deterministic` with `cascade`.

- Error from `heuristic` → improve the heuristic.
- Error from `phase` → inspect activity rules and the parser.
- Error from `memory` → inspect metadata and phase lifecycle.
- Semantic error or unresolved fallback → inspect the embedding ranking.

Example for reading rankings if `jq` is available:

```bash
jq '.records[]
  | select(.ambiguous == false)
  | select(.semantic_only.all_scores != null)
  | {
      id,
      expected: .expected_mode,
      selected: .semantic_only.mode,
      source: .semantic_only.source,
      margin: .semantic_only.margin,
      ranking: .semantic_only.all_scores
    }' codex-calibration-semantic.json
```

Do not assess only the final mode: `implement` may be correctly selected as a fallback even though semantics accepted no class.

### Step 3: improve descriptions and examples

Good examples describe **the activity and the mode's boundaries**, not the repository's subject matter.

Example distinctions:

```text
plan:
“Develop a testing strategy; do not write tests yet.”

test:
“Add unit tests for the validator and run them.”

review:
“Assess the validator's correctness and describe issues without editing.”

implement:
“Add input validation to the existing module.”
```

Recommendations:

- use representative Polish and English phrasing;
- add varied paraphrases, not dozens of nearly identical sentences;
- explain distinctions in `description`;
- do not use preferred model names as classification clues;
- do not copy `holdout` cases into `examples`;
- avoid copying evaluation prompts verbatim even from calibration — use them to identify missing categories of phrasing.

If the correct class regularly ranks second, lowering `threshold` alone usually will not help: the wrong class will still win.

### Step 4: select threshold and margin

Once descriptions are stable, test a small grid of values, for example:

```text
threshold:  0.45, 0.50, 0.51, 0.55, 0.60
min_margin: 0.02, 0.05, 0.08, 0.10
```

This is **an example experimental range**, not a recommended optimum. A different embedding model may require a different range.

- Higher `threshold`: usually fewer acceptances and more fallbacks.
- Lower `threshold`: more acceptances, but potentially incorrect confident decisions.
- Higher `min_margin`: rejects competing classes with similar scores.
- Lower `min_margin`: accepts more borderline cases.

Example of creating a candidate configuration:

```bash
jq '.settings.semantic.threshold = 0.55
    | .settings.semantic.min_margin = 0.08' \
  llm_router_plugins/resources/routing/agentic_routing_codex.json \
  > codex-candidate.json
```

Then:

```bash
python -m llm_router_plugins.utils.routing.agentic_routing.codex.evaluation \
  --config codex-candidate.json \
  --dataset tests/data/codex_routing_quality.json \
  --split calibration \
  > codex-candidate-calibration.json
```

The CLI has no separate `--threshold` or `--min-margin` options and no automatic tuning. Supply parameters through configuration.

**Cost optimization:** when only thresholds change, the `all_scores` ranking remains the same. You can screen threshold pairs offline using saved rankings instead of loading the model every time. However, confirm the selected configuration with an actual replay of the complete cascade.

### Step 5: tune other parameters only afterward

Check these separately:

- `top_k`, such as `1`, `3`, `5`;
- intent and phase section lengths;
- `chunk_size` and `chunk_overlap`;
- other embedding models.

Do not change everything at once — you will lose track of what caused an improvement.

After changing the embedding model, calibrate thresholds again. The same cosine value need not have the same diagnostic meaning for different models.

### Step 6: select configuration by quality, not acceptance count

Priorities:

1. Fewer incorrect modes in main work.
2. Fewer unnecessary and missed switches.
3. No degradation in particularly important classes.
4. No regressions in explicit overrides, Plan Mode, titles, or compaction.
5. Acceptable routing cost and latency.

**Fewer fallbacks do not automatically mean better routing.** An incorrect semantic decision may be worse than a cautious fallback.

## 9. Calibrating heuristics

Tune heuristics separately, starting with a `--no-semantic` report.

Common actions:

- remove overly broad keywords that capture unrelated activities;
- add more unambiguous phrases;
- reduce the weight of an ambiguous signal;
- increase `heuristic_min_margin` if competing classes have similar scores;
- check local negations, such as “do not run tests”;
- account for Polish spelling both with and without diacritics.

`CodexModeScorer.rank_modes()` exposes a ranking with raw scores and `SignalMatch` matches. This is the right place to investigate “which word added points.” The CLI report does not contain a complete heuristic ranking.

Not all configured modes are heuristic candidates. In particular, merely adding keywords to `plan` does not make it a heuristic winner in the current cascade; planning has structural signals and a semantic path.

After improving heuristics, rerun the embedding evaluation too — changing an earlier layer changes the requests that reach semantics.

## 10. Confirming results and deploying configuration

After choosing a candidate on `calibration`, run the old and new configurations on **the same independent holdout**. Compare not just summaries, but individual lost and recovered cases.

If you change configuration to address errors after inspecting the holdout, that dataset is no longer an independent check. Keep it as regression coverage and prepare a new holdout.

The current corpus is small and has already been analyzed while implementing fixes. It is useful for regressions, but is not strong evidence of quality on new sessions.

### Deployment

Example of selecting a ready configuration file:

```bash
export LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_CONFIG=/path/to/codex-candidate.json
```

Selected production overrides:

```bash
export LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_SEMANTIC_ENABLED=true
export LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_SIMILARITY_THRESHOLD=0.55
export LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_MODEL=/path/to/embedding-model
```

Here, `MODEL` means the embedding model. `MODEL_TEST`, `MODEL_REVIEW`, etc. mean response-generating models assigned to modes.

`semantic.min_margin`, `aggregation`, and section budgets currently have no individual environment variable overrides; set them in `JSON`.

Changes are loaded when the plugin is created, so they require recreating the plugin / restarting workers.

### Persistent index

If you use `vector_store_path` or `PERSIST_DIR`, after changing:

- the embedding model;
- the set of modes;
- descriptions or examples;
- text chunking parameters;

**build a new index**. The safest approach is to specify a new, versioned directory. The current engine does not guarantee automatic index invalidation after every configuration change.

Changing only `threshold`, `min_margin`, `top_k`, or query budgets does not require re-embedding descriptions.

The `FAISS` index and Redis memory are independent. Do not clear all of Redis when calibrating embeddings.

## 11. What the current report tells you

In `codex-routing-holdout-verification.json` from the earlier verification:

| Metric | `deterministic` | `stateful` |
|---|---:|---:|
| Mode accuracy | `26/35 = 74.3%` | `29/35 = 82.9%` |
| Main-work accuracy | `23/32 = 71.9%` | `26/32 = 81.25%` |
| Unnecessary switches | `2` | `0` |
| Missed switches | `1` | `0` |
| `review` recall | `25%` | `25%` |
| `plan` recall | `33.3%` | `33.3%` |

This demonstrates an improvement from memory on these cases. **It does not demonstrate embedding quality** — neither variant uses embeddings, and `semantic_acceptance_rate` is `0`.

Low `review` and `plan` recall identifies areas to investigate, not an automatic reason to lower the threshold. First check whether the incorrect mode comes from a heuristic or whether the correct mode ranks highly in the semantic ranking.

The current baseline also has unverified historical provenance. Use it as a frozen regression snapshot, not as evidence of results “before the entire implementation.” `--baseline` compares only common cases with consistent labels.

## 12. Short workflow checklist

```text
1. Check that the interpreter and embedding model work.
2. Prepare calibration and holdout sets separated by session.
3. Save results for the current configuration without semantics.
4. Save results for the current configuration with semantics.
5. Identify the layer responsible for errors.
6. Improve descriptions / examples or the relevant layer's rules.
7. Tune threshold and min_margin on calibration.
8. Check incorrect modes, precision/recall, and switches.
9. Confirm the selected candidate on an independent holdout.
10. Deploy configuration and a fresh index, then inspect production logs.
```

**The key rule: optimize correctness of the current work mode, not high similarity, low fallback usage, or model-name accuracy.**