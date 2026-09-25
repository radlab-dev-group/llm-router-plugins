# Claude Code Model Swap (`agentic_routing_claude_code`)

Model-swapping plugin for the **Anthropic-Messages-style** requests emitted by the **Claude Code** CLI. It looks at the
model every request asks for, finds the Claude Code **tier** that owns that name, and rewrites the model to the one this
router actually serves. Sonnet traffic, Opus traffic, the Haiku calls Claude Code runs in the background and the
`opusplan` Plan Mode phase can each land on their own model, without any per-developer configuration.

The plugin exists because Claude Code has no gateway-side notion of a model mapping. Pointing it at your own backend
used to mean setting one environment variable per tier on every host — `ANTHROPIC_DEFAULT_FABLE_MODEL`,
`ANTHROPIC_DEFAULT_OPUS_MODEL`, `ANTHROPIC_DEFAULT_SONNET_MODEL`, `ANTHROPIC_DEFAULT_HAIKU_MODEL`,
`CLAUDE_CODE_SUBAGENT_MODEL` — and repeating it on every laptop, container image and CI runner. Here that mapping is
one JSON file the router reads once.

- **Plugin name (registry key):** `agentic_routing_claude_code`
- **Class:** `llm_router_plugins.utils.routing.agentic_routing.claude_code.plugin.ClaudeCodeRoutingPlugin`
- **Default config:** `llm_router_plugins/resources/routing/agentic_routing_claude_code.json`
- **Env prefix:** `LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CLAUDE_CODE_`
- **Dependencies:** none beyond the standard library — no embedding model, no FAISS

---

## What the plugin changes

`apply()` rewrites the payload in place and touches the model key(s) plus one annotation:

| Key                             | Value                                                          |
| ------------------------------- | -------------------------------------------------------------- |
| `model`                         | `model_name` of the matched tier, when the payload carried it  |
| `model_name`                    | the same target, when the payload carried it                   |
| `routing`                       | `plugin`, `similarity`, `mode`, `original_model`, `matched_model`, `match_type`, `field` |

Only keys the payload already carries are rewritten, and only those configured in `settings.model_fields` are considered
at all: a payload that names its model `model_name` never grows a `model` key, and a key outside `model_fields` keeps its
value. Everything else — `messages`, `system`, `tools`, `max_tokens`, `metadata` — is forwarded unchanged. The plugin
never rejects a request; it only ever picks a model.

**Fail-open, and silent about it.** A payload that is not a dict, carries no configured model key, whose model no tier
claims, or whose tier has no `model_name`, is returned as **the same object it came in as** — no rewrite, no log line.
There is deliberately nothing to grep for on that path: an unmatched model is normal traffic, not a signal. Anything
raised inside the plugin is caught, logged once as a warning, and the request still goes out unchanged.

**Fail-hard on configuration only.** No tiers, duplicated tier names, an embedded wildcard, or one exact model name
claimed by two tiers all raise at plugin construction, so the router refuses to start rather than misroute quietly.

---

## Quick start

### 1. Install the plugin package

```bash
# from a checkout of this repository, into the environment the router runs in
pip install -e .              # this plugin needs nothing extra
```

Unlike the semantic and Codex routing plugins, this one has no `[ml]` extra: it never embeds anything.

### 2. Load the plugin into the router

The plugin is a *utils* plugin: it runs inside the llm-router request pipeline and is selected by a comma-separated
list of plugin identifiers.

```bash
export LLM_ROUTER_UTILS_PLUGINS_PIPELINE="agentic_routing_claude_code"
```

The router resolves the name against `llm_router_plugins.utils.registry.MAIN_UTILS_REGISTRY`, instantiates it **once per
process** (`UtilsRegistry`) and calls `apply()` on every prepared request. Order matters when several utils plugins are
wired together: the first one to rewrite the model wins the downstream dispatch, so put this plugin before any semantic
routing plugin that might answer the same request.

### 3. Declare the models in the router

Every value in `claude_code_modes[].model_name` must be a normal, active model in the llm-router model config
(`LLM_ROUTER_MODELS_CONFIG`) with a reachable provider. A name the router does not know is a hard downstream failure
(`model not found`), *not* a plugin error.

Check one thing in your router: **whether it resolves the requested model before or after the utils pipeline.** If it
resolves first, the Claude Code IDs themselves (`claude-sonnet-5`, `claude-opus-5-5`, …) must also be declared — as
aliases is cleanest — otherwise the request is refused before this plugin ever sees it.

### 4. Point Claude Code at the router

```bash
export ANTHROPIC_BASE_URL="http://<router-host>:8081/anthropic"
export ANTHROPIC_AUTH_TOKEN="<router API key>"
claude
```

Keep the Claude Code tier variables at the model IDs your users expect to see (`/model opus` should still *say* Opus);
the swap happens on the way through. That is the whole point: the CLI keeps its tier vocabulary, the router decides what
serves it. To move a tier to another model, edit this plugin's config and restart the router — no client changes.

### 5. Verify

One log line per swapped request:

```text
Claude Code model swap: claude-sonnet-5 -> qwen/Qwen3.8-Flash-Next (mode=sonnet, matched=claude-sonnet-5, match_type=literal, field=model)
```

No line at all means the model matched no tier — see [Troubleshooting](#troubleshooting). For a check without a CLI or a
running router, see [Verifying the plugin](#verifying-the-plugin).

---

## How it works

### Step 1 — normalize the requested name

Claude Code resolves its own `fable` / `opus` / `sonnet` / `haiku` aliases before sending, so the gateway sees a
versioned ID — and the same tier arrives under several spellings. `normalize_model_name()` reduces all of them to one
key by lower-casing and stripping, in order: a `[1m]`-style window suffix, a provider version (`:0`, `:v2`) or
`@YYYYMMDD` revision, a vendor prefix (`us.anthropic.`, `anthropic/`), a leading path (`models/…`), a `-vN` revision and
a `-YYYYMMDD` release date.

```python
>>> from llm_router_plugins.utils.routing.agentic_routing.claude_code import normalize_model_name as n
>>> n("us.anthropic.claude-sonnet-4-5-20250929-v1:0")
'claude-sonnet-4-5'
>>> n("claude-opus-5-5[1m]")
'claude-opus-5-5'
```

### Step 2 — match it against the configured tiers

`ModelMatcher` tries four layers in strict order, most specific first:

| #   | Layer       | Configured entry         | Request                        | Why it matters                                    |
| --- | ----------- | ------------------------ | ------------------------------ | ------------------------------------------------- |
| 1   | `literal`   | `claude-sonnet-4-5`      | `claude-sonnet-4-5`            | a pin always wins, exactly as written             |
| 2   | `exact`     | `claude-sonnet-4-5`      | `claude-sonnet-4-5[1m]`        | the pin also covers its normalized spellings      |
| 3   | `wildcard`  | `claude-opus-*`          | `claude-opus-7-7`              | a future release of a known line still resolves   |
| 4   | `family`    | `claude-haiku-*`         | `claude-3-5-haiku-20241022`    | legacy names order the family differently         |

Within the wildcard layer the longest literal prefix wins, so `claude-opus-4-*` beats `claude-opus-*` beats
`claude-*`. An exact pin never claims a family: the family layer is reachable only through an explicit `*` declaration.
Matching is case-insensitive; ties resolve by configuration order (first declaration wins).

### Step 3 — rewrite and annotate

The matched tier's `model_name` is written into every configured model key the payload carries, and the decision goes
into `payload["routing"]`:

```python
from llm_router_plugins.utils.routing.agentic_routing.claude_code import ClaudeCodeRoutingPlugin

plugin = ClaudeCodeRoutingPlugin()
plugin.apply({"model": "claude-opus-9-9", "messages": [{"role": "user", "content": "hi"}]})
# {
#   "model": "qwen/Qwen3.8-Flash-Next",
#   "messages": [...],
#   "routing": {
#     "plugin": "agentic_routing_claude_code", "similarity": 1.0,
#     "mode": "opus", "original_model": "claude-opus-9-9",
#     "matched_model": "claude-opus-*", "match_type": "wildcard", "field": "model"
#   }
# }
```

`similarity` is a constant `1.0`: the decision is deterministic, and the value keeps the `routing` block shaped like the
semantic plugins' for anything that reads all of them. `match_type` is the layer from the table above, which is how you
tell a pin from a family guess when auditing a config.

---

## Configuration

### Config file layout

```jsonc
// llm_router_plugins/resources/routing/agentic_routing_claude_code.json
{
  "description": "…",
  "settings": {
    "enabled": true,             // master switch: false = plugin does nothing at all
    "match_families": true,      // false = exact matching only, wildcards are not even indexed
    "model_fields": ["model", "model_name"],   // payload keys read *and* rewritten
    "provider_prefixes": ["us.anthropic.", "anthropic.", "anthropic/"]
  },
  "claude_code_modes": [
    {
      "name": "opus",            // tier id, reported as routing.mode
      "model_name": "qwen/Qwen3.8-Flash-Next",  // target; "" = pass matching requests through
      "description": "…",        // documentation only, never matched against
      "models": ["claude-opus-5-5", "claude-opus-*", "opus"]
    }
  ]
}
```

### Tiers and what each one replaces

| Tier       | Claude Code requests                                              | Claude Code setting it replaces                  |
| ---------- | ----------------------------------------------------------------- | ------------------------------------------------ |
| `fable`    | `claude-fable-5-1`, `claude-fable-5`, the `best` alias            | `ANTHROPIC_DEFAULT_FABLE_MODEL`                  |
| `opus`     | `claude-opus-5-5` … `claude-opus-4-6`, pinned Opus IDs            | `ANTHROPIC_DEFAULT_OPUS_MODEL`                   |
| `plan`     | `opusplan`, if it ever reaches the gateway unresolved             | the Plan Mode phase of `model: "opusplan"`       |
| `sonnet`   | `claude-sonnet-5`, `claude-sonnet-4-6`, `claude-sonnet-4-5`       | `ANTHROPIC_DEFAULT_SONNET_MODEL`                 |
| `haiku`    | `claude-haiku-4-5`, `claude-3-5-haiku-*`, background calls        | `ANTHROPIC_DEFAULT_HAIKU_MODEL` (`ANTHROPIC_SMALL_FAST_MODEL`) |

Claude Code normally resolves `opusplan` and the bare aliases itself, so the `plan` tier and the alias entries
(`opus`, `sonnet`, `haiku`, `best`) are a safety net for gateways and wrappers that forward them verbatim. Subagent
traffic needs no extra tier: whatever `CLAUDE_CODE_SUBAGENT_MODEL` names lands in the tier that owns that name.

The two target models in the shipped file (`qwen/Qwen3.8-Flash-Next`, `qwen/Qwen3.8-27B`) are examples taken from the
Codex plugin's config. Replace them with what your router serves.

### Environment variables

All of them carry the `LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CLAUDE_CODE_` prefix and beat the JSON file.

| Variable               | Effect                                                                              |
| ---------------------- | ----------------------------------------------------------------------------------- |
| `CONFIG`               | the whole config — a raw JSON string *or* a path to a JSON file                     |
| `ENABLED`              | `1/0`, `true/false`, `yes/no`, `on/off` — switch the whole plugin off                |
| `MATCH_FAMILIES`       | turn wildcard and family matching off, leaving exact names only                      |
| `FIELDS`               | payload keys to read and rewrite, e.g. `model\|model_name`                          |
| `MODEL_<TIER>`         | target model of one tier, e.g. `MODEL_OPUS=qwen/Big`                                |
| `MODELS`               | several targets at once: `opus=model-a\|haiku=model-b`                              |
| `MODES`                | whitelist of tier names, e.g. `sonnet\|haiku`                                       |
| `MODE_<tier>_MODELS`   | replaces the names one tier answers, e.g. `MODE_SONNET_MODELS=claude-sonnet-9\|foo` |

List variables accept `|` or `,` as separators. Unknown tier names are logged as warnings and otherwise ignored; unknown
*models* are not an error, they simply never match.

### Precedence and linting

`CONFIG` (or the bundled file) is parsed first, then env overrides are applied in place, then the result is validated.
Validation raises on: no tiers, an empty or duplicated tier name, an empty `model_fields`, an embedded wildcard, a bare
`*`, and one exact model name claimed by two tiers.

`lint_signals()` reports, as warnings only:

- a tier with no `model_name` — matching requests pass through, which is usually a typo;
- a tier with no `models` — it can never be reached;
- the same wildcard declared by two tiers — only the first is ever selected.

---

## Tuning and gotchas

- **A wildcard needs a literal prefix.** `claude-*` is fine, `*` raises: a rule that matches every model in the router
  is a outage waiting to happen, not a default.
- **`match_families: false` also drops the wildcards.** Exact-only mode is exact for real: `claude-opus-*` entries are
  not indexed at all, so every new model ID has to be pinned by name.
- **Normalization strips paths.** `qwen/llama` and `other-org/llama` collapse to the same key, so they cannot both be
  pinned in one config — validation reports the collision instead of picking one at match time.
- **Ordering is a wildcard's only risk.** `claude-*` in an earlier tier beats nothing, but `claude-*` *alone* in the
  first tier swallows every family: put broad wildcards in a late tier, or pin the exceptions before it.
- **Retired and future IDs keep working** as long as the family wildcard is there, which is the reason to prefer
  `claude-sonnet-*` plus pins over pinning every version.
- **The router still owns model availability.** A tier pointing at a model the router does not serve is a downstream
  `model not found`, not a plugin error.

---

## Verifying the plugin

### Offline dry run (no network, no router)

```bash
python - <<'PY'
from llm_router_plugins.utils.routing.agentic_routing.claude_code import ClaudeCodeRoutingPlugin

plugin = ClaudeCodeRoutingPlugin()
for model in ["claude-sonnet-5", "claude-opus-5-5", "claude-haiku-4-5",
              "claude-opus-4-8[1m]", "opusplan", "gpt-5"]:
    print(f"{model:28} -> {plugin.resolve(model)}")
print(plugin.apply({"model": "claude-sonnet-5", "messages": []}))
PY
```

`resolve()` answers exactly what `apply()` would decide, without a payload: the winning tier, the configured entry that
matched and the layer it came from. `gpt-5` printing `None` is the expected result — foreign traffic is untouched.

### Router smoke test

Send an Anthropic-Messages request for a covered model and read the response's usage block, or simply watch the router
log for `Claude Code model swap:`. Then flip `…_ENABLED=false`, restart, and confirm the request goes out with its
original model name.

### Automated tests

```bash
pytest tests/test_agentic_routing_claude_code.py -v
```

160 tests cover the normalizer, all four match layers and their precedence, config loading and validation, every env
override, and the plugin's pass-through paths — including the assertion that a miss returns the *same object* and logs
nothing at all.

---

## Troubleshooting

| Symptom                                          | Cause                                                                     | Fix                                                            |
| ------------------------------------------------ | ------------------------------------------------------------------------- | -------------------------------------------------------------- |
| Nothing logged, model unchanged                  | the model matches no tier                                                  | `plugin.resolve("<model>")`, then add a pin or a family wildcard |
| Router answers `model not found` for `claude-*`  | it resolves the model before the utils pipeline                            | declare the incoming IDs as aliases too                          |
| Startup raises `claimed by modes`                | two tiers pin the same exact name                                          | keep one owner, or make one a wildcard                           |
| Startup raises `no modes defined`                | `MODES` filtered everything out, or the config is empty                     | check `MODES` and `claude_code_modes`                            |
| Every tier answers the same model                | a broad wildcard in the first tier                                         | move it to a late tier and pin the exceptions                    |
| A tier never matches                             | it has no `models`, or `match_families` is off with only wildcards          | read the lint warnings in the startup log                        |
| A tier never swaps, but matches                  | its `model_name` is empty                                                  | set it, or via `MODEL_<TIER>`                                    |

---

## Reference

### Module map

| Module         | Responsibility                                                                |
| -------------- | ----------------------------------------------------------------------------- |
| `mapping.py`   | `normalize_model_name`, `model_family`, `validate_pattern`, `ModelMatcher`, config audits |
| `config.py`    | `ClaudeCodeMode`, `ClaudeCodeRoutingConfig` (loading, env overrides, linting) |
| `plugin.py`    | `ClaudeCodeRoutingPlugin` — resolve, rewrite, annotate                        |

### Defaults at a glance

| Setting                      | Default                                                     |
| ---------------------------- | ----------------------------------------------------------- |
| `enabled`                    | `true`                                                      |
| `match_families`             | `true`                                                      |
| `model_fields`               | `("model", "model_name")`                                    |
| `provider_prefixes`          | `us.` / `eu.` / `apac.` / `global.anthropic.`, `anthropic.`, `anthropic/` |
| `routing.similarity`         | `1.0` (constant — the decision is deterministic)             |
| tiers                        | `fable`, `opus`, `plan`, `sonnet`, `haiku`                    |

### See also

- [Codex CLI Routing](../codex/README.md) — work-mode routing for the Codex CLI's `/v1/responses` traffic
- [Bi-Encoder Routing](../../semantic_biencoder/README.md) — embedding similarity routing for `auto`
- [Main README](../../../../../README.md) — plugin architecture, pipelines and registration
