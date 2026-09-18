# Semantic Routing Plugins

Three routing plugins are available for **model selection** in the **LLM‑Router** system.

- `simple_semantic_routing` and `semantic_biencoder_routing` activate when `payload["model"] == "auto"`.
- `agentic_routing` activates when `payload["model"]` equals its own trigger value (`"agentic"` by default) and routes
  on the agent's **work mode**.

---

## Table of Contents

- [0. Shared Routing Layer](#0-shared-routing-layer)
- [1. Simple Semantic Routing (Heuristic)](#1-simple-semantic-routing-heuristic)
- [2. Bi-Encoder Semantic Routing (Embedding-based)](#2-bi-encoder-semantic-routing-embedding-based)
- [3. Agentic Routing (Agent Work Mode)](#3-agentic-routing-agent-work-mode)
- [4. Comparison: Which Plugin to Use?](#4-comparison-which-plugin-to-use)
- [5. File Locations](#5-file-locations)

---

## 0. Shared Routing Layer

`semantic_biencoder_routing` and `agentic_routing` share a common layer in
`llm_router_plugins/utils/routing/`:

| Module | Contents |
|--------|----------|
| `embedder.py` | `EmbeddingRouter` (BiEncoder + FAISS: chunking, index build/load/persist, `route()`) and `EmbeddingRouterConfig` — the formal config contract the router duck-types |
| `target.py` | `RoutingTarget` — the shared target/mode dataclass (`name`, `model_name`, `description`, `examples`); `AgentMode` is a subclass of it |
| `common.py` | `RoutingConfigBase` (shared `from_file`/`from_json` protocol: `..._CONFIG` env var holding raw JSON or a file path, optional default config location), `env_int`/`env_float`/`env_bool`/`resolve_persist_dir` env helpers, `build_embedding_router` + `check_router_has_vectors`, `should_route` (the `payload["model"]` trigger gate) and `annotate_routing` (writes `payload["model"]` + `payload["routing"]`) |

Backward compatibility: `semantic_biencoder/embedder.py` and
`semantic_biencoder/config.py` re-export `EmbeddingRouter` /
`EmbeddingRouterConfig` / `RoutingTarget` from the shared modules, so existing
imports keep working.

All three plugins also share the text-extraction helper
`llm_router_plugins/utils/text_extractor.py::extract_user_text`
(`messages[-1].content` → `user_last_statement` → `query` → `prompt` → `input`).

## 1. Simple Semantic Routing (Heuristic)

The **Simple Semantic Routing plugin** (`simple_semantic_routing`) performs
two-stage heuristic model selection: it classifies the user's intent
(code, math, creative, general) via weighted keywords, multi-word phrases, and
regex patterns, then estimates input complexity (token count) to pick the most
appropriate model from a configured pool.

**No embedding model is required** — routing is a fast, pure-text classification.

### 1.1 Architecture

#### Data Flow

```
User Input
    │
    ▼
classify_intent(text)  ← keywords + phrases + regex patterns
    │
    ▼
estimate_tokens(text)  ← word_count × 1.25
    │
    ▼
complexity_level(tokens)  ← simple / medium / complex
    │
    ▼
select_model(intent, complexity)  ← weighted score → model index
    │
    ▼
Updated payload["model"]
```

#### Intent Classification Algorithm

Each intent is defined in `simple_semantic.json` with **four complementary signal types**:

##### a) Keywords — single words

Each keyword has an optional weight. If no weight is specified, the default is **1.0**.

```json
"keywords": ["code", "debug", "funkcja"],
"weights": {"debug": 5, "kod": 4, "code": 2}
```

Matching `"debug"` adds **5** to the score; matching `"code"` adds **2**.

##### b) Phrases — multi-word expressions

Phrases use the format `"text:weight"`. The default weight is **2.0** when omitted.

```json
"phrases": ["write code:5", "fix bug:4", "napisz funkcję:4"]
```

Matching `"write code"` adds **5** to the score.

##### c) Regex Patterns — structural detection

Patterns add a flat **+3.0** per match, useful for detecting code structures, math formulas, and question patterns.

```json
"patterns": [
"def\\s+\\w+\\s*\\(",
"class\\s+\\w+\\s*[(:]",
"\\b\\d+\\s*[+\\-*/^]\\s*\\d+\\b",
"\\bwhat\\s+(is|are|does)\\b"
]
```

##### d) Weights — keyword importance

The `weights` object maps individual keywords to boost factors, allowing fine-grained control:

```json
"weights": {
"debug": 5, // Very strong signal for code intent
"błąd": 5, // Very strong signal for code intent
"kod": 2, // Moderate signal
"python": 2   // Moderate signal
}
```

### 1.2 Intent Categories

| Intent         | Description                                  | Examples                                                        |
|----------------|----------------------------------------------|-----------------------------------------------------------------|
| **`code`**     | Programming, debugging, implementation       | "napisz funkcję", "fix bug", "debug code", "git commit"         |
| **`math`**     | Mathematics, calculations, statistics        | "oblicz prawdopodobieństwo", "solve equation", "calculate mean" |
| **`creative`** | Creative writing, editing, brainstorming     | "napisz opowiadanie", "write a story", "brainstorm"             |
| **`general`**  | General questions, explanations, comparisons | "wyjaśnij jak", "what is the difference", "compare"             |
| **`none`**     | No specific intent (greetings, etc.)         | "cześć", "hello", "dziękuję"                                    |

### 1.3 Model Selection Algorithm

```
1. complexity_map = {"simple": 0, "medium": n//2, "complex": n-1}
2. idx = complexity_map[complexity]

3. if intent has a target in intent_adjustment:
       intent_idx = model_index_for(intent.target)
       idx = max(idx, intent_idx)      // boost toward intent's target
   else:
       idx -= 1                          // demote (no intent)

4. idx = clamp(idx, 0, n-1)
5. return models[idx]
```

**Boost / Demote Examples:**

| Intent                | Complexity      | Base Index | Intent Adjustment      | Result            |
|-----------------------|-----------------|------------|------------------------|-------------------|
| `code` → `medium`     | simple (idx=0)  | 0          | intent targets index 1 | **max(0, 1) = 1** |
| `none`                | complex (idx=2) | 2          | no target → demote     | **2-1 = 1**       |
| `math` → `medium`     | complex (idx=3) | 3          | intent targets index 1 | **max(3, 1) = 3** |
| `creative` → `simple` | medium (idx=1)  | 1          | intent targets index 0 | **max(1, 0) = 1** |

### 1.4 Configuration

#### JSON Config (`simple_semantic.json`)

All configuration lives in [
`llm_router_plugins/resources/routing/simple_semantic.json`](../resources/routing/simple_semantic.json).

```json
{
  "settings": {
    "len_thresholds_max": {
      "simple": 25,
      "medium": 150
    },
    "default_models": {
      "simple": "gpt-oss:120b",
      "medium": "qwen3.6:35b"
    },
    "intent_adjustment": {
      "code": "medium",
      "math": "medium",
      "creative": "simple",
      "general": "simple",
      "none": ""
    }
  },
  "intents": {
    "code": {
      "keywords": [
        "code",
        "debug",
        "funkcja"
      ],
      "phrases": [
        "write code:5",
        "fix bug:4"
      ],
      "patterns": [
        "def\\s+\\w+\\s*\\(",
        "class\\s+\\w+"
      ],
      "weights": {
        "debug": 5,
        "błąd": 4
      }
    },
    "math": {
      "keywords": [
        "calculate",
        "equation"
      ],
      "phrases": [
        "calculate:4",
        "solve equation:4"
      ],
      "patterns": [
        "\\b\\d+\\s*[+\\-*/^]\\s*\\d+\\b"
      ],
      "weights": {
        "calculate": 4,
        "solve": 4
      }
    },
    "creative": {
      "keywords": [
        "write",
        "story",
        "napisz"
      ],
      "phrases": [
        "write a story:4",
        "napisz wiersz:4"
      ],
      "patterns": [
        "napisz\\s+(mi|ci|go|ją)"
      ],
      "weights": {
        "napisz": 4,
        "write": 3
      }
    },
    "general": {
      "keywords": [
        "explain",
        "difference",
        "wyjaśnij"
      ],
      "phrases": [
        "what is:4",
        "how to:4"
      ],
      "patterns": [
        "\\bwhat\\s+(is|are|does)\\b"
      ],
      "weights": {
        "wyjaśnij": 3,
        "help": 2
      }
    }
  },
  "none": {
    "keywords": [
      "hello",
      "cześć",
      "thanks"
    ],
    "phrases": [
      "hello:1",
      "thanks:1"
    ],
    "patterns": [
      "^\\b(hello|hi|cześć)\\b"
    ],
    "weights": {}
  }
}
```

#### Configurable Values

| Setting                      | Description                           | Default        |
|------------------------------|---------------------------------------|----------------|
| `len_thresholds_max.simple`  | Max tokens for "simple" complexity    | `25`           |
| `len_thresholds_max.medium`  | Max tokens for "medium" complexity    | `150`          |
| `default_models.simple`      | Model for simple complexity           | `gpt-oss:120b` |
| `default_models.medium`      | Model for medium complexity           | `qwen3.6:35b`  |
| `intent_adjustment.code`     | Target complexity for code intent     | `medium`       |
| `intent_adjustment.math`     | Target complexity for math intent     | `medium`       |
| `intent_adjustment.creative` | Target complexity for creative intent | `simple`       |
| `intent_adjustment.general`  | Target complexity for general intent  | `simple`       |

### 1.5 Environment Variable Overrides

All environment variables are prefixed with `LLM_ROUTER_ROUTING_`.

| Variable                                   | Format                               | Description                                     |
|--------------------------------------------|--------------------------------------|-------------------------------------------------|
| `LLM_ROUTER_ROUTING_MODELS`                | `model-a\|model-b\|model-c`          | Comma-separated model pool (pipe-delimited)     |
| `LLM_ROUTER_ROUTING_COMPLEXITY_THRESHOLDS` | `simple_threshold\|medium_threshold` | Token count thresholds (e.g. `10\|50`)          |
| `LLM_ROUTER_ROUTING_DEFAULT_MODEL`         | `model-name`                         | Fallback model when no text content is found    |
| `LLM_ROUTER_ROUTING_INTENT_<CATEGORY>`     | `kw1\|kw2:5\|phrase:3`               | Override intent keywords/phrases for a category |

**Example:**

```bash
export LLM_ROUTER_ROUTING_MODELS="tiny-model\|medium-model\|large-model"
export LLM_ROUTER_ROUTING_COMPLEXITY_THRESHOLDS="10\|50"
export LLM_ROUTER_ROUTING_INTENT_CODE="code\|debug\|implement"
export LLM_ROUTER_ROUTING_DEFAULT_MODEL="fallback-model"
```

### 1.6 Usage Examples

#### Basic Usage

```python
from llm_router_plugins.utils.routing.simple_semantic.simple_semantic_routing import SimpleSemanticRoutingPlugin

plugin = SimpleSemanticRoutingPlugin()

payload = {
    "model": "auto",
    "messages": [{"role": "user", "content": "Napisz funkcję w Pythonie"}]
}

result = plugin.apply(payload)
# result["model"] → selected model (e.g., "qwen3.6:35b")
```

#### Payload Text Sources

The plugin extracts text from the payload in this priority order:

1. `payload["messages"][-1]["content"]` — last message content
2. `payload["user_last_statement"]`
3. `payload["query"]`
4. `payload["prompt"]`
5. `payload["input"]`

#### Logging

When a logger is provided, the plugin logs routing decisions:

```python
import logging

logger = logging.getLogger("llm_router")
plugin = SimpleSemanticRoutingPlugin(logger=logger)

# On apply:
# INFO: Semantic routing: intent=code, complexity=medium (12 tokens) -> qwen3.6:35b
```

### 1.7 Weight Tuning Guidelines

| Weight Range | Meaning         | Use Case                                                              |
|--------------|-----------------|-----------------------------------------------------------------------|
| **1**        | Weak / baseline | Generic terms that appear in many contexts                            |
| **2–3**      | Moderate        | Common but not definitive signals                                     |
| **4–5**      | Strong          | Highly specific, unambiguous signals (e.g., "debug", "błąd" for code) |
| **1 (none)** | Neutral         | Greetings and filler phrases that shouldn't influence routing         |

### 1.8 Phrase Format Guidelines

| Format          | Weight        | Example          |
|-----------------|---------------|------------------|
| `"text:weight"` | Explicit      | `"write code:5"` |
| `"text"`        | Default (2.0) | `"fix bug"`      |

### 1.9 Complexity Levels

| Level     | Token Range                                     | Model Selection                     |
|-----------|-------------------------------------------------|-------------------------------------|
| `simple`  | 0 – `thresholds[simple]`                        | Cheapest / simplest model (index 0) |
| `medium`  | `thresholds[simple]` + 1 – `thresholds[medium]` | Middle model (index n // 2)         |
| `complex` | > `thresholds[medium]`                          | Strongest model (index n - 1)       |

Token estimation: `round(word_count × 1.25)`

### 1.10 Built-in Patterns Reference

#### Code Patterns

| Pattern                                     | Matches                     |
|---------------------------------------------|-----------------------------|
| `def\s+\w+\s*\(`                            | Python function definitions |
| `class\s+\w+\s*[(:]`                        | Python/class definitions    |
| `import\s+\w+`                              | Import statements           |
| `try\s*:`, `except\s*:`                     | Try/except blocks           |
| `git\s+(commit\|push\|pull\|branch\|merge)` | Git commands                |
| `npm\s+(install\|run\|start\|build)`        | NPM commands                |
| `docker\s+(build\|run\|compose)`            | Docker commands             |
| `SELECT\s+.*\s+FROM`                        | SQL SELECT queries          |

#### Math Patterns

| Pattern                                  | Matches                  |
|------------------------------------------|--------------------------|
| `\b\d+\s*[+\-*/^]\s*\d+\b`               | Arithmetic expressions   |
| `\bsin\s*\(`, `\bcos\s*\(`, `\btan\s*\(` | Trig functions           |
| `\bsqrt\s*\(`, `\blog\s*\(`              | Math functions           |
| `\bf\(x\)\s*=`                           | Function notation        |
| `\b∑\b`, `\b∫\b`                         | Sigma / integral symbols |

#### General Question Patterns

| Pattern                      | Matches                     |
|------------------------------|-----------------------------|
| `\bwhat\s+(is\|are\|does)\b` | What questions              |
| `\bhow\s+(to\|does\|do)\b`   | How questions               |
| `\bwhy\s+(is\|does)\b`       | Why questions               |
| `\bporównaj\s+`              | Polish comparison questions |

### 1.11 Adding New Intents

To add a new intent category:

1. **Add the intent to `simple_semantic.json`:**

```json
{
  "intents": {
    "my_intent": {
      "keywords": [
        "keyword1",
        "keyword2"
      ],
      "phrases": [
        "phrase one:4",
        "phrase two:3"
      ],
      "patterns": [
        "pattern\\s+here"
      ],
      "weights": {
        "keyword1": 5,
        "keyword2": 2
      }
    }
  },
  "settings": {
    "intent_adjustment": {
      "my_intent": "medium"
    }
  }
}
```

2. **Or override via environment variable:**

```bash
export LLM_ROUTER_ROUTING_INTENT_MY_INTENT="keyword1\|keyword2:5\|phrase:4"
```

### 1.12 Running Tests

```bash
pytest tests/test_simple_semantic_routing.py -v
```

---

## 2. Bi-Encoder Semantic Routing (Embedding-based)

The **Bi-Encoder routing plugin** (`semantic_biencoder_routing`) uses a neural embedding model
(**google/embeddinggemma-300m**) to compute semantic embeddings for a set of pre-configured routing targets.
Each target has a `name`, a `model_name` (the model to route to), a `description`, and a list of `examples`.
At query time the user message is embedded and matched against all stored target embeddings using FAISS
(`IndexFlatIP` on L2-normalised vectors = cosine similarity). The best-matching target determines the selected model.

### 2.1 Index Building (on first load or when the persist directory is missing)

- For each target, its `description` and `examples` are combined into text.
- The text is split into overlapping **token chunks** using a sliding window (`chunk_size` tokens, `chunk_overlap`
  tokens overlap).
- Each chunk is embedded via the BiEncoder model (e.g. `google/embeddinggemma-300m`).
- All embedding vectors are **L2-normalised** to unit length.
- Vectors are inserted into a `faiss.IndexFlatIP` index (inner product).
- A docstore maps each FAISS doc ID to its target name (for reverse lookup).

### 2.2 Routing (query)

- The user message is embedded and L2-normalised.
- FAISS performs a nearest-neighbor search returning the `top_k` closest chunks.
- Scores are **aggregated per target**: the mean cosine similarity of all chunks belonging to the same target is
  computed.
- The target with the **highest mean similarity** wins and its `model_name` is returned.

### 2.3 Persistence

The FAISS index and docstore are saved to disk (files `index.faiss` and `docstore.pkl`) under the configured persist
directory. On subsequent starts the index is loaded from disk — embeddings are **not recomputed**.
If the embedding model changes (different output dimension) the index is automatically rebuilt.

### 2.4 Configuration

Configuration is loaded from [
`llm_router_plugins/resources/routing/semantic_biencoder.json`](../resources/routing/semantic_biencoder.json).

Example JSON configuration:

```json
{
  "embedding_model": "google/embeddinggemma-300m",
  "settings": {
    "chunk_size": 256,
    "chunk_overlap": 64,
    "similarity_threshold": 0.0,
    "top_k": 1,
    "vector_store_path": ""
  },
  "routing_targets": [
    {
      "name": "code-generation",
      "model_name": "qwen3.6:35b",
      "description": "Model specialized for code-related tasks.",
      "examples": [
        "Write a Python function...",
        "..."
      ]
    },
    {
      "name": "math-reasoning",
      "model_name": "gpt-oss:120b",
      "description": "Model for mathematical reasoning and calculations.",
      "examples": [
        "Calculate the derivative of...",
        "..."
      ]
    }
  ]
}
```

#### Environment Variables

| Variable                                              | Purpose                               |
|-------------------------------------------------------|---------------------------------------|
| `LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_MODEL`         | Override the embedding model name     |
| `LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_TARGETS`       | Pipe-separated list of target names   |
| `LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_CHUNK_SIZE`    | Override chunk size                   |
| `LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_CHUNK_OVERLAP` | Override chunk overlap                |
| `LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_PERSIST_DIR`   | Directory for FAISS index persistence |

### 2.5 Usage Examples

#### Basic Usage

```python
from llm_router_plugins.utils.routing.semantic_biencoder.semantic_biencoder_routing import
    SemanticBiEncoderRoutingPlugin

plugin = SemanticBiEncoderRoutingPlugin()

payload = {
    "model": "auto",
    "messages": [{"role": "user", "content": "Napisz funkcję w Pythonie"}]
}

result = plugin.apply(payload)
# result["model"] → selected model (e.g., "qwen3.6:35b")
```

#### Target Examples with Embeddings

Each routing target has `description` + `examples`. The plugin embeds both and computes the mean vector as the
target's embedding. When a user query arrives:

1. Query text is embedded → **vector Q** (L2-normalised).
2. FAISS searches for closest vectors to **Q** among all target embeddings.
3. Best match → that target's `model_name` is selected.

**Example targets:**

| Target Name       | Model        | Description                            | Examples                                                      |
|-------------------|--------------|----------------------------------------|---------------------------------------------------------------|
| code-generation   | qwen3.6:35b  | Write and debug code                   | "Write a Python function to sort a list", "Fix my CSS layout" |
| math-reasoning    | gpt-oss:120b | Mathematical problems and calculations | "Calculate the integral of x²", "Solve for x in 2x+3=7"       |
| creative-writing  | gpt-oss:120b | Creative writing and storytelling      | "Write a short story about...", "Create a poem about..."      |
| general-knowledge | gpt-oss:120b | General questions and explanations     | "What is the capital of France?", "How does DNA work?"        |

### 2.6 Scoring Details

For each user query, FAISS returns `top_k` closest chunks with similarity scores. These scores are **aggregated per
target** (mean of all matching chunk scores). The final score table looks like:

| Target            | Mean Cosine Similarity | Selected? |
|-------------------|------------------------|-----------|
| code-generation   | 0.85                   | ✅ Yes     |
| math-reasoning    | 0.42                   |           |
| creative-writing  | 0.31                   |           |
| general-knowledge | 0.28                   |           |

The target with the **highest mean similarity** wins. If its score falls below
`similarity_threshold` (default 0.0), the payload is returned unchanged —
`payload["model"]` is left as it was.

### 2.7 Running Tests

```bash
pytest tests/test_semantic_biencoder_routing.py -v
```

---

## 3. Agentic Routing (Agent Work Mode)

The **Agentic Routing plugin** (`agentic_routing`) routes on the **work mode of the agent** — is it planning,
coding, reviewing, testing, debugging, researching or summarizing — instead of classifying a free-form intent.
Each mode owns its own model, so a coding agent gets a strong code model while a summarizing step stays on a
cheaper general model.

The plugin activates only when `payload["model"]` is a string whose trimmed value is in the trigger list
(`"agentic"` by default, case-sensitive). It rewrites `payload["model"]` and adds `payload["agent_mode"]` plus
`payload["routing"]` metadata; temperature, `max_tokens`, prompts and tools are never modified.

Selection is **deterministic first, semantic last**: whatever can be decided from declared facts (an explicit
mode, a rule, session memory, a keyword) is decided without ML, and embeddings are consulted only when the
deterministic layers stay silent. With `SEMANTIC_ENABLED=false` no ML library is imported at all, and the plugin
still routes every request that carries a signal.

### 3.1 Detection Cascade

Mode resolution is strictly ordered — the first layer that answers wins:

| # | Layer           | `source`                  | Behaviour                                                                        | ML    |
|---|-----------------|---------------------------|----------------------------------------------------------------------------------|-------|
| 1 | **Explicit**    | `explicit`                | First present key wins: `agent_mode` → `mode` → `agent.mode` → `metadata.agent_mode` | no    |
| 2 | **Rules**       | `rules`                   | Declarative AND-conditions over the request signals (§3.3)                        | no    |
| 3 | **Affinity**    | `affinity`                | The mode already chosen for this `session_id` (§3.5)                              | no    |
| 4 | **Heuristic**   | `heuristic`               | Keywords (weight 1), `text:weight` phrases (default 2.0), regex patterns (+3.0)   | no    |
| 5 | **Semantic**    | `semantic`                | Bi-encoder + FAISS over mode descriptions/examples, accepted at `similarity >= threshold` | yes   |
| 6 | **Fallback**    | `fallback` / `empty_text` | The configured `fallback_mode`                                                    | no    |

Details worth knowing:

- Layers 1–4 are pure Python: no embedding model, no network, no FAISS. A payload that declares a `task`, an
  `agent`, `tools` or a known `session_id` always gets a reproducible decision.
- **Rules intentionally beat affinity**: a rule is the operator's policy for *this* request, affinity is only the
  memory of the conversation. Set `rules_enabled=false` to make sessions stick unconditionally.
- Heuristics and semantics form a single inference step — heuristics run first, semantics only pick up text that
  no keyword, phrase or pattern recognized.
- An explicit mode name is normalised — trimmed, lower-cased, `-` and spaces converted to `_` — so `"Plan"` resolves
  to `plan`, and a custom `deep_research` mode also matches `"Deep Research"` and `"deep-research"`. An unknown name
  logs a warning and the cascade continues; it is never a hard error.
- The heuristic layer scores every mode and the highest score above 0 wins, using exactly the same keyword /
  phrase / pattern semantics as the Simple plugin (§1.1). Ties are won by the mode defined first.
- An explicit mode is honoured even when the payload carries no text. Whitespace-only text is stripped, so a
  request without usable text resolves to the fallback mode with `source = "empty_text"` — blank text never
  reaches FAISS.

The reported `similarity` is `1.0` for `explicit`, `rules` and `affinity`, `score / (score + 1)` for a heuristic
hit, the FAISS score for a semantic hit, and `0.0` for `fallback` / `empty_text`.

### 3.2 Request Signals (Agent-Aware Payload)

`signals.py` is the only place in the plugin that knows *where* agent metadata lives inside a payload. Every other
layer works on the normalized, immutable `RequestSignals` snapshot, which keeps routing independent of the wire
format of individual clients (OpenAI chat, Codex, Ollama, custom agents).

Locations are consulted in priority order, first non-empty declaration wins: **top level** → `metadata` → `agent`.

```json
{
  "model": "agentic",
  "agent": "codex",
  "session_id": "abc123",
  "task": "coding",
  "tools": true,
  "reasoning": true,
  "context_tokens": 42000
}
```

normalizes to `RequestSignals(agent="codex", session_id="abc123", task="coding", tools=True, tool_count=1,
reasoning=True, context_tokens=42000)`.

| Signal                      | Accepted payload keys                                                                                 |
|-----------------------------|-------------------------------------------------------------------------------------------------------|
| `agent`                     | `agent` (string), `agent.name`, `agent_name`, `metadata.agent`                                         |
| `session_id`                | `session_id`, `conversation_id`, `thread_id`                                                           |
| `task`                      | `task`                                                                                                 |
| `tools` / `tool_count`      | `tools` as a boolean **or** a list/tuple/dict of tool definitions (`tool_count` is derived)             |
| `reasoning`                 | `reasoning`, `thinking`, or a non-empty `reasoning_effort`                                              |
| `vision`                    | `vision` flag, or an image part (`image_url`, `image`, `input_image`) in the message content            |
| `structured_output`         | `structured_output` flag, or a `response_format` other than `text`                                     |
| `parallel_tools`            | `parallel_tools`, `parallel_tool_calls`                                                                |
| `context_tokens`            | `context_tokens`, otherwise an estimate (`characters // 4` of the text plus conversation history)       |
| `metadata`                  | the raw `metadata` mapping, so rules can test it without touching the payload                          |

Agent, task and session values are normalized like mode names (trimmed, lower-cased, `-`/space → `_`). Reading a
signal **never raises**: a value that cannot be interpreted keeps its field default, so an unusual payload degrades
into the inference layers instead of failing the request.

### 3.3 Declarative Rules

`rules.py` applies the operator's policy before any guesswork. A rule is a pure condition over `RequestSignals`
plus the user text — it either matches or it does not — which makes the decision reproducible and auditable:
`routing.rule_id` names the rule that fired.

```json
{
  "rules": [
    {
      "id": "task-coding",
      "priority": 100,
      "when": { "task": ["coding", "implement", "refactor"] },
      "then": { "mode": "code" }
    },
    {
      "id": "needs-tools",
      "priority": 80,
      "when": { "requires_tools": true },
      "then": { "mode": "code" }
    }
  ]
}
```

`id` defaults to `rule-<index>`, `priority` to `0` (higher wins; equal priorities keep their JSON order), and
`mode` may be written directly instead of `then.mode`. Every condition present in `when` must hold — logical
**AND**:

| `when` key                                                                                                       | Meaning                                                      |
|------------------------------------------------------------------------------------------------------------------|---|
| `agent`, `task`                                                                                                  | String or list of strings (a list is an OR)                   |
| `requires_tools` / `requires_reasoning` / `requires_vision` / `requires_structured_output` / `requires_parallel_tools` | `true` requires the signal to be set, `false` requires it unset |
| `min_context_tokens` / `max_context_tokens`                                                                      | Inclusive bounds on `context_tokens`                          |
| `text_contains_any`                                                                                              | List of substrings, case-insensitive (OR)                     |
| `text_matches`                                                                                                   | Regex (`re.search`, case-insensitive); an invalid expression never matches |
| `metadata`                                                                                                       | Key/value pairs that must all be present and equal            |

An empty or missing `when` is a catch-all (a warning is logged). A rule that targets a non-existent mode logs a
warning and is ignored, while an **unknown `when` key raises `ValueError` at startup** naming the exact JSON path:
a typo must fail the plugin build, not silently send everything to the fallback.

### 3.4 Capabilities and Escalation

A model is not only a string. Each mode may declare what its model can actually do, and a request declares what it
needs through its signals. `capabilities.py` is a **gate, not a scorer**: it never picks the "best" mode, it only
rejects modes that cannot serve the request.

```json
"capabilities": {
  "tool_calling": true,
  "parallel_tool_calls": true,
  "reasoning": true,
  "vision": false,
  "structured_output": true,
  "context_window": 131072
}
```

- A capability **absent** on a mode means "unknown" and counts as satisfied — modes without capability metadata
  keep the legacy behaviour and are never penalized. Only *active* requirements constrain the choice.
- `context_window` compares numbers; a non-numeric declared value is unknown, therefore satisfied.
- When the selected mode cannot serve the request, `escalate()` replaces it with the capable candidate that has
  the **largest declared `context_window`** (missing counts as `0`, ties keep configuration order). The
  `fallback_mode` is skipped unless it is the only capable mode left.
- If no mode can serve the request, the original mode is kept and a warning is logged — routing never fails.
- An **explicit** mode is never overridden; the gate only warns
  `"... does not satisfy <capabilities> — keeping the explicit choice"`, because an explicit declaration is the
  intent of the caller.
- Escalation keeps the layer and the confidence of the original decision and adds `"escalated": true` plus
  `"escalated_from": "<mode>"` to `routing`. A cached session entry that no longer satisfies the requirements is
  invalidated and the request is re-routed.

Candidates are ranked only by deterministic comparisons, so escalation is reproducible: the same input always
lands on the same model.

### 3.5 Session Affinity

`session_affinity.py` keeps one conversation on one model: the decision of the first turn is remembered under
`session_id` and reused (`source = "affinity"`), so a follow-up such as "and now add error handling" does not jump
from a code model to a chat model.

- Storage is an in-process, thread-safe LRU cache. `ttl_seconds` (default **900**) is refreshed on every hit,
  `max_entries` (default **1024**) evicts the least recently used entry.
- `fallback` and `empty_text` decisions are **never cached** — a turn that carried no information must not pin a
  bad mode to the whole conversation.
- A cached mode that disappeared from the configuration, or that can no longer satisfy the request, is dropped and
  the cascade continues.
- When a `session_id` is present and affinity is enabled, the annotation gains
  `"session": {"session_id": …, "mode": …, "model": …, "reused": <bool>}`; `reused` is `true` only on a cache hit.
- `AgenticRoutingPlugin.reset_sessions()` clears the cache — useful in tests and after a config reload.

The cache is per process. Behind several router workers, either pin sessions at the gateway level or set
`SESSION_AFFINITY_ENABLED=false` to avoid divergent per-worker memory.

### 3.6 Configuration

Default configuration lives in [
`llm_router_plugins/resources/routing/agentic_routing.json`](../../../llm_router_plugins/resources/routing/agentic_routing.json).

Example JSON configuration (abridged to one rule and one mode):

```json
{
  "settings": {
    "trigger": ["agentic"],
    "fallback_mode": "fallback",
    "vector_store_path": "",
    "rules_enabled": true,
    "capabilities_enabled": true,
    "escalation_enabled": true,
    "session_affinity": { "enabled": true, "ttl_seconds": 900, "max_entries": 1024 },
    "semantic": {
      "enabled": true,
      "threshold": 0.55,
      "top_k": 3,
      "chunk_size": 256,
      "chunk_overlap": 64
    }
  },
  "rules": [
    { "id": "task-coding", "priority": 100, "when": { "task": ["coding"] }, "then": { "mode": "code" } }
  ],
  "agent_modes": [
    {
      "name": "code",
      "model_name": "gpt-oss:120b",
      "description": "Agent works in coding mode: writing, implementing and refactoring source code.",
      "examples": ["Implement a Python function that parses this CSV file", "..."],
      "keywords": ["implement", "refactor", "function", "..."],
      "phrases": ["write a function:5", "refactor this:5", "..."],
      "patterns": ["\\bcode\\b", "\\bimplement\\w*\\b"],
      "weights": {"implement": 3, "refactor": 3},
      "capabilities": {
        "tool_calling": true,
        "parallel_tool_calls": true,
        "reasoning": true,
        "vision": false,
        "structured_output": true,
        "context_window": 131072
      }
    }
  ]
}
```

The eight shipped modes are `plan`, `code`, `review`, `test`, `debug`, `research`, `summarize` and `fallback`
(`plan`/`research`/`summarize`/`fallback` → `qwen3.6:35b`, `code`/`debug` → `gpt-oss:120b`,
`review`/`test` → `granite3.3:8b`); several modes may legitimately share the same `model_name`. The nine shipped
rules are the seven `task-*` alias rules (priority 100), `needs-tools` (priority 80 → `code`) and
`needs-reasoning` (priority 70 → `plan`).

### 3.7 Defining Custom Modes and Rules

Point the plugin at your own file (or inline JSON) and describe the modes your agents actually use:

```bash
export LLM_ROUTER_ROUTING_AGENTIC_CONFIG=/etc/llm-router/my_agent_modes.json
# or inline, when the value starts with "{" or "["
export LLM_ROUTER_ROUTING_AGENTIC_CONFIG='{"settings": {...}, "rules": [...], "agent_modes": [...]}'
```

A mode only requires `name`, `model_name` and `description`; `examples`, `keywords`, `phrases`, `patterns`,
`weights` and `capabilities` are optional. Rules enforced at startup (a `ValueError` naming both the JSON key and
the env var):

- `agent_modes` must be non-empty, mode names unique, and every `model_name` non-empty.
- `fallback_mode` must exist among the modes — if you whitelist modes with `MODES`, keep the fallback in the list
  or override `FALLBACK_MODE` too.
- `embedding_model` must be set when the semantic layer is enabled; `top_k >= 1`, `chunk_size > 0`,
  `chunk_overlap >= 0`.
- every `when` key of every rule must be supported (§3.3).

`MODES` whitelists *mode names* and keeps the rest of the config coherent: rules pointing at a removed mode are
dropped with a warning, exactly like a rule naming an unknown mode.

### 3.8 Environment Variable Overrides

| Env variable                                          | Purpose                                                  |
|-------------------------------------------------------|----------------------------------------------------------|
| `LLM_ROUTER_ROUTING_AGENTIC_CONFIG`                   | Path to a custom JSON config, or a raw JSON string        |
| `LLM_ROUTER_ROUTING_AGENTIC_TRIGGER`                  | Pipe-separated trigger values (default `agentic`)         |
| `LLM_ROUTER_ROUTING_AGENTIC_MODEL`                    | Override the **embedding** model name                     |
| `LLM_ROUTER_ROUTING_AGENTIC_MODELS`                   | Per-mode models, e.g. `plan=model_a\|code=model_b`        |
| `LLM_ROUTER_ROUTING_AGENTIC_MODES`                    | Whitelist of mode names to keep                           |
| `LLM_ROUTER_ROUTING_AGENTIC_RULES_ENABLED`            | `true`/`false` — toggle the declarative rules layer       |
| `LLM_ROUTER_ROUTING_AGENTIC_CAPABILITIES_ENABLED`     | `true`/`false` — derive requirements from the signals     |
| `LLM_ROUTER_ROUTING_AGENTIC_ESCALATION_ENABLED`       | `true`/`false` — move uncappable modes to a capable one   |
| `LLM_ROUTER_ROUTING_AGENTIC_SESSION_AFFINITY_ENABLED` | `true`/`false` — toggle per-session model stickiness      |
| `LLM_ROUTER_ROUTING_AGENTIC_SESSION_TTL_SECONDS`      | Affinity entry lifetime, refreshed on hit (default `900`) |
| `LLM_ROUTER_ROUTING_AGENTIC_SESSION_MAX_ENTRIES`      | Affinity cache size (default `1024`)                      |
| `LLM_ROUTER_ROUTING_AGENTIC_SEMANTIC_ENABLED`         | `true`/`false` — toggle the embedding layer               |
| `LLM_ROUTER_ROUTING_AGENTIC_SIMILARITY_THRESHOLD`     | Minimum cosine similarity for a semantic hit              |
| `LLM_ROUTER_ROUTING_AGENTIC_TOP_K`                    | Chunks retrieved per query                                |
| `LLM_ROUTER_ROUTING_AGENTIC_CHUNK_SIZE`               | Token chunk size used when indexing modes                 |
| `LLM_ROUTER_ROUTING_AGENTIC_CHUNK_OVERLAP`            | Token overlap between chunks                              |
| `LLM_ROUTER_ROUTING_AGENTIC_PERSIST_DIR`              | Directory for FAISS index persistence                     |
| `LLM_ROUTER_ROUTING_AGENTIC_FALLBACK_MODE`            | Mode used when nothing matches                            |
| `LLM_ROUTER_ROUTING_AGENTIC_MODE_<name>_KEYWORDS`     | Pipe-separated keyword override for a single mode         |

Unknown values never crash the router: an unrecognized boolean or a malformed threshold is logged and ignored,
overrides naming an unknown mode are skipped with a warning, and a `SESSION_TTL_SECONDS` or
`SESSION_MAX_ENTRIES` below `1` is ignored with a warning.

### 3.9 Usage Example

```python
from llm_router_plugins.utils.routing.agentic_routing import AgenticRoutingPlugin

plugin = AgenticRoutingPlugin()

result = plugin.apply({
    "model": "agentic",
    "prompt": "Please refactor this function to add caching",
})
# result["model"]      -> "gpt-oss:120b"
# result["agent_mode"] -> "code"
# result["routing"]    -> {"plugin": "agentic_routing", "similarity": 0.9,
#                          "agent_mode": "code", "source": "heuristic"}
```

An orchestrator that already knows the mode can skip detection entirely:

```python
result = plugin.apply({"model": "agentic", "agent_mode": "REVIEW", "messages": [...]})
# result["model"] -> granite3.3:8b, routing["source"] == "explicit"
```

An agent-aware payload is decided by the rules layer, with no model inference involved:

```python
result = plugin.apply({
    "model": "agentic",
    "agent": "codex",
    "session_id": "abc123",
    "task": "coding",
    "tools": True,
    "reasoning": True,
    "context_tokens": 42000,
    "messages": [...],
})
# result["model"]            -> "gpt-oss:120b"
# result["routing"]["source"]  -> "rules"
# result["routing"]["rule_id"] -> "task-coding"
# result["routing"]["session"] -> {"session_id": "abc123", "mode": "code",
#                                  "model": "gpt-oss:120b", "reused": False}
```

The next turn of the same session stays on the same model, even when its text looks unrelated:

```python
result = plugin.apply({"model": "agentic", "session_id": "abc123", "messages": [...]})
# routing["source"] -> "affinity", routing["session"]["reused"] -> True
```

A request whose target mode lacks a required capability is escalated instead of broken:

```python
result = plugin.apply({"model": "agentic", "task": "review", "tools": True, "prompt": "..."})
# review cannot call tools -> result["model"] -> "qwen3.6:35b"
# routing["source"] -> "rules", routing["rule_id"] -> "task-review",
# routing["escalated"] -> True, routing["escalated_from"] -> "review"
# result["agent_mode"] -> "plan" — the payload reports the escalated mode, review stays visible in escalated_from
```

### 3.10 Module Layout

| Module                | Responsibility                                                                       |
|-----------------------|--------------------------------------------------------------------------------------|
| `signals.py`          | `RequestSignals` — payload → normalized, immutable facts (§3.2)                        |
| `rules.py`            | `RoutingRule`, `parse_rules`, `match_rule`, `describe_rule` (§3.3)                     |
| `capabilities.py`     | `requirements_from_signals`, `satisfies`, `missing_capabilities`, `filter_modes`, `escalate` (§3.4) |
| `session_affinity.py` | `SessionAffinitySettings`, `CachedDecision`, `SessionAffinityCache` (§3.5)             |
| `heuristics.py`       | `detect_heuristic`, `score_mode`, `score_to_similarity` — layer 4                      |
| `semantic.py`         | `SemanticLayer` — availability gate and lookup for layer 5                             |
| `config.py`           | `AgenticRoutingConfig`, `AgentMode`, defaults, validation, env overrides               |
| `agentic_routing.py`  | `AgenticRoutingPlugin` — the cascade itself and the payload annotation                 |

The deterministic layers (1–4) import no ML library, and `semantic.py` only needs a router that is built elsewhere,
so `SEMANTIC_ENABLED=false` keeps `faiss` and `sentence-transformers` out of the process entirely.

### 3.11 Running Tests

```bash
pytest tests/test_agentic_routing.py tests/test_agentic_routing_deterministic.py -v
```

FAISS-backed cases are skipped automatically when `faiss` is not installed; the default JSON config keeps
`semantic.enabled = true`, so tests that build a real plugin set `LLM_ROUTER_ROUTING_AGENTIC_SEMANTIC_ENABLED=false`.
`tests/test_agentic_routing_deterministic.py` covers the deterministic layers — signals, rules, capabilities and
escalation, session affinity and the cascade order — and runs with no ML dependency at all.

---

## 4. Comparison: Which Plugin to Use?

| Feature                 | Simple Semantic Routing          | Bi-Encoder Semantic Routing           | Agentic Routing                                      |
|-------------------------|----------------------------------|---------------------------------------|------------------------------------------------------|
| **Trigger**             | `model == "auto"`                | `model == "auto"`                     | `model == "agentic"`                                 |
| **Decides from**        | Intent + token complexity        | Nearest embedding target              | Declared agent facts + work mode                     |
| **Approach**            | Heuristic (keyword/phrase)       | Neural embeddings (FAISS)             | Explicit → rules → affinity → heuristic → embeddings |
| **Accuracy**            | Rule-based, limited context      | Semantic understanding of meaning     | Keyword precision, mode-level granularity            |
| **Model Required**      | ❌ None                          | ✅ Bi-encoder embeddings              | Optional — layers 1–4 need no ML                     |
| **Speed**               | Very fast (~0.1ms)               | Slower (~50-200ms, model dependent)   | Fast; embeddings only when facts are silent          |
| **Config Complexity**   | JSON keywords/phrases/patterns   | JSON targets + examples               | JSON rules + modes + capabilities                    |
| **Scalability**         | Linear keyword search            | FAISS index (efficient at scale)      | Linear keyword search; FAISS at scale                |
| **Caller Control**      | None                             | None                                  | ✅ Pass `agent_mode`/`task`/`session_id`             |
| **Persistence**         | N/A                              | ✅ FAISS index saved to disk          | ✅ FAISS index saved to disk                         |
| **Use Case**            | Fast, lightweight routing        | High-quality semantic matching        | Agents that switch activities per step               |

### Recommendation

- Use **Simple Semantic Routing** when:
    - You need fast, deterministic routing with no external model dependency.
    - Your routing categories are well-defined by keywords and phrases.
    - You want minimal infrastructure.

- Use **Bi-Encoder Semantic Routing** when:
    - You need semantic understanding beyond keywords (e.g., synonyms, paraphrasing).
    - You have diverse, nuanced use cases that keyword matching can't capture.
    - You can afford the embedding model latency and dependencies.

- Use **Agentic Routing** when:
    - An agent (or your orchestrator) moves between distinct activities such as planning, coding and reviewing.
    - The caller can declare facts about the step (`agent_mode`, `task`, `tools`, `session_id`) and expects that
      declaration to win over guessing.
    - You want deterministic, auditable decisions first — declarative rules and session affinity resolve most
      requests without any model inference — and semantic matching only as the last resort.
    - A cheap mode must be escalated automatically when the step needs capabilities it does not declare.
    - You need per-mode model assignment without touching temperature, tokens or tools.

---

## 5. File Locations

| File                                                           | Purpose                             |
|----------------------------------------------------------------|-------------------------------------|
| `llm_router_plugins/utils/routing/embedder.py`                 | Shared `EmbeddingRouter` (BiEncoder + FAISS) |
| `llm_router_plugins/utils/routing/target.py`                   | Shared `RoutingTarget` dataclass    |
| `llm_router_plugins/utils/routing/common.py`                   | Shared config base, env helpers, router factory |
| `llm_router_plugins/utils/routing/simple_semantic/`            | SimpleSemanticRoutingPlugin code    |
| `llm_router_plugins/utils/routing/semantic_biencoder/`         | SemanticBiEncoderRoutingPlugin code |
| `llm_router_plugins/utils/routing/agentic_routing/`            | AgenticRoutingPlugin code           |
| `llm_router_plugins/utils/routing/agentic_routing/signals.py`  | Layer 0: agent-aware request signals (`RequestSignals`) |
| `llm_router_plugins/utils/routing/agentic_routing/rules.py`    | Layer 1: declarative `when`/`then` rules (`RoutingRule`, `match_rule`) |
| `llm_router_plugins/utils/routing/agentic_routing/session_affinity.py` | Layer 2: TTL/LRU `session_id` → mode cache |
| `llm_router_plugins/utils/routing/agentic_routing/heuristics.py` | Layer 3: deterministic keyword/phrase/pattern scoring |
| `llm_router_plugins/utils/routing/agentic_routing/semantic.py` | Layer 4: optional FAISS fallback (`SemanticLayer`) |
| `llm_router_plugins/utils/routing/agentic_routing/capabilities.py` | Capability checks and model escalation |
| `llm_router_plugins/utils/routing/agentic_routing/config.py`   | Modes, rules and settings loading   |
| `llm_router_plugins/utils/text_extractor.py`                   | Shared payload text extraction      |
| `llm_router_plugins/resources/routing/simple_semantic.json`    | Intent definitions & config         |
| `llm_router_plugins/resources/routing/semantic_biencoder.json` | Embedding routing config            |
| `llm_router_plugins/resources/routing/agentic_routing.json`    | Agent mode definitions & config     |
| `tests/test_routing_common.py`                                 | Unit tests (shared layer)           |
| `tests/test_simple_semantic_routing.py`                        | Unit tests (Simple)                 |
| `tests/test_semantic_biencoder_routing.py`                     | Unit tests (Bi-Encoder)             |
| `tests/test_agentic_routing.py`                                | Unit tests (Agentic)                |
| `tests/test_agentic_routing_deterministic.py`                    | Unit tests (Agentic deterministic layers: signals, rules, capabilities, affinity) |
