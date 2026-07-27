# Semantic Routing Plugins

Two routing plugins are available for **heuristic-based model selection** in the **LLM‑Router** system.  
Both activate when `payload["model"] == "auto"`.

---

## Table of Contents

- [1. Simple Semantic Routing (Heuristic)](#1-simple-semantic-routing-heuristic)
- [2. Bi-Encoder Semantic Routing (Embedding-based)](#2-bi-encoder-semantic-routing-embedding-based)

---

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
  "debug": 5,   // Very strong signal for code intent
  "błąd": 5,    // Very strong signal for code intent
  "kod": 2,     // Moderate signal
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

All configuration lives in [`llm_router_plugins/resources/routing/simple_semantic.json`](../resources/routing/simple_semantic.json).

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
      "keywords": ["code", "debug", "funkcja"],
      "phrases": ["write code:5", "fix bug:4"],
      "patterns": ["def\\s+\\w+\\s*\\(", "class\\s+\\w+"],
      "weights": {"debug": 5, "błąd": 4}
    },
    "math": {
      "keywords": ["calculate", "equation"],
      "phrases": ["calculate:4", "solve equation:4"],
      "patterns": ["\\b\\d+\\s*[+\\-*/^]\\s*\\d+\\b"],
      "weights": {"calculate": 4, "solve": 4}
    },
    "creative": {
      "keywords": ["write", "story", "napisz"],
      "phrases": ["write a story:4", "napisz wiersz:4"],
      "patterns": ["napisz\\s+(mi|ci|go|ją)"],
      "weights": {"napisz": 4, "write": 3}
    },
    "general": {
      "keywords": ["explain", "difference", "wyjaśnij"],
      "phrases": ["what is:4", "how to:4"],
      "patterns": ["\\bwhat\\s+(is|are|does)\\b"],
      "weights": {"wyjaśnij": 3, "help": 2}
    }
  },
  "none": {
    "keywords": ["hello", "cześć", "thanks"],
    "phrases": ["hello:1", "thanks:1"],
    "patterns": ["^\\b(hello|hi|cześć)\\b"],
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
      "keywords": ["keyword1", "keyword2"],
      "phrases": ["phrase one:4", "phrase two:3"],
      "patterns": ["pattern\\s+here"],
      "weights": {"keyword1": 5, "keyword2": 2}
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
(**radlab/semantic-euro-bert-encoder-v1**) to compute semantic embeddings for a set of pre-configured routing targets.
Each target has a `name`, a `model_name` (the model to route to), a `description`, and a list of `examples`.
At query time the user message is embedded and matched against all stored target embeddings using FAISS
(`IndexFlatIP` on L2-normalised vectors = cosine similarity). The best-matching target determines the selected model.

### 2.1 Index Building (on first load or when the persist directory is missing)

- For each target, its `description` and `examples` are combined into text.
- The text is split into overlapping **token chunks** using a sliding window (`chunk_size` tokens, `chunk_overlap`
  tokens overlap).
- Each chunk is embedded via the BiEncoder model (e.g. `radlab/semantic-euro-bert-encoder-v1`).
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

Configuration is loaded from [`llm_router_plugins/resources/routing/semantic_biencoder.json`](../resources/routing/semantic_biencoder.json).

Example JSON configuration:

```json
{
  "embedding_model": "radlab/semantic-euro-bert-encoder-v1",
  "settings": {
    "chunk_size": 256,
    "chunk_overlap": 64,
    "similarity_threshold": 0.0,
    "top_k": 1
  },
  "routing_targets": [
    {
      "name": "code-generation",
      "model_name": "qwen3.6:35b",
      "description": "Model specialized for code-related tasks.",
      "examples": ["Write a Python function...", "..."]
    },
    {
      "name": "math-reasoning",
      "model_name": "gpt-oss:120b",
      "description": "Model for mathematical reasoning and calculations.",
      "examples": ["Calculate the derivative of...", "..."]
    }
  ]
}
```

#### Environment Variables

| Variable                                          | Purpose                               |
|---------------------------------------------------|---------------------------------------|
| `LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_MODEL`     | Override the embedding model name     |
| `LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_TARGETS`   | Pipe-separated list of target names   |
| `LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_CHUNK_SIZE`| Override chunk size                   |
| `LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_CHUNK_OVERLAP`| Override chunk overlap             |
| `LLM_ROUTER_ROUTING_SEMANTIC_BIENCODER_PERSIST_DIR`| Directory for FAISS index persistence|

### 2.5 Usage Examples

#### Basic Usage

```python
from llm_router_plugins.utils.routing.semantic_biencoder.semantic_biencoder_routing import SemanticBiEncoderRoutingPlugin

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

| Target Name        | Model           | Description                          | Examples                                          |
|--------------------|-----------------|--------------------------------------|---------------------------------------------------|
| code-generation    | qwen3.6:35b     | Write and debug code                 | "Write a Python function to sort a list", "Fix my CSS layout" |
| math-reasoning     | gpt-oss:120b    | Mathematical problems and calculations | "Calculate the integral of x²", "Solve for x in 2x+3=7" |
| creative-writing   | gpt-oss:120b    | Creative writing and storytelling    | "Write a short story about...", "Create a poem about..." |
| general-knowledge  | gpt-oss:120b    | General questions and explanations   | "What is the capital of France?", "How does DNA work?" |

### 2.6 Scoring Details

For each user query, FAISS returns `top_k` closest chunks with similarity scores. These scores are **aggregated per
target** (mean of all matching chunk scores). The final score table looks like:

| Target             | Mean Cosine Similarity | Selected? |
|--------------------|------------------------|-----------|
| code-generation    | 0.85                   | ✅ Yes    |
| math-reasoning     | 0.42                   |           |
| creative-writing   | 0.31                   |           |
| general-knowledge  | 0.28                   |           |

The target with the **highest mean similarity** wins. If all scores fall below `similarity_threshold` (default 0.0),
the default model is used.

### 2.7 Running Tests

```bash
pytest tests/test_semantic_biencoder_routing.py -v
```

---

## 3. Comparison: Which Plugin to Use?

| Feature                | Simple Semantic Routing        | Bi-Encoder Semantic Routing       |
|------------------------|--------------------------------|------------------------------------|
| **Approach**           | Heuristic (keyword/phrase)     | Neural embeddings (FAISS)          |
| **Model Required**     | ❌ None                        | ✅ `radlab/semantic-euro-bert-v1` |
| **Speed**              | Very fast (~0.1ms)             | Slower (~50-200ms, model dependent)|
| **Accuracy**           | Rule-based, limited context    | Semantic understanding of meaning  |
| **Config Complexity**  | JSON keywords/phrases/patterns | JSON targets + examples            |
| **Scalability**        | Linear keyword search          | FAISS index (efficient at scale)   |
| **Persistence**        | N/A                            | ✅ FAISS index saved to disk       |
| **Use Case**           | Fast, lightweight routing      | High-quality semantic matching     |

### Recommendation

- Use **Simple Semantic Routing** when:
  - You need fast, deterministic routing with no external model dependency.
  - Your routing categories are well-defined by keywords and phrases.
  - You want minimal infrastructure.

- Use **Bi-Encoder Semantic Routing** when:
  - You need semantic understanding beyond keywords (e.g., synonyms, paraphrasing).
  - You have diverse, nuanced use cases that keyword matching can't capture.
  - You can afford the embedding model latency and dependencies.

---

## 4. File Locations

| File                                                        | Purpose                           |
|-------------------------------------------------------------|-----------------------------------|
| `llm_router_plugins/utils/routing/simple_semantic/`         | SimpleSemanticRoutingPlugin code  |
| `llm_router_plugins/utils/routing/semantic_biencoder/`      | SemanticBiEncoderRoutingPlugin code |
| `llm_router_plugins/resources/routing/simple_semantic.json` | Intent definitions & config       |
| `llm_router_plugins/resources/routing/semantic_biencoder.json` | Embedding routing config        |
| `tests/test_simple_semantic_routing.py`                     | Unit tests (Simple)               |
| `tests/test_semantic_biencoder_routing.py`                  | Unit tests (Bi-Encoder)           |
