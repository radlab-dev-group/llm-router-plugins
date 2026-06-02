# Semantic Routing Plugin — `simple_semantic_routing`

Heuristic-based model selection plugin for the **LLM‑Router** system.  
When `payload["model"] == "auto"`, the plugin analyses the user's input and selects the most appropriate model from a
configured pool.

---

## 1. Overview

### 1.1 What it does

The plugin performs **two-stage heuristic routing**:

| Stage                     | Description                                                                                                                                            |
|---------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Intent Classification** | Scans the text for keywords, multi-word phrases, and regex patterns to determine the user's intent (`code`, `math`, `creative`, `general`, or `none`). |
| **Complexity Analysis**   | Estimates token count (word count × 1.25) and maps it to a complexity level (`simple`, `medium`, or `complex`).                                        |
| **Model Selection**       | Combines the intent target with the complexity level to pick the best model from the pool.                                                             |

### 1.2 Intent Categories

| Intent         | Description                                  | Examples                                                        |
|----------------|----------------------------------------------|-----------------------------------------------------------------|
| **`code`**     | Programming, debugging, implementation       | "napisz funkcję", "fix bug", "debug code", "git commit"         |
| **`math`**     | Mathematics, calculations, statistics        | "oblicz prawdopodobieństwo", "solve equation", "calculate mean" |
| **`creative`** | Creative writing, editing, brainstorming     | "napisz opowiadanie", "write a story", "brainstorm"             |
| **`general`**  | General questions, explanations, comparisons | "wyjaśnij jak", "what is the difference", "compare"             |
| **`none`**     | No specific intent (greetings, etc.)         | "cześć", "hello", "dziękuję"                                    |

---

## 2. Architecture

### 2.1 Data Flow

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

### 2.2 Intent Classification Algorithm

Each intent is defined in `simple.json` with **four complementary signal types**:

#### a) Keywords — single words

Each keyword has an optional weight. If no weight is specified, the default is **1.0**.

```json
"keywords": ["code", "debug", "funkcja"],
"weights": {"debug": 5, "kod": 4, "code": 2}
```

Matching `"debug"` adds **5** to the score; matching `"code"` adds **2**.

#### b) Phrases — multi-word expressions

Phrases use the format `"text:weight"`. The default weight is **2.0** when omitted.

```json
"phrases": ["write code:5", "fix bug:4", "napisz funkcję:4"]
```

Matching `"write code"` adds **5** to the score.

#### c) Regex Patterns — structural detection

Patterns add a flat **+3.0** per match, useful for detecting code structures, math formulas, and question patterns.

```json
"patterns": [
"def\\s+\\w+\\s*\\(",
"class\\s+\\w+\\s*[(:]",
"\\b\\d+\\s*[+\\-*/^]\\s*\\d+\\b",
"\\bwhat\\s+(is|are|does)\\b"
]
```

#### d) Weights — keyword importance

The `weights` object maps individual keywords to boost factors, allowing fine-grained control:

```json
"weights": {
"debug": 5, // Very strong signal for code intent
"błąd": 5, // Very strong signal for code intent
"kod": 2, // Moderate signal
"python": 2    // Moderate signal
}
```

### 2.3 Model Selection Algorithm

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

---

## 3. Configuration

### 3.1 JSON Config (`simple.json`)

All configuration lives in `llm_router_plugins/resources/routing/semantic/simple.json`.

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
        ...
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
        ...
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
        ...
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
        ...
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

### 3.2 Configurable Values

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

### 3.3 Environment Variable Overrides

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

---

## 4. Usage

### 4.1 Basic Usage

```python
from llm_router_plugins.utils.routing.semantic.simple import DefaultSemanticRoutingPlugin

plugin = DefaultSemanticRoutingPlugin()

payload = {
    "model": "auto",
    "messages": [{"role": "user", "content": "Napisz funkcję w Pythonie"}]
}

result = plugin.apply(payload)
# result["model"] → selected model (e.g., "qwen3.6:35b")
```

### 4.2 Payload Text Sources

The plugin extracts text from the payload in this priority order:

1. `payload["messages"][-1]["content"]` — last message content
2. `payload["user_last_statement"]`
3. `payload["query"]`
4. `payload["prompt"]`
5. `payload["input"]`

### 4.3 Logging

When a logger is provided, the plugin logs routing decisions:

```python
import logging

logger = logging.getLogger("llm_router")
plugin = DefaultSemanticRoutingPlugin(logger=logger)

# On apply:
# INFO: Semantic routing: intent=code, complexity=medium (12 tokens) -> qwen3.6:35b
```

---

## 5. Adding New Intents

To add a new intent category:

1. **Add the intent to `simple.json`:**

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

### 5.1 Weight Tuning Guidelines

| Weight Range | Meaning         | Use Case                                                              |
|--------------|-----------------|-----------------------------------------------------------------------|
| **1**        | Weak / baseline | Generic terms that appear in many contexts                            |
| **2–3**      | Moderate        | Common but not definitive signals                                     |
| **4–5**      | Strong          | Highly specific, unambiguous signals (e.g., "debug", "błąd" for code) |
| **1 (none)** | Neutral         | Greetings and filler phrases that shouldn't influence routing         |

### 5.2 Phrase Format Guidelines

| Format          | Weight        | Example          |
|-----------------|---------------|------------------|
| `"text:weight"` | Explicit      | `"write code:5"` |
| `"text"`        | Default (2.0) | `"fix bug"`      |

---

## 6. Complexity Levels

| Level     | Token Range                                     | Model Selection                     |
|-----------|-------------------------------------------------|-------------------------------------|
| `simple`  | 0 – `thresholds[simple]`                        | Cheapest / simplest model (index 0) |
| `medium`  | `thresholds[simple]` + 1 – `thresholds[medium]` | Middle model (index n // 2)         |
| `complex` | > `thresholds[medium]`                          | Strongest model (index n - 1)       |

Token estimation: `round(word_count × 1.25)`

---

## 7. Built-in Patterns Reference

### 7.1 Code Patterns

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

### 7.2 Math Patterns

| Pattern                                  | Matches                  |
|------------------------------------------|--------------------------|
| `\b\d+\s*[+\-*/^]\s*\d+\b`               | Arithmetic expressions   |
| `\bsin\s*\(`, `\bcos\s*\(`, `\btan\s*\(` | Trig functions           |
| `\bsqrt\s*\(`, `\blog\s*\(`              | Math functions           |
| `\bf\(x\)\s*=`                           | Function notation        |
| `\b∑\b`, `\b∫\b`                         | Sigma / integral symbols |

### 7.3 General Question Patterns

| Pattern                      | Matches                     |
|------------------------------|-----------------------------|
| `\bwhat\s+(is\|are\|does)\b` | What questions              |
| `\bhow\s+(to\|does\|do)\b`   | How questions               |
| `\bwhy\s+(is\|does)\b`       | Why questions               |
| `\bporównaj\s+`              | Polish comparison questions |

---

## 8. Running Tests

```bash
pytest tests/test_default_semantic_routing.py -v
```

---

## 9. File Locations

| File                                                        | Purpose                     |
|-------------------------------------------------------------|-----------------------------|
| `llm_router_plugins/utils/routing/semantic/simple.py`       | Plugin implementation       |
| `llm_router_plugins/resources/routing/semantic/simple.json` | Intent definitions & config |
| `tests/test_default_semantic_routing.py`                    | Unit tests                  |
