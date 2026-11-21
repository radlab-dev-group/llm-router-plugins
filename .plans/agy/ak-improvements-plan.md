# Plan Usprawnień Wtyczki `agentic_knowledge` (Matching & Retrieval)
## Kompleksowa optymalizacja wyszukiwania, dopasowywania i serwowania wiedzy z bazy SSE

**Wersja dokumentu:** 2.0  
**Data:** 21 listopada 2026  
**Status:** Propozycja architektoniczna do zatwierdzenia (/plan)  
**Ścieżka pliku:** `.plans/agy/ak-improvements-plan.md`  
**Dotyczy:** `llm_router_plugins/utils/knowledge/agentic_knowledge/` (`query.py`, `assembler.py`, `sse.py`, `doc_context.py`, `miner.py`, `config.py`)

---

## 1. Wprowadzenie i Kontekst Architektoniczny

Wtyczka `agentic_knowledge` jest kluczowym komponentem warstwy pośredniczącej `llm-router`, działającym jako transparentne proxy wstrzykujące kontekstualną wiedzę do promptów agentów programistycznych (Codex, Claude Code).

```
                      ┌───────────────────────────────────────────────┐
                      │              Zapytanie Agenta                 │
                      └──────────────────────┬────────────────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │  Wykrywanie trybu i stanu │
                               │ (debug, test, plan, itp.) │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │   QueryBuilder (query.py) │
                               └─────────────┬─────────────┘
                                             │ SearchRequest
                                             ▼
                               ┌───────────────────────────┐
                               │ KnowledgeStore (sse.py)   │
                               │  - Milvus (dense vector)  │
                               │  - PostgreSQL (FTS/BM25)  │
                               │  - Cross-Encoder (rerank) │
                               └─────────────┬─────────────┘
                                             │ StoreHit[]
                                             ▼
                               ┌───────────────────────────┐
                               │ ContextAssembler          │
                               │  - normalizacja skali     │
                               │  - wagi i bonusy          │
                               │  - dopasowanie budżetu    │
                               └─────────────┬─────────────┘
                                             │ KnowledgeContext
                                             ▼
                               ┌───────────────────────────┐
                               │  Wstrzyknięcie do promptu │
                               │   <agentic_knowledge>     │
                               └───────────────────────────┘
```

Podczas gdy wtyczka posiada już zaawansowaną infrastrukturę (obsługę wielu źródeł, integrację z SSE, cache generacji i zręby uczenia), **jakość dopasowywania wiedzy (retrieval precision & recall)** napotyka na kilka fundamentalnych wąskich gardeł:
1. **Zaszumienie i ułomności zapytań (Query Construction)**: Pomijanie tekstu użytkownika przy ekstrakcji słów kluczowych BM25, zlepianie ścieżek i surowych tracebacków w gęstym wektorze.
2. **Nieoptymalna współpraca z silnikiem Semantic Search Engine (SSE)**: Agresywny oversampling (Milvus ×10 przy sztucznej podłodze `max_results=15` skutkuje pobieraniem 150 wektorów na zapytanie), co powoduje zapychanie kolejki CPU i opóźnienia do 2–7 sekund.
3. **Sztywne lub zerujące się filtry zakresu (`path_scope_for`)**: Wyznaczanie wspólnego prefiksu ścieżek zeruje filtr przy typowej pracy nad kodem i testem (`src/` vs `tests/`).
4. **Statyczne i skokowe bonusy w asemblerze**: Sztywne stałe (`SIGNATURE_BONUS=0.2`, `PATH_BONUS=0.08`), brak ciągłego oceniania podobieństwa sygnatur błędu.
5. **Dylemat formatowania i przycinania**: Brutalne ucinanie lub spłaszczanie kodu przy wyczerpaniu budżetu zamiast dynamicznej priorytetyzacji kluczowych diffów.

Poniższy plan definiuje precyzyjną ścieżkę modernizacji modułów dopasowywania wiedzy.

---

## 2. Diagnoza Stanu Obecnego i Identyfikacja Wąskich Gardeł

### 2.1. Błąd w ekstrakcji słów kluczowych BM25 (`query.py:516-544`)
W funkcji `_extract_keywords` zdefiniowano sygnaturę:
```python
def _extract_keywords(
    failure: str,
    cmd: str,
    paths: Sequence[str],
    text: Optional[str] = None,
) -> List[str]:
```
Jednak zmienna `text` (zawierająca ostatnią wypowiedź użytkownika) **nie jest nigdzie używana wewnątrz funkcji**!
- **Konsekwencja**: W trybach `plan`, `implement` lub zapytaniach ogólnych (gdzie nie ma `failure` ani `cmd`), jeśli agent nie dotknął jeszcze żadnych plików (`paths` puste), `keywords` zwraca **pustą listę**! Wyszukiwanie pełnotekstowe w SSE nie otrzymuje żadnych słów kluczowych, a hybrydowy RRF traci całkowicie nogę leksykalną.

### 2.2. Problem oversamplingu i podłogi `max_results` (`query.py:213` vs SSE Milvus)
W `query.py:213`:
```python
max_results=max(15, limits.get(name, 3))
```
Silnik SSE ma ustawioną stałą klasową:
```python
SEMANTIC_SEARCH_FACTOR_MAX_RESULTS = 10  # milvus.py:73
```
- **Konsekwencja**: Zapytanie o `max_results=15` wymusza na Milvusie pobranie `15 * 10 = 150` wektorów i chunków z bazy danych wraz z deserializacją `text_str`. Następnie FTS w Postgresie generuje wektory w locie, a RRF łączy setki wyników tylko po to, by asembler wybrał z nich **2–3 dokumenty**.
- **Wpływ na wydajność**: Jedno zapytanie trwa 1.7–2.9 s, a 3 równoległe zapytania kolejkują się na CPU do 7.3 s!

### 2.3. Wadliwe wyznaczanie zakresu katalogowego (`path_scope_for` w `query.py:260-286`)
Funkcja oblicza `common directory prefix`:
```python
common = paths[0].rsplit("/", 1)[0].split("/")
for path in paths[1:]:
    ...
```
- **Przypadek brzegowy**: Gdy agent edytuje `src/auth/jwt.py` oraz `tests/auth/test_jwt.py`, wspólnym prefiksem jest `""` (pusty string).
- **Konsekwencja**: Filtr `relative_path_contains` zostaje wyzerowany. Zamiast szukać receptur dotyczących modułu `auth`, silnik przeszukuje całe repozytorium (szum ze starych, niepowiązanych modułów).

### 2.4. Zaszumienie gęstych wektorów (Dense Embeddings Noise)
W `query.py:340-360` zapytania są sklejane w prosty ciąg:
- `debug`: `[failure, text]`
- `test`: `[command, paths, text]`
- `implement`: `[text, paths, failure]`
Wrzucenie 400 znaków surowego zrzutu tracebacka lub 10 ścieżek pliku do modelu `embeddinggemma-300m` (który ma ograniczoną pojemność i preferuje naturalny język semantyczny) drastycznie obniża podobieństwo kosinusowe do dobrze opisanych receptur i playbooków.

### 2.5. Sztywne ograniczenie do jednego filtra metadanych w SSE (`sse.py:973-1004`)
W `sse.py` wtyczka narzuca priorytetyzację:
`thread_id` > `modes` > `phases` > `doc_types`.
Nie ma możliwości jednoczesnego zapytania: „daj mi dokumenty z fazy `test` ORAZ typu `guide`”.

### 2.6. Skokowy charakter bonusów i brak uwzględnienia bliskości sygnatury (`assembler.py:573-589`)
`SIGNATURE_BONUS = 0.2` jest przyznawany binarnie: `meta.signature_hash == self.signature_hash`.
Jeśli w nowej wersji biblioteki błąd zmienił nieznacznie komunikat (np. `KeyError: 'user_id'` vs `KeyError: 'userId'`), hash sygnatury jest inny, bonus wynosi `0.0`, a sprawdzona receptura spada na sam dół rankingu.

---

## 3. Szczegółowy Plan Zmian Technicznych

---

### Moduł 1: Inteligentna Konstrukcja Zapytań (`query.py`)

#### 1.1. Naprawa i rozszerzenie `_extract_keywords`
- **Zmiana**: Włączenie parametru `text` do ekstrakcji tokenów leksykalnych. Ekstrakcja kluczowych identyfikatorów, symboli w camelCase/snake_case oraz terminów dziedzinowych z promptu użytkownika.
- **Odfiltrowanie szumu**: Zastosowanie listy `_STOPWORDS` wzbogaconej o typowe tokeny poleceń agenta (`please`, `fix`, `run`, `check`, `update`).

```python
# Proponowany fragment llm_router_plugins/utils/knowledge/agentic_knowledge/query.py
def _extract_keywords(
    failure: str,
    cmd: str,
    paths: Sequence[str],
    text: Optional[str] = None,
) -> List[str]:
    tokens: List[str] = []
    # 1. Wyjątki i klasy błędów
    for exc in re.findall(
        r"\b[A-Z][a-zA-Z0-9]*(?:Error|Exception|Timeout|Failure|Warning)\b",
        failure or "",
    ):
        if exc not in tokens:
            tokens.append(exc)
            
    # 2. Nazwy bazowe plików
    for path in paths[:8]:
        name = str(path).rsplit("/", 1)[-1]
        stem = name.split(".", 1)[0] if "." in name else name
        if stem and stem.lower() not in _STOPWORDS and stem not in tokens:
            tokens.append(stem)
            
    # 3. Identyfikatory ze zrzutu błędu i polecenia
    combined = f"{failure or ''} {cmd or ''}"
    for token in re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b", combined):
        lower = token.lower()
        if lower not in _STOPWORDS and token not in tokens and not token.isdigit():
            tokens.append(token)
        if len(tokens) >= 12:
            break

    # 4. NOWOŚĆ: Kluczowe pojęcia z tekstu użytkownika (identyfikatory, backtiki, camelCase)
    if text and len(tokens) < 15:
        # Ekstrakcja słów w backtickach: `my_variable` lub `UserService`
        for quoted in re.findall(r"`([^`]+)`", text):
            clean_q = re.sub(r"[^\w\.-]", "", quoted)
            if clean_q and clean_q.lower() not in _STOPWORDS and clean_q not in tokens:
                tokens.append(clean_q)
        # Ekstrakcja terminów camelCase/snake_case z tekstu
        for token in re.findall(r"\b(?:[a-z]+(?:_[a-z0-9]+)+|[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+)\b", text):
            if token.lower() not in _STOPWORDS and token not in tokens:
                tokens.append(token)
            if len(tokens) >= 15:
                break

    return tokens[:15]
```

#### 1.2. Wielokomponentowe formowanie zapytania semantycznego
Zamiast wrzucać do embeddera surowy stacktrace:
- Dla trybu `debug`:
  - Pobieramy znormalizowany nagłówek i ostatnią linię błędu (`KeyError: 'session_id'`), a nie 20 linii wywołań wewnętrznych biblioteki.
  - Generujemy naturalne zapytanie: `Fix for {ErrorClass}: {ErrorMessage} in {TouchedModule}`.
- Dla trybu `test`:
  - `Resolve test failure in {TestName} for command {TestCommand}`.
- Dla trybu `implement` i `plan`:
  - Pierwsze 2 zdania z `latest_user_text` + nazwy kluczowych modułów.

#### 1.3. Usprawnienie określania zakresu ścieżki (`path_scope_for`)
- Zamiast sztywnego `commonprefix` od roota repozytorium:
  - Ignorujemy prefiksy źródłowe i testowe: `src/`, `lib/`, `tests/`, `test/`, `app/`.
  - Szukamy wspólnego korzenia modułu (np. dla `src/auth/jwt.py` i `tests/auth/test_jwt.py` wykrywamy wspólny komponent `auth`).
  - Jeśli pliki są z różnych modułów, sprawdzamy najczęściej występujący katalog nadrzędny (majority voting).

#### 1.4. Redukcja podłogi pobierania `max_results`
- Zmiana w `query.py:213`:
```python
# Było: max_results=max(15, limits.get(name, 3))
# Propozycja:
max_results=max(6, min(limits.get(name, 3) * 2, 12))
```
- **Zysk**: Zmniejszenie obciążenia Milvus o 60%, skrócenie czasu oczekiwania na pojedyncze zapytanie z 1.7 s do ~500 ms, eliminacja kolejkowania CPU przy równoległych zapytaniach trybów.

---

### Moduł 2: Protokół Wyszukiwania i Konfiguracja SSE (`sse.py` & `config.py`)

#### 2.1. Dynamiczny wybór trybu wyszukiwania (Lexical vs Dense vs Hybrid)
- **Problem**: Zapytania składające się wyłącznie z symbolu błędu lub identyfikatora (np. `ERR_CONN_REFUSED` lub `TypeError: NoneType`) nie potrzebują embeddingu gęstego.
- **Rozwiązanie**: Wzbogacenie `_wants_lexical(query)` w `sse.py`:
  - Jeśli udział identyfikatorów przekracza `fts_only_min_identifier_share` (domyślnie 0.5) LUB zapytanie jest bardzo krótkim kodem błędu, zapytanie idzie w trybie `lexical_only=True`.
  - Omija to model SentenceTransformer na CPU, skracając czas odpowiedzi do < 50 ms.

#### 2.2. Selektywny (warunkowy) Reranking Cross-Encoderem
- **Założenie**: Reranking wszystkich zapytań jest zbyt wolny na CPU. Brak rerankingu powoduje dużą liczbę false positives.
- **Rozwiązanie**: Wprowadzenie progu pewności:
  - Reranker włączany tylko dla zapytań w trybie `debug` oraz gdy najlepszy wynik RRF z pierwszej fazy ma relatywnie niski wynik pewności (kandydaci są blisko siebie, np. różnica między top 1 a top 3 < 0.05).
  - Ograniczenie liczby rerankowanych kandydatów do `top_k_rerank = 5` (zamiast wszystkich 20–40).

#### 2.3. Wsparcie dla zapytań ze złożonymi filtrami
- W `sse.py` przygotowanie zapytań kompozytowych, a tam gdzie SSE nie wspiera przecięcia (`AND` pomiędzy różnymi rodzinami metadanych), wdrożenie dwuetapowego filtrowania:
  1. Zapytanie SSE z filtrem o najwyższej selektywności (np. `phase` lub `thread_id`).
  2. Post-filtering w pamięci wtyczki w `assembler.py` dla pozostałych wymiarów (`doc_type`, `mode`).

---

### Moduł 3: Precyzyjny Ranking i Składanie Kontekstu (`assembler.py`)

#### 3.1. Ciągłe (płynne) bonusy podobieństwa sygnatur
Zamiast binarnego bonusu `SIGNATURE_BONUS = 0.2` dla identycznego hasha:
- Obliczanie podobieństwa leksykalnego (token-based Jaccard lub Levenshtein) pomiędzy surowym/znormalizowanym tekstem błędu a sygnaturą dokumentu.
```python
def _signature_similarity(sig_a: str, sig_b: str) -> float:
    if not sig_a or not sig_b:
        return 0.0
    if sig_a == sig_b:
        return 1.0
    # Obliczenie współczynnika Jaccarda na tokenach błędu
    tokens_a = set(re.findall(r"\w+", sig_a.lower()))
    tokens_b = set(re.findall(r"\w+", sig_b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / float(len(tokens_a | tokens_b))
```
- Bonus skalowany: `bonus += SIGNATURE_BONUS * _signature_similarity(self.signature, meta.signature)`.
- Zapewnia to, że niemal identyczne błędy (np. różniące się tylko nazwą zmiennej lub ścieżką) otrzymują wysoki bonus (~0.16–0.19) zamiast 0.0.

#### 3.2. Wykrywanie sprzeczności i przestarzałych receptur (Superseded Recipes)
- Gdy w wynikach pojawiają się dwie receptury modyfikujące ten sam plik:
  - Sprawdzamy datę (`meta.date`) oraz wskaźniki reputacji (`upvotes` / `downvotes`).
  - Jeśli nowsza receptura ma zweryfikowany status i wysoki feedback, starsza receptura otrzymuje karę tłumienia (`suppression_penalty = 0.5`), zapobiegając wstrzykiwaniu sprzecznych instrukcji do jednego promptu.

#### 3.3. Adaptacyjne zarządzanie budżetem (`max_chars`)
- Obecnie `_QUOTE_CHARS = 1800` ucina każdy dokument jednakowo.
- Nowe podejście: **Elastic Quota Allocation**:
  - Jeśli mamy tylko 1 bardzo trafny `recipe` (np. vote > 0.8), pozwalamy mu zająć do 3000 znaków (aby zmieścił się pełny diff).
  - Jeśli mamy 3 słabsze dokumenty, dzielimy budżet dynamicznie, obcinając do 600–800 znaków.
  - Zawsze zachowujemy integralność składniową bloków ````diff` i ````python` (funkcja `_truncate_with_code_blocks`).

#### 3.4. Udoskonalenie promptu Markdown dla Agenta
- Wzbogacenie szablonu o dyrektywne instrukcje:
```markdown
<agentic_knowledge>
> [!IMPORTANT]
> The following verified fixes and conventions were automatically retrieved from the repository's knowledge base.
> Use them to avoid repeating past mistakes.

## Verified fixes in this repo
- [recipe · in test · 2026-11-15 · src/auth/jwt.py]
  Failure: `KeyError: 'alg'`
  Solution diff:
  ```diff
  - algorithm = options['alg']
  + algorithm = options.get('alg', 'HS256')
  ```

## Known dead ends (DO NOT ATTEMPT)
- [deadend · in implement · 2026-11-14] Tried using `jwt.decode(..., verify=False)` — failed security audit tests.
</agentic_knowledge>
```
- Jasne oznaczenie sekcji `deadend` jako `DO NOT ATTEMPT` znacząco redukuje skłonność modeli (szczególnie Qwen, Claude, GPT-4) do powtarzania zarzuconych ścieżek.

---

### Moduł 4: Zapis i Indeksowanie Wiedzy (`miner.py` & `doc_context.py`)

#### 4.1. Poszerzenie kryteriów sukcesu w miningu (`miner.py`)
- Obecnie `_is_green_test` wymaga formalnego testu zakończonego kodem 0.
- **Rozszerzenie**:
  - Rejestracja pomyślnych kompilacji (`cargo check`, `tsc --noEmit`, `go build`).
  - Rejestracja czystych przebiegów linterów (`ruff check`, `eslint`, `flake8`) po wcześniejszych błędach składniowych.
  - Włączenie opcji `task_solutions_enabled=True` dla kompletnych rozwiązań wieloplikowych po zakończeniu sesji.

#### 4.2. Sanityzacja i czyszczenie diffów
- Filtrowanie zmian w plikach czysto konfiguracyjnych lub lockfile'ach (`package-lock.json`, `poetry.lock`, `Cargo.lock`, pliki `.pyc`), które sztucznie pompują rozmiar receptury i zużywają cenny budżet znakowy.

#### 4.3. Wzbogacenie stojącego kontekstu dokumentacji (`doc_context.py`)
- Oprócz samych nagłówków (`_MAX_HEADINGS_PER_FILE = 6`), dodanie kluczowych zadeklarowanych symboli (nazwy klas/interfejsów) zawartych w nagłówkach drugiego stopnia (`## Class UserService`).
- Zapewnia to agentowi precyzyjną mapę API bez konieczności czytania całych plików dokumentacji.

---

## 4. Harmonogram Wdrożenia (Fazy)

```mermaid
flowchart TD
    subgraph Faza 1: Szybkie Wygrane [Faza 1: Quick Wins]
        A1["Naprawa _extract_keywords (obsługa text)"] --> A2["Obniżenie podłogi max_results (z 15 do 6)"]
        A2 --> A3["Poprawa path_scope_for (moduły zamiast prefixu)"]
    end

    subgraph Faza 2: Optymalizacja SSE [Faza 2: SSE & Retrieval]
        B1["Dynamiczny wybór lexical vs dense"] --> B2["Warunkowy Reranking top-5 na CPU"]
        B2 --> B3["Dwuetapowe filtrowanie metadanych"]
    end

    subgraph Faza 3: Zaawansowany Asembler [Faza 3: Ranking & Formatting]
        C1["Płynne bonusy podobieństwa sygnatur"] --> C2["Tłumienie sprzecznych receptur"]
        C2 --> C3["Elastyczny budżet znakowy i czytelny Markdown"]
    end

    subgraph Faza 4: Rozszerzenie Minera [Faza 4: Write Path]
        D1["Wykrywanie sukcesów kompilatorów i linterów"] --> D2["Czyszczenie diffów z plików lock"]
        D2 --> D3["Wzbogacenie doc_context o symbole API"]
    end

    Faza 1 --> Faza 2 --> Faza 3 --> Faza 4
```

| Faza | Zakres zadań | Szacowany czas | Ryzyko | Kryterium sukcesu |
|---|---|---|---|---|
| **Faza 1: Szybkie wygrane** | Naprawa `_extract_keywords`, optymalizacja `max_results`, poprawka `path_scope_for` | 1-2 dni | Niskie | Wzrost słów kluczowych BM25 w testach, spadek latencji o 50% |
| **Faza 2: Protokół SSE** | Selektywny reranking, reguły FTS-only dla symboli, filtry złożone | 2-3 dni | Średnie | Brak spowolnień powyżej 1.5 s, zero błędów filtrów w SSE |
| **Faza 3: Asembler** | Ciągłe podobieństwo sygnatur, deduplikacja sprzeczności, elastyczny budżet | 2-3 dni | Niskie | Wyższy `recipe_hit_rate` w `evaluation.py`, brak obciętych kluczowych diffów |
| **Faza 4: Mining i Ingestion** | Mining kompilacji/linterów, sanityzacja diffów, doc symbols | 2-3 dni | Niskie | Wzrost liczby minowanych receptur o 40–60% bez utraty jakości |

---

## 5. Plan Weryfikacji i Metryki Sukcesu

### 5.1. Testy automatyczne jednostkowe i regresyjne
- Rozszerzenie `tests/test_agentic_knowledge_query.py`:
  - Test ekstrakcji słów kluczowych z `latest_user_text` przy pustych `failure` i `cmd`.
  - Test zachowania `path_scope_for` dla ścieżek `src/auth/jwt.py` i `tests/auth/test_jwt.py` (sprawdzenie czy nie zwraca `""`).
- Rozszerzenie `tests/test_agentic_knowledge_assembler.py`:
  - Test płynnego bonusu dla sygnatur częściowo zbieżnych.
  - Test tłumienia przestarzałych receptur.
  - Test integralności bloków kodu przy przycinaniu elastycznym.

### 5.2. Replay kontrafaktyczny (`evaluation.py`)
Uruchomienie skryptu ewaluacyjnego na zbiorze sesji:
```bash
PYTHONPATH=. python3 scripts/agentic-kb-eval.sh
```
**Oczekiwane rezultaty KPI**:
1. **`recipe_hit_rate`**: Wzrost z obecnego poziomu bazowego o min. **+15–25% względnych**.
2. **`injected_chars_per_turn`**: Utrzymanie budżetu lub spadek o 5–10% dzięki lepszej selekcji i eliminacji szumu.
3. **`char_efficiency` (hit_rate / kchars)**: Wzrost o min. **+20%**.
4. **Mediana czasu odpowiedzi SSE**: Spadek z ~1.7 s do **< 0.6 s**.
