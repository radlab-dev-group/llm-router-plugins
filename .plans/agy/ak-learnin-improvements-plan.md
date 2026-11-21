# Plan Usprawnień Procesu Uczenia `agentic_knowledge` (Continuous Learning)
## Architektura adaptacyjnego rankingu, wielokrokowej atrybucji nagród, kontekstowych bandytów i przeciwdziałania zjawisku cold-start

**Wersja dokumentu:** 2.0  
**Data:** 21 listopada 2026  
**Status:** Propozycja architektoniczna do zatwierdzenia (/plan)  
**Ścieżka pliku:** `.plans/agy/ak-learnin-improvements-plan.md`  
**Dotyczy:** `llm_router_plugins/utils/knowledge/agentic_knowledge/` (`learning.py`, `plugin.py`, `cache.py`, `assembler.py`, `miner.py`, `evaluation.py`) oraz CLI (`learner.py`, `tune.py`, `stats.py`)

---

## 1. Wprowadzenie i Stan Obecny Systemu Uczenia

Wtyczka `agentic_knowledge` posiada unikalną na tle całego ekosystemu `llm-router` architekturę uczenia statystycznego. W odróżnieniu od statycznych wag konfiguracji, warstwa ta potrafi adaptować swoje zachowanie na podstawie obserwowanych efektów działania agenta (sukces weryfikacji, powrót błędu, pętla).

```
                 ŚCIEŻKA PROXY (Online, fail-open, bez opóźnień)
   ┌────────────────────────────────────────────────────────────────────────┐
   │ 1. Assembler pobiera snapshot modelu z Redis (cache lokalny 60 s)      │
   │ 2. Odczyt reputacji dokumentu (Beta-posterior + Thompson sampling)     │
   │ 3. Obliczenie mnożnika: m = clamp(rep × ranker_mult, 0.3, 1.5)        │
   │ 4. Zapis zdarzenia ekspozycji do Redis Stream (agentic_kb:training)    │
   └───────────────────────────────────┬────────────────────────────────────┘
                                       │ Redis Stream (XADD)
                                       ▼
                 PROCES WORKERA (Offline, cykliczny / cron)
   ┌────────────────────────────────────────────────────────────────────────┐
   │ 1. llm-router-agentic-kb-learner drenuje strumień treningowy           │
   │ 2. Rozliczenie etykiet (kontrafaktyczne per-dokument)                  │
   │ 3. Trening ContextRanker (regresja logistyczna ridge L2, Newton-step)  │
   │ 4. Weryfikacja jakości na holdoucie (warunek: holdout_auc >= 0.60)     │
   │ 5. Publikacja nowego snapshotu z wersjonowaną historią (rollback)      │
   └────────────────────────────────────────────────────────────────────────┘
```

### 1.1. Co zostało zrealizowane w ostatnich iteracjach (stan na gałęzi `feat/learning-loop-2`):
- **Trzypoziomowy model**:
  1. *Reputacja dokumentu*: Rozkład Beta-Bernoulli z czasem połowicznego zaniku (`half_life_seconds: 30 dni`) i próbkowaniem Thompsona.
  2. *Model cech kandydata*: Ridge Logistic Regression z macierzą Hessego (Laplace approximation) dostarczającą wariancję wag jako składnik eksploracyjny.
  3. *Bandyta źródeł*: `SourceBandit` estymujący wartość źródła per znak w parach `(mode, kind)`.
- **Zarządzanie stanem i bezpieczeństwo**:
  - Przechowywanie reputacji i logów w Redis (z zachowaniem działania in-memory w razie braku połączenia).
  - Wersjonowana historia snapshotów per scope (`snapshot_history_size: 8`) i dwustronny mechanizm rollbacku w CLI (`--rollback`).
  - Bramki jakościowe: przy braku etykiet (`min_samples: 200`) lub zbyt niskim AUC (`min_holdout_auc: 0.6`) model publikuje bezpieczny *neutral snapshot* (`weights = [0.0]`, mnożnik neutralny `1.0`).
- **Narzędzia analityczne**:
  - `stats.py`: Dashboard monitorujący stopień eksploracji, rozkład cech, dryf danych i ryzyko samowzmocnienia (self-reinforcement).
  - `tune.py`: Replay-based grid search optymalizujący progi polityki pod kątem `recipe_hit_rate - λ * chars` z ograniczeniem na `escalation_precision` i `escalation_recall`.

### 1.2. Diagnoza kluczowych ograniczeń procesu uczenia
Mimo solidnych fundamentów matematycznych, audyt kodu ujawnił istotne bariery blokujące pełne wykorzystanie potencjału uczenia maszynowego:
1. **Jednoturowe okno atrybucji kredytu (Short Attribution Horizon)**: Po wstrzyknięciu wiedzy w turze $T$, `_settle_feedback` oczekuje rozliczenia natychmiast w turze $T+1$. Jeśli agent najpierw edytuje kod w $T+1$, a komendę testową uruchamia w $T+2$, etykieta przepada bezpowrotnie.
2. **Krytyczna rzadkość danych (Extreme Label Sparsity & Cold Start)**: W realnych projektach liczba zweryfikowanych sukcesów w mikroskali pojedynczych sesji jest niska. Wymóg 200–1000 etykiet powoduje, że model miesiącami tkwi w trybie neutralnym.
3. **Płaska i ograniczona przestrzeń cech**: Z 15 zdefiniowanych cech, aż 8 jest zablokowanych na sztywno (`_LEARNABLE = (1, 5, 6, 7, 8, 9, 13)`). Brakuje kluczowych cech semantycznych, miar dopasowania intencji oraz kontekstu stanu agenta.
4. **Liniowość modelu predykcyjnego**: Klasyczna regresja logistyczna nie jest w stanie uchwycić nieliniowych interakcji cech (np. wysoki score semantyczny + zerowe dopasowanie ścieżek to zazwyczaj fałszywy alarm).
5. **Bezkontekstowy bandyta źródeł**: `SourceBandit` zlicza nagrody per para `(mode, kind)` globalnie, ignorując kontekst zapytania (np. rodzaj zadania, stopień skomplikowania repozytorium).
6. **Dyskretna kalibracja w `tune.py`**: Sztywny grid-search po kilku punktach zamiast ciągłej, wielowymiarowej optymalizacji bayesowskiej.

---

## 2. Szczegółowy Plan Rozwoju Architektury Uczenia

---

### Filar 1: Wielokrokowa Atrybucja Nagród i Lepsze Etykietowanie (Credit Assignment)

#### 1.1. Rozszerzenie horyzontu czasowego atrybucji (Multi-turn Attribution Window)
- **Problem**: Obecnie w `plugin.py:967`:
  ```python
  pending = self._pending_feedback.pop(turn.session_id, None)
  ```
  Każda tura niszczy stan oczekującego feedbacku. Jeśli agent wykonał operację pośrednią (np. odczyt pliku `read_file` lub edycję `edit_file`), a weryfikację przeprowadził krok później, tracimy sygnał o przydatności receptury.
- **Rozwiązanie**: Wprowadzenie bufora atrybucji z oknem przesuwnym o długości $K=3$ tur lub do momentu zakończenia zadania:
  - Zapisujemy wstrzyknięte dokumenty w strukturze FIFO z czasem życia: `Tuple[turn_index, injected_docs, initial_signature]`.
  - Weryfikacja zielonego testu w turze $T+1$, $T+2$ lub $T+3$ przypisuje nagrodę z geometrycznym dyskontem czasowym:
    $$\gamma^{\Delta t}, \quad \text{gdzie } \gamma = 0.85, \; \Delta t = \text{turn\_curr} - \text{turn\_injected}$$
  - Jeśli w międzyczasie nastąpiła zmiana celu lub reset sesji, bufor jest bezpiecznie czyszczony.

#### 1.2. Weryfikacja adaptacji kodu (Code Adoption & Diff Overlap)
- **Problem**: Etykieta `+0.5` przyznawana jest, jeśli zminowano naprawę i dokument dzieli wspólną ścieżkę pliku. Prowadzi to do fałszywego nagradzania dokumentów, które agent widział, ale całkowicie zignorował.
- **Rozwiązanie**: Porównanie semantyczne i tokenowe faktycznie zastosowanego diffu z diffem zawartym w recepturze:
  - Ekstrakcja kluczowych zmienionych linii z receptury (`+ linia`).
  - Obliczenie współczynnika pokrycia tokenów (Token Overlap / Jaccard):
    $$\text{AdoptionScore} = \frac{|\text{Tokens}(\text{TurnDiff}) \cap \text{Tokens}(\text{RecipeDiff})|}{|\text{Tokens}(\text{RecipeDiff})|}$$
  - Jeśli $\text{AdoptionScore} > 0.4$, przyznajemy pełną etykietę `+1.0` (udowodniona adopcja). Jeśli $\text{AdoptionScore} < 0.05$ przy wspólnej ścieżce, etykieta wynosi `0.0` (unattributed – brak dowodu na wpływ).

#### 1.3. Jawne wykrywanie negatywnego sprzężenia (Reversion & Rejection Detection)
- **Sytuacja**: Agent czyta recepturę, aplikuje łatkę, po czym uruchamia polecenie, widzi nowy błąd i cofa zmiany (`git checkout`, `git restore` lub ręczny revert).
- **Detekcja w `miner.py`**:
  - Jeśli odwrócony diff anuluje zmiany wprowadzone po wstrzyknięciu receptury, dokument otrzymuje jednoznaczną, silną karę: `label = -1.0`, `weight = 1.0` z powodem `recipe_reverted`.
  - Daje to bezcenny negatywny sygnał trenujący bez konieczności czekania na powtórzenie identycznej sygnatury błędu.

---

### Filar 2: Przełamywanie Rzadkości Danych i Cold Start

```
                         STRATEGIA COLD-START
   ┌───────────────────────────────────────────────────────────────┐
   │ 1. Global Meta-Prior (Wytrenowany na korpusie multi-repo)    │
   │    Wagi początkowe W_0 zamiast [0, 0, ...]                    │
   └───────────────────────────────┬───────────────────────────────┘
                                   │
                                   ▼
   ┌───────────────────────────────────────────────────────────────┐
   │ 2. Syntetyczny Bootstrap z historii Git                       │
   │    Mining przeszłych bug-fix commitów w repozytorium          │
   └───────────────────────────────┬───────────────────────────────┘
                                   │
                                   ▼
   ┌───────────────────────────────────────────────────────────────┐
   │ 3. Hierarchical Empirical Bayes                               │
   │    Płynne przejście z Meta-Prior do wag lokalnych             │
   │    W_scope = (N * W_local + M * W_global) / (N + M)           │
   └───────────────────────────────────────────────────────────────┘
```

#### 2.1. Hierarchiczny Prior Bayesowski (Hierarchical Empirical Bayes)
- Zamiast zerowego punktu startowego ($w_i = 0$), definiujemy **Globalny Prior Wiedzy** obliczony na zagregowanych danych ze wszystkich projektów organizacji:
  $$\mathbf{w}_{\text{prior}} = \mu_{\text{global}}, \quad \mathbf{\Sigma}_{\text{prior}} = \text{diag}(\sigma^2_{\text{global}})$$
- Przy estymacji wag dla nowego scope'u, regularyzacja L2 w Newton-step przyciąga wagi nie do zera, lecz do $\mathbf{w}_{\text{prior}}$:
  $$\mathcal{L}(\mathbf{w}) = \mathcal{L}_{\text{data}}(\mathbf{w}) + \frac{1}{2} (\mathbf{w} - \mathbf{w}_{\text{prior}})^T \mathbf{\Sigma}_{\text{prior}}^{-1} (\mathbf{w} - \mathbf{w}_{\text{prior}})$$
- **Efekt**: Model w nowym repozytorium od pierwszej etykiety potrafi sensownie ważyć dopasowanie sygnatury czy polecenia testowego, zamiast czekać na 200 lokalnych próbek.

#### 2.2. Syntetyczny Bootstrap z Historii Git (Git History Replay Ingestion)
- Implementacja nowego polecenia CLI: `llm-router-agentic-kb-bootstrap --repo-path /path/to/repo`.
- Narzędzie analizuje historię gita:
  - Wyszukuje commity ze słowami kluczowymi `fix`, `resolve`, `bug`, `closes`.
  - Kojarzy commit naprawiający ze stanem kodu przed poprawką (tworząc syntetyczne pary: stan błędu $\to$ rozwiązujący diff).
  - Wstrzykuje te pary do kolekcji jako gotowe `recipe` oraz syntetyczne etykiety treningowe do strumienia `agentic_kb:training`.
- Pozwala to na zainicjowanie bazy wiedzy setkami wiarygodnych receptur i etykiet natychmiast po dołączeniu nowego projektu.

---

### Filar 3: Wzbogacenie Przestrzeni Cech (Feature Engineering 2.0)

Rozszerzenie wektora cech `CandidateFeatures` z 15 cech do 22 wysoce informatywnych wskaźników, ze szczególnym uwzględnieniem cech interakcyjnych:

| Indeks | Nazwa cechy | Typ | Znaczenie biznesowe i matematyczne | Uczalna (`_LEARNABLE`) |
|---|---|---|---|:---:|
| 0 | `bias` | float | Punkt przecięcia modelu (intercept) | Nie (stała) |
| 1 | `score` | [0, 1] | Znormalizowany wynik z silnika SSE | **Tak** |
| 2 | `score_sq` | [0, 1] | Nieliniowy człon wyniku ($score^2$) | Nie (regularyzowany) |
| 3 | `scale_rrf` | 0/1 | Flaga skali Reciprocal Rank Fusion | Nie |
| 4 | `scale_reranker` | 0/1 | Flaga skali Cross-Encodera | Nie |
| 5 | `signature_match` | [0, 1] | **NOWOŚĆ: Ciągłe podobieństwo sygnatur** (0.0 do 1.0) | **Tak** |
| 6 | `path_match` | 0/1 | Przynajmniej jedna wspólna ścieżka pliku | **Tak** |
| 7 | `path_stem_match` | [0, 1] | Znormalizowany udział wspólnych stemów modułów | **Tak** |
| 8 | `command_match` | 0/1 | Zgodność polecenia testowego | **Tak** |
| 9 | `age_days_log` | float | $\log(1 + \text{age})$ – spadek wartości starej wiedzy | **Tak** |
| 10 | `upvotes_log` | float | $\log(1 + \text{upvotes})$ w Redis | Nie |
| 11 | `downvotes_log` | float | $\log(1 + \text{downvotes})$ w Redis | Nie |
| 12 | `exposure_log` | float | $\log(1 + \text{exposures})$ | Nie |
| 13 | `rank` | [0, 1] | Pozycja w wynikach SSE ($\text{rank} / 20$) | **Tak** |
| 14 | `size_share` | [0, 1] | Udział wielkości dokumentu w budżecie znakowym | Nie |
| 15 | `score_x_path` | [0, 1] | **NOWOŚĆ: Interakcja**: $score \times path\_match$ | **Tak** |
| 16 | `sig_x_cmd` | [0, 1] | **NOWOŚĆ: Interakcja**: $signature \times command$ (pewny fix) | **Tak** |
| 17 | `agent_loop_level` | [0, 1] | **NOWOŚĆ: Stan agenta**: znormalizowany `loop_count` | **Tak** |
| 18 | `is_human_doc` | 0/1 | **NOWOŚĆ: Źródło**: dokumentacja vs auto-mined | **Tak** |
| 19 | `patch_sim_prior` | [0, 1] | **NOWOŚĆ**: Podobieństwo leksykalne diffu do dotkniętych linii | **Tak** |
| 20 | `query_intent_align`| [0, 1] | **NOWOŚĆ**: Zbieżność intencji zapytania z typem dokumentu | **Tak** |
| 21 | `error_class_match`| 0/1 | **NOWOŚĆ**: Identyczna klasa błędu (np. `TimeoutError`) | **Tak** |

- **Interakcje cech (`score_x_path`, `sig_x_cmd`)**: Rozwiązują problem liniowości bez konieczności porzucania wypukłej optymalizacji Newtona. Pozwalają modelowi precyzyjnie karać kandydatów o wysokim podobieństwie tekstu, ale całkowicie obcych plikach.
- **Kontekst stanu agenta (`agent_loop_level`)**: Pozwala modelowi dynamicznie zmieniać preferencje: w stanie pętli (`loop_count >= 2`) preferowane są wyłącznie sprawdzone receptury z identyczną sygnaturą, a dokumentacja ogólna jest tłumiona.

---

### Filar 4: Nowoczesna Architektura Modelu Rankującego (Learning-to-Rank)

Obecny klasyfikator logistyczny traktuje każdego kandydata niezależnie (pointwise). W wyszukiwaniu kluczowy jest **ranking relatywny (pairwise/listwise)** wewnątrz grupy kandydatów dla danego zapytania.

#### 4.1. Przejście na Pairwise Logistic Loss (RankNet formulation)
- Zamiast minimalizować stratę binarną dla pojedynczego dokumentu, optymalizujemy prawdopodobieństwo, że dokument trafny $A$ powinien znaleźć się wyżej niż dokument nietrafny $B$:
  $$P(A \succ B) = \sigma(\mathbf{w}^T (\mathbf{x}_A - \mathbf{x}_B))$$
- **Funkcja celu z regularyzacją L2**:
  $$\mathcal{L}_{\text{pairwise}}(\mathbf{w}) = \sum_{(A, B) \in \mathcal{P}} - \log \sigma(\mathbf{w}^T (\mathbf{x}_A - \mathbf{x}_B)) + \frac{\lambda}{2} \|\mathbf{w}\|_2^2$$
- **Zalety**:
  - Gradient i Hesjan mają analityczną, ściśle wypukłą postać – zachowujemy algorytm Newtona-Raphsona i natychmiastową zbieżność w 4–6 krokach.
  - Model bezpośrednio maksymalizuje miary rankingowe (MRR, NDCG), eliminując problem kalibracji bezwzględnych prawdopodobieństw.
  - Czas inferencji w proxy pozostaje rzędu **< 0.05 ms** (jeden iloczyn skalarny wektorów o długości 22).

#### 4.2. Opcjonalny wariant drzewiasty (LightGBM LambdaRank) w Workerze
- W przypadku repozytoriów o bardzo dużym ruchu (> 5000 etykiet), wdrożenie w workerze `learner.py` opcjonalnego silnika GBDT (LightGBM).
- Wygenerowany model jest eksportowany do prostego formatu tablicowego (Decision Tree Arrays) i wykonywany w proxy przez miniaturowy, bezdependencyjny interpreter w czystym Pythonie:
  ```python
  def predict_tree_fast(features: Sequence[float], tree_nodes: List[Tuple]) -> float:
      node_idx = 0
      while True:
          feat_idx, threshold, left, right, value = tree_nodes[node_idx]
          if feat_idx == -1:
              return value
          node_idx = left if features[feat_idx] <= threshold else right
  ```
- Gwarantuje to obsługę dowolnie skomplikowanych nieliniowości przy zerowym narzucie pamięciowym i braku zewnętrznych zależności C++ w procesie proxy.

---

### Filar 5: Kontekstowe Bandyty Źródeł (Contextual Bandits)

#### 5.1. Zastąpienie statycznego `SourceBandit` modelem LinUCB / Contextual Thompson
- **Stan obecny**: `SourceBandit` zlicza sumę nagród i zużytych znaków niezależnie dla każdej pary `(mode, kind)`.
- **Nowy model (LinUCB)**:
  Wartość źródła $a$ (np. `recipe`, `deadend`, `doc`) zależy od cech kontekstu zapytania $\mathbf{z}$ (np. długość promptu, czy wystąpił błąd, czy tryb to debug):
  $$\hat{r}_a(\mathbf{z}) = \mathbf{z}^T \boldsymbol{\theta}_a + \alpha \sqrt{\mathbf{z}^T \mathbf{A}_a^{-1} \mathbf{z}}$$
  gdzie:
  - $\mathbf{A}_a = \mathbf{I} + \sum \mathbf{z} \mathbf{z}^T$ (macierz kowariancji cech dla ramienia $a$),
  - $\boldsymbol{\theta}_a$ to wektor wag nagrody per znak.
- **Działanie w proxy**:
  - Na początku zapytania wyznaczamy oczekiwaną wartość na znak dla każdego źródła w danym kontekście.
  - Źródła o najwyższej oczekiwanej wartości otrzymują większy udział w budżecie znakowym (`budget_chars`).

---

### Filar 6: Ciągła Kalibracja i Optymalizacja Bayesowska (`tune.py`)

#### 6.1. Zastąpienie siatki dyskretnej optymalizacją bayesowską (Optuna)
- Obecne narzędzie `tune.py` wykonuje brute-force po z góry ustalonej siatce kilku progów.
- **Propozycja**: Integracja silnika TPE (Tree-structured Parzen Estimator) z biblioteki Optuna (lub lekkiej implementacji bayesowskiej):
  - Ciągła przestrzeń poszukiwań:
    - `min_similarity` $\in [0.20, 0.60]$
    - `strong_recipe_score` $\in [0.30, 0.80]$
    - `loop_threshold` $\in [1, 5]$
    - wagi per źródło $\in [0.1, 2.0]$
  - **Wielokryterialna funkcja celu**:
    $$\text{Maximize } \mathcal{F} = \text{recipe\_hit\_rate} + 0.3 \cdot \text{escalation\_recall} - \lambda \cdot \frac{\text{chars}}{1000} - 1.5 \cdot \max(0, \text{prec\_floor} - \text{escalation\_precision})$$
- Automatyczne generowanie gotowego pliku konfiguracyjnego `agentic_knowledge.tuned.json` wraz z pełnym raportem Markdown do dołączenia do Pull Requesta.

---

## 3. Architektura Przepływu Danych i Nowy Układ Redis

```
┌────────────────────────────────────────────────────────────────────────┐
│                              Redis Layout                              │
├───────────────────────────────────┬────────────────────────────────────┤
│ Klucz                             │ Zawartość i Przeznaczenie           │
├───────────────────────────────────┼────────────────────────────────────┤
│ {prefix}:training                 │ Bounded Stream (XADD) z ekspozycjami│
│ {prefix}:rep:{doc}                │ DocReputation hash (a, b, e, t)    │
│ {prefix}:model:{scope}            │ Live Model Snapshot (Pairwise wagi)│
│ {prefix}:model_history:{scope}    │ Lista poprzednich modeli (rollback)│
│ {prefix}:bandit:contextual        │ Macierze A_a i b_a dla LinUCB      │
│ {prefix}:meta_prior               │ Globalne wagi bazowe dla cold-start│
│ {prefix}:attr_buffer:{session_id} │ Multi-turn attribution sliding buf │
└───────────────────────────────────┴────────────────────────────────────┘
```

---

## 4. Harmonogram Wdrożenia (Fazy Projektu)

```mermaid
flowchart TD
    subgraph Faza 1: Rozszerzenie Atrybucji i Etykiet [Faza 1: Atrybucja & Etykiety]
        P1["Wielokrokowe okno atrybucji (K=3 tury)"] --> P2["Weryfikacja adopcji kodu (Diff Overlap)"]
        P2 --> P3["Detekcja wycofania zmian (Revert Detection)"]
    end

    subgraph Faza 2: Cechy 2.0 i Cold-Start [Faza 2: Cechy & Cold Start]
        Q1["Implementacja 22-elementowego wektora cech"] --> Q2["Global Meta-Prior (Hierarchical Bayes)"]
        Q2 --> Q3["Narzędzie CLI do bootstrapu z Git"]
    end

    subgraph Faza 3: Nowy Model Rankingowy [Faza 3: Pairwise RankNet]
        R1["Pairwise logistic loss w ContextRanker"] --> R2["Aktualizacja worker learner.py"]
        R2 --> R3["Kontekstowy bandyta LinUCB dla źródeł"]
    end

    subgraph Faza 4: Optymalizacja i CI/CD [Faza 4: Optuna & Replay]
        S1["Bayesian Optimization w tune.py"] --> S2["Replay benchmark & testy dryfu"]
        S2 --> S3["Wdrożenie i walidacja produkcyjna"]
    end

    Faza 1 --> Faza 2 --> Faza 3 --> Faza 4
```

| Faza | Zakres zadań | Czas | Ryzyko | Kryterium akceptacji |
|---|---|---|---|---|
| **Faza 1: Lepsza atrybucja** | Bufor multi-turn, diff overlap, revert penalty w `plugin.py` i `miner.py` | 2-3 dni | Niskie | Wzrost liczby poprawnie powiązanych etykiet o min. +50% |
| **Faza 2: Cechy i Cold Start** | Wektor 22 cech z interakcjami, Hierarchical Bayes, git bootstrap | 3-4 dni | Średnie | Model uczy się na < 50 etykietach, wysokie AUC od 1. fitu |
| **Faza 3: Pairwise RankNet** | Sformułowanie parowe w `learning.py`, kontekstowy bandyta LinUCB | 3-4 dni | Średnie | Wzrost NDCG/AUC na holdoucie powyżej 0.75 w testach replay |
| **Faza 4: Optymalizacja** | Integracja Optuna w `tune.py`, raporty automatyczne PR, drift tests | 2 dni | Niskie | Automatyczne strojenie polityki znajdujące optymalny punkt Pareto |

---

## 5. Plan Weryfikacji, Testy i Metryki Sukcesu

### 5.1. Zestaw testów jednostkowych i integracyjnych
- `tests/test_agentic_knowledge_learning.py`:
  - Test poprawności gradientu i Hesjanu dla sformułowania parowego (pairwise loss).
  - Test wielokrokowego bufora atrybucji: wstrzyknięcie w turze 1, edycja w turze 2, zielony test w turze 3 $\implies$ przypisanie etykiety z dyskontem $\gamma^2$.
  - Test blokowania etykiety, gdy diff nie wykazuje adopcji (`AdoptionScore < 0.05`).
  - Test zachowania hierarchicznego priora przy małej liczbie próbek ($N < 20$).
- `tests/test_agentic_knowledge_tune.py`:
  - Test wielowymiarowej optymalizacji i zachowania podłogi precyzji eskalacji.

### 5.2. Ewaluacja na zbiorze Replay (`scripts/agentic-kb-eval.sh`)
Porównanie wariantu `read_learned_v2` z obecnym `read_learned`:
1. **Holdout AUC**: Wzrost z obecnego progu minimalnego 0.60 do stabilnego **0.75–0.85**.
2. **`recipe_hit_rate`**: Wzrost o min. **+20–30%** dzięki parowemu rankingowi i cechom interakcyjnym.
3. **Redukcja szumu**: Spadek liczby wstrzykniętych nieprzydatnych znaków (`unattributed exposures`) o **35%**.
4. **Odporność na dryf**: Test statystyczny w `stats.py` wykazujący brak samowzmocnienia (`explored_share > 0.15`).
