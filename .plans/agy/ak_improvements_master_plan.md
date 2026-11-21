# Plan Wdrożenia Usprawnień Warstwy `agentic_knowledge` (Matching & Continuous Learning)

## Opis Celu
Kompleksowa modernizacja i podniesienie skuteczności wtyczki `agentic_knowledge` w dwóch kluczowych obszarach:
1. **Dopasowywanie i wyszukiwanie wiedzy (Matching & Retrieval)**: Wyeliminowanie zaszumienia zapytań, optymalizacja interakcji z silnikiem Semantic Search Engine (SSE), usunięcie problemu zerowania zakresu ścieżkowego (`path_scope_for`), naprawa ekstrakcji słów kluczowych BM25 oraz wdrożenie elastycznego przydziału budżetu znakowego.
2. **Proces ciągłego uczenia (Continuous Learning & Attribution)**: Rozwiązanie problemu wąskiego okna atrybucji nagród (przejście na bufor wielokrokowy), weryfikacja faktycznego zastosowania kodu przez agenta (Diff Overlap), przełamanie problemu zimnego startu (Hierarchical Empirical Bayes + Git Bootstrap), rozszerzenie wektora cech o interakcje nieliniowe, przejście na parowy model rankingowy (Pairwise Logistic / RankNet) oraz wielowymiarowe strojenie hiperparametrów w `tune.py`.

Pełne, szczegółowe dokumenty analityczno-projektowe zostały zapisane w repozytorium:
- [ak-improvements-plan.md](file:///mnt/data2/dev/develop/llm-router-plugins/.plans/agy/ak-improvements-plan.md) — Plan usprawnień wyszukiwania i dopasowywania wiedzy z bazy.
- [ak-learnin-improvements-plan.md](file:///mnt/data2/dev/develop/llm-router-plugins/.plans/agy/ak-learnin-improvements-plan.md) — Plan usprawnień procesu uczenia, atrybucji i modeli rankingowych.

---

## User Review Required

> [!IMPORTANT]
> **Kluczowe decyzje architektoniczne do akceptacji przed rozpoczęciem prac implementacyjnych:**
> 1. **Redukcja podłogi `max_results` w `query.py` z 15 na 6–8**:
>    Obecna wartość 15 w połączeniu z mnożnikiem `SEMANTIC_SEARCH_FACTOR_MAX_RESULTS = 10` po stronie silnika Milvus powoduje każdorazowe pobieranie i deserializację 150 wektorów i chunków, generując latencję rzędu 1.7–2.9 s na zapytanie (i do 7.3 s przy kolejkowaniu). Zmniejszenie podłogi drastycznie skróci czas odpowiedzi do < 600 ms bez utraty jakości (bo asembler i tak wybiera max 2–3 dokumenty na źródło).
> 2. **Wielokrokowe okno atrybucji (Multi-turn Attribution Horizon, K=3 tury)**:
>    Obecnie `pending_feedback` jest bezpowrotnie niszczony po 1 turze. Proponujemy bufor $K=3$ tur z dyskontem geometrycznym $\gamma=0.85$, co pozwoli na nagradzanie wiedzy, gdy agent najpierw edytuje pliki, a test uruchamia 1–2 tury później.
> 3. **Rozszerzenie wektora cech z 15 do 22**:
>    Wprowadzenie cech interakcyjnych ($score \times path\_match$, $signature \times command$) oraz stanu pętli agenta. Zapewni to zachowanie pełnej wstecznej kompatybilności dzięki wersjonowaniu schematu snapshotów (`schema: 2`).

---

## Open Questions

> [!NOTE]
> 1. Czy preferujesz wdrożenie dwuetapowe (najpierw Faza Matching w `query.py`/`assembler.py`, a w osobnym PR Faza Learning), czy spójne wdrożenie całościowe na gałęzi roboczej?
> 2. Czy w procesie uczenia chcemy dopuścić opcjonalny lekki silnik drzewiasty (Fast Decision Tree array) dla dużych korpusów, czy pozostać ściśle przy rozszerzonym sformułowaniu analitycznym (Pairwise Logistic z krokiem Newtona)? *(Rekomendacja: Pairwise Logistic jako domyślny, bez zewnętrznych zależności).*

---

## Proponowane Zmiany w Modułach

### Komponent: Retrieval & Query Engineering (`query.py`, `sse.py`)
Poprawa jakości zapytań i odciążenie silnika SSE.

#### [MODIFY] `llm_router_plugins/utils/knowledge/agentic_knowledge/query.py`
- Uwzględnienie `text` w funkcji `_extract_keywords`, aby w trybach `plan` i `implement` zapytania BM25 nie były puste.
- Przeprojektowanie `path_scope_for`: usunięcie błędu zerowania prefiksu przy plikach `src/` i `tests/` – wyznaczanie wspólnego modułu dziedzinowego.
- Obniżenie podłogi `max_results` z `max(15, limits)` do `max(6, min(limits * 2, 12))`.
- Selektywna dekompozycja zapytania semantycznego (oddzielenie czystego komunikatu błędu od zaszumionego tracebacka).

#### [MODIFY] `llm_router_plugins/utils/knowledge/agentic_knowledge/sse.py`
- Dodanie reguły kierującej zapytania czysto symboliczne bezpośrednio do FTS (`lexical_only=True`) z pominięciem CPU embeddingu.
- Warunkowy Reranking Cross-Encoderem dla spornych wyników w trybie `debug` z ograniczeniem do top-5.

---

### Komponent: Ranking & Context Assembly (`assembler.py`)
Maksymalizacja trafności wstrzykiwanego bloku wiedzy.

#### [MODIFY] `llm_router_plugins/utils/knowledge/agentic_knowledge/assembler.py`
- Wprowadzenie ciągłego bonusu podobieństwa sygnatur błędu (Jaccard similarity na tokenach błędu) zamiast sztywnego binarnego 0.2 / 0.0.
- Wykrywanie i tłumienie przestarzałych/sprzecznych receptur dla tych samych plików.
- Adaptacyjny budżet znakowy: możliwość przydzielenia do 3000 znaków dla pojedynczej, wysoce trafnej receptury przy zachowaniu twardego limitu trybu.
- Dyrektywne tagi w szablonie Markdown (`[!IMPORTANT]`, `Known dead ends (DO NOT ATTEMPT)`).

---

### Komponent: Pętla Uczenia i Modele Rankingowe (`learning.py`, `plugin.py`, `miner.py`)
Przekształcenie systemu uczenia w pełnoprawny, odporny na cold-start silnik LTR.

#### [MODIFY] `llm_router_plugins/utils/knowledge/agentic_knowledge/plugin.py`
- Przebudowa `_settle_feedback`: bufor atrybucji wielokrokowej ($K=3$ tury z dyskontem $\gamma$).
- Weryfikacja faktycznej adopcji zmian (Diff Overlap / Token Matching) przed przyznaniem etykiety `+1.0`/`+0.5`.

#### [MODIFY] `llm_router_plugins/utils/knowledge/agentic_knowledge/miner.py`
- Rejestracja sukcesu po kompilacjach (`cargo check`, `tsc`) i linterach (`ruff`, `eslint`).
- Wykrywanie wycofania zmian przez agenta (revert/restore) i generowanie jawnego sygnału negatywnego (`-1.0`).

#### [MODIFY] `llm_router_plugins/utils/knowledge/agentic_knowledge/learning.py`
- Rozszerzenie wektora `CandidateFeatures` do 22 cech (dodanie cech interakcyjnych i stanu pętli agenta).
- Wdrożenie parowej funkcji straty (Pairwise Logistic Loss / RankNet) optymalizowanej analitycznym krokiem Newtona.
- Hierarchiczny Prior Bayesowski (Hierarchical Empirical Bayes) zapobiegający publikowaniu neutralnych zerowych wag przy małej liczbie próbek.
- Wdrożenie kontekstowego bandyty LinUCB dla dynamicznego przydziału budżetu źródeł.

---

### Komponent: Narzędzia CLI i Strojenie (`learner.py`, `tune.py`)

#### [MODIFY] `llm_router_plugins/cli/plugins/utils/agentic_knowledge/tune.py`
- Integracja wielowymiarowej optymalizacji bayesowskiej zamiast rzadkiej siatki dyskretnej.
- Automatyczne generowanie gotowej propozycji polityki `agentic_knowledge.json` z uzasadnieniem w Markdown.

---

## Verification Plan

### Automated Tests
1. **Testy jednostkowe komponentów matchingowych**:
   ```bash
   PYTHONPATH=. pytest tests/test_agentic_knowledge_query.py tests/test_agentic_knowledge_assembler.py -v
   ```
   Weryfikacja:
   - `test_extract_keywords_includes_user_text_when_no_failure`
   - `test_path_scope_module_detection_across_src_and_tests`
   - `test_signature_continuous_bonus_similarity`
   - `test_flexible_character_budget_allocation`

2. **Testy jednostkowe uczenia i atrybucji**:
   ```bash
   PYTHONPATH=. pytest tests/test_agentic_knowledge_learning.py tests/test_agentic_knowledge_learning_integration.py -v
   ```
   Weryfikacja:
   - `test_multi_turn_attribution_sliding_window`
   - `test_pairwise_newton_step_convergence`
   - `test_hierarchical_prior_shrinkage_on_small_samples`
   - `test_diff_adoption_verification_filter`

3. **Całościowy Replay kontrafaktyczny (Evaluation Replay)**:
   ```bash
   PYTHONPATH=. python3 scripts/agentic-kb-eval.sh
   ```
   Kryteria akceptacji:
   - Wzrost `recipe_hit_rate` o min. **+20%**.
   - Wzrost wskaźnika `char_efficiency` o min. **+25%**.
   - Spadek średniego czasu zapytania z ~1.7 s do **< 600 ms**.
   - Utrzymanie wskaźnika fałszywych eskalacji poniżej zadanego progu (`escalation_precision >= 0.85`).

### Manual Verification
- Uruchomienie narzędzia `llm-router-agentic-kb-stats --learning --scope <test_scope>` w celu inspekcji rozkładu cech, stopnia eksploracji i braku zjawiska samowzmocnienia.
- Przetestowanie procedury rollbacku: publikacja snapshotu testowego, wykonanie `--rollback` i weryfikacja przywrócenia poprzedniego modelu.
