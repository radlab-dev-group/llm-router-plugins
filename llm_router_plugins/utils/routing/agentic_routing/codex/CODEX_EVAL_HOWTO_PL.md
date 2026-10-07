# Kalibracja i ewaluacja routingu Codexa

English version: [CODEX_EVAL_HOWTO_EN.md](CODEX_EVAL_HOWTO_EN.md).

## 1. Co właściwie kalibrujesz

Ta instrukcja dotyczy przede wszystkim `agentic_routing_codex` i implementacji w tym repozytorium opisanej podczas weryfikacji z 6–7 października 2026 r. Wartości i wyniki przytoczone poniżej są punktem odniesienia z tego przeglądu, nie gwarancją aktualności po kolejnych zmianach konfiguracji lub kodu.

**Cel: poprawnie rozpoznać bieżący tryb pracy, a dopiero potem wybrać przypisany do niego model.** Nie stroisz routingu pod nazwę konkretnego modelu generującego odpowiedź.

Są trzy niezależne mechanizmy:

1. **Reguły strukturalne i faza pracy** — jawny tryb, Plan Mode, tytuł, kompaktowanie, wykonane narzędzia i ich wyniki.
2. **Heurystyki** — punktacja słów, fraz i wyrażeń regularnych.
3. **Semantyka** — podobieństwo embeddingów aktualnej czynności do opisów i przykładów trybów.

Do tego dochodzi opcjonalna pamięć fazy w Redisie.

Kolejność rozstrzygania:

```text
jawny tryb
→ klasa żądania: tytuł / kompaktowanie
→ deklaracja Plan Mode
→ aktualna faza / wiarygodna pamięć fazy
→ heurystyki
→ embeddingi
→ fallback
→ mapowanie trybu na model
```

**Ważne:** jeśli zła decyzja zapadnie już w heurystykach, obniżenie progu embeddingów jej nie naprawi. Semantyka nie jest wtedy pytana.

Kalibracja w obecnym systemie **nie oznacza trenowania ani fine-tuningu modelu embeddingowego**. Oznacza dobór modelu embeddingowego, opisów, przykładów, parametrów indeksowania i progów akceptacji.

## 2. Do czego służą poszczególne konfiguracje i pliki

| Plik / ustawienie | Rola |
|---|---|
| `llm_router_plugins/resources/routing/agentic_routing_codex.json` | Tryby Codexa, reguły, przykłady semantyczne, progi i mapowanie trybów na modele. |
| `tests/data/codex_routing_quality.json` | Zbiór żądań z oczekiwanymi trybami. Służy do pomiaru, nie jest automatycznie dodawany do indeksu embeddingów. |
| `tests/data/codex_routing_baseline.json` | Zamrożone wcześniejsze predykcje do porównań na wspólnych przypadkach. |
| `codex-routing-holdout-verification.json` | Zapisany wynik konkretnego replayu. To raport, nie konfiguracja. |
| `LLM_ROUTER_MODELS_CONFIG` | Konfiguracja modeli po stronie aplikacji routera: dostępność, dostawca, parametry obsługi itd. Nie definiuje przykładów do klasyfikowania trybów. |
| `simple_semantic.json` | Konfiguracja osobnego pluginu heurystycznego. Mimo nazwy nie wymaga embeddingów. |
| `semantic_biencoder.json` | Konfiguracja osobnego pluginu embeddingowego do ogólnego routingu. Nie zastępuje konfiguracji Codexa. |
| `agentic_routing_claude_code.json` | Konfiguracja innego pluginu, dla Claude Code. |

Zmiana `semantic_biencoder.json` **nie kalibruje automatycznie Codexa**. Współdzielony jest silnik embeddingowy, ale konfiguracje i warstwy decyzyjne są osobne.

### Najważniejsze pola konfiguracji Codexa

| Pole | Znaczenie |
|---|---|
| `embedding_model` | Model tworzący wektory; ścieżka lokalna lub identyfikator modelu. To nie model odpowiadający użytkownikowi. |
| `settings.trigger_model` | Alias uruchamiający plugin, domyślnie `auto_codex`. |
| `settings.fallback_mode` | Tryb wybierany, gdy żaden mechanizm nie rozstrzygnie, obecnie `implement`. |
| `codex_modes[].name` | Nazwa trybu, np. `review`. |
| `codex_modes[].model_name` | Model używany po wybraniu tego trybu. |
| `codex_modes[].description` | Opis znaczenia trybu, indeksowany semantycznie. |
| `codex_modes[].examples` | Przykłady czynności należących do trybu, indeksowane semantycznie. |
| `keywords`, `phrases`, `patterns`, `weights` | Sygnały dla heurystyk, a nie przykłady embeddingowe. |
| `settings.phase` | Reguły interpretacji czynności: komendy, narzędzia, ścieżki testów, zapowiedzi itd. |
| `settings.semantic` | Parametry indeksowania i akceptacji odpowiedzi semantycznej. |
| `settings.vector_store_path` | Opcjonalny katalog trwałego indeksu semantycznego. To nie pamięć sesji. |

**Redis jest osobnym mechanizmem.** Jego połączenie konfigurujesz niezależnymi zmiennymi `ENV`, nie dopisujesz danych połączenia do konfiguracji embeddingów ani nie korzystasz automatycznie z ustawień Redis dla auth.

## 3. Jak rozumieć score i similarity

### Punktacja heurystyczna

Aktualne domyślne wagi:

```text
keyword = 1
phrase  = 2
pattern = 3
```

Można je zmieniać przez `settings.heuristic_weights`, a także nadpisywać wagę konkretnego słowa lub frazy.

Przykład składni pól pojedynczego trybu:

```json
{
  "keywords": ["pytest"],
  "weights": {"pytest": 3},
  "phrases": ["uruchom testy jednostkowe:4"],
  "patterns": ["\\buruchom\\s+testy\\b"]
}
```

Nakładające się trafienia są deduplikowane: **nie zakładaj, że wszystkie pasujące reguły zawsze się zsumują**. Powtarzanie jednego słowa również nie nabija punktacji bez końca.

Akceptacja wymaga:

```text
najlepszy score > 0
najlepszy score >= heuristic_min_score
najlepszy score > drugi score
najlepszy score - drugi score >= heuristic_min_margin
```

Aktualnie w konfiguracji:

```text
heuristic_min_score  = 3.0
heuristic_min_margin = 1.0
```

Dla heurystyk pole `routing.similarity` jest przeliczeniem:

```text
similarity = score / (score + 1)

score = 3 → similarity = 0.75
```

**To nie znaczy „75% szans, że tryb jest poprawny”.**

### Wynik embeddingowy

Opisy i przykłady są zamieniane na znormalizowane wektory. Silnik porównuje je z wektorem żądania za pomocą podobieństwa kosinusowego.

Przy `aggregation: "per_target_top_k"`:

1. Wyszukiwane są podobieństwa wszystkich fragmentów indeksu.
2. Dla każdego trybu wybierana jest taka sama liczba najlepszych fragmentów.
3. Ich podobieństwa są uśredniane.
4. Powstaje ranking trybów `all_scores`.

Efektywna liczba fragmentów na tryb wynosi:

```text
min(top_k, liczba fragmentów w najmniejszej klasie)
```

Dlatego `top_k: 3` **nie oznacza trzech trybów w wyniku**. Oznacza do trzech najlepszych fragmentów na każdy tryb.

Akceptacja wymaga:

```text
s1 >= threshold
s1 > s2
s1 - s2 >= min_margin
```

Gdzie `s1` to wynik najlepszego trybu, a `s2` — drugiego.

Obecne ustawienia:

```json
{
  "threshold": 0.51,
  "min_margin": 0.05,
  "aggregation": "per_target_top_k"
}
```

Przykłady:

| Najlepszy wynik | Drugi wynik | Decyzja przy obecnych progach |
|---:|---:|---|
| `0.62` | `0.54` | Akceptacja: wystarczający wynik i przewaga. |
| `0.62` | `0.60` | Odrzucenie: za mała przewaga. |
| `0.49` | `0.35` | Odrzucenie: za niski wynik. |
| `0.62` | `0.62` | Odrzucenie: remis. |

**Kosinus `0.62` też nie oznacza 62% prawdopodobieństwa poprawności.** Nie porównuj go bezpośrednio z heurystycznym `0.75` ani strukturalnym `1.0`.

## 4. Co rzeczywiście trafia do embeddingów

Indeks trybów zawiera:

- nazwę trybu i jego `description`;
- teksty z `examples`.

Nazwy przypisanych modeli generujących nie są dowodem semantycznym. `aux_title` i `compaction` są rozstrzygane strukturalnie i nie uczestniczą w semantycznym rankingu głównych trybów.

Zapytanie semantyczne obejmuje osobno:

- intencję obecnego polecenia użytkownika;
- kontekst aktualnej fazy: ostatnią wypowiedź agenta i strukturalny opis czynności narzędzia.

**Surowa treść odczytanego pliku albo wyniku narzędzia nie jest dodawana jako opis fazy.** Dzięki temu temat pliku nie powinien zastępować informacji, co agent robi.

Sekcje intencji i fazy są embedowane oddzielnie, następnie ich wektory są uśredniane i normalizowane.

| Parametr | Co kontroluje |
|---|---|
| `intent_max_chars` | Limit znaków sekcji intencji; obecnie `2000`. |
| `phase_max_chars` | Limit znaków sekcji fazy; obecnie `2000`. |
| `classify_max_chars` | Osobny budżet parsera historii; nie jest wagą intencji względem fazy. |
| `chunk_size` | Rozmiar fragmentów indeksowanych opisów/przykładów; obecnie `256`. |
| `chunk_overlap` | Nakładanie fragmentów; obecnie `64`. |
| `top_k` | Liczba najlepszych fragmentów uwzględnianych na tryb; obecnie `3`. |

Zwiększenie `phase_max_chars` nie jest bezpośrednim ustawieniem „daj fazie dwa razy większą wagę”. Zmienia ilość dostępnego tekstu.

## 5. Jak przygotować dobry zbiór do kalibracji

Masz już `tests/data/codex_routing_quality.json`. Możesz użyć go jako punktu wyjścia, ale do mocniejszego pomiaru potrzebujesz większej liczby niezależnych sesji.

Minimalny przykład osobnego datasetu:

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
              "text": "Przejrzyj obsługę błędów tego modułu i opisz problemy. Nie zmieniaj plików."
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
              "text": "Oceń poprawność walidatora i wskaż ryzyka, bez przygotowywania poprawki."
            }
          ]
        }
      ]
    }
  ]
}
```

Dla sekwencji faz dodajesz:

- wspólną wartość `sequence`;
- metadane `session_id`, `thread_id`, `agent_name`, `turn_id` w polu przypadku `metadata`;
- rzeczywiste prefiksy historii: wywołania narzędzi, identyfikatory i wyniki;
- opcjonalne `payload` dla pozostałych pól żądania, np. `tools`, formatu odpowiedzi i metadanych Codexa.

**Uwaga:** pole przypadku `metadata` jest w replayu dokładane do `client_metadata` żądania. Jawny override trybu można zachować w `payload`.

Zasady etykietowania:

1. Oczekiwany tryb wynika wyłącznie z informacji dostępnych **przed decyzją**.
2. Etykieta opisuje bieżącą czynność, nie cały temat zadania.
3. Historyczna decyzja routera nie jest automatycznie prawidłową etykietą.
4. Pełna sesja należy albo do `calibration`, albo do `holdout` — nie dziel prawie identycznych prefiksów między oba zbiory.
5. Rzeczywiście nierozstrzygalne przypadki oznaczaj `ambiguous: true`; nie używaj tego do ukrywania błędów.
6. Usuń sekrety, ale zachowaj strukturę potrzebną do rozpoznania fazy.

Uwzględniaj trudne kontrasty, np.:

- dodanie mocka w kodzie produkcyjnym kontra pisanie testów;
- integracja z GitHub kontra analiza historii Git;
- opracowanie strategii testowania kontra uruchamianie testów;
- ocena kodu kontra przygotowanie poprawki;
- neutralny odczyt pliku po uruchomieniu testów;
- zmiana fazy `implement → test → debug → implement`.

## 6. Jak uruchamiać evaluate

### Automatyczne strojenie i eval

Skrypt Bash uruchamia całą procedurę (wymaga `jq`, Pythona z projektem i zależnościami `[ml]`):

```bash
bash scripts/codex-tune-eval.sh \
  --python /sciezka/do/venv/bin/python \
  --thresholds "0.45 0.50 0.51 0.55 0.60" \
  --margins "0.02 0.05 0.08 0.10" \
  --output-dir ./workdir/codex-tuning-01
```

Opcje `--config`, `--dataset` i `--baseline` pozwalają wskazać własne pliki; `--help` opisuje wszystkie opcje. Katalog wynikowy musi być nowy. Bez `--output-dir` powstaje nowy katalog `codex-eval-*` w repozytorium. Ścieżki podane przez użytkownika są względne do katalogu uruchomienia; skrypt można uruchomić z dowolnego katalogu.

Skrypt najpierw ocenia konfigurację wyjściową, potem każdą parę `threshold`/`min_margin` **tylko na calibration**. Wybiera najwyższą `cascade.main_mode_accuracy`, następnie mniej pominiętych i zbędnych przełączeń trybów, następnie wyższą `mode_accuracy`. Pełny remis zachowuje konfigurację wyjściową. Dopiero po zamrożeniu wyboru uruchamia holdout dla wyjściowej i wybranej konfiguracji. Nie trenuje embeddingów, nie poprawia przykładów ani nie stroi heurystyk.

Wyniki: `selection.json` (ranking kandydatów), `selected-config.json`, `holdout-comparison.json`, pełne raporty i osobne logi dla każdego replayu. Zachowuje kopie źródłowej konfiguracji i datasetu. Nie nadpisuje produkcyjnej konfiguracji ani nie wdraża zwycięzcy — przed wdrożeniem sprawdź także precision/recall, specjalne przypadki i regresje na holdout.

Każdy kandydat wykonuje rzeczywisty replay i ponownie ładuje model embeddingowy, więc domyślna siatka może być kosztowna. Błąd przerywa procedurę; log i częściowy raport pozostają do diagnozy. Do samej ewaluacji bez embeddingów użyj `--no-semantic`: pomija strojenie i raportuje warianty deterministyczne na obu zbiorach. Skrypt nie instaluje zależności i nie korzysta z produkcyjnego Redisa.

Poniższe komendy uruchamiasz z katalogu głównego repozytorium. `python` powinien wskazywać stabilny interpreter środowiska projektu w PyCharm.

Dla semantyki potrzebne są opcjonalne zależności:

```bash
python -m pip install -e '.[ml]'
```

To instrukcja instalacji dla Ciebie, nie zapis wykonanej instalacji. Wcześniejsza weryfikacja semantyki była zablokowana przez środowisko z Pythonem `3.11.0rc1`; do tych pomiarów użyj kompatybilnego, stabilnego interpretera.

### A. Punkt odniesienia bez embeddingów

```bash
python -m llm_router_plugins.utils.routing.agentic_routing.codex.evaluation \
  --config llm_router_plugins/resources/routing/agentic_routing_codex.json \
  --dataset tests/data/codex_routing_quality.json \
  --split calibration \
  --no-semantic \
  > codex-calibration-deterministic.json
```

Dostaniesz warianty `deterministic` i `stateful`.

### B. Pomiar embeddingów na kalibracji

```bash
python -m llm_router_plugins.utils.routing.agentic_routing.codex.evaluation \
  --config llm_router_plugins/resources/routing/agentic_routing_codex.json \
  --dataset tests/data/codex_routing_quality.json \
  --split calibration \
  > codex-calibration-semantic.json
```

Ten pomiar dodaje `cascade` i `semantic_only`.

### C. Końcowy pomiar na holdout

Po zakończeniu strojenia:

```bash
python -m llm_router_plugins.utils.routing.agentic_routing.codex.evaluation \
  --config llm_router_plugins/resources/routing/agentic_routing_codex.json \
  --dataset tests/data/codex_routing_quality.json \
  --split holdout \
  --baseline tests/data/codex_routing_baseline.json \
  > codex-holdout-semantic.json
```

`--split holdout` jest domyślne, ale warto podawać je jawnie. `--split all` służy do diagnostyki całości, nie do wybierania progów na zbiorze kontrolnym.

### Ważne zachowania evaluate

- Czyta wskazany plik `JSON`; nie stosuje normalnych override’ów konfiguracji routingu z `ENV`.
- Buduje świeży indeks w pamięci; nie korzysta z produkcyjnego trwałego indeksu.
- Nie wywołuje modeli generujących odpowiedzi.
- Nie korzysta z produkcyjnego Redisa. Pamięć odtwarza w izolowanym magazynie lokalnym na sekwencję.
- Bez `--no-semantic` wymaga włączonej semantyki i `aggregation: "per_target_top_k"`.
- Błąd ładowania embeddingów lub lookupu ma przerwać pomiar, zamiast udawać poprawną ewaluację fallbacku.

**Dlatego ustawienie progu przez `ENV` nie zmienia wyniku tego CLI. Do eksperymentu zapisujesz próg w pliku przekazanym przez `--config`.**

## 7. Jak czytać warianty raportu

| Wariant | Co mierzy |
|---|---|
| `deterministic` | Kaskadę bez embeddingów i bez pamięci sesji. |
| `stateful` | Kaskadę deterministyczną z odtwarzaniem pamięci fazy. |
| `cascade` | Kaskadę z embeddingami, ale bez pamięci sesji. |
| `semantic_only` | Embeddingi dla głównych żądań; dla tytułów i kompaktowania nadal routing strukturalny. |

**Obecny evaluator nie raportuje osobnego wariantu „pamięć + semantyka”.** Nie traktuj `cascade` jako kompletnej symulacji produkcji z włączonym Redisem.

`semantic_only` pomaga zobaczyć jakość samego rankingu. Nie jest „maksymalnym możliwym wynikiem”, bo celowo pomija część mocniejszych sygnałów strukturalnych.

### Najważniejsze metryki

| Metryka | Interpretacja |
|---|---|
| `mode_accuracy` | Udział poprawnych trybów w oznaczonych przypadkach. |
| `main_mode_accuracy` | Trafność dla głównej pracy, bez tytułów i kompaktowania. |
| `per_mode.precision` | Spośród wskazań danego trybu: ile było poprawnych? |
| `per_mode.recall` | Spośród przypadków wymagających danego trybu: ile wykryto? |
| `mode_confusion` | Wiersz: tryb oczekiwany; kolumna: tryb przewidziany. |
| `sources` | Która warstwa podejmowała decyzje. |
| `fallback_reasons` | Powody decyzji fallbackowych zapisane przez wariant. |
| `semantic_acceptance_rate` | Udział oznaczonych przypadków zakończonych przez źródło `semantic`; nie precision semantyki i nie udział akceptacji tylko wśród lookupów. |
| `special_cases` | Osobny wynik tytułów i kompaktowania. |
| `ambiguous_count` | Przypadki wyłączone z trafności z powodu niejednoznaczności. |
| `mean_routing_ms` | Średni czas replayu danego wariantu, nie pełna latencja produkcji. |

Przykładowo: wysoki recall `test` i niska precision `test` oznaczają, że testy są wykrywane, ale tryb przechwytuje też niepasujące zadania.

Dla sekwencji szczególnie ważne są:

- `unnecessary_switches` — zmiana trybu, gdy oczekiwana faza się nie zmieniła;
- `missed_switches` — brak zmiany przy oczekiwanym przejściu;
- `mean_switch_delay` — opóźnienie wykrycia nowej fazy liczone w kolejnych żądaniach, nie w milisekundach;
- `censored_switches` — przejścia, dla których poprawnej fazy nie osiągnięto przed końcem dostępnego fragmentu.

Same poprawne liczby przełączeń nie dowodzą poprawnego wyboru trybów. Zawsze czytaj je razem z trafnością i macierzą pomyłek.

**Nie wybieraj konfiguracji według `model_accuracy`.** Jeżeli kilka trybów ma ten sam model, błędny tryb może wyglądać jak poprawny wybór modelu.

## 8. Praktyczna procedura kalibracji embeddingów

### Krok 1: zamroź punkt odniesienia

Zapisz raport dla obecnej konfiguracji na `calibration`. Zachowaj konfigurację i dataset użyte w pomiarze.

Raport zawiera m.in. ich hashe w `metadata`. Dodatkowo zanotuj wersję kodu, rewizję wag embeddingowych i wersje zależności — hash konfiguracji nie wykryje podmiany plików modelu pod tą samą ścieżką.

### Krok 2: sprawdź, gdzie powstały błędy

Najpierw porównaj `deterministic` z `cascade`.

- Błąd ze źródła `heuristic` → poprawiaj heurystykę.
- Błąd ze źródła `phase` → sprawdzaj reguły czynności i parser.
- Błąd ze źródła `memory` → sprawdzaj metadane i cykl życia fazy.
- Błąd semantyczny albo nierozstrzygnięty fallback → oglądaj ranking embeddingowy.

Przykład odczytu rankingów, jeśli masz `jq`:

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

Nie oceniaj tylko końcowego trybu: `implement` może zostać poprawnie wskazany jako fallback, mimo że semantyka nie zaakceptowała żadnej klasy.

### Krok 3: poprawiaj opisy i przykłady

Dobre przykłady opisują **czynność i granice trybu**, a nie temat repozytorium.

Przykładowa różnica:

```text
plan:
„Opracuj strategię testowania; na razie nie pisz testów.”

test:
„Dodaj testy jednostkowe walidatora i uruchom je.”

review:
„Oceń poprawność walidatora i opisz problemy bez edycji.”

implement:
„Dodaj walidację danych wejściowych w istniejącym module.”
```

Zalecenia:

- używaj reprezentatywnych sformułowań polskich i angielskich;
- dodawaj różne parafrazy, nie dziesiątki prawie identycznych zdań;
- opisuj rozróżnienia w `description`;
- nie wpisuj nazw preferowanych modeli jako wskazówek klasyfikacji;
- nie kopiuj przypadków `holdout` do `examples`;
- unikaj dosłownego kopiowania promptów ewaluacyjnych także z kalibracji — używaj ich do identyfikacji brakujących kategorii sformułowań.

Jeżeli poprawna klasa regularnie jest druga w rankingu, samo obniżenie `threshold` zazwyczaj nie pomoże: nadal wygra zła klasa.

### Krok 4: dobierz threshold i margin

Po ustabilizowaniu opisów przetestuj niewielką siatkę wartości, np.:

```text
threshold:  0.45, 0.50, 0.51, 0.55, 0.60
min_margin: 0.02, 0.05, 0.08, 0.10
```

To **przykładowy zakres eksperymentu**, nie zalecane optimum. Dla innego modelu embeddingowego sensowny zakres może być inny.

- Wyższy `threshold`: zwykle mniej akceptacji, więcej fallbacków.
- Niższy `threshold`: więcej akceptacji, ale możliwe błędne pewne decyzje.
- Wyższy `min_margin`: odrzuca konkurujące klasy o podobnym wyniku.
- Niższy `min_margin`: akceptuje więcej przypadków granicznych.

Przykład utworzenia konfiguracji kandydującej:

```bash
jq '.settings.semantic.threshold = 0.55
    | .settings.semantic.min_margin = 0.08' \
  llm_router_plugins/resources/routing/agentic_routing_codex.json \
  > codex-candidate.json
```

Następnie:

```bash
python -m llm_router_plugins.utils.routing.agentic_routing.codex.evaluation \
  --config codex-candidate.json \
  --dataset tests/data/codex_routing_quality.json \
  --split calibration \
  > codex-candidate-calibration.json
```

CLI nie ma osobnych opcji `--threshold`, `--min-margin` ani automatycznego strojenia. Parametry podajesz przez konfigurację.

**Optymalizacja kosztu:** gdy zmieniasz wyłącznie progi, ranking `all_scores` pozostaje ten sam. Możesz przesiewać pary progów offline na zapisanych rankingach, zamiast za każdym razem ładować model. Wybraną konfigurację potwierdź jednak rzeczywistym replayem całej kaskady.

### Krok 5: dopiero potem strojenie pozostałych parametrów

Sprawdzaj oddzielnie:

- `top_k`, np. `1`, `3`, `5`;
- długości sekcji intencji i fazy;
- `chunk_size` i `chunk_overlap`;
- inne modele embeddingowe.

Nie zmieniaj wszystkiego naraz — utracisz informację, co przyniosło poprawę.

Po zmianie modelu embeddingowego ponownie dobierz progi. Ten sam kosinus nie musi mieć takiej samej wartości diagnostycznej dla różnych modeli.

### Krok 6: wybierz konfigurację według jakości, nie liczby akceptacji

Priorytety:

1. Mniej błędnych trybów w głównej pracy.
2. Mniej zbędnych i pominiętych przełączeń.
3. Brak pogorszenia klas szczególnie istotnych.
4. Brak regresji jawnych override’ów, Plan Mode, tytułów i kompaktowania.
5. Akceptowalny koszt i czas routingu.

**Mniej fallbacków nie oznacza automatycznie lepszego routingu.** Zła decyzja semantyczna może być gorsza niż ostrożny fallback.

## 9. Jak kalibrować heurystyki

Heurystyki stroisz osobno, zaczynając od raportu z `--no-semantic`.

Najczęstsze działania:

- usuń zbyt ogólne słowa przechwytujące niepasujące czynności;
- dodaj bardziej jednoznaczne frazy;
- zmniejsz wagę sygnału wieloznacznego;
- zwiększ `heuristic_min_margin`, jeżeli konkurujące klasy mają podobne wyniki;
- sprawdź lokalne negacje, np. „nie uruchamiaj testów”;
- uwzględnij pisownię z polskimi znakami i bez nich.

`CodexModeScorer.rank_modes()` udostępnia ranking z surowymi punktami i dopasowaniami `SignalMatch`. To właściwe miejsce do analizy „które słowo dodało punkty”. Raport CLI nie zawiera pełnego rankingu heurystyk.

Nie wszystkie tryby z konfiguracji są kandydatami heurystycznymi. W szczególności samo dopisywanie słów do `plan` nie zmienia go w heurystycznego zwycięzcę obecnej kaskady; planowanie ma sygnały strukturalne i ścieżkę semantyczną.

Po poprawce heurystyk ponów także pomiar z embeddingami — zmiana wcześniejszej warstwy zmienia zestaw żądań, które docierają do semantyki.

## 10. Jak potwierdzić wynik i wdrożyć konfigurację

Po wyborze kandydata na `calibration` uruchom stary i nowy config na **tym samym, niezależnym holdout**. Porównuj nie tylko podsumowania, ale też konkretne utracone i odzyskane przypadki.

Jeżeli po obejrzeniu błędów z holdout zmieniasz konfigurację pod te błędy, zbiór przestaje być niezależnym sprawdzianem. Zachowaj go jako regresje i przygotuj nowy holdout.

Obecny korpus jest mały i był już analizowany podczas poprawek. Nadaje się do regresji, ale nie jest mocnym dowodem jakości na nowych sesjach.

### Wdrożenie

Przykład wyboru gotowego pliku konfiguracji:

```bash
export LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_CONFIG=/sciezka/codex-candidate.json
```

Wybrane override’y produkcyjne:

```bash
export LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_SEMANTIC_ENABLED=true
export LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_SIMILARITY_THRESHOLD=0.55
export LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_MODEL=/sciezka/do/modelu-embeddingowego
```

`MODEL` oznacza tu model embeddingowy. `MODEL_TEST`, `MODEL_REVIEW` itd. oznaczają modele generujące przypisane do trybów.

`semantic.min_margin`, `aggregation` oraz budżety sekcji nie mają obecnie osobnych override’ów `ENV`; ustawiasz je w `JSON`.

Zmiany są ładowane przy tworzeniu pluginu, więc wymagają jego ponownego utworzenia / restartu workerów.

### Trwały indeks

Jeśli używasz `vector_store_path` albo `PERSIST_DIR`, po zmianie:

- modelu embeddingowego;
- zestawu trybów;
- opisów lub przykładów;
- parametrów dzielenia tekstu na fragmenty;

**zbuduj nowy indeks**. Najbezpieczniej wskaż nowy, wersjonowany katalog. Aktualny silnik nie gwarantuje automatycznej invalidacji indeksu po każdej zmianie konfiguracji.

Sama zmiana `threshold`, `min_margin`, `top_k` lub budżetów zapytania nie wymaga ponownego embedowania opisów.

Indeks `FAISS` i pamięć w Redisie są niezależne. Nie czyść całego Redisa podczas kalibracji embeddingów.

## 11. Co mówi obecny raport

W raporcie `codex-routing-holdout-verification.json` z wcześniejszej weryfikacji:

| Metryka | `deterministic` | `stateful` |
|---|---:|---:|
| Trafność trybów | `26/35 = 74,3%` | `29/35 = 82,9%` |
| Trafność głównej pracy | `23/32 = 71,9%` | `26/32 = 81,25%` |
| Zbędne przełączenia | `2` | `0` |
| Pominięte przełączenia | `1` | `0` |
| Recall `review` | `25%` | `25%` |
| Recall `plan` | `33,3%` | `33,3%` |

To pokazuje poprawę dzięki pamięci na tych przypadkach. **Nie pokazuje jakości embeddingów** — w tych dwóch wariantach ich nie ma, a `semantic_acceptance_rate` wynosi `0`.

Niski recall `review` i `plan` wskazuje obszary do diagnozy, nie automatycznie nakaz obniżenia progu. Najpierw sprawdź, czy błędny tryb pochodzi z heurystyki, czy poprawny tryb jest wysoko w rankingu semantycznym.

Dodatkowo obecny baseline ma niezweryfikowane historyczne pochodzenie. Używaj go jako zamrożonego snapshotu regresyjnego, nie jako dowodu wyniku „przed całą implementacją”. `--baseline` porównuje tylko wspólne, zgodnie oznaczone przypadki.

## 12. Skrócona kolejność pracy

```text
1. Sprawdź działający interpreter i model embeddingowy.
2. Przygotuj rozdzielone sesjami calibration i holdout.
3. Zapisz wynik obecnej konfiguracji bez semantyki.
4. Zapisz wynik obecnej konfiguracji z semantyką.
5. Ustal warstwę odpowiedzialną za błędy.
6. Popraw opisy / przykłady albo reguły właściwej warstwy.
7. Dobierz threshold i min_margin na calibration.
8. Sprawdź błędne tryby, precision/recall i przełączenia.
9. Potwierdź wybranego kandydata na niezależnym holdout.
10. Wdróż config i świeży indeks, sprawdź logi produkcyjne.
```

**Najważniejsza zasada: stroisz poprawność bieżącego trybu pracy, nie wysoki similarity, niski fallback ani trafność nazwy modelu.**