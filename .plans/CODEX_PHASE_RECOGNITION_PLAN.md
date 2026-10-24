# Plan poprawy rozpoznawania faz/trybów w pluginie `agentic_routing_codex`

Dokument opisuje **metodę** poprawy rozpoznawania trybów pracy (faz), nie rozbudowę listy
fraz w konfiguracji. Punktem wyjścia są pomiary na realnych payloadach Codex CLI
(`agents-conversation/codex/`, 793 żądania).

Status: do wdrożenia. Poprzedza go weryfikacja działania pluginu (sekcja 1).

---

## 1. Diagnoza: co naprawdę działa dziś

Kaskada w `classifier.py:283`:
`explicit → class → collaboration_mode → memory → phase → heuristic → semantic → fallback`

Pomiar na 793 sparsowanych payloadach, konfiguracja produkcyjna, `semantic.enabled=false`:

| warstwa              | decyzji | udział |
|----------------------|---------|--------|
| `phase`              | 555     | 70%    |
| `collaboration_mode` | 103     | 13%    |
| `class`              | 29      | 4%     |
| `heuristic`          | 25      | 3%     |
| `fallback`           | 17      | 2%     |

| tryb          | decyzji |
|---------------|---------|
| `git_review`  | 303     |
| `implement`   | 156     |
| `test`        | 138     |
| `plan`        | 103     |
| `compaction`  | 19      |
| `aux_title`   | 10      |
| `review`      | **0**   |
| `debug`       | **0**   |

Wnioski pomiarowe:

1. **Warstwa heurystyczna jest praktycznie martwa.** Na 700 requestach `main`:
   627 punktuje poniżej `heuristic_min_score=3.0`, 72 nie mają żadnego sygnału,
   1 abstynuje przez remis. Tylko 25 decyzji na całym kopusie.
2. **`review` i `debug` są konstrukcyjnie osiągalne** — syntetyczne żądania
   przechodzą poprawnie (`przejrzyj ten kod… → review(heuristic)`,
   `napraw błąd… → debug(heuristic)`). Problemem jest jakość sygnału, nie brak reguły.
3. **Detekcja porażki testów nie odpala nigdy.** 149 payloadów zawiera przegrany
   `pytest` (np. „7 failed”, `Process exited with code 2`). `_execution_status()`
   rozpoznaje status poprawnie (`False` dla 177/177 takich outputów), ale
   `collect_phase_evidence()` tworzy dowód `test_failure` jedynie przy warunku
   `linked[1] == len(evidence) - 1` (`phase.py:1093`), czyli gdy test był **ostatnim**
   dowodem. W rzeczywistej turze Codex emituje wiele wywołań równolegle
   (`function_call`, `function_call`, `function_call_output`, …) i warunek nigdy nie
   zachodzi → `debug` nie powstaje (2/793).
4. **Intencja użytkownika przegrywa z ostatnią komendą agenta.** `phase` siedzi nad
   `heuristic`, więc żądanie „co o tym sądzisz” + pierwsze `git log` → `git_review`.
   Dominuje *aktywność*, nie *prośba*.
5. **Tekst intencji jest skażony.** `latest_user_text` to cała wiadomość użytkownika:
   p50 = 5282 znaki, p99 = 12 893 (limit `classify_max_chars=4000` przycina),
   **42% zawiera bloki kodu**, 2% to przeklejone plany innego agenta.
   Udokumentowany przypadek: payload `llm-router-tests-by-agent.log:104387`
   (wklejony *plan o trybie `git_review`*) dostaje `git_review = 235.0` pkt vs
   `review = 11.0`. Metoda myli **opis czegoś** z **prośbą o coś**.
6. **Morfologia nie działa** — dopasowanie jest dosłowne. Frazy `zgodna z README:3`
   nie trafia „zgodny z opisem w README”. To główne źródło 627 abstynencji.
7. **Semantyka jest ostatnią deską ratunku, nie głosem.** Przy `semantic.enabled=true`
   odpowiada za 16/729 decyzji (2%), a to *jedyna* warstwa, która wydała `review`.
   Margines `min_margin_relative=0.004` (0.4%) akceptuje szum, a warstwa i tak
   nie jest pytana, gdy heurystyka „zgadnie”.
8. **Brak pamięci domyślnie** (`memory.enabled=false`) → ~10 decyzji w turze liczonych
   od zera, oscylacja trybów w obrębie jednej pracy.
9. **Skala `similarity` nieporównywalna między warstwami** (phase = 1.000, heuristic =
   siła reguły, semantic = cosinus) → nie da się progować globalnie ani mierzyć ufności.

---

## 2. Zasada przewodnia

Rozdzielić **dwa ortogonalne sygnały** i świadomie je łączyć, zamiast ukrywać wybór
w kolejności kaskady:

- **INTENCJA** — o co prosi użytkownik (z tekstu, oczyszczonego z szumu).
- **AKTYWNOŚĆ** — co agent faktycznie robi (dowody z komend/patchy/statusów).

Zasada łączenia: *aktywność może przesunąć intencję wyłącznie, gdy jest świeża
i nie jest z nią sprzeczna*. Dziś aktywność przesłania intencję zawsze.

Druga zasada: **każda warstwa zwraca oszacowanie ufności w wspólnej skali**, a decyzją
jest argmax po łączonej ufności z jawną abstynencją. To obsługuje punkty 5–9 powyżej.

---

## 3. Etapy wdrożenia

Kolejność wg stosunek zysku do ryzyka. Każdy etap osobno, za flagą, z bramką replay.

### Faza 0 — Pomiar przed zmianami (wymóg, nie opcja)

Bez tego nie da się udowodnić „lepiej”.

- Zbudować etykietowany kopus z `agents-conversation/`: request prefix → tryb
  oczekiwany przez człowieka patrzącego **tylko** na to, co przed decyzją.
  Format zgodny z `tests/data/codex_routing_quality.json` (`split`, `sequence`,
  `metadata`, `ambiguous`).
- **Podział splitów po `thread_id`**, nigdy po wierszu pliku (ta sama tura nie może
  znaleźć się w kalibracji i holdout).
- Wymuszyć obsadę klas rzadkich: `debug`, `review`, `aux_title`, `compaction`
  (obecny korpus: 73 przypadki, w tym `debug: 3` i `review: 5` → przy takiej
  liczebności nie da
  się ich ulepszać mierzalnie).
- Dodać metryki celowane, nie tylko `mode_accuracy`:
  - per-mode precision/recall i macierz pomyłek,
  - `debug` recall wśród żądań z przegranym testem (dziś 0),
  - FP `git_review` wśród żądań czysto implementacyjnych,
  - `unnecessary_switches` (metryka już istnieje w `evaluation.py`),
  - koszt: udział żądań na modelu tanim vs drogim.
- Zamrozić baseline (`--baseline`) i traktować go jako bramkę wejściową każdej fazy.
- Narzędzia ekstrakcji i replaya są już w repo (`scripts/plugins/auto_codex/eval/`) — pozostaje je wykorzystywać, nie pisać.

**Done gdy:** kopus ≥ ~300 przypadków z obsadzonymi `debug`/`review`, raport
`deterministic` i `cascade` reproducible, baseline w repo.

### Faza 1 — Naprawa dowodu „test nie przeszedł” (największy defekt)

Cel: `debug` ma być rozpoznawany, gdy testy faktycznie padły.

- Usunąć zależność od pozycji w `phase.py:1093`. Wiązać output z wywołaniem po
  **`call_id`** i zapamiętać indeks dowodu wywołania w momencie rejestracji, a nie
  wnioskować z `len(evidence) - 1`.
- Poprawić kolejność przetwarzania dla **równoległych wywołań**: kilka `function_call` przed
  pierwszym `function_call_output` to przypadek normalny (potwierdzony w logach),
  nie wyjątek.
- Uczyć, że `test_failure` nie musi być *ostatnim* dowodem, by liczył się przy
  dominacji w oknie (`detect_phase_evidence`, `evidence_window`).
- Rozszerzyć pojęcie niepowodzenia poza `exec_command`:
  - `apply_patch`, który zwraca błąd,
  - komendy budowania/wdrożeniowe (`make`, `tox`, `npm run build`) jako osobne
    `EVIDENCE_*`, bo ich porażka to też faza `debug`.
- Zachować ostrożność z `phase.py:949`: napis „failed” w *wczytywanym pliku* to dane,
  nie porażka bieżącego wywołania. Rozpoznawać wyłącznie statusy/obwiednie wykonania.

**Done gdy:** wśród 149 payloadów z przegranym pytestem `debug > 0`, a `test`/`implement`
bez regresji; testy jednostkowe na wiązaniu równoległych `call_id`.

### Faza 2 — Intencja jako anker, aktywność jako korekta

Cel: żądanie użytkownika nie przegrywa z pierwszą napotkaną komendą.

- Rozdzielić wprost dwa wyniki klasyfikacji: `intent_mode` (warstwy tekstowe) i
  `activity_mode` (dowody). Zamiast „phase wygrywa, bo jest wyżej w kaskadzie” —
  jawna reguła łączenia:
  - `activity` przejmuje, gdy dowód jest w ostatnich `N` zdarzeniach **oraz**
    nie jest w sprzeczności z intencją,
  - przy sprzeczności wygrywa intencja, a `activity` idzie do adnotacji
    (`reason`/`evidence`) jako dowód drugiego stopnia.
- Zdefiniować tabelę konfliktów jako dane (para intencja/aktywność → tryb), nie jako
  kod — pozwala kalibrować bez zmian w logice i utrzymuje determinizm.
- Zachować prymat `class`/`collaboration_mode`/`explicit` (warstwy strukturalne, dziś
  działające bezbłędnie — 100% zgodności z niezależną regułą).

**Done gdy:** przypadek „co o tym sądzisz” + `git log` → `review`/`implement`, nie
`git_review`; spadek `unnecessary_switches`; brak regresji `plan`/`aux_title`.

### Faza 3 — Morfologia i dopasowanie zbliżeniowe zamiast literałów

Cel: to rdzeń „metody” — zastępuje dosypywanie fraz.

- Wprowadzić **normalizację morfologiczną** tekstu i sygnałów w `scoring.py`:
  deterministyczny stemper/lematizer PL (redukcja sufiksów) + Porter dla EN,
  czysto w Pythonie, bez nowych zależności wymagalnych (fail-open: brak
  normalizatora = obecne zachowanie).
- Dopasowywać **pnie**, nie pełne wyrazy: `zgodn*` → zgodny/zgodna/zgodne/zgodność,
  `test*` → testy/testów/testowi, `przejrz*` → przeglądaj/przejrzyj/przegląd.
- Dla fraz wielowyrazowych **dopasowanie oknowe** (współwystąpienie znormalizowanych
  tokenów w oknie `K` słów) zamiast ciągłego podłańcucha — dziś „zgodny z opisem w
  README” gubi się, bo fraza konfiguracji jest inna sekwencją liter.
- Dopuszczalna reprezentacja sygnału w konfigie jako **zestaw rdzeni do
  współwystąpienia** (np. `["zgodn", "readme"]`), a nie zdanie-pamięć.
- Przelicz/przeproguj `heuristic_min_score`/`heuristic_min_margin` po zmianie skali —
  obecne `3.0/1.0` były strojonе pod literały.

**Done gdy:** spadek liczby abstynencji (627 „poniżej progu” + 72 „bez sygnału”),
`review`/`debug` > 0 w wariancie **deterministycznym**, bez wzrostu FP.

### Faza 4 — Redukcja szumu w tekście intencji (kto to napisał i po co)

Cel: nie punktować tego, czego użytkownik *nie prosi*.

- **Maskować cytaty przed scoringiem** w `intent_text` (`payload.py:237`,
  `_latest_user_text` przy `payload.py:639`): bloki kodu, diffy, logi, treść plików,
  outputy narzędzi. 42% requestów `main` zawiera bloki kodu — dziś punktowane jak
  проза.
- **Atrybucja autorstwa**: rozróżnić `user` / `agent` (poprzedni plan, „A previous
  agent produced the plan below…”) / `system` (nagłówki CLI). Plan innego agenta to
  kontekst, nie intencja. Rozszerzyć `_INTERNAL_CONTEXT_RE` (`payload.py:108`) lub
  wprowadzić osobne pole `intent_source`, żeby nie mieszać z istniejącą semantyką.
- Dla żądań typu handoff: brać **polecenie** (sekcja „Task”, zdania rozkazujące,
  ostatnie zdanie), nie akapity opisowe.
- **Waga pozycji**: początek i koniec wiadomości ważą więcej niż środek — polecenie
  jest na wierzchu, dokumentacja w środku.

**Done gdy:** payload `llm-router-tests-by-agent.log:104387` nie daje `git_review`;
brak regresji na krótkich, zwykłych żądaniach.

### Faza 5 — Wspólna skala ufności, jawna abstynencja, histereza bez Redisa

Cel: decyzja mierzalna i stabilna w turze.

- **Kalibrować `similarity` per warstwa** do wspólnej skali (monotoniczna mapa
  wyznaczona na kopusie — izotoniczna albo tablica progów per tryb), bo dziś
  `phase=1.000`, heuristic = siła reguły, semantic = cosinus nie są porównywalne.
- **Jawna abstynencja**: gdy żadna warstwa nie przekracza progu ufności → tryb
  `implement` (fallback) z adnotacją `low_confidence`, bez podnoszenia modelu.
- **Histerza w pamięci procesowej** (LRU per `thread_id`/`turn_id`, bez Redisa):
  w obrębie tury zmiana trybu tylko przy przekroczeniu progu. Dziś
  `memory.enabled=false` i ~10 decyzji w turze liczonych od zera. Uwaga: znany defekt
  diagnostyczny — `plugin.py:373` `getattr(outcome, "reason", "")` na wartości
  tekstowej, więc przyczyna konfliktu nigdy nie trafia do loga (naprawić przy okazji).
- Histereza musi zachować determinizm: stan wejściowy = wyłącznie sygnały z payloadu,
  brak zależności od kolejności żądań między procesami.

**Done gdy:** `unnecessary_switches` w dół, `similarity` progowalne globalnie,
testy determinizmu (ta sama decyzja przy iteracji forward/reverse) nadal zielone.

### Faza 6 — Semantyka jako głos, nie ostatnia deska

Cel: embeddings używane wtedy, gdy coś wnoszą.

- Pytać warstwę semantyczną **równolegle** do heurystyki i łączyć rangi
  (reciprocal rank fusion) zamiast „heurystyka zgadła → nie pytamy”. Dziś 2% decyzji.
- Kalibrować progi z dystrybuanty, nie z intuicji: `threshold=0.44`,
  `min_margin_relative=0.004` to akceptacja różnic rzędu 0.4% — to szum.
  Narzędzie gotowe: `scripts/codex-tune-eval.sh` (grid threshold × margin, wybór na
  kalibracji, raport na holdout).
- Indeksować **reprezentantów z realnych żądań** (zdania użytkowników z kopusu), nie
  tylko opisy trybów — to zmienia jakość cosine similarity bardziej niż strojenie progu.
- Zostawić `semantic_only` jako diagnostykę (już jest w `evaluation.py`).

**Done gdy:** udział semantyki > 2% bez spadku dokładności; `review`/`debug`
osiągalne także bez warstwy deterministycznej (czerwona linia: semantyka nie może być
jedynym źródłem `review`, bo deployment z `semantic.enabled=false` musi działać).

### Faza 7 — Poprawki drobnych ziaren konfiguracji (konsekwencje metody)

Nie „więcej fraz”, tylko naprawa źle zdefiniowanych reguł w
`llm_router_plugins/resources/routing/agentic_routing_codex.json`:

- `git ls-files`, `git status` → neutralne (`mode: null`), nie `git_review` —
  wylistowanie plików to nie przegląd zmian (12 takich decyzji na kopusie).
- Rozdzielić „operacje na repo” (`add`/`commit`/`push`) od „analizy historii”
  (`log`/`blame`/`show`) — to różny kontekst i różny model kosztowo.
- `python3 script.py` to nie `test`; testem jest wywołanie frameworka testowego.
- Przegląd mapek pod kątem „czy ta komenda *jest* tą pracą, czy tylko jej towarzyszy”.

---

## 4. Bramka wdrożeniowa (jednakowa dla każdej fazy)

1. Replay deterministyczny na kopusie + `tests/` (`968 passed`, `18 skipped` baseline).
2. `python3 -m llm_router_plugins.utils.routing.agentic_routing.codex.evaluation \
     --config <nowa> --dataset <kopus> --split holdout --baseline <zamrożony>`
3. Kryteria dopuszczenia:
   - `mode_accuracy` nie niżej niż baseline,
   - per-mode recall docelowej klasy w górę,
   - brak regresji `plan` / `aux_title` / `compaction` (dziś 100%),
   - brak regresji integralności: passthrough zwraca **ten sam obiekt** bez mutacji,
     payload routowany dostaje wyłącznie `model`, `agent_mode`, `routing`,
   - zero wyjątków z `apply()` na całym kopusie (fail-open).
4. Rollout: najpierw `deterministic` (bez embeddings), potem `cascade`.
5. Rollback = wyłączenie flagi fazy; żaden etap nie może zmieniać kluczy payloadu
   ani kolejności warstw strukturalnych.

---

## 5. Za kulisami: czego **nie** robić

- Nie rozbudowywać listy fraz jako głównej metody — to jest droga do 627 abstynencji,
  w których zainstalowano już kilka iteracji „poprawek”.
- Nie ruszać warstw strukturalnych (`class`, `collaboration_mode`, `explicit`) —
  pomiary: 100% zgodności z niezależną regułą, 0 błędów.
- Nie wnosić zależnych wagowo modeli ML do ścieżki krytycznej bez bramy replay.
- Nie mieszać „opisu czegoś” z „prośbą o coś” — to najszybciej rosnące źródło błędów
  przy agentach przeklejaszowych (plany, diffy, logi).

---

## 6. Załącznik: odtwarzalność pomiarów

Narzędzia pomiarowe są w repo: `scripts/plugins/auto_codex/eval/`
(`README.md` opisuje cały potok). Uruchamiane z katalogu głównego:

```bash
EVAL=scripts/plugins/auto_codex/eval

# 1) Ekstrakcja payloadów + parowanie z zapisaną decyzją (single-pass po logach)
python3 $EVAL/extract_payloads.py -o /tmp/corpus.jsonl

# 2) Replay bieżącej implementacji + kontrole integralności i determinizmu
python3 $EVAL/replay.py /tmp/corpus.jsonl -o /tmp/decisions.jsonl --determinism
python3 $EVAL/replay.py /tmp/corpus.jsonl --semantic -o /tmp/decisions-sem.jsonl

# 3) Porównanie z decyzjami zapisanymi w logach (diagnostyka przesunięć)
python3 $EVAL/compare_history.py /tmp/decisions.jsonl --recorded --show 5
python3 $EVAL/compare_history.py /tmp/decisions.jsonl --list-unrouted

# 4) Podgląd pojedynczego przypadku: kontekst obok decyzji
python3 $EVAL/inspect_case.py /tmp/corpus.jsonl /tmp/decisions.jsonl --line 104387
python3 $EVAL/inspect_case.py /tmp/corpus.jsonl /tmp/decisions.jsonl --layer phase

# 5) Ewaluacja na korpusie etykietowanym (accuracy, nie „co się przesunęło")
python3 -m llm_router_plugins.utils.routing.agentic_routing.codex.evaluation \
  --dataset tests/data/codex_routing_quality.json --split all --no-semantic

# 6) Testy
python3 -m pytest tests/test_agentic_routing_codex.py tests/test_codex_routing_phase.py \
  tests/test_codex_routing_state.py tests/test_codex_semantic_ranking.py \
  tests/test_codex_memory_configuration.py tests/test_codex_routing_quality.py -q
```

Wyniki odniesienia z bieżącego checkoutu (793 requesty, `semantic.enabled=false`):
729 routowanych, 0 wyjątków, `passthrough` 64/64 bez mutacji i tym samym
obiektem, `+agent_mode`/`+routing` jako jedyne dodane klucze, decyzje identyczne
przy replayu wstecz. Zgodność z zapisem historycznym: 349/727 (48,0%) — zgodnie
z sekcją 1 nie jest to wynik oceny, tylko miara przesunięcia implementacji.

Ograniczenia parowania (opisane w README skryptów): decyzje dobierane są
porządkiem strumienia, więc dwa requesty z obcego modelu (`google/gemma-…`)
zostały sparowane z sąsiednimi decyzjami; `--list-unrouted` pokazuje je osobno,
żeby nie fałszowały liczby zgodności.

Kluczowe lokalizacje kodu:

- `llm_router_plugins/utils/routing/agentic_routing/codex/classifier.py:283` — kaskada
- `llm_router_plugins/utils/routing/agentic_routing/codex/classifier.py:678` — progi heurystyki
- `llm_router_plugins/utils/routing/agentic_routing/codex/phase.py:991` — dowody z aktywności
- `llm_router_plugins/utils/routing/agentic_routing/codex/phase.py:1093` — warunek `test_failure`
- `llm_router_plugins/utils/routing/agentic_routing/codex/payload.py:237` — `intent_text`
- `llm_router_plugins/utils/routing/agentic_routing/codex/scoring.py:103` — dopasowanie sygnałów
- `llm_router_plugins/resources/routing/agentic_routing_codex.json` — konfiguracja i mappiki
