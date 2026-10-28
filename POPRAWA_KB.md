# POPRAWA_KB — plan naprawy warstwy wiedzy (`agentic_knowledge`)

Data przeglądu: 2026-10-25 · Zakres: `llm_router_plugins/utils/knowledge/agentic_knowledge/`,
`llm_router_plugins/cli/plugins/utils/agentic_knowledge/`, polityka
`llm_router_plugins/resources/knowledge/agentic_knowledge.json`.
Metoda: replay 841 nagrań z `agents-conversation/codex/payloads-gold` przez prawdziwy plugin
(fałszywy tylko silnik i strumień) + testy właściwości na kodzie syntetycznym + testy
jednostkowe repozytorium (`593 passed, 1 skipped`).

## Ustalenia, które determinują plan

| miara z korpusu | wartość |
|---|---|
| dowody faz: `command` / `test_failure` / `patch` | 23 854 / 35 / **4** |
| **różne** wywołania, które zakończyły się porażką testu | **1** |
| wywołania z kopertą `*** Begin Patch` przy 61 381 `function_call` | 5 |
| narzędzie `apply_patch` na liście narzędzi tego buildu CLI | brak (edycja przez `exec_command`) |
| zapisane zdarzenia: `recipe` / `playbook` / `deadend` | **0 / 0 / 0** |
| zapisane zdarzenia: `session_archive` / `doc` | 1 877 / 57 |
| `loop_count` maksimum (próg eskalacji 3) | 1 |
| żądania wykonujące zapytania do silnika | 177 / 841 (429 wywołań, 2–3 na żądanie) |
| kwerendy z sygnaturą błędu / ze ścieżkami | 0 / 841 · 4 / 841 |
| tryby zwrócone przez router w 776 decyzjach | `git_review` 332, `implement` 171, `test` 138, `plan` 103, `compaction` 19, `aux_title` 13 — **`review` 0, `debug` 0** |

Wniosek nadrzędny: **baza nie uczy się, bo nie widzi edycji agenta** (nie dlatego, że miner jest
zły), a **odczyt płaci latencją za zbiory, które są puste**. Wszystko poniżej jest podporządkowane
naprawieniu tych dwóch rzeczy, przy zachowaniu dwóch invariantów: warstwa nigdy nie podnosi
żądania i nigdy nie myli repozytoriów.

## Zasady obowiązujące przy wszystkich zmianach

1. Fail-open na ścieżce żądania pozostaje nienaruszalny; nowe ścieżki błędu kończą się adnotacją,
   nie wyjątkiem.
2. Każda zmiana ma test regresji *przed* zmianą kodu (tam, gdzie da się go zrobić bez silnika).
3. Klucze filtrów (`thread_id`, `scope`, `kind`, `mode`, `phase`, `category`, `source_file`,
   `doc_type`) pozostają zapisywane bez maskowania.
4. Zmiany w `phase.py` / `payload.py` dotykają też routingu — wymagają przejścia pełnego
   `pytest -k agentic` (nie tylko `agentic_knowledge`).

---

## ETAP 0 — kliny i utrata danych (P0, jeden dzień, ryzyko niskie)

Te cztery poprawki nie zmieniają zachowania zdrowego ruchu; eliminują stany, w których warstwa
przestaje cokolwiek robić albo kasuje własną wiedzę.

### 0.1 Worker: jedno zdarzenie zawiesza cały strumień

`worker.py:386` — `self._config.collection(event.scope, event.kind)` jest wywołane **poza**
`try`, więc `ValueError` z `config.py:141-143` ucieka ponad pętlą batcha: zdarzenie nigdy nie
trafia do `XACK`, licznik prób nie rośnie i każdy kolejny odczyt przetwarza batch od nowa.

- **Zmiana:** przenieść wyliczenie kolekcji do `try` i potraktować wyjątek jak zwykłe
  `_settle_failure(entry_id, event, exc)`; dodatkowo log z `entry_id` i rodzajem przyczyny
  (bez treści).
- **Test:** zdarzenie z `scope="-"` w strumieniu → batch kończy pracę, zdarzenie po limicie prób
  zostaje odrzucone z logiem, pozostałe indeksują się normalnie.
- **Kryterium:** `XPENDING` nie rośnie monotonicznie w czasie.

### 0.2 CLI dokumentów: `--scope` trafia do silnika bez normalizacji

`docs.py:219` bierze `args.scope` dosłownie; `--scope "-"` albo `" "` produkuje scope, którego
`config.collection()` odmawia (czyli dokładnie sytuację z 0.1).

- **Zmiana:** przepuścić `--scope` przez `scope_key(...)`/ten sam mechanizm co ścieżka proxy i
  przerwać z jasnym komunikatem, gdy wynik jest pusty (komunikat już istnieje dla przypadku
  domyślnego — rozszerzyć na jawne `--scope`).
- **Test:** `--scope "-"`, `--scope " "`, `--scope "../x"` → błąd walidacji przed jakimkolwiek
  zapisem do Redis.

### 0.3 Cofnięcie edycji pliku usuwa jego dokumentację

`docs_ingestor.py:1016-1049`: odcisk dokumentu jest liczony z treści, a `events.py` deduplikuje
po `SET NX <fingerprint> EX 604800`. Sekwencja `v1 → v2 → git checkout` w oknie dedupu publikuje
`docs_prune` na nazwie od `v2`, a publikacja kopii `v1` zostaje **cicho** odrzucona jako duplikat.
Dokument znika z kolekcji i nie wraca, dopóki treść się nie zmieni.

- **Zmiana (dwuczęściowa):**
  1. `docs_prune` publikowany wyłącznie dla plików, których dokument zastępczy faktycznie
     przeszedł (`published`, nie `attempted` — por. `docs_ingestor.py:1040`);
  2. przy publikacji `doc` odświeżać TTL istniejącego claimu zamiast polegać na jego
     nieistnieniu (albo: claim per `(fingerprint, document_name)` z nadpisaniem).
- **Test:** stan `v1` → skan (`v2`) → przywrócenie `v1` → skan; asercja: w kolekcji dokładnie
  jeden aktualny dokument pliku, a `state` wskazuje nazwę, która istnieje.

### 0.4 Scope: nazwy spoza ASCII tracą wiedzę albo mieszają repozytoria

`scope.py:67,303` — normalizacja `[^a-z0-9._-]+` → `-` kasuje znaki nie-ASCII.

Zmierzone: `/home/u/проект`, `/home/u/日本`, `/home/u/-` → **scope pusty** (cała warstwa bezgłośnie
niedziała); `/home/u/repo-ą` i `/home/u/repo-ę` → **to samo kolekcja** `agentic_kb_repo`;
`ścieżki` → `cie-ki`.

- **Zmiana:** transliteracja `unicodedata.normalize("NFKD", …)` + mapowanie znaku na ASCII przed
  usunięciem; gdy po transliteracji wynik jest pusty — deterministyczny skrót od pełnej ścieżki
  (`sha1(path)[:8]`) jako scope, nigdy pusty; kolizja dwóch różnych ścieżek do tego samego klucza
  jest błędem konfiguracyjnym logowanym raz na proces (ewentualnie wymuszającym `scope.overrides`).
- **Test:** tabela `проект`/`日本`/`ścieżki`/`repo-ą`/`repo-ę` — wszystkie dają niepusty klucz,
  różne ścieżki dają różne klucze, a `collection()` nigdy nie rzuca.

---

## ETAP 1 — przywrócenie widzialności edycji (P0, wartość, 2–4 dni)

Bez tego etapu pozostałe poprawki nie mają czego odzyskiwać: przy 4 dowodach patcha na 841 żądań
`recipe`, `playbook` i `deadend` pozostaną puste.

### 1.1 Rejestr dowodów patcha dla edycji przez `exec_command`

Obecnie dowód `patch` dają wyłącznie koperty `*** Begin Patch` (`README`: „a quoted envelope in a
command that applies nothing does not count" — zasada zostaje). W tym buildzie CLI edycje idą
przez `cat > plik <<'EOF'`, `tee`, `sed -i`, `python - <<'EOF' … write_text(...)`.

- **Zmiana:** wprowadzić **dwa poziomy** sygnału edycji:
  - `strong` (obecny) — koperta patcha: pozwala pisać `recipe`, `playbook`, `deadend`;
  - `weak` — strukturalne zapisy pliku rozpoznawane z samej komendy i jej argumentów:
    `cat > path <<EOF`, `… | tee path`, `sed -i`, `open(path, "w")`/`write_text(` w heredoku
    pythona; dowód `weak` pozwala pisać **tylko `playbook`** i nigdy nie domyka weryfikacji
    przepisu.
  Nowe dowody trafiają do `phase.py` (wspólne z routingiem) z osobnym `reason=`, żeby dało się
  liczyć, ile z nich to fałszywe trafienia.
- **Ryzyko i hamulec:** fałszywy patch → fałszywy playbook. Dlatego `weak` nie startuje
  weryfikacji (`_verifies` wymaga zielonego przebiegu po `strong`), a `miner` oznacza dokument
  polem `metadata.edited_by="exec"`; polityka czytająca może go później odrzucić jednym filtrem.
- **Test:** dla 10 nagrań z korpusu, w których agent zapisał plik przez `cat >`, powstaje dowód
  `weak`; dla `cat > notes.md <<EOF` **bez** zmian kodu — nie powstaje `strong`.

### 1.2 Mierzalnik: dlaczego epizod nie powstał

Zanim dołożymy cokolwiek dalej, niech mining mówi, czego zabrakło — dziś „brak zdarzeń" i „brak
dowodów" wyglądają identycznie.

- **Zmiana:** liczniki w adnotacji `routing.knowledge.write`: `no_failure`, `no_patch`,
  `not_verified`, `covered_by_recipe`, `published`.
- **Kryterium:** po 1.1 na korpusie `no_patch` spada z 100% do wartości odzwierciedlającej realne
  braki, a replay syntetyczny (`failure → patch → green`) daje `recipe`.

### 1.3 Nie płakać nad pustym repozytorium

Repozytorium, które nie ma jeszcze żadnego wiedzy wydobytej, nie powinno płacić za jego
przeszukanie.

- **Zmiana:** licznik zdarzeń zapisanych per scope w Redis (albo flaga „kolekcja istnieje");
  dopóki zero — pomijamy szukanie po `recipe`/`playbook`/`deadend`, zostawiając `doc` i
  `session_archive`.
- **Kryterium:** na korpusie liczba wywołań silnika na żądanie spada z 2,4 do ~1,2 bez utraty
  czegokolwiek, co dziś wraca.

---

## ETAP 2 — odczyt: jakość odpowiedzi i koszt (P1, 3–5 dni)

### 2.1 Kwerenda bez tego, co wiadomo

Zmierzone: 0/841 kwerend ma sygnaturę błędu, 4/841 ma ścieżki — mimo że obie rzeczy są policzone
w minerze i w adapterze.

- **Zmiana A:** wynieść sygnaturę z `miner._loop` (wynik miningu już ją zwraca jako
  `signature_hash`) do `AgentTurn`, a `QueryBuilder` ma dokładać do kwerendy `debug` czytelne
  `signature` (nie sam hash) + `test_command`.
- **Zmiana B:** `git_review`/`review`/`test` mają dokładać `touched_paths`; skoro są puste przy
  braku dowodów patcha, po etapie 1.1 pojawią się naturalnie.
- **Test:** na syntetycznym `debug` z znaną sygnaturą kwerenda zawiera jej tekst; asercja
  długości — kwerenda nie przekracza 512 znaków i nie wciąga całego okna.

### 2.2 Skala wyników: przestać mieszać RRF z bonusami

`assembler.py:400` liczy `vote = base * weight + bonus`, przy czym `base` to `score_normalized`,
czyli **przy domyślnej konfiguracji (`hybrid_search: true`, `rerank_results: false`) ranga RRF
o maksimum ~0,033**, a bonusy to 0,20–0,44 (`assembler.py:64-84`). Wykonany test: przy
`score = 0,033` wynik to 0,033, `has_strong_recipe = False`, a maksymalny osiągalny wynik
przepisu to ~0,47 przy progu `strong_recipe_score: 0.75`.

Konsekwencje dziś: (a) o kolejności decyduje waga źródła i trafienie ścieżką, nie podobieństwo;
(b) po włączeniu eskalacji zasada „najpierw daj szansę przepisowi" nigdy nie zadziała.

- **Zmiana:** wprowadź `score_scale` (już zwracany przez silnik, patrzy na to `sse.py`) jako
  jawny argument wejściowy assemblera i **znormalizuj głos w ramach skali**: dla `rrf` przelicz rangę na
  `0…1` (albo użyj kolejności rang i traktuj bonusy jako przesunięcia o ułamek szerokości
  skali), dla `cosine`/`reranker` zostaw obecne progi. Progi konfiguracyjne (`min_similarity`,
  `strong_recipe_score`) muszą być walidowane względem skali — brak zgodności to ostrzeżenie na
  starcie, nie cichy brak trafień.
- **Dodatkowo:** `min_similarity` pozostaje progiem **nogi wektorowej po stronie silnika** (udokumentowane
  w `sse.py`), więc assembler nie może go ponownie nakładać na `score_normalized`; jeśli dziś
  gdzieś nakłada — usunąć.
- **Test:** dwa dokumenty, jeden z bazą 10× lepszą, drugi z trafieniem ścieżką — przy `rrf`
  wygrywa lepszy bazą (bonus nie może przykryć rzędu wielkości); przy włączonym rerankerze próg
  0,75 jest osiągalny i `keep` zamiast `escalate`.

### 2.3 Polityki trybów, których router nie emituje

`review` i `debug` nie pojawiły się ani razu w 776 zagranych decyzjach, a `git_review` (332 żądania)
**nie ma źródła `doc`** w ogóle. Czyli polityka pisana pod przegląd kodu (jedyne `security` +
`contributing`) nigdy nie startuje.

- **Zmiana:** w polityce `resources/knowledge/agentic_knowledge.json`:
  - dodać `doc` (`guide`, `contributing`, `security`) do `git_review`;
  - oznaczyć `review` jako alias `git_review` (albo przenieść jego treść do `git_review` i
    usunąć wpis), żeby nie utrzymywać martwej konfiguracji;
  - przemyśleć `escalation_model_name` dla `debug` w sytuacji, gdy `debug` nie jest emitowany
    (wtedy realną ścieżkę eskalacji ma tylko `test`).
- **Test:** dla każdego trybu, który router jest w stanie zwrócić (wyczerpująca lista z
  `resources/routing/agentic_routing_codex.json`), istnieje polityka z co najmniej jednym źródłem
  nie pustym. Test wylicza tryby z konfiguracji routingu, żeby lista się nie rozjechała ponownie.

### 2.4 Latencja: jeden budżet, mniej wywołań, równoległość

Dziś 2–3 wywołania seryjnie przy `read.timeout_seconds: 6.0` i medianie ~1,7 s na hybrid search
(własny pomiar README) — czyli ogon 4–5 s na 21% żądań.

- **Zmiana:** (a) uruchamiać zapytania do różnych kolekcji równolegle (krótka pula wątków,
  limit = liczba źródeł, bez zmian kontraktu `KnowledgeStore.search`); (b) obniżyć
  `read.timeout_seconds` do ~3,0 i `SSE_API_TIMEOUT` do ~1,5 po pomiarze na prawdziwym silniku;
  (c) kolejność źródeł w trybie tak, aby najtańsze i najbardziej wartościowe szło pierwsze, a
  porzucone przy wyczerpaniu budżetu było najmniej wartościowe (dziś kolejność = kolejność w
  polityce, czyli ofiarą pada najczęściej dokumentacja).
- **Kryterium:** p95 dodatkowego czasu warstwy na żądanie < 400 ms przy włączonym cache i
  < 1,5 s przy pudle; liczba wywołań na żądanie ≤ 1,5.

### 2.5 Cache generacji a zmiana trybu

Klucz unieważnia każda zmiana `mode`, a tryb w trakcie jednej komendy skacze (pudło ~48% żądań,
~60% par w obrębie tej samej generacji).

- **Zmiana:** nie unieważniać przy zmianie trybie, jeśli nie zmienił się fingerprint dowodów;
  przy pudle z powodu trybu — przeliczać tylko sekcje, których tryb dotyczy, zamiast całego
  bloku. Atrybuty rozjechania bajtów pod provider prompt-cache opisane w README mają wtedy
  szansę być spełnione naprawdę.
- **Test:** ta sama generacja, dwa tryby, te same dowody → identyczny blok i zero wywołań.

---

## ETAP 3 — dokumenty: retencja, spójność stanu, szczelność (P2, 2–3 dni)

| # | problem | zmiana |
|---|---|---|
| 3.1 | `docs_ingestor.py:1048` zapisuje stan po skanie uciętym przez `max_files`; `_next_entries` zostawia tylko sczytane pliki, więc dokumenty pozostałych są nieusuwalne | nie zapisywać stanu przy `truncated`, albo zapisywać z flagą „lista niepełna" i przy niej nie prune'ować nigdy |
| 3.2 | retencja sprząta wyłącznie `_sessions` (`worker.py:962-992`); `_docs` nie starzeje się w ogóle | objąć retencją `_docs` (TTL liczony od publikacji dokumentu) albo udokumentować, że `_docs` czyści wyłącznie prune |
| 3.3 | `doc_context.py:208-227`: śledzenie fence trzyma 3 znaki i domyka po `startswith`, więc blok ````` ``` `````zamyka się na przykładzie z ```` ``` ```` i komentarze `# …` stają się „nagłówkami" wstrzykiwanymi bez maskowania | pełne porównanie długości znaku otwierającego; test na przykładach z README w README; asercja, że w dyżescie nie ma linii zaczynającej się od `#` wewnątrz bloku kodu |
| 3.4 | pliki binarne/nie-UTF-8 dekodowane z `errors="replace"` trafiają do indeksu jako mojibake | odmówić publikacji pliku, który nie jest tekstem (stosunek znaków niedrukowalnych) |
| 3.5 | `--force` przy częściowej porażce (część `xadd` nie weszła) kończy się, ale `state` opisuje stan **sprzed** operacji | nie ruszać stanu przy statusie `unavailable`; komunikat ma mówić, ile zdarzeń weszło |
| 3.6 | `docs_ingestor.py:1311` liczy kubełek nazwy po `date.today()`, worker po `created_at` — skan i indeksowanie przez północ zostawiają dokument, którego prune nigdy nie znajdzie | jedna strefa/timestamp: nazwa liczona z `created_at` zdarzenia |
| 3.7 | `--force` znosi wszystkie limity bez zastępczego zabezpieczenia (cały markdown do pamięci i na strumień) | zostawić możliwość, ale wymagać jawnego `--yes-unbounded` przy przekroczeniu np. 20 MB i logować rozmiar |

---

## ETAP 4 — ewaluacja i obserwowalność (P2, 2 dni)

Obecna ewaluacja nie jest w stanie pokazać żadnej z powyższych regresji: `_FakeStore`
(`evaluation.py:113-144`) indeksuje **tylko przepisy**, ignoruje `doc_types`, kolekcje i
`min_similarity`, a `assemble()` wywoływany jest bez `doc_context`, więc `injected_chars_per_turn`
zaniża to, co realnie jedzie do modelu (mapa dokumentów to do 700 znaków na żądanie, domyślnie
włączone).

- **4.1** Fake store: respektuj kolekcję, `doc_types`, `min_similarity` i kolejność rang;
  indeksuj wszystkie rodzaje, nie tylko `recipe`.
- **4.2** `assemble(...)` w replay dostarczaj `doc_context`; mierz `injected_chars_per_turn` z nim
  i bez, jako dwie metryki.
- **4.3** Zestaw syntetyczny (bo korpus prawdziwych przepisów nie zawiera): `failure → patch →
  green`, `failure → patch → failure → patch → green`, `patch → green`, `patch → green → failure`,
  edycja przez `cat >` — z oczekiwanymi rodzajami i odciskami dokumentów. To ten zestaw ma pilnować
  etapu 1.
- **4.4** Metryki produkcyjne do logów (bez treści): `searches_per_request`, `read_budget_skips`,
  `cache_miss_reason` (`mode` / `evidence` / `cold`), `mined_kinds`, `no_patch`,
  `docs_publish_dropped_as_duplicate`, `pending_stream_age`.
- **4.5** Jedna linia w README na każdy zmieniony kontrakt (skala wyników, `--scope`, scope
  nie-ASCII, widzialność edycji).

---

## Kolejność i oszacowanie

| kolejność | etap | dni | ryzyko | co odblokowuje |
|---|---|---|---|---|
| 1 | 0.1–0.4 (kliny, utrata dokumentów, scope) | 1 | niskie | bezpieczeństwo pracy workera i CLI |
| 2 | 1.1–1.3 (widzialność edycji, liczniki, zimne repo) | 2–4 | średnie (fałszywe `weak` patche) | jakikolwiek mined knowledge |
| 3 | 2.2 (skala wyników) | 1–2 | niskie | sensowna kolejność i przyszła eskalacja |
| 4 | 2.1 + 2.3 (kwerenda, polityki trybów) | 1–2 | niskie | trafność |
| 5 | 2.4–2.5 (latencja, cache) | 2 | średnie (równoległość) | obietnica „szybciej" |
| 6 | 3.1–3.7 (dokumenty) | 2–3 | niskie | trwałość wiedzy |
| 7 | 4.1–4.5 (ewaluacja, metryki) | 2 | niskie | obrona przed regresjami |

## Kryteria sukcesu (do zmierzenia replayem + logami, nie deklaracją)

1. `XPENDING` strumienia wiedzy nie rośnie; zdarzenia odrzucone mają limity prób i log.
2. Na korpusie syntetycznym: `recipe > 0`, `playbook > 0`, `deadend > 0`; na korpusie prawdziwym
   `no_patch` < 20% żądań z dowodami poleceń.
3. p95 dodatkowej latencji warstwy < 400 ms (cache trafiony) / < 1,5 s (pudło); ≤ 1,5 wywołania
   silnika na żądanie.
4. Żaden dokument nie znika po `git checkout` pliku markdown; żaden dokument nie zostaje
   osierocony przez ucięty skan.
5. Zero kolizji scope i zero pustych scope'ów dla ścieżek z polskimi i diakrytycznymi nazwami.
6. Po włączeniu `escalation.enabled`: dla przepisu spełniającego warunki decyzja to `keep`, a nie
   `loop_without_recipe` (czyli 2.2 zostało dowiezione).

## Świadomie poza zakresem

- Przebudowa routingu (`agentic_routing_codex`): rejestrowane decyzje nie emitują `review` ani
  `debug`; to osobna decyzja projektowa, tu tylko domykamy polityki wiedzy do realnych trybów.
- Zgoda na „patcha" rozpoznawanego semantycznie z treści polecenia — zostajemy przy
  rozróżnieniu `strong` (koperta) / `weak` (skierowany zapis pliku) i przy tym, że przepis
  domyka wyłącznie zielony przebieg w tym samym zakresie (`_verifies`, sprawdzone: `cd … && pytest`
  celowo **nie** weryfikuje).
- Włączanie eskalacji na produkcji — najwcześniej po 2.2 i po 4.3.
