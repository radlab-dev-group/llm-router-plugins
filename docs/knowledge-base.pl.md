# Przewodnik operatora bazy wiedzy

Jak warstwa `agentic_knowledge` przechowuje, czyta i indeksuje wiedzę oraz co
robić, gdy wyszukiwanie wraca puste.

## Jedna kolekcja na repozytorium

Wszystkie rodzaje dokumentów repozytorium leżą w jednej kolekcji,
`agentic_kb_<scope>`. `scope` to nazwa katalogu repozytorium zredukowana do
alfabetu silnika (myślnik staje się podkreśleniem), więc `llm-router-plugins`
to `agentic_kb_llm_router_plugins`.

Rodzaje są rozdzielane **kategorią** silnika, nie nazwą kolekcji:

| Kategoria           | Co to jest                                       |
| ------------------- | ------------------------------------------------ |
| `recipe`            | zweryfikowana naprawa: błąd, zmiana, zielony test |
| `deadend`           | zmiana, która nie naprawiła błędu                |
| `playbook`          | zweryfikowana procedura, bez błędu po drodze     |
| `session_archive`   | okno zarchiwizowane przy kompaktacji             |
| `doc`               | własne markdowny repozytorium                    |

Kategoria to granica tak samo twarda jak kolekcja: silnik rozwija ją do zbioru
nazw dokumentów *przed* jakimkolwiek scoringiem, więc wyszukiwanie, które nie
poprosiło o kategorię, jej nie widzi, a jego `max_results` jest wydane
wewnątrz tego, co przeżyło wycięcie.

`collection_layout` (`single`, domyślnie, albo `split`) decyduje, do jakiej
kolekcji czyta reader i pisze worker. `split` to starsza notacja —
`<prefix>_<scope>` dla wiedzy wydobytej, `<prefix>_<scope>_sessions` dla
archiwum, `<prefix>_<scope>_docs` dla dokumentacji — zachowana, żeby
repozytorium zaindeksowane w tej notacji nadal było czytane. Nadpisanie
polityki: `LLM_ROUTER_KNOWLEDGE_AGENTIC_COLLECTION_LAYOUT`.

## Jak czytane jest żądanie

Tryb zadaje jedno pytanie na grupę źródeł i grupy nigdy się nie łączą:

- **Wiedza wydobyta** — pytana własnym zapytaniem trybu (sygnatura błędu w
  `debug`), bez filtra metadanych.
- **Archiwum** — pytane tym samym zapytaniem, ale ograniczone do `thread_id`
  żądania, żeby jedna rozmowa nie czytała okna innej.
- **Dokumentacja** — pytana tematem tury (słowa użytkownika i ścieżki, których
  dotyczy, nie błędem) i ograniczona przez `doc_type`.

Silnik przecina rodziny filtrów, ale sumuje filtry wewnątrz rodziny
metadanych, więc żądanie niesie dokładnie jeden filtr metadanych. Dlatego
archiwum nigdy nie jest łączone z wiedzą wydobytą: `thread_id` przy takim
żądaniu wyciąłoby wiedzę do dokumentów tego jednego wątku, cicho pomijając
każdy przepis wydobyty w poprzedniej sesji.

Warstwa jest **fail-open**: nieosiągalny silnik, kolekcja nigdy
nieindeksowana, filtr bez trafień — żądanie idzie dalej tak, jakby baza wiedzy
była pusta, a adnotacja pod `routing.knowledge` (`injected`, `cache`, `status`)
mówi, co się stało.

### Jakie rodzaje dokumentacji czyta dany tryb

Źródło `doc` trybu wypisuje `doc_types`, które czyta; pusta lista czyta każdy
rodzaj. Domyślna polityka:

| Tryb          | `doc_types`                                |
| ------------- | ------------------------------------------ |
| `debug`       | `guide`, `note`                            |
| `test`        | `guide`, `contributing`                    |
| `implement`   | `readme`, `guide`, `contributing`          |
| `plan`        | `readme`, `guide`, `changelog`, `note`     |
| `review`      | `guide`, `contributing`, `security`        |
| `git_review`  | (bez źródła dokumentacji)                  |
| `*`           | każdy rodzaj                               |

Repozytorium z samym `README` nie da więc `debug` i `test` nic do
przeczytania, za to `implement` znajdzie `readme`. To decyzja polityki, nie
usterka — ale to pierwsza rzecz do sprawdzenia, gdy wyszukiwanie wraca puste w
jednym trybie, a w innym nie.

## Jak klasyfikowana jest dokumentacja

Dowodem jest nazwa pliku. `README` to `readme`, `CONTRIBUTING` to
`contributing`, `CHANGELOG` to `changelog`, `SECURITY` to `security`, `LICENSE`
to `license`. Wszystko inne w najwyższym katalogu `docs/`, `doc/`, `wiki/`,
`documentation/`, `guides/` albo `book/` to **`guide`**; wszystko inne to
`note`.

Żeby dokumentacja repozytorium była czytelna w `debug` i `test`, pisz ją pod
`docs/` — przewodnik o module wygrywa z README, a właśnie `debug` potrzebuje
przewodnika.

## Jak zapisywane są dokumenty

- Odcinek wydobyty (przepis, ślepy zaułek, procedura) jest publikowany przez
  proxy na strumień Redisa w trakcie żądania;
  `llm-router-agentic-kb-worker` go maskuje i indeksuje, tworząc kolekcję
  repozytorium przy pierwszym użyciu.
- `session_archive` zapisywane jest tylko wtedy, gdy przez proxy przechodzi
  żądanie sklasyfikowane jako `compaction`; nie ma komendy, która by je
  wygenerowała.
- Markdowny repozytorium publikuje `llm-router-agentic-kb-docs`: pliki
  niezmienione kosztują jeden digest, zmieniane są republikowane, a usunięty
  plik wycofuje swój dokument komunikatem prune.

## Polecenia

`llm-router-agentic-kb-worker` — indeksuje to, co jest na strumieniu.

- `--once` — przetwarza jedną partię i wychodzi (styl cron)
- `--no-retain` — pomija cykliczną retencję
- `--config PATH` — inna polityka niż wbudowana

`llm-router-agentic-kb-docs` — publikuje dokumentację repozytorium.

- `--path DIR` — repozytorium do zaindeksowania (domyślnie `.`)
- `--scope NAME` — zakres (scope), do którego indeksować (domyślnie: to, co
  proxy wyprowadziłoby z tego katalogu)
- `--dry-run` — wypisuje, co zostałoby opublikowane, nic nie publikuje
- `--force` — republikuje każdy plik, nawet niezmienione
- `--no-prune` — zachowuje wycofane dokumenty
- `--json` — wynik jako JSON

## Retencja

Przebieg zamiata kolekcję repozytorium (i nadal stare kolekcje `..._sessions`,
które zostawił deployment w notacji `split`), ale usuwa wyłącznie te rodzaje
dokumentów, które sam nazwie. `session_archive` starzeje się po
`write.archive.ttl_seconds`; README starzeje się, gdy człowiek usunie plik, a
przepis wycofuje się przez zastąpienie. Jedna kolekcja trzyma wszystkie te
rodzaje na różnych zegarach, więc usunięcie nazywa swoją kategorię zamiast
ufać dniu wpisanemu w nazwę dokumentu.

## Gdy wyszukiwanie wraca puste

1. **`doc_types` trybu** wykluczają to, co ma repozytorium — patrz tabela
   powyżej. W logu silnika pojawia się wtedy `No documents to search!`, co nie
   jest błędem: to filtr kategorii rozwinięty do zera dokumentów.
2. **Repozytorium nigdy nie było zaindeksowane** — młody projekt odpowiada
   pustką zamiast błędem; worker tworzy kolekcję przy pierwszym evencie.
3. **Archiwum jest puste** — nic nie jest archiwizowane, póki nie dojdzie do
   kompaktacji.
4. **Kolekcja została przebudowana** — stan skanu i silnik chwilowo się nie
   zgadzają; `llm-router-agentic-kb-docs --force` republikuje.

Najpierw czytaj adnotację: `routing.knowledge.status` odróżnia `ok`
(odpowiedziano, choćby pustką) od `disabled` i `unavailable`.
