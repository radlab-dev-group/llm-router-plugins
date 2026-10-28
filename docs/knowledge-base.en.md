# Knowledge base operator guide

How the `agentic_knowledge` layer stores, reads and indexes knowledge, and
what to do when a search comes back empty.

## One collection per repository

Every kind of document of a repository lives in a single collection,
`agentic_kb_<scope>`. The `scope` is the repository's directory name reduced to
the engine's character set (a hyphen becomes an underscore), so
`llm-router-plugins` is `agentic_kb_llm_router_plugins`.

The kinds are separated by their engine **category**, not by the name of the
collection:

| Category          | What it is                                    |
| ----------------- | --------------------------------------------- |
| `recipe`          | a verified fix: failure, change, green run    |
| `deadend`         | a change that did not fix the failure         |
| `playbook`        | a verified procedure, no failure in between   |
| `session_archive` | the window archived around a compaction       |
| `doc`             | the repository's own markdown                 |

A category is a boundary of the same hardness a collection was: the engine
resolves it to a set of document names *before* it scores anything, so a search
that did not ask for a category never sees it, and its `max_results` is spent
within what survived the cut.

`collection_layout` (`single`, the default, or `split`) decides the collection
a reader asks and a worker writes. `split` is the older spelling —
`<prefix>_<scope>` for mined knowledge, `<prefix>_<scope>_sessions` for the
archive, `<prefix>_<scope>_docs` for the documentation — kept so a repository
indexed with it keeps being read. Set
`LLM_ROUTER_KNOWLEDGE_AGENTIC_COLLECTION_LAYOUT` to override the policy.

## How a request is read

A mode asks one question per source group, and the groups are never merged:

- **Mined knowledge** — asked with the mode's own query (the failure signature
  in `debug`), no metadata filter.
- **The archive** — asked with the same query but restricted to the request's
  `thread_id`, so one conversation cannot read another's window.
- **The documentation** — asked with the turn's subject (the user's words and
  the paths it touches, not the failure) and restricted by `doc_type`.

The engine intersects filter families but unions the filters inside the
metadata family, so a request carries exactly one metadata filter. That is why
the archive is never merged with mined knowledge: a `thread_id` on such a
request would cut the mined knowledge to the documents of this one thread,
silently dropping every recipe mined in an earlier session.

The layer is **fail-open**: the engine unreachable, a collection never indexed,
a filter that matched nothing — the request proceeds as if the knowledge base
were empty, and the annotation under `routing.knowledge` (`injected`, `cache`,
`status`) says what happened.

### Which documentation kinds a mode reads

The `doc` source of a mode lists the `doc_types` it reads; an empty list reads
every kind. The shipped policy:

| Mode         | `doc_types`                             |
| ------------ | --------------------------------------- |
| `debug`      | `guide`, `note`                         |
| `test`       | `guide`, `contributing`                 |
| `implement`  | `readme`, `guide`, `contributing`       |
| `plan`       | `readme`, `guide`, `changelog`, `note`  |
| `review`     | `guide`, `contributing`, `security`     |
| `git_review` | `guide`, `contributing`, `security`     |
| `*`          | every kind                              |

A repository that has only a `README` therefore gives `debug` and `test`
nothing to read, while `implement` finds the `readme`. `git_review` reads
the same pages as `review` — it is its alias. That is a policy decision,
not a failure — but it is the first thing to check when a search comes back
empty in one mode and not in another.

## On what scale to read a score

A hybrid search returns a fused (RRF) rank that lives on its own scale —
its ceiling is the best hit actually in hand, not one. Before a hit goes
under a mode's weight or a threshold, the rank is remapped onto 0–1 inside
the result set that produced it, and every hit carries the name of the
scale its score lives on (`score_scale`: `rrf`, `reranker` or `cosine`).
`strong_recipe_score` is a threshold on that remapped scale; `min_similarity`
stays where the engine applies it.

## What a read costs

A mode's sources — mined knowledge, the archive, the documentation — are
read in parallel inside one call under one shared budget, not one after
another: a slow source no longer spends the budget the others were to use.
The shipped `read.timeout_seconds` is 3.0. The annotation under
`routing.knowledge` carries a `metrics` block with how many searches the
read made for the request and how many the budget could not start — the
number a deployment reads to see how much the layer costs the engine per
request.

## How documentation is classified

The file name is the evidence. `README` is a `readme`, `CONTRIBUTING` a
`contributing`, `CHANGELOG` a `changelog`, `SECURITY` a `security`, `LICENSE` a
`license`. Anything else under a top-level `docs/`, `doc/`, `wiki/`,
`documentation/`, `guides/` or `book/` directory is a **`guide`**; anything else
is a `note`.

To make a repository's documentation readable in `debug` and `test`, write it
under `docs/` — a guide about a module outranks the README, and `debug` is the
mode that needs a guide.

## How a document is written

- A mined episode (recipe, dead end, playbook) is published by the proxy to a
  Redis stream while the request happens; `llm-router-agentic-kb-worker` masks
  it and indexes it, creating the repository's collection on first use.
- A `session_archive` is written only when a request classified `compaction`
  goes through the proxy; there is no command that produces one.
- The repository's markdown is published by `llm-router-agentic-kb-docs`:
  unchanged files cost one digest, changed files are republished, and a deleted
  file retires its document through a prune message.

## Commands

`llm-router-agentic-kb-worker` — index what the stream holds.

- `--once` — process one batch and exit (cron-style drain)
- `--no-retain` — skip the periodic retention pass
- `--config PATH` — a policy other than the bundled one

`llm-router-agentic-kb-docs` — publish a repository's documentation.

- `--path DIR` — the repository to index (default `.`)
- `--scope NAME` — the scope to index into (default: what the proxy would
  derive from this directory)
- `--dry-run` — list what would be published, publish nothing
- `--force` — republish every file, even unchanged ones
- `--no-prune` — keep the superseded documents
- `--json` — the outcome as JSON

## Retention

The pass sweeps the repository's collection (and still the legacy
`..._sessions` collections a split deployment left behind), but it deletes only
the document kinds it names. `session_archive` ages on
`write.archive.ttl_seconds`; a README ages when a human deletes the file, and a
recipe retires by being superseded. One collection holds all of these on
different clocks, so the deletion names its kind instead of trusting the day in
a document's name.

## When a search comes back empty

1. **The mode's `doc_types`** exclude what the repository has — see the table
   above. The engine log then says `No documents to search!`, which is not an
   error: it is the category filter resolving to zero documents.
2. **The repository was never indexed** — a young project answers with nothing
   rather than with a failure; the worker creates the collection on its first
   event.
3. **The archive is empty** — nothing is archived until a compaction happens.
4. **The collection was rebuilt** — the scan state and the engine disagree for
   a moment; `llm-router-agentic-kb-docs --force` republishes.

Read the annotation first: `routing.knowledge.status` distinguishes `ok`
(answered, possibly empty) from `disabled` and `unavailable`.
