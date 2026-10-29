"""
Configuration of the agentic knowledge layer.

Two kinds of setting live in two places, on purpose — the same split the Codex
session memory uses:

* the **policy** is JSON: which adapters are served, which sources each work
  mode reads and at what weight, how many characters a mode may spend on
  knowledge, which model escalation switches to, how long anything is kept.
  This is reviewable, versionable, and safe to put in a repository;
* every **connection** comes from the environment under
  ``LLM_ROUTER_KNOWLEDGE_AGENTIC_``: the Semantic Search Engine host and token,
  the Redis host and credentials.  A config file that names a host is a secret
  waiting to be committed.

Precedence is environment, then JSON, then the defaults below.  A value that
is present but malformed is an error rather than a silent default, except for
the connection settings, where a missing host simply means the feature is not
available and the plugin passes requests through.

JSON structure::

    {
      "adapters": ["codex", "claude_code"],
      "collection_prefix": "agentic_kb",
      "collection_layout": "single",
      "scope": {"strategy": "repo_basename", "overrides": {}, "default": ""},
      "read": {"enabled": true, "timeout_seconds": 1.5, "min_similarity": 0.35,
               "hybrid_search": true, "rerank_results": false,
               "cache_ttl_seconds": 3600, "max_lookups_per_generation": 20},
      "modes": {
        "debug": {"max_chars": 6000, "escalation_model_name": "…", "sources": [
          {"category": "recipe", "weight": 1.0, "max_items": 3}]},
        …
      },
      "claude_code": {"mode_map": {}},
      "write": {"enabled": true, "stream_key": "agentic_kb:events",
                "max_stream_len": 100000, "dedup_ttl_seconds": 604800,
                "max_patch_chars": 4000,
                "archive": {"enabled": true, "max_chunk_chars": 2000,
                            "ttl_seconds": 1209600}},
      "escalation": {"enabled": false, "loop_threshold": 3,
                     "strong_recipe_score": 0.75},
      "docs": {"enabled": true, "auto_index": true,
               "patterns": ["*.md"], "excluded_dirs": ["node_modules"],
               "max_files": 200, "max_file_bytes": 200000,
               "max_total_bytes": 2000000,
               "rescan_interval_seconds": 21600,
               "state_ttl_seconds": 7776000}
    }
"""

import json
import os
import pathlib
import re

from dataclasses import dataclass, field, replace
from typing import Any, Callable, ClassVar, Dict, Mapping, Optional, Tuple

from llm_router_plugins.utils.knowledge.agentic_knowledge.scope import (
    SCOPE_STRATEGIES,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.state import (
    RedisConnectionSettings,
    connection_from_env,
    validate_connection,
)

__all__ = [
    "AGENTIC_KNOWLEDGE_ENV_PREFIX",
    "AgenticKnowledgeConfig",
    "ModePolicy",
    "SourcePolicy",
    "WILDCARD_MODE",
    "WorkerSettings",
    "is_collection_of",
]

#: Mode key of the policy that serves every mode without one of its own.
#: Routing falls back to ``implement`` rather than to nothing, so a request
#: reaches this plugin with an empty mode only when routing did not run at
#: all — a mode this policy cannot name, and a reason not to skip the search.
WILDCARD_MODE = "*"

#: Filename patterns the documentation scan picks up by default.  Markdown is
#: the lingua franca of a repository's own documentation; an .rst tree is worth
#: adding by an operator who has one.
DEFAULT_DOC_PATTERNS: Tuple[str, ...] = ("*.md", "*.markdown", "*.mdx")

#: Directories a documentation scan never descends into.  Everything here is
#: either a dependency's documentation (retrieved as the team's own, which it
#: is not), a build artifact, or a tool's cache — noise in all three cases, and
#: in the first two a great deal of it.
DEFAULT_DOC_EXCLUDED_DIRS: Tuple[str, ...] = (
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    ".env",
    ".idea",
    ".vscode",
    "node_modules",
    "bower_components",
    "venv",
    "env",
    "dist",
    "build",
    "target",
    "site-packages",
    "__pycache__",
    "vendor",
    "coverage",
    "htmlcov",
    "site",
)

#: Prefix of every environment variable this plugin reads.
AGENTIC_KNOWLEDGE_ENV_PREFIX = "LLM_ROUTER_KNOWLEDGE_AGENTIC_"

#: Empty means "ask the client library", which resolves its own default
#: (``api``, or ``SSE_API_PREFIX``) — see :meth:`KnowledgeStore._ensure_client`.
#: A guessed ``/api/v1`` here would 404 against an unversioned deployment and
#: fail open into a silently knowledge-less proxy.
_DEFAULT_SSE_PREFIX = ""
_DEFAULT_STREAM_KEY = "agentic_kb:events"
_DEFAULT_KEY_PREFIX = "agentic-kb"


def _engine_collection_fragment(value: str, label: str) -> str:
    """
    Reduce *value* to what the engine accepts in a collection name.

    The engine refuses to create a collection whose name is not
    identifier-like: a hyphen, or a name starting with a digit, is a 500 at
    creation time.  This is the one spelling the plugin creates, queries
    and prunes with, so it must survive all three.
    """
    text = str(value).strip().strip("-_")
    if not text:
        raise ValueError(f"{label} must not be empty")
    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", text)
    text = re.sub(r"\.(?!\w)", "_", text)
    text = text.replace("-", "_").strip("._")
    text = re.sub(r"^[0-9]+", "", text)
    return text or "_"


_DEFAULT_COLLECTION_PREFIX = _engine_collection_fragment(
    "agentic-kb", "collection_prefix"
)

#: One collection per repository: every document kind lives inside it, told
#: apart by its engine ``category``.  A category is as hard a boundary as a
#: collection of its own was — the engine resolves it to document names
#: *before* the vector leg runs (``semantic.py``: the category becomes a list
#: of names handed to ``search_in_documents``), so ``max_results`` is spent
#: within what survived the cut and a README cannot push a recipe out of a
#: search that did not ask for a README.
COLLECTION_LAYOUT_SINGLE = "single"

#: The older layout: mined knowledge, session archives and documentation in
#: three collections per repository — ``<base>``, ``<base>_sessions`` and
#: ``<base>_docs``.  Kept so a deployment can be read by the code that wrote
#: it, and so a layout change stays reversible without re-indexing.
COLLECTION_LAYOUT_SPLIT = "split"

#: Layouts an operator may name, in ``collection_layout`` and in the env.
COLLECTION_LAYOUTS: Tuple[str, ...] = (
    COLLECTION_LAYOUT_SINGLE,
    COLLECTION_LAYOUT_SPLIT,
)

_DEFAULT_COLLECTION_LAYOUT = COLLECTION_LAYOUT_SINGLE

#: Suffix of the archive collection of the split layout.  Retention used to
#: find archives by it; under one collection it finds them by document kind,
#: and this stays only to recognise what the split layout left behind.
_SESSIONS_SUFFIX = "sessions"

#: Suffix of the documentation collection of the split layout.
_DOCS_SUFFIX = "docs"


def is_collection_of(
    name: str, prefix: str, layout: str = COLLECTION_LAYOUT_SPLIT
) -> bool:
    """
    Whether a collection *name* belongs to this plugin under *prefix*.

    The split layout gives every repository three names — the bare one plus
    the ``_sessions`` and ``_docs`` companions — and the single layout gives it
    one.  The caller asks one question ("is this mine, and of what kind?") and
    gets an answer that is true of either spelling: a hyphenated name from a
    release before the engine's character set was enforced is still this
    plugin's name, and a ``_sessions`` collection left behind by a layout
    change is still this plugin's, so it can be swept.

    Returns
    -------
    str or None
        The kind of the collection — ``"repository"``, ``"sessions"`` or
        ``"docs"`` — or ``None`` when the name is not this plugin's at all.
    """
    for separator in ("_", "-"):
        if not name.startswith(f"{prefix}{separator}"):
            continue
        rest = name[len(prefix) + 1 :]
        if rest.endswith(f"_{_SESSIONS_SUFFIX}") or rest.endswith(
            f"-{_SESSIONS_SUFFIX}"
        ):
            return "sessions"
        if rest.endswith(f"_{_DOCS_SUFFIX}") or rest.endswith(f"-{_DOCS_SUFFIX}"):
            return "docs"
        return "repository"
    return None


def _collection_layout(value: Any, label: str) -> str:
    """Read one of :data:`COLLECTION_LAYOUTS`; anything else is a typo."""
    text = str(value or "").strip().lower()
    if text not in COLLECTION_LAYOUTS:
        raise ValueError(
            f"{label} must be one of {list(COLLECTION_LAYOUTS)}, got {value!r}"
        )
    return text


#: Ceiling of one read call, so a typo cannot stall a request.
_MAX_SSE_TIMEOUT_SECONDS = 10.0
#: Ceiling of one call to the engine.  A hybrid search of a real collection
#: measured a median of 1.7 s (engine and plugin on separate hosts), so a call
#: allowed 1.5 s loses most of the searches it makes to its own timeout; 2 s is
#: the shortest number that answers more often than it times out.
_DEFAULT_SSE_TIMEOUT_SECONDS = 2.0
#: Budget of the read path: what one request may spend on the engine in total.
#: A mode asks the engine several questions (one per collection), and they are
#: asked one after another, so the request's ceiling has to be a sum — two
#: calls of :data:`_DEFAULT_SSE_TIMEOUT_SECONDS` is what the common mode does.
_DEFAULT_READ_TIMEOUT_SECONDS = 4.0
#: Room the budget keeps over a single call.  A search starts only while a
#: whole call still fits before the deadline, and the deadline is read from
#: the clock a moment before the call is — so a budget *equal* to one call is
#: spent by the microseconds between those two readings, and every search of
#: every request comes back "not started".  This is the difference between a
#: deployment that reads its knowledge base and one that only warns about it.
_MIN_READ_MARGIN_SECONDS = 0.25
_DEFAULT_MIN_SIMILARITY = 0.35

#: Both legs of the engine fused by Reciprocal Rank Fusion is what its
#: collections are built for, so it is the default.  The cross-encoder reranker
#: is not: it is a second model call over every retrieved chunk, and an engine
#: without a reranker bound to the collection answers nothing with one on.
_DEFAULT_HYBRID_SEARCH = True
_DEFAULT_RERANK_RESULTS = False
_DEFAULT_CACHE_TTL_SECONDS = 3600
_DEFAULT_MAX_LOOKUPS = 20
_DEFAULT_MAX_CHARS = 6000
_DEFAULT_MAX_STREAM_LEN = 100_000
_DEFAULT_DEDUP_TTL_SECONDS = 604_800
_DEFAULT_MAX_PATCH_CHARS = 4_000
_DEFAULT_ARCHIVE_CHUNK_CHARS = 2_000
_DEFAULT_ARCHIVE_TTL_SECONDS = 1_209_600
_DEFAULT_LOOP_THRESHOLD = 3
_DEFAULT_STRONG_RECIPE_SCORE = 0.75
_DEFAULT_RETENTION_INTERVAL_SECONDS = 21_600
_DEFAULT_MAX_PENDING = 1_000
_DEFAULT_WORKER_BATCH = 32

#: Chunking of a collection: declared when the worker **creates** one and
#: repeated on every indexing call, because the engine only splits a document
#: the way the call says, and a document indexed with different numbers than
#: the collection declares is a document split unlike its neighbours.  The
#: engine cuts the *embedded* text at 508 tokens however a collection is
#: declared, so a chunk of 500 is truncated by the embedder while one of 400 is
#: not: at 400 tokens with a 60-token overlap a chunk and its neighbour still
#: share a sentence or two, which is what makes a recipe's failure and its fix
#: retrieve together.  ``0`` means "say nothing" at creation and at indexing
#: alike, which leaves the engine's own default on both sides.
_DEFAULT_CHUNK_TOKENS = 400
_DEFAULT_CHUNK_OVERLAP = 60
_DEFAULT_WORKER_BLOCK_MS = 5_000

#: Ceilings of the documentation scan.  They are budgets, not expectations: a
#: scan that runs out of one stops and says so, because pruning against a
#: partial listing would delete the documentation the scan never reached.
_DEFAULT_DOC_MAX_FILES = 200
_DEFAULT_DOC_MAX_FILE_BYTES = 200_000
_DEFAULT_DOC_MAX_TOTAL_BYTES = 2_000_000

#: How often one process looks at the repository's documentation again.
_DEFAULT_DOC_RESCAN_INTERVAL_SECONDS = 21_600

#: Lifetime of a scope's scan state.  It has to outlive a redeploy — losing it
#: means re-publishing every document — while a scope that has gone quiet should
#: not hold its listing in Redis forever.
_DEFAULT_DOC_STATE_TTL_SECONDS = 7_776_000

#: Budget of the stable documentation digest the session carries.  It is always
#: present, so it must be short enough to pay for itself on every request of a
#: long session; the deep chunks are what a matching query buys.
_DEFAULT_DOC_CONTEXT_ENABLED = True
_DEFAULT_DOC_CONTEXT_CHARS = 700
_DEFAULT_DOC_CONTEXT_TTL_SECONDS = 86_400

_INDEX_TYPES = ("HNSW", "IVF_FLAT")


@dataclass(frozen=True)
class SourcePolicy:
    """
    One retrieval source of a work mode.

    Parameters
    ----------
    category : str
        Document category to ask the search for, e.g. ``recipe``.
    weight : float
        Multiplier of a hit's normalized score when the sources vote.
    max_items : int
        Most hits this source may contribute, whatever its score.
    phases : Tuple[str, ...]
        Work phases this source is read in; empty — the default — reads it in
        every phase.  A phase is the work the episode was doing (`test`,
        `implement`), the document kind is what it produced (`recipe`).
    doc_types : Tuple[str, ...]
        Documentation kinds this source reads; empty — the default — reads
        every kind.  A second metadata dimension for the ``doc`` category,
        which otherwise would have none: the engine unions several
        ``metadata_filters`` with each other, so the list is the one that
        is put in the slot.  Allowed on the ``doc`` source only, because
        it is the only category whose documents carry a ``doc_type``.
    modes : Tuple[str, ...]
        Routing modes this source is read in; empty — the default — reads it in
        every mode.  Mutually exclusive with *phases* for sources retrieved by
        one request: the engine offers a single metadata slot and unions
        whatever is put in it, so a second dimension would widen the result
        instead of narrowing it.
    """

    category: str
    weight: float = 1.0
    max_items: int = 3
    phases: Tuple[str, ...] = ()
    modes: Tuple[str, ...] = ()
    doc_types: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ModePolicy:
    """
    What one work mode reads and what it may escalate to.

    Parameters
    ----------
    name : str
        Work mode this policy belongs to.
    max_chars : int
        Hard budget of injected characters for the mode.
    sources : Tuple[SourcePolicy, ...]
        Sources the mode reads, in the order they are cited.
    escalation_model_name : str
        Model escalation switches to when the mode is looping with no known
        fix; ``""`` means the mode never escalates.
    cheap_model_name : str
        Model to prefer when a strong recipe was found; ``""`` leaves routing
        alone, which is the default.
    """

    name: str
    max_chars: int = _DEFAULT_MAX_CHARS
    sources: Tuple[SourcePolicy, ...] = ()
    escalation_model_name: str = ""
    cheap_model_name: str = ""

    def policy_signature(self) -> str:
        """
        What a switch to another mode would actually change.

        The identity of the questions this mode asks of the knowledge base:
        the categories and their weights and caps, the character budget, and
        the filters that shape the requests.  Two modes that read the same
        sources with the same budget answer the same question, and a switch
        between them changes the *label* of the answer, not the answer —
        which is what the generation cache may keep serving.
        """
        import hashlib
        import json

        parts = [
            self.max_chars,
            [
                [
                    source.category,
                    float(source.weight),
                    int(source.max_items),
                    list(source.phases),
                    list(source.modes),
                    list(source.doc_types),
                ]
                for source in self.sources
            ],
        ]
        digest = hashlib.sha256(
            json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return digest[:16]


@dataclass(frozen=True)
class WorkerSettings:
    """
    Settings of the indexing worker that are not policy.

    Parameters
    ----------
    group : str
        Consumer group the worker reads under.
    consumer : str
        Name of this consumer inside the group.
    batch : int
        Events claimed per read.
    block_ms : int
        How long a read waits for work before returning empty.
    embedder : str
        Embedder name the SSE backend serves; required to create a collection.
    reranker : str
        Reranker name the SSE backend serves.
    index_type : str
        Vector index type for a new collection.
    chunk_tokens : int
        Chunk size, in tokens: declared when a collection is created and sent
        again on every indexing call, so a document is split the way the
        collection says it is split.  Changing it never re-splits a collection
        that already exists; ``0`` declares nothing on either side and leaves
        the engine's own default.
    chunk_overlap : int
        Chunk overlap, in tokens, sent with the same two-sided rule and always
        smaller than *chunk_tokens*.
    retention_interval_seconds : int
        Seconds between retention passes; ``0`` disables them.
    max_pending : int
        Pending-entry list length above which the worker warns.
    """

    group: str = "agentic-kb"
    consumer: str = ""
    batch: int = _DEFAULT_WORKER_BATCH
    block_ms: int = _DEFAULT_WORKER_BLOCK_MS
    embedder: str = ""
    reranker: str = ""
    index_type: str = "HNSW"
    chunk_tokens: int = _DEFAULT_CHUNK_TOKENS
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP
    retention_interval_seconds: int = _DEFAULT_RETENTION_INTERVAL_SECONDS
    max_pending: int = _DEFAULT_MAX_PENDING


@dataclass
class AgenticKnowledgeConfig:
    """
    Resolved configuration of one plugin instance.

    Mutable so :meth:`override_from_env` can apply the environment in place,
    exactly as the routing configs do; read it only after loading.
    """

    _ENV_PREFIX: ClassVar[str] = AGENTIC_KNOWLEDGE_ENV_PREFIX

    enabled: bool = True
    adapters: Tuple[str, ...] = ("codex",)
    collection_prefix: str = _DEFAULT_COLLECTION_PREFIX
    #: How a repository's documents are spread over collections; see
    #: :data:`COLLECTION_LAYOUT_SINGLE` and :data:`COLLECTION_LAYOUT_SPLIT`.
    #: It is a property of the **data at rest**, not of a request: the layout
    #: the worker indexed with is the layout the reader must ask with, so a
    #: deployment switching it switches both sides at once and re-indexes.
    collection_layout: str = _DEFAULT_COLLECTION_LAYOUT
    key_prefix: str = _DEFAULT_KEY_PREFIX

    scope_strategy: str = "repo_basename"
    scope_overrides: Dict[str, str] = field(default_factory=dict)
    scope_default: str = ""

    read_enabled: bool = True
    #: Budget of the read path: how long the knowledge layer may spend on the
    #: engine for **one request in total**.  A mode asks one question per
    #: collection and the questions are asked one after another, so this is a
    #: sum and :attr:`sse_timeout_seconds` is the share one of them may take:
    #: a search is not started unless a whole call still fits before the budget
    #: runs out.  The number the operator wrote is therefore the ceiling a
    #: request actually waits, not a per-call figure multiplied by the number
    #: of sources the mode reads.
    read_timeout_seconds: float = _DEFAULT_READ_TIMEOUT_SECONDS
    min_similarity: float = _DEFAULT_MIN_SIMILARITY
    #: Fuse the vector leg with the engine's full-text leg.
    hybrid_search: bool = _DEFAULT_HYBRID_SEARCH
    #: Re-score what the engine retrieved with its cross-encoder.
    rerank_results: bool = _DEFAULT_RERANK_RESULTS
    cache_ttl_seconds: int = _DEFAULT_CACHE_TTL_SECONDS
    max_lookups_per_generation: int = _DEFAULT_MAX_LOOKUPS

    write_enabled: bool = True
    stream_key: str = _DEFAULT_STREAM_KEY
    max_stream_len: int = _DEFAULT_MAX_STREAM_LEN
    dedup_ttl_seconds: int = _DEFAULT_DEDUP_TTL_SECONDS
    max_patch_chars: int = _DEFAULT_MAX_PATCH_CHARS

    archive_enabled: bool = True
    archive_max_chunk_chars: int = _DEFAULT_ARCHIVE_CHUNK_CHARS
    archive_ttl_seconds: int = _DEFAULT_ARCHIVE_TTL_SECONDS

    #: Whether the repository's own documentation belongs to the knowledge base.
    docs_enabled: bool = True
    #: Whether a request may trigger a documentation scan.  Off means the
    #: documentation is indexed by hand, with the operator's own command.
    docs_auto_index: bool = True
    #: Filename patterns a scan picks up.
    docs_patterns: Tuple[str, ...] = DEFAULT_DOC_PATTERNS
    #: Directory names a scan does not descend into.
    docs_excluded_dirs: Tuple[str, ...] = DEFAULT_DOC_EXCLUDED_DIRS
    docs_max_files: int = _DEFAULT_DOC_MAX_FILES
    docs_max_file_bytes: int = _DEFAULT_DOC_MAX_FILE_BYTES
    docs_max_total_bytes: int = _DEFAULT_DOC_MAX_TOTAL_BYTES
    docs_rescan_interval_seconds: int = _DEFAULT_DOC_RESCAN_INTERVAL_SECONDS
    docs_state_ttl_seconds: int = _DEFAULT_DOC_STATE_TTL_SECONDS
    #: Whether the session carries a stable digest of the repository's own
    #: documentation, injected on every request of the session.
    doc_context_enabled: bool = _DEFAULT_DOC_CONTEXT_ENABLED
    #: Most characters the digest may occupy; the README's head and the
    #: repository's top-level headings, trimmed to this.
    doc_context_max_chars: int = _DEFAULT_DOC_CONTEXT_CHARS
    #: How long a session's digest survives; a session, not a generation.
    doc_context_ttl_seconds: int = _DEFAULT_DOC_CONTEXT_TTL_SECONDS

    escalation_enabled: bool = False
    loop_threshold: int = _DEFAULT_LOOP_THRESHOLD
    strong_recipe_score: float = _DEFAULT_STRONG_RECIPE_SCORE

    claude_code_mode_map: Dict[str, str] = field(default_factory=dict)
    modes: Dict[str, ModePolicy] = field(default_factory=dict)

    sse_host: str = ""
    sse_prefix: str = _DEFAULT_SSE_PREFIX
    sse_token: str = ""
    #: Username and password of an engine that authenticates by login rather
    #: than by a pre-issued token; the client then logs in on its first call.
    #: An empty password is no credential, and a token, when set, wins.
    sse_username: str = ""
    sse_password: str = ""
    #: Ceiling of one call to the engine; always within the read budget.
    sse_timeout_seconds: float = _DEFAULT_SSE_TIMEOUT_SECONDS
    sse_verify: bool = True
    #: Ask the engine not to detect the language of the query.  The engine uses
    #: what it detects twice: as a metadata filter on the vector leg, and as
    #: the name of the PostgreSQL full-text configuration.  A knowledge base is
    #: indexed in the language of the code and asked in the language of the
    #: operator, so the first filters the answers away and the second asks the
    #: cluster for a configuration it may not have — which is a 500, not an
    #: empty result.
    sse_ignore_language_detect: bool = True

    redis: RedisConnectionSettings = field(default_factory=RedisConnectionSettings)
    worker: WorkerSettings = field(default_factory=WorkerSettings)
    #: Why the Redis connection was discarded, when :meth:`override_from_env`
    #: dropped a malformed one; empty when the settings were accepted.
    redis_error: str = ""

    # ---------------------------------------------------------------- load ---
    @classmethod
    def from_file(
        cls, path: Optional[pathlib.Path] = None
    ) -> "AgenticKnowledgeConfig":
        """
        Load the policy from the ``…_CONFIG`` env var, a path, or the bundle.

        Parameters
        ----------
        path : pathlib.Path, optional
            Config file to read when the env var is unset.

        Returns
        -------
        AgenticKnowledgeConfig
            The parsed policy, with connection defaults from the environment.

        Raises
        ------
        FileNotFoundError
            If the named file does not exist.
        ValueError
            If the policy is inconsistent.
        """
        raw = _read_raw(path, cls._ENV_PREFIX)
        return cls._from_raw(raw)

    @classmethod
    def from_json(cls, raw_json: str) -> "AgenticKnowledgeConfig":
        """
        Load the policy from a JSON string.

        Parameters
        ----------
        raw_json : str
            A JSON object with the documented shape.

        Returns
        -------
        AgenticKnowledgeConfig
            The parsed policy.

        Raises
        ------
        ValueError
            If the policy is inconsistent.
        """
        return cls._from_raw(json.loads(raw_json or "{}"))

    @classmethod
    def _from_raw(cls, raw: Mapping[str, Any]) -> "AgenticKnowledgeConfig":
        """Validate one decoded policy document into a config."""
        data = dict(raw or {})
        unknown = set(data) - _POLICY_KEYS
        if unknown:
            raise ValueError(f"Unknown knowledge config keys: {sorted(unknown)}")

        read = _section(data, "read")
        write = _section(data, "write")
        archive = _section(write, "archive")
        docs = _section(data, "docs")
        escalation = _section(data, "escalation")
        scope = _section(data, "scope")
        claude = _section(data, "claude_code")

        strategy = str(scope.get("strategy", "repo_basename"))
        if strategy not in SCOPE_STRATEGIES:
            raise ValueError(
                f"scope.strategy must be one of {list(SCOPE_STRATEGIES)}, "
                f"got {strategy!r}"
            )
        config = cls(
            enabled=_boolean(data.get("enabled", True), "enabled"),
            adapters=tuple(
                _string_list(data.get("adapters", ["codex"]), "adapters")
            ),
            collection_prefix=_safe(
                str(data.get("collection_prefix", _DEFAULT_COLLECTION_PREFIX)),
                "collection_prefix",
            ),
            collection_layout=_collection_layout(
                data.get("collection_layout", _DEFAULT_COLLECTION_LAYOUT),
                "collection_layout",
            ),
            scope_strategy=strategy,
            scope_overrides=_overrides(scope.get("overrides", {})),
            scope_default=str(scope.get("default", "") or ""),
            read_enabled=_boolean(read.get("enabled", True), "read.enabled"),
            read_timeout_seconds=_positive_float(
                read.get("timeout_seconds", _DEFAULT_READ_TIMEOUT_SECONDS),
                "read.timeout_seconds",
            ),
            min_similarity=_fraction(
                read.get("min_similarity", _DEFAULT_MIN_SIMILARITY),
                "read.min_similarity",
            ),
            hybrid_search=_boolean(
                read.get("hybrid_search", _DEFAULT_HYBRID_SEARCH),
                "read.hybrid_search",
            ),
            rerank_results=_boolean(
                read.get("rerank_results", _DEFAULT_RERANK_RESULTS),
                "read.rerank_results",
            ),
            cache_ttl_seconds=_positive(
                read.get("cache_ttl_seconds", _DEFAULT_CACHE_TTL_SECONDS),
                "read.cache_ttl_seconds",
            ),
            max_lookups_per_generation=_positive(
                read.get("max_lookups_per_generation", _DEFAULT_MAX_LOOKUPS),
                "read.max_lookups_per_generation",
            ),
            write_enabled=_boolean(write.get("enabled", True), "write.enabled"),
            stream_key=str(write.get("stream_key", _DEFAULT_STREAM_KEY))
            or _DEFAULT_STREAM_KEY,
            max_stream_len=_positive(
                write.get("max_stream_len", _DEFAULT_MAX_STREAM_LEN),
                "write.max_stream_len",
            ),
            dedup_ttl_seconds=_positive(
                write.get("dedup_ttl_seconds", _DEFAULT_DEDUP_TTL_SECONDS),
                "write.dedup_ttl_seconds",
            ),
            max_patch_chars=_positive(
                write.get("max_patch_chars", _DEFAULT_MAX_PATCH_CHARS),
                "write.max_patch_chars",
            ),
            archive_enabled=_boolean(
                archive.get("enabled", True), "archive.enabled"
            ),
            archive_max_chunk_chars=_positive(
                archive.get("max_chunk_chars", _DEFAULT_ARCHIVE_CHUNK_CHARS),
                "write.archive.max_chunk_chars",
            ),
            archive_ttl_seconds=_positive(
                archive.get("ttl_seconds", _DEFAULT_ARCHIVE_TTL_SECONDS),
                "write.archive.ttl_seconds",
            ),
            docs_enabled=_boolean(docs.get("enabled", True), "docs.enabled"),
            docs_auto_index=_boolean(
                docs.get("auto_index", True), "docs.auto_index"
            ),
            docs_patterns=_string_list(
                docs.get("patterns", DEFAULT_DOC_PATTERNS), "docs.patterns"
            ),
            docs_excluded_dirs=_string_list(
                docs.get("excluded_dirs", DEFAULT_DOC_EXCLUDED_DIRS),
                "docs.excluded_dirs",
            ),
            docs_max_files=_positive(
                docs.get("max_files", _DEFAULT_DOC_MAX_FILES), "docs.max_files"
            ),
            docs_max_file_bytes=_positive(
                docs.get("max_file_bytes", _DEFAULT_DOC_MAX_FILE_BYTES),
                "docs.max_file_bytes",
            ),
            docs_max_total_bytes=_positive(
                docs.get("max_total_bytes", _DEFAULT_DOC_MAX_TOTAL_BYTES),
                "docs.max_total_bytes",
            ),
            docs_rescan_interval_seconds=_non_negative(
                docs.get(
                    "rescan_interval_seconds", _DEFAULT_DOC_RESCAN_INTERVAL_SECONDS
                ),
                "docs.rescan_interval_seconds",
            ),
            docs_state_ttl_seconds=_positive(
                docs.get("state_ttl_seconds", _DEFAULT_DOC_STATE_TTL_SECONDS),
                "docs.state_ttl_seconds",
            ),
            doc_context_enabled=_boolean(
                docs.get("context_enabled", _DEFAULT_DOC_CONTEXT_ENABLED),
                "docs.context_enabled",
            ),
            doc_context_max_chars=_positive(
                docs.get("context_max_chars", _DEFAULT_DOC_CONTEXT_CHARS),
                "docs.context_max_chars",
            ),
            doc_context_ttl_seconds=_positive(
                docs.get("context_ttl_seconds", _DEFAULT_DOC_CONTEXT_TTL_SECONDS),
                "docs.context_ttl_seconds",
            ),
            escalation_enabled=_boolean(
                escalation.get("enabled", False), "escalation.enabled"
            ),
            loop_threshold=_positive(
                escalation.get("loop_threshold", _DEFAULT_LOOP_THRESHOLD),
                "escalation.loop_threshold",
            ),
            strong_recipe_score=_fraction(
                escalation.get("strong_recipe_score", _DEFAULT_STRONG_RECIPE_SCORE),
                "escalation.strong_recipe_score",
            ),
            claude_code_mode_map=_mode_map(claude.get("mode_map", {})),
            modes=_modes(data.get("modes", {})),
        )
        config.validate()
        return config

    # ----------------------------------------------------------------- env ---
    def override_from_env(self, logger: Any = None) -> "AgenticKnowledgeConfig":
        """
        Apply the plugin's environment over the loaded policy, in place.

        Connection variables always win, because a deployment is not a policy.
        A policy variable that is set but malformed raises rather than falls
        back, so a typo cannot silently double the token budget.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger used for notes nobody should have to guess at.

        Returns
        -------
        AgenticKnowledgeConfig
            This config.
        """
        prefix = self._ENV_PREFIX
        for name, target, cast in (
            ("ENABLED", "enabled", _boolean),
            ("READ_ENABLED", "read_enabled", _boolean),
            ("WRITE_ENABLED", "write_enabled", _boolean),
            ("ESCALATION_ENABLED", "escalation_enabled", _boolean),
            ("ARCHIVE_ENABLED", "archive_enabled", _boolean),
            # The engine's own character set: an env prefix is reduced to a
            # collection-name fragment, not merely checked for emptiness, or
            # a hyphen from the environment would reach a creation the engine
            # refuses with a 500.
            ("COLLECTION_PREFIX", "collection_prefix", _engine_collection_fragment),
            # The layout is validated, not reduced: a typo would silently read
            # one repository's documents out of a collection nobody indexes.
            ("COLLECTION_LAYOUT", "collection_layout", _collection_layout),
            ("KEY_PREFIX", "key_prefix", _safe),
            ("SSE_API_HOST", "sse_host", _text),
            ("SSE_API_PREFIX", "sse_prefix", _text),
            ("SSE_API_TOKEN", "sse_token", _text),
            ("SSE_API_USERNAME", "sse_username", _text),
            ("SSE_API_PASSWORD", "sse_password", _text),
            ("SSE_VERIFY", "sse_verify", _boolean),
        ):
            raw = _env_override(prefix, name)
            if raw is None:
                continue
            setattr(self, target, cast(raw, f"{prefix}{name}"))

        # One annotated tuple rather than a literal one: the casts differ in
        # what they return, and unpacking a heterogeneous literal makes mypy
        # hold the first entry's type for the whole loop.
        casts: Tuple[Tuple[str, str, Callable[[Any, str], Any]], ...] = (
            ("SSE_API_TIMEOUT", "sse_timeout_seconds", _positive_float),
            ("READ_TIMEOUT_SECONDS", "read_timeout_seconds", _positive_float),
            (
                "SSE_IGNORE_LANGUAGE_DETECT",
                "sse_ignore_language_detect",
                _boolean,
            ),
            ("MIN_SIMILARITY", "min_similarity", _fraction),
            ("HYBRID_SEARCH", "hybrid_search", _boolean),
            ("RERANK_RESULTS", "rerank_results", _boolean),
            ("CACHE_TTL_SECONDS", "cache_ttl_seconds", _positive),
            ("MAX_LOOKUPS_PER_GENERATION", "max_lookups_per_generation", _positive),
            ("MAX_STREAM_LEN", "max_stream_len", _positive),
            ("DEDUP_TTL_SECONDS", "dedup_ttl_seconds", _positive),
            ("MAX_PATCH_CHARS", "max_patch_chars", _positive),
            ("ARCHIVE_MAX_CHUNK_CHARS", "archive_max_chunk_chars", _positive),
            ("ARCHIVE_TTL_SECONDS", "archive_ttl_seconds", _positive),
            ("DOCS_ENABLED", "docs_enabled", _boolean),
            ("DOCS_AUTO_INDEX", "docs_auto_index", _boolean),
            ("DOCS_MAX_FILES", "docs_max_files", _positive),
            ("DOCS_MAX_FILE_BYTES", "docs_max_file_bytes", _positive),
            ("DOCS_MAX_TOTAL_BYTES", "docs_max_total_bytes", _positive),
            (
                "DOCS_RESCAN_INTERVAL_SECONDS",
                "docs_rescan_interval_seconds",
                _non_negative,
            ),
            ("DOCS_STATE_TTL_SECONDS", "docs_state_ttl_seconds", _positive),
            ("DOCS_CONTEXT_ENABLED", "doc_context_enabled", _boolean),
            ("DOCS_CONTEXT_MAX_CHARS", "doc_context_max_chars", _positive),
            (
                "DOCS_CONTEXT_TTL_SECONDS",
                "doc_context_ttl_seconds",
                _positive,
            ),
            ("LOOP_THRESHOLD", "loop_threshold", _positive),
            ("STRONG_RECIPE_SCORE", "strong_recipe_score", _fraction),
        )
        for name, target, cast in casts:
            raw = _env_override(prefix, name)
            if raw is None:
                continue
            setattr(self, target, cast(raw, f"{prefix}{name}"))

        # A request path may not wait forever, whatever the operator set, and
        # the two numbers mean different things: the budget is what a request
        # waits, the timeout what one of its calls may take.  They used to be
        # one number, which made a mode that reads three sources wait three
        # times what the operator wrote — or, at the old 1.5 s default, lose
        # every search to a timeout the engine cannot beat.
        self.sse_timeout_seconds = min(
            max(self.sse_timeout_seconds, 0.1), _MAX_SSE_TIMEOUT_SECONDS
        )
        self.read_timeout_seconds = min(
            max(self.read_timeout_seconds, 0.1), _MAX_SSE_TIMEOUT_SECONDS
        )
        self.worker = _worker_from_env(self.worker, prefix)
        try:
            connection = connection_from_env(prefix)
            # `connection_from_env` only parses; this is where an out-of-range
            # port, database, protocol or a non-positive timeout is caught, so
            # a typo surfaces here instead of as a warning on every request.
            validate_connection(connection)
            self.redis = connection
        except ValueError as exc:
            # A malformed connection setting does not invalidate the policy;
            # the feature it belongs to is simply unavailable.  It is said out
            # loud, though: an operator who set a port and got a write path that
            # never writes has to be able to find out why without a debugger,
            # and the worker is the process that would notice first.
            self.redis = RedisConnectionSettings()
            self.redis_error = str(exc)
            if logger is not None:
                logger.warning("Agentic knowledge Redis connection ignored: %s", exc)
        return self

    # ----------------------------------------------------------- validation -
    def validate(self) -> None:
        """
        Check the resolved policy.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            On an empty prefix, an unknown adapter, an unusable scope default,
            or a threshold outside its interval.
        """
        if not self.collection_prefix.strip():
            raise ValueError("collection_prefix must not be empty")
        if not self.key_prefix.strip():
            raise ValueError("key_prefix must not be empty")
        unknown = [
            name for name in self.adapters if name not in ("codex", "claude_code")
        ]
        if unknown:
            raise ValueError(f"Unknown adapters: {sorted(unknown)}")
        if self.min_similarity < 0.0 or self.min_similarity > 1.0:
            raise ValueError("min_similarity must be in [0, 1]")
        # One whole call has to fit the request's read budget *with room to
        # spare*, or the layer reads nothing at all and reports it as an empty
        # knowledge base: the store refuses a call that cannot finish before
        # the deadline, and a deadline equal to one call never leaves room for
        # it.  An operator who sets ``SSE_API_TIMEOUT`` and
        # ``READ_TIMEOUT_SECONDS`` to the same number hits exactly that, which
        # is why the budget is raised here rather than trusted.  This invariant
        # belongs to the resolved policy, not to the environment path that
        # happens to set the numbers, so it is enforced here.
        self.sse_timeout_seconds = min(
            max(self.sse_timeout_seconds, 0.1), _MAX_SSE_TIMEOUT_SECONDS
        )
        self.read_timeout_seconds = min(
            max(self.read_timeout_seconds, 0.1), _MAX_SSE_TIMEOUT_SECONDS
        )
        self.read_timeout_seconds = max(
            self.read_timeout_seconds,
            self.sse_timeout_seconds + _MIN_READ_MARGIN_SECONDS,
        )
        if not 0.0 < self.strong_recipe_score <= 1.0:
            raise ValueError("strong_recipe_score must be in (0, 1]")
        self._warn_scale_mismatch()
        if self.docs_max_file_bytes < 1024:
            raise ValueError("docs.max_file_bytes must be at least 1024 bytes")
        if self.docs_max_total_bytes < self.docs_max_file_bytes:
            raise ValueError(
                "docs.max_total_bytes must not be below docs.max_file_bytes"
            )
        if not any(str(pattern).strip() for pattern in self.docs_patterns):
            raise ValueError("docs.patterns must name at least one file pattern")
        # The same pair the worker declares on a collection and repeats on every
        # indexing call; the engine refuses ``overlap >= chunk`` and a policy
        # that says so is a deployment that indexes nothing.
        if self.worker.chunk_tokens and not (
            0 <= self.worker.chunk_overlap < self.worker.chunk_tokens
        ):
            raise ValueError(
                "worker.chunk_overlap must be at least 0 and below"
                f" worker.chunk_tokens, got {self.worker.chunk_overlap} for"
                f" {self.worker.chunk_tokens}"
            )
        if self.doc_context_max_chars < 120:
            raise ValueError(
                "docs.context_max_chars must be at least 120 characters"
            )
        for policy in self.modes.values():
            if policy.max_chars < 200:
                raise ValueError(
                    f"modes.{policy.name}.max_chars must be at least 200 characters"
                )
            for source in policy.sources:
                if not source.category.strip():
                    raise ValueError(
                        f"modes.{policy.name} has a source without a category"
                    )
                if source.max_items < 1:
                    raise ValueError(
                        f"modes.{policy.name} source {source.category} "
                        "must allow at least one item"
                    )
                if not 0.0 < float(source.weight) <= 1.0:
                    # A source's weight multiplies a normalized score, so it
                    # has to stay inside the interval the escalation threshold
                    # (``strong_recipe_score``) is read against: a weight above
                    # 1 would inflate a weak recipe past it, a negative one
                    # would rank a hit below nothing.
                    raise ValueError(
                        f"modes.{policy.name} source {source.category} "
                        f"weight must be in (0, 1], got {source.weight}"
                    )
                if source.category == "doc" and (source.phases or source.modes):
                    # The doc collection is searched with its own request
                    # and its own doc_type filter; a phase or mode filter
                    # would land on the mined-knowledge request the doc
                    # source never rides on.
                    raise ValueError(
                        f"modes.{policy.name} source {source.category} is "
                        "searched by doc_type and cannot restrict phases "
                        "or modes"
                    )
                if source.doc_types and source.category != "doc":
                    # doc_type is a field only the documentation header
                    # writes; a mined recipe carries nothing to match it
                    # against, so the filter would read as empty and the
                    # source would silently return nothing.
                    raise ValueError(
                        f"modes.{policy.name} source {source.category} has "
                        "doc_types but only the doc source carries them"
                    )
                if source.category == "session_archive" and (
                    source.phases or source.modes
                ):
                    # The archive is read through a thread filter, and that
                    # filter owns the metadata slot; a phase or mode filter
                    # would be unioned with it and return every thread of the
                    # repository, so it is a misconfiguration rather than a
                    # restriction.
                    raise ValueError(
                        f"modes.{policy.name} source {source.category} is read "
                        "through a thread filter and cannot restrict phases "
                        "or modes"
                    )
            shared = [
                source
                for source in policy.sources
                if source.category not in ("session_archive", "doc")
            ]
            if any(source.phases for source in shared) and any(
                source.modes for source in shared
            ):
                raise ValueError(
                    f"modes.{policy.name} restricts both phases and modes: one "
                    "request carries one metadata dimension, the engine unions "
                    "several filters instead of intersecting them"
                )

    def _warn_scale_mismatch(self) -> None:
        """
        Warn once when a threshold is set for a scale it cannot be met on.

        ``strong_recipe_score`` is compared against a recipe's normalized
        retrieval score, and the scale that score lives on depends on which
        retrieval leg the read runs with.  In the default deployment it is a
        fused rank rescaled onto ``[0, 1]`` only *within* the result set that
        produced it; with the reranker leg it is a cross-encoder logit the
        engine squashed onto ``[0, 1]`` — a different number under the same
        name, and one a threshold tuned for the other scale does not apply
        to.  A mismatch that prints a warning at start is one an operator can
        fix; a silent one is one nobody ever finds.
        """
        import logging

        logger = logging.getLogger("agentic_knowledge")
        if self.rerank_results:
            # A squashed cross-encoder logit tops out at 1.0 in principle,
            # but the band its relevant documents actually sit in is lower:
            # measured against a real engine, a recipe carrying the very
            # signature it was asked about scored 0.731 with the reranker on
            # and 1.0 with it off.  A threshold at or above that band is
            # therefore not a high bar, it is a bar the engine's own answers
            # rarely clear, and the identity of a matching recipe — not its
            # score — is what decides it.
            if self.strong_recipe_score >= 0.75:
                logger.warning(
                    "escalation.strong_recipe_score=%s is set for a scale "
                    "the read no longer runs on: rerank_results is on, so a "
                    "recipe's score is a squashed cross-encoder logit whose "
                    "relevant band the reference engine puts near 0.5–0.75, "
                    "and the threshold is applied to that number.  Lower "
                    "the threshold to the reranker's band or turn the "
                    "reranker off; a recipe that matches the failure's own "
                    "signature still counts, and the decision names it "
                    "(reason strong_recipe_identity).",
                    self.strong_recipe_score,
                )
            return
        # The fused rank of a hybrid search rescales the set's best hit to
        # 1.0, so any threshold in (0, 1] is reachable in principle.  What
        # is not reachable is a threshold the *engine* cannot express before
        # the rescale — but that is a property of the engine, not of this
        # policy, and the one number an operator can be told is the one
        # they set: warn when the default scale is in play and the
        # threshold sits in a band a two-leg fusion rarely reaches.
        if self.strong_recipe_score > 0.9:
            logger.warning(
                "escalation.strong_recipe_score=%s is only reachable when "
                "the engine reranks; a hybrid search rescales each result "
                "set onto [0, 1], so a recipe must be the set's best hit "
                "with very little between it and the rest.  If the "
                "escalation rule never fires, that is why.",
                self.strong_recipe_score,
            )

    # ------------------------------------------------------------ readiness -
    @property
    def read_available(self) -> bool:
        """Whether knowledge may be injected into a request."""
        return bool(self.enabled and self.read_enabled and self.sse_host.strip())

    @property
    def write_available(self) -> bool:
        """Whether episodes may be published for later indexing."""
        return bool(self.enabled and self.write_enabled and self.redis.configured)

    @property
    def archive_available(self) -> bool:
        """Whether conversation windows may be archived."""
        return bool(self.write_available and self.archive_enabled)

    @property
    def docs_available(self) -> bool:
        """
        Whether the repository's documentation may be indexed.

        Documentation is written through the same stream as mined knowledge:
        the proxy reads files and publishes events, the worker masks and
        indexes them — so no Redis means no documentation either.
        """
        return bool(self.write_available and self.docs_enabled)

    @property
    def escalation_available(self) -> bool:
        """Whether a request's model may be escalated."""
        return bool(self.enabled and self.escalation_enabled)

    def policy_for(self, mode: str) -> Optional[ModePolicy]:
        """
        Return the policy of *mode*, or ``None`` when the mode has none.

        A mode with no policy of its own falls back to the one named
        :data:`WILDCARD_MODE`, when the configuration provides one, so a
        request whose mode routing did not resolve is still asked of the
        knowledge base instead of being served empty.  A mode's own policy
        always wins over the fallback.

        Parameters
        ----------
        mode : str
            Work mode resolved by routing.

        Returns
        -------
        ModePolicy or None
            The policy, or ``None`` when this mode is not served.
        """
        return self.modes.get(mode or "") or self.modes.get(WILDCARD_MODE)

    def collection(self, scope: str, kind: str) -> str:
        """
        Return the collection one document kind lives in.

        Session archives are separated from mined knowledge because they age on
        a different clock and outnumber everything else; the repository's own
        documentation gets a third collection, for the same separation and so
        a mode's ``doc_types`` restriction is one metadata request, not two.

        Parameters
        ----------
        scope : str
            Repository scope key.
        kind : str
            Document kind.

        Returns
        -------
        str
            A collection name, in the spelling the engine accepts.
        """
        # The scope arrives from a directory name and may hold anything a
        # checkout is called — `llm-router` included.  The engine refuses to
        # create a name with a hyphen (a 500, and nothing indexes ever), so
        # the same reduction the prefix went through applies here; two scopes
        # that differ only by punctuation are the same repository as far as a
        # collection name is concerned.
        fragment = _engine_collection_fragment(scope, "scope") if scope else ""
        base = (
            f"{self.collection_prefix}_{fragment}"
            if fragment
            else self.collection_prefix
        )
        if self.collection_layout == COLLECTION_LAYOUT_SINGLE:
            # One collection per repository, every kind inside it.  What keeps
            # the kinds apart is their engine ``category``, which a search
            # names and the engine resolves to a document set *before* it
            # scores — a boundary as hard as a collection of one's own, asked
            # for per request rather than baked into a name.  What this gives
            # up is nothing the reader used: one embedder, one chunking and one
            # ``min_similarity`` already served all three collections, and
            # isolation between repositories was never a filter, it was the
            # fact that there is one collection per repository.
            return base
        if kind == "session_archive":
            return f"{base}_{_SESSIONS_SUFFIX}"
        if kind == "doc":
            # Documentation lives apart from mined knowledge: it is written
            # on a different clock (a human edits a README) and it is the
            # only kind a mode restricts by doc_type, which is one metadata
            # request the engine answers in one collection.
            return f"{base}_{_DOCS_SUFFIX}"
        return base

    @property
    def uses_single_collection(self) -> bool:
        """Whether every kind of this repository shares one collection.

        Two consumers must ask this rather than assume a name shape: retention,
        which under one collection may only age the documents of the archive's
        kind, and the worker, which can no longer discover an archive by the
        ``_sessions`` suffix of its name.
        """
        return self.collection_layout == COLLECTION_LAYOUT_SINGLE

    @property
    def served_adapters(self) -> Tuple[str, ...]:
        """Adapters this configuration serves, in resolution order."""
        return tuple(self.adapters)

    def reason_unavailable(self) -> str:
        """
        Return why the plugin is idle, for a single log line.

        Returns
        -------
        str
            ``""`` when the plugin is available, otherwise a short reason.
        """
        if not self.enabled:
            return "disabled by configuration"
        if not self.sse_host.strip():
            return f"{self._ENV_PREFIX}SSE_API_HOST is not set"
        return ""

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        """Return a representation without tokens, hosts or credentials."""
        return (
            f"AgenticKnowledgeConfig(enabled={self.enabled}, "
            f"adapters={self.adapters!r}, modes={len(self.modes)}, "
            f"read={self.read_available}, write={self.write_available}, "
            f"escalation={self.escalation_available})"
        )


_POLICY_KEYS = frozenset(
    {
        "description",
        "adapters",
        "collection_prefix",
        "collection_layout",
        "scope",
        "read",
        "modes",
        "claude_code",
        "write",
        "escalation",
        "docs",
    }
)


def _read_raw(path: Optional[pathlib.Path], prefix: str) -> Dict[str, Any]:
    """Read the policy from the env var, the given path, or the bundle."""
    raw_json = os.getenv(f"{prefix}CONFIG", "").strip()
    if raw_json:
        if raw_json.startswith("{"):
            return dict(json.loads(raw_json))
        return _load_file(pathlib.Path(raw_json))
    if path is not None:
        return _load_file(path)
    return _load_file(_default_config_path())


def _load_file(path: pathlib.Path) -> Dict[str, Any]:
    """Decode one JSON policy file."""
    with open(path, "r", encoding="utf-8") as handle:
        return dict(json.load(handle))


def _default_config_path() -> pathlib.Path:
    """Locate the bundled policy of this plugin."""
    return (
        pathlib.Path(__file__).resolve().parents[3]
        / "resources"
        / "knowledge"
        / "agentic_knowledge.json"
    )


def _section(raw: Mapping[str, Any], name: str) -> Dict[str, Any]:
    """Return one sub-object, tolerating its absence."""
    value = raw.get(name, {})
    return dict(value) if isinstance(value, Mapping) else {}


def _modes(raw: Any) -> Dict[str, ModePolicy]:
    """Validate the per-mode policies."""
    if not isinstance(raw, Mapping):
        raise ValueError("modes must be an object")
    policies: Dict[str, ModePolicy] = {}
    for name, value in raw.items():
        if not isinstance(value, Mapping):
            raise ValueError(f"modes.{name} must be an object")
        unknown = set(value) - {
            "max_chars",
            "sources",
            "escalation_model_name",
            "cheap_model_name",
        }
        if unknown:
            raise ValueError(f"Unknown modes.{name} keys: {sorted(unknown)}")
        policies[str(name)] = ModePolicy(
            name=str(name),
            max_chars=_positive(
                value.get("max_chars", _DEFAULT_MAX_CHARS), f"modes.{name}.max_chars"
            ),
            sources=_sources(value.get("sources", ()), name),
            escalation_model_name=str(value.get("escalation_model_name", "") or ""),
            cheap_model_name=str(value.get("cheap_model_name", "") or ""),
        )
    return policies


def _sources(raw: Any, mode: str) -> Tuple[SourcePolicy, ...]:
    """Validate one mode's source list."""
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"modes.{mode}.sources must be a list")
    sources = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise ValueError(f"modes.{mode}.sources entries must be objects")
        unknown = set(entry) - {
            "category",
            "weight",
            "max_items",
            "phases",
            "modes",
            "doc_types",
        }
        if unknown:
            raise ValueError(f"Unknown modes.{mode}.sources keys: {sorted(unknown)}")
        sources.append(
            SourcePolicy(
                category=str(entry.get("category", "")),
                weight=float(entry.get("weight", 1.0)),
                max_items=_positive(
                    entry.get("max_items", 3), f"modes.{mode}.sources.max_items"
                ),
                phases=_phase_names(
                    entry.get("phases", ()), f"modes.{mode}.sources.phases"
                ),
                modes=_phase_names(
                    entry.get("modes", ()), f"modes.{mode}.sources.modes"
                ),
                doc_types=_phase_names(
                    entry.get("doc_types", ()),
                    f"modes.{mode}.sources.doc_types",
                ),
            )
        )
    return tuple(sources)


def _phase_names(raw: Any, label: str) -> Tuple[str, ...]:
    """Validate one source's phase restriction."""
    if raw in (None, "", [], ()):
        return ()
    if isinstance(raw, str) or not isinstance(raw, (list, tuple)):
        raise ValueError(f"{label} must be a list of phase names")
    names = []
    for value in raw:
        name = str(value).strip()
        if not name:
            raise ValueError(f"{label} must not hold an empty phase name")
        if name not in names:
            names.append(name)
    return tuple(names)


def _overrides(raw: Any) -> Dict[str, str]:
    """Validate the declared-path to scope-key table."""
    if not isinstance(raw, Mapping):
        raise ValueError("scope.overrides must be an object")
    return {str(key): str(value) for key, value in raw.items() if str(value).strip()}


def _mode_map(raw: Any) -> Dict[str, str]:
    """Validate Claude Code's tier to work-mode table."""
    if not isinstance(raw, Mapping):
        raise ValueError("claude_code.mode_map must be an object")
    return {str(key): str(value) for key, value in raw.items() if str(value).strip()}


def _env_override(prefix: str, name: str) -> Optional[str]:
    """
    Read one variable of this plugin's prefix, ``None`` when it is not set.

    A variable that is present but empty is an override attempt like any other
    and goes on to be validated — the contract the Codex memory settings use
    (``state._resolve``) and the reason a mistyped value cannot quietly leave
    the policy at its default.  Only a variable that is genuinely absent from
    the environment falls back.
    """
    raw = os.environ.get(f"{prefix}{name}")
    return None if raw is None else raw.strip()


def _worker_from_env(worker: WorkerSettings, prefix: str) -> WorkerSettings:
    """Build worker settings over the environment."""
    values: Dict[str, Any] = {}
    for name, target, cast in (
        ("WORKER_GROUP", "group", _text),
        ("WORKER_CONSUMER", "consumer", _text),
        ("WORKER_BATCH", "batch", _positive),
        ("WORKER_BLOCK_MS", "block_ms", _positive),
        ("SSE_EMBEDDER", "embedder", _text),
        ("SSE_RERANKER", "reranker", _text),
        ("SSE_INDEX_TYPE", "index_type", _text),
        ("SSE_CHUNK_TOKENS", "chunk_tokens", _non_negative),
        ("SSE_CHUNK_OVERLAP", "chunk_overlap", _non_negative),
        ("RETENTION_INTERVAL_SECONDS", "retention_interval_seconds", _non_negative),
        ("MAX_PENDING", "max_pending", _positive),
    ):
        raw = _env_override(prefix, name)
        if raw is None:
            continue
        values[target] = cast(raw, f"{prefix}{name}")
    index_type = values.get("index_type", worker.index_type)
    if index_type not in _INDEX_TYPES:
        raise ValueError(
            f"{prefix}SSE_INDEX_TYPE must be one of {list(_INDEX_TYPES)}, "
            f"got {index_type!r}"
        )
    settings = replace(worker, **values)
    # An overlap of the chunk size or more makes every chunk a copy of its
    # neighbour: the engine accepts the pair and the collection indexes the
    # same text over and over, so the combination is refused rather than
    # discovered a week later in a knowledge base of duplicates.
    if settings.chunk_tokens and settings.chunk_overlap >= settings.chunk_tokens:
        raise ValueError(
            f"{prefix}SSE_CHUNK_OVERLAP ({settings.chunk_overlap}) must be "
            f"smaller than SSE_CHUNK_TOKENS ({settings.chunk_tokens})"
        )
    return settings


def _boolean(value: Any, label: str) -> bool:
    """Read a boolean from config or environment."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{label} must be a boolean, got {value!r}")


def _positive(value: Any, label: str) -> int:
    """Read a strictly positive integer."""
    number = _number(value, label)
    if int(number) <= 0:
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    return int(number)


def _positive_float(value: Any, label: str) -> float:
    """
    Read a strictly positive duration.

    A separate cast because a timeout is not a count: reading ``1.5`` as an
    integer quietly makes it ``1``, and reading a deliberate ``0.5`` makes it
    ``0`` and then an error, so the sub-second budgets a request path actually
    wants would be the one thing the configuration refused to say.
    """
    number = _number(value, label)
    if float(number) <= 0:
        raise ValueError(f"{label} must be a positive number, got {value!r}")
    return float(number)


def _non_negative(value: Any, label: str) -> int:
    """Read an integer that may be zero."""
    number = _number(value, label)
    if int(number) < 0:
        raise ValueError(f"{label} must not be negative, got {value!r}")
    return int(number)


def _fraction(value: Any, label: str) -> float:
    """Read a float in ``[0, 1]``."""
    number = _number(value, label)
    if not 0.0 <= float(number) <= 1.0:
        raise ValueError(f"{label} must be in [0, 1], got {value!r}")
    return float(number)


def _number(value: Any, label: str) -> float:
    """Coerce a config or environment value to a real number."""
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a number, got {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number, got {value!r}") from exc


def _text(value: Any, label: str) -> str:
    """Read a plain string from the environment, empty allowed."""
    return str(value).strip()


def _safe_text(value: Any, label: str) -> str:
    """Read a non-empty identifier-ish string."""
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} must not be empty")
    return text


def _safe(value: str, label: str) -> str:
    """Strip a prefix down to something a collection name accepts."""
    text = value.strip().strip("-_")
    if not text:
        raise ValueError(f"{label} must not be empty")
    return text


def _string_list(value: Any, label: str) -> Tuple[str, ...]:
    """Read a list of non-empty strings."""
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a list of names")
    items = tuple(str(item).strip() for item in value if str(item).strip())
    if not items:
        raise ValueError(f"{label} must not be empty")
    return items
