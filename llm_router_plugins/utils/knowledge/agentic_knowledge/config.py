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
      "collection_prefix": "agentic-kb",
      "scope": {"strategy": "repo_basename", "overrides": {}, "default": ""},
      "read": {"enabled": true, "timeout_seconds": 1.5, "min_similarity": 0.35,
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
                     "strong_recipe_score": 0.75}
    }
"""

import json
import os
import pathlib
import re

from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Dict, Mapping, Optional, Tuple

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
    "WorkerSettings",
]

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

#: Ceiling of one read call, so a typo cannot stall a request.
_MAX_SSE_TIMEOUT_SECONDS = 10.0
_DEFAULT_READ_TIMEOUT_SECONDS = 1.5
_DEFAULT_MIN_SIMILARITY = 0.35
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
_DEFAULT_WORKER_BLOCK_MS = 5_000

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
        `implement`), the document kind is what it produced (`recipe`); the
        engine filters them with different options and they intersect.
    """

    category: str
    weight: float = 1.0
    max_items: int = 3
    phases: Tuple[str, ...] = ()


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
        Chunk size declared for a new collection; ``0`` leaves the server default.
    chunk_overlap : int
        Chunk overlap declared for a new collection.
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
    chunk_tokens: int = 0
    chunk_overlap: int = 0
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
    key_prefix: str = _DEFAULT_KEY_PREFIX

    scope_strategy: str = "repo_basename"
    scope_overrides: Dict[str, str] = field(default_factory=dict)
    scope_default: str = ""

    read_enabled: bool = True
    #: Budget of the read path: how long the knowledge layer may spend on the
    #: engine for one request.  It drives the engine client's own timeout, so
    #: the number the operator wrote is the number a request waits.
    read_timeout_seconds: float = _DEFAULT_READ_TIMEOUT_SECONDS
    min_similarity: float = _DEFAULT_MIN_SIMILARITY
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
    sse_timeout_seconds: float = _DEFAULT_READ_TIMEOUT_SECONDS
    sse_verify: bool = True

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

        # Whether the operator pinned the transport timeout apart from the read
        # budget; `SSE_API_TIMEOUT` is one name among many below, so the flag
        # has to start false rather than appear with it.
        sse_timeout_overridden = False
        for name, target, cast in (
            ("SSE_API_TIMEOUT", "sse_timeout_seconds", _positive_float),
            ("READ_TIMEOUT_SECONDS", "read_timeout_seconds", _positive_float),
            ("MIN_SIMILARITY", "min_similarity", _fraction),
            ("CACHE_TTL_SECONDS", "cache_ttl_seconds", _positive),
            ("MAX_LOOKUPS_PER_GENERATION", "max_lookups_per_generation", _positive),
            ("MAX_STREAM_LEN", "max_stream_len", _positive),
            ("DEDUP_TTL_SECONDS", "dedup_ttl_seconds", _positive),
            ("MAX_PATCH_CHARS", "max_patch_chars", _positive),
            ("ARCHIVE_MAX_CHUNK_CHARS", "archive_max_chunk_chars", _positive),
            ("ARCHIVE_TTL_SECONDS", "archive_ttl_seconds", _positive),
            ("LOOP_THRESHOLD", "loop_threshold", _positive),
            ("STRONG_RECIPE_SCORE", "strong_recipe_score", _fraction),
        ):
            raw = _env_override(prefix, name)
            if raw is None:
                continue
            setattr(self, target, cast(raw, f"{prefix}{name}"))
            if name == "SSE_API_TIMEOUT":
                sse_timeout_overridden = True

        # The read budget is what a request waits for the engine; the transport
        # timeout is the same number unless the operator said otherwise.  Left
        # as two fields because the worker indexes with a far longer timeout
        # than the proxy searches with, and shares this one.
        if not sse_timeout_overridden:
            self.sse_timeout_seconds = self.read_timeout_seconds
        # A request path may not wait forever, whatever the operator set.
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
        if not 0.0 < self.strong_recipe_score <= 1.0:
            raise ValueError("strong_recipe_score must be in (0, 1]")
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
    def escalation_available(self) -> bool:
        """Whether a request's model may be escalated."""
        return bool(self.enabled and self.escalation_enabled)

    def policy_for(self, mode: str) -> Optional[ModePolicy]:
        """
        Return the policy of *mode*, or ``None`` when the mode has none.

        Parameters
        ----------
        mode : str
            Work mode resolved by routing.

        Returns
        -------
        ModePolicy or None
            The policy, or ``None`` when this mode is not served.
        """
        return self.modes.get(mode or "")

    def collection(self, scope: str, kind: str) -> str:
        """
        Return the collection one document kind lives in.

        Session archives are separated from mined knowledge because they age on
        a different clock and outnumber everything else.

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
        base = f"{self.collection_prefix}_{fragment}" if fragment else self.collection_prefix
        return f"{base}_sessions" if kind == "session_archive" else base

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
        "scope",
        "read",
        "modes",
        "claude_code",
        "write",
        "escalation",
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
        unknown = set(entry) - {"category", "weight", "max_items", "phases"}
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
    return replace(worker, **values)


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
