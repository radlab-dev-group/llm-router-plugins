"""
Optional shared session memory for the Codex routing cascade.

Why it exists
-------------
Codex runs one action per model call.  Between two calls the user's command
does not change, but the evidence does: the test suite has now run, the patch
has landed, the failure arrived.  A stateless cascade re-reads whatever the
client happened to send and can lose the thread between two requests of the
same action — especially when the history arrives incrementally instead of as a
full transcript.

This module remembers, per session, the last *reliable* phase of the current
command generation, so a neutral request in the middle of an action keeps it.
The memory is shared between Gunicorn workers through Redis, which is why
there is no local-process cache: two workers holding two different ideas of the
same session's phase is worse than no memory at all.

What it never does
------------------
- It never overrides a fresh, unambiguous signal.  The caller applies it only
  where the deterministic layers stayed silent.
- It never stores a fallback, a weak semantic match or a special request
  (title generation, compaction).
- It stores no model names, no conversation text, no tool output and no
  credentials — a mode name, the kind and reason of the evidence that produced
  it, bounded event identifiers and a version counter.
- A missing, expired, malformed or foreign record is a miss, not an error.
  A Redis outage is a miss too: routing continues statelessly and the request
  is never failed because of this module.

Identity
--------
A key isolates the plugin's namespace, the session, thread and agent.
Installations must choose distinct ``MEMORY_KEY_PREFIX`` values. ``turn_id`` is
the *generation*: it distinguishes the
current command from the one before it inside one thread, and it is what makes
an older or causally incomparable update a conflict instead of an overwrite.
Without enough identifiers to build a key, routing stays stateless.
"""

import hashlib
import json
import logging
import math
import os
import re
import threading
import time
import traceback

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Optional, Tuple

__all__ = [
    "MEMORY_BACKEND_MEMORY",
    "MEMORY_BACKEND_REDIS",
    "RedisConnectionSettings",
    "CodexMemoryConfig",
    "SessionRoutingState",
    "PendingRoutingCall",
    "MemoryResolution",
    "resolve_memory",
    "remember_decision",
    "memory_config_from_raw",
    "connection_from_env",
    "validate_connection",
    "merge_state",
    "record_state",
    "RoutingStateStore",
    "InMemoryRoutingStateStore",
    "RedisRoutingStateStore",
    "MemoryStatus",
    "session_key",
    "build_state_store",
]

#: Process-local store.  Isolated replay and contract tests only, never a
#: production fallback: a per-worker cache splits the state between workers.
MEMORY_BACKEND_MEMORY = "memory"

#: Shared Redis-backed store — the only backend two workers can agree on.
MEMORY_BACKEND_REDIS = "redis"

VALID_MEMORY_BACKENDS = (MEMORY_BACKEND_MEMORY, MEMORY_BACKEND_REDIS)

#: Bump when the stored record layout changes; an unknown version is a miss.
STATE_SCHEMA_VERSION = 1

#: Characters allowed in a key component.  Anything else is fingerprinted, so a
#: client-supplied identifier can never forge a key or escape the namespace.
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9._-]{1,128}")

#: Fallback fingerprint length for an unsafe identifier.
_FINGERPRINT_CHARS = 16

#: Environment variables holding connection settings, read the way the host
#: application reads its own auth Redis: separate variables, ``os.environ.get``,
#: explicit numeric conversion, empty password normalized to ``None``.
#:
#: Deliberately *not* read from the JSON config and never inherited from the
#: host application's ``AUTH_REDIS_*`` or generic ``REDIS_*`` variables.
CONNECTION_ENV_FIELDS = (
    "host", "port", "db", "password", "protocol", "username", "ssl",
    "ssl_ca_certs", "ssl_certfile", "ssl_keyfile", "ssl_cert_reqs",
    "socket_connect_timeout", "socket_timeout",
)


@dataclass(frozen=True)
class MemoryStatus:
    """
    Why the memory did or did not contribute to one decision.

    Parameters
    ----------
    state : str
        ``disabled``, ``unconfigured``, ``unavailable``, ``hit``, ``miss``,
        ``expired``, ``conflict`` or ``written``.
        The distinction matters operationally: ``unavailable`` is a broken
        deployment while ``miss`` is an ordinary cold start.
        Expired records are reported as ``expired`` rather than ``miss`` only
        when the record was found and found stale.
    detail : str
        Short reason code, safe to log: never a credential, never a full
        session identifier.
    """

    state: str
    detail: str = ""


#: Memory contributed nothing; the plugin runs statelessly.
STATUS_DISABLED = MemoryStatus("disabled")

#: Memory is enabled but has no host, or the client library is missing.
STATUS_UNCONFIGURED = MemoryStatus("unconfigured")

#: The store is configured but the operation failed.
STATUS_UNAVAILABLE = MemoryStatus("unavailable")


@dataclass(frozen=True)
class RedisConnectionSettings:
    """
    Redis connection parameters, read exclusively from the plugin's own ENV.

    Mirrors the host application's auth-Redis contract: one variable per
    parameter under the plugin prefix, an empty password meaning *no
    password*, and TLS verification on by default — enabling TLS never
    silently disables certificate checks.

    Parameters
    ----------
    host : str
        Server host.  Empty means "not configured", which is the default: no
        memory, no error.
    port : int
        Server port.
    db : int
        Database number.
    password : str or None
        AUTH password; ``None`` when unset or empty.
    protocol : int
        RESP protocol version.
    username : str or None
        ACL username; ``None`` for the default user.
    ssl : bool
        Whether to connect over TLS.
    ssl_ca_certs, ssl_certfile, ssl_keyfile : str or None
        TLS material paths.
    ssl_cert_reqs : str
        ``required`` (default), ``optional`` or ``none``.
    socket_connect_timeout, socket_timeout : float
        Short positive timeouts, in seconds.
    """

    host: str = ""
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    protocol: int = 3
    username: Optional[str] = None
    ssl: bool = False
    ssl_ca_certs: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    ssl_cert_reqs: str = "required"
    socket_connect_timeout: float = 1.0
    socket_timeout: float = 1.0

    @property
    def configured(self) -> bool:
        """Whether a connection can be attempted at all."""
        return bool(self.host)

    def client_kwargs(self) -> Dict[str, Any]:
        """
        Return the keyword arguments for :class:`redis.Redis`.

        Returns
        -------
        dict
            Connection arguments, including the TLS block only when TLS is on
            and the short socket timeouts that keep a dead Redis from stalling
            a routing decision.
        """
        arguments: Dict[str, Any] = {
            "host": self.host,
            "port": self.port,
            "db": self.db,
            "password": self.password,
            "protocol": self.protocol,
            "decode_responses": True,
            "socket_connect_timeout": self.socket_connect_timeout,
            "socket_timeout": self.socket_timeout,
        }
        if self.username:
            arguments["username"] = self.username
        if self.ssl:
            arguments["ssl"] = True
            arguments["ssl_cert_reqs"] = self.ssl_cert_reqs
            for name in ("ssl_ca_certs", "ssl_certfile", "ssl_keyfile"):
                value = getattr(self, name)
                if value:
                    arguments[name] = value
        return arguments


@dataclass(frozen=True)
class CodexMemoryConfig:
    """
    Memory policy: whether it is on, where it lives and how big it may get.

    Non-secret settings resolve ENV → explicit JSON → safe default.  Connection
    settings have **no** JSON counterpart at all, so a config file can never
    carry a Redis URL or a password.

    Parameters
    ----------
    enabled : bool
        Off unless explicitly turned on, by ENV or by an explicit JSON section.
    backend : str
        :data:`MEMORY_BACKEND_REDIS` in production;
        :data:`MEMORY_BACKEND_MEMORY` for isolated replay and tests only.
    ttl_seconds : int
        Lifetime of one session record.
    max_sessions : int
        Cap on the sessions this plugin's namespace may hold, enforced across
        workers without scanning the keyspace.
    max_events : int
        Cap on the event identifiers remembered per session.
    max_calls : int
        Cap on the unresolved call identifiers remembered per session.
    key_prefix : str
        Namespace of the plugin's own keys.  Pruning never touches anything
        outside it.
    max_retries : int
        Attempts on a version conflict before the request continues statelessly.
    connection : RedisConnectionSettings
        Where to connect.
    """

    enabled: bool = False
    backend: str = MEMORY_BACKEND_REDIS
    ttl_seconds: int = 900
    max_sessions: int = 10000
    max_events: int = 64
    max_calls: int = 32
    key_prefix: str = "llm-router:codex-routing"
    max_retries: int = 1
    connection: RedisConnectionSettings = field(default_factory=RedisConnectionSettings)

    @property
    def usable(self) -> bool:
        """Whether the policy is on and points at something connectable."""
        if not self.enabled:
            return False
        if self.backend == MEMORY_BACKEND_REDIS:
            return self.connection.configured
        return True


def _env_text(prefix: str, name: str, default: str = "") -> str:
    """Read a stripped string variable."""
    return (os.environ.get(f"{prefix}{name}") or default).strip()


def _env_secret(prefix: str, name: str) -> Optional[str]:
    """Read a secret variable, normalizing an empty value to ``None``."""
    value = (os.environ.get(f"{prefix}{name}") or "").strip()
    return value or None


def _env_int(prefix: str, name: str, default: int) -> int:
    """Read an integer variable, falling back to *default* when unparsable."""
    raw = (os.environ.get(f"{prefix}{name}") or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{prefix}{name} must be an integer, got {raw!r}") from exc


def _env_bool(prefix: str, name: str, default: bool) -> bool:
    """Read a boolean variable; unrecognized values are an error, not a guess."""
    raw = (os.environ.get(f"{prefix}{name}") or "").strip().lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{prefix}{name} must be a boolean, got {raw!r}")


def _env_float(prefix: str, name: str, default: float) -> float:
    """Read a positive float variable."""
    raw = (os.environ.get(f"{prefix}{name}") or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{prefix}{name} must be a number, got {raw!r}") from exc


def _resolve(raw: Dict[str, Any], key: str, env_name: str, default):
    """
    Apply the ENV → explicit JSON → default order of a non-secret setting.

    ``MEMORY_*`` variables always win over the configuration file, which is the
    documented precedence; a variable that is present but empty still counts as
    an explicit override attempt and is validated, not silently ignored.
    """
    if env_name in os.environ:
        return os.environ[env_name]
    if isinstance(raw, dict) and key in raw:
        return raw[key]
    return default


def connection_from_env(prefix: str) -> RedisConnectionSettings:
    """
    Build the connection settings from the plugin's own environment variables.

    Parameters
    ----------
    prefix : str
        The plugin's environment prefix, e.g.
        ``LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_``.

    Returns
    -------
    RedisConnectionSettings
        Values with the documented defaults.

    Raises
    ------
    ValueError
        On a malformed numeric, boolean or certificate-verification value.
    """
    cert_reqs = _env_text(prefix, "REDIS_SSL_CERT_REQS", "required").lower()
    if cert_reqs not in ("required", "optional", "none"):
        raise ValueError(
            f"{prefix}REDIS_SSL_CERT_REQS must be required, optional or none"
        )
    return RedisConnectionSettings(
        host=_env_text(prefix, "REDIS_HOST"),
        port=_env_int(prefix, "REDIS_PORT", 6379),
        db=_env_int(prefix, "REDIS_DB", 0),
        password=_env_secret(prefix, "REDIS_PASSWORD"),
        protocol=_env_int(prefix, "REDIS_PROTOCOL", 3),
        username=_env_secret(prefix, "REDIS_USERNAME"),
        ssl=_env_bool(prefix, "REDIS_SSL", False),
        ssl_ca_certs=_env_secret(prefix, "REDIS_SSL_CA_CERTS"),
        ssl_certfile=_env_secret(prefix, "REDIS_SSL_CERTFILE"),
        ssl_keyfile=_env_secret(prefix, "REDIS_SSL_KEYFILE"),
        ssl_cert_reqs=cert_reqs,
        socket_connect_timeout=_env_float(prefix, "REDIS_SOCKET_CONNECT_TIMEOUT", 1.0),
        socket_timeout=_env_float(prefix, "REDIS_SOCKET_TIMEOUT", 1.0),
    )


def memory_config_from_raw(raw: Any, prefix: str) -> CodexMemoryConfig:
    """
    Resolve the memory policy from an optional JSON section plus the ENV.

    An absent section means "off": an older configuration file keeps the
    stateless behaviour without being rewritten.

    Parameters
    ----------
    raw : Any
        ``settings.memory`` from the supplied JSON, or ``None``.
    prefix : str
        Environment prefix providing the ``MEMORY_*`` variables.

    Returns
    -------
    CodexMemoryConfig
        The resolved policy, with connection settings read from the ENV only.

    Raises
    ------
    ValueError
        On an unknown backend or an out-of-range limit.
    """
    data: Dict[str, Any] = raw if isinstance(raw, dict) else {}
    unknown = set(data) - {
        "enabled", "backend", "ttl_seconds", "max_sessions", "max_events",
        "max_calls", "key_prefix", "max_retries",
    }
    if unknown:
        raise ValueError(f"Unknown settings.memory fields: {sorted(unknown)}")

    def pick(key: str, env_name: str, default):
        return _resolve(data, key, f"{prefix}{env_name}", default)

    enabled = _as_bool(pick("enabled", "MEMORY_ENABLED", False))
    backend = str(
        pick("backend", "MEMORY_BACKEND", MEMORY_BACKEND_REDIS)
    ).strip().lower()
    config = CodexMemoryConfig(
        enabled=enabled,
        backend=backend,
        ttl_seconds=_positive(
            pick("ttl_seconds", "MEMORY_TTL_SECONDS", 900), "ttl_seconds"
        ),
        max_sessions=_positive(
            pick("max_sessions", "MEMORY_MAX_SESSIONS", 10000), "max_sessions"
        ),
        max_events=_positive(
            pick("max_events", "MEMORY_MAX_EVENTS", 64), "max_events"
        ),
        max_calls=_positive(pick("max_calls", "MEMORY_MAX_CALLS", 32), "max_calls"),
        key_prefix=str(pick(
            "key_prefix", "MEMORY_KEY_PREFIX", "llm-router:codex-routing"
        )).strip(),
        max_retries=_non_negative(
            pick("max_retries", "MEMORY_MAX_RETRIES", 1), "max_retries"
        ),
        connection=connection_from_env(prefix),
    )
    if backend not in VALID_MEMORY_BACKENDS:
        raise ValueError(
            f"settings.memory.backend must be one of {VALID_MEMORY_BACKENDS}"
        )
    if not config.key_prefix or any(
        char.isspace() for char in config.key_prefix
    ):
        raise ValueError(
            "settings.memory.key_prefix must be a non-empty, space-free prefix"
        )
    return config


def _as_bool(value: Any) -> bool:
    """Interpret a JSON or ENV boolean-ish value."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
    raise ValueError(f"memory enabled must be a boolean, got {value!r}")


def _positive(value: Any, label: str) -> int:
    """Validate a strictly positive integer limit."""
    number = _integer(value, label)
    if number < 1:
        raise ValueError(f"settings.memory.{label} must be >= 1")
    return number


def _non_negative(value: Any, label: str) -> int:
    """Validate a non-negative integer."""
    number = _integer(value, label)
    if number < 0:
        raise ValueError(f"settings.memory.{label} must be >= 0")
    return number


def _integer(value: Any, label: str) -> int:
    """
    Coerce an int-ish value, rejecting bools, fractional floats and text.

    A number arriving from the environment is a string, so a digit string is a
    legitimate value here; anything that is not a whole number is a
    configuration error rather than something to guess at.
    """
    if isinstance(value, bool):
        raise ValueError(f"settings.memory.{label} must be an integer")
    if isinstance(value, str):
        try:
            value = int(value.strip())
        except ValueError as exc:
            raise ValueError(
                f"settings.memory.{label} must be an integer, got {value!r}"
            ) from exc
    if not isinstance(value, (int, float)):
        raise ValueError(f"settings.memory.{label} must be an integer")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"settings.memory.{label} must be an integer")
    return int(value)


def validate_connection(connection: RedisConnectionSettings) -> None:
    """
    Validate resolved connection settings.

    Parameters
    ----------
    connection : RedisConnectionSettings
        The settings to check.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        On an out-of-range port, database, protocol or timeout.
    """
    if not 1 <= connection.port <= 65535:
        raise ValueError(f"Redis port must be in [1, 65535], got {connection.port}")
    if not 0 <= connection.db <= 15:
        raise ValueError(f"Redis db must be in [0, 15], got {connection.db}")
    if connection.protocol not in (2, 3):
        raise ValueError(f"Redis protocol must be 2 or 3, got {connection.protocol}")
    if connection.ssl_cert_reqs not in ("required", "optional", "none"):
        raise ValueError("Redis ssl_cert_reqs must be required, optional or none")
    for name in ("socket_connect_timeout", "socket_timeout"):
        value = getattr(connection, name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or (
            not math.isfinite(value)
        ) or value <= 0:
            raise ValueError(f"Redis {name} must be a positive number of seconds")


def _safe_component(value: str) -> str:
    """
    Return a key-safe form of *value*, fingerprinting anything unusual.

    Identifiers come from a client payload.  A component that is not plainly
    alphanumeric is replaced by a digest of itself, which keeps the key stable
    for the same input while making it impossible to inject a separator.
    """
    text = (value or "").strip()
    if not text:
        return "-"
    if _SAFE_COMPONENT.fullmatch(text):
        return text
    return "~" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:_FINGERPRINT_CHARS]


def session_key(
    key_prefix: str,
    session_id: str,
    thread_id: str,
    agent_name: str,
) -> Optional[str]:
    """
    Build the isolation key of one agent's session, or ``None``.

    A session is identified by the CLI session and thread; the agent is part of
    the key because a sub-agent working in the same thread has its own current
    action.  Without both a session and a thread identifier there is nothing to
    isolate, and routing must stay stateless rather than guess.

    Parameters
    ----------
    key_prefix : str
        The plugin's own namespace.
    session_id : str
        ``client_metadata.session_id`` of the request.
    thread_id : str
        ``client_metadata.thread_id`` of the request.
    agent_name : str
        Agent emitting the request; the root agent uses its own name.

    Returns
    -------
    Optional[str]
        The key, or ``None`` when the identifiers are insufficient.
    """
    if not session_id or not thread_id:
        return None
    return ":".join((
        key_prefix,
        f"v{STATE_SCHEMA_VERSION}",
        _safe_component(session_id),
        _safe_component(thread_id),
        _safe_component(agent_name),
    ))


@dataclass(frozen=True)
class PendingRoutingCall:
    """Bounded structural evidence; never command text or tool output."""

    call_id: str
    name: str
    mode: str
    kind: str
    reason: str
    event_id: str
    token: str


@dataclass(frozen=True)
class SessionRoutingState:
    """
    What one session currently remembers about its work phase.

    Parameters
    ----------
    mode : str
        The remembered work mode.
    kind : str
        Kind of evidence that produced it, e.g. ``command`` or ``patch``.
    reason : str
        Reason code of that evidence.
    generation : str
        ``turn_id`` of the command generation this state belongs to.
    event_id : str
        Identifier of the payload item that produced the evidence.
    fingerprint : str
        Digest of the evidence, used to notice a repeated payload.
    pending_calls : Tuple[str, ...]
        Call identifiers opened and not yet settled, bounded by the policy.
    seen_events : Tuple[str, ...]
        Recently accounted event identifiers, newest last, bounded by the
        policy.  Replaying a full transcript therefore does not re-count a
        result it has already applied.
    version : int
        Monotonic compare-and-set counter, assigned by the store.
    updated_at : float
        Wall-clock second of the last evidence change or generation reset.
        Metadata-only writes preserve this time and the original TTL deadline.
    """

    mode: str = ""
    kind: str = ""
    reason: str = ""
    generation: str = ""
    event_id: str = ""
    fingerprint: str = ""
    pending_calls: Tuple[str, ...] = ()
    seen_events: Tuple[str, ...] = ()
    version: int = 0
    updated_at: float = 0.0
    pending_evidence: Tuple[PendingRoutingCall, ...] = ()
    history: Tuple[str, ...] = ()
    action_token: str = ""
    retired_generations: Tuple[str, ...] = ()
    context_id: str = ""
    reset: bool = False
    seen_calls: Tuple[str, ...] = ()

    def set_version(self, version: int, updated_at: float) -> "SessionRoutingState":
        """
        Return a copy stamped with the version and evidence time the store assigns.

        Parameters
        ----------
        version : int
            The version this record is written under.
        updated_at : float
            Wall-clock second of the write.

        Returns
        -------
        SessionRoutingState
            This record with :attr:`version` and :attr:`updated_at` set.
        """
        return replace(self, version=version, updated_at=updated_at)

    def to_json(self) -> str:
        """
        Serialize the record.

        Returns
        -------
        str
            A compact JSON object tagged with :data:`STATE_SCHEMA_VERSION`.
        """
        return json.dumps({
            "v": STATE_SCHEMA_VERSION,
            "mode": self.mode,
            "kind": self.kind,
            "reason": self.reason,
            "gen": self.generation,
            "ev": self.event_id,
            "fp": self.fingerprint,
            "pending": list(self.pending_calls),
            "seen": list(self.seen_events),
            "ver": self.version,
            "ts": self.updated_at,
            "calls": [vars(call) for call in self.pending_evidence],
            "history": list(self.history),
            "action": self.action_token,
            "retired": list(self.retired_generations),
            "context": self.context_id,
            "reset": self.reset,
            "seen_calls": list(self.seen_calls),
        }, separators=(",", ":"), ensure_ascii=True)

    @classmethod
    def from_json(cls, raw: Any) -> Optional["SessionRoutingState"]:
        """
        Parse a stored record, returning ``None`` for anything unusable.

        A damaged, unknown-version or non-object record is a miss: the memory
        degrades to stateless routing instead of acting on a guess, and it never
        evaluates or imports what it read.

        Parameters
        ----------
        raw : Any
            The stored string.

        Returns
        -------
        Optional[SessionRoutingState]
            The record, or ``None``.
        """
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8")
            except UnicodeError:
                return None
        if not isinstance(raw, str) or not raw or len(raw) > 262144:
            return None
        try:
            data = json.loads(raw)
        except (TypeError, ValueError, RecursionError, OverflowError):
            return None
        if not isinstance(data, dict) or type(data.get("v")) is not int:
            return None
        if data.get("v") != STATE_SCHEMA_VERSION:
            return None
        mode = data.get("mode")
        reset = data.get("reset", False)
        if not isinstance(reset, bool):
            return None
        if not isinstance(mode, str) or (not mode and not (reset and data.get("gen"))):
            return None
        strings = ("mode", "kind", "reason", "gen", "ev", "fp", "action", "context")
        if any(not isinstance(data.get(key, ""), str) for key in strings):
            return None
        for key in ("pending", "seen", "history", "retired", "seen_calls"):
            values = data.get(key, [])
            if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
                return None
        version = data.get("ver", 0)
        timestamp = data.get("ts", 0.0)
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            return None
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
            return None
        try:
            if not math.isfinite(timestamp) or timestamp < 0:
                return None
            calls = tuple(PendingRoutingCall(**call) for call in data.get("calls", []))
        except (TypeError, ValueError, OverflowError):
            return None
        if any(
            not isinstance(value, str)
            for call in calls for value in vars(call).values()
        ):
            return None
        return cls(
            mode=mode,
            kind=_text(data.get("kind")),
            reason=_text(data.get("reason")),
            generation=_text(data.get("gen")),
            event_id=_text(data.get("ev")),
            fingerprint=_text(data.get("fp")),
            pending_calls=_texts(data.get("pending")),
            seen_events=_texts(data.get("seen")),
            version=version,
            updated_at=timestamp,
            pending_evidence=calls,
            history=_texts(data.get("history")),
            action_token=_text(data.get("action")),
            retired_generations=_texts(data.get("retired")),
            context_id=_text(data.get("context")),
            reset=reset,
            seen_calls=_texts(data.get("seen_calls")),
        )


def _text(value: Any) -> str:
    """Return *value* when it is a string, else ``""``."""
    return value if isinstance(value, str) else ""


def _texts(value: Any) -> Tuple[str, ...]:
    """Return the string entries of *value* as a tuple."""
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(entry for entry in value if isinstance(entry, str))


def _number(value: Any) -> float:
    """Return *value* when it is a finite number, else ``0.0``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value) if math.isfinite(float(value)) else 0.0


class RoutingStateStore:
    """
    Interface the cascade needs from a session memory.

    Implementations must be safe to share between threads and workers, must
    never raise on a routing path (a failure is a miss reported through
    :class:`MemoryStatus`), and must never touch keys outside the plugin's own
    namespace.
    """

    def read(self, key: str) -> Tuple[Optional[SessionRoutingState], MemoryStatus]:
        """
        Return the state stored under *key*.

        Parameters
        ----------
        key : str
            Key from :func:`session_key`.

        Returns
        -------
        Tuple[Optional[SessionRoutingState], MemoryStatus]
            The record (or ``None``) and why.
        """
        raise NotImplementedError

    def write(
        self,
        key: str,
        state: SessionRoutingState,
        expected_version: int,
    ) -> MemoryStatus:
        """
        Store *state* only if the record is still at *expected_version*.

        Parameters
        ----------
        key : str
            Key from :func:`session_key`.
        state : SessionRoutingState
            The record to store.
        expected_version : int
            Version read with this state; ``0`` when the key was absent.

        Returns
        -------
        MemoryStatus
            ``written`` on success, ``conflict`` when another worker moved
            first, ``unavailable`` on an infrastructure error.
        """
        raise NotImplementedError

    def clear(self, key: str) -> MemoryStatus:
        """
        Forget one session.

        Parameters
        ----------
        key : str
            Key from :func:`session_key`.

        Returns
        -------
        MemoryStatus
            Why the key was or was not removed.
        """
        raise NotImplementedError

    @property
    def available(self) -> bool:
        """Whether the store can be used at all."""
        return True


def _limit_state(
    state: SessionRoutingState, config: CodexMemoryConfig,
) -> SessionRoutingState:
    """Apply this worker's limits also to records written by older policies."""
    return replace(
        state, pending_calls=_bounded(state.pending_calls, config.max_calls),
        pending_evidence=(
            tuple(state.pending_evidence[-config.max_calls:]) if config.max_calls else ()
        ),
        seen_events=_bounded(state.seen_events, config.max_events),
        history=_bounded(state.history, config.max_events),
        retired_generations=_bounded(state.retired_generations, config.max_events),
        seen_calls=_bounded(state.seen_calls, config.max_events),
    )


class InMemoryRoutingStateStore(RoutingStateStore):
    """
    Process-local store for isolated replay and for testing the contract.

    Explicitly *not* a production fallback: each worker would keep its own
    idea of a session, which is the failure mode shared memory exists to
    avoid.  It implements the same versioned compare-and-set semantics as the
    Redis adapter so a test of the contract means something.

    Parameters
    ----------
    config : CodexMemoryConfig
        Policy providing the TTL, the session cap and the key prefix.
    clock : callable
        Seconds-source, injectable to test expiry without sleeping.
    """

    def __init__(
        self,
        config: Optional[CodexMemoryConfig] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Store the policy and start with an empty set of sessions."""
        self._config = config or CodexMemoryConfig(backend=MEMORY_BACKEND_MEMORY)
        self._clock = clock
        self._lock = threading.Lock()
        self._records: Dict[str, Tuple[SessionRoutingState, float]] = {}
        self._fail_read = False
        self._fail_write = False

    def fail_reads(self, enabled: bool = True) -> None:
        """
        Make reads fail, to exercise the fail-open path.

        Parameters
        ----------
        enabled : bool
            Whether :meth:`read` should behave as an infrastructure failure.

        Returns
        -------
        None
        """
        self._fail_read = enabled

    def fail_writes(self, enabled: bool = True) -> None:
        """
        Make writes fail, to exercise the fail-open path.

        Parameters
        ----------
        enabled : bool
            Whether :meth:`write` should behave as an infrastructure failure.

        Returns
        -------
        None
        """
        self._fail_write = enabled

    def read(self, key: str) -> Tuple[Optional[SessionRoutingState], MemoryStatus]:
        """Return the live record under *key*, applying the TTL."""
        if self._fail_read:
            return None, MemoryStatus("unavailable", "injected read failure")
        with self._lock:
            stored = self._records.get(key)
            if stored is None:
                return None, MemoryStatus("miss")
            state, expires_at = stored
            if expires_at <= self._clock():
                del self._records[key]
                return None, MemoryStatus("expired")
            return state, MemoryStatus("hit")

    def write(
        self,
        key: str,
        state: SessionRoutingState,
        expected_version: int,
    ) -> MemoryStatus:
        """
        Store the record only while *key* still holds *expected_version*.

        The version lives inside the record — exactly as in the Redis adapter,
        where a server-side script reads it — so a test of this store exercises
        the same compare-and-set contract rather than a friendlier imitation.
        """
        if self._fail_write:
            return MemoryStatus("unavailable", "injected write failure")
        with self._lock:
            stored = self._records.get(key)
            if stored is not None and stored[1] <= self._clock():
                del self._records[key]
                stored = None
            current = stored[0].version if stored else 0
            if stored is not None and current != expected_version:
                return MemoryStatus("conflict", f"version {current}")
            if stored is None and expected_version != 0:
                return MemoryStatus("conflict", "record disappeared")
            preserve = stored is not None and all(
                getattr(stored[0], name) == getattr(state, name)
                for name in (
                    "generation", "mode", "kind", "reason", "event_id",
                    "fingerprint", "action_token", "reset",
                )
            )
            now = self._clock()
            self._records[key] = (
                _limit_state(state, self._config).set_version(
                    current + 1, stored[0].updated_at if preserve else now,
                ),
                stored[1] if preserve else now + self._config.ttl_seconds,
            )
            self._prune()
            return MemoryStatus("written")

    def clear(self, key: str) -> MemoryStatus:
        """Remove one session's record."""
        with self._lock:
            if self._records.pop(key, None) is None:
                return MemoryStatus("miss")
            return MemoryStatus("written")

    def _prune(self) -> None:
        """Enforce the session cap and drop expired records, oldest first."""
        now = self._clock()
        expired = [
            key for key, (_, expires) in self._records.items() if expires <= now
        ]
        for key in expired:
            self._records.pop(key, None)
        overflow = len(self._records) - self._config.max_sessions
        if overflow <= 0:
            return
        oldest = sorted(self._records, key=lambda key: self._records[key][1])[:overflow]
        for key in oldest:
            self._records.pop(key, None)


def build_state_store(
    config: CodexMemoryConfig,
    client: Any = None,
    logger: Any = None,
) -> Tuple[Optional[RoutingStateStore], MemoryStatus]:
    """
    Build the store the policy asks for, or explain why there is none.

    Enabled Redis memory is checked with PING before returning the store.
    A failed check logs a warning and leaves routing stateless.

    Parameters
    ----------
    config : CodexMemoryConfig
        Resolved memory policy.
    client : Any, optional
        Pre-built Redis client.  Injected clients are used as-is, which is how
        a test or an embedding application supplies its own connection.
    logger : Any, optional
        Logger for explanations and credential-masked failure tracebacks.

    Returns
    -------
    Tuple[Optional[RoutingStateStore], MemoryStatus]
        The store (or ``None``) and the reason.
    """
    if not config.enabled:
        return None, STATUS_DISABLED
    if logger is None:
        logger = logging.getLogger(__name__)
    if config.backend == MEMORY_BACKEND_MEMORY:
        logger.warning(
            "Codex routing memory backend is process-local: isolated replay "
            "and tests only; use Redis for shared production workers."
        )
        return (
            InMemoryRoutingStateStore(config),
            MemoryStatus("written", "memory backend"),
        )
    if not config.connection.configured:
        status = MemoryStatus(STATUS_UNCONFIGURED.state, "no host configured")
        if logger is not None:
            logger.warning(
                "Codex routing memory is enabled but %s is unset — routing "
                "stays stateless.",
                "REDIS_HOST",
            )
        return None, status
    redis_client = None
    try:
        import redis  # noqa: PLC0415 - optional dependency, imported on demand
    except ImportError:
        status = MemoryStatus(STATUS_UNCONFIGURED.state, "redis client not installed")
        if logger is not None:
            logger.warning(
                "Codex routing memory needs the redis client — install it or "
                "set the memory backend to a stateless configuration."
            )
        return None, status
    try:
        validate_connection(config.connection)
        redis_client = client if client is not None else redis.Redis(
            **config.connection.client_kwargs()
        )
        redis_client.ping()
    except Exception as exc:  # a bad deployment must not break routing
        status = MemoryStatus("unavailable", _reason(exc))
        if logger is not None:
            logger.warning(
                "Codex routing memory disabled: %s\n%s", _reason(exc),
                _failure_traceback(exc, config.connection),
            )
        return None, status
    finally:
        if client is None and redis_client is not None:
            try:
                redis_client.close()
                redis_client.connection_pool.disconnect()
            except Exception:
                pass
    return (
        RedisRoutingStateStore(config, client, logger=logger),
        MemoryStatus("written", "redis backend"),
    )


def _reason(exc: BaseException) -> str:
    """Return a short, credential-free description of an exception."""
    return type(exc).__name__


def _failure_traceback(exc: BaseException, connection: RedisConnectionSettings) -> str:
    """Format the exception chain and stack without Redis connection credentials."""
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    text = re.sub(r"(rediss?://)[^\s/@]+@", r"\1[REDACTED]@", text)
    for secret in (connection.password, connection.username):
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text.rstrip()


#: Compare-and-set executed server-side, so two workers cannot interleave a
#: read, a merge and a write.  The version stored inside the record is the
#: optimistic lock; the index is a per-namespace sorted set of session keys
#: scored by expiry, which is what makes a session cap possible without ever
#: scanning the keyspace.  Every key it touches is one it wrote itself.
_ATOMIC_WRITE_SCRIPT = """
local current = redis.call('GET', KEYS[1])
local version = 0
local previous = nil
if current then
  local ok, decoded = pcall(cjson.decode, current)
  if ok and type(decoded) == 'table' and decoded['ver'] then
    previous = decoded
    version = tonumber(decoded['ver']) or 0
  end
end
if version ~= tonumber(ARGV[2]) then
  return {'conflict', tostring(version)}
end
local ttl = tonumber(ARGV[3])
local payload = ARGV[1]
local incoming = cjson.decode(payload)
local preserve = previous ~= nil
for _, name in ipairs({'gen', 'mode', 'kind', 'reason', 'ev', 'fp', 'action', 'reset'}) do
  if not previous or previous[name] ~= incoming[name] then
    preserve = false
    break
  end
end
local expires = tonumber(ARGV[4])
if preserve then
  local remaining = redis.call('PTTL', KEYS[1])
  if remaining <= 0 then
    return {'conflict', tostring(version)}
  end
  local timestamp = string.match(current, '"ts"%s*:%s*([^,}]+)')
  payload = string.gsub(payload, '"ts":[^,}]+', '"ts":' .. timestamp, 1)
  redis.call('SET', KEYS[1], payload, 'PX', remaining)
  expires = tonumber(ARGV[6]) + remaining / 1000
else
  redis.call('SET', KEYS[1], payload, 'EX', ttl)
end
redis.call('ZADD', KEYS[2], expires, KEYS[1])
redis.call('EXPIRE', KEYS[2], ttl + 60)
redis.call('ZREMRANGEBYSCORE', KEYS[2], '-inf', tonumber(ARGV[6]))
local overflow = redis.call('ZCARD', KEYS[2]) - tonumber(ARGV[5])
if overflow > 0 then
  local victims = redis.call('ZRANGE', KEYS[2], 0, overflow - 1)
  for index = 1, #victims do
    if string.sub(victims[index], 1, string.len(ARGV[7])) == ARGV[7] then
      redis.call('DEL', victims[index])
    end
  end
  redis.call('ZREMRANGEBYRANK', KEYS[2], 0, overflow - 1)
end
return {'written', tostring(version + 1)}
"""


class RedisRoutingStateStore(RoutingStateStore):
    """
    Shared session memory over Redis, fail-open by construction.

    Every method swallows infrastructure errors and reports them through
    :class:`MemoryStatus`: a routing decision must never depend on a cache, and
    a Redis that is down degrades the plugin to the stateless cascade it had
    before this module existed.

    The client is created on first use rather than at plugin construction, so a
    worker forked from a preloaded application builds its own connection pool
    instead of inheriting its parent's sockets.

    Parameters
    ----------
    config : CodexMemoryConfig
        Policy providing the TTL, the caps and the key prefix.
    client : Any, optional
        Redis client to use as-is (tests, or an embedding application that
        owns its connections).  When omitted, one is built lazily from the
        connection settings.
    logger : Any, optional
        Logger for failures with exception tracebacks; defaults to this module's logger.
    """

    def __init__(
        self,
        config: CodexMemoryConfig,
        client: Any = None,
        logger: Any = None,
    ) -> None:
        """Store the policy, the client (or the means to build one) and the logger."""
        self._config = config
        self._client = client
        self._pid = os.getpid()
        self._logger = logger if logger is not None else logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._script: Any = None
        self._factory: Optional[Callable[[], Any]] = None
        if client is None:
            def _build() -> Any:
                import redis  # noqa: PLC0415 - optional dependency

                return redis.Redis(**config.connection.client_kwargs())
            self._factory = _build

    def _connection(self) -> Any:
        """
        Return the Redis client, building it once per process.

        Returns
        -------
        Any
            The client.

        Raises
        ------
        Exception
            Whatever the client constructor raises; callers translate it into
            an ``unavailable`` status.
        """
        pid = os.getpid()
        if self._pid != pid:
            self._lock = threading.Lock()
            self._script = None
            if self._factory is not None:
                self._client = None
            elif self._client is not None:
                self._client.connection_pool.reset()
            self._pid = pid
        client = self._client
        if client is not None:
            return client
        with self._lock:
            if self._client is None and self._factory is not None:
                self._client = self._factory()
            return self._client

    def _session_index(self) -> str:
        """Return the index key holding this namespace's session keys."""
        return f"{self._config.key_prefix}:v{STATE_SCHEMA_VERSION}:sessions"

    def read(self, key: str) -> Tuple[Optional[SessionRoutingState], MemoryStatus]:
        """
        Read one session's record.

        Parameters
        ----------
        key : str
            Key from :func:`session_key`.

        Returns
        -------
        Tuple[Optional[SessionRoutingState], MemoryStatus]
            ``hit`` with the record, ``miss`` when there is none, ``expired``
            when it lived here but the TTL won, and ``unavailable`` when Redis
            could not be reached.
        """
        try:
            raw = self._connection().get(key)
        except Exception as exc:
            return None, self._unavailable("read", exc)
        if raw is None:
            return None, MemoryStatus("miss")
        state = SessionRoutingState.from_json(raw)
        if state is None:
            # Unreadable or from another schema version: forget it and continue.
            self._drop(key, raw)
            return None, MemoryStatus("miss", "unreadable record")
        return _limit_state(state, self._config), MemoryStatus("hit")

    def write(
        self,
        key: str,
        state: SessionRoutingState,
        expected_version: int,
    ) -> MemoryStatus:
        """
        Store the record through an atomic, versioned update.

        Parameters
        ----------
        key : str
            Key from :func:`session_key`.
        state : SessionRoutingState
            The record to store; its version field is assigned by the store.
        expected_version : int
            Version this record was read at, ``0`` when it did not exist.

        Returns
        -------
        MemoryStatus
            ``written``, ``conflict`` when another worker moved first, or
            ``unavailable``.
        """
        now = time.time()
        payload = _limit_state(state, self._config).set_version(
            expected_version + 1, now,
        ).to_json()
        try:
            client = self._connection()
            with self._lock:
                if self._script is None:
                    self._script = client.register_script(_ATOMIC_WRITE_SCRIPT)
                script = self._script
            if not callable(script):
                return STATUS_UNAVAILABLE
            result = script(
                keys=[key, self._session_index()],
                args=[
                    payload,
                    expected_version,
                    self._config.ttl_seconds,
                    now + self._config.ttl_seconds,
                    self._config.max_sessions,
                    now,
                    f"{self._config.key_prefix}:v{STATE_SCHEMA_VERSION}:",
                ],
            )
        except Exception as exc:
            return self._unavailable("write", exc)
        status, version = _script_result(result)
        if status == "conflict":
            return MemoryStatus("conflict", f"version {version}")
        return MemoryStatus(status)

    def clear(self, key: str) -> MemoryStatus:
        """Forget one session's record and remove it from the namespace index."""
        try:
            client = self._connection()
            removed = client.eval(
                "local removed = redis.call('DEL', KEYS[1]); "
                "redis.call('ZREM', KEYS[2], KEYS[1]); return removed",
                2, key, self._session_index(),
            )
        except Exception as exc:
            return self._unavailable("clear", exc)
        return MemoryStatus("written" if removed else "miss")

    def _drop(self, key: str, raw: Any) -> None:
        """Best-effort removal of a record this version cannot read."""
        try:
            self._connection().eval(
                "if redis.call('GET', KEYS[1]) == ARGV[1] then "
                "redis.call('DEL', KEYS[1]); "
                "redis.call('ZREM', KEYS[2], KEYS[1]); return 1 end; return 0",
                2, key, self._session_index(), raw,
            )
        except Exception as exc:  # a cleanup must never surface on the routing path
            self._unavailable("cleanup", exc)

    def _unavailable(self, operation: str, exc: BaseException) -> MemoryStatus:
        """Log and wrap an infrastructure failure as an ``unavailable`` status."""
        status = MemoryStatus(STATUS_UNAVAILABLE.state, f"{operation}: {_reason(exc)}")
        if self._logger is not None:
            self._logger.warning(
                "Codex routing memory %s\n%s", status.detail,
                _failure_traceback(exc, self._config.connection),
            )
        return status


def _script_result(result: Any) -> Tuple[str, int]:
    """Normalize a Lua reply into ``(status, version)``."""
    if not isinstance(result, (list, tuple)) or not result:
        return "unavailable", 0
    status = result[0]
    if isinstance(status, bytes):
        status = status.decode("utf-8", "replace")
    version = 0
    if len(result) > 1:
        try:
            version = int(result[1])
        except (TypeError, ValueError):
            version = 0
    return (status if status in ("written", "conflict") else "unavailable", version)


def _bounded(values: Tuple[str, ...], limit: int) -> Tuple[str, ...]:
    """Keep the last *limit* entries of *values*, newest last."""
    if limit <= 0:
        return ()
    return tuple(values[-limit:])


def merge_state(
    previous: Optional[SessionRoutingState],
    generation: str,
    same_generation: bool,
) -> Optional[SessionRoutingState]:
    """
    Decide whether a remembered state still applies to this request.

    A new command resets the phase: what the agent was doing under the previous
    instruction is not evidence about the current one.  Within one generation,
    a remembered state survives while it is still fresh and comparable; a
    request carrying a different generation, or no generation at all, cannot be
    compared and therefore does not inherit anything.

    Parameters
    ----------
    previous : Optional[SessionRoutingState]
        The record read from the store, if any.
    generation : str
        ``turn_id`` of the request being routed.
    same_generation : bool
        Whether the request belongs to the same command generation as before,
        as decided by the caller from identifiers it actually has.

    Returns
    -------
    Optional[SessionRoutingState]
        The state to carry forward, or ``None``.
    """
    if previous is None or not previous.mode:
        return None
    if not generation or not same_generation:
        return None
    if previous.generation != generation:
        return None
    return previous


def record_state(
    mode: str,
    kind: str,
    reason: str,
    generation: str,
    event_id: str,
    fingerprint: str,
    previous: Optional[SessionRoutingState],
    config: CodexMemoryConfig,
) -> SessionRoutingState:
    """
    Build the record for one decision, bounded by the policy's limits.

    Parameters
    ----------
    mode : str
        Resolved work mode.
    kind : str
        Kind of evidence that decided it.
    reason : str
        Reason code of that evidence.
    generation : str
        ``turn_id`` of the command this decision belongs to.
    event_id : str
        Identifier of the payload item that carried the evidence.
    fingerprint : str
        Digest identifying this evidence, so a replayed payload can be spotted.
    previous : Optional[SessionRoutingState]
        The record being replaced, whose identifier lists are extended.
    config : CodexMemoryConfig
        Policy providing the event and call caps.

    Returns
    -------
    SessionRoutingState
        The record to store.
    """
    event_id = _identifier(event_id)
    generation = _identifier(generation)
    seen = tuple(previous.seen_events) if previous else ()
    if event_id:
        seen = tuple(entry for entry in seen if entry != event_id) + (event_id,)
    return SessionRoutingState(
        mode=mode,
        kind=kind,
        reason=reason,
        generation=generation,
        event_id=event_id,
        fingerprint=fingerprint,
        pending_calls=_bounded(
            tuple(previous.pending_calls) if previous else (), config.max_calls
        ),
        seen_events=_bounded(seen, config.max_events),
    )


def fingerprint(*parts: str) -> str:
    """
    Return a short digest identifying a piece of evidence.

    Parameters
    ----------
    *parts : str
        Structural components of the evidence — tool name, command shape,
        status — never its content.

    Returns
    -------
    str
        A hex digest prefix, stable for equal input.
    """
    digest = hashlib.sha256("\x00".join(parts).encode("utf-8"))
    return digest.hexdigest()[:_FINGERPRINT_CHARS]


def _identifier(value: str) -> str:
    """Bound wire identifiers without silently conflating long prefixes."""
    return value if len(value) <= 128 else "hash-" + fingerprint(value)


@dataclass(frozen=True)
class MemoryResolution:
    """Pure routing preview; persist only after routing has succeeded.

    ``evidence`` is safe current phase evidence, including a linked incremental
    result. ``carried`` is absent for stale/incomparable histories. ``state`` is
    the proposed bounded record, not a write. A conflict permits stateless
    routing from the payload but must not contribute memory or overwrite it.
    """

    evidence: Any = None
    carried: Optional[SessionRoutingState] = None
    state: Optional[SessionRoutingState] = None
    status: MemoryStatus = field(default_factory=lambda: MemoryStatus("miss"))


def _event_token(item: Any) -> str:
    """Identify an event without retaining its text, even without a wire ID."""
    if item.event_id:
        return fingerprint("event", item.event_id)
    return fingerprint(
        item.kind, item.event_id, item.call_id, item.name,
        "" if item.event_id else item.text,
    )


def _context_id(request: Any) -> str:
    return fingerprint(request.window_id, request.context_window_id) if (
        request.window_id or request.context_window_id
    ) else ""


def resolve_memory(
    previous: Optional[SessionRoutingState],
    request: Any,
    rules: Any,
    config: CodexMemoryConfig,
    evidence: Any = None,
) -> MemoryResolution:
    """Resolve bounded pending calls and prove history continuity before carry.

    Call before choosing phase/memory, with the record read from the store and
    ``config.phase`` rules. ``rules=None`` supports old writers which provide
    only their chosen ``evidence``; new integrations should always pass rules.
    Disjoint histories cannot be ordered using opaque event/turn identifiers.
    Only a linked pending output or an overlap at the frontier proves an
    incremental continuation. Structurally neutral assistant-only requests in
    the same context may carry memory without changing the record or its TTL.
    IDs evicted by the bound no longer prove order.
    """
    from .phase import (
        EVIDENCE_COMMAND, EVIDENCE_TEST_FAILURE, PhaseEvidence,
        _execution_status, collect_phase_evidence,
    )

    generation = _identifier(getattr(request, "turn_id", ""))
    if not generation or getattr(request, "request_class", "main") != "main":
        return MemoryResolution(status=MemoryStatus("miss", "no main generation"))
    if previous and generation in previous.retired_generations:
        return MemoryResolution(status=MemoryStatus("conflict", "retired generation"))
    same = previous is not None and previous.generation == generation
    base = previous if same else None
    unique = {}
    for item in getattr(request, "activity", ()):
        unique.setdefault(_event_token(item), item)
    tokens = tuple(unique)
    activity = tuple(unique.values())
    context = _context_id(request)
    history = base.history if base else ()
    pending = {call.call_id: call for call in base.pending_evidence} if base else {}
    start = 0
    relation = "new generation" if previous and not same else "fresh"
    if base and history:
        frontier = history[-1]
        if not tokens:
            if context != base.context_id:
                return MemoryResolution(
                    status=MemoryStatus("conflict", "context changed"),
                )
            return MemoryResolution(
                carried=base if base.mode else None, state=base,
                status=MemoryStatus("hit" if base.mode else "miss", "neutral"),
            )
        if frontier in tokens:
            start = tokens.index(frontier) + 1
            overlap = tuple(token for token in tokens[:start] if token in history)
            expected = tuple(token for token in history if token in overlap)
            if overlap != expected or any(
                token in history for token in tokens[start:]
            ):
                return MemoryResolution(
                    status=MemoryStatus("conflict", "reordered history"),
                )
            relation = "extension" if start < len(tokens) else "replay"
        elif all(token in history for token in tokens):
            return MemoryResolution(status=MemoryStatus("conflict", "stale history"))
        elif (
            context == base.context_id and rules is not None and evidence is None
            and all(item.kind == "assistant" for item in activity)
            and not any(token in history for token in tokens)
            and not collect_phase_evidence(activity, rules)
        ):
            return MemoryResolution(
                carried=base if base.mode else None, state=base,
                status=MemoryStatus("hit" if base.mode else "miss", "neutral"),
            )
        elif all(
            item.kind == "function_call_output"
            and _identifier(item.call_id) in pending
            and (
                not item.name or item.name == pending[_identifier(item.call_id)].name
            )
            for item in activity
        ):
            relation = "incremental output"
        else:
            return MemoryResolution(
                status=MemoryStatus("conflict", "incomparable history"),
            )
    elif base and not base.reset:
        return MemoryResolution(status=MemoryStatus("conflict", "no causal frontier"))

    current = base
    fresh = None
    seen = base.seen_events if base else ()
    seen_calls = base.seen_calls if base else ()
    action_token = base.action_token if base else ""
    retired = previous.retired_generations if previous else ()
    if previous and not same and previous.generation:
        retired = _bounded(retired + (previous.generation,), config.max_events)
    for item, token in zip(activity[start:], tokens[start:]):
        if token in history:
            continue
        produced = None
        if item.kind == "function_call_output":
            linked = pending.get(_identifier(item.call_id))
            if not linked or (item.name and item.name != linked.name):
                continue
            if linked:
                succeeded = _execution_status(item.text)
                if succeeded is not None:
                    pending.pop(_identifier(item.call_id))
                test_mode = rules.test_mode if rules else "test"
                if (
                    linked.token == action_token and linked.mode == test_mode
                    and linked.kind == EVIDENCE_COMMAND and succeeded is False
                ):
                    produced = PhaseEvidence(
                        mode=rules.failure_mode if rules else "debug",
                        kind=EVIDENCE_TEST_FAILURE, reason="test command failed",
                        event_id=item.event_id, call_id=linked.call_id,
                        completed=True, succeeded=False,
                    )
        else:
            call_id = _identifier(item.call_id)
            if item.kind == "function_call" and call_id and call_id in seen_calls:
                continue
            found = collect_phase_evidence((item,), rules) if rules else ()
            if found:
                produced = found[-1]
            elif evidence is not None and (
                (evidence.event_id and evidence.event_id == item.event_id)
                or (evidence.call_id and evidence.call_id == item.call_id)
                or (
                    not evidence.event_id and not evidence.call_id
                    and item.kind == "assistant"
                )
            ):
                produced = evidence
            if produced is not None:
                action_token = token
                if produced.call_id and not produced.completed:
                    pending[_identifier(produced.call_id)] = PendingRoutingCall(
                        _identifier(produced.call_id), item.name,
                        produced.mode, produced.kind,
                        produced.reason, _identifier(produced.event_id), token,
                    )
                if produced.call_id:
                    seen_calls = _bounded(
                        seen_calls + (_identifier(produced.call_id),), config.max_events,
                    )
        history = _bounded(history + (token,), config.max_events)
        if item.event_id:
            event_id = _identifier(item.event_id)
            seen = _bounded(
                tuple(v for v in seen if v != event_id) + (event_id,), config.max_events,
            )
        if produced is not None:
            fresh = produced
            current = record_state(
                produced.mode, produced.kind, produced.reason, generation,
                produced.event_id,
                fingerprint(
                    produced.mode, produced.kind, produced.reason,
                    produced.event_id, produced.call_id, str(produced.completed),
                    str(produced.succeeded), action_token,
                ),
                current, config,
            )
        pending = (
            dict(list(pending.items())[-config.max_calls:]) if config.max_calls else {}
        )
    if relation == "replay" and rules:
        found = collect_phase_evidence(activity, rules)
        fresh = found[-1] if found else None
    if current is None and previous and not same:
        current = SessionRoutingState(generation=generation, reset=True)
    if current is not None:
        current = replace(
            current, history=history, seen_events=seen,
            pending_calls=tuple(pending), pending_evidence=tuple(pending.values()),
            action_token=action_token, retired_generations=retired,
            context_id=(
                context if not base or history != base.history else base.context_id
            ),
            version=previous.version if previous else 0,
            updated_at=previous.updated_at if previous else 0.0,
            seen_calls=seen_calls,
        )
    return MemoryResolution(
        evidence=fresh, carried=base if base and base.mode else None, state=current,
        status=MemoryStatus("hit" if base else "miss", relation),
    )


def remember_decision(
    store: Optional[RoutingStateStore],
    config: CodexMemoryConfig,
    request: Any,
    decision: Any,
    rules: Any = None,
    expected_version: Optional[int] = None,
) -> str:
    """
    Write a phase decision to the session memory, if it deserves to be written.

    Only a phase decided from this turn's own evidence is carried forward.  A
    fallback, a weak semantic match and a special request are not facts about
    what the agent is doing; storing one would let a single uninformative
    request pin the mode for the rest of a command generation.  Replaying the
    same history is a no-op: identical evidence in the same generation does not
    bump the version, so a retry cannot make a session look fresher than it is.
    Changes to history or pending calls alone may bump the version, but both
    stores preserve the previous evidence time and TTL deadline atomically.

    Parameters
    ----------
    store : Optional[RoutingStateStore]
        The session memory, or ``None`` when routing is stateless.
    config : CodexMemoryConfig
        Policy providing the key prefix and the retry budget.
    request : Any
        The parsed request, read for its session identifiers.
    decision : Any
        The routing decision, read for ``source``, ``mode``, ``reason`` and
        ``evidence``.
    rules : Any
        Phase taxonomy (``routing_config.phase``). Supply it to persist all
        recognized pending calls and resolve results using configured modes.
    expected_version : Optional[int]
        Version read before classification, or zero for a miss. Passing it
        rejects a write if another request changed the record in the meantime.

    Returns
    -------
    str
        ``"written"``, ``"unchanged"``, ``"skipped"``, ``"conflict"`` or
        ``"unavailable"`` — for diagnostics, never raised.
    """
    if store is None or getattr(request, "request_class", "main") != "main":
        return "skipped"
    key = session_key(
        config.key_prefix,
        getattr(request, "session_id", ""),
        getattr(request, "thread_id", ""),
        getattr(request, "agent_name", ""),
    )
    generation = _identifier(getattr(request, "turn_id", ""))
    if key is None or not generation:
        return "skipped"
    evidence = getattr(decision, "evidence", None)
    reliable = getattr(decision, "source", "") in ("phase", "memory")
    observed_generation = None
    for attempt in range(config.max_retries + 1):
        stored, status = store.read(key)
        if status.state not in ("hit", "miss", "expired"):
            return "unavailable"
        current_generation = stored.generation if stored else ""
        if attempt and current_generation != observed_generation:
            return "conflict"
        observed_generation = current_generation
        if expected_version is not None and expected_version != (
            stored.version if stored else 0
        ):
            return "conflict"
        if not reliable and (stored is None or stored.generation == generation):
            return "skipped"
        resolved = resolve_memory(stored, request, rules, config, evidence=evidence)
        if resolved.status.state == "conflict":
            return "conflict"
        record = resolved.state
        if not reliable:
            record = SessionRoutingState(
                generation=generation, reset=True,
                retired_generations=_bounded(
                    stored.retired_generations + (stored.generation,),
                    config.max_events,
                ),
                context_id=_context_id(request),
            )
        if record is None:
            return "skipped"
        if not record.reset and record.mode != decision.mode:
            return "conflict"
        if stored == record:
            return "unchanged"
        written = store.write(
            key,
            record,
            stored.version if stored else 0,
        )
        if written.state != "conflict":
            return written.state
        if attempt >= config.max_retries:
            return "conflict"
    return "conflict"
