"""
Configuration dataclass for the Agentic routing plugin.

JSON structure::

    {
      "description": "Agentic routing configuration — ...",
      "embedding_model": "google/embeddinggemma-300m",
      "settings": {
        "trigger": ["agentic"],
        "fallback_mode": "fallback",
        "vector_store_path": "",
        "rules_enabled": true,
        "capabilities_enabled": true,
        "escalation_enabled": true,
        "session_affinity": {
          "enabled": true, "ttl_seconds": 900, "max_entries": 1024
        },
        "semantic": {
          "enabled": true,
          "threshold": 0.55,
          "top_k": 3,
          "chunk_size": 256,
          "chunk_overlap": 64
        }
      },
      "rules": [
        {
          "id": "task-coding",
          "priority": 100,
          "when": { "task": ["coding", "implement"] },
          "then": { "mode": "code" }
        }
      ],
      "agent_modes": [
        {
          "name": "plan",
          "model_name": "qwen3.6:35b",
          "description": "Agent works in planning mode: ...",
          "examples": ["Plan the migration of this service...", ...],
          "keywords": ["planning", "roadmap", ...],
          "phrases": ["plan działania:5", ...],
          "patterns": ["\\\\bplan\\\\b", ...],
          "weights": { "planning": 3, "roadmap": 3, ... },
          "capabilities": {
            "tool_calling": true, "reasoning": true, "context_window": 131072
          }
        }
      ]
    }
"""

import logging
import os
import pathlib

from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from llm_router_plugins.utils.routing.constants import AGENTIC_ROUTING_PREFIX
from llm_router_plugins.utils.routing.common import (
    RoutingConfigBase,
    env_bool,
    env_float,
    env_int,
    resolve_persist_dir,
)
from llm_router_plugins.utils.routing.target import RoutingTarget
from llm_router_plugins.utils.routing.agentic_routing.general.rules import (
    RoutingRule,
    parse_rules,
)
from llm_router_plugins.utils.routing.agentic_routing.general.session_affinity import (
    SessionAffinitySettings,
)

# Re-exported for backward compatibility — ``RoutingTarget`` is now a shared
# routing concept (see ``llm_router_plugins.utils.routing.target``).
__all__ = [
    "RoutingTarget",
    "AgenticRoutingConfig",
    "AgentMode",
    "RoutingRule",
    "SessionAffinitySettings",
]


def _session_affinity_from_settings(
    settings: Dict[str, Any],
) -> SessionAffinitySettings:
    """
    Build :class:`SessionAffinitySettings` from the raw ``settings`` mapping.

    Parameters
    ----------
    settings : Dict[str, Any]
        The raw ``settings`` object of the JSON config.

    Returns
    -------
    SessionAffinitySettings
        The parsed settings; defaults are used when the key is absent.

    Raises
    ------
    ValueError
        If ``settings.session_affinity`` is present but is not an object.
    """
    raw = settings.get("session_affinity")
    if raw is None:
        return SessionAffinitySettings()
    if not isinstance(raw, dict):
        raise ValueError(
            "AgenticRouting: 'settings.session_affinity' must be an object, "
            f"got {type(raw).__name__} — check 'settings.session_affinity' in "
            "the JSON config"
        )
    return SessionAffinitySettings(
        enabled=bool(raw.get("enabled", True)),
        ttl_seconds=int(raw.get("ttl_seconds", SessionAffinitySettings.ttl_seconds)),
        max_entries=int(raw.get("max_entries", SessionAffinitySettings.max_entries)),
    )


@dataclass
class AgenticRoutingConfig(RoutingConfigBase):
    """
    Snapshot of agentic routing configuration.

    Holds the agent work-mode definitions loaded from the JSON config file
    together with the trigger, fallback and semantic-detection settings.  The
    dataclass is mutable on purpose: :py:meth:`_override_from_env` applies
    environment variable overrides in place before the config is consumed.

    Parameters
    ----------
    trigger : List[str]
        Values of ``payload["model"]`` that activate the plugin.
    fallback_mode : str
        Name of the mode used when no other mode can be detected.
    vector_store_path : str or None
        Directory path for persisting the FAISS index and doc_store.
        If ``None``, the index is kept in memory only.
    semantic_enabled : bool
        Whether the semantic (biencoder + FAISS) detection layer is active.
    similarity_threshold : float
        Minimum cosine similarity required for a semantic match to be accepted.
    top_k : int
        Number of nearest neighbours to retrieve during routing queries.
    chunk_size : int
        Number of tokens per chunk when splitting mode text.
    chunk_overlap : int
        Number of tokens overlapping between adjacent chunks.
    embedding_model : str
        The HuggingFace model identifier used to compute embeddings.
    agent_modes : Tuple[AgentMode, ...]
        Immutable sequence of :class:`AgentMode` dataclasses, one per work mode.
    rules : Tuple[RoutingRule, ...]
        Declarative deterministic rules, already sorted by descending priority.
    rules_enabled : bool
        Whether the declarative rule layer participates in the cascade.
    capabilities_enabled : bool
        Whether declared mode capabilities gate the selected mode.
    escalation_enabled : bool
        Whether a mode lacking a required capability may be escalated to a
        more capable mode.
    session_affinity : SessionAffinitySettings
        Settings of the session-affinity (sticky session) layer.
    """

    # RoutingConfigBase hooks (ClassVar — not dataclass fields)
    _ENV_PREFIX: ClassVar[str] = AGENTIC_ROUTING_PREFIX
    _DEFAULT_CONFIG_PATH: ClassVar[Optional[pathlib.Path]] = (
        pathlib.Path(__file__).resolve().parent.parent.parent.parent.parent
        / "resources"
        / "routing"
        / "agentic_routing.json"
    )

    trigger: List[str]
    fallback_mode: str
    vector_store_path: Optional[str]
    semantic_enabled: bool
    similarity_threshold: float
    top_k: int
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    agent_modes: Tuple["AgentMode", ...]
    rules: Tuple[RoutingRule, ...] = ()
    rules_enabled: bool = True
    capabilities_enabled: bool = True
    escalation_enabled: bool = True
    session_affinity: SessionAffinitySettings = field(
        default_factory=SessionAffinitySettings
    )

    @property
    def mode_names(self) -> List[str]:
        """
        Return the names of all configured agent work modes.

        Returns
        -------
        List[str]
            A list of mode name strings, in the order defined in config.
        """
        return [m.name for m in self.agent_modes]

    @property
    def mode_by_name(self) -> Dict[str, "AgentMode"]:
        """
        Return a mapping from mode name to :class:`AgentMode`.

        Returns
        -------
        Dict[str, AgentMode]
            A dictionary mapping each mode name to its configuration object.
        """
        return {m.name: m for m in self.agent_modes}

    def override_from_env(self, logger: Optional[logging.Logger] = None) -> None:
        self._override_from_env(logger=logger)

    def validate_args(self) -> None:
        self._validate_args()

    @classmethod
    def _from_raw(cls, raw: Dict[str, Any]) -> "AgenticRoutingConfig":
        """Parse and validate the decoded JSON dict (``RoutingConfigBase`` hook)."""
        for required_key in ("settings", "agent_modes"):
            if required_key not in raw:
                raise KeyError(
                    f"Missing required top-level key '{required_key}' in config. "
                    f"Available keys: {list(raw.keys())}"
                )

        settings = raw["settings"]
        for setting_key in ("trigger", "fallback_mode", "semantic"):
            if setting_key not in settings:
                raise KeyError(
                    f"Missing required field '{setting_key}' in settings. "
                    f"Available fields: {list(settings.keys())}"
                )

        semantic = settings["semantic"]
        for semantic_key in (
            "enabled",
            "threshold",
            "top_k",
            "chunk_size",
            "chunk_overlap",
        ):
            if semantic_key not in semantic:
                raise KeyError(
                    f"Missing required field '{semantic_key}' in settings.semantic. "
                    f"Available fields: {list(semantic.keys())}"
                )

        for idx, mode in enumerate(raw["agent_modes"]):
            for key in ("name", "model_name", "description"):
                if key not in mode:
                    raise KeyError(
                        f"Missing required field '{key}' in agent_modes[{idx}]. "
                        f"Available fields: {list(mode.keys())}"
                    )

        agent_modes = tuple(
            AgentMode(
                name=m["name"],
                model_name=m["model_name"],
                description=m["description"],
                examples=tuple(m.get("examples", [])),
                keywords=list(m.get("keywords", [])),
                phrases=list(m.get("phrases", [])),
                patterns=list(m.get("patterns", [])),
                weights=m.get("weights", {}) or {},
                capabilities=m.get("capabilities", {}) or {},
            )
            for m in raw["agent_modes"]
        )

        chunk_size = semantic["chunk_size"]
        chunk_overlap = semantic["chunk_overlap"]
        top_k = semantic["top_k"]
        RoutingConfigBase.validate_semantic_params(chunk_size, chunk_overlap, top_k)

        return AgenticRoutingConfig(
            trigger=list(settings["trigger"]),
            fallback_mode=settings["fallback_mode"],
            vector_store_path=raw.get("vector_store_path")
            or settings.get("vector_store_path"),
            semantic_enabled=bool(semantic["enabled"]),
            similarity_threshold=float(semantic["threshold"]),
            top_k=top_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=raw.get("embedding_model", ""),
            agent_modes=agent_modes,
            rules=parse_rules(raw.get("rules", []), [m.name for m in agent_modes]),
            rules_enabled=bool(settings.get("rules_enabled", True)),
            capabilities_enabled=bool(settings.get("capabilities_enabled", True)),
            escalation_enabled=bool(settings.get("escalation_enabled", True)),
            session_affinity=_session_affinity_from_settings(settings),
        )

    def _override_from_env(self, logger: Optional[logging.Logger] = None) -> None:
        """
        Apply environment variable overrides to the config **in place**.

        Supported environment variables (prefix
        ``LLM_ROUTER_ROUTING_AGENTIC_``):

        - ``TRIGGER`` — pipe-separated list of ``payload["model"]`` triggers
        - ``MODEL`` — override the embedding model name
        - ``MODELS`` — per-mode model mapping, e.g. ``plan=model_a|code=model_b``
        - ``MODES`` — pipe-separated whitelist of mode names
        - ``SEMANTIC_ENABLED`` — ``1/0``, ``true/false``, ``yes/no``, ``on/off``
        - ``SIMILARITY_THRESHOLD`` — override the semantic similarity threshold
        - ``TOP_K`` / ``CHUNK_SIZE`` / ``CHUNK_OVERLAP`` — chunking/retrieval params
        - ``PERSIST_DIR`` — directory for FAISS index persistence
        - ``FALLBACK_MODE`` — override the fallback mode name
        - ``RULES_ENABLED`` / ``CAPABILITIES_ENABLED`` / ``ESCALATION_ENABLED``
          — enable/disable the deterministic cascade layers
        - ``SESSION_AFFINITY_ENABLED`` — enable/disable session affinity
        - ``SESSION_TTL_SECONDS`` / ``SESSION_MAX_ENTRIES`` — affinity cache bounds
        - ``MODE_<name>_KEYWORDS`` — pipe-separated keyword list for one mode

        Unknown mode names are logged as warnings and otherwise ignored.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger instance used to report applied overrides.

        Returns
        -------
        None
        """
        trigger_env = os.getenv(f"{AGENTIC_ROUTING_PREFIX}TRIGGER")
        if trigger_env:
            triggers = [t.strip() for t in trigger_env.split("|") if t.strip()]
            if triggers:
                self.trigger = triggers
                if logger:
                    logger.info("Overriding trigger: %s", trigger_env)

        model_env = os.getenv(f"{AGENTIC_ROUTING_PREFIX}MODEL")
        if model_env:
            self.embedding_model = model_env
            if logger:
                logger.info("Overriding embedding model: %s", model_env)

        models_env = os.getenv(f"{AGENTIC_ROUTING_PREFIX}MODELS")
        if models_env:
            mapping: Dict[str, str] = {}
            for pair in models_env.split("|"):
                if "=" not in pair:
                    continue
                mode_name, _, model_name = pair.partition("=")
                mode_name, model_name = mode_name.strip().lower(), model_name.strip()
                if mode_name and model_name:
                    mapping[mode_name] = model_name
            known = self.mode_by_name
            for mode_name, model_name in mapping.items():
                mode = known.get(mode_name)
                if mode is None:
                    if logger:
                        logger.warning(
                            "Ignoring MODELS override for unknown mode '%s'",
                            mode_name,
                        )
                    continue
                self.agent_modes = tuple(
                    replace(m, model_name=model_name) if m.name == mode_name else m
                    for m in self.agent_modes
                )
                if logger:
                    logger.info(
                        "Overriding model for mode '%s': %s",
                        mode_name,
                        model_name,
                    )

        modes_env = os.getenv(f"{AGENTIC_ROUTING_PREFIX}MODES")
        if modes_env:
            allowed = set(self.mode_names)
            selected = [
                m.strip()
                for m in modes_env.split("|")
                if m.strip() and m.strip() in allowed
            ]
            if selected and selected != self.mode_names:
                self.agent_modes = tuple(
                    m for m in self.agent_modes if m.name in selected
                )
                kept_modes = set(selected)
                dropped = [r.id for r in self.rules if r.mode not in kept_modes]
                if dropped:
                    self.rules = tuple(r for r in self.rules if r.mode in kept_modes)
                    if logger:
                        logger.warning(
                            "Dropped rules referencing removed modes: %s",
                            ", ".join(dropped),
                        )
                if logger:
                    logger.info(
                        "Overriding agent modes: %s",
                        "|".join(selected),
                    )

        semantic_enabled = env_bool(
            AGENTIC_ROUTING_PREFIX, "SEMANTIC_ENABLED", logger
        )
        if semantic_enabled is not None:
            self.semantic_enabled = semantic_enabled

        threshold = env_float(AGENTIC_ROUTING_PREFIX, "SIMILARITY_THRESHOLD")
        if threshold is not None:
            self.similarity_threshold = threshold

        for env_suffix, field_name in (
            ("TOP_K", "top_k"),
            ("CHUNK_SIZE", "chunk_size"),
            ("CHUNK_OVERLAP", "chunk_overlap"),
        ):
            env_value = env_int(AGENTIC_ROUTING_PREFIX, env_suffix)
            if env_value is not None:
                setattr(self, field_name, env_value)

        persist_dir = resolve_persist_dir(
            AGENTIC_ROUTING_PREFIX, self.vector_store_path, logger
        )
        if persist_dir is not None:
            self.vector_store_path = persist_dir

        fallback_env = os.getenv(f"{AGENTIC_ROUTING_PREFIX}FALLBACK_MODE")
        if fallback_env:
            self.fallback_mode = fallback_env.strip()
            if logger:
                logger.info(
                    "Overriding fallback mode: %s",
                    self.fallback_mode,
                )

        for flag_suffix, attr_name in (
            ("RULES_ENABLED", "rules_enabled"),
            ("CAPABILITIES_ENABLED", "capabilities_enabled"),
            ("ESCALATION_ENABLED", "escalation_enabled"),
        ):
            flag = env_bool(AGENTIC_ROUTING_PREFIX, flag_suffix, logger)
            if flag is not None:
                setattr(self, attr_name, flag)
                if logger:
                    logger.info(
                        "Overriding %s: %s",
                        flag_suffix.lower(),
                        flag,
                    )

        affinity_enabled = env_bool(
            AGENTIC_ROUTING_PREFIX, "SESSION_AFFINITY_ENABLED", logger
        )
        if affinity_enabled is not None:
            self.session_affinity = replace(
                self.session_affinity, enabled=affinity_enabled
            )

        affinity_overrides: List[str] = []
        ttl_seconds = env_int(AGENTIC_ROUTING_PREFIX, "SESSION_TTL_SECONDS")
        if ttl_seconds is not None:
            if ttl_seconds < 1:
                if logger:
                    logger.warning(
                        "Ignoring %sSESSION_TTL_SECONDS: value must be >= 1, got %s",
                        AGENTIC_ROUTING_PREFIX,
                        ttl_seconds,
                    )
            else:
                self.session_affinity = replace(
                    self.session_affinity, ttl_seconds=ttl_seconds
                )
                affinity_overrides.append(f"ttl_seconds={ttl_seconds}")

        max_entries = env_int(AGENTIC_ROUTING_PREFIX, "SESSION_MAX_ENTRIES")
        if max_entries is not None:
            if max_entries < 1:
                if logger:
                    logger.warning(
                        "Ignoring %sSESSION_MAX_ENTRIES: value must be >= 1, got %s",
                        AGENTIC_ROUTING_PREFIX,
                        max_entries,
                    )
            else:
                self.session_affinity = replace(
                    self.session_affinity, max_entries=max_entries
                )
                affinity_overrides.append(f"max_entries={max_entries}")

        if affinity_overrides and logger:
            logger.info(
                "Overriding session affinity: %s",
                ", ".join(affinity_overrides),
            )

        keywords_prefix = f"{AGENTIC_ROUTING_PREFIX}MODE_"
        for env_name, env_keywords in os.environ.items():
            if not env_name.startswith(keywords_prefix) or not env_name.endswith(
                "_KEYWORDS"
            ):
                continue
            mode_name = env_name[len(keywords_prefix) : -len("_KEYWORDS")].lower()
            mode = self.mode_by_name.get(mode_name)
            if mode is None:
                if logger:
                    logger.warning(
                        "Ignoring %s override for unknown mode '%s'",
                        env_name,
                        mode_name,
                    )
                continue
            keywords = [k.strip() for k in env_keywords.split("|") if k.strip()]
            self.agent_modes = tuple(
                replace(m, keywords=keywords) if m.name == mode_name else m
                for m in self.agent_modes
            )
            if logger:
                logger.info(
                    "Overriding keywords for mode '%s': %s",
                    mode_name,
                    env_keywords,
                )

    def _validate_args(self) -> None:
        """
        Validate that all required agentic routing configuration is present
        after env overrides.

        Raises
        ------
        ValueError
            If the trigger, the mode list, or any mode definition is missing,
            duplicated, or inconsistent with ``fallback_mode``.
        """
        if not self.trigger:
            raise ValueError(
                "AgenticRouting: no trigger configured — "
                "check 'settings.trigger' in the JSON config or the "
                f"{AGENTIC_ROUTING_PREFIX}TRIGGER environment variable"
            )

        if not self.agent_modes:
            raise ValueError(
                "AgenticRouting: no agent modes defined — "
                "check 'agent_modes' in the JSON config or the "
                f"{AGENTIC_ROUTING_PREFIX}MODES environment variable"
            )

        names = self.mode_names
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"AgenticRouting: duplicate agent mode names {duplicates} — "
                "check 'agent_modes' in the JSON config"
            )

        for mode in self.agent_modes:
            if not mode.model_name:
                raise ValueError(
                    f"AgenticRouting: mode '{mode.name}' has no model_name — "
                    "check 'agent_modes[].model_name' in the JSON config or the "
                    f"{AGENTIC_ROUTING_PREFIX}MODELS environment variable"
                )

        if self.fallback_mode not in self.mode_by_name:
            raise ValueError(
                f"AgenticRouting: fallback_mode '{self.fallback_mode}' is not a "
                "defined agent mode — check 'settings.fallback_mode' in the JSON "
                f"config or the {AGENTIC_ROUTING_PREFIX}FALLBACK_MODE "
                "environment variable"
            )

        if self.semantic_enabled and not self.embedding_model:
            raise ValueError(
                "AgenticRouting: no embedding_model configured — "
                "check 'embedding_model' in the JSON config or the "
                f"{AGENTIC_ROUTING_PREFIX}MODEL environment variable"
            )

        if self.chunk_size <= 0:
            raise ValueError(
                f"AgenticRouting: chunk_size must be > 0, got {self.chunk_size}"
            )

        if self.chunk_overlap < 0:
            raise ValueError(
                f"AgenticRouting: chunk_overlap must be >= 0, got "
                f"{self.chunk_overlap}"
            )

        if self.top_k < 1:
            raise ValueError(f"AgenticRouting: top_k must be >= 1, got {self.top_k}")

        known = self.mode_by_name
        dangling = sorted(r.id for r in self.rules if r.mode not in known)
        if dangling:
            raise ValueError(
                f"AgenticRouting: rules {dangling} reference unknown agent modes — "
                "check 'rules[].then.mode' in the JSON config against the "
                f"defined modes {sorted(known)}"
            )

        if self.session_affinity.ttl_seconds < 1:
            raise ValueError(
                "AgenticRouting: settings.session_affinity.ttl_seconds must be "
                f">= 1, got {self.session_affinity.ttl_seconds}"
            )

        if self.session_affinity.max_entries < 1:
            raise ValueError(
                "AgenticRouting: settings.session_affinity.max_entries must be "
                f">= 1, got {self.session_affinity.max_entries}"
            )


@dataclass(frozen=True)
class AgentMode(RoutingTarget):
    """
    Definition of a single agent work mode.

    Extends the shared :class:`RoutingTarget` (name, model, description,
    examples) with the signals used by the heuristic detection layer.

    Parameters
    ----------
    name : str
        Unique identifier of the mode (used in ``agent_mode`` in results).
    model_name : str
        The model name to select when this mode is the best match.
    description : str
        Human-readable description used for embedding.
    examples : Tuple[str, ...]
        Example prompts representative of this mode, used for embedding.
    keywords : List[str]
        Keywords for the heuristic scorer (substring matches).
    phrases : List[str]
        Multi-word expressions for the heuristic scorer, optionally suffixed
        with ``":weight"`` (default weight 2.0).
    patterns : List[str]
        Regex patterns for the heuristic scorer (each match adds 3.0).
    weights : Dict[str, Any]
        Per-keyword weights overriding the default keyword weight of 1.
    capabilities : Dict[str, Any]
        Declared model capabilities of this mode, e.g. ``tool_calling``,
        ``reasoning``, ``vision``, ``structured_output``, ``parallel_tools``
        (booleans) and ``context_window`` (int tokens).
    """

    keywords: List[str] = field(default_factory=list)
    phrases: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    weights: Dict[str, Any] = field(default_factory=dict)
    capabilities: Dict[str, Any] = field(default_factory=dict)
