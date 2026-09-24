"""
Configuration dataclass for the Codex routing plugin.

JSON structure::

    {
      "description": "Codex routing configuration — ...",
      "embedding_model": "google/embeddinggemma-300m",
      "settings": {
        "trigger_model": "auto_codex",
        "fallback_mode": "implement",
        "heuristic_enabled": true,
        "heuristic_min_score": 3.0,
        "classify_max_chars": 4000,
        "vector_store_path": "",
        "semantic": {
          "enabled": true,
          "threshold": 0.51,
          "top_k": 3,
          "chunk_size": 256,
          "chunk_overlap": 64
        }
      },
      "codex_modes": [
        {
          "name": "plan",
          "model_name": "qwen/Qwen3.8-Flash-Next",
          "description": "The agent plans before touching any file ...",
          "examples": ["Zaplanuj migrację tej usługi...", ...],
          "keywords": ["plan", "roadmap", ...],
          "phrases": ["plan działania:3", ...],
          "patterns": ["\\bplan\\b", ...],
          "weights": { "plan": 2, "roadmap": 2, ... }
        }
      ]
    }
"""

import logging
import os
import re
import pathlib

from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from llm_router_plugins.utils.routing.agentic_routing.codex.payload import (
    DEFAULT_CLASSIFY_MAX_CHARS,
)
from llm_router_plugins.utils.routing.constants import AGENTIC_CODEX_ROUTING_PREFIX
from llm_router_plugins.utils.routing.common import (
    RoutingConfigBase,
    env_bool,
    env_float,
    env_int,
    resolve_persist_dir,
)
from llm_router_plugins.utils.routing.target import RoutingTarget

# Re-exported for convenience — ``RoutingTarget`` is a shared routing concept
# (see ``llm_router_plugins.utils.routing.target``).
__all__ = [
    "RoutingTarget",
    "CodexRoutingConfig",
    "CodexMode",
]

#: Upper-case letters, which a pattern keeps but the scored text does not.
_UPPERCASE_RE = re.compile("[A-Z]")


@dataclass
class CodexRoutingConfig(RoutingConfigBase):
    """
    Snapshot of Codex routing configuration.

    Holds the work-mode definitions loaded from the JSON config file together
    with the trigger, fallback and heuristic-detection settings.  The dataclass
    is mutable on purpose: :py:meth:`_override_from_env` applies environment
    variable overrides in place before the config is consumed.

    Parameters
    ----------
    trigger_model : str
        Value of ``payload["model"]`` that activates the plugin.
    fallback_mode : str
        Name of the mode used when no other mode can be detected.
    heuristic_enabled : bool
        Whether the keyword layer participates in the decision cascade.
    heuristic_min_score : float
        Minimum keyword score required to accept a heuristic match.
    vector_store_path : str or None
        Directory path for persisting the FAISS index and doc store.
        If ``None``, the index is kept in memory only.
    semantic_enabled : bool
        Whether the semantic (biencoder + FAISS) similarity layer is active.
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
    codex_modes : Tuple[CodexMode, ...]
        Immutable sequence of :class:`CodexMode` dataclasses, one per work mode.
    classify_max_chars : int
        Character budget of the text the keyword and semantic layers see: the
        newest user message is always kept whole, older ones are appended while
        the budget holds.
    """

    # RoutingConfigBase hooks (ClassVar — not dataclass fields)
    _ENV_PREFIX: ClassVar[str] = AGENTIC_CODEX_ROUTING_PREFIX
    _DEFAULT_CONFIG_PATH: ClassVar[Optional[pathlib.Path]] = (
        pathlib.Path(__file__).resolve().parents[4]
        / "resources"
        / "routing"
        / "agentic_routing_codex.json"
    )

    trigger_model: str
    fallback_mode: str
    heuristic_enabled: bool
    heuristic_min_score: float
    vector_store_path: Optional[str]
    semantic_enabled: bool
    similarity_threshold: float
    top_k: int
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    codex_modes: Tuple["CodexMode", ...]
    classify_max_chars: int = DEFAULT_CLASSIFY_MAX_CHARS

    @property
    def mode_names(self) -> List[str]:
        """
        Return the names of all configured Codex work modes.

        Returns
        -------
        List[str]
            A list of mode name strings, in the order defined in config.
        """
        return [mode.name for mode in self.codex_modes]

    @property
    def mode_by_name(self) -> Dict[str, "CodexMode"]:
        """
        Return a mapping from mode name to :class:`CodexMode`.

        Returns
        -------
        Dict[str, CodexMode]
            A dictionary mapping each mode name to its configuration object.
        """
        return {mode.name: mode for mode in self.codex_modes}

    def override_from_env(self, logger: Optional[logging.Logger] = None) -> None:
        self._override_from_env(logger=logger)

    def validate_args(self) -> None:
        self._validate_args()

    @classmethod
    def _from_raw(cls, raw: Dict[str, Any]) -> "CodexRoutingConfig":
        """Parse and validate the decoded JSON dict (``RoutingConfigBase`` hook)."""
        for required_key in ("settings", "codex_modes"):
            if required_key not in raw:
                raise KeyError(
                    f"Missing required top-level key '{required_key}' in config. "
                    f"Available keys: {list(raw.keys())}"
                )

        settings = raw["settings"]
        for setting_key in ("trigger_model", "fallback_mode"):
            if setting_key not in settings:
                raise KeyError(
                    f"Missing required field '{setting_key}' in settings. "
                    f"Available fields: {list(settings.keys())}"
                )

        for idx, mode in enumerate(raw["codex_modes"]):
            for key in ("name", "model_name", "description"):
                if key not in mode:
                    raise KeyError(
                        f"Missing required field '{key}' in codex_modes[{idx}]. "
                        f"Available fields: {list(mode.keys())}"
                    )

        codex_modes = tuple(
            CodexMode(
                name=m["name"],
                model_name=m["model_name"],
                description=m["description"],
                examples=tuple(m.get("examples", [])),
                keywords=tuple(m.get("keywords", [])),
                phrases=tuple(m.get("phrases", [])),
                patterns=tuple(m.get("patterns", [])),
                weights=m.get("weights", {}) or {},
            )
            for m in raw["codex_modes"]
        )

        semantic = settings.get("semantic", {}) or {}
        chunk_size = int(semantic.get("chunk_size", 256))
        chunk_overlap = int(semantic.get("chunk_overlap", 64))
        top_k = int(semantic.get("top_k", 3))
        RoutingConfigBase.validate_semantic_params(chunk_size, chunk_overlap, top_k)

        return CodexRoutingConfig(
            trigger_model=str(settings["trigger_model"]),
            fallback_mode=str(settings["fallback_mode"]),
            heuristic_enabled=bool(settings.get("heuristic_enabled", True)),
            heuristic_min_score=float(settings.get("heuristic_min_score", 3.0)),
            vector_store_path=raw.get("vector_store_path")
            or settings.get("vector_store_path"),
            semantic_enabled=bool(semantic.get("enabled", True)),
            similarity_threshold=float(semantic.get("threshold", 0.51)),
            top_k=top_k,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=str(raw.get("embedding_model", "") or ""),
            codex_modes=codex_modes,
            classify_max_chars=int(
                settings.get("classify_max_chars", DEFAULT_CLASSIFY_MAX_CHARS)
            ),
        )

    def _override_from_env(self, logger: Optional[logging.Logger] = None) -> None:
        """
        Apply environment variable overrides to the config **in place**.

        Supported environment variables (prefix
        ``LLM_ROUTER_ROUTING_SEMANTIC_AGENTIC_CODEX_``):

        - ``CONFIG`` — the full config, as a raw JSON string or a file path
        - ``TRIGGER`` — the single ``payload["model"]`` trigger value
        - ``MODEL_<MODE>`` — per-mode model override, e.g. ``MODEL_PLAN``
        - ``MODELS`` — per-mode model mapping, e.g. ``plan=model_a|test=model_b``
        - ``MODES`` — pipe-separated whitelist of mode names
        - ``FALLBACK_MODE`` — override the fallback mode name
        - ``HEURISTIC_ENABLED`` — ``1/0``, ``true/false``, ``yes/no``, ``on/off``
        - ``HEURISTIC_MIN_SCORE`` — override the heuristic acceptance threshold
        - ``CLASSIFY_MAX_CHARS`` — character budget of the classified user text
        - ``MODEL`` — embedding model used by the semantic similarity layer
        - ``SEMANTIC_ENABLED`` — turn the semantic similarity layer on/off
        - ``SIMILARITY_THRESHOLD`` — minimum cosine similarity to accept a match
        - ``TOP_K``/``CHUNK_SIZE``/``CHUNK_OVERLAP`` — embedding router knobs
        - ``PERSIST_DIR`` — directory holding the persisted FAISS index
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
        trigger_env = os.getenv(f"{AGENTIC_CODEX_ROUTING_PREFIX}TRIGGER")
        if trigger_env and trigger_env.strip():
            self.trigger_model = trigger_env.strip()
            if logger:
                logger.info(
                    "Overriding trigger model: %s",
                    self.trigger_model,
                )

        for mode in self.codex_modes:
            model_env = os.getenv(
                f"{AGENTIC_CODEX_ROUTING_PREFIX}MODEL_{mode.name.upper()}"
            )
            if model_env and model_env.strip():
                self._replace_mode(
                    mode.name,
                    logger,
                    "model",
                    model_name=model_env.strip(),
                )

        models_env = os.getenv(f"{AGENTIC_CODEX_ROUTING_PREFIX}MODELS")
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
                if mode_name not in known:
                    if logger:
                        logger.warning(
                            "Ignoring MODELS override for unknown mode '%s'",
                            mode_name,
                        )
                    continue
                self._replace_mode(
                    mode_name,
                    logger,
                    "model",
                    model_name=model_name,
                )

        modes_env = os.getenv(f"{AGENTIC_CODEX_ROUTING_PREFIX}MODES")
        if modes_env:
            allowed = set(self.mode_names)
            selected = [
                name.strip()
                for name in modes_env.split("|")
                if name.strip() and name.strip() in allowed
            ]
            if selected and selected != self.mode_names:
                self.codex_modes = tuple(
                    mode for mode in self.codex_modes if mode.name in selected
                )
                if logger:
                    logger.info(
                        "Overriding agent modes: %s",
                        "|".join(selected),
                    )

        fallback_env = os.getenv(f"{AGENTIC_CODEX_ROUTING_PREFIX}FALLBACK_MODE")
        if fallback_env and fallback_env.strip():
            self.fallback_mode = fallback_env.strip()
            if logger:
                logger.info(
                    "Overriding fallback mode: %s",
                    self.fallback_mode,
                )

        heuristic_enabled = env_bool(
            AGENTIC_CODEX_ROUTING_PREFIX, "HEURISTIC_ENABLED", logger
        )
        if heuristic_enabled is not None:
            self.heuristic_enabled = heuristic_enabled

        min_score = env_float(AGENTIC_CODEX_ROUTING_PREFIX, "HEURISTIC_MIN_SCORE")
        if min_score is not None:
            self.heuristic_min_score = min_score

        classify_max_chars = env_int(
            AGENTIC_CODEX_ROUTING_PREFIX, "CLASSIFY_MAX_CHARS"
        )
        if classify_max_chars is not None:
            self.classify_max_chars = classify_max_chars

        model_env = os.getenv(f"{AGENTIC_CODEX_ROUTING_PREFIX}MODEL")
        if model_env and model_env.strip():
            self.embedding_model = model_env.strip()
            if logger:
                logger.info(
                    "Overriding embedding model: %s",
                    self.embedding_model,
                )

        semantic_enabled = env_bool(
            AGENTIC_CODEX_ROUTING_PREFIX, "SEMANTIC_ENABLED", logger
        )
        if semantic_enabled is not None:
            self.semantic_enabled = semantic_enabled

        threshold = env_float(AGENTIC_CODEX_ROUTING_PREFIX, "SIMILARITY_THRESHOLD")
        if threshold is not None:
            self.similarity_threshold = threshold

        for env_suffix, attr in (
            ("TOP_K", "top_k"),
            ("CHUNK_SIZE", "chunk_size"),
            ("CHUNK_OVERLAP", "chunk_overlap"),
        ):
            env_value = env_int(AGENTIC_CODEX_ROUTING_PREFIX, env_suffix)
            if env_value is not None:
                setattr(self, attr, env_value)

        persist_dir = resolve_persist_dir(
            AGENTIC_CODEX_ROUTING_PREFIX, self.vector_store_path, logger
        )
        if persist_dir is not None:
            self.vector_store_path = persist_dir

        keywords_prefix = f"{AGENTIC_CODEX_ROUTING_PREFIX}MODE_"
        for env_name, env_keywords in os.environ.items():
            if not env_name.startswith(keywords_prefix) or not env_name.endswith(
                "_KEYWORDS"
            ):
                continue
            mode_name = env_name[len(keywords_prefix) : -len("_KEYWORDS")].lower()
            if mode_name not in self.mode_by_name:
                if logger:
                    logger.warning(
                        "Ignoring %s override for unknown mode '%s'",
                        env_name,
                        mode_name,
                    )
                continue
            keywords = tuple(k.strip() for k in env_keywords.split("|") if k.strip())
            self._replace_mode(
                mode_name,
                logger,
                "keywords",
                keywords=keywords,
            )

    def _replace_mode(
        self,
        mode_name: str,
        logger: Optional[logging.Logger],
        label: str,
        **changes: Any,
    ) -> None:
        """
        Replace one mode in :attr:`codex_modes` and log the applied override.

        Parameters
        ----------
        mode_name : str
            Name of the mode to rebuild; must already be configured.
        logger : logging.Logger, optional
            Logger instance used to report the applied override.
        label : str
            What was overridden, e.g. ``"model"`` or ``"keywords"``; reported
            with the mode name and the new value.
        **changes : Any
            Fields to overwrite on the mode (a single field per call).

        Returns
        -------
        None

        Raises
        ------
        None
        """
        self.codex_modes = tuple(
            replace(mode, **changes) if mode.name == mode_name else mode
            for mode in self.codex_modes
        )
        if logger:
            logger.info(
                "Overriding %s for mode '%s': %s",
                label,
                mode_name,
                next(iter(changes.values())),
            )

    def _validate_args(self) -> None:
        """
        Validate that all required Codex routing configuration is present
        after env overrides.

        A mode with an empty ``model_name`` is *valid*: the plugin passes such
        requests through untouched instead of rewriting them.

        Raises
        ------
        ValueError
            If the trigger is missing, no mode is defined, a mode name is
            duplicated, ``fallback_mode`` does not name a configured mode,
            semantic similarity is enabled without an embedding model, the
            embedding router parameters are out of range, or the classified
            text budget is not positive.
        """
        if not self.trigger_model:
            raise ValueError(
                "CodexRouting: no trigger configured — "
                "check 'settings.trigger_model' in the JSON config or the "
                f"{AGENTIC_CODEX_ROUTING_PREFIX}TRIGGER environment variable"
            )

        if not self.codex_modes:
            raise ValueError(
                "CodexRouting: no Codex modes defined — "
                "check 'codex_modes' in the JSON config or the "
                f"{AGENTIC_CODEX_ROUTING_PREFIX}MODES environment variable"
            )

        names = self.mode_names
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"CodexRouting: duplicate Codex mode names {duplicates} — "
                "check 'codex_modes' in the JSON config"
            )

        if self.fallback_mode not in self.mode_by_name:
            raise ValueError(
                f"CodexRouting: fallback_mode '{self.fallback_mode}' is not a "
                "defined Codex mode — check 'settings.fallback_mode' in the JSON "
                f"config or the {AGENTIC_CODEX_ROUTING_PREFIX}FALLBACK_MODE "
                "environment variable"
            )

        if self.semantic_enabled and not self.embedding_model:
            raise ValueError(
                "CodexRouting: no embedding_model configured — "
                "check 'embedding_model' in the JSON config or the "
                f"{AGENTIC_CODEX_ROUTING_PREFIX}MODEL environment variable"
            )

        if self.chunk_size <= 0:
            raise ValueError(
                f"CodexRouting: chunk_size must be > 0, got {self.chunk_size}"
            )

        if self.classify_max_chars <= 0:
            raise ValueError(
                "CodexRouting: classify_max_chars must be > 0, got "
                f"{self.classify_max_chars}"
            )

        if self.chunk_overlap < 0:
            raise ValueError(
                f"CodexRouting: chunk_overlap must be >= 0, got "
                f"{self.chunk_overlap}"
            )

        if self.top_k < 1:
            raise ValueError(f"CodexRouting: top_k must be >= 1, got {self.top_k}")

    def lint_signals(self, logger: Optional[logging.Logger] = None) -> None:
        """
        Report configured signals that can never score, as warnings.

        Keyword scoring stays silent about unusable settings: a pattern that
        does not compile is dropped and a pattern with upper-case letters
        never matches, because the text is lower-cased before scoring.  This
        lint surfaces both, along with a ``chunk_overlap`` that is not below
        ``chunk_size`` and a mode without a ``model_name`` (requests resolved
        to it are passed through untouched).  Nothing is changed and nothing
        is raised: the routing decision stays exactly what :meth:`validate_args`
        accepts.

        Parameters
        ----------
        logger : logging.Logger, optional
            Logger instance used to report the findings.  When ``None`` the
            lint does nothing.

        Returns
        -------
        None
        """
        if logger is None:
            return

        if self.chunk_size > 0 and self.chunk_overlap >= self.chunk_size:
            logger.warning(
                "CodexRouting: chunk_overlap (%d) is not below chunk_size "
                "(%d) — the embedding router clamps it, check "
                "'settings.semantic' in the JSON config",
                self.chunk_overlap,
                self.chunk_size,
            )

        for mode in self.codex_modes:
            if not mode.model_name:
                logger.warning(
                    "CodexRouting: mode '%s' has no model_name — requests "
                    "resolved to it are passed through with the trigger model",
                    mode.name,
                )
            for pattern in mode.patterns:
                try:
                    re.compile(pattern)
                except (re.error, TypeError):
                    logger.warning(
                        "CodexRouting: mode '%s' declares the unusable pattern "
                        "%r — keyword scoring ignores it",
                        mode.name,
                        pattern,
                    )
                    continue
                if isinstance(pattern, str) and _UPPERCASE_RE.search(pattern):
                    logger.warning(
                        "CodexRouting: mode '%s' pattern %r contains "
                        "upper-case letters — the text is lower-cased before "
                        "scoring, so it can never match",
                        mode.name,
                        pattern,
                    )


@dataclass(frozen=True)
class CodexMode(RoutingTarget):
    """
    Definition of a single Codex work mode.

    Extends the shared :class:`RoutingTarget` (name, model, description,
    examples) with the signals used by the deterministic keyword layer.

    Parameters
    ----------
    name : str
        Unique identifier of the mode (used in ``agent_mode`` in results).
    model_name : str
        The model name to select when this mode wins; an empty value makes the
        plugin pass the request through unchanged.
    description : str
        Human-readable description of the mode.
    examples : Tuple[str, ...]
        Example prompts representative of this mode.
    keywords : Tuple[str, ...]
        Keywords for the scorer (word-start matches, default weight 1.0).
    phrases : Tuple[str, ...]
        Multi-word expressions for the scorer, optionally suffixed with
        ``":weight"`` (default weight 2.0).
    patterns : Tuple[str, ...]
        Regex patterns for the scorer (each match adds 3.0).
    weights : Dict[str, Any]
        Per-keyword weights overriding the default keyword weight of 1.0.
    """

    keywords: Tuple[str, ...] = field(default_factory=tuple)
    phrases: Tuple[str, ...] = field(default_factory=tuple)
    patterns: Tuple[str, ...] = field(default_factory=tuple)
    weights: Dict[str, Any] = field(default_factory=dict)
