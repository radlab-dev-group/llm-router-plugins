"""
Shared plumbing for the routing plugins.

This module centralises the parts of the routing plugins that are pure
boilerplate today — duplicated across ``semantic_biencoder`` and
``agentic_routing``:

- :class:`RoutingConfigBase` — the ``from_file`` / ``from_json`` loading
  protocol (``..._CONFIG`` env var holding a raw JSON string *or* a file
  path, optional default config location, ``_from_raw`` hook);
- :func:`env_int`, :func:`env_float`, :func:`env_bool`,
  :func:`resolve_persist_dir` — environment-variable override helpers;
- :func:`build_embedding_router`, :func:`check_router_has_vectors` — building
  and validating the shared :class:`EmbeddingRouter`;
- :func:`should_route` — the ``payload["model"]`` trigger gate;
- :func:`annotate_routing` — writing the routing decision into the payload.
"""

import json
import os
import pathlib

from typing import Any, Collection, Dict, Optional

from llm_router_plugins.utils.routing.embedder import (
    EmbeddingRouter,
    EmbeddingRouterConfig,
)


# Accepted textual representations of booleans coming from env vars.
_BOOL_TRUE_VALUES = ("1", "true", "yes", "on")
_BOOL_FALSE_VALUES = ("0", "false", "no", "off")


class RoutingConfigBase:
    """
    Base class for routing config dataclasses.

    Subclasses are dataclasses that define two class attributes and implement
    the :meth:`_from_raw` classmethod:

    - ``_ENV_PREFIX`` — the environment-variable prefix of the plugin
      (e.g. ``"LLM_ROUTER_ROUTING_AGENTIC_"``);
    - ``_DEFAULT_CONFIG_PATH`` — the bundled default JSON config, or
      ``None`` when the plugin has no default location;
    - ``_from_raw(raw) -> config`` — parse and validate the decoded JSON dict.

    Parameters
    ----------
    (none — the base class adds no fields)
    """

    _ENV_PREFIX: str = ""
    _DEFAULT_CONFIG_PATH: Optional[pathlib.Path] = None

    @classmethod
    def _config_json_env(cls) -> str:
        """Return the env var name that can hold the full config as JSON."""
        return f"{cls._ENV_PREFIX}CONFIG"

    @classmethod
    def from_file(
        cls, path: Optional[pathlib.Path] = None
    ) -> "RoutingConfigBase":
        """
        Load configuration from a JSON file or from the ``..._CONFIG`` env var.

        The ``{prefix}CONFIG`` env var supports **two forms**:

        1. **Raw JSON string** — value starts with ``{`` or ``[`` → parsed
           directly.
        2. **File path** — anything else → opened as a JSON config file.

        When the env var is set (non-empty) it takes priority over *path* and
        the default location.  There is no silent fall-through: if the
        specified file does not exist or contains invalid JSON the error
        propagates so the user sees exactly what went wrong.

        When *no* env var is present the file is loaded from *path* if given,
        or from ``_DEFAULT_CONFIG_PATH`` when defined.

        Parameters
        ----------
        path : pathlib.Path or None, optional
            Path to the JSON config file. Used only when the env var is unset
            or empty.

        Returns
        -------
        RoutingConfigBase
            A config dataclass populated from the JSON source.

        Raises
        ------
        FileNotFoundError
            If the env var points to a file that does not exist, or if no env
            var is set and the default config is missing.
        KeyError
            If the JSON (from env or file) is missing required fields.
        json.JSONDecodeError
            If the env var value or config file contains invalid JSON.
        ValueError
            If no config source can be resolved, or the parameters are out of
            range (see :meth:`_from_raw`).
        """
        # ---- env-var shortcut (raw JSON string or file path) -----------------
        raw_json = os.environ.get(cls._config_json_env())
        if raw_json is not None:
            stripped = raw_json.strip()
            # --- Case A: raw JSON string (starts with { or [) -----------------
            if stripped and stripped[0] in ("{", "["):
                return cls.from_json(stripped)

            # --- Case B: file path supplied via env var ------------------------
            if stripped:
                # No fall-through — raise immediately if the file can't be read
                with open(stripped, "r", encoding="utf-8") as fh:
                    raw = json.load(fh)
                return cls._from_raw(raw)

        if path is None:
            if cls._DEFAULT_CONFIG_PATH is None:
                raise ValueError(
                    f"{cls.__name__}.from_file: no config path provided — "
                    f"set the {cls._config_json_env()} env var to a valid "
                    "JSON object or a file path (not an empty string)"
                )
            path = cls._DEFAULT_CONFIG_PATH

        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)

        return cls._from_raw(raw)

    @classmethod
    def from_json(cls, raw: str) -> "RoutingConfigBase":
        """
        Parse configuration from a raw JSON string.

        Parameters
        ----------
        raw : str
            A valid JSON string.

        Returns
        -------
        RoutingConfigBase
            A config dataclass populated from the parsed JSON.

        Raises
        ------
        KeyError
            If required fields are missing.
        json.JSONDecodeError
            If *raw* is not valid JSON.
        ValueError
            If the config string is empty, or the parameters are out of range
            (see :meth:`_from_raw`).
        """
        if not raw:
            raise ValueError(
                f"{cls.__name__}.from_json: empty config string — "
                f"check that {cls._config_json_env()} env var is set to a "
                "valid JSON object or a file path (not an empty string)"
            )
        parsed = json.loads(raw)
        return cls._from_raw(parsed)

    @staticmethod
    def validate_semantic_params(
        chunk_size: int, chunk_overlap: int, top_k: int
    ) -> None:
        """
        Validate the shared semantic-routing parameters.

        Parameters
        ----------
        chunk_size : int
            Must be > 0.
        chunk_overlap : int
            Must be >= 0.
        top_k : int
            Must be >= 1.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If any of the parameters is out of range.
        """
        if chunk_size <= 0:
            raise ValueError(f"Expected 'chunk_size' > 0, got {chunk_size}.")
        if chunk_overlap < 0:
            raise ValueError(
                f"Expected 'chunk_overlap' >= 0, got {chunk_overlap}."
            )
        if top_k < 1:
            raise ValueError(f"Expected 'top_k' >= 1, got {top_k}.")


def env_int(prefix: str, suffix: str) -> Optional[int]:
    """
    Read an integer-typed environment variable ``{prefix}{suffix}``.

    Parameters
    ----------
    prefix : str
        The environment-variable prefix (e.g. ``"LLM_ROUTER_ROUTING_AGENTIC_"``).
    suffix : str
        The env var suffix (e.g. ``"TOP_K"``).

    Returns
    -------
    int or None
        The parsed value, or ``None`` when the variable is unset, empty or
        not a valid integer (invalid values are silently ignored, matching
        the plugins' historical behaviour).
    """
    value = os.getenv(f"{prefix}{suffix}")
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def env_float(prefix: str, suffix: str) -> Optional[float]:
    """
    Read a float-typed environment variable ``{prefix}{suffix}``.

    Parameters
    ----------
    prefix : str
        The environment-variable prefix.
    suffix : str
        The env var suffix (e.g. ``"SIMILARITY_THRESHOLD"``).

    Returns
    -------
    float or None
        The parsed value, or ``None`` when the variable is unset, empty or
        not a valid float (invalid values are silently ignored).
    """
    value = os.getenv(f"{prefix}{suffix}")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def env_bool(
    prefix: str, suffix: str, logger: Any = None
) -> Optional[bool]:
    """
    Read a boolean-typed environment variable ``{prefix}{suffix}``.

    Accepted true values: ``1``, ``true``, ``yes``, ``on``; accepted false
    values: ``0``, ``false``, ``no``, ``off`` (case-insensitive).

    Parameters
    ----------
    prefix : str
        The environment-variable prefix.
    suffix : str
        The env var suffix (e.g. ``"SEMANTIC_ENABLED"``).
    logger : logging.Logger, optional
        Logger instance used to report unrecognized values.

    Returns
    -------
    bool or None
        The parsed value, or ``None`` when the variable is unset, empty or
        unrecognized (unrecognized values are logged as warnings).
    """
    value = os.getenv(f"{prefix}{suffix}")
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in _BOOL_TRUE_VALUES:
        return True
    if lowered in _BOOL_FALSE_VALUES:
        return False
    if logger:
        logger.warning(
            "Ignoring unrecognized %s%s value '%s'", prefix, suffix, value
        )
    return None


def resolve_persist_dir(
    prefix: str,
    configured: Optional[str],
    logger: Any = None,
) -> Optional[str]:
    """
    Resolve the FAISS index persistence directory.

    The ``{prefix}PERSIST_DIR`` environment variable takes priority over the
    *configured* value from the JSON config.

    Parameters
    ----------
    prefix : str
        The environment-variable prefix.
    configured : str or None
        The value from the JSON config (may be empty or ``None``).
    logger : logging.Logger, optional
        Logger instance used to report the applied override.

    Returns
    -------
    str or None
        The persistence directory, or ``None`` for in-memory mode.
    """
    persist_env = os.getenv(f"{prefix}PERSIST_DIR")
    if persist_env:
        if logger:
            logger.info("Overriding vector store path: %s", persist_env)
        return persist_env
    return configured or None


def build_embedding_router(
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    routing_targets: Collection[Any],
    logger: Any = None,
    persist_dir: Optional[str] = None,
    missing_deps_hint: Optional[str] = None,
) -> EmbeddingRouter:
    """
    Build and initialize the shared BiEncoder + FAISS :class:`EmbeddingRouter`.

    The heavy ML dependencies (``faiss``, ``sentence_transformers``) are
    imported lazily so the routing plugins keep working without them as long
    as the semantic layer is not used.

    Parameters
    ----------
    embedding_model : str
        The HuggingFace model identifier (or local path) for embeddings.
    chunk_size : int
        Number of tokens per chunk when splitting target text.
    chunk_overlap : int
        Number of tokens overlapping between adjacent chunks.
    top_k : int
        Number of nearest neighbours to retrieve during routing queries.
    routing_targets : Collection
        Routing targets to index (``RoutingTarget`` or a subclass).
    logger : logging.Logger, optional
        Logger instance.
    persist_dir : str, optional
        Directory where the FAISS index and docstore are saved.
    missing_deps_hint : str, optional
        Extra context added to the error message raised when the ML
        dependencies are not installed.

    Returns
    -------
    EmbeddingRouter
        An initialized router whose index contains at least one vector.

    Raises
    ------
    ValueError
        If ``faiss`` / ``sentence_transformers`` are not importable, or if
        the resulting index contains no vectors.
    """
    hint = (
        missing_deps_hint
        or (
            "semantic routing is enabled but the sentence-transformers / FAISS "
            "dependencies are not installed — install them or disable semantic "
            "routing"
        )
    )
    try:
        import faiss  # noqa: F401
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except ImportError as exc:
        raise ValueError(hint) from exc

    router = EmbeddingRouter(
        config=EmbeddingRouterConfig(
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            routing_targets=tuple(routing_targets),
        ),
        logger=logger,
        persist_dir=persist_dir,
    )

    try:
        router.initialize()
    except ImportError as exc:
        raise ValueError(hint) from exc

    check_router_has_vectors(router)
    return router


def check_router_has_vectors(router: Any, label: str = "router") -> None:
    """
    Verify that *router* contains at least one vector.

    Parameters
    ----------
    router : Any
        The router to check (must expose the ``has_vectors`` property).
    label : str
        Short plugin name used in the error message.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the router has no vectors.
    """
    if not getattr(router, "has_vectors", False):
        raise ValueError(
            f"{label}: router loaded but has no vectors — check the routing "
            "targets have non-empty descriptions/examples and the FAISS index "
            "was built successfully"
        )


def should_route(payload: Dict[str, Any], triggers: Collection[str]) -> bool:
    """
    Return ``True`` when *payload* should be routed by this plugin.

    The plugin activates only when ``payload["model"]`` is a string whose
    trimmed value is listed in *triggers*.

    Parameters
    ----------
    payload : dict
        The incoming message payload.
    triggers : Collection[str]
        The configured trigger values (e.g. ``("auto",)`` or the agentic
        ``trigger`` list).

    Returns
    -------
    bool
        ``True`` when the payload should be routed, ``False`` otherwise.
    """
    model = payload.get("model")
    if not isinstance(model, str):
        return False
    return model.strip() in triggers


def annotate_routing(
    payload: Dict[str, Any],
    plugin_name: str,
    model_name: str,
    similarity: float,
    **extra: Any,
) -> Dict[str, Any]:
    """
    Write the routing decision into *payload*.

    Sets ``payload["model"]`` to the selected *model_name* and adds
    ``payload["routing"]`` with the base keys ``"plugin"`` and
    ``"similarity"`` plus any *extra* fields (e.g. ``target_name`` for the
    biencoder plugin, ``agent_mode`` / ``source`` for the agentic plugin).

    Parameters
    ----------
    payload : dict
        The payload to annotate.
    plugin_name : str
        The ``name`` of the plugin that made the decision.
    model_name : str
        The selected model name.
    similarity : float
        Confidence score of the decision.
    **extra : Any
        Additional ``"routing"`` metadata entries.

    Returns
    -------
    dict
        The annotated payload.
    """
    payload["model"] = model_name
    routing: Dict[str, Any] = {
        "plugin": plugin_name,
        "similarity": float(similarity),
    }
    routing.update(extra)
    payload["routing"] = routing
    return payload
