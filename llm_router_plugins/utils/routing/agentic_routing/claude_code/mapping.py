"""
Model-name normalization and matching for the Claude Code model swap.

Claude Code resolves its own aliases (``opus``, ``sonnet``, ``haiku``,
``fable``, ``opusplan``) before it sends a request, so what reaches a gateway
is a versioned model ID — and the same tier arrives under several spellings:

==========================================  ==========  ======================
Name seen by the gateway                    Tier        Spelling artefact
==========================================  ==========  ======================
``claude-sonnet-5``                         sonnet      plain API model ID
``claude-sonnet-5[1m]``                     sonnet      requested 1M window
``claude-sonnet-4-5-20250929``              sonnet      dated release ID
``us.anthropic.claude-sonnet-4-5-20250929-v1:0``  sonnet  Bedrock inference ID
``models/claude-sonnet-4-5``                sonnet      deployment-style path
``claude-3-5-haiku-20241022``               haiku       legacy name ordering
==========================================  ==========  ======================

:func:`normalize_model_name` reduces a name to its comparable core so the
spellings above collapse to one key, and :class:`ModelMatcher` resolves a name
to the mode that configured it through four deterministic layers, most specific
first: literal exact, normalized exact, longest wildcard prefix, family token.

Matching is case-insensitive and never raises: an unmatched name yields
``None`` and the caller passes the request through untouched.
"""

import re

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Match kinds reported by :attr:`ModelMatch.kind`, in precedence order.
MATCH_LITERAL = "literal"
MATCH_EXACT = "exact"
MATCH_WILDCARD = "wildcard"
MATCH_FAMILY = "family"

#: Payload keys inspected for the model name, in the order they are tried.
DEFAULT_MODEL_FIELDS: Tuple[str, ...] = ("model", "model_name")

#: Vendor prefixes stripped from a model name before comparison.  Bedrock,
#: Vertex and friends prefix the Anthropic ID with a vendor label that says
#: nothing about which tier the request belongs to.
DEFAULT_PROVIDER_PREFIXES: Tuple[str, ...] = (
    "us.anthropic.",
    "eu.anthropic.",
    "apac.anthropic.",
    "global.anthropic.",
    "anthropic.",
    "anthropic/",
)

#: Model families known to Claude Code, used by the family fallback layer.
#: An unknown family never reaches that layer, which costs nothing.
KNOWN_MODEL_FAMILIES: Tuple[str, ...] = ("fable", "opus", "sonnet", "haiku")

# Requested context window, e.g. the "[1m]" of "claude-opus-5-5[1m]".
_WINDOW_SUFFIX_RE = re.compile(r"\[[^\[\]]*\]$")
# Provider-side version suffix, e.g. the ":0" of "...-v1:0" (Bedrock).
_PROVIDER_VERSION_RE = re.compile(r":v?\d+$")
# Release revision: "@YYYYMMDD" (Vertex) or "-vN" (Bedrock).
_VERTEX_REVISION_RE = re.compile(r"@\d{4,8}$")
_REVISION_RE = re.compile(r"-v\d+$")
# Release date suffix, e.g. the "-20250929" of "claude-sonnet-4-5-20250929".
_DATE_SUFFIX_RE = re.compile(r"-\d{8}$")
# Any family token, anywhere in the name, so that legacy spellings such as
# "claude-3-5-haiku-20241022" still resolve to the Haiku family.
_FAMILY_ALTERNATIVES = "|".join(re.escape(family) for family in KNOWN_MODEL_FAMILIES)
_FAMILY_RE = re.compile(rf"(?:^|[-.])(?:{_FAMILY_ALTERNATIVES})(?:[-.]|$)")


def normalize_model_name(
    value: Any,
    provider_prefixes: Tuple[str, ...] = DEFAULT_PROVIDER_PREFIXES,
) -> str:
    """
    Reduce a model name to the key the matcher compares.

    The reduction is deliberately lossy: it removes what an operator should not
    have to repeat per tier in the configuration, and nothing else.  After
    lower-casing it strips, in order, a trailing ``[...]`` window suffix, a
    provider version suffix (``:0``, ``:v2``), an ``@YYYYMMDD`` revision, a
    known vendor prefix, any leading path (``models/…``), a ``-vN`` revision
    and a ``-YYYYMMDD`` release date.

    Parameters
    ----------
    value : Any
        The model name to normalize.  Anything that is not a string yields an
        empty result.
    provider_prefixes : Tuple[str, ...]
        Vendor prefixes to strip, compared case-insensitively.

    Returns
    -------
    str
        The normalized name, or an empty string when nothing is left to
        compare.

    Examples
    --------
    >>> normalize_model_name("us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    'claude-sonnet-4-5'
    >>> normalize_model_name("claude-opus-5-5[1m]")
    'claude-opus-5-5'
    """
    if not isinstance(value, str):
        return ""

    name = value.strip().lower()
    if not name:
        return ""

    name = _WINDOW_SUFFIX_RE.sub("", name)
    name = _PROVIDER_VERSION_RE.sub("", name)
    name = _VERTEX_REVISION_RE.sub("", name)
    name = _strip_provider_prefix(name, provider_prefixes)
    name = name.rsplit("/", 1)[-1]
    name = _REVISION_RE.sub("", name)
    name = _DATE_SUFFIX_RE.sub("", name)

    return name.strip()


def model_family(value: Any) -> Optional[str]:
    """
    Return the Claude Code family carried by *value*, if any.

    Parameters
    ----------
    value : Any
        A model name, normalized or not.

    Returns
    -------
    str or None
        One of :data:`KNOWN_MODEL_FAMILIES`, or ``None`` when the name does not
        name a known family.

    Examples
    --------
    >>> model_family("claude-3-5-haiku-20241022")
    'haiku'
    """
    if not isinstance(value, str):
        return None
    found = _FAMILY_RE.search(value.strip().lower())
    if found is None:
        return None
    return next(
        (family for family in KNOWN_MODEL_FAMILIES if family in found.group(0)), None
    )


def validate_pattern(pattern: Any) -> str:
    """
    Validate one configured model pattern.

    Parameters
    ----------
    pattern : Any
        A configured entry such as ``"claude-opus-5-5"`` or
        ``"claude-sonnet-*"``.  A wildcard is accepted only as the single
        trailing character.

    Returns
    -------
    str
        The pattern without surrounding whitespace.

    Raises
    ------
    ValueError
        If *pattern* is not a non-empty string, has no literal prefix in front
        of its wildcard, or embeds ``*`` anywhere other than at the end.
    """
    if not isinstance(pattern, str) or not pattern.strip():
        raise ValueError(
            f"Model pattern must be a non-empty string, got {pattern!r}"
        )
    cleaned = pattern.strip()
    if "*" in cleaned[:-1]:
        raise ValueError(
            f"Model pattern {pattern!r} embeds a wildcard — only a single "
            "trailing '*' is supported, e.g. 'claude-opus-*'"
        )
    if cleaned == "*":
        raise ValueError(
            "Model pattern '*' matches every model this router serves — name "
            "the family instead, e.g. 'claude-*'"
        )
    return cleaned


def is_wildcard(pattern: str) -> bool:
    """
    Return ``True`` when *pattern* is a trailing-wildcard pattern.

    Parameters
    ----------
    pattern : str
        A validated pattern.

    Returns
    -------
    bool
        ``True`` when the pattern ends with ``*``.
    """
    return pattern.endswith("*")


def _strip_provider_prefix(name: str, provider_prefixes: Tuple[str, ...]) -> str:
    """
    Remove a leading vendor prefix from *name*, longest prefix first.

    Parameters
    ----------
    name : str
        The lower-cased model name.
    provider_prefixes : Tuple[str, ...]
        Configured vendor prefixes.

    Returns
    -------
    str
        *name* without its vendor prefix.
    """
    for prefix in sorted(provider_prefixes, key=len, reverse=True):
        cleaned = prefix.strip().lower()
        if cleaned and name.startswith(cleaned):
            return name[len(cleaned) :]
    return name


@dataclass(frozen=True)
class ModelMatch:
    """
    A single successful match of a model name against the configuration.

    Parameters
    ----------
    mode_name : str
        Name of the mode (tier) that configured the winning entry.
    pattern : str
        The configured entry as authored, reported for diagnostics.
    kind : str
        How it matched: :data:`MATCH_LITERAL`, :data:`MATCH_EXACT`,
        :data:`MATCH_WILDCARD` or :data:`MATCH_FAMILY`.
    """

    mode_name: str
    pattern: str
    kind: str


class ModelMatcher:
    """
    Resolve a model name to the mode that configured it.

    Built once from the flat list of configured patterns, it answers lookups
    without consulting the configuration again.  Four layers are tried in
    strict order, so an entry naming one model always beats an entry naming a
    family:

    1. **literal** — the lower-cased, trimmed name equals a configured entry;
    2. **exact** — the :func:`normalize_model_name` reduction equals the
       reduction of a configured entry, which is what makes
       ``us.anthropic.claude-sonnet-4-5-20250929-v1:0`` hit
       ``claude-sonnet-4-5``;
    3. **wildcard** — a ``claude-opus-*`` style entry, longest literal prefix
       first, so a version pin beats the family wildcard of the same tier;
    4. **family** — the family token declared by a wildcard, which keeps legacy
       spellings such as ``claude-3-5-haiku-20241022`` on the Haiku mode.

    Layers 3 and 4 — and the wildcard entries themselves — are skipped when
    *families_enabled* is ``False``, leaving exact matching only.  Ties resolve
    by configuration order: the first declaration wins, and
    :func:`find_ambiguous_wildcards` / :func:`find_duplicate_literals` report
    the cases worth telling an operator about.

    Parameters
    ----------
    entries : Iterable[Tuple[str, Any]]
        ``(mode_name, pattern)`` pairs, in configuration order.
    provider_prefixes : Tuple[str, ...]
        Vendor prefixes used by the normalization layer.
    families_enabled : bool
        Whether wildcard and family matching participate.
    """

    def __init__(
        self,
        entries: Iterable[Tuple[str, Any]],
        provider_prefixes: Tuple[str, ...] = DEFAULT_PROVIDER_PREFIXES,
        families_enabled: bool = True,
    ) -> None:
        """
        Index *entries* for matching.

        Raises
        ------
        ValueError
            If any pattern is empty or embeds a wildcard, as reported by
            :func:`validate_pattern`.  Nothing is indexed until every pattern
            validates.
        """
        self._provider_prefixes = tuple(provider_prefixes)
        self._families_enabled = bool(families_enabled)
        self._literal: Dict[str, ModelMatch] = {}
        self._normalized: Dict[str, ModelMatch] = {}
        self._wildcards: List[Tuple[str, str, ModelMatch]] = []
        self._family: Dict[str, ModelMatch] = {}

        for mode_name, pattern in entries:
            cleaned = validate_pattern(pattern)
            match = ModelMatch(
                mode_name=str(mode_name),
                pattern=cleaned,
                kind=MATCH_LITERAL,
            )
            if is_wildcard(cleaned):
                if not self._families_enabled:
                    continue
                prefix = cleaned[:-1].strip().lower()
                self._wildcards.append(
                    (
                        prefix,
                        normalize_model_name(prefix, self._provider_prefixes),
                        match,
                    )
                )
                family = model_family(prefix)
                if family and family not in self._family:
                    self._family[family] = match
                continue

            self._literal.setdefault(match.pattern.lower(), match)
            normalized = normalize_model_name(
                cleaned, provider_prefixes=self._provider_prefixes
            )
            if normalized:
                self._normalized.setdefault(normalized, match)

        self._wildcards.sort(key=lambda item: len(item[0]), reverse=True)

    @property
    def families_enabled(self) -> bool:
        """
        Whether the wildcard and family layers participate in matching.

        Returns
        -------
        bool
            ``True`` when wildcard and family matching are active.
        """
        return self._families_enabled

    def match(self, value: Any) -> Optional[ModelMatch]:
        """
        Resolve *value* to a configured mode.

        Parameters
        ----------
        value : Any
            The model name as it appears in the payload.

        Returns
        -------
        ModelMatch or None
            The winning match, or ``None`` when no configured entry covers the
            name — the caller then passes the request through untouched.
        """
        if not isinstance(value, str) or not value.strip():
            return None

        literal = value.strip().lower()
        found = self._literal.get(literal)
        if found is not None:
            return _as_kind(found, MATCH_LITERAL)

        normalized = normalize_model_name(
            value, provider_prefixes=self._provider_prefixes
        )
        if not normalized:
            return None

        found = self._normalized.get(normalized)
        if found is not None:
            return _as_kind(found, MATCH_EXACT)

        return self._match_family_layers(normalized, literal)

    def _match_family_layers(
        self, normalized: str, literal: str
    ) -> Optional[ModelMatch]:
        """
        Resolve *normalized* through the wildcard and then the family layer.

        Parameters
        ----------
        normalized : str
            The normalized model name, compared against normalized prefixes.
        literal : str
            The lower-cased name as sent, compared against literal prefixes.

        Returns
        -------
        ModelMatch or None
            The winning match, or ``None`` when neither layer covers the name.
        """
        if not self._families_enabled:
            return None

        for prefix, normalized_prefix, candidate in self._wildcards:
            if normalized.startswith(normalized_prefix) or literal.startswith(
                prefix
            ):
                return _as_kind(candidate, MATCH_WILDCARD)

        family = model_family(normalized) or model_family(literal)
        if family is None:
            return None
        found = self._family.get(family)
        if found is None:
            return None
        return _as_kind(found, MATCH_FAMILY)


def _as_kind(match: ModelMatch, kind: str) -> ModelMatch:
    """
    Return *match* relabeled with the *kind* that resolved it.

    Parameters
    ----------
    match : ModelMatch
        The indexed match a wildcard or family layer found.
    kind : str
        The kind to report instead of the indexed :data:`MATCH_LITERAL`.

    Returns
    -------
    ModelMatch
        A copy of *match* with :attr:`ModelMatch.kind` replaced.
    """
    return replace(match, kind=kind)


def iter_mode_entries(
    modes: Iterable[Any],
    models_attribute: str = "models",
) -> List[Tuple[str, str]]:
    """
    Flatten ``(mode_name, pattern)`` pairs out of configured modes.

    Parameters
    ----------
    modes : Iterable[Any]
        Mode objects, each carrying a sequence of patterns.
    models_attribute : str
        Name of the attribute holding the patterns.

    Returns
    -------
    List[Tuple[str, str]]
        Pairs in mode order, skipping modes that declare no patterns.
    """
    pairs: List[Tuple[str, str]] = []
    for mode in modes:
        for pattern in getattr(mode, models_attribute, ()) or ():
            pairs.append((str(mode.name), pattern))
    return pairs


def find_duplicate_literals(
    entries: Iterable[Tuple[str, Any]],
    provider_prefixes: Tuple[str, ...] = DEFAULT_PROVIDER_PREFIXES,
) -> Dict[str, Tuple[str, str]]:
    """
    Report exact patterns that more than one mode claims.

    Parameters
    ----------
    entries : Iterable[Tuple[str, Any]]
        ``(mode_name, pattern)`` pairs, in configuration order.
    provider_prefixes : Tuple[str, ...]
        Vendor prefixes used by the normalization layer.

    Returns
    -------
    Dict[str, Tuple[str, str]]
        Mapping from the collating key to the two mode names that both declare
        it.  One exact model name claimed by two modes is a configuration
        mistake: only one of the two targets can ever be reached.
    """
    seen: Dict[str, str] = {}
    duplicates: Dict[str, Tuple[str, str]] = {}
    for mode_name, pattern in entries:
        try:
            cleaned = validate_pattern(pattern)
        except ValueError:
            continue
        if is_wildcard(cleaned):
            continue
        keys = (
            cleaned.lower(),
            normalize_model_name(cleaned, provider_prefixes),
        )
        for key in keys:
            if not key:
                continue
            owner = seen.get(key)
            if owner is None:
                seen[key] = str(mode_name)
            elif owner != str(mode_name) and key not in duplicates:
                duplicates[key] = (owner, str(mode_name))
    return duplicates


def find_ambiguous_wildcards(
    entries: Iterable[Tuple[str, Any]],
) -> Dict[str, Tuple[str, ...]]:
    """
    Report identical wildcards declared by several modes.

    Parameters
    ----------
    entries : Iterable[Tuple[str, Any]]
        ``(mode_name, pattern)`` pairs, in configuration order.

    Returns
    -------
    Dict[str, Tuple[str, ...]]
        Mapping from wildcard pattern to every mode that declares it.  The
        first declaration wins at match time, so the others are dead weight
        rather than an error.
    """
    owners: Dict[str, List[str]] = {}
    for mode_name, pattern in entries:
        try:
            cleaned = validate_pattern(pattern)
        except ValueError:
            continue
        if is_wildcard(cleaned):
            owners.setdefault(cleaned, []).append(str(mode_name))
    return {
        pattern: tuple(names)
        for pattern, names in owners.items()
        if len(set(names)) > 1
    }
