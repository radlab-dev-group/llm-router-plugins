"""
Deterministic keyword scoring for the Codex routing plugin.

Scoring model
-------------
A :class:`~llm_router_plugins.utils.routing.agentic_routing.codex.config.
CodexMode` carries three kinds of signals, all matched case-insensitively.
Keywords and phrases must start at a word boundary, so Polish inflected
forms still match (``testow`` matches ``testów``) while mid-word hits
do not (``protest`` does not match ``test``); patterns are the exception,
matching a compiled regex:

===========  ==========================================  ==============
Signal       Weight                                      Source
===========  ==========================================  ==============
keyword      ``mode.weights[keyword]`` or ``1.0``        ``weights``
phrase       weight suffix ``":w"`` or ``2.0``           ``phrases``
pattern      ``3.0``                                     ``patterns``
===========  ==========================================  ==============

Scores of all matched signals are summed per mode.  The comparison used by
:func:`detect_mode` is strictly greater-than while iterating the modes in
config declaration order, so **ties are won by the mode declared first**.

The score is mapped onto a confidence with
:func:`score_to_similarity` (``score / (score + 1)``), a saturating function
that keeps every confidence inside ``[0.0, 1.0)``.

Scoring runs off a per-mode *plan* — every signal prepared once and cached.
Keyword and phrase signals are matched with a plain substring scan plus a word
boundary test instead of one regex per signal, and are probed in a first pass
that lets a mode score ``0.0`` without a single boundary check.  The scan is
equivalence-preserving: it reports exactly the matches the word-start regexes
report, which keeps the scoring model and every score unchanged.

No machine learning, no embeddings, no network access — the same input always
produces the same score.
"""

import re

from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

from llm_router_plugins.utils.routing.agentic_routing.codex.config import CodexMode

__all__ = [
    "DEFAULT_KEYWORD_WEIGHT",
    "DEFAULT_PHRASE_WEIGHT",
    "PATTERN_WEIGHT",
    "score_mode",
    "score_to_similarity",
    "detect_mode",
]

#: Weight of a keyword that has no explicit entry in ``mode.weights``.
DEFAULT_KEYWORD_WEIGHT = 1.0

#: Weight of a phrase that has no ``":weight"`` suffix.
DEFAULT_PHRASE_WEIGHT = 2.0

#: Weight of a regular-expression match.
PATTERN_WEIGHT = 3.0


def _signal_weight(value: str, default: float) -> Tuple[str, float]:
    """
    Split an optional ``":weight"`` suffix off a signal.

    Parameters
    ----------
    value : str
        The raw signal, e.g. ``"run the tests:3"`` or ``"testy"``.
    default : float
        Weight used when no parsable suffix is present.

    Returns
    -------
    Tuple[str, float]
        The lower-cased signal text and its weight.  A malformed suffix is
        treated as part of the signal text.

    Raises
    ------
    None
    """
    text, separator, suffix = value.rpartition(":")
    if separator and text:
        try:
            return text.strip().lower(), float(suffix)
        except ValueError:
            pass
    return value.strip().lower(), default


def _keyword_weight(keyword: str, weights: Dict[str, Any]) -> float:
    """
    Return the configured weight of *keyword*, falling back to the default.

    Parameters
    ----------
    keyword : str
        The keyword as declared in the mode definition.
    weights : Dict[str, Any]
        The per-keyword weight mapping of the mode.

    Returns
    -------
    float
        The configured weight, or :data:`DEFAULT_KEYWORD_WEIGHT` when the
        keyword has no entry or the entry is not a number.

    Raises
    ------
    None
    """
    candidates = (keyword, keyword.lower())
    for candidate in candidates:
        if candidate in weights:
            try:
                return float(weights[candidate])
            except (TypeError, ValueError):
                return DEFAULT_KEYWORD_WEIGHT
    return DEFAULT_KEYWORD_WEIGHT


class _ModePlan(NamedTuple):
    """
    Precompiled scoring program for one mode.

    Attributes
    ----------
    literals : Tuple[Tuple[str, float], ...]
        Lower-cased keyword and phrase needles with their weights, in
        declaration order.
    patterns : Tuple[re.Pattern, ...]
        Compiled regular expressions of the mode, each worth
        :data:`PATTERN_WEIGHT` on a match.
    """

    literals: Tuple[Tuple[str, float], ...] = ()
    patterns: Tuple["re.Pattern[str]", ...] = ()


def _is_word_char(char: str) -> bool:
    """
    Return whether *char* is a ``\\w`` character.

    Parameters
    ----------
    char : str
        A single character preceding a candidate match.

    Returns
    -------
    bool
        ``True`` for exactly the characters ``re`` treats as word characters:
        alphanumerics (letters and numbers of any script) and ``"_"``.

    Raises
    ------
    None
    """
    return char.isalnum() or char == "_"


def _literal_matches(text_lower: str, needle: str) -> bool:
    """
    Return whether *needle* occurs in *text_lower* at a word start.

    Stands in for a ``(?<!\\w)re.escape(needle)`` search: an occurrence counts
    only when the character in front of it is not a word character, so
    inflected forms match (``testow`` finds ``testów``) while mid-word hits do
    not (``protest`` does not find ``test``).  Occurrences failing the boundary
    test are skipped and the scan continues.

    Parameters
    ----------
    text_lower : str
        The lower-cased text to scan.
    needle : str
        The lower-cased literal to look for.

    Returns
    -------
    bool
        ``True`` when at least one occurrence starts a word.

    Raises
    ------
    None
    """
    start = 0
    while True:
        found = text_lower.find(needle, start)
        if found < 0:
            return False
        if found == 0 or not _is_word_char(text_lower[found - 1]):
            return True
        start = found + 1


def _compile_pattern(pattern: Any) -> Optional["re.Pattern[str]"]:
    """
    Compile a configured regular expression, ``None`` when unusable.

    Parameters
    ----------
    pattern : Any
        The pattern as declared in the mode definition.

    Returns
    -------
    re.Pattern or None
        The compiled pattern, or ``None`` for an invalid expression or a value
        that is not a pattern at all.

    Raises
    ------
    None
    """
    try:
        return re.compile(pattern)
    except (re.error, TypeError):
        return None


def _mode_signals(mode: CodexMode) -> _ModePlan:
    """
    Prepare the literals and the compiled patterns of *mode*.

    Parameters
    ----------
    mode : CodexMode
        The mode to prepare.

    Returns
    -------
    _ModePlan
        Keyword and phrase needles with their weights, followed by the valid
        regular expressions; empty needles and invalid patterns are dropped,
        exactly as the scorer skips them.

    Raises
    ------
    None
    """
    weights = mode.weights if isinstance(mode.weights, dict) else {}
    literals: List[Tuple[str, float]] = []
    for keyword in mode.keywords:
        needle = keyword.strip().lower()
        if needle:
            literals.append((needle, _keyword_weight(keyword, weights)))
    for phrase in mode.phrases:
        needle, weight = _signal_weight(phrase, DEFAULT_PHRASE_WEIGHT)
        if needle:
            literals.append((needle, weight))
    patterns: List["re.Pattern[str]"] = []
    for pattern in mode.patterns:
        compiled = _compile_pattern(pattern)
        if compiled is not None:
            patterns.append(compiled)
    return _ModePlan(literals=tuple(literals), patterns=tuple(patterns))


#: Plans compiled so far, keyed by the signal signature of the mode.
_PLAN_CACHE: Dict[Tuple[Any, ...], _ModePlan] = {}

#: Plans kept before the cache is dropped; configs hold a handful of modes.
_PLAN_CACHE_MAX = 64


def _plan_signature(mode: CodexMode) -> Tuple[Any, ...]:
    """
    Return the cache key identifying the signals of *mode*.

    :class:`~...config.CodexMode` is not hashable because it carries a mapping,
    so the plan is keyed by the values it is built from instead.

    Raises
    ------
    None
    """
    weights = mode.weights if isinstance(mode.weights, dict) else {}
    return (
        mode.name,
        tuple(mode.keywords),
        tuple(mode.phrases),
        tuple(mode.patterns),
        tuple(sorted((str(key), repr(value)) for key, value in weights.items())),
    )


def _mode_plan(mode: CodexMode) -> _ModePlan:
    """
    Return the cached scoring plan of *mode*.

    Parameters
    ----------
    mode : CodexMode
        The mode to look up.

    Returns
    -------
    _ModePlan
        The plan for the mode's current signal values; rebuilt when a signal
        or a weight changed.

    Raises
    ------
    None
    """
    signature = _plan_signature(mode)
    plan = _PLAN_CACHE.get(signature)
    if plan is None:
        plan = _mode_signals(mode)
        if len(_PLAN_CACHE) >= _PLAN_CACHE_MAX:
            _PLAN_CACHE.clear()
        _PLAN_CACHE[signature] = plan
    return plan


def score_mode(mode: CodexMode, text_lower: str) -> float:
    """
    Return the heuristic score of *mode* for already lower-cased *text_lower*.

    Parameters
    ----------
    mode : CodexMode
        The mode whose signals are matched against the text.
    text_lower : str
        The text to match, lower-cased by the caller.

    Returns
    -------
    float
        The sum of the weights of every matched signal (``0.0`` when the text
        is empty or matches nothing).  Keywords and phrases match at a word
        start (suffix inflection allowed), patterns as compiled regexes;
        invalid regular expressions are skipped.

    Raises
    ------
    None
    """
    if not text_lower:
        return 0.0

    plan = _mode_plan(mode)
    score = 0.0
    if any(needle in text_lower for needle, _ in plan.literals):
        for needle, weight in plan.literals:
            if _literal_matches(text_lower, needle):
                score += weight
    for pattern in plan.patterns:
        if pattern.search(text_lower):
            score += PATTERN_WEIGHT
    return score


def score_to_similarity(score: float) -> float:
    """
    Map a heuristic score onto a confidence in ``[0.0, 1.0)``.

    Parameters
    ----------
    score : float
        The raw heuristic score.

    Returns
    -------
    float
        ``0.0`` for a non-positive score, otherwise ``score / (score + 1)``.

    Raises
    ------
    None
    """
    if score <= 0:
        return 0.0
    return score / (score + 1.0)


def detect_mode(
    text: str,
    modes: Iterable[CodexMode],
) -> Tuple[Optional[CodexMode], float]:
    """
    Return the best-scoring mode for *text*.

    Parameters
    ----------
    text : str
        The text to classify (normally ``CodexRequest.latest_user_text``).
    modes : Iterable[CodexMode]
        Candidate modes, scanned in declaration order.

    Returns
    -------
    Tuple[Optional[CodexMode], float]
        The winning mode and its score, or ``(None, 0.0)`` when nothing
        matches.  Ties are won by the mode declared first.

    Raises
    ------
    None
    """
    text_lower = text.lower()
    best_mode: Optional[CodexMode] = None
    best_score = 0.0
    for mode in modes:
        score = score_mode(mode, text_lower)
        if score > best_score:
            best_mode, best_score = mode, score
    return best_mode, best_score
