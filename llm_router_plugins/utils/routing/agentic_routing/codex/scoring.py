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

No machine learning, no embeddings, no network access — the same input always
produces the same score.
"""

import re
from functools import lru_cache

from typing import Any, Dict, Iterable, Optional, Tuple

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


@lru_cache(maxsize=512)
def _word_start_pattern(needle: str) -> "re.Pattern[str]":
    """
    Compile a leading word-boundary match for *needle*.

    Parameters
    ----------
    needle : str
        A lower-cased keyword or phrase with surrounding whitespace removed.

    Returns
    -------
    re.Pattern
        A compiled regex that matches *needle* only at a word start, allowing
        inflected suffixes (``testow`` matches ``testów``) while rejecting
        mid-word hits (``protest`` does not match ``test``).

    Raises
    ------
    None
    """
    return re.compile(r"(?<!\w)" + re.escape(needle))


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

    weights = mode.weights if isinstance(mode.weights, dict) else {}
    score = 0.0
    for keyword in mode.keywords:
        needle = keyword.strip().lower()
        if needle and _word_start_pattern(needle).search(text_lower):
            score += _keyword_weight(keyword, weights)
    for phrase in mode.phrases:
        needle, weight = _signal_weight(phrase, DEFAULT_PHRASE_WEIGHT)
        if needle and _word_start_pattern(needle).search(text_lower):
            score += weight
    for pattern in mode.patterns:
        try:
            if re.search(pattern, text_lower):
                score += PATTERN_WEIGHT
        except re.error:
            continue
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
