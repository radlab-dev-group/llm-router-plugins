"""
Deterministic keyword/phrase/regex scoring for the Agentic routing cascade.

This module owns the *deterministic* text-classification layer: given the
configured agent modes it scores every mode against the request text and
returns the best match.  It has no external dependencies — no embedding model,
no vector store — which makes it fast, cheap and reproducible, and therefore a
natural predecessor of the semantic layer.

Scoring model (highest score wins):

1. **Keywords** — case-insensitive substring match; the weight is looked up in
   ``mode.weights`` (default ``1``).
2. **Phrases** — multi-word expressions optionally suffixed with
   ``":weight"`` (default weight ``2.0``).
3. **Patterns** — regex patterns; every match adds ``3.0``.

Ties are won by the mode that appears first in the configuration, so the
outcome depends only on the config and the text.
"""

import re

from typing import Iterable, List, Optional, Tuple

from llm_router_plugins.utils.routing.agentic_routing.config import AgentMode

__all__ = [
    "DEFAULT_KEYWORD_WEIGHT",
    "DEFAULT_PHRASE_WEIGHT",
    "PATTERN_WEIGHT",
    "score_mode",
    "score_to_similarity",
    "detect_heuristic",
]

#: Weight of a plain keyword match when ``weights`` has no entry for it.
DEFAULT_KEYWORD_WEIGHT = 1
#: Weight of a phrase match when the phrase carries no ``":weight"`` suffix.
DEFAULT_PHRASE_WEIGHT = 2.0
#: Weight added by a single regex pattern match.
PATTERN_WEIGHT = 3.0


def score_mode(mode: AgentMode, text_lower: str) -> float:
    """
    Return the heuristic score of *mode* for lower-cased *text_lower*.

    Parameters
    ----------
    mode : AgentMode
        The mode definition (keywords, phrases, patterns, weights).
    text_lower : str
        The request text, already lower-cased by the caller.

    Returns
    -------
    float
        The accumulated score; ``0.0`` when the mode does not match at all.

    Raises
    ------
    None
    """
    score = 0.0

    for keyword in mode.keywords:
        weight = mode.weights.get(keyword, DEFAULT_KEYWORD_WEIGHT)
        if keyword in text_lower:
            score += float(weight)

    for phrase in mode.phrases:
        phrase_text, weight = _split_phrase(phrase)
        if phrase_text in text_lower:
            score += weight

    for pattern in mode.patterns:
        try:
            if re.search(pattern, text_lower):
                score += PATTERN_WEIGHT
        except re.error:
            continue

    return score


def _split_phrase(phrase: str) -> Tuple[str, float]:
    """
    Split a ``"text:weight"`` phrase definition into its parts.

    Parameters
    ----------
    phrase : str
        The raw phrase definition; the weight suffix is optional.

    Returns
    -------
    Tuple[str, float]
        Lower-cased phrase text and its weight (``2.0`` when absent or
        not parseable).

    Raises
    ------
    None
    """
    if isinstance(phrase, str) and ":" in phrase:
        text, _, raw_weight = phrase.rpartition(":")
        try:
            return text.strip().lower(), float(raw_weight.strip())
        except ValueError:
            return phrase.strip().lower(), DEFAULT_PHRASE_WEIGHT
    return str(phrase).strip().lower(), DEFAULT_PHRASE_WEIGHT


def score_to_similarity(score: float) -> float:
    """
    Map an unbounded heuristic score onto a ``[0, 1)`` confidence value.

    Parameters
    ----------
    score : float
        The heuristic score of the winning mode.

    Returns
    -------
    float
        ``score / (score + 1)``, or ``0.0`` for a non-positive score.

    Raises
    ------
    None
    """
    if score <= 0:
        return 0.0
    return float(score) / (float(score) + 1.0)


def detect_heuristic(
    text: str,
    modes: Iterable[AgentMode],
) -> Tuple[Optional[AgentMode], float]:
    """
    Detect the best-matching mode by deterministic text scoring.

    Parameters
    ----------
    text : str
        The text to classify.
    modes : Iterable[AgentMode]
        The configured modes, in configuration order.

    Returns
    -------
    Tuple[Optional[AgentMode], float]
        The highest-scoring mode and its score, or ``(None, 0.0)`` when no
        mode scores above zero.  Ties are won by the mode defined first.

    Raises
    ------
    None
    """
    if not text:
        return None, 0.0

    text_lower = text.lower()
    best_mode: Optional[AgentMode] = None
    best_score = 0.0

    for mode in modes:
        score = score_mode(mode, text_lower)
        if score > best_score:
            best_score = score
            best_mode = mode

    return best_mode, best_score


def scored_modes(
    text: str, modes: Iterable[AgentMode]
) -> List[Tuple[AgentMode, float]]:
    """
    Return all modes with a positive heuristic score, best score first.

    Parameters
    ----------
    text : str
        The text to classify.
    modes : Iterable[AgentMode]
        The configured modes, in configuration order.

    Returns
    -------
    List[Tuple[AgentMode, float]]
        ``(mode, score)`` pairs with ``score > 0``, sorted by descending score
        while preserving configuration order for ties.

    Raises
    ------
    None
    """
    if not text:
        return []

    text_lower = text.lower()
    scored = [(mode, score_mode(mode, text_lower)) for mode in modes]
    ranked = [(mode, score) for mode, score in scored if score > 0]
    ranked.sort(key=lambda item: -item[1])
    return ranked
