"""
Deterministic keyword scoring for the Codex routing plugin.

:class:`CodexModeScorer` scores the signals of
:class:`~llm_router_plugins.utils.routing.agentic_routing.codex.config.CodexMode`
against request text: :meth:`CodexModeScorer.detect_mode` returns the
best-scoring mode, :meth:`CodexModeScorer.score_mode` the score of a single
one, and :meth:`CodexModeScorer.rank_modes` exposes all candidates and evidence.
The reported similarity is heuristic strength, not calibrated probability.

Scoring model
-------------
A :class:`~llm_router_plugins.utils.routing.agentic_routing.codex.config.
CodexMode` carries three kinds of signals, all matched case-insensitively.
Keywords and phrases must start at a word boundary, so Polish inflected
forms still match (``test`` matches ``testów``) while mid-word hits
do not (``protest`` does not match ``test``); patterns are the exception,
matching a compiled regex:

===========  ==========================================  ==============
Signal       Weight                                      Source
===========  ==========================================  ==============
keyword      ``mode.weights[keyword]`` or ``1.0``        ``weights``
phrase       weight suffix ``":w"`` or ``2.0``           ``phrases``
pattern      ``3.0``                                     ``patterns``
===========  ==========================================  ==============

Within a mode, strongest matches win; equally weighted matches prefer the
longest span. Accepted spans cannot overlap and each configured signal counts
at most once. Forbidden action clauses are ignored locally. Ties between modes
abstain rather than favouring declaration order.

The score is mapped onto a strength with
:meth:`CodexModeScorer.score_to_similarity` (``score / (score + 1)``), a
saturating function that keeps every strength inside ``[0.0, 1.0)``.

Scoring runs off a per-mode *plan* — every signal prepared once and cached on
the scorer. Keyword and phrase spans use a substring scan with a word-start
boundary check; patterns use compiled regexes. Ranking and matches are returned
per call, never retained as request-specific state.

No machine learning, no embeddings, no network access — the same input always
produces the same score.
"""

import math
import re

from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

from llm_router_plugins.utils.routing.agentic_routing.codex.config import (
    CodexMode,
    CodexRoutingConfig,
)

__all__ = [
    "DEFAULT_PLAN_CACHE_MAX",
    "CodexModeScorer",
    "SignalMatch",
    "ModeScore",
]

#: Plans a scorer keeps before dropping its cache; configs hold a handful of
#: modes, so the default is never reached by a running plugin.
DEFAULT_PLAN_CACHE_MAX = 64


class SignalMatch(NamedTuple):
    """Accepted configured signal and its span in the lower-cased input."""

    signal: str
    start: int
    end: int
    weight: float


class ModeScore(NamedTuple):
    """One candidate's score and independent evidence, including zero scores."""

    mode: CodexMode
    score: float
    matches: Tuple[SignalMatch, ...]


class _ModePlan(NamedTuple):
    """
    Precompiled scoring program for one mode.

    Attributes
    ----------
    literals : Tuple[Tuple[str, float], ...]
        Lower-cased keyword and phrase needles with their weights, in
        declaration order.
    patterns : Tuple[re.Pattern, ...]
        Compiled regular expressions of the mode, each worth the configured
        pattern weight on a match.
    """

    literals: Tuple[Tuple[str, float], ...] = ()
    patterns: Tuple["re.Pattern[str]", ...] = ()


class CodexModeScorer:
    """
    Score the signals of the Codex work modes against request text.

    The scorer compiles one :class:`_ModePlan` per mode — lower-cased literals
    with their weights plus the validated regular expressions — and keeps it in
    a cache of its own, so scoring never recompiles a signal and one scorer
    never shares state with another.  Everything it does is a pure function of
    ``(mode, text)``: the instance is stateless apart from that cache and is
    safely shared by every request a plugin handles.

    Parameters
    ----------
    cache_max : int
        Number of compiled plans the scorer keeps before dropping the cache.
        A non-positive value rebuilds the plans on every call.
    negation_pattern : str
        Regex matching complete forbidden action spans. Empty/zero-width
        matches do not exclude any text.
    weights : Dict[str, float]
        Required keyword, phrase and pattern weights from the supplied config.
    """

    def __init__(
        self, *, negation_pattern: str, weights: Dict[str, float],
        cache_max: int = DEFAULT_PLAN_CACHE_MAX,
    ) -> None:
        """
        Create an empty plan cache with the given capacity.

        Parameters
        ----------
        cache_max : int
            Maximum number of cached plans.

        Returns
        -------
        None

        Raises
        ------
        None
        """
        self._cache_max = cache_max
        self._plan_cache: Dict[Tuple[Any, ...], _ModePlan] = {}
        self._negation = re.compile(negation_pattern, re.IGNORECASE)
        self._weights = CodexRoutingConfig._heuristic_weights(weights)

    def detect_mode(
        self,
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
            matches. On a positive tie returns ``(None, best_score)``.

        Raises
        ------
        None
        """
        ranking = self.rank_modes(text, modes)
        if not ranking or ranking[0].score <= 0:
            return None, 0.0
        best = ranking[0]
        if len(ranking) > 1 and best.score == ranking[1].score:
            return None, best.score
        return best.mode, best.score

    def rank_modes(self, text: str, modes: Iterable[CodexMode]) -> Tuple[ModeScore, ...]:
        """Return every candidate, descending by score, then by mode name.

        Ranking order is diagnostic only: it never resolves a tied decision.
        Spans refer to the lower-cased input, without logging the input itself.
        """
        text_lower = text.lower()
        blocked = tuple(
            match.span() for match in self._negation.finditer(text_lower)
            if match.end() > match.start()
        )
        scores = [self._score_evidence(mode, text_lower, blocked) for mode in modes]
        return tuple(sorted(scores, key=lambda item: (-item.score, item.mode.name)))

    def score_mode(self, mode: CodexMode, text_lower: str) -> float:
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
            Sum of independent, non-overlapping signals outside forbidden
            action clauses. Each declared signal contributes at most once.
            Invalid regexes and zero-width matches are skipped.

        Raises
        ------
        None
        """
        return self.rank_modes(text_lower, (mode,))[0].score

    def _score_evidence(
        self, mode: CodexMode, text: str, blocked: Tuple[Tuple[int, int], ...],
    ) -> ModeScore:
        plan = self._mode_plan(mode)
        candidates = []

        def allowed(start, end):
            return start < end and not any(
                start < right and left < end for left, right in blocked
            )

        for needle, weight in plan.literals:
            start = 0
            while needle and start < len(text):
                found = text.find(needle, start)
                if found < 0:
                    break
                if (
                    (found == 0 or not self._is_word_char(text[found - 1]))
                    and allowed(found, found + len(needle))
                ):
                    candidates.append(SignalMatch(
                        "literal:" + needle, found, found + len(needle), weight
                    ))
                    break
                start = found + 1
        for pattern in plan.patterns:
            for match in pattern.finditer(text):
                if allowed(match.start(), match.end()):
                    candidates.append(SignalMatch(
                        "pattern:" + pattern.pattern, match.start(), match.end(),
                        self._weights["pattern"],
                    ))
                    break
        accepted = []
        used = set()
        for match in sorted(
            (item for item in candidates if math.isfinite(item.weight) and item.weight > 0),
            key=lambda item: (
                -item.weight, -(item.end - item.start), item.start, item.signal
            ),
        ):
            if match.signal in used:
                continue
            if any(match.start < item.end and item.start < match.end for item in accepted):
                continue
            accepted.append(match)
            used.add(match.signal)
        matches = tuple(sorted(accepted, key=lambda item: (item.start, item.end)))
        return ModeScore(mode, sum((item.weight for item in matches), 0.0), matches)

    @staticmethod
    def score_to_similarity(score: float) -> float:
        """
        Map score to heuristic strength, not a calibrated probability.

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

    def clear_cache(self) -> None:
        """
        Drop every cached plan, forcing the next score to rebuild them.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        None
        """
        self._plan_cache.clear()

    def _mode_plan(self, mode: CodexMode) -> _ModePlan:
        """
        Return the cached scoring plan of *mode*.

        Parameters
        ----------
        mode : CodexMode
            The mode to look up.

        Returns
        -------
        _ModePlan
            The plan for the mode's current signal values; rebuilt when a
            signal or a weight changed.

        Raises
        ------
        None
        """
        signature = self._plan_signature(mode)
        plan = self._plan_cache.get(signature)
        if plan is None:
            plan = self._mode_signals(mode, self._weights)
            if len(self._plan_cache) >= self._cache_max:
                self._plan_cache.clear()
            self._plan_cache[signature] = plan
        return plan

    @staticmethod
    def _plan_signature(mode: CodexMode) -> Tuple[Any, ...]:
        """
        Return the cache key identifying the signals of *mode*.

        :class:`~...config.CodexMode` is not hashable because it carries a
        mapping, so the plan is keyed by the values it is built from instead.

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

    @classmethod
    def _mode_signals(cls, mode: CodexMode, default_weights) -> _ModePlan:
        """
        Prepare the literals and the compiled patterns of *mode*.

        Parameters
        ----------
        mode : CodexMode
            The mode to prepare.

        Returns
        -------
        _ModePlan
            Keyword and phrase needles with their weights, followed by the
            valid regular expressions; empty needles and invalid patterns are
            dropped, exactly as the scorer skips them.

        Raises
        ------
        None
        """
        weights = mode.weights if isinstance(mode.weights, dict) else {}
        literals: List[Tuple[str, float]] = []
        for keyword in mode.keywords:
            needle = keyword.strip().lower()
            if needle:
                literals.append((needle, cls._keyword_weight(
                    keyword, weights, default_weights["keyword"]
                )))
        for phrase in mode.phrases:
            needle, weight = cls._signal_weight(phrase, default_weights["phrase"])
            if needle:
                literals.append((needle, weight))
        patterns: List["re.Pattern[str]"] = []
        for pattern in mode.patterns:
            compiled = cls._compile_pattern(pattern)
            if compiled is not None:
                patterns.append(compiled)
        return _ModePlan(literals=tuple(literals), patterns=tuple(patterns))

    @staticmethod
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
            The compiled pattern, or ``None`` for an invalid expression or a
            value that is not a pattern at all.

        Raises
        ------
        None
        """
        try:
            return re.compile(pattern)
        except (re.error, TypeError):
            return None

    @staticmethod
    def _keyword_weight(
        keyword: str, weights: Dict[str, Any], default: float
    ) -> float:
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
            The configured weight, or the supplied default when the
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
                    return default
        return default

    @staticmethod
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

    @staticmethod
    def _literal_matches(text_lower: str, needle: str) -> bool:
        """
        Return whether *needle* occurs in *text_lower* at a word start.

        Stands in for a ``(?<!\\w)re.escape(needle)`` search: an occurrence
        counts only when the character in front of it is not a word character,
        so inflected forms match (``test`` finds ``testów``) while mid-word
        hits do not (``protest`` does not find ``test``).  Occurrences failing
        the boundary test are skipped and the scan continues.

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
            if found == 0 or not CodexModeScorer._is_word_char(
                text_lower[found - 1]
            ):
                return True
            start = found + 1

    @staticmethod
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
            ``True`` for exactly the characters ``re`` treats as word
            characters: alphanumerics (letters and numbers of any script) and
            ``"_"``.

        Raises
        ------
        None
        """
        return char.isalnum() or char == "_"
