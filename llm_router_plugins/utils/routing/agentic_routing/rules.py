"""
Declarative routing rules — the deterministic layer of the cascade.

A rule is a pure condition over the normalized :class:`RequestSignals` of a
request plus the user text.  No embeddings, no scoring, no guessing: a rule
either matches or it does not, which makes the decision reproducible and
auditable.  Rules are evaluated **before** the semantic layer, so a declared
``task``, a known ``agent`` or a hard requirement such as ``tools: true``
always beats a text-similarity guess.

JSON shape::

    "rules": [
      {
        "id": "task-coding",
        "priority": 100,
        "when": { "task": ["coding", "implement"] },
        "then": { "mode": "code" }
      },
      {
        "id": "needs-tools",
        "priority": 80,
        "when": { "requires_tools": true },
        "then": { "mode": "code" }
      },
      {
        "id": "urgent-flag",
        "priority": 110,
        "when": { "metadata": { "tenant": "batch" }, "text_matches": "^fw:" },
        "mode": "fallback"
      }
    ]

Supported ``when`` conditions (all present conditions must hold — logical
**AND**):

``agent``
    Normalized name of the calling agent (string or list of strings).
``task``
    Normalized declared task (string or list of strings — OR).
``requires_tools`` / ``requires_reasoning`` / ``requires_vision`` /
``requires_structured_output`` / ``requires_parallel_tools``
    ``true`` requires the signal to be set, ``false`` requires it to be unset.
``min_context_tokens`` / ``max_context_tokens``
    Inclusive bounds on ``RequestSignals.context_tokens``.
``text_contains_any``
    List of substrings, matched case-insensitively against the user text
    (OR).
``text_matches``
    Regular expression matched case-insensitively against the user text.
    An invalid expression never matches.
``metadata``
    Mapping of key/value pairs that must all be present and equal in
    ``RequestSignals.metadata``.

``mode`` may also be written as ``then.mode``.  ``id`` defaults to
``rule-<index>`` and ``priority`` to ``0``.  Rules are evaluated in descending
priority order; rules of equal priority keep their JSON order.  An empty or
missing ``when`` is a catch-all (a warning is logged when a logger is given).

Configuration mistakes raise :class:`ValueError` naming the exact JSON path,
so a broken config fails fast at plugin construction instead of silently
routing everything to the fallback mode.
"""

import logging
import re

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from llm_router_plugins.utils.routing.agentic_routing.signals import (
    RequestSignals,
    _normalize_token,
)

__all__ = ["RoutingRule", "parse_rules", "match_rule", "describe_rule"]

_REQUIRES_FIELDS: Dict[str, str] = {
    "requires_tools": "tools",
    "requires_reasoning": "reasoning",
    "requires_vision": "vision",
    "requires_structured_output": "structured_output",
    "requires_parallel_tools": "parallel_tools",
}

_SUPPORTED_WHEN_KEYS: Tuple[str, ...] = (
    "agent",
    "task",
    *_REQUIRES_FIELDS,
    "min_context_tokens",
    "max_context_tokens",
    "text_contains_any",
    "text_matches",
    "metadata",
)


@dataclass(frozen=True)
class RoutingRule:
    """
    A single deterministic routing rule.

    Parameters
    ----------
    id : str
        Stable identifier reported as ``routing.rule_id``.
    priority : int
        Evaluation order key — higher priority wins.
    when : Dict[str, Any]
        Raw condition mapping as written in the configuration.
    mode : str
        Name of the agent mode selected when the rule matches.
    """

    id: str
    priority: int
    when: Dict[str, Any]
    mode: str

    def matches(self, signals: RequestSignals, text: str = "") -> bool:
        """
        Return ``True`` when every condition of the rule holds.

        Parameters
        ----------
        signals : RequestSignals
            Normalized request signals.
        text : str, optional
            The extracted user text, used by the text conditions.

        Returns
        -------
        bool
            ``True`` when all conditions are satisfied.

        Raises
        ------
        None
        """
        return _matches(self.when, signals, text)


def parse_rules(
    raw: Any,
    known_modes: Iterable[str],
    logger: Optional[logging.Logger] = None,
) -> Tuple[RoutingRule, ...]:
    """
    Validate *raw* rule definitions and return them in evaluation order.

    Parameters
    ----------
    raw : Any
        The value of the top-level ``rules`` key.  ``None`` and empty lists
        produce an empty tuple.
    known_modes : Iterable[str]
        Names of the configured agent modes — a rule referencing anything else
        is a configuration error.
    logger : logging.Logger, optional
        Logger used to report catch-all rules.

    Returns
    -------
    Tuple[RoutingRule, ...]
        Rules sorted by descending priority, JSON order preserved on ties.

    Raises
    ------
    ValueError
        If ``rules`` is not a list, an item or condition is malformed, a mode
        is unknown, or rule ids are duplicated.
    """
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise ValueError(
            "AgenticRouting: 'rules' must be a list of rule objects, got "
            f"{type(raw).__name__} — check 'rules' in the JSON config"
        )

    available = list(known_modes)
    parsed: List[RoutingRule] = []
    seen: set = set()

    for index, item in enumerate(raw):
        rule = _parse_rule(index, item, available)
        if rule.id in seen:
            raise ValueError(
                f"AgenticRouting: duplicate rule id '{rule.id}' at "
                f"rules[{index}].id — rule ids must be unique"
            )
        seen.add(rule.id)
        if not rule.when and logger is not None:
            logger.warning(
                "AgenticRouting: rule '%s' has an empty 'when' condition — "
                "it matches every request (catch-all)",
                rule.id,
            )
        parsed.append(rule)

    return tuple(sorted(parsed, key=lambda item: -item.priority))


def match_rule(
    rules: Iterable[RoutingRule],
    signals: RequestSignals,
    text: str = "",
) -> Optional[RoutingRule]:
    """
    Return the first rule matching the request, or ``None``.

    Parameters
    ----------
    rules : Iterable[RoutingRule]
        Rules in evaluation order (see :func:`parse_rules`).
    signals : RequestSignals
        Normalized request signals.
    text : str, optional
        The extracted user text.

    Returns
    -------
    Optional[RoutingRule]
        The matching rule with the highest priority, or ``None`` when no rule
        applies.

    Raises
    ------
    None
    """
    for rule in rules:
        if rule.matches(signals, text):
            return rule
    return None


def describe_rule(rule: RoutingRule) -> str:
    """
    Return a compact, human-readable description of *rule*.

    Parameters
    ----------
    rule : RoutingRule
        The rule to describe.

    Returns
    -------
    str
        A string such as ``"task-coding(p=100) -> code"``.

    Raises
    ------
    None
    """
    return f"{rule.id}(p={rule.priority}) -> {rule.mode}"


def _parse_rule(
    index: int,
    item: Any,
    known_modes: List[str],
) -> RoutingRule:
    """
    Validate a single rule definition.

    Parameters
    ----------
    index : int
        Position in the ``rules`` list, used to build error paths.
    item : Any
        The raw rule definition.
    known_modes : List[str]
        Names of the configured agent modes.

    Returns
    -------
    RoutingRule
        The validated rule.

    Raises
    ------
    ValueError
        If the definition is malformed or references an unknown mode.
    """
    if not isinstance(item, Mapping):
        raise ValueError(
            f"AgenticRouting: rules[{index}] must be an object, got "
            f"{type(item).__name__} — check 'rules' in the JSON config"
        )

    mode, mode_path = _rule_mode(index, item, known_modes)
    when = _rule_conditions(index, item)
    priority = _rule_priority(index, item)
    rule_id = item.get("id")
    if not isinstance(rule_id, str) or not rule_id.strip():
        if rule_id is not None:
            raise ValueError(
                f"AgenticRouting: rules[{index}].id must be a non-empty "
                f"string, got {rule_id!r}"
            )
        rule_id = f"rule-{index}"

    return RoutingRule(
        id=rule_id.strip(),
        priority=priority,
        when=when,
        mode=mode,
    )


def _rule_mode(
    index: int, item: Mapping[str, Any], known_modes: List[str]
) -> Tuple[str, str]:
    """
    Resolve and validate the target mode of a rule definition.

    Parameters
    ----------
    index : int
        Position in the ``rules`` list.
    item : Mapping[str, Any]
        The raw rule definition.
    known_modes : List[str]
        Names of the configured agent modes.

    Returns
    -------
    Tuple[str, str]
        The validated mode name and the config path it was read from.

    Raises
    ------
    ValueError
        If ``then`` is malformed, the mode is missing, not a string, or not a
        configured agent mode.
    """
    path = f"rules[{index}].mode"
    source = item

    if "then" in item:
        path = f"rules[{index}].then.mode"
        then = item["then"]
        if not isinstance(then, Mapping):
            raise ValueError(
                f"AgenticRouting: rules[{index}].then must be an object with a "
                f"'mode' key, got {type(then).__name__}"
            )
        source = then

    value = source.get("mode")
    if value is None:
        raise ValueError(
            f"AgenticRouting: {path} is missing — a rule must select one of "
            f"the configured modes {known_modes}"
        )
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"AgenticRouting: {path} must be a non-empty string, got {value!r}"
        )

    mode = value.strip().lower().replace("-", "_").replace(" ", "_")
    if mode not in known_modes:
        raise ValueError(
            f"AgenticRouting: {path} references unknown mode '{value}' — "
            f"available modes: {known_modes}"
        )
    return mode, path


def _rule_conditions(index: int, item: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Validate the ``when`` mapping of a rule definition.

    Parameters
    ----------
    index : int
        Position in the ``rules`` list.
    item : Mapping[str, Any]
        The raw rule definition.

    Returns
    -------
    Dict[str, Any]
        A copy of the conditions (empty for a catch-all rule).

    Raises
    ------
    ValueError
        If ``when`` is not an object or contains an unsupported key.
    """
    when = item.get("when", {})
    if when is None:
        return {}
    if not isinstance(when, Mapping):
        raise ValueError(
            f"AgenticRouting: rules[{index}].when must be an object, got "
            f"{type(when).__name__} — supported conditions: "
            f"{list(_SUPPORTED_WHEN_KEYS)}"
        )

    unknown = [key for key in when if key not in _SUPPORTED_WHEN_KEYS]
    if unknown:
        paths = [f"rules[{index}].when.{key}" for key in unknown]
        raise ValueError(
            f"AgenticRouting: unsupported rule condition(s) {paths} — "
            f"supported conditions: {list(_SUPPORTED_WHEN_KEYS)}"
        )

    for key in ("min_context_tokens", "max_context_tokens"):
        if key in when and _to_int(when[key]) is None:
            raise ValueError(
                f"AgenticRouting: rules[{index}].when.{key} must be an "
                f"integer, got {when[key]!r}"
            )

    if "metadata" in when and not isinstance(when["metadata"], Mapping):
        raise ValueError(
            f"AgenticRouting: rules[{index}].when.metadata must be an object "
            f"of key/value pairs, got {type(when['metadata']).__name__}"
        )

    return dict(when)


def _rule_priority(index: int, item: Mapping[str, Any]) -> int:
    """
    Validate the ``priority`` of a rule definition.

    Parameters
    ----------
    index : int
        Position in the ``rules`` list.
    item : Mapping[str, Any]
        The raw rule definition.

    Returns
    -------
    int
        The priority, ``0`` when not declared.

    Raises
    ------
    ValueError
        If ``priority`` is present but not an integer.
    """
    if "priority" not in item or item["priority"] is None:
        return 0

    value = item["priority"]
    if isinstance(value, bool) or _to_int(value) is None:
        raise ValueError(
            f"AgenticRouting: rules[{index}].priority must be an integer, "
            f"got {value!r}"
        )
    return _to_int(value) or 0


def _to_int(value: Any) -> Optional[int]:
    """
    Return *value* as an int, or ``None`` when it is not numeric.

    Parameters
    ----------
    value : Any
        Candidate numeric value.

    Returns
    -------
    Optional[int]
        The integer value, or ``None``.

    Raises
    ------
    None
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _matches(when: Mapping[str, Any], signals: RequestSignals, text: str) -> bool:
    """
    Evaluate all conditions of a rule against the request.

    Parameters
    ----------
    when : Mapping[str, Any]
        The rule conditions.
    signals : RequestSignals
        Normalized request signals.
    text : str
        The extracted user text.

    Returns
    -------
    bool
        ``True`` when every condition holds.

    Raises
    ------
    None
    """
    if "agent" in when and not _name_in(when["agent"], signals.agent):
        return False

    if "task" in when and not _name_in(when["task"], signals.task):
        return False

    for key, field_name in _REQUIRES_FIELDS.items():
        if key in when and bool(getattr(signals, field_name)) != bool(when[key]):
            return False

    context = signals.context_tokens
    if "min_context_tokens" in when and context < (
        _to_int(when["min_context_tokens"]) or 0
    ):
        return False
    if "max_context_tokens" in when:
        maximum = _to_int(when["max_context_tokens"])
        if maximum is not None and context > maximum:
            return False

    if "text_contains_any" in when and not _contains_any(
        when["text_contains_any"], text
    ):
        return False

    if "text_matches" in when and not _regex_matches(when["text_matches"], text):
        return False

    if "metadata" in when and not _metadata_matches(
        when["metadata"], signals.metadata
    ):
        return False

    return True


def _name_in(expected: Any, actual: str) -> bool:
    """
    Return ``True`` when *actual* equals *expected* (or any list member).

    Parameters
    ----------
    expected : Any
        A name or a list of names as written in the configuration.
    actual : str
        The normalized signal value.

    Returns
    -------
    bool
        ``True`` when the name matches.

    Raises
    ------
    None
    """
    if not actual:
        return False
    candidates = expected if isinstance(expected, (list, tuple, set)) else [expected]
    return any(_normalize_token(candidate) == actual for candidate in candidates)


def _contains_any(expected: Any, text: str) -> bool:
    """
    Return ``True`` when the text contains any of the *expected* substrings.

    Parameters
    ----------
    expected : Any
        A substring or a list of substrings.
    text : str
        The user text.

    Returns
    -------
    bool
        ``True`` when at least one substring occurs in the text.

    Raises
    ------
    None
    """
    if not text:
        return False
    needles = expected if isinstance(expected, (list, tuple, set)) else [expected]
    lowered = text.lower()
    return any(str(needle).lower() in lowered for needle in needles if str(needle))


def _regex_matches(pattern: Any, text: str) -> bool:
    """
    Return ``True`` when *pattern* matches *text* (case-insensitively).

    Parameters
    ----------
    pattern : Any
        The regular expression.
    text : str
        The user text.

    Returns
    -------
    bool
        ``True`` on a match; an invalid pattern never matches.

    Raises
    ------
    None
    """
    if not text or not isinstance(pattern, str) or not pattern:
        return False
    try:
        return re.search(pattern, text, re.IGNORECASE) is not None
    except re.error:
        return False


def _metadata_matches(expected: Any, metadata: Mapping[str, Any]) -> bool:
    """
    Return ``True`` when every expected key/value pair is present.

    Parameters
    ----------
    expected : Any
        Mapping of required metadata entries.
    metadata : Mapping[str, Any]
        The request metadata.

    Returns
    -------
    bool
        ``True`` when all pairs are equal.

    Raises
    ------
    None
    """
    if not isinstance(expected, Mapping):
        return False
    if not expected:
        return True
    return all(metadata.get(key) == value for key, value in expected.items())
