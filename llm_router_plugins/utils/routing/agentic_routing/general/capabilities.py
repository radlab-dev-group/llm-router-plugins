"""
Capability routing — the deterministic hard-constraint layer.

A model is not only a string: every agent mode declares what its model can
actually do (see the ``capabilities`` block of a mode in the JSON config)::

    "capabilities": {
      "tool_calling": true,
      "parallel_tool_calls": true,
      "reasoning": true,
      "vision": false,
      "structured_output": true,
      "context_window": 131072
    }

An incoming request declares what it *needs* through its
:class:`~llm_router_plugins.utils.routing.agentic_routing.general.signals.RequestSignals`
(``tools: true``, ``context_tokens: 80000`` ...).  This module turns those
signals into a requirement set and checks it against the capabilities of a
mode.  It is a **gate**, not a scorer: it never decides *which* mode is the
best match for a request, it only rejects modes that cannot serve it.

Resolution rules:

- a capability key that is **absent** (or ``None``) on a mode means "unknown",
  which is treated as satisfied — modes without capability metadata keep the
  legacy behaviour and are never penalized;
- a requirement value of ``False``/absent is always satisfied — only *active*
  requirements constrain the choice;
- ``context_window`` compares numbers; a non-numeric capability value is
  treated as unknown and therefore satisfied.

When the mode selected by the deterministic or semantic layers cannot serve
the request, :func:`escalate` moves it to the most capable alternative — the
same decision every time, because candidates are ordered by the configuration
and only deterministic comparisons (declared context window) are used.
"""

import logging

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from llm_router_plugins.utils.routing.agentic_routing.general.config import AgentMode
from llm_router_plugins.utils.routing.agentic_routing.general.signals import RequestSignals

__all__ = [
    "REQUIREMENT_KEYS",
    "requirements_from_signals",
    "satisfies",
    "missing_capabilities",
    "filter_modes",
    "Escalation",
    "escalate",
]

#: Capability/requirement keys understood by this layer, in reporting order.
REQUIREMENT_KEYS: Tuple[str, ...] = (
    "tool_calling",
    "parallel_tool_calls",
    "reasoning",
    "vision",
    "structured_output",
    "context_window",
)

#: Maps a requirement key to the :class:`RequestSignals` attribute behind it.
_SIGNAL_ATTRS: Dict[str, str] = {
    "tool_calling": "tools",
    "parallel_tool_calls": "parallel_tools",
    "reasoning": "reasoning",
    "vision": "vision",
    "structured_output": "structured_output",
}


@dataclass(frozen=True)
class Escalation:
    """
    Result of a capability check on an already selected mode.

    Parameters
    ----------
    mode : AgentMode
        The mode to use — the original one, or a capable alternative.
    changed : bool
        ``True`` when the mode was replaced by another one.
    reason : str
        Short human-readable explanation, empty when nothing changed.
    """

    mode: AgentMode
    changed: bool
    reason: str = ""


def requirements_from_signals(signals: RequestSignals) -> Dict[str, Any]:
    """
    Build the requirement set implied by *signals*.

    Only **active** requirements are returned: a request that declares no
    tools produces no ``tool_calling`` entry, a request with
    ``context_tokens == 0`` produces no ``context_window`` entry.  An empty
    mapping therefore means "this request can be served by anything".

    Parameters
    ----------
    signals : RequestSignals
        Normalized request signals.

    Returns
    -------
    Dict[str, Any]
        Mapping of requirement key to required value.

    Raises
    ------
    None
    """
    requirements: Dict[str, Any] = {}

    for key, attr in _SIGNAL_ATTRS.items():
        if getattr(signals, attr, False):
            requirements[key] = True

    context_tokens = int(getattr(signals, "context_tokens", 0) or 0)
    if context_tokens > 0:
        requirements["context_window"] = context_tokens

    return requirements


def satisfies(
    capabilities: Optional[Mapping[str, Any]],
    requirements: Optional[Mapping[str, Any]],
) -> bool:
    """
    Return ``True`` when *capabilities* meet all *requirements*.

    Parameters
    ----------
    capabilities : Mapping[str, Any], optional
        Capability metadata of a mode.  ``None``/empty means "unknown" and is
        always satisfied.
    requirements : Mapping[str, Any], optional
        Required capabilities.  ``None``/empty is trivially satisfied.

    Returns
    -------
    bool
        ``True`` when no requirement is violated.

    Raises
    ------
    None
    """
    return not missing_capabilities(capabilities, requirements)


def missing_capabilities(
    capabilities: Optional[Mapping[str, Any]],
    requirements: Optional[Mapping[str, Any]],
) -> List[str]:
    """
    Return the list of *requirements* not met by *capabilities*.

    Parameters
    ----------
    capabilities : Mapping[str, Any], optional
        Capability metadata of a mode.  Absent keys count as satisfied.
    requirements : Mapping[str, Any], optional
        Required capabilities; falsy values are ignored.

    Returns
    -------
    List[str]
        Names of the violated capabilities in :data:`REQUIREMENT_KEYS` order.

    Raises
    ------
    None
    """
    if not requirements:
        return []

    caps = capabilities if isinstance(capabilities, Mapping) else {}
    missing: List[str] = []

    for key in REQUIREMENT_KEYS:
        if key not in requirements or not requirements[key]:
            continue
        if key == "context_window":
            if _violates_context_window(caps.get(key), requirements[key]):
                missing.append(key)
        elif key in caps and not bool(caps[key]):
            missing.append(key)

    return missing


def filter_modes(
    modes: Iterable[AgentMode],
    requirements: Optional[Mapping[str, Any]],
) -> List[AgentMode]:
    """
    Return the *modes* that satisfy all *requirements*, keeping config order.

    Parameters
    ----------
    modes : Iterable[AgentMode]
        Candidate modes.
    requirements : Mapping[str, Any], optional
        Required capabilities.

    Returns
    -------
    List[AgentMode]
        Capable modes; empty when nothing can serve the request.

    Raises
    ------
    None
    """
    if not requirements:
        return list(modes)
    return [mode for mode in modes if satisfies(mode.capabilities, requirements)]


def escalate(
    mode: AgentMode,
    modes: Iterable[AgentMode],
    requirements: Optional[Mapping[str, Any]],
    fallback_mode: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Escalation:
    """
    Replace *mode* with a capable alternative when it cannot serve the request.

    Candidates are the configured modes that satisfy *requirements*, evaluated
    in configuration order and ranked by their declared ``context_window``
    (largest wins, a missing value counts as ``0``, ties keep config order).
    *fallback_mode* is excluded from the candidates unless it is the only
    capable mode left — a request should not be escalated to the mode that
    catches undetectable requests.  When no mode can serve the request the
    original mode is kept and a warning is logged.

    Parameters
    ----------
    mode : AgentMode
        The mode selected by the preceding cascade layers.
    modes : Iterable[AgentMode]
        All configured modes.
    requirements : Mapping[str, Any], optional
        Capabilities required by the request.
    fallback_mode : str, optional
        Name of the fallback mode to avoid escalating to.
    logger : logging.Logger, optional
        Logger used for reporting.

    Returns
    -------
    Escalation
        The mode to use, whether it changed, and why.

    Raises
    ------
    None
    """
    if not requirements:
        return Escalation(mode=mode, changed=False)

    missing = missing_capabilities(mode.capabilities, requirements)
    if not missing:
        return Escalation(mode=mode, changed=False)

    all_modes = list(modes)
    candidates = [
        candidate
        for candidate in all_modes
        if candidate.name != mode.name
        and candidate.name != fallback_mode
        and satisfies(candidate.capabilities, requirements)
    ]
    if not candidates:
        candidates = [
            candidate
            for candidate in all_modes
            if candidate.name == fallback_mode
            and satisfies(candidate.capabilities, requirements)
        ]

    if not candidates:
        if logger:
            logger.warning(
                "AgenticRouting: mode '%s' cannot satisfy %s and no capable "
                "alternative exists — keeping '%s'",
                mode.name,
                ", ".join(missing),
                mode.name,
            )
        return Escalation(mode=mode, changed=False)

    best = max(candidates, key=_declared_context)

    if logger:
        logger.info(
            "AgenticRouting: escalating '%s' -> '%s' (missing: %s)",
            mode.name,
            best.name,
            ", ".join(missing),
        )
    return Escalation(
        mode=best, changed=True, reason="missing: " + ", ".join(missing)
    )


def _violates_context_window(declared: Any, required: Any) -> bool:
    """
    Return ``True`` when a declared context window is too small.

    Parameters
    ----------
    declared : Any
        ``context_window`` as configured on a mode; non-numeric means unknown.
    required : Any
        Required number of tokens.

    Returns
    -------
    bool
        ``True`` when the declared window is known and smaller than required.

    Raises
    ------
    None
    """
    declared_value = _as_int(declared)
    required_value = _as_int(required)
    if declared_value is None or required_value is None:
        return False
    return declared_value < required_value


def _declared_context(mode: AgentMode) -> int:
    """
    Return the numeric ``context_window`` of *mode*, ``0`` when unknown.

    Parameters
    ----------
    mode : AgentMode
        The mode to inspect.

    Returns
    -------
    int
        Declared context window, ``0`` when absent or not a number.

    Raises
    ------
    None
    """
    value = _as_int(mode.capabilities.get("context_window"))
    return value if value is not None else 0


def _as_int(value: Any) -> Optional[int]:
    """
    Return *value* as an ``int``, or ``None`` when it is not a number.

    Parameters
    ----------
    value : Any
        Value to convert.

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
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
