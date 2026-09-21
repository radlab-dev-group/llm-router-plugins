"""
Request signals — the deterministic input of the agentic routing cascade.

This module is the **only** place in the plugin that knows *where* agent
metadata lives inside an incoming payload.  Every other layer (rules,
capabilities, session affinity, heuristics, semantic detection) works on the
normalized, immutable :class:`RequestSignals` snapshot created here, which
keeps the deterministic layers independent from the wire format of individual
agents (OpenAI chat, Codex, Ollama, custom clients).

Locations are consulted in priority order, first non-empty declaration wins:

1. top level — ``{"task": "coding", "tools": true}``
2. ``metadata`` — ``{"metadata": {"task": "coding", "tools": true}}``
3. ``agent`` — ``{"agent": {"name": "codex", "task": "coding"}}``

The agent document from the routing design::

    {
      "model": "agentic",
      "agent": "codex",
      "session_id": "abc123",
      "task": "coding",
      "tools": true,
      "reasoning": true,
      "context_tokens": 42000
    }

normalizes to::

    RequestSignals(agent="codex", session_id="abc123", task="coding",
                   tools=True, tool_count=1, reasoning=True,
                   context_tokens=42000)

Reading a signal never raises: a value that cannot be interpreted keeps the
field default, so an unusual payload degrades into the semantic layers instead
of failing the request.

Field notes
-----------
agent : str
    Name of the calling agent/client (``agent`` as a string, ``agent.name``,
    ``agent_name``, ``metadata.agent``).  Normalized: lower-cased, trimmed,
    hyphens and spaces converted to underscores.
session_id : str
    Conversation/session identifier (``session_id``, ``conversation_id``,
    ``thread_id``) — the affinity key.
task : str
    Declared work task, the anchor of deterministic routing.
tools / tool_count
    Whether the request carries tools and how many.  ``tools`` accepts a
    boolean or a list/tuple/dict of tool definitions.
reasoning
    ``reasoning``, ``thinking`` or a non-empty ``reasoning_effort``.
vision
    Explicit flag or an image part in the message content.
structured_output
    ``structured_output`` flag or a ``response_format`` other than ``text``.
parallel_tools
    ``parallel_tool_calls`` / ``parallel_tools``.
context_tokens : int
    Declared context size, or an estimate of the whole conversation
    (``character count // 4``) when nothing is declared.
metadata : dict
    The raw ``metadata`` mapping, so rule conditions on metadata need no
    access to the payload itself.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

__all__ = ["RequestSignals"]

# Rough heuristic used when the caller does not declare its context size.
_CHARS_PER_TOKEN = 4

_IMAGE_PART_TYPES = frozenset({"image_url", "image", "input_image"})

_FLAG_FALSE_STRINGS = frozenset(
    {"", "0", "false", "no", "off", "none", "null", "text"}
)

_TEXT_PART_KEYS = ("text", "content")


@dataclass(frozen=True)
class RequestSignals:
    """
    Normalized, immutable view of the routing-relevant request metadata.

    All fields carry defaults so that a bare payload yields a valid, empty
    signal set (everything falsy, ``context_tokens`` estimated from text).

    Raises
    ------
    None
    """

    agent: str = ""
    session_id: str = ""
    task: str = ""
    tools: bool = False
    tool_count: int = 0
    reasoning: bool = False
    vision: bool = False
    structured_output: bool = False
    parallel_tools: bool = False
    context_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_payload(payload: Any, text: str = "") -> "RequestSignals":
        """
        Build :class:`RequestSignals` from a raw request *payload*.

        Parameters
        ----------
        payload : Any
            The request payload.  Anything that is not a mapping produces
            default signals.
        text : str, optional
            The user text already extracted from the payload, used only to
            estimate ``context_tokens`` when none is declared.

        Returns
        -------
        RequestSignals
            The normalized signal snapshot.

        Raises
        ------
        None
        """
        top = _as_mapping(payload)
        metadata = _as_mapping(top.get("metadata"))
        agent = _as_mapping(top.get("agent"))
        sources: Tuple[Mapping[str, Any], ...] = (top, metadata, agent)

        tools, tool_count = _resolve_tools(sources)
        explicit_context = _resolve_positive_int(
            _lookup(sources, ("context_tokens",))
        )

        return RequestSignals(
            agent=_resolve_agent(top, metadata, agent),
            session_id=_resolve_session_id(sources),
            task=_normalize_token(_lookup(sources, ("task",))),
            tools=tools,
            tool_count=tool_count,
            reasoning=_as_flag(
                _lookup(sources, ("reasoning", "thinking", "reasoning_effort"))
            ),
            vision=_as_flag(_lookup(sources, ("vision",)))
            or _payload_contains_images(top),
            structured_output=_as_flag(
                _lookup(
                    sources,
                    ("structured_output", "response_format", "output_format"),
                )
            ),
            parallel_tools=_as_flag(
                _lookup(sources, ("parallel_tool_calls", "parallel_tools"))
            ),
            context_tokens=(
                explicit_context
                if explicit_context > 0
                else _estimate_context_tokens(top, text)
            ),
            metadata=dict(metadata),
        )


def _as_mapping(value: Any) -> Dict[str, Any]:
    """
    Return *value* as a dict, or an empty dict when it is not a mapping.

    Parameters
    ----------
    value : Any
        Candidate value read from a payload.

    Returns
    -------
    dict
        The mapping itself, or ``{}``.

    Raises
    ------
    None
    """
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _normalize_token(value: Any) -> str:
    """
    Normalize an agent/task name for comparison.

    Parameters
    ----------
    value : Any
        Raw name; non-string values normalize to an empty string.

    Returns
    -------
    str
        Lower-cased, trimmed name with hyphens and spaces as underscores.

    Raises
    ------
    None
    """
    if not isinstance(value, str):
        return ""
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _lookup(
    sources: Tuple[Mapping[str, Any], ...],
    keys: Tuple[str, ...],
) -> Any:
    """
    Return the first declared value among *keys*, scanning *sources* in order.

    Parameters
    ----------
    sources : Tuple[Mapping[str, Any], ...]
        Payload locations to scan, highest precedence first.
    keys : Tuple[str, ...]
        Equivalent key names, highest precedence first.

    Returns
    -------
    Any
        The first value whose key is present, or ``None``.

    Raises
    ------
    None
    """
    for source in sources:
        for key in keys:
            if key in source:
                return source[key]
    return None


def _as_flag(value: Any) -> bool:
    """
    Interpret *value* as a boolean signal.

    Booleans pass through, containers are true when non-empty, ``None`` is
    false, and strings such as ``"false"``, ``"off"`` or ``"text"`` are false
    while any other non-empty string (``"high"``, ``"json_schema"``) is true.

    Parameters
    ----------
    value : Any
        Raw signal value.

    Returns
    -------
    bool
        The interpreted flag.

    Raises
    ------
    None
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in _FLAG_FALSE_STRINGS
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return bool(value)


def _resolve_agent(
    top: Mapping[str, Any],
    metadata: Mapping[str, Any],
    agent: Mapping[str, Any],
) -> str:
    """
    Resolve the calling agent name from the payload.

    ``agent`` as a plain string (``{"agent": "codex"}``) is used directly;
    otherwise the first non-empty name among ``agent.name``/``agent.agent``,
    top-level ``agent_name`` and ``metadata.agent`` wins.  A non-string
    ``agent`` value carries no name.

    Parameters
    ----------
    top : Mapping[str, Any]
        The payload itself.
    metadata : Mapping[str, Any]
        The ``metadata`` mapping.
    agent : Mapping[str, Any]
        The ``agent`` mapping (empty when ``agent`` is not a mapping).

    Returns
    -------
    str
        The normalized agent name, or an empty string.

    Raises
    ------
    None
    """
    raw_agent = top.get("agent")
    if isinstance(raw_agent, str) and raw_agent.strip():
        return _normalize_token(raw_agent)

    for candidate in (
        _lookup((agent,), ("name", "agent", "id")),
        top.get("agent_name"),
        metadata.get("agent"),
    ):
        normalized = _normalize_token(candidate)
        if normalized:
            return normalized
    return ""


def _resolve_session_id(sources: Tuple[Mapping[str, Any], ...]) -> str:
    """
    Resolve the session/conversation identifier used for affinity.

    Parameters
    ----------
    sources : Tuple[Mapping[str, Any], ...]
        Payload locations to scan.

    Returns
    -------
    str
        The trimmed identifier, or an empty string when absent.

    Raises
    ------
    None
    """
    value = _lookup(sources, ("session_id", "conversation_id", "thread_id"))
    if isinstance(value, str) and value.strip():
        return value.strip()
    return ""


def _resolve_tools(
    sources: Tuple[Mapping[str, Any], ...],
) -> Tuple[bool, int]:
    """
    Resolve the tools flag and the number of declared tools.

    The first location declaring ``tools`` decides the flag: a boolean is
    honored as-is, a list/tuple/dict of tool definitions is true when
    non-empty.  An explicit ``tool_count`` can only raise the count.

    Parameters
    ----------
    sources : Tuple[Mapping[str, Any], ...]
        Payload locations to scan.

    Returns
    -------
    Tuple[bool, int]
        ``(tools, tool_count)``.

    Raises
    ------
    None
    """
    value = _lookup(sources, ("tools",))

    tools = False
    tool_count = 0
    if isinstance(value, bool):
        tools = value
        tool_count = 1 if value else 0
    elif isinstance(value, (list, tuple)):
        tool_count = len(value)
        tools = tool_count > 0
    elif isinstance(value, dict):
        tool_count = len(value)
        tools = tool_count > 0
    elif value is not None:
        tools = _as_flag(value)
        tool_count = 1 if tools else 0

    declared = _resolve_positive_int(
        _lookup(sources, ("tool_count", "number_of_tools"))
    )
    if declared > tool_count:
        tool_count = declared
        tools = True

    return tools, tool_count


def _resolve_positive_int(value: Any) -> int:
    """
    Return *value* as a positive int, otherwise ``0``.

    Parameters
    ----------
    value : Any
        Raw numeric value (ints and numeric strings are accepted).

    Returns
    -------
    int
        The positive integer, or ``0`` when unusable.

    Raises
    ------
    None
    """
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value if value > 0 else 0
    if isinstance(value, float):
        return int(value) if value > 0 else 0
    if isinstance(value, str):
        try:
            parsed = int(value.strip())
        except ValueError:
            return 0
        return parsed if parsed > 0 else 0
    return 0


def _estimate_context_tokens(payload: Mapping[str, Any], text: str) -> int:
    """
    Estimate the context size of the whole conversation.

    Parameters
    ----------
    payload : Mapping[str, Any]
        The payload, consulted for ``messages``.
    text : str
        The already extracted user text (the last message).

    Returns
    -------
    int
        Estimated token count (``characters // 4``).

    Raises
    ------
    None
    """
    total = len(text)

    messages = payload.get("messages")
    if isinstance(messages, (list, tuple)):
        for message in messages[:-1]:
            total += _content_length(message)

    return total // _CHARS_PER_TOKEN


def _content_length(message: Any) -> int:
    """
    Return the character length of a single message's content.

    Parameters
    ----------
    message : Any
        A message entry; non-mapping entries are counted by ``str`` length.

    Returns
    -------
    int
        The character length of the content.

    Raises
    ------
    None
    """
    if not isinstance(message, Mapping):
        return len(str(message))

    content = message.get("content")
    if isinstance(content, str):
        return len(content)
    if isinstance(content, (list, tuple)):
        total = 0
        for part in content:
            if not isinstance(part, Mapping):
                total += len(str(part))
                continue
            value = _lookup((part,), _TEXT_PART_KEYS)
            if value is not None:
                total += len(str(value))
        return total
    return len(str(content)) if content is not None else 0


def _payload_contains_images(payload: Mapping[str, Any]) -> bool:
    """
    Detect an image content part anywhere in ``messages``.

    Parameters
    ----------
    payload : Mapping[str, Any]
        The payload to inspect.

    Returns
    -------
    bool
        ``True`` when at least one image part is present.

    Raises
    ------
    None
    """
    messages = payload.get("messages")
    if not isinstance(messages, (list, tuple)):
        return False

    for message in messages:
        if not isinstance(message, Mapping):
            continue
        parts = message.get("content")
        if not isinstance(parts, (list, tuple)):
            continue
        for part in parts:
            if not isinstance(part, Mapping):
                continue
            if str(part.get("type", "")).lower() in _IMAGE_PART_TYPES:
                return True
    return False
