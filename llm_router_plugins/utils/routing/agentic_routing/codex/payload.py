"""
Codex payload normalization — the deterministic input of the Codex cascade.

This module is the only place in the Codex routing plugin that knows *where*
Codex CLI metadata lives inside an OpenAI-Responses-style request.  Every
other layer (scoring, classification, the plugin itself) works on the
normalized, immutable :class:`CodexRequest` snapshot created here, which keeps
the decision logic independent from the wire format of the client.

Request classes (strict priority ``compaction > aux_title > main``)
-------------------------------------------------------------------
compaction
    ``request_kind == "compaction"`` — the CLI summarizes the conversation so
    far, so the request carries the whole (huge) transcript.
aux_title
    ``request_kind == "turn"`` with ``thread_source == "system"`` — the
    auto-generated one-line thread title: no tools at all and a
    ``codex_output_schema`` JSON schema in ``text``.
main
    Everything else — a regular agent turn, running in Plan or in Default
    collaboration mode.

Metadata locations
------------------
``client_metadata`` carries ``turn_id``, ``thread_id``, ``session_id``,
``root_turn_id``, ``x-codex-window-id`` and ``x-codex-turn-metadata``, the
last one being a **JSON string** holding ``request_kind``, ``thread_source``,
``sandbox_mode``, ``context_window_id`` and ``agent_name``.

Reading a payload never raises: a value that cannot be interpreted keeps the
field default, so an unexpected request degrades into the fallback mode
instead of failing.  ``parse_codex_payload`` never mutates its argument.
"""

import json
import re

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "REQUEST_CLASS_MAIN",
    "REQUEST_CLASS_AUX_TITLE",
    "REQUEST_CLASS_COMPACTION",
    "COLLABORATION_MODE_PLAN",
    "COLLABORATION_MODE_DEFAULT",
    "CodexRequest",
    "parse_codex_payload",
]

#: Rough character-to-token ratio used when only a character count is known.
_CHARS_PER_TOKEN = 4

#: The turn-metadata header of the Codex CLI is a JSON-encoded string.
_TURN_METADATA_KEY = "x-codex-turn-metadata"

REQUEST_CLASS_MAIN = "main"
REQUEST_CLASS_AUX_TITLE = "aux_title"
REQUEST_CLASS_COMPACTION = "compaction"

COLLABORATION_MODE_PLAN = "plan"
COLLABORATION_MODE_DEFAULT = "default"

#: First line of the Plan Mode block injected by the Codex CLI.
_PLAN_MODE_HEADING = "# Plan Mode"
#: First line of the Default Mode block injected by the Codex CLI.
_DEFAULT_MODE_HEADING = "# Collaboration Mode: Default"

# Two capture groups, so consumers must read ``match.group(1)``.
_COLLABORATION_MODE_RE = re.compile(
    r"<collaboration_mode>(.*?)(</collaboration_mode>|\Z)",
    re.DOTALL,
)


@dataclass(frozen=True)
class CodexRequest:
    """
    Immutable snapshot of the Codex fields relevant to routing.

    Parameters
    ----------
    session_id : str
        Identifier of the CLI session (``client_metadata.session_id``).
    thread_id : str
        Identifier of the conversation thread.
    turn_id : str
        Identifier of the current turn.
    root_turn_id : str
        Identifier of the turn that started the current branch.
    window_id : str
        Identifier of the context window (``<thread_id>:<number>``).
    agent_name : str
        Name of the agent emitting the request (``/root`` for the main agent).
    thread_source : str
        ``user`` for agent turns, ``system`` for CLI-generated helper calls.
    sandbox_mode : str
        Sandbox declared by the CLI, e.g. ``danger-full-access``.
    request_kind : str
        Kind declared by the CLI: ``turn`` or ``compaction``.
    context_window_id : str
        Identifier of the context window entry being summarized/extended.
    tool_names : Tuple[str, ...]
        Names of the advertised tools, in request order.  Entries without a
        ``name`` fall back to their ``type``.
    has_tools : bool
        Whether the request advertises at least one tool.
    structured_output : bool
        Whether ``text.format`` requests a ``json_schema`` response.
    reasoning_effort : str or None
        Declared reasoning effort (``reasoning.effort``), if any.
    parallel_tool_calls : bool
        Whether the request allows parallel tool calls.
    context_chars : int
        Serialized size of ``instructions`` plus ``input``, in characters.
    context_tokens : int
        Estimate of :attr:`context_chars` in tokens (``chars // 4``).
    collaboration_mode : str
        ``plan``, ``default`` or ``""`` when no block is present.
    latest_user_text : str
        Text of the most recent genuine user message.
    request_class : str
        One of ``main``, ``aux_title``, ``compaction``.
    """

    session_id: str = ""
    thread_id: str = ""
    turn_id: str = ""
    root_turn_id: str = ""
    window_id: str = ""
    agent_name: str = ""
    thread_source: str = ""
    sandbox_mode: str = ""
    request_kind: str = ""
    context_window_id: str = ""
    tool_names: Tuple[str, ...] = ()
    has_tools: bool = False
    structured_output: bool = False
    reasoning_effort: Optional[str] = None
    parallel_tool_calls: bool = False
    context_chars: int = 0
    context_tokens: int = 0
    collaboration_mode: str = ""
    latest_user_text: str = ""
    request_class: str = REQUEST_CLASS_MAIN


def parse_codex_payload(payload: Dict[str, Any]) -> "CodexRequest":
    """
    Normalize an OpenAI-Responses-style Codex payload.

    Parameters
    ----------
    payload : dict
        The incoming payload.  It is only read, never modified.

    Returns
    -------
    CodexRequest
        The normalized snapshot; every field falls back to its default when
        the payload does not carry (or does not carry a usable) value.

    Raises
    ------
    None
    """
    body: Dict[str, Any] = payload if isinstance(payload, dict) else {}
    client_metadata = _as_mapping(body.get("client_metadata"))
    turn_metadata = _decode_turn_metadata(client_metadata.get(_TURN_METADATA_KEY))
    items = _input_items(body)

    context_chars = _serialized_length(
        body.get("instructions")
    ) + _serialized_length(items)
    tool_names = _tool_names(body.get("tools"))

    return CodexRequest(
        session_id=_text(client_metadata.get("session_id"))
        or _text(turn_metadata.get("session_id")),
        thread_id=_text(client_metadata.get("thread_id"))
        or _text(turn_metadata.get("thread_id")),
        turn_id=_text(client_metadata.get("turn_id"))
        or _text(turn_metadata.get("turn_id")),
        root_turn_id=_text(client_metadata.get("root_turn_id"))
        or _text(turn_metadata.get("root_turn_id")),
        window_id=_text(client_metadata.get("x-codex-window-id"))
        or _text(turn_metadata.get("window_id")),
        agent_name=_text(turn_metadata.get("agent_name")),
        thread_source=_text(turn_metadata.get("thread_source")),
        sandbox_mode=_text(turn_metadata.get("sandbox_mode")),
        request_kind=_text(turn_metadata.get("request_kind")),
        context_window_id=_text(turn_metadata.get("context_window_id")),
        tool_names=tool_names,
        has_tools=bool(tool_names),
        structured_output=_is_structured_output(body.get("text")),
        reasoning_effort=_reasoning_effort(body.get("reasoning")),
        parallel_tool_calls=bool(body.get("parallel_tool_calls", False)),
        context_chars=context_chars,
        context_tokens=context_chars // _CHARS_PER_TOKEN,
        collaboration_mode=_collaboration_mode(items),
        latest_user_text=_latest_user_text(items, body),
        request_class=_request_class(
            _text(turn_metadata.get("request_kind")),
            _text(turn_metadata.get("thread_source")),
        ),
    )


def _decode_turn_metadata(raw: Any) -> Dict[str, Any]:
    """
    Decode the JSON-string turn-metadata header.

    Parameters
    ----------
    raw : Any
        The raw ``x-codex-turn-metadata`` value, normally a JSON string.

    Returns
    -------
    dict
        The decoded mapping, or an empty dict when it is absent or malformed.

    Raises
    ------
    None
    """
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _as_mapping(value: Any) -> Dict[str, Any]:
    """
    Return *value* when it is a dict, otherwise an empty dict.

    Parameters
    ----------
    value : Any
        Candidate mapping read from the payload.

    Returns
    -------
    dict
        *value* itself or ``{}``.

    Raises
    ------
    None
    """
    return value if isinstance(value, dict) else {}


def _input_items(payload: Dict[str, Any]) -> List[Any]:
    """
    Return the ``input`` items of *payload* as a list.

    Parameters
    ----------
    payload : dict
        The payload body.

    Returns
    -------
    list
        The ``input`` list, or an empty list when absent or not a list.

    Raises
    ------
    None
    """
    items = payload.get("input")
    return items if isinstance(items, list) else []


def _text(value: Any) -> str:
    """
    Return a stripped string for *value*, or ``""`` for anything else.

    Parameters
    ----------
    value : Any
        Candidate identifier value.

    Returns
    -------
    str
        The trimmed text or an empty string.

    Raises
    ------
    None
    """
    if isinstance(value, str):
        return value.strip()
    return ""


def _input_texts(item: Any) -> List[str]:
    """
    Return the ``input_text`` parts of a message item.

    Parameters
    ----------
    item : Any
        A single ``input`` entry.

    Returns
    -------
    List[str]
        The text parts of the item, empty for non-message entries.

    Raises
    ------
    None
    """
    if not isinstance(item, dict):
        return []
    content = item.get("content")
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    texts: List[str] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "input_text":
            text = part.get("text")
            if isinstance(text, str):
                texts.append(text)
    return texts


def _tool_names(tools: Any) -> Tuple[str, ...]:
    """
    Extract the advertised tool names.

    Parameters
    ----------
    tools : Any
        The ``tools`` list; entries without a ``name`` (for example
        ``{"type": "web_search"}` fall back to their ``type``.

    Returns
    -------
    Tuple[str, ...]
        The tool names in request order.

    Raises
    ------
    None
    """
    if not isinstance(tools, list):
        return ()
    names: List[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = _text(tool.get("name")) or _text(tool.get("type"))
        if name:
            names.append(name)
    return tuple(names)


def _is_structured_output(text_config: Any) -> bool:
    """
    Return whether *text_config* requests a ``json_schema`` response.

    Parameters
    ----------
    text_config : Any
        The optional ``text`` object of the payload.

    Returns
    -------
    bool

    Raises
    ------
    None
    """
    text_format = _as_mapping(text_config).get("format")
    return _text(_as_mapping(text_format).get("type")) == "json_schema"


def _reasoning_effort(reasoning: Any) -> Optional[str]:
    """
    Return the declared reasoning effort, or ``None``.

    Parameters
    ----------
    reasoning : Any
        The optional ``reasoning`` object of the payload.

    Returns
    -------
    str or None

    Raises
    ------
    None
    """
    effort = _text(_as_mapping(reasoning).get("effort"))
    return effort or None


def _collaboration_mode(items: List[Any]) -> str:
    """
    Detect the collaboration mode declared in the developer messages.

    Every ``input_text`` part of every ``role == "developer"`` message is
    scanned and the **last** declaration wins, so a Plan → Default switch in
    the middle of a session reclassifies the request correctly.

    Parameters
    ----------
    items : List[Any]
        The ``input`` items of the payload.

    Returns
    -------
    str
        ``plan``, ``default`` or ``""`` when no block is declared.

    Raises
    ------
    None
    """
    blocks: List[str] = []
    for item in items:
        if not isinstance(item, dict) or item.get("role") != "developer":
            continue
        for text in _input_texts(item):
            blocks.extend(
                match.group(1) for match in _COLLABORATION_MODE_RE.finditer(text)
            )
    if not blocks:
        return ""
    return _mode_heading(blocks[-1])


def _mode_heading(block: str) -> str:
    """
    Map a collaboration-mode block onto its mode label.

    Parameters
    ----------
    block : str
        The body of the last ``<collaboration_mode>`` block.

    Returns
    -------
    str
        ``plan``, ``default`` or ``""`` for an unrecognized heading.

    Raises
    ------
    None
    """
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(_PLAN_MODE_HEADING):
            return COLLABORATION_MODE_PLAN
        if stripped.startswith(_DEFAULT_MODE_HEADING):
            return COLLABORATION_MODE_DEFAULT
        return ""
    return ""


def _latest_user_text(items: List[Any], payload: Dict[str, Any], only_first: bool = False) -> str:
    """
    Return the text of the user message.

    When only_first is set to `False`, then the text is concatenated from the
    original user message and other LLM-generated messages (with the ` user ` role).

    Parameters
    ----------
    items : List[Any]
        The ``input`` items of the payload, scanned in reverse order.
    payload : dict
        The payload body, consulted for a legacy ``prompt`` fallback.

    Returns
    -------
    str
        The user text, or ``""`` when the request carries none.  Messages that
        only inject an ``<environment_context>`` block are skipped.

    Raises
    ------
    None
    """
    _full_user_msg = ""
    for item in reversed(items):
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message" or item.get("role") != "user":
            continue

        text = "\n".join(_input_texts(item)).strip()
        if not text or text.startswith("<environment_context>"):
            continue

        if not all_messages:
            return text

        _full_user_msg += text + "\n\n"

    _full_user_msg = _full_user_msg.strip()
    if len(_full_user_msg):
        return _full_user_msg

    return _text(payload.get("prompt"))


def _request_class(request_kind: str, thread_source: str) -> str:
    """
    Classify the request, with ``compaction`` ranked above ``aux_title``.

    Parameters
    ----------
    request_kind : str
        The decoded ``request_kind`` of the turn metadata.
    thread_source : str
        The decoded ``thread_source`` of the turn metadata.

    Returns
    -------
    str
        One of ``compaction``, ``aux_title``, ``main``.

    Raises
    ------
    None
    """
    if request_kind == REQUEST_CLASS_COMPACTION:
        return REQUEST_CLASS_COMPACTION
    if request_kind == "turn" and thread_source == "system":
        return REQUEST_CLASS_AUX_TITLE
    return REQUEST_CLASS_MAIN


def _serialized_length(value: Any) -> int:
    """
    Return the serialized length of *value* in characters.

    Parameters
    ----------
    value : Any
        A payload fragment (a string or a JSON-serializable structure).

    Returns
    -------
    int
        The character count, ``0`` for an absent value.

    Raises
    ------
    None
    """
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    try:
        return len(json.dumps(value, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return len(str(value))
