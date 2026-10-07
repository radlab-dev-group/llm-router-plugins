"""
Codex payload normalization — the deterministic input of the Codex cascade.

This module is the only place in the Codex routing plugin that knows *where*
Codex CLI metadata lives inside an OpenAI-Responses-style request.  Every
other layer (scoring, classification, the plugin itself) works on the
normalized, immutable :class:`CodexRequest` snapshot created by
:class:`CodexPayloadParser`, which keeps the decision logic independent from
the wire format of the client.

Request classes (strict priority ``compaction > aux_title > main``)
-------------------------------------------------------------------
compaction
    ``request_kind == "compaction"`` — the CLI summarizes the conversation so
    far, so the request carries the whole (huge) transcript.
aux_title
    The auto-generated one-line thread title: no tools at all and a
    ``codex_output_schema`` JSON schema in ``text``.  Declared by the CLI as
    ``thread_source == "system"``, and — CLI releases disagree on that header —
    recognized from that shape alone when the metadata stays silent.
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
instead of failing.  :meth:`CodexPayloadParser.parse` never mutates its
argument.
"""

import json
import re

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

__all__ = [
    "REQUEST_CLASS_MAIN",
    "REQUEST_CLASS_AUX_TITLE",
    "REQUEST_CLASS_COMPACTION",
    "COLLABORATION_MODE_PLAN",
    "COLLABORATION_MODE_DEFAULT",
    "DEFAULT_CLASSIFY_MAX_CHARS",
    "CodexActivity",
    "CodexRequest",
    "CodexPayloadParser",
]

#: Rough character-to-token ratio used when only a character count is known.
_CHARS_PER_TOKEN = 4

#: Default character budget for the text assembled for classification.
DEFAULT_CLASSIFY_MAX_CHARS = 4000

#: Characters separating commands in the optional history context.
_MESSAGE_SEPARATOR_CHARS = 2

#: The turn-metadata header of the Codex CLI is a JSON-encoded string.
_TURN_METADATA_KEY = "x-codex-turn-metadata"

#: ``thread_source`` of the helper calls the CLI emits on its own behalf.
_THREAD_SOURCE_SYSTEM = "system"

#: Opening of the instruction the CLI sends to generate the thread title.  The
#: instruction is always the first sentence of the user text of the request,
#: which is why the pattern is anchored there instead of searched for.
_AUX_TITLE_PROMPT_RE = re.compile(
    r"^(?:generate|produce|create|write|draft|suggest)\b[^\n]{0,120}\btitle\b",
    re.IGNORECASE,
)

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

_FOLLOW_UP_RE = re.compile(
    r"(?:tak|ok(?:ay)?|yes|sure|continue|kontynuuj|dalej|zrób to|zrob to|"
    r"do it|go ahead|proceed|(?:tak|ok|yes)[,\s]+(?:zrób to|zrob to|do it|go ahead))"
    r"[.!\s]*",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CodexActivity:
    """
    One assistant utterance or tool event in the active user turn.

    Parameters
    ----------
    kind : str
        ``assistant``, ``function_call`` or ``function_call_output``.
    text : str
        The utterance, the raw call arguments, or the tool output.
    name : str
        Tool name for a call; the name of the calling tool for an output, when
        a call with that ``call_id`` was seen in this turn.
    call_id : str
        Identifier linking a call to its output.
    event_id : str
        Identifier of the ``input`` item itself (``id``), when the payload
        carried one.  It survives retries of a *different* call, so it is what
        lets a session remember that this exact event was already accounted for
        while the same conversation history is replayed.
    """

    kind: str
    text: str
    name: str = ""
    call_id: str = ""
    event_id: str = ""


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
        Content characters of ``instructions`` plus ``input``: the length of
        every string in both fragments, JSON syntax (keys, braces, quoting)
        excluded.
    context_tokens : int
        Estimate of :attr:`context_chars` in tokens (``chars // 4``).
    collaboration_mode : str
        ``plan``, ``default`` or ``""`` when no block is present.
    latest_user_text : str
        Only the newest genuine user command, kept whole.
    user_history : Tuple[str, ...]
        Earlier commands, newest first, within the parser's character budget.
    assistant_messages:
        Copies of assistant output text after the newest genuine user command.
    activity : Tuple[CodexActivity, ...]
        Assistant and tool events after that command, oldest first.
    classify_max_chars : int
        Separate history budget and total semantic-query character cap.
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
    assistant_messages: Optional[List[Dict[str, Any]]] = None
    request_class: str = REQUEST_CLASS_MAIN
    user_history: Tuple[str, ...] = ()
    activity: Tuple[CodexActivity, ...] = ()
    classify_max_chars: int = DEFAULT_CLASSIFY_MAX_CHARS

    @property
    def intent_text(self) -> str:
        """Use earlier commands only to resolve a short, referential reply."""
        texts = [self.latest_user_text]
        if _FOLLOW_UP_RE.fullmatch(self.latest_user_text.strip()):
            for text in self.user_history:
                texts.append(text)
                if not _FOLLOW_UP_RE.fullmatch(text.strip()):
                    break
        return "\n\n".join(texts)


class CodexPayloadParser:
    """
    Normalize an OpenAI-Responses-style Codex payload into a CodexRequest.

    The parser owns every wire-format detail of the Codex CLI: where the
    metadata headers live, how the ``input`` items are read, and how the user
    text is assembled for classification.  It is stateless apart from its
    character budget, so a single instance is safely shared by every request
    (and by every thread) of a plugin.

    Reading never raises: a value that cannot be interpreted keeps its field
    default, so an unexpected payload degrades into the fallback mode instead
    of failing the request.  The payload is only read, never modified.

    Parameters
    ----------
    max_chars : int
        Character budget for :attr:`CodexRequest.latest_user_text`.  The
        newest user message is always kept whole; older ones are appended
        while the assembled text stays within the budget.  A non-positive
        value leaves the assembly unbounded.
    """

    def __init__(self, max_chars: int = DEFAULT_CLASSIFY_MAX_CHARS) -> None:
        """
        Store the character budget used to assemble the user text.

        Parameters
        ----------
        max_chars : int
            Character budget for :attr:`CodexRequest.latest_user_text`.

        Returns
        -------
        None

        Raises
        ------
        None
        """
        self._max_chars = max_chars

    def parse(self, payload: Any) -> CodexRequest:
        """
        Normalize an OpenAI-Responses-style Codex payload.

        Parameters
        ----------
        payload : Any
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
        client_metadata = self._as_mapping(body.get("client_metadata"))
        turn_metadata = self._decode_turn_metadata(
            client_metadata.get(_TURN_METADATA_KEY)
        )
        items = self._input_items(body)
        request_kind = self._text(turn_metadata.get("request_kind"))
        thread_source = self._text(turn_metadata.get("thread_source"))

        context_chars = self._content_length(body.get("instructions"))
        context_chars += self._content_length(items)
        tool_names = self._tool_names(body.get("tools"))
        has_tools = bool(tool_names)
        structured_output = self._is_structured_output(body.get("text"))
        latest_user_text = self._latest_user_text(items, body)
        user_turns = []
        for index, item in enumerate(items):
            text = self._user_text(item)
            if text:
                user_turns.append((index, text))
        active_items = items[user_turns[-1][0] + 1:] if user_turns else []
        assistant_messages = self._assistant_messages(active_items)

        return CodexRequest(
            session_id=self._text(client_metadata.get("session_id"))
            or self._text(turn_metadata.get("session_id")),
            thread_id=self._text(client_metadata.get("thread_id"))
            or self._text(turn_metadata.get("thread_id")),
            turn_id=self._text(client_metadata.get("turn_id"))
            or self._text(turn_metadata.get("turn_id")),
            root_turn_id=self._text(client_metadata.get("root_turn_id"))
            or self._text(turn_metadata.get("root_turn_id")),
            window_id=self._text(client_metadata.get("x-codex-window-id"))
            or self._text(turn_metadata.get("window_id")),
            agent_name=self._text(client_metadata.get("agent_name"))
            or self._text(turn_metadata.get("agent_name")),
            thread_source=thread_source,
            sandbox_mode=self._text(turn_metadata.get("sandbox_mode")),
            request_kind=request_kind,
            context_window_id=self._text(turn_metadata.get("context_window_id")),
            tool_names=tool_names,
            has_tools=has_tools,
            structured_output=structured_output,
            reasoning_effort=self._reasoning_effort(body.get("reasoning")),
            parallel_tool_calls=bool(body.get("parallel_tool_calls", False)),
            context_chars=context_chars,
            context_tokens=context_chars // _CHARS_PER_TOKEN,
            collaboration_mode=self._collaboration_mode(items),
            latest_user_text=latest_user_text,
            assistant_messages=assistant_messages,
            user_history=self._user_history(user_turns),
            activity=self._activity(active_items),
            classify_max_chars=self._max_chars,
            request_class=self._request_class(
                request_kind,
                thread_source,
                has_tools,
                structured_output,
                latest_user_text,
            ),
        )

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @classmethod
    def _tool_names(cls, tools: Any) -> Tuple[str, ...]:
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
            name = cls._text(tool.get("name")) or cls._text(tool.get("type"))
            if name:
                names.append(name)
        return tuple(names)

    @classmethod
    def _is_structured_output(cls, text_config: Any) -> bool:
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
        text_format = cls._as_mapping(text_config).get("format")
        return cls._text(cls._as_mapping(text_format).get("type")) == "json_schema"

    @classmethod
    def _reasoning_effort(cls, reasoning: Any) -> Optional[str]:
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
        effort = cls._text(cls._as_mapping(reasoning).get("effort"))
        return effort or None

    @classmethod
    def _collaboration_mode(cls, items: List[Any]) -> str:
        """
        Detect the collaboration mode declared in the developer messages.

        Every ``input_text`` part of every ``role == "developer"`` message is
        scanned and the **last** declaration wins, so a Plan → Default switch
        in the middle of a session reclassifies the request correctly.

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
            for text in cls._input_texts(item):
                blocks.extend(
                    match.group(1) for match in _COLLABORATION_MODE_RE.finditer(text)
                )
        if not blocks:
            return ""
        return cls._mode_heading(blocks[-1])

    @staticmethod
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

    def _latest_user_text(self, items: List[Any], payload: Dict[str, Any]) -> str:
        """Return only the newest command, or the legacy prompt fallback."""
        for item in reversed(items):
            text = self._user_text(item)
            if text:
                return text
        return self._text(payload.get("prompt"))

    @classmethod
    def _user_text(cls, item: Any) -> str:
        """Exclude environment-only messages without losing attached commands."""
        if not isinstance(item, dict):
            return ""
        if item.get("type") != "message" or item.get("role") != "user":
            return ""
        text = "\n".join(cls._input_texts(item)).strip()
        if (
            text.startswith("<environment_context>")
            and "</environment_context>" not in text
        ):
            return ""
        return re.sub(
            r"<environment_context>.*?</environment_context>", "", text, flags=re.DOTALL
        ).strip()

    def _user_history(self, turns: List[Tuple[int, str]]) -> Tuple[str, ...]:
        """Keep a bounded, separate history; never truncate the current command."""
        messages: List[str] = []
        remaining = self._max_chars
        for _, text in reversed(turns[:-1]):
            if self._max_chars > 0:
                if remaining <= 0:
                    break
                text = text[:remaining]
                remaining -= len(text) + _MESSAGE_SEPARATOR_CHARS
            messages.append(text)
        return tuple(messages)

    def _activity(self, items: List[Any]) -> Tuple[CodexActivity, ...]:
        """Copy ordered tool events, linking outputs only to calls in this turn."""
        events: List[CodexActivity] = []
        calls: Dict[str, str] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            event_id = self._text(item.get("id"))
            if kind == "message" and item.get("role") == "assistant":
                messages = self._assistant_messages([item]) or []
                for message in messages:
                    text = "\n".join(part["text"] for part in message["content"])
                    events.append(CodexActivity("assistant", text, event_id=event_id))
            elif kind in ("function_call", "custom_tool_call"):
                name = self._text(item.get("name"))
                call_id = self._text(item.get("call_id"))
                if call_id:
                    calls[call_id] = name
                arguments = (
                    item.get("input") if kind == "custom_tool_call"
                    else item.get("arguments")
                )
                events.append(
                    CodexActivity(
                        "function_call", self._text(arguments), name, call_id,
                        event_id,
                    )
                )
            elif kind in ("function_call_output", "custom_tool_call_output"):
                call_id = self._text(item.get("call_id"))
                text = self._text(item.get("output"))
                if self._max_chars > 0:
                    text = text[:self._max_chars]
                events.append(
                    CodexActivity(
                        "function_call_output", text, calls.get(call_id, ""), call_id,
                        event_id,
                    )
                )
        return tuple(events)

    @staticmethod
    def _assistant_messages(items: List[Any]) -> Optional[List[Dict[str, Any]]]:
        """
        Return copies of the assistant messages of *items*.

        Only ``type == "message"`` / ``role == "assistant"`` items count, and
        each returned copy keeps just its ``output_text`` content parts, so
        the semantic layer sees agent utterances and nothing else.  The
        original payload is never touched — the copies are new dicts — and an
        item without any ``output_text`` part is skipped.

        Parameters
        ----------
        items : List[Any]
            The ``input`` items of the payload.

        Returns
        -------
        Optional[List[Dict[str, Any]]]
            The new message dicts, oldest first, or ``None`` when the request
            carries no assistant message.
        """
        messages: List[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message" or item.get("role") != "assistant":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            output_parts = [
                part
                for part in content
                if isinstance(part, dict)
                and part.get("type") == "output_text"
                and isinstance(part.get("text"), str)
            ]
            if not output_parts:
                continue
            messages.append(
                {
                    "type": "message", "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": part["text"]}
                        for part in output_parts
                    ],
                }
            )
        return messages

    @staticmethod
    def _is_title_call(
        has_tools: bool, structured_output: bool, latest_user_text: str
    ) -> bool:
        """
        Recognize the title request from its shape, ignoring the metadata.

        CLI releases do not agree on what they declare in
        ``x-codex-turn-metadata`` — recent ones emit the title on a ``user``
        thread — so the call is also recognized by what makes it unmistakable:
        no advertised tools, a ``json_schema`` response, and the title
        instruction as the first sentence of the user text.  A regular agent
        turn advertises tools, so it can never match here.

        Parameters
        ----------
        has_tools : bool
            Whether the request advertises at least one tool.
        structured_output : bool
            Whether the request asks for a ``json_schema`` response.
        latest_user_text : str
            The assembled user text, newest message first.

        Returns
        -------
        bool
            ``True`` when all three properties of the title call are met.

        Raises
        ------
        None
        """
        if has_tools or not structured_output or not latest_user_text:
            return False
        return _AUX_TITLE_PROMPT_RE.match(latest_user_text) is not None

    @classmethod
    def _request_class(
        cls,
        request_kind: str,
        thread_source: str,
        has_tools: bool,
        structured_output: bool,
        latest_user_text: str,
    ) -> str:
        """
        Classify the request, with ``compaction`` ranked above ``aux_title``.

        The ``system`` thread source is enough on its own — every helper call
        the CLI emits for itself belongs on the auxiliary model — and the shape
        of the title call (:meth:`_is_title_call`) covers the CLI releases that
        do not mark it in the metadata at all.

        Parameters
        ----------
        request_kind : str
            The decoded ``request_kind`` of the turn metadata.
        thread_source : str
            The decoded ``thread_source`` of the turn metadata.
        has_tools : bool
            Whether the request advertises tools.
        structured_output : bool
            Whether the request asks for a ``json_schema`` response.
        latest_user_text : str
            The assembled user text, newest message first.

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
        if thread_source == _THREAD_SOURCE_SYSTEM:
            return REQUEST_CLASS_AUX_TITLE
        if cls._is_title_call(has_tools, structured_output, latest_user_text):
            return REQUEST_CLASS_AUX_TITLE
        return REQUEST_CLASS_MAIN

    @staticmethod
    def _content_length(value: Any) -> int:
        """
        Return the number of content characters in *value*.

        Walks the structure once, counting every string it meets (mapping keys
        included) plus the text of the numbers, booleans and byte strings in
        it, instead of JSON-serializing the fragment to measure it.  A round
        trip through :func:`json.dumps` costs milliseconds on a full Codex
        transcript — long enough to dominate request parsing — while it only
        ever produced a rough size estimate, which this reproduces within about
        half a percent (the JSON syntax it leaves out).

        Parameters
        ----------
        value : Any
            A payload fragment (a string or a nested JSON-like structure).

        Returns
        -------
        int
            The content character count, ``0`` for an absent value.  Containers
            already counted are skipped, so a self-referencing payload
            terminates instead of looping forever, and values that are neither
            text, a number, a boolean nor a container contribute nothing.

        Raises
        ------
        None
        """
        if value is None:
            return 0
        if isinstance(value, str):
            return len(value)

        total = 0
        counted: Set[int] = set()
        stack: List[Any] = [value]
        while stack:
            current = stack.pop()
            if isinstance(current, str):
                total += len(current)
            elif isinstance(current, dict):
                if id(current) in counted:
                    continue
                counted.add(id(current))
                for key, entry in current.items():
                    if isinstance(key, str):
                        total += len(key)
                    stack.append(entry)
            elif isinstance(current, (list, tuple)):
                if id(current) in counted:
                    continue
                counted.add(id(current))
                stack.extend(current)
            elif isinstance(current, bytes):
                total += len(current)
            elif isinstance(current, (int, float)):
                total += len(str(current))
        return total
