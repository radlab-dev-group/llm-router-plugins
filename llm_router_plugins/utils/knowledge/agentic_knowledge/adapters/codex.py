"""
Codex adapter: the knowledge layer's view of the Codex CLI.

Codex speaks the OpenAI Responses wire format and — decisively for this plugin —
replays the whole turn in every request.  One request therefore contains the
user's command, every tool call the agent made since, and every result they
returned, which is enough to reconstruct what happened without the plugin
keeping any state of its own.

Rather than parse that again, the adapter reuses the routing plugin's own
normalizer (:class:`CodexPayloadParser`) and its phase rules
(:func:`collect_phase_evidence`), so "the agent is patching" means one thing in
both plugins.  A contract test pins the two structures the knowledge layer
depends on; if routing ever renames them, the test fails here instead of a
recipe quietly never being written again.

Knowledge is injected as one extra ``developer`` item at the very end of
``input``.  Not in ``instructions``: that is the stable prefix of the prompt,
and appending to it invalidates the provider's prompt cache on every request.
An item at the end is appended to the cache instead, and it disappears with the
request that carried it, which is why the same set of documents has to be
re-injected for every request of one command generation.
"""

import hashlib
import json
import os
import re

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llm_router_plugins.utils.knowledge.agentic_knowledge.adapters.base import (
    AgentAdapter,
    AgentTurn,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.scope import (
    RepositoryScope,
    ScopeResolver,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.payload import (
    CodexActivity,
    CodexPayloadParser,
    CodexRequest,
    REQUEST_CLASS_MAIN,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.phase import (
    CodexPhaseConfig,
    PhaseEvidence,
    collect_phase_evidence,
)

__all__ = [
    "CodexAgentAdapter",
    "KNOWLEDGE_MARKER",
    "load_phase_rules",
    "patch_bodies",
]

#: Routing plugin identifier that marks a request as Codex's.
CODEX_ROUTING_PLUGIN = "agentic_routing_codex"

#: Marker wrapped around injected knowledge; also its idempotency detector.
KNOWLEDGE_MARKER = "agentic_knowledge"

_INJECTION_OPEN = f"<{KNOWLEDGE_MARKER}>"
_INJECTION_CLOSE = f"</{KNOWLEDGE_MARKER}>"

#: Env var pointing at an alternative Codex routing config for phase rules.
PHASE_CONFIG_ENV = "LLM_ROUTER_KNOWLEDGE_AGENTIC_CODEX_PHASE_CONFIG"

_TURN_METADATA_KEY = "x-codex-turn-metadata"
_CODEX_METADATA_KEYS = (
    "session_id",
    "thread_id",
    "turn_id",
    "root_turn_id",
    "x-codex-window-id",
    "agent_name",
)

_PATCH_FILE_RE = re.compile(
    r"^\*\*\* (?:Add|Update|Delete) File: (.+)$", re.MULTILINE
)
_PATCH_MOVE_RE = re.compile(r"^\*\*\* Move to: (.+)$", re.MULTILINE)

#: Argument keys a patch envelope can ride in, in the order Codex sends them.
_PATCH_ARGUMENT_KEYS = ("patch", "input", "cmd", "command")

#: The first line of a well-formed patch envelope.
_PATCH_BEGIN = "*** Begin Patch"

#: A here-document introducer and its terminating word.
_HERE_DOCUMENT_RE = re.compile(r"<<<?-?\s*(['\"]?)(\w+)\1")

#: Ceiling of the serialized window of a compaction request.
_MAX_WINDOW_CHARS = 400_000

#: Ceiling of a client-supplied identifier before it is hashed down.  Same bound
#: the Claude Code adapter puts on its session identifiers.
_MAX_ID_CHARS = 128


def load_phase_rules(
    path: Optional[Path] = None, logger: Any = None
) -> Optional[CodexPhaseConfig]:
    """
    Return the phase rules the Codex router uses, or ``None``.

    The rules are read from the routing plugin's own configuration so that one
    definition of "this call is a patch" serves both plugins.  Nothing here is
    fatal: without rules the knowledge layer still injects, it simply records
    no episodes — which is worth a log line, since "no episodes" is otherwise
    indistinguishable from a knowledge base with nothing in it.

    Parameters
    ----------
    path : pathlib.Path, optional
        Alternative configuration file; defaults to
        :data:`PHASE_CONFIG_ENV`, then to the bundled Codex routing config.
    logger : logging.Logger, optional
        Where to report rules that could not be read.

    Returns
    -------
    CodexPhaseConfig or None
        The rules, or ``None`` when they cannot be read, are malformed, or are
        switched off.
    """
    candidate = path or _phase_config_path()
    if candidate is None:
        return None
    try:
        with open(candidate, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        phase = raw.get("settings", {}).get("phase")
        if not isinstance(phase, dict):
            return None
        rules = CodexPhaseConfig.from_raw(phase)
    except Exception as exc:  # a foreign config must not break this plugin
        if logger is not None:
            logger.warning(
                "agentic knowledge: episode mining off, phase rules unreadable"
                " (%s: %s)",
                type(exc).__name__,
                candidate,
            )
        return None
    if not rules.enabled and logger is not None:
        logger.info("agentic knowledge: episode mining off, phase rules disabled")
    return rules if rules.enabled else None


def _phase_config_path() -> Optional[Path]:
    """Locate the configuration the phase rules should come from."""
    declared = os.getenv(PHASE_CONFIG_ENV, "").strip()
    if declared:
        return Path(declared)
    bundled = (
        Path(__file__).resolve().parents[4]
        / "resources"
        / "routing"
        / "agentic_routing_codex.json"
    )
    return bundled if bundled.is_file() else None


class CodexAgentAdapter(AgentAdapter):
    """
    Reads and enriches Codex CLI requests.

    Parameters
    ----------
    parser : CodexPayloadParser, optional
        Payload normalizer.  A pre-built one is injected by tests.
    phase_rules : CodexPhaseConfig, optional
        Phase rules; loaded from the Codex routing config when omitted.
    scope_resolver : ScopeResolver, optional
        Repository scope resolver.
    """

    name = "codex"

    def __init__(
        self,
        parser: Optional[CodexPayloadParser] = None,
        phase_rules: Optional[CodexPhaseConfig] = None,
        scope_resolver: Optional[ScopeResolver] = None,
    ) -> None:
        """Bind the collaborators, loading the phase rules when needed."""
        self._parser = parser or CodexPayloadParser()
        self._phase_rules = (
            phase_rules if phase_rules is not None else load_phase_rules()
        )
        self._scope_resolver = scope_resolver or ScopeResolver()

    @property
    def phase_rules(self) -> Optional[CodexPhaseConfig]:
        """The phase rules in force, ``None`` when episode mining is off."""
        return self._phase_rules

    def matches(self, payload: Dict[str, Any]) -> bool:
        """
        Return whether *payload* is a Codex request.

        Routing already marks the requests it classified; anything else is
        recognized structurally, so the knowledge layer also works when the
        routing plugin is not in the pipeline.

        Parameters
        ----------
        payload : Dict[str, Any]
            The request payload.

        Returns
        -------
        bool
            ``True`` when this adapter claims the payload.
        """
        if not isinstance(payload, dict):
            return False
        routing = payload.get("routing")
        if (
            isinstance(routing, dict)
            and routing.get("plugin") == CODEX_ROUTING_PLUGIN
        ):
            return True
        if "input" not in payload:
            return False
        metadata = payload.get("client_metadata")
        if not isinstance(metadata, dict):
            return False
        if _TURN_METADATA_KEY in metadata:
            return True
        return any(metadata.get(key) for key in _CODEX_METADATA_KEYS)

    def parse(self, payload: Dict[str, Any]) -> Optional[AgentTurn]:
        """
        Return the turn *payload* describes.

        Parameters
        ----------
        payload : Dict[str, Any]
            The request payload.

        Returns
        -------
        AgentTurn or None
            The turn, or ``None`` when the payload carries neither a work mode
            nor any activity to learn from.
        """
        request: CodexRequest = self._parser.parse(payload)
        scope = self._scope_resolver.resolve(payload)
        routing = (
            payload.get("routing")
            if isinstance(payload.get("routing"), dict)
            else {}
        )
        evidence = self._evidence(request)
        mode = str(payload.get("agent_mode") or routing.get("agent_mode") or "")
        if not mode and not request.activity:
            return None
        paths = _touched_paths(request.activity, scope)
        return AgentTurn(
            agent=self.name,
            scope=scope.key,
            repo_root=scope.repo_root,
            session_id=request.session_id,
            thread_id=request.thread_id,
            turn_id=request.turn_id,
            root_turn_id=request.root_turn_id,
            generation=_identifier(request),
            request_class=request.request_class,
            mode=mode,
            routing_source=str(routing.get("source") or ""),
            model=str(payload.get("model") or ""),
            latest_user_text=request.latest_user_text,
            window_id=request.window_id or request.context_window_id,
            activity=request.activity,
            evidence=evidence,
            window_text=(
                _window_text(payload)
                if request.request_class != REQUEST_CLASS_MAIN
                else ""
            ),
            touched_paths=paths,
        )

    def inject(self, payload: Dict[str, Any], text: str) -> Dict[str, Any]:
        """
        Append the knowledge block to ``input`` as a ``developer`` message.

        Idempotent: a payload that already carries the block is returned as
        already enriched, so a retry of one request cannot stack two contexts.

        Parameters
        ----------
        payload : Dict[str, Any]
            The request payload.
        text : str
            The rendered knowledge block, markers included.

        Returns
        -------
        Dict[str, Any]
            The payload.
        """
        block = text if _INJECTION_OPEN in text else f"{_INJECTION_OPEN}\n{text}"
        body = payload.get("input")
        if isinstance(body, str):
            payload["input"] = [
                {"role": "user", "content": [{"type": "input_text", "text": body}]},
                _developer_item(block),
            ]
        elif isinstance(body, list):
            if _already_injected(body):
                return payload
            body.append(_developer_item(block))
        # Any other shape is not an input this adapter understands.  Replacing
        # it would drop whatever the agent actually sent to carry one context
        # block, so the request goes out as it came in and the block is skipped.
        return payload

    def set_model(self, payload: Dict[str, Any], model_name: str) -> None:
        """
        Point *payload* at *model_name*.

        Parameters
        ----------
        payload : Dict[str, Any]
            The request payload.
        model_name : str
            The model to request instead.
        """
        payload["model"] = model_name

    # ----------------------------------------------------------- internals -
    def _evidence(self, request: CodexRequest) -> Tuple[PhaseEvidence, ...]:
        """Collect phase evidence, or nothing when there are no rules."""
        if self._phase_rules is None or not request.activity:
            return ()
        try:
            return collect_phase_evidence(request.activity, self._phase_rules)
        except Exception:  # a phase-rule change must not break enrichment
            return ()


def _developer_item(text: str) -> Dict[str, Any]:
    """Build one ``developer`` message of the Responses wire format."""
    return {
        "type": "message",
        "role": "developer",
        "content": [{"type": "input_text", "text": text}],
    }


def _already_injected(items: List[Any]) -> bool:
    """
    Whether this adapter's own block already ends the input.

    Only the tail is read, and only in our own message shape.  Scanning every
    item for the markers would treat a conversation that merely quotes them —
    an agent working on this very plugin, say — as a request already enriched,
    and knowledge would quietly stop being injected in the one repository where
    it is being written.
    """
    if not items:
        return False
    last = items[-1]
    if not isinstance(last, dict) or last.get("role") != "developer":
        return False
    content = last.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(part, dict)
        and isinstance(part.get("text"), str)
        and _INJECTION_OPEN in part["text"]
        for part in content
    )


def _identifier(request: CodexRequest) -> str:
    """
    Identity of the command generation, ``turn_id`` where the CLI gives one.

    The value comes straight from the client's metadata and becomes part of a
    Redis key, so an over-long one is hashed rather than truncated: a client
    controlling this field must not be able to inflate keys, and a truncation
    would let two different long ids collide onto one generation's cached
    context.  Hashing keeps them distinct and bounded, matching the fallback
    branch and the Claude Code adapter's own cap.
    """
    if request.turn_id:
        turn_id = request.turn_id
        if len(turn_id) <= _MAX_ID_CHARS:
            return turn_id
        return "t" + hashlib.sha256(turn_id.encode("utf-8")).hexdigest()[:32]
    if request.thread_id and request.input_items:
        # No turn identifier: fall back to something that still distinguishes
        # the requests of one thread from each other, and from other threads.
        return (
            "i"
            + hashlib.sha256(
                f"{request.thread_id}\x00{request.input_items}".encode("utf-8")
            ).hexdigest()[:16]
        )
    return ""


def patch_bodies(text: str) -> List[str]:
    """
    Return the texts of one tool call that can hold a patch envelope.

    A Codex patch does not arrive as plain text: the arguments are JSON-encoded,
    so the envelope's line breaks are ``\\n`` escapes inside one string — and,
    more often still, the patch is a here-document inside an ``exec_command``,
    where several patches of one turn can ride together.  Matching the raw
    arguments with a line-anchored pattern therefore finds nothing, which is
    why the envelopes are read back out here: the ``patch``/``input``
    arguments, the shell commands, and the body of each here-document a command
    feeds to ``apply_patch``.  A here-document counts only then — a file the
    agent writes can quote a patch, and quoting one patched nothing.

    A text that is not JSON at all is an already-unencoded envelope — the
    shape of older captures — and is returned as it stands.

    Parameters
    ----------
    text : str
        The raw arguments of one ``function_call``.

    Returns
    -------
    List[str]
        The decoded envelope candidates, in the order the call carries them,
        empty only for arguments that carry no text at all.
    """
    bodies: List[str] = []
    for candidate in _argument_texts(text):
        if not candidate.strip():
            continue
        # A patch envelope is never shell: splitting it would read a ``<<``
        # inside the diff as a here-document and eat the rest of the patch.
        if "<<" not in candidate or candidate.lstrip().startswith(_PATCH_BEGIN):
            if candidate not in bodies:
                bodies.append(candidate)
            continue
        shell, documents = _split_here_documents(candidate)
        if shell and shell not in bodies:
            bodies.append(shell)
        for body in documents:
            # A here-document is data for the program it feeds, and most of
            # what the agent pipes is not a patch at all: it counts only when
            # the command applies one, or announces one in its first line.
            if "apply_patch" not in shell and not body.lstrip().startswith(
                _PATCH_BEGIN
            ):
                continue
            if body not in bodies:
                bodies.append(body)
    return bodies


def _argument_texts(text: str) -> List[str]:
    """
    Return the argument values of one call that can carry a patch envelope.

    Parameters
    ----------
    text : str
        The raw arguments of the call, JSON-encoded in real Codex traffic.

    Returns
    -------
    List[str]
        The decoded values, or the raw text when it is not JSON at all — the
        raw envelope then being the only candidate there is.
    """
    try:
        arguments = json.loads(text)
    except (ValueError, TypeError):
        return [text]
    if isinstance(arguments, str):
        return [arguments]
    if not isinstance(arguments, dict):
        return [text]
    texts: List[str] = []
    for key in _PATCH_ARGUMENT_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip() and value not in texts:
            texts.append(value)
    # No patch-carrying key: fall back to the raw text.  A JSON string cannot
    # contain a bare line break, so the line-anchored patch markers cannot
    # match it — but a shape with an unexpected key keeps working as before.
    return texts or [text]


def _split_here_documents(command: str) -> Tuple[str, List[str]]:
    """
    Split *command* into its shell text and the bodies it pipes.

    A here-document is data, not shell syntax, and the split is what makes a
    patch readable: the lines of a body are real lines of the command string,
    so scanning the whole command would read a patch *example* written into a
    file as a patch applied to it.  Each body is handed back separately, and
    the shell text keeps a placeholder where it stood.

    Parameters
    ----------
    command : str
        A shell command, unencoded.

    Returns
    -------
    Tuple[str, List[str]]
        The shell text and the bodies, in the order they run.
    """
    if "<<" not in command:
        return command, []
    shell: List[str] = []
    bodies: List[str] = []
    lines = command.splitlines()
    index = 0
    while index < len(lines):
        match = _HERE_DOCUMENT_RE.search(lines[index])
        if match is None:
            shell.append(lines[index])
            index += 1
            continue
        shell.append(lines[index][: match.start()] + " - ")
        closing = match.group(2)
        body: List[str] = []
        index += 1
        while index < len(lines):
            if lines[index].strip() == closing:
                index += 1
                break
            body.append(lines[index])
            index += 1
        bodies.append("\n".join(body))
    return "\n".join(shell), bodies


def _touched_paths(
    activity: Tuple[CodexActivity, ...], scope: RepositoryScope
) -> Tuple[str, ...]:
    """
    Read the repository-relative paths this turn patched.

    The envelopes come out of :func:`patch_bodies`, so a replayed turn names
    each file once, in the order it was first touched.
    """
    found: List[str] = []
    for item in activity:
        if item.kind != "function_call" or not item.text:
            continue
        for body in patch_bodies(item.text):
            for match in _PATCH_FILE_RE.finditer(body):
                path = scope.relative(match.group(1).strip())
                if path and path not in found:
                    found.append(path)
            for match in _PATCH_MOVE_RE.finditer(body):
                path = scope.relative(match.group(1).strip())
                if path and path not in found:
                    found.append(path)
    return tuple(found[:40])


def _window_text(payload: Dict[str, Any]) -> str:
    """
    Serialize the conversation window of a compaction request.

    ``CodexRequest`` deliberately keeps only what routing needs, and a window
    archive is precisely the text that was left out — so it is read from the
    payload itself, item by item, in the order the CLI sent it.

    Parameters
    ----------
    payload : Dict[str, Any]
        The compaction request.

    Returns
    -------
    str
        Role-tagged window text, capped in length.
    """
    items = payload.get("input")
    if not isinstance(items, list):
        return ""
    parts: List[str] = []
    total = 0
    for item in items:
        for role, text in _item_parts(item):
            chunk = f"[{role}]\n{text}\n"
            parts.append(chunk)
            total += len(chunk)
            if total >= _MAX_WINDOW_CHARS:
                return "".join(parts)[:_MAX_WINDOW_CHARS]
    return "".join(parts)


def _item_parts(item: Any) -> List[Tuple[str, str]]:
    """Return ``(role, text)`` pairs carried by one ``input`` item."""
    if isinstance(item, str):
        return [("input", item)] if item.strip() else []
    if not isinstance(item, dict):
        return []
    role = str(item.get("role") or item.get("type") or "input")
    content = item.get("content")
    texts: List[str] = []
    if isinstance(content, str):
        texts.append(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
    elif isinstance(item.get("output"), str):
        texts.append(item["output"])
    return [(role, text.strip()) for text in texts if text and text.strip()]
