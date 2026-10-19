"""
Claude Code adapter: the basic half of the knowledge layer.

Claude Code speaks the Anthropic Messages shape: a separate ``system`` block
instead of a stable ``instructions`` prefix, and a ``messages`` list instead of
a replayed ``input``.  The basic adapter covers the parts every agent needs —
recognize the request, identify its repository, read the newest user command,
and put knowledge into ``system`` — and deliberately does not cover the parts
that depend on Codex's replay: there is no phase evidence in a Messages body,
so this adapter mines nothing and escalates nothing.  The knowledge base it
reads is the same one Codex writes, which is the point of a shared layer.
"""

import hashlib

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from llm_router_plugins.utils.knowledge.agentic_knowledge.adapters.base import (
    AgentAdapter,
    AgentTurn,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.scope import (
    ScopeResolver,
)

__all__ = ["ClaudeCodeAgentAdapter", "CLAUDE_ROUTING_PLUGIN"]

#: Routing plugin identifier that marks a request as Claude Code's.
CLAUDE_ROUTING_PLUGIN = "agentic_routing_claude_code"

#: Marker wrapped around injected knowledge; also its idempotency detector.
_INJECTION_OPEN = "<agentic_knowledge>"
_INJECTION_CLOSE = "</agentic_knowledge>"

#: Work mode assumed when routing did not name one.
_DEFAULT_MODE = "implement"

#: Ceiling of an identifier read out of the body; it is a cache key, not data.
_MAX_ID_CHARS = 128


@dataclass(frozen=True)
class ClaudeCodeAgentAdapter(AgentAdapter):
    """
    Reads and enriches Claude Code requests.

    Parameters
    ----------
    scope_resolver : ScopeResolver, optional
        Repository scope resolver.
    mode_map : Dict[str, str], optional
        Translation of a routing mode into a work mode this plugin serves.
    """

    scope_resolver: Optional[ScopeResolver] = None
    mode_map: Dict[str, str] = None  # type: ignore[assignment]

    name = "claude_code"

    def __post_init__(self) -> None:
        """Give the optional collaborators their defaults."""
        if self.scope_resolver is None:
            object.__setattr__(self, "scope_resolver", ScopeResolver())
        if self.mode_map is None:
            object.__setattr__(self, "mode_map", {})

    def matches(self, payload: Dict[str, Any]) -> bool:
        """
        Return whether *payload* is a Claude Code request.

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
            and routing.get("plugin") == CLAUDE_ROUTING_PLUGIN
        ):
            return True
        # Anthropic Messages shape: a system block and a messages list.
        return "messages" in payload and ("system" in payload or "model" in payload)

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
            The turn, or ``None`` when the payload is not a request this
            plugin can do anything with.
        """
        if not isinstance(payload, dict) or not self.matches(payload):
            return None
        scope = self.scope_resolver.resolve(payload)
        routing = payload.get("routing")
        routing = routing if isinstance(routing, dict) else {}
        declared = str(routing.get("mode") or "")
        mode = self.mode_map.get(declared, declared) or _DEFAULT_MODE
        # Identity comes from the body, not from ``routing``: the Claude Code
        # routing plugin records how it matched a model and nothing else, so
        # reading session or turn identifiers out of its annotation yields an
        # empty string for every request.
        session_id = _session_id(payload)
        latest_user_text = _latest_user_text(payload)
        return AgentTurn(
            agent=self.name,
            scope=scope.key,
            repo_root=scope.repo_root,
            session_id=session_id,
            thread_id="",
            turn_id="",
            root_turn_id="",
            generation=_generation(session_id, latest_user_text),
            request_class="main",
            mode=mode,
            routing_source=str(routing.get("source") or ""),
            model=str(payload.get("model") or ""),
            latest_user_text=latest_user_text,
            window_id="",
            activity=(),
            evidence=(),
            window_text="",
            touched_paths=(),
        )

    def inject(self, payload: Dict[str, Any], text: str) -> Dict[str, Any]:
        """
        Return *payload* with the knowledge block added to ``system``.

        Parameters
        ----------
        payload : Dict[str, Any]
            The request payload.
        text : str
            The rendered knowledge block.

        Returns
        -------
        Dict[str, Any]
            The payload, enriched.
        """
        if not isinstance(payload, dict) or not text:
            return payload
        block = f"{_INJECTION_OPEN}\n{text}\n{_INJECTION_CLOSE}"
        system = payload.get("system")
        if isinstance(system, str):
            if _INJECTION_OPEN in system:
                return payload
            payload["system"] = f"{system}\n\n{block}" if system.strip() else block
        elif isinstance(system, list):
            if _already_injected(system):
                return payload
            system.append({"type": "text", "text": block})
        else:
            payload["system"] = block
        return payload

    def set_model(self, payload: Dict[str, Any], model_name: str) -> None:
        """
        Point *payload* at *model_name*.

        Parameters
        ----------
        payload : Dict[str, Any]
            The request payload, modified in place.
        model_name : str
            The model to request instead.
        """
        payload["model"] = model_name


def _already_injected(system: List[Any]) -> bool:
    """
    Whether this adapter's block already ends the ``system`` list.

    Only the tail is read: a prompt that merely quotes the markers is not a
    request this plugin has already enriched, and mistaking it for one would
    switch knowledge off without a trace.
    """
    if not system:
        return False
    last = system[-1]
    return (
        isinstance(last, dict)
        and isinstance(last.get("text"), str)
        and _INJECTION_OPEN in last["text"]
    )


def _latest_user_text(payload: Dict[str, Any]) -> str:
    """Return the newest genuine user command of a Messages body."""
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"].strip())
            if parts:
                return "\n".join(parts)
    return ""


def _session_id(payload: Dict[str, Any]) -> str:
    """
    Return the session identity the client declares, or ``""``.

    Claude Code sends ``metadata.user_id``, sometimes as a JSON string and
    sometimes as an object; it is opaque, so it is read for identity only and
    never indexed.
    """
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return ""
    declared = metadata.get("user_id")
    if isinstance(declared, dict):
        return _first_text(declared, ("session_id", "thread_id", "id"))
    if not isinstance(declared, str):
        return ""
    text = declared.strip()
    if text.startswith("{"):
        import json

        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            return text[:_MAX_ID_CHARS]
        if isinstance(parsed, dict):
            return _first_text(parsed, ("session_id", "thread_id", "id"))
    return text[:_MAX_ID_CHARS]


def _first_text(values: Dict[str, Any], keys: Tuple[str, ...]) -> str:
    """Return the first non-empty string among *keys*."""
    for key in keys:
        value = values.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:_MAX_ID_CHARS]
    return ""


def _generation(session_id: str, latest_user_text: str) -> str:
    """
    Identity of one command generation of a Claude Code conversation.

    Anthropic Messages carries no turn identifier, so the stable handle is the
    newest user message: it repeats unchanged across the requests an agent makes
    while working on it and changes when the user speaks again.  Hashing the
    whole history instead gives every request a fresh identity, which makes the
    generation cache miss every time and its lookup ceiling never bind.
    """
    if not latest_user_text:
        return ""
    seed = f"{session_id}\x00{latest_user_text}"
    return "c" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
