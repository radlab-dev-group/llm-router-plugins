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
from llm_router_plugins.utils.knowledge.agentic_knowledge.documents import (
    HEADER_TAG,
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

#: Ceiling of the serialized window of a compaction request.
_MAX_WINDOW_CHARS = 400_000


def load_phase_rules(path: Optional[Path] = None) -> Optional[CodexPhaseConfig]:
    """
    Return the phase rules the Codex router uses, or ``None``.

    The rules are read from the routing plugin's own configuration so that one
    definition of "this call is a patch" serves both plugins.  Nothing here is
    fatal: without rules the knowledge layer still injects, it simply records
    no episodes.

    Parameters
    ----------
    path : pathlib.Path, optional
        Alternative configuration file; defaults to
        :data:`PHASE_CONFIG_ENV`, then to the bundled Codex routing config.

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
    except Exception:  # a foreign config must not break this plugin
        return None
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
        else:
            payload["input"] = [_developer_item(block)]
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
    """Whether one of the ``input`` items is this plugin's own block."""
    for item in items:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if (
                isinstance(part, dict)
                and isinstance(part.get("text"), str)
                and (
                    _INJECTION_OPEN in part["text"]
                    or f"<{HEADER_TAG} " in part["text"]
                )
            ):
                return True
    return False


def _identifier(request: CodexRequest) -> str:
    """Identity of the command generation, ``turn_id`` where the CLI gives one."""
    if request.turn_id:
        return request.turn_id
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


def _touched_paths(
    activity: Tuple[CodexActivity, ...], scope: RepositoryScope
) -> Tuple[str, ...]:
    """Read the repository-relative paths this turn patched."""
    found: List[str] = []
    for item in activity:
        if item.kind != "function_call" or not item.text:
            continue
        for match in _PATCH_FILE_RE.finditer(item.text):
            path = scope.relative(match.group(1).strip())
            if path and path not in found:
                found.append(path)
        for match in _PATCH_MOVE_RE.finditer(item.text):
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
