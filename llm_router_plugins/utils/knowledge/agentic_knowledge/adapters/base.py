"""
What the knowledge layer expects from an agent, and what it gives back.

The knowledge itself does not care which CLI produced a request; the payload
shape does, and that is the whole reason an adapter exists.  Codex speaks the
OpenAI Responses wire format and replays the whole turn in every request,
Claude Code speaks Anthropic Messages with a separate ``system`` block, and a
third agent will speak something else again.

An adapter answers four questions about a payload: is this yours
(:meth:`AgentAdapter.matches`), what did the agent do
(:meth:`AgentAdapter.parse`), where does knowledge go
(:meth:`AgentAdapter.inject`), and how is the model changed
(:meth:`AgentAdapter.set_model`).  Everything else — what to search, how to
rank, when to escalate — stays in the plugin.

:meth:`AgentAdapter.parse` returns an :class:`AgentTurn`, a snapshot of one
command generation of one agent: who, where, which work mode, what the user
asked, and what the agent has done since.  It is the only structure the rest
of the package reads, which is what lets the Codex miner and a hypothetical
future one share a knowledge base.
"""

import abc

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional, Tuple

from llm_router_plugins.utils.routing.agentic_routing.codex.payload import (
    CodexActivity,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.phase import (
    PhaseEvidence,
)

__all__ = ["AgentAdapter", "AgentTurn", "KnowledgeEvent"]


@dataclass(frozen=True)
class AgentTurn:
    """
    One command generation of one agent, ready to be enriched.

    Parameters
    ----------
    agent : str
        Which agent produced the request; the adapter's ``name``.
    scope : str
        Repository the work happens in, as a path-free key; ``""`` when the
        request declared none.
    repo_root : str
        The declared working directory.  Used to relativize paths, never sent
        anywhere.
    session_id : str
        CLI session identifier, when the agent declares one.
    thread_id : str
        Conversation thread; the unit a session archive belongs to.
    turn_id : str
        Identifier of this command generation.
    root_turn_id : str
        Identifier of the turn this one branches from, when the agent says.
    generation : str
        Identity of the generation the cache keys on; ``turn_id`` where the
        agent provides one, a digest of the request otherwise.
    request_class : str
        ``main``, ``aux_title`` or ``compaction``.
    mode : str
        Work mode resolved by routing; ``""`` when routing did not run.
    routing_source : str
        How routing decided, e.g. ``explicit``; ``""`` when it did not decide.
    model : str
        Model the request will go to, after routing.
    latest_user_text : str
        The newest genuine user command.
    window_id : str
        Context window the request was made from; the unit an archived window
        belongs to, and what a later request filters the archive by together
        with :attr:`thread_id`.
    activity : Tuple[CodexActivity, ...]
        Tool and assistant events of this turn, oldest first.  Empty for an
        agent whose events the plugin cannot interpret.
    evidence : Tuple[PhaseEvidence, ...]
        Recognized phase evidence over :attr:`activity`, oldest first.
    window_text : str
        Serialized conversation window; filled for ``compaction`` only.
    touched_paths : Tuple[str, ...]
        Repository-relative paths of the patches this turn changed.
    """

    agent: str = ""
    scope: str = ""
    repo_root: str = ""
    session_id: str = ""
    thread_id: str = ""
    turn_id: str = ""
    root_turn_id: str = ""
    generation: str = ""
    request_class: str = "main"
    mode: str = ""
    routing_source: str = ""
    model: str = ""
    latest_user_text: str = ""
    window_id: str = ""
    activity: Tuple[CodexActivity, ...] = ()
    evidence: Tuple[PhaseEvidence, ...] = ()
    window_text: str = ""
    touched_paths: Tuple[str, ...] = ()

    @property
    def cache_generation(self) -> str:
        """Identity of this generation for caching and loop accounting."""
        for candidate in (self.generation, self.turn_id, self.root_turn_id):
            if candidate:
                return candidate
        return ""

    def latest_failure(self) -> Optional[PhaseEvidence]:
        """
        Return the most recent test failure of this turn, if any.

        Returns
        -------
        PhaseEvidence or None
            The last ``test_failure`` evidence, or ``None`` when the turn has
            not failed a test.
        """
        for item in reversed(self.evidence):
            if item.kind == "test_failure":
                return item
        return None


@dataclass(frozen=True)
class KnowledgeEvent:
    """
    One episode worth remembering, on its way to the indexing worker.

    The event is the wire format between the request path and the worker: it
    is JSON, versioned, and carries everything the worker needs so that no
    request state has to outlive the request.  ``text`` is pre-masking —
    masking is the worker's job, so that a change in masking rules can be
    applied to a backlog.

    Parameters
    ----------
    version : int
        Schema version of the event.
    kind : str
        One of the miner's kinds: ``recipe``, ``deadend``, ``session_archive``.
    fingerprint : str
        Stable digest of the episode; the deduplication key.
    scope : str
        Repository the episode belongs to.
    agent : str
        Agent that produced it.
    mode : str
        Work mode the episode happened in.
    model : str
        Model that did the work.
    thread_id : str
        Conversation thread, for the session archive.
    generation : str
        Command generation the episode came from.
    created_at : float
        Wall-clock second the event was published.
    signature : str
        Normalized failure the episode is about; ``""`` for an archive.
    signature_hash : str
        Digest of :attr:`signature`; empty when there is none.
    text : str
        The document to index, header included.
    metadata : Dict[str, Any]
        Structured fields for filtering: paths, window, position, attempts.
    """

    version: int = 1
    kind: str = ""
    fingerprint: str = ""
    scope: str = ""
    agent: str = ""
    mode: str = ""
    model: str = ""
    thread_id: str = ""
    generation: str = ""
    created_at: float = 0.0
    signature: str = ""
    signature_hash: str = ""
    text: str = ""
    metadata: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Give a missing metadata mapping a real, empty dict."""
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})

    def to_dict(self) -> Dict[str, Any]:
        """
        Return the JSON-serializable form of this event.

        Returns
        -------
        Dict[str, Any]
            Every field, under its own name.
        """
        return {
            "version": self.version,
            "kind": self.kind,
            "fingerprint": self.fingerprint,
            "scope": self.scope,
            "agent": self.agent,
            "mode": self.mode,
            "model": self.model,
            "thread_id": self.thread_id,
            "generation": self.generation,
            "created_at": self.created_at,
            "signature": self.signature,
            "signature_hash": self.signature_hash,
            "text": self.text,
            "metadata": dict(self.metadata or {}),
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "KnowledgeEvent":
        """
        Rebuild an event from :meth:`to_dict` output.

        Unknown keys are ignored, so a worker can be older than the proxy that
        produced the event; a missing key falls back to its default.

        Parameters
        ----------
        raw : Dict[str, Any]
            One decoded event from the stream.

        Returns
        -------
        KnowledgeEvent
            The decoded event.
        """
        fields = {field.name for field in cls.__dataclass_fields__.values()}
        known = {key: value for key, value in (raw or {}).items() if key in fields}
        metadata = known.get("metadata")
        if not isinstance(metadata, dict):
            known["metadata"] = {}
        return cls(**known)


class AgentAdapter(abc.ABC):
    """
    The payload-specific half of the knowledge layer.

    Attributes
    ----------
    name : ClassVar[str]
        Identifier recorded in the request annotation.
    """

    name: ClassVar[str] = ""

    @abc.abstractmethod
    def matches(self, payload: Dict[str, Any]) -> bool:
        """
        Return whether *payload* is a request of this adapter's agent.

        Parameters
        ----------
        payload : Dict[str, Any]
            The request payload.

        Returns
        -------
        bool
            ``True`` when this adapter claims the payload.
        """

    @abc.abstractmethod
    def parse(self, payload: Dict[str, Any]) -> Optional[AgentTurn]:
        """
        Read the turn out of *payload*, or ``None`` when unreadable.

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

    @abc.abstractmethod
    def inject(self, payload: Dict[str, Any], text: str) -> Dict[str, Any]:
        """
        Return *payload* with the knowledge block added.

        Injection must be idempotent: passing a payload this adapter already
        enriched returns it unchanged, so a retry does not stack contexts.

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

    @abc.abstractmethod
    def set_model(self, payload: Dict[str, Any], model_name: str) -> None:
        """
        Point *payload* at *model_name*, in the field this agent reads.

        Parameters
        ----------
        payload : Dict[str, Any]
            The request payload, modified in place.
        model_name : str
            The model to request instead.
        """
