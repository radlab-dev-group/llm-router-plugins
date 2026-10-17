"""
Mining knowledge out of a turn the agent has already performed.

The agent will not say what it learned — it is a CLI, not a knowledge base, and
its next session starts empty.  But it leaves the evidence behind: the command
that failed, the patch it wrote in response, the same command green.  Read
together, those three are a verified fix, and a verified fix is worth more than
anything an offline extractor would dream up, because the agent's own test run
proved it.

Three episodes are recognizable from the phase evidence of one turn:

* **recipe** — a failure, a change, and a test run that passed; recorded once,
  with the failure it fixed, the files it touched and the command that proved
  it;
* **dead end** — the same failure again after a change; recorded so the next
  attempt in this repository does not walk into the same wall;
* **window archive** — for a compaction request, the conversation that is about
  to be summarized, chunked along message boundaries so an exact error message
  survives the summarization.

A fourth signal is not a document at all: **the loop** — the same failure N
times in one generation.  It is a reason to escalate, not a fact to store, so
it is returned to the caller and never published.

The miner is stateless.  Codex replays the whole turn on every request, so
episodes are recomputed each time and deduplicated downstream, by fingerprint,
which is what makes a retried request harmless.
"""

import datetime
import re

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from llm_router_plugins.utils.knowledge.agentic_knowledge.adapters.base import (
    AgentTurn,
    KnowledgeEvent,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.documents import (
    KIND_DEADEND,
    KIND_RECIPE,
    KIND_SESSION_ARCHIVE,
    build_document_text,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.signature import (
    ErrorSignature,
    signature_of,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.payload import (
    REQUEST_CLASS_COMPACTION,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.phase import (
    EVIDENCE_COMMAND,
    EVIDENCE_PATCH,
    EVIDENCE_TEST_FAILURE,
    PhaseEvidence,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.state import fingerprint

__all__ = ["EpisodeMiner", "MiningResult"]

#: Test command of a recipe has to have settled successfully.
_RECIPE_MIN_PATCHES = 1

_WINDOW_CHUNK_RE = re.compile(r"\n(?=\[[a-z_]+\]\n)")


@dataclass(frozen=True)
class MiningResult:
    """
    What one turn yielded.

    Parameters
    ----------
    events : Tuple[KnowledgeEvent, ...]
        Documents to publish; empty for a turn that proved nothing.
    loop_count : int
        Times the same failure was seen in this generation.
    signature_hash : str
        Identity of that failure; ``""`` when there was none.
    tried : Tuple[str, ...]
        One-line summaries of what this turn already attempted, for the
        "do not repeat this" section of the injection.
    """

    events: Tuple[KnowledgeEvent, ...] = ()
    loop_count: int = 0
    signature_hash: str = ""
    tried: Tuple[str, ...] = ()


@dataclass
class _Failure:
    """One recognized test failure, with what happened after it."""

    evidence: PhaseEvidence
    signature: Optional[ErrorSignature]
    patches: List[PhaseEvidence] = field(default_factory=list)
    later_failures: List[PhaseEvidence] = field(default_factory=list)
    resolved_by: Optional[PhaseEvidence] = None


class EpisodeMiner:
    """
    Turn phase evidence into knowledge events.

    Parameters
    ----------
    max_patch_chars : int
        Ceiling of the diff quoted into a recipe or a dead end.
    archive_enabled : bool
        Whether compaction windows are archived at all.
    archive_max_chunk_chars : int
        Ceiling of one archived fragment.
    archive_ttl_seconds : int
        Retention of the archive, recorded for the worker.
    test_mode : str
        Name of the work mode the phase rules call a test run.
    """

    def __init__(
        self,
        max_patch_chars: int = 4000,
        archive_enabled: bool = True,
        archive_max_chunk_chars: int = 2000,
        archive_ttl_seconds: int = 1209600,
        test_mode: str = "test",
    ) -> None:
        """Bind the budgets the miner writes into its documents."""
        self._max_patch_chars = max_patch_chars
        self._archive_enabled = archive_enabled
        self._archive_max_chunk_chars = archive_max_chunk_chars
        self._archive_ttl_seconds = archive_ttl_seconds
        self._test_mode = test_mode

    def mine(self, turn: AgentTurn) -> MiningResult:
        """
        Return everything worth keeping from *turn*.

        Parameters
        ----------
        turn : AgentTurn
            The request's turn, with its evidence.

        Returns
        -------
        MiningResult
            Events to publish and the loop signal.
        """
        if turn.request_class == REQUEST_CLASS_COMPACTION:
            return MiningResult(events=self._archive(turn))
        if not turn.evidence:
            return MiningResult()
        failures = self._episodes(turn)
        events = [
            event
            for event in (
                [
                    *(
                        self._recipe(turn, item)
                        for item in failures
                        if item.resolved_by
                    ),
                    *(
                        self._deadend(turn, item)
                        for item in failures
                        if item.later_failures
                    ),
                ]
            )
            if event is not None
        ]
        loop = self._loop(turn, failures)
        tried = tuple(
            tried
            for tried in (self._summary(turn, item) for item in failures)
            if tried
        )
        return MiningResult(
            events=tuple(events),
            loop_count=loop.loop_count,
            signature_hash=loop.signature_hash,
            tried=tried,
        )

    # ------------------------------------------------------------- episodes -
    def _episodes(self, turn: AgentTurn) -> List[_Failure]:
        """Group the turn's evidence into failures and what followed them."""
        texts = _output_texts(turn)
        failures: List[_Failure] = []
        for evidence in turn.evidence:
            if evidence.kind == EVIDENCE_TEST_FAILURE:
                signature = signature_of(texts.get(evidence.call_id, ""))
                previous = failures[-1] if failures else None
                if previous is not None and _same_failure(previous, signature):
                    # The same wall again: this is what makes it a dead end,
                    # and it does not start a new episode.
                    previous.later_failures.append(evidence)
                    continue
                failures.append(_Failure(evidence=evidence, signature=signature))
                continue
            if not failures:
                continue
            latest = failures[-1]
            if evidence.kind == EVIDENCE_PATCH:
                latest.patches.append(evidence)
            elif self._is_green_test(evidence):
                latest.resolved_by = evidence
        return failures

    def _failure_evidence(self, turn: AgentTurn) -> List[PhaseEvidence]:
        """Every test failure of this turn, including repeats."""
        return [item for item in turn.evidence if item.kind == EVIDENCE_TEST_FAILURE]

    def _is_green_test(self, evidence: PhaseEvidence) -> bool:
        """Whether one evidence item is a test command that passed."""
        return bool(
            evidence.kind == EVIDENCE_COMMAND
            and evidence.succeeded is True
            and (not evidence.mode or evidence.mode == self._test_mode)
        )

    def _loop(self, turn: AgentTurn, failures: List[_Failure]) -> MiningResult:
        """Count the repetitions of the most frequent failure of this turn."""
        texts = _output_texts(turn)
        counts: Dict[str, int] = {}
        for evidence in self._failure_evidence(turn):
            signature = signature_of(texts.get(evidence.call_id, ""))
            key = signature.signature_hash if signature else ""
            if not key:
                key = evidence.call_id or evidence.event_id or "?"
            counts[key] = counts.get(key, 0) + 1
        if not counts:
            return MiningResult()
        key = max(counts, key=lambda name: (counts[name], name))
        return MiningResult(loop_count=counts[key], signature_hash=key)

    def _summary(self, turn: AgentTurn, failure: _Failure) -> str:
        """One line saying what this turn already tried for this failure."""
        if not failure.patches:
            return ""
        paths = ", ".join(turn.touched_paths[:6]) or "unknown files"
        return f"patched {paths}"

    # -------------------------------------------------------------- recipe --
    def _recipe(
        self, turn: AgentTurn, failure: _Failure
    ) -> Optional[KnowledgeEvent]:
        """Build the document of a failure this turn fixed."""
        if len(failure.patches) < _RECIPE_MIN_PATCHES or failure.signature is None:
            return None
        patch_text = self._patch_text(turn, failure.patches)
        command = self._command_text(turn, failure.resolved_by)
        body = "\n".join(
            part
            for part in (
                f"Failure: {failure.signature.signature}",
                f"Fixed by: {', '.join(turn.touched_paths[:12]) or 'unknown files'}",
                f"Verified by: {command or 'the test command of the turn'}",
                "",
                "Diff that made it pass:",
                patch_text,
            )
            if part is not None
        )
        identities = tuple(
            sorted(
                {
                    item.call_id or item.event_id
                    for item in (*failure.patches, failure.resolved_by)
                    if item is not None and (item.call_id or item.event_id)
                }
            )
        )
        return self._event(
            turn,
            KIND_RECIPE,
            body,
            signature=failure.signature,
            parts=("recipe", failure.signature.signature_hash, *identities),
            test_command=command,
            paths=_merge_paths(turn.touched_paths, failure.signature.paths),
        )

    # ------------------------------------------------------------- deadend --
    def _deadend(
        self, turn: AgentTurn, failure: _Failure
    ) -> Optional[KnowledgeEvent]:
        """Build the document of a change that did not fix the failure."""
        if not failure.patches or failure.signature is None:
            return None
        patch_text = self._patch_text(turn, failure.patches)
        body = "\n".join(
            (
                f"Failure: {failure.signature.signature}",
                f"Attempted: {', '.join(turn.touched_paths[:12]) or 'unknown files'}",
                "Outcome: the same failure was still there afterwards.",
                "",
                "Change that did not fix it:",
                patch_text,
            )
        )
        identities = tuple(
            sorted(
                {
                    item.call_id or item.event_id
                    for item in (*failure.patches, *failure.later_failures)
                    if item.call_id or item.event_id
                }
            )
        )
        return self._event(
            turn,
            KIND_DEADEND,
            body,
            signature=failure.signature,
            parts=("deadend", failure.signature.signature_hash, *identities),
            test_command=self._command_text(turn, failure.evidence),
            paths=_merge_paths(turn.touched_paths, failure.signature.paths),
        )

    # ------------------------------------------------------------ archive ---
    def _archive(self, turn: AgentTurn) -> Tuple[KnowledgeEvent, ...]:
        """Split a compaction window into searchable fragments."""
        if not self._archive_enabled or not turn.window_text.strip():
            return ()
        chunks = _split_window(turn.window_text, self._archive_max_chunk_chars)
        events = []
        for position, (role, text) in enumerate(chunks):
            metadata = {
                "thread_id": turn.thread_id,
                "window_id": turn.window_id,
                "position": position,
                "fragment_kind": role,
                "ttl_seconds": self._archive_ttl_seconds,
            }
            body = build_document_text(
                KIND_SESSION_ARCHIVE,
                text,
                date=_today(),
                thread_id=turn.thread_id,
                window_id=turn.window_id,
                position=position,
                agent=turn.agent,
                mode=turn.mode,
            )
            events.append(
                KnowledgeEvent(
                    kind=KIND_SESSION_ARCHIVE,
                    fingerprint=fingerprint(
                        "session_archive",
                        turn.thread_id,
                        turn.window_id,
                        str(position),
                        text[:200],
                    ),
                    scope=turn.scope,
                    agent=turn.agent,
                    mode=turn.mode,
                    model=turn.model,
                    thread_id=turn.thread_id,
                    generation=turn.cache_generation,
                    created_at=_now(),
                    text=body,
                    metadata=metadata,
                )
            )
        return tuple(events)

    # ------------------------------------------------------------- helpers --
    def _event(
        self,
        turn: AgentTurn,
        kind: str,
        body: str,
        *,
        signature: Optional[ErrorSignature],
        parts: Sequence[str],
        test_command: str = "",
        paths: Tuple[str, ...] = (),
    ) -> KnowledgeEvent:
        """Assemble one publishable event out of the parts."""
        digest = fingerprint(*parts)
        text = build_document_text(
            kind,
            body,
            date=_today(),
            signature=signature.signature if signature else "",
            paths=paths,
            thread_id=turn.thread_id,
            agent=turn.agent,
            mode=turn.mode,
            test_command=test_command,
        )
        metadata: Dict[str, Any] = {
            "paths": list(paths),
            "thread_id": turn.thread_id,
            "mode": turn.mode,
            "agent": turn.agent,
        }
        if test_command:
            metadata["test_command"] = test_command
        return KnowledgeEvent(
            kind=kind,
            fingerprint=digest,
            scope=turn.scope,
            agent=turn.agent,
            mode=turn.mode,
            model=turn.model,
            thread_id=turn.thread_id,
            generation=turn.cache_generation,
            created_at=_now(),
            signature=signature.signature if signature else "",
            signature_hash=signature.signature_hash if signature else "",
            text=text,
            metadata=metadata,
        )

    def _patch_text(self, turn: AgentTurn, patches: Sequence[PhaseEvidence]) -> str:
        """Quote the patches of an episode, capped."""
        arguments = _call_arguments(turn)
        chunks = []
        total = 0
        for item in patches:
            text = arguments.get(item.call_id) or arguments.get(item.event_id) or ""
            if not text.strip():
                continue
            chunks.append(text.strip())
            total += len(text)
            if total >= self._max_patch_chars:
                break
        joined = "\n".join(chunks)
        return joined[: self._max_patch_chars]

    def _command_text(
        self, turn: AgentTurn, evidence: Optional[PhaseEvidence]
    ) -> str:
        """Return the command line an evidence item stands for."""
        if evidence is None:
            return ""
        arguments = _call_arguments(turn)
        raw = (
            arguments.get(evidence.call_id) or arguments.get(evidence.event_id) or ""
        )
        return _command_line(raw)


def _output_texts(turn: AgentTurn) -> Dict[str, str]:
    """Map ``call_id`` to the tool output of that call."""
    texts: Dict[str, str] = {}
    for item in turn.activity:
        if item.kind == "function_call_output" and item.call_id:
            texts[item.call_id] = item.text
    return texts


def _call_arguments(turn: AgentTurn) -> Dict[str, str]:
    """Map ``call_id`` and ``event_id`` to the arguments of that call."""
    arguments: Dict[str, str] = {}
    for item in turn.activity:
        if item.kind != "function_call":
            continue
        if item.call_id:
            arguments[item.call_id] = item.text
        if item.event_id:
            arguments.setdefault(item.event_id, item.text)
    return arguments


def _command_line(raw: str) -> str:
    """Read the command out of a tool call's JSON arguments."""
    if not raw:
        return ""
    text = raw.strip()
    if not text.startswith("{"):
        return text[:200]
    import json

    try:
        decoded = json.loads(text)
    except (TypeError, ValueError):
        return text[:200]
    if isinstance(decoded, dict):
        for key in ("cmd", "command", "input"):
            value = decoded.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:200]
    return ""


def _same_failure(previous: _Failure, signature: Optional[ErrorSignature]) -> bool:
    """Whether a new failure is the one :attr:`_Failure` already describes."""
    if signature is None or previous.signature is None:
        return False
    return signature.signature_hash == previous.signature.signature_hash


def _merge_paths(
    touched: Tuple[str, ...], mentioned: Tuple[str, ...], limit: int = 12
) -> Tuple[str, ...]:
    """
    Join patched paths with the files a failure named, without duplicates.

    A traceback names ``test_a.py`` where a patch names ``tests/test_a.py``;
    the longer form says more, so a bare name already covered by a patched
    path is dropped rather than listed twice.

    Parameters
    ----------
    touched : Tuple[str, ...]
        Paths the turn patched.
    mentioned : Tuple[str, ...]
        Paths the failure report named.
    limit : int
        Most paths to keep.

    Returns
    -------
    Tuple[str, ...]
        The merged paths.
    """
    merged: List[str] = []
    for path in touched:
        if path not in merged:
            merged.append(path)
    for path in mentioned:
        if not path or path in merged:
            continue
        if any(known.endswith(f"/{path}") for known in merged):
            continue
        merged.append(path)
    return tuple(merged[:limit])


def _split_window(text: str, max_chars: int) -> Tuple[Tuple[str, str], ...]:
    """Cut a serialized window into fragments along message boundaries."""
    blocks = [block for block in _WINDOW_CHUNK_RE.split(text) if block.strip()]
    fragments: List[Tuple[str, str]] = []
    for block in blocks:
        role = _role_of(block)
        for piece in _fit(block, max_chars):
            fragments.append((role, piece))
    return tuple(fragments)


def _role_of(block: str) -> str:
    """Read the ``[role]`` tag a serialized window block opens with."""
    first = block.strip().splitlines()[0] if block.strip() else ""
    match = re.match(r"^\[([a-z_]+)\]$", first.strip())
    return match.group(1) if match else "message"


def _fit(block: str, max_chars: int) -> List[str]:
    """Split a block that is still longer than the chunk budget."""
    if max_chars <= 0 or len(block) <= max_chars:
        return [block.strip()]
    return [
        block[start : start + max_chars].strip()
        for start in range(0, len(block), max_chars)
    ]


def _today() -> datetime.date:
    """Return today, isolated for the tests."""
    return datetime.date.today()


def _now() -> float:
    """Return the current second, isolated for the tests."""
    return datetime.datetime.now(tz=datetime.timezone.utc).timestamp()
