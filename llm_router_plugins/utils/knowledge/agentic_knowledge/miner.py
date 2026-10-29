"""
Mining knowledge out of a turn the agent has already performed.

The agent will not say what it learned — it is a CLI, not a knowledge base, and
its next session starts empty.  But it leaves the evidence behind: the command
that failed, the patch it wrote in response, the same command green.  Read
together, those three are a verified fix, and a verified fix is worth more than
anything an offline extractor would dream up, because the agent's own test run
proved it.

Four episodes are recognizable from the phase evidence of one turn:

* **recipe** — a failure, a change, and a test run that passed; recorded once,
  with the failure it fixed, the files it touched and the command that proved
  it;
* **dead end** — the same failure again after a change; recorded so the next
  attempt in this repository does not walk into the same wall;
* **playbook** — a change whose tests passed with no failure in between; the
  correct steps of a task are knowledge too, and a turn that went right from
  the first try would otherwise teach this repository nothing;
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
import shlex

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

from llm_router_plugins.utils.knowledge.agentic_knowledge.adapters.base import (
    AgentTurn,
    KnowledgeEvent,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.documents import (
    KIND_DEADEND,
    KIND_PLAYBOOK,
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
    STRENGTH_WEAK,
    PhaseEvidence,
    patch_envelopes,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.state import fingerprint

__all__ = ["EpisodeMiner", "MiningResult"]

#: Test command of a recipe has to have settled successfully.
_RECIPE_MIN_PATCHES = 1

_WINDOW_CHUNK_RE = re.compile(r"\n(?=\[[a-z_]+\]\n)")

#: How pytest is launched as a module: the ``-m`` of an invocation does
#: not narrow anything, and must not be read as the ``-m`` that selects markers.
_MODULE_LAUNCHER_RE = re.compile(
    r"^\s*\S*python\d*(?:\.\d*)?\s+-m\s+(?=pytest\b|unittest\b)"
)


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
    signature : str
        Human-readable text of the failure :attr:`signature_hash` digests,
        ``""`` when the turn has none.  The reader asks the knowledge base
        with the failure it can show, not with the digest only the writer
        can compare.
    tried : Tuple[str, ...]
        One-line summaries of what this turn already attempted, for the
        "do not repeat this" section of the injection.
    """

    events: Tuple[KnowledgeEvent, ...] = ()
    loop_count: int = 0
    signature_hash: str = ""
    signature: str = ""
    tried: Tuple[str, ...] = ()
    #: Why nothing (or less than expected) was written, counted per turn.
    #: ``no_failure`` — a turn with commands but no test failure to key an
    #: episode on; ``no_patch`` — a failure the turn never changed anything for;
    #: ``not_verified`` — a change with no green re-run of the same failure;
    #: ``covered_by_recipe`` — a procedure a recipe already describes;
    #: ``published`` — documents this turn actually emitted.
    diagnostics: Tuple[Tuple[str, int], ...] = ()


@dataclass
class _Failure:
    """One recognized test failure, with what happened after it."""

    evidence: PhaseEvidence
    signature: Optional[ErrorSignature]
    #: The command whose output failed, read back out of the activity; what a
    #: green run has to be compared against to count as a verification.
    command: str = ""
    patches: List[PhaseEvidence] = field(default_factory=list)
    #: Position of each patch in the evidence sequence; a test run verifies a
    #: fix only if it came after the last of them.
    patch_positions: List[int] = field(default_factory=list)
    later_failures: List[PhaseEvidence] = field(default_factory=list)
    resolved_by: Optional[PhaseEvidence] = None
    #: Command line of the green run; "" when there was none.
    green_command: str = ""
    #: Position of the green run, ``-1`` when there was none.
    resolved_at: int = -1
    #: Closed attempts retain only patches preceding their repeated failure.
    failed_attempts: List["_Failure"] = field(default_factory=list)
    attempt_start: int = 0

    @property
    def first_identity(self) -> str:
        """Identity of the first attempted patch; the episode's own fingerprint.

        The *first* patch, not the set of them: Codex replays the whole turn on
        every request, so a set grows as the turn goes on and a fingerprint built
        from it changes mid-episode — which publishes the same episode once per
        attempt and leaves the superseded documents indexed.
        """
        if not self.patches:
            return ""
        first = self.patches[0]
        return first.call_id or first.event_id or ""

    def verified(self) -> bool:
        """Whether a green run after the last patch actually re-ran this failure."""
        if self.resolved_by is None or not self.patch_positions:
            return False
        if self.resolved_at < max(self.patch_positions):
            return False
        return _verifies(self.command, self.green_command)


@dataclass
class _Procedure:
    """One stretch of work between two green runs, with what it contains."""

    patches: List[PhaseEvidence] = field(default_factory=list)
    items: List[PhaseEvidence] = field(default_factory=list)
    failures: int = 0
    verified_by: Optional[PhaseEvidence] = None
    verified_at: int = -1


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
        procedures, covered_by_recipe = self._procedures(
            turn, _verified_patches(failures)
        )
        events = [
            event
            for event in (
                *(self._recipe(turn, item) for item in failures if item.verified()),
                *(
                    self._deadend(turn, attempt)
                    for item in failures
                    for attempt in item.failed_attempts
                ),
                *(self._playbook(turn, window) for window in procedures),
            )
            if event is not None
        ]
        loop = self._loop(turn, failures)
        tried = _deduplicated(
            tried
            for tried in (self._summary(turn, item) for item in failures)
            if tried
        )
        return MiningResult(
            events=tuple(events),
            loop_count=loop.loop_count,
            signature_hash=loop.signature_hash,
            signature=loop.signature,
            tried=tried,
            diagnostics=self._diagnostics(turn, failures, events, covered_by_recipe),
        )

    def _diagnostics(
        self,
        turn: AgentTurn,
        failures: List["_Failure"],
        events: List[KnowledgeEvent],
        covered_by_recipe: int = 0,
    ) -> Tuple[Tuple[str, int], ...]:
        """
        Count why a turn wrote what it wrote, so an empty knowledge base can
        be read: a turn with commands but no failure is not the same as a
        failure the agent never changed, and neither is the same as a change
        nobody re-ran.  The counts are per turn and never carry content.
        """
        kinds = [item.kind for item in turn.evidence]
        has_failure = EVIDENCE_TEST_FAILURE in kinds
        has_patch = EVIDENCE_PATCH in kinds
        has_command = EVIDENCE_COMMAND in kinds
        no_failure = int(has_command and not has_failure)
        no_patch = int(has_failure and not has_patch)
        not_verified = sum(
            1
            for item in failures
            if item.patches and not item.verified() and not item.failed_attempts
        )
        published = len(events)
        return (
            ("no_failure", no_failure),
            ("no_patch", no_patch),
            ("not_verified", not_verified),
            ("covered_by_recipe", covered_by_recipe),
            ("published", published),
        )

    # ------------------------------------------------------------- episodes -
    def _episodes(self, turn: AgentTurn) -> List[_Failure]:
        """
        Group the turn's evidence into failures and what followed them.

        Episodes are keyed by failure *signature*, not by position: an agent that
        ping-pongs between two errors (`A, patch, B, patch, A`) has two episodes,
        each with its own dead end, and comparing only with the newest one would
        report neither.

        A repeated signature joins the episode it belongs to and takes back any
        green run in between — a test that passed once and then failed again on
        the same signature did not verify anything, and publishing it as a recipe
        would both write a fix that does not work and hand the escalation policy
        a "strong recipe" that keeps the looping model in place.

        Each repeat closes the pending attempt.  Its dead end is a snapshot,
        while the episode keeps the cumulative diff for a later verified recipe.
        """
        texts = _output_texts(turn)
        commands = _call_arguments(turn)
        episodes: List[_Failure] = []
        by_signature: Dict[str, _Failure] = {}
        latest: Optional[_Failure] = None
        for position, evidence in enumerate(turn.evidence):
            if evidence.kind == EVIDENCE_TEST_FAILURE:
                signature = signature_of(texts.get(evidence.call_id, ""))
                known = (
                    by_signature.get(signature.signature_hash)
                    if signature is not None
                    else None
                )
                if known is not None:
                    if len(known.patches) > known.attempt_start:
                        previous = (
                            known.later_failures[-1]
                            if known.later_failures
                            else known.evidence
                        )
                        known.failed_attempts.append(
                            _Failure(
                                evidence=previous,
                                signature=known.signature,
                                command=_command_line(
                                    commands.get(previous.call_id) or ""
                                ),
                                patches=known.patches[known.attempt_start :],
                                patch_positions=known.patch_positions[
                                    known.attempt_start :
                                ],
                                later_failures=[evidence],
                            )
                        )
                    known.attempt_start = len(known.patches)
                    known.later_failures.append(evidence)
                    known.resolved_by = None
                    known.resolved_at = -1
                    known.green_command = ""
                    latest = known
                    continue
                latest = _Failure(
                    evidence=evidence,
                    signature=signature,
                    command=_command_line(commands.get(evidence.call_id) or ""),
                )
                episodes.append(latest)
                if signature is not None:
                    by_signature[signature.signature_hash] = latest
                continue
            if latest is None:
                continue
            if evidence.kind == EVIDENCE_PATCH:
                if evidence.strength == STRENGTH_WEAK:
                    # The episode is closed by envelopes only: a shell write
                    # is a weak signal, and a "verified fix" built on one is a
                    # guess the escalation policy would trust as proof.  The
                    # write still counts for a procedure, which says less.
                    continue
                latest.patches.append(evidence)
                latest.patch_positions.append(position)
            elif self._is_green_test(evidence):
                latest.resolved_by = evidence
                latest.resolved_at = position
                latest.green_command = _command_line(
                    commands.get(evidence.call_id) or ""
                )
        return episodes

    def _procedures(
        self, turn: AgentTurn, covered: Optional[FrozenSet[str]] = None
    ) -> Tuple[List["_Procedure"], int]:
        """
        Split the turn into stretches of work that a green run just accepted.

        A stretch is the evidence between two green runs, so a turn that
        implemented, tested, then extended and tested again yields two
        procedures rather than one document describing the whole turn.  A
        stretch that holds a test failure belongs to a recipe instead — that
        failure and its fix are already recorded, with the failure as the
        entry point — and a stretch without a patch proves nothing a document
        could repeat.

        *covered* names the patches a verified recipe has already published.
        An episode keeps every patch that followed its failure, so the change
        of a last, wider test run is usually inside that recipe's diff
        already; writing a procedure for it too would spend a mode's budget
        saying the same thing twice.

        The green run is the whole epistemics of this kind: a change nobody
        ran is an intention, and an intention indexed as a verified procedure
        is a guess the next session will trust.
        """
        already = covered or frozenset()
        procedures: List[_Procedure] = []
        covered_count = 0
        current = _Procedure()
        for position, evidence in enumerate(turn.evidence):
            if evidence.kind == EVIDENCE_TEST_FAILURE:
                current.failures += 1
                current.items.append(evidence)
                continue
            if evidence.kind == EVIDENCE_PATCH:
                current.patches.append(evidence)
                current.items.append(evidence)
                continue
            if self._is_green_test(evidence):
                recorded = {
                    item.call_id or item.event_id or "" for item in current.patches
                }
                if (
                    current.patches
                    and not current.failures
                    and not recorded <= already
                ):
                    current.verified_by = evidence
                    current.verified_at = position
                    current.items.append(evidence)
                    procedures.append(current)
                elif (
                    current.patches
                    and not current.failures
                    and recorded
                    and recorded <= already
                ):
                    # The recipe that owns these patches already says this;
                    # counted so a silent "no playbook" can be read as "the
                    # recipe covered it" rather than as a mining gap.
                    covered_count += 1
                current = _Procedure()
                continue
            current.items.append(evidence)
        return procedures, covered_count

    def _playbook(
        self, turn: AgentTurn, procedure: "_Procedure"
    ) -> Optional[KnowledgeEvent]:
        """
        Build the document of a change this turn made and its tests accepted.

        A recipe is read back in `debug`, where a failure is the question; a
        procedure is read back in `implement`, where the question is how this
        repository does the thing that has not gone wrong yet.  So the body
        leads with what was asked for, then what was changed, then the command
        that proved the change — the same three facts, in the order a task
        arrives in.
        """
        patches = procedure.patches
        if not patches or procedure.verified_by is None:
            return None
        patch_text = self._patch_text(turn, patches)
        if not patch_text.strip():
            return None
        command = self._command_text(turn, procedure.verified_by)[:200]
        paths = _merge_paths(
            _patched_paths(patch_text),
            _deduplicated(item for patch in patches for item in patch.paths),
        )
        task = (turn.latest_user_text or "").strip().splitlines()
        body = "\n".join(
            part
            for part in (
                f"Task: {task[0][:200]}" if task else "",
                f"Changed: {', '.join(paths) or 'unknown files'}",
                f"Verified by: {command or 'a test run of this turn'}",
                "",
                "Steps that worked:",
                patch_text,
            )
            if part
        )
        return self._event(
            turn,
            KIND_PLAYBOOK,
            body,
            signature=None,
            parts=self._procedure_parts(turn, procedure),
            test_command=command,
            paths=paths,
            phase=_phase_of_items(turn, procedure.items),
            edited_by=(
                "exec"
                if any(patch.strength == STRENGTH_WEAK for patch in patches)
                else None
            ),
        )

    @staticmethod
    def _procedure_parts(
        turn: AgentTurn, procedure: "_Procedure"
    ) -> Tuple[str, ...]:
        """
        The fingerprint parts of one accepted stretch of work.

        The identity of the first patch names the stretch and never changes
        while Codex replays the turn, which is what keeps one document per
        procedure rather than one per request; the green run is not part of
        it, because the run that closes a procedure is decided by where the
        turn happens to be when it is replayed.
        """
        first = procedure.patches[0] if procedure.patches else None
        identity = (first.call_id or first.event_id or "") if first else ""
        return tuple(part for part in ("playbook", turn.scope, identity) if part)

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
        """
        Count the repetitions of the failure this turn is stuck on.

        On a tie the *most recently seen* failure wins: a turn that failed A
        twice and then B twice ended on B, and the loop identity is what the
        injected advice and the escalation annotation are about.  Breaking the
        tie by hash would make it arbitrary — same loop, arbitrary identity.
        """
        texts = _output_texts(turn)
        counts: Dict[str, int] = {}
        seen_at: Dict[str, int] = {}
        for position, evidence in enumerate(self._failure_evidence(turn)):
            signature = signature_of(texts.get(evidence.call_id, ""))
            key = signature.signature_hash if signature else ""
            if not key:
                key = evidence.call_id or evidence.event_id or "?"
            counts[key] = counts.get(key, 0) + 1
            seen_at[key] = position
        if not counts:
            return MiningResult()
        key = max(counts, key=lambda name: (counts[name], seen_at[name]))
        matched = None
        for evidence in self._failure_evidence(turn):
            candidate = signature_of(texts.get(evidence.call_id, ""))
            if candidate is not None and candidate.signature_hash == key:
                matched = candidate
                break
        return MiningResult(
            loop_count=counts[key],
            signature_hash=key,
            signature=matched.signature if matched else "",
        )

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
        command = (
            failure.green_command or self._command_text(turn, failure.resolved_by)
        )[:200]
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
        return self._event(
            turn,
            KIND_RECIPE,
            body,
            signature=failure.signature,
            parts=self._episode_parts(turn, "recipe", failure),
            test_command=command,
            paths=_merge_paths(turn.touched_paths, failure.signature.paths),
            phase=_episode_phase(turn, failure),
        )

    # ------------------------------------------------------------- deadend --
    def _deadend(
        self, turn: AgentTurn, failure: _Failure
    ) -> Optional[KnowledgeEvent]:
        """Build the document of a change that did not fix the failure."""
        if not failure.patches or failure.signature is None:
            return None
        patch_text = self._patch_text(turn, failure.patches)
        paths = _patched_paths(patch_text)
        body = "\n".join(
            (
                f"Failure: {failure.signature.signature}",
                f"Attempted: {', '.join(paths[:12]) or 'unknown files'}",
                "Outcome: the same failure was still there afterwards.",
                "",
                "Change that did not fix it:",
                patch_text,
            )
        )
        return self._event(
            turn,
            KIND_DEADEND,
            body,
            signature=failure.signature,
            parts=self._episode_parts(turn, "deadend", failure),
            test_command=(
                failure.command or self._command_text(turn, failure.evidence)
            )[:200],
            paths=_merge_paths(paths, failure.signature.paths),
            phase=_episode_phase(turn, failure),
        )

    def _episode_parts(
        self, turn: AgentTurn, kind: str, failure: _Failure
    ) -> Tuple[str, ...]:
        """
        The fingerprint parts of one episode: what it is, where, of what.

        Scope belongs in the digest because the deduplication key does not carry
        it — without it, two repositories that hit the same normalized failure
        would compute the same fingerprint and the second one's episode would be
        dropped as a duplicate of the first.  The identity of the *first* patch
        belongs in it because it names this episode and never changes while the
        turn replays; the patches and failures that came after it do change, and
        a digest that covered them would publish one document per attempt.
        """
        parts: List[str] = [
            kind,
            failure.signature.signature_hash if failure.signature else "",
            turn.scope,
            failure.first_identity,
        ]
        return tuple(part for part in parts if part)

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
                "phase": turn.mode,
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
                phase=turn.mode,
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
        phase: str = "",
        edited_by: Optional[str] = None,
    ) -> KnowledgeEvent:
        """Assemble one publishable event out of the parts."""
        digest = fingerprint(*parts)
        text = build_document_text(
            kind,
            body,
            date=_today(),
            signature=signature.signature if signature else "",
            signature_hash=signature.signature_hash if signature else "",
            paths=paths,
            thread_id=turn.thread_id,
            agent=turn.agent,
            mode=turn.mode,
            test_command=test_command,
            phase=phase,
        )
        metadata: Dict[str, Any] = {
            "paths": list(paths),
            "thread_id": turn.thread_id,
            "mode": turn.mode,
            "agent": turn.agent,
        }
        if phase:
            metadata["phase"] = phase
        # A document built on a weak edit names its source, so a reader — or a
        # policy — can weigh it, or drop it, in one filter: the paths it
        # carries came off a shell command, not off a patch envelope.
        if edited_by:
            metadata["edited_by"] = edited_by
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
        """Quote the patches of an episode, capped.

        The envelope is what a reader needs, and a CLI release decides how it
        is carried — an ``apply_patch`` argument today, a here-document piped
        through ``exec_command`` yesterday — so the arguments are decoded first
        and quoted raw only when they hold no envelope at all.
        """
        arguments = _call_arguments(turn)
        chunks = []
        total = 0
        for item in patches:
            text = arguments.get(item.call_id) or arguments.get(item.event_id) or ""
            if not text.strip():
                continue
            bodies = patch_envelopes(text) or [text]
            for body in bodies:
                if not body.strip():
                    continue
                chunks.append(body.strip())
                total += len(body)
                if total >= self._max_patch_chars:
                    break
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
        return text
    import json

    try:
        decoded = json.loads(text)
    except (TypeError, ValueError):
        return text
    if isinstance(decoded, dict):
        for key in ("cmd", "command", "input"):
            value = decoded.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _same_failure(previous: _Failure, signature: Optional[ErrorSignature]) -> bool:
    """Whether a new failure is the one :attr:`_Failure` already describes."""
    if signature is None or previous.signature is None:
        return False
    return signature.signature_hash == previous.signature.signature_hash


def _deduplicated(lines: Iterable[str]) -> Tuple[str, ...]:
    """Keep the first of every equal line, in order."""
    seen: set = set()
    kept: List[str] = []
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        kept.append(line)
    return tuple(kept)


def _verification_scope(command: str) -> Optional[Tuple[str, ...]]:
    """Read an explicit scope, abstaining on shell syntax or unknown options."""
    if not command or re.search(r"[;&|<>$`\n\r*?]", command):
        return None
    try:
        tokens = shlex.split(_MODULE_LAUNCHER_RE.sub("", command))
    except ValueError:
        return None
    if not tokens:
        return None
    if tokens[0] != "pytest":
        # Other runners must match exactly, with no unrecognized options.
        if any(token.startswith("-") for token in tokens[1:]):
            return None
        return tuple(tokens)
    targets: List[str] = []
    for token in tokens[1:]:
        if re.fullmatch(r"-[qv]+", token) or token in (
            "-s",
            "--quiet",
            "--verbose",
            "--disable-warnings",
            "--no-header",
            "--no-summary",
        ):
            continue
        if token.startswith("-") or token.startswith("@"):
            return None
        targets.append(token)
    return ("pytest", *sorted(set(targets)))


def _verifies(failing: str, green: str) -> bool:
    """
    Whether a green run is the failing test, run again.

    A recipe is the strongest claim this layer makes — it says *this* change made
    *this* failure pass — and the escalation policy trusts it: a recipe scoring
    above the strong-recipe threshold is what keeps a looping model in place.  So
    "a command exited 0 afterwards" is not enough.  ``pytest -k 'not test_total'``,
    a different test file, and ``--collect-only`` all go green while proving
    nothing, and a recipe written from one of them is worse than no recipe at all.

    Parameters
    ----------
    failing : str
        Command whose output produced the failure.
    green : str
        Command that exited successfully after the fix.

    Returns
    -------
    bool
        ``True`` for equivalent explicit scopes.  Pytest targets (including node
        IDs and directories) may be reordered and reporting flags may differ.
        Selection flags, unknown options and shell wrappers are not proof, even
        if identical; no coverage of omitted tests is inferred from exit status.
    """
    left = _verification_scope(failing)
    right = _verification_scope(green)
    return left is not None and left == right


def _verified_patches(failures: Sequence[_Failure]) -> FrozenSet[str]:
    """
    Return the patches that a verified recipe already stands for.

    Parameters
    ----------
    failures : Sequence[_Failure]
        The turn's failure episodes, verified or not.

    Returns
    -------
    FrozenSet[str]
        Identities of the patches of every episode that a test run closed.
    """
    return frozenset(
        patch.call_id or patch.event_id or ""
        for failure in failures
        if failure.verified()
        for patch in failure.patches
    )


def _patched_paths(patch_text: str) -> Tuple[str, ...]:
    """Return the file paths one rendered patch touches, in order."""
    found = re.findall(
        r"^\*\*\* (?:(?:Update|Add|Delete) File|Move to): (.+)$",
        patch_text,
        re.MULTILINE,
    )
    return _deduplicated(found)


def _phase_of_items(turn: AgentTurn, items: Sequence[PhaseEvidence]) -> str:
    """Return the work phase a stretch of evidence happened in."""
    counted: Dict[str, int] = {}
    for item in items:
        if item.mode:
            counted[item.mode] = counted.get(item.mode, 0) + 1
    if not counted:
        return turn.mode
    best = max(counted.values())
    for item in reversed(items):
        if counted.get(item.mode, 0) == best:
            return item.mode
    return turn.mode


def _episode_phase(turn: AgentTurn, failure: _Failure) -> str:
    """
    Return the work phase one episode happened in.

    The mode routing resolved for the whole turn is not it: a turn that started
    by debugging, patched, and ended running tests produced its recipe while
    testing.  The phase is read from the episode's own evidence — the mode that
    names most of its items, the newest one breaking a tie — and falls back to
    the turn's mode when the episode names none, which is what an operator
    filters on when nothing was recorded.
    """
    items = [failure.evidence, *failure.patches, *failure.later_failures]
    if failure.resolved_by is not None:
        items.append(failure.resolved_by)
    return _phase_of_items(turn, items)


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
