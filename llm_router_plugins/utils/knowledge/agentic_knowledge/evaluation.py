"""
Counterfactual replay: what the knowledge layer would have changed.

Before switching writing and escalation on for live traffic, replay recorded
sessions and ask the question the production annotation cannot answer: given
only what the agent had done *so far*, would the knowledge base have handed it
the fix it eventually found, and how many of its spins around the same failure
would a stronger model have ended?

The replay is deliberately leak-free.  A turn is scored against knowledge mined
from strictly earlier turns of the same session — the recipe that a turn itself
produces is invisible to it, exactly as it would be in production, where a fix
is indexed after the request that discovered it.  A dedicated assertion in the
tests checks that no scored turn ever saw a document from its own or a later
turn, so "would have helped" can never quietly mean "already knew the answer".

Three variants are measured over the same replay: no knowledge, read-only, and
read-plus-escalation.  The deltas between them are the honest estimate of what
each half of the layer is worth.
"""

import argparse
import json
import sys

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from llm_router_plugins.utils.knowledge.agentic_knowledge.adapters.base import (
    AgentTurn,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.assembler import (
    ContextAssembler,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
    AgenticKnowledgeConfig,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.escalation import (
    ACTION_ESCALATE,
    EscalationPolicy,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.miner import EpisodeMiner
from llm_router_plugins.utils.knowledge.agentic_knowledge.query import QueryBuilder
from llm_router_plugins.utils.knowledge.agentic_knowledge.documents import (
    KIND_RECIPE,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.sse import (
    STATUS_OK,
    SearchRequest,
    SearchResult,
    StoreHit,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.payload import (
    CodexActivity,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.phase import (
    EVIDENCE_COMMAND,
    EVIDENCE_PATCH,
    EVIDENCE_TEST_FAILURE,
    PhaseEvidence,
)

__all__ = ["evaluate", "load_sessions", "ReplayEngine"]

#: The three configurations the replay measures.
VARIANT_NONE = "no_kb"
VARIANT_READ = "read"
VARIANT_ESCALATE = "read_escalate"

_TOKENS = __import__("re").compile(r"[a-z0-9_.:/-]+")

#: Fraction of the query a document must cover to be returned at all.  A
#: retrieval over a short query and a long document is scored by how much of
#: the query the document contains, not by symmetric overlap: that is what a
#: reranker rewards, and it is what makes a recipe match the failure it fixes.
_MIN_OVERLAP = 0.2


class _FakeStore:
    """
    A deterministic, in-memory stand-in for the search engine.

    Retrieval is token overlap, computed identically on every machine and every
    run, because an evaluation that drifts is worth nothing.  A document is
    returned for a request only from the requested collection and category, and
    only when the query and the document share enough of their words.
    """

    def __init__(self) -> None:
        """Start with an empty index."""
        self.documents: List[Dict[str, Any]] = []

    def index(self, kind: str, collection: str, text: str, mined_at: str) -> None:
        """Add one document to the index."""
        self.documents.append(
            {
                "collection": collection,
                "category": kind,
                "text": text,
                "mined_at": mined_at,
            }
        )

    def search(self, request: SearchRequest) -> SearchResult:
        """Return deterministic overlap-scored hits for *request*."""
        query = _tokens(request.query)
        if not query:
            return SearchResult(STATUS_OK)
        scored: List[Tuple[float, StoreHit, str]] = []
        for document in self.documents:
            if document["collection"] != request.collection:
                continue
            categories = request.categories or ()
            if categories and document["category"] not in categories:
                continue
            score = _overlap(query, _tokens(document["text"]))
            if score < _MIN_OVERLAP:
                continue
            scored.append(
                (
                    score,
                    StoreHit(
                        collection=document["collection"],
                        document_name=f"{document['category']}:{document['mined_at']}",
                        text=document["text"],
                        score_normalized=score,
                    ),
                    document["mined_at"],
                )
            )
        scored.sort(key=lambda item: (-item[0], item[2]))
        hits = tuple(item[1] for item in scored[: request.max_results])
        return SearchResult(STATUS_OK, hits=hits)

    def mined_at_for(self, document_name: str) -> str:
        """Map a returned document name back to the turn that mined it."""
        for document in self.documents:
            if f"{document['category']}:{document['mined_at']}" == document_name:
                return document["mined_at"]
        return ""


class ReplayEngine:
    """
    Replays sessions through the real read path and a fake store.

    Parameters
    ----------
    config : AgenticKnowledgeConfig
        The policy the replay runs under: budgets, sources, escalation thresholds.
    """

    def __init__(self, config: AgenticKnowledgeConfig) -> None:
        """Build the components the read path needs."""
        self.config = config
        self._miner = EpisodeMiner(
            max_patch_chars=config.max_patch_chars,
            archive_enabled=False,
        )
        self._query_builder = QueryBuilder(
            collection_namer=config.collection,
            min_similarity=config.min_similarity,
        )
        self._escalation = EscalationPolicy(config)

    def replay(self, sessions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Score every session under all three variants.

        Parameters
        ----------
        sessions : Sequence[Dict[str, Any]]
            Loaded sessions, each an ordered list of turns.

        Returns
        -------
        Dict[str, Any]
            The report: loop metrics and one block per variant.
        """
        variants = {
            VARIANT_NONE: _VariantTally(),
            VARIANT_READ: _VariantTally(),
            VARIANT_ESCALATE: _VariantTally(),
        }
        loop = _LoopTally()
        per_turn: List[Dict[str, Any]] = []

        for session in sessions:
            store = _FakeStore()
            turns = session.get("turns") or []
            mined: Dict[str, str] = {}
            for index, raw in enumerate(turns):
                turn = _build_turn(session, raw, self.config)
                mined_result = self._miner.mine(turn)
                # Publish the fix this turn produced *after* scoring it, so a
                # turn can never retrieve what it has not yet earned.
                policy = self.config.policy_for(turn.mode)
                context, retrieved = self._read(turn, policy, store, variants)
                for event in mined_result.events:
                    if event.kind == KIND_RECIPE:
                        store.index(
                            event.kind,
                            self.config.collection(turn.scope, event.kind),
                            event.text,
                            str(raw.get("id", index)),
                        )
                        mined[str(raw.get("id", index))] = event.signature_hash
                self._score(variants, raw, context, retrieved, store, index)
                self._escalate(variants, turn, context, mined_result, raw, policy)
                loop.record(raw, mined_result.loop_count, self.config.loop_threshold)
                per_turn.append(
                    {
                        "session": session.get("id", ""),
                        "turn": str(raw.get("id", index)),
                        "mode": turn.mode,
                        "loop_count": mined_result.loop_count,
                        "injected": not context.empty,
                        "retrieved_mined_at": retrieved,
                    }
                )
        return _report(sessions, loop, variants, per_turn)

    def _read(
        self,
        turn: AgentTurn,
        policy: Any,
        store: _FakeStore,
        variants: Dict[str, Any],
    ) -> Tuple[Any, List[str]]:
        """Run the read path for one turn; the no-KB variant reads nothing."""
        if policy is None:
            return _empty_context(), []
        assembler = ContextAssembler(touched_paths=turn.touched_paths)
        results = []
        retrieved: List[str] = []
        for request in self._query_builder.build(turn, policy):
            result = store.search(request)
            results.append((result, request.categories))
            for hit in result.hits:
                mined = store.mined_at_for(hit.document_name)
                if mined:
                    retrieved.append(mined)
        context = assembler.assemble(results, policy)
        return context, retrieved

    def _score(
        self,
        variants: Dict[str, Any],
        raw: Dict[str, Any],
        context: Any,
        retrieved: List[str],
        store: _FakeStore,
        index: int,
    ) -> None:
        """Record hit, save and injection metrics for the read variants."""
        saved_by = raw.get("saved_by")
        applicable = bool(saved_by)
        hit = bool(saved_by) and str(saved_by) in retrieved
        for name in (VARIANT_READ, VARIANT_ESCALATE):
            tally = variants[name]
            tally.turns += 1
            tally.injected_chars += context.chars
            if applicable:
                tally.applicable += 1
                if hit:
                    tally.hits += 1
                    tally.turns_saved += 1
        if applicable and hit:
            variants[VARIANT_NONE].missed_saves += 1
        variants[VARIANT_NONE].turns += 1

    def _escalate(
        self,
        variants: Dict[str, Any],
        turn: AgentTurn,
        context: Any,
        mined_result: Any,
        raw: Dict[str, Any],
        policy: Any,
    ) -> None:
        """Count the escalations the read-plus-escalate variant would make."""
        decision = self._escalation.decide(
            turn,
            context,
            policy,
            mined_result.loop_count,
            mined_result.signature_hash,
        )
        if decision.action == ACTION_ESCALATE:
            variants[VARIANT_ESCALATE].escalations += 1


@dataclass
class _VariantTally:
    """Accumulators for one variant."""

    turns: int = 0
    applicable: int = 0
    hits: int = 0
    turns_saved: int = 0
    injected_chars: int = 0
    escalations: int = 0
    missed_saves: int = 0

    def report(self) -> Dict[str, Any]:
        """The variant's metrics, rounded for a stable report."""
        return {
            "turns": self.turns,
            "recipe_hit_rate": _ratio(self.hits, self.applicable),
            "turns_potentially_saved": self.turns_saved,
            "missed_saves": self.missed_saves,
            "escalations": self.escalations,
            "injected_chars_per_turn": _ratio(self.injected_chars, self.turns),
        }


@dataclass
class _LoopTally:
    """Confusion counts for loop detection against the ground truth."""

    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    def record(self, raw: Dict[str, Any], loop_count: int, threshold: int) -> None:
        """Compare the detected loop with the turn's ground-truth label."""
        detected = loop_count >= int(threshold)
        expected = bool(raw.get("is_loop", False))
        if detected and expected:
            self.true_positive += 1
        elif detected and not expected:
            self.false_positive += 1
        elif expected and not detected:
            self.false_negative += 1

    def report(self) -> Dict[str, Any]:
        """Precision and recall of loop detection."""
        total_detected = self.true_positive + self.false_positive
        total_expected = self.true_positive + self.false_negative
        return {
            "precision": _ratio(self.true_positive, total_detected),
            "recall": _ratio(self.true_positive, total_expected),
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
        }


def _report(
    sessions: Sequence[Dict[str, Any]],
    loop: _LoopTally,
    variants: Dict[str, _VariantTally],
    per_turn: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Assemble the final report dictionary."""
    turns = sum(len(session.get("turns") or ()) for session in sessions)
    return {
        "sessions": len(sessions),
        "turns": turns,
        "loop_detection": loop.report(),
        "variants": {name: tally.report() for name, tally in variants.items()},
        "turn_details": per_turn,
    }


def _build_turn(
    session: Dict[str, Any], raw: Dict[str, Any], config: AgenticKnowledgeConfig
) -> AgentTurn:
    """Turn one dataset entry into an :class:`AgentTurn` with synthesized evidence."""
    activity, evidence = _synthesize(raw)
    scope = str(session.get("scope") or raw.get("scope") or "repo")
    return AgentTurn(
        agent="codex",
        scope=scope,
        repo_root=f"/repo/{scope}",
        session_id=str(session.get("id") or ""),
        thread_id=str(session.get("id") or ""),
        turn_id=str(raw.get("id") or ""),
        root_turn_id=str(raw.get("id") or ""),
        generation=str(raw.get("id") or ""),
        request_class="main",
        mode=str(raw.get("mode") or ""),
        routing_source=str(raw.get("source") or "phase"),
        model=str(raw.get("model") or "small-model"),
        latest_user_text=str(raw.get("text") or ""),
        window_id=str(raw.get("id") or ""),
        activity=activity,
        evidence=evidence,
    )


def _synthesize(
    raw: Dict[str, Any],
) -> Tuple[Tuple[CodexActivity, ...], Tuple[PhaseEvidence, ...]]:
    """
    Build the activity and evidence a recorded turn would have produced.

    A recorded turn is a compact description — a failure, maybe a patch,
    whether the test then passed, and how many times the failure repeated.  This
    expands it into the tool-call sequence and phase evidence the miner reads,
    so the replay exercises the real miner rather than a stub.
    """
    activity: List[CodexActivity] = []
    evidence: List[PhaseEvidence] = []
    failure = str(raw.get("failure") or "")
    patch = str(raw.get("patch") or "")
    test_command = str(raw.get("test_command") or "pytest")
    repeats = int(raw.get("repeat") or 1)

    for occurrence in range(max(1, repeats)):
        call = f"run-{occurrence}"
        activity.append(
            CodexActivity(
                kind="function_call",
                text=json.dumps({"command": test_command}),
                name="exec_command",
                call_id=call,
                event_id=f"call-{occurrence}",
            )
        )
        activity.append(
            CodexActivity(
                kind="function_call_output",
                text=failure,
                call_id=call,
                event_id=f"out-{occurrence}",
            )
        )
        evidence.append(
            PhaseEvidence(
                mode="test",
                kind=EVIDENCE_TEST_FAILURE,
                reason="eval",
                event_id=f"out-{occurrence}",
                call_id=call,
                completed=True,
                succeeded=False,
            )
        )
    if patch:
        activity.append(
            CodexActivity(
                kind="function_call",
                text=patch,
                name="apply_patch",
                call_id="patch-0",
                event_id="patch-evt",
            )
        )
        evidence.append(
            PhaseEvidence(
                mode="implement",
                kind=EVIDENCE_PATCH,
                reason="eval",
                event_id="patch-evt",
                call_id="patch-0",
                completed=True,
            )
        )
    if raw.get("test_passed"):
        activity.append(
            CodexActivity(
                kind="function_call",
                text=json.dumps({"command": test_command}),
                name="exec_command",
                call_id="green-0",
                event_id="green-evt",
            )
        )
        activity.append(
            CodexActivity(
                kind="function_call_output",
                text="1 passed",
                call_id="green-0",
                event_id="green-out",
            )
        )
        evidence.append(
            PhaseEvidence(
                mode="test",
                kind=EVIDENCE_COMMAND,
                reason="eval",
                event_id="green-out",
                call_id="green-0",
                completed=True,
                succeeded=True,
            )
        )
    return tuple(activity), tuple(evidence)


def _empty_context() -> Any:
    """An empty knowledge context, for turns with no mode policy."""
    from llm_router_plugins.utils.knowledge.agentic_knowledge.assembler import (
        KnowledgeContext,
    )

    return KnowledgeContext(cache="miss", status="disabled")


def _tokens(text: str) -> frozenset:
    """Lowercased word set of a text, the basis of the overlap score."""
    return frozenset(
        match.group(0) for match in _TOKENS.finditer((text or "").lower())
    )


def _overlap(query: frozenset, document: frozenset) -> float:
    """Fraction of the query's words that appear in the document."""
    if not query or not document:
        return 0.0
    return len(query & document) / float(len(query))


def _ratio(numerator: int, denominator: int) -> float:
    """A rounded ratio, ``0.0`` for an empty denominator."""
    if not denominator:
        return 0.0
    return round(numerator / denominator, 4)


def load_sessions(path: Path) -> List[Dict[str, Any]]:
    """
    Read sessions from a dataset file.

    Parameters
    ----------
    path : pathlib.Path
        JSON file with a ``sessions`` array.

    Returns
    -------
    List[Dict[str, Any]]
        The sessions, in file order.
    """
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    sessions = data.get("sessions") if isinstance(data, dict) else data
    return list(sessions or [])


def evaluate(
    config: AgenticKnowledgeConfig, sessions: Sequence[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Replay *sessions* and return the report.

    Parameters
    ----------
    config : AgenticKnowledgeConfig
        The policy to replay under.
    sessions : Sequence[Dict[str, Any]]
        The recorded sessions.

    Returns
    -------
    Dict[str, Any]
        The evaluation report.
    """
    return ReplayEngine(config=config).replay(sessions)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Run the evaluation from the command line.

    Parameters
    ----------
    argv : Sequence[str], optional
        Command-line arguments; ``sys.argv`` when omitted.

    Returns
    -------
    int
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        prog="agentic-kb-eval",
        description="Replay recorded sessions through the knowledge layer.",
    )
    parser.add_argument("--dataset", required=True, help="Path to the eval JSON.")
    parser.add_argument("--config", default=None, help="Path to a policy JSON.")
    parser.add_argument("--output", default=None, help="Write the report here.")
    args = parser.parse_args(argv)

    if args.config:
        config = AgenticKnowledgeConfig.from_file(Path(args.config))
    else:
        config = AgenticKnowledgeConfig.from_file()
    sessions = load_sessions(Path(args.dataset))
    report = evaluate(config, sessions)
    text = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    sys.exit(main())
