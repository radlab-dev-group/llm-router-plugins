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
import random
import sys
import time

from dataclasses import dataclass, field, replace
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
from llm_router_plugins.utils.knowledge.agentic_knowledge.learning import (
    CandidateFeatures,
    FeedbackSample,
    LearningPolicy,
    publish_snapshot,
    train_ranker,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.query import QueryBuilder
from llm_router_plugins.utils.knowledge.agentic_knowledge.documents import (
    KIND_RECIPE,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.doc_context import (
    DocContext,
    DocFile,
    build_doc_context,
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

#: The five configurations the replay measures.  The learned variant is the
#: same read path with the learner attached, fitted on the labels the replay
#: itself has earned so far: it measures what the model adds on top of the
#: static counters, on the same sessions, with the same anti-leak rule the
#: rest of the replay lives by.  The reputation variant is the learner with
#: one loop held back — the document's Beta posterior on its own, no fitted
#: weights and no source arm — because a learned variant that lost ground
#: says nothing about which of the three loops lost it.
VARIANT_NONE = "no_kb"
VARIANT_READ = "read"
VARIANT_ESCALATE = "read_escalate"
VARIANT_LEARNED = "read_learned"
VARIANT_REPUTATION = "read_reputation"

#: The learned variant is fitted on the labels the session has earned by the
#: time a request is scored.  The gates below are the replay's, not the
#: deployment's: a session of a few turns cannot fill ``min_samples`` of the
#: policy, and a fit the holdout cannot separate better than a coin must not
#: reorder the block it was meant to measure.  The seed is fixed because an
#: evaluation that drifts is worth nothing, and the exploration draws are
#: part of the measurement: the share of the votes a model spent exploring is
#: itself a metric of the variant.
_LEARNED_CONFIG_SEED = 11
_LEARNED_MIN_SAMPLES = 12
_LEARNED_MIN_EVIDENCE = 8.0
_LEARNED_MIN_HOLDOUT_AUC = 0.6
_LEARNED_HOLDOUT_FRACTION = 0.2
_LEARNED_EXPLORE_RATE = 0.1
_LEARNED_BOOTSTRAP = 200
#: The origin of the replay's fit clock.  A snapshot's id carries the second
#: it was fitted, and a report that carries the second it was produced is a
#: report that cannot be diffed against itself; the replay numbers its fits
#: from a fixed point instead.
_LEARNED_FIT_CLOCK_ORIGIN = 1_700_000_000
#: Shorter than one tick of the replay clock, so the snapshot a fit published
#: is the snapshot the next turn reads: the variant measures the model the
#: labels earned, not a model a cache held onto.
_LEARNED_SNAPSHOT_TTL_SECONDS = 0.001

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

    def index(
        self,
        kind: str,
        collection: str,
        text: str,
        mined_at: str,
        phase: str = "",
        doc_type: str = "",
    ) -> None:
        """
        Add one document of any kind to the index.

        The index mirrors the engine the read path talks to: a document is
        filed under its kind, its collection, the phase it was mined in and,
        for documentation, the kind of page it is.  Indexing only recipes
        used to make the replay blind to the rest of the knowledge base —
        the dead ends, the procedures and the documentation the read path
        actually injects — so the metrics it produced understated the block.
        """
        self.documents.append(
            {
                "collection": collection,
                "category": kind,
                "text": text,
                "mined_at": mined_at,
                "phase": phase,
                "doc_type": doc_type,
            }
        )

    def search(
        self, request: SearchRequest, min_similarity: float = 0.0
    ) -> SearchResult:
        """
        Return deterministic overlap-scored hits for *request*.

        The request's metadata restrictions are honored the way the engine
        honors them: the collection, the categories, the phases and — for a
        documentation search — the kinds of page asked for.  A document
        below the *min_similarity* floor is not returned at all, which is
        the vector leg's cut and not the caller's.
        """
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
            if request.phases and document["phase"] not in request.phases:
                continue
            if (
                request.doc_types
                and document.get("doc_type")
                and document["doc_type"] not in request.doc_types
            ):
                continue
            score = _overlap(query, _tokens(document["text"]))
            if score < _MIN_OVERLAP or score < float(min_similarity):
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


class _MemoryRedis:
    """
    The in-memory store the learned replay learns against.

    The learner's three readers want a client: the reputation wants a hash per
    document, the bandit a hash per arm, the log a stream, and the snapshot a
    string.  Nothing here is a Redis; it is the smallest object that satisfies
    those four readers in one process, and the only reason it exists is that
    the replay must fit and read a model without a deployment to borrow from.
    """

    def __init__(self) -> None:
        """Start with an empty namespace."""
        self.kv: Dict[str, str] = {}
        self.hashes: Dict[str, Dict[str, str]] = {}
        self.streams: Dict[str, List[Dict[str, Any]]] = {}

    def get(self, key: str) -> Optional[str]:
        return self.kv.get(key)

    def set(self, key: str, value: str, **_: Any) -> bool:
        self.kv[key] = str(value)
        return True

    def hset(self, key: str, field: str, value: Any) -> int:
        self.hashes.setdefault(key, {})[str(field)] = str(value)
        return 1

    def hgetall(self, key: str) -> Dict[str, str]:
        return dict(self.hashes.get(key, {}))

    def hincrby(self, key: str, field: str, amount: int = 1) -> int:
        bucket = self.hashes.setdefault(key, {})
        current = int(bucket.get(str(field), "0")) + int(amount)
        bucket[str(field)] = str(current)
        return current

    def expire(self, key: str, seconds: Any) -> bool:
        return True

    def xadd(self, key: str, fields: Dict[str, Any], maxlen: Any = None, **_: Any) -> str:
        stream = self.streams.setdefault(key, [])
        stream.append(dict(fields))
        return str(len(stream))


class _LearnedReplay:
    """
    The learned variant for one scope: its log, its fits, its policy.

    One instance per scope, which is what production keys this state by, and
    one more reason than tidiness: a holdout split is by session, so a corpus
    of one session has nothing to hold out and the gate that is supposed to
    refuse a fit that cannot separate new situations would grade the fit on
    the labels it trained on.  A scope accumulates the sessions the replay
    has walked through, in order, and the anti-leak rule stays a rule about
    time: the model a request may see was fitted on labels that had already
    been earned when the request was scored, and on nothing earned later.  A
    scope with no labels yet gets the neutral snapshot, which reads as ``1.0``
    everywhere and reorders nothing, so the variant degrades to the read
    variant rather than to a guess.

    The gates are the replay's, not the deployment's: a dataset of a few
    sessions cannot fill ``learning.min_samples``, and the question the
    variant asks is whether the learned path *would* have ranked better, not
    whether one dataset is a statistically legal corpus for a production
    publish.  What must not move is determinism: one seeded generator per
    scope, and a clock that is a counter rather than the wall clock, because
    a model id built from the second the fit ran is a report that changes when
    nothing about the replay did.
    """

    def __init__(self, scope: str, config: AgenticKnowledgeConfig) -> None:
        """Build the learners; fit nothing."""
        self.scope = scope
        self.config = replace(
            config.learning,
            enabled=True,
            min_samples=_LEARNED_MIN_SAMPLES,
            min_evidence=_LEARNED_MIN_EVIDENCE,
            min_holdout_auc=_LEARNED_MIN_HOLDOUT_AUC,
            explore_rate=_LEARNED_EXPLORE_RATE,
            model_ttl_seconds=_LEARNED_SNAPSHOT_TTL_SECONDS,
        )
        self.client = _MemoryRedis()
        self.rng = random.Random(_LEARNED_CONFIG_SEED)
        # A counter, not the wall clock: a snapshot is cached for a TTL, and
        # the replay must see the snapshot it published between two turns,
        # which a clock that advances a second per turn guarantees without
        # asking the policy for a TTL of zero it refuses to have.
        self._tick = 0.0
        self.policy = LearningPolicy(
            config=self.config,
            client=self.client,
            rng=self.rng,
            time_fn=lambda: self._tick,
        )
        #: The reputation loop on its own: the document's Beta posterior with
        #: no fitted weights and no source arm, in a namespace of its own so
        #: neither the snapshot nor the arms of the full learner leak into it.
        #: It exists because a learned variant that loses ground says nothing
        #: about which of the three loops lost it, and the plan's phase-2
        #: criterion is about this loop alone.
        self.reputation_policy = LearningPolicy(
            config=replace(self.config, explore_rate=0.0),
            client=_MemoryRedis(),
            rng=random.Random(_LEARNED_CONFIG_SEED + 1),
            time_fn=lambda: self._tick,
        )
        self.samples: List[FeedbackSample] = []
        #: How many times the fit ran and what it said; the report reads
        #: these rather than re-fitting, because the fit is the expensive
        #: half of the replay and it is the one thing that must not drift.
        self.fits = 0
        self.accepted = 0
        self.refusals: Dict[str, int] = {}
        self.last_auc = 0.0
        self.model_ids: List[str] = []
        #: The votes the policy spent exploring, and the votes it spent at
        #: all: the share is a metric of the variant, because a model that
        #: never explores is a model that never changes its mind.
        self.explore_reads = 0
        self.candidate_reads = 0
        self.mults: List[float] = []

    def snapshot(self) -> Dict[str, Any]:
        """
        What the learner knew right now, for the per-turn record.

        The count of settled records and the id of the model in force are the
        two numbers a leak assertion needs, and they are recorded from the
        request's point of view rather than reconstructed afterwards: a report
        that says "the model that served turn seven was fitted on these
        fourteen records" is a report a reader can check.
        """
        served = self.policy.annotation(self.scope) or {}
        return {
            "samples": len(self.samples),
            "labelled": sum(1 for sample in self.samples if sample.label != 0.0),
            "model_id": str(served.get("model_id") or ""),
            "holdout_auc": round(float(served.get("holdout_auc") or 0.0), 3),
        }

    def tick(self) -> None:
        """Advance the replay clock so the next request reads a fresh model."""
        self._tick += 1.0

    def observe(
        self,
        session_id: str,
        generation: str,
        context: Any,
        features: Dict[str, CandidateFeatures],
        saved_by: str,
        is_loop: bool,
    ) -> None:
        """
        Settle one scored turn into the corpus, then re-fit if anything moved.

        The labels are the turn's own ground truth, settled *after* the turn
        was scored, which is the whole anti-leak argument: the model that
        ranked this request never saw this request's label, and the next
        request of the session gets a model that did.  A turn that earned its
        fix from a document on screen credits that document at full strength
        and leaves the rest unlabelled, because "was on screen" is not "was
        followed"; a turn that looped with the block on screen rejects it at a
        third, which is the graded label of the settle path and not an
        invention of this replay.
        """
        labelled = False
        for hit in context.hits:
            name = hit.document_name
            if not name or name not in features:
                continue
            if saved_by and name.endswith(f":{saved_by}"):
                label, kind = 1.0, "counterfactual"
            elif is_loop:
                label, kind = -0.3, "coarse"
            else:
                label, kind = 0.0, "counterfactual"
            sample = FeedbackSample(
                scope=self.scope,
                session_id=session_id,
                generation=generation,
                document_name=name,
                features=features[name],
                label=label,
                label_kind=kind,
            )
            self.samples.append(sample)
            self.policy.settle(sample)
            # The reputation-only loop gets the same verdict and nothing
            # else: no bandit observation, no fit, no snapshot.  Its state is
            # the Beta posterior and that is the whole of what the variant is
            # allowed to know.
            if sample.label:
                self.reputation_policy.reputation.observe(name, sample.label)
            labelled = labelled or label != 0.0
        if labelled:
            self._fit()

    def _fit(self) -> None:
        """Re-fit on the corpus so far and publish what the gates accept."""
        self.fits += 1
        snapshot, report = train_ranker(
            self.samples,
            self.scope,
            config=self.config,
            holdout_fraction=_LEARNED_HOLDOUT_FRACTION,
            rng=self.rng,
            # A counter, not the wall clock: the id of a model is part of the
            # report, and a report that changes between two runs of the same
            # dataset reports the clock rather than the replay.
            now=float(_LEARNED_FIT_CLOCK_ORIGIN + self.fits),
        )
        if report.get("accepted"):
            self.accepted += 1
            self.model_ids.append(snapshot.model_id)
        else:
            reason = str(report.get("refused") or "refused")
            self.refusals[reason] = self.refusals.get(reason, 0) + 1
        self.last_auc = float(report.get("holdout_auc", 0.0))
        publish_snapshot(self.client, snapshot, key_prefix=self.config.key_prefix)

    def note_vote(self, learning_detail: Dict[str, Any]) -> None:
        """Remember what one candidate's vote cost, for the variant's metrics."""
        self.candidate_reads += 1
        if learning_detail.get("explore"):
            self.explore_reads += 1
        mult = learning_detail.get("mult")
        if isinstance(mult, (int, float)):
            self.mults.append(float(mult))

    def report(self) -> Dict[str, Any]:
        """What the fitting did, reported apart from the variant's tally."""
        return {
            "fits": self.fits,
            "accepted": self.accepted,
            "refusals": dict(self.refusals),
            "holdout_auc": round(self.last_auc, 3),
            "model_ids": list(self.model_ids),
            "explore_share": _ratio(self.explore_reads, self.candidate_reads),
            "mult_p95": _percentile(self.mults, 0.95),
        }


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
            VARIANT_LEARNED: _VariantTally(),
            VARIANT_REPUTATION: _VariantTally(),
        }
        loop = _LoopTally()
        per_turn: List[Dict[str, Any]] = []
        # One learned replay per scope — the unit the model is fitted per —
        # and the report folds their metrics together at the end.  The
        # confidence interval is still resampled per session, because a
        # session is the unit of evidence even when it is not the unit of the
        # state.
        learned: Dict[str, _LearnedReplay] = {}

        for session in sessions:
            store = _FakeStore()
            turns = session.get("turns") or []
            mined: Dict[str, str] = {}
            scope = str(session.get("scope") or "repo")
            learned_state = learned.get(scope)
            if learned_state is None:
                learned_state = learned[scope] = _LearnedReplay(scope, self.config)
            # A session may declare the documentation its repository had
            # written down; the digest of those files is what the read path
            # injects on every request of the session, and a replay that
            # ignored it understated the block by the whole map.
            doc_context = _doc_context(session, self.config)
            for index, raw in enumerate(turns):
                turn = _build_turn(session, raw, self.config)
                learned_state.tick()
                mined_result = self._miner.mine(turn)
                # Publish the fix this turn produced *after* scoring it, so a
                # turn can never retrieve what it has not yet earned.
                policy = self.config.policy_for(turn.mode)
                context, retrieved, _ = self._read(
                    turn, policy, store, variants, doc_context
                )
                # The same turn, the same store, the same moment — and a
                # learner attached, holding only the labels this session had
                # already earned when this request was scored.
                learned_context, learned_retrieved, learned_features = self._read(
                    turn,
                    policy,
                    store,
                    variants,
                    doc_context,
                    learner=learned_state.policy,
                )
                # The same turn again, this time with only the document's
                # reputation moved: the delta of this variant against ``read``
                # is the phase-2 criterion, and the delta of ``read_learned``
                # against it is what the fitted model and the arms add.
                reputation_context, reputation_retrieved, _ = self._read(
                    turn,
                    policy,
                    store,
                    variants,
                    doc_context,
                    learner=learned_state.reputation_policy,
                )
                for hit in learned_context.hits:
                    learned_state.note_vote(dict(getattr(hit, "learning", {}) or {}))
                # Recorded before the turn's own verdict is settled, which is
                # exactly the claim the leak assertion in the tests checks:
                # the corpus a request was ranked with is the corpus the
                # earlier turns had left behind, and nothing more.
                learned_before = learned_state.snapshot()
                for event in mined_result.events:
                    # Every mined kind is indexed, not only recipes: a dead
                    # end the replay cannot see is a regression the metrics
                    # cannot catch, and the block the read path injects is
                    # made of more than recipes.
                    store.index(
                        event.kind,
                        self.config.collection(turn.scope, event.kind),
                        event.text,
                        str(raw.get("id", index)),
                        phase=str(event.metadata.get("phase", "")),
                        doc_type=str(event.metadata.get("doc_type", "")),
                    )
                    if event.kind == KIND_RECIPE:
                        mined[str(raw.get("id", index))] = event.signature_hash
                self._score(
                    variants,
                    raw,
                    context,
                    retrieved,
                    store,
                    index,
                    session_id=str(session.get("id") or ""),
                )
                self._score(
                    variants,
                    raw,
                    learned_context,
                    learned_retrieved,
                    store,
                    index,
                    variant_names=(VARIANT_LEARNED,),
                    session_id=str(session.get("id") or ""),
                    count_baseline=False,
                )
                self._score(
                    variants,
                    raw,
                    reputation_context,
                    reputation_retrieved,
                    store,
                    index,
                    variant_names=(VARIANT_REPUTATION,),
                    session_id=str(session.get("id") or ""),
                    count_baseline=False,
                )
                self._escalate(variants, turn, context, mined_result, raw, policy)
                loop.record(raw, mined_result.loop_count, self.config.loop_threshold)
                # The turn's own verdict is settled only now, after both reads
                # were scored: the label this turn produced belongs to the
                # next request's model, never to this one's.
                learned_state.observe(
                    session_id=str(session.get("id") or ""),
                    generation=str(raw.get("id", index)),
                    context=learned_context,
                    features=learned_features,
                    saved_by=str(raw.get("saved_by") or ""),
                    is_loop=bool(raw.get("is_loop", False)),
                )
                learned_after = learned_state.snapshot()
                per_turn.append(
                    {
                        "session": session.get("id", ""),
                        "turn": str(raw.get("id", index)),
                        "mode": turn.mode,
                        "loop_count": mined_result.loop_count,
                        "injected": not context.empty,
                        "retrieved_mined_at": retrieved,
                        "learned": dict(
                            learned_before,
                            scope=scope,
                            added=learned_after["samples"] - learned_before["samples"],
                        ),
                    }
                )
        return _report(sessions, loop, variants, per_turn, learned)

    def _read(
        self,
        turn: AgentTurn,
        policy: Any,
        store: _FakeStore,
        variants: Dict[str, Any],
        doc_context: Optional[DocContext] = None,
        learner: Optional[LearningPolicy] = None,
    ) -> Tuple[Any, List[str], Dict[str, CandidateFeatures]]:
        """
        Run the read path for one turn; the no-KB variant reads nothing.

        The features the assembler showed the learner are returned with the
        context.  They are what the next turn's fit trains on, and they are
        the reason the learned variant must run the same assembler rather
        than a stand-in: a model trained on features the read path never
        produced is a model fitted on the replay's invention, and its
        uplift measures the same invention.
        """
        if policy is None:
            return _empty_context(), [], {}
        assembler = ContextAssembler(
            touched_paths=turn.touched_paths,
            learning=learner,
            scope=turn.scope if learner is not None else "",
        )
        results = []
        retrieved: List[str] = []
        for request in self._query_builder.build(turn, policy):
            result = store.search(request, min_similarity=self.config.min_similarity)
            results.append((result, request.categories))
            for hit in result.hits:
                mined = store.mined_at_for(hit.document_name)
                if mined:
                    retrieved.append(mined)
        context = assembler.assemble(results, policy, doc_context=doc_context)
        return context, retrieved, dict(assembler.candidate_features)

    def _score(
        self,
        variants: Dict[str, Any],
        raw: Dict[str, Any],
        context: Any,
        retrieved: List[str],
        store: _FakeStore,
        index: int,
        variant_names: Tuple[str, ...] = (VARIANT_READ, VARIANT_ESCALATE),
        session_id: str = "",
        count_baseline: bool = True,
    ) -> None:
        """
        Record hit, save and injection metrics for the named variants.

        The no-KB baseline is one answer about the dataset, not one per
        variant: it is counted on the read that represents the layer without
        learning, and the learned read must not be allowed to move it, or a
        variant could "improve" the baseline by being expensive.
        """
        saved_by = raw.get("saved_by")
        applicable = bool(saved_by)
        hit = bool(saved_by) and str(saved_by) in retrieved
        # The rank is taken in the order the search returned the documents,
        # which is the question the metric asks ("did retrieval surface it"),
        # and on a document's first appearance, because a document retrieved
        # twice was still found once.
        in_top_3, reciprocal_rank = (
            _ranking_quality(str(saved_by), retrieved) if applicable else (0, 0.0)
        )
        for name in variant_names:
            tally = variants[name]
            tally.turns += 1
            tally.injected_chars += context.chars
            tally.retrieved_chars += _retrieved_chars(context)
            if getattr(context, "doc_context", None):
                tally.turns_with_doc_context += 1
            if applicable:
                tally.applicable += 1
                tally.relevant_at_3 += in_top_3
                tally.reciprocal_rank_sum += reciprocal_rank
                if hit:
                    tally.hits += 1
                    tally.turns_saved += 1
            # Per session, because the confidence interval of "did the model
            # help" is resampled over sessions: a dataset is a handful of
            # sessions, and a turn-level interval would report the size of
            # the sessions rather than the size of the evidence.
            if session_id:
                session_counts = tally.per_session.setdefault(session_id, _SessionCount())
                session_counts.turns += 1
                if applicable:
                    session_counts.applicable += 1
                    if hit:
                        session_counts.hits += 1
        if count_baseline:
            if applicable:
                variants[VARIANT_NONE].applicable += 1
            if applicable and hit:
                variants[VARIANT_NONE].missed_saves += 1
            variants[VARIANT_NONE].turns += 1
            if session_id:
                session_counts = variants[VARIANT_NONE].per_session.setdefault(
                    session_id, _SessionCount()
                )
                session_counts.turns += 1
                if applicable:
                    session_counts.applicable += 1
                    if hit:
                        session_counts.hits += 1

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
        # The denominator of the recall the tune is asked to keep is the
        # loops the ground truth marks, whatever this variant chose to do
        # with them: a variant that escalated on none of them has a recall of
        # zero, and a precision floor alone would never have said so, because
        # a point that escalates on nothing also escalates on nothing wrong.
        if raw.get("is_loop"):
            variants[VARIANT_ESCALATE].true_loops += 1
        decision = self._escalation.decide(
            turn,
            context,
            policy,
            mined_result.loop_count,
            mined_result.signature_hash,
        )
        if decision.action == ACTION_ESCALATE:
            variants[VARIANT_ESCALATE].escalations += 1
            # Whether the escalation was the right call is a question only the
            # ground truth answers, and the replay is the one place the ground
            # truth is: an escalation on a turn that was not a loop is the
            # false positive the precision floor exists to keep bounded, and
            # it is what a threshold the tune moves can break without the
            # replay noticing.
            if raw.get("is_loop"):
                variants[VARIANT_ESCALATE].escalations_on_true_loop += 1


@dataclass
class _SessionCount:
    """One session's contribution to a variant, the unit of the bootstrap."""

    turns: int = 0
    applicable: int = 0
    hits: int = 0


@dataclass
class _VariantTally:
    """Accumulators for one variant."""

    turns: int = 0
    applicable: int = 0
    hits: int = 0
    turns_saved: int = 0
    injected_chars: int = 0
    escalations: int = 0
    #: Of the escalations the variant made, the ones the ground truth marks as
    #: loops: the numerator of the precision floor a tune is asked to keep.
    escalations_on_true_loop: int = 0
    #: The turns the ground truth marks as loops, whatever the variant did:
    #: the denominator of the recall a tune is asked to keep, which is a
    #: property of the dataset rather than of the policy, and is counted
    #: here, where every turn of the replay passes, rather than reconstructed
    #: from the report afterwards.
    true_loops: int = 0
    missed_saves: int = 0
    #: Characters of retrieved content, before the standing documentation
    #: map the block may also carry; the fixed and the earned halves of the
    #: injection are reported apart.
    retrieved_chars: int = 0
    turns_with_doc_context: int = 0
    #: Ranking against the document that eventually closed the failure: the
    #: plan's Precision@3 and MRR.  They are measured here rather than in
    #: production because a live request has no ground truth — only the replay
    #: knows which document the agent went on to need.
    relevant_at_3: int = 0
    reciprocal_rank_sum: float = 0.0
    #: Per session, for the bootstrap: the uplift the learned variant claims
    #: is only as trustworthy as the number of sessions it was measured over.
    per_session: Dict[str, "_SessionCount"] = field(default_factory=dict)

    def report(self) -> Dict[str, Any]:
        """The variant's metrics, rounded for a stable report."""
        return {
            "turns": self.turns,
            "recipe_hit_rate": _ratio(self.hits, self.applicable),
            "turns_potentially_saved": self.turns_saved,
            "missed_saves": self.missed_saves,
            "escalations": self.escalations,
            # None when the variant never escalated: precision with no
            # escalations is a constraint that is met by doing nothing, and
            # reporting it as 1.0 would let a variant that escalates on
            # nothing read as one that escalates perfectly.
            "escalation_precision": (
                _ratio(self.escalations_on_true_loop, self.escalations)
                if self.escalations
                else None
            ),
            # None when the dataset marked no loops: a recall with no loops
            # to recall is a constraint met by definition, and the tune's
            # floor reads that the way it reads the precision's empty case —
            # as met, never as perfect-by-accident of the arithmetic.
            "escalation_recall": (
                _ratio(self.escalations_on_true_loop, self.true_loops)
                if self.true_loops
                else None
            ),
            # The block a request is shown: retrieved content and, where the
            # mode reads documentation, the standing map the block carries.
            # The two are measured apart, because the map is a fixed cost the
            # operator opted into and the retrieved content is the variable
            # part the read path earns.
            "injected_chars_per_turn": _ratio(self.injected_chars, self.turns),
            "retrieved_chars_per_turn": _ratio(self.retrieved_chars, self.turns),
            "turns_with_doc_context": self.turns_with_doc_context,
            # Of the turns that had a document to find, how often the search
            # put it in the first three, and how near the top it sat on
            # average.  Both are over the applicable turns, so a turn with no
            # fix to find dilutes neither.
            "precision_at_3": _ratio(self.relevant_at_3, self.applicable),
            "mrr": _ratio(self.reciprocal_rank_sum, self.applicable),
            # Turns saved per character the block cost: the plan's
            # ``char_efficiency``, and the number to maximize, because
            # knowledge is paid for in tokens.  A variant that saved every
            # turn and flooded the context is not the winner, and this is the
            # metric that says so.  Reported next to its inverse, which reads
            # more naturally in a report: what one save cost.
            "char_efficiency": _efficiency(self.turns_saved, self.injected_chars, self.turns),
            "chars_per_turn_saved": _ratio(self.injected_chars, self.turns_saved),
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
    learned: Optional[Dict[str, _LearnedReplay]] = None,
) -> Dict[str, Any]:
    """Assemble the final report dictionary."""
    turns = sum(len(session.get("turns") or ()) for session in sessions)
    report = {
        "sessions": len(sessions),
        "turns": turns,
        "loop_detection": loop.report(),
        "variants": {name: tally.report() for name, tally in variants.items()},
        "turn_details": per_turn,
    }
    if learned:
        report["learning"] = _learned_report(list(learned.values()))
        report["variants"][VARIANT_LEARNED][
            "model_uplift_vs_read"
        ] = _bootstrap_uplift(
            variants.get(VARIANT_READ), variants.get(VARIANT_LEARNED)
        )
        # The reputation loop measured on its own, against the same read:
        # the cheap half of the learned path, and the half that has to work
        # before a fitted model is worth its cost.
        report["variants"][VARIANT_REPUTATION][
            "reputation_uplift_vs_read"
        ] = _bootstrap_uplift(
            variants.get(VARIANT_READ), variants.get(VARIANT_REPUTATION)
        )
    return report


def _learned_report(states: Sequence[_LearnedReplay]) -> Dict[str, Any]:
    """
    Fold the per-session learners into one view.

    The counts add, the shares are weighted by the votes that produced them,
    and the refusals are kept per reason: "the model never fitted" and "the
    model fitted and was refused by its own gates" are different answers to
    the same empty report, and an operator reading the difference decides
    different things.
    """
    fits = sum(state.fits for state in states)
    accepted = sum(state.accepted for state in states)
    refusals: Dict[str, int] = {}
    candidate_reads = sum(state.candidate_reads for state in states)
    explore_reads = sum(state.explore_reads for state in states)
    mults: List[float] = []
    aucs = []
    model_ids: List[str] = []
    for state in states:
        for reason, count in state.refusals.items():
            refusals[reason] = refusals.get(reason, 0) + count
        mults.extend(state.mults)
        if state.fits:
            aucs.append(state.last_auc)
        model_ids.extend(state.model_ids)
    return {
        "scopes": len(states),
        "fits": fits,
        "accepted": accepted,
        "refusals": refusals,
        "holdout_auc_mean": _ratio(sum(aucs), len(aucs)) if aucs else 0.0,
        "model_ids": sorted(set(model_ids)),
        "explore_share": _ratio(explore_reads, candidate_reads),
        "mult_p95": _percentile(mults, 0.95),
        "mult_min": round(min(mults), 4) if mults else 0.0,
        "mult_max": round(max(mults), 4) if mults else 0.0,
    }


def _bootstrap_uplift(
    read: Optional[_VariantTally], learned: Optional[_VariantTally]
) -> Dict[str, Any]:
    """
    How much the model moved the hit rate, with a confidence interval.

    The point estimate is the difference of the two aggregate rates.  What
    makes it a measurement rather than an anecdote is the interval: sessions
    are resampled with replacement and the difference recomputed on each
    draw, because the turns of one session are not independent evidence —
    they share a repository, a failure and often the same document — and an
    interval computed over turns would claim the precision of a hundred
    observations from ten sessions that said the same thing once.

    The seed is fixed.  An interval that moves between two runs of the same
    dataset cannot be quoted in a decision, and this report exists to be
    quoted.
    """
    if read is None or learned is None:
        return {}
    point = _ratio(learned.hits, learned.applicable) - _ratio(
        read.hits, read.applicable
    )
    sessions = sorted(set(read.per_session) | set(learned.per_session))
    if len(sessions) < 2:
        # One session is one observation of the difference, and one
        # observation has no spread.  Saying so is the report; inventing an
        # interval out of a single session is not.
        return {
            "point": round(point, 4),
            "ci_low": None,
            "ci_high": None,
            "bootstrap": 0,
            "sessions": len(sessions),
            "note": "a single session carries no interval",
        }
    rng = random.Random(_LEARNED_CONFIG_SEED)
    draws: List[float] = []
    for _ in range(_LEARNED_BOOTSTRAP):
        hits_read = applicable_read = 0
        hits_learned = applicable_learned = 0
        for _ in range(len(sessions)):
            name = sessions[rng.randrange(len(sessions))]
            read_counts = read.per_session.get(name)
            learned_counts = learned.per_session.get(name)
            if read_counts is not None:
                applicable_read += read_counts.applicable
                hits_read += read_counts.hits
            if learned_counts is not None:
                applicable_learned += learned_counts.applicable
                hits_learned += learned_counts.hits
        draws.append(
            _ratio(hits_learned, applicable_learned)
            - _ratio(hits_read, applicable_read)
        )
    return {
        "point": round(point, 4),
        "ci_low": round(_percentile(draws, 0.025), 4),
        "ci_high": round(_percentile(draws, 0.975), 4),
        "bootstrap": _LEARNED_BOOTSTRAP,
        "sessions": len(sessions),
    }


def _percentile(values: Sequence[float], fraction: float) -> float:
    """
    A linear-interpolated percentile; ``0.0`` for nothing to rank.

    The nearest-rank definition would report a value the sample actually
    contained, which for a p95 of a dozen votes is the largest vote there
    was — a number that moves with the last sample rather than with the
    distribution the metric is meant to describe.
    """
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return round(ordered[0], 4)
    position = _clamp(float(fraction), 0.0, 1.0) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return round(ordered[lower] * (1.0 - weight) + ordered[upper] * weight, 4)


def _clamp(value: float, low: float, high: float) -> float:
    """Constrain to the interval; the interval wins when it is empty."""
    return max(low, min(high, value))


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


def _doc_context(
    session: Dict[str, Any], config: AgenticKnowledgeConfig
) -> Optional[DocContext]:
    """
    The standing documentation digest of a recorded session, if it declared one.

    A session declares the documentation its repository had written down —
    path, kind, headings — and the digest built from it is what the read path
    injects on every request of that session.  A session that declares none
    replays without a map, exactly as a repository with no markdown reads.
    """
    raw = session.get("docs") or ()
    if not raw:
        return None
    files = tuple(
        DocFile(
            path=str(item.get("path") or "").strip(),
            doc_type=str(item.get("doc_type") or ""),
            headings=tuple(
                str(heading).strip()
                for heading in (item.get("headings") or ())
                if str(heading).strip()
            ),
            content_hash=str(item.get("hash") or ""),
        )
        for item in raw
        if str(item.get("path") or "").strip()
    )
    if not files:
        return None
    return build_doc_context(
        str(session.get("scope") or "repo"),
        files,
        max_chars=config.doc_context_max_chars,
    )


def _retrieved_chars(context: Any) -> int:
    """Characters of retrieved content in one block, without the map."""
    if context is None or context.empty:
        return 0
    digest = getattr(context, "doc_context", None)
    if not digest:
        return int(context.chars)
    return max(0, int(context.chars) - len(digest.text))


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


def _rank_of(needle: str, retrieved: Sequence[str]) -> int:
    """
    The 1-based position of *needle* in *retrieved*, ``0`` when absent.

    Duplicates collapse onto their first appearance: a document the search
    returned twice was found at its earliest position, and counting the later
    one would punish a retrieval for a redundancy that helped it.
    """
    for position, name in enumerate(retrieved, start=1):
        if str(name) == needle:
            return position
    return 0


def _ranking_quality(needle: str, retrieved: Sequence[str]) -> Tuple[int, float]:
    """
    ``(in_top_3, reciprocal_rank)`` of the document the turn needed.

    Split out of the tally because this is the whole of the two metrics'
    arithmetic, and it is the part worth pinning without depending on which
    order a fake search happens to return its hits in.
    """
    rank = _rank_of(needle, retrieved)
    if not rank:
        return 0, 0.0
    return (1 if rank <= 3 else 0), 1.0 / rank


def _efficiency(saved: int, injected_chars: int, turns: int) -> float:
    """
    Turns saved per character of injection the variant cost on average.

    The plan's formula is ``turns_potentially_saved / injected_chars_per_turn``
    and it is written out here rather than composed from the rounded report
    numbers, because a ratio of two rounded rates is a third rounding of a
    number nobody can reproduce.
    """
    per_turn = _ratio(injected_chars, turns)
    if per_turn <= 0.0:
        return 0.0
    return round(saved / per_turn, 4)


def _ratio(numerator: float, denominator: float) -> float:
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
