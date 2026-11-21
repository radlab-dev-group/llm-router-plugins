"""
The knowledge layer plugin: read, write and escalate, without ever breaking a request.

This is the module the pipeline actually loads.  It stands behind the routing
plugin in the utility pipeline and does three things with what routing left in
the payload — the work mode, the source, the thread and turn identifiers — and
what the adapter can still read out of the body itself:

* **read**: search the repository's knowledge base for what is already known
  about the situation, assemble it into one block and inject it;
* **write**: mine the turn for episodes worth remembering and hand them to the
  indexing worker across the stream;
* **escalate**: when the agent is looping with no known fix, switch the model
  to the one the operator named for the mode.

Every one of the three is optional and fail-open: a missing engine, a missing
stream, an unreadable body, each is a status on the annotation and a request
that proceeds, never an exception.  The only invariant is that a request the
plugin cannot serve comes back the very object it was given.
"""

import logging
import threading
import time

from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

from llm_router_plugins.plugin_interface import PluginInterface
from llm_router_plugins.utils.knowledge.agentic_knowledge.adapters.base import (
    AgentAdapter,
    AgentTurn,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.adapters.claude_code import (
    ClaudeCodeAgentAdapter,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.adapters.codex import (
    CodexAgentAdapter,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.assembler import (
    ContextAssembler,
    KnowledgeContext,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.metrics import (
    KnowledgeMetrics,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.cache import (
    CachedContext,
    FeedbackTracker,
    GenerationContextCache,
    InMemoryContextCache,
    LookupLimiter,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.redisclient import (
    LazyRedisClient,
    RedisClientHandle,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
    AgenticKnowledgeConfig,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.documents import (
    KIND_PLAYBOOK,
    KIND_RECIPE,
    KIND_TASK_SOLUTION,
    KIND_TOOL_RECIPE,
    MINED_KINDS,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.debug_capture import (
    capture_payload as _capture_payload,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.docs_ingestor import (
    DocsIndexer,
    ScanOutcome,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.escalation import (
    ACTION_ESCALATE,
    EscalationDecision,
    EscalationPolicy,
    REASON_LOOP_WITHOUT_RECIPE,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.learning import (
    CandidateFeatures,
    FeedbackSample,
    LearningPolicy,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.events import (
    KnowledgeEventPublisher,
    PublishStatus,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.miner import EpisodeMiner
from llm_router_plugins.utils.knowledge.agentic_knowledge.query import QueryBuilder
from llm_router_plugins.utils.knowledge.agentic_knowledge.scope import ScopeResolver
from llm_router_plugins.utils.knowledge.agentic_knowledge.sse import (
    REASON_MEMO,
    REASON_READ_BUDGET,
    KnowledgeStore,
    STATUS_DISABLED,
    STATUS_OK,
    STATUS_UNAVAILABLE,
)

#: Kinds of knowledge whose injection is worth scoring: they propose a way
#: forward, so what the agent did next says whether they helped.  A document
#: that only recalls (an archive, a dead end) is not credited either way.
_SCORED_KINDS = (KIND_RECIPE, KIND_PLAYBOOK, KIND_TASK_SOLUTION, KIND_TOOL_RECIPE)

#: How many sessions may wait one turn for their injection to be scored.
_FEEDBACK_PENDING_LIMIT = 512

__all__ = ["AgenticKnowledgePlugin"]

#: How often one repository is told that its read budget starved its searches.
_BUDGET_WARNING_INTERVAL_SECONDS = 300.0

#: How many repositories that note is remembered for, so a proxy serving many
#: repositories cannot grow the table without bound.
_BUDGET_WARNING_SCOPES = 512

#: Request classes the plugin does not enrich.
_REQUEST_CLASS_AUX = "aux_title"
_REQUEST_CLASS_COMPACTION = "compaction"


@dataclass(frozen=True)
class _InjectedDoc:
    """
    What one request was shown, kept until the next turn judges it.

    The name alone would be enough for the counters, which vote a whole block
    up or down; it is not enough for the learner, which credits the documents
    that could have caused what this turn did and leaves the rest unlabelled.
    So the three facts a verdict is read against — the failure the document is
    about, the files it names, the command it declares — are kept next to the
    features the ranking weighed, for one turn, in this process only.
    """

    document_name: str = ""
    signature_hash: str = ""
    paths: Tuple[str, ...] = ()
    test_command: str = ""
    features: Optional[CandidateFeatures] = None


class AgenticKnowledgePlugin(PluginInterface):
    """
    Enrich coding-agent requests with repository knowledge.

    Parameters
    ----------
    logger : logging.Logger, optional
        Where the plugin reports failures, without payload content.
    config : AgenticKnowledgeConfig, optional
        Resolved policy; loaded from the bundle and environment when omitted.
    store : KnowledgeStore, optional
        Pre-built engine facade; tests pass a fake one.
    cache : GenerationContextCache, optional
        Generation cache; an in-process one is used when Redis is unavailable.
    publisher : KnowledgeEventPublisher, optional
        Stream publisher; tests pass a fake one.
    adapters : Tuple[AgentAdapter, ...], optional
        Adapters, in resolution order; the configured ones are built when
        omitted.
    redis_client : Any, optional
        A Redis client the shared state should use instead of one built from
        the policy's connection settings; tests pass a fake one.  It goes
        through the same fork-aware handle the request path uses, so a proxy
        that preloads and then forks still never hands a worker its parent's
        connection pool.

    Raises
    ------
    ValueError
        If the policy is inconsistent: a weight outside ``(0, 1]``, a zero
        budget, a threshold no hit could ever reach. A bad policy is a
        configuration error, not a runtime condition, so it surfaces here at
        startup instead of being caught on every request and leaving the layer
        silently doing nothing.
    """

    name = "agentic_knowledge"

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        config: Optional[AgenticKnowledgeConfig] = None,
        store: Optional[KnowledgeStore] = None,
        cache: Optional[GenerationContextCache] = None,
        publisher: Optional[KnowledgeEventPublisher] = None,
        adapters: Optional[Tuple[AgentAdapter, ...]] = None,
        metrics: Optional[KnowledgeMetrics] = None,
        redis_client: Any = None,
    ) -> None:
        """Resolve the policy and wire the collaborators; open no connection."""
        super().__init__(logger=logger)
        self._config = config if config is not None else _load_config(self._logger)
        self._store = store
        self._cache = cache
        self._publisher = publisher
        self._adapters = adapters
        self._redis_client = redis_client
        self._miner: Optional[EpisodeMiner] = None
        self._query_builder: Optional[QueryBuilder] = None
        self._escalation: Optional[EscalationPolicy] = None
        self._limiter: Optional[LookupLimiter] = None
        self._docs: Optional[DocsIndexer] = None
        self._feedback_tracker: Optional[FeedbackTracker] = None
        self._redis: Optional[RedisClientHandle] = None
        self._learning: Optional[LearningPolicy] = None
        #: Per session: what the previous request of this session was shown —
        #: the document, the evidence it names and the features the learner saw
        #: — so the verdict of the next turn can be credited per document
        #: instead of per block.
        self._pending_injections: "OrderedDict[str, Tuple[_InjectedDoc, ...]]" = (
            OrderedDict()
        )
        self._metrics: Optional[KnowledgeMetrics] = metrics
        #: Per session: the knowledge documents injected on the previous
        #: request, and the failure signature the request was stuck on.
        self._pending_feedback: "OrderedDict[str, Tuple[Tuple[str, ...], str]]" = (
            OrderedDict()
        )
        #: When the last push actually reached the engine; ``None`` is "never".
        #: It cannot be ``0.0``: the origin of ``time.monotonic`` is
        #: unspecified, and on a freshly booted runner the clock reads less
        #: than the interval, so a zero sentinel would hold the first verdict
        #: of every boot back until the clock climbs past it.
        self._feedback_pushed_at: Optional[float] = None
        self._feedback_flush_lock = threading.Lock()
        self._feedback_thread: Optional[threading.Thread] = None
        self._budget_warnings: Dict[str, float] = {}
        self._wire(self._config)

    # -------------------------------------------------------------- wiring -
    def _wire(self, config: AgenticKnowledgeConfig) -> None:
        """
        Build what the policy asks for, opening no connection.

        The engine and Redis clients are held lazily by their own holders, so
        wiring costs nothing at startup and a plugin built before its process
        forks still connects with sockets of its own.  What is *not* deferred is
        the policy: reading it and validating it is the point at which a
        misconfigured deployment fails.
        """
        if self._store is None:
            self._store = KnowledgeStore(
                host=config.sse_host,
                api_prefix=config.sse_prefix,
                token=config.sse_token,
                username=config.sse_username,
                password=config.sse_password,
                timeout_seconds=config.sse_timeout_seconds,
                verify=config.sse_verify,
                min_similarity=config.min_similarity,
                hybrid_search=config.hybrid_search,
                rerank_results=config.rerank_results,
                fts_only_min_identifier_share=config.fts_only_min_identifier_share,
                ignore_language_detect=config.sse_ignore_language_detect,
                logger=self._logger,
            )
        if self._publisher is None:
            self._publisher = KnowledgeEventPublisher(
                connection=config.redis,
                stream_key=config.stream_key,
                key_prefix=config.key_prefix,
                dedup_ttl_seconds=config.dedup_ttl_seconds,
                max_stream_len=config.max_stream_len,
                logger=self._logger,
            )
        if self._cache is None:
            self._cache = _build_cache(config, self._logger)
        if self._adapters is None:
            self._adapters = _build_adapters(config, self._logger)
        self._miner = EpisodeMiner(
            max_patch_chars=config.max_patch_chars,
            archive_enabled=config.archive_enabled,
            archive_max_chunk_chars=config.archive_max_chunk_chars,
            archive_ttl_seconds=config.archive_ttl_seconds,
            test_mode=_test_mode(self._adapters),
            task_solutions_enabled=config.task_solutions_enabled,
            tool_recipes_enabled=config.tool_recipes_enabled,
        )
        self._query_builder = QueryBuilder(
            collection_namer=config.collection,
            min_similarity=config.min_similarity,
        )
        self._escalation = EscalationPolicy(config)
        self._limiter = LookupLimiter(
            max_per_generation=config.max_lookups_per_generation
        )
        # Built whether or not documentation is enabled: the indexer is the one
        # that says why it did nothing, and a plugin whose documentation path
        # quietly never existed is the failure this layer keeps avoiding.
        self._docs = DocsIndexer(
            config=config, publisher=self._publisher, logger=self._logger
        )
        # One fork-aware holder for every Redis the request path touches: the
        # votes, the reputation, the bandit, the training stream and the model
        # snapshots all live in the same place, and a proxy that preloads and
        # then forks must not hand a worker its parent's sockets.  It is built
        # even when Redis is not configured, so every reader has one thing to
        # ask and gets ``None`` back: a deployment without Redis keeps the
        # in-process state it always had, and one with Redis shares that state
        # across workers and across restarts.
        self._redis = RedisClientHandle(
            connection=config.redis,
            client=self._redis_client,
            logger=self._logger,
        )
        self._feedback_tracker = FeedbackTracker(
            client=LazyRedisClient(self._redis),
            key_prefix=config.key_prefix,
        )
        # The learned path.  Built whether or not it is enabled, for the same
        # reason the indexer is: the one that says *why* it did nothing is the
        # one that did nothing.  When disabled every multiplier is 1.0 and the
        # training log still records what a request was shown, because the
        # labels of today are the corpus of next month and they cannot be
        # recovered after the fact.
        self._learning = LearningPolicy(
            config=config.learning,
            client=LazyRedisClient(self._redis),
            logger=self._logger,
        )
        if self._metrics is None:
            # Built on the same connection the cache and the publisher use, so
            # a deployment without Redis still counts in-process: the numbers
            # describe a run rather than a history, which is what a deployment
            # without a place to keep history can honestly report.
            self._metrics = KnowledgeMetrics(
                connection=config.redis if config.redis.configured else None,
                key_prefix=config.key_prefix,
                logger=self._logger,
            )
        if self._logger is not None:
            # One line per process, at a level an operator actually reads.  The
            # layer fails open everywhere, so without it a plugin that loaded
            # and is not allowed to write looks exactly like one that never
            # loaded, and an empty knowledge base has no other way to say which
            # of the two it is.
            self._logger.info(
                "agentic knowledge wired: %s",
                _wiring_summary(config, self._adapters),
            )

    # ---------------------------------------------------------------- apply -
    def apply(self, payload: Any, **kwargs: Any) -> Any:
        """
        Enrich one agent request with the knowledge of its repository.

        Parameters
        ----------
        payload : Any
            The request payload.  Anything that is not a dict, or that no
            adapter claims, is returned unchanged, as the very same object.
        **kwargs : Any
            Accepted for interface compatibility; unused by this plugin.

        Returns
        -------
        Any
            The payload, enriched when the plugin served it.
        """
        if not isinstance(payload, dict):
            return payload
        config = self._config
        if config is None or not config.enabled:
            return payload

        adapter = self._matching_adapter(payload)
        if adapter is None:
            return payload
        _capture_payload(payload, self._logger)
        try:
            turn = adapter.parse(payload)
        except Exception as exc:
            self._warn("agentic knowledge parse failed: %s", type(exc).__name__)
            return payload
        if turn is None:
            return payload

        if turn.request_class == _REQUEST_CLASS_AUX:
            return payload

        self._debug(
            "agentic knowledge request: adapter=%s class=%s scope=%s mode=%s"
            " %s %s",
            adapter.name,
            turn.request_class or "-",
            turn.scope or "-",
            turn.mode or "-",
            _payload_shape(payload),
            _turn_shape(turn),
        )
        mining = self._mine(turn)
        publish = self._publish(mining, config, turn)
        docs = self._index_docs(turn, config)

        if turn.request_class == _REQUEST_CLASS_COMPACTION:
            # A window about to be summarized is written, never read from.
            self._annotate(
                payload,
                adapter,
                turn,
                None,
                EscalationDecision(),
                publish,
                docs,
                mining,
            )
            return payload
        return self._serve_read(
            adapter, turn, payload, config, mining, publish, docs
        )

    # ----------------------------------------------------------- read path -
    def _serve_read(
        self,
        adapter: AgentAdapter,
        turn: AgentTurn,
        payload: Dict[str, Any],
        config: AgenticKnowledgeConfig,
        mining: Any,
        publish: PublishStatus,
        docs: Optional[ScanOutcome] = None,
    ) -> Any:
        """Search, inject, decide escalation, and annotate — in one pass."""
        read_on = config.read_available
        policy = config.policy_for(turn.mode) if read_on else None
        cached = self._cached(turn)
        # The generation cache outlives the switch: an operator who turns the
        # read path off must stop getting knowledge injected, not keep serving
        # whatever a previous request cached for this generation.  The cache is
        # still read below, because escalation persistence lives in it and has
        # its own switch.
        fresh = (
            read_on and cached is not None and self._is_fresh(cached, turn, config)
        )
        cache_miss_reason = ""
        if not fresh and cached is not None:
            # The block a request is shown is one thing, and why the cache did
            # not serve it is another: a mode switch that reads the same
            # question is a hit, a mode that reads a different question is a
            # miss the operator can read as "the policy changed", and new
            # evidence is a miss that is the turn actually moving.
            cache_miss_reason = (
                "evidence"
                if _evidence_moved(cached.evidence_fingerprint, turn)
                else "mode"
            )
        # What the previous request of this session was shown is scored by
        # what happened since: this turn's mining either mined a fix (the
        # injected knowledge held) or repeats the failure it was shown how
        # to avoid (it did not).  Settled before this request assembles, so
        # the verdict already weighs what this request ranks.
        self._settle_feedback(turn, mining)
        self._flush_feedback(turn)
        candidate_features: Dict[str, CandidateFeatures] = {}
        if fresh:
            context = KnowledgeContext(
                hits=cached.context.hits,
                rendered=cached.context.rendered,
                chars=cached.context.chars,
                best_recipe_score=cached.context.best_recipe_score,
                deadends=cached.context.deadends,
                cache="hit",
                status=cached.context.status,
                doc_context=cached.context.doc_context,
            )
        else:
            context, candidate_features = self._assemble(
                turn, config, policy, mining, self._doc_context(turn, docs)
            )
            # Nothing cached for this generation is a cold read; a switch to a
            # mode that reads a different question is a policy change; new
            # evidence is the turn moving.
            if cache_miss_reason:
                context = _with_miss_reason(context, cache_miss_reason)
            elif context.cache == "miss":
                context = _with_miss_reason(context, "cold")

        if not context.empty:
            payload = adapter.inject(payload, context.rendered)

        recipe_trials = dict(cached.recipe_trials) if cached is not None else {}
        decision = (
            self._escalation.decide(
                turn,
                context,
                policy,
                mining.loop_count,
                mining.signature_hash,
                recipe_offered_at=recipe_trials.get(mining.signature_hash),
            )
            if self._escalation is not None
            else EscalationDecision()
        )

        # An explicit routing choice is never overridden, so the whole
        # escalation step — including the persisted-escalation carry-over below
        # — stands down for it.  Reading the persisted escalation before this
        # check used to record `escalate` in the annotation for a request whose
        # model was deliberately left alone.
        explicit = (turn.routing_source or "").lower() == "explicit"
        target = decision.model_name if decision.action == ACTION_ESCALATE else ""
        if not target and config.escalation_available and not explicit:
            # A loop escalated earlier in this generation stays escalated even
            # when this request's own evidence no longer repeats the loop.  Only
            # while escalation is on: the cache outlives a restart, so an
            # operator who switched the mechanism off would otherwise keep
            # handing out the escalation model until the entry ages out.
            target = self._persisted_escalation(cached, turn)
            if target:
                decision = EscalationDecision(
                    ACTION_ESCALATE,
                    REASON_LOOP_WITHOUT_RECIPE,
                    target,
                    decision.loop_count,
                    decision.signature_hash,
                )
        if target and not explicit:
            # The previous model is what the annotation reports as `from`; it
            # may be unknown (a payload that carries no model and no routed
            # one), which is not a reason to skip the switch — only a reason not
            # to claim which model was replaced.
            previous = str(payload.get("model") or turn.model)
            if previous != target:
                adapter.set_model(payload, target)
                if previous:
                    self._note_model_before_knowledge(payload, previous)

        if read_on:
            # Frozen, not overwritten, while the read path is off: the entry
            # this generation built under a live read path stays intact, so
            # switching reads back on resumes instead of finding an empty block
            # it would treat as fresh.
            if (
                self._escalation is not None
                and mining.loop_count > 0
                and self._escalation.has_strong_recipe(
                    context, mining.signature_hash
                )
            ):
                recipe_trials.setdefault(mining.signature_hash, mining.loop_count)
            self._remember_context(turn, context, target, recipe_trials, config)
        self._remember_injected(turn, context, mining, candidate_features)
        self._record_metrics(turn, context, decision)
        self._annotate(
            payload, adapter, turn, context, decision, publish, docs, mining
        )
        return payload

    def _record_metrics(
        self,
        turn: AgentTurn,
        context: KnowledgeContext,
        decision: EscalationDecision,
    ) -> None:
        """
        Count this request on its repository's scoreboard.

        One call per request that reached the read, so the denominators of the
        rates a deployment reads are requests rather than searches: the
        question an operator asks is "of the turns we showed knowledge to, how
        many got fixed", and a turn is the unit that question counts.

        It runs after the payload is already final — a counter is not allowed
        to be a reason a request is late or fails, so a recorder that raises is
        swallowed here as well as inside the recorder itself.
        """
        if self._metrics is None or not turn.scope:
            return
        # A loop is what the layer itself calls one: the same failure repeated
        # as many times as the threshold says.  Counting every turn that met a
        # failure as a loop would report a single stumble as the thing the
        # layer exists to end, so the two are counted apart.
        loop_threshold = getattr(self._config, "loop_threshold", 1)
        try:
            self._metrics.record(
                turn.scope,
                turns=1,
                injected_turns=1 if context.hits else 0,
                injected_docs=len(context.hits),
                failure_turns=1 if decision.loop_count > 0 else 0,
                loops=1 if decision.loop_count >= loop_threshold else 0,
                escalations=1 if decision.action == ACTION_ESCALATE else 0,
            )
        except Exception:  # a scoreboard is never a reason to fail a request
            pass

    def _assemble(
        self,
        turn: AgentTurn,
        config: AgenticKnowledgeConfig,
        policy: Any,
        mining: Any,
        doc_context: Any = None,
    ) -> Tuple[KnowledgeContext, Dict[str, CandidateFeatures]]:
        """
        Search the sources and assemble the block; fail-open to an empty one.

        The second value is what the learner saw per document, so the caller can
        note the exposure with the same numbers the ranking weighed — computing
        them a second time would be a second answer to the same question.
        """
        if policy is None:
            return (
                KnowledgeContext(
                    cache="disabled", status="disabled", doc_context=doc_context
                ),
                {},
            )
        if not turn.scope:
            return KnowledgeContext(cache="miss", status="disabled"), {}
        generation = turn.cache_generation
        if self._limiter is not None and not self._limiter.allow(
            turn.scope, generation
        ):
            return KnowledgeContext(cache="miss", status="ok"), {}

        results = []
        statuses = set()
        # One budget for the whole read, not one per source: a mode asks one
        # question per collection and they are asked one after another, so
        # without a shared ceiling a request pays the transport timeout as
        # many times as the mode has sources.  The store starts a search only
        # while a whole call fits before this moment, which is what makes
        # `read_timeout_seconds` the number a request actually waits.
        deadline = time.monotonic() + self._config.read_timeout_seconds
        # The miner read the failure out of the turn on the writer's own
        # normalization; the query asks with it, so the search is a question
        # about the failure the agent is looking at and not a paraphrase of
        # the user's sentence.
        builder = _query_builder_with_failure(
            self._query_builder,
            getattr(mining, "signature", "") or "",
        )
        requests = builder.build(turn, policy)
        # A scope with nothing mined has nothing for the knowledge bucket to
        # return, and the search of it is pure latency on the way to an agent.
        # The counter is the worker's; unknown is not zero, and an unknown
        # counter costs the search it always cost.
        requests = self._skip_cold_knowledge(turn, requests)
        # The requests of one mode are independent: they ask different
        # collections, with different filters, the same question.  Asked one
        # after another a slow engine makes the request pay the transport
        # timeout as many times as the mode has sources; asked together they
        # pay it once.  The budget is the same, and the store still refuses a
        # call that cannot fit before the deadline.
        results = list(
            self._search_requests_in_parallel(
                requests,
                deadline,
            )
        )
        results = self._widen_empty_scoped_searches(requests, results, deadline)
        skipped = sum(
            1 for result, _ in results if result.reason == REASON_READ_BUDGET
        )
        memo_hits = sum(1 for result, _ in results if result.reason == REASON_MEMO)
        for result, _ in results:
            statuses.add(result.status)
        if self._metrics is not None and turn.scope:
            # The failures are counted here rather than where they are logged,
            # because a search that returned nothing and a search that could
            # not be reached are the same empty answer to the assembler and
            # very different facts to an operator reading the scoreboard: this
            # is the one place both are in hand, statuses and all.
            unavailable = sum(
                1 for result, _ in results if result.status == STATUS_UNAVAILABLE
            )
            counts = {}
            if unavailable:
                counts["read_failures"] = unavailable
            if memo_hits:
                # The number that says the memo earned its staleness window:
                # engine calls this process did not make.  It is read against
                # `searches_per_request`, which counts the question whether or
                # not the engine was asked it.
                counts["memo_hits"] = memo_hits
            if counts:
                try:
                    self._metrics.record(turn.scope, **counts)
                except Exception:  # a scoreboard is never a reason to fail a read
                    pass
        if skipped:
            # The reads the operator will otherwise conclude never happened.
            # Both numbers go in the line because the usual cause is not a
            # small budget: it is a per-call timeout the budget cannot hold
            # more than one of, which looks like an idle engine and an empty
            # knowledge base at once.  Rate-limited per repository, because a
            # starved configuration starves every request of every session,
            # and a warning that prints on every request is a warning nobody
            # reads.
            self._warn_budget(turn.scope, skipped, len(results))
        status = _status_of(statuses)

        assembler = ContextAssembler(
            signature_hash=mining.signature_hash,
            signature=getattr(mining, "signature", "") or "",
            touched_paths=turn.touched_paths,
            local_deadends=mining.tried,
            feedback=self._feedback_weights(),
            learning=self._learning,
            scope=turn.scope,
            failing_command=getattr(mining, "failing_command", "") or "",
            budget_chars=getattr(policy, "max_chars", 0) or 0,
        )
        context = assembler.assemble(results, policy, doc_context=doc_context)
        # The exposure this request made, with the features the learner saw:
        # logged before the verdict exists, so a turn that never proves anything
        # still leaves a record of what it was shown — the record the credit
        # assignment of the next turn needs and that cannot be written later.
        self._log_exposure(turn, assembler, context)
        return KnowledgeContext(
            hits=context.hits,
            rendered=context.rendered,
            chars=context.chars,
            best_recipe_score=context.best_recipe_score,
            deadends=context.deadends,
            cache="miss",
            status=status,
            doc_context=context.doc_context,
            searches=len(results),
            budget_skipped=skipped,
            memo_hits=memo_hits,
            cache_miss_reason="cold",
        ), dict(getattr(assembler, "candidate_features", None) or {})

    def _skip_cold_knowledge(self, turn: AgentTurn, requests) -> tuple:
        """
        Drop the mined-knowledge search of a scope that has none.

        The reader pays a hybrid search for a bucket a repository that has
        never been mined cannot answer.  The worker counts the mined documents
        it writes per scope, and a scope whose count is known to be zero skips
        that one search; documentation and archived windows are asked for as
        always, because they are written by other paths and on other clocks.
        An unknown count is not a zero: a Redis that will not answer, or a
        deployment predating the counter, keeps searching.
        """
        if not requests or not turn.scope:
            return requests
        if self._publisher is None:
            return requests
        try:
            count = self._publisher.mined_count(turn.scope)
        except Exception:  # a counter that cannot be read is not a cold scope
            return requests
        if count != 0:
            return requests
        remaining = tuple(
            request for request in requests if not _is_mined_only(request)
        )
        return remaining if remaining else ()

    def _search_requests_in_parallel(
        self,
        requests: Tuple[Any, ...],
        deadline: float,
    ):
        """
        Search every request at once, keeping the shared read budget.

        A short-lived pool, one worker per request: the work is bounded by
        the mode's own source list, and the pool is torn down before the
        method returns, so a slow engine cannot hold a thread past the
        request.  One request that fails or overruns the budget answers with
        the same shape the serial loop used to produce — the other requests'
        results are not lost with it.
        """
        if not requests:
            return
        from concurrent.futures import ThreadPoolExecutor

        def one(request):
            try:
                result = self._store.search(request, deadline=deadline)
            except Exception as exc:
                self._warn("knowledge search failed: %s", type(exc).__name__)
                from llm_router_plugins.utils.knowledge.agentic_knowledge.sse import (
                    SearchResult as _SR,
                )

                result = _SR(STATUS_UNAVAILABLE, reason="search")
            return (result, request.categories)

        with ThreadPoolExecutor(max_workers=len(requests)) as pool:
            for outcome in pool.map(one, requests):
                yield outcome

    # ---------------------------------------------------------- cache path -
    def _cached(self, turn: AgentTurn) -> Optional[CachedContext]:
        """Return the cached entry of this generation, fresh or not."""
        cache = self._cache
        if cache is None or not turn.scope or not turn.cache_generation:
            return None
        return cache.get(turn.scope, turn.cache_generation)

    @staticmethod
    def _is_fresh(cached: CachedContext, turn: AgentTurn, config: Any) -> bool:
        """Whether the cached block still describes this turn."""
        if _evidence_moved(cached.evidence_fingerprint, turn):
            # New evidence is new facts: the block was built against a
            # different turn than the one in front of it, whatever the mode.
            return False
        if cached.mode == turn.mode:
            return True
        # The mode switched, but a mode is a reading of the same facts:
        # two modes that read the same sources on the same budget ask the
        # same question of the knowledge base, and a switch between them
        # changes the label of the answer, not the answer.  The block may
        # keep being served.  A mode whose policy reads anything different
        # is a different question, and the cache is a miss for it.
        policy = config.policy_for(turn.mode) if config is not None else None
        if policy is None or not cached.policy_signature:
            return False
        return cached.policy_signature == policy.policy_signature()

    @staticmethod
    def _persisted_escalation(
        cached: Optional[CachedContext], turn: AgentTurn
    ) -> str:
        """The model an earlier request of this generation escalated to."""
        if cached is None or cached.mode != turn.mode:
            return ""
        return cached.escalated_model or ""

    def _widen_empty_scoped_searches(
        self, requests: Tuple[Any, ...], results: list, deadline: float
    ) -> list:
        """
        Retry a path-scoped search that answered nothing, without its scope.

        The directory of the touched paths is the narrowest question worth
        asking first, and an empty answer to it says nothing about the rest
        of the repository: one unscoped retry keeps the narrow first look
        free of false negatives, at the price of a second call only for the
        searches that came back empty.  The store refuses a call that no
        longer fits the read budget, so the widening never overspends it.
        """
        if not requests:
            return results
        widened = list(results)
        for index, request in enumerate(requests):
            if not getattr(request, "relative_path_contains", ""):
                continue
            result = widened[index][0]
            if result.hits or result.status != STATUS_OK:
                continue
            try:
                retry = self._store.search(
                    replace(request, relative_path_contains=""),
                    deadline=deadline,
                )
            except Exception as exc:
                self._warn("knowledge search failed: %s", type(exc).__name__)
                continue
            widened[index] = (retry, request.categories)
        return widened

    def _flush_feedback(self, turn: AgentTurn) -> None:
        """
        Start a push of the settled verdicts, without waiting for it.

        The verdicts of one process are of use to nobody but the engine, and
        the push is one HTTP call per document: run here, it would make one
        request in every interval wait seconds for work that has nothing to do
        with answering it.  So the request path only decides *whether* to push
        — the interval has passed, there is something queued, this turn knows
        a collection — and a daemon thread carries the calls.  One flush at a
        time: a push still running is a push that will finish and drain the
        queue, not a reason to start a second one behind it.
        """
        tracker = self._feedback_tracker
        if tracker is None or self._store is None or not turn.scope:
            return
        if not tracker.touched():
            return
        interval = int(
            getattr(self._config, "feedback_push_interval_seconds", 0) or 0
        )
        if (
            self._feedback_pushed_at is not None
            and time.monotonic() - self._feedback_pushed_at < interval
        ):
            return
        if not self._feedback_flush_lock.acquire(blocking=False):
            return
        try:
            self._feedback_thread = threading.Thread(
                target=self._push_feedback,
                args=(tracker, turn.scope),
                name="agentic-kb-feedback",
                daemon=True,
            )
            self._feedback_thread.start()
        except Exception:  # a thread that will not start is a push not made
            self._feedback_flush_lock.release()

    def _push_feedback(self, tracker: Any, scope: str) -> None:
        """
        Push the queued verdicts; runs off the request path, never raises.

        The engine's counters are monotonic and the endpoint takes absolute
        totals, so a name is sent as the deployment has counted it and a
        refused name simply stays queued for the next flush.  The interval
        clock moves only when something actually reached the engine, so a
        deployment whose endpoint is down costs one thread per interval and
        no verdicts.
        """
        pushed: List[str] = []
        try:
            for name in tracker.touched():
                collection = tracker.collection_of(name)
                if not collection:
                    continue
                totals = tracker.get_feedback(name)
                up = int(totals.get("upvotes", 0))
                down = int(totals.get("downvotes", 0))
                if not (up or down):
                    continue
                try:
                    if self._store.push_quality(collection, name, up, down):
                        pushed.append(name)
                except Exception:  # one document is not the flush's end
                    continue
            if pushed:
                tracker.pushed(*pushed)
                self._feedback_pushed_at = time.monotonic()
        except Exception:
            pass
        finally:
            self._feedback_flush_lock.release()

    def _feedback_weights(self) -> Optional[Dict[str, Dict[str, int]]]:
        """Reinforcement recorded so far; a debunked recipe ranks lower."""
        if self._feedback_tracker is None:
            return None
        try:
            return self._feedback_tracker.all_feedback() or None
        except Exception:
            return None

    def _settle_feedback(self, turn: AgentTurn, mining: Any) -> None:
        """
        Score the knowledge injected on the previous request of this session.

        The turn carries what happened since the injection: a recipe or a
        playbook mined out of it means a failure was verified fixed with the
        injected guidance on screen; the very same failure signature coming
        back means the loop survived it.  A turn that proves neither — the
        agent simply moved on — neither credits nor blames, and the pending
        entry is spent either way: one turn is the whole window of
        attribution the plugin claims.

        The coarse verdict above is one bit for the whole block, and the block
        is several documents: crediting all of them with a fix one of them
        describes teaches the layer to inject more of everything.  So each
        document is graded against the evidence of *this* turn — its signature,
        the files it names, the command it declares — and the graded label is
        what the learner folds in.  The counters keep the coarse bit, because
        the engine's quality signal is a vote and a vote has two values.
        """
        tracker = self._feedback_tracker
        if tracker is None:
            return
        pending = self._pending_feedback.pop(turn.session_id, None)
        injected = self._pending_injections.pop(turn.session_id, ())
        if not pending:
            return
        doc_names, previous_signature = pending
        repeated = bool(previous_signature) and (
            mining.signature_hash == previous_signature
        )
        resolved = any(
            event.kind in (KIND_RECIPE, KIND_PLAYBOOK) for event in mining.events
        )
        if not doc_names or not (repeated or resolved):
            return
        for doc_name in doc_names:
            tracker.record_feedback(
                doc_name,
                success=resolved,
                collection=tracker.collection_of(doc_name),
            )
        self._learn_from_verdict(
            turn,
            mining,
            injected,
            resolved=resolved,
            repeated=repeated,
            previous_signature=previous_signature,
        )
        if self._metrics is not None and turn.scope:
            # The turn's own verdict on what it was shown last request, counted
            # once for the request rather than once per document: adoption is a
            # property of a turn, the per-document tally the tracker keeps is a
            # property of a document, and the two answer different questions.
            try:
                counts = {
                    "feedback_up": 1 if resolved else 0,
                    "feedback_down": 0 if resolved else 1,
                    "green_after_injection": 1 if resolved else 0,
                }
                self._metrics.record(turn.scope, **counts)
            except Exception:
                pass

    def _learn_from_verdict(
        self,
        turn: AgentTurn,
        mining: Any,
        injected: Tuple["_InjectedDoc", ...],
        resolved: bool,
        repeated: bool,
        previous_signature: str = "",
    ) -> None:
        """
        Fold this turn's verdict into the learner, one document at a time.

        The label is a reading of the evidence this turn produced, per document:

        * ``+1.0`` — the document was about the very failure that has now been
          verified fixed, so it is credited with the fix it describes;
        * ``+0.5`` — the turn was resolved and the document speaks about a file
          the turn touched or declares the command that just went green: it
          plausibly held, without the evidence proving it;
        * ``0.0`` — the turn was resolved and this document has nothing in
          common with what the turn did.  That is not a failure and must not be
          trained as one, so it is logged as an exposure and fitted as nothing;
        * ``-1.0`` — the same failure came back and this document was about it:
          the advice was on screen and the loop survived it;
        * ``-0.3`` — the failure came back and this document is unrelated to
          it.  Weak evidence against a document that was merely along for the
          block, so it weighs a third of a judgement.

        A document that names neither signature nor file of its own — the
        common case for a chunk the engine matched by text — falls back to the
        block's verdict, marked ``coarse``: the credit is collective, and the
        model is told so, rather than the exposure being dropped from the
        corpus the next fit trains on.  Never raises: a learner that cannot be
        fed is a learner that did not learn, not a request that failed.
        """
        policy = self._learning
        if policy is None or not injected:
            return
        signature = getattr(mining, "signature_hash", "") or ""
        failing_command = str(getattr(mining, "failing_command", "") or "").strip()
        touched = {str(path) for path in turn.touched_paths or () if path}
        touched_names = {path.rsplit("/", 1)[-1] for path in touched}
        for record in injected:
            if not record.document_name:
                continue
            # The document's own account of which failure it is about.  A mined
            # recipe of a failure keeps that signature in its metadata; a hit
            # that carries neither a signature nor a file cannot be matched to
            # a verdict by anything more than the block it was in, and is
            # settled as the block was — marked ``coarse`` so a fit can tell
            # the two kinds of credit apart.
            about_previous = bool(
                record.signature_hash
                and previous_signature
                and record.signature_hash == previous_signature
            )
            about_current = bool(
                record.signature_hash
                and signature
                and record.signature_hash == signature
            )
            paths = {str(path) for path in record.paths if path}
            shares_path = bool(
                paths
                and (
                    paths & touched
                    or {path.rsplit("/", 1)[-1] for path in paths} & touched_names
                )
            )
            command_matches = bool(
                record.test_command
                and failing_command
                and (
                    record.test_command.strip() == failing_command
                    or record.test_command.strip().split()[-1]
                    == failing_command.split()[-1]
                )
            )
            if about_previous or about_current or shares_path or command_matches:
                if resolved:
                    if about_current:
                        label, kind = 1.0, "confirmed"
                    else:
                        label, kind = 0.5, "plausible"
                elif repeated and about_previous:
                    label, kind = -1.0, "debunked"
                elif repeated:
                    label, kind = -0.3, "rejected"
                else:
                    continue
            else:
                if not (resolved or repeated):
                    continue
                # The block's verdict, unattributed to anything in this
                # document's own header: a collective credit, and a fit that
                # sees it can weigh it accordingly.
                label, kind = (1.0 if resolved else -1.0), "coarse"
            try:
                policy.settle(
                    FeedbackSample(
                        scope=turn.scope,
                        session_id=turn.session_id,
                        generation=turn.cache_generation,
                        document_name=record.document_name,
                        features=record.features or CandidateFeatures(),
                        label=label,
                        label_kind=kind,
                    )
                )
            except Exception:  # a learner that will not eat is not a failed turn
                continue

    def _log_exposure(
        self, turn: AgentTurn, assembler: Any, context: KnowledgeContext
    ) -> None:
        """
        Record what this request was shown, before it has a verdict.

        Kept off the annotated path and never allowed to raise: the record is
        worth having and a training log is never worth a request.  Written even
        when learning is switched off, because a label can only be attached to
        an exposure that was written down while it happened.
        """
        policy = self._learning
        if policy is None or not turn.session_id:
            return
        features = getattr(assembler, "candidate_features", None) or {}
        if not features:
            return
        shown = [
            (
                hit.document_name,
                features[hit.document_name],
                bool(getattr(hit, "learning", {}) and (hit.learning or {}).get("explore")),
            )
            for hit in context.hits
            if hit.document_name and hit.document_name in features
        ]
        if not shown:
            return
        try:
            policy.log.append_injection(
                scope=turn.scope,
                session_id=turn.session_id,
                generation=turn.cache_generation,
                model_id=str(policy.annotation(turn.scope).get("model_id") or ""),
                features=[item for _, item, _ in shown],
                document_names=[name for name, _, _ in shown],
                explorations=["explore" if e else "greedy" for _, _, e in shown],
            )
        except Exception:  # an unwritten log line is a lost sample, not a failure
            return

    def _remember_injected(
        self,
        turn: AgentTurn,
        context: KnowledgeContext,
        mining: Any,
        candidate_features: Optional[Dict[str, CandidateFeatures]] = None,
    ) -> None:
        """Remember what this request injected so the next one can score it."""
        if self._feedback_tracker is None or not turn.session_id:
            return
        doc_names = tuple(
            hit.document_name
            for hit in context.hits
            if hit.document_name and hit.category in _SCORED_KINDS
        )
        if not doc_names:
            return
        features = candidate_features or {}
        recorded = tuple(
            _InjectedDoc(
                document_name=hit.document_name,
                signature_hash=hit.signature_hash,
                paths=tuple(hit.paths or ()),
                test_command=hit.test_command,
                features=features.get(hit.document_name),
            )
            for hit in context.hits
            if hit.document_name and hit.category in _SCORED_KINDS
        )
        self._pending_injections[turn.session_id] = recorded
        while len(self._pending_injections) > _FEEDBACK_PENDING_LIMIT:
            self._pending_injections.popitem(last=False)
        tracker = self._feedback_tracker
        for hit in context.hits:
            if hit.document_name and hit.category in _SCORED_KINDS:
                # The collection is the config's answer, not the hit's: a hit
                # carries no collection because the search that produced it was
                # built from this same name, so re-deriving it is exact and
                # costs the assembled block no new field.
                tracker.remember_collection(
                    hit.document_name,
                    self._config.collection(turn.scope, hit.category),
                )
        self._pending_feedback[turn.session_id] = (doc_names, mining.signature_hash)
        while len(self._pending_feedback) > _FEEDBACK_PENDING_LIMIT:
            self._pending_feedback.popitem(last=False)

    def _remember_context(
        self,
        turn: AgentTurn,
        context: KnowledgeContext,
        escalated_model: str,
        recipe_trials: Optional[Dict[str, int]] = None,
        config: Any = None,
    ) -> None:
        """Remember the block for the rest of the generation, if caching is on."""
        cache = self._cache
        if cache is None or not turn.scope or not turn.cache_generation:
            return
        cache.put(
            turn.scope,
            turn.cache_generation,
            CachedContext(
                context=context,
                evidence_fingerprint=_evidence_fingerprint(turn),
                mode=turn.mode,
                policy_signature=(
                    config.policy_for(turn.mode).policy_signature()
                    if config is not None
                    and config.policy_for(turn.mode) is not None
                    else ""
                ),
                escalated=bool(escalated_model),
                escalated_model=escalated_model,
                recipe_trials=dict(recipe_trials or {}),
                injected_doc_ids=tuple(
                    hit.document_name for hit in context.hits if hit.document_name
                ),
            ),
        )

    # ---------------------------------------------------------- write path -
    def _mine(self, turn: AgentTurn) -> Any:
        """Mine the turn once, for the read assembly and the write together."""
        from llm_router_plugins.utils.knowledge.agentic_knowledge.miner import (
            MiningResult,
        )

        if self._miner is None:
            return MiningResult()
        try:
            result = self._miner.mine(turn)
        except Exception as exc:
            self._warn("knowledge mining failed: %s", type(exc).__name__)
            return MiningResult()
        self._debug(
            "agentic knowledge mined: events=%d kinds=%s loop=%d",
            len(result.events),
            ",".join(sorted({str(event.kind) for event in result.events})) or "-",
            result.loop_count,
        )
        return result

    def _publish(
        self, mining: Any, config: AgenticKnowledgeConfig, turn: AgentTurn
    ) -> PublishStatus:
        """
        Hand mined episodes to the worker; never blocks or raises on Redis.

        A request whose repository is unknown publishes nothing.  Its documents
        would have no repository's collection to live in — the reader asks for
        a scope's collection and never looks at the bare prefix — and the
        conversations of every other scope-less checkout would meet there,
        which is the isolation the scope exists to prevent.
        """
        events = getattr(mining, "events", ())
        if not events or not config.write_available or self._publisher is None:
            if events:
                self._debug(
                    "agentic knowledge write skipped: %s",
                    (
                        "write disabled"
                        if not config.write_enabled
                        else "no Redis configured"
                    ),
                )
            return PublishStatus()
        if not turn.scope:
            self._debug(
                "agentic knowledge write skipped: repository scope unknown"
                " (%d event(s) dropped)",
                len(events),
            )
            return PublishStatus(STATUS_DISABLED, reason="no scope")
        try:
            return self._publisher.publish(events)
        except Exception as exc:
            self._warn("knowledge publish failed: %s", type(exc).__name__)
            return PublishStatus(STATUS_UNAVAILABLE, reason="publish")

    # -------------------------------------------------------------- helpers -
    def _matching_adapter(self, payload: Dict[str, Any]) -> Optional[AgentAdapter]:
        """Return the first adapter that claims the payload, if any."""
        if not self._adapters:
            return None
        for adapter in self._adapters:
            try:
                if adapter.matches(payload):
                    return adapter
            except Exception:
                continue
        return None

    def _note_model_before_knowledge(
        self, payload: Dict[str, Any], previous: str
    ) -> None:
        """Record the routed model the knowledge layer is replacing."""
        routing = payload.setdefault("routing", {})
        if isinstance(routing, dict):
            routing.setdefault("model_before_knowledge", previous)

    def _annotate(
        self,
        payload: Dict[str, Any],
        adapter: AgentAdapter,
        turn: AgentTurn,
        context: Optional[KnowledgeContext],
        decision: EscalationDecision,
        publish: Optional[PublishStatus],
        docs: Optional[ScanOutcome] = None,
        mining: Any = None,
    ) -> None:
        """Write the knowledge layer's decision into ``payload["routing"]``."""
        routing = payload.get("routing")
        if routing is None:
            routing = {}
            payload["routing"] = routing
        elif not isinstance(routing, dict):
            # Something else owns this key.  Replacing it to leave a note would
            # destroy that owner's data, and the note is diagnostics, not data.
            return
        knowledge: Dict[str, Any] = {
            "plugin": self.name,
            "adapter": adapter.name,
            "scope": turn.scope,
            "injected": len(context.hits) if context else 0,
            "chars": context.chars if context else 0,
            "sources": _source_counts(context),
            "cache": context.cache if context else "disabled",
            "loop_count": decision.loop_count,
        }
        # The numbers a deployment needs to read the layer without its
        # contents: how many searches the read made, how many the budget
        # could not start, why the cache did not serve, what the turn mined,
        # and how much of the write the stream already knew.
        metrics: Dict[str, Any] = {
            "searches_per_request": int(
                getattr(context, "searches", 0) if context else 0
            ),
            "read_budget_skips": int(
                getattr(context, "budget_skipped", 0) if context else 0
            ),
            "memo_hits": int(getattr(context, "memo_hits", 0) if context else 0),
            "cache_miss_reason": (
                context.cache_miss_reason if context is not None else ""
            ),
            "mined_kinds": (
                sorted({str(event.kind) for event in getattr(mining, "events", ())})
                if mining is not None
                else []
            ),
            "docs_publish_dropped_as_duplicate": int(
                publish.deduplicated if publish is not None else 0
            ),
        }
        knowledge["metrics"] = metrics
        if decision.action != "none":
            knowledge["escalation"] = {
                "action": decision.action,
                "reason": decision.reason,
                "from": turn.model,
                "to": decision.model_name,
            }
        if publish is not None:
            knowledge["write"] = {
                "published": publish.published,
                "deduplicated": publish.deduplicated,
                "errors": publish.errors,
                "status": publish.status,
            }
        if docs is not None:
            knowledge["docs"] = docs.as_dict()
        if self._learning is not None:
            # Why a recipe was first must be answerable from the response, not
            # from a log nobody kept: the model that served this request, its
            # evidence, the range its multipliers may move a vote through, and
            # — per candidate — the multiplier and the reasons for it, so a
            # recipe that ranked above a stronger one can be explained from
            # the request that served it, without a log nobody kept.
            learning_view = self._learning.annotation(turn.scope)
            if context is not None:
                per_candidate = {
                    hit.document_name: dict(getattr(hit, "learning", {}) or {})
                    for hit in context.hits
                    if hit.document_name and getattr(hit, "learning", None)
                }
                if per_candidate:
                    learning_view["candidates"] = per_candidate
            knowledge["learning"] = learning_view
            votes = self._feedback_tracker.all_feedback() if (
                self._feedback_tracker
            ) else {}
            knowledge["metrics"]["feedback"] = {
                "docs": len(votes),
                "up": sum(int(v.get("upvotes", 0) or 0) for v in votes.values()),
                "down": sum(int(v.get("downvotes", 0) or 0) for v in votes.values()),
                "source": "redis" if self._redis is not None else "memory",
            }
        if context is not None and getattr(context, "doc_context", None):
            digest = context.doc_context
            knowledge["doc_context"] = {
                "level": "digest",
                "chars": digest.chars,
                "source_files": list(digest.source_files)[:8],
            }
        routing["knowledge"] = knowledge

    def _index_docs(
        self, turn: AgentTurn, config: AgenticKnowledgeConfig
    ) -> Optional[ScanOutcome]:
        """
        Offer the repository's documentation to the knowledge base.

        Returns ``None`` — no annotation at all — when documentation is not part
        of this deployment, or when the request named no repository: reading a
        filesystem is only worth doing when its result has a collection to live
        in.  The indexer itself is cooldown-bound and never raises; this is the
        belt for the case where it does.
        """
        if not config.docs_available or self._docs is None:
            return None
        if not turn.scope or not turn.repo_root:
            return None
        try:
            return self._docs.index(
                turn.scope,
                turn.repo_root,
                agent=turn.agent,
                model=turn.model,
                generation=turn.cache_generation,
            )
        except Exception as exc:
            self._warn("knowledge docs scan failed: %s", type(exc).__name__)
            return ScanOutcome(status=STATUS_UNAVAILABLE, reason="scan")

    def _doc_context(self, turn: AgentTurn, docs: Optional[ScanOutcome]) -> Any:
        """
        Return the standing documentation digest for this request.

        The scan that just ran carries the digest it refreshed; every other
        request of the session reads the same map from the store, so a long
        session sees one stable digest of what the repository has written down
        rather than re-walking it per command.  ``None`` — no standing context —
        when documentation is off, no repository is known, or nothing has been
        scanned yet.  Never raises: this is context, not a dependency.
        """
        if not self._config or not self._config.doc_context_enabled:
            return None
        if not self._config.docs_available or self._docs is None:
            return None
        if not turn.scope:
            return None
        if docs is not None and docs.context is not None:
            return docs.context
        try:
            return self._docs.context(turn.scope)
        except Exception as exc:
            self._warn("knowledge doc context read failed: %s", type(exc).__name__)
            return None

    def _warn_budget(self, scope: str, skipped: int, total: int) -> None:
        """
        Report searches the read budget could not start, once per interval.

        The first line of a starved deployment is the only one that carries
        information; the thousandth is noise that hides the rest of the log.
        """
        logger = self._logger
        if logger is None:
            return
        now = time.monotonic()
        key = scope or "?"
        last = self._budget_warnings.get(key)
        if last is not None and now - last < _BUDGET_WARNING_INTERVAL_SECONDS:
            return
        if len(self._budget_warnings) > _BUDGET_WARNING_SCOPES:
            self._budget_warnings.clear()
        self._budget_warnings[key] = now
        logger.warning(
            "knowledge read budget of %.1fs spent before %d of %d search(es) of"
            " %s could start (one call is allowed %.1fs): raise"
            " READ_TIMEOUT_SECONDS above SSE_API_TIMEOUT, or lower"
            " SSE_API_TIMEOUT — a budget equal to one call leaves room for no"
            " search at all",
            self._config.read_timeout_seconds,
            skipped,
            total,
            key,
            self._config.sse_timeout_seconds,
        )

    def _warn(self, message: str, *args: Any) -> None:
        """Report one failure when a logger was given."""
        if self._logger is not None:
            self._logger.warning(message, *args)

    def _debug(self, message: str, *args: Any) -> None:
        """Report one request-path detail when debug logging is on."""
        if self._logger is not None:
            self._logger.debug(message, *args)


def _status_of(statuses: set) -> str:
    """Collapse per-source search statuses into one retrieval status."""
    if STATUS_OK in statuses:
        return STATUS_OK
    if STATUS_UNAVAILABLE in statuses:
        return STATUS_UNAVAILABLE
    return "disabled"


def _payload_shape(payload: Any) -> str:
    """
    Count the wire items of one request, naming no content.

    The adapter works on a normalized view; when that view is empty the
    question is what the request carried at all.  Item types answer it: they
    show whether a tool call arrived as ``function_call``, as
    ``custom_tool_call``, or under a type no parser here knows about.
    """
    items = payload.get("input") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return "items=-"
    shapes: Dict[str, int] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type") or ("message" if item.get("role") else "?"))
        name = item.get("name")
        if name:
            kind = f"{kind}:{name}"
        shapes[kind] = shapes.get(kind, 0) + 1
    return "items=" + (
        ",".join(f"{kind}:{count}" for kind, count in sorted(shapes.items())) or "0"
    )


def _turn_shape(turn: Any) -> str:
    """
    Describe what the adapter saw, naming no argument and no output text.

    A recipe needs three things in *one* request: a failed test, a patch after
    it, and a green run after the patch.  When nothing is ever mined, the
    question is always which of those three the request did not carry — and
    `activity` plus the per-tool counts answer that, because a Codex request
    replays the whole turn and therefore should show every tool call of it.
    """
    calls: Dict[str, int] = {}
    for item in getattr(turn, "activity", ()) or ():
        if getattr(item, "kind", "") == "function_call":
            name = getattr(item, "name", "") or "?"
            calls[name] = calls.get(name, 0) + 1
    tools = (
        ",".join(f"{name}:{count}" for name, count in sorted(calls.items())) or "-"
    )
    states = []
    for evidence in list(getattr(turn, "evidence", ()) or ())[-8:]:
        outcome = getattr(evidence, "succeeded", None)
        states.append(
            f"{getattr(evidence, 'kind', '?')}"
            f":{getattr(evidence, 'mode', '') or '-'}"
            f":{'ok' if outcome else 'fail' if outcome is False else '-'}"
        )
    return (
        f"activity={len(getattr(turn, 'activity', ()) or ())}"
        f" tools=[{tools}]"
        f" evidence={len(states)}"
        f"[{'+'.join(states) or '-'}]"
    )


def _on_off(value: bool) -> str:
    """Render a switch the way the wiring line reads."""
    return "on" if value else "off"


def _wiring_summary(
    config: AgenticKnowledgeConfig, adapters: Optional[Tuple[AgentAdapter, ...]]
) -> str:
    """
    Describe one process's wiring in a line, holding no host and no secret.

    Answers the two questions an empty knowledge base always raises: is this
    layer alive at all, and is it allowed to write.  The phase rules are called
    out by name because they decide whether anything is mined at all — with
    them unread, every other switch can be on and the collections stay empty
    forever, in silence.
    """
    rules_on = any(
        getattr(adapter, "phase_rules", None) is not None
        for adapter in adapters or ()
    )
    return (
        f"adapters={[getattr(adapter, 'name', '?') for adapter in adapters or ()]}"
        f", engine={'configured' if config.sse_host.strip() else 'NOT configured'}"
        f", redis={'configured' if config.redis.configured else 'NOT configured'}"
        f", read={config.read_available}"
        f", retrieval=hybrid:{_on_off(config.hybrid_search)}"
        f"/rerank:{_on_off(config.rerank_results)}"
        f", write={config.write_available}"
        f", archive={config.archive_available}"
        f", docs={_on_off(config.docs_enabled)}"
        f"/{'auto' if config.docs_auto_index else 'manual'}"
        f", escalation={config.escalation_available}"
        f", phase_rules={'on' if rules_on else 'OFF (nothing will be mined)'}"
        f", stream={config.stream_key}"
    )


# ------------------------------------------------------------------ wiring -
def _load_config(logger: Optional[logging.Logger]) -> AgenticKnowledgeConfig:
    """
    Read the policy the deployment ships with, and fail on it if it is wrong.

    Bundled JSON, then the environment over it, then the checks that only the
    resolved whole can pass.  Every error leaves this function rather than being
    caught: a proxy that came up with an unreadable policy answers requests with
    no knowledge at all, and looks exactly like a knowledge base that is empty.

    Raises
    ------
    ValueError
        On a malformed or inconsistent policy.
    FileNotFoundError
        When ``…_CONFIG`` names a file that is not there.
    """
    config = AgenticKnowledgeConfig.from_file()
    config.override_from_env(logger)
    config.validate()
    return config


def _build_adapters(
    config: AgenticKnowledgeConfig, logger: Optional[logging.Logger]
) -> Tuple[AgentAdapter, ...]:
    """Build the adapters the configuration serves."""
    adapters: list = []
    if "codex" in config.served_adapters:
        adapters.append(
            CodexAgentAdapter(
                scope_resolver=ScopeResolver(
                    strategy=config.scope_strategy,
                    overrides=config.scope_overrides,
                    default_scope=config.scope_default,
                ),
                logger=logger,
            )
        )
    if "claude_code" in config.served_adapters:
        adapters.append(
            ClaudeCodeAgentAdapter(
                scope_resolver=ScopeResolver(
                    strategy=config.scope_strategy,
                    overrides=config.scope_overrides,
                    default_scope=config.scope_default,
                ),
                mode_map=dict(config.claude_code_mode_map),
            )
        )
    return tuple(adapters)


def _test_mode(adapters: Optional[Tuple[AgentAdapter, ...]]) -> str:
    """
    The mode name the loaded phase rules call a test run.

    The miner decides whether a green command closes an episode by its mode; the
    rules that classified the evidence in the first place own that name.  Reading
    it from the adapter keeps an operator's own ruleset — where the test mode is
    called something else — from yielding zero recipes in silence.
    """
    for adapter in adapters or ():
        rules = getattr(adapter, "phase_rules", None)
        mode = str(getattr(rules, "test_mode", "") or "")
        if mode:
            return mode
    return "test"


def _build_cache(
    config: AgenticKnowledgeConfig, logger: Optional[logging.Logger]
) -> GenerationContextCache:
    """Return the generation cache: Redis when configured, in-process otherwise."""
    if not config.redis.configured:
        return InMemoryContextCache(ttl_seconds=config.cache_ttl_seconds)
    # The connection is opened by the cache on first use, in the process that
    # uses it: a proxy that preloads the plugin and forks its workers must not
    # hand each of them the parent's Redis connection pool.
    return GenerationContextCache(
        connection=config.redis,
        key_prefix=config.key_prefix,
        ttl_seconds=config.cache_ttl_seconds,
        logger=logger,
    )


# ------------------------------------------------------------- annotations -
def _is_mined_only(request) -> bool:
    """Whether one search asks only for mined knowledge."""
    categories = tuple(str(kind) for kind in request.categories or ())
    return bool(categories) and all(kind in MINED_KINDS for kind in categories)


def _source_counts(context: Optional[KnowledgeContext]) -> Dict[str, int]:
    """Count the injected hits by their document kind."""
    counts: Dict[str, int] = {}
    if context is None:
        return counts
    for hit in context.hits:
        key = hit.category or "doc"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _with_miss_reason(context: KnowledgeContext, reason: str) -> KnowledgeContext:
    """The context, saying why the cache did not serve it."""
    import dataclasses

    return dataclasses.replace(context, cache_miss_reason=reason)


def _query_builder_with_failure(builder: Any, signature: str) -> Any:
    """The query builder, carrying the failure the miner read off the turn."""
    if not signature:
        return builder
    import dataclasses

    return dataclasses.replace(builder, signature=signature)


def _evidence_fingerprint(turn: AgentTurn) -> str:
    """Identity of the turn's evidence, to detect when it moves."""
    if not turn.evidence:
        return ""
    last = turn.evidence[-1]
    parts = [last.kind, last.call_id or last.event_id, str(last.succeeded)]
    if last.mode:
        parts.append(last.mode)
    import hashlib

    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:16]


def _evidence_moved(previous: str, turn: AgentTurn) -> bool:
    """Whether the turn produced new evidence since the block was cached."""
    return _evidence_fingerprint(turn) != previous
