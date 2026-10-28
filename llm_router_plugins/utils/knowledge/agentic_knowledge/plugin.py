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
import time

from typing import Any, Dict, Optional, Tuple

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
from llm_router_plugins.utils.knowledge.agentic_knowledge.cache import (
    CachedContext,
    GenerationContextCache,
    InMemoryContextCache,
    LookupLimiter,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
    AgenticKnowledgeConfig,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.documents import (
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
from llm_router_plugins.utils.knowledge.agentic_knowledge.events import (
    KnowledgeEventPublisher,
    PublishStatus,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.miner import EpisodeMiner
from llm_router_plugins.utils.knowledge.agentic_knowledge.query import QueryBuilder
from llm_router_plugins.utils.knowledge.agentic_knowledge.scope import ScopeResolver
from llm_router_plugins.utils.knowledge.agentic_knowledge.sse import (
    REASON_READ_BUDGET,
    KnowledgeStore,
    STATUS_DISABLED,
    STATUS_OK,
    STATUS_UNAVAILABLE,
)

__all__ = ["AgenticKnowledgePlugin"]

#: How often one repository is told that its read budget starved its searches.
_BUDGET_WARNING_INTERVAL_SECONDS = 300.0

#: How many repositories that note is remembered for, so a proxy serving many
#: repositories cannot grow the table without bound.
_BUDGET_WARNING_SCOPES = 512

#: Request classes the plugin does not enrich.
_REQUEST_CLASS_AUX = "aux_title"
_REQUEST_CLASS_COMPACTION = "compaction"


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
    ) -> None:
        """Resolve the policy and wire the collaborators; open no connection."""
        super().__init__(logger=logger)
        self._config = config if config is not None else _load_config(self._logger)
        self._store = store
        self._cache = cache
        self._publisher = publisher
        self._adapters = adapters
        self._miner: Optional[EpisodeMiner] = None
        self._query_builder: Optional[QueryBuilder] = None
        self._escalation: Optional[EscalationPolicy] = None
        self._limiter: Optional[LookupLimiter] = None
        self._docs: Optional[DocsIndexer] = None
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
                payload, adapter, turn, None, EscalationDecision(), publish, docs
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
        fresh = read_on and cached is not None and self._is_fresh(cached, turn)
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
            context = self._assemble(
                turn, config, policy, mining, self._doc_context(turn, docs)
            )

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
            self._remember_context(turn, context, target, recipe_trials)
        self._annotate(payload, adapter, turn, context, decision, publish, docs)
        return payload

    def _assemble(
        self,
        turn: AgentTurn,
        config: AgenticKnowledgeConfig,
        policy: Any,
        mining: Any,
        doc_context: Any = None,
    ) -> KnowledgeContext:
        """Search the sources and assemble the block; fail-open to an empty one."""
        if policy is None:
            return KnowledgeContext(
                cache="disabled", status="disabled", doc_context=doc_context
            )
        if not turn.scope:
            return KnowledgeContext(cache="miss", status="disabled")
        generation = turn.cache_generation
        if self._limiter is not None and not self._limiter.allow(
            turn.scope, generation
        ):
            return KnowledgeContext(cache="miss", status="ok")

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
        skipped = sum(
            1 for result, _ in results if result.reason == REASON_READ_BUDGET
        )
        for result, _ in results:
            statuses.add(result.status)
        if skipped:
            # The reads the operator will otherwise conclude never happened.
            # Both numbers go in the line because the usual cause is not a
            # small budget: it is a per-call timeout the budget cannot hold
            # more than one of, which looks like an idle engine and an empty
            # knowledge base at once.  Rate-limited per repository, because a
            # starved configuration starves every request of every session,
            # and a warning that prints on every request is a warning nobody
            # reads.
            self._warn_budget(turn.scope, skipped, len(results) + skipped)
        status = _status_of(statuses)

        assembler = ContextAssembler(
            signature_hash=mining.signature_hash,
            touched_paths=turn.touched_paths,
            local_deadends=mining.tried,
        )
        context = assembler.assemble(results, policy, doc_context=doc_context)
        return KnowledgeContext(
            hits=context.hits,
            rendered=context.rendered,
            chars=context.chars,
            best_recipe_score=context.best_recipe_score,
            deadends=context.deadends,
            cache="miss",
            status=status,
            doc_context=context.doc_context,
        )

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
    def _is_fresh(cached: CachedContext, turn: AgentTurn) -> bool:
        """Whether the cached block still describes this turn."""
        return cached.mode == turn.mode and not _evidence_moved(
            cached.evidence_fingerprint, turn
        )

    @staticmethod
    def _persisted_escalation(
        cached: Optional[CachedContext], turn: AgentTurn
    ) -> str:
        """The model an earlier request of this generation escalated to."""
        if cached is None or cached.mode != turn.mode:
            return ""
        return cached.escalated_model or ""

    def _remember_context(
        self,
        turn: AgentTurn,
        context: KnowledgeContext,
        escalated_model: str,
        recipe_trials: Optional[Dict[str, int]] = None,
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
                escalated=bool(escalated_model),
                escalated_model=escalated_model,
                recipe_trials=dict(recipe_trials or {}),
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
    return bool(categories) and all(
        kind in MINED_KINDS for kind in categories
    )


def _source_counts(context: Optional[KnowledgeContext]) -> Dict[str, int]:
    """Count the injected hits by their document kind."""
    counts: Dict[str, int] = {}
    if context is None:
        return counts
    for hit in context.hits:
        key = hit.category or "doc"
        counts[key] = counts.get(key, 0) + 1
    return counts


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
