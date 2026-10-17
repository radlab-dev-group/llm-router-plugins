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
    KnowledgeStore,
    STATUS_OK,
    STATUS_UNAVAILABLE,
)

__all__ = ["AgenticKnowledgePlugin"]

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
        """Bind the collaborators, building the configured ones when needed."""
        super().__init__(logger=logger)
        self._config = config
        self._store = store
        self._cache = cache
        self._publisher = publisher
        self._adapters = adapters
        self._miner: Optional[EpisodeMiner] = None
        self._query_builder: Optional[QueryBuilder] = None
        self._escalation: Optional[EscalationPolicy] = None
        self._limiter: Optional[LookupLimiter] = None
        self._configured = False

    # -------------------------------------------------------------- wiring -
    def _ensure_configured(self) -> Optional[AgenticKnowledgeConfig]:
        """Load the policy and the collaborators on first use."""
        if self._configured:
            return self._config
        if self._config is None:
            self._config = AgenticKnowledgeConfig.from_file()
            self._config.override_from_env(self._logger)
        config = self._config
        if self._store is None:
            self._store = KnowledgeStore(
                host=config.sse_host,
                api_prefix=config.sse_prefix,
                token=config.sse_token,
                timeout_seconds=config.sse_timeout_seconds,
                verify=config.sse_verify,
                min_similarity=config.min_similarity,
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
        )
        self._query_builder = QueryBuilder(
            collection_namer=config.collection,
            min_similarity=config.min_similarity,
        )
        self._escalation = EscalationPolicy(config)
        self._limiter = LookupLimiter(
            max_per_generation=config.max_lookups_per_generation
        )
        self._configured = True
        return config

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
        try:
            config = self._ensure_configured()
        except Exception as exc:
            self._warn("agentic knowledge config failed: %s", type(exc).__name__)
            return payload
        if config is None or not config.enabled:
            return payload

        adapter = self._matching_adapter(payload)
        if adapter is None:
            return payload
        try:
            turn = adapter.parse(payload)
        except Exception as exc:
            self._warn("agentic knowledge parse failed: %s", type(exc).__name__)
            return payload
        if turn is None:
            return payload

        if turn.request_class == _REQUEST_CLASS_AUX:
            return payload

        mining = self._mine(turn)
        publish = self._publish(mining, config)

        if turn.request_class == _REQUEST_CLASS_COMPACTION:
            # A window about to be summarized is written, never read from.
            self._annotate(
                payload, adapter, turn, None, EscalationDecision(), publish
            )
            return payload
        return self._serve_read(adapter, turn, payload, config, mining, publish)

    # ----------------------------------------------------------- read path -
    def _serve_read(
        self,
        adapter: AgentAdapter,
        turn: AgentTurn,
        payload: Dict[str, Any],
        config: AgenticKnowledgeConfig,
        mining: Any,
        publish: PublishStatus,
    ) -> Any:
        """Search, inject, decide escalation, and annotate — in one pass."""
        policy = config.policy_for(turn.mode) if config.read_available else None
        cached = self._cached(turn)
        fresh = cached is not None and self._is_fresh(cached, turn)
        if fresh:
            context = KnowledgeContext(
                hits=cached.context.hits,
                rendered=cached.context.rendered,
                chars=cached.context.chars,
                best_recipe_score=cached.context.best_recipe_score,
                deadends=cached.context.deadends,
                cache="hit",
                status=cached.context.status,
            )
        else:
            context = self._assemble(turn, config, policy, mining)

        if not context.empty:
            payload = adapter.inject(payload, context.rendered)

        decision = (
            self._escalation.decide(
                turn,
                context,
                policy,
                mining.loop_count,
                mining.signature_hash,
            )
            if self._escalation is not None
            else EscalationDecision()
        )

        target = decision.model_name if decision.action == ACTION_ESCALATE else ""
        if not target:
            # A loop escalated earlier in this generation stays escalated even
            # when this request's own evidence no longer repeats the loop.
            target = self._persisted_escalation(cached, turn)
            if target:
                decision = EscalationDecision(
                    ACTION_ESCALATE,
                    REASON_LOOP_WITHOUT_RECIPE,
                    target,
                    decision.loop_count,
                    decision.signature_hash,
                )
        if target and (turn.routing_source or "").lower() != "explicit":
            previous = str(payload.get("model") or turn.model)
            if previous and previous != target:
                adapter.set_model(payload, target)
                self._note_model_before_knowledge(payload, previous)

        self._remember_context(turn, context, target)
        self._annotate(payload, adapter, turn, context, decision, publish)
        return payload

    def _assemble(
        self,
        turn: AgentTurn,
        config: AgenticKnowledgeConfig,
        policy: Any,
        mining: Any,
    ) -> KnowledgeContext:
        """Search the sources and assemble the block; fail-open to an empty one."""
        if policy is None:
            return KnowledgeContext(cache="disabled", status="disabled")
        if not turn.scope:
            return KnowledgeContext(cache="miss", status="disabled")
        generation = turn.cache_generation
        if self._limiter is not None and not self._limiter.allow(
            turn.scope, generation
        ):
            return KnowledgeContext(cache="miss", status="ok")

        results = []
        statuses = set()
        for request in self._query_builder.build(turn, policy):
            try:
                result = self._store.search(request)
            except Exception as exc:
                self._warn("knowledge search failed: %s", type(exc).__name__)
                from llm_router_plugins.utils.knowledge.agentic_knowledge.sse import (
                    SearchResult as _SR,
                )

                result = _SR(STATUS_UNAVAILABLE, reason="search")
            statuses.add(result.status)
            results.append((result, request.categories))
        status = _status_of(statuses)

        assembler = ContextAssembler(
            signature_hash=mining.signature_hash,
            touched_paths=turn.touched_paths,
            local_deadends=mining.tried,
        )
        context = assembler.assemble(results, policy)
        return KnowledgeContext(
            hits=context.hits,
            rendered=context.rendered,
            chars=context.chars,
            best_recipe_score=context.best_recipe_score,
            deadends=context.deadends,
            cache="miss",
            status=status,
        )

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
        self, turn: AgentTurn, context: KnowledgeContext, escalated_model: str
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
            return self._miner.mine(turn)
        except Exception as exc:
            self._warn("knowledge mining failed: %s", type(exc).__name__)
            return MiningResult()

    def _publish(self, mining: Any, config: AgenticKnowledgeConfig) -> PublishStatus:
        """Hand mined episodes to the worker; never blocks or raises on Redis."""
        events = getattr(mining, "events", ())
        if not events or not config.write_available or self._publisher is None:
            return PublishStatus()
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
    ) -> None:
        """Write the knowledge layer's decision into ``payload["routing"]``."""
        routing = payload.setdefault("routing", {})
        if not isinstance(routing, dict):
            routing = {}
            payload["routing"] = routing
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
                "status": publish.status,
            }
        routing["knowledge"] = knowledge

    def _warn(self, message: str, *args: Any) -> None:
        """Report one failure when a logger was given."""
        if self._logger is not None:
            self._logger.warning(message, *args)


def _status_of(statuses: set) -> str:
    """Collapse per-source search statuses into one retrieval status."""
    if STATUS_OK in statuses:
        return STATUS_OK
    if STATUS_UNAVAILABLE in statuses:
        return STATUS_UNAVAILABLE
    return "disabled"


# ------------------------------------------------------------------ wiring -
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
                )
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


def _build_cache(
    config: AgenticKnowledgeConfig, logger: Optional[logging.Logger]
) -> GenerationContextCache:
    """Return the generation cache: Redis when configured, in-process otherwise."""
    if not config.redis.configured:
        return InMemoryContextCache(ttl_seconds=config.cache_ttl_seconds)
    try:
        import redis  # noqa: PLC0415 - optional dependency

        client = redis.Redis(**config.redis.client_kwargs())
    except Exception:
        if logger is not None:
            logger.warning(
                "agentic knowledge cache: redis unavailable, using in-process cache"
            )
        return InMemoryContextCache(ttl_seconds=config.cache_ttl_seconds)
    return GenerationContextCache(
        client=client,
        key_prefix=config.key_prefix,
        ttl_seconds=config.cache_ttl_seconds,
        logger=logger,
    )


# ------------------------------------------------------------- annotations -
def _source_counts(context: Optional[KnowledgeContext]) -> Dict[str, int]:
    """Count the injected hits by their document kind."""
    counts: Dict[str, int] = {}
    if context is None:
        return counts
    for hit in context.hits:
        key = hit.category or "doc"
        counts[key] = counts.get(key, 0) + 1
    return counts


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
