"""
The generation cache and the stateless lookup limiter.

The cache keeps one command generation's block stable so the context does not
change between two steps of one thought, and it is where escalation remembers
itself.  The in-memory cache is used when Redis is unavailable; the lookup
limiter bounds how much work one generation may cause in this process.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from llm_router_plugins.utils.knowledge.agentic_knowledge.assembler import (
    KnowledgeContext,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.cache import (
    CachedContext,
    InMemoryContextCache,
    LookupLimiter,
)


def entry(rendered="a block", escalated_model="big-model", mode="debug"):
    context = KnowledgeContext(rendered=rendered, chars=len(rendered))
    return CachedContext(
        context=context,
        evidence_fingerprint="f1",
        mode=mode,
        escalated=bool(escalated_model),
        escalated_model=escalated_model,
    )


def test_put_then_get_round_trips():
    cache = InMemoryContextCache()
    assert cache.put("repo", "gen", entry())
    fetched = cache.get("repo", "gen")
    assert fetched is not None
    assert fetched.context.rendered == "a block"
    assert fetched.escalated_model == "big-model"
    assert fetched.escalated


def test_miss_when_never_stored():
    cache = InMemoryContextCache()
    assert cache.get("repo", "gen") is None


def test_entries_are_isolated_by_scope_and_generation():
    cache = InMemoryContextCache()
    cache.put("repo", "gen-1", entry(rendered="one"))
    cache.put("repo", "gen-2", entry(rendered="two"))
    cache.put("other", "gen-1", entry(rendered="three"))
    assert cache.get("repo", "gen-1").context.rendered == "one"
    assert cache.get("repo", "gen-2").context.rendered == "two"
    assert cache.get("other", "gen-1").context.rendered == "three"


def test_empty_scope_or_generation_is_never_stored():
    cache = InMemoryContextCache()
    assert not cache.put("", "gen", entry())
    assert not cache.put("repo", "", entry())
    assert cache.get("", "gen") is None


def test_lookup_limiter_allows_up_to_the_ceiling_then_stops():
    limiter = LookupLimiter(max_per_generation=2)
    assert limiter.allow("repo", "gen")
    assert limiter.allow("repo", "gen")
    assert not limiter.allow("repo", "gen")
    # A different generation is unaffected.
    assert limiter.allow("repo", "other-gen")


def test_lookup_limiter_isolated_per_generation():
    limiter = LookupLimiter(max_per_generation=1)
    assert limiter.allow("repo", "gen")
    assert not limiter.allow("repo", "gen")
    assert limiter.allow("other", "gen")
