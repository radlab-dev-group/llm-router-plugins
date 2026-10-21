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


def test_decode_drops_malformed_entries_instead_of_raising():
    # A cache value is foreign input: an older build sharing this schema number,
    # or anything else with the key prefix, can leave a field of the wrong type.
    # Decoding must miss, never raise — a raise propagates out of apply().
    import json

    from llm_router_plugins.utils.knowledge.agentic_knowledge.cache import _decode

    assert _decode(json.dumps({"schema": 1, "chars": "not-an-int"})) is None
    assert (
        _decode(
            json.dumps(
                {"schema": 1, "hits": [{"score": "high", "category": "recipe"}]}
            )
        )
        is None
    )
    # A non-list `hits` is not a crash either — it is no hits.
    degraded = _decode(json.dumps({"schema": 1, "hits": 5, "rendered": "x"}))
    assert degraded is None or degraded.context.hits == ()


def test_decode_still_reads_a_good_entry():
    import json

    from llm_router_plugins.utils.knowledge.agentic_knowledge.cache import (
        _decode,
        _encode,
    )

    decoded = _decode(_encode(entry()))
    assert decoded is not None and decoded.mode == "debug"


def _doc_hit_hit(text="the fix"):
    from llm_router_plugins.utils.knowledge.agentic_knowledge.assembler import (
        KnowledgeHit,
    )

    return KnowledgeHit(
        category="doc",
        document_name="doc:20261021:aa",
        text=text,
        score=0.8,
        weight=0.64,
        source="docs/guide.md",
        doc_type="guide",
        heading="Deploy",
    )


def test_the_documentation_fields_of_a_hit_survive_the_cache():
    """
    The citation of a documentation section is its file and heading.

    A cache that silently dropped them would render the same block — the
    rendered text carries them — but report a hit without a source, and the
    annotation would say the quote came from nowhere.
    """
    from llm_router_plugins.utils.knowledge.agentic_knowledge.assembler import (
        KnowledgeContext,
    )

    cache = InMemoryContextCache()
    context = KnowledgeContext(hits=(_doc_hit_hit(),), rendered="a block", chars=7)
    entry = CachedContext(
        context=context,
        evidence_fingerprint="f1",
        mode="implement",
        escalated=False,
        escalated_model="",
    )
    cache.put("repo", "gen", entry)
    fetched = cache.get("repo", "gen")
    assert fetched.context.hits[0].source == "docs/guide.md"
    assert fetched.context.hits[0].doc_type == "guide"
    assert fetched.context.hits[0].heading == "Deploy"


def test_the_standing_digest_survives_the_cache():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.assembler import (
        KnowledgeContext,
    )
    from llm_router_plugins.utils.knowledge.agentic_knowledge.doc_context import (
        DocContext,
    )

    cache = InMemoryContextCache()
    digest = DocContext(
        scope="repo",
        text="- README.md (readme): Demo",
        source_files=("README.md",),
        fingerprint="deadbeef",
    )
    context = KnowledgeContext(rendered="a block", chars=7, doc_context=digest)
    entry = CachedContext(
        context=context,
        evidence_fingerprint="f1",
        mode="implement",
        escalated=False,
        escalated_model="",
    )
    cache.put("repo", "gen", entry)
    fetched = cache.get("repo", "gen")
    assert fetched.context.doc_context is not None
    assert fetched.context.doc_context.text == digest.text
    assert fetched.context.doc_context.fingerprint == digest.fingerprint


def test_an_entry_without_a_digest_decodes_without_one():
    import json

    from llm_router_plugins.utils.knowledge.agentic_knowledge.cache import (
        _decode,
    )

    decoded = _decode(
        json.dumps(
            {
                "schema": 2,
                "rendered": "x",
                "chars": 1,
                "hits": [],
                "mode": "implement",
            }
        )
    )
    assert decoded is not None
    assert decoded.context.doc_context is None
