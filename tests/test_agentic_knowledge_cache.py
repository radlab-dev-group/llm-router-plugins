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


def test_recipe_provenance_and_trial_counts_survive_serialization():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.assembler import (
        KnowledgeHit,
    )
    from llm_router_plugins.utils.knowledge.agentic_knowledge.cache import (
        _decode,
        _encode,
    )

    original = CachedContext(
        context=KnowledgeContext(
            hits=(
                KnowledgeHit(
                    category="recipe",
                    score=0.95,
                    signature_hash="failure-a",
                    test_command="pytest tests/test_a.py",
                ),
            ),
            rendered="a tested fix",
        ),
        recipe_trials={"failure-a": 2, "failure-b": 3},
    )
    decoded = _decode(_encode(original))
    assert decoded.recipe_trials == original.recipe_trials
    assert decoded.context.hits[0].signature_hash == "failure-a"
    assert decoded.context.hits[0].test_command == "pytest tests/test_a.py"


def test_legacy_cache_does_not_invent_recipe_verification():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.cache import _decode

    decoded = _decode(
        {
            "schema": 2,
            "rendered": "legacy fix",
            "hits": [{"category": "recipe", "score": 0.99}],
        }
    )
    assert decoded.recipe_trials == {}
    assert decoded.context.hits[0].signature_hash == ""
    assert decoded.context.hits[0].test_command == ""


def test_invalid_recipe_trial_counts_are_not_used():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.cache import _decode

    decoded = _decode(
        {
            "schema": 2,
            "recipe_trials": {"a": "2", "b": True, "c": -1, "d": 0, "e": 2},
        }
    )
    assert decoded.recipe_trials == {"e": 2}


# ------------------------------------------------------------ feedback -----

class _FakeHashClient:
    """A redis-shaped client that remembers one hash per key."""

    def __init__(self):
        self.hashes = {}
        self.increments = []

    def hincrby(self, key, field, amount):
        bucket = self.hashes.setdefault(key, {})
        bucket[field] = int(bucket.get(field, 0)) + amount
        self.increments.append((key, field, amount))

    def hgetall(self, key):
        return dict(self.hashes.get(key, {}))


def test_feedback_tracker_counts_upvotes_and_downvotes():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.cache import (
        FeedbackTracker,
    )

    tracker = FeedbackTracker()
    tracker.record_feedback("recipe:20260101:abc", success=True)
    tracker.record_feedback("recipe:20260101:abc", success=True)
    tracker.record_feedback("recipe:20260101:abc", success=False)
    stats = tracker.get_feedback("recipe:20260101:abc")
    assert stats == {"upvotes": 2, "downvotes": 1}
    assert tracker.all_feedback() == {
        "recipe:20260101:abc": {"upvotes": 2, "downvotes": 1}
    }
    # A document nobody scored is an empty record, not an error.
    assert tracker.get_feedback("recipe:20260101:other") == {
        "upvotes": 0,
        "downvotes": 0,
    }
    assert tracker.get_feedback("") == {"upvotes": 0, "downvotes": 0}


def test_feedback_tracker_writes_through_to_redis():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.cache import (
        FeedbackTracker,
    )

    client = _FakeHashClient()
    tracker = FeedbackTracker(client=client, key_prefix="agentic-kb")
    tracker.record_feedback("recipe:20260101:abc", success=False)
    tracker.record_feedback("recipe:20260101:abc", success=False)
    assert client.hashes["agentic-kb:fb:recipe:20260101:abc"] == {
        "downvotes": 2
    }
    assert len(client.increments) == 2


def test_feedback_tracker_reads_redis_when_local_is_empty():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.cache import (
        FeedbackTracker,
    )

    client = _FakeHashClient()
    client.hashes["agentic-kb:fb:recipe:20260101:abc"] = {
        "upvotes": 5,
        "downvotes": 1,
    }
    tracker = FeedbackTracker(client=client, key_prefix="agentic-kb")
    assert tracker.get_feedback("recipe:20260101:abc") == {
        "upvotes": 5,
        "downvotes": 1,
    }


def test_a_broken_client_fails_open():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.cache import (
        FeedbackTracker,
    )

    class _Broken:
        def hincrby(self, *args):
            raise RuntimeError("down")

        def hgetall(self, *args):
            raise RuntimeError("down")

    tracker = FeedbackTracker(client=_Broken())
    tracker.record_feedback("recipe:20260101:abc", success=True)
    # The local count survives a dead client.
    assert tracker.get_feedback("recipe:20260101:abc") == {
        "upvotes": 1,
        "downvotes": 0,
    }


def test_injected_document_ids_survive_the_cache_round_trip():
    from llm_router_plugins.utils.knowledge.agentic_knowledge.assembler import (
        KnowledgeContext,
        KnowledgeHit,
    )
    from llm_router_plugins.utils.knowledge.agentic_knowledge.cache import (
        _decode,
        _encode,
    )

    original = CachedContext(
        context=KnowledgeContext(
            hits=(
                KnowledgeHit(
                    category="recipe",
                    document_name="recipe:20260101:abc",
                    score=0.9,
                ),
            ),
            rendered="a fix",
        ),
        injected_doc_ids=("recipe:20260101:abc",),
    )
    decoded = _decode(_encode(original))
    assert decoded.injected_doc_ids == ("recipe:20260101:abc",)
    # A cache entry written before the field existed decodes without one.
    legacy = _decode(
        '{"schema": 3, "evidence_fingerprint": "", "mode": "debug", '
        '"escalated": false, "escalated_model": "", "recipe_trials": {}, '
        '"chars": 1, "status": "ok", "rendered": ""}'
    )
    assert legacy.injected_doc_ids == ()
