"""
The contract from a mined episode to an injected block, with the real masker.

Each half of the layer is tested on its own, and that is exactly how the two
stopped agreeing: the miner wrote a header the reader could parse, the worker
masked the document whole and turned that header into masker tokens, and the
assembler compared a signature by hashing it again — which does not reproduce
the hash of the original output.  Every one of those passed its own tests.

So this file walks the whole road on real collaborators: a failure output goes
to the signature, the signature to a document, the document through the worker's
masking to what the engine would store, and the stored text back through the
assembler — where the recipe for the failure the agent is looking at has to come
out on top.
"""

import datetime
import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from llm_router_plugins.cli.plugins.utils.agentic_knowledge.worker import (
    KnowledgeIndexWorker,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.adapters.base import (
    KnowledgeEvent,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.assembler import (
    SIGNATURE_BONUS,
    ContextAssembler,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
    ModePolicy,
    SourcePolicy,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.documents import (
    KIND_DOC,
    KIND_RECIPE,
    build_document_text,
    parse_document_text,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.signature import (
    signature_of,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.sse import (
    STATUS_OK,
    SearchResult,
    StoreHit,
)

DAY = datetime.date(2026, 10, 17)
CREATED = datetime.datetime(2026, 10, 17, tzinfo=datetime.timezone.utc).timestamp()

#: The shapes a coding agent actually reports, with a secret in each.
FAILURES = {
    "traceback": (
        "Traceback (most recent call last):\n"
        '  File "/home/dev/repo/app/service.py", line 42, in load\n'
        "    assert items\n"
        "AssertionError: expected items for user@example.com"
    ),
    "pytest": (
        "FAILED tests/test_load.py::test_load - AssertionError: expected "
        "items from 192.168.1.9\n1 failed in 0.42s"
    ),
    "rust": "error[E0308]: mismatched types\n --> src/main.rs:12:5",
}

_POLICY = ModePolicy(
    name="debug",
    max_chars=6000,
    sources=(SourcePolicy("recipe", 1.0, 3), SourcePolicy("doc", 0.8, 3)),
)


def _worker():
    """A worker with the real masker and no connections to make."""
    return KnowledgeIndexWorker(
        logger=None, redis_client=object(), sse_client=object(), config=_config()
    )


def _config():
    from agentic_knowledge_support import knowledge_config

    return knowledge_config()


def _indexed(raw_output):
    """Return (signature, text-as-the-engine-would-store-it) for one failure."""
    signature = signature_of(raw_output)
    assert signature is not None, f"{raw_output!r} produced no signature at all"
    document = build_document_text(
        KIND_RECIPE,
        f"Failure: {signature.signature}\n\nFixed by: app/service.py",
        date=DAY,
        signature=signature.signature,
        signature_hash=signature.signature_hash,
        paths=("app/service.py",),
        thread_id="thread-1",
    )
    event = KnowledgeEvent(
        kind=KIND_RECIPE,
        fingerprint="abc123",
        scope="webapp",
        agent="codex",
        mode="debug",
        model="model-x",
        thread_id="thread-1",
        generation="gen-1",
        created_at=CREATED,
        signature=signature.signature,
        signature_hash=signature.signature_hash,
        text=document,
        metadata={"paths": ["app/service.py"]},
    )
    return signature, _worker()._prepare(event)


@pytest.mark.parametrize("label", sorted(FAILURES))
def test_the_header_survives_masking(label):
    """What the reader parses has to be readable after the worker is done."""
    signature, indexed = _indexed(FAILURES[label])
    meta, body = parse_document_text(indexed)
    assert meta.kind == KIND_RECIPE
    assert meta.date == DAY, "a masked date leaves the citation without a date"
    assert meta.signature_hash == signature.signature_hash
    assert meta.paths == ("app/service.py",)
    assert meta.thread_id == "thread-1"
    assert body.strip()


@pytest.mark.parametrize("label", sorted(FAILURES))
def test_masking_still_masks(label):
    """Keeping the header legible is not a licence to index the secret."""
    _, indexed = _indexed(FAILURES[label])
    assert "user@example.com" not in indexed
    assert "192.168.1.9" not in indexed


@pytest.mark.parametrize("label", sorted(FAILURES))
def test_a_recipe_for_this_failure_outranks_a_convention(label):
    """
    The signature bonus has to fire, or the read path ranks blindly.

    Scores are set so a convention page wins on retrieval alone (0.9 * 0.8)
    and the recipe wins once the failure it describes is the one being asked
    about (0.61 * 1.0 + 0.2).  No bonus, no ordering — which is the bug.
    """
    signature, indexed = _indexed(FAILURES[label])
    recipe = StoreHit(
        collection="agentic_kb_webapp",
        document_name=f"recipe:{DAY.strftime('%Y%m%d')}:abc123",
        text=indexed,
        score=0.61,
        score_normalized=0.61,
    )
    convention = StoreHit(
        collection="agentic_kb_webapp",
        document_name="conventions.md",
        text="Use four spaces and keyword arguments.",
        score=0.9,
        score_normalized=0.9,
    )
    result = SearchResult(STATUS_OK, hits=(convention, recipe))

    context = ContextAssembler(signature_hash=signature.signature_hash).assemble(
        [(result, (KIND_RECIPE, KIND_DOC))], _POLICY
    )

    assert [hit.category for hit in context.hits][0] == KIND_RECIPE
    assert context.hits[0].weight == pytest.approx(0.61 + SIGNATURE_BONUS, abs=1e-4)
    assert context.best_recipe_score >= SIGNATURE_BONUS


def test_an_unrelated_failure_gets_no_signature_bonus():
    """The bonus is for this failure; another one must not borrow it."""
    signature, indexed = _indexed(FAILURES["traceback"])
    recipe = StoreHit(
        collection="agentic_kb_webapp",
        document_name=f"recipe:{DAY.strftime('%Y%m%d')}:abc123",
        text=indexed,
        score=0.61,
        score_normalized=0.61,
    )
    context = ContextAssembler(signature_hash="some-other-failure").assemble(
        [(_ok(recipe), (KIND_RECIPE,))], _POLICY
    )
    assert context.hits[0].weight == pytest.approx(0.61, abs=1e-4)


def _ok(hit):
    return SearchResult(STATUS_OK, hits=(hit,))
