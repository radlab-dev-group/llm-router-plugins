"""
The read path: the options the engine is sent, and what comes back from it.

Two contract details live here and neither is visible from the plugin's own
types.  The first is what the two score numbers mean: ``min_similarity`` is a
cosine cutoff the engine applies to the vector leg before the full-text leg
runs, while the ``score_normalized`` of a hit is whatever the final stage
produced — a sigmoid of a cross-encoder logit under ``rerank_results=True`` —
and the engine says which one it was in ``results.score_scale``.  Reading that
field is the only way a consumer can tell.  The second is that a document's
path is keyed ``relative_filepath`` in the reformatted answer the endpoint
serves and ``relative_path`` in the unreformatted row.
"""

import sys
import types
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from llm_router_plugins.utils.knowledge.agentic_knowledge.sse import (
    STATUS_OK,
    STATUS_UNAVAILABLE,
    KnowledgeStore,
    SearchRequest,
    _flatten,
    _hits,
    _options,
    _score_scale,
)


class _Reply:
    """The object ``sse_lib`` hands back: reformatted hits and the scale."""

    def __init__(self, hits, score_scale="reranker"):
        self.hits = hits
        self.results = {"score_scale": score_scale}


class _Hit:
    """One ``sse_lib.models.SearchHit``."""

    def __init__(self, **values):
        self.text = values.get("text")
        self.document_name = values.get("document_name", "")
        self.relative_filepath = values.get("relative_filepath", "")
        self.score = values.get("score")
        self.score_normalized = values.get("score_normalized")
        self.raw = values


class _Client:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def search(self, collection, query, options):
        self.calls.append((collection, query, options))
        return self.reply


# ------------------------------------------------------------------ login --
def test_a_login_is_handed_to_the_client_that_builds_itself(monkeypatch):
    """
    An engine behind login has no token to put in an Authorization header.

    The client logs in on its first call when it was built with a username
    and a password, so the pair has to reach the constructor rather than
    stop at the store.
    """
    built = {}

    class _Built:
        def __init__(self, host, **kwargs):
            built["host"] = host
            built.update(kwargs)

    module = types.ModuleType("sse_lib")
    module.SSEClient = _Built
    monkeypatch.setitem(sys.modules, "sse_lib", module)
    store = KnowledgeStore(
        host="http://engine.test", username="default_user", password="secret"
    )
    store._ensure_client()
    assert built["host"] == "http://engine.test"
    assert built["username"] == "default_user"
    assert built["password"] == "secret"


def test_no_credentials_leaves_the_client_anonymous(monkeypatch):
    """Without a credential the client must not be told to log in at all."""
    built = {}

    class _Built:
        def __init__(self, host, **kwargs):
            built.update(kwargs)

    module = types.ModuleType("sse_lib")
    module.SSEClient = _Built
    monkeypatch.setitem(sys.modules, "sse_lib", module)
    KnowledgeStore(host="http://engine.test")._ensure_client()
    assert built["username"] is None
    assert built["password"] is None


# --------------------------------------------------------------- the options -
def test_the_thread_filter_and_the_categories_intersect():
    """
    ``use_and_operator`` combines the filter families, and must be on.

    Left at the engine's default the categories and the thread filter would be
    unioned, so an archived window would be answered with the whole collection.
    """
    options = _options(
        SearchRequest(
            collection="agentic_kb_webapp_sessions",
            query="boom",
            categories=("session_archive",),
            thread_id="thread-1",
        ),
        0.35,
    )
    assert options["use_and_operator"] is True
    assert options["categories"] == ["session_archive"]
    assert options["metadata_filters"] == [
        {"operator": "eq", "field": {"thread_id": "thread-1"}}
    ]


# ----------------------------------------------------------- missing collection -
class _EngineError(Exception):
    """The shape ``sse_lib.SSEAPIError`` carries: HTTP 200 and a code in the body."""

    def __init__(self, error_code, message):
        super().__init__(message)
        self.errors = [{"error_code": error_code, "message": message}]


class _FailingClient:
    def __init__(self, error):
        self.error = error

    def search(self, collection, query, options):
        raise self.error


def test_a_repository_not_indexed_yet_answers_as_an_empty_one():
    """
    A collection is created when the project's first episode is indexed, so
    every earlier request of that project meets a collection that does not
    exist.  The engine says so with an error code, not with an empty list; to a
    reader both mean the same thing, and neither is worth a warning per request.
    """
    store = KnowledgeStore(
        client=_FailingClient(
            _EngineError("engine_001", "Kolekcja nie istnieje lub brak dostępu!")
        )
    )
    result = store.search(SearchRequest(collection="agentic_kb_webapp", query="fix"))
    assert result.status == STATUS_OK
    assert result.hits == ()
    assert result.reason == "no collection"


def test_another_engine_error_is_still_an_unavailable_store():
    store = KnowledgeStore(
        client=_FailingClient(
            _EngineError("engine_004", "Błąd modelu lub konfiguracji!")
        )
    )
    result = store.search(SearchRequest(collection="agentic_kb_webapp", query="fix"))
    assert result.status == STATUS_UNAVAILABLE


def test_an_error_without_engine_codes_is_not_a_missing_collection():
    store = KnowledgeStore(client=_FailingClient(RuntimeError("boom")))
    result = store.search(SearchRequest(collection="agentic_kb_webapp", query="fix"))
    assert result.status == STATUS_UNAVAILABLE


def test_a_request_without_a_thread_sends_no_metadata_filter():
    options = _options(SearchRequest(collection="c", query="q"), 0.35)
    assert "metadata_filters" not in options


# ---------------------------------------------------------------- the scale -
def test_the_score_scale_of_the_answer_is_carried_through():
    store = KnowledgeStore(client=_Client(_Reply([], score_scale="rrf")))
    assert (
        store.search(SearchRequest(collection="c", query="q")).score_scale == "rrf"
    )


def test_the_score_scale_is_read_from_both_shapes():
    assert _score_scale(_Reply([], "reranker")) == "reranker"
    assert _score_scale({"results": {"score_scale": "cosine"}}) == "cosine"
    assert _score_scale({"score_scale": "cosine"}) == "cosine"
    assert _score_scale({}) == ""
    assert _score_scale(None) == ""


def test_a_scale_is_copied_onto_every_hit():
    """The scale is read once per answer and stamped on the hits it produced."""
    reply = _Reply([_Hit(text="t", score=2.1, score_normalized=0.9)], "rrf")
    assert tuple(_hits("c", reply, "rrf"))[0].score_scale == "rrf"
    assert tuple(_hits("c", reply))[0].score_scale == "rrf"


# ------------------------------------------------------------------- the path -
def test_the_reformatted_hit_names_its_path_relative_filepath():
    """The served key is ``relative_filepath``, not ``relative_path``."""
    entry = {
        "score": 1.0,
        "score_normalized": 0.8,
        "document_name": "recipe:20261016:abc",
        "relative_filepath": "recipe:20261016:abc",
        "text_str": "body",
    }
    assert _flatten(entry).relative_filepath == "recipe:20261016:abc"


def test_an_unreformatted_row_keeps_its_relative_path():
    entry = {
        "result": {
            "text": {
                "document_name": "recipe:20261016:abc",
                "relative_path": "src/a.py",
                "text_str": "body",
                "score": 0.4,
            }
        }
    }
    assert _flatten(entry).relative_filepath == "src/a.py"


def test_a_flattened_entry_reaches_the_caller_as_a_hit():
    body = {
        "results": {
            "score_scale": "reranker",
            "detailed_results": {
                "0": {
                    "score": 2.5,
                    "score_normalized": 0.92,
                    "document_name": "recipe:20261016:abc",
                    "relative_filepath": "recipe:20261016:abc",
                    "text_str": "the fix",
                }
            },
        }
    }
    store = KnowledgeStore(client=_Client(body))
    result = store.search(SearchRequest(collection="agentic_kb_webapp", query="q"))
    assert result.status == STATUS_OK
    assert result.score_scale == "reranker"
    assert result.hits[0].relative_path == "recipe:20261016:abc"
    assert result.hits[0].text == "the fix"
    assert result.hits[0].score_normalized == 0.92


# ------------------------------------------------------------------ phase --
def test_a_phase_restriction_is_one_metadata_filter_per_phase():
    """
    The engine unions several ``metadata_filters``, and here that is the
    feature: N filters read as "the phase is one of these N".
    """
    options = _options(
        SearchRequest(collection="c", query="q", phases=("test", "implement")),
        0.35,
    )
    assert options["metadata_filters"] == [
        {"operator": "eq", "field": {"phase": "test"}},
        {"operator": "eq", "field": {"phase": "implement"}},
    ]


def test_a_thread_filter_wins_over_the_phases():
    """
    Two metadata filters would be unioned, so an archived window filtered by
    both would come back as every thread of the repository.
    """
    options = _options(
        SearchRequest(
            collection="c", query="q", thread_id="thread-1", phases=("test",)
        ),
        0.35,
    )
    assert options["metadata_filters"] == [
        {"operator": "eq", "field": {"thread_id": "thread-1"}}
    ]


def test_the_retrieval_options_are_the_ones_configured():
    """The legs and the reranker are the policy's, not the client's opinion."""
    options = _options(
        SearchRequest(collection="c", query="q"),
        0.35,
        hybrid_search=False,
        rerank_results=True,
    )
    assert options["hybrid_search"] is False
    assert options["rerank_results"] is True


def test_a_search_goes_out_with_the_store_default_legs():
    """Hybrid on, reranker off: the score comes back on the RRF scale."""
    client = _Client(_Reply([], score_scale="rrf"))
    store = KnowledgeStore(client=client)
    assert (
        store.search(SearchRequest(collection="c", query="q")).score_scale == "rrf"
    )
    (_collection, _query, options), = client.calls
    assert options["hybrid_search"] is True
    assert options["rerank_results"] is False


def test_a_store_switches_every_search_it_makes():
    client = _Client(_Reply([], score_scale="reranker"))
    store = KnowledgeStore(client=client, hybrid_search=False, rerank_results=True)
    store.search(SearchRequest(collection="c", query="q"))
    (_collection, _query, options), = client.calls
    assert options["hybrid_search"] is False
    assert options["rerank_results"] is True
