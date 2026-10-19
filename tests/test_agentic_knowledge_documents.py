"""
The document codec: a name and a header that survive the engine's round trip.

The search engine returns a document's name and text but never its metadata,
so a knowledge document carries its kind, retention day, failure signature and
paths in the name and in a one-line header inside the indexed text.  A text
without a header (indexed elsewhere, or a mid-document chunk) parses to an
empty metadata and the text itself, so a reader always gets usable content.

The name is not the other half of the story either: the engine stores an
indexed text as ``basename(<filepath>) + ".input_text"``, so the name
retention lists is a longer string than the one the plugin built, and anything
with a ``/`` in it would be cut by that ``basename``.
"""

import sys
import pathlib
import datetime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from llm_router_plugins.utils.knowledge.agentic_knowledge.documents import (
    ENGINE_NAME_SUFFIX,
    KIND_RECIPE,
    KIND_SESSION_ARCHIVE,
    build_document_text,
    document_name,
    is_knowledge_document,
    parse_document_name,
    parse_document_text,
)

DAY = datetime.date(2026, 10, 16)


def _engine_name(name: str) -> str:
    """The name the engine would report for a document sent as *name*."""
    return pathlib.PurePosixPath(name).name + ENGINE_NAME_SUFFIX


def test_build_then_parse_round_trips_the_fields():
    day = datetime.date(2026, 10, 16)
    text = build_document_text(
        KIND_RECIPE,
        "fix body",
        date=day,
        signature="AssertionError: boom",
        paths=("a.py", "b.py"),
        thread_id="thread-1",
    )
    meta, body = parse_document_text(text)
    assert meta.kind == KIND_RECIPE
    assert meta.date == day
    assert meta.signature == "AssertionError: boom"
    assert meta.paths == ("a.py", "b.py")
    assert meta.thread_id == "thread-1"
    assert body.strip() == "fix body"


def test_document_name_carries_kind_date_and_fingerprint():
    name = document_name(KIND_RECIPE, datetime.date(2026, 10, 16), "deadbeef")
    assert name == "recipe:20261016:deadbeef"
    parsed = parse_document_name(name)
    assert parsed.kind == KIND_RECIPE
    assert parsed.date == datetime.date(2026, 10, 16)


def test_a_session_archive_name_round_trips():
    name = document_name(KIND_SESSION_ARCHIVE, datetime.date(2026, 10, 16), "abc")
    assert name.startswith("session_archive:")
    assert parse_document_name(name).kind == KIND_SESSION_ARCHIVE


def test_is_knowledge_document_recognizes_plugin_documents_only():
    assert is_knowledge_document("recipe:20261016:deadbeef")
    assert not is_knowledge_document("recipe:notadate:deadbeef")
    assert not is_knowledge_document("my-notes.md")
    assert not is_knowledge_document("recipe:20261016")  # three parts required


def test_text_without_a_header_yields_empty_meta_and_the_text():
    meta, body = parse_document_text("just some plain text")
    assert meta.kind == ""
    assert body == "just some plain text"


def test_special_characters_in_a_signature_are_escaped():
    text = build_document_text(KIND_RECIPE, "body", signature='has "quotes"')
    meta, _ = parse_document_text(text)
    assert meta.signature  # the quote did not end the attribute early


# ------------------------------------------------- the engine's own name form -
# ``add_documents_from_list_of_doc_as_dict`` builds the stored path as
# ``<filepath> + ".input_text"`` and ``add_document_from_dict`` stores
# ``Document.name = os.path.basename(that)``, so the name ``list_document_details``
# serves is the built name plus the suffix.  Retention only ever sees that form.


def test_the_suffixed_name_the_engine_serves_parses_the_same():
    built = document_name(KIND_RECIPE, DAY, "abc123")
    served = _engine_name(built)
    assert served == "recipe:20261016:abc123.input_text"
    assert parse_document_name(served) == parse_document_name(built)
    assert parse_document_name(served).kind == KIND_RECIPE
    assert parse_document_name(served).date == DAY


def test_retention_recognises_and_ages_the_suffixed_name():
    """A document is only deleted if the listed name is still recognised."""
    old = _engine_name(document_name(KIND_SESSION_ARCHIVE, DAY, "abc"))
    assert is_knowledge_document(old)
    assert parse_document_name(old).date < datetime.date.today()
    # An operator's own file stays untouched, suffix or not.
    assert not is_knowledge_document("runbook.md" + ENGINE_NAME_SUFFIX)
    assert not is_knowledge_document("notes:20261016.input_text")


def test_a_fingerprint_with_a_slash_cannot_lose_the_kind():
    """
    ``basename`` would cut a name at the last slash, kind included.

    A fingerprint of ``webapp/abc123`` arrives as ``abc123.input_text`` — no
    kind, no date, and a document that never expires.  The slash is therefore
    not allowed into the name in the first place.
    """
    name = document_name(KIND_RECIPE, DAY, "webapp/abc123")
    assert "/" not in name
    assert _engine_name(name) == name + ENGINE_NAME_SUFFIX
    assert parse_document_name(_engine_name(name)).kind == KIND_RECIPE
    assert is_knowledge_document(_engine_name(name))


def test_a_fingerprint_cannot_add_a_field_or_be_truncated():
    """A name is three fields with no slash in it, whatever it was told."""
    for fingerprint in ("a:b:c", ":", "..", "  ", "a b\\c", "%2f%2e", "a/b/c"):
        name = document_name(KIND_RECIPE, DAY, fingerprint)
        assert name.count(":") == 2, fingerprint
        assert "/" not in name, fingerprint
        meta = parse_document_name(_engine_name(name))
        assert meta.kind == KIND_RECIPE, fingerprint
        assert meta.date == DAY, fingerprint


def test_distinct_fingerprints_keep_distinct_names_after_basename():
    """
    ``basename`` is a no-op on a built name, so it cannot merge two documents.

    Two episodes sharing a day and a kind must land in two documents; if the
    stored name were the tail of a longer path, ``a/x`` and ``b/x`` would index
    as one and overwrite each other.
    """
    built = [
        document_name(KIND_RECIPE, DAY, "webapp/aa"),
        document_name(KIND_RECIPE, DAY, "libweb/bb"),
        document_name(KIND_RECIPE, DAY, "aa"),
    ]
    stored = [_engine_name(name) for name in built]
    assert len(set(stored)) == 3
    assert all(
        name == _engine_name(name)[: -len(ENGINE_NAME_SUFFIX)] for name in built
    )


def test_an_old_name_whose_fingerprint_holds_a_separator_still_parses():
    """A document written before the fingerprint was sanitized is not orphaned."""
    legacy = "recipe:20261016:webapp/abc123"
    meta = parse_document_name(legacy)
    assert meta.kind == KIND_RECIPE
    assert meta.date == DAY
    assert is_knowledge_document(legacy)
