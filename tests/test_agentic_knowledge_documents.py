"""
The document codec: a name and a header that survive the engine's round trip.

The search engine returns a document's name and text but never its metadata,
so a knowledge document carries its kind, retention day, failure signature and
paths in the name and in a one-line header inside the indexed text.  A text
without a header (indexed elsewhere, or a mid-document chunk) parses to an
empty metadata and the text itself, so a reader always gets usable content.
"""

import sys
import pathlib
import datetime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from llm_router_plugins.utils.knowledge.agentic_knowledge.documents import (
    KIND_RECIPE,
    KIND_SESSION_ARCHIVE,
    build_document_text,
    document_name,
    is_knowledge_document,
    parse_document_name,
    parse_document_text,
)


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
