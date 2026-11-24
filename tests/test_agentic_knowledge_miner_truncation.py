"""
The quote budget of the miner: what fits, what is cut, and what is said.

A document whose diff did not fit used to end mid-hunk and read as complete.
These tests pin the replacement: source files are quoted before bookkeeping,
a cut lands at a file boundary and never inside a hunk, every omitted file is
named in a marker, the kinds that carry a whole task quote more than a single
failure's fix, and every cut leaves a counter behind.
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agentic_knowledge_support import (
    FAILING,
    GREEN,
    _command,
    _command_output,
    codex_payload,
)

from llm_router_plugins.utils.knowledge.agentic_knowledge.adapters.codex import (
    CodexAgentAdapter,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.documents import (
    KIND_PLAYBOOK,
    KIND_RECIPE,
    parse_document_text,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.miner import (
    EpisodeMiner,
    _cut_patch_at_header,
    _is_source_patch,
)

_OMITTED_MARKER = "[... quote ends here; not quoted:"
_BODY_MARKER = "was cut at a file boundary"


def envelope(entries):
    """An apply_patch envelope whose file sections carry *entries* fillers."""
    body = "*** Begin Patch\n"
    for path, lines in entries:
        body += f"*** Update File: {path}\n@@\n"
        body += "".join(
            f"+filler line {index} for {path}\n" for index in range(lines)
        )
    body += "*** End Patch"
    return body


def patch_item(call_id, event_id, text):
    return {
        "type": "function_call",
        "name": "apply_patch",
        "call_id": call_id,
        "id": event_id,
        "arguments": text,
    }


def procedure_payload(entries):
    """A change its test run accepted — the playbook shape — one patch."""
    return [
        patch_item("p1", "pe", envelope(entries)),
        _command("c1", "e1", "pytest tests/test_a.py"),
        _command_output("c1", "oe", GREEN, code=0),
    ]


def failure_payload(entries):
    """A failure, the patch, and the green run — the recipe shape."""
    return [
        _command("c0", "e0", "pytest tests/test_a.py"),
        _command_output("c0", "o0", FAILING, code=1),
        patch_item("p1", "pe", envelope(entries)),
        _command("c1", "e1", "pytest tests/test_a.py"),
        _command_output("c1", "oe", GREEN, code=0),
    ]


def mine(payload, **miner_kwargs):
    return EpisodeMiner(**miner_kwargs).mine(
        CodexAgentAdapter().parse(codex_payload(payload))
    )


def playbook_body(events):
    event = next(e for e in events if e.kind == KIND_PLAYBOOK)
    return parse_document_text(event.text)[1]


# ------------------------------------------------- the cut and its markers --


def test_an_omitted_file_is_named_in_the_quote():
    """What does not fit is listed, so a partial document never reads whole."""
    body = playbook_body(
        mine(
            procedure_payload(
                [("src/a.py", 12), ("src/b.py", 12), ("src/c.py", 12)]
            ),
            max_patch_chars=750,
            multifile_patch_budget=750,
        ).events
    )
    assert _OMITTED_MARKER in body
    # The budget spent first on the files it reached; the rest named.
    assert "src/a.py" in body.split(_OMITTED_MARKER)[0]
    assert "src/c.py" in body.split(_OMITTED_MARKER)[1]


def test_source_files_are_quoted_before_generated_ones():
    """When the budget runs out it runs out on bookkeeping, not on the fix."""
    events = mine(
        procedure_payload([("dist/bundle.js", 12), ("src/app.py", 12)]),
        max_patch_chars=500,
        multifile_patch_budget=500,
    ).events
    body = playbook_body(events)
    quoted, _, omitted = body.partition(_OMITTED_MARKER)
    assert "src/app.py" in quoted
    assert "dist/bundle.js" in omitted


def test_a_huge_single_file_is_cut_at_a_file_boundary_and_says_so():
    """One oversized patch is cut whole-section, with a marker, never mid-hunk."""
    body = playbook_body(
        mine(
            procedure_payload([("src/big.py", 60)]),
            max_patch_chars=700,
            multifile_patch_budget=700,
        ).events
    )
    assert _BODY_MARKER in body
    kept = body.split(_BODY_MARKER)[0]
    assert "Update File: src/big.py" in kept
    # Cut mid-file at a hunk boundary: some fillers in, the rest not, no
    # second file header to split.
    assert "+filler" in kept
    assert kept.count("+filler") < 60


def test_cut_patch_at_header_never_splits_a_section():
    """The helper's own contract: the result ends at a header, or is whole."""
    text = envelope([("a.py", 4), ("b.py", 4), ("c.py", 4)])
    cut = _cut_patch_at_header(text, 200)
    assert len(cut) <= 200
    assert "Update File: a.py" in cut
    assert "Update File: c.py" not in cut
    sections_kept = cut.count("*** Update File:")
    assert 1 <= sections_kept < 3
    short = envelope([("a.py", 2)])
    assert _cut_patch_at_header(short, 10_000) == short


def test_a_headerless_body_is_cut_at_the_ceiling():
    """A raw hunk with no envelope has no boundary to honour — hard cut."""
    raw = "x" * 500
    assert _cut_patch_at_header(raw, 100) == "x" * 100


def test_nothing_is_cut_when_everything_fits():
    """The honest budget: no marker, no counter, quote as is."""
    events = mine(procedure_payload([("src/a.py", 3)]), max_patch_chars=4000).events
    body = playbook_body(events)
    assert _OMITTED_MARKER not in body
    assert _BODY_MARKER not in body
    counts = dict(d for d in mine(procedure_payload([("src/a.py", 3)])).diagnostics)
    assert counts.get("truncated") == 0


# ----------------------------------------------------- budgets per kind ----


def big_entries(count=10, lines=16):
    return [(f"src/mod{i}.py", lines) for i in range(count)]


def test_the_whole_task_kinds_quote_more_than_a_recipe():
    """A change the size of a task fits a playbook but not a recipe quote."""
    entries = big_entries()  # well over 4000, under 8000 characters
    playbook = playbook_body(mine(procedure_payload(entries)).events)
    assert _OMITTED_MARKER not in playbook
    assert _BODY_MARKER not in playbook

    recipe_events = mine(failure_payload(entries)).events
    recipe = next(
        parse_document_text(e.text)[1]
        for e in recipe_events
        if e.kind == KIND_RECIPE
    )
    cut = _OMITTED_MARKER in recipe or _BODY_MARKER in recipe
    assert cut
    assert len(recipe) < len(playbook)


def test_an_explicit_larger_ceiling_is_honoured_for_recipes():
    """The operator's number still owns the recipe budget when it is bigger."""
    entries = big_entries()
    events = mine(failure_payload(entries), max_patch_chars=9000).events
    recipe = next(
        parse_document_text(e.text)[1] for e in events if e.kind == KIND_RECIPE
    )
    assert _OMITTED_MARKER not in recipe
    assert _BODY_MARKER not in recipe


# --------------------------------------------------------- diagnostics -----


def test_diagnostics_count_the_truncated_documents():
    result = mine(
        procedure_payload([("src/a.py", 10), ("src/b.py", 10)]),
        max_patch_chars=300,
        multifile_patch_budget=300,
    )
    counts = dict(result.diagnostics)
    assert counts.get("published") == 1
    assert counts.get("truncated") == 1


# ------------------------------------------------------ source ranking -----


def test_generated_paths_sort_as_bookkeeping():
    assert _is_source_patch(envelope([("src/app.py", 1)])) is True
    assert _is_source_patch(envelope([("dist/bundle.js", 1)])) is False
    assert _is_source_patch(envelope([("a.lock", 1), ("b.png", 1)])) is False
    # A nameless hunk is treated as source: nothing marks it otherwise.
    assert _is_source_patch("@@\n+x = 1\n") is True
