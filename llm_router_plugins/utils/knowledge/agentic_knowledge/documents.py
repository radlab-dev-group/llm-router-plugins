"""
Codec of one knowledge document: its name and its embedded header.

The Semantic Search Engine serves document *names* with every search hit but
never the document's own metadata (``sse_lib.models.SearchHit`` has no metadata
field, and ``list_document_details`` reports only ``id``, ``name``,
``use_in_search`` and ``indexing_status``).  A knowledge document therefore
carries what the reader needs twice:

* in its **name** — ``<kind>:<YYYYMMDD>:<fingerprint>`` — which yields the type
  for the section layout and the date bucket the retention pass deletes by;
* in a **header line** inside the indexed text — kind, date, error signature,
  touched paths, thread and window identifiers — which yields the fields a name
  cannot hold without becoming unreadable.

What the engine reports back as a name is not byte-identical to what
:func:`document_name` returned: an indexed text is stored as
``Document.name = os.path.basename(<filepath>) + ENGINE_NAME_SUFFIX``
(``sse_api.engine.controllers.database.relational_db``), while its
``relative_path`` is the path the plugin sent, without that suffix.  The codec
is written for both name forms — see :func:`parse_document_name` — and
:func:`document_name` keeps a name to a character set that survives the
``basename``.

The header is one line so a chunk that starts at the document start still
carries it; a chunk cut out of the middle falls back to the fields in the name.

Example
-------
::

    text = build_document_text("recipe", "fix …", date=date(2026, 10, 16),
                               signature="assertion failed on empty list")
    name = document_name("recipe", date(2026, 10, 16), "9f2c1a7b")
    meta, body = parse_document_text(text)
    meta.kind, meta.signature  # "recipe", "assertion failed on empty list"
"""

import datetime
import re

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

#: A verified fix: this error, this change, this green test run.
KIND_RECIPE = "recipe"

#: A change that did not fix the error — recorded so it is not repeated.
KIND_DEADEND = "deadend"

#: A fragment of a conversation window archived around compaction.
KIND_SESSION_ARCHIVE = "session_archive"

#: Curated documentation or conventions; never produced by the miner.
KIND_DOC = "doc"

#: Kinds the miner may publish.
KNOWLEDGE_KINDS: Tuple[str, ...] = (KIND_RECIPE, KIND_DEADEND, KIND_SESSION_ARCHIVE)

#: All kinds a reader may ask for.
DOCUMENT_KINDS: Tuple[str, ...] = KNOWLEDGE_KINDS + (KIND_DOC,)

#: Tag of the header line embedded in the indexed text.
HEADER_TAG = "agentic-kb"

#: Separator of the fields of a document name.
_NAME_SEPARATOR = ":"

#: Format of the date bucket inside a document name.
_NAME_DATE = "%Y%m%d"

#: What the engine appends to the name of an indexed text.  A name read back
#: from ``list_document_details`` or from a search hit carries it; the name the
#: plugin built does not.
ENGINE_NAME_SUFFIX = ".input_text"

#: Characters a name field may not hold.  ``/`` would let ``basename`` cut the
#: kind off the name, ``:`` would add a field the codec does not expect, and a
#: space or a control character has nothing to do in an identifier.
_UNSAFE_FIELD = re.compile(r"[^A-Za-z0-9._-]+")

_HEADER_RE = re.compile(
    rf"^<{HEADER_TAG}[ \t]+(?P<fields>.*?)[ \t]*/>[ \t]*\r?\n?",
    re.DOTALL,
)
_FIELD_RE = re.compile(r'(?P<key>[a-z_]+)="(?P<value>[^"]*)"')
_KIND_IN_NAME_RE = re.compile(r"^(?P<kind>[a-z_]+)$")


@dataclass(frozen=True)
class DocumentMeta:
    """
    What a knowledge document says about itself.

    Parameters
    ----------
    kind : str
        One of :data:`DOCUMENT_KINDS`, or ``""`` when the document carries no
        header and its name does not resolve to a known kind.
    date : datetime.date or None
        Day the document was published; the retention pass compares it with
        the configured TTL.
    signature : str
        Normalized failure signature the document is about, when it has one.
        Readable, and never compared: the comparison happens on
        :attr:`signature_hash`, because normalizing an already normalized
        signature does not reproduce it.
    signature_hash : str
        Digest of the signature the document is about; what a reader matches
        its own failure against.
    paths : Tuple[str, ...]
        Repository-relative paths the document speaks about.
    thread_id : str
        Conversation thread of a session archive fragment; ``""`` otherwise.
    window_id : str
        Context window the fragment was archived from.
    position : int
        Order of the fragment inside its window; ``-1`` when unspecified.
    agent : str
        Agent that produced the document.
    mode : str
        Work mode the episode happened in.
    test_command : str
        Command that decided the outcome of a recipe or dead end.
    attempts : int
        How many times the same failure was seen; ``0`` when unspecified.
    source : str
        Name the fragment came from, for the rendered citation.
    doc_type : str
        Kind of documentation file — ``readme``, ``guide``, ``changelog`` …;
        ``""`` for a document the miner produced.
    heading : str
        Heading path a documentation section sits under, for the citation.
    """

    kind: str = ""
    date: Optional[datetime.date] = None
    signature: str = ""
    signature_hash: str = ""
    paths: Tuple[str, ...] = ()
    thread_id: str = ""
    window_id: str = ""
    position: int = -1
    agent: str = ""
    mode: str = ""
    test_command: str = ""
    attempts: int = 0
    source: str = ""
    phase: str = ""
    doc_type: str = ""
    heading: str = ""


def _attribute(key: str, value: str) -> str:
    """Render one header attribute, escaping what would end it early."""
    text = (value or "").replace('"', "'").replace("\n", " ").replace("\r", " ")
    return f'{key}="{text}"'


def build_document_text(
    kind: str,
    body: str,
    *,
    date: Optional[datetime.date] = None,
    signature: str = "",
    signature_hash: str = "",
    paths: Sequence[str] = (),
    thread_id: str = "",
    window_id: str = "",
    position: Optional[int] = None,
    agent: str = "",
    mode: str = "",
    phase: str = "",
    test_command: str = "",
    attempts: int = 0,
    source: str = "",
    doc_type: str = "",
    heading: str = "",
) -> str:
    """
    Return the text to index: one header line followed by *body*.

    Parameters
    ----------
    kind : str
        Document kind; one of :data:`DOCUMENT_KINDS` or :data:`KIND_DOC`.
    body : str
        The knowledge itself, already reduced to what a model needs.
    date : datetime.date, optional
        Publication day; defaults to today.
    signature : str, optional
        Normalized failure signature the document is about.
    signature_hash : str, optional
        Digest of :attr:`signature`.  This is what a reader matches on, so it
        survives anything that rewrites the signature's text.
    paths : Sequence[str], optional
        Repository-relative paths the document speaks about.
    thread_id, window_id : str, optional
        Identity of an archived window, for session fragments only.
    position : int, optional
        Order of the fragment inside its window.
    agent, mode : str, optional
        Who produced the episode and in which work mode.
    phase : str, optional
        Work phase the episode was mined in; a subset of the turn's mode.
    test_command : str, optional
        Command that decided the outcome.
    attempts : int, optional
        Times the same failure was observed.
    source : str, optional
        File a documentation section came from, repository-relative.
    doc_type : str, optional
        What kind of documentation file that is; see
        :func:`.docs_ingestor.classify_document`.
    heading : str, optional
        Heading path the section sits under, joined with ``>``.

    Returns
    -------
    str
        Header line plus the body.
    """
    day = date or datetime.date.today()
    fields = [
        _attribute("kind", kind),
        _attribute("date", day.isoformat()),
    ]
    if signature:
        fields.append(_attribute("signature", signature))
    if signature_hash:
        fields.append(_attribute("sig_hash", signature_hash))
    if paths:
        fields.append(_attribute("paths", ",".join(paths)))
    if thread_id:
        fields.append(_attribute("thread", thread_id))
    if window_id:
        fields.append(_attribute("window", window_id))
    if position is not None:
        fields.append(_attribute("position", str(position)))
    if agent:
        fields.append(_attribute("agent", agent))
    if mode:
        fields.append(_attribute("mode", mode))
    if phase:
        fields.append(_attribute("phase", phase))
    if source:
        fields.append(_attribute("src", source))
    if doc_type:
        fields.append(_attribute("doc_type", doc_type))
    if heading:
        fields.append(_attribute("heading", heading))
    if test_command:
        fields.append(_attribute("test_command", test_command))
    if attempts:
        fields.append(_attribute("attempts", str(attempts)))
    header = f"<{HEADER_TAG} {' '.join(fields)}/>"
    return f"{header}\n{(body or '').strip()}\n"


def parse_document_text(text: str) -> Tuple[DocumentMeta, str]:
    """
    Split *text* into the header it carries and the knowledge body.

    A text without a header — one indexed outside this plugin, or a chunk cut
    out of the middle of a document — yields an empty :class:`DocumentMeta`
    and the text itself, so the caller always gets usable content.

    Parameters
    ----------
    text : str
        The retrieved chunk.

    Returns
    -------
    Tuple[DocumentMeta, str]
        The declared metadata (empty when there is no header) and the body.
    """
    body = text or ""
    match = _HEADER_RE.match(body)
    if match is None:
        return DocumentMeta(), body
    fields = {}
    for field in _FIELD_RE.finditer(match.group("fields")):
        fields[field.group("key")] = field.group("value")
    return DocumentMeta(**_meta_arguments(fields)), body[match.end() :]


def document_body(text: str) -> str:
    """
    Return *text* with the header line it carries taken off.

    The counterpart of :func:`build_document_text`: a writer that has to mask
    the knowledge — the indexing worker, for one — strips the header here,
    masks what remains, and rebuilds the header from its own structured fields,
    so a masker never rewrites a value a reader later parses.

    Parameters
    ----------
    text : str
        A document text, with or without a header.

    Returns
    -------
    str
        The knowledge body; *text* itself when it carries no header.
    """
    body = text or ""
    match = _HEADER_RE.match(body)
    return body[match.end() :] if match else body


def _meta_arguments(fields: dict) -> dict:
    """Turn parsed header attributes into :class:`DocumentMeta` arguments."""
    arguments: dict = {}
    if fields.get("kind"):
        arguments["kind"] = fields["kind"]
    day = _parse_date(fields.get("date", ""))
    if day is not None:
        arguments["date"] = day
    for target, source in (
        ("signature", "signature"),
        ("signature_hash", "sig_hash"),
        ("thread_id", "thread"),
        ("window_id", "window"),
        ("agent", "agent"),
        ("mode", "mode"),
        ("test_command", "test_command"),
        ("phase", "phase"),
        ("source", "src"),
        ("doc_type", "doc_type"),
        ("heading", "heading"),
    ):
        if fields.get(source):
            arguments[target] = fields[source]
    if fields.get("paths"):
        arguments["paths"] = tuple(
            path for path in fields["paths"].split(",") if path
        )
    if fields.get("position"):
        arguments["position"] = _whole(fields["position"])
    if fields.get("attempts"):
        arguments["attempts"] = _whole(fields["attempts"])
    return arguments


def _whole(value: str) -> int:
    """Parse an integer attribute, falling back to ``0``."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _parse_date(value: str) -> Optional[datetime.date]:
    """Parse an ISO day, or ``None`` when the value is not one."""
    if not value:
        return None
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        return None


def document_name(kind: str, date: datetime.date, fingerprint: str) -> str:
    """
    Return the SSE document name of one knowledge entry.

    The name is the only part of a document the search always reports, so it
    carries the kind and the retention day bucket next to the fingerprint.

    Two engine behaviours decide what may go into it.  The engine stores the
    name as ``os.path.basename(<filepath>) + ENGINE_NAME_SUFFIX``, so the name
    a reader sees carries a suffix this function does not add.  And
    ``basename`` throws away everything up to the last ``/``: a fingerprint
    holding a slash would arrive as its last segment alone, kind and date
    bucket gone with the front of it.  A ``:`` in the fingerprint would turn
    the three fields the codec splits on into four.  Both are replaced here, so
    a name always reaches the reader as exactly three fields in the agreed
    order — and two episodes whose raw identifiers differ cannot collide by
    one of them dissolving.

    Parameters
    ----------
    kind : str
        Document kind; one of :data:`DOCUMENT_KINDS`.
    date : datetime.date
        Publication day.
    fingerprint : str
        Stable digest of the episode the document describes; made safe here.

    Returns
    -------
    str
        ``<kind>:<YYYYMMDD>:<fingerprint>``, without the suffix the engine will
        append to it.
    """
    day = date.strftime(_NAME_DATE)
    return _NAME_SEPARATOR.join(
        (
            _safe_field(kind, KIND_DOC),
            day,
            _safe_field(fingerprint, "-"),
        )
    )


def _safe_field(value: str, fallback: str) -> str:
    """Reduce one name field to characters that survive ``basename`` and ``:``."""
    text = _UNSAFE_FIELD.sub("-", str(value or "")).strip("-.")
    return text or fallback


def parse_document_name(name: str) -> DocumentMeta:
    """
    Read back what :func:`document_name` wrote.

    Both spellings of a name parse to the same metadata: the bare name the
    plugin built (``recipe:20261016:abc123``) and the one the engine serves for
    an indexed text (``recipe:20261016:abc123.input_text``, the same three
    fields with :data:`ENGINE_NAME_SUFFIX` hanging off the last).  The suffix
    needs no special case — it falls inside the fingerprint field, which
    retention never reads — but it is why the split is bounded to two cuts:
    the fingerprint is the field allowed to be opaque, and the kind and the day
    are always the first two.  A fingerprint that still holds a separator, from
    a document written before :func:`document_name` sanitized it, then lands
    whole in the third field instead of breaking the shape and hiding the
    document from retention for good.  Retention lists documents through
    ``list_document_details``, which only ever serves the suffixed form, so
    this is what makes it recognise its own documents.

    Unknown or legacy names resolve to an empty metadata with whatever kind
    prefix could be recognized, never to an error.

    Parameters
    ----------
    name : str
        Document name as served by the search or by ``list_document_details``.

    Returns
    -------
    DocumentMeta
        At most :attr:`DocumentMeta.kind` and :attr:`DocumentMeta.date`.
    """
    parts = (name or "").split(_NAME_SEPARATOR, 2)
    if len(parts) != 3:
        return DocumentMeta()
    kind, day, _ = parts
    if not _KIND_IN_NAME_RE.match(kind):
        return DocumentMeta()
    date = _parse_date(day)
    if date is None and len(day) == 8:
        try:
            date = datetime.datetime.strptime(day, _NAME_DATE).date()
        except ValueError:
            date = None
    return DocumentMeta(kind=kind, date=date)


def is_knowledge_document(name: str) -> bool:
    """
    Whether *name* was produced by this plugin.

    Only used by the retention pass, which must never delete documents an
    operator indexed themselves into the same collection.  The names it passes
    here come from ``list_document_details`` and therefore carry
    :data:`ENGINE_NAME_SUFFIX`; :func:`parse_document_name` accepts both that
    and the bare form, so a document is aged whether it was listed by the
    engine or named by the plugin.

    Parameters
    ----------
    name : str
        Document name as served by ``list_document_details``.

    Returns
    -------
    bool
        ``True`` when the name has the plugin's three-part shape and a known
        kind.
    """
    meta = parse_document_name(name)
    return meta.kind in DOCUMENT_KINDS and meta.date is not None
