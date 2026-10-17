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
    """

    kind: str = ""
    date: Optional[datetime.date] = None
    signature: str = ""
    paths: Tuple[str, ...] = ()
    thread_id: str = ""
    window_id: str = ""
    position: int = -1
    agent: str = ""
    mode: str = ""
    test_command: str = ""
    attempts: int = 0
    source: str = ""


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
    paths: Sequence[str] = (),
    thread_id: str = "",
    window_id: str = "",
    position: Optional[int] = None,
    agent: str = "",
    mode: str = "",
    test_command: str = "",
    attempts: int = 0,
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
    paths : Sequence[str], optional
        Repository-relative paths the document speaks about.
    thread_id, window_id : str, optional
        Identity of an archived window, for session fragments only.
    position : int, optional
        Order of the fragment inside its window.
    agent, mode : str, optional
        Who produced the episode and in which work mode.
    test_command : str, optional
        Command that decided the outcome.
    attempts : int, optional
        Times the same failure was observed.

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
        ("thread_id", "thread"),
        ("window_id", "window"),
        ("agent", "agent"),
        ("mode", "mode"),
        ("test_command", "test_command"),
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

    Parameters
    ----------
    kind : str
        Document kind.
    date : datetime.date
        Publication day.
    fingerprint : str
        Stable digest of the episode the document describes.

    Returns
    -------
    str
        ``<kind>:<YYYYMMDD>:<fingerprint>``.
    """
    day = date.strftime(_NAME_DATE)
    return _NAME_SEPARATOR.join((kind or KIND_DOC, day, fingerprint or "-"))


def parse_document_name(name: str) -> DocumentMeta:
    """
    Read back what :func:`document_name` wrote.

    Unknown or legacy names resolve to an empty metadata with whatever kind
    prefix could be recognized, never to an error.

    Parameters
    ----------
    name : str
        Document name as served by the search.

    Returns
    -------
    DocumentMeta
        At most :attr:`DocumentMeta.kind` and :attr:`DocumentMeta.date`.
    """
    parts = (name or "").split(_NAME_SEPARATOR)
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
    operator indexed themselves into the same collection.

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
