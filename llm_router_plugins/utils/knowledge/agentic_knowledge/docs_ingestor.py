"""
Indexing the repository's own documentation as retrievable knowledge.

A recipe is something an agent learned; documentation is something it walked in
knowing, and knew only because a human had written it down.  The layer has read
the ``doc`` category from the start — ``plan``, ``implement`` and ``review`` ask
for it — while the miner never produces one, so the category stayed empty and
the conventions section rendered nothing.

This module fills it from the side the proxy can see.  Every request declares
the directory the agent works in; where the proxy runs on that machine, the
repository's markdown is the cheapest, freshest, most project-specific
knowledge available, and someone has already written it.

Three rules shape the implementation.

* **A file is one document.**  The file is read whole and published to the
  stream the miner already uses; the engine splits it into chunks with the
  collection's own chunking, and the worker masks and indexes it exactly as it
  masks a recipe, so documentation inherits the privacy properties of
  everything else in the knowledge base.  A heading is a citation, not a cut:
  the document's title travels in its header, and a chunk cut out of the middle
  falls back to the fields in the document name.
* **A file is republished when its content changes**, not when it is looked at.
  The digest of the whole file is the document's fingerprint and the scan state
  compares digests, so an unchanged file costs one hash and no embedding, and a
  ``git checkout`` across ten branches publishes nothing.
* **A scan that did not see the whole repository prunes nothing.**  Replacing a
  document is only safe against a complete listing, so a scan that ran out of
  ``docs.max_files`` reports itself truncated and leaves the old documents in
  place.

What a scan knows — which file is currently indexed under which document names
— is kept in Redis, one key per scope, as a cache of what the engine holds.
Losing it costs one republication of that repository's documentation.
"""

import datetime
import fnmatch
import hashlib
import json
import os
import posixpath
import re
import threading
import time

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from llm_router_plugins.utils.knowledge.agentic_knowledge.adapters.base import (
    KnowledgeEvent,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.doc_context import (
    DocContext,
    DocContextStore,
    build_doc_context,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
    DEFAULT_DOC_EXCLUDED_DIRS,
    DEFAULT_DOC_PATTERNS,
    AgenticKnowledgeConfig,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.documents import (
    KIND_DOC,
    build_document_text,
    document_name,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.events import (
    KnowledgeEventPublisher,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.redisclient import (
    RedisClientHandle,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.sse import (
    STATUS_DISABLED,
    STATUS_OK,
    STATUS_UNAVAILABLE,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.state import (
    RedisConnectionSettings,
    fingerprint as digest,
)

__all__ = [
    "CONTROL_DOCS_PRUNE",
    "DocRecord",
    "DocsIndexer",
    "DocsScanState",
    "ScanOutcome",
    "ScanReport",
    "classify_document",
    "doc_events",
    "document_fingerprint",
    "prune_event",
    "scan_repository",
]

#: How the indexer knows it has no standing digest to serve yet.
_NO_CONTEXT = None

#: Stream message that asks the worker to forget documents this scope no longer
#: has.  A control message rather than a document: it is never indexed, and a
#: worker older than this module indexes it as an empty document, which is
#: harmless and self-correcting once a current worker runs.
CONTROL_DOCS_PRUNE = "docs_prune"

#: A scan ran and published what it found.
STATUS_SKIPPED = "skipped"

#: The heading path is quoted into the document header; a heading longer than
#: this is a paragraph wearing a hash.
_MAX_HEADING_CHARS = 160

#: How deep below the declared directory a scan descends.  Deep enough for any
#: documentation tree, shallow enough that a symlink farm or a generated SDK
#: cannot turn one request into a walk of a filesystem.
_MAX_SCAN_DEPTH = 12

#: Layout version of a scope's scan state, and the shape of its payload.
#: Version ``2`` added the ``collection`` the listing was written for.
_STATE_VERSION = 2

#: Never descended into, whatever the policy says.  A version control directory
#: is not documentation and a scan of one reads history the knowledge base has
#: no business holding; an operator's ``docs.excluded_dirs`` is a list added to
#: this floor, not a replacement for it.
_ALWAYS_EXCLUDED = frozenset({".git", ".hg", ".svn"})

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.*\S)[ \t]*$")
_FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})")

#: Basename (lowercased, suffix removed) to documentation type.  The type is a
#: policy handle, not a claim about content: it lets a mode read the conventions
#: without reading the changelog.
_DOC_TYPE_NAMES: Tuple[Tuple[str, str], ...] = (
    ("readme", "readme"),
    ("changelog", "changelog"),
    ("changes", "changelog"),
    ("news", "changelog"),
    ("history", "changelog"),
    ("release-notes", "changelog"),
    ("contributing", "contributing"),
    ("code-of-conduct", "contributing"),
    ("license", "license"),
    ("licence", "license"),
    ("copying", "license"),
    ("notice", "license"),
    ("security", "security"),
    ("agents", "agent"),
    ("claude", "agent"),
    ("gemini", "agent"),
    ("cursorrules", "agent"),
    ("codex", "agent"),
)

#: Top-level directories whose contents are guides rather than notes.
_GUIDE_DIRECTORIES = ("docs", "doc", "wiki", "documentation", "guides", "book")


@dataclass(frozen=True)
class DocRecord:
    """
    One documentation file, ready to become one document.

    Parameters
    ----------
    source_file : str
        Repository-relative path, with POSIX separators.
    doc_type : str
        What kind of file it is; see :func:`classify_document`.
    heading : str
        The document's title: its first heading, ``""`` for a file that opens
        without one.  A citation, not a cut — the engine chunks the text.
    position : int
        Always ``0``; one file is one document, and the field is kept so an
        older event on the stream still decodes.
    text : str
        The whole file, headings included.
    content_hash : str
        Digest of the **whole file**; the document's fingerprint is built on it,
        so one changed line marks that file for republication and only that
        file's.
    fingerprint : str
        Identity of the document: scope, file and file digest.  It is the
        publisher's deduplication key and the document name's last field.
    size : int
        File size in bytes, as the scan saw it.
    mtime : float
        File modification second, as the scan saw it.  Recorded for operators;
        republication is decided on :attr:`content_hash`, because a checkout
        rewrites mtimes and must not re-embed a file whose text did not change.
    """

    source_file: str = ""
    doc_type: str = ""
    heading: str = ""
    position: int = 0
    text: str = ""
    content_hash: str = ""
    fingerprint: str = ""
    size: int = 0
    mtime: float = 0.0


@dataclass(frozen=True)
class ScanReport:
    """
    What one walk of a repository found.

    Parameters
    ----------
    records : Tuple[DocRecord, ...]
        One record per file that was read: the file is the document, and the
        engine is what splits it into chunks.
    headings_by_file : Dict[str, Tuple[str, ...], ...]
        The heading outline of each file, in document order, as it was written.
        A heading is a citation and the map a session is given to navigate by,
        not a cut: the whole file is indexed as one document.
    files : int
        Documentation files read.
    skipped : int
        Files that matched but could not be used — unreadable, or above the
        per-file ceiling.
    truncated : bool
        ``True`` when a ceiling stopped the scan, which makes the listing
        partial and therefore useless for pruning.
    reason : str
        Why nothing was scanned at all; ``""`` otherwise.
    """

    records: Tuple[DocRecord, ...] = ()
    headings_by_file: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    files: int = 0
    skipped: int = 0
    truncated: bool = False
    reason: str = ""


@dataclass(frozen=True)
class ScanOutcome:
    """
    What one :meth:`DocsIndexer.index` call did, for the request annotation.

    Parameters
    ----------
    status : str
        :data:`STATUS_OK`, :data:`STATUS_SKIPPED`, :data:`STATUS_DISABLED` or
        :data:`STATUS_UNAVAILABLE`.
    reason : str
        Why not, when it did not run: ``disabled``, ``manual``, ``no scope``,
        ``no root``, ``cooldown``, ``no-such-directory``, ``state``, ``publish``.
        Only the last two are a dependency failure.
    files : int
        Documentation files the repository has.
    published : int
        Section documents handed to the worker.
    pruned : int
        Documents the worker was asked to forget.
    unchanged : int
        Files already indexed with this exact content.
    truncated : bool
        Whether a ceiling cut the scan short.
    context : DocContext, optional
        The standing digest the scan refreshed; ``None`` when documentation is
        off or the scan produced nothing to map.
    """

    status: str = STATUS_DISABLED
    reason: str = ""
    files: int = 0
    published: int = 0
    pruned: int = 0
    unchanged: int = 0
    truncated: bool = False
    context: Optional[DocContext] = None

    def as_dict(self) -> Dict[str, Any]:
        """
        Return the annotation form of this outcome.

        Returns
        -------
        Dict[str, Any]
            Status, counts and reason — no paths, no file names, no content.
        """
        result: Dict[str, Any] = {
            "status": self.status,
            "files": self.files,
            "published": self.published,
            "pruned": self.pruned,
        }
        if self.reason:
            result["reason"] = self.reason
        if self.truncated:
            result["truncated"] = True
        if self.context is not None and not self.context.empty:
            result["context"] = {
                "chars": self.context.chars,
                "files": len(self.context.source_files),
            }
        return result


def classify_document(source_file: str) -> str:
    """
    Return what kind of documentation *source_file* is.

    The name is the only evidence: ``README`` is an introduction, ``CONTRIBUTING``
    is rules for changing the code, ``CHANGELOG`` is history.  Anything under a
    ``docs`` tree is a guide, and the rest is a note.

    Parameters
    ----------
    source_file : str
        Repository-relative path.

    Returns
    -------
    str
        One of ``readme``, ``changelog``, ``contributing``, ``license``,
        ``security``, ``agent``, ``guide``, ``note``.
    """
    path = str(source_file or "").replace("\\", "/")
    name = posixpath.basename(path).lower()
    stem = posixpath.splitext(name)[0]
    for candidate, kind in _DOC_TYPE_NAMES:
        if stem == candidate or name == candidate:
            return kind
    top = path.split("/", 1)[0].lower()
    if top in _GUIDE_DIRECTORIES:
        return "guide"
    return "note"


def scan_repository(
    repo_root: str,
    *,
    patterns: Sequence[str] = DEFAULT_DOC_PATTERNS,
    excluded_dirs: Sequence[str] = DEFAULT_DOC_EXCLUDED_DIRS,
    max_files: int = 200,
    max_file_bytes: int = 200_000,
    max_total_bytes: int = 2_000_000,
    scope: str = "",
) -> ScanReport:
    """
    Read the documentation of the repository rooted at *repo_root*.

    Reads are bounded by the three ceilings, in bytes and files: a scan runs on
    the way to answering a live agent, and a repository with five thousand
    generated markdown files is a job for the operator's own command, not for a
    request.  Hitting a ceiling sets :attr:`ScanReport.truncated`, which is what
    tells the caller that the listing is not the whole repository and that
    pruning against it would delete documents that are still true.

    Symbolic links are neither followed nor read, and a directory named in
    *excluded_dirs* is not descended into.

    Parameters
    ----------
    repo_root : str
        Directory to walk; the request's declared working directory.
    patterns : Sequence[str]
        Filename patterns to pick up, matched case-insensitively.
    excluded_dirs : Sequence[str]
        Directory names to skip at any depth.
    max_files : int
        Most files to read.
    max_file_bytes : int
        Largest file to read; a bigger one is counted as skipped.
    max_total_bytes : int
        Most bytes to read in one scan.
    scope : str
        Repository key the records belong to; part of every fingerprint.

    Returns
    -------
    ScanReport
        One record per documentation file, and what stopped the scan.
    """
    root = os.path.abspath(os.path.expanduser(str(repo_root or "")))
    if not root or not os.path.isdir(root):
        return ScanReport(reason="no-such-directory")
    wanted = tuple(str(pattern).lower() for pattern in patterns if pattern)
    skip = {str(name) for name in excluded_dirs if str(name).strip()}
    skip |= set(_ALWAYS_EXCLUDED)

    records: List[DocRecord] = []
    outlines: Dict[str, Tuple[str, ...]] = {}
    files = 0
    skipped = 0
    total = 0
    truncated = False
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        relative = directory[len(root) :]
        if relative.count(os.sep) >= _MAX_SCAN_DEPTH:
            dirnames[:] = []
        dirnames[:] = sorted(name for name in dirnames if name not in skip)
        for name in sorted(filenames):
            if not _matches(name, wanted):
                continue
            path = os.path.join(directory, name)
            if os.path.islink(path) or not os.path.isfile(path):
                continue
            try:
                info = os.stat(path)
            except OSError:
                skipped += 1
                continue
            if info.st_size > max_file_bytes:
                skipped += 1
                continue
            if files + 1 > int(max_files) or total + info.st_size > int(
                max_total_bytes
            ):
                truncated = True
                break
            text = _read_text(path)
            if text is None:
                skipped += 1
                continue
            files += 1
            total += info.st_size
            outlines[
                posixpath.join(
                    _relative(root, directory), name.replace(os.sep, "/")
                ).lstrip("/")
            ] = document_headings(text)
            records.append(
                _file_record(
                    posixpath.join(
                        _relative(root, directory), name.replace(os.sep, "/")
                    ).lstrip("/"),
                    text,
                    size=info.st_size,
                    mtime=info.st_mtime,
                    scope=scope,
                )
            )
        if truncated:
            break
    return ScanReport(
        records=tuple(records),
        headings_by_file=outlines,
        files=files,
        skipped=skipped,
        truncated=truncated,
    )


def document_headings(text: str) -> Tuple[str, ...]:
    """
    Return the heading outline of one markdown document, in order.

    Every heading, at every level, outside fenced code blocks — the document's
    own table of contents.  Nothing here cuts a document: the file is indexed
    whole and the engine chunks it, so an outline is only ever a map, and a map
    that dropped a heading would be a worse one.
    """
    titles: List[str] = []
    fence = ""
    for line in str(text or "").splitlines():
        opening = _FENCE_RE.match(line)
        if fence:
            if line.strip().startswith(fence):
                fence = ""
            continue
        if opening:
            fence = opening.group(1)[:3]
            continue
        title = _HEADING_RE.match(line)
        if title is None:
            continue
        heading = title.group(2).strip()
        if heading:
            titles.append(heading)
    return tuple(titles)


def _matches(name: str, patterns: Sequence[str]) -> bool:
    """Whether a file name is documentation, by its extension."""
    lowered = name.lower()
    return any(fnmatch.fnmatch(lowered, pattern) for pattern in patterns)


def _relative(root: str, directory: str) -> str:
    """Path of *directory* under *root*, POSIX-style and ``""`` at the root."""
    tail = directory[len(root) :].strip(os.sep)
    return tail.replace(os.sep, "/")


def _read_text(path: str) -> Optional[str]:
    """Read one file as text, or ``None`` when the filesystem says no."""
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError:
        return None
    return raw.decode("utf-8", errors="replace")


def _file_record(
    source_file: str,
    text: str,
    *,
    size: int,
    mtime: float,
    scope: str,
) -> DocRecord:
    """
    Turn one file into the one document it is indexed as.

    The file is a whole: the engine chunks it with the collection's own
    chunking, and a section the author split it into is not a retrieval unit
    this layer owns.  What the record keeps is the identity — scope, file,
    digest — and the title, which a retrieved chunk cites.
    """
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return DocRecord(
        source_file=source_file,
        doc_type=classify_document(source_file),
        heading=_document_title(text),
        position=0,
        text=text,
        content_hash=content_hash,
        fingerprint=document_fingerprint(scope, source_file, content_hash),
        size=size,
        mtime=float(mtime),
    )


def _document_title(text: str) -> str:
    """The first heading of a document; ``""`` for one that opens without one."""
    headings = document_headings(text)
    return str(headings[0])[:_MAX_HEADING_CHARS] if headings else ""


def document_fingerprint(scope: str, source_file: str, content_hash: str) -> str:
    """
    Return the identity of one file's document.

    The scope is in it because two repositories index into two collections and
    the digest must not collide between them; the file digest is in it because
    the fingerprint is also the document name, and a new name is what makes a
    changed file replace itself rather than sit beside itself.

    Parameters
    ----------
    scope : str
        Repository key.
    source_file : str
        Repository-relative path.
    content_hash : str
        Digest of the whole file.

    Returns
    -------
    str
        A short hex digest, stable for equal content.
    """
    return digest("doc", scope, source_file, content_hash)


def doc_events(
    records: Iterable[DocRecord],
    *,
    scope: str,
    agent: str = "",
    model: str = "",
    generation: str = "",
    today: Optional[datetime.date] = None,
    now: Optional[float] = None,
) -> Tuple[KnowledgeEvent, ...]:
    """
    Build the events that carry *records* to the indexing worker.

    The header says what the file is and what it is titled, so a retrieved
    chunk names its source even when the engine cut it out of the middle; the
    metadata repeats the file so the worker can key its own bookkeeping on it.
    No work mode is claimed: documentation is not an episode, and a mode filter
    attached to it would hide a guide from the one mode that needs it.

    Parameters
    ----------
    records : Iterable[DocRecord]
        Files to publish, one document each.
    scope : str
        Repository the documentation belongs to.
    agent : str, optional
        Agent whose request triggered the scan, for the citation.
    model : str, optional
        Model of that request; recorded, never used to filter.
    generation : str, optional
        Command generation the scan ran in.
    today : datetime.date, optional
        Publication day; the document name's retention bucket.
    now : float, optional
        Publication second.

    Returns
    -------
    Tuple[KnowledgeEvent, ...]
        One event per file, kind ``doc``.
    """
    day = today or datetime.date.today()
    stamp = time.time() if now is None else float(now)
    events: List[KnowledgeEvent] = []
    for record in records:
        if not record.text.strip() or not record.fingerprint:
            continue
        # The file goes out exactly as it was written: the engine splits it,
        # so a heading quoted in here twice would only be read twice.
        text = build_document_text(
            KIND_DOC,
            record.text,
            date=day,
            paths=(record.source_file,),
            agent=agent,
            source=record.source_file,
            doc_type=record.doc_type,
            heading=record.heading,
        )
        metadata: Dict[str, Any] = {
            "kind": KIND_DOC,
            "scope": scope,
            "source_file": record.source_file,
            "doc_type": record.doc_type,
            "heading": record.heading,
            "position": int(record.position),
            "content_hash": record.content_hash,
            "paths": [record.source_file],
        }
        events.append(
            KnowledgeEvent(
                kind=KIND_DOC,
                fingerprint=record.fingerprint,
                scope=scope,
                agent=agent,
                mode="",
                model=model,
                generation=generation,
                created_at=stamp,
                text=text,
                metadata=metadata,
            )
        )
    return tuple(events)


def prune_event(
    scope: str,
    names: Iterable[str],
    *,
    agent: str = "",
    generation: str = "",
    now: Optional[float] = None,
) -> Optional[KnowledgeEvent]:
    """
    Build the control message that retires *names*, or ``None`` when empty.

    The names are the ones this plugin gave the documents, which the worker
    matches against the engine's listing — the engine stores a name with its own
    suffix, so asking to delete ``doc:20261021:abc`` retires the document the
    engine calls ``doc:20261021:abc.input_text``.

    Parameters
    ----------
    scope : str
        Repository whose documentation changed.
    names : Iterable[str]
        Document names that are no longer current.
    agent : str, optional
        Agent whose request triggered the scan.
    generation : str, optional
        Command generation the scan ran in.
    now : float, optional
        Publication second.

    Returns
    -------
    KnowledgeEvent or None
        The event, or ``None`` when there was nothing to retire.
    """
    wanted = tuple(sorted({str(name) for name in names if str(name)}))
    if not wanted:
        return None
    stamp = time.time() if now is None else float(now)
    return KnowledgeEvent(
        kind=CONTROL_DOCS_PRUNE,
        fingerprint=digest(CONTROL_DOCS_PRUNE, scope, *wanted),
        scope=scope,
        agent=agent,
        generation=generation,
        created_at=stamp,
        text="",
        metadata={"names": list(wanted), "count": len(wanted)},
    )


class DocsScanState:
    """
    What a scope's documentation currently looks like in the knowledge base.

    One Redis key per scope, holding ``file → {digest, document names, size,
    mtime}``.  It is a cache of the engine's contents, not their source: losing
    it republishes a repository's documentation once, and a stale entry only
    delays a replacement until the digest comparison notices.

    A listing also names the **collection** it was written for.  A policy that
    moves the documentation to another collection — as it moved to
    ``<prefix>_<scope>_docs`` — must not find a listing that claims every file
    is already indexed, because it is indexed in a collection nobody reads: an
    entry naming a different collection is no entry at all, and the next scan
    republishes the repository where it now belongs.

    Parameters
    ----------
    connection : RedisConnectionSettings
        Where the state lives.
    key_prefix : str
        Namespace of the keys.
    ttl_seconds : int
        How long a scope's listing survives without a scan.
    client : Any, optional
        Redis client to use as-is; tests pass a fake.
    logger : Any, optional
        Where failures are reported, without paths.
    """

    def __init__(
        self,
        connection: Optional[RedisConnectionSettings] = None,
        key_prefix: str = "agentic-kb",
        ttl_seconds: int = 7_776_000,
        client: Any = None,
        logger: Any = None,
    ) -> None:
        """Bind the destination; build no connection until it is used."""
        self._handle = RedisClientHandle(connection, client, logger)
        self._key_prefix = key_prefix
        self._ttl = int(ttl_seconds)
        self._logger = logger

    @property
    def available(self) -> bool:
        """Whether the state can be read and written at all."""
        return self._handle.configured

    def key(self, scope: str) -> str:
        """Return the Redis key of one scope's listing."""
        return f"{self._key_prefix}:docs:{scope}"

    def load(self, scope: str, collection: str = "") -> Dict[str, Dict[str, Any]]:
        """
        Return the listing of *scope*; empty when there is none to read.

        Parameters
        ----------
        scope : str
            Repository the listing belongs to.
        collection : str, optional
            The collection the caller is indexing into.  A listing written for
            another one is treated as absent: it describes documents that are
            no longer read from, and trusting it would leave a repository with
            no documentation in the place the reader looks.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            ``file → entry``; empty on a miss, a malformed value, a listing
            written for a different collection, or a Redis that did not answer
            — each of which only means every file will look new, which is the
            safe direction.
        """
        client = self._handle.get()
        if client is None:
            return {}
        try:
            raw = client.get(self.key(scope))
        except Exception as exc:
            self._warn("knowledge docs state unreadable: %s", type(exc).__name__)
            return {}
        if not raw:
            return {}
        try:
            payload = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
        except (TypeError, ValueError):
            return {}
        if not isinstance(payload, dict):
            return {}
        if collection and str(payload.get("collection") or "") != collection:
            return {}
        files = payload.get("files")
        if not isinstance(files, dict):
            return {}
        return {
            str(name): dict(entry)
            for name, entry in files.items()
            if isinstance(entry, dict)
        }

    def save(
        self,
        scope: str,
        entries: Dict[str, Dict[str, Any]],
        collection: str = "",
    ) -> bool:
        """
        Write the listing of *scope*.

        Parameters
        ----------
        scope : str
            Repository the listing belongs to.
        entries : Dict[str, Dict[str, Any]]
            ``file → entry`` for every file the repository now has.
        collection : str, optional
            The collection these documents were indexed into; recorded so a
            later policy that moves them notices.

        Returns
        -------
        bool
            ``True`` when the listing was written; ``False`` means the next scan
            will republish this repository's documentation.
        """
        client = self._handle.get()
        if client is None:
            return False
        payload = json.dumps(
            {"v": _STATE_VERSION, "collection": collection, "files": entries},
            ensure_ascii=False,
        )
        try:
            client.set(self.key(scope), payload, ex=self._ttl)
        except Exception as exc:
            self._warn("knowledge docs state unwritable: %s", type(exc).__name__)
            return False
        return True

    def _warn(self, message: str, *args: Any) -> None:
        """Report one failure when a logger was given."""
        if self._logger is not None:
            self._logger.warning(message, *args)


@dataclass
class DocsIndexer:
    """
    Keep a repository's documentation in the knowledge base.

    The indexer is the whole documentation write path: scan, compare with what
    is already indexed, publish what is new or changed, ask for what is gone to
    be retired, remember the result.  It is called from the request path and
    therefore never raises, never blocks past a Redis timeout, and never reads
    one repository twice within its configured rescan interval.

    Parameters
    ----------
    config : AgenticKnowledgeConfig
        Policy: patterns, ceilings, interval, and whether documentation is part
        of the knowledge base at all.
    publisher : KnowledgeEventPublisher, optional
        Where events go; ``None`` means the write path is off.
    state : DocsScanState, optional
        Where the listing lives; built from the configuration when omitted.
    context_store : DocContextStore, optional
        Where the standing digest lives; built from the configuration when
        omitted.
    logger : Any, optional
        Where progress is reported, without file names.
    clock : callable, optional
        Current second; a seam for the cooldown's tests.
    """

    config: AgenticKnowledgeConfig
    publisher: Optional[KnowledgeEventPublisher] = None
    state: Optional[DocsScanState] = None
    context_store: Optional[DocContextStore] = None
    logger: Any = None
    clock: Any = field(default=time.time, repr=False)

    def __post_init__(self) -> None:
        """Build the state stores when the caller did not pass them."""
        if self.state is None:
            self.state = DocsScanState(
                connection=self.config.redis,
                key_prefix=self.config.key_prefix,
                ttl_seconds=self.config.docs_state_ttl_seconds,
                logger=self.logger,
            )
        if self.context_store is None:
            self.context_store = DocContextStore(
                connection=self.config.redis,
                key_prefix=self.config.key_prefix,
                ttl_seconds=self.config.doc_context_ttl_seconds,
                logger=self.logger,
            )
        self._last_scan: Dict[str, float] = {}
        self._lock = threading.Lock()

    def index(
        self,
        scope: str,
        repo_root: str,
        *,
        agent: str = "",
        model: str = "",
        generation: str = "",
        force: bool = False,
        allow_prune: bool = True,
    ) -> ScanOutcome:
        """
        Bring one repository's documentation up to date.

        Parameters
        ----------
        scope : str
            Repository key; empty means the knowledge base has no collection to
            write into and nothing is read.
        repo_root : str
            The declared working directory, read from this machine.
        agent : str, optional
            Agent whose request triggered the scan.
        model : str, optional
            Model of that request.
        generation : str, optional
            Command generation the scan ran in.
        force : bool, optional
            Ignore the in-process cooldown; the operator's command uses it.
        allow_prune : bool, optional
            ``False`` retires nothing: the scan publishes new and changed files
            and leaves the superseded documents where they are.

        Returns
        -------
        ScanOutcome
            What happened; never an exception.
        """
        if not self.config.docs_enabled or self.publisher is None:
            return ScanOutcome(reason="disabled")
        if not force and not self.config.docs_auto_index:
            # ``docs.auto_index`` off means the operator indexes documentation
            # by hand, so a request must not quietly do it instead.
            return ScanOutcome(status=STATUS_SKIPPED, reason="manual")
        if not scope:
            # Nothing identifies a collection to write into; this is not a
            # failure of the deployment, it is a request with no repository.
            return ScanOutcome(status=STATUS_SKIPPED, reason="no scope")
        if not repo_root:
            return ScanOutcome(status=STATUS_SKIPPED, reason="no root")
        if not force and self._in_cooldown(scope, repo_root):
            return ScanOutcome(status=STATUS_SKIPPED, reason="cooldown")
        self._mark_scanned(scope, repo_root)

        report = scan_repository(
            repo_root,
            patterns=self.config.docs_patterns,
            excluded_dirs=self.config.docs_excluded_dirs,
            max_files=self.config.docs_max_files,
            max_file_bytes=self.config.docs_max_file_bytes,
            max_total_bytes=self.config.docs_max_total_bytes,
            scope=scope,
        )
        if report.reason:
            # The router does not always run where the agent works, and then it
            # cannot read the repository at all.  That is a deployment shape,
            # not an outage, and it is reported as one.
            return ScanOutcome(status=STATUS_SKIPPED, reason=report.reason)
        if self.state is None or not self.state.available:
            return ScanOutcome(
                status=STATUS_UNAVAILABLE,
                reason="state",
                files=report.files,
                truncated=report.truncated,
            )

        # The collection is part of the listing's identity: a policy that
        # moves the documentation elsewhere has to republish rather than
        # believe a listing of a collection nobody reads.
        collection = self.config.collection(scope, KIND_DOC)
        entries = self.state.load(scope, collection)
        by_file = _group(report.records)
        changed = _changed_files(entries, by_file)
        stale = _superseded_names(entries, by_file, changed)

        published: int = 0
        if changed:
            events = doc_events(
                _records_of(by_file, changed),
                scope=scope,
                agent=agent,
                model=model,
                generation=generation,
            )
            attempted = self._publish(events)
            if attempted is None:
                return ScanOutcome(
                    status=STATUS_UNAVAILABLE,
                    reason="publish",
                    files=report.files,
                    truncated=report.truncated,
                )
            published = attempted

        pruned = 0
        if stale and allow_prune and not report.truncated:
            pruned = self._publish_prune(
                scope, stale, agent=agent, generation=generation
            )

        if not self.state.save(
            scope, _next_entries(entries, by_file, changed), collection
        ):
            self._warn("knowledge docs state not saved for %d file(s)", len(by_file))
        context = self._refresh_context(scope, report, by_file)
        return ScanOutcome(
            status=STATUS_OK,
            files=len(by_file),
            published=published,
            pruned=pruned,
            unchanged=len(by_file) - len(changed),
            truncated=report.truncated,
            context=context,
        )

    def context(self, scope: str) -> Optional[DocContext]:
        """
        Return the standing digest of *scope*, read from the store.

        This is what a request path calls when its own scan was on cooldown:
        the digest is refreshed by whichever request happened to scan, and
        every other request of the session reads the same map without walking
        the repository again.  ``None`` when there is no digest or the store is
        off; the caller injects no standing context, which is not an error.

        Parameters
        ----------
        scope : str
            Repository whose digest to read.

        Returns
        -------
        DocContext or None
            The digest, or ``None`` when there is none to read.
        """
        if not self.config.doc_context_enabled or self.context_store is None:
            return None
        return self.context_store.get(scope)

    def _refresh_context(
        self, scope: str, report: "ScanReport", by_file: Dict[str, Any]
    ) -> Optional[DocContext]:
        """
        Rebuild and store one repository's standing digest.

        The digest names the files and their headings and nothing else, so it
        is a map, not a summary, and it can be rebuilt from the same listing the
        scan already produced at no extra cost.  It is written whatever the
        publish path did — a repository whose documents are unchanged still
        wants a map — and only when the scan saw the files, never from a
        cooldown or a failed read.

        Returns ``None`` when documentation context is off or the store cannot
        hold it; the scan itself is unaffected.
        """
        if not self.config.doc_context_enabled or self.context_store is None:
            return None
        if report.reason or not by_file:
            return None
        context = build_doc_context(
            scope,
            _context_files(report),
            max_chars=self.config.doc_context_max_chars,
            truncated=report.truncated,
        )
        if context.empty:
            return None
        self.context_store.put(context)
        return context

    # ------------------------------------------------------------ internals -
    def _in_cooldown(self, scope: str, repo_root: str) -> bool:
        """Whether this process looked at this repository too recently."""
        interval = int(self.config.docs_rescan_interval_seconds)
        if interval <= 0:
            return False
        key = f"{scope}\x00{repo_root}"
        with self._lock:
            last = self._last_scan.get(key)
        if last is None:
            return False
        return (float(self.clock()) - float(last)) < interval

    def _mark_scanned(self, scope: str, repo_root: str) -> None:
        """Record that this process looked at this repository now."""
        with self._lock:
            self._last_scan[f"{scope}\x00{repo_root}"] = float(self.clock())

    def _publish(self, events: Sequence[KnowledgeEvent]) -> Optional[int]:
        """Hand events to the worker; ``None`` when the stream refused them."""
        if not events:
            return 0
        try:
            status = self.publisher.publish(events)  # type: ignore[union-attr]
        except Exception as exc:
            self._warn("knowledge docs publish failed: %s", type(exc).__name__)
            return None
        if getattr(status, "status", "") == STATUS_UNAVAILABLE:
            return None
        return int(getattr(status, "published", 0))

    def _publish_prune(
        self, scope: str, names: Iterable[str], *, agent: str, generation: str
    ) -> int:
        """Ask for retired documents; a refused control message is not fatal."""
        event = prune_event(scope, names, agent=agent, generation=generation)
        if event is None:
            return 0
        if self._publish((event,)) is None:
            return 0
        return len({str(name) for name in names if str(name)})

    def _warn(self, message: str, *args: Any) -> None:
        """Report one failure when a logger was given."""
        if self.logger is not None:
            self.logger.warning(message, *args)


def _context_files(report: "ScanReport") -> List[Any]:
    """
    Return one entry per documentation file, for the standing digest.

    A file's headings are the outline the scan collected, in order;
    the first is the file's own title.  The entry carries the file digest so the
    digest's fingerprint moves only when the file's text does.
    """
    from llm_router_plugins.utils.knowledge.agentic_knowledge.doc_context import (
        DocFile,
    )

    ordered: List[str] = []
    doc_type: Dict[str, str] = {}
    content_hash: Dict[str, str] = {}
    for record in report.records:
        path = record.source_file
        if path not in doc_type:
            ordered.append(path)
            doc_type[path] = record.doc_type
            content_hash[path] = record.content_hash
    outlines = report.headings_by_file
    return [
        DocFile(
            path=path,
            doc_type=doc_type[path],
            headings=tuple(outlines.get(path) or ())
            or _headings_of(records_of(report, path)),
            content_hash=content_hash[path],
        )
        for path in ordered
    ]


def records_of(report: "ScanReport", path: str) -> List[DocRecord]:
    """The record of one file; one record, because one file is one document."""
    return [record for record in report.records if record.source_file == path]


def _headings_of(records: Sequence[DocRecord]) -> Tuple[str, ...]:
    """
    Fall back to the documents' titles when no outline was collected.

    A caller that builds a :class:`ScanReport` from records alone — a test, a
    tool, a future writer — still gets a map, drawn from the title each record
    carries.  It is a thinner map than the outline the scan collects, because a
    document has one title and many headings.
    """
    titles: List[str] = []
    for record in records:
        if record.heading and record.heading not in titles:
            titles.append(record.heading)
    return tuple(titles)


def _group(records: Iterable[DocRecord]) -> Dict[str, List[DocRecord]]:
    """Group scanned records by the file they came from."""
    grouped: Dict[str, List[DocRecord]] = {}
    for record in records:
        grouped.setdefault(record.source_file, []).append(record)
    return grouped


def _changed_files(
    entries: Dict[str, Dict[str, Any]],
    by_file: Dict[str, List[DocRecord]],
) -> List[str]:
    """Files that are new, or whose content no longer matches what is indexed."""
    changed: List[str] = []
    for source_file, records in by_file.items():
        known = entries.get(source_file) or {}
        current = records[0].content_hash if records else ""
        if str(known.get("h") or "") != current:
            changed.append(source_file)
    return changed


def _superseded_names(
    entries: Dict[str, Dict[str, Any]],
    by_file: Dict[str, List[DocRecord]],
    changed: Sequence[str],
) -> List[str]:
    """
    Names of documents the repository no longer has, or has in a new version.

    A changed file has new names — its digest is part of every one of them — and
    a file that is gone has no names at all.  Both are asked for by the same
    control message, and both are only safe to compute against a complete scan,
    which is the caller's concern, not this function's.
    """
    superseded: List[str] = []
    changing = set(changed)
    for source_file, entry in entries.items():
        names = entry.get("n") if isinstance(entry, dict) else None
        if not isinstance(names, list):
            continue
        if source_file not in by_file or source_file in changing:
            superseded.extend(str(name) for name in names if str(name))
    return superseded


def _records_of(
    by_file: Dict[str, List[DocRecord]], files: Sequence[str]
) -> List[DocRecord]:
    """Documents to publish: one per file that changed."""
    wanted = set(files)
    return [
        record
        for name, records in by_file.items()
        if name in wanted
        for record in records
    ]


def _next_entries(
    entries: Dict[str, Dict[str, Any]],
    by_file: Dict[str, List[DocRecord]],
    changed: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    """
    The listing to remember after a scan.

    Unchanged files keep their entry, document names included — they are what a
    later scan will retire when the file changes.  Changed files get the names
    this scan just published; files that are gone are dropped, their documents
    having been handed to the worker to delete.
    """
    next_entries: Dict[str, Dict[str, Any]] = {}
    changing = set(changed)
    for source_file, records in by_file.items():
        if source_file in changing:
            record = records[0] if records else None
            if record is None:
                continue
            next_entries[source_file] = {
                "h": record.content_hash,
                "s": int(record.size),
                "m": float(record.mtime),
                "n": [
                    document_name(KIND_DOC, datetime.date.today(), item.fingerprint)
                    for item in records
                ],
            }
            continue
        known = entries.get(source_file)
        if known:
            next_entries[source_file] = known
    return next_entries
