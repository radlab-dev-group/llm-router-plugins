"""
The repository's standing documentation context, one per session's view of it.

A retrieved chunk answers a question; a digest of the repository's own
documentation answers the question the agent has before it has a question —
what this project is, where its documentation lives, and what the pages
cover.  The layer has always retrieved documentation per query; what was
missing was the always-present map, so every command generation of a long
session re-learns the same thing: this is a service, it has a deploy guide,
auth is documented.

The digest is **structural, not prose**.  It names the documentation files,
their kinds and their headings, and nothing else.  The prose of a repository
is what the write path masks before it indexes, and a digest built on the
request path would carry the README's sentences unmasked into a request — an
API key in a ``Getting started`` would reach a model the rest of the layer
keeps away from.  Paths, kinds and headings carry no such payload (the write
path already files paths unmasked), and they are what an agent actually needs
to aim a question: it reads ``docs/export.md › Export helpers`` exists and
asks for it, and the retrieval leg delivers the text.

Stability is the point.  A digest lives per repository, not per request or
per generation: it changes only when the documentation it names changes, so
the block an agent is reading does not move under it between two commands.
It is kept in Redis, one key per scope, refreshed by the scan that already
runs on the request path and by ``llm-router-agentic-kb-docs`` on a machine
the router cannot see the repository from.
"""

import json

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple

from llm_router_plugins.utils.knowledge.agentic_knowledge.redisclient import (
    RedisClientHandle,
)
from llm_router_plugins.utils.routing.agentic_routing.codex.state import (
    RedisConnectionSettings,
    fingerprint as digest,
)

__all__ = [
    "DocContext",
    "DocContextStore",
    "DocFile",
    "build_doc_context",
    "render_doc_context",
]

#: Most files the digest names.  A map of forty files is a directory listing,
#: not a map: past this the agent has no better use for the names than a
#: ``ls`` would have given it, and the budget is spent on paths it will not
#: read this session.
_MAX_CONTEXT_FILES = 12

#: Most headings quoted per file.
_MAX_HEADINGS_PER_FILE = 6

#: A heading longer than this is a paragraph, not a heading.
_MAX_HEADING_CHARS = 60

#: Said when the map is not the whole repository.  A reader who is not told
#: will conclude a page does not exist and rewrite it.
_PARTIAL_NOTE = "\n(… more documentation not listed; search it)"


@dataclass(frozen=True)
class DocContext:
    """
    One repository's standing documentation context.

    Parameters
    ----------
    scope : str
        Repository the context describes.
    text : str
        The rendered digest, ready for injection.
    source_files : Tuple[str, ...]
        Documentation files the digest names; the citation and the
        annotation's ``sources``.
    fingerprint : str
        Digest of what the map is made of; a changed map is a new fingerprint,
        and the annotation says which one was injected.
    truncated : bool
        ``True`` when a ceiling left files out of the map; the digest then
        says so, so the agent knows it is looking at a part.
    """

    scope: str = ""
    text: str = ""
    source_files: Tuple[str, ...] = ()
    fingerprint: str = ""
    truncated: bool = False

    @property
    def chars(self) -> int:
        """Length of :attr:`text`; what the digest costs a request."""
        return len(self.text)

    @property
    def empty(self) -> bool:
        """Whether the digest has nothing to say."""
        return not self.text


@dataclass(frozen=True)
class DocFile:
    """
    One documentation file as the standing digest sees it.

    Parameters
    ----------
    path : str
        Repository-relative path.
    doc_type : str
        What kind of documentation it is; see
        :func:`.docs_ingestor.classify_document`.
    headings : Tuple[str, ...]
        Headings inside the file, in order; the first is its title.
    content_hash : str
        Digest of the whole file, so the map's fingerprint moves only when a
        file's text does.
    """

    path: str = ""
    doc_type: str = ""
    headings: Tuple[str, ...] = ()
    content_hash: str = ""


def build_doc_context(
    scope: str,
    files: Iterable[DocFile],
    *,
    max_chars: int = 700,
    truncated: bool = False,
) -> DocContext:
    """
    Return the standing digest of a repository's documentation.

    Parameters
    ----------
    scope : str
        Repository the documentation belongs to.
    files : Iterable[DocFile]
        The documentation files, any order; the digest sorts them, because a
        map that rearranges itself between two requests is a map nobody can
        learn.
    max_chars : int, optional
        Ceiling of the rendered digest.
    truncated : bool, optional
        Whether the scan that saw these files did not see the whole
        repository; the digest then says it is partial.

    Returns
    -------
    DocContext
        The digest; empty when the repository has no documentation at all.
    """
    limit = max(120, int(max_chars))
    entries = sorted(
        (
            DocFile(
                path=str(entry.path).strip(),
                doc_type=str(entry.doc_type or ""),
                headings=tuple(
                    heading.strip()[:_MAX_HEADING_CHARS]
                    for heading in entry.headings
                    if heading.strip()
                ),
                content_hash=str(entry.content_hash or ""),
            )
            for entry in files
            if str(entry.path or "").strip()
        ),
        # The introduction first, then the rest by path: a reader meets the
        # project before its pages.  Everything else is a deterministic path
        # order, and the file digest is deliberately not a sort key — it moves
        # when the prose does, and a map must not rearrange itself over a
        # typo.
        key=lambda entry: (
            0 if entry.doc_type == "readme" else 1,
            entry.path,
        ),
    )
    if not entries:
        return DocContext(scope=scope)

    # There are more files than the map can hold in two different ways — the
    # file ceiling and the character ceiling — and a partial map that reads as
    # complete is worse than a short one, so the note that says "part" has its
    # room reserved before the first file is written rather than appended if
    # there happens to be space left over.
    partial = truncated or len(entries) > _MAX_CONTEXT_FILES
    room = limit - (len(_PARTIAL_NOTE) + 1 if partial else 0)

    lines: List[str] = []
    used = 0
    named = 0
    for entry in entries:
        if named == _MAX_CONTEXT_FILES:
            break
        line = f"- {entry.path}"
        if entry.doc_type:
            line += f" ({entry.doc_type})"
        if entry.headings:
            quoted = "; ".join(entry.headings[:_MAX_HEADINGS_PER_FILE])
            line += f": {quoted}"
        if used + len(line) + 1 > room and named:
            break
        lines.append(line)
        used += len(line) + 1
        named += 1

    body = "\n".join(lines)
    truncated = partial or named < len(entries)
    if truncated:
        body += _PARTIAL_NOTE

    fingerprint = digest(
        "docctx",
        scope,
        # The map as it reads, and the revisions it was drawn from: a new
        # heading is a new map, and so is a file whose text moved under a
        # heading the map still quotes.
        body,
        *(
            f"{entry.path}|{entry.doc_type}|{entry.content_hash}"
            for entry in entries
        ),
    )
    return DocContext(
        scope=scope,
        text=body,
        source_files=tuple(entry.path for entry in entries[:named]),
        fingerprint=fingerprint,
        truncated=truncated,
    )


def render_doc_context(context: DocContext) -> str:
    """
    Return the digest as the block injects it.

    The wording is deliberate: this is a map of what the repository has
    written down, and the one instruction that matters is that it can be
    asked for.

    Parameters
    ----------
    context : DocContext
        The digest to render.

    Returns
    -------
    str
        The rendered digest; ``""`` for an empty one.
    """
    if context.empty:
        return ""
    return (
        "This repository documents itself; the map below is what it has "
        "written down, and any of it can be asked for in full.\n"
        f"{context.text}"
    )


class DocContextStore:
    """
    Where a repository's standing digest lives between requests.

    One Redis key per scope.  Losing it costs one scan's worth of re-reading
    the repository's own files; a stale one costs a digest that is one edit
    behind, which the retrieval leg corrects the moment it is asked about.

    Parameters
    ----------
    connection : RedisConnectionSettings, optional
        Where the digest lives.
    key_prefix : str
        Namespace of the keys.
    ttl_seconds : int
        How long a digest survives without a refresh.
    client : Any, optional
        Redis client to use as-is; tests pass a fake.
    logger : Any, optional
        Where failures are reported, without content.
    """

    def __init__(
        self,
        connection: Optional[RedisConnectionSettings] = None,
        key_prefix: str = "agentic-kb",
        ttl_seconds: int = 86_400,
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
        """Whether the digest can be read and written at all."""
        return self._handle.configured

    def key(self, scope: str) -> str:
        """Return the Redis key of one scope's digest."""
        return f"{self._key_prefix}:docs:context:{scope}"

    def get(self, scope: str) -> Optional[DocContext]:
        """
        Return the digest of *scope*; ``None`` when there is none to read.

        Returns
        -------
        DocContext or None
            The digest, or ``None`` on a miss, a malformed value, or a Redis
            that did not answer — each of which only means no standing context
            this request, which is the layer's normal state.
        """
        if not scope:
            return None
        client = self._handle.get()
        if client is None:
            return None
        try:
            raw = client.get(self.key(scope))
        except Exception as exc:
            self._warn("knowledge doc context unreadable: %s", type(exc).__name__)
            return None
        if not raw:
            return None
        try:
            payload = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
        except (TypeError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        text = str(payload.get("text") or "").strip()
        if not text:
            return None
        files = payload.get("files")
        return DocContext(
            scope=scope,
            text=text,
            source_files=(
                tuple(str(path) for path in files if str(path).strip())
                if isinstance(files, (list, tuple))
                else ()
            ),
            fingerprint=str(payload.get("fingerprint") or ""),
            truncated=bool(payload.get("truncated")),
        )

    def put(self, context: DocContext) -> bool:
        """
        Write one scope's digest.

        Parameters
        ----------
        context : DocContext
            What to remember.

        Returns
        -------
        bool
            ``True`` when the digest was written; ``False`` means the next
            request simply has no standing context, which is not an error.
        """
        if context.empty:
            return False
        client = self._handle.get()
        if client is None:
            return False
        payload = json.dumps(
            {
                "v": 1,
                "text": context.text,
                "files": list(context.source_files),
                "fingerprint": context.fingerprint,
                "truncated": context.truncated,
            },
            ensure_ascii=False,
        )
        try:
            client.set(self.key(context.scope), payload, ex=self._ttl)
        except Exception as exc:
            self._warn("knowledge doc context unwritable: %s", type(exc).__name__)
            return False
        return True

    def _warn(self, message: str, *args: Any) -> None:
        """Log a warning when a logger was given."""
        if self._logger is not None:
            self._logger.warning(message, *args)
