"""
Repository scope of one agent request, and the collections it maps to.

Knowledge is worth retrieving only inside the repository it was produced in: a
verified fix to a pytest fixture in one service says nothing about the next.
The CLI declares no repository, so the only usable identifier is the one the
agent itself prints in the ``<environment_context>`` block of its first user
message — ``<cwd>`` for Codex, the ``Working directory`` line for Claude Code.
That message is not the first item of every request: a compaction re-sends the
whole conversation, so the leading messages are searched rather than assumed.

The declared path is turned into a stable, path-free *scope key* (by default
the repository's directory name, with an override table for organizations and
checkout directories that share a basename).  One collection per scope keeps
the isolation out of the query: the search of a repository cannot return a
document of another even if a metadata filter is forgotten.
"""

import hashlib
import logging
import os
import re
import posixpath
import unicodedata

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Optional, Tuple

__all__ = [
    "STRATEGY_REPO_BASENAME",
    "STRATEGY_PARENT_DIRECTORY",
    "SCOPE_STRATEGIES",
    "RepositoryScope",
    "ScopeResolver",
    "declared_working_directory",
    "normalize_scope",
    "scope_key",
]

#: Scope key is the name of the checked-out directory.
STRATEGY_REPO_BASENAME = "repo_basename"

#: Scope key is ``<parent>/<repo>``, for monorepos under a common parent.
STRATEGY_PARENT_DIRECTORY = "parent_directory"

SCOPE_STRATEGIES: Tuple[str, ...] = (
    STRATEGY_REPO_BASENAME,
    STRATEGY_PARENT_DIRECTORY,
)

_CWD_TAG_RE = re.compile(r"<cwd>\s*(.+?)\s*</cwd>")
_ROOT_TAG_RE = re.compile(r"<workspace_roots>\s*<root>\s*(.+?)\s*</root>")
_WORKING_DIRECTORY_RE = re.compile(
    r"[Ww]orking directory:?\s*[\"']?([^\n\"']+)", re.MULTILINE
)
_CONTEXT_RE = re.compile(r"<(?:environment_context|system)\b")

#: Characters of one message searched for a declaration.  A compaction request
#: re-sends the whole conversation, so the CLI's context block sits tens of
#: kilobytes into the payload, behind the developer instructions; a search of
#: this much text costs well under a millisecond.
_SCAN_MESSAGE_CHARS = 256_000

#: Ceiling on that search for one request.  A declaration is in one of the
#: leading messages; a longer payload is not read to the end, so a request path
#: never pays for a history it does not need.
_SCAN_TOTAL_CHARS = 1_000_000

#: Characters a collection name may carry; SSE replaces spaces itself, but the
#: plugin normalizes the rest so the name it derives is the name it queries.
_UNSAFE_RE = re.compile(r"[^a-z0-9._-]+")

_MAX_SCOPE_CHARS = 48

_LOGGER = logging.getLogger("agentic-kb-scope")

#: One warning per (key, first-seen name) per process: two distinct
#: repositories sharing a scope key is a configuration error the operator
#: should see once, not a line in every request.
_seen: dict = {}


def _clean_path(value: str) -> str:
    """Normalize a declared path without resolving symlinks it cannot see."""
    text = (value or "").strip().strip("\"'")
    if not text:
        return ""
    return posixpath.normpath(text.replace(os.sep, "/"))


def _message_texts(payload: Any) -> Iterator[str]:
    """
    Yield the texts of a payload in which a directory may be declared.

    Message texts only, in the order the agent sent them: the CLI writes its
    context block into a message, while a tool output can quote the ``<cwd>``
    of some other checkout, and a quoted path is not a declaration.  Nothing
    is copied here — the caller decides how much of each text it reads.
    """
    instructions = payload.get("instructions")
    if isinstance(instructions, str):
        yield instructions
    system = payload.get("system")
    if isinstance(system, str):
        yield system
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                yield block["text"]
    items = payload.get("input")
    if isinstance(items, list):
        for item in items:
            if isinstance(item, str):
                yield item
                continue
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, str):
                yield content
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        yield part["text"]


def declared_working_directory(payload: Any) -> str:
    """
    Return the working directory the agent declared, or ``""``.

    Prefers the machine-readable ``<cwd>`` tag, then the ``<workspace_roots>``
    entry Codex adds, then the prose line Claude Code sends.  Messages are read
    in the order they were sent and the first declaration wins, so a directory
    quoted later — from another checkout, inside a tool output — cannot stand
    in for the one this request was made in.

    Parameters
    ----------
    payload : Any
        The request payload.

    Returns
    -------
    str
        A normalized POSIX path, or ``""`` when the payload declares none.
    """
    if not isinstance(payload, dict):
        return ""
    remaining = _SCAN_TOTAL_CHARS
    for text in _message_texts(payload):
        if remaining <= 0:
            break
        end = min(len(text), remaining, _SCAN_MESSAGE_CHARS)
        remaining -= end
        for pattern in (_CWD_TAG_RE, _ROOT_TAG_RE, _WORKING_DIRECTORY_RE):
            match = pattern.search(text, 0, end)
            if match:
                path = _clean_path(match.group(1))
                if path:
                    return path
    return ""


@dataclass(frozen=True)
class RepositoryScope:
    """
    The knowledge scope of one request.

    Parameters
    ----------
    key : str
        Scope identifier used in collection names and cache keys; ``""`` when
        the request declares no repository and no scope can be assumed.
    repo_root : str
        The declared working directory, used to relativize paths.  Never
        indexed: only :attr:`key` reaches the search engine.
    """

    key: str = ""
    repo_root: str = ""

    @property
    def known(self) -> bool:
        """Whether a repository was identified."""
        return bool(self.key)

    def relative(self, path: str) -> str:
        """
        Return *path* relative to :attr:`repo_root`.

        A path outside the repository — or a path that cannot be placed — is
        reduced to its basename, so no absolute filesystem layout is ever
        written into a knowledge document.

        Parameters
        ----------
        path : str
            A path as it appeared in a patch or a tool output.

        Returns
        -------
        str
            A repository-relative path, a basename, or ``""``.
        """
        text = _clean_path(path)
        if not text:
            return ""
        root = self.repo_root
        if root and (text == root or text.startswith(root + "/")):
            return text[len(root) + 1 :] or posixpath.basename(text)
        if text.startswith("/"):
            return posixpath.basename(text)
        return text


@dataclass(frozen=True)
class ScopeResolver:
    """
    Turn a payload into a :class:`RepositoryScope`.

    Parameters
    ----------
    strategy : str
        One of :data:`SCOPE_STRATEGIES`.
    overrides : Mapping[str, str]
        Explicit ``declared path -> scope key`` table, consulted first, for
        checkouts whose directory name is not a usable identifier.
    default_scope : str
        Scope used when nothing matches and the operator wants a shared
        knowledge base; ``""`` keeps the plugin stateless instead.
    """

    strategy: str = STRATEGY_REPO_BASENAME
    overrides: Mapping[str, str] = field(default_factory=dict)
    default_scope: str = ""

    def resolve(self, payload: Any) -> RepositoryScope:
        """
        Return the scope of *payload*.

        Parameters
        ----------
        payload : Any
            The request payload.

        Returns
        -------
        RepositoryScope
            The identified scope, or one with an empty key.
        """
        root = declared_working_directory(payload)
        if not root:
            return RepositoryScope(key=_normalize(self.default_scope, digest=True))
        return RepositoryScope(
            key=scope_key(
                root,
                strategy=self.strategy,
                overrides=self.overrides,
                default=self.default_scope,
            ),
            repo_root=root,
        )


def scope_key(
    root: str,
    *,
    strategy: str = STRATEGY_REPO_BASENAME,
    overrides: Optional[Mapping[str, str]] = None,
    default: str = "",
) -> str:
    """
    Return the scope key of the repository rooted at *root*.

    Split out of :meth:`ScopeResolver.resolve` because a scope is not always
    declared in a payload: the operator's documentation command knows the
    directory it was pointed at and has to derive the same key the proxy would
    have derived from a request made in it.  Both go through here, so a
    repository indexed by hand and a repository an agent works in are the same
    collection rather than two that never meet.

    Parameters
    ----------
    root : str
        Directory of the checkout, absolute or relative.
    strategy : str
        One of :data:`SCOPE_STRATEGIES`.
    overrides : Mapping[str, str], optional
        Explicit ``directory → scope key`` table, consulted first.
    default : str, optional
        Key to fall back to when nothing can be derived.

    Returns
    -------
    str
        The scope key, possibly ``""``.
    """
    text = _clean_path(root)
    given = overrides or {}
    override = given.get(text) or given.get(f"{text}/")
    if override:
        return _normalize(override)
    parts = [part for part in text.split("/") if part and part != "."]
    if not parts:
        return _normalize(default)
    name = parts[-1]
    if strategy == STRATEGY_PARENT_DIRECTORY and len(parts) > 1:
        name = f"{parts[-2]}-{name}"
    # A scope the operator never typed may not be empty: a repository whose
    # name has no ASCII at all still needs a collection to live in, so the
    # digest fallback stands in for a name that cannot be read.
    return _normalize(name, digest=True) or _normalize(default)


def _transliterate(value: str) -> str:
    """
    Reduce *value* to the ASCII it contains, best effort.

    A name is lowercased and its safe characters kept; the non-ASCII that
    survives is decomposed (NFKD) and its diacritics dropped, so ``repo-ą``
    reads as ``repo-a`` and two repositories that differ only by a diacritic
    stay apart.  A name with no ASCII at all (``проект``, ``日本``) reduces to
    nothing, and the caller replaces it with a deterministic digest of the
    original — a scope may not be empty, and a collision of two different
    names is worse than a name no one can read.
    """
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(
        char for char in decomposed if unicodedata.category(char) != "Mn"
    )


def _collides(key: str, original: str) -> None:
    """Warn once per process when a different name lands on a taken key."""
    first = _seen.get(key)
    if first is None:
        _seen[key] = original
        return
    if original != first and (key, first) not in _collides._warned:
        _collides._warned.add((key, first))
        _LOGGER.warning(
            "knowledge scope collision: %r and %r both normalize to %r; "
            "configure scope.overrides to keep them apart",
            first,
            original,
            key,
        )


_collides._warned = set()


def _normalize(value: Optional[str], *, digest: bool = False) -> str:
    """
    Make an out of a path fragment something a collection name accepts.

    A *derived* scope (``digest=True``) must never be empty, because an empty
    key is a scope the engine cannot name and the whole layer goes silently
    off for that repository; a name with no ASCII at all then stands behind a
    deterministic digest of itself.  An *operator-supplied* scope
    (``digest=False``) is a typo when it carries no readable identity, and the
    caller refuses it rather than inventing a key the operator never asked for.
    """
    original = (value or "").strip()
    if not original:
        return ""
    # Transliterate before the unsafe substitution: a name that is only
    # non-ASCII (``проект``, ``日本``) must not be emptied by the substitution
    # and then left with no key to fall back on.  The transliteration is what
    # decides whether a readable ASCII name survives at all.
    text = _transliterate(original.lower())
    key = _UNSAFE_RE.sub("-", text).strip("-._")[:_MAX_SCOPE_CHARS].strip("-._")
    if not key:
        if not digest:
            return ""
        # Nothing readable survived: a deterministic digest of the original
        # name is the only key that is both stable and never empty.  It is a
        # fallback, not a preference — a readable name is always preferred.
        return hashlib.sha1(original.encode("utf-8")).hexdigest()[:8]
    _collides(key, original)
    return key


def normalize_scope(value: str) -> str:
    """
    Reduce one operator-supplied scope to what a collection name accepts.

    The proxy path derives its scope from a directory name and normalizes it
    on the way in; a scope typed into ``llm-router-agentic-kb-docs --scope``
    must take the same road, because the engine refuses a collection name with
    a character the normalization would not produce — and a scope that cannot
    be named is a scope the worker cannot index into.  An empty result is
    returned as ``""`` so the caller can refuse the command before it writes
    anything.

    Parameters
    ----------
    value : str
        The scope the operator typed; ``""`` when none was given.

    Returns
    -------
    str
        The normalized scope key, possibly ``""``.
    """
    return _normalize(value)
