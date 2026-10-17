"""
Repository scope of one agent request, and the collections it maps to.

Knowledge is worth retrieving only inside the repository it was produced in: a
verified fix to a pytest fixture in one service says nothing about the next.
The CLI declares no repository, so the only usable identifier is the one the
agent itself prints in the ``<environment_context>`` block it prepends to the
first ``input`` item — ``<cwd>`` for Codex, the ``Working directory`` line for
Claude Code.

The declared path is turned into a stable, path-free *scope key* (by default
the repository's directory name, with an override table for organizations and
checkout directories that share a basename).  One collection per scope keeps
the isolation out of the query: the search of a repository cannot return a
document of another even if a metadata filter is forgotten.
"""

import os
import re
import posixpath

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

__all__ = [
    "STRATEGY_REPO_BASENAME",
    "STRATEGY_PARENT_DIRECTORY",
    "SCOPE_STRATEGIES",
    "RepositoryScope",
    "ScopeResolver",
    "declared_working_directory",
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

#: Characters a collection name may carry; SSE replaces spaces itself, but the
#: plugin normalizes the rest so the name it derives is the name it queries.
_UNSAFE_RE = re.compile(r"[^a-z0-9._-]+")

_MAX_SCOPE_CHARS = 48


def _clean_path(value: str) -> str:
    """Normalize a declared path without resolving symlinks it cannot see."""
    text = (value or "").strip().strip("\"'")
    if not text:
        return ""
    return posixpath.normpath(text.replace(os.sep, "/"))


def _candidate_text(payload: Any) -> str:
    """
    Return the text of a payload that may declare the working directory.

    Only the leading items are scanned: the context block is prepended by the
    CLI, and a long history is neither scanned nor copied into memory.
    """
    if not isinstance(payload, dict):
        return ""
    parts = []
    instructions = payload.get("instructions")
    if isinstance(instructions, str):
        parts.append(instructions[:8000])
    system = payload.get("system")
    if isinstance(system, str):
        parts.append(system[:8000])
    elif isinstance(system, list):
        for block in system[:4]:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"][:8000])
    items = payload.get("input")
    if isinstance(items, list):
        for item in items[:3]:
            if isinstance(item, str):
                parts.append(item[:8000])
                continue
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, str):
                parts.append(content[:8000])
            elif isinstance(content, list):
                for part in content[:3]:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        parts.append(part["text"][:8000])
    return "\n".join(parts)


def declared_working_directory(payload: Any) -> str:
    """
    Return the working directory the agent declared, or ``""``.

    Prefers the machine-readable ``<cwd>`` tag, then the ``<workspace_roots>``
    entry Codex adds, then the prose line Claude Code sends.

    Parameters
    ----------
    payload : Any
        The request payload.

    Returns
    -------
    str
        A normalized POSIX path, or ``""`` when the payload declares none.
    """
    text = _candidate_text(payload)
    if not text:
        return ""
    for pattern in (_CWD_TAG_RE, _ROOT_TAG_RE, _WORKING_DIRECTORY_RE):
        match = pattern.search(text)
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
            return RepositoryScope(key=_normalize(self.default_scope))
        override = self.overrides.get(root) or self.overrides.get(f"{root}/")
        if override:
            return RepositoryScope(key=_normalize(override), repo_root=root)
        return RepositoryScope(key=self._key_from_root(root), repo_root=root)

    def _key_from_root(self, root: str) -> str:
        """Derive a scope key from an absolute path."""
        parts = [part for part in root.split("/") if part and part != "."]
        if not parts:
            return _normalize(self.default_scope)
        name = parts[-1]
        if self.strategy == STRATEGY_PARENT_DIRECTORY and len(parts) > 1:
            name = f"{parts[-2]}-{name}"
        return _normalize(name) or _normalize(self.default_scope)


def _normalize(value: Optional[str]) -> str:
    """Make an out of a path fragment something a collection name accepts."""
    text = _UNSAFE_RE.sub("-", (value or "").strip().lower()).strip("-._")
    return text[:_MAX_SCOPE_CHARS].strip("-._")
