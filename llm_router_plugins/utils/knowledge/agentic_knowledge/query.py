"""
What to ask the search, for each work mode.

A question that is good for one mode is noise for another.  In ``debug`` the
agent is staring at a failure, so the failure's own signature — not the user's
phrasing of it — is what matches a recipe written months ago.  In ``plan`` there
is no failure yet and the user's sentence is all there is.  In ``git_review``
the work is about files, so the files decide.

The builder also does the bookkeeping the engine cannot: sources live in
different collections (mined knowledge apart from the session archive apart
from the repository's documentation), the archive is restricted to the current
thread, and a mode's character budget is turned into a sane ``max_results`` so
the search does not return fifty chunks that the assembler will throw away.

Documentation is searched with a query of its own.  A normalized stack trace
retrieves a recipe written from the same failure and nothing else, while a
conventions page matches the module the turn is editing and what the user asked
for — so the documentation request is built from the user's text and the paths,
and never carries the failure signature.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from llm_router_plugins.utils.knowledge.agentic_knowledge.adapters.base import (
    AgentTurn,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.config import ModePolicy
from llm_router_plugins.utils.knowledge.agentic_knowledge.documents import (
    KIND_DOC,
    KIND_SESSION_ARCHIVE,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.sse import SearchRequest

__all__ = ["QueryBuilder"]

#: Ceiling of the free-text part of a query.
_MAX_TEXT_CHARS = 600

#: Ceiling of one query overall.
_MAX_QUERY_CHARS = 900

#: Most paths spelled out in a documentation query.  A documentation file is
#: named after the module it documents, so the paths are the strongest term the
#: turn has; past a large diff the list stops adding meaning and starts
#: diluting the user's own words.
_MAX_QUERY_PATHS = 8


@dataclass(frozen=True)
class QueryBuilder:
    """
    Turn a turn and a mode policy into search requests.

    Parameters
    ----------
    collection_namer : callable
        ``(scope, kind) -> collection name``, normally
        :meth:`AgenticKnowledgeConfig.collection`.
    min_similarity : float
        Cosine floor carried into every request.
    """

    collection_namer: object
    min_similarity: float = 0.35

    def build(
        self, turn: AgentTurn, policy: ModePolicy
    ) -> Tuple[SearchRequest, ...]:
        """
        Return the searches this turn asks for.

        Parameters
        ----------
        turn : AgentTurn
            The request's turn.
        policy : ModePolicy
            The policy of its work mode.

        Returns
        -------
        Tuple[SearchRequest, ...]
            One request per collection; empty when the turn has nothing
            searchable or the scope is unknown.
        """
        if not turn.scope:
            return ()
        query = self.query_for(turn)
        if not query.strip():
            return ()
        grouped: Dict[str, List[str]] = {}
        limits: Dict[str, int] = {}
        # A missing key is "restrictions of the sources seen so far"; ``None``
        # is "any".  One source that does not restrict makes the whole
        # collection unrestricted, because both are retrieved by one request
        # and the other's filter would silently drop half of what the mode
        # asked to read.
        phase_limits: Dict[str, Optional[List[str]]] = {}
        mode_limits: Dict[str, Optional[List[str]]] = {}
        doc_type_limits: Dict[str, Optional[List[str]]] = {}
        for source in policy.sources:
            kind = source.category
            archived = kind == KIND_SESSION_ARCHIVE
            if archived and not turn.thread_id:
                # Without a thread there is nothing to restrict the archive to,
                # and an unfiltered archive is another team's conversation.
                continue
            name = str(self.collection_namer(turn.scope, kind))
            kinds = grouped.setdefault(name, [])
            # A policy may list one category twice; keeping it once means the
            # "archive-only?" test below reads the set of kinds, not their
            # multiplicity, so a repeated `session_archive` cannot turn the
            # thread filter off and expose the whole repository's windows.
            if kind not in kinds:
                kinds.append(kind)
            limits[name] = limits.get(name, 0) + int(source.max_items)
            if not source.phases:
                phase_limits[name] = None
            elif name not in phase_limits:
                phase_limits[name] = list(source.phases)
            elif phase_limits[name] is not None:
                phase_limits[name] = list(
                    dict.fromkeys(phase_limits[name] + list(source.phases))
                )
            if not source.modes:
                mode_limits[name] = None
            elif name not in mode_limits:
                mode_limits[name] = list(source.modes)
            elif mode_limits[name] is not None:
                mode_limits[name] = list(
                    dict.fromkeys(mode_limits[name] + list(source.modes))
                )
            if not source.doc_types:
                doc_type_limits[name] = None
            elif name not in doc_type_limits:
                doc_type_limits[name] = list(source.doc_types)
            else:
                known = doc_type_limits[name]
                if known is not None:
                    doc_type_limits[name] = list(
                        dict.fromkeys(known + list(source.doc_types))
                    )
        requests: List[SearchRequest] = []
        for name, categories in grouped.items():
            archived = KIND_SESSION_ARCHIVE in categories
            documented = _is_documentation(categories)
            restricted = phase_limits.get(name)
            restricted_modes = mode_limits.get(name)
            restricted_types = doc_type_limits.get(name)
            requests.append(
                SearchRequest(
                    collection=name,
                    # Documentation is retrieved on the turn's subject, not on
                    # its failure; everything else keeps the mode-shaped query.
                    query=self.docs_query_for(turn) if documented else query,
                    categories=tuple(dict.fromkeys(categories)),
                    max_results=max(1, limits.get(name, 3)),
                    thread_id=turn.thread_id if archived else "",
                    # An archived window is already narrowed to one thread, and
                    # the engine cannot intersect two metadata filters.
                    phases=() if archived else tuple(restricted or ()),
                    modes=() if archived else tuple(restricted_modes or ()),
                    doc_types=() if archived else tuple(restricted_types or ()),
                )
            )
        return tuple(requests)

    def docs_query_for(self, turn: AgentTurn) -> str:
        """
        Return the query that retrieves documentation, not mined knowledge.

        Documentation is prose about the code: it answers "how does this
        repository do X" and is filed under the module and the concept, never
        under a failure signature.  So the query is the user's own words plus
        the paths of the turn — the whole path, because ``docs/export.md``
        quotes ``src/export.py`` verbatim more often than not, and the stem,
        because a file is usually named after the module it explains.

        Parameters
        ----------
        turn : AgentTurn
            The request's turn.

        Returns
        -------
        str
            The query; ``""`` when the turn says nothing about any file.
        """
        text = (turn.latest_user_text or "").strip()[:_MAX_TEXT_CHARS]
        paths = [path for path in turn.touched_paths[:_MAX_QUERY_PATHS] if path]
        stems = _stems(paths)
        joined = " \n".join(
            part for part in (text, " ".join(paths), " ".join(stems)) if part
        )
        return joined.strip()[:_MAX_QUERY_CHARS]

    def query_for(self, turn: AgentTurn) -> str:
        """
        Return the query text of one turn, shaped by its work mode.

        Parameters
        ----------
        turn : AgentTurn
            The request's turn.

        Returns
        -------
        str
            The query; ``""`` when there is nothing to search for.
        """
        failure = _failure_signature(turn)
        text = (turn.latest_user_text or "").strip()[:_MAX_TEXT_CHARS]
        paths = " ".join(turn.touched_paths[:12])
        mode = turn.mode

        if mode == "debug":
            parts = [failure, text]
        elif mode == "test":
            parts = [_test_command(turn), paths, text]
        elif mode == "implement":
            parts = [text, paths, failure]
        elif mode in ("git_review", "review"):
            parts = [paths, text]
        else:
            parts = [text, paths, failure]
        joined = " \n".join(part for part in parts if part and part.strip())
        return joined.strip()[:_MAX_QUERY_CHARS]


def _is_documentation(categories: Sequence[str]) -> bool:
    """
    Return whether one collection holds documentation only.

    The ``doc`` category lives in its own collection, so a group that contains
    it contains nothing else.  Anything unexpected is treated as mined
    knowledge: a query carrying the failure signature is the older, safer
    default.
    """
    return tuple(str(kind) for kind in categories) == (KIND_DOC,)


def _stems(paths: Sequence[str]) -> List[str]:
    """Return the file-name stems of *paths*, de-duplicated, order kept."""
    stems: List[str] = []
    for path in paths:
        name = str(path).rsplit("/", 1)[-1]
        stem = name.rsplit(".", 1)[0] if "." in name else name
        stem = stem.strip()
        if stem and stem not in stems:
            stems.append(stem)
    return stems


def _failure_signature(turn: AgentTurn) -> str:
    """Return the failure this turn is stuck on, already normalized."""
    from llm_router_plugins.utils.knowledge.agentic_knowledge.signature import (
        signature_of,
    )

    evidence = turn.latest_failure()
    if evidence is None:
        return ""
    for item in reversed(turn.activity):
        if item.kind == "function_call_output" and item.call_id == evidence.call_id:
            signature = signature_of(item.text)
            return signature.signature if signature else ""
    return ""


def _test_command(turn: AgentTurn) -> str:
    """Return the test command of the turn, if it ran one."""
    from llm_router_plugins.utils.routing.agentic_routing.codex.phase import (
        EVIDENCE_COMMAND,
    )

    for item in reversed(turn.evidence):
        if item.kind != EVIDENCE_COMMAND:
            continue
        for activity in reversed(turn.activity):
            if (
                activity.kind == "function_call"
                and activity.call_id
                and activity.call_id == item.call_id
            ):
                return _first_line(activity.text)
    return ""


def _first_line(raw: str) -> str:
    """Read a command out of a tool call's arguments."""
    text = (raw or "").strip()
    if not text:
        return ""
    if text.startswith("{"):
        import json

        try:
            decoded = json.loads(text)
        except (TypeError, ValueError):
            decoded = None
        if isinstance(decoded, dict):
            for key in ("cmd", "command", "input"):
                value = decoded.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip().splitlines()[0][:160]
        return ""
    return text.splitlines()[0][:160]
