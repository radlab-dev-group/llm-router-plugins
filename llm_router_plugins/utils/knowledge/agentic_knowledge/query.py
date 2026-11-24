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
import re
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

#: The three groups of sources one turn asks about; see :func:`_bucket`.  They
#: key the grouping of requests and are never sent to the engine, which sees
#: categories and metadata filters instead.
_BUCKET_KNOWLEDGE = "knowledge"
_BUCKET_ARCHIVE = "archive"
_BUCKET_DOC = "doc"

#: Ceiling of one query overall.
_MAX_QUERY_CHARS = 900

#: Most paths spelled out in a documentation query.  A documentation file is
#: named after the module it documents, so the paths are the strongest term the
#: turn has; past a large diff the list stops adding meaning and starts
#: diluting the user's own words.
_MAX_QUERY_PATHS = 8

#: A turn that named no file, met no failure and ran no command asks the
#: knowledge base with its own sentence alone.  Below this many characters
#: that sentence is an acknowledgement, not a question, and three searches of
#: it are three payments for nothing; the read is skipped instead.  It applies
#: only in the absence of the structured signals a query is built from, so a
#: short question that names a file is still a question.
_MIN_QUESTION_CHARS = 12

#: The engine multiplies ``max_results`` by its own factor on every leg of a
#: hybrid search, so this number is not how many chunks the assembler reads —
#: it is how many the engine embeds, deserializes and ranks first.  A floor of
#: fifteen bought one hundred and fifty vectors per request for a block that
#: quotes two or three documents, and paid for it in queueing latency.  The
#: ceiling keeps a policy that genuinely asks for more audible; the floor
#: keeps a narrow policy from retrieving so little that fusion has nothing to
#: fuse.
_MAX_RESULTS_FLOOR = 6
_MAX_RESULTS_CEILING = 12

#: Directory names that say "where the tree keeps code", not "which module
#: this is".  A `src/` tree and its `tests/` mirror share a module name under
#: them, and the plain common prefix of the two is the repository root — that
#: is, no scope at all.  Dropping them is what lets a turn that edits
#: `src/auth/jwt.py` and `tests/auth/test_jwt.py` still ask about `auth`.
_ROOT_DIRECTORIES = frozenset(
    {
        "src",
        "lib",
        "libs",
        "app",
        "apps",
        "pkg",
        "packages",
        "test",
        "tests",
        "spec",
        "specs",
        "integration",
        "integrationtest",
        "e2e",
    }
)


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
    signature : str
        Human-readable text of the failure this turn is stuck on, ``""``
        when there is none.  A digest only the writer can compare is not a
        question, and a query that carries the failure the agent is looking
        at retrieves the recipe that was written for it.
    test_command : str
        The command that failed, ``""`` when the turn ran none; the recipe
        names it, so it is a term the search can match on.
    """

    collection_namer: object
    min_similarity: float = 0.35
    signature: str = ""
    test_command: str = ""

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
            One request per source group; empty when the turn has nothing
            searchable or the scope is unknown.
        """
        if not turn.scope:
            return ()
        # A question is a failure, a path, a command — or a sentence worth
        # searching.  None of the three structured signals and a sentence
        # too short to be a question is a turn that has nothing to ask the
        # knowledge base, and the read is skipped rather than paid for.
        if (
            self._trivial_turn(turn)
            and len((turn.latest_user_text or "").strip()) < _MIN_QUESTION_CHARS
        ):
            return ()
        query = self.query_for(turn)
        if not query.strip():
            return ()
        grouped: Dict[str, List[str]] = {}
        limits: Dict[str, int] = {}
        # Which collection each group is asked of.  Grouping is by what a
        # request has to *be*, not by where the documents live: with one
        # collection per repository the three groups share a name, and only
        # their categories and filters differ.
        collections: Dict[str, str] = {}
        # A missing key is "restrictions of the sources seen so far"; ``None``
        # is "any".  One source that does not restrict makes the whole
        # collection unrestricted, because both are retrieved by one request
        # and the other's filter would silently drop half of what the mode
        # asked to read.
        phase_limits: Dict[str, Optional[List[str]]] = {}
        mode_limits: Dict[str, Optional[List[str]]] = {}
        doc_type_limits: Dict[str, Optional[List[str]]] = {}
        keywords_match_limits: Dict[str, str] = {}
        for source in policy.sources:
            kind = source.category
            archived = kind == KIND_SESSION_ARCHIVE
            if archived and not turn.thread_id:
                # Without a thread there is nothing to restrict the archive to,
                # and an unfiltered archive is another team's conversation.
                continue
            name = _bucket(kind)
            collections.setdefault(
                name, str(self.collection_namer(turn.scope, kind))
            )
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
            match = source.keywords_match or "and"
            if name not in keywords_match_limits:
                keywords_match_limits[name] = match
            elif keywords_match_limits[name] != "and":
                # A bucket is asked once: the engine combines the keywords one
                # way, so the strictest source in it wins.  A source that
                # wants ``any`` next to one that wants ``and`` is a
                # contradiction the engine cannot express, and the fix is the
                # one that retrieves less noise, not the one that retrieves
                # more.
                keywords_match_limits[name] = "and"
        requests: List[SearchRequest] = []
        for name, categories in grouped.items():
            archived = KIND_SESSION_ARCHIVE in categories
            documented = _is_documentation(categories)
            restricted = phase_limits.get(name)
            restricted_modes = mode_limits.get(name)
            restricted_types = doc_type_limits.get(name)
            requests.append(
                SearchRequest(
                    collection=collections.get(name, name),
                    # Documentation is retrieved on the turn's subject, not on
                    # its failure; everything else keeps the mode-shaped query.
                    query=self.docs_query_for(turn) if documented else query,
                    categories=tuple(dict.fromkeys(categories)),
                    max_results=_max_results_for(limits.get(name, 3)),
                    thread_id=turn.thread_id if archived else "",
                    # An archived window is already narrowed to one thread, and
                    # the engine cannot intersect two metadata filters.
                    phases=() if archived else tuple(restricted or ()),
                    modes=() if archived else tuple(restricted_modes or ()),
                    doc_types=() if archived else tuple(restricted_types or ()),
                    keywords=self.keywords_for(turn),
                    keywords_match=keywords_match_limits.get(name, "and"),
                    relative_path_contains=(
                        self.path_scope_for(turn)
                        if not archived and not documented
                        else ""
                    ),
                    # A debug turn is choosing between recipes by their order:
                    # the first one read is the one the agent tries next, so
                    # that one question is worth the cross-encoder when the
                    # first pass is undecided.  Archive and documentation
                    # orders are not chosen the way a fix is.
                    rerank_if_contested=(
                        not archived
                        and not documented
                        and turn.mode in ("debug", "test")
                    ),
                )
            )
        return tuple(requests)

    @staticmethod
    def _trivial_turn(turn: AgentTurn) -> bool:
        """
        Whether the turn carries no structured signal to search with.

        The three facts a query is built from are the failure the miner read,
        the files the turn touched and the command it ran; a turn with none of
        them has only its own sentence, which is what the character ceiling in
        :meth:`build` then judges.  The archive is not a signal here: it is a
        source the query is asked of, not a fact the query is built from.
        """
        if turn.touched_paths:
            return False
        if (_failure_signature(turn) or "").strip():
            return False
        if (_test_command(turn) or "").strip():
            return False
        return True

    def keywords_for(self, turn: AgentTurn) -> Tuple[str, ...]:
        """Extract lexical/BM25 keywords: exception types, symbols, paths."""
        failure = self.signature or _failure_signature(turn)
        cmd = self.test_command or _test_command(turn)
        return tuple(
            _extract_keywords(
                failure, cmd, turn.touched_paths, turn.latest_user_text
            )
        )

    def path_scope_for(self, turn: AgentTurn) -> str:
        """
        Return the module the touched paths belong to, or the empty string.

        The narrowest scope that holds every touched path is what a scoped
        search wants, and for a turn that edits a module and its test the
        plain common prefix is the repository root: `src/auth/jwt.py` and
        `tests/auth/test_jwt.py` share nothing above the first slash, so the
        filter the turn would ask for is the one that filters nothing, and
        the widening fallback then reads the whole repository.

        So the scope is read as a *module*, not as a prefix.  The directory
        names that say where a tree keeps code (`src`, `tests`, `lib`, `app`)
        carry no meaning about which part of it a turn is working on; under
        them the two paths name the same module, and that is the scope.  When
        the paths still disagree the prefix rule decides, and when even that
        is empty the search is asked unscoped — a search with no hits widens
        back to the whole repository in the reader, and a wrong narrow scope
        has no such remedy.
        """
        if not turn.touched_paths:
            return ""
        paths = [str(path) for path in turn.touched_paths if path and "/" in str(path)]
        if not paths:
            return ""
        return _module_scope(paths) or _common_prefix(paths)

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
            The query; ``""`` when the turn says nothing at all.
        """
        # The miner already read the failure out of the turn, on the
        # writer's normalization; asking again would be a second reader with
        # a second opinion, and the two could disagree on what the failure
        # is.  ``_failure_signature`` remains for a caller that has no
        # miner in hand.
        failure = self.signature or _failure_signature(turn)
        text = (turn.latest_user_text or "").strip()[:_MAX_TEXT_CHARS]
        paths = " ".join(turn.touched_paths[:12])
        mode = turn.mode

        if mode == "debug":
            # A failure the agent is staring at is two things: the sentence
            # that names it, and the twenty lines of somebody else's
            # internals below it.  The first is what a recipe was written
            # from; the second is a library's stack, and embedding it pulls
            # the query toward every document that mentions the same
            # framework.  The keywords still harvest the whole normalized
            # signature — the sparse leg wants the frames — while the dense
            # leg is asked with the head of it.
            parts: list = [_failure_head(failure), text]
        elif mode == "test":
            parts = [self.test_command or _test_command(turn), paths, text]
        elif mode == "implement":
            parts = [text, paths, failure]
        elif mode in ("git_review", "review"):
            parts = [paths, text]
        else:
            parts = [text, paths, failure]
        joined = _join(parts)
        if not joined:
            # A mode's own shape can come up empty — a review of a turn that
            # named no path, a test run with neither a command nor a word from
            # the user — and an empty query is a request that never asked the
            # knowledge base anything at all.  What it then asks with is the
            # composition every other mode already uses: the same three facts,
            # none of them invented, just not ordered for a mode that had
            # nothing to order.
            joined = _join([text, paths, failure])
        return joined[:_MAX_QUERY_CHARS]


#: A line that says a command lost, not what broke: an exit status, a frame
#: of somebody else's library, the seam of a traceback.  Every failure has
#: them, no document is written from them, and as the head of a dense query
#: they ask the vector store for the harness instead of for the bug.
_NOISE_LINE = re.compile(
    r"""^(?:"""
    r"""process\s+exited|"""
    # The writer's own digest of which files a trace crossed: a list of paths
    # under a label no document uses, useful to a human reading a log.
    r"""chain\s*:|"""
    r"""exit\s+(?:code|status)\b|"""
    r"""command\s+(?:failed|returned)\b|"""
    r"""traceback\s*\(most\s+recent\s+call\s+last\)|"""
    r"""file\s+"|"""
    r"""at\s+[\w.$]+\(|"""
    r"""\^+|"""
    r"""\.{3,})""",
    re.IGNORECASE,
)

#: A line that names a failure: an exception class, a pytest verdict, an
#: assertion, or the one ``word: message`` shape a normalized signature leads
#: with.  The first line matching this is the sentence a recipe was written
#: from, which is exactly what the dense leg is looking for.
_FAILURE_LINE = re.compile(
    r"""(?:\b[A-Z][A-Za-z0-9_]*(?:Error|Exception|Failure|Timeout|Warning)\b"""
    r"""|\b[A-Z][A-Za-z_]*Error\b"""
    r"""|^FAILED\b"""
    r"""|^E\s+"""
    r"""|^FAIL\b"""
    r"""|\bassert\b"""
    r"""|:\s*\S.*$)""",
    re.IGNORECASE,
)


def _failure_head(signature: str) -> str:
    """
    Return the line of a failure that names it, without the noise around it.

    A normalized signature is not ordered by informativeness.  The exit status
    of the command and the frames of the library that raised sit in the same
    block as ``KeyError: 'alg'``, and which one lands first depends on the
    tool that produced the output — so the head is *chosen*, not taken: the
    first line that names a failure, skipping the ones that only say that
    something lost.

    Retrieval wants that line because a recipe was written from it.  The
    frames below are shared by every failure that ever passed through the same
    library, so embedding them asks the store for the library; the exit status
    is in every failure ever recorded, so embedding it asks for nothing.  When
    no line names a failure the whole signature is the question — a head made
    up of noise asks about nothing, and the unshaped text is the older, safer
    default.

    The keywords are unaffected: the sparse leg still harvests the whole
    normalized signature, frames and all, because a frame is an exact token
    and the text index reads exact tokens well.
    """
    text = str(signature or "").strip()
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) <= 1:
        return text
    for line in lines:
        if _NOISE_LINE.match(line):
            continue
        if _FAILURE_LINE.search(line):
            return line[:_MAX_TEXT_CHARS]
    return text


def _join(parts: Sequence[str]) -> str:
    """Join the pieces of a query, dropping the ones that are empty."""
    return " \n".join(part.strip() for part in parts if part and part.strip())[
        :_MAX_QUERY_CHARS
    ]


def _is_documentation(categories: Sequence[str]) -> bool:
    """
    Return whether one collection holds documentation only.

    The ``doc`` category lives in its own collection, so a group that contains
    it contains nothing else.  Anything unexpected is treated as mined
    knowledge: a query carrying the failure signature is the older, safer
    default.
    """
    return tuple(str(kind) for kind in categories) == (KIND_DOC,)


def _bucket(kind: str) -> str:
    """
    Return the group of sources one document kind belongs to.

    Three groups, and they are not a restatement of the collections: each is a
    different **request**, because it needs a different thing of the engine.
    Mined knowledge is asked with the mode's own query and no metadata filter;
    an archived window is asked with the same query but restricted to one
    thread; documentation is asked with the turn's subject — the user's words
    and the paths — and restricted by ``doc_type``.  Under the split layout the
    groups happen to land in three collections; under one collection per
    repository they share a name, and only the categories and the filters tell
    them apart.

    The groups never merge, and that is not a stylistic choice.  A request
    carries at most one metadata filter, and the engine intersects filter
    families while unioning the filters inside the metadata family — so an
    archive request that also asked for mined knowledge would cut the mined
    documents to the ones sharing this ``thread_id``, silently dropping every
    recipe mined in an earlier session.
    """
    if kind == KIND_SESSION_ARCHIVE:
        return _BUCKET_ARCHIVE
    if kind == KIND_DOC:
        return _BUCKET_DOC
    return _BUCKET_KNOWLEDGE


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


def _max_results_for(limit: int) -> int:
    """
    Translate a policy's item caps into the ceiling to ask the engine for.

    The policy says how many documents of a source the assembler may *quote*;
    the engine is asked for more than that, because fusion, the signature
    bonus and the per-source caps all throw most of what comes back away.
    Asking for double with a ceiling is the whole compromise: enough candidates
    for a mode that reads four sources, not one hundred and fifty vectors for
    a mode that quotes two documents.
    """
    return max(_MAX_RESULTS_FLOOR, min(int(limit) * 2, _MAX_RESULTS_CEILING))


def _directories(path: str) -> list:
    """Directory components of a repository-relative path, innermost last."""
    head = str(path).replace("\\", "/").rsplit("/", 1)[0]
    return [part for part in head.split("/") if part]


def _without_roots(directory: Sequence[str]) -> List[str]:
    """Drop the leading directory names that name a tree, not a module."""
    parts = list(directory)
    while parts and parts[0].lower() in _ROOT_DIRECTORIES:
        parts = parts[1:]
    return parts


def _common_of(sequences: Sequence[Sequence[str]]) -> List[str]:
    """Longest leading run of components every sequence shares."""
    if not sequences:
        return []
    common = list(sequences[0])
    for sequence in sequences[1:]:
        depth = 0
        while (
            depth < len(common)
            and depth < len(sequence)
            and common[depth] == sequence[depth]
        ):
            depth += 1
        common = common[:depth]
    return common


def _common_prefix(paths: Sequence[str]) -> str:
    """Deepest directory every path shares; ``""`` when they share none."""
    directories = [_directories(path) for path in paths]
    return "/".join(_common_of(directories))


def _module_scope(paths: Sequence[str]) -> str:
    """
    Return the module directory the paths agree on, or ``""``.

    Each path is reduced to the directory it lives in with the tree names
    (`src`, `tests`, …) stripped from the front, which is the part that says
    what the file *is* rather than which half of the repository keeps it.  A
    module is worth scoping on only when every path agrees on it: scoping on
    one file's module because the others disagreed would hide the recipes
    about the rest of the turn, and an unscoped search is the honest answer
    to a turn that really is about several modules at once.
    """
    if len(paths) < 2:
        return ""
    stripped = [_without_roots(_directories(path)) for path in paths]
    if any(not parts for parts in stripped):
        return ""
    shared = _common_of(stripped)
    if not shared:
        return ""
    prefix = "/".join(shared)
    # A scope the plain prefix already provides adds nothing; returning the
    # stripped one would only be right when it is *narrower* than the prefix,
    # and a stripped path is never deeper than the directory it came from.
    if prefix and prefix in _common_prefix(paths):
        return ""
    return prefix


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


_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "that",
        "this",
        "it",
        "not",
        "by",
        "as",
        "from",
        "failed",
        "error",
        "exception",
        "assert",
        "test",
        "tests",
        # The verbs of an instruction to an agent say what the reader should
        # do, never what the code is: they are in every prompt and in no
        # document, so as BM25 terms they only dilute the ones that matter.
        "please",
        "fix",
        "run",
        "check",
        "update",
        "add",
        "make",
        "should",
        "would",
        "could",
        "will",
        "need",
        "needs",
        "want",
        "wants",
        "use",
        "using",
        "try",
        "tried",
        "still",
        "does",
        "did",
        "have",
        "has",
        "had",
        "can",
        "cant",
    }
)

#: Ceiling of the BM25 term list.  The engine scores every term it is given;
#: past a dozen or so the extra ones are the tail of the user's sentence, and
#: a long conjunction of weak terms retrieves the average document.
_MAX_KEYWORDS = 15


def _extract_keywords(
    failure: str,
    cmd: str,
    paths: Sequence[str],
    text: Optional[str] = None,
) -> List[str]:
    """
    Extract distinct lexical tokens for the sparse BM25 branch.

    Four harvests, strongest first.  An exception class and a file stem are
    terms a document was written with; an identifier in the failure's own
    text is next; and the user's sentence is last, because in `plan` and
    `implement` it is the only signal the turn has — a mode with no failure
    and no command used to send the engine an empty keyword list and rely on
    the vector leg alone for a question whose terms were spelled out in the
    prompt.

    The user's words are harvested for the shapes code is written in, not for
    English: a backticked term is a name the author chose deliberately, and a
    `snake_case` or `CamelCase` token is a symbol.  Prose in between is where
    the stopword list earns its keep, and it is why the agent's own verbs —
    *please*, *fix*, *run* — are stopwords too: they are in every request and
    in no document, so as terms they only dilute the ones that identify
    something.
    """
    tokens: List[str] = []
    # 1. Exception class names (e.g. ConnectionTimeoutError, TypeError)
    for exc in re.findall(
        r"\b[A-Z][a-zA-Z0-9]*(?:Error|Exception|Timeout|Failure|Warning)\b",
        failure or "",
    ):
        if exc not in tokens:
            tokens.append(exc)
    # 2. File stems
    for path in paths[:8]:
        name = str(path).rsplit("/", 1)[-1]
        stem = name.split(".", 1)[0] if "." in name else name
        if stem and stem.lower() not in _STOPWORDS and stem not in tokens:
            tokens.append(stem)
    # 3. Key identifiers in failure and command
    combined = f"{failure or ''} {cmd or ''}"
    for token in re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b", combined):
        lower = token.lower()
        if lower not in _STOPWORDS and token not in tokens and not token.isdigit():
            tokens.append(token)
        if len(tokens) >= _MAX_KEYWORDS:
            break
    # 4. Names the user spelled out: a backticked term, a snake_case or a
    # CamelCase identifier.  Prose words are not terms; these are.
    if text and len(tokens) < _MAX_KEYWORDS:
        for quoted in re.findall(r"`([^`]+)`", text):
            clean = re.sub(r"[^\w.\-]", "", quoted)
            if (
                len(clean) > 2
                and clean.lower() not in _STOPWORDS
                and clean not in tokens
            ):
                tokens.append(clean)
        for token in re.findall(
            r"\b(?:[a-z]+(?:_[a-z0-9]+)+|[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+)\b",
            text,
        ):
            if token.lower() not in _STOPWORDS and token not in tokens:
                tokens.append(token)
            if len(tokens) >= _MAX_KEYWORDS:
                break
    return tokens[:_MAX_KEYWORDS]
