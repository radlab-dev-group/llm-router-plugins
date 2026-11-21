"""
Choosing what to inject, and saying it in a fixed voice.

Several sources answer one question, and their scores are not comparable out of
the box: a reranker logit for a documentation page and a fused rank for a recipe
are both "relevance" only in the loosest sense.  The engine normalizes them onto
``[0, 1]``, the mode's policy says how much each source is worth, and two things
this plugin knows and the engine does not add the final margin: the document
names the failure the agent is *actually* looking at, and it names a file this
turn is actually touching.

After the votes comes subtraction, which matters more than the ranking.  A
prompt is not free, an agent reads what it is given in order, and a wall of
retrieved text teaches it to skim.  So: one entry per document, a hard character
budget per mode, and a fixed section order — a fix, then what not to repeat,
then what the summary forgot, then conventions.

The rendered text is machine-stable: the same hits always produce the same
block, which is what lets it be cached for a whole command generation without
the prompt changing under the provider's own cache.
"""

import datetime
import math
import re

from dataclasses import dataclass, field
from dataclasses import replace as _replace_meta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from llm_router_plugins.utils.knowledge.agentic_knowledge.config import ModePolicy
from llm_router_plugins.utils.knowledge.agentic_knowledge.learning import (
    CandidateFeatures,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.doc_context import (
    DocContext,
    render_doc_context,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.documents import (
    DOCUMENT_KINDS,
    KIND_DEADEND,
    KIND_DOC,
    KIND_PLAYBOOK,
    KIND_RECIPE,
    KIND_SESSION_ARCHIVE,
    KIND_TASK_SOLUTION,
    KIND_TOOL_RECIPE,
    DocumentMeta,
    parse_document_name,
    parse_document_text,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.sse import (
    SCALE_RERANKER,
    SCALE_RRF,
    SearchResult,
    StoreHit,
)

__all__ = [
    "ContextAssembler",
    "KnowledgeContext",
    "KnowledgeHit",
    "best_recipe_score",
    "deadends_of",
]

#: Marker around the injected block; also the adapter's idempotency detector.
BLOCK_OPEN = "<agentic_knowledge>"
BLOCK_CLOSE = "</agentic_knowledge>"

#: Bonus of a document that names the failure the agent is looking at.
SIGNATURE_BONUS = 0.2

#: Bonus of a document that names a file this turn is touching, per file.
PATH_BONUS = 0.08

#: Ceiling of the path bonus, so a big diff cannot outvote the failure.
_MAX_PATH_BONUS = 0.24

#: Bonus of a documentation file that documents a file this turn is touching —
#: `src/export.py` edited, `docs/export.md` retrieved.  Neither name contains
#: the other, so the shared stem is the only link between a module and its
#: description, and it is worth more than a shared directory: documentation
#: earns its place by being about the code in front of the agent.
DOC_STEM_BONUS = 0.12

#: Ceiling of the documentation affinity bonus, for a diff that touches half
#: the modules a big design document describes.
_MAX_DOC_BONUS = 0.24

#: Vote penalty of a recipe that an older recipe supersedes: both change the
#: same file, the newer one is verified and carries more judgement, and a
#: block that quoted both would hand the agent two conflicting fixes for one
#: bug and leave it to pick.  Half a vote keeps the older recipe readable in
#: a debug — it is still about the right file — without letting it compete
#: for the slot the newer one is owed.
_SUPERSEDED_PENALTY = 0.5

#: Most characters of a chunk quoted into the block.
_QUOTE_CHARS = 1800

#: Ceiling of the quote one lone, high-confidence recipe may spend: a verified
#: fix whose diff is cut in half cannot be applied, and the block that
#: survives the mode's budget is worth more than one that is short of it.
_WIDE_QUOTE_CHARS = 3000

#: Vote at and above which a recipe is "the" answer rather than one of a
#: field, and may spend the wide ceiling when the budget has room for it.
_ELASTIC_VOTE = 0.8

#: Fewest characters of a recipe worth quoting when a block is squeezed; below
#: this the entry is dropped rather than rendered as a truncated fragment.
_MIN_QUOTE_CHARS = 80

_INTRO = (
    "> [!IMPORTANT]\n"
    "Verified fixes, known dead ends and conventions retrieved from this "
    "repository's own record of earlier work. Check them against the code in "
    "front of you and ignore any part that no longer applies — do not repeat a "
    "dead end."
)

_SECTIONS: Tuple[Tuple[str, str], ...] = (
    (KIND_RECIPE, "## Verified fixes in this repo"),
    (KIND_TASK_SOLUTION, "## Verified task solutions in this repo"),
    (KIND_PLAYBOOK, "## Verified procedures in this repo"),
    (KIND_TOOL_RECIPE, "## Recommended tool commands in this repo"),
    (KIND_DEADEND, "## Known dead ends (DO NOT ATTEMPT)"),
    (KIND_SESSION_ARCHIVE, "## Previous Conversation Milestones"),
    (KIND_DOC, "## Conventions"),
)

#: Heading the standing documentation digest renders under.  Its own section,
#: ahead of the retrieved ones: the map of what a repository has written down
#: is read before the one paragraph a query happened to match.
DOC_CONTEXT_TITLE = "## Documentation in this repository"

#: Documentation names that describe the directory they sit in rather than a
#: module: a repository holds many `README.md` files and they share one stem,
#: so a stem match between a `README.md` and anything else says nothing.  What
#: links one is the directory it documents.
_GENERIC_DOCUMENT_NAMES = frozenset(
    {
        "agents",
        "changelog",
        "contributing",
        "faq",
        "index",
        "license",
        "notes",
        "readme",
        "releasing",
        "security",
        "todo",
    }
)


@dataclass(frozen=True)
class KnowledgeHit:
    """
    One document selected for injection.

    Parameters
    ----------
    category : str
        Document kind.
    document_name : str
        Engine-level name, for logging and citation.
    text : str
        The quoted content, header removed.
    score : float
        Normalized relevance the engine reported.
    weight : float
        The source's weight times the bonuses; what decided the order.
    created_at : float or None
        Publication second, when the document declared a date.
    paths : Tuple[str, ...]
        Files the document names.
    signature : str
        Failure it is about, when it has one.
    phase : str
        Work the episode was doing when it was learned; ``""`` when the
        document predates the attribute.
    source : str
        File a documentation section came from, repository-relative; ``""``
        for knowledge the miner produced.
    doc_type : str
        Kind of documentation file — ``readme``, ``guide``, ``changelog``;
        ``""`` outside the documentation collection.
    heading : str
        Heading path the section sits under, for the citation.
    signature_hash : str
        Exact failure identity, independent of retrieval relevance.
    test_command : str
        Verification command declared by a mined recipe.
    score_scale : str
        The scale :attr:`score` was produced on — ``rrf`` in a hybrid
        search, ``reranker`` under a cross-encoder, ``cosine`` for a
        vector-only run — so a threshold read against the score knows
        what the number means.  ``""`` when the answer did not state it.
    learning : Dict[str, Any]
        What the learner said about this one candidate, when it said anything:
        the multiplier it applied, the reasons for it, the model it read, and
        whether the read was an exploration.  ``{}`` when the learner had no
        say, which is the common case — no learner, no scope, or a policy
        switched off — and the only reason a field exists at all on a hit
        that never asked one.  The response must answer "why was this recipe
        first" per candidate, not per request, so the reasons travel with the
        candidate rather than living in a log nobody kept.
    """

    category: str = ""
    document_name: str = ""
    text: str = ""
    score: float = 0.0
    weight: float = 0.0
    created_at: Optional[float] = None
    paths: Tuple[str, ...] = ()
    signature: str = ""
    phase: str = ""
    source: str = ""
    doc_type: str = ""
    heading: str = ""
    signature_hash: str = ""
    test_command: str = ""
    score_scale: str = ""
    learning: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KnowledgeContext:
    """
    The assembled answer for one request.

    Parameters
    ----------
    hits : Tuple[KnowledgeHit, ...]
        What was injected, in the order it is rendered.
    rendered : str
        The block to inject; ``""`` when there is nothing to say.
    chars : int
        Length of :attr:`rendered`.
    doc_context : DocContext, optional
        The standing documentation digest that was injected, when any; the
        annotation reads its level, size and source files from here.
    best_recipe_score : float
        Highest retrieval relevance among recipes, not a success estimate.
    deadends : Tuple[KnowledgeHit, ...]
        The dead ends among the hits.
    cache : str
        ``hit``, ``miss``, ``disabled`` or ``unavailable``.
    status : str
        Retrieval status: ``ok``, ``disabled`` or ``unavailable``.
    """

    hits: Tuple[KnowledgeHit, ...] = ()
    rendered: str = ""
    chars: int = 0
    best_recipe_score: float = 0.0
    deadends: Tuple[KnowledgeHit, ...] = ()
    cache: str = "miss"
    status: str = "ok"
    doc_context: Optional[DocContext] = None
    #: How many engine searches the read made for this request, the ones the
    #: budget could not start included; the number a deployment reads to see
    #: how much the layer spends on the engine per request.
    searches: int = 0
    #: How many of those searches the read budget could not start.
    budget_skipped: int = 0
    #: How many of those searches were answered from this process's memo of an
    #: identical question instead of from the engine: the saving, not a cost.
    memo_hits: int = 0
    #: Why the cache did not serve this request: ``mode`` for a switch to a
    #: mode that reads a different question, ``evidence`` for new evidence,
    #: ``cold`` for nothing cached at all, ``""`` when the cache did serve.
    cache_miss_reason: str = ""

    @property
    def empty(self) -> bool:
        """Whether nothing was injected."""
        return not self.rendered


@dataclass
class _Candidate:
    """Internal: one hit with its vote and parsed metadata."""

    hit: KnowledgeHit
    meta: DocumentMeta
    vote: float = 0.0
    source_max: int = 3
    section: int = len(_SECTIONS)
    #: Ceiling of the rendered line; 0 means the entry renders whole.
    render_limit: int = 0


@dataclass(frozen=True)
class ContextAssembler:
    """
    Rank, trim and render the knowledge block.

    Parameters
    ----------
    signature_hash : str
        Failure this turn is stuck on; the strongest available bonus.
    touched_paths : Tuple[str, ...]
        Files this turn changed, for the secondary bonus.
    local_deadends : Tuple[str, ...]
        One-line summaries of what this turn already tried.
    """

    signature_hash: str = ""
    #: The failure's own words, for the continuous part of the signature bonus:
    #: a recipe written from ``KeyError: 'alg'`` and a turn stuck on
    #: ``KeyError: 'algn'`` are not the same document, but they are close
    #: enough that the binary hash says nothing and the words say something.
    signature: str = ""
    touched_paths: Tuple[str, ...] = ()
    local_deadends: Tuple[str, ...] = ()
    feedback: Optional[Dict[str, Dict[str, int]]] = None
    #: The learned multiplier of the layer, when the deployment runs one.  It
    #: answers ``multiplier_for(features, document_name, scope)`` and is asked
    #: only for candidates the ranking actually reaches; ``None`` — or a policy
    #: that is switched off — leaves every multiplier ``0.0``, and a ``0.0``
    #: means "the learner has no say": the vote is then the one this file
    #: computed before the learner existed, static counters and all.  A
    #: switched-off policy must not answer ``1.0``, because ``1.0`` is a vote,
    #: and a vote of exactly one is how a disabled layer would still silence
    #: the counters it used to read.
    learning: Optional[Any] = None
    #: Repository the block is assembled for; the key the model snapshot lives
    #: under.  Empty means no learner can be consulted, which is the same as
    #: having none.
    scope: str = ""
    #: Command this turn ran and lost with, when it names one; the learner's
    #: strongest "this recipe is about that failure" feature.
    failing_command: str = ""
    #: The lines this turn changed, read out of its patches: the side of the
    #: adoption comparison the credit assignment wants, kept here so the
    #: feature is computed once per candidate, not per verdict.  ``()`` when
    #: the turn carried no patch, and then the adoption feature is zero,
    #: which is the true statement: nothing was changed, so nothing was
    #: adopted.
    turn_patch_lines: Tuple[str, ...] = ()
    #: Times the same failure has come back this session; ``0`` is a turn on
    #: its first try and ``2`` and up is the loop.  A document worth reading
    #: changes with it: the first try wants the general and the third wants
    #: the specific, the verified, the about-this-failure.
    loop_count: int = 0
    #: Character budget the mode spends the block on; the denominator of the
    #: cost feature, because a document's worth is per character it occupies.
    #: ``0`` means "not charged", and the cost feature reads as zero rather
    #: than dividing by it.
    budget_chars: int = 0
    #: What each ranked candidate looked like to the learner, by document name.
    #: Filled while ranking so the caller can log the exposure it just made
    #: without computing the features a second time; empty when no learner is
    #: attached.
    candidate_features: Dict[str, Any] = field(default_factory=dict)

    def assemble(
        self,
        results: Sequence[Tuple[SearchResult, Sequence[str]]],
        policy: ModePolicy,
        *,
        extra_lines: Iterable[str] = (),
        doc_context: Optional[DocContext] = None,
    ) -> KnowledgeContext:
        """
        Return the block to inject for one request.

        Parameters
        ----------
        results : Sequence[Tuple[SearchResult, Sequence[str]]]
            One search result per request, with the request's categories.
        policy : ModePolicy
            The mode's policy: weights, per-source caps, character budget.
        extra_lines : Iterable[str], optional
            Lines from this turn itself (what it already tried), placed in the
            dead-ends section without competing for retrieval.
        doc_context : DocContext, optional
            The repository's standing documentation digest.  Rendered first and
            charged against the budget before anything is retrieved, so the
            always-present map is what the mode pays for first and the deep
            chunks spend what is left.

        Returns
        -------
        KnowledgeContext
            The block and what it is made of.
        """
        candidates = self._candidates(results, policy)
        extra = _deduplicated(
            [
                line.strip()
                for line in (*self.local_deadends, *extra_lines)
                if line and line.strip()
            ]
        )
        # The digest is a fixed cost the operator opted into, so it is measured
        # against the mode's budget before the variable parts: what is left is
        # what retrieval may spend.  When even the digest will not fit, it is
        # not injected at all rather than squeezing the block out from under it.
        digest = render_doc_context(doc_context) if doc_context else ""
        if digest and len(digest) + len(DOC_CONTEXT_TITLE) + 3 > policy.max_chars:
            digest = ""
        reserved = len(digest) + len(DOC_CONTEXT_TITLE) + 3 if digest else 0
        # The lines this turn supplies itself are rendered whatever else fits,
        # so they are charged against the budget first: without that a long
        # "already tried" list would push the block past the mode's budget,
        # which is the one number the operator can reason about.
        extra = _fit_extra(extra, policy.max_chars - _block_overhead() - reserved)
        # A mined fix and a written rule about the same module are two halves
        # of one answer, and neither search connects them: the recipe names
        # `src/export.py`, the page is `docs/export.md`.  The documentation map
        # the request already carries resolves the link, so an entry points at
        # the page whether or not the query happened to retrieve it.
        doc_files = tuple(doc_context.source_files) if doc_context else ()
        chosen = self._budget(
            candidates,
            policy.max_chars,
            _extra_cost(extra) + reserved,
            doc_files,
        )
        rendered = self._render(chosen, extra, digest, doc_files)
        hits = tuple(candidate.hit for candidate in chosen)
        return KnowledgeContext(
            hits=hits,
            rendered=rendered,
            chars=len(rendered),
            best_recipe_score=best_recipe_score(hits),
            deadends=deadends_of(hits),
            doc_context=doc_context if digest else None,
        )

    # ------------------------------------------------------------ ranking ---
    def _candidates(
        self,
        results: Sequence[Tuple[SearchResult, Sequence[str]]],
        policy: ModePolicy,
    ) -> List[_Candidate]:
        """Score every hit once, keep the best version of every document."""
        weights = {
            source.category: float(source.weight) for source in policy.sources
        }
        caps = {source.category: int(source.max_items) for source in policy.sources}
        types = {
            source.category: tuple(source.doc_types)
            for source in policy.sources
            if source.doc_types
        }
        order = {kind: index for index, (kind, _) in enumerate(_SECTIONS)}
        best: Dict[str, _Candidate] = {}
        for result, categories in results:
            scale = result.score_scale
            # Rank scales are only comparable inside the result set that
            # produced them: a fused rank says "this is the first, second,
            # third of what the engine could fuse", and the ceiling of that
            # statement is the best hit actually in hand.
            ceiling = _rank_ceiling(result.hits, scale)
            for position, hit in enumerate(result.hits):
                candidate = self._candidate(
                    hit,
                    categories,
                    weights,
                    caps,
                    order,
                    types,
                    scale,
                    ceiling,
                    position,
                    policy.name,
                )
                if candidate is None:
                    continue
                key = candidate.hit.document_name or candidate.hit.text[:120]
                current = best.get(key)
                if current is None or candidate.vote > current.vote:
                    best[key] = candidate
        self._suppress_superseded(best.values())
        ordered = sorted(
            best.values(),
            key=lambda item: (-item.vote, item.section, item.hit.document_name),
        )
        return ordered

    def _candidate(
        self,
        hit: StoreHit,
        categories: Sequence[str],
        weights: Dict[str, float],
        caps: Dict[str, int],
        order: Dict[str, int],
        types: Optional[Dict[str, Tuple[str, ...]]] = None,
        score_scale: str = "",
        rank_ceiling: float = 0.0,
        rank: int = 0,
        mode: str = "",
    ) -> Optional[_Candidate]:
        """Turn one retrieved chunk into a candidate, or drop it."""
        meta, body = self._meta(hit)
        kind = meta.kind or _kind_from_categories(hit, categories)
        if kind not in DOCUMENT_KINDS:
            return None
        declared = str(meta.doc_type or "").strip()
        allowed = (types or {}).get(kind, ())
        if allowed and declared and declared not in allowed:
            # The engine already filters the documentation collection on
            # doc_type; this is the same rule restated where it can also be
            # seen — a chunk indexed before the attribute existed, or a page
            # the reader found in a collection that predates the split.  A
            # document that declares no type at all is not another type, and
            # excluding documentation for a field it never wrote would lose a
            # guide over a header format.
            return None
        doc_type = _document_type(meta) if kind == KIND_DOC else ""
        base = (
            hit.score_normalized if hit.score_normalized is not None else hit.score
        )
        base = float(base) if base is not None else 0.0
        scale = score_scale or hit.score_scale
        # A fused rank and a cosine similarity are different numbers that
        # happen to share a name.  A bonus is a *shift* in the scale the
        # base is measured on: against a cosine the 0.2 of a matching
        # failure is a real margin, against a fused rank of ~0.03 it is an
        # order of magnitude that decides the ordering by itself.  Rank
        # scales are remapped onto [0, 1] before anything is added to
        # them; similarity scales already live there.
        if scale == SCALE_RRF and rank_ceiling > 0.0:
            base = min(1.0, base / rank_ceiling)
        elif scale == SCALE_RERANKER:
            normalized = hit.score_normalized
            if normalized is not None and 0.0 <= float(normalized) <= 1.0:
                base = float(normalized)
            else:
                try:
                    base = 1.0 / (1.0 + math.exp(-float(base)))
                except OverflowError:
                    base = 0.0 if base < 0 else 1.0
        weight = weights.get(kind, min(weights.values()) if weights else 1.0)
        learned, learning_detail = self._learning_multiplier(
            hit=hit,
            meta=meta,
            kind=kind,
            base=base,
            scale=scale,
            rank=rank,
            body=body,
            mode=mode,
        )
        # The learner's reputation and the counter below are two readings of
        # the same evidence, so only one of them ever weighs a vote: with an
        # enabled learner the counters stop multiplying, which is what keeps
        # the influence of feedback inside the range the policy names.  A
        # learner that answers ``0.0`` — switched off, or failed to read —
        # has no say at all, and the counters weigh the vote as they did
        # before the learner existed.
        feedback_mult = (
            1.0 if learned > 0.0 else self._feedback_multiplier(hit.document_name)
        )
        vote = (base * weight + self._bonus(meta, kind)) * feedback_mult * (
            learned if learned > 0.0 else 1.0
        )
        return _Candidate(
            hit=KnowledgeHit(
                category=kind,
                document_name=hit.document_name,
                text=_clean(body),
                score=base,
                weight=round(vote, 4),
                created_at=_timestamp(meta.date),
                paths=meta.paths,
                signature=meta.signature,
                phase=meta.phase,
                source=meta.source if kind == KIND_DOC else "",
                doc_type=doc_type,
                heading=meta.heading if kind == KIND_DOC else "",
                signature_hash=meta.signature_hash,
                test_command=meta.test_command,
                score_scale=scale,
                learning=dict(learning_detail or {}),
            ),
            meta=meta,
            vote=vote,
            source_max=caps.get(kind, 3),
            section=order.get(kind, len(_SECTIONS)),
        )

    def _meta(self, hit: StoreHit) -> Tuple[DocumentMeta, str]:
        """
        Read the header, filling what it lacks from the document name.

        Only the fields a name can supply are filled from it.  `source` is not
        one of them: it is the repository-relative file a documentation section
        was cut from, and the engine's document name is no substitute — it would
        turn `docs/guide.md` into `doc:20261016:9f2c` and cost the citation, and
        the kind inference, the only path it has.
        """
        meta, body = parse_document_text(hit.text)
        if not meta.kind or meta.date is None:
            from_name = parse_document_name(hit.document_name)
            meta = _replace_meta(
                meta,
                kind=meta.kind or from_name.kind,
                date=meta.date or from_name.date,
            )
        return meta, body

    def _bonus(self, meta: DocumentMeta, kind: str = "") -> float:
        """Reward the facts the engine cannot rank."""
        bonus = 0.0
        if self.signature_hash and meta.signature_hash == self.signature_hash:
            bonus += SIGNATURE_BONUS
        elif self.signature and meta.signature:
            # The hash is all-or-nothing, and a failure that differs by one
            # variable name or one path drops out of it entirely.  The words
            # of the two signatures still agree on most of their tokens, and
            # that agreement is worth a share of the full bonus: a near-identical
            # failure is most of the evidence an identical one is.
            bonus += SIGNATURE_BONUS * _signature_similarity(
                self.signature, meta.signature
            )
        if self.touched_paths and meta.paths:
            touched = {path.rsplit("/", 1)[-1] for path in self.touched_paths}
            shared = sum(
                1
                for path in meta.paths
                if path in self.touched_paths or path.rsplit("/", 1)[-1] in touched
            )
            bonus += min(_MAX_PATH_BONUS, shared * PATH_BONUS)
        if kind == KIND_DOC:
            bonus += self._doc_bonus(meta)
        return bonus

    def _doc_bonus(self, meta: DocumentMeta) -> float:
        """
        Reward documentation that documents a file this turn is touching.

        A guide is named after the module it explains more often than it names
        the module in its text: `docs/export.md` describes `src/export.py`, and
        neither string contains the other, so no retrieval leg connects them.
        The shared file stem does, and it is the difference between a page of
        house style and the one page about the code the agent is editing.
        """
        if not self.touched_paths:
            return 0.0
        touched = tuple(str(path) for path in self.touched_paths if str(path))
        if not touched:
            return 0.0
        shared = sum(
            1
            for document in _documentation_paths(meta)
            if _documents_path(document, touched)
        )
        if not shared:
            return 0.0
        return min(_MAX_DOC_BONUS, shared * DOC_STEM_BONUS)

    # ---------------------------------------------------------- supersession --
    def _suppress_superseded(self, candidates: Iterable[_Candidate]) -> None:
        """
        Dampen the older of two recipes that fix the same file.

        A file that was fixed twice has two recipes, and a block that quotes
        both is not two pieces of knowledge: it is a contradiction the agent
        must resolve and cannot, because both are worded as verified.  The
        newer one is what the repository does now; the older one is what it
        did until something turned out to be wrong with it.  The older stays
        available — it is still about the right file, and a debug turn may
        want the history — but at half a vote, so the slot it would otherwise
        win is the one the newer recipe is owed.

        Only recipes supersede each other.  A recipe and a dead end on the
        same file are not in conflict, they are the two sides of one lesson.
        """
        by_path: Dict[str, List[_Candidate]] = {}
        for candidate in candidates:
            if candidate.hit.category != KIND_RECIPE:
                continue
            for path in candidate.hit.paths or ():
                by_path.setdefault(str(path), []).append(candidate)
        for group in by_path.values():
            if len(group) < 2:
                continue
            for candidate in group:
                others = [item for item in group if item is not candidate]
                if any(self._supersedes(other, candidate) for other in others):
                    candidate.vote -= _SUPERSEDED_PENALTY

    def _supersedes(self, newer: _Candidate, older: _Candidate) -> bool:
        """
        Whether *newer* is the current answer to the file *older* also claims.

        Superseding is a statement about the repository, not about the
        ranking: the document has to be later, and the repository has to
        believe in it more.  The vote itself is not the evidence — it is what
        this adjusts — so the belief is read where the layer keeps it: the
        judgement a document has earned.  A newer recipe nobody has credited
        yet has not replaced anything, it merely arrived later, and demoting
        the recipe that has been working in favour of the untested one is the
        wrong way round.
        """
        new_date = newer.hit.created_at
        old_date = older.hit.created_at
        if new_date is None or old_date is None:
            # Without a date there is no order to supersede in.  A document
            # that does not say when it was learned is neither newer nor
            # older, and letting a missing field decide which fix the agent
            # reads would be a ranking by an accident of indexing.
            return False
        if new_date <= old_date:
            return False
        new_up, _ = self._votes(newer.hit.document_name)
        _, old_down = self._votes(older.hit.document_name)
        return new_up > old_down

    def _votes(self, document_name: str) -> Tuple[int, int]:
        """Recorded judgements of one document; ``(0, 0)`` when it has none."""
        stats = (self.feedback or {}).get(document_name) or {}
        return (
            int(stats.get("upvotes", 0) or 0),
            int(stats.get("downvotes", 0) or 0),
        )

    # ------------------------------------------------------------- budget ---
    def _budget(
        self,
        candidates: Sequence[_Candidate],
        max_chars: int,
        reserved: int = 0,
        doc_files: Sequence[str] = (),
    ) -> List[_Candidate]:
        """
        Apply per-source caps and the mode's character budget.

        The budget is charged for everything the block is made of: the markers,
        the intro, the title of every section that ends up with an entry, each
        entry, and whatever this turn contributes itself (:attr:`reserved`).
        Counting only the entries let a mode's documented budget be exceeded by
        the section headings alone.

        The quote ceiling is elastic, and that is the point of a budget: a
        single high-confidence recipe is the one answer a mode has, and a diff
        cut in half to fit the ceiling is a fix that cannot be applied; a
        field of weaker candidates shares the ceiling, because five full
        documents at the cost of one are five ways to be wrong.  The mode's
        ``max_chars`` is the hard limit in both cases — the elastic part is
        which candidates may spend it, never how much of it exists.
        """
        used: Dict[str, int] = {}
        opened: set = set()
        chosen: List[_Candidate] = []
        spent = _block_overhead() + reserved
        for candidate in candidates:
            kind = candidate.hit.category
            if used.get(kind, 0) >= candidate.source_max:
                continue
            head_room = max_chars - spent - (
                len(_section_title(kind)) + 1 if kind not in opened else 0
            )
            ceiling = self._quote_ceiling(candidate, head_room, doc_files)
            cost = len(_entry(candidate, limit=ceiling, doc_files=doc_files)) + 1
            if kind not in opened:
                cost += len(_section_title(kind)) + 1
            if spent + cost > max_chars:
                continue
            chosen.append(candidate)
            opened.add(kind)
            # The ceiling the entry was charged at is the ceiling it renders
            # to, so the cost the budget spent and the text the block holds
            # are the same number, not two numbers that happened to agree.
            candidate.render_limit = ceiling
            used[kind] = used.get(kind, 0) + 1
            spent += cost
        if not chosen:
            chosen = _force_one_recipe(candidates, max_chars - spent, opened)
        return chosen

    def _quote_ceiling(
        self, candidate: _Candidate, room: int, doc_files: Sequence[str] = ()
    ) -> int:
        """
        How much of the mode's budget this entry may spend on its quote.

        ``room`` is what the budget has left before this entry, after the
        section title it may open.  The default is the flat ceiling.  A
        recipe that is the strongest candidate in the field — the vote the
        ranking gave it puts it first, not a cap it merely survived — may
        spend what the room actually is, up to the wide ceiling: a diff that
        fits is worth the characters, and the room is the mode's own number,
        so spending it is not an overrun.  Anything else keeps the flat
        ceiling, because a field of candidates is a field, and each of them
        is owed the same line.

        A ceiling below the entry's own head is a ceiling that quotes
        nothing, and ``_entry`` would render it as empty — so the ceiling
        never comes back smaller than the head, and an entry that cannot be
        quoted at all is dropped by the caller, not broken into a fragment.
        """
        head_len = _entry_head_len(candidate, doc_files)
        if head_len >= _WIDE_QUOTE_CHARS:
            return max(_MIN_QUOTE_CHARS, head_len)
        if candidate.vote >= _ELASTIC_VOTE and room - 1 > head_len:
            return max(head_len, min(_WIDE_QUOTE_CHARS, room - 1))
        return _QUOTE_CHARS

    def _learning_multiplier(
        self,
        hit: StoreHit,
        meta: DocumentMeta,
        kind: str,
        base: float,
        scale: str,
        rank: int,
        body: str,
        mode: str = "",
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Ask the learner what this candidate is worth, and remember what it asked.

        Returns ``(0.0, {})`` when the learner has no say — no learner, no
        scope, a policy switched off, or a learner that raised — and the
        caller then weighs the candidate exactly as it did before the learner
        existed, static counters included.  ``0.0`` means "not the learner's
        vote", never "a vote of one", and that distinction is what keeps a
        disabled deployment from silently substituting a multiplier of exactly
        one for the counters it used to read.  The second value is the
        learner's own account of the number — the reasons, the model it read,
        the exploration flag — which the candidate carries on
        :attr:`KnowledgeHit.learning` so the response can answer why this one
        ranked where it did.  The features are recorded whatever the answer
        was, including from a switched-off policy: an injection whose exposure
        was never logged cannot be credited or blamed a turn later, and the
        log is the one part of this that cannot be reconstructed afterwards —
        collecting while learning is off is how the first corpus comes to
        exist at all.
        """
        learner = self.learning
        if learner is None or not self.scope or not hit.document_name:
            return 0.0, {}
        features = self._features(
            meta=meta,
            kind=kind,
            base=base,
            scale=scale,
            rank=rank,
            body=body,
            mode=mode,
            document_name=hit.document_name,
        )
        self.candidate_features[hit.document_name] = features
        if not getattr(learner, "enabled", True):
            return 0.0, {}
        try:
            multiplier, detail = learner.multiplier_for(
                features, document_name=hit.document_name, scope=self.scope
            )
        except Exception:  # a learner is never a reason to fail a read
            return 0.0, {}
        if not multiplier or multiplier <= 0.0:
            return 0.0, dict(detail or {})
        return float(multiplier), dict(detail or {})

    def _features(
        self,
        meta: DocumentMeta,
        kind: str,
        base: float,
        scale: str,
        rank: int,
        body: str,
        mode: str = "",
        document_name: str = "",
    ) -> CandidateFeatures:
        """
        Describe one candidate for the learner, in numbers only.

        Every value is one the ranking above already computed or the document
        header declares.  Nothing here is content: no text, no path, no command
        — only whether the command matched.  The record goes to the training
        stream, and the worker masks the documents it is fitted on precisely so
        that the layer never keeps content it cannot re-identify.
        """
        touched = self.touched_paths
        shared = 0
        stems = 0
        if touched and meta.paths:
            touched_names = {path.rsplit("/", 1)[-1] for path in touched}
            for path in meta.paths:
                if path in touched:
                    shared += 1
                elif path.rsplit("/", 1)[-1] in touched_names:
                    stems += 1
        age_days = 0.0
        if meta.date is not None:
            today = datetime.date.today()
            age_days = float(max(0, (today - meta.date).days))
        stats = (self.feedback or {}).get(document_name, {}) or {}
        up = int(stats.get("upvotes", 0) or 0)
        down = int(stats.get("downvotes", 0) or 0)
        signature_match = bool(
            self.signature_hash and meta.signature_hash == self.signature_hash
        )
        # The words of the two failures, when the hash is silent: a failure
        # apart by one variable name is not the same document, but it is the
        # same bug, and the Jaccard of their tokens says how close.
        similarity = 0.0
        if not signature_match and self.signature and meta.signature:
            similarity = _signature_similarity(self.signature, meta.signature)
        doc_class = _exception_class(meta.signature)
        return CandidateFeatures(
            kind=kind,
            mode=mode,
            score=float(base),
            score_scale=scale,
            signature_match=signature_match,
            path_match=shared,
            path_stem_match=min(stems, 4),
            command_match=self._command_matches(meta),
            age_days=age_days,
            upvotes=up,
            downvotes=down,
            # Judgements, not impressions.  The request path counts what the
            # deployment has *decided* about a document, never how often it was
            # displayed: an exposure count would cost one Redis read per
            # candidate, and the read path is the one place this layer is not
            # allowed to spend a round-trip.
            exposure=up + down,
            rank=rank,
            size_chars=len(body or ""),
            budget_chars=self.budget_chars,
            signature_similarity=similarity,
            patch_sim=_adoption(body, self.turn_patch_lines),
            intent_align=_intent_aligned(kind, mode),
            error_class=_exception_class(self.signature),
            doc_error_class=doc_class,
            loop_count=self.loop_count,
        )

    def _command_matches(self, meta: DocumentMeta) -> bool:
        """
        Whether the recipe's verification command is the one now failing.

        A mined recipe names the command that turned it green; the turn names
        the command that went red.  When they are the same command, the recipe
        is about this failure and not merely about text that resembles it — the
        strongest evidence available without asking a model anything.
        """
        declared = str(meta.test_command or "").strip()
        failing = str(self.failing_command or "").strip()
        if not declared or not failing:
            return False
        if declared == failing:
            return True
        return declared.split()[-1] == failing.split()[-1]

    def _feedback_multiplier(self, document_name: str) -> float:
        """Weight multiplier based on recorded reinforcement feedback."""
        if not self.feedback or not document_name:
            return 1.0
        stats = self.feedback.get(document_name)
        if not stats:
            return 1.0
        up = stats.get("upvotes", 0)
        down = stats.get("downvotes", 0)
        score = (up + 2) / (up + down + 2)
        return round(0.3 + 1.2 * score, 4)

    def _anti_patterns_block(self, chosen: Sequence[_Candidate]) -> str:
        """Render pre-emptive anti-pattern warning for dead ends on touched files."""
        if not self.touched_paths:
            return ""
        touched = {p.rsplit("/", 1)[-1] for p in self.touched_paths if p}
        matched = []
        for cand in chosen:
            if cand.hit.category == KIND_DEADEND:
                paths = cand.hit.paths or ()
                if any(
                    p in self.touched_paths or p.rsplit("/", 1)[-1] in touched
                    for p in paths
                ):
                    matched.append(cand)
        if not matched and not self.local_deadends:
            return ""
        lines = [
            "<anti_patterns_warning>",
            "CRITICAL: Previous attempts in touched files failed:",
        ]
        for cand in matched[:3]:
            path_str = (
                f"In `{', '.join(cand.hit.paths[:2])}`: " if cand.hit.paths else ""
            )
            text = cand.hit.text
            summary = text.splitlines()[0] if text else "Attempted fix failed"
            if len(summary) > 160:
                summary = summary[:160] + "…"
            lines.append(f"- {path_str}{summary}")
        for deadend in self.local_deadends[:2]:
            lines.append(f"- This turn already tried: {deadend}")
        lines.append("DO NOT repeat these approaches.")
        lines.append("</anti_patterns_warning>")
        return "\n".join(lines)

    # -------------------------------------------------------------- render --
    def _render(
        self,
        chosen: Sequence[_Candidate],
        extra_lines: Sequence[str],
        doc_context: str = "",
        doc_files: Sequence[str] = (),
    ) -> str:
        """Render the block, sections in the fixed order."""
        if not chosen and not extra_lines and not doc_context:
            return ""
        grouped: Dict[int, List[str]] = {}
        for candidate in chosen:
            grouped.setdefault(candidate.section, []).append(
                _entry(candidate, doc_files=doc_files)
            )
        if extra_lines:
            deadend_section = _section_index(KIND_DEADEND)
            grouped.setdefault(deadend_section, []).extend(
                _extra_line(line) for line in extra_lines
            )
        parts = [BLOCK_OPEN, _INTRO]
        anti_patterns = self._anti_patterns_block(chosen)
        if anti_patterns:
            parts.append(anti_patterns)
        if doc_context:
            parts.extend((DOC_CONTEXT_TITLE, doc_context))
        for index, (_, title) in enumerate(_SECTIONS):
            entries = grouped.get(index)
            if not entries:
                continue
            parts.extend((title, *entries))
        parts.append(BLOCK_CLOSE)
        return "\n".join(parts)


def _format_entry_content(text: str, category: str) -> str:
    """Format recipe, playbook, deadend, or doc content with clean markdown."""
    clean_text = (text or "").strip()
    if not clean_text:
        return ""
    if category in (KIND_RECIPE, KIND_PLAYBOOK, KIND_DEADEND, KIND_TASK_SOLUTION):
        for marker in (
            "Diff that made it pass:",
            "Steps that worked:",
            "Solution diff:",
            "Change that did not fix it:",
        ):
            if marker in clean_text:
                parts = clean_text.split(marker, 1)
                prefix = parts[0].strip()
                code_part = parts[1].strip()
                if code_part and "```" not in code_part:
                    code_part = f"```diff\n{code_part}\n```"
                return f"{prefix}\n{marker}\n{code_part}".strip()
    return clean_text


def _format_archive_milestones(text: str) -> str:
    """Extract decisions, environment context, and milestones from an archive."""
    lines = (text or "").splitlines()
    milestones = []
    env_blocks = []
    in_env = False
    for line in lines:
        stripped = line.strip()
        if "environment" in stripped.lower() or "context" in stripped.lower():
            in_env = True
        if in_env:
            env_blocks.append(stripped)
            if len(env_blocks) > 5:
                in_env = False
        markers = ("milestone", "decision", "goal", "plan:", "summary:")
        if any(marker in stripped.lower() for marker in markers):
            milestones.append(stripped)
    if milestones or env_blocks:
        parts = []
        if env_blocks:
            parts.append("Environment Context: " + " ".join(env_blocks[:4]))
        if milestones:
            listed = "\n".join(f"- {m}" for m in milestones[:5])
            parts.append(f"Key Milestones:\n{listed}")
        return "\n".join(parts)
    return text


def _truncate_with_code_blocks(text: str, limit: int) -> str:
    """Truncate text to limit, preserving markdown code block syntax integrity."""
    if len(text) <= limit:
        return text
    truncated = text[:limit].rstrip() + "…"
    fences = [m.start() for m in re.finditer(r"```", truncated)]
    if len(fences) % 2 != 0:
        close_fence = "\n```"
        if len(truncated) + len(close_fence) <= limit:
            truncated += close_fence
        else:
            avail = max(0, limit - len(close_fence) - 1)
            truncated = text[:avail].rstrip() + "…" + close_fence
    return truncated


def _entry(
    candidate: _Candidate,
    limit: Optional[int] = None,
    doc_files: Sequence[str] = (),
) -> str:
    """
    Render one hit as one bullet of its section.

    ``limit`` cuts the quoted text so the rendered line is at most that long;
    it defaults to the ceiling the candidate was given when it was selected,
    and is used only when the block has room for nothing else and the
    alternative is injecting no recipe at all.  ``doc_files`` is the scope's
    documentation map: a mined entry names the page that documents the module
    it changed, which turns "here is the fix" into "here is the fix, and here
    is what this repository says a module like this must do".
    """
    hit = candidate.hit
    if hit.category == KIND_DOC:
        citation = [KIND_DOC, hit.doc_type, _documentation_citation(hit)]
    else:
        citation = [hit.category, f"in {hit.phase}" if hit.phase else ""]
        if hit.created_at:
            citation.append(
                datetime.datetime.fromtimestamp(
                    hit.created_at, tz=datetime.timezone.utc
                )
                .date()
                .isoformat()
            )
        if hit.paths:
            citation.append(", ".join(hit.paths[:4]))
    head = f"- [{' · '.join(part for part in citation if part)}] "
    text = _format_entry_content(hit.text, hit.category)
    if hit.category == KIND_SESSION_ARCHIVE:
        text = _format_archive_milestones(text)
    if len(text) > _QUOTE_CHARS:
        text = _truncate_with_code_blocks(text, _QUOTE_CHARS)
    pointer = _documentation_pointer(hit, doc_files)
    ceiling = candidate.render_limit if limit is None else limit
    if 0 < ceiling < len(head) + len(text) + len(pointer):
        room = ceiling - len(head) - len(pointer) - 1
        if room < _MIN_QUOTE_CHARS:
            return ""
        text = _truncate_with_code_blocks(text, room)
    return head + text + pointer


def _documentation_pointer(hit: KnowledgeHit, doc_files: Sequence[str]) -> str:
    """
    Return the ` → docs/export.md` tail one mined entry deserves, or ``""``.

    A verified fix says what worked; the repository's own page about that
    module says what it has to keep doing.  Nothing in the retrieval connects
    the two — a recipe is filed under a failure, a page under a module — so
    the shared file stem is resolved here, against the documentation map the
    request already carries rather than against another search.

    Documentation entries point at nothing (they are the pointed-at), and a
    path with no documentation about it gets no pointer: a citation is worth
    its characters only when it leads somewhere.
    """
    if hit.category == KIND_DOC or not hit.paths or not doc_files:
        return ""
    paths = tuple(str(path) for path in hit.paths if str(path))
    if not paths:
        return ""
    for path in doc_files:
        if path in paths:
            # A citation is not a pointer: the entry already says where the
            # thing is, so pointing back at it spends characters on nothing.
            continue
        if _documents_path(path, paths):
            return f" → {path}"
    return ""


_ERROR_CLASS = re.compile(
    r"\b([A-Z][A-Za-z0-9_]*(?:Error|Exception|Failure|Timeout|Warning))\b"
)


def _exception_class(signature: str) -> str:
    """
    The class of the exception a failure names, ``""`` when it names none.

    The class is the part of a failure that is stable from one occurrence to
    the next: the message changes with the input, the class does not.  A
    document written from ``KeyError: 'alg'`` and a turn stuck on
    ``KeyError: 'session_id'`` are two failures of one kind, and the kind is
    what a recipe is most likely to be filed under after the signature.
    """
    match = _ERROR_CLASS.search(str(signature or ""))
    return match.group(1) if match else ""


#: The kinds of document each mode is asking for: the engine matches on text,
#: and text matching cannot see whether the match is the kind of document the
#: turn wants.  A debug turn is looking for a fix, a plan for a procedure or
#: a page, an implement for either; a review is looking for a page.
_INTENT: Dict[str, Tuple[str, ...]] = {
    "debug": (KIND_RECIPE, KIND_TOOL_RECIPE),
    "test": (KIND_RECIPE, KIND_TOOL_RECIPE),
    "plan": (KIND_PLAYBOOK, KIND_DOC, KIND_TASK_SOLUTION),
    "implement": (KIND_RECIPE, KIND_PLAYBOOK, KIND_DOC, KIND_TASK_SOLUTION),
    "git_review": (KIND_DOC,),
    "review": (KIND_DOC,),
}


def _intent_aligned(kind: str, mode: str) -> bool:
    """Whether *kind* is what *mode* is asking the knowledge base for."""
    wanted = _INTENT.get(str(mode))
    if wanted is None:
        # A mode the table does not know is not a mode the table can judge;
        # the honest answer is "not a misalignment", and that is the feature
        # reading as false rather than as true.
        return False
    return str(kind) in wanted


def _adoption(body: str, turn_lines: Sequence[str]) -> float:
    """
    How much of the document's change the turn's own change repeats.

    The comparison is on the lines a fix adds — the ``+`` lines of the
    document's diff — against the lines the turn changed, whatever the turn's
    tool spelled them.  A turn that applied the fix the document describes
    repeats most of those lines; a turn that edited the same file for another
    reason repeats almost none.  The denominator is the document's lines,
    because the question is "did the turn do what this document says", and a
    document with no added lines has nothing to be adopted.

    ``0.0`` when either side is empty, which is the true statement in both
    cases: a document that adds nothing is not a fix to adopt, and a turn
    that changed nothing adopted nothing.
    """
    document_lines = _added_lines(body)
    if not document_lines or not turn_lines:
        return 0.0
    changed = {str(line) for line in turn_lines}
    overlap = sum(1 for line in document_lines if line in changed)
    return overlap / float(len(document_lines))


def _added_lines(text: str) -> set:
    """The lines a patch adds, minus their ``+``, deduplicated."""
    return {
        line[1:].strip()
        for line in str(text or "").splitlines()
        if line.startswith("+") and not line.startswith("+++") and line[1:].strip()
    }


def _signature_similarity(signature: str, other: str) -> float:
    """
    How much two failure signatures agree, between 0 and 1.

    A token of a failure is a word, a number, an operator, or the class of
    the exception — the parts that name what broke.  The Jaccard of those two
    token sets is the overlap a document's failure and a turn's failure share,
    independent of the words a human would add or drop around them.  One
    variable name apart, two signatures share most of their tokens and read
    as the same bug; two different bugs in the same file share almost none.

    ``1.0`` is reserved for the identical pair, which the hash already
    rewards at full strength; this function is only asked about the pairs the
    hash refused.
    """
    a = str(signature or "").strip()
    b = str(other or "").strip()
    if not a or not b or a == b:
        return 0.0
    tokens_a = set(re.findall(r"\w+|[+\-*/=!<>]+", a.lower()))
    tokens_b = set(re.findall(r"\w+|[+\-*/=!<>]+", b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / float(len(tokens_a | tokens_b))


def _documents_path(document: str, paths: Sequence[str]) -> bool:
    """
    Return whether *document* is the page about one of *paths*.

    Two conventions connect a page to code.  A named page is named after the
    module it explains — `docs/export.md` for `src/export.py` — which a shared
    file stem catches, neither string containing the other.  A generic page
    (`README.md`, `AGENTS.md`) documents the directory it sits in, and shares
    its stem with every other one in the repository, so it matches only inside
    its own directory; the root `README.md` would otherwise be documentation of
    everything, and a pointer to it would be noise on every entry.
    """
    text = str(document).replace("\\", "/")
    name = text.rsplit("/", 1)[-1]
    stem = (name.split(".", 1)[0] if "." in name else name).strip().lower()
    directory = text.rsplit("/", 1)[0] + "/" if "/" in text else ""
    if stem and stem not in _GENERIC_DOCUMENT_NAMES:
        return stem in _stems_of(paths)
    if not directory or directory == "/":
        return False
    return any(str(path).replace("\\", "/").startswith(directory) for path in paths)


def _documentation_citation(hit: KnowledgeHit) -> str:
    """
    Render where a documentation section came from: `docs/deploy.md › Production`.

    A mined recipe is cited by the files it fixed; a documentation section is
    cited by the file it was cut from and the heading it sits under, which is
    how an agent gets from a quoted paragraph back to the page it should read
    in full.  With no file name left the section is useless to attribute, and
    an empty part is dropped by the caller.
    """
    source = str(hit.source or "").strip()
    if not source:
        return ""
    heading = str(hit.heading or "").strip()
    return f"{source} › {heading}" if heading else source


def _entry_head_len(candidate: _Candidate, doc_files: Sequence[str] = ()) -> int:
    """The part of an entry that is not its quote: bullet, citation, pointer."""
    hit = candidate.hit
    citation = [hit.category, f"in {hit.phase}" if hit.phase else ""]
    if hit.created_at:
        citation.append(
            datetime.datetime.fromtimestamp(
                hit.created_at, tz=datetime.timezone.utc
            )
            .date()
            .isoformat()
        )
    if hit.paths:
        citation.append(", ".join(hit.paths[:4]))
    head = f"- [{' · '.join(part for part in citation if part)}] "
    return len(head) + len(_documentation_pointer(hit, doc_files))


def _block_overhead() -> int:
    """What the block costs before any entry: markers, intro, blank lines."""
    return len(BLOCK_OPEN) + len(BLOCK_CLOSE) + len(_INTRO) + 8


def _section_title(kind: str) -> str:
    """The heading one kind renders under, or ``""`` when it opens no section."""
    for name, title in _SECTIONS:
        if name == kind:
            return title
    return ""


def _extra_line(line: str) -> str:
    """Render one "this turn already tried" line."""
    return f"- this turn already: {line.strip()}"


def _extra_cost(lines: Sequence[str]) -> int:
    """What the turn's own lines will cost in the rendered block."""
    if not lines:
        return 0
    cost = len(_section_title(KIND_DEADEND)) + 1
    return cost + sum(len(_extra_line(line)) + 1 for line in lines)


def _fit_extra(lines: Sequence[str], budget: int) -> Tuple[str, ...]:
    """Keep as many of the turn's own lines as the mode's budget allows."""
    if budget <= 0:
        return ()
    kept: List[str] = []
    spent = len(_section_title(KIND_DEADEND)) + 1
    for line in lines:
        cost = len(_extra_line(line)) + 1
        if spent + cost > budget:
            break
        kept.append(line)
        spent += cost
    return tuple(kept)


def _force_one_recipe(
    candidates: Sequence[_Candidate], room: int, opened: set
) -> List[_Candidate]:
    """
    Make room for the best recipe when nothing else fit.

    A mode that asked for recipes and has one should not answer with an empty
    block because the first candidate was verbose: the entry is shortened to
    what is left rather than dropped.
    """
    for candidate in candidates:
        if candidate.hit.category != KIND_RECIPE:
            continue
        available = room
        if KIND_RECIPE not in opened:
            available -= len(_section_title(KIND_RECIPE)) + 1
        # A limit of 0 means "no ceiling" to `_entry`, so it must never be what
        # a too-small `available` collapses to — that would render the recipe at
        # full length and blow the mode's budget.  No room means no recipe.
        limit = available - 1
        if limit <= 0:
            return []
        if not _entry(candidate, limit=limit):
            return []
        return [_replace_meta(candidate, render_limit=limit)]
    return []


def _section_index(kind: str) -> int:
    """Position of one section in the fixed order."""
    for index, (name, _) in enumerate(_SECTIONS):
        if name == kind:
            return index
    return len(_SECTIONS)


def _rank_ceiling(hits: Sequence[StoreHit], score_scale: str) -> float:
    """
    Return the best fused rank of one result set, ``0.0`` when it has none.

    A hybrid search returns ranks whose absolute values are meaningless
    outside the set — the first hit of a two-leg fusion sits at ~0.033 — so
    the scale has to be read from what the engine actually returned.  A set
    whose best rank is not a positive number cannot be normalized and is
    left alone (``0.0`` means "do not rescale").
    """
    if score_scale != SCALE_RRF:
        return 0.0
    ranks = [
        float(value)
        for value in (
            hit.score_normalized if hit.score_normalized is not None else hit.score
            for hit in hits
        )
        if value is not None and math.isfinite(float(value)) and float(value) > 0.0
    ]
    return max(ranks) if ranks else 0.0


def best_recipe_score(hits: Iterable[KnowledgeHit]) -> float:
    """
    Return how strongly the best recipe matches, on the engine's own scale.

    This is the number the escalation policy compares against
    ``strong_recipe_score``, so it is the *normalized relevance the engine
    reported* and not the weighted vote that ordered the block: a source's
    weight and the signature and path bonuses decide what is read first, while
    the threshold has to mean the same thing in every mode's policy.
    """
    recipes = [hit.score for hit in hits if (hit.category or "") == KIND_RECIPE]
    return max(recipes) if recipes else 0.0


def deadends_of(hits: Iterable[KnowledgeHit]) -> Tuple[KnowledgeHit, ...]:
    """Return the dead ends among *hits*, in the order they were rendered."""
    return tuple(hit for hit in hits if (hit.category or "") == KIND_DEADEND)


def _kind_from_categories(hit: StoreHit, categories: Sequence[str]) -> str:
    """
    Read a kind from a chunk that carries no header.

    The document name is the reliable signal — this plugin names every document
    it writes ``<kind>:<day>:<fingerprint>``.  Failing that, a search that asked
    for exactly one kind says what came back; a search that asked for several
    says nothing, and guessing the first would file a conventions page under
    “verified fixes”, at the recipes' weight and as a recipe for the escalation
    policy to believe.
    """
    match = re.match(r"^([a-z_]+):", hit.document_name or "")
    if match and match.group(1) in DOCUMENT_KINDS:
        return match.group(1)
    named = [kind for kind in DOCUMENT_KINDS if kind in categories]
    if len(categories) == 1 and named:
        return named[0]
    return KIND_DOC


def _documentation_paths(meta: DocumentMeta) -> Tuple[str, ...]:
    """Return every path a documentation chunk is known to come from."""
    paths = [str(meta.source or "")]
    paths.extend(str(path) for path in meta.paths)
    return tuple(path for path in paths if path)


def _stems_of(paths: Sequence[str]) -> set[str]:
    """
    Return the file stems of *paths*, without their extensions.

    `docs/export.md` and `src/export.py` share the stem `export`; `README` and
    `readme` share one lowercased stem, because a file is named `README.md` and
    written about as “the readme”.  A path with no directory and no suffix is
    its own stem.
    """
    stems = set()
    for path in paths:
        name = str(path).replace("\\", "/").rsplit("/", 1)[-1]
        stem = name.split(".", 1)[0] if "." in name else name
        stem = stem.strip().lower()
        if len(stem) > 1:
            stems.add(stem)
    return stems


def _document_type(meta: DocumentMeta) -> str:
    """
    Return the kind of documentation a chunk came from, for its citation.

    The header says it.  A chunk that predates the attribute still names its
    file, and a file named `CHANGELOG.md` is a changelog whatever its header
    forgot — so the classification is repeated from the path rather than the
    citation losing the word.
    """
    declared = str(meta.doc_type or "").strip()
    if declared:
        return declared
    path = str(meta.source or "") or (meta.paths[0] if meta.paths else "")
    if not path:
        return ""
    from llm_router_plugins.utils.knowledge.agentic_knowledge.docs_ingestor import (
        classify_document,
    )

    return classify_document(path)


def _deduplicated(lines: Iterable[str]) -> Tuple[str, ...]:
    """Keep the first of every equal line, preserving order."""
    seen = set()
    kept: List[str] = []
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        kept.append(line)
    return tuple(kept)


def _clean(body: str) -> str:
    """Clean the noise of a quoted chunk while preserving line and code structure."""
    if not body:
        return ""
    text = body.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _timestamp(date: Optional[datetime.date]) -> Optional[float]:
    """Turn a publication day into a comparable second."""
    if date is None:
        return None
    moment = datetime.datetime(
        date.year, date.month, date.day, tzinfo=datetime.timezone.utc
    )
    return moment.timestamp()
