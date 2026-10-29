"""
Knowledge-or-escalate: the retrieval answer steers the model, not just the text.

An agent that keeps hitting the same wall is either one good answer away from
a fix or spinning in a loop that no more of the same model will break.  The
knowledge base tells the two apart: a strong recipe for exactly this failure
earns one chance for the routed model.  A further failure after that recipe
was offered, or a loop with no matching verified fix, switches the request to
the mode's escalation model.  Retrieval relevance alone is not evidence of
success, and replaying the same evidence is not a failed attempt.

The decision runs after routing, on top of it: routing chose a model from the
agent's observed state, and this step only ever *replaces* it with the model
the operator named for the mode, and only when the operator switched the whole
mechanism on.  It is a deliberate, fully annotated change, never a silent one.
"""

from dataclasses import dataclass
import math
from typing import Optional

from llm_router_plugins.utils.knowledge.agentic_knowledge.adapters.base import (
    AgentTurn,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.assembler import (
    KnowledgeContext,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.config import (
    AgenticKnowledgeConfig,
    ModePolicy,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.sse import (
    SCALE_COSINE,
    SCALE_RERANKER,
    SCALE_RRF,
)

__all__ = [
    "EscalationDecision",
    "EscalationPolicy",
    "ACTION_NONE",
    "ACTION_KEEP",
    "ACTION_ESCALATE",
    "REASON_DISABLED",
    "REASON_EXPLICIT_MODE",
    "REASON_NO_LOOP",
    "REASON_NO_ESCALATION_MODEL",
    "REASON_STRONG_RECIPE",
    "REASON_STRONG_RECIPE_IDENTITY",
    "REASON_LOOP_WITHOUT_RECIPE",
    "REASON_RECIPE_FAILED",
    "REASON_RECIPE_TRIAL_LIMIT",
]

#: No escalation consideration was made at all.
ACTION_NONE = "none"

#: The loop was recognised, but a strong recipe keeps the routed model.
ACTION_KEEP = "keep"

#: The loop was recognised and the model is switched to the mode's escalation model.
ACTION_ESCALATE = "escalate"

#: The mechanism is switched off; nothing is decided.
REASON_DISABLED = "disabled"

#: Routing was explicit, which the operator has said never to override.
REASON_EXPLICIT_MODE = "explicit_mode"

#: The failure did not repeat enough to count as a loop.
REASON_NO_LOOP = "no_loop"

#: The mode names no escalation model, so there is nothing to switch to.
REASON_NO_ESCALATION_MODEL = "no_escalation_model"

#: A strong recipe for the failure was found; the context already has the fix.
REASON_STRONG_RECIPE = "strong_recipe"

#: A recipe for exactly this failure was found and verified, and the retrieval
#: scale it was scored on cannot hold a threshold, so its identity decided.
REASON_STRONG_RECIPE_IDENTITY = "strong_recipe_identity"

#: The loop went on with no known fix; the model is replaced.
REASON_LOOP_WITHOUT_RECIPE = "loop_without_recipe"

#: The failure recurred after a matching recipe was offered in this generation.
REASON_RECIPE_FAILED = "recipe_did_not_resolve_loop"

#: Without exposure history, an advanced loop cannot receive unlimited trials.
REASON_RECIPE_TRIAL_LIMIT = "recipe_trial_limit"


@dataclass(frozen=True)
class EscalationDecision:
    """
    What the knowledge layer decided to do with the model.

    Parameters
    ----------
    action : str
        :data:`ACTION_NONE`, :data:`ACTION_KEEP` or :data:`ACTION_ESCALATE`.
    reason : str
        The branch of the policy that fired, for the request annotation.
    model_name : str or None
        The model the request should go to; ``None`` when the model is
        untouched.
    loop_count : int
        How many times the same failure was seen in the generation.
    signature_hash : str
        Identity of that failure; ``""`` when there was none.
    """

    action: str = ACTION_NONE
    reason: str = REASON_DISABLED
    model_name: Optional[str] = None
    loop_count: int = 0
    signature_hash: str = ""


class EscalationPolicy:
    """
    Decides whether a looping request gets the mode's escalation model.

    Parameters
    ----------
    config : AgenticKnowledgeConfig
        The resolved policy: the switch, the loop threshold and the score a
        recipe needs to count as strong.
    """

    def __init__(self, config: AgenticKnowledgeConfig) -> None:
        """Bind the policy."""
        self._config = config

    def decide(
        self,
        turn: AgentTurn,
        context: KnowledgeContext,
        policy: Optional[ModePolicy],
        loop_count: int,
        signature_hash: str,
        recipe_offered_at: Optional[int] = None,
    ) -> EscalationDecision:
        """
        Return the model decision for one request.

        Parameters
        ----------
        turn : AgentTurn
            The request's turn; its routing source is read from it.
        context : KnowledgeContext
            What the retrieval answered, including the best recipe score.
        policy : ModePolicy or None
            The policy of the turn's work mode, or ``None`` when the mode has
            none.
        loop_count : int
            Repetitions of the turn's most frequent failure, from the miner.
        signature_hash : str
            Identity of that failure.
        recipe_offered_at : int or None
            Failure count when a matching recipe was first injected in this
            generation.  A higher count is evidence that the loop continued,
            not proof that the agent applied the recipe.

        Returns
        -------
        EscalationDecision
            Always safe to apply: an action other than :data:`ACTION_ESCALATE`
            leaves the model alone.
        """
        base = EscalationDecision(
            loop_count=loop_count, signature_hash=signature_hash
        )
        if not self._config.escalation_available:
            return base
        if (turn.routing_source or "").lower() == "explicit":
            # The operator picked this model by hand; the knowledge layer does
            # not second-guess an explicit choice.
            return EscalationDecision(
                ACTION_NONE, REASON_EXPLICIT_MODE, None, loop_count, signature_hash
            )
        if policy is None or not policy.escalation_model_name:
            return EscalationDecision(
                ACTION_NONE,
                REASON_NO_ESCALATION_MODEL,
                None,
                loop_count,
                signature_hash,
            )
        if loop_count < self._config.loop_threshold:
            return EscalationDecision(
                ACTION_NONE, REASON_NO_LOOP, None, loop_count, signature_hash
            )
        if recipe_offered_at is not None and loop_count > recipe_offered_at:
            return EscalationDecision(
                ACTION_ESCALATE,
                REASON_RECIPE_FAILED,
                policy.escalation_model_name,
                loop_count,
                signature_hash,
            )
        recipe_reason = self.strong_recipe_reason(context, signature_hash)
        if recipe_reason and (
            recipe_offered_at is not None
            or loop_count == self._config.loop_threshold
        ):
            return EscalationDecision(
                ACTION_KEEP,
                recipe_reason,
                None,
                loop_count,
                signature_hash,
            )
        return EscalationDecision(
            ACTION_ESCALATE,
            (
                REASON_RECIPE_TRIAL_LIMIT
                if recipe_reason
                else REASON_LOOP_WITHOUT_RECIPE
            ),
            policy.escalation_model_name,
            loop_count,
            signature_hash,
        )

    def has_strong_recipe(
        self, context: KnowledgeContext, signature_hash: str
    ) -> bool:
        """Whether a recipe in *context* counts as a fix for this failure."""
        return bool(self.strong_recipe_reason(context, signature_hash))

    def strong_recipe_reason(
        self, context: KnowledgeContext, signature_hash: str
    ) -> str:
        """
        Return why a recipe in *context* is a fix, or ``""`` when none is.

        A recipe is a candidate for the fix of this failure when it carries
        this failure's signature and a verification command: identity, not
        similarity.  The command is a minimum provenance requirement, not an
        independent confirmation count; legacy recipes without it remain
        readable, but cannot suppress escalation just because retrieval ranked
        them highly.

        Whether being *retrieved* is enough is a question about the scale the
        retrieval scored on, and the engine names that scale on every hit.
        ``strong_recipe_score`` is a relevance threshold, and it means
        something on a cosine similarity (an absolute relevance) and on a
        fused rank the assembler has already remapped onto ``0…1`` (how near
        the top of its set a recipe sits).  It means nothing on a cross-encoder
        logit squashed onto ``[0, 1]``: measured against a real engine, a
        recipe carrying the very signature that was asked about scored 0.731
        with the reranker on and 1.0 with it off, so a threshold set for one
        scale silently never fires on the other — the loop is then reported as
        having no known fix while the block injected in front of the model
        holds exactly that fix.  On such a scale identity and verification
        decide, and the reason names which of the two routes was taken so
        that the annotation shows it.  A hit whose scale the engine did not
        state keeps the threshold, the conservative reading.
        """
        if context.empty or not signature_hash:
            return ""
        threshold = self._config.strong_recipe_score
        for hit in context.hits:
            if (hit.category or "") != "recipe":
                continue
            if hit.signature_hash != signature_hash:
                continue
            if not bool(hit.test_command.strip()):
                continue
            if not math.isfinite(hit.score):
                continue
            if hit.score_scale == SCALE_RERANKER:
                return REASON_STRONG_RECIPE_IDENTITY
            if hit.score >= threshold:
                return REASON_STRONG_RECIPE
        return ""
