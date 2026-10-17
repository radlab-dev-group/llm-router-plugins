"""
Knowledge-or-escalate: the retrieval answer steers the model, not just the text.

An agent that keeps hitting the same wall is either one good answer away from
a fix or spinning in a loop that no more of the same model will break.  The
knowledge base tells the two apart: a strong recipe for exactly this failure
means the context already carries the way out, so the routed model stays; the
same loop with no known fix means the model is the limiting factor, so the
request is switched to the mode's escalation model and the agent is told what
it already tried.

The decision runs after routing, on top of it: routing chose a model from the
agent's observed state, and this step only ever *replaces* it with the model
the operator named for the mode, and only when the operator switched the whole
mechanism on.  It is a deliberate, fully annotated change, never a silent one.
"""

from dataclasses import dataclass
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
    "REASON_LOOP_WITHOUT_RECIPE",
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

#: The loop went on with no known fix; the model is replaced.
REASON_LOOP_WITHOUT_RECIPE = "loop_without_recipe"


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
        if context.best_recipe_score >= self._config.strong_recipe_score:
            return EscalationDecision(
                ACTION_KEEP,
                REASON_STRONG_RECIPE,
                None,
                loop_count,
                signature_hash,
            )
        return EscalationDecision(
            ACTION_ESCALATE,
            REASON_LOOP_WITHOUT_RECIPE,
            policy.escalation_model_name,
            loop_count,
            signature_hash,
        )
