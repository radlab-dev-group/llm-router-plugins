"""
The knowledge-or-escalate decision: loop without fix escalates, a strong recipe
keeps the model, explicit routing and disabled modes never do.

The policy reads the retrieval answer (best recipe score), the miner's loop
signal and the mode policy.  It must be deterministic and safe to apply: an
action other than escalate leaves the model alone.
"""

import sys
import pathlib
from dataclasses import replace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agentic_knowledge_support import knowledge_config

from llm_router_plugins.utils.knowledge.agentic_knowledge.adapters.base import (
    AgentTurn,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.assembler import (
    KnowledgeContext,
    KnowledgeHit,
)
from llm_router_plugins.utils.knowledge.agentic_knowledge.escalation import (
    ACTION_ESCALATE,
    ACTION_KEEP,
    ACTION_NONE,
    EscalationPolicy,
)


def turn(**overrides):
    base = dict(
        agent="codex",
        scope="webapp",
        mode="debug",
        routing_source="phase",
        model="small-model",
    )
    base.update(overrides)
    return AgentTurn(**base)


def context(recipe_score=0.0):
    hits = ()
    if recipe_score > 0:
        hits = (
            KnowledgeHit(
                category="recipe",
                document_name="recipe:1:2",
                text="x",
                score=recipe_score,
                weight=1.0,
                signature_hash="h",
                test_command="pytest tests/test_a.py",
            ),
        )
    return KnowledgeContext(
        hits=hits,
        rendered="x",
        chars=1,
        best_recipe_score=recipe_score,
        status="ok",
    )


def policy_for(config, mode="debug"):
    return config.modes.get(mode)


def decide(config, candidate=None, loop=3, score=0.0, signature="h"):
    """Run the policy for one turn/context pair, defaults where unspecified."""
    candidate = candidate if candidate is not None else turn()
    knowledge = context(recipe_score=score)
    return EscalationPolicy(config).decide(
        candidate, knowledge, policy_for(config, candidate.mode), loop, signature
    )


def test_loop_without_recipe_escalates_to_the_mode_model():
    config = knowledge_config(escalation_enabled=True)
    decision = decide(config, loop=3, score=0.0)
    assert decision.action == ACTION_ESCALATE
    assert decision.model_name == config.modes["debug"].escalation_model_name
    assert decision.loop_count == 3


def test_strong_recipe_keeps_the_routed_model():
    config = knowledge_config(escalation_enabled=True)
    decision = decide(config, loop=config.loop_threshold, score=0.9)
    assert decision.action == ACTION_KEEP
    assert decision.model_name is None


def test_recipe_below_the_threshold_still_escalates():
    config = knowledge_config(escalation_enabled=True, strong_recipe_score=0.75)
    decision = decide(config, loop=4, score=0.74)
    assert decision.action == ACTION_ESCALATE


def test_below_the_loop_threshold_does_not_escalate():
    config = knowledge_config(escalation_enabled=True, loop_threshold=3)
    decision = decide(config, loop=2, score=0.0)
    assert decision.action == ACTION_NONE
    assert decision.model_name is None


def test_explicit_routing_is_never_overridden():
    config = knowledge_config(escalation_enabled=True)
    decision = decide(config, candidate=turn(routing_source="explicit"), loop=5)
    assert decision.action == ACTION_NONE


def test_disabled_mechanism_decides_nothing():
    config = knowledge_config(escalation_enabled=False)
    decision = decide(config, loop=5)
    assert decision.action == ACTION_NONE
    assert decision.model_name is None


def test_mode_without_an_escalation_model_does_not_escalate():
    config = knowledge_config(escalation_enabled=True)
    # 'implement' has no escalation_model_name in the bundled policy.
    decision = decide(config, candidate=turn(mode="implement"), loop=5)
    assert decision.action == ACTION_NONE


def test_a_mode_not_in_the_policy_never_escalates():
    config = knowledge_config(escalation_enabled=True)
    decision = decide(config, candidate=turn(mode="unknown"), loop=5)
    assert decision.action == ACTION_NONE


@pytest.mark.parametrize(
    "changes",
    [
        {"signature_hash": "other"},
        {"signature_hash": ""},
        {"test_command": ""},
        {"category": "doc"},
    ],
)
def test_similarity_without_matching_verification_does_not_keep_model(changes):
    config = knowledge_config(escalation_enabled=True)
    knowledge = context(0.95)
    knowledge = replace(knowledge, hits=(replace(knowledge.hits[0], **changes),))
    decision = EscalationPolicy(config).decide(
        turn(), knowledge, policy_for(config), config.loop_threshold, "h"
    )
    assert decision.action == ACTION_ESCALATE


def test_new_failure_after_recipe_exposure_escalates():
    config = knowledge_config(escalation_enabled=True)
    decision = EscalationPolicy(config).decide(
        turn(), context(0.95), policy_for(config), 4, "h", recipe_offered_at=3
    )
    assert decision.action == ACTION_ESCALATE
    assert decision.reason == "recipe_did_not_resolve_loop"


def test_replayed_evidence_does_not_exhaust_recipe_trial():
    config = knowledge_config(escalation_enabled=True)
    decision = EscalationPolicy(config).decide(
        turn(), context(0.95), policy_for(config), 3, "h", recipe_offered_at=3
    )
    assert decision.action == ACTION_KEEP


def test_missing_cache_cannot_give_recipe_unlimited_trials():
    config = knowledge_config(escalation_enabled=True)
    decision = decide(config, loop=config.loop_threshold + 1, score=0.95)
    assert decision.action == ACTION_ESCALATE


def test_a_recipe_not_injected_cannot_hold_back_escalation():
    config = knowledge_config(escalation_enabled=True)
    decision = EscalationPolicy(config).decide(
        turn(), replace(context(0.95), rendered=""), policy_for(config), 3, "h"
    )
    assert decision.action == ACTION_ESCALATE


def test_recipe_on_the_reranker_scale_counts_by_its_identity():
    # The reranker's squashed logit tops out below a cosine-tuned threshold
    # (measured 0.731 against a real engine for a matching recipe), so the
    # threshold cannot be the deciding number there; identity decides, and
    # the reason names it.
    config = knowledge_config(escalation_enabled=True, rerank_results=True)
    hits = (
        KnowledgeHit(
            category="recipe",
            document_name="recipe:1:2",
            text="x",
            score=0.73,
            weight=1.0,
            signature_hash="h",
            test_command="pytest tests/test_a.py",
            score_scale="reranker",
        ),
    )
    knowledge = KnowledgeContext(
        hits=hits, rendered="x", chars=1, best_recipe_score=0.73, status="ok"
    )
    decision = EscalationPolicy(config).decide(
        turn(), knowledge, policy_for(config, "debug"), 3, "h"
    )
    assert decision.action == ACTION_KEEP
    assert decision.reason == "strong_recipe_identity"
    assert EscalationPolicy(config).has_strong_recipe(knowledge, "h")


def test_a_reranker_scale_recipe_without_a_matching_signature_still_escalates():
    config = knowledge_config(escalation_enabled=True, rerank_results=True)
    hits = (
        KnowledgeHit(
            category="recipe",
            document_name="recipe:1:2",
            text="x",
            score=0.99,
            weight=1.0,
            signature_hash="other",
            test_command="pytest tests/test_a.py",
            score_scale="reranker",
        ),
    )
    knowledge = KnowledgeContext(
        hits=hits, rendered="x", chars=1, best_recipe_score=0.99, status="ok"
    )
    decision = EscalationPolicy(config).decide(
        turn(), knowledge, policy_for(config, "debug"), 3, "h"
    )
    assert decision.action == ACTION_ESCALATE
    assert decision.reason == "loop_without_recipe"


def test_a_reranker_recipe_without_verification_does_not_keep():
    config = knowledge_config(escalation_enabled=True, rerank_results=True)
    hits = (
        KnowledgeHit(
            category="recipe",
            document_name="recipe:1:2",
            text="x",
            score=0.99,
            weight=1.0,
            signature_hash="h",
            test_command="",
            score_scale="reranker",
        ),
    )
    knowledge = KnowledgeContext(
        hits=hits, rendered="x", chars=1, best_recipe_score=0.99, status="ok"
    )
    decision = EscalationPolicy(config).decide(
        turn(), knowledge, policy_for(config, "debug"), 3, "h"
    )
    assert decision.action == ACTION_ESCALATE
