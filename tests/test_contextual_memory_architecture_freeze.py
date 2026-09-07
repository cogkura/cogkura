"""Consolidated architecture-freeze invariants for the 0.16 contextual-memory line."""

from __future__ import annotations

import pytest
from context_test_helpers import (
    QUERY,
    TENANT,
    StubMatchCache,
    T,
    build_store,
    context_match,
    diagnostic_observations,
    episode_candidates,
    semantic,
)
from test_context_observability import _candidate, _diagnostics, _reinstatement
from test_encoding_context_aggregation import _stored

from cogkura import ObservationInput, RetrievalContext
from cogkura.algorithms.context_matching import DeterministicContextMatcher
from cogkura.algorithms.context_observability import (
    DeterministicRetrievalContextPolicy,
    assign_context_ranks,
)
from cogkura.algorithms.context_reinstatement import DeterministicContextReinstatementPolicy
from cogkura.algorithms.episodic import DeterministicEpisodicEncoder
from cogkura.algorithms.semantic_support_context import DeterministicSemanticSupportContextPolicy
from cogkura.models import (
    ContextMatchState,
    ContextReinstatementReason,
    MemoryContextSignature,
    MemoryKind,
    RecallInspectionDisposition,
    RetrievalContextState,
    SemanticDerivationInput,
    SemanticDerivationRelation,
    SemanticSupportContextReason,
)
from cogkura.observations.encoding_context import ObservationContext

_RETRIEVAL_CONTEXT = RetrievalContext(
    conversation_id="arch-42",
    thread_id="queue-selection",
    goal="reduce-operational-complexity",
    activity="architecture-decision",
    domain="payments-api",
    temporal_context=("queue-redesign",),
)


@pytest.mark.asyncio
async def test_encoding_context_optional_and_concept_ids_reserved() -> None:
    memory = await build_store(diagnostic_observations())
    episodes = await memory.list_episodes(tenant_id=TENANT)
    assert episodes
    assert all(episode.encoding_context.concept_ids == () for episode in episodes)


def test_structured_context_not_inferred_from_text() -> None:
    encoder = DeterministicEpisodicEncoder()
    episode = encoder.encode(
        [
            _stored(
                obs_id="obs-1",
                content="Goal: reduce-operational-complexity in payments-api architecture thread.",
                metadata={"entity_ids": ("redis",)},
            )
        ]
    )[0]
    signature = episode.encoding_context
    assert signature.goals == ()
    assert signature.activities == ()
    assert signature.domains == ()
    assert signature.conversation_ids == ()


def test_matching_only_supplied_dimensions_and_positive_only_reinstatement() -> None:
    matcher = DeterministicContextMatcher()
    encoding = MemoryContextSignature(
        domains=("payments-api", "analytics-api"),
        activities=("architecture-decision",),
    )
    match = matcher.match(RetrievalContext(domain="payments-api"), encoding)
    assert match.score == pytest.approx(1.0)
    assert any(dimension.state is ContextMatchState.MATCH for dimension in match.dimensions)

    mismatch = matcher.match(RetrievalContext(domain="warehouse-api"), encoding)
    assert mismatch.score == pytest.approx(0.0)

    policy = DeterministicContextReinstatementPolicy()
    applied = policy.evaluate(
        match=match,
        weight=0.5,
        episodic=True,
        has_retrieval_context=True,
    )
    assert applied.strength == pytest.approx(1.0)
    assert applied.activation_contribution == pytest.approx(0.5)

    zero = policy.evaluate(
        match=mismatch,
        weight=0.5,
        episodic=True,
        has_retrieval_context=True,
    )
    assert zero.strength == 0.0
    assert zero.activation_contribution == 0.0
    assert zero.reason is ContextReinstatementReason.ZERO_MATCH


def test_semantic_generalization_strength_invariants() -> None:
    policy = DeterministicSemanticSupportContextPolicy()
    memory = semantic(
        derivations=tuple(
            SemanticDerivationInput(
                episode_id=f"ep-{index}",
                relation=SemanticDerivationRelation.SUPPORTS,
                contribution_score=0.9,
            )
            for index in range(1, 5)
        )
    )

    all_match = policy.evaluate(
        memory=memory,
        retrieval_context=RetrievalContext(domain="payments-api"),
        weight=0.25,
        match_cache=StubMatchCache(
            {f"ep-{index}": (context_match(score=1.0), 1.0, False) for index in range(1, 5)}
        ),
    )
    assert all_match.strength == pytest.approx(1.0)

    one_match = policy.evaluate(
        memory=memory,
        retrieval_context=RetrievalContext(domain="payments-api"),
        weight=0.25,
        match_cache=StubMatchCache(
            {
                "ep-1": (context_match(score=1.0), 1.0, False),
                "ep-2": (None, 0.0, True),
                "ep-3": (None, 0.0, True),
                "ep-4": (None, 0.0, True),
            }
        ),
    )
    assert one_match.strength == pytest.approx(0.25)

    mixed_unavailable = policy.evaluate(
        memory=memory,
        retrieval_context=RetrievalContext(domain="payments-api"),
        weight=0.25,
        match_cache=StubMatchCache(
            {
                "ep-1": (context_match(score=1.0), 1.0, False),
                "ep-2": (context_match(score=1.0), 1.0, False),
                "ep-3": (None, 0.0, True),
                "ep-4": (None, 0.0, True),
            }
        ),
    )
    assert mixed_unavailable.strength == pytest.approx(0.5)


def test_context_state_tree_including_conflict() -> None:
    policy = DeterministicRetrievalContextPolicy()
    not_provided = policy.evaluate(
        retrieval_context=None,
        candidates=(),
        underspecified_margin=0.0,
    )
    assert not_provided.state is RetrievalContextState.CONTEXT_NOT_PROVIDED

    unavailable = policy.evaluate_recall_pool(
        retrieval_context=RetrievalContext(domain="payments-api"),
        candidates=(),
        underspecified_margin=0.0,
    )
    assert unavailable.state is RetrievalContextState.CONTEXT_UNAVAILABLE


@pytest.mark.asyncio
async def test_relevance_boundary_perfect_context_does_not_admit_irrelevant_memory() -> None:
    memory = await build_store(
        [
            ObservationInput(
                tenant_id=TENANT,
                subject_id="developer-1",
                source_namespace="github",
                source_record_id="unrelated",
                source_type="discussion",
                content="Weekly lunch menu options for the cafeteria were updated.",
                observed_at=T,
                metadata={"entity_ids": ("cafeteria",)},
                context=ObservationContext(
                    conversation_id="arch-42",
                    domain="payments-api",
                    activity="architecture-decision",
                ),
            )
        ]
    )
    inspection = await memory.inspect_recall(
        QUERY,
        tenant_id=TENANT,
        retrieval_context=_RETRIEVAL_CONTEXT,
        limit=10,
    )
    unrelated = [
        candidate
        for candidate in inspection.returned
        if candidate.memory_kind is MemoryKind.EPISODE
        and "cafeteria" in candidate.memory.statement.lower()
    ]
    assert not unrelated


@pytest.mark.asyncio
async def test_observability_does_not_change_recall_or_working_memory() -> None:
    memory = await build_store()
    recall = await memory.recall(
        QUERY,
        tenant_id=TENANT,
        retrieval_context=_RETRIEVAL_CONTEXT,
    )
    inspection = await memory.inspect_recall(
        QUERY,
        tenant_id=TENANT,
        retrieval_context=_RETRIEVAL_CONTEXT,
        limit=10,
    )
    recall_rows = {
        (result.memory.memory_key, result.memory_kind): round(result.score, 6) for result in recall
    }
    returned_rows = {
        (candidate.memory.memory_key, candidate.memory_kind): round(candidate.score, 6)
        for candidate in inspection.returned
    }
    assert recall_rows.keys() == returned_rows.keys()
    for key in recall_rows:
        assert recall_rows[key] == pytest.approx(returned_rows[key], abs=1e-5)

    wm = await memory.prepare_context(
        QUERY,
        tenant_id=TENANT,
        retrieval_context=_RETRIEVAL_CONTEXT,
        prompt_budget_tokens=512,
    )
    wm_again = await memory.prepare_context(
        QUERY,
        tenant_id=TENANT,
        retrieval_context=_RETRIEVAL_CONTEXT,
        prompt_budget_tokens=512,
    )
    assert wm.render() == wm_again.render()


@pytest.mark.asyncio
async def test_inspection_contract_fields_present_on_episodic_candidate() -> None:
    memory = await build_store()
    inspection = await memory.inspect_recall(
        QUERY,
        tenant_id=TENANT,
        retrieval_context=_RETRIEVAL_CONTEXT,
        limit=10,
    )
    assert inspection.context is not None
    assert inspection.retrieval_context == _RETRIEVAL_CONTEXT
    assert inspection.context.comparable_candidate_count >= 0
    assert inspection.context.matching_candidate_count >= 0

    episodes = episode_candidates(inspection)
    assert episodes
    candidate = episodes[0]
    assert candidate.diagnostics is not None
    diagnostics = candidate.diagnostics
    assert diagnostics.context_match is not None
    assert diagnostics.context_reinstatement is not None
    assert diagnostics.activation_before_context is not None
    assert candidate.rank_before_context is not None
    assert candidate.rank_after_context is not None
    assert candidate.context_rank_delta is not None


def test_rank_delta_positive_means_improved_rank() -> None:
    reinstatement = _reinstatement(
        strength=1.0,
        activation_contribution=0.5,
        applied=True,
        reason=ContextReinstatementReason.APPLIED,
    )
    improved = _candidate(
        memory_key="candidate-a",
        disposition=RecallInspectionDisposition.RETURNED,
        activation=0.2,
        diagnostics=_diagnostics(
            activation_before_context=-1.0,
            activation=0.2,
            context_reinstatement=reinstatement,
        ),
        rank=1,
    )
    declined = _candidate(
        memory_key="candidate-b",
        disposition=RecallInspectionDisposition.RETURNED,
        activation=0.1,
        diagnostics=_diagnostics(
            activation_before_context=0.5,
            activation=0.1,
        ),
        rank=2,
    )
    ranks = assign_context_ranks((improved, declined))
    assert ranks["candidate-a"].rank_before_context == 2
    assert ranks["candidate-a"].rank_after_context == 1
    assert ranks["candidate-a"].context_rank_delta == 1


def test_semantic_support_not_evaluated_vs_unavailable() -> None:
    policy = DeterministicSemanticSupportContextPolicy()
    memory = semantic(
        derivations=(
            SemanticDerivationInput(
                episode_id="ep-1",
                relation=SemanticDerivationRelation.SUPPORTS,
                contribution_score=0.9,
            ),
        )
    )
    not_evaluated = policy.evaluate(
        memory=memory,
        retrieval_context=RetrievalContext(domain="payments-api"),
        weight=0.25,
        match_cache=None,
    )
    assert not_evaluated.reason is SemanticSupportContextReason.NOT_EVALUATED
    assert not_evaluated.unavailable_support_count == 0

    no_cue = policy.evaluate(
        memory=memory,
        retrieval_context=None,
        weight=0.25,
        match_cache=None,
    )
    assert no_cue.reason is SemanticSupportContextReason.NO_RETRIEVAL_CONTEXT
    assert no_cue.unavailable_support_count == 0

    unavailable = policy.evaluate(
        memory=memory,
        retrieval_context=RetrievalContext(domain="payments-api"),
        weight=0.25,
        match_cache=StubMatchCache({"ep-1": (None, 0.0, True)}),
    )
    assert unavailable.reason is SemanticSupportContextReason.NO_COMPARABLE_CONTEXT
    assert unavailable.unavailable_support_count == 1
