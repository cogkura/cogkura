"""Behaviour tests for 0.16.5 context hardening and diagnostic semantics."""

from __future__ import annotations

from datetime import timedelta

import pytest
from context_test_helpers import (
    QUERY,
    T0,
    TENANT,
    build_store,
    diagnostic_observations,
    episode_candidates,
    reference,
    semantic,
    support_episode,
)

from cogkura import RetrievalContext
from cogkura.algorithms.activation import (
    ACTRDeclarativeActivator,
    activation_candidate_from_semantic,
    build_episode_support_index,
)
from cogkura.models import (
    ActivationConfig,
    ContextObservabilityReason,
    MemoryContextSignature,
    RetrievalContextState,
    RetrievalCue,
    SemanticDerivationInput,
    SemanticDerivationRelation,
    SemanticSupportContextReason,
)

_CONFLICT_CONTEXT = RetrievalContext(domain="warehouse-api")


@pytest.mark.asyncio
async def test_domain_mismatch_produces_context_conflict() -> None:
    memory = await build_store()
    inspection = await memory.inspect_recall(
        QUERY,
        tenant_id=TENANT,
        retrieval_context=_CONFLICT_CONTEXT,
        limit=10,
    )
    assert inspection.context is not None
    assert inspection.context.state is RetrievalContextState.CONTEXT_CONFLICT
    assert ContextObservabilityReason.NO_CONTEXTUAL_MATCH in inspection.context.reasons
    assert inspection.context.matching_candidate_count == 0
    assert inspection.context.comparable_candidate_count > 0


@pytest.mark.asyncio
async def test_positive_match_is_not_context_conflict() -> None:
    memory = await build_store()
    inspection = await memory.inspect_recall(
        QUERY,
        tenant_id=TENANT,
        retrieval_context=RetrievalContext(domain="payments-api"),
        limit=10,
    )
    assert inspection.context is not None
    assert inspection.context.state is not RetrievalContextState.CONTEXT_CONFLICT
    assert inspection.context.matching_candidate_count > 0


@pytest.mark.asyncio
async def test_mismatch_and_unavailable_candidates_still_conflict() -> None:
    memory = await build_store()
    inspection = await memory.inspect_recall(
        QUERY,
        tenant_id=TENANT,
        retrieval_context=_CONFLICT_CONTEXT,
        limit=10,
    )
    episodes = episode_candidates(inspection)
    comparable = [
        candidate
        for candidate in episodes
        if candidate.diagnostics
        and candidate.diagnostics.context_match is not None
        and candidate.diagnostics.context_match.score is not None
    ]
    assert comparable
    assert all(
        candidate.diagnostics.context_reinstatement.strength == 0.0  # type: ignore[union-attr]
        for candidate in comparable
        if candidate.diagnostics and candidate.diagnostics.context_reinstatement
    )
    assert inspection.context is not None
    assert inspection.context.state is RetrievalContextState.CONTEXT_CONFLICT


@pytest.mark.asyncio
async def test_default_encoder_leaves_concept_ids_empty() -> None:
    memory = await build_store(diagnostic_observations())
    episodes = await memory.list_episodes(tenant_id=TENANT)
    assert episodes
    assert all(episode.encoding_context.concept_ids == () for episode in episodes)


def test_semantic_mixed_support_emits_partial_conflict_reason() -> None:
    semantic_memory = semantic(
        derivations=tuple(
            SemanticDerivationInput(
                episode_id=episode_id,
                relation=SemanticDerivationRelation.SUPPORTS,
                contribution_score=0.9,
            )
            for episode_id in ("ep-1", "ep-2", "ep-3")
        )
    )
    support_episodes = [
        support_episode(
            episode_id="ep-1",
            encoding_context=MemoryContextSignature(domains=("payments-api",)),
        ),
        support_episode(
            episode_id="ep-2",
            encoding_context=MemoryContextSignature(domains=("analytics-api",)),
        ),
        support_episode(
            episode_id="ep-3",
            encoding_context=MemoryContextSignature(domains=("payments-api",)),
        ),
    ]
    activator = ACTRDeclarativeActivator()
    cue = RetrievalCue(
        text="redis",
        entity_ids=("redis",),
        retrieval_context=RetrievalContext(domain="payments-api"),
    )
    candidate = activation_candidate_from_semantic(semantic_memory)
    config = ActivationConfig(
        retrieval_threshold=-10.0,
        semantic_context_reinstatement_weight=0.25,
        enable_spreading_activation=False,
        enable_duplicate_collapse=False,
        enable_text_entity_seeding=False,
        enable_semantic_slot_admission=False,
    )
    results = activator.rank(
        candidates=[candidate],
        cue=cue,
        references={candidate.identity: (reference(),)},
        as_of=T0 + timedelta(seconds=601),
        config=config,
        limit=10,
        episode_support_index=build_episode_support_index([semantic_memory]),
        episode_by_id={episode.id: episode for episode in support_episodes},
    )
    diagnostics = results[0].diagnostics
    assert diagnostics is not None
    assert diagnostics.support_context is not None
    assert diagnostics.support_context.matching_support_count == 2
    assert diagnostics.support_context.conflicting_support_count == 1
    assert diagnostics.support_context.strength == pytest.approx(2.0 / 3.0)


@pytest.mark.asyncio
async def test_no_retrieval_context_semantic_support_counts_not_evaluated() -> None:
    semantic_memory = semantic(
        derivations=(
            SemanticDerivationInput(
                episode_id="ep-1",
                relation=SemanticDerivationRelation.SUPPORTS,
                contribution_score=0.9,
            ),
        )
    )
    support_episodes = [
        support_episode(
            episode_id="ep-1",
            encoding_context=MemoryContextSignature(domains=("payments-api",)),
        )
    ]
    activator = ACTRDeclarativeActivator()
    cue = RetrievalCue(text="redis", entity_ids=("redis",))
    candidate = activation_candidate_from_semantic(semantic_memory)
    config = ActivationConfig(
        retrieval_threshold=-10.0,
        semantic_context_reinstatement_weight=0.25,
        enable_spreading_activation=False,
        enable_duplicate_collapse=False,
        enable_text_entity_seeding=False,
        enable_semantic_slot_admission=False,
    )
    results = activator.rank(
        candidates=[candidate],
        cue=cue,
        references={candidate.identity: (reference(),)},
        as_of=T0 + timedelta(seconds=601),
        config=config,
        limit=10,
        episode_support_index=build_episode_support_index([semantic_memory]),
        episode_by_id={episode.id: episode for episode in support_episodes},
    )
    diagnostics = results[0].diagnostics
    assert diagnostics is not None
    assert diagnostics.support_context is not None
    assert diagnostics.support_context.reason is SemanticSupportContextReason.NO_RETRIEVAL_CONTEXT
    assert diagnostics.support_context.unavailable_support_count == 0
    assert diagnostics.support_context.comparable_support_count == 0
