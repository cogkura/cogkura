"""Behaviour and regression tests for 0.17.0 cue-competition diagnostics."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import pytest

from cogkura import Memory, ObservationInput, RetrievalContext
from cogkura.algorithms.semantic import ComplementaryLearningSemanticConsolidator
from cogkura.models import (
    CompetitionConfig,
    CompetitionDirection,
    MemoryKind,
)

_TENANT = "competition-tenant"
_SUBJECT = "operator-1"
_T = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)


def _memory(**kwargs: object) -> Memory:
    defaults = {
        "semantic_consolidator": ComplementaryLearningSemanticConsolidator(
            minimum_supporting_episodes=1,
        ),
    }
    defaults.update(kwargs)
    return Memory(**defaults)


def _semantic_observation(
    *,
    source_record_id: str,
    predicate: str,
    object_value: str,
    subject_entity_id: str,
    observed_at: datetime,
    content: str,
    cardinality: str = "one",
) -> ObservationInput:
    return ObservationInput(
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        source_namespace="facts",
        source_record_id=source_record_id,
        source_type="fact",
        content=content,
        observed_at=observed_at,
        metadata={
            "entity_ids": [subject_entity_id],
            "semantic_facts": [
                {
                    "predicate": predicate,
                    "object_value": object_value,
                    "subject_entity_id": subject_entity_id,
                    "cardinality": cardinality,
                    "polarity": "affirm",
                    "qualifiers": {},
                }
            ],
        },
    )


async def _setup_deployment_competitors(memory: Memory) -> None:
    """Two coexisting deployment memories (cardinality=many) remain in the candidate set."""
    await memory.observe(
        _semantic_observation(
            source_record_id="jenkins",
            predicate="deployment_system",
            object_value="jenkins",
            subject_entity_id="payments-api",
            observed_at=_T - timedelta(days=120),
            content="payments-api deployment_system jenkins",
            cardinality="many",
        )
    )
    await memory.observe(
        _semantic_observation(
            source_record_id="gha",
            predicate="deployment_system",
            object_value="github_actions",
            subject_entity_id="payments-api",
            observed_at=_T - timedelta(days=10),
            content="payments-api deployment_system github_actions",
            cardinality="many",
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_T)


@pytest.mark.asyncio
async def test_same_semantic_slot_reports_strong_competition() -> None:
    memory = _memory()
    await _setup_deployment_competitors(memory)
    inspection = await memory.inspect_recall(
        "payments-api deployment_system",
        tenant_id=_TENANT,
        limit=10,
        as_of=_T,
    )
    assert inspection.competition is not None
    with_competition = [
        candidate
        for candidate in (*inspection.returned, *inspection.rejected)
        if candidate.competition is not None and candidate.competition.competitor_count > 0
    ]
    assert with_competition
    sample = with_competition[0]
    evidence = sample.competition.competitors[0]
    assert evidence.same_semantic_slot
    assert evidence.strength >= 0.45


@pytest.mark.asyncio
async def test_same_subject_predicate_without_slot_still_competes() -> None:
    memory = _memory()
    await memory.observe(
        _semantic_observation(
            source_record_id="flat",
            predicate="preferred_drink",
            object_value="flat_white",
            subject_entity_id="customer_42",
            observed_at=_T - timedelta(days=30),
            content="customer_42 preferred_drink flat_white",
            cardinality="many",
        )
    )
    await memory.observe(
        _semantic_observation(
            source_record_id="black",
            predicate="preferred_drink",
            object_value="black_coffee",
            subject_entity_id="customer_42",
            observed_at=_T - timedelta(days=5),
            content="customer_42 preferred_drink black_coffee",
            cardinality="many",
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_T)
    inspection = await memory.inspect_recall(
        "What coffee does customer_42 usually drink?",
        tenant_id=_TENANT,
        limit=10,
        as_of=_T,
    )
    competitors = [
        candidate
        for candidate in (*inspection.returned, *inspection.rejected)
        if candidate.competition and candidate.competition.competitor_count > 0
    ]
    assert competitors
    assert any(
        evidence.same_predicate and evidence.same_subject
        for candidate in competitors
        for evidence in candidate.competition.competitors
    )


@pytest.mark.asyncio
async def test_entity_overlap_without_predicate_fit_is_weak() -> None:
    memory = _memory()
    await memory.observe(
        _semantic_observation(
            source_record_id="db",
            predicate="production_database",
            object_value="postgresql",
            subject_entity_id="platform",
            observed_at=_T - timedelta(days=20),
            content="platform production_database postgresql",
        )
    )
    await memory.observe(
        _semantic_observation(
            source_record_id="backup",
            predicate="backup_schedule",
            object_value="six_hours",
            subject_entity_id="postgresql",
            observed_at=_T - timedelta(days=10),
            content="postgresql backup_schedule six_hours",
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_T)
    inspection = await memory.inspect_recall(
        "What is the production database?",
        tenant_id=_TENANT,
        limit=10,
        as_of=_T,
    )
    strong_pairs = [
        evidence
        for candidate in (*inspection.returned, *inspection.rejected)
        if candidate.competition
        for evidence in candidate.competition.competitors
        if evidence.strength >= 0.45
    ]
    assert not strong_pairs


@pytest.mark.asyncio
async def test_different_subjects_do_not_compete_for_subject_specific_query() -> None:
    memory = _memory()
    await memory.observe(
        _semantic_observation(
            source_record_id="pay",
            predicate="deployment_system",
            object_value="github_actions",
            subject_entity_id="payments-api",
            observed_at=_T - timedelta(days=10),
            content="payments-api deployment_system github_actions",
        )
    )
    await memory.observe(
        _semantic_observation(
            source_record_id="id",
            predicate="deployment_system",
            object_value="jenkins",
            subject_entity_id="identity-service",
            observed_at=_T - timedelta(days=10),
            content="identity-service deployment_system jenkins",
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_T)
    inspection = await memory.inspect_recall(
        "How is payments-api deployed?",
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        limit=10,
        as_of=_T,
    )
    cross_service = [
        evidence
        for candidate in (*inspection.returned, *inspection.rejected)
        if candidate.competition
        for evidence in candidate.competition.competitors
        if evidence.competitor_identity.memory_kind is MemoryKind.SEMANTIC
        and "identity-service" in str(candidate.memory.statement + evidence.shared_entity_ids)
    ]
    assert not cross_service


@pytest.mark.asyncio
async def test_context_separation_reduces_competition() -> None:
    memory = _memory()
    await memory.observe(
        ObservationInput(
            tenant_id=_TENANT,
            subject_id=_SUBJECT,
            source_namespace="facts",
            source_record_id="a",
            source_type="fact",
            content="project_a deployment_system github_actions",
            observed_at=_T - timedelta(days=5),
            metadata={
                "entity_ids": ["project_a"],
                "semantic_facts": [
                    {
                        "predicate": "deployment_system",
                        "object_value": "github_actions",
                        "subject_entity_id": "project_a",
                        "cardinality": "one",
                        "polarity": "affirm",
                        "qualifiers": {},
                    }
                ],
                "domain": "project_a",
            },
        )
    )
    await memory.observe(
        ObservationInput(
            tenant_id=_TENANT,
            subject_id=_SUBJECT,
            source_namespace="facts",
            source_record_id="b",
            source_type="fact",
            content="project_b deployment_system jenkins",
            observed_at=_T - timedelta(days=5),
            metadata={
                "entity_ids": ["project_b"],
                "semantic_facts": [
                    {
                        "predicate": "deployment_system",
                        "object_value": "jenkins",
                        "subject_entity_id": "project_b",
                        "cardinality": "one",
                        "polarity": "affirm",
                        "qualifiers": {},
                    }
                ],
                "domain": "project_b",
            },
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_T)
    inspection = await memory.inspect_recall(
        "How is project_a deployed?",
        tenant_id=_TENANT,
        retrieval_context=RetrievalContext(domain="project_a"),
        limit=10,
        as_of=_T,
    )
    strong_cross_project = [
        evidence
        for candidate in (*inspection.returned, *inspection.rejected)
        if candidate.competition
        for evidence in candidate.competition.competitors
        if "project_b" in str(candidate.memory.statement) and evidence.strength >= 0.45
    ]
    assert not strong_cross_project


@pytest.mark.asyncio
async def test_proactive_and_retroactive_directions() -> None:
    memory = _memory()
    await _setup_deployment_competitors(memory)
    inspection = await memory.inspect_recall(
        "payments-api deployment_system",
        tenant_id=_TENANT,
        limit=10,
        as_of=_T,
    )
    directions = {
        evidence.direction
        for candidate in (*inspection.returned, *inspection.rejected)
        if candidate.competition
        for evidence in candidate.competition.competitors
    }
    assert (
        CompetitionDirection.PROACTIVE in directions
        or CompetitionDirection.RETROACTIVE in directions
    )


@pytest.mark.asyncio
async def test_historical_valid_at_excludes_future_competitors() -> None:
    memory = _memory()
    historical = datetime(2025, 6, 1, tzinfo=UTC)
    await memory.observe(
        _semantic_observation(
            source_record_id="mysql",
            predicate="production_database",
            object_value="mysql",
            subject_entity_id="platform",
            observed_at=datetime(2025, 1, 1, tzinfo=UTC),
            content="platform production_database mysql",
            cardinality="many",
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=historical)
    await memory.observe(
        _semantic_observation(
            source_record_id="pg",
            predicate="production_database",
            object_value="postgresql",
            subject_entity_id="platform",
            observed_at=datetime(2026, 1, 1, tzinfo=UTC),
            content="platform production_database postgresql",
            cardinality="many",
        )
    )
    inspection = await memory.inspect_recall(
        "platform production_database",
        tenant_id=_TENANT,
        valid_at=historical,
        as_of=historical,
        limit=10,
    )
    candidate_statements = {
        candidate.memory.statement for candidate in (*inspection.returned, *inspection.rejected)
    }
    assert all("postgresql" not in statement for statement in candidate_statements)
    competitor_statements = {
        evidence.competitor_identity.memory_key
        for candidate in (*inspection.returned, *inspection.rejected)
        if candidate.competition
        for evidence in candidate.competition.competitors
    }
    assert competitor_statements == set()


@pytest.mark.asyncio
async def test_disabled_competition_has_no_diagnostics() -> None:
    memory = _memory(competition_config=CompetitionConfig(enabled=False))
    await _setup_deployment_competitors(memory)
    inspection = await memory.inspect_recall(
        "payments-api deployment_system",
        tenant_id=_TENANT,
        limit=10,
        as_of=_T,
    )
    assert inspection.competition is None
    assert all(candidate.competition is None for candidate in inspection.returned)


@pytest.mark.asyncio
async def test_candidate_ordering_is_invariant() -> None:
    memory = _memory()
    await _setup_deployment_competitors(memory)
    query = "payments-api deployment_system"
    baseline = await memory.inspect_recall(query, tenant_id=_TENANT, limit=10, as_of=_T)
    baseline_map = {
        candidate.memory.memory_key: candidate.competition
        for candidate in (*baseline.returned, *baseline.rejected)
        if candidate.competition is not None
    }
    for seed in (1, 7, 42):
        random.seed(seed)
        shuffled = await memory.inspect_recall(query, tenant_id=_TENANT, limit=10, as_of=_T)
        shuffled_map = {
            candidate.memory.memory_key: candidate.competition
            for candidate in (*shuffled.returned, *shuffled.rejected)
            if candidate.competition is not None
        }
        assert shuffled_map == baseline_map


@pytest.mark.asyncio
async def test_bounded_competitors_respects_config() -> None:
    memory = _memory(competition_config=CompetitionConfig(max_competitors_per_candidate=1))
    for index in range(4):
        await memory.observe(
            _semantic_observation(
                source_record_id=f"drink-{index}",
                predicate="preferred_drink",
                object_value=f"drink_{index}",
                subject_entity_id="customer_42",
                observed_at=_T - timedelta(days=index + 1),
                content=f"customer_42 preferred_drink drink_{index}",
                cardinality="many",
            )
        )
    await memory.process(tenant_id=_TENANT, as_of=_T)
    inspection = await memory.inspect_recall(
        "What does customer_42 drink?",
        tenant_id=_TENANT,
        limit=20,
        as_of=_T,
    )
    assert inspection.competition is not None
    assert inspection.competition.maximum_competitors_for_candidate <= 1
    for candidate in (*inspection.returned, *inspection.rejected):
        if candidate.competition is not None:
            assert candidate.competition.competitor_count <= 1
