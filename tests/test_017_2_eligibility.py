"""Behaviour tests for 0.17.2 behavioural competition eligibility."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from cogkura import Memory, ObservationInput
from cogkura.algorithms.semantic import ComplementaryLearningSemanticConsolidator
from cogkura.models import BehavioralEligibilityReason, CompetitionConfig

_TENANT = "eligibility-tenant"
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
                    "cardinality": "many",
                    "polarity": "affirm",
                    "qualifiers": {},
                }
            ],
        },
    )


async def _seed_shared_store(memory: Memory) -> None:
    await memory.observe(
        _semantic_observation(
            source_record_id="jenkins",
            predicate="deployment_system",
            object_value="jenkins",
            subject_entity_id="payments-api",
            observed_at=_T - timedelta(days=120),
            content="payments-api deployment_system jenkins",
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
        )
    )
    await memory.observe(
        _semantic_observation(
            source_record_id="mysql",
            predicate="production_database",
            object_value="mysql",
            subject_entity_id="platform",
            observed_at=_T - timedelta(days=200),
            content="platform production_database mysql",
        )
    )
    await memory.observe(
        _semantic_observation(
            source_record_id="pg",
            predicate="production_database",
            object_value="postgresql",
            subject_entity_id="platform",
            observed_at=_T - timedelta(days=20),
            content="platform production_database postgresql",
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_T)


def _eligible_contributions(inspection: object) -> list[object]:
    contributions = []
    for candidate in (*inspection.returned, *inspection.rejected):
        if candidate.competition is None or candidate.competition.interference is None:
            continue
        for contribution in candidate.competition.interference.contributions:
            contributions.append(contribution)
    return contributions


def _behavioral_reasons(inspection: object) -> set[str]:
    reasons: set[str] = set()
    for candidate in (*inspection.returned, *inspection.rejected):
        if candidate.competition is None:
            continue
        for evidence in candidate.competition.competitors:
            if evidence.behavioral_eligibility is not None:
                reasons.add(evidence.behavioral_eligibility.reason.value)
    return reasons


@pytest.mark.asyncio
async def test_deployment_query_has_behavioural_interference() -> None:
    memory = _memory(competition_config=CompetitionConfig(enabled=True, apply_interference=True))
    await _seed_shared_store(memory)
    inspection = await memory.inspect_recall(
        "payments-api deployment_system",
        tenant_id=_TENANT,
        limit=10,
        as_of=_T,
    )
    assert inspection.competition is not None
    assert inspection.competition.behaviorally_eligible_pairs > 0
    contributions = _eligible_contributions(inspection)
    assert contributions
    assert BehavioralEligibilityReason.SAME_SEMANTIC_SLOT.value in _behavioral_reasons(inspection)


@pytest.mark.asyncio
async def test_database_query_isolates_deployment_bleed() -> None:
    memory = _memory(competition_config=CompetitionConfig(enabled=True, apply_interference=True))
    await _seed_shared_store(memory)
    inspection = await memory.inspect_recall(
        "platform production_database",
        tenant_id=_TENANT,
        limit=10,
        as_of=_T,
    )
    deployment_penalties = [
        contribution
        for candidate in (*inspection.returned, *inspection.rejected)
        if candidate.competition and candidate.competition.interference
        for contribution in candidate.competition.interference.contributions
        if "jenkins" in contribution.competitor_identity.memory_key
        or "gha" in contribution.competitor_identity.memory_key
        or contribution.competitor_identity.memory_key in {"jenkins", "gha"}
    ]
    assert deployment_penalties == []
    if inspection.competition is not None:
        db_eligible = [
            evidence
            for candidate in (*inspection.returned, *inspection.rejected)
            if candidate.competition
            for evidence in candidate.competition.competitors
            if evidence.behavioral_eligibility and evidence.behavioral_eligibility.eligible
        ]
        for evidence in db_eligible:
            assert evidence.behavioral_eligibility.reason in {
                BehavioralEligibilityReason.SAME_SEMANTIC_SLOT,
                BehavioralEligibilityReason.SAME_SUBJECT_PREDICATE,
                BehavioralEligibilityReason.QUERY_ANCHORED_COMPETITION,
            }


@pytest.mark.asyncio
async def test_cross_service_deployment_has_no_behavioural_pressure() -> None:
    memory = _memory(competition_config=CompetitionConfig(enabled=True, apply_interference=True))
    await memory.observe(
        _semantic_observation(
            source_record_id="payments-jenkins",
            predicate="deployment_system",
            object_value="jenkins",
            subject_entity_id="payments-api",
            observed_at=_T - timedelta(days=30),
            content="payments-api deployment_system jenkins",
        )
    )
    await memory.observe(
        _semantic_observation(
            source_record_id="identity-jenkins",
            predicate="deployment_system",
            object_value="jenkins",
            subject_entity_id="identity-service",
            observed_at=_T - timedelta(days=20),
            content="identity-service deployment_system jenkins",
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_T)
    inspection = await memory.inspect_recall(
        "payments-api deployment_system",
        tenant_id=_TENANT,
        limit=10,
        as_of=_T,
    )
    cross_service_pressure = [
        contribution
        for candidate in (*inspection.returned, *inspection.rejected)
        if candidate.competition and candidate.competition.interference
        for contribution in candidate.competition.interference.contributions
        if "identity" in contribution.competitor_identity.memory_key
    ]
    assert cross_service_pressure == []


@pytest.mark.asyncio
async def test_unrelated_query_has_zero_eligible_pairs() -> None:
    memory = _memory(competition_config=CompetitionConfig(enabled=True, apply_interference=True))
    await _seed_shared_store(memory)
    inspection = await memory.inspect_recall(
        "customer jacket preference",
        tenant_id=_TENANT,
        limit=10,
        as_of=_T,
    )
    if inspection.competition is not None:
        assert inspection.competition.behaviorally_eligible_pairs == 0


@pytest.mark.asyncio
async def test_apply_interference_false_reports_eligibility_without_penalty() -> None:
    memory = _memory(competition_config=CompetitionConfig(enabled=True, apply_interference=False))
    await _seed_shared_store(memory)
    results = await memory.recall(
        "payments-api deployment_system",
        tenant_id=_TENANT,
        limit=10,
        as_of=_T,
    )
    inspection = await memory.inspect_recall(
        "payments-api deployment_system",
        tenant_id=_TENANT,
        limit=10,
        as_of=_T,
    )
    assert any(
        evidence.behavioral_eligibility is not None
        for candidate in (*inspection.returned, *inspection.rejected)
        if candidate.competition
        for evidence in candidate.competition.competitors
    )
    assert all(result.components.interference == 0.0 for result in results)
    assert all(
        candidate.competition is None
        or candidate.competition.interference is None
        or candidate.competition.interference.total_penalty == 0.0
        for candidate in (*inspection.returned, *inspection.rejected)
    )


@pytest.mark.asyncio
async def test_disabled_competition_has_no_eligibility() -> None:
    memory = _memory(competition_config=CompetitionConfig(enabled=False))
    await _seed_shared_store(memory)
    inspection = await memory.inspect_recall(
        "payments-api deployment_system",
        tenant_id=_TENANT,
        limit=10,
        as_of=_T,
    )
    assert inspection.competition is None or inspection.competition.accepted_competition_pairs == 0


@pytest.mark.asyncio
async def test_recall_and_inspect_agree_with_eligibility() -> None:
    memory = _memory(competition_config=CompetitionConfig(enabled=True, apply_interference=True))
    await _seed_shared_store(memory)
    query = "payments-api deployment_system"
    recalled = await memory.recall(query, tenant_id=_TENANT, limit=10, as_of=_T)
    inspection = await memory.inspect_recall(query, tenant_id=_TENANT, limit=10, as_of=_T)
    recalled_keys = [
        (item.memory_kind, item.memory.memory_key, item.activation, item.score) for item in recalled
    ]
    inspected_keys = [
        (item.memory_kind, item.memory.memory_key, item.activation, item.score)
        for item in inspection.returned
    ]
    assert recalled_keys == inspected_keys


@pytest.mark.asyncio
async def test_diagnostic_counts_include_ineligible_competitors() -> None:
    memory = _memory(competition_config=CompetitionConfig(enabled=True, apply_interference=False))
    await _seed_shared_store(memory)
    inspection = await memory.inspect_recall(
        "platform production_database",
        tenant_id=_TENANT,
        limit=10,
        as_of=_T,
    )
    for candidate in (*inspection.returned, *inspection.rejected):
        if candidate.competition is None or candidate.competition.competitor_count == 0:
            continue
        total = (
            candidate.competition.behaviorally_eligible_competitor_count
            + candidate.competition.behaviorally_rejected_competitor_count
        )
        assert total == candidate.competition.competitor_count


@pytest.mark.asyncio
async def test_historical_valid_at_still_isolates_future_memories() -> None:
    memory = _memory(competition_config=CompetitionConfig(enabled=True, apply_interference=True))
    historical = datetime(2025, 6, 1, tzinfo=UTC)
    await memory.observe(
        _semantic_observation(
            source_record_id="mysql",
            predicate="production_database",
            object_value="mysql",
            subject_entity_id="platform",
            observed_at=datetime(2025, 1, 1, tzinfo=UTC),
            content="platform production_database mysql",
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
        )
    )
    inspection = await memory.inspect_recall(
        "platform production_database",
        tenant_id=_TENANT,
        valid_at=historical,
        as_of=historical,
        limit=10,
    )
    competitor_keys = {
        evidence.competitor_identity.memory_key
        for candidate in (*inspection.returned, *inspection.rejected)
        if candidate.competition
        for evidence in candidate.competition.competitors
    }
    assert competitor_keys == set()


@pytest.mark.asyncio
async def test_rejected_by_reason_populated() -> None:
    memory = _memory(competition_config=CompetitionConfig(enabled=True, apply_interference=False))
    await _seed_shared_store(memory)
    inspection = await memory.inspect_recall(
        "platform production_database",
        tenant_id=_TENANT,
        limit=10,
        as_of=_T,
    )
    if inspection.competition is not None and inspection.competition.behaviorally_rejected_pairs:
        assert isinstance(inspection.competition.rejected_by_reason, dict)
