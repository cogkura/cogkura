"""Behaviour tests for 0.17.1 competition hardening and transient interference."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from cogkura import Memory, ObservationInput, RetrievalContext
from cogkura.algorithms.semantic import ComplementaryLearningSemanticConsolidator
from cogkura.models import CompetitionConfig, CompetitionDirection

_TENANT = "interference-tenant"
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
    cardinality: str = "many",
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
    await memory.process(tenant_id=_TENANT, as_of=_T)


@pytest.mark.asyncio
async def test_proactive_and_retroactive_interference_penalties() -> None:
    memory = _memory(competition_config=CompetitionConfig(enabled=True, apply_interference=True))
    await _setup_deployment_competitors(memory)
    inspection = await memory.inspect_recall(
        "payments-api deployment_system",
        tenant_id=_TENANT,
        limit=10,
        as_of=_T,
    )
    with_interference = [
        candidate
        for candidate in (*inspection.returned, *inspection.rejected)
        if candidate.competition
        and candidate.competition.interference is not None
        and candidate.competition.interference.total_penalty < 0.0
    ]
    assert with_interference
    directions = {
        contribution.direction
        for candidate in with_interference
        for contribution in candidate.competition.interference.contributions
    }
    assert (
        CompetitionDirection.PROACTIVE in directions
        or CompetitionDirection.RETROACTIVE in directions
    )


@pytest.mark.asyncio
async def test_asymmetric_interference_weights() -> None:
    memory = _memory(
        competition_config=CompetitionConfig(
            enabled=True,
            apply_interference=True,
            proactive_weight=0.4,
            retroactive_weight=0.1,
        )
    )
    await _setup_deployment_competitors(memory)
    inspection = await memory.inspect_recall(
        "payments-api deployment_system",
        tenant_id=_TENANT,
        limit=10,
        as_of=_T,
    )
    penalties = [
        candidate.competition.interference.total_penalty
        for candidate in (*inspection.returned, *inspection.rejected)
        if candidate.competition and candidate.competition.interference is not None
    ]
    assert penalties
    assert any(penalty < 0.0 for penalty in penalties)


@pytest.mark.asyncio
async def test_apply_interference_false_preserves_recall_behaviour() -> None:
    baseline = _memory(competition_config=CompetitionConfig(enabled=True, apply_interference=False))
    await _setup_deployment_competitors(baseline)
    query = "payments-api deployment_system"
    baseline_results = await baseline.recall(query, tenant_id=_TENANT, limit=10, as_of=_T)
    diagnostic = _memory(
        competition_config=CompetitionConfig(enabled=True, apply_interference=False)
    )
    await _setup_deployment_competitors(diagnostic)
    diagnostic_results = await diagnostic.recall(query, tenant_id=_TENANT, limit=10, as_of=_T)
    assert [
        (item.memory_kind, item.memory.memory_key, item.activation, item.score)
        for item in baseline_results
    ] == [
        (item.memory_kind, item.memory.memory_key, item.activation, item.score)
        for item in diagnostic_results
    ]


@pytest.mark.asyncio
async def test_recall_and_inspect_agree_when_interference_enabled() -> None:
    memory = _memory(competition_config=CompetitionConfig(enabled=True, apply_interference=True))
    await _setup_deployment_competitors(memory)
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
async def test_interference_diagnostics_on_returned_results() -> None:
    memory = _memory(competition_config=CompetitionConfig(enabled=True, apply_interference=True))
    await _setup_deployment_competitors(memory)
    results = await memory.recall(
        "payments-api deployment_system",
        tenant_id=_TENANT,
        limit=10,
        as_of=_T,
    )
    with_diagnostics = [
        result
        for result in results
        if result.diagnostics is not None
        and result.diagnostics.activation_before_interference is not None
    ]
    assert with_diagnostics
    assert any(result.components.interference < 0.0 for result in with_diagnostics)


@pytest.mark.asyncio
async def test_disabled_competition_has_no_interference() -> None:
    memory = _memory(competition_config=CompetitionConfig(enabled=False))
    await _setup_deployment_competitors(memory)
    results = await memory.recall(
        "payments-api deployment_system",
        tenant_id=_TENANT,
        limit=10,
        as_of=_T,
    )
    assert all(result.components.interference == 0.0 for result in results)


@pytest.mark.asyncio
async def test_historical_valid_at_isolates_future_interference() -> None:
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
async def test_context_match_reduces_interference() -> None:
    memory = _memory(competition_config=CompetitionConfig(enabled=True, apply_interference=True))
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
                        "cardinality": "many",
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
                        "cardinality": "many",
                        "polarity": "affirm",
                        "qualifiers": {},
                    }
                ],
                "domain": "project_b",
            },
        )
    )
    await memory.process(tenant_id=_TENANT, as_of=_T)
    without_context = await memory.inspect_recall(
        "How is project_a deployed?",
        tenant_id=_TENANT,
        limit=10,
        as_of=_T,
    )
    with_context = await memory.inspect_recall(
        "How is project_a deployed?",
        tenant_id=_TENANT,
        retrieval_context=RetrievalContext(domain="project_a"),
        limit=10,
        as_of=_T,
    )

    def strongest_penalty(inspection: object) -> float:
        penalties = [
            candidate.competition.interference.total_penalty
            for candidate in (*inspection.returned, *inspection.rejected)
            if candidate.competition and candidate.competition.interference is not None
        ]
        return min(penalties) if penalties else 0.0

    assert strongest_penalty(with_context) >= strongest_penalty(without_context)


@pytest.mark.asyncio
async def test_inspect_interference_rank_deltas_present() -> None:
    memory = _memory(competition_config=CompetitionConfig(enabled=True, apply_interference=True))
    await _setup_deployment_competitors(memory)
    inspection = await memory.inspect_recall(
        "payments-api deployment_system",
        tenant_id=_TENANT,
        limit=10,
        as_of=_T,
    )
    returned = [
        candidate
        for candidate in inspection.returned
        if candidate.rank_before_interference is not None
    ]
    assert returned
