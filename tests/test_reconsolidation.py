"""Unit tests for semantic reconsolidation."""

from __future__ import annotations

from datetime import UTC, datetime
from types import MappingProxyType

import pytest

from cogkura import Memory, ObservationInput
from cogkura.algorithms.reconsolidation import (
    DeterministicSemanticReconciler,
    classify_update_relation,
    compare_temporal_validity,
    revision_valid_at,
)
from cogkura.algorithms.semantic import (
    ComplementaryLearningSemanticConsolidator,
    generate_revision_key,
)
from cogkura.models import (
    EpisodeEvidenceInput,
    SemanticCardinality,
    SemanticFactCandidate,
    SemanticMemoryStatus,
    SemanticPolarity,
    SemanticRevisionCandidate,
    SemanticRevisionInput,
    SemanticUpdateRelation,
    StoredEpisode,
)

_T0 = datetime(2026, 1, 1, tzinfo=UTC)
_T1 = datetime(2026, 6, 1, tzinfo=UTC)
_T2 = datetime(2027, 1, 1, tzinfo=UTC)
_AS_OF = datetime(2026, 8, 5, 12, 0, tzinfo=UTC)


def _episode(*, episode_id: str) -> StoredEpisode:
    return StoredEpisode(
        id=episode_id,
        tenant_id="company_123",
        subject_id="customer_42",
        memory_key=f"key-{episode_id}",
        statement="Episode statement.",
        started_at=_T0,
        ended_at=_T0,
        confidence=0.9,
        importance=0.7,
        is_active=True,
        evidence=(
            EpisodeEvidenceInput(
                observation_id=f"obs-{episode_id}",
                observation_revision=1,
                sequence_number=0,
            ),
        ),
        entities=(),
        metadata=MappingProxyType({"episode": {"source_namespaces": ["chat.messages"]}}),
        created_at=_T0,
        updated_at=_T0,
    )


def _fact(*, episode_id: str) -> SemanticFactCandidate:
    return SemanticFactCandidate(
        tenant_id="company_123",
        source_episode_id=episode_id,
        subject_entity_id="customer_42",
        predicate="preferred_database",
        object_value="postgresql",
        object_entity_id="postgresql",
        polarity=SemanticPolarity.AFFIRM,
        cardinality=SemanticCardinality.ONE,
        confidence=0.9,
        observed_at=_T0,
        qualifiers=MappingProxyType({"environment": "production"}),
    )


def _revision_candidate(
    *,
    memory_key: str,
    object_value: str,
    object_entity_id: str,
    valid_from: datetime | None = None,
    valid_until: datetime | None = None,
    cardinality: SemanticCardinality = SemanticCardinality.ONE,
) -> SemanticRevisionCandidate:
    revision_key = generate_revision_key(
        memory_key,
        valid_from=valid_from,
        valid_until=valid_until,
    )
    return SemanticRevisionCandidate(
        tenant_id="company_123",
        memory_key=memory_key,
        slot_key="customer_42:preferred_database:one:environment=production",
        revision_key=revision_key,
        statement=f"Customer prefers {object_value}.",
        subject_id="customer_42",
        subject_entity_id="customer_42",
        predicate="preferred_database",
        object_value=object_value,
        object_entity_id=object_entity_id,
        polarity=SemanticPolarity.AFFIRM,
        cardinality=cardinality,
        qualifiers=MappingProxyType({"environment": "production"}),
        valid_from=valid_from,
        valid_until=valid_until,
        support_count=1,
        first_supported_at=_T0,
        last_supported_at=_T0,
        support_confidence=0.8,
        importance=0.7,
        derivations=(),
        observation_evidence=(),
        metadata=MappingProxyType(
            {
                "semantic": {
                    "slot_key": "customer_42:preferred_database:one:environment=production",
                    "content_fingerprint": f"fp-{memory_key}",
                    "support_mass": 1.0,
                    "source_namespaces": ["chat.messages"],
                }
            }
        ),
    )


def _reconcile(candidates, *, existing_memories=(), existing_revisions=()):
    return DeterministicSemanticReconciler().reconcile(
        candidates=candidates,
        existing_memories=existing_memories,
        existing_revisions=existing_revisions,
        as_of=_AS_OF,
    )


def test_compare_temporal_validity_half_open_boundary() -> None:
    assert compare_temporal_validity(None, _T1, _T1, None).value == "before"


def test_unknown_validity_conflicts_not_supersedes() -> None:
    left = _revision_candidate(
        memory_key="mem-a",
        object_value="postgresql",
        object_entity_id="postgresql",
    )
    right = _revision_candidate(
        memory_key="mem-b",
        object_value="mysql",
        object_entity_id="mysql",
    )
    assert (
        classify_update_relation(existing=left, incoming=right) is SemanticUpdateRelation.CONFLICTS
    )


def test_sequential_validity_supersedes() -> None:
    predecessor = _revision_candidate(
        memory_key="mem-a",
        object_value="postgresql",
        object_entity_id="postgresql",
        valid_until=_T1,
    )
    successor = _revision_candidate(
        memory_key="mem-b",
        object_value="mysql",
        object_entity_id="mysql",
        valid_from=_T1,
    )
    assert (
        classify_update_relation(existing=predecessor, incoming=successor)
        is SemanticUpdateRelation.SUPERSEDES
    )


def test_many_cardinality_coexists() -> None:
    left = _revision_candidate(
        memory_key="mem-a",
        object_value="postgresql",
        object_entity_id="postgresql",
    )
    right = _revision_candidate(
        memory_key="mem-b",
        object_value="redis",
        object_entity_id="redis",
        cardinality=SemanticCardinality.MANY,
    )
    assert (
        classify_update_relation(existing=left, incoming=right) is SemanticUpdateRelation.COEXISTS
    )


def test_reconcile_unknown_chronology_marks_contested() -> None:
    candidates = [
        _revision_candidate(
            memory_key="mem-a",
            object_value="postgresql",
            object_entity_id="postgresql",
        ),
        _revision_candidate(
            memory_key="mem-b",
            object_value="mysql",
            object_entity_id="mysql",
        ),
    ]
    plan = _reconcile(candidates)
    statuses = {memory.status for memory in plan.current_memories}
    assert SemanticMemoryStatus.CONTESTED in statuses
    assert plan.conflict_count >= 1


def test_reconcile_supersession_preserves_predecessor_revision() -> None:
    candidates = [
        _revision_candidate(
            memory_key="mem-a",
            object_value="postgresql",
            object_entity_id="postgresql",
            valid_until=_T1,
        ),
        _revision_candidate(
            memory_key="mem-b",
            object_value="mysql",
            object_entity_id="mysql",
            valid_from=_T1,
        ),
    ]
    plan = _reconcile(candidates)
    assert plan.superseded_count == 1
    predecessor = next(revision for revision in plan.revisions if revision.memory_key == "mem-a")
    assert predecessor.status is SemanticMemoryStatus.SUPERSEDED
    assert predecessor.valid_until == _T1


def test_reconcile_a_b_a_creates_three_revisions() -> None:
    candidates = [
        _revision_candidate(
            memory_key="mem-acme",
            object_value="Acme",
            object_entity_id="acme",
            valid_from=_T0,
            valid_until=_T1,
        ),
        _revision_candidate(
            memory_key="mem-beta",
            object_value="Beta",
            object_entity_id="beta",
            valid_from=_T1,
            valid_until=_T2,
        ),
        _revision_candidate(
            memory_key="mem-acme",
            object_value="Acme",
            object_entity_id="acme",
            valid_from=_T2,
            valid_until=None,
        ),
    ]
    plan = _reconcile(candidates)
    assert len(plan.revisions) == 3
    assert len([revision for revision in plan.revisions if revision.memory_key == "mem-acme"]) == 2
    preserved = next(
        revision
        for revision in plan.revisions
        if revision.memory_key == "mem-acme" and revision.valid_from == _T0
    )
    assert preserved.status is SemanticMemoryStatus.SUPERSEDED


def test_revision_valid_at_half_open() -> None:
    revision = SemanticRevisionInput(
        tenant_id="company_123",
        memory_key="mem-a",
        revision_key="rev-a",
        revision_number=1,
        status=SemanticMemoryStatus.ACTIVE,
        valid_from=_T0,
        valid_until=_T1,
        confidence=0.8,
        importance=0.7,
        support_count=1,
        contradiction_count=0,
        first_supported_at=_T0,
        last_supported_at=_T0,
        derivations=(),
    )
    assert revision_valid_at(revision, _T0)
    assert not revision_valid_at(revision, _T1)


@pytest.mark.asyncio
async def test_memory_valid_at_recall_selects_historical_revision() -> None:
    memory = Memory(
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            minimum_supporting_episodes=1,
        )
    )
    await memory.observe(
        ObservationInput(
            tenant_id="company_123",
            subject_id="customer_42",
            source_namespace="chat.messages",
            source_record_id="message_1",
            event_type="message",
            content="Preferred Acme.",
            observed_at=_T0,
            metadata={
                "conversation_id": "conv-1",
                "semantic_facts": [
                    {
                        "predicate": "preferred_vendor",
                        "object_value": "Acme",
                        "object_entity_id": "acme",
                        "cardinality": "one",
                        "polarity": "affirm",
                        "valid_from": _T0.isoformat(),
                        "valid_until": _T1.isoformat(),
                    }
                ],
            },
        )
    )
    await memory.observe(
        ObservationInput(
            tenant_id="company_123",
            subject_id="customer_42",
            source_namespace="chat.messages",
            source_record_id="message_2",
            event_type="message",
            content="Preferred Beta.",
            observed_at=_T1,
            metadata={
                "conversation_id": "conv-2",
                "semantic_facts": [
                    {
                        "predicate": "preferred_vendor",
                        "object_value": "Beta",
                        "object_entity_id": "beta",
                        "cardinality": "one",
                        "polarity": "affirm",
                        "valid_from": _T1.isoformat(),
                        "valid_until": _T2.isoformat(),
                    }
                ],
            },
        )
    )
    await memory.encode_episodes(tenant_id="company_123")
    await memory.consolidate_semantics(tenant_id="company_123")

    historical = await memory.list_semantic_memories(
        tenant_id="company_123",
        valid_at=_T0,
    )
    current = await memory.list_semantic_memories(tenant_id="company_123")
    assert len(historical) >= 1
    assert any(item.object_value.lower() == "acme" for item in historical)
    assert any(item.object_value.lower() == "beta" for item in current)


def test_consolidator_emits_revision_candidates() -> None:
    consolidator = ComplementaryLearningSemanticConsolidator(minimum_supporting_episodes=1)
    episodes = [_episode(episode_id="ep-1")]
    candidates = [_fact(episode_id="ep-1")]
    revisions = consolidator.consolidate(episodes, candidates)
    assert len(revisions) == 1
    assert revisions[0].revision_key
    assert revisions[0].support_count == 1
