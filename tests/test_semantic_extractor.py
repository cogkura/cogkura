"""Unit tests for metadata semantic extraction."""

from __future__ import annotations

from datetime import UTC, datetime
from types import MappingProxyType

import pytest

from cogkura.algorithms.semantic import MetadataSemanticExtractor
from cogkura.models import (
    EpisodeEvidenceInput,
    SemanticCardinality,
    SemanticPolarity,
    StoredEpisode,
)
from cogkura.observations.models import StoredObservation


def _observation(
    *,
    obs_id: str,
    semantic_facts: object,
    subject_id: str = "customer_42",
) -> StoredObservation:
    return StoredObservation(
        id=obs_id,
        tenant_id="company_123",
        subject_id=subject_id,
        actor_id=subject_id,
        source_type="application",
        source_namespace="chat.messages",
        source_record_id=f"record-{obs_id}",
        source_version="v1",
        event_type="message",
        content="PostgreSQL is preferred.",
        content_hash=f"hash-{obs_id}",
        metadata=MappingProxyType({"semantic_facts": semantic_facts}),
        source_created_at=None,
        source_updated_at=None,
        observed_at=datetime(2026, 8, 5, 10, 0, tzinfo=UTC),
        current_revision=1,
        is_deleted=False,
    )


def _episode(*, episode_id: str, observation_id: str) -> StoredEpisode:
    now = datetime(2026, 8, 5, 10, 0, tzinfo=UTC)
    return StoredEpisode(
        id=episode_id,
        tenant_id="company_123",
        subject_id="customer_42",
        memory_key=f"key-{episode_id}",
        statement="Episode.",
        started_at=now,
        ended_at=now,
        confidence=0.9,
        importance=0.7,
        is_active=True,
        evidence=(
            EpisodeEvidenceInput(
                observation_id=observation_id,
                observation_revision=1,
                sequence_number=0,
            ),
        ),
        entities=(),
        metadata=MappingProxyType({}),
        created_at=now,
        updated_at=now,
    )


@pytest.mark.asyncio
async def test_extracts_metadata_facts() -> None:
    extractor = MetadataSemanticExtractor()
    observation = _observation(
        obs_id="obs-1",
        semantic_facts=[
            {
                "predicate": "preferred_database",
                "object_value": "postgresql",
                "object_entity_id": "postgresql",
                "cardinality": "one",
                "polarity": "affirm",
                "qualifiers": {"environment": "production"},
            }
        ],
    )
    episode = _episode(episode_id="ep-1", observation_id="obs-1")
    result = await extractor.extract([episode], observations={"obs-1": observation})
    assert len(result.candidates) == 1
    assert result.candidates[0].predicate == "preferred_database"
    assert result.candidates[0].subject_entity_id == "customer_42"
    assert result.candidates[0].polarity is SemanticPolarity.AFFIRM
    assert result.candidates[0].cardinality is SemanticCardinality.ONE


@pytest.mark.asyncio
async def test_malformed_fact_is_counted_as_failed() -> None:
    extractor = MetadataSemanticExtractor()
    observation = _observation(obs_id="obs-1", semantic_facts=[{"predicate": "only_predicate"}])
    episode = _episode(episode_id="ep-1", observation_id="obs-1")
    result = await extractor.extract([episode], observations={"obs-1": observation})
    assert result.candidates == ()
    assert result.failed == 1
