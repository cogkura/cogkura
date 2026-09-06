"""Persistence tests for observation and episode encoding context."""

from __future__ import annotations

from datetime import UTC, datetime
from types import MappingProxyType

import pytest

from cogkura import Memory, ObservationInput
from cogkura.models import EpisodeEvidenceInput, EpisodeInput, MemoryContextSignature
from cogkura.observations.encoding_context import ObservationContext
from cogkura.storage.in_memory_episode import InMemoryEpisodeStore
from cogkura.storage.in_memory_observation import InMemoryObservationStore


def _observation_input(**overrides: object) -> ObservationInput:
    defaults = {
        "tenant_id": "acme",
        "subject_id": "developer-1",
        "source_namespace": "github",
        "source_record_id": "record-1",
        "content": "Redis was discussed.",
        "observed_at": datetime(2026, 8, 4, 10, 0, tzinfo=UTC),
    }
    defaults.update(overrides)
    return ObservationInput(**defaults)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_observation_context_round_trip_in_memory() -> None:
    store = InMemoryObservationStore()
    context = ObservationContext(
        conversation_id="arch-42",
        goal="reduce-operational-complexity",
    )
    observation = _observation_input(context=context)
    memory = Memory(observation_store=store)
    await memory.observe(observation)
    stored = await store.get_by_source(
        tenant_id="acme",
        source_namespace="github",
        source_record_id="record-1",
    )
    assert stored is not None
    assert stored.context.to_canonical_dict() == context.to_canonical_dict()


@pytest.mark.asyncio
async def test_context_only_update_creates_revision() -> None:
    store = InMemoryObservationStore()
    memory = Memory(observation_store=store)
    await memory.observe(_observation_input())
    updated = _observation_input(
        context=ObservationContext(goal="reduce-operational-complexity"),
    )
    status = await memory.observe(updated)
    stored = await store.get_by_source(
        tenant_id="acme",
        source_namespace="github",
        source_record_id="record-1",
    )
    assert status == "updated"
    assert stored is not None
    assert stored.current_revision == 2
    assert stored.context.goal == "reduce-operational-complexity"


@pytest.mark.asyncio
async def test_episode_encoding_context_round_trip_in_memory() -> None:
    episode_store = InMemoryEpisodeStore()
    signature = MemoryContextSignature(
        conversation_ids=("arch-42",),
        goals=("reduce-operational-complexity",),
    )
    episode = EpisodeInput(
        tenant_id="acme",
        subject_id="developer-1",
        memory_key="episode-key",
        statement="Episode statement.",
        started_at=datetime(2026, 8, 4, 10, 0, tzinfo=UTC),
        ended_at=datetime(2026, 8, 4, 10, 30, tzinfo=UTC),
        confidence=0.9,
        importance=0.7,
        evidence=(
            EpisodeEvidenceInput(
                observation_id="obs-1",
                observation_revision=1,
                sequence_number=0,
            ),
        ),
        metadata=MappingProxyType(
            {"episode": {"content_fingerprint": "fp-1", "encoding_version": "v1"}}
        ),
        encoding_context=signature,
    )
    await episode_store.upsert(episode)
    episodes = await episode_store.list(tenant_id="acme")
    assert episodes[0].encoding_context.to_canonical_dict() == signature.to_canonical_dict()


@pytest.mark.asyncio
async def test_episode_unchanged_requires_matching_encoding_context() -> None:
    episode_store = InMemoryEpisodeStore()
    base = EpisodeInput(
        tenant_id="acme",
        subject_id="developer-1",
        memory_key="episode-key",
        statement="Episode statement.",
        started_at=datetime(2026, 8, 4, 10, 0, tzinfo=UTC),
        ended_at=datetime(2026, 8, 4, 10, 30, tzinfo=UTC),
        confidence=0.9,
        importance=0.7,
        evidence=(
            EpisodeEvidenceInput(
                observation_id="obs-1",
                observation_revision=1,
                sequence_number=0,
            ),
        ),
        metadata=MappingProxyType(
            {"episode": {"content_fingerprint": "fp-1", "encoding_version": "v1"}}
        ),
        encoding_context=MemoryContextSignature(goals=("goal-a",)),
    )
    await episode_store.upsert(base)
    unchanged = await episode_store.upsert(base)
    assert unchanged == "unchanged"
    changed = await episode_store.upsert(
        EpisodeInput(
            tenant_id=base.tenant_id,
            subject_id=base.subject_id,
            memory_key=base.memory_key,
            statement=base.statement,
            started_at=base.started_at,
            ended_at=base.ended_at,
            confidence=base.confidence,
            importance=base.importance,
            evidence=base.evidence,
            metadata=base.metadata,
            encoding_context=MemoryContextSignature(goals=("goal-b",)),
        )
    )
    assert changed == "updated"
