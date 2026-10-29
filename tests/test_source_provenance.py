"""Revision-aware, read-only source-provenance coverage."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cogkura import (
    Memory,
    MemoryKind,
    MemoryProvenanceRepresentation,
    ObservationInput,
    ObservationRevisionIdentity,
    SourceEvidenceRelation,
    SourceEvidenceResolutionStatus,
)
from cogkura.observations.retention import ObservationRetentionMode, apply_retention
from cogkura.storage.in_memory_observation import InMemoryObservationStore


def _observation(
    *,
    tenant_id: str = "tenant-a",
    record_id: str = "adr-0001",
    version: str = "1",
    source_ref: str | None = "docs/adr/0001.md",
    semantic_fact: bool = False,
    is_deleted: bool = False,
    observed_at: datetime | None = None,
) -> ObservationInput:
    metadata: dict[str, object] = {}
    if source_ref is not None:
        metadata["source_ref"] = source_ref
    if semantic_fact:
        metadata["semantic_facts"] = {
            "predicate": "deployment_policy",
            "object_value": "migrations run",
            "cardinality": "one",
            "polarity": "affirm",
        }
    return ObservationInput(
        tenant_id=tenant_id,
        subject_id="service",
        source_namespace="repository",
        source_record_id=record_id,
        source_version=version,
        content=f"Migrations run during deployment before traffic ({record_id}, v{version}).",
        observed_at=observed_at or datetime(2026, 10, 1, tzinfo=UTC),
        metadata=metadata,
        is_deleted=is_deleted,
    )


async def _episode_context(memory: Memory) -> object:
    await memory.encode_episodes(
        tenant_id="tenant-a",
        subject_id="service",
        as_of=datetime(2026, 10, 2, tzinfo=UTC),
    )
    return await memory.prepare_context(
        "deployment migrations",
        tenant_id="tenant-a",
        subject_id="service",
        as_of=datetime(2026, 10, 3, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_episode_provenance_uses_the_encoded_revision_after_source_update() -> None:
    memory = Memory()
    await memory.observe(_observation(version="1", source_ref="docs/adr/original.md"))
    context = await _episode_context(memory)
    rendered_before = context.render()

    await memory.observe(
        _observation(
            version="2",
            source_ref="docs/adr/revised.md",
            observed_at=datetime(2026, 10, 4, tzinfo=UTC),
        )
    )
    provenance = await memory.resolve_provenance(context)

    source = provenance.items[0].sources[0]
    assert source.observation_revision == 1
    assert source.source_version == "1"
    assert source.source_ref == "docs/adr/original.md"
    assert source.is_latest_known_revision is False
    assert source.resolution_status is SourceEvidenceResolutionStatus.RESOLVED
    assert context.render() == rendered_before


@pytest.mark.asyncio
async def test_missing_revision_is_reported_without_current_revision_substitution() -> None:
    store = InMemoryObservationStore()
    memory = Memory(observation_store=store)
    await memory.observe(_observation())
    context = await _episode_context(memory)
    observation_id = context.items[0].memory.evidence[0].observation_id
    store.revisions[observation_id].clear()

    provenance = await memory.resolve_provenance(context)
    source = provenance.items[0].sources[0]
    assert source.resolution_status is SourceEvidenceResolutionStatus.REVISION_UNAVAILABLE
    assert source.source_version is None
    assert source.source_ref is None
    assert source.observed_at is None
    assert provenance.items[0].is_complete is False


@pytest.mark.asyncio
async def test_deleted_current_source_keeps_resolvable_historical_evidence() -> None:
    memory = Memory()
    await memory.observe(_observation(version="1"))
    context = await _episode_context(memory)
    await memory.observe(
        _observation(
            version="2",
            observed_at=datetime(2026, 10, 4, tzinfo=UTC),
            is_deleted=True,
        )
    )

    source = (await memory.resolve_provenance(context)).items[0].sources[0]
    assert source.resolution_status is SourceEvidenceResolutionStatus.RESOLVED
    assert source.observation_revision == 1
    assert source.is_latest_known_revision is False


@pytest.mark.asyncio
async def test_missing_source_ref_keeps_resolved_evidence_and_tenant_scope_is_enforced() -> None:
    store = InMemoryObservationStore()
    memory = Memory(observation_store=store)
    observation = _observation(source_ref=None)
    await memory.observe(observation)
    context = await _episode_context(memory)

    provenance = await memory.resolve_provenance(context)
    source = provenance.items[0].sources[0]
    assert source.resolution_status is SourceEvidenceResolutionStatus.SOURCE_REF_MISSING
    assert source.source_namespace == "repository"
    assert source.source_record_id == "adr-0001"
    assert source.observed_at == observation.observed_at

    foreign = _observation(tenant_id="tenant-b")
    await store.ingest(
        foreign,
        retained=apply_retention(foreign, mode=ObservationRetentionMode.FULL),
    )
    foreign_stored = await store.get_by_source(
        tenant_id="tenant-b",
        source_namespace="repository",
        source_record_id="adr-0001",
    )
    assert foreign_stored is not None
    assert (
        await store.get_revisions_many(
            tenant_id="tenant-a",
            references=(ObservationRevisionIdentity(foreign_stored.id, 1),),
        )
        == {}
    )


@pytest.mark.asyncio
async def test_semantic_support_and_chunk_member_representation_are_explicit() -> None:
    memory = Memory()
    await memory.observe(
        _observation(
            record_id="adr-0001",
            semantic_fact=True,
            observed_at=datetime(2026, 10, 1, tzinfo=UTC),
        )
    )
    await memory.observe(
        _observation(
            record_id="adr-0002",
            semantic_fact=True,
            observed_at=datetime(2026, 10, 2, tzinfo=UTC),
        )
    )
    await memory.process(
        tenant_id="tenant-a",
        subject_id="service",
        as_of=datetime(2026, 10, 3, tzinfo=UTC),
    )
    context = await memory.prepare_context(
        "deployment policy",
        tenant_id="tenant-a",
        subject_id="service",
        as_of=datetime(2026, 10, 4, tzinfo=UTC),
    )

    provenance = await memory.resolve_provenance(context, max_evidence=16)
    semantic = next(
        item for item in provenance.items if item.memory_identity.memory_kind is MemoryKind.SEMANTIC
    )
    support = next(
        item
        for item in provenance.items
        if item.representation
        is MemoryProvenanceRepresentation.CHUNK_MEMBER_NOT_INDEPENDENTLY_RENDERED
    )
    assert semantic.distinct_source_count == 2
    assert {source.relation for source in semantic.sources} == {
        SourceEvidenceRelation.SEMANTIC_SUPPORT
    }
    assert support.memory_identity.memory_kind is MemoryKind.EPISODE
    assert context.render().count(support.sources[0].source_record_id or "") == 0

    truncated = await memory.resolve_provenance(context, max_evidence=1)
    assert truncated.is_truncated is True
    assert any(item.is_truncated for item in truncated.items)
