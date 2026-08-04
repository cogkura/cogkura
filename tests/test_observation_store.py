"""Unit tests for in-memory observation store."""

from datetime import UTC, datetime

import pytest

from cognema.observations.models import IngestStatus, ObservationInput
from cognema.observations.pipeline import ObservationPipeline
from cognema.observations.retention import ObservationRetentionMode, apply_retention
from cognema.storage.in_memory_observation import InMemoryObservationStore


def _observation(
    *,
    content: str = "George prefers PostgreSQL for production services.",
    source_version: str = "v1",
    is_deleted: bool = False,
) -> ObservationInput:
    return ObservationInput(
        tenant_id="company_123",
        subject_id="user_george",
        source_namespace="public.messages",
        source_record_id="msg-1",
        source_version=source_version,
        event_type="message",
        content=content,
        observed_at=datetime(2026, 8, 4, 10, 0, tzinfo=UTC),
        is_deleted=is_deleted,
    )


@pytest.mark.asyncio
async def test_create_and_unchanged() -> None:
    store = InMemoryObservationStore()
    obs = _observation()
    retained = apply_retention(obs, mode=ObservationRetentionMode.FULL)
    status = await store.ingest(obs, retained=retained)
    assert status is IngestStatus.CREATED
    status = await store.ingest(obs, retained=retained)
    assert status is IngestStatus.UNCHANGED


@pytest.mark.asyncio
async def test_update_and_delete_and_restore() -> None:
    store = InMemoryObservationStore()
    obs = _observation(source_version="v1")
    retained = apply_retention(obs, mode=ObservationRetentionMode.FULL)
    await store.ingest(obs, retained=retained)

    updated = _observation(source_version="v2", content="Updated content here.")
    retained_updated = apply_retention(updated, mode=ObservationRetentionMode.FULL)
    status = await store.ingest(updated, retained=retained_updated)
    assert status is IngestStatus.UPDATED

    deleted = _observation(source_version="v3", is_deleted=True)
    retained_deleted = apply_retention(deleted, mode=ObservationRetentionMode.FULL)
    status = await store.ingest(deleted, retained=retained_deleted)
    assert status is IngestStatus.DELETED

    restored = _observation(source_version="v4", content="Restored content here.")
    retained_restored = apply_retention(restored, mode=ObservationRetentionMode.FULL)
    status = await store.ingest(restored, retained=retained_restored)
    assert status is IngestStatus.RESTORED


@pytest.mark.asyncio
async def test_pipeline_rejects_short_content() -> None:
    store = InMemoryObservationStore()
    pipeline = ObservationPipeline(store)
    obs = _observation(content="ok")
    status = await pipeline.ingest(obs)
    assert status is None
