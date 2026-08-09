"""Unit tests for episode stores."""

from __future__ import annotations

from datetime import UTC, datetime
from types import MappingProxyType

import pytest

from cogkura.models import EpisodeEntity, EpisodeEvidenceInput, EpisodeInput, EpisodeWriteStatus
from cogkura.storage.in_memory_episode import InMemoryEpisodeStore


def _episode(
    *,
    memory_key: str = "episode-key-1",
    fingerprint: str = "fp-1",
    subject_id: str | None = "user_1",
) -> EpisodeInput:
    return EpisodeInput(
        tenant_id="company_123",
        subject_id=subject_id,
        memory_key=memory_key,
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
        entities=(EpisodeEntity(entity_id="user_1", role="subject"),),
        metadata=MappingProxyType(
            {
                "episode": {
                    "content_fingerprint": fingerprint,
                    "encoding_version": "tulving-deterministic-v1",
                }
            }
        ),
    )


@pytest.mark.asyncio
async def test_create_update_and_unchanged() -> None:
    store = InMemoryEpisodeStore()
    episode = _episode()
    status = await store.upsert(episode)
    assert status is EpisodeWriteStatus.CREATED

    status = await store.upsert(episode)
    assert status is EpisodeWriteStatus.UNCHANGED

    updated = _episode(fingerprint="fp-2", memory_key="episode-key-1")
    status = await store.upsert(updated)
    assert status is EpisodeWriteStatus.UPDATED


@pytest.mark.asyncio
async def test_list_filters_by_tenant_and_subject() -> None:
    store = InMemoryEpisodeStore()
    await store.upsert(_episode(memory_key="k1", subject_id="user_1"))
    await store.upsert(_episode(memory_key="k2", subject_id="user_2"))

    episodes = await store.list(tenant_id="company_123", subject_id="user_1")
    assert len(episodes) == 1
    assert episodes[0].subject_id == "user_1"


@pytest.mark.asyncio
async def test_deactivate_missing() -> None:
    store = InMemoryEpisodeStore()
    await store.upsert(_episode(memory_key="k1"))
    await store.upsert(_episode(memory_key="k2"))

    deactivated = await store.deactivate_missing(
        tenant_id="company_123",
        subject_id=None,
        active_memory_keys={"k1"},
    )
    assert deactivated == 1
    active = await store.list(tenant_id="company_123")
    assert len(active) == 1
    assert active[0].memory_key == "k1"
    inactive = await store.list(tenant_id="company_123", include_inactive=True)
    assert len(inactive) == 2


@pytest.mark.asyncio
async def test_clear_removes_tenant_episodes() -> None:
    store = InMemoryEpisodeStore()
    await store.upsert(_episode())
    await store.clear(tenant_id="company_123")
    assert await store.list(tenant_id="company_123") == []
