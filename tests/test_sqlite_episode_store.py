"""SQLite episode persistence and evidence tests."""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from cogkura import Memory, ObservationInput
from cogkura.exceptions import StorageError
from cogkura.models import (
    EpisodeEntity,
    EpisodeEvidenceInput,
    EpisodeInput,
    EpisodeWriteStatus,
    MemoryContextSignature,
)
from cogkura.observations.retention import apply_retention
from cogkura.storage.sqlite import SQLiteObservationStore, SQLiteStorage
from cogkura.storage.sqlite_episode import SQLiteEpisodeStore


@pytest.mark.asyncio
async def test_episode_restart_and_isolation(tmp_path: Path) -> None:
    path = tmp_path / "db"
    storage = await SQLiteStorage.open(path)
    observations = SQLiteObservationStore(storage)
    obs = ObservationInput(
        tenant_id="a",
        source_namespace="src",
        source_record_id="one",
        content="A durable example observation.",
        observed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    await observations.ingest(obs, retained=apply_retention(obs))
    observation_id = (await observations.list(tenant_id="a"))[0].id
    now = datetime(2026, 1, 1, tzinfo=UTC)
    episode = EpisodeInput(
        tenant_id="a",
        subject_id="person",
        memory_key="key",
        statement="Example",
        started_at=now,
        ended_at=now,
        confidence=0.8,
        importance=0.5,
        evidence=(EpisodeEvidenceInput(observation_id, 1, 0),),
        entities=(EpisodeEntity("person", "subject"),),
        metadata={"episode": {"content_fingerprint": "one"}},
    )
    store = SQLiteEpisodeStore(storage)
    assert await store.upsert(episode, as_of=now) is EpisodeWriteStatus.CREATED
    assert await store.upsert(episode, as_of=now) is EpisodeWriteStatus.UNCHANGED
    episode = replace(episode, encoding_context=MemoryContextSignature(goals=("durability",)))
    assert await store.upsert(episode, as_of=now) is EpisodeWriteStatus.UPDATED
    original = (await store.list(tenant_id="a"))[0]
    assert original.encoding_context.goals == ("durability",)
    invalid = replace(
        episode,
        evidence=(EpisodeEvidenceInput("missing", 1, 0),),
        metadata={"episode": {"content_fingerprint": "bad"}},
    )
    with pytest.raises(StorageError):
        await store.upsert(invalid, as_of=now)
    assert (await store.list(tenant_id="a"))[0] == original
    assert await store.get_evidence_many(tenant_id="b", episode_ids={original.id}) == {}
    assert await store.get_evidence_many(tenant_id="a", episode_ids={original.id}) == {
        original.id: episode.evidence
    }
    foreign = replace(episode, tenant_id="b")
    with pytest.raises(StorageError):
        await store.upsert(foreign)
    assert not await store.list(tenant_id="b")
    await storage.close()
    storage = await SQLiteStorage.open(path)
    store = SQLiteEpisodeStore(storage)
    assert (await store.list(tenant_id="a"))[0] == original
    assert (
        await store.deactivate_missing(
            tenant_id="a", subject_id=None, active_memory_keys=set(), as_of=now
        )
        == 1
    )
    assert await store.list(tenant_id="a") == []
    assert not (await store.list(tenant_id="a", include_inactive=True))[0].is_active
    changed = replace(episode, metadata={"episode": {"content_fingerprint": "two"}})
    assert await store.upsert(changed, as_of=now) is EpisodeWriteStatus.UPDATED
    assert (await store.list(tenant_id="a"))[0].id == original.id
    await storage.close()


@pytest.mark.asyncio
async def test_memory_encode_reopen(tmp_path: Path) -> None:
    path = tmp_path / "db"
    storage = await SQLiteStorage.open(path)
    observations = SQLiteObservationStore(storage)
    episodes = SQLiteEpisodeStore(storage)
    memory = Memory(observation_store=observations, episode_store=episodes)
    item = ObservationInput(
        tenant_id="a",
        source_namespace="src",
        source_record_id="one",
        content="Alice prefers a durable cognitive memory engine.",
        observed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    await memory.observe(item)
    await memory.encode_episodes(tenant_id="a")
    before = await memory.list_episodes(tenant_id="a")
    assert before and before[0].evidence[0].observation_revision == 1
    await storage.close()
    storage = await SQLiteStorage.open(path)
    memory = Memory(
        observation_store=SQLiteObservationStore(storage), episode_store=SQLiteEpisodeStore(storage)
    )
    assert await memory.list_episodes(tenant_id="a") == before
    await storage.close()


@pytest.mark.asyncio
async def test_upgrade_preserves_observations_and_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cogkura.storage import sqlite as sqlite_module
    from cogkura.storage.sqlite import SQLiteCheckpointStore

    paths = sqlite_module._migrations()
    monkeypatch.setattr(sqlite_module, "_migrations", lambda: paths[:1])
    path = tmp_path / "db"
    storage = await SQLiteStorage.open(path)
    item = ObservationInput(
        tenant_id="a",
        source_namespace="src",
        source_record_id="one",
        content="A source record survives a migration.",
        observed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    await SQLiteObservationStore(storage).ingest(item, retained=apply_retention(item))
    before = (await SQLiteObservationStore(storage).list(tenant_id="a"))[0]
    await SQLiteCheckpointStore(storage).set(tenant_id="a", connector_id="c", checkpoint={"x": 1})
    await storage.close()
    monkeypatch.setattr(sqlite_module, "_migrations", lambda: paths)
    storage = await SQLiteStorage.open(path)
    assert (await SQLiteObservationStore(storage).list(tenant_id="a"))[0] == before
    assert await SQLiteCheckpointStore(storage).get(tenant_id="a", connector_id="c") == {"x": 1}
    assert not await SQLiteEpisodeStore(storage).list(tenant_id="a")
    await storage.close()
