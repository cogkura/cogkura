"""SQLite storage contract, persistence and failure tests."""

import asyncio
import sqlite3
import threading
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from cogkura.exceptions import StorageError
from cogkura.observations.encoding_context import ObservationContext
from cogkura.observations.models import IngestStatus, ObservationInput, ObservationRevisionIdentity
from cogkura.observations.retention import ObservationRetentionMode, apply_retention
from cogkura.storage.in_memory_observation import InMemoryObservationStore
from cogkura.storage.sqlite import SQLiteCheckpointStore, SQLiteObservationStore, SQLiteStorage


def observation(**changes: object) -> ObservationInput:
    values: dict[str, object] = {
        "tenant_id": "a",
        "source_namespace": "source",
        "source_record_id": "1",
        "content": "original content",
        "source_version": "1",
        "observed_at": datetime(2026, 1, 1, tzinfo=UTC),
        "metadata": {"version": 1},
    }
    values.update(changes)
    return ObservationInput(**values)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_parity_and_restart(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    storage = await SQLiteStorage.open(path)
    sqlite = SQLiteObservationStore(storage)
    memory = InMemoryObservationStore()
    original = observation()
    changes = [
        original,
        original,
        replace(original, metadata={"ignored": True}),
        replace(original, context=ObservationContext(goal="test")),
        replace(original, source_version="2", content="updated", metadata={"version": 2}),
        replace(original, source_version="3", is_deleted=True),
        replace(original, source_version="4", content="restored"),
    ]
    statuses = [
        IngestStatus.CREATED,
        IngestStatus.UNCHANGED,
        IngestStatus.UNCHANGED,
        IngestStatus.UPDATED,
        IngestStatus.UPDATED,
        IngestStatus.DELETED,
        IngestStatus.RESTORED,
    ]
    for item, status in zip(changes, statuses, strict=True):
        retained = apply_retention(item)
        assert await sqlite.ingest(item, retained=retained) == status
        assert await memory.ingest(item, retained=retained) == status
    current = (await sqlite.list(tenant_id="a"))[0]
    reference = ObservationRevisionIdentity(current.id, 1)
    history = await sqlite.get_revisions_many(tenant_id="a", references=[reference, reference])
    assert history[reference].metadata == {"version": 1}
    assert history[reference].current_revision == 5
    assert not await sqlite.get_revisions_many(tenant_id="b", references=[reference])
    assert (await sqlite.get_many(tenant_id="a", observation_ids={current.id}))[0] == current
    assert await sqlite.list(tenant_id="a", subject_id="other") == []
    other = observation(tenant_id="b")
    await sqlite.ingest(other, retained=apply_retention(other))
    checkpoints = SQLiteCheckpointStore(storage)
    await checkpoints.set(tenant_id="a", connector_id="x", checkpoint={"cursor": [1, "é"]})
    await checkpoints.set(tenant_id="b", connector_id="x", checkpoint={"cursor": 9})
    await storage.close()
    await storage.close()
    storage = await SQLiteStorage.open(path)
    sqlite = SQLiteObservationStore(storage)
    assert (await sqlite.list(tenant_id="a"))[0] == current
    assert (await sqlite.get_revisions_many(tenant_id="a", references=[reference])) == history
    assert await SQLiteCheckpointStore(storage).get(tenant_id="a", connector_id="x") == {
        "cursor": [1, "é"]
    }
    await sqlite.clear(tenant_id="a")
    assert not await sqlite.get_revisions_many(tenant_id="a", references=[reference])
    assert len(await sqlite.list(tenant_id="b")) == 1
    assert await SQLiteCheckpointStore(storage).get(tenant_id="b", connector_id="x") == {
        "cursor": 9
    }
    await storage.close()


@pytest.mark.asyncio
async def test_retention_and_deleted_filter(tmp_path: Path) -> None:
    storage = await SQLiteStorage.open(tmp_path / "db")
    store = SQLiteObservationStore(storage)
    item = observation()
    retained = apply_retention(item, mode=ObservationRetentionMode.HASH_ONLY)
    await store.ingest(item, retained=retained)
    assert (await store.list(tenant_id="a"))[0].content is None
    deleted = replace(item, is_deleted=True)
    await store.ingest(deleted, retained=apply_retention(deleted))
    assert not await store.list(tenant_id="a")
    assert len(await store.list(tenant_id="a", include_deleted=True)) == 1
    await storage.close()


@pytest.mark.asyncio
async def test_revision_write_failure_rolls_back(tmp_path: Path) -> None:
    storage = await SQLiteStorage.open(tmp_path / "db")
    store = SQLiteObservationStore(storage)
    item = observation()
    await store.ingest(item, retained=apply_retention(item))
    await storage._run(
        lambda: storage._conn().execute("""CREATE TRIGGER fail_revision
        BEFORE INSERT ON observation_revisions BEGIN SELECT RAISE(ABORT, 'injected'); END""")
    )
    changed = replace(item, source_version="2")
    with pytest.raises(StorageError, match="injected"):
        await store.ingest(changed, retained=apply_retention(changed))
    assert (await store.list(tenant_id="a"))[0].current_revision == 1
    await storage.close()


@pytest.mark.asyncio
async def test_lock_timeout_and_event_loop(tmp_path: Path) -> None:
    path = tmp_path / "db"
    first = await SQLiteStorage.open(path, busy_timeout=0.1)
    second = await SQLiteStorage.open(path, busy_timeout=0.1)
    await first._run(lambda: first._conn().execute("BEGIN IMMEDIATE"))
    item = observation()
    task = asyncio.create_task(
        SQLiteObservationStore(second).ingest(item, retained=apply_retention(item))
    )
    await asyncio.sleep(0.02)
    assert not task.done()
    with pytest.raises(StorageError, match="locked"):
        await task
    await first._run(lambda: first._conn().rollback())
    assert (
        await SQLiteObservationStore(second).ingest(item, retained=apply_retention(item))
        == IngestStatus.CREATED
    )
    await first.close()
    await second.close()


@pytest.mark.asyncio
async def test_cancelled_operation_drains_on_close(tmp_path: Path) -> None:
    storage = await SQLiteStorage.open(tmp_path / "db")
    entered, release = threading.Event(), threading.Event()

    def blocked() -> None:
        entered.set()
        release.wait(2)
        storage._conn().execute("INSERT INTO connector_checkpoints VALUES ('a','x','{}')")

    task = asyncio.create_task(storage._run(blocked))
    assert await asyncio.to_thread(entered.wait, 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    closing = asyncio.create_task(storage.close())
    await asyncio.sleep(0)
    assert not closing.done()
    release.set()
    await closing
    reopened = await SQLiteStorage.open(tmp_path / "db")
    assert await SQLiteCheckpointStore(reopened).get(tenant_id="a", connector_id="x") == {}
    await reopened.close()


@pytest.mark.asyncio
async def test_incompatible_database_and_future_migration(tmp_path: Path) -> None:
    path = tmp_path / "db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE customer (id INTEGER)")
    with pytest.raises(StorageError, match="compatible"):
        await SQLiteStorage.open(path)
    path = tmp_path / "other"
    storage = await SQLiteStorage.open(path)
    await storage._run(
        lambda: storage._conn().execute("INSERT INTO schema_migrations VALUES ('999_future')")
    )
    await storage.close()
    with pytest.raises(StorageError, match="version"):
        await SQLiteStorage.open(path)


@pytest.mark.asyncio
async def test_concurrent_initialization_and_writes(tmp_path: Path) -> None:
    path = tmp_path / "db"
    first, second = await asyncio.gather(SQLiteStorage.open(path), SQLiteStorage.open(path))
    item = observation()
    statuses = await asyncio.gather(
        SQLiteObservationStore(first).ingest(item, retained=apply_retention(item)),
        SQLiteObservationStore(second).ingest(item, retained=apply_retention(item)),
    )
    assert sorted(statuses) == sorted([IngestStatus.CREATED, IngestStatus.UNCHANGED])
    await asyncio.gather(first.close(), second.close())


@pytest.mark.asyncio
async def test_failed_migration_rolls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from cogkura.storage import sqlite as sqlite_module

    path = tmp_path / "db"
    storage = await SQLiteStorage.open(path)
    await storage.close()
    failing = tmp_path / "005_fail.sql"
    failing.write_text("CREATE TABLE temporary_test (id INTEGER);\nINVALID SQL;\n")
    paths = sqlite_module._migrations()
    monkeypatch.setattr(sqlite_module, "_migrations", lambda: [*paths, failing])
    with pytest.raises(StorageError):
        await SQLiteStorage.open(path)
    with sqlite3.connect(path) as conn:
        assert conn.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall() == [
            ("001_initial",),
            ("002_episodes",),
            ("003_semantic",),
            ("004_cognitive_state",),
        ]
        assert not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='temporary_test'"
        ).fetchall()
    monkeypatch.setattr(sqlite_module, "_migrations", lambda: paths)
    storage = await SQLiteStorage.open(path)
    await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
async def test_observation_checkpoint_contract(
    backend: str, tmp_path: Path, postgres_memory_url: str | None
) -> None:
    from uuid import uuid4

    from cogkura.storage.base import CheckpointStore, ObservationStore
    from cogkura.storage.in_memory_observation import InMemoryCheckpointStore

    owner = None
    engine = None
    tenant = str(uuid4())
    if backend == "sqlite":
        owner = await SQLiteStorage.open(tmp_path / "contract.db")
        store: ObservationStore = SQLiteObservationStore(owner)
        checkpoints: CheckpointStore = SQLiteCheckpointStore(owner)
    elif backend == "postgres":
        if postgres_memory_url is None:
            pytest.skip("COGKURA_POSTGRES_MEMORY_URL is not set")
        from sqlalchemy.ext.asyncio import create_async_engine

        from cogkura.migrations import apply_migrations
        from cogkura.storage.postgres import PostgresCheckpointStore, PostgresObservationStore

        engine = create_async_engine(postgres_memory_url)
        await apply_migrations(engine)
        store = PostgresObservationStore(engine)
        checkpoints = PostgresCheckpointStore(engine)
    else:
        store = InMemoryObservationStore()
        checkpoints = InMemoryCheckpointStore()
    try:
        item = observation(tenant_id=tenant, subject_id="user")
        assert await store.ingest(item, retained=apply_retention(item)) is IngestStatus.CREATED
        assert await store.ingest(item, retained=apply_retention(item)) is IngestStatus.UNCHANGED
        changed = replace(item, source_version="2", metadata={"version": 2})
        assert (
            await store.ingest(changed, retained=apply_retention(changed)) is IngestStatus.UPDATED
        )
        current = (await store.list(tenant_id=tenant, subject_id="user"))[0]
        identity = ObservationRevisionIdentity(current.id, 1)
        revisions = await store.get_revisions_many(tenant_id=tenant, references=[identity])
        assert revisions[identity].metadata == {"version": 1}
        assert revisions[identity].current_revision == 2
        assert not await store.get_revisions_many(tenant_id="other", references=[identity])
        deleted = replace(changed, source_version="3", is_deleted=True)
        assert (
            await store.ingest(deleted, retained=apply_retention(deleted)) is IngestStatus.DELETED
        )
        assert not await store.list(tenant_id=tenant)
        assert len(await store.list(tenant_id=tenant, include_deleted=True)) == 1
        restored = replace(deleted, source_version="4", is_deleted=False)
        assert (
            await store.ingest(restored, retained=apply_retention(restored))
            is IngestStatus.RESTORED
        )
        await checkpoints.set(tenant_id=tenant, connector_id="c", checkpoint={"offset": 1})
        await checkpoints.set(tenant_id=tenant, connector_id="c", checkpoint={"offset": 2})
        assert await checkpoints.get(tenant_id=tenant, connector_id="c") == {"offset": 2}
        assert await checkpoints.get(tenant_id="other", connector_id="c") is None
    finally:
        await store.clear(tenant_id=tenant)
        if owner is not None:
            await owner.close()
        if engine is not None:
            from sqlalchemy import text

            async with engine.begin() as conn:
                await conn.execute(
                    text("DELETE FROM cogkura.connector_checkpoints WHERE tenant_id=:tenant"),
                    {"tenant": tenant},
                )
            await engine.dispose()


@pytest.mark.asyncio
async def test_memory_ingestion_replay_after_failed_batch(tmp_path: Path) -> None:
    from collections.abc import AsyncIterator
    from typing import Any

    from cogkura import Memory

    class Source:
        connector_id = "connector"

        async def records(self, checkpoint: dict[str, Any] | None) -> AsyncIterator[int]:
            start = checkpoint["offset"] if checkpoint else 0
            for number in range(start + 1, 4):
                yield number

        def checkpoint_for(self, record: int) -> dict[str, Any]:
            return {"offset": record}

    class Mapper:
        fail = True

        def map(self, record: int) -> ObservationInput:
            if self.fail and record == 2:
                raise ValueError("injected mapper failure")
            return observation(source_record_id=str(record))

    storage = await SQLiteStorage.open(tmp_path / "db")
    observations = SQLiteObservationStore(storage)
    checkpoints = SQLiteCheckpointStore(storage)
    memory = Memory(observation_store=observations, checkpoint_store=checkpoints)
    mapper = Mapper()
    result = await memory.ingest(source=Source(), mapper=mapper, tenant_id="a", batch_size=3)
    assert result.created == 1 and result.failed == 1
    assert await checkpoints.get(tenant_id="a", connector_id="connector") is None
    mapper.fail = False
    result = await memory.ingest(source=Source(), mapper=mapper, tenant_id="a", batch_size=3)
    assert result.unchanged == 1 and result.created == 2
    assert await checkpoints.get(tenant_id="a", connector_id="connector") == {"offset": 3}
    await storage.close()


@pytest.mark.asyncio
async def test_large_revision_batch_and_missing_ids(tmp_path: Path) -> None:
    storage = await SQLiteStorage.open(tmp_path / "db")
    store = SQLiteObservationStore(storage)
    item = observation()
    await store.ingest(item, retained=apply_retention(item))
    current = (await store.list(tenant_id="a"))[0]
    identity = ObservationRevisionIdentity(current.id, 1)
    missing = [ObservationRevisionIdentity(f"missing-{i}", 1) for i in range(1000)]
    assert set(await store.get_revisions_many(tenant_id="a", references=[*missing, identity])) == {
        identity
    }
    assert (
        len(
            await store.get_many(
                tenant_id="a", observation_ids={r.observation_id for r in missing} | {current.id}
            )
        )
        == 1
    )
    await storage.close()


@pytest.mark.asyncio
async def test_missing_schema_table_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "db"
    storage = await SQLiteStorage.open(path)
    await storage.close()
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE connector_checkpoints")
    with pytest.raises(StorageError, match="required tables"):
        await SQLiteStorage.open(path)
