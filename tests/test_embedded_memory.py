"""Embedded runtime lifecycle, persistence, and concurrency."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cogkura import ActivationConfig, EmbeddedMemory, Memory, ObservationInput
from cogkura.exceptions import StorageError, ValidationError
from cogkura.models import LearningFeedback, LearningOutcome, MemoryFeedback, MemoryIdentity
from cogkura.storage.sqlite_cognitive import SQLiteEntityRelationshipStore, SQLiteInhibitionStore
from cogkura.storage.sqlite_semantic import SQLiteSemanticMemoryStore


@pytest.mark.asyncio
async def test_context_manager_persists_memory_and_cognitive_state(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    now = datetime(2026, 1, 1, tzinfo=UTC)
    owner = await EmbeddedMemory.open(
        path, activation_config=ActivationConfig(retrieval_threshold=-10)
    )
    async with owner as memory:
        assert isinstance(memory, Memory)
        assert isinstance(memory._semantic_inspection_store, SQLiteSemanticMemoryStore)
        assert isinstance(memory._inhibition_store, SQLiteInhibitionStore)
        assert isinstance(memory._entity_relationship_store, SQLiteEntityRelationshipStore)
        await memory.observe(
            ObservationInput(
                tenant_id="a",
                source_namespace="src",
                source_record_id="1",
                content="The deployment service uses a database.",
                observed_at=now,
                metadata={
                    "relationships": [
                        {
                            "source_entity_id": "service",
                            "relation_type": "uses",
                            "target_entity_id": "database",
                        }
                    ]
                },
            )
        )
        await memory.process(tenant_id="a", as_of=now)
        results = await memory.recall(
            "deployment service database", tenant_id="a", as_of=now + timedelta(days=1)
        )
        assert results
        await memory.record_access(
            results, tenant_id="a", referenced_at=now + timedelta(days=1), request_id="r"
        )
        identity = MemoryIdentity(results[0].memory_kind, results[0].memory.memory_key)
        feedback = LearningFeedback(
            tenant_id="a",
            feedback_id="f",
            occurred_at=now + timedelta(days=1),
            items=(MemoryFeedback(identity=identity, outcome=LearningOutcome.HELPFUL),),
        )
        await memory.learn(feedback)
        before = await memory.recall(
            "deployment service database", tenant_id="a", as_of=now + timedelta(days=2)
        )
        states = await memory.list_learning_state(tenant_id="a", identities=[identity])
        relationships = await memory.list_entity_relationships(tenant_id="a")
    with pytest.raises(StorageError, match="closed"):
        _ = owner.memory
    with pytest.raises(StorageError, match="closed"):
        await memory.list_episodes(tenant_id="a")
    await owner.close()
    async with await EmbeddedMemory.open(
        path, activation_config=ActivationConfig(retrieval_threshold=-10)
    ) as reopened:
        assert (
            await reopened.recall(
                "deployment service database", tenant_id="a", as_of=now + timedelta(days=2)
            )
            == before
        )
        assert await reopened.list_learning_state(tenant_id="a", identities=[identity]) == states
        assert await reopened.list_entity_relationships(tenant_id="a") == relationships
        assert (await reopened.learn(feedback)).unchanged


@pytest.mark.asyncio
async def test_startup_failure_cleanup_and_reopen(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    with pytest.raises(TypeError):
        await EmbeddedMemory.open(path, unknown_option=True)
    async with await EmbeddedMemory.open(path) as memory:
        assert await memory.list_episodes(tenant_id="a") == []
    with pytest.raises(StorageError):
        await EmbeddedMemory.open(tmp_path / "missing" / "db")
    unrelated = tmp_path / "unrelated.db"
    with sqlite3.connect(unrelated) as conn:
        conn.execute("CREATE TABLE unrelated (id INTEGER)")
    with pytest.raises(StorageError, match="compatible"):
        await EmbeddedMemory.open(unrelated)


@pytest.mark.asyncio
async def test_validation_rejects_ephemeral_and_store_override(tmp_path: Path) -> None:
    for path in ("", " ", ":memory:"):
        with pytest.raises(ValidationError, match="file path"):
            await EmbeddedMemory.open(path)
    for timeout in (-1.0, float("nan"), float("inf"), True):
        with pytest.raises(ValidationError, match="busy_timeout"):
            await EmbeddedMemory.open(tmp_path / "db", busy_timeout=timeout)
    with pytest.raises(ValidationError, match="own stores"):
        await EmbeddedMemory.open(tmp_path / "db", episode_store=None)
    assert not (tmp_path / "db").exists()


@pytest.mark.asyncio
async def test_close_drains_cancelled_work_and_rejects_new_access(tmp_path: Path) -> None:
    owner = await EmbeddedMemory.open(tmp_path / "db")
    entered, release = threading.Event(), threading.Event()

    def blocked() -> None:
        entered.set()
        release.wait(2)

    task = asyncio.create_task(owner._storage._run(blocked))
    assert await asyncio.to_thread(entered.wait, 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    closing = asyncio.create_task(owner.close())
    await asyncio.sleep(0)
    with pytest.raises(StorageError, match="closed"):
        _ = owner.memory
    assert not closing.done()
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    release.set()
    await asyncio.gather(owner.close(), owner.close())


@pytest.mark.asyncio
async def test_two_owners_read_during_write_and_timeout(tmp_path: Path) -> None:
    path = tmp_path / "db"
    first = await EmbeddedMemory.open(path, busy_timeout=0.1)
    second = await EmbeddedMemory.open(path, busy_timeout=0.1)
    await first._storage._run(lambda: first._storage._conn().execute("BEGIN IMMEDIATE"))
    assert await second.memory.list_episodes(tenant_id="a") == []
    obs = ObservationInput(
        tenant_id="a",
        source_namespace="src",
        source_record_id="1",
        content="A valid and sufficiently long observation.",
        observed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    with pytest.raises(StorageError, match="locked"):
        await second.memory.observe(obs)
    await first._storage._run(lambda: first._storage._conn().rollback())
    assert await second.memory.observe(obs) is not None
    await asyncio.gather(first.close(), second.close())


@pytest.mark.asyncio
async def test_fresh_process_reopen(tmp_path: Path) -> None:
    path = tmp_path / "db"
    async with await EmbeddedMemory.open(path) as memory:
        await memory.observe(
            ObservationInput(
                tenant_id="a",
                source_namespace="src",
                source_record_id="1",
                content="Persistent memory survives a process restart.",
                observed_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
        )
        await memory.process(tenant_id="a")
        episode_id = (await memory.list_episodes(tenant_id="a"))[0].id
    program = """import asyncio, sys
from cogkura import EmbeddedMemory
async def main():
    async with await EmbeddedMemory.open(sys.argv[1]) as memory:
        print((await memory.list_episodes(tenant_id="a"))[0].id)
asyncio.run(main())
"""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        program,
        os.fspath(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    assert process.returncode == 0, stderr.decode()
    assert stdout.decode().strip() == episode_id


@pytest.mark.asyncio
async def test_upgrade_existing_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from cogkura.storage import sqlite as sqlite_module

    path = tmp_path / "old.db"
    paths = sqlite_module._migrations()
    monkeypatch.setattr(sqlite_module, "_migrations", lambda: paths[:1])
    when = datetime(2026, 1, 1, tzinfo=UTC)
    async with await EmbeddedMemory.open(path) as memory:
        await memory.observe(
            ObservationInput(
                tenant_id="a",
                source_namespace="src",
                source_record_id="1",
                content="An old database contains important observations.",
                observed_at=when,
            )
        )
    monkeypatch.setattr(sqlite_module, "_migrations", lambda: paths)
    async with await EmbeddedMemory.open(path) as memory:
        await memory.process(tenant_id="a", as_of=when)
        assert len(await memory.list_episodes(tenant_id="a")) == 1
        rows = await memory._observation_store.list(tenant_id="a")
        assert rows[0].source_record_id == "1"


@pytest.mark.asyncio
async def test_independent_processes_serialize_writes(tmp_path: Path) -> None:
    path = tmp_path / "db"
    owner = await EmbeddedMemory.open(path)
    program = """import asyncio, sys
from datetime import UTC, datetime
from cogkura import EmbeddedMemory, ObservationInput
async def main():
    async with await EmbeddedMemory.open(sys.argv[1]) as memory:
        for i in range(12):
            await memory.observe(ObservationInput(tenant_id="a", source_namespace="src",
                source_record_id=f"child-{i}", content="An observation from a different process.",
                observed_at=datetime(2026,1,1,tzinfo=UTC)))
asyncio.run(main())
"""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        program,
        os.fspath(path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    async def parent_writes() -> None:
        for i in range(12):
            await owner.memory.observe(
                ObservationInput(
                    tenant_id="a",
                    source_namespace="src",
                    source_record_id=f"parent-{i}",
                    content="An observation from the parent process.",
                    observed_at=datetime(2026, 1, 1, tzinfo=UTC),
                )
            )

    await parent_writes()
    _, stderr = await process.communicate()
    assert process.returncode == 0, stderr.decode()
    assert len(await owner.memory._observation_store.list(tenant_id="a")) == 24
    await owner.close()
    async with await EmbeddedMemory.open(path) as memory:
        assert len(await memory._observation_store.list(tenant_id="a")) == 24


@pytest.mark.asyncio
async def test_context_manager_closes_on_exception(tmp_path: Path) -> None:
    owner = await EmbeddedMemory.open(tmp_path / "db")
    with pytest.raises(RuntimeError, match="application failure"):
        async with owner:
            raise RuntimeError("application failure")
    with pytest.raises(StorageError, match="closed"):
        _ = owner.memory
