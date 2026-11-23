"""Historical activation must not score evidence from after its cognitive clock."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cogkura import ActivationConfig, EmbeddedMemory, Memory, ObservationInput

_T0 = datetime(2026, 1, 1, tzinfo=UTC)
_T1 = _T0 + timedelta(days=1)
_T2 = _T0 + timedelta(days=2)


async def _observe(memory: Memory, record: str, text: str, when: datetime, **metadata) -> None:
    await memory.observe(
        ObservationInput(
            tenant_id="historical-candidate",
            subject_id="service",
            source_namespace="history-test",
            source_record_id=record,
            content=text,
            observed_at=when,
            metadata=metadata,
        )
    )
    await memory.encode_episodes(tenant_id="historical-candidate", as_of=when)


async def _exercise(memory: Memory) -> None:
    tenant = "historical-candidate"
    await _observe(memory, "known", "migration strategy stable", _T0)
    before = {
        valid_at: await memory.recall(
            "migration strategy", tenant_id=tenant, as_of=_T1, valid_at=valid_at, limit=20
        )
        for valid_at in (None, _T1, _T2)
    }
    assert all(before.values())
    await _observe(memory, "future", "migration strategy future", _T2)
    for valid_at in (None, _T1, _T2):
        after = await memory.recall(
            "migration strategy", tenant_id=tenant, as_of=_T1, valid_at=valid_at, limit=20
        )
        assert after == before[valid_at]
        inspection = await memory.inspect_recall(
            "migration strategy", tenant_id=tenant, as_of=_T1, valid_at=valid_at, limit=20
        )
        assert inspection.considered_count == 1
        await memory.prepare_context(
            "migration strategy", tenant_id=tenant, as_of=_T1, valid_at=valid_at
        )
    later = await memory.recall("migration strategy", tenant_id=tenant, as_of=_T2, limit=20)
    assert any("future" in result.memory.statement for result in later)


@pytest.mark.parametrize("idf", [True, False])
async def test_historical_candidate_visibility_memory_and_sqlite(tmp_path: Path, idf: bool) -> None:
    config = ActivationConfig(retrieval_threshold=-10, enable_candidate_idf=idf)
    await _exercise(Memory(activation_config=config))
    async with await EmbeddedMemory.open(
        tmp_path / f"{idf}.db", activation_config=config
    ) as memory:
        await _exercise(memory)


async def _exercise_mixed_episode(memory: Memory) -> None:
    await _observe(memory, "first", "migration known", _T0, session_id="shared")
    before = await memory.recall("migration", tenant_id="historical-candidate", as_of=_T1)
    assert len(before) == 1
    await _observe(memory, "second", "migration future secret", _T2, session_id="shared")
    results = await memory.recall("migration", tenant_id="historical-candidate", as_of=_T1)
    assert results == before
    inspection = await memory.inspect_recall(
        "migration", tenant_id="historical-candidate", as_of=_T1
    )
    assert inspection.considered_count == 1
    assert "future secret" not in repr(inspection)
    assert await memory.recall("migration", tenant_id="historical-candidate", as_of=_T2)


async def test_episode_spanning_as_of_memory_and_sqlite(tmp_path: Path) -> None:
    config = ActivationConfig(retrieval_threshold=-10)
    await _exercise_mixed_episode(Memory(activation_config=config))
    async with await EmbeddedMemory.open(
        tmp_path / "history.db", activation_config=config
    ) as memory:
        await _exercise_mixed_episode(memory)
    async with await EmbeddedMemory.open(
        tmp_path / "history.db", activation_config=config
    ) as memory:
        restored = await memory.recall("migration", tenant_id="historical-candidate", as_of=_T1)
        assert len(restored) == 1
        assert "future secret" not in restored[0].memory.statement


async def _exercise_write_between_history_check_and_read(
    reader: Memory, writer: Memory, monkeypatch: pytest.MonkeyPatch
) -> None:
    tenant = "historical-candidate"
    await _observe(writer, "first", "migration known", _T0, session_id="shared")
    expected = await reader.recall("migration", tenant_id=tenant, as_of=_T1)
    assert len(expected) == 1
    checked = asyncio.Event()
    resume = asyncio.Event()
    original = reader._check_history_order

    async def pause_after_check(*, tenant_id: str, as_of: datetime | None) -> None:
        await original(tenant_id=tenant_id, as_of=as_of)
        checked.set()
        await resume.wait()

    monkeypatch.setattr(reader, "_check_history_order", pause_after_check)
    pending = asyncio.create_task(reader.recall("migration", tenant_id=tenant, as_of=_T1))
    try:
        await asyncio.wait_for(checked.wait(), timeout=5)
        await _observe(writer, "second", "migration future secret", _T2, session_id="shared")
    finally:
        resume.set()
    result = await asyncio.wait_for(pending, timeout=5)
    assert result == expected
    assert "future secret" not in repr(result)


async def test_forward_write_between_history_check_and_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ActivationConfig(retrieval_threshold=-10)
    memory = Memory(activation_config=config)
    await _exercise_write_between_history_check_and_read(memory, memory, monkeypatch)
    path = tmp_path / "concurrent-history.db"
    async with await EmbeddedMemory.open(path, activation_config=config) as reader:
        async with await EmbeddedMemory.open(path, activation_config=config) as writer:
            await _exercise_write_between_history_check_and_read(reader, writer, monkeypatch)


async def _exercise_backdated_write_after_history_check(
    reader: Memory, writer: Memory, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cogkura import HistoricalStateUnavailableError

    tenant = "historical-candidate"
    await _observe(writer, "first", "migration known", _T0, session_id="shared")
    await _observe(writer, "later", "migration later", _T2, session_id="shared")
    checked, resume = asyncio.Event(), asyncio.Event()
    original = reader._check_history_order

    async def pause_after_check(*, tenant_id: str, as_of: datetime | None) -> None:
        await original(tenant_id=tenant_id, as_of=as_of)
        checked.set()
        await resume.wait()

    monkeypatch.setattr(reader, "_check_history_order", pause_after_check)
    pending = asyncio.create_task(reader.recall("migration", tenant_id=tenant, as_of=_T1))
    try:
        await asyncio.wait_for(checked.wait(), timeout=5)
        await _observe(
            writer,
            "backdated",
            "migration backdated",
            _T0 + timedelta(hours=12),
            session_id="shared",
        )
        with pytest.raises(HistoricalStateUnavailableError):
            await original(tenant_id=tenant, as_of=_T1)
    finally:
        resume.set()
    try:
        await asyncio.wait_for(pending, timeout=5)
    except HistoricalStateUnavailableError:
        # A stricter implementation may reject the already-started read too.
        pass
    monkeypatch.setattr(reader, "_check_history_order", original)
    with pytest.raises(HistoricalStateUnavailableError):
        await reader.recall("migration", tenant_id=tenant, as_of=_T1)


async def test_backdated_write_after_history_check_marks_later_reads_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ActivationConfig(retrieval_threshold=-10)
    memory = Memory(activation_config=config)
    await _exercise_backdated_write_after_history_check(memory, memory, monkeypatch)
    path = tmp_path / "concurrent-backdated-history.db"
    async with await EmbeddedMemory.open(path, activation_config=config) as reader:
        async with await EmbeddedMemory.open(path, activation_config=config) as writer:
            await _exercise_backdated_write_after_history_check(reader, writer, monkeypatch)


@pytest.mark.postgres
async def test_postgres_write_between_history_check_and_read(
    postgres_memory_url: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not postgres_memory_url:
        pytest.skip("COGKURA_POSTGRES_MEMORY_URL is not set")
    from sqlalchemy.ext.asyncio import create_async_engine

    from cogkura.migrations import apply_migrations
    from cogkura.storage.postgres import (
        PostgresEpisodeStore,
        PostgresObservationStore,
        PostgresSemanticMemoryStore,
    )

    engine = create_async_engine(postgres_memory_url)
    await apply_migrations(engine)

    def make_memory() -> Memory:
        return Memory(
            observation_store=PostgresObservationStore(engine),
            episode_store=PostgresEpisodeStore(engine),
            semantic_store=PostgresSemanticMemoryStore(engine),
            activation_config=ActivationConfig(retrieval_threshold=-10),
        )

    reader, writer = make_memory(), make_memory()
    try:
        await writer.clear(tenant_id="historical-candidate")
        await _exercise_write_between_history_check_and_read(reader, writer, monkeypatch)
        await writer.clear(tenant_id="historical-candidate")
        # Use a fresh reader so the previous pause wrapper is not retained.
        reader = make_memory()
        await _exercise_backdated_write_after_history_check(reader, writer, monkeypatch)
    finally:
        await writer.clear(tenant_id="historical-candidate")
        await engine.dispose()


async def _exercise_inactive_episode(memory: Memory) -> None:
    tenant = "historical-candidate"
    await _observe(memory, "removed", "migration known", _T0, session_id="inactive")
    before = await memory.recall("migration", tenant_id=tenant, as_of=_T1)
    assert len(before) == 1
    await memory.observe(
        ObservationInput(
            tenant_id=tenant,
            subject_id="service",
            source_namespace="history-test",
            source_record_id="removed",
            content="",
            observed_at=_T2,
            is_deleted=True,
        )
    )
    await memory.encode_episodes(tenant_id=tenant, as_of=_T2)
    assert not await memory.recall("migration", tenant_id=tenant, as_of=_T2)
    assert await memory.recall("migration", tenant_id=tenant, as_of=_T1) == before
    assert (
        await memory.inspect_recall("migration", tenant_id=tenant, as_of=_T1)
    ).considered_count == 1


async def test_inactive_episode_history_memory_and_sqlite(tmp_path: Path) -> None:
    config = ActivationConfig(retrieval_threshold=-10)
    await _exercise_inactive_episode(Memory(activation_config=config))
    path = tmp_path / "inactive.db"
    async with await EmbeddedMemory.open(path, activation_config=config) as memory:
        await _exercise_inactive_episode(memory)
    async with await EmbeddedMemory.open(path, activation_config=config) as memory:
        assert await memory.recall("migration", tenant_id="historical-candidate", as_of=_T1)


async def test_episode_history_fails_closed_when_writes_are_backdated() -> None:
    memory = Memory(activation_config=ActivationConfig(retrieval_threshold=-10))
    await _observe(memory, "first", "migration known", _T0, session_id="shared")
    await _observe(memory, "second", "migration future secret", _T2, session_id="shared")
    await _observe(memory, "late", "migration arrived later", _T1, session_id="shared")
    from cogkura import HistoricalStateUnavailableError

    with pytest.raises(HistoricalStateUnavailableError):
        await memory.recall("migration", tenant_id="historical-candidate", as_of=_T1)


async def test_episode_history_does_not_invent_pre_migration_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cogkura.storage import sqlite as sqlite_module

    paths = sqlite_module._migrations()
    monkeypatch.setattr(sqlite_module, "_migrations", lambda: paths[:4])
    path = tmp_path / "old.db"
    config = ActivationConfig(retrieval_threshold=-10)
    async with await EmbeddedMemory.open(path, activation_config=config) as memory:
        await _observe(memory, "first", "migration known", _T0, session_id="shared")
    monkeypatch.setattr(sqlite_module, "_migrations", lambda: paths)
    async with await EmbeddedMemory.open(path, activation_config=config) as memory:
        await _observe(memory, "second", "migration future", _T2, session_id="shared")
        assert await memory.recall("migration", tenant_id="historical-candidate", as_of=_T1) == []


async def test_episode_snapshot_failure_rolls_back_sqlite_write(tmp_path: Path) -> None:
    from cogkura.exceptions import StorageError

    config = ActivationConfig(retrieval_threshold=-10)
    owner = await EmbeddedMemory.open(tmp_path / "atomic.db", activation_config=config)
    async with owner as memory:
        await _observe(memory, "first", "migration known", _T0, session_id="shared")
        before = await memory.list_episodes(tenant_id="historical-candidate")
        await owner._storage._run(
            lambda: owner._storage._conn().execute(
                "CREATE TRIGGER fail_history BEFORE INSERT ON episode_history "
                "BEGIN SELECT RAISE(ABORT, 'injected failure'); END"
            )
        )
        with pytest.raises(StorageError):
            await _observe(memory, "second", "migration future", _T2, session_id="shared")
        assert await memory.list_episodes(tenant_id="historical-candidate") == before


@pytest.mark.postgres
async def test_episode_snapshot_failure_rolls_back_postgres_write(
    postgres_memory_url: str | None,
) -> None:
    if not postgres_memory_url:
        pytest.skip("COGKURA_POSTGRES_MEMORY_URL is not set")
    from sqlalchemy import text
    from sqlalchemy.exc import SQLAlchemyError
    from sqlalchemy.ext.asyncio import create_async_engine

    from cogkura.migrations import apply_migrations
    from cogkura.storage.postgres import (
        PostgresEpisodeStore,
        PostgresObservationStore,
        PostgresSemanticMemoryStore,
    )

    tenant = "historical-candidate"
    engine = create_async_engine(postgres_memory_url)
    await apply_migrations(engine)
    memory = Memory(
        observation_store=PostgresObservationStore(engine),
        episode_store=PostgresEpisodeStore(engine),
        semantic_store=PostgresSemanticMemoryStore(engine),
        activation_config=ActivationConfig(retrieval_threshold=-10),
    )
    try:
        await memory.clear(tenant_id=tenant)
        await _observe(memory, "first", "migration known", _T0, session_id="shared")
        before = await memory.list_episodes(tenant_id=tenant)
        order_before = await memory._episode_store.historical_order_status(tenant_id=tenant)
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "CREATE FUNCTION cogkura.test_fail_episode_history() "
                    "RETURNS trigger LANGUAGE plpgsql AS $$ "
                    "BEGIN RAISE EXCEPTION 'injected failure'; END; $$"
                )
            )
            await conn.execute(
                text(
                    "CREATE TRIGGER test_fail_episode_history BEFORE INSERT "
                    "ON cogkura.episode_history FOR EACH ROW "
                    "EXECUTE FUNCTION cogkura.test_fail_episode_history()"
                )
            )
        with pytest.raises(SQLAlchemyError):
            await _observe(memory, "second", "migration future", _T2, session_id="shared")
        assert await memory.list_episodes(tenant_id=tenant) == before
        assert await memory._episode_store.historical_order_status(tenant_id=tenant) == order_before
        # Observation ingestion is a separate committed stage, not part of the failed episode write.
        assert len(await memory._observation_store.list(tenant_id=tenant)) == 2
        async with engine.begin() as conn:
            await conn.execute(
                text("DROP TRIGGER test_fail_episode_history ON cogkura.episode_history")
            )
            await conn.execute(text("DROP FUNCTION cogkura.test_fail_episode_history()"))
        await memory.encode_episodes(tenant_id=tenant, as_of=_T2)
        assert any(
            "future" in item.statement for item in await memory.list_episodes(tenant_id=tenant)
        )
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DROP TRIGGER IF EXISTS test_fail_episode_history ON cogkura.episode_history")
            )
            await conn.execute(text("DROP FUNCTION IF EXISTS cogkura.test_fail_episode_history()"))
        await memory.clear(tenant_id=tenant)
        await engine.dispose()


async def test_episode_history_equal_times_and_tenant_scope() -> None:
    from dataclasses import replace

    from cogkura.storage.historical_snapshots import (
        encode_memory_snapshot,
        select_episode_snapshot,
    )
    from cogkura.storage.in_memory_episode import InMemoryEpisodeStore

    memory = Memory(activation_config=ActivationConfig(retrieval_threshold=-10))
    await _observe(memory, "first", "migration known", _T0, session_id="shared")
    original = (await memory.list_episodes(tenant_id="historical-candidate"))[0]
    later = replace(original, statement="migration revised")
    versions = [(_T0, encode_memory_snapshot(original)), (_T0, encode_memory_snapshot(later))]
    assert select_episode_snapshot(versions, as_of=_T0) == later
    assert select_episode_snapshot([*versions, (_T1, None)], as_of=_T1) is None
    assert select_episode_snapshot([*versions, (_T1, "corrupt")], as_of=_T1) is None

    store = memory._episode_store
    assert isinstance(store, InMemoryEpisodeStore)
    assert (
        await store.historical_episodes(tenant_id="other", episode_ids={original.id}, as_of=_T0)
        == {}
    )


async def _exercise_equal_time_episode_updates(memory: Memory) -> None:
    tenant = "historical-candidate"
    await _observe(memory, "equal-first", "migration first", _T0, session_id="equal-time")
    first = (await memory.list_episodes(tenant_id=tenant))[0]
    await _observe(memory, "equal-second", "migration second", _T0, session_id="equal-time")
    latest = (await memory.list_episodes(tenant_id=tenant))[0]
    assert latest.id == first.id
    assert latest.memory_key == first.memory_key
    assert len(latest.evidence) == 2
    snapshots = await memory._episode_store.historical_episodes(
        tenant_id=tenant, episode_ids={first.id}, as_of=_T0
    )
    assert snapshots[first.id] == latest
    assert (
        await memory._episode_store.historical_episodes(
            tenant_id="another-tenant", episode_ids={first.id}, as_of=_T0
        )
        == {}
    )


async def test_equal_time_episode_updates_memory_and_sqlite(tmp_path: Path) -> None:
    await _exercise_equal_time_episode_updates(Memory())
    async with await EmbeddedMemory.open(tmp_path / "equal-time.db") as memory:
        await _exercise_equal_time_episode_updates(memory)


async def _exercise_clear_history_tenant_scope(memory: Memory) -> None:
    tenants = ("history-clear-alpha", "history-clear-beta")
    for tenant in tenants:
        await memory.observe(
            ObservationInput(
                tenant_id=tenant,
                subject_id="service",
                source_namespace="history-clear",
                source_record_id="first",
                content="Database is PostgreSQL.",
                observed_at=_T0,
                metadata={
                    "semantic_facts": [
                        {
                            "predicate": "database",
                            "object_value": "postgresql",
                            "cardinality": "one",
                            "authority": {"basis": "application_asserted"},
                        }
                    ]
                },
            )
        )
        await memory.process(tenant_id=tenant, as_of=_T0)

    identities = {}
    for tenant in tenants:
        episodes = await memory.list_episodes(tenant_id=tenant)
        semantics = await memory.list_semantic_memories(tenant_id=tenant)
        revisions = await memory.list_semantic_revisions(tenant_id=tenant)
        assert len(episodes) == len(semantics) == len(revisions) == 1
        identities[tenant] = (episodes[0].id, semantics[0].id, revisions[0].revision_key)

    async def history(tenant: str) -> tuple[object, object, object]:
        episode_id, semantic_id, revision_key = identities[tenant]
        return (
            await memory._episode_store.historical_episodes(
                tenant_id=tenant, episode_ids={episode_id}, as_of=_T1
            ),
            await memory._semantic_store.historical_semantics(
                tenant_id=tenant, memory_ids={semantic_id}, as_of=_T1
            ),
            await memory._semantic_store.historical_revisions(
                tenant_id=tenant, revision_keys={revision_key}, as_of=_T1
            ),
        )

    assert all(await history(tenants[0]))
    beta_before = await history(tenants[1])
    assert all(beta_before)
    await memory.clear(tenant_id=tenants[0])
    assert not any(await history(tenants[0]))
    assert await history(tenants[1]) == beta_before
    assert not await memory.list_episodes(tenant_id=tenants[0])
    assert await memory.list_episodes(tenant_id=tenants[1])


async def test_clear_history_is_tenant_scoped_memory_and_sqlite(tmp_path: Path) -> None:
    from cogkura import ComplementaryLearningSemanticConsolidator

    options = {
        "semantic_consolidator": ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        )
    }
    await _exercise_clear_history_tenant_scope(Memory(**options))
    async with await EmbeddedMemory.open(tmp_path / "clear-history.db", **options) as memory:
        await _exercise_clear_history_tenant_scope(memory)


@pytest.mark.postgres
async def test_clear_history_is_tenant_scoped_postgres(postgres_memory_url: str | None) -> None:
    if not postgres_memory_url:
        pytest.skip("COGKURA_POSTGRES_MEMORY_URL is not set")
    from sqlalchemy.ext.asyncio import create_async_engine

    from cogkura import ComplementaryLearningSemanticConsolidator
    from cogkura.migrations import apply_migrations
    from cogkura.storage.postgres import (
        PostgresEpisodeStore,
        PostgresObservationStore,
        PostgresSemanticMemoryStore,
    )

    engine = create_async_engine(postgres_memory_url)
    await apply_migrations(engine)
    memory = Memory(
        observation_store=PostgresObservationStore(engine),
        episode_store=PostgresEpisodeStore(engine),
        semantic_store=PostgresSemanticMemoryStore(engine),
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
    )
    try:
        for tenant in ("history-clear-alpha", "history-clear-beta"):
            await memory.clear(tenant_id=tenant)
        await _exercise_clear_history_tenant_scope(memory)
    finally:
        for tenant in ("history-clear-alpha", "history-clear-beta"):
            await memory.clear(tenant_id=tenant)
        await engine.dispose()


@pytest.mark.postgres
@pytest.mark.parametrize("idf", [True, False])
async def test_historical_candidate_visibility_postgres(
    postgres_memory_url: str | None, idf: bool
) -> None:
    if not postgres_memory_url:
        pytest.skip("COGKURA_POSTGRES_MEMORY_URL is not set")
    from sqlalchemy.ext.asyncio import create_async_engine

    from cogkura.migrations import apply_migrations
    from cogkura.storage.postgres import (
        PostgresEpisodeStore,
        PostgresObservationStore,
        PostgresSemanticMemoryStore,
    )

    engine = create_async_engine(postgres_memory_url)
    await apply_migrations(engine)
    memory = Memory(
        observation_store=PostgresObservationStore(engine),
        episode_store=PostgresEpisodeStore(engine),
        semantic_store=PostgresSemanticMemoryStore(engine),
        activation_config=ActivationConfig(retrieval_threshold=-10, enable_candidate_idf=idf),
    )
    try:
        await memory.clear(tenant_id="historical-candidate")
        await _exercise(memory)
        await memory.clear(tenant_id="historical-candidate")
        await _exercise_mixed_episode(memory)
        await memory.clear(tenant_id="historical-candidate")
        await _exercise_inactive_episode(memory)
    finally:
        await memory.clear(tenant_id="historical-candidate")
        await engine.dispose()


async def _exercise_revision_relations(memory: Memory) -> None:
    from cogkura import SemanticSlotQuery

    tenant = "historical-candidate"
    fact = {
        "predicate": "database",
        "object_value": "postgresql",
        "cardinality": "one",
        "authority": {"basis": "application_asserted"},
    }
    await _observe(memory, "original", "Service database PostgreSQL.", _T0, semantic_facts=[fact])
    await memory.consolidate_semantics(tenant_id=tenant, as_of=_T0)
    original = (await memory.list_semantic_revisions(tenant_id=tenant))[0]
    earlier_time = _T0 + timedelta(hours=12)
    before = await memory.recall("database PostgreSQL", tenant_id=tenant, as_of=earlier_time)
    before_valid = await memory.recall(
        "database PostgreSQL", tenant_id=tenant, as_of=earlier_time, valid_at=earlier_time
    )
    assert before
    before_inspection = await memory.inspect_recall(
        "database PostgreSQL", tenant_id=tenant, as_of=earlier_time
    )
    assert before_inspection.considered_count == 2
    slot = SemanticSlotQuery(subject_entity_id="service", predicate="database")
    before_slot = await memory.inspect_semantic_slot(slot, tenant_id=tenant, as_of=earlier_time)
    await _observe(
        memory, "reinforcement", "PostgreSQL is the database.", _T1, semantic_facts=[fact]
    )
    await memory.consolidate_semantics(tenant_id=tenant, as_of=_T1)
    correction = {
        **fact,
        "object_value": "mysql",
        "correction": {
            "replaces_revision_key": original.revision_key,
            "effective_at": _T2.isoformat(),
        },
    }
    await _observe(
        memory, "correction", "Service database MySQL.", _T2, semantic_facts=[correction]
    )
    await memory.consolidate_semantics(tenant_id=tenant, as_of=_T2)
    earlier = await memory._semantic_store.historical_revisions(
        tenant_id=tenant, revision_keys={original.revision_key}, as_of=earlier_time
    )
    assert earlier == {original.revision_key: original}
    assert (
        await memory.recall("database PostgreSQL", tenant_id=tenant, as_of=earlier_time) == before
    )
    assert (
        await memory.recall(
            "database PostgreSQL", tenant_id=tenant, as_of=earlier_time, valid_at=earlier_time
        )
        == before_valid
    )
    assert (
        await memory.inspect_recall("database PostgreSQL", tenant_id=tenant, as_of=earlier_time)
    ).considered_count == before_inspection.considered_count
    after_slot = await memory.inspect_semantic_slot(slot, tenant_id=tenant, as_of=earlier_time)
    assert after_slot == before_slot
    assert await memory._semantic_store.historical_relations(tenant_id=tenant, as_of=_T1) == []
    relations = await memory._semantic_store.historical_relations(tenant_id=tenant, as_of=_T2)
    assert len(relations) == 1
    assert relations[0].left_revision_key == original.revision_key
    assert relations[0].relation.value == "supersedes"
    assert await memory._semantic_store.historical_relations(tenant_id="other", as_of=_T2) == []


async def test_revision_relation_history_memory_and_sqlite(tmp_path: Path) -> None:
    from cogkura import ComplementaryLearningSemanticConsolidator, SemanticCorrectionConfig

    kwargs = {
        "semantic_consolidator": ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
        "semantic_correction_config": SemanticCorrectionConfig(enabled=True),
    }
    await _exercise_revision_relations(Memory(**kwargs))
    path = tmp_path / "revision-history.db"
    async with await EmbeddedMemory.open(path, **kwargs) as memory:
        await _exercise_revision_relations(memory)
    async with await EmbeddedMemory.open(path, **kwargs) as memory:
        revisions = await memory.list_semantic_revisions(tenant_id="historical-candidate")
        snapshots = await memory._semantic_store.historical_revisions(
            tenant_id="historical-candidate",
            revision_keys={r.revision_key for r in revisions},
            as_of=_T0 + timedelta(hours=12),
        )
        assert len(snapshots) == 1
        assert (
            len(
                await memory._semantic_store.historical_relations(
                    tenant_id="historical-candidate", as_of=_T2
                )
            )
            == 1
        )


async def test_future_semantic_is_not_visible_before_support() -> None:
    from cogkura import ComplementaryLearningSemanticConsolidator

    memory = Memory(
        activation_config=ActivationConfig(retrieval_threshold=-10),
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
    )
    await _observe(
        memory,
        "future-fact",
        "Service database is PostgreSQL.",
        _T2,
        semantic_facts=[
            {
                "predicate": "database",
                "object_value": "PostgreSQL",
                "cardinality": "one",
                "authority": {"basis": "application_asserted"},
            }
        ],
    )
    await memory.consolidate_semantics(tenant_id="historical-candidate", as_of=_T2)
    for valid_at in (None, _T1, _T2):
        assert not await memory.recall(
            "database PostgreSQL", tenant_id="historical-candidate", as_of=_T1, valid_at=valid_at
        )
    inspection = await memory.inspect_recall(
        "database PostgreSQL", tenant_id="historical-candidate", as_of=_T2
    )
    assert inspection.considered_count == 2


async def test_future_semantic_support_does_not_change_earlier_episode() -> None:
    from cogkura import ComplementaryLearningSemanticConsolidator

    memory = Memory(
        activation_config=ActivationConfig(retrieval_threshold=-10),
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
    )
    fact = {
        "predicate": "database",
        "object_value": "postgresql",
        "cardinality": "one",
        "authority": {"basis": "application_asserted"},
    }
    await _observe(memory, "known-fact", "Service database PostgreSQL.", _T0, semantic_facts=[fact])
    before = await memory.recall("database PostgreSQL", tenant_id="historical-candidate", as_of=_T1)
    await _observe(memory, "later-fact", "Service database PostgreSQL.", _T2, semantic_facts=[fact])
    await memory.consolidate_semantics(tenant_id="historical-candidate", as_of=_T2)
    after = await memory.recall("database PostgreSQL", tenant_id="historical-candidate", as_of=_T1)
    assert after == before


async def test_knowledge_time_does_not_exclude_known_future_validity() -> None:
    from cogkura import ComplementaryLearningSemanticConsolidator

    memory = Memory(
        activation_config=ActivationConfig(retrieval_threshold=-10),
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
    )
    await _observe(
        memory,
        "known-scheduled",
        "Service will use PostgreSQL.",
        _T0,
        semantic_facts=[
            {
                "predicate": "database",
                "object_value": "postgresql",
                "cardinality": "one",
                "authority": {"basis": "application_asserted"},
                "valid_from": _T2.isoformat(),
            }
        ],
    )
    await memory.consolidate_semantics(tenant_id="historical-candidate", as_of=_T0)
    inspection = await memory.inspect_recall(
        "database PostgreSQL", tenant_id="historical-candidate", as_of=_T1, valid_at=_T2
    )
    assert inspection.considered_count == 2


async def test_mixed_time_semantic_support_is_restored_from_snapshot() -> None:
    from cogkura import ComplementaryLearningSemanticConsolidator

    memory = Memory(
        activation_config=ActivationConfig(retrieval_threshold=-10),
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
    )
    fact = {
        "predicate": "database",
        "object_value": "postgresql",
        "cardinality": "one",
        "authority": {"basis": "application_asserted"},
    }
    await _observe(memory, "known", "Service database PostgreSQL.", _T0, semantic_facts=[fact])
    await memory.consolidate_semantics(tenant_id="historical-candidate", as_of=_T0)
    before = await memory.inspect_recall(
        "database PostgreSQL", tenant_id="historical-candidate", as_of=_T1
    )
    assert before.considered_count == 2
    await _observe(memory, "later", "PostgreSQL remains the database.", _T2, semantic_facts=[fact])
    await memory.consolidate_semantics(tenant_id="historical-candidate", as_of=_T2)
    for valid_at in (None, _T1, _T2):
        after = await memory.inspect_recall(
            "database PostgreSQL", tenant_id="historical-candidate", as_of=_T1, valid_at=valid_at
        )
        assert after.considered_count == 2


async def _exercise_semantic_snapshots(memory: Memory) -> None:
    fact = {
        "predicate": "database",
        "object_value": "postgresql",
        "cardinality": "one",
        "authority": {"basis": "application_asserted"},
    }
    tenant = "historical-candidate"
    await _observe(memory, "known", "Service database PostgreSQL.", _T0, semantic_facts=[fact])
    await memory.consolidate_semantics(tenant_id=tenant, as_of=_T0)
    before = (await memory.list_semantic_memories(tenant_id=tenant))[0]
    before_revision = (await memory.list_semantic_revisions(tenant_id=tenant))[0]
    before_recall = await memory.recall("database PostgreSQL", tenant_id=tenant, as_of=_T1)
    await _observe(memory, "later", "PostgreSQL remains the database.", _T2, semantic_facts=[fact])
    await memory.consolidate_semantics(tenant_id=tenant, as_of=_T2)
    after = (await memory.list_semantic_memories(tenant_id=tenant))[0]
    assert after.support_count > before.support_count
    snapshot = await memory._semantic_store.historical_semantics(
        tenant_id=tenant, memory_ids={before.id}, as_of=_T1
    )
    assert snapshot == {before.id: before}
    assert await memory._semantic_store.historical_revisions(
        tenant_id=tenant, revision_keys={before_revision.revision_key}, as_of=_T1
    ) == {before_revision.revision_key: before_revision}
    assert (
        await memory._semantic_store.historical_semantics(
            tenant_id="other", memory_ids={before.id}, as_of=_T1
        )
        == {}
    )
    assert await memory.recall("database PostgreSQL", tenant_id=tenant, as_of=_T1) == before_recall
    inspection = await memory.inspect_recall("database PostgreSQL", tenant_id=tenant, as_of=_T1)
    assert inspection.considered_count == 2


async def _exercise_linked_mixed_episode(memory: Memory) -> None:
    tenant = "historical-candidate"
    fact = {
        "predicate": "database",
        "object_value": "postgresql",
        "cardinality": "one",
        "authority": {"basis": "application_asserted"},
    }
    await _observe(
        memory,
        "known",
        "Service database PostgreSQL.",
        _T0,
        session_id="shared",
        semantic_facts=[fact],
    )
    await memory.consolidate_semantics(tenant_id=tenant, as_of=_T0)
    before = await memory.recall("database PostgreSQL", tenant_id=tenant, as_of=_T1)
    before_valid = await memory.recall(
        "database PostgreSQL", tenant_id=tenant, as_of=_T1, valid_at=_T1
    )
    assert before
    assert (
        await memory.inspect_recall("database PostgreSQL", tenant_id=tenant, as_of=_T1)
    ).considered_count == 2
    await _observe(
        memory,
        "later",
        "PostgreSQL remains the future database.",
        _T2,
        session_id="shared",
        semantic_facts=[fact],
    )
    await memory.consolidate_semantics(tenant_id=tenant, as_of=_T2)
    after = await memory.recall("database PostgreSQL", tenant_id=tenant, as_of=_T1)
    assert after == before
    assert (
        await memory.recall("database PostgreSQL", tenant_id=tenant, as_of=_T1, valid_at=_T1)
        == before_valid
    )
    assert "future" not in repr(after)
    assert (
        await memory.inspect_recall("database PostgreSQL", tenant_id=tenant, as_of=_T1)
    ).considered_count == 2


async def test_linked_mixed_episode_memory_and_sqlite(tmp_path: Path) -> None:
    from cogkura import ComplementaryLearningSemanticConsolidator

    consolidator = ComplementaryLearningSemanticConsolidator(
        allow_authoritative_single_episode=True
    )
    config = ActivationConfig(retrieval_threshold=-10)
    await _exercise_linked_mixed_episode(
        Memory(activation_config=config, semantic_consolidator=consolidator)
    )
    async with await EmbeddedMemory.open(
        tmp_path / "linked.db", activation_config=config, semantic_consolidator=consolidator
    ) as memory:
        await _exercise_linked_mixed_episode(memory)


async def _exercise_linked_mixed_before_correction(memory: Memory) -> None:
    tenant = "historical-candidate"
    fact = {
        "predicate": "database",
        "object_value": "postgresql",
        "cardinality": "one",
        "authority": {"basis": "application_asserted"},
    }
    await _observe(
        memory,
        "known",
        "Service database PostgreSQL.",
        _T0,
        session_id="shared",
        semantic_facts=[fact],
    )
    await memory.consolidate_semantics(tenant_id=tenant, as_of=_T0)
    predecessor = (await memory.list_semantic_revisions(tenant_id=tenant))[0].revision_key
    before = await memory.recall("database PostgreSQL", tenant_id=tenant, as_of=_T1)
    assert before
    before_valid = await memory.recall(
        "database PostgreSQL", tenant_id=tenant, as_of=_T1, valid_at=_T1
    )
    await _observe(
        memory,
        "later",
        "PostgreSQL future evidence.",
        _T2,
        session_id="shared",
        semantic_facts=[fact],
    )
    await memory.consolidate_semantics(tenant_id=tenant, as_of=_T2)
    correction_at = _T2 + timedelta(days=1)
    replacement = {
        **fact,
        "object_value": "mysql",
        "correction": {
            "replaces_revision_key": predecessor,
            "effective_at": correction_at.isoformat(),
        },
    }
    await _observe(
        memory, "correction", "Service database MySQL.", correction_at, semantic_facts=[replacement]
    )
    await memory.consolidate_semantics(tenant_id=tenant, as_of=correction_at)
    assert await memory.recall("database PostgreSQL", tenant_id=tenant, as_of=_T1) == before
    assert (
        await memory.recall("database PostgreSQL", tenant_id=tenant, as_of=_T1, valid_at=_T1)
        == before_valid
    )


async def test_linked_mixed_episode_before_later_correction(tmp_path: Path) -> None:
    from cogkura import ComplementaryLearningSemanticConsolidator, SemanticCorrectionConfig

    kwargs = {
        "activation_config": ActivationConfig(retrieval_threshold=-10),
        "semantic_consolidator": ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
        "semantic_correction_config": SemanticCorrectionConfig(enabled=True),
    }
    await _exercise_linked_mixed_before_correction(Memory(**kwargs))
    async with await EmbeddedMemory.open(tmp_path / "linked-correction.db", **kwargs) as memory:
        await _exercise_linked_mixed_before_correction(memory)


async def _exercise_inactive_linked_episode(memory: Memory) -> None:
    tenant = "historical-candidate"
    fact = {
        "predicate": "database",
        "object_value": "postgresql",
        "cardinality": "one",
        "authority": {"basis": "application_asserted"},
    }
    await _observe(memory, "removed", "Service database PostgreSQL.", _T0, semantic_facts=[fact])
    await memory.consolidate_semantics(tenant_id=tenant, as_of=_T0)
    before = await memory.recall("database PostgreSQL", tenant_id=tenant, as_of=_T1)
    await memory.observe(
        ObservationInput(
            tenant_id=tenant,
            subject_id="service",
            source_namespace="history-test",
            source_record_id="removed",
            content="",
            observed_at=_T2,
            is_deleted=True,
        )
    )
    await memory.encode_episodes(tenant_id=tenant, as_of=_T2)
    assert await memory.recall("database PostgreSQL", tenant_id=tenant, as_of=_T1) == before
    assert (
        await memory.inspect_recall("database PostgreSQL", tenant_id=tenant, as_of=_T1)
    ).considered_count == 2


async def test_inactive_linked_episode_memory_and_sqlite(tmp_path: Path) -> None:
    from cogkura import ComplementaryLearningSemanticConsolidator

    consolidator = ComplementaryLearningSemanticConsolidator(
        allow_authoritative_single_episode=True
    )
    config = ActivationConfig(retrieval_threshold=-10)
    await _exercise_inactive_linked_episode(
        Memory(activation_config=config, semantic_consolidator=consolidator)
    )
    async with await EmbeddedMemory.open(
        tmp_path / "inactive-linked.db",
        activation_config=config,
        semantic_consolidator=consolidator,
    ) as memory:
        await _exercise_inactive_linked_episode(memory)


async def _exercise_inactive_semantic(memory: Memory) -> None:
    tenant = "historical-candidate"
    fact = {
        "predicate": "database",
        "object_value": "postgresql",
        "cardinality": "one",
        "authority": {"basis": "application_asserted"},
    }
    await _observe(memory, "known", "Service database PostgreSQL.", _T0, semantic_facts=[fact])
    await memory.consolidate_semantics(tenant_id=tenant, as_of=_T0)
    before = await memory.recall("database PostgreSQL", tenant_id=tenant, as_of=_T1)
    before_valid = await memory.recall(
        "database PostgreSQL", tenant_id=tenant, as_of=_T1, valid_at=_T1
    )
    assert (
        await memory.inspect_recall("database PostgreSQL", tenant_id=tenant, as_of=_T1)
    ).considered_count == 2
    await memory._semantic_store.deactivate_missing(
        tenant_id=tenant, subject_id=None, active_memory_keys=set(), as_of=_T2
    )
    assert await memory.recall("database PostgreSQL", tenant_id=tenant, as_of=_T1) == before
    assert (
        await memory.recall("database PostgreSQL", tenant_id=tenant, as_of=_T1, valid_at=_T1)
        == before_valid
    )
    assert (
        await memory.inspect_recall(
            "database PostgreSQL", tenant_id=tenant, as_of=_T2, valid_at=_T2
        )
    ).considered_count == 1
    assert (
        await memory.inspect_recall("database PostgreSQL", tenant_id=tenant, as_of=_T2)
    ).considered_count == 1


async def test_inactive_semantic_memory_and_sqlite(tmp_path: Path) -> None:
    from cogkura import ComplementaryLearningSemanticConsolidator

    consolidator = ComplementaryLearningSemanticConsolidator(
        allow_authoritative_single_episode=True
    )
    config = ActivationConfig(retrieval_threshold=-10)
    await _exercise_inactive_semantic(
        Memory(activation_config=config, semantic_consolidator=consolidator)
    )
    async with await EmbeddedMemory.open(
        tmp_path / "inactive-semantic.db",
        activation_config=config,
        semantic_consolidator=consolidator,
    ) as memory:
        await _exercise_inactive_semantic(memory)


async def test_semantic_snapshots_memory_and_sqlite(tmp_path: Path) -> None:
    from cogkura import ComplementaryLearningSemanticConsolidator

    consolidator = ComplementaryLearningSemanticConsolidator(
        allow_authoritative_single_episode=True
    )
    await _exercise_semantic_snapshots(Memory(semantic_consolidator=consolidator))
    async with await EmbeddedMemory.open(
        tmp_path / "semantic-history.db", semantic_consolidator=consolidator
    ) as memory:
        await _exercise_tenant_history_order(memory)
        await _exercise_semantic_snapshots(memory)
    async with await EmbeddedMemory.open(
        tmp_path / "semantic-history.db", semantic_consolidator=consolidator
    ) as memory:
        inspection = await memory.inspect_recall(
            "database PostgreSQL", tenant_id="historical-candidate", as_of=_T1
        )
        assert inspection.considered_count == 2


async def test_semantic_history_fails_closed_when_reconciliation_is_backdated() -> None:
    from cogkura import ComplementaryLearningSemanticConsolidator

    memory = Memory(
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        )
    )
    await _exercise_semantic_snapshots(memory)
    await memory.consolidate_semantics(tenant_id="historical-candidate", as_of=_T0)
    from cogkura import HistoricalStateUnavailableError

    with pytest.raises(HistoricalStateUnavailableError):
        await memory.inspect_recall(
            "database PostgreSQL", tenant_id="historical-candidate", as_of=_T1
        )


async def test_semantic_snapshot_failure_rolls_back_sqlite_reconciliation(tmp_path: Path) -> None:
    from cogkura import ComplementaryLearningSemanticConsolidator
    from cogkura.exceptions import StorageError

    fact = {
        "predicate": "database",
        "object_value": "postgresql",
        "cardinality": "one",
        "authority": {"basis": "application_asserted"},
    }
    owner = await EmbeddedMemory.open(
        tmp_path / "semantic-atomic.db",
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
    )
    async with owner as memory:
        await _observe(memory, "known", "Service database PostgreSQL.", _T0, semantic_facts=[fact])
        await memory.consolidate_semantics(tenant_id="historical-candidate", as_of=_T0)
        before = await memory.list_semantic_memories(tenant_id="historical-candidate")
        await _observe(
            memory, "later", "PostgreSQL remains the database.", _T2, semantic_facts=[fact]
        )
        await owner._storage._run(
            lambda: owner._storage._conn().execute(
                "CREATE TRIGGER fail_semantic_history BEFORE INSERT ON semantic_history "
                "BEGIN SELECT RAISE(ABORT, 'injected failure'); END"
            )
        )
        with pytest.raises(StorageError):
            await memory.consolidate_semantics(tenant_id="historical-candidate", as_of=_T2)
        assert await memory.list_semantic_memories(tenant_id="historical-candidate") == before


@pytest.mark.parametrize(
    ("action", "history_table"),
    [
        ("correction", "semantic_history"),
        ("correction", "semantic_revision_history"),
        ("correction", "semantic_relation_history"),
        ("withdrawal", "semantic_history"),
        ("withdrawal", "semantic_revision_history"),
    ],
)
async def test_transition_history_failure_rolls_back_sqlite_semantic_transaction(
    tmp_path: Path, action: str, history_table: str
) -> None:
    from cogkura import ComplementaryLearningSemanticConsolidator, SemanticCorrectionConfig
    from cogkura.exceptions import StorageError

    tenant = "historical-candidate"
    fact = {
        "predicate": "database",
        "object_value": "postgresql",
        "cardinality": "one",
        "authority": {"basis": "application_asserted"},
    }
    owner = await EmbeddedMemory.open(
        tmp_path / "correction-atomic.db",
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
        semantic_correction_config=SemanticCorrectionConfig(enabled=True),
    )
    async with owner as memory:
        await _observe(memory, "original", "database PostgreSQL", _T0, semantic_facts=[fact])
        await memory.consolidate_semantics(tenant_id=tenant, as_of=_T0)
        original = (await memory.list_semantic_revisions(tenant_id=tenant))[0]
        if action == "correction":
            correction = {
                **fact,
                "object_value": "mysql",
                "correction": {
                    "replaces_revision_key": original.revision_key,
                    "effective_at": _T2.isoformat(),
                },
            }
            await _observe(memory, "correction", "database MySQL", _T2, semantic_facts=[correction])
        else:
            await _observe(
                memory,
                "withdrawal",
                "database guidance withdrawn",
                _T2,
                semantic_withdrawals=[
                    {
                        "target_revision_key": original.revision_key,
                        "effective_at": _T2.isoformat(),
                        "authority": {"basis": "application_asserted"},
                    }
                ],
            )
        tables = (
            "semantic_memories",
            "semantic_revisions",
            "semantic_relations",
            "semantic_derivations",
            "semantic_evidence",
            "semantic_entities",
            "semantic_history",
            "semantic_revision_history",
            "semantic_relation_history",
            "history_order",
        )

        def snapshot() -> dict[str, list[tuple]]:
            conn = owner._storage._conn()
            return {
                name: [tuple(row) for row in conn.execute(f"SELECT * FROM {name}")]
                for name in tables
            }

        before = await owner._storage._run(snapshot)
        await owner._storage._run(
            lambda: owner._storage._conn().execute(
                f"CREATE TRIGGER fail_history BEFORE INSERT ON {history_table} "
                "BEGIN SELECT RAISE(ABORT, 'injected failure'); END"
            )
        )
        with pytest.raises(StorageError):
            await memory.consolidate_semantics(tenant_id=tenant, as_of=_T2)
        assert await owner._storage._run(snapshot) == before
        await owner._storage._run(
            lambda: owner._storage._conn().execute("DROP TRIGGER fail_history")
        )
        result = await memory.consolidate_semantics(tenant_id=tenant, as_of=_T2)
        if action == "correction":
            assert result.authoritative_corrections_applied == 1
        else:
            assert result.authoritative_withdrawals_applied == 1


@pytest.mark.postgres
@pytest.mark.parametrize(
    ("action", "history_table"),
    [
        ("correction", "semantic_history"),
        ("correction", "semantic_revision_history"),
        ("correction", "semantic_relation_history"),
        ("withdrawal", "semantic_history"),
        ("withdrawal", "semantic_revision_history"),
    ],
)
async def test_transition_history_failure_rolls_back_postgres_semantic_transaction(
    postgres_memory_url: str | None, action: str, history_table: str
) -> None:
    if not postgres_memory_url:
        pytest.skip("COGKURA_POSTGRES_MEMORY_URL is not set")
    from sqlalchemy import text
    from sqlalchemy.exc import SQLAlchemyError
    from sqlalchemy.ext.asyncio import create_async_engine

    from cogkura import ComplementaryLearningSemanticConsolidator, SemanticCorrectionConfig
    from cogkura.migrations import apply_migrations
    from cogkura.storage.postgres import (
        PostgresEpisodeStore,
        PostgresObservationStore,
        PostgresSemanticMemoryStore,
    )

    tenant = "historical-candidate"
    engine = create_async_engine(postgres_memory_url)
    await apply_migrations(engine)
    memory = Memory(
        observation_store=PostgresObservationStore(engine),
        episode_store=PostgresEpisodeStore(engine),
        semantic_store=PostgresSemanticMemoryStore(engine),
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
        semantic_correction_config=SemanticCorrectionConfig(enabled=True),
    )
    tables = (
        "memories",
        "semantic_claims",
        "memory_derivations",
        "semantic_claim_revisions",
        "semantic_revision_relations",
        "semantic_history",
        "semantic_revision_history",
        "semantic_relation_history",
        "history_order",
    )

    async def snapshot() -> dict[str, list[str]]:
        async with engine.connect() as conn:
            return {
                name: sorted(
                    map(
                        repr,
                        (
                            await conn.execute(
                                text(f"SELECT * FROM cogkura.{name} WHERE tenant_id=:tenant"),
                                {"tenant": tenant},
                            )
                        ).fetchall(),
                    )
                )
                for name in tables
            }

    try:
        await memory.clear(tenant_id=tenant)
        fact = {
            "predicate": "database",
            "object_value": "postgresql",
            "cardinality": "one",
            "authority": {"basis": "application_asserted"},
        }
        await _observe(memory, "original", "database PostgreSQL", _T0, semantic_facts=[fact])
        await memory.consolidate_semantics(tenant_id=tenant, as_of=_T0)
        original = (await memory.list_semantic_revisions(tenant_id=tenant))[0]
        if action == "correction":
            correction = {
                **fact,
                "object_value": "mysql",
                "correction": {
                    "replaces_revision_key": original.revision_key,
                    "effective_at": _T2.isoformat(),
                },
            }
            await _observe(memory, "correction", "database MySQL", _T2, semantic_facts=[correction])
        else:
            await _observe(
                memory,
                "withdrawal",
                "database guidance withdrawn",
                _T2,
                semantic_withdrawals=[
                    {
                        "target_revision_key": original.revision_key,
                        "effective_at": _T2.isoformat(),
                        "authority": {"basis": "application_asserted"},
                    }
                ],
            )
        before = await snapshot()
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "CREATE FUNCTION cogkura.test_fail_history() RETURNS trigger LANGUAGE plpgsql "
                    "AS $$ BEGIN RAISE EXCEPTION 'injected failure'; END; $$"
                )
            )
            await conn.execute(
                text(
                    f"CREATE TRIGGER test_fail_history BEFORE INSERT ON cogkura.{history_table} "
                    "FOR EACH ROW EXECUTE FUNCTION cogkura.test_fail_history()"
                )
            )
        with pytest.raises(SQLAlchemyError):
            await memory.consolidate_semantics(tenant_id=tenant, as_of=_T2)
        assert await snapshot() == before
        async with engine.begin() as conn:
            await conn.execute(text(f"DROP TRIGGER test_fail_history ON cogkura.{history_table}"))
            await conn.execute(text("DROP FUNCTION cogkura.test_fail_history()"))
        result = await memory.consolidate_semantics(tenant_id=tenant, as_of=_T2)
        if action == "correction":
            assert result.authoritative_corrections_applied == 1
        else:
            assert result.authoritative_withdrawals_applied == 1
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text(f"DROP TRIGGER IF EXISTS test_fail_history ON cogkura.{history_table}")
            )
            await conn.execute(text("DROP FUNCTION IF EXISTS cogkura.test_fail_history()"))
        await memory.clear(tenant_id=tenant)
        await engine.dispose()


@pytest.mark.postgres
async def test_semantic_snapshots_postgres(postgres_memory_url: str | None) -> None:
    if not postgres_memory_url:
        pytest.skip("COGKURA_POSTGRES_MEMORY_URL is not set")
    from sqlalchemy.ext.asyncio import create_async_engine

    from cogkura import ComplementaryLearningSemanticConsolidator, SemanticCorrectionConfig
    from cogkura.migrations import apply_migrations
    from cogkura.storage.postgres import (
        PostgresEpisodeStore,
        PostgresObservationStore,
        PostgresSemanticMemoryStore,
    )

    engine = create_async_engine(postgres_memory_url)
    await apply_migrations(engine)
    memory = Memory(
        observation_store=PostgresObservationStore(engine),
        episode_store=PostgresEpisodeStore(engine),
        semantic_store=PostgresSemanticMemoryStore(engine),
        activation_config=ActivationConfig(retrieval_threshold=-10),
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
    )
    try:
        await memory.clear(tenant_id="historical-candidate")
        await _exercise_equal_time_episode_updates(memory)
        await memory.clear(tenant_id="historical-candidate")
        await _exercise_tenant_history_order(memory)
        await _exercise_semantic_snapshots(memory)
        await memory.clear(tenant_id="historical-candidate")
        await _exercise_linked_mixed_episode(memory)
        await memory.clear(tenant_id="historical-candidate")
        await _exercise_inactive_linked_episode(memory)
        await memory.clear(tenant_id="historical-candidate")
        await _exercise_inactive_semantic(memory)
        await memory.clear(tenant_id="historical-candidate")
        correction_memory = Memory(
            observation_store=PostgresObservationStore(engine),
            episode_store=PostgresEpisodeStore(engine),
            semantic_store=PostgresSemanticMemoryStore(engine),
            activation_config=ActivationConfig(retrieval_threshold=-10),
            semantic_consolidator=ComplementaryLearningSemanticConsolidator(
                allow_authoritative_single_episode=True
            ),
            semantic_correction_config=SemanticCorrectionConfig(enabled=True),
        )
        await _exercise_revision_relations(correction_memory)
        await correction_memory.clear(tenant_id="historical-candidate")
        await _exercise_linked_mixed_before_correction(correction_memory)
        await correction_memory.clear(tenant_id="historical-candidate")
        await _exercise_contested_revision_snapshot(correction_memory)
        await correction_memory.clear(tenant_id="historical-candidate")
        await _exercise_shared_contested_history(correction_memory)
        await correction_memory.clear(tenant_id="historical-candidate")
        await _exercise_shared_contested_history(correction_memory, remove_snapshot=True)
        await correction_memory.clear(tenant_id="historical-candidate")
        await _exercise_shared_contested_history(correction_memory, corrupt_snapshot=True)
        await correction_memory.clear(tenant_id="historical-candidate")
        await _exercise_mixed_episode_after_correction(correction_memory)
        await correction_memory.clear(tenant_id="historical-candidate")
        await _exercise_mixed_multiple_revisions(correction_memory)
    finally:
        await memory.clear(tenant_id="historical-candidate")
        await engine.dispose()


async def _exercise_tenant_history_order(memory: Memory) -> None:
    tenant = "historical-candidate"
    status = memory._episode_store.historical_order_status
    await _observe(memory, "first", "first claim", _T0)
    await _observe(memory, "same-time", "another claim", _T0)
    assert (await status(tenant_id=tenant)).complete
    assert not (await status(tenant_id=tenant)).ambiguous
    await _observe(memory, "later", "later claim", _T2)
    await _observe(memory, "backdated", "backdated claim", _T1)
    result = await status(tenant_id=tenant)
    assert result.ambiguous
    assert result.last_recorded_at == _T2
    from cogkura import HistoricalStateUnavailableError

    with pytest.raises(HistoricalStateUnavailableError):
        await memory.recall("claim", tenant_id=tenant, as_of=_T1)
    with pytest.raises(HistoricalStateUnavailableError):
        await memory.inspect_recall("claim", tenant_id=tenant, as_of=_T1)
    with pytest.raises(HistoricalStateUnavailableError):
        await memory.prepare_context("claim", tenant_id=tenant, as_of=_T1)
    assert await memory.recall("claim", tenant_id=tenant, as_of=_T2)
    assert not (await status(tenant_id="other-tenant")).ambiguous
    await memory.clear(tenant_id=tenant)
    assert not (await status(tenant_id=tenant)).ambiguous
    assert (await status(tenant_id=tenant)).last_recorded_at is None

    fact = {
        "subject_entity_id": "service",
        "predicate": "database",
        "object_value": "postgresql",
        "cardinality": "one",
        "authority": {"basis": "application_asserted"},
    }
    await _observe(memory, "semantic", "service database postgresql", _T0, semantic_facts=[fact])
    await memory.consolidate_semantics(tenant_id=tenant, as_of=_T2)
    assert (await status(tenant_id=tenant)).last_recorded_at == _T2
    await _observe(memory, "older-episode", "older other memory", _T1)
    assert (await status(tenant_id=tenant)).ambiguous
    await memory.clear(tenant_id=tenant)


async def test_tenant_history_order_memory_and_sqlite(tmp_path: Path) -> None:
    from cogkura import ComplementaryLearningSemanticConsolidator

    kwargs = {
        "semantic_consolidator": ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        )
    }
    await _exercise_tenant_history_order(Memory(**kwargs))
    path = tmp_path / "history-order.db"
    async with await EmbeddedMemory.open(path, **kwargs) as memory:
        await _exercise_tenant_history_order(memory)
        await _observe(memory, "later", "later claim", _T2)
        await _observe(memory, "backdated", "backdated claim", _T1)
    async with await EmbeddedMemory.open(path) as memory:
        result = await memory._episode_store.historical_order_status(
            tenant_id="historical-candidate"
        )
        assert result.ambiguous
        assert result.last_recorded_at == _T2


async def test_history_order_migration_marks_existing_tenants_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cogkura.storage.sqlite as sqlite_module

    paths = sqlite_module._migrations()
    monkeypatch.setattr(sqlite_module, "_migrations", lambda: paths[:7])
    path = tmp_path / "legacy-order.db"
    async with await EmbeddedMemory.open(path) as memory:
        await _observe(memory, "existing", "existing claim", _T0)
    monkeypatch.setattr(sqlite_module, "_migrations", lambda: paths)
    async with await EmbeddedMemory.open(path) as memory:
        status = memory._episode_store.historical_order_status
        assert not (await status(tenant_id="historical-candidate")).complete
        assert (await status(tenant_id="new-tenant")).complete
        await memory.clear(tenant_id="historical-candidate")
        assert (await status(tenant_id="historical-candidate")).complete


async def test_history_order_update_rolls_back_with_snapshot(tmp_path: Path) -> None:
    owner = await EmbeddedMemory.open(tmp_path / "order-rollback.db")
    async with owner as memory:
        await _observe(memory, "later", "later claim", _T2)

        def rollback() -> None:
            conn = owner._storage._conn()
            conn.execute("BEGIN")
            conn.execute(
                "INSERT INTO episode_history(tenant_id,episode_id,recorded_at,snapshot) "
                "SELECT tenant_id,episode_id,?,snapshot FROM episode_history LIMIT 1",
                (_T0.isoformat(),),
            )
            conn.rollback()

        await owner._storage._run(rollback)
        status = await memory._episode_store.historical_order_status(
            tenant_id="historical-candidate"
        )
        assert not status.ambiguous
        assert status.last_recorded_at == _T2


async def _exercise_contested_revision_snapshot(memory: Memory) -> None:
    from cogkura import SemanticKnowledgeState, SemanticSlotQuery

    tenant = "historical-candidate"
    for value in ("startup", "pipeline"):
        for number in range(2):
            await memory.observe(
                ObservationInput(
                    tenant_id=tenant,
                    subject_id="service",
                    source_namespace="history-test",
                    source_record_id=f"{value}-{number}",
                    content=f"Deployment policy is {value} ({number}).",
                    observed_at=_T0,
                    metadata={
                        "conversation_id": f"{value}-{number}",
                        "semantic_facts": [
                            {
                                "subject_entity_id": "service",
                                "predicate": "deployment_policy",
                                "object_value": value,
                                "cardinality": "one",
                                "valid_from": _T0.isoformat(),
                                "valid_until": (_T0 + timedelta(days=10)).isoformat(),
                            }
                        ],
                    },
                )
            )
    await memory.process(tenant_id=tenant, as_of=_T0)
    slot = SemanticSlotQuery("service", "deployment_policy")
    before_slot = await memory.inspect_semantic_slot(slot, tenant_id=tenant, as_of=_T1)
    assert before_slot.state is SemanticKnowledgeState.CONTESTED
    before = {
        factual_time: await memory.inspect_recall(
            "deployment policy", tenant_id=tenant, as_of=_T1, valid_at=factual_time, limit=20
        )
        for factual_time in (None, _T1, _T2)
    }
    await memory.observe(
        ObservationInput(
            tenant_id=tenant,
            subject_id="service",
            source_namespace="history-test",
            source_record_id="startup-later",
            content="Deployment policy is startup (later).",
            observed_at=_T2,
            metadata={
                "conversation_id": "startup-later",
                "semantic_facts": [
                    {
                        "subject_entity_id": "service",
                        "predicate": "deployment_policy",
                        "object_value": "startup",
                        "cardinality": "one",
                        "valid_from": _T0.isoformat(),
                        "valid_until": (_T0 + timedelta(days=10)).isoformat(),
                    }
                ],
            },
        )
    )
    await memory.process(tenant_id=tenant, as_of=_T2)
    assert await memory.inspect_semantic_slot(slot, tenant_id=tenant, as_of=_T1) == before_slot
    for factual_time, expected in before.items():
        assert (
            await memory.inspect_recall(
                "deployment policy", tenant_id=tenant, as_of=_T1, valid_at=factual_time, limit=20
            )
            == expected
        )


async def test_contested_revision_snapshots_memory_and_sqlite(tmp_path: Path) -> None:
    kwargs = {"activation_config": ActivationConfig(retrieval_threshold=-10)}
    await _exercise_contested_revision_snapshot(Memory(**kwargs))
    async with await EmbeddedMemory.open(tmp_path / "contested.db", **kwargs) as memory:
        await _exercise_contested_revision_snapshot(memory)


async def _exercise_shared_contested_history(
    memory: Memory,
    *,
    remove_snapshot: bool = False,
    corrupt_snapshot: bool = False,
    snapshot_payload: str = '{"version": 999}',
) -> None:
    from cogkura import SemanticKnowledgeState, SemanticSlotQuery

    tenant = "historical-candidate"
    facts = [
        {
            "subject_entity_id": "service",
            "predicate": "deployment_policy",
            "object_value": value,
            "cardinality": "one",
            "valid_from": _T0.isoformat(),
            "valid_until": (_T0 + timedelta(days=10)).isoformat(),
        }
        for value in ("startup", "pipeline")
    ]

    async def observe(record: str, when: datetime, session: str) -> None:
        await memory.observe(
            ObservationInput(
                tenant_id=tenant,
                subject_id="service",
                source_namespace="history-test",
                source_record_id=record,
                content=f"deployment policy startup pipeline {record}",
                observed_at=when,
                metadata={"conversation_id": session, "semantic_facts": facts},
            )
        )

    await observe("first", _T0, "shared")
    await observe("other", _T0, "other")
    await memory.process(tenant_id=tenant, as_of=_T0)
    slot = SemanticSlotQuery("service", "deployment_policy")
    before_slot = await memory.inspect_semantic_slot(slot, tenant_id=tenant, as_of=_T1)
    assert before_slot.state is SemanticKnowledgeState.CONTESTED
    before = {
        factual_time: await memory.inspect_recall(
            "deployment policy", tenant_id=tenant, as_of=_T1, valid_at=factual_time, limit=20
        )
        for factual_time in (None, _T1, _T2)
    }
    assert all(result.considered_count == 4 for result in before.values())
    await observe("future secret", _T2, "shared")
    await memory.process(tenant_id=tenant, as_of=_T2)
    if remove_snapshot or corrupt_snapshot:
        semantics = await memory.list_semantic_memories(tenant_id=tenant)
        if hasattr(memory._semantic_store, "_storage"):
            storage = memory._semantic_store._storage

            def remove_sqlite_history() -> None:
                conn = storage._conn()
                if corrupt_snapshot:
                    conn.execute(
                        "UPDATE semantic_history SET snapshot=? WHERE tenant_id=? AND memory_id=?",
                        (snapshot_payload, tenant, semantics[0].id),
                    )
                else:
                    conn.execute(
                        "DELETE FROM semantic_history WHERE tenant_id=? AND memory_id=?",
                        (tenant, semantics[0].id),
                    )
                conn.commit()

            await storage._run(remove_sqlite_history)
        elif hasattr(memory._semantic_store, "_engine"):
            from sqlalchemy import text

            engine = memory._semantic_store._engine
            async with engine.begin() as conn:
                if corrupt_snapshot:
                    await conn.execute(
                        text(
                            "UPDATE cogkura.semantic_history SET snapshot=CAST(:snapshot AS jsonb) "
                            "WHERE tenant_id=:tenant AND memory_id=CAST(:memory_id AS uuid)"
                        ),
                        {
                            "snapshot": snapshot_payload,
                            "tenant": tenant,
                            "memory_id": semantics[0].id,
                        },
                    )
                else:
                    await conn.execute(
                        text(
                            "DELETE FROM cogkura.semantic_history "
                            "WHERE tenant_id=:tenant AND memory_id=CAST(:memory_id AS uuid)"
                        ),
                        {"tenant": tenant, "memory_id": semantics[0].id},
                    )
        else:
            history = memory._semantic_store._semantic_history
            key = (tenant, semantics[0].id)
            if corrupt_snapshot:
                history[key] = [(at, snapshot_payload) for at, _ in history[key]]
            else:
                history.pop(key)
        result = await memory.inspect_recall("deployment policy", tenant_id=tenant, as_of=_T1)
        candidates = (*result.returned, *result.rejected)
        assert all("future secret" not in candidate.memory.statement for candidate in candidates)
        assert all("first" not in candidate.memory.statement for candidate in candidates)
        return
    assert await memory.inspect_semantic_slot(slot, tenant_id=tenant, as_of=_T1) == before_slot
    for factual_time, expected in before.items():
        assert (
            await memory.inspect_recall(
                "deployment policy", tenant_id=tenant, as_of=_T1, valid_at=factual_time, limit=20
            )
            == expected
        )


async def test_shared_contested_history_memory_and_sqlite(tmp_path: Path) -> None:
    kwargs = {"activation_config": ActivationConfig(retrieval_threshold=-10)}
    await _exercise_shared_contested_history(Memory(**kwargs))
    await _exercise_shared_contested_history(Memory(**kwargs), remove_snapshot=True)
    await _exercise_shared_contested_history(Memory(**kwargs), corrupt_snapshot=True)
    await _exercise_shared_contested_history(
        Memory(**kwargs), corrupt_snapshot=True, snapshot_payload="not-json"
    )
    async with await EmbeddedMemory.open(tmp_path / "shared-contested.db", **kwargs) as memory:
        await _exercise_shared_contested_history(memory)
    async with await EmbeddedMemory.open(tmp_path / "missing-shared.db", **kwargs) as memory:
        await _exercise_shared_contested_history(memory, remove_snapshot=True)
    async with await EmbeddedMemory.open(tmp_path / "corrupt-shared.db", **kwargs) as memory:
        await _exercise_shared_contested_history(memory, corrupt_snapshot=True)
    async with await EmbeddedMemory.open(tmp_path / "invalid-shared.db", **kwargs) as memory:
        await _exercise_shared_contested_history(
            memory, corrupt_snapshot=True, snapshot_payload="not-json"
        )


async def _exercise_mixed_episode_after_correction(memory: Memory) -> None:
    from cogkura import SemanticSlotQuery

    tenant = "historical-candidate"
    fact = {
        "predicate": "database",
        "object_value": "postgresql",
        "cardinality": "one",
        "authority": {"basis": "application_asserted"},
    }
    await _observe(memory, "original", "Service database postgresql.", _T0, semantic_facts=[fact])
    await memory.consolidate_semantics(tenant_id=tenant, as_of=_T0)
    predecessor = (await memory.list_semantic_revisions(tenant_id=tenant))[0].revision_key
    replacement = {
        **fact,
        "object_value": "mysql",
        "correction": {"replaces_revision_key": predecessor, "effective_at": _T1.isoformat()},
    }
    await _observe(
        memory,
        "correction",
        "Service database mysql.",
        _T1,
        session_id="shared",
        semantic_facts=[replacement],
    )
    await memory.consolidate_semantics(tenant_id=tenant, as_of=_T1)
    successor = next(
        revision
        for revision in await memory.list_semantic_revisions(tenant_id=tenant)
        if revision.revision_key != predecessor
    )
    earlier = _T1 + timedelta(hours=12)
    slot = SemanticSlotQuery("service", "database")
    before_slot = await memory.inspect_semantic_slot(slot, tenant_id=tenant, as_of=earlier)
    before = {
        valid_at: await memory.inspect_recall(
            "database mysql", tenant_id=tenant, as_of=earlier, valid_at=valid_at, limit=20
        )
        for valid_at in (None, earlier, _T2)
    }
    await _observe(
        memory,
        "later",
        "Service database mysql future.",
        _T2,
        session_id="shared",
        semantic_facts=[{**fact, "object_value": "mysql"}],
    )
    await memory.consolidate_semantics(tenant_id=tenant, as_of=_T2)
    withdrawal_at = _T2 + timedelta(days=1)
    await _observe(
        memory,
        "withdraw",
        "Database claim withdrawn.",
        withdrawal_at,
        semantic_facts=[],
        semantic_withdrawals=[
            {
                "target_revision_key": successor.revision_key,
                "effective_at": withdrawal_at.isoformat(),
                "authority": {"basis": "application_asserted"},
            }
        ],
    )
    await memory.consolidate_semantics(tenant_id=tenant, as_of=withdrawal_at)
    assert await memory.inspect_semantic_slot(slot, tenant_id=tenant, as_of=earlier) == before_slot
    for valid_at, expected in before.items():
        assert (
            await memory.inspect_recall(
                "database mysql", tenant_id=tenant, as_of=earlier, valid_at=valid_at, limit=20
            )
            == expected
        )
    relations = await memory._semantic_store.historical_relations(
        tenant_id=tenant, as_of=withdrawal_at
    )
    assert {relation.relation.value for relation in relations} >= {"supersedes"}


async def test_mixed_episode_after_correction_memory_and_sqlite(tmp_path: Path) -> None:
    from cogkura import ComplementaryLearningSemanticConsolidator, SemanticCorrectionConfig

    kwargs = {
        "activation_config": ActivationConfig(retrieval_threshold=-10),
        "semantic_consolidator": ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
        "semantic_correction_config": SemanticCorrectionConfig(enabled=True),
    }
    await _exercise_mixed_episode_after_correction(Memory(**kwargs))
    async with await EmbeddedMemory.open(tmp_path / "mixed-transition.db", **kwargs) as memory:
        await _exercise_mixed_episode_after_correction(memory)


async def _exercise_mixed_multiple_revisions(memory: Memory) -> None:
    from cogkura import SemanticSlotQuery

    tenant = "historical-candidate"
    fact = {
        "predicate": "database",
        "object_value": "postgresql",
        "cardinality": "one",
        "authority": {"basis": "application_asserted"},
    }
    await _observe(memory, "original", "Service database postgresql.", _T0, semantic_facts=[fact])
    await memory.consolidate_semantics(tenant_id=tenant, as_of=_T0)
    original = (await memory.list_semantic_revisions(tenant_id=tenant))[0]
    mysql = {
        **fact,
        "object_value": "mysql",
        "correction": {
            "replaces_revision_key": original.revision_key,
            "effective_at": _T1.isoformat(),
        },
    }
    await _observe(memory, "mysql", "Service database mysql.", _T1, semantic_facts=[mysql])
    await memory.consolidate_semantics(tenant_id=tenant, as_of=_T1)
    middle = next(
        revision
        for revision in await memory.list_semantic_revisions(tenant_id=tenant)
        if revision.revision_key != original.revision_key
    )
    returned = {
        **fact,
        "correction": {
            "replaces_revision_key": middle.revision_key,
            "effective_at": _T2.isoformat(),
        },
    }
    await _observe(
        memory,
        "return",
        "Service database postgresql again.",
        _T2,
        session_id="shared",
        semantic_facts=[returned],
    )
    await memory.consolidate_semantics(tenant_id=tenant, as_of=_T2)
    revisions = await memory.list_semantic_revisions(tenant_id=tenant)
    assert (
        len([revision for revision in revisions if revision.memory_key == original.memory_key]) == 2
    )
    current = next(
        item
        for item in await memory.list_semantic_memories(tenant_id=tenant)
        if item.memory_key == original.memory_key
    )
    current_revision = next(r for r in revisions if r.revision_key == current.revision_key)
    assert set(current.derivations) == set(current_revision.derivations)
    assert len(current.derivations) == len(set(current.derivations))
    as_of = _T2 + timedelta(hours=12)
    later = _T2 + timedelta(days=1)
    before = {
        valid_at: await memory.inspect_recall(
            "database postgresql", tenant_id=tenant, as_of=as_of, valid_at=valid_at, limit=20
        )
        for valid_at in (None, _T1, as_of)
    }
    slot = SemanticSlotQuery("service", "database")
    before_slot = await memory.inspect_semantic_slot(slot, tenant_id=tenant, as_of=as_of)
    await _observe(
        memory,
        "future",
        "Service database postgresql future.",
        later,
        session_id="shared",
        semantic_facts=[fact],
    )
    await memory.consolidate_semantics(tenant_id=tenant, as_of=later)
    assert await memory.inspect_semantic_slot(slot, tenant_id=tenant, as_of=as_of) == before_slot
    for valid_at, expected in before.items():
        assert (
            await memory.inspect_recall(
                "database postgresql", tenant_id=tenant, as_of=as_of, valid_at=valid_at, limit=20
            )
            == expected
        )


async def test_mixed_multiple_revisions_memory_and_sqlite(tmp_path: Path) -> None:
    from cogkura import ComplementaryLearningSemanticConsolidator, SemanticCorrectionConfig

    kwargs = {
        "activation_config": ActivationConfig(retrieval_threshold=-10),
        "semantic_consolidator": ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
        "semantic_correction_config": SemanticCorrectionConfig(enabled=True),
    }
    await _exercise_mixed_multiple_revisions(Memory(**kwargs))
    async with await EmbeddedMemory.open(tmp_path / "multiple-revisions.db", **kwargs) as memory:
        await _exercise_mixed_multiple_revisions(memory)
