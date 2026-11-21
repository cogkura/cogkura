"""Historical activation must not score evidence from after its cognitive clock."""

from __future__ import annotations

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


async def test_episode_history_fails_closed_when_writes_are_backdated() -> None:
    memory = Memory(activation_config=ActivationConfig(retrieval_threshold=-10))
    await _observe(memory, "first", "migration known", _T0, session_id="shared")
    await _observe(memory, "second", "migration future secret", _T2, session_id="shared")
    await _observe(memory, "late", "migration arrived later", _T1, session_id="shared")
    assert await memory.recall("migration", tenant_id="historical-candidate", as_of=_T1) == []


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
    finally:
        await memory.clear(tenant_id="historical-candidate")
        await engine.dispose()


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


async def test_mixed_time_semantic_support_is_omitted_from_historical_recall() -> None:
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
        # With explicit factual time the store can project an earlier semantic
        # revision; without it the mixed current aggregate must be excluded.
        assert after.considered_count == (1 if valid_at is None else 2)
