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
    inspection = await memory.inspect_recall(
        "database PostgreSQL", tenant_id="historical-candidate", as_of=_T1
    )
    assert inspection.considered_count == 1


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
    finally:
        await memory.clear(tenant_id="historical-candidate")
        await engine.dispose()
