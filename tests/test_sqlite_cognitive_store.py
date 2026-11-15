"""SQLite cognitive storage parity and restart checks."""

import importlib
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from cogkura.algorithms.learning import DeterministicLearningProcessor
from cogkura.exceptions import StorageError
from cogkura.models import (
    ActivationReferenceKind,
    LearningConfig,
    LearningFeedback,
    LearningOutcome,
    MemoryFeedback,
    MemoryIdentity,
    MemoryKind,
    MemoryReference,
    MemoryRetentionState,
    StoredMemoryDynamics,
)
from cogkura.storage.sqlite import SQLiteStorage
from cogkura.storage.sqlite_cognitive import (
    SQLiteActivationStore,
    SQLiteInhibitionStore,
    SQLiteLearningStore,
    SQLiteMemoryDynamicsStore,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module", "name", "store_name", "store_type"),
    [
        (
            "test_postgres_activation_store",
            "test_postgres_activation_append_and_list",
            "PostgresActivationStore",
            SQLiteActivationStore,
        ),
        (
            "test_postgres_dynamics_store",
            "test_postgres_dynamics_upsert_get_and_reactivate",
            "PostgresMemoryDynamicsStore",
            SQLiteMemoryDynamicsStore,
        ),
        (
            "test_postgres_dynamics_store",
            "test_postgres_activation_compaction",
            "PostgresActivationStore",
            SQLiteActivationStore,
        ),
        (
            "test_postgres_learning_store",
            "test_postgres_learning_store_matches_in_memory",
            "PostgresLearningStore",
            SQLiteLearningStore,
        ),
        (
            "test_postgres_inhibition_store",
            "test_postgres_inhibition_round_trip_and_request_idempotency",
            "PostgresInhibitionStore",
            SQLiteInhibitionStore,
        ),
        (
            "test_postgres_inhibition_store",
            "test_postgres_repeated_use_without_request_id",
            "PostgresInhibitionStore",
            SQLiteInhibitionStore,
        ),
        (
            "test_postgres_inhibition_store",
            "test_postgres_as_of_order_and_scope_round_trip",
            "PostgresInhibitionStore",
            SQLiteInhibitionStore,
        ),
        (
            "test_postgres_inhibition_store",
            "test_postgres_list_for_memories_matches_bounds",
            "PostgresInhibitionStore",
            SQLiteInhibitionStore,
        ),
    ],
)
async def test_existing_storage_contracts(
    module: str,
    name: str,
    store_name: str,
    store_type: type,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("sqlalchemy")
    contract = importlib.import_module(module)
    storage = await SQLiteStorage.open(tmp_path / "db")
    monkeypatch.setattr(contract, store_name, lambda _engine: store_type(storage))
    try:
        await getattr(contract, name)(None)
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_state_reopen_and_learning_rollback(tmp_path: Path) -> None:
    path = tmp_path / "db"
    storage = await SQLiteStorage.open(path)
    activation = SQLiteActivationStore(storage)
    dynamics = SQLiteMemoryDynamicsStore(storage)
    learning = SQLiteLearningStore(storage)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    identity = MemoryIdentity(MemoryKind.EPISODE, "episode")
    ref = MemoryReference(
        tenant_id="a",
        memory_kind=identity.memory_kind,
        memory_key=identity.memory_key,
        reference_kind=ActivationReferenceKind.RETRIEVED,
        referenced_at=now,
        request_id="request",
    )
    await activation.append_references([ref, ref])
    state = StoredMemoryDynamics(
        tenant_id="a",
        memory_kind=identity.memory_kind,
        memory_key=identity.memory_key,
        retention_state=MemoryRetentionState.FORGOTTEN,
        last_base_level=-2,
        last_retention_score=0.1,
        below_threshold_since=now,
        forgotten_at=now,
        evaluated_at=now,
        updated_at=now,
    )
    await dynamics.upsert_many([state])
    other_identity = MemoryIdentity(MemoryKind.SEMANTIC, "semantic")
    feedback = LearningFeedback(
        tenant_id="a",
        feedback_id="feedback",
        occurred_at=now,
        items=(
            MemoryFeedback(identity=identity, outcome=LearningOutcome.HELPFUL),
            MemoryFeedback(identity=other_identity, outcome=LearningOutcome.HELPFUL),
        ),
    )
    plan = DeterministicLearningProcessor().plan(feedback=feedback, config=LearningConfig())
    await storage._run(
        lambda: (
            storage._conn().execute("""CREATE TRIGGER fail_learning BEFORE INSERT ON learning_state
        BEGIN SELECT RAISE(ABORT, 'injected'); END""")
        )
    )
    with pytest.raises(StorageError, match="injected"):
        await learning.apply(plan)
    await storage._run(lambda: storage._conn().execute("DROP TRIGGER fail_learning"))
    assert (await learning.apply(plan)).created
    assert (await learning.apply(plan)).unchanged
    conflict = replace(plan, feedback_fingerprint="different")
    with pytest.raises(StorageError, match="Conflicting"):
        await learning.apply(conflict)
    await storage.close()
    storage = await SQLiteStorage.open(path)
    activation = SQLiteActivationStore(storage)
    dynamics = SQLiteMemoryDynamicsStore(storage)
    learning = SQLiteLearningStore(storage)
    assert (
        len(
            (
                await activation.list_reference_traces(
                    tenant_id="a", identities=[identity], before_or_at=now
                )
            )[identity]
        )
        == 1
    )
    assert (await dynamics.get_many(tenant_id="a", identities=[identity]))[identity] == state
    associations = await learning.list_associations(
        tenant_id="a", identities=[identity, other_identity]
    )
    assert len(associations) == 1 and associations[0].coactivation_count == 1
    assert (
        await learning.list_states(tenant_id="a", identities=[identity], context_keys=["global"])
    )[0].helpful_count == 1
    assert (
        len(
            (
                await learning.list_reinforcement_traces(
                    tenant_id="a", identities=[identity], before_or_at=now
                )
            )[identity]
        )
        == 1
    )
    assert not await learning.list_reinforcement_traces(
        tenant_id="a", identities=[identity], before_or_at=now - timedelta(microseconds=1)
    )
    await dynamics.reactivate(tenant_id="a", identities=[identity], at=now + timedelta(seconds=1))
    assert (await dynamics.get_many(tenant_id="a", identities=[identity]))[
        identity
    ].retention_state is MemoryRetentionState.ACTIVE
    await storage.close()


@pytest.mark.asyncio
async def test_recall_and_cognitive_state_survive_restart(tmp_path: Path) -> None:
    from cogkura import ActivationConfig, Memory, ObservationInput
    from cogkura.storage.sqlite import SQLiteCheckpointStore, SQLiteObservationStore
    from cogkura.storage.sqlite_cognitive import SQLiteEntityRelationshipStore
    from cogkura.storage.sqlite_episode import SQLiteEpisodeStore
    from cogkura.storage.sqlite_semantic import SQLiteSemanticMemoryStore

    path = tmp_path / "cognitive.db"
    now = datetime(2026, 1, 1, tzinfo=UTC)

    def memory_for(storage: SQLiteStorage) -> Memory:
        return Memory(
            observation_store=SQLiteObservationStore(storage),
            checkpoint_store=SQLiteCheckpointStore(storage),
            episode_store=SQLiteEpisodeStore(storage),
            semantic_store=SQLiteSemanticMemoryStore(storage),
            activation_store=SQLiteActivationStore(storage),
            dynamics_store=SQLiteMemoryDynamicsStore(storage),
            learning_store=SQLiteLearningStore(storage),
            inhibition_store=SQLiteInhibitionStore(storage),
            entity_relationship_store=SQLiteEntityRelationshipStore(storage),
            activation_config=ActivationConfig(retrieval_threshold=-10),
        )

    storage = await SQLiteStorage.open(path)
    memory = memory_for(storage)
    await memory.observe(
        ObservationInput(
            tenant_id="a",
            source_namespace="test",
            source_record_id="1",
            content="Alice maintains the deployment database service.",
            observed_at=now,
            metadata={
                "relationships": [
                    {
                        "source_entity_id": "alice",
                        "relation_type": "maintains",
                        "target_entity_id": "database",
                    }
                ]
            },
        )
    )
    await memory.process(tenant_id="a", as_of=now)
    first = await memory.recall("deployment database", tenant_id="a", as_of=now + timedelta(days=1))
    assert first
    await memory.record_access(
        first, tenant_id="a", referenced_at=now + timedelta(days=1), request_id="r1"
    )
    identity = MemoryIdentity(first[0].memory_kind, first[0].memory.memory_key)
    feedback = LearningFeedback(
        tenant_id="a",
        feedback_id="f1",
        occurred_at=now + timedelta(days=1),
        items=(MemoryFeedback(identity=identity, outcome=LearningOutcome.HELPFUL),),
    )
    assert (await memory.learn(feedback)).created
    evaluated = now + timedelta(days=2)
    before = await memory.recall("deployment database", tenant_id="a", as_of=evaluated)
    learning = await memory.list_learning_state(tenant_id="a", identities=[identity])
    relationships = await memory.list_entity_relationships(tenant_id="a")
    assert len(relationships) == 1
    await storage.close()
    storage = await SQLiteStorage.open(path)
    memory = memory_for(storage)
    assert await memory.recall("deployment database", tenant_id="a", as_of=evaluated) == before
    assert await memory.list_learning_state(tenant_id="a", identities=[identity]) == learning
    assert await memory.list_entity_relationships(tenant_id="a") == relationships
    assert (await memory.learn(feedback)).unchanged
    await storage.close()


@pytest.mark.asyncio
async def test_concurrent_learning_and_tenant_isolation(tmp_path: Path) -> None:
    import asyncio

    path = tmp_path / "db"
    first, second = await asyncio.gather(SQLiteStorage.open(path), SQLiteStorage.open(path))
    identity = MemoryIdentity(MemoryKind.EPISODE, "same")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    feedback = LearningFeedback(
        tenant_id="a",
        feedback_id="same",
        occurred_at=now,
        items=(MemoryFeedback(identity=identity, outcome=LearningOutcome.HELPFUL),),
    )
    processor = DeterministicLearningProcessor()
    plan = processor.plan(feedback=feedback, config=LearningConfig())
    results = await asyncio.gather(
        SQLiteLearningStore(first).apply(plan), SQLiteLearningStore(second).apply(plan)
    )
    assert sum(item.created for item in results) == 1
    assert sum(item.unchanged for item in results) == 1
    other = processor.plan(feedback=replace(feedback, tenant_id="b"), config=LearningConfig())
    assert (await SQLiteLearningStore(second).apply(other)).created
    await SQLiteLearningStore(first).clear(tenant_id="a")
    assert not await SQLiteLearningStore(second).list_states(
        tenant_id="a", identities=[identity], context_keys=["global"]
    )
    assert (
        len(
            await SQLiteLearningStore(second).list_states(
                tenant_id="b", identities=[identity], context_keys=["global"]
            )
        )
        == 1
    )
    await first.close()
    await second.close()


@pytest.mark.asyncio
async def test_inhibition_survives_restart_with_scope_and_bounds(tmp_path: Path) -> None:
    path = tmp_path / "db"
    from cogkura.models import (
        BehavioralStructuralAnchor,
        CompetitionDirection,
        InhibitionScopeSignature,
        InhibitoryTrace,
    )

    now = datetime(2026, 1, 1, tzinfo=UTC)
    first = InhibitoryTrace(
        id="11111111-1111-1111-1111-111111111111",
        tenant_id="a",
        inhibited_identity=MemoryIdentity(MemoryKind.SEMANTIC, "loser"),
        selected_identity=MemoryIdentity(MemoryKind.SEMANTIC, "winner"),
        direction=CompetitionDirection.CO_TEMPORAL,
        scope=InhibitionScopeSignature(
            structural_anchor=BehavioralStructuralAnchor.SUBJECT_PREDICATE,
            subject_id="api",
            predicate="deploy",
        ),
        competition_strength=0.8,
        competitor_accessibility=0.7,
        induction_pressure=0.56,
        retrieval_evaluated_at=now,
        induced_at=now,
        request_id="same",
    )
    repeated = replace(first, id="22222222-2222-2222-2222-222222222222")
    storage = await SQLiteStorage.open(path)
    await SQLiteInhibitionStore(storage).append_traces([first, repeated])
    await storage.close()
    storage = await SQLiteStorage.open(path)
    traces = await SQLiteInhibitionStore(storage).list_for_memories(
        tenant_id=first.tenant_id,
        identities=[first.inhibited_identity],
        before_or_at=first.induced_at,
        limit_per_memory=1,
    )
    assert traces[first.inhibited_identity] == (first,)
    assert not await SQLiteInhibitionStore(storage).list_for_memories(
        tenant_id="other", identities=[first.inhibited_identity], before_or_at=first.induced_at
    )
    assert not await SQLiteInhibitionStore(storage).list_for_memories(
        tenant_id=first.tenant_id,
        identities=[first.inhibited_identity],
        after=first.induced_at,
        before_or_at=first.induced_at,
    )
    await storage.close()


@pytest.mark.asyncio
async def test_upgrade_from_populated_0191(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from cogkura import Memory, ObservationInput
    from cogkura.storage import sqlite as sqlite_module
    from cogkura.storage.sqlite import SQLiteObservationStore
    from cogkura.storage.sqlite_episode import SQLiteEpisodeStore
    from cogkura.storage.sqlite_semantic import SQLiteSemanticMemoryStore

    paths = sqlite_module._migrations()
    monkeypatch.setattr(sqlite_module, "_migrations", lambda: paths[:3])
    path = tmp_path / "old.db"
    storage = await SQLiteStorage.open(path)
    memory = Memory(
        observation_store=SQLiteObservationStore(storage),
        episode_store=SQLiteEpisodeStore(storage),
        semantic_store=SQLiteSemanticMemoryStore(storage),
    )
    when = datetime(2026, 1, 1, tzinfo=UTC)
    await memory.observe(
        ObservationInput(
            tenant_id="a",
            source_namespace="src",
            source_record_id="1",
            content="Alice maintains a durable database.",
            observed_at=when,
        )
    )
    await memory.process(tenant_id="a", as_of=when)
    episodes = await memory.list_episodes(tenant_id="a")
    assert episodes
    await storage.close()
    monkeypatch.setattr(sqlite_module, "_migrations", lambda: paths)
    storage = await SQLiteStorage.open(path)
    assert await SQLiteEpisodeStore(storage).list(tenant_id="a") == episodes
    assert (
        await SQLiteActivationStore(storage).list_reference_traces(
            tenant_id="a", identities=[], before_or_at=when
        )
        == {}
    )
    await storage.close()
