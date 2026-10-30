"""PostgreSQL integration tests."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from cogkura.memory import Memory
from cogkura.migrations import apply_migrations
from cogkura.models import MemoryKind, SourceEvidenceRelation, SourceEvidenceResolutionStatus
from cogkura.observations.models import (
    IngestStatus,
    ObservationInput,
    ObservationRevisionIdentity,
)
from cogkura.sources.postgres import PostgresTableSource
from cogkura.storage.postgres import (
    PostgresCheckpointStore,
    PostgresEpisodeStore,
    PostgresObservationStore,
)

pytestmark = pytest.mark.postgres

TENANT = "company_123"
USER_ID = "11111111-1111-1111-1111-111111111111"
CONVERSATION_ID = "22222222-2222-2222-2222-222222222222"


class MessageMapper:
    def __init__(self, tenant_id: str) -> None:
        self.tenant_id = tenant_id

    def map(self, row: Mapping[str, Any]) -> ObservationInput:
        return ObservationInput(
            tenant_id=self.tenant_id,
            subject_id=str(row["user_id"]),
            actor_id=str(row["user_id"]),
            source_type="postgres",
            source_namespace="public.messages",
            source_record_id=str(row["id"]),
            source_version=row["updated_at"].isoformat(),
            event_type="message",
            content=row["body"],
            source_created_at=row["created_at"],
            source_updated_at=row["updated_at"],
            observed_at=row["updated_at"],
            metadata={
                "conversation_id": str(row["conversation_id"]),
                "sender_type": row["sender_type"],
            },
            is_deleted=row.get("deleted_at") is not None,
        )


class FailingMapper:
    def __init__(self, tenant_id: str, fail_on_id: str) -> None:
        self.tenant_id = tenant_id
        self.fail_on_id = fail_on_id

    def map(self, row: Mapping[str, Any]) -> ObservationInput:
        if str(row["id"]) == self.fail_on_id:
            raise RuntimeError("mapping failed")
        return MessageMapper(self.tenant_id).map(row)


@pytest.fixture
async def memory_engine(postgres_memory_url: str | None) -> AsyncIterator[AsyncEngine]:
    if postgres_memory_url is None:
        pytest.skip("COGKURA_POSTGRES_MEMORY_URL is not set")
    engine = create_async_engine(postgres_memory_url)
    await apply_migrations(engine)
    yield engine
    await engine.dispose()


@pytest.fixture
async def source_engine(postgres_source_url: str | None) -> AsyncIterator[AsyncEngine]:
    if postgres_source_url is None:
        pytest.skip("COGKURA_POSTGRES_SOURCE_URL is not set")
    engine = create_async_engine(postgres_source_url)
    yield engine
    await engine.dispose()


@pytest.fixture
async def source_admin_engine() -> AsyncIterator[AsyncEngine]:
    import os

    url = os.environ.get(
        "COGKURA_POSTGRES_SOURCE_ADMIN_URL",
        "postgresql+asyncpg://postgres:postgres@localhost:5432/cogkura_source",
    )
    engine = create_async_engine(url)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception:
        await engine.dispose()
        pytest.skip("PostgreSQL admin source URL is unavailable")
    yield engine
    await engine.dispose()


@pytest.fixture
async def same_db_engine() -> AsyncIterator[AsyncEngine]:
    """Same database with both public source tables and cogkura schema."""
    import os

    url = os.environ.get(
        "COGKURA_POSTGRES_SAME_DB_URL",
        "postgresql+asyncpg://postgres:postgres@localhost:5432/cogkura_source",
    )
    engine = create_async_engine(url)
    try:
        await apply_migrations(engine)
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1 FROM public.messages LIMIT 1"))
    except Exception:
        await engine.dispose()
        pytest.skip("Same-DB PostgreSQL URL is unavailable")
    yield engine
    await engine.dispose()


async def _insert_message(
    engine: AsyncEngine,
    *,
    message_id: str,
    body: str,
    updated_at: datetime,
    deleted_at: datetime | None = None,
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                """
                INSERT INTO public.messages (
                    id, conversation_id, user_id, sender_type, body,
                    created_at, updated_at, deleted_at
                ) VALUES (
                    :id, :conversation_id, :user_id, 'user', :body,
                    :updated_at, :updated_at, :deleted_at
                )
                ON CONFLICT (id) DO UPDATE SET
                    body = EXCLUDED.body,
                    updated_at = EXCLUDED.updated_at,
                    deleted_at = EXCLUDED.deleted_at
                """
            ),
            {
                "id": message_id,
                "conversation_id": CONVERSATION_ID,
                "user_id": USER_ID,
                "body": body,
                "updated_at": updated_at,
                "deleted_at": deleted_at,
            },
        )


@pytest.mark.asyncio
async def test_postgres_observation_insert_and_idempotency(memory_engine: AsyncEngine) -> None:
    store = PostgresObservationStore(memory_engine)
    record_id = f"integration-{uuid4().hex[:8]}"
    obs = ObservationInput(
        tenant_id=TENANT,
        subject_id="user_george",
        source_namespace="public.messages",
        source_record_id=record_id,
        source_version="v1",
        event_type="message",
        content="George prefers PostgreSQL for production services.",
        observed_at=datetime.now(UTC),
    )
    memory = Memory(observation_store=store)
    status = await memory.observe(obs)
    assert status is IngestStatus.CREATED
    status = await memory.observe(obs)
    assert status is IngestStatus.UNCHANGED


@pytest.mark.asyncio
async def test_postgres_source_ingest(
    memory_engine: AsyncEngine,
    source_engine: AsyncEngine,
) -> None:
    observation_store = PostgresObservationStore(memory_engine)
    checkpoint_store = PostgresCheckpointStore(memory_engine)
    memory = Memory(
        observation_store=observation_store,
        checkpoint_store=checkpoint_store,
    )
    source = PostgresTableSource(
        connector_id="integration-messages",
        engine=source_engine,
        table="public.messages",
        columns=(
            "id",
            "conversation_id",
            "user_id",
            "sender_type",
            "body",
            "created_at",
            "updated_at",
        ),
        soft_delete_column="deleted_at",
        cursor_columns=("updated_at", "id"),
        batch_size=100,
    )
    assert "deleted_at" in source.selected_columns
    result = await memory.ingest(
        source=source,
        mapper=MessageMapper(TENANT),
        tenant_id=TENANT,
    )
    assert result.discovered > 0
    assert result.created + result.unchanged + result.updated + result.deleted > 0


@pytest.mark.asyncio
async def test_failed_batch_does_not_advance_checkpoint(
    memory_engine: AsyncEngine,
    source_engine: AsyncEngine,
) -> None:
    observation_store = PostgresObservationStore(memory_engine)
    checkpoint_store = PostgresCheckpointStore(memory_engine)
    memory = Memory(
        observation_store=observation_store,
        checkpoint_store=checkpoint_store,
    )
    source = PostgresTableSource(
        connector_id="integration-fail-batch",
        engine=source_engine,
        table="public.messages",
        soft_delete_column="deleted_at",
        cursor_columns=("updated_at", "id"),
        batch_size=2,
    )
    mapper = FailingMapper(TENANT, fail_on_id="00000000-0000-0000-0000-000000000002")
    await memory.ingest(source=source, mapper=mapper, tenant_id=TENANT, batch_size=2)
    checkpoint = await checkpoint_store.get(
        tenant_id=TENANT,
        connector_id="integration-fail-batch",
    )
    assert checkpoint is None or checkpoint.get("id") != "00000000-0000-0000-0000-000000000002"


@pytest.mark.asyncio
async def test_tenant_isolation(memory_engine: AsyncEngine) -> None:
    store = PostgresObservationStore(memory_engine)
    obs = ObservationInput(
        tenant_id="tenant_a",
        source_namespace="public.messages",
        source_record_id="tenant-test-1",
        content="Tenant A observation content.",
        observed_at=datetime.now(UTC),
    )
    memory = Memory(observation_store=store)
    await memory.observe(obs)
    other = await store.get_by_source(
        tenant_id="tenant_b",
        source_namespace="public.messages",
        source_record_id="tenant-test-1",
    )
    assert other is None


@pytest.mark.asyncio
async def test_identical_timestamps_compound_cursor(
    memory_engine: AsyncEngine,
    source_engine: AsyncEngine,
    source_admin_engine: AsyncEngine,
) -> None:
    ts = datetime(2026, 8, 4, 15, 0, 0, tzinfo=UTC)
    first_id = str(uuid4())
    second_id = str(uuid4())
    # Lexicographic order matters for the id tie-breaker.
    ids = sorted([first_id, second_id])
    await _insert_message(
        source_admin_engine,
        message_id=ids[0],
        body="Identical timestamp message A for compound cursor.",
        updated_at=ts,
    )
    await _insert_message(
        source_admin_engine,
        message_id=ids[1],
        body="Identical timestamp message B for compound cursor.",
        updated_at=ts,
    )

    observation_store = PostgresObservationStore(memory_engine)
    checkpoint_store = PostgresCheckpointStore(memory_engine)
    memory = Memory(
        observation_store=observation_store,
        checkpoint_store=checkpoint_store,
    )
    connector_id = f"identical-ts-{uuid4().hex[:8]}"
    source = PostgresTableSource(
        connector_id=connector_id,
        engine=source_engine,
        table="public.messages",
        soft_delete_column="deleted_at",
        cursor_columns=("updated_at", "id"),
        batch_size=1,
    )
    result = await memory.ingest(
        source=source,
        mapper=MessageMapper(TENANT),
        tenant_id=TENANT,
        batch_size=1,
    )
    stored_a = await observation_store.get_by_source(
        tenant_id=TENANT,
        source_namespace="public.messages",
        source_record_id=ids[0],
    )
    stored_b = await observation_store.get_by_source(
        tenant_id=TENANT,
        source_namespace="public.messages",
        source_record_id=ids[1],
    )
    assert stored_a is not None
    assert stored_b is not None
    assert result.created >= 2


@pytest.mark.asyncio
async def test_soft_delete_restore_and_revision_history(
    memory_engine: AsyncEngine,
) -> None:
    store = PostgresObservationStore(memory_engine)
    memory = Memory(observation_store=store)
    record_id = f"rev-{uuid4().hex[:8]}"
    base = ObservationInput(
        tenant_id=TENANT,
        subject_id="user_george",
        source_namespace="public.messages",
        source_record_id=record_id,
        source_version="v1",
        event_type="message",
        content="Release planned for Thursday originally stated.",
        observed_at=datetime(2026, 8, 4, 16, 0, tzinfo=UTC),
    )
    assert await memory.observe(base) is IngestStatus.CREATED

    updated = ObservationInput(
        tenant_id=TENANT,
        subject_id="user_george",
        source_namespace="public.messages",
        source_record_id=record_id,
        source_version="v2",
        event_type="message",
        content="Release planned for Friday after correction.",
        observed_at=datetime(2026, 8, 4, 16, 1, tzinfo=UTC),
    )
    assert await memory.observe(updated) is IngestStatus.UPDATED

    deleted = ObservationInput(
        tenant_id=TENANT,
        subject_id="user_george",
        source_namespace="public.messages",
        source_record_id=record_id,
        source_version="v3",
        event_type="message",
        content="Release planned for Friday after correction.",
        observed_at=datetime(2026, 8, 4, 16, 2, tzinfo=UTC),
        is_deleted=True,
    )
    assert await memory.observe(deleted) is IngestStatus.DELETED

    restored = ObservationInput(
        tenant_id=TENANT,
        subject_id="user_george",
        source_namespace="public.messages",
        source_record_id=record_id,
        source_version="v4",
        event_type="message",
        content="Release planned for Friday after correction.",
        observed_at=datetime(2026, 8, 4, 16, 3, tzinfo=UTC),
        is_deleted=False,
    )
    assert await memory.observe(restored) is IngestStatus.RESTORED

    stored = await store.get_by_source(
        tenant_id=TENANT,
        source_namespace="public.messages",
        source_record_id=record_id,
    )
    assert stored is not None
    assert stored.current_revision == 4
    assert stored.is_deleted is False

    async with memory_engine.connect() as conn:
        result = await conn.execute(
            text(
                """
                SELECT change_type, revision_number
                FROM cogkura.observation_revisions
                WHERE observation_id = :observation_id
                ORDER BY revision_number
                """
            ),
            {"observation_id": stored.id},
        )
        changes = [(row[0], row[1]) for row in result]
    assert changes == [
        ("created", 1),
        ("updated", 2),
        ("deleted", 3),
        ("restored", 4),
    ]


@pytest.mark.asyncio
async def test_same_database_schema_mode(same_db_engine: AsyncEngine) -> None:
    observation_store = PostgresObservationStore(same_db_engine)
    checkpoint_store = PostgresCheckpointStore(same_db_engine)
    memory = Memory(
        observation_store=observation_store,
        checkpoint_store=checkpoint_store,
    )
    source = PostgresTableSource(
        connector_id=f"same-db-{uuid4().hex[:8]}",
        engine=same_db_engine,
        table="public.messages",
        soft_delete_column="deleted_at",
        cursor_columns=("updated_at", "id"),
        batch_size=50,
    )
    result = await memory.ingest(
        source=source,
        mapper=MessageMapper(TENANT),
        tenant_id=TENANT,
    )
    assert result.discovered > 0
    assert result.created + result.unchanged + result.updated + result.deleted > 0


@pytest.mark.asyncio
async def test_read_only_source_role(source_engine: AsyncEngine) -> None:
    async with source_engine.connect() as conn:
        with pytest.raises(Exception):  # noqa: B017
            await conn.execute(
                text(
                    """
                    INSERT INTO public.messages (
                        id, conversation_id, user_id, sender_type, body
                    ) VALUES (
                        :id, :conversation_id, :user_id, 'user', 'should fail'
                    )
                    """
                ),
                {
                    "id": str(uuid4()),
                    "conversation_id": CONVERSATION_ID,
                    "user_id": USER_ID,
                },
            )
            await conn.commit()


@pytest.mark.asyncio
async def test_postgres_encode_episodes(memory_engine: AsyncEngine) -> None:
    observation_store = PostgresObservationStore(memory_engine)
    episode_store = PostgresEpisodeStore(memory_engine)
    memory = Memory(
        observation_store=observation_store,
        checkpoint_store=PostgresCheckpointStore(memory_engine),
        episode_store=episode_store,
    )
    await memory.observe(
        ObservationInput(
            tenant_id=TENANT,
            subject_id=USER_ID,
            actor_id=USER_ID,
            source_namespace="direct",
            source_record_id=str(uuid4()),
            event_type="message",
            content="PostgreSQL episodic encoding integration test message one.",
            observed_at=datetime.now(UTC),
            metadata={"conversation_id": CONVERSATION_ID},
        )
    )
    await memory.observe(
        ObservationInput(
            tenant_id=TENANT,
            subject_id=USER_ID,
            actor_id=USER_ID,
            source_namespace="direct",
            source_record_id=str(uuid4()),
            event_type="message",
            content="PostgreSQL episodic encoding integration test message two.",
            observed_at=datetime.now(UTC),
            metadata={"conversation_id": CONVERSATION_ID, "terminal_event": True},
        )
    )
    result = await memory.encode_episodes(tenant_id=TENANT, subject_id=USER_ID)
    episodes = await memory.list_episodes(tenant_id=TENANT, subject_id=USER_ID)
    assert result.created == 1
    assert len(episodes) == 1
    assert len(episodes[0].evidence) == 2


@pytest.mark.asyncio
async def test_postgres_provenance_resolves_historical_revision_and_tenant_scope(
    memory_engine: AsyncEngine,
) -> None:
    """Provenance reads exact PostgreSQL revision rows and remains tenant-scoped."""
    observation_store = PostgresObservationStore(memory_engine)
    memory = Memory(
        observation_store=observation_store,
        episode_store=PostgresEpisodeStore(memory_engine),
    )
    subject_id = f"provenance-revision-{uuid4().hex}"
    record_id = f"adr-{uuid4().hex}"

    await memory.observe(
        ObservationInput(
            tenant_id=TENANT,
            subject_id=subject_id,
            source_namespace="repository",
            source_record_id=record_id,
            source_version="1",
            event_type="document",
            content="Migrations run during deployment before traffic.",
            observed_at=datetime(2026, 10, 1, tzinfo=UTC),
            metadata={"source_ref": "docs/adr/original.md"},
        )
    )
    await memory.encode_episodes(
        tenant_id=TENANT,
        subject_id=subject_id,
        as_of=datetime(2026, 10, 2, tzinfo=UTC),
    )
    context = await memory.prepare_context(
        "deployment migrations",
        tenant_id=TENANT,
        subject_id=subject_id,
        as_of=datetime(2026, 10, 3, tzinfo=UTC),
    )

    await memory.observe(
        ObservationInput(
            tenant_id=TENANT,
            subject_id=subject_id,
            source_namespace="repository",
            source_record_id=record_id,
            source_version="2",
            event_type="document",
            content="Migrations run during deployment after verification.",
            observed_at=datetime(2026, 10, 4, tzinfo=UTC),
            metadata={"source_ref": "docs/adr/revised.md"},
        )
    )

    source = (await memory.resolve_provenance(context)).items[0].sources[0]
    assert source.observation_revision == 1
    assert source.source_version == "1"
    assert source.source_ref == "docs/adr/original.md"
    assert source.is_latest_known_revision is False
    assert source.resolution_status is SourceEvidenceResolutionStatus.RESOLVED

    stored = await observation_store.get_by_source(
        tenant_id=TENANT,
        source_namespace="repository",
        source_record_id=record_id,
    )
    assert stored is not None
    reference = ObservationRevisionIdentity(stored.id, 1)
    assert (
        await observation_store.get_revisions_many(
            tenant_id=f"other-{uuid4().hex}",
            references=(reference,),
        )
        == {}
    )


@pytest.mark.asyncio
async def test_postgres_provenance_resolves_semantic_support_evidence(
    memory_engine: AsyncEngine,
) -> None:
    """Semantic provenance obtains SUPPORT evidence through PostgreSQL episodes."""
    memory = Memory(
        observation_store=PostgresObservationStore(memory_engine),
        episode_store=PostgresEpisodeStore(memory_engine),
    )
    subject_id = f"provenance-semantic-{uuid4().hex}"
    semantic_fact = {
        "predicate": "deployment_policy",
        "object_value": "migrations run",
        "cardinality": "one",
        "polarity": "affirm",
    }
    for record_id, observed_at in (
        (f"adr-{uuid4().hex}", datetime(2026, 10, 1, tzinfo=UTC)),
        (f"adr-{uuid4().hex}", datetime(2026, 10, 2, tzinfo=UTC)),
    ):
        await memory.observe(
            ObservationInput(
                tenant_id=TENANT,
                subject_id=subject_id,
                source_namespace="repository",
                source_record_id=record_id,
                source_version="1",
                event_type="document",
                content="Migrations run during deployment before traffic.",
                observed_at=observed_at,
                metadata={
                    "source_ref": f"docs/adr/{record_id}.md",
                    "semantic_facts": semantic_fact,
                },
            )
        )

    await memory.process(
        tenant_id=TENANT,
        subject_id=subject_id,
        as_of=datetime(2026, 10, 3, tzinfo=UTC),
    )
    context = await memory.prepare_context(
        "deployment policy",
        tenant_id=TENANT,
        subject_id=subject_id,
        as_of=datetime(2026, 10, 4, tzinfo=UTC),
    )

    provenance = await memory.resolve_provenance(context)
    semantic = next(
        item for item in provenance.items if item.memory_identity.memory_kind is MemoryKind.SEMANTIC
    )
    assert semantic.distinct_source_count == 2
    assert {source.relation for source in semantic.sources} == {
        SourceEvidenceRelation.SEMANTIC_SUPPORT
    }
    assert all(
        source.resolution_status is SourceEvidenceResolutionStatus.RESOLVED
        for source in semantic.sources
    )
