"""PostgreSQL observation ingestion and episodic encoding demo."""

from __future__ import annotations

import asyncio
import os

from sqlalchemy.ext.asyncio import create_async_engine

from cogkura import Memory
from cogkura.sources.postgres import PostgresTableSource
from cogkura.storage.postgres import (
    PostgresActivationStore,
    PostgresCheckpointStore,
    PostgresEntityRelationshipStore,
    PostgresEpisodeStore,
    PostgresObservationStore,
    PostgresSemanticMemoryStore,
)

SOURCE_URL = os.environ.get(
    "COGKURA_POSTGRES_SOURCE_URL",
    "postgresql+asyncpg://cogkura_reader:cogkura_reader@localhost:5432/cogkura_source",
)
MEMORY_URL = os.environ.get(
    "COGKURA_POSTGRES_MEMORY_URL",
    "postgresql+asyncpg://cogkura_writer:cogkura_writer@localhost:5432/cogkura_memory",
)
TENANT_ID = "company_123"


class MessageMapper:
    def __init__(self, tenant_id: str) -> None:
        self.tenant_id = tenant_id

    def map(self, row: dict[str, object]) -> object:
        from datetime import UTC

        from cogkura import ObservationInput

        updated_at = row["updated_at"]
        if getattr(updated_at, "tzinfo", None) is None:
            updated_at = updated_at.replace(tzinfo=UTC)  # type: ignore[union-attr]
        return ObservationInput(
            tenant_id=self.tenant_id,
            subject_id=str(row["user_id"]),
            actor_id=str(row["user_id"]),
            source_type="postgres",
            source_namespace="public.messages",
            source_record_id=str(row["id"]),
            source_version=updated_at.isoformat(),  # type: ignore[union-attr]
            event_type="message",
            content=str(row["body"]),
            source_created_at=row["created_at"],  # type: ignore[arg-type]
            source_updated_at=updated_at,  # type: ignore[arg-type]
            observed_at=updated_at,  # type: ignore[arg-type]
            metadata={
                "conversation_id": str(row["conversation_id"]),
                "sender_type": row["sender_type"],
            },
            is_deleted=row["deleted_at"] is not None,
        )


async def main() -> None:
    source_engine = create_async_engine(SOURCE_URL)
    memory_engine = create_async_engine(MEMORY_URL)
    observation_store = PostgresObservationStore(memory_engine)
    checkpoint_store = PostgresCheckpointStore(memory_engine)
    episode_store = PostgresEpisodeStore(memory_engine)
    semantic_store = PostgresSemanticMemoryStore(memory_engine)
    activation_store = PostgresActivationStore(memory_engine)
    entity_relationship_store = PostgresEntityRelationshipStore(memory_engine)
    memory = Memory(
        observation_store=observation_store,
        checkpoint_store=checkpoint_store,
        episode_store=episode_store,
        semantic_store=semantic_store,
        activation_store=activation_store,
        entity_relationship_store=entity_relationship_store,
    )
    source = PostgresTableSource(
        connector_id="application-messages",
        engine=source_engine,
        table="public.messages",
        cursor_columns=("updated_at", "id"),
        soft_delete_column="deleted_at",
        batch_size=500,
    )
    result = await memory.ingest(
        source=source,
        mapper=MessageMapper(TENANT_ID),
        tenant_id=TENANT_ID,
    )
    print(result)

    encoding = await memory.encode_episodes(tenant_id=TENANT_ID)
    episodes = await memory.list_episodes(tenant_id=TENANT_ID)
    print(encoding)
    print(f"episodes={len(episodes)}")

    consolidation = await memory.consolidate_semantics(tenant_id=TENANT_ID)
    semantic_memories = await memory.list_semantic_memories(tenant_id=TENANT_ID)
    print(consolidation)
    print(f"semantic_memories={len(semantic_memories)}")

    recall = await memory.recall("PostgreSQL operational complexity", tenant_id=TENANT_ID)
    print(f"recall_results={len(recall)}")
    if recall:
        print(recall[0].memory_kind, recall[0].score, recall[0].reason)

    await source_engine.dispose()
    await memory_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
