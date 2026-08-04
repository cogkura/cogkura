"""PostgreSQL observation ingestion and episodic encoding demo."""

from __future__ import annotations

import asyncio
import os

from sqlalchemy.ext.asyncio import create_async_engine

from cognema import Memory
from cognema.sources.postgres import PostgresTableSource
from cognema.storage.postgres import (
    PostgresCheckpointStore,
    PostgresEpisodeStore,
    PostgresObservationStore,
)

SOURCE_URL = os.environ.get(
    "COGNEMA_POSTGRES_SOURCE_URL",
    "postgresql+asyncpg://cognema_reader:cognema_reader@localhost:5432/cognema_source",
)
MEMORY_URL = os.environ.get(
    "COGNEMA_POSTGRES_MEMORY_URL",
    "postgresql+asyncpg://cognema_writer:cognema_writer@localhost:5432/cognema_memory",
)
TENANT_ID = "company_123"


class MessageMapper:
    def __init__(self, tenant_id: str) -> None:
        self.tenant_id = tenant_id

    def map(self, row: dict[str, object]) -> object:
        from datetime import UTC

        from cognema import ObservationInput

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
    memory = Memory(
        observation_store=observation_store,
        checkpoint_store=checkpoint_store,
        episode_store=episode_store,
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

    await source_engine.dispose()
    await memory_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
