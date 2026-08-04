"""PostgreSQL observation and checkpoint storage."""

from __future__ import annotations

import json
from datetime import UTC
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from cognema.exceptions import StorageError
from cognema.observations.models import IngestStatus, ObservationInput, StoredObservation
from cognema.observations.retention import RetainedObservation
from cognema.storage.base import CheckpointStore, ObservationStore


class PostgresObservationStore(ObservationStore):
    """PostgreSQL-backed observation store."""

    def __init__(self, engine: AsyncEngine, *, schema: str = "cognema") -> None:
        self._engine = engine
        self._schema = schema

    def _table(self, name: str) -> str:
        return f"{self._schema}.{name}"

    async def ingest(
        self,
        observation: ObservationInput,
        *,
        retained: RetainedObservation,
    ) -> IngestStatus:
        existing = await self.get_by_source(
            tenant_id=observation.tenant_id,
            source_namespace=observation.source_namespace,
            source_record_id=observation.source_record_id,
        )
        if existing is None:
            return await self._create(observation, retained=retained)
        return await self._update(existing, observation, retained=retained)

    async def _create(
        self,
        observation: ObservationInput,
        *,
        retained: RetainedObservation,
    ) -> IngestStatus:
        obs_id = str(uuid4())
        revision_id = str(uuid4())
        observed_at = observation.observed_at.astimezone(UTC)
        metadata_json = json.dumps(retained.metadata)
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    INSERT INTO {self._table("observations")} (
                        id, tenant_id, subject_id, actor_id,
                        source_type, source_namespace, source_record_id, source_version,
                        event_type, content, content_hash, metadata,
                        source_created_at, source_updated_at,
                        first_observed_at, last_observed_at,
                        current_revision, is_deleted, created_at, updated_at
                    ) VALUES (
                        :id, :tenant_id, :subject_id, :actor_id,
                        :source_type, :source_namespace, :source_record_id, :source_version,
                        :event_type, :content, :content_hash, CAST(:metadata AS jsonb),
                        :source_created_at, :source_updated_at,
                        :first_observed_at, :last_observed_at,
                        1, :is_deleted, :now, :now
                    )
                    """
                ),
                self._observation_params(
                    obs_id,
                    observation,
                    retained,
                    observed_at=observed_at,
                    revision=1,
                ),
            )
            await conn.execute(
                text(
                    f"""
                    INSERT INTO {self._table("observation_revisions")} (
                        id, observation_id, revision_number, source_version,
                        content, content_hash, metadata, change_type, observed_at
                    ) VALUES (
                        :revision_id, :observation_id, 1, :source_version,
                        :content, :content_hash, CAST(:metadata AS jsonb),
                        'created', :observed_at
                    )
                    """
                ),
                {
                    "revision_id": revision_id,
                    "observation_id": obs_id,
                    "source_version": observation.source_version,
                    "content": retained.content,
                    "content_hash": retained.content_hash,
                    "metadata": metadata_json,
                    "observed_at": observed_at,
                },
            )
        return IngestStatus.CREATED

    async def _update(
        self,
        existing: StoredObservation,
        observation: ObservationInput,
        *,
        retained: RetainedObservation,
    ) -> IngestStatus:
        unchanged = (
            existing.source_version == observation.source_version
            and existing.content_hash == retained.content_hash
            and existing.is_deleted == observation.is_deleted
        )
        if unchanged:
            return IngestStatus.UNCHANGED

        if existing.is_deleted and not observation.is_deleted:
            change_type = "restored"
            status = IngestStatus.RESTORED
        elif observation.is_deleted and not existing.is_deleted:
            change_type = "deleted"
            status = IngestStatus.DELETED
        else:
            change_type = "updated"
            status = IngestStatus.UPDATED

        revision_number = existing.current_revision + 1
        revision_id = str(uuid4())
        observed_at = observation.observed_at.astimezone(UTC)
        metadata_json = json.dumps(retained.metadata)
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    UPDATE {self._table("observations")}
                    SET
                        subject_id = :subject_id,
                        actor_id = :actor_id,
                        source_type = :source_type,
                        source_version = :source_version,
                        event_type = :event_type,
                        content = :content,
                        content_hash = :content_hash,
                        metadata = CAST(:metadata AS jsonb),
                        source_created_at = :source_created_at,
                        source_updated_at = :source_updated_at,
                        last_observed_at = :last_observed_at,
                        current_revision = :current_revision,
                        is_deleted = :is_deleted,
                        updated_at = :now
                    WHERE id = :id
                    """
                ),
                {
                    "id": existing.id,
                    "subject_id": observation.subject_id,
                    "actor_id": observation.actor_id,
                    "source_type": observation.source_type,
                    "source_version": observation.source_version,
                    "event_type": observation.event_type,
                    "content": retained.content,
                    "content_hash": retained.content_hash,
                    "metadata": metadata_json,
                    "source_created_at": observation.source_created_at,
                    "source_updated_at": observation.source_updated_at,
                    "last_observed_at": observed_at,
                    "current_revision": revision_number,
                    "is_deleted": observation.is_deleted,
                    "now": observed_at,
                },
            )
            await conn.execute(
                text(
                    f"""
                    INSERT INTO {self._table("observation_revisions")} (
                        id, observation_id, revision_number, source_version,
                        content, content_hash, metadata, change_type, observed_at
                    ) VALUES (
                        :revision_id, :observation_id, :revision_number, :source_version,
                        :content, :content_hash, CAST(:metadata AS jsonb),
                        :change_type, :observed_at
                    )
                    """
                ),
                {
                    "revision_id": revision_id,
                    "observation_id": existing.id,
                    "revision_number": revision_number,
                    "source_version": observation.source_version,
                    "content": retained.content,
                    "content_hash": retained.content_hash,
                    "metadata": metadata_json,
                    "change_type": change_type,
                    "observed_at": observed_at,
                },
            )
        return status

    def _observation_params(
        self,
        obs_id: str,
        observation: ObservationInput,
        retained: RetainedObservation,
        *,
        observed_at: Any,
        revision: int,
    ) -> dict[str, Any]:
        return {
            "id": obs_id,
            "tenant_id": observation.tenant_id,
            "subject_id": observation.subject_id,
            "actor_id": observation.actor_id,
            "source_type": observation.source_type,
            "source_namespace": observation.source_namespace,
            "source_record_id": observation.source_record_id,
            "source_version": observation.source_version,
            "event_type": observation.event_type,
            "content": retained.content,
            "content_hash": retained.content_hash,
            "metadata": json.dumps(retained.metadata),
            "source_created_at": observation.source_created_at,
            "source_updated_at": observation.source_updated_at,
            "first_observed_at": observed_at,
            "last_observed_at": observed_at,
            "is_deleted": observation.is_deleted,
            "now": observed_at,
            "revision": revision,
        }

    async def get_by_source(
        self,
        *,
        tenant_id: str,
        source_namespace: str,
        source_record_id: str,
    ) -> StoredObservation | None:
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT
                        id, tenant_id, subject_id, actor_id,
                        source_type, source_namespace, source_record_id, source_version,
                        event_type, content, content_hash, metadata,
                        source_created_at, source_updated_at, last_observed_at,
                        current_revision, is_deleted
                    FROM {self._table("observations")}
                    WHERE tenant_id = :tenant_id
                      AND source_namespace = :source_namespace
                      AND source_record_id = :source_record_id
                    """
                ),
                {
                    "tenant_id": tenant_id,
                    "source_namespace": source_namespace,
                    "source_record_id": source_record_id,
                },
            )
            row = result.mappings().first()
        if row is None:
            return None
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        return StoredObservation(
            id=str(row["id"]),
            tenant_id=row["tenant_id"],
            subject_id=row["subject_id"],
            actor_id=row["actor_id"],
            source_type=row["source_type"],
            source_namespace=row["source_namespace"],
            source_record_id=row["source_record_id"],
            source_version=row["source_version"],
            event_type=row["event_type"],
            content=row["content"],
            content_hash=row["content_hash"],
            metadata=MappingProxyType(dict(metadata)),
            source_created_at=row["source_created_at"],
            source_updated_at=row["source_updated_at"],
            observed_at=row["last_observed_at"],
            current_revision=row["current_revision"],
            is_deleted=row["is_deleted"],
        )


class PostgresCheckpointStore(CheckpointStore):
    """PostgreSQL-backed connector checkpoint store."""

    def __init__(self, engine: AsyncEngine, *, schema: str = "cognema") -> None:
        self._engine = engine
        self._schema = schema

    def _table(self) -> str:
        return f"{self._schema}.connector_checkpoints"

    async def get(
        self,
        *,
        tenant_id: str,
        connector_id: str,
    ) -> dict[str, Any] | None:
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT checkpoint
                    FROM {self._table()}
                    WHERE tenant_id = :tenant_id AND connector_id = :connector_id
                    """
                ),
                {"tenant_id": tenant_id, "connector_id": connector_id},
            )
            row = result.first()
        if row is None:
            return None
        checkpoint = row[0]
        if isinstance(checkpoint, str):
            parsed: dict[str, Any] = json.loads(checkpoint)
            return parsed
        if isinstance(checkpoint, dict):
            return dict(checkpoint)
        return None

    async def set(
        self,
        *,
        tenant_id: str,
        connector_id: str,
        checkpoint: dict[str, Any],
    ) -> None:
        try:
            async with self._engine.begin() as conn:
                await conn.execute(
                    text(
                        f"""
                        INSERT INTO {self._table()} (
                            tenant_id, connector_id, checkpoint, updated_at
                        ) VALUES (
                            :tenant_id, :connector_id, CAST(:checkpoint AS jsonb), now()
                        )
                        ON CONFLICT (tenant_id, connector_id)
                        DO UPDATE SET
                            checkpoint = EXCLUDED.checkpoint,
                            updated_at = now()
                        """
                    ),
                    {
                        "tenant_id": tenant_id,
                        "connector_id": connector_id,
                        "checkpoint": json.dumps(checkpoint),
                    },
                )
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"Failed to persist checkpoint: {exc}") from exc
