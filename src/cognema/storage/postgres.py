"""PostgreSQL observation and checkpoint storage."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from cognema.exceptions import StorageError
from cognema.models import (
    EpisodeEntity,
    EpisodeEvidenceInput,
    EpisodeInput,
    EpisodeWriteStatus,
    SemanticCardinality,
    SemanticDerivationInput,
    SemanticDerivationRelation,
    SemanticMemoryInput,
    SemanticMemoryStatus,
    SemanticPolarity,
    SemanticWriteStatus,
    StoredEpisode,
    StoredSemanticMemory,
)
from cognema.observations.models import IngestStatus, ObservationInput, StoredObservation
from cognema.observations.retention import RetainedObservation
from cognema.storage.base import (
    CheckpointStore,
    EpisodeStore,
    ObservationStore,
    SemanticMemoryStore,
)


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
                        current_revision, is_deleted,
                        attention_score, retention_class, policy_reasons,
                        created_at, updated_at
                    ) VALUES (
                        :id, :tenant_id, :subject_id, :actor_id,
                        :source_type, :source_namespace, :source_record_id, :source_version,
                        :event_type, :content, :content_hash, CAST(:metadata AS jsonb),
                        :source_created_at, :source_updated_at,
                        :first_observed_at, :last_observed_at,
                        1, :is_deleted,
                        :attention_score, :retention_class, CAST(:policy_reasons AS jsonb),
                        :now, :now
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
                        attention_score = :attention_score,
                        retention_class = :retention_class,
                        policy_reasons = CAST(:policy_reasons AS jsonb),
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
                    "attention_score": retained.attention_score,
                    "retention_class": retained.retention_class,
                    "policy_reasons": json.dumps(list(retained.policy_reasons)),
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
            "attention_score": retained.attention_score,
            "retention_class": retained.retention_class,
            "policy_reasons": json.dumps(list(retained.policy_reasons)),
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
                        current_revision, is_deleted,
                        attention_score, retention_class, policy_reasons
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
        return self._to_stored(row)

    async def list(
        self,
        *,
        tenant_id: str,
        subject_id: str | None = None,
        include_deleted: bool = False,
    ) -> list[StoredObservation]:
        clauses = ["tenant_id = :tenant_id"]
        params: dict[str, Any] = {"tenant_id": tenant_id}
        if subject_id is not None:
            clauses.append("subject_id = :subject_id")
            params["subject_id"] = subject_id
        if not include_deleted:
            clauses.append("is_deleted = FALSE")
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT
                        id, tenant_id, subject_id, actor_id,
                        source_type, source_namespace, source_record_id, source_version,
                        event_type, content, content_hash, metadata,
                        source_created_at, source_updated_at, last_observed_at,
                        current_revision, is_deleted,
                        attention_score, retention_class, policy_reasons
                    FROM {self._table("observations")}
                    WHERE {" AND ".join(clauses)}
                    ORDER BY last_observed_at, id
                    """
                ),
                params,
            )
            rows = result.mappings().all()
        return [self._to_stored(row) for row in rows]

    async def get_many(
        self,
        *,
        tenant_id: str,
        observation_ids: set[str],
    ) -> Sequence[StoredObservation]:
        if not observation_ids:
            return []
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT
                        id, tenant_id, subject_id, actor_id,
                        source_type, source_namespace, source_record_id, source_version,
                        event_type, content, content_hash, metadata,
                        source_created_at, source_updated_at, last_observed_at,
                        current_revision, is_deleted,
                        attention_score, retention_class, policy_reasons
                    FROM {self._table("observations")}
                    WHERE tenant_id = :tenant_id
                      AND id = ANY(CAST(:observation_ids AS uuid[]))
                    ORDER BY last_observed_at, id
                    """
                ),
                {"tenant_id": tenant_id, "observation_ids": list(observation_ids)},
            )
            rows = result.mappings().all()
        return [self._to_stored(row) for row in rows]

    async def clear(self, *, tenant_id: str) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text(f"DELETE FROM {self._table('observations')} WHERE tenant_id = :tenant_id"),
                {"tenant_id": tenant_id},
            )

    def _to_stored(self, row: Any) -> StoredObservation:
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        policy_reasons = row.get("policy_reasons", [])
        if isinstance(policy_reasons, str):
            policy_reasons = json.loads(policy_reasons)
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
            attention_score=float(row.get("attention_score", 0.5)),
            retention_class=row.get("retention_class", "full"),
            policy_reasons=tuple(policy_reasons),
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


class PostgresEpisodeStore(EpisodeStore):
    """PostgreSQL-backed episodic memory store."""

    def __init__(self, engine: AsyncEngine, *, schema: str = "cognema") -> None:
        self._engine = engine
        self._schema = schema

    def _table(self, name: str) -> str:
        return f"{self._schema}.{name}"

    async def upsert(self, episode: EpisodeInput) -> EpisodeWriteStatus:
        fingerprint = episode.metadata["episode"]["content_fingerprint"]
        existing = await self._get_by_memory_key(
            tenant_id=episode.tenant_id,
            memory_key=episode.memory_key,
        )
        if existing is not None:
            existing_fingerprint = existing.metadata["episode"]["content_fingerprint"]
            if existing_fingerprint == fingerprint:
                return EpisodeWriteStatus.UNCHANGED
            await self._update(existing.id, episode)
            return EpisodeWriteStatus.UPDATED

        await self._create(episode)
        return EpisodeWriteStatus.CREATED

    async def _get_by_memory_key(
        self,
        *,
        tenant_id: str,
        memory_key: str,
    ) -> StoredEpisode | None:
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT id
                    FROM {self._table("memories")}
                    WHERE tenant_id = :tenant_id
                      AND memory_type = 'episodic'
                      AND memory_key = :memory_key
                    """
                ),
                {"tenant_id": tenant_id, "memory_key": memory_key},
            )
            row = result.first()
        if row is None:
            return None
        return await self._load_episode(str(row[0]))

    async def _load_episode(self, memory_id: str) -> StoredEpisode:
        async with self._engine.connect() as conn:
            memory = await conn.execute(
                text(
                    f"""
                    SELECT
                        id, tenant_id, subject_id, memory_key, statement,
                        confidence, importance, valid_from, valid_until,
                        is_active, metadata, created_at, updated_at
                    FROM {self._table("memories")}
                    WHERE id = :id
                    """
                ),
                {"id": memory_id},
            )
            memory_row = memory.mappings().first()
            if memory_row is None:
                raise StorageError(f"Episode {memory_id} not found.")

            evidence_result = await conn.execute(
                text(
                    f"""
                    SELECT
                        observation_id, observation_revision,
                        sequence_number, contribution_score
                    FROM {self._table("memory_evidence")}
                    WHERE memory_id = :memory_id
                    ORDER BY sequence_number
                    """
                ),
                {"memory_id": memory_id},
            )
            entity_result = await conn.execute(
                text(
                    f"""
                    SELECT entity_id, entity_role
                    FROM {self._table("memory_entities")}
                    WHERE memory_id = :memory_id
                    ORDER BY entity_id, entity_role
                    """
                ),
                {"memory_id": memory_id},
            )
        return self._to_stored(
            memory_row, evidence_result.mappings().all(), entity_result.mappings().all()
        )

    async def _create(self, episode: EpisodeInput) -> None:
        memory_id = str(uuid4())
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    INSERT INTO {self._table("memories")} (
                        id, tenant_id, subject_id, memory_type, memory_key,
                        statement, confidence, importance,
                        valid_from, valid_until, is_active, metadata,
                        created_at, updated_at
                    ) VALUES (
                        :id, :tenant_id, :subject_id, 'episodic', :memory_key,
                        :statement, :confidence, :importance,
                        :valid_from, :valid_until, TRUE, CAST(:metadata AS jsonb),
                        :now, :now
                    )
                    """
                ),
                self._memory_params(memory_id, episode),
            )
            await self._replace_evidence(conn, memory_id, episode.evidence)
            await self._replace_entities(conn, memory_id, episode.entities)

    async def _update(self, memory_id: str, episode: EpisodeInput) -> None:
        metadata_json = json.dumps(dict(episode.metadata))
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    UPDATE {self._table("memories")}
                    SET
                        subject_id = :subject_id,
                        statement = :statement,
                        confidence = :confidence,
                        importance = :importance,
                        valid_from = :valid_from,
                        valid_until = :valid_until,
                        is_active = TRUE,
                        metadata = CAST(:metadata AS jsonb),
                        updated_at = :now
                    WHERE id = :id
                    """
                ),
                {
                    "id": memory_id,
                    "subject_id": episode.subject_id,
                    "statement": episode.statement,
                    "confidence": episode.confidence,
                    "importance": episode.importance,
                    "valid_from": episode.started_at,
                    "valid_until": episode.ended_at,
                    "metadata": metadata_json,
                    "now": datetime.now(UTC),
                },
            )
            await self._replace_evidence(conn, memory_id, episode.evidence)
            await self._replace_entities(conn, memory_id, episode.entities)

    async def _replace_evidence(
        self,
        conn: Any,
        memory_id: str,
        evidence: tuple[EpisodeEvidenceInput, ...],
    ) -> None:
        await conn.execute(
            text(f"DELETE FROM {self._table('memory_evidence')} WHERE memory_id = :memory_id"),
            {"memory_id": memory_id},
        )
        for item in evidence:
            await conn.execute(
                text(
                    f"""
                    INSERT INTO {self._table("memory_evidence")} (
                        memory_id, observation_id, observation_revision,
                        sequence_number, contribution_score
                    ) VALUES (
                        :memory_id, :observation_id, :observation_revision,
                        :sequence_number, :contribution_score
                    )
                    """
                ),
                {
                    "memory_id": memory_id,
                    "observation_id": item.observation_id,
                    "observation_revision": item.observation_revision,
                    "sequence_number": item.sequence_number,
                    "contribution_score": item.contribution_score,
                },
            )

    async def _replace_entities(
        self,
        conn: Any,
        memory_id: str,
        entities: tuple[EpisodeEntity, ...],
    ) -> None:
        await conn.execute(
            text(f"DELETE FROM {self._table('memory_entities')} WHERE memory_id = :memory_id"),
            {"memory_id": memory_id},
        )
        for entity in entities:
            await conn.execute(
                text(
                    f"""
                    INSERT INTO {self._table("memory_entities")} (
                        memory_id, entity_id, entity_role
                    ) VALUES (
                        :memory_id, :entity_id, :entity_role
                    )
                    """
                ),
                {
                    "memory_id": memory_id,
                    "entity_id": entity.entity_id,
                    "entity_role": entity.role,
                },
            )

    def _memory_params(self, memory_id: str, episode: EpisodeInput) -> dict[str, Any]:
        now = datetime.now(UTC)
        return {
            "id": memory_id,
            "tenant_id": episode.tenant_id,
            "subject_id": episode.subject_id,
            "memory_key": episode.memory_key,
            "statement": episode.statement,
            "confidence": episode.confidence,
            "importance": episode.importance,
            "valid_from": episode.started_at,
            "valid_until": episode.ended_at,
            "metadata": json.dumps(dict(episode.metadata)),
            "now": now,
        }

    async def list(
        self,
        *,
        tenant_id: str,
        subject_id: str | None = None,
        include_inactive: bool = False,
        limit: int | None = None,
    ) -> list[StoredEpisode]:
        clauses = ["tenant_id = :tenant_id", "memory_type = 'episodic'"]
        params: dict[str, Any] = {"tenant_id": tenant_id}
        if subject_id is not None:
            clauses.append("subject_id = :subject_id")
            params["subject_id"] = subject_id
        if not include_inactive:
            clauses.append("is_active = TRUE")
        limit_clause = ""
        if limit is not None:
            limit_clause = " LIMIT :limit"
            params["limit"] = limit
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT id
                    FROM {self._table("memories")}
                    WHERE {" AND ".join(clauses)}
                    ORDER BY valid_from, id
                    {limit_clause}
                    """
                ),
                params,
            )
            memory_ids = [str(row[0]) for row in result.all()]
        episodes: list[StoredEpisode] = []
        for memory_id in memory_ids:
            episodes.append(await self._load_episode(memory_id))
        return episodes

    async def deactivate_missing(
        self,
        *,
        tenant_id: str,
        subject_id: str | None,
        active_memory_keys: set[str],
    ) -> int:
        clauses = [
            "tenant_id = :tenant_id",
            "memory_type = 'episodic'",
            "is_active = TRUE",
        ]
        params: dict[str, Any] = {"tenant_id": tenant_id}
        if subject_id is not None:
            clauses.append("subject_id = :subject_id")
            params["subject_id"] = subject_id
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT id, memory_key
                    FROM {self._table("memories")}
                    WHERE {" AND ".join(clauses)}
                    """
                ),
                params,
            )
            rows = result.mappings().all()

        deactivated = 0
        for row in rows:
            memory_key = row["memory_key"]
            if memory_key in active_memory_keys:
                continue
            async with self._engine.begin() as conn:
                await conn.execute(
                    text(
                        f"""
                        UPDATE {self._table("memories")}
                        SET is_active = FALSE, updated_at = :now
                        WHERE id = :id
                        """
                    ),
                    {"id": str(row["id"]), "now": datetime.now(UTC)},
                )
            deactivated += 1
        return deactivated

    async def clear(self, *, tenant_id: str) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    DELETE FROM {self._table("memories")}
                    WHERE tenant_id = :tenant_id AND memory_type = 'episodic'
                    """
                ),
                {"tenant_id": tenant_id},
            )

    def _to_stored(
        self,
        row: Any,
        evidence_items: Sequence[Any],
        entity_items: Sequence[Any],
    ) -> StoredEpisode:
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        evidence = tuple(
            EpisodeEvidenceInput(
                observation_id=str(item["observation_id"]),
                observation_revision=int(item["observation_revision"]),
                sequence_number=int(item["sequence_number"]),
                contribution_score=float(item["contribution_score"] or 1.0),
            )
            for item in evidence_items
        )
        entities = tuple(
            EpisodeEntity(entity_id=item["entity_id"], role=item["entity_role"])
            for item in entity_items
        )
        return StoredEpisode(
            id=str(row["id"]),
            tenant_id=row["tenant_id"],
            subject_id=row["subject_id"],
            memory_key=row["memory_key"],
            statement=row["statement"],
            started_at=row["valid_from"],
            ended_at=row["valid_until"],
            confidence=float(row["confidence"]),
            importance=float(row["importance"]),
            is_active=row["is_active"],
            evidence=evidence,
            entities=entities,
            metadata=MappingProxyType(dict(metadata)),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class PostgresSemanticMemoryStore(SemanticMemoryStore):
    """PostgreSQL-backed semantic memory store."""

    def __init__(self, engine: AsyncEngine, *, schema: str = "cognema") -> None:
        self._engine = engine
        self._schema = schema

    def _table(self, name: str) -> str:
        return f"{self._schema}.{name}"

    async def upsert(self, memory: SemanticMemoryInput) -> SemanticWriteStatus:
        fingerprint = memory.metadata["semantic"]["content_fingerprint"]
        existing = await self._get_by_memory_key(
            tenant_id=memory.tenant_id,
            memory_key=memory.memory_key,
        )
        if existing is not None:
            existing_fingerprint = existing.metadata["semantic"]["content_fingerprint"]
            if existing_fingerprint == fingerprint:
                return SemanticWriteStatus.UNCHANGED
            await self._update(existing.id, memory)
            return SemanticWriteStatus.UPDATED

        await self._create(memory)
        return SemanticWriteStatus.CREATED

    async def _get_by_memory_key(
        self,
        *,
        tenant_id: str,
        memory_key: str,
    ) -> StoredSemanticMemory | None:
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT id
                    FROM {self._table("memories")}
                    WHERE tenant_id = :tenant_id
                      AND memory_type = 'semantic'
                      AND memory_key = :memory_key
                    """
                ),
                {"tenant_id": tenant_id, "memory_key": memory_key},
            )
            row = result.first()
        if row is None:
            return None
        return await self._load_semantic(str(row[0]))

    async def _load_semantic(self, memory_id: str) -> StoredSemanticMemory:
        async with self._engine.connect() as conn:
            memory = await conn.execute(
                text(
                    f"""
                    SELECT
                        m.id, m.tenant_id, m.subject_id, m.memory_key, m.statement,
                        m.confidence, m.importance, m.valid_from, m.valid_until,
                        m.is_active, m.metadata, m.created_at, m.updated_at,
                        c.slot_key, c.subject_entity_id, c.predicate, c.object_value,
                        c.object_entity_id, c.polarity, c.cardinality, c.qualifiers,
                        c.status, c.support_count, c.contradiction_count,
                        c.first_supported_at, c.last_supported_at
                    FROM {self._table("memories")} AS m
                    JOIN {self._table("semantic_claims")} AS c
                      ON c.memory_id = m.id
                    WHERE m.id = :id
                    """
                ),
                {"id": memory_id},
            )
            memory_row = memory.mappings().first()
            if memory_row is None:
                raise StorageError(f"Semantic memory {memory_id} not found.")

            derivation_result = await conn.execute(
                text(
                    f"""
                    SELECT source_memory_id, relation, contribution_score
                    FROM {self._table("memory_derivations")}
                    WHERE target_memory_id = :memory_id
                    ORDER BY source_memory_id, relation
                    """
                ),
                {"memory_id": memory_id},
            )
            evidence_result = await conn.execute(
                text(
                    f"""
                    SELECT
                        observation_id, observation_revision,
                        sequence_number, contribution_score
                    FROM {self._table("memory_evidence")}
                    WHERE memory_id = :memory_id
                    ORDER BY sequence_number
                    """
                ),
                {"memory_id": memory_id},
            )
            entity_result = await conn.execute(
                text(
                    f"""
                    SELECT entity_id, entity_role
                    FROM {self._table("memory_entities")}
                    WHERE memory_id = :memory_id
                    ORDER BY entity_id, entity_role
                    """
                ),
                {"memory_id": memory_id},
            )
        return self._to_stored(
            memory_row,
            derivation_result.mappings().all(),
            evidence_result.mappings().all(),
            entity_result.mappings().all(),
        )

    async def _create(self, memory: SemanticMemoryInput) -> None:
        memory_id = str(uuid4())
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    INSERT INTO {self._table("memories")} (
                        id, tenant_id, subject_id, memory_type, memory_key,
                        statement, confidence, importance,
                        valid_from, valid_until, is_active, metadata,
                        created_at, updated_at
                    ) VALUES (
                        :id, :tenant_id, :subject_id, 'semantic', :memory_key,
                        :statement, :confidence, :importance,
                        :valid_from, :valid_until, TRUE, CAST(:metadata AS jsonb),
                        :now, :now
                    )
                    """
                ),
                self._memory_params(memory_id, memory),
            )
            await self._upsert_claim(conn, memory_id, memory)
            await self._replace_derivations(conn, memory.tenant_id, memory_id, memory.derivations)
            await self._replace_evidence(conn, memory_id, memory.observation_evidence)
            await self._replace_entities(conn, memory_id, memory.entities)

    async def _update(self, memory_id: str, memory: SemanticMemoryInput) -> None:
        metadata_json = json.dumps(dict(memory.metadata))
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    UPDATE {self._table("memories")}
                    SET
                        subject_id = :subject_id,
                        statement = :statement,
                        confidence = :confidence,
                        importance = :importance,
                        valid_from = :valid_from,
                        valid_until = :valid_until,
                        is_active = TRUE,
                        metadata = CAST(:metadata AS jsonb),
                        updated_at = :now
                    WHERE id = :id
                    """
                ),
                {
                    "id": memory_id,
                    "subject_id": memory.subject_id,
                    "statement": memory.statement,
                    "confidence": memory.confidence,
                    "importance": memory.importance,
                    "valid_from": memory.first_supported_at,
                    "valid_until": memory.last_supported_at,
                    "metadata": metadata_json,
                    "now": datetime.now(UTC),
                },
            )
            await self._upsert_claim(conn, memory_id, memory)
            await self._replace_derivations(conn, memory.tenant_id, memory_id, memory.derivations)
            await self._replace_evidence(conn, memory_id, memory.observation_evidence)
            await self._replace_entities(conn, memory_id, memory.entities)

    async def _upsert_claim(
        self,
        conn: Any,
        memory_id: str,
        memory: SemanticMemoryInput,
    ) -> None:
        qualifiers_json = json.dumps(dict(memory.qualifiers))
        await conn.execute(
            text(
                f"""
                INSERT INTO {self._table("semantic_claims")} (
                    tenant_id, memory_id, slot_key,
                    subject_entity_id, predicate, object_value, object_entity_id,
                    polarity, cardinality, qualifiers,
                    status, support_count, contradiction_count,
                    first_supported_at, last_supported_at
                ) VALUES (
                    :tenant_id, :memory_id, :slot_key,
                    :subject_entity_id, :predicate, :object_value, :object_entity_id,
                    :polarity, :cardinality, CAST(:qualifiers AS jsonb),
                    :status, :support_count, :contradiction_count,
                    :first_supported_at, :last_supported_at
                )
                ON CONFLICT (memory_id) DO UPDATE SET
                    slot_key = EXCLUDED.slot_key,
                    subject_entity_id = EXCLUDED.subject_entity_id,
                    predicate = EXCLUDED.predicate,
                    object_value = EXCLUDED.object_value,
                    object_entity_id = EXCLUDED.object_entity_id,
                    polarity = EXCLUDED.polarity,
                    cardinality = EXCLUDED.cardinality,
                    qualifiers = EXCLUDED.qualifiers,
                    status = EXCLUDED.status,
                    support_count = EXCLUDED.support_count,
                    contradiction_count = EXCLUDED.contradiction_count,
                    first_supported_at = EXCLUDED.first_supported_at,
                    last_supported_at = EXCLUDED.last_supported_at
                """
            ),
            {
                "tenant_id": memory.tenant_id,
                "memory_id": memory_id,
                "slot_key": memory.slot_key,
                "subject_entity_id": memory.subject_entity_id,
                "predicate": memory.predicate,
                "object_value": memory.object_value,
                "object_entity_id": memory.object_entity_id,
                "polarity": memory.polarity.value,
                "cardinality": memory.cardinality.value,
                "qualifiers": qualifiers_json,
                "status": memory.status.value,
                "support_count": memory.support_count,
                "contradiction_count": memory.contradiction_count,
                "first_supported_at": memory.first_supported_at,
                "last_supported_at": memory.last_supported_at,
            },
        )

    async def _replace_derivations(
        self,
        conn: Any,
        tenant_id: str,
        memory_id: str,
        derivations: tuple[SemanticDerivationInput, ...],
    ) -> None:
        await conn.execute(
            text(
                f"""
                DELETE FROM {self._table("memory_derivations")}
                WHERE target_memory_id = :memory_id
                """
            ),
            {"memory_id": memory_id},
        )
        for derivation in derivations:
            await conn.execute(
                text(
                    f"""
                    INSERT INTO {self._table("memory_derivations")} (
                        tenant_id, target_memory_id, source_memory_id,
                        relation, contribution_score
                    ) VALUES (
                        :tenant_id, :target_memory_id, :source_memory_id,
                        :relation, :contribution_score
                    )
                    """
                ),
                {
                    "tenant_id": tenant_id,
                    "target_memory_id": memory_id,
                    "source_memory_id": derivation.episode_id,
                    "relation": derivation.relation.value,
                    "contribution_score": derivation.contribution_score,
                },
            )

    async def _replace_evidence(
        self,
        conn: Any,
        memory_id: str,
        evidence: tuple[EpisodeEvidenceInput, ...],
    ) -> None:
        await conn.execute(
            text(f"DELETE FROM {self._table('memory_evidence')} WHERE memory_id = :memory_id"),
            {"memory_id": memory_id},
        )
        for item in evidence:
            await conn.execute(
                text(
                    f"""
                    INSERT INTO {self._table("memory_evidence")} (
                        memory_id, observation_id, observation_revision,
                        sequence_number, contribution_score
                    ) VALUES (
                        :memory_id, :observation_id, :observation_revision,
                        :sequence_number, :contribution_score
                    )
                    """
                ),
                {
                    "memory_id": memory_id,
                    "observation_id": item.observation_id,
                    "observation_revision": item.observation_revision,
                    "sequence_number": item.sequence_number,
                    "contribution_score": item.contribution_score,
                },
            )

    async def _replace_entities(
        self,
        conn: Any,
        memory_id: str,
        entities: tuple[EpisodeEntity, ...],
    ) -> None:
        await conn.execute(
            text(f"DELETE FROM {self._table('memory_entities')} WHERE memory_id = :memory_id"),
            {"memory_id": memory_id},
        )
        for entity in entities:
            await conn.execute(
                text(
                    f"""
                    INSERT INTO {self._table("memory_entities")} (
                        memory_id, entity_id, entity_role
                    ) VALUES (
                        :memory_id, :entity_id, :entity_role
                    )
                    """
                ),
                {
                    "memory_id": memory_id,
                    "entity_id": entity.entity_id,
                    "entity_role": entity.role,
                },
            )

    def _memory_params(self, memory_id: str, memory: SemanticMemoryInput) -> dict[str, Any]:
        now = datetime.now(UTC)
        return {
            "id": memory_id,
            "tenant_id": memory.tenant_id,
            "subject_id": memory.subject_id,
            "memory_key": memory.memory_key,
            "statement": memory.statement,
            "confidence": memory.confidence,
            "importance": memory.importance,
            "valid_from": memory.first_supported_at,
            "valid_until": memory.last_supported_at,
            "metadata": json.dumps(dict(memory.metadata)),
            "now": now,
        }

    async def list(
        self,
        *,
        tenant_id: str,
        subject_id: str | None = None,
        include_inactive: bool = False,
        status: SemanticMemoryStatus | None = None,
        limit: int | None = None,
    ) -> list[StoredSemanticMemory]:
        clauses = ["m.tenant_id = :tenant_id", "m.memory_type = 'semantic'"]
        params: dict[str, Any] = {"tenant_id": tenant_id}
        if subject_id is not None:
            clauses.append("m.subject_id = :subject_id")
            params["subject_id"] = subject_id
        if not include_inactive:
            clauses.append("m.is_active = TRUE")
        if status is not None:
            clauses.append("c.status = :status")
            params["status"] = status.value
        limit_clause = ""
        if limit is not None:
            limit_clause = " LIMIT :limit"
            params["limit"] = limit
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT m.id
                    FROM {self._table("memories")} AS m
                    JOIN {self._table("semantic_claims")} AS c
                      ON c.memory_id = m.id
                    WHERE {" AND ".join(clauses)}
                    ORDER BY c.first_supported_at, m.id
                    {limit_clause}
                    """
                ),
                params,
            )
            memory_ids = [str(row[0]) for row in result.all()]
        memories: list[StoredSemanticMemory] = []
        for memory_id in memory_ids:
            memories.append(await self._load_semantic(memory_id))
        return memories

    async def deactivate_missing(
        self,
        *,
        tenant_id: str,
        subject_id: str | None,
        active_memory_keys: set[str],
    ) -> int:
        clauses = [
            "tenant_id = :tenant_id",
            "memory_type = 'semantic'",
            "is_active = TRUE",
        ]
        params: dict[str, Any] = {"tenant_id": tenant_id}
        if subject_id is not None:
            clauses.append("subject_id = :subject_id")
            params["subject_id"] = subject_id
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT id, memory_key
                    FROM {self._table("memories")}
                    WHERE {" AND ".join(clauses)}
                    """
                ),
                params,
            )
            rows = result.mappings().all()

        deactivated = 0
        for row in rows:
            memory_key = row["memory_key"]
            if memory_key in active_memory_keys:
                continue
            async with self._engine.begin() as conn:
                await conn.execute(
                    text(
                        f"""
                        UPDATE {self._table("memories")}
                        SET is_active = FALSE, updated_at = :now
                        WHERE id = :id
                        """
                    ),
                    {"id": str(row["id"]), "now": datetime.now(UTC)},
                )
            deactivated += 1
        return deactivated

    async def clear(self, *, tenant_id: str) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    DELETE FROM {self._table("memories")}
                    WHERE tenant_id = :tenant_id AND memory_type = 'semantic'
                    """
                ),
                {"tenant_id": tenant_id},
            )

    def _to_stored(
        self,
        row: Any,
        derivation_items: Sequence[Any],
        evidence_items: Sequence[Any],
        entity_items: Sequence[Any],
    ) -> StoredSemanticMemory:
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        qualifiers = row["qualifiers"]
        if isinstance(qualifiers, str):
            qualifiers = json.loads(qualifiers)
        derivations = tuple(
            SemanticDerivationInput(
                episode_id=str(item["source_memory_id"]),
                relation=SemanticDerivationRelation(item["relation"]),
                contribution_score=float(item["contribution_score"]),
            )
            for item in derivation_items
        )
        evidence = tuple(
            EpisodeEvidenceInput(
                observation_id=str(item["observation_id"]),
                observation_revision=int(item["observation_revision"]),
                sequence_number=int(item["sequence_number"]),
                contribution_score=float(item["contribution_score"] or 1.0),
            )
            for item in evidence_items
        )
        entities = tuple(
            EpisodeEntity(entity_id=item["entity_id"], role=item["entity_role"])
            for item in entity_items
        )
        return StoredSemanticMemory(
            id=str(row["id"]),
            tenant_id=row["tenant_id"],
            subject_id=row["subject_id"],
            memory_key=row["memory_key"],
            slot_key=row["slot_key"],
            statement=row["statement"],
            subject_entity_id=row["subject_entity_id"],
            predicate=row["predicate"],
            object_value=row["object_value"],
            object_entity_id=row["object_entity_id"],
            polarity=SemanticPolarity(row["polarity"]),
            cardinality=SemanticCardinality(row["cardinality"]),
            qualifiers=MappingProxyType(dict(qualifiers)),
            confidence=float(row["confidence"]),
            importance=float(row["importance"]),
            status=SemanticMemoryStatus(row["status"]),
            support_count=int(row["support_count"]),
            contradiction_count=int(row["contradiction_count"]),
            first_supported_at=row["first_supported_at"],
            last_supported_at=row["last_supported_at"],
            is_active=row["is_active"],
            derivations=derivations,
            observation_evidence=evidence,
            entities=entities,
            metadata=MappingProxyType(dict(metadata)),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
