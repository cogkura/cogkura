"""PostgreSQL observation and checkpoint storage."""

from __future__ import annotations

import builtins
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Any, cast
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from cogkura.algorithms.reconsolidation import revision_valid_at
from cogkura.exceptions import StorageError
from cogkura.models import (
    ActivationReferenceKind,
    ActivationReferenceTrace,
    EpisodeEntity,
    EpisodeEvidenceInput,
    EpisodeInput,
    EpisodeWriteStatus,
    LearningOutcome,
    LearningPlan,
    LearningWriteResult,
    MemoryIdentity,
    MemoryKind,
    MemoryReference,
    MemoryRetentionState,
    ReferenceCompactionResult,
    SemanticCardinality,
    SemanticDerivationInput,
    SemanticDerivationRelation,
    SemanticMemoryInput,
    SemanticMemoryStatus,
    SemanticPolarity,
    SemanticReconciliationPlan,
    SemanticReconciliationWriteResult,
    SemanticWriteStatus,
    StoredEpisode,
    StoredMemoryAssociation,
    StoredMemoryDynamics,
    StoredMemoryLearningState,
    StoredSemanticMemory,
    StoredSemanticRevision,
)
from cogkura.observations.models import IngestStatus, ObservationInput, StoredObservation
from cogkura.observations.retention import RetainedObservation
from cogkura.storage.activation_compaction import compaction_representative_time
from cogkura.storage.base import (
    ActivationStore,
    CheckpointStore,
    EpisodeStore,
    LearningStore,
    MemoryDynamicsStore,
    ObservationStore,
    SemanticMemoryStore,
)


class PostgresObservationStore(ObservationStore):
    """PostgreSQL-backed observation store."""

    def __init__(self, engine: AsyncEngine, *, schema: str = "cogkura") -> None:
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

    def __init__(self, engine: AsyncEngine, *, schema: str = "cogkura") -> None:
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

    def __init__(self, engine: AsyncEngine, *, schema: str = "cogkura") -> None:
        self._engine = engine
        self._schema = schema

    def _table(self, name: str) -> str:
        return f"{self._schema}.{name}"

    async def upsert(
        self,
        episode: EpisodeInput,
        *,
        as_of: datetime | None = None,
    ) -> EpisodeWriteStatus:
        fingerprint = episode.metadata["episode"]["content_fingerprint"]
        existing = await self._get_by_memory_key(
            tenant_id=episode.tenant_id,
            memory_key=episode.memory_key,
        )
        if existing is not None:
            existing_fingerprint = existing.metadata["episode"]["content_fingerprint"]
            if existing_fingerprint == fingerprint:
                return EpisodeWriteStatus.UNCHANGED
            await self._update(existing.id, episode, as_of=as_of)
            return EpisodeWriteStatus.UPDATED

        await self._create(episode, as_of=as_of)
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

    async def _create(self, episode: EpisodeInput, *, as_of: datetime | None = None) -> None:
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
                self._memory_params(memory_id, episode, as_of=as_of),
            )
            await self._replace_evidence(conn, memory_id, episode.evidence)
            await self._replace_entities(conn, memory_id, episode.entities)

    async def _update(
        self,
        memory_id: str,
        episode: EpisodeInput,
        *,
        as_of: datetime | None = None,
    ) -> None:
        metadata_json = json.dumps(dict(episode.metadata))
        now = as_of.astimezone(UTC) if as_of is not None else datetime.now(UTC)
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
                    "now": now,
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

    def _memory_params(
        self,
        memory_id: str,
        episode: EpisodeInput,
        *,
        as_of: datetime | None = None,
    ) -> dict[str, Any]:
        now = as_of.astimezone(UTC) if as_of is not None else datetime.now(UTC)
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
        as_of: datetime | None = None,
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

        now = as_of.astimezone(UTC) if as_of is not None else datetime.now(UTC)
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
                    {"id": str(row["id"]), "now": now},
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

    def __init__(self, engine: AsyncEngine, *, schema: str = "cogkura") -> None:
        self._engine = engine
        self._schema = schema

    def _table(self, name: str) -> str:
        return f"{self._schema}.{name}"

    async def upsert(
        self,
        memory: SemanticMemoryInput,
        *,
        as_of: datetime | None = None,
    ) -> SemanticWriteStatus:
        fingerprint = memory.metadata["semantic"]["content_fingerprint"]
        existing = await self._get_by_memory_key(
            tenant_id=memory.tenant_id,
            memory_key=memory.memory_key,
        )
        if existing is not None:
            existing_fingerprint = existing.metadata["semantic"]["content_fingerprint"]
            if existing_fingerprint == fingerprint:
                return SemanticWriteStatus.UNCHANGED
            await self._update(existing.id, memory, as_of=as_of)
            return SemanticWriteStatus.UPDATED

        await self._create(memory, as_of=as_of)
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
                        c.first_supported_at, c.last_supported_at,
                        c.current_revision_key,
                        COALESCE(r.revision_number, 1) AS revision_number
                    FROM {self._table("memories")} AS m
                    JOIN {self._table("semantic_claims")} AS c
                      ON c.memory_id = m.id
                    LEFT JOIN {self._table("semantic_claim_revisions")} AS r
                      ON r.revision_key = c.current_revision_key
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

    async def _create(self, memory: SemanticMemoryInput, *, as_of: datetime | None = None) -> None:
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
                self._memory_params(memory_id, memory, as_of=as_of),
            )
            await self._upsert_claim(conn, memory_id, memory)
            await self._replace_derivations(
                conn,
                memory.tenant_id,
                memory_id,
                memory.revision_key,
                memory.derivations,
            )
            await self._replace_evidence(conn, memory_id, memory.observation_evidence)
            await self._replace_entities(conn, memory_id, memory.entities)

    async def _update(
        self,
        memory_id: str,
        memory: SemanticMemoryInput,
        *,
        as_of: datetime | None = None,
    ) -> None:
        metadata_json = json.dumps(dict(memory.metadata))
        now = as_of.astimezone(UTC) if as_of is not None else datetime.now(UTC)
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
                    "valid_from": memory.valid_from,
                    "valid_until": memory.valid_until,
                    "metadata": metadata_json,
                    "now": now,
                },
            )
            await self._upsert_claim(conn, memory_id, memory)
            await self._replace_derivations(
                conn,
                memory.tenant_id,
                memory_id,
                memory.revision_key,
                memory.derivations,
            )
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
                    first_supported_at, last_supported_at, current_revision_key
                ) VALUES (
                    :tenant_id, :memory_id, :slot_key,
                    :subject_entity_id, :predicate, :object_value, :object_entity_id,
                    :polarity, :cardinality, CAST(:qualifiers AS jsonb),
                    :status, :support_count, :contradiction_count,
                    :first_supported_at, :last_supported_at, :current_revision_key
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
                    last_supported_at = EXCLUDED.last_supported_at,
                    current_revision_key = EXCLUDED.current_revision_key
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
                "current_revision_key": memory.revision_key,
            },
        )

    async def _replace_derivations(
        self,
        conn: Any,
        tenant_id: str,
        memory_id: str,
        revision_key: str,
        derivations: tuple[SemanticDerivationInput, ...],
    ) -> None:
        await conn.execute(
            text(
                f"""
                DELETE FROM {self._table("memory_derivations")}
                WHERE revision_key = :revision_key
                """
            ),
            {"revision_key": revision_key},
        )
        for derivation in derivations:
            await conn.execute(
                text(
                    f"""
                    INSERT INTO {self._table("memory_derivations")} (
                        tenant_id, target_memory_id, source_memory_id,
                        relation, contribution_score, revision_key
                    ) VALUES (
                        :tenant_id, :target_memory_id, :source_memory_id,
                        :relation, :contribution_score, :revision_key
                    )
                    """
                ),
                {
                    "tenant_id": tenant_id,
                    "target_memory_id": memory_id,
                    "source_memory_id": derivation.episode_id,
                    "relation": derivation.relation.value,
                    "contribution_score": derivation.contribution_score,
                    "revision_key": revision_key,
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

    def _memory_params(
        self,
        memory_id: str,
        memory: SemanticMemoryInput,
        *,
        as_of: datetime | None = None,
    ) -> dict[str, Any]:
        now = as_of.astimezone(UTC) if as_of is not None else datetime.now(UTC)
        return {
            "id": memory_id,
            "tenant_id": memory.tenant_id,
            "subject_id": memory.subject_id,
            "memory_key": memory.memory_key,
            "statement": memory.statement,
            "confidence": memory.confidence,
            "importance": memory.importance,
            "valid_from": memory.valid_from,
            "valid_until": memory.valid_until,
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
        valid_at: datetime | None = None,
    ) -> builtins.list[StoredSemanticMemory]:
        if valid_at is not None:
            return await self._list_at_valid_time(
                tenant_id=tenant_id,
                subject_id=subject_id,
                valid_at=valid_at,
                limit=limit,
            )
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
        else:
            clauses.append("c.status <> 'superseded'")
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

    async def list_revisions(
        self,
        *,
        tenant_id: str,
        memory_key: str | None = None,
        subject_id: str | None = None,
        valid_at: datetime | None = None,
        limit: int | None = None,
    ) -> builtins.list[StoredSemanticRevision]:
        clauses = ["r.tenant_id = :tenant_id"]
        params: dict[str, Any] = {"tenant_id": tenant_id}
        if memory_key is not None:
            clauses.append("r.memory_key = :memory_key")
            params["memory_key"] = memory_key
        if subject_id is not None:
            clauses.append("m.subject_id = :subject_id")
            params["subject_id"] = subject_id
        limit_clause = ""
        if limit is not None:
            limit_clause = " LIMIT :limit"
            params["limit"] = limit
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT
                        r.revision_key, r.memory_key, r.tenant_id, r.revision_number,
                        r.status, r.valid_from, r.valid_until, r.confidence,
                        r.importance, r.support_count, r.contradiction_count,
                        r.first_supported_at, r.last_supported_at,
                        r.created_at, r.updated_at
                    FROM {self._table("semantic_claim_revisions")} AS r
                    JOIN {self._table("memories")} AS m
                      ON m.id = r.memory_id
                    WHERE {" AND ".join(clauses)}
                    ORDER BY r.memory_key, r.revision_number
                    {limit_clause}
                    """
                ),
                params,
            )
            rows = result.mappings().all()
        revisions: list[StoredSemanticRevision] = []
        for row in rows:
            revision = StoredSemanticRevision(
                revision_key=row["revision_key"],
                memory_key=row["memory_key"],
                tenant_id=row["tenant_id"],
                revision_number=int(row["revision_number"]),
                status=SemanticMemoryStatus(row["status"]),
                valid_from=row["valid_from"],
                valid_until=row["valid_until"],
                confidence=float(row["confidence"]),
                importance=float(row["importance"]),
                support_count=int(row["support_count"]),
                contradiction_count=int(row["contradiction_count"]),
                first_supported_at=row["first_supported_at"],
                last_supported_at=row["last_supported_at"],
                derivations=await self._load_revision_derivations(row["revision_key"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            if valid_at is not None and not revision_valid_at(revision, valid_at):
                continue
            revisions.append(revision)
        return revisions

    async def apply_reconciliation(
        self,
        plan: SemanticReconciliationPlan,
        *,
        as_of: datetime | None = None,
    ) -> SemanticReconciliationWriteResult:
        now = as_of.astimezone(UTC) if as_of is not None else datetime.now(UTC)
        created = updated = unchanged = 0
        revisions_created = revisions_updated = 0
        relations_written = 0
        async with self._engine.begin() as conn:
            memory_ids: dict[tuple[str, str], str] = {}
            for memory in plan.current_memories:
                key = (memory.tenant_id, memory.memory_key)
                existing = await conn.execute(
                    text(
                        f"""
                        SELECT m.id, m.metadata
                        FROM {self._table("memories")} AS m
                        WHERE m.tenant_id = :tenant_id
                          AND m.memory_type = 'semantic'
                          AND m.memory_key = :memory_key
                        """
                    ),
                    {"tenant_id": memory.tenant_id, "memory_key": memory.memory_key},
                )
                row = existing.mappings().first()
                fingerprint = memory.metadata["semantic"]["content_fingerprint"]
                if row is None:
                    memory_id = str(uuid4())
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
                    created += 1
                else:
                    memory_id = str(row["id"])
                    existing_metadata = row["metadata"]
                    if isinstance(existing_metadata, str):
                        existing_metadata = json.loads(existing_metadata)
                    existing_fingerprint = existing_metadata.get("semantic", {}).get(
                        "content_fingerprint"
                    )
                    if existing_fingerprint == fingerprint:
                        unchanged += 1
                    else:
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
                                "valid_from": memory.valid_from,
                                "valid_until": memory.valid_until,
                                "metadata": json.dumps(dict(memory.metadata)),
                                "now": now,
                            },
                        )
                        updated += 1
                memory_ids[key] = memory_id
                await self._upsert_claim(conn, memory_id, memory)
                await self._replace_derivations(
                    conn,
                    memory.tenant_id,
                    memory_id,
                    memory.revision_key,
                    memory.derivations,
                )
                await self._replace_evidence(conn, memory_id, memory.observation_evidence)
                await self._replace_entities(conn, memory_id, memory.entities)

            for revision in plan.revisions:
                resolved_memory_id = memory_ids.get((revision.tenant_id, revision.memory_key))
                if resolved_memory_id is None:
                    lookup = await conn.execute(
                        text(
                            f"""
                            SELECT id
                            FROM {self._table("memories")}
                            WHERE tenant_id = :tenant_id
                              AND memory_type = 'semantic'
                              AND memory_key = :memory_key
                            """
                        ),
                        {
                            "tenant_id": revision.tenant_id,
                            "memory_key": revision.memory_key,
                        },
                    )
                    found = lookup.first()
                    if found is None:
                        continue
                    resolved_memory_id = str(found[0])
                    memory_ids[(revision.tenant_id, revision.memory_key)] = resolved_memory_id
                existing_revision = await conn.execute(
                    text(
                        f"""
                        SELECT revision_key
                        FROM {self._table("semantic_claim_revisions")}
                        WHERE revision_key = :revision_key
                        """
                    ),
                    {"revision_key": revision.revision_key},
                )
                if existing_revision.first() is None:
                    revisions_created += 1
                else:
                    revisions_updated += 1
                await conn.execute(
                    text(
                        f"""
                        INSERT INTO {self._table("semantic_claim_revisions")} (
                            revision_key, tenant_id, memory_id, memory_key,
                            revision_number, status, valid_from, valid_until,
                            confidence, importance, support_count, contradiction_count,
                            first_supported_at, last_supported_at, created_at, updated_at
                        ) VALUES (
                            :revision_key, :tenant_id, :memory_id, :memory_key,
                            :revision_number, :status, :valid_from, :valid_until,
                            :confidence, :importance, :support_count, :contradiction_count,
                            :first_supported_at, :last_supported_at, :now, :now
                        )
                        ON CONFLICT (revision_key) DO UPDATE SET
                            status = EXCLUDED.status,
                            valid_from = EXCLUDED.valid_from,
                            valid_until = EXCLUDED.valid_until,
                            confidence = EXCLUDED.confidence,
                            importance = EXCLUDED.importance,
                            support_count = EXCLUDED.support_count,
                            contradiction_count = EXCLUDED.contradiction_count,
                            first_supported_at = EXCLUDED.first_supported_at,
                            last_supported_at = EXCLUDED.last_supported_at,
                            updated_at = EXCLUDED.updated_at
                        """
                    ),
                    {
                        "revision_key": revision.revision_key,
                        "tenant_id": revision.tenant_id,
                        "memory_id": resolved_memory_id,
                        "memory_key": revision.memory_key,
                        "revision_number": revision.revision_number,
                        "status": revision.status.value,
                        "valid_from": revision.valid_from,
                        "valid_until": revision.valid_until,
                        "confidence": revision.confidence,
                        "importance": revision.importance,
                        "support_count": revision.support_count,
                        "contradiction_count": revision.contradiction_count,
                        "first_supported_at": revision.first_supported_at,
                        "last_supported_at": revision.last_supported_at,
                        "now": now,
                    },
                )
                await self._replace_derivations(
                    conn,
                    revision.tenant_id,
                    resolved_memory_id,
                    revision.revision_key,
                    revision.derivations,
                )

            for relation in plan.relations:
                result = await conn.execute(
                    text(
                        f"""
                        INSERT INTO {self._table("semantic_revision_relations")} (
                            tenant_id, left_revision_key, right_revision_key,
                            relation, effective_at
                        ) VALUES (
                            :tenant_id, :left_revision_key, :right_revision_key,
                            :relation, :effective_at
                        )
                        ON CONFLICT DO NOTHING
                        """
                    ),
                    {
                        "tenant_id": relation.tenant_id,
                        "left_revision_key": relation.left_revision_key,
                        "right_revision_key": relation.right_revision_key,
                        "relation": relation.relation.value,
                        "effective_at": relation.effective_at,
                    },
                )
                if result.rowcount:
                    relations_written += 1

        return SemanticReconciliationWriteResult(
            created=created,
            updated=updated,
            unchanged=unchanged,
            revisions_created=revisions_created,
            revisions_updated=revisions_updated,
            relations_written=relations_written,
        )

    async def _list_at_valid_time(
        self,
        *,
        tenant_id: str,
        subject_id: str | None,
        valid_at: datetime,
        limit: int | None,
    ) -> builtins.list[StoredSemanticMemory]:
        revisions = await self.list_revisions(
            tenant_id=tenant_id,
            subject_id=subject_id,
            valid_at=valid_at,
        )
        results: list[StoredSemanticMemory] = []
        for revision in revisions:
            memory = await self._get_by_memory_key(
                tenant_id=tenant_id,
                memory_key=revision.memory_key,
            )
            if memory is None:
                continue
            results.append(
                StoredSemanticMemory(
                    id=memory.id,
                    tenant_id=memory.tenant_id,
                    subject_id=memory.subject_id,
                    memory_key=memory.memory_key,
                    slot_key=memory.slot_key,
                    revision_key=revision.revision_key,
                    revision_number=revision.revision_number,
                    statement=memory.statement,
                    subject_entity_id=memory.subject_entity_id,
                    predicate=memory.predicate,
                    object_value=memory.object_value,
                    object_entity_id=memory.object_entity_id,
                    polarity=memory.polarity,
                    cardinality=memory.cardinality,
                    qualifiers=memory.qualifiers,
                    confidence=revision.confidence,
                    importance=revision.importance,
                    status=revision.status,
                    support_count=revision.support_count,
                    contradiction_count=revision.contradiction_count,
                    first_supported_at=revision.first_supported_at,
                    last_supported_at=revision.last_supported_at,
                    valid_from=revision.valid_from,
                    valid_until=revision.valid_until,
                    is_active=True,
                    derivations=revision.derivations,
                    observation_evidence=memory.observation_evidence,
                    entities=memory.entities,
                    metadata=memory.metadata,
                    created_at=memory.created_at,
                    updated_at=memory.updated_at,
                )
            )
        results.sort(key=lambda item: (item.first_supported_at, item.id))
        if limit is not None:
            return results[:limit]
        return results

    async def _load_revision_derivations(
        self,
        revision_key: str,
    ) -> tuple[SemanticDerivationInput, ...]:
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT source_memory_id, relation, contribution_score
                    FROM {self._table("memory_derivations")}
                    WHERE revision_key = :revision_key
                    ORDER BY source_memory_id, relation
                    """
                ),
                {"revision_key": revision_key},
            )
            rows = result.mappings().all()
        return tuple(
            SemanticDerivationInput(
                episode_id=str(row["source_memory_id"]),
                relation=SemanticDerivationRelation(row["relation"]),
                contribution_score=float(row["contribution_score"]),
            )
            for row in rows
        )

    async def deactivate_missing(
        self,
        *,
        tenant_id: str,
        subject_id: str | None,
        active_memory_keys: set[str],
        as_of: datetime | None = None,
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

        now = as_of.astimezone(UTC) if as_of is not None else datetime.now(UTC)
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
                    {"id": str(row["id"]), "now": now},
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
            revision_key=row.get("current_revision_key") or f"legacy:{row['id']}",
            revision_number=int(row.get("revision_number") or 1),
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
            valid_from=row["valid_from"],
            valid_until=row["valid_until"],
            is_active=row["is_active"],
            derivations=derivations,
            observation_evidence=evidence,
            entities=entities,
            metadata=MappingProxyType(dict(metadata)),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


class PostgresActivationStore(ActivationStore):
    """PostgreSQL-backed activation reference store."""

    def __init__(self, engine: AsyncEngine, *, schema: str = "cogkura") -> None:
        self._engine = engine
        self._schema = schema

    def _table(self, name: str) -> str:
        return f"{self._schema}.{name}"

    async def append_references(self, references: Sequence[MemoryReference]) -> None:
        if not references:
            return
        async with self._engine.begin() as conn:
            for reference in references:
                metadata_json = json.dumps(dict(reference.metadata))
                await conn.execute(
                    text(
                        f"""
                        INSERT INTO {self._table("memory_activation_references")} (
                            id, tenant_id, memory_kind, memory_key,
                            reference_kind, referenced_at, request_id,
                            weight, metadata, created_at
                        ) VALUES (
                            :id, :tenant_id, :memory_kind, :memory_key,
                            :reference_kind, :referenced_at, :request_id,
                            :weight, CAST(:metadata AS jsonb), :created_at
                        )
                        ON CONFLICT DO NOTHING
                        """
                    ),
                    {
                        "id": str(uuid4()),
                        "tenant_id": reference.tenant_id,
                        "memory_kind": reference.memory_kind.value,
                        "memory_key": reference.memory_key,
                        "reference_kind": reference.reference_kind.value,
                        "referenced_at": reference.referenced_at,
                        "request_id": reference.request_id,
                        "weight": reference.weight,
                        "metadata": metadata_json,
                        "created_at": datetime.now(UTC),
                    },
                )

    async def list_reference_traces(
        self,
        *,
        tenant_id: str,
        identities: Sequence[MemoryIdentity],
        before_or_at: datetime,
    ) -> Mapping[MemoryIdentity, tuple[ActivationReferenceTrace, ...]]:
        if not identities:
            return {}
        kinds = [identity.memory_kind.value for identity in identities]
        keys = [identity.memory_key for identity in identities]
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT memory_kind, memory_key, referenced_at, weight
                    FROM {self._table("memory_activation_references")}
                    WHERE tenant_id = :tenant_id
                      AND referenced_at <= :before_or_at
                      AND (memory_kind, memory_key) IN (
                          SELECT * FROM unnest(
                              CAST(:kinds AS text[]),
                              CAST(:keys AS text[])
                          )
                      )
                    ORDER BY referenced_at
                    """
                ),
                {
                    "tenant_id": tenant_id,
                    "before_or_at": before_or_at,
                    "kinds": kinds,
                    "keys": keys,
                },
            )
            rows = result.mappings().all()
        grouped: dict[MemoryIdentity, list[ActivationReferenceTrace]] = {}
        for row in rows:
            identity = MemoryIdentity(
                memory_kind=MemoryKind(row["memory_kind"]),
                memory_key=row["memory_key"],
            )
            grouped.setdefault(identity, []).append(
                ActivationReferenceTrace(
                    referenced_at=row["referenced_at"],
                    weight=int(row["weight"]),
                )
            )
        return {identity: tuple(traces) for identity, traces in grouped.items()}

    async def compact_references(
        self,
        *,
        tenant_id: str,
        before: datetime,
        bucket_seconds: float,
    ) -> ReferenceCompactionResult:
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT id, memory_kind, memory_key, reference_kind,
                           referenced_at, weight, metadata
                    FROM {self._table("memory_activation_references")}
                    WHERE tenant_id = :tenant_id
                      AND referenced_at < :before
                    ORDER BY referenced_at
                    """
                ),
                {"tenant_id": tenant_id, "before": before},
            )
            rows = result.mappings().all()

        if not rows:
            return ReferenceCompactionResult(references_compacted=0)

        buckets: dict[
            tuple[str, str, str, datetime],
            list[dict[str, Any]],
        ] = {}
        for row in rows:
            referenced_at = row["referenced_at"].astimezone(UTC)
            bucket_start = _postgres_bucket_start(referenced_at, bucket_seconds)
            key = (
                row["memory_kind"],
                row["memory_key"],
                row["reference_kind"],
                bucket_start,
            )
            buckets.setdefault(key, []).append(dict(row))

        compacted_count = 0
        async with self._engine.begin() as conn:
            for references in buckets.values():
                if len(references) == 1 and int(references[0]["weight"]) == 1:
                    continue
                compacted_count += len(references)
                for reference in references:
                    await conn.execute(
                        text(
                            f"""
                            DELETE FROM {self._table("memory_activation_references")}
                            WHERE id = :id
                            """
                        ),
                        {"id": str(reference["id"])},
                    )
                total_weight = sum(int(reference["weight"]) for reference in references)
                bucket_start = _postgres_bucket_start(
                    references[0]["referenced_at"].astimezone(UTC),
                    bucket_seconds,
                )
                representative_at = compaction_representative_time(
                    [
                        MemoryReference(
                            tenant_id=tenant_id,
                            memory_kind=MemoryKind(reference["memory_kind"]),
                            memory_key=reference["memory_key"],
                            reference_kind=ActivationReferenceKind(reference["reference_kind"]),
                            referenced_at=reference["referenced_at"],
                            weight=int(reference["weight"]),
                        )
                        for reference in references
                    ],
                    bucket_start=bucket_start,
                    as_of=before.astimezone(UTC),
                )
                metadata_json = json.dumps(dict(references[0]["metadata"]))
                await conn.execute(
                    text(
                        f"""
                        INSERT INTO {self._table("memory_activation_references")} (
                            id, tenant_id, memory_kind, memory_key,
                            reference_kind, referenced_at, request_id,
                            weight, metadata, created_at
                        ) VALUES (
                            :id, :tenant_id, :memory_kind, :memory_key,
                            :reference_kind, :referenced_at, NULL,
                            :weight, CAST(:metadata AS jsonb), :created_at
                        )
                        """
                    ),
                    {
                        "id": str(uuid4()),
                        "tenant_id": tenant_id,
                        "memory_kind": references[0]["memory_kind"],
                        "memory_key": references[0]["memory_key"],
                        "reference_kind": references[0]["reference_kind"],
                        "referenced_at": representative_at,
                        "weight": total_weight,
                        "metadata": metadata_json,
                        "created_at": datetime.now(UTC),
                    },
                )

        return ReferenceCompactionResult(references_compacted=compacted_count)

    async def clear(self, *, tenant_id: str) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    DELETE FROM {self._table("memory_activation_references")}
                    WHERE tenant_id = :tenant_id
                    """
                ),
                {"tenant_id": tenant_id},
            )


def _postgres_bucket_start(referenced_at: datetime, bucket_seconds: float) -> datetime:
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    elapsed = (referenced_at.astimezone(UTC) - epoch).total_seconds()
    bucket_index = int(elapsed // bucket_seconds)
    return epoch + timedelta(seconds=bucket_index * bucket_seconds)


class PostgresMemoryDynamicsStore(MemoryDynamicsStore):
    """PostgreSQL-backed memory dynamics store."""

    def __init__(self, engine: AsyncEngine, *, schema: str = "cogkura") -> None:
        self._engine = engine
        self._schema = schema

    def _table(self, name: str) -> str:
        return f"{self._schema}.{name}"

    async def get_many(
        self,
        *,
        tenant_id: str,
        identities: Sequence[MemoryIdentity],
    ) -> Mapping[MemoryIdentity, StoredMemoryDynamics]:
        if not identities:
            return {}
        kinds = [identity.memory_kind.value for identity in identities]
        keys = [identity.memory_key for identity in identities]
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT *
                    FROM {self._table("memory_dynamics")}
                    WHERE tenant_id = :tenant_id
                      AND (memory_kind, memory_key) IN (
                          SELECT * FROM unnest(
                              CAST(:kinds AS text[]),
                              CAST(:keys AS text[])
                          )
                      )
                    """
                ),
                {"tenant_id": tenant_id, "kinds": kinds, "keys": keys},
            )
            rows = result.mappings().all()
        return {
            MemoryIdentity(
                memory_kind=MemoryKind(row["memory_kind"]),
                memory_key=row["memory_key"],
            ): _dynamics_from_row(dict(row))
            for row in rows
        }

    async def upsert_many(self, dynamics: Sequence[StoredMemoryDynamics]) -> None:
        if not dynamics:
            return
        async with self._engine.begin() as conn:
            for record in dynamics:
                await conn.execute(
                    text(
                        f"""
                        INSERT INTO {self._table("memory_dynamics")} (
                            tenant_id, memory_kind, memory_key,
                            retention_state, last_base_level,
                            last_retention_score, below_threshold_since,
                            forgotten_at, evaluated_at, updated_at
                        ) VALUES (
                            :tenant_id, :memory_kind, :memory_key,
                            :retention_state, :last_base_level,
                            :last_retention_score, :below_threshold_since,
                            :forgotten_at, :evaluated_at, :updated_at
                        )
                        ON CONFLICT (tenant_id, memory_kind, memory_key)
                        DO UPDATE SET
                            retention_state = EXCLUDED.retention_state,
                            last_base_level = EXCLUDED.last_base_level,
                            last_retention_score = EXCLUDED.last_retention_score,
                            below_threshold_since = EXCLUDED.below_threshold_since,
                            forgotten_at = EXCLUDED.forgotten_at,
                            evaluated_at = EXCLUDED.evaluated_at,
                            updated_at = EXCLUDED.updated_at
                        """
                    ),
                    {
                        "tenant_id": record.tenant_id,
                        "memory_kind": record.memory_kind.value,
                        "memory_key": record.memory_key,
                        "retention_state": record.retention_state.value,
                        "last_base_level": record.last_base_level,
                        "last_retention_score": record.last_retention_score,
                        "below_threshold_since": record.below_threshold_since,
                        "forgotten_at": record.forgotten_at,
                        "evaluated_at": record.evaluated_at,
                        "updated_at": record.updated_at,
                    },
                )

    async def reactivate(
        self,
        *,
        tenant_id: str,
        identities: Sequence[MemoryIdentity],
        at: datetime,
    ) -> None:
        if not identities:
            return
        kinds = [identity.memory_kind.value for identity in identities]
        keys = [identity.memory_key for identity in identities]
        timestamp = at.astimezone(UTC)
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    UPDATE {self._table("memory_dynamics")}
                    SET retention_state = :active_state,
                        below_threshold_since = NULL,
                        forgotten_at = NULL,
                        evaluated_at = :at,
                        updated_at = :at
                    WHERE tenant_id = :tenant_id
                      AND (memory_kind, memory_key) IN (
                          SELECT * FROM unnest(
                              CAST(:kinds AS text[]),
                              CAST(:keys AS text[])
                          )
                      )
                    """
                ),
                {
                    "tenant_id": tenant_id,
                    "kinds": kinds,
                    "keys": keys,
                    "active_state": MemoryRetentionState.ACTIVE.value,
                    "at": timestamp,
                },
            )

    async def clear(self, *, tenant_id: str) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    DELETE FROM {self._table("memory_dynamics")}
                    WHERE tenant_id = :tenant_id
                    """
                ),
                {"tenant_id": tenant_id},
            )


def _dynamics_from_row(row: Mapping[str, Any]) -> StoredMemoryDynamics:
    return StoredMemoryDynamics(
        tenant_id=row["tenant_id"],
        memory_kind=MemoryKind(row["memory_kind"]),
        memory_key=row["memory_key"],
        retention_state=MemoryRetentionState(row["retention_state"]),
        last_base_level=float(row["last_base_level"]),
        last_retention_score=float(row["last_retention_score"]),
        below_threshold_since=row["below_threshold_since"],
        forgotten_at=row["forgotten_at"],
        evaluated_at=row["evaluated_at"],
        updated_at=row["updated_at"],
    )


class PostgresLearningStore(LearningStore):
    """PostgreSQL-backed learning and reinforcement store."""

    def __init__(self, engine: AsyncEngine, *, schema: str = "cogkura") -> None:
        self._engine = engine
        self._schema = schema

    def _table(self, name: str) -> str:
        return f"{self._schema}.{name}"

    async def apply(self, plan: LearningPlan) -> LearningWriteResult:
        async with self._engine.begin() as conn:
            existing = await conn.execute(
                text(
                    f"""
                    SELECT feedback_fingerprint
                    FROM {self._table("memory_learning_events")}
                    WHERE tenant_id = :tenant_id
                      AND feedback_id = :feedback_id
                    """
                ),
                {"tenant_id": plan.tenant_id, "feedback_id": plan.feedback_id},
            )
            row = existing.first()
            if row is not None:
                if row[0] == plan.feedback_fingerprint:
                    return LearningWriteResult(
                        created=False,
                        unchanged=True,
                        helpful=0,
                        unhelpful=0,
                        incorrect=0,
                        associations_reinforced=0,
                    )
                raise StorageError(
                    f"Conflicting feedback fingerprint for feedback_id {plan.feedback_id!r}."
                )

            helpful = unhelpful = incorrect = 0
            for item in plan.items:
                if item.outcome is LearningOutcome.HELPFUL:
                    helpful += 1
                elif item.outcome is LearningOutcome.UNHELPFUL:
                    unhelpful += 1
                else:
                    incorrect += 1

            await conn.execute(
                text(
                    f"""
                    INSERT INTO {self._table("memory_learning_events")} (
                        tenant_id, feedback_id, feedback_fingerprint,
                        subject_id, context_key, occurred_at, metadata, created_at
                    ) VALUES (
                        :tenant_id, :feedback_id, :feedback_fingerprint,
                        :subject_id, :context_key, :occurred_at,
                        CAST(:metadata AS jsonb), :created_at
                    )
                    """
                ),
                {
                    "tenant_id": plan.tenant_id,
                    "feedback_id": plan.feedback_id,
                    "feedback_fingerprint": plan.feedback_fingerprint,
                    "subject_id": plan.subject_id,
                    "context_key": plan.context_key,
                    "occurred_at": plan.occurred_at,
                    "metadata": json.dumps(dict(plan.metadata)),
                    "created_at": datetime.now(UTC),
                },
            )

            for item in plan.items:
                await conn.execute(
                    text(
                        f"""
                        INSERT INTO {self._table("memory_learning_feedback")} (
                            tenant_id, feedback_id, memory_kind, memory_key,
                            revision_key, outcome, metadata, created_at
                        ) VALUES (
                            :tenant_id, :feedback_id, :memory_kind, :memory_key,
                            :revision_key, :outcome, CAST(:metadata AS jsonb), :created_at
                        )
                        """
                    ),
                    {
                        "tenant_id": plan.tenant_id,
                        "feedback_id": plan.feedback_id,
                        "memory_kind": item.identity.memory_kind.value,
                        "memory_key": item.identity.memory_key,
                        "revision_key": item.revision_key,
                        "outcome": item.outcome.value,
                        "metadata": json.dumps(dict(item.metadata)),
                        "created_at": datetime.now(UTC),
                    },
                )
                await self._increment_state(
                    conn,
                    tenant_id=plan.tenant_id,
                    context_key=plan.context_key,
                    identity=item.identity,
                    outcome=item.outcome,
                    at=plan.occurred_at,
                )

            associations_reinforced = 0
            for left, right in plan.association_pairs:
                await self._increment_association(
                    conn,
                    tenant_id=plan.tenant_id,
                    left=left,
                    right=right,
                    at=plan.occurred_at,
                )
                associations_reinforced += 1

        return LearningWriteResult(
            created=True,
            unchanged=False,
            helpful=helpful,
            unhelpful=unhelpful,
            incorrect=incorrect,
            associations_reinforced=associations_reinforced,
        )

    async def _increment_state(
        self,
        conn: Any,
        *,
        tenant_id: str,
        context_key: str,
        identity: MemoryIdentity,
        outcome: LearningOutcome,
        at: datetime,
    ) -> None:
        timestamp = at.astimezone(UTC)
        helpful_delta = 1 if outcome is LearningOutcome.HELPFUL else 0
        unhelpful_delta = 1 if outcome is LearningOutcome.UNHELPFUL else 0
        incorrect_delta = 1 if outcome is LearningOutcome.INCORRECT else 0
        await conn.execute(
            text(
                f"""
                INSERT INTO {self._table("memory_learning_state")} (
                    tenant_id, context_key, memory_kind, memory_key,
                    helpful_count, unhelpful_count, incorrect_count,
                    first_feedback_at, last_feedback_at, updated_at
                ) VALUES (
                    :tenant_id, :context_key, :memory_kind, :memory_key,
                    :helpful_count, :unhelpful_count, :incorrect_count,
                    :timestamp, :timestamp, :timestamp
                )
                ON CONFLICT (tenant_id, context_key, memory_kind, memory_key)
                DO UPDATE SET
                    helpful_count = {self._table("memory_learning_state")}.helpful_count
                        + EXCLUDED.helpful_count,
                    unhelpful_count = {self._table("memory_learning_state")}.unhelpful_count
                        + EXCLUDED.unhelpful_count,
                    incorrect_count = {self._table("memory_learning_state")}.incorrect_count
                        + EXCLUDED.incorrect_count,
                    last_feedback_at = EXCLUDED.last_feedback_at,
                    updated_at = EXCLUDED.updated_at
                """
            ),
            {
                "tenant_id": tenant_id,
                "context_key": context_key,
                "memory_kind": identity.memory_kind.value,
                "memory_key": identity.memory_key,
                "helpful_count": helpful_delta,
                "unhelpful_count": unhelpful_delta,
                "incorrect_count": incorrect_delta,
                "timestamp": timestamp,
            },
        )

    async def _increment_association(
        self,
        conn: Any,
        *,
        tenant_id: str,
        left: MemoryIdentity,
        right: MemoryIdentity,
        at: datetime,
    ) -> None:
        timestamp = at.astimezone(UTC)
        await conn.execute(
            text(
                f"""
                INSERT INTO {self._table("memory_learned_associations")} (
                    tenant_id,
                    left_memory_kind, left_memory_key,
                    right_memory_kind, right_memory_key,
                    coactivation_count,
                    first_reinforced_at, last_reinforced_at, updated_at
                ) VALUES (
                    :tenant_id,
                    :left_memory_kind, :left_memory_key,
                    :right_memory_kind, :right_memory_key,
                    1,
                    :timestamp, :timestamp, :timestamp
                )
                ON CONFLICT (
                    tenant_id,
                    left_memory_kind,
                    left_memory_key,
                    right_memory_kind,
                    right_memory_key
                )
                DO UPDATE SET
                    coactivation_count = (
                        {self._table("memory_learned_associations")}.coactivation_count + 1
                    ),
                    last_reinforced_at = EXCLUDED.last_reinforced_at,
                    updated_at = EXCLUDED.updated_at
                """
            ),
            {
                "tenant_id": tenant_id,
                "left_memory_kind": left.memory_kind.value,
                "left_memory_key": left.memory_key,
                "right_memory_kind": right.memory_kind.value,
                "right_memory_key": right.memory_key,
                "timestamp": timestamp,
            },
        )

    async def list_states(
        self,
        *,
        tenant_id: str,
        identities: Sequence[MemoryIdentity],
        context_keys: Sequence[str],
    ) -> Sequence[StoredMemoryLearningState]:
        if not identities or not context_keys:
            return ()
        kinds = [identity.memory_kind.value for identity in identities]
        keys = [identity.memory_key for identity in identities]
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT tenant_id, context_key, memory_kind, memory_key,
                           helpful_count, unhelpful_count, incorrect_count,
                           first_feedback_at, last_feedback_at, updated_at
                    FROM {self._table("memory_learning_state")}
                    WHERE tenant_id = :tenant_id
                      AND context_key = ANY(CAST(:context_keys AS text[]))
                      AND (memory_kind, memory_key) IN (
                          SELECT * FROM unnest(
                              CAST(:kinds AS text[]),
                              CAST(:keys AS text[])
                          )
                      )
                    """
                ),
                {
                    "tenant_id": tenant_id,
                    "context_keys": list(context_keys),
                    "kinds": kinds,
                    "keys": keys,
                },
            )
            rows = result.mappings().all()
        return tuple(_learning_state_from_row(cast(Mapping[str, Any], row)) for row in rows)

    async def list_associations(
        self,
        *,
        tenant_id: str,
        identities: Sequence[MemoryIdentity],
    ) -> Sequence[StoredMemoryAssociation]:
        if not identities:
            return ()
        kinds = [identity.memory_kind.value for identity in identities]
        keys = [identity.memory_key for identity in identities]
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT tenant_id,
                           left_memory_kind, left_memory_key,
                           right_memory_kind, right_memory_key,
                           coactivation_count,
                           first_reinforced_at, last_reinforced_at, updated_at
                    FROM {self._table("memory_learned_associations")}
                    WHERE tenant_id = :tenant_id
                      AND left_memory_kind = ANY(CAST(:kinds AS text[]))
                      AND left_memory_key = ANY(CAST(:keys AS text[]))
                      AND right_memory_kind = ANY(CAST(:kinds AS text[]))
                      AND right_memory_key = ANY(CAST(:keys AS text[]))
                    """
                ),
                {"tenant_id": tenant_id, "kinds": kinds, "keys": keys},
            )
            rows = result.mappings().all()
        associations = tuple(_association_from_row(cast(Mapping[str, Any], row)) for row in rows)
        identity_set = set(identities)
        return tuple(
            association
            for association in associations
            if association.left in identity_set and association.right in identity_set
        )

    async def list_reinforcement_traces(
        self,
        *,
        tenant_id: str,
        identities: Sequence[MemoryIdentity],
        before_or_at: datetime,
    ) -> Mapping[MemoryIdentity, tuple[ActivationReferenceTrace, ...]]:
        if not identities:
            return {}
        kinds = [identity.memory_kind.value for identity in identities]
        keys = [identity.memory_key for identity in identities]
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(
                    f"""
                    SELECT feedback.memory_kind, feedback.memory_key, events.occurred_at
                    FROM {self._table("memory_learning_feedback")} AS feedback
                    JOIN {self._table("memory_learning_events")} AS events
                      ON events.tenant_id = feedback.tenant_id
                     AND events.feedback_id = feedback.feedback_id
                    WHERE feedback.tenant_id = :tenant_id
                      AND feedback.outcome = :helpful
                      AND events.occurred_at <= :before_or_at
                      AND (feedback.memory_kind, feedback.memory_key) IN (
                          SELECT * FROM unnest(
                              CAST(:kinds AS text[]),
                              CAST(:keys AS text[])
                          )
                      )
                    ORDER BY events.occurred_at
                    """
                ),
                {
                    "tenant_id": tenant_id,
                    "helpful": LearningOutcome.HELPFUL.value,
                    "before_or_at": before_or_at,
                    "kinds": kinds,
                    "keys": keys,
                },
            )
            rows = result.mappings().all()
        grouped: dict[MemoryIdentity, list[ActivationReferenceTrace]] = {}
        for row in rows:
            identity = MemoryIdentity(
                memory_kind=MemoryKind(row["memory_kind"]),
                memory_key=row["memory_key"],
            )
            grouped.setdefault(identity, []).append(
                ActivationReferenceTrace(referenced_at=row["occurred_at"], weight=1)
            )
        return {identity: tuple(traces) for identity, traces in grouped.items()}

    async def clear(self, *, tenant_id: str) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"""
                    DELETE FROM {self._table("memory_learning_events")}
                    WHERE tenant_id = :tenant_id
                    """
                ),
                {"tenant_id": tenant_id},
            )


def _learning_state_from_row(row: Mapping[str, Any]) -> StoredMemoryLearningState:
    return StoredMemoryLearningState(
        tenant_id=row["tenant_id"],
        context_key=row["context_key"],
        memory_kind=MemoryKind(row["memory_kind"]),
        memory_key=row["memory_key"],
        helpful_count=int(row["helpful_count"]),
        unhelpful_count=int(row["unhelpful_count"]),
        incorrect_count=int(row["incorrect_count"]),
        first_feedback_at=row["first_feedback_at"],
        last_feedback_at=row["last_feedback_at"],
        updated_at=row["updated_at"],
    )


def _association_from_row(row: Mapping[str, Any]) -> StoredMemoryAssociation:
    return StoredMemoryAssociation(
        tenant_id=row["tenant_id"],
        left=MemoryIdentity(
            memory_kind=MemoryKind(row["left_memory_kind"]),
            memory_key=row["left_memory_key"],
        ),
        right=MemoryIdentity(
            memory_kind=MemoryKind(row["right_memory_kind"]),
            memory_key=row["right_memory_key"],
        ),
        coactivation_count=int(row["coactivation_count"]),
        first_reinforced_at=row["first_reinforced_at"],
        last_reinforced_at=row["last_reinforced_at"],
        updated_at=row["updated_at"],
    )
