"""In-memory observation store for tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from cognema.observations.models import IngestStatus, ObservationInput, StoredObservation
from cognema.observations.retention import RetainedObservation
from cognema.storage.base import CheckpointStore, ObservationStore


@dataclass(slots=True)
class _Revision:
    revision_number: int
    source_version: str | None
    content: str | None
    content_hash: str
    metadata: dict[str, Any]
    change_type: str
    observed_at: Any


class InMemoryObservationStore(ObservationStore):
    """In-memory observation store with revision tracking."""

    def __init__(self) -> None:
        self._observations: dict[tuple[str, str, str], StoredObservation] = {}
        self._revisions: dict[str, list[_Revision]] = {}

    @property
    def revisions(self) -> dict[str, list[_Revision]]:
        return self._revisions

    def _key(
        self,
        tenant_id: str,
        source_namespace: str,
        source_record_id: str,
    ) -> tuple[str, str, str]:
        return (tenant_id, source_namespace, source_record_id)

    async def ingest(
        self,
        observation: ObservationInput,
        *,
        retained: RetainedObservation,
    ) -> IngestStatus:
        key = self._key(
            observation.tenant_id,
            observation.source_namespace,
            observation.source_record_id,
        )
        existing = self._observations.get(key)

        if existing is None:
            obs_id = str(uuid4())
            stored = StoredObservation(
                id=obs_id,
                tenant_id=observation.tenant_id,
                subject_id=observation.subject_id,
                actor_id=observation.actor_id,
                source_type=observation.source_type,
                source_namespace=observation.source_namespace,
                source_record_id=observation.source_record_id,
                source_version=observation.source_version,
                event_type=observation.event_type,
                content=retained.content,
                content_hash=retained.content_hash,
                metadata=MappingProxyType(dict(retained.metadata)),
                source_created_at=observation.source_created_at,
                source_updated_at=observation.source_updated_at,
                observed_at=observation.observed_at.astimezone(UTC),
                current_revision=1,
                is_deleted=observation.is_deleted,
            )
            self._observations[key] = stored
            self._revisions[obs_id] = [
                _Revision(
                    revision_number=1,
                    source_version=observation.source_version,
                    content=retained.content,
                    content_hash=retained.content_hash,
                    metadata=dict(retained.metadata),
                    change_type="created",
                    observed_at=observation.observed_at,
                )
            ]
            return IngestStatus.CREATED

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
        stored = StoredObservation(
            id=existing.id,
            tenant_id=observation.tenant_id,
            subject_id=observation.subject_id,
            actor_id=observation.actor_id,
            source_type=observation.source_type,
            source_namespace=observation.source_namespace,
            source_record_id=observation.source_record_id,
            source_version=observation.source_version,
            event_type=observation.event_type,
            content=retained.content,
            content_hash=retained.content_hash,
            metadata=MappingProxyType(dict(retained.metadata)),
            source_created_at=observation.source_created_at,
            source_updated_at=observation.source_updated_at,
            observed_at=observation.observed_at.astimezone(UTC),
            current_revision=revision_number,
            is_deleted=observation.is_deleted,
        )
        self._observations[key] = stored
        self._revisions[existing.id].append(
            _Revision(
                revision_number=revision_number,
                source_version=observation.source_version,
                content=retained.content,
                content_hash=retained.content_hash,
                metadata=dict(retained.metadata),
                change_type=change_type,
                observed_at=observation.observed_at,
            )
        )
        return status

    async def get_by_source(
        self,
        *,
        tenant_id: str,
        source_namespace: str,
        source_record_id: str,
    ) -> StoredObservation | None:
        return self._observations.get(self._key(tenant_id, source_namespace, source_record_id))


class InMemoryCheckpointStore(CheckpointStore):
    """In-memory checkpoint store for tests."""

    def __init__(self) -> None:
        self._checkpoints: dict[tuple[str, str], dict[str, Any]] = {}

    async def get(
        self,
        *,
        tenant_id: str,
        connector_id: str,
    ) -> dict[str, Any] | None:
        return self._checkpoints.get((tenant_id, connector_id))

    async def set(
        self,
        *,
        tenant_id: str,
        connector_id: str,
        checkpoint: dict[str, Any],
    ) -> None:
        self._checkpoints[(tenant_id, connector_id)] = dict(checkpoint)
