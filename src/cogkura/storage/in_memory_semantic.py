"""In-memory semantic memory store for tests."""

from __future__ import annotations

from datetime import UTC, datetime
from types import MappingProxyType
from uuid import uuid4

from cogkura.algorithms.reconsolidation import revision_valid_at
from cogkura.models import (
    SemanticMemoryInput,
    SemanticMemoryStatus,
    SemanticReconciliationPlan,
    SemanticReconciliationWriteResult,
    SemanticWriteStatus,
    StoredSemanticMemory,
    StoredSemanticRevision,
)
from cogkura.storage.base import SemanticMemoryStore

_List = list  # avoid shadowing by SemanticMemoryStore.list


class InMemorySemanticMemoryStore(SemanticMemoryStore):
    """In-memory semantic memory store."""

    def __init__(self) -> None:
        self._memories: dict[tuple[str, str], StoredSemanticMemory] = {}
        self._revisions: dict[tuple[str, str], StoredSemanticRevision] = {}
        self._relations: dict[tuple[str, str, str, str], tuple[str, str, str, str | None]] = {}

    def _memory_key(self, tenant_id: str, memory_key: str) -> tuple[str, str]:
        return (tenant_id, memory_key)

    def _revision_key(self, tenant_id: str, revision_key: str) -> tuple[str, str]:
        return (tenant_id, revision_key)

    async def upsert(
        self,
        memory: SemanticMemoryInput,
        *,
        as_of: datetime | None = None,
    ) -> SemanticWriteStatus:
        key = self._memory_key(memory.tenant_id, memory.memory_key)
        existing = self._memories.get(key)
        now = as_of.astimezone(UTC) if as_of is not None else datetime.now(UTC)
        fingerprint = memory.metadata["semantic"]["content_fingerprint"]
        if existing is not None:
            existing_fingerprint = existing.metadata["semantic"]["content_fingerprint"]
            if existing_fingerprint == fingerprint:
                return SemanticWriteStatus.UNCHANGED
            stored = _stored_from_input(
                memory, memory_id=existing.id, created_at=existing.created_at, now=now
            )
            self._memories[key] = stored
            return SemanticWriteStatus.UPDATED

        stored = _stored_from_input(memory, memory_id=str(uuid4()), created_at=now, now=now)
        self._memories[key] = stored
        return SemanticWriteStatus.CREATED

    async def list(
        self,
        *,
        tenant_id: str,
        subject_id: str | None = None,
        include_inactive: bool = False,
        status: SemanticMemoryStatus | None = None,
        limit: int | None = None,
        valid_at: datetime | None = None,
    ) -> _List[StoredSemanticMemory]:
        if valid_at is not None:
            return await self._list_at_valid_time(
                tenant_id=tenant_id,
                subject_id=subject_id,
                valid_at=valid_at,
                limit=limit,
            )
        results: list[StoredSemanticMemory] = []
        for memory in self._memories.values():
            if memory.tenant_id != tenant_id:
                continue
            if subject_id is not None and memory.subject_id != subject_id:
                continue
            if not include_inactive and not memory.is_active:
                continue
            if status is not None and memory.status != status:
                continue
            if (
                memory.status is SemanticMemoryStatus.SUPERSEDED
                and status is not SemanticMemoryStatus.SUPERSEDED
            ):
                continue
            results.append(memory)
        results.sort(key=lambda item: (item.first_supported_at, item.id))
        if limit is not None:
            return results[:limit]
        return results

    async def list_revisions(
        self,
        *,
        tenant_id: str,
        memory_key: str | None = None,
        subject_id: str | None = None,
        valid_at: datetime | None = None,
        limit: int | None = None,
    ) -> _List[StoredSemanticRevision]:
        results: list[StoredSemanticRevision] = []
        for revision in self._revisions.values():
            if revision.tenant_id != tenant_id:
                continue
            if memory_key is not None and revision.memory_key != memory_key:
                continue
            if valid_at is not None and not revision_valid_at(revision, valid_at):
                continue
            if subject_id is not None:
                memory = self._memories.get(self._memory_key(tenant_id, revision.memory_key))
                if memory is None or memory.subject_id != subject_id:
                    continue
            results.append(revision)
        results.sort(key=lambda item: (item.memory_key, item.revision_number))
        if limit is not None:
            return results[:limit]
        return results

    async def apply_reconciliation(
        self,
        plan: SemanticReconciliationPlan,
        *,
        as_of: datetime | None = None,
    ) -> SemanticReconciliationWriteResult:
        memories = dict(self._memories)
        revisions = dict(self._revisions)
        relations = dict(self._relations)
        now = as_of.astimezone(UTC) if as_of is not None else datetime.now(UTC)
        created = 0
        updated = 0
        unchanged = 0
        revisions_created = 0
        revisions_updated = 0
        relations_written = 0

        for revision in plan.revisions:
            key = self._revision_key(revision.tenant_id, revision.revision_key)
            existing = revisions.get(key)
            stored_revision = StoredSemanticRevision(
                revision_key=revision.revision_key,
                memory_key=revision.memory_key,
                tenant_id=revision.tenant_id,
                revision_number=revision.revision_number,
                status=revision.status,
                valid_from=revision.valid_from,
                valid_until=revision.valid_until,
                confidence=revision.confidence,
                importance=revision.importance,
                support_count=revision.support_count,
                contradiction_count=revision.contradiction_count,
                first_supported_at=revision.first_supported_at,
                last_supported_at=revision.last_supported_at,
                derivations=revision.derivations,
                created_at=existing.created_at if existing is not None else now,
                updated_at=now,
            )
            if existing is None:
                revisions_created += 1
            else:
                revisions_updated += 1
            revisions[key] = stored_revision

        for memory in plan.current_memories:
            key = self._memory_key(memory.tenant_id, memory.memory_key)
            existing_memory = memories.get(key)
            fingerprint = memory.metadata["semantic"]["content_fingerprint"]
            if (
                existing_memory is not None
                and existing_memory.metadata["semantic"]["content_fingerprint"] == fingerprint
            ):
                unchanged += 1
            elif existing_memory is None:
                created += 1
            else:
                updated += 1
            memories[key] = _stored_from_input(
                memory,
                memory_id=existing_memory.id if existing_memory is not None else str(uuid4()),
                created_at=existing_memory.created_at if existing_memory is not None else now,
                now=now,
            )

        for relation in plan.relations:
            rel_key = (
                relation.tenant_id,
                relation.left_revision_key,
                relation.right_revision_key,
                relation.relation.value,
            )
            if rel_key not in relations:
                relations_written += 1
            relations[rel_key] = (
                relation.left_revision_key,
                relation.right_revision_key,
                relation.relation.value,
                relation.effective_at.isoformat() if relation.effective_at else None,
            )

        self._memories = memories
        self._revisions = revisions
        self._relations = relations
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
    ) -> _List[StoredSemanticMemory]:
        revision_matches = await self.list_revisions(
            tenant_id=tenant_id,
            subject_id=subject_id,
            valid_at=valid_at,
        )
        results: list[StoredSemanticMemory] = []
        for revision in revision_matches:
            memory = self._memories.get(self._memory_key(tenant_id, revision.memory_key))
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

    async def deactivate_missing(
        self,
        *,
        tenant_id: str,
        subject_id: str | None,
        active_memory_keys: set[str],
        as_of: datetime | None = None,
    ) -> int:
        deactivated = 0
        now = as_of.astimezone(UTC) if as_of is not None else datetime.now(UTC)
        for key, memory in list(self._memories.items()):
            if memory.tenant_id != tenant_id:
                continue
            if subject_id is not None and memory.subject_id != subject_id:
                continue
            if not memory.is_active:
                continue
            if memory.memory_key in active_memory_keys:
                continue
            self._memories[key] = StoredSemanticMemory(
                id=memory.id,
                tenant_id=memory.tenant_id,
                subject_id=memory.subject_id,
                memory_key=memory.memory_key,
                slot_key=memory.slot_key,
                revision_key=memory.revision_key,
                revision_number=memory.revision_number,
                statement=memory.statement,
                subject_entity_id=memory.subject_entity_id,
                predicate=memory.predicate,
                object_value=memory.object_value,
                object_entity_id=memory.object_entity_id,
                polarity=memory.polarity,
                cardinality=memory.cardinality,
                qualifiers=memory.qualifiers,
                confidence=memory.confidence,
                importance=memory.importance,
                status=memory.status,
                support_count=memory.support_count,
                contradiction_count=memory.contradiction_count,
                first_supported_at=memory.first_supported_at,
                last_supported_at=memory.last_supported_at,
                valid_from=memory.valid_from,
                valid_until=memory.valid_until,
                is_active=False,
                derivations=memory.derivations,
                observation_evidence=memory.observation_evidence,
                entities=memory.entities,
                metadata=memory.metadata,
                created_at=memory.created_at,
                updated_at=now,
            )
            deactivated += 1
        return deactivated

    async def clear(self, *, tenant_id: str) -> None:
        memory_keys = [key for key in self._memories if key[0] == tenant_id]
        for key in memory_keys:
            del self._memories[key]
        revision_keys = [key for key in self._revisions if key[0] == tenant_id]
        for revision_key in revision_keys:
            del self._revisions[revision_key]
        relation_keys = [key for key in self._relations if key[0] == tenant_id]
        for relation_key in relation_keys:
            del self._relations[relation_key]


def _stored_from_input(
    memory: SemanticMemoryInput,
    *,
    memory_id: str,
    created_at: datetime,
    now: datetime,
) -> StoredSemanticMemory:
    return StoredSemanticMemory(
        id=memory_id,
        tenant_id=memory.tenant_id,
        subject_id=memory.subject_id,
        memory_key=memory.memory_key,
        slot_key=memory.slot_key,
        revision_key=memory.revision_key,
        revision_number=memory.revision_number,
        statement=memory.statement,
        subject_entity_id=memory.subject_entity_id,
        predicate=memory.predicate,
        object_value=memory.object_value,
        object_entity_id=memory.object_entity_id,
        polarity=memory.polarity,
        cardinality=memory.cardinality,
        qualifiers=memory.qualifiers,
        confidence=memory.confidence,
        importance=memory.importance,
        status=memory.status,
        support_count=memory.support_count,
        contradiction_count=memory.contradiction_count,
        first_supported_at=memory.first_supported_at,
        last_supported_at=memory.last_supported_at,
        valid_from=memory.valid_from,
        valid_until=memory.valid_until,
        is_active=True,
        derivations=memory.derivations,
        observation_evidence=memory.observation_evidence,
        entities=memory.entities,
        metadata=MappingProxyType(dict(memory.metadata)),
        created_at=created_at,
        updated_at=now,
    )
