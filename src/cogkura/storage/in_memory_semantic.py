"""In-memory semantic memory store for tests."""

from __future__ import annotations

from datetime import UTC, datetime
from types import MappingProxyType
from uuid import uuid4

from cogkura.models import (
    SemanticMemoryInput,
    SemanticMemoryStatus,
    SemanticWriteStatus,
    StoredSemanticMemory,
)
from cogkura.storage.base import SemanticMemoryStore


class InMemorySemanticMemoryStore(SemanticMemoryStore):
    """In-memory semantic memory store."""

    def __init__(self) -> None:
        self._memories: dict[tuple[str, str], StoredSemanticMemory] = {}

    def _key(self, tenant_id: str, memory_key: str) -> tuple[str, str]:
        return (tenant_id, memory_key)

    async def upsert(self, memory: SemanticMemoryInput) -> SemanticWriteStatus:
        key = self._key(memory.tenant_id, memory.memory_key)
        existing = self._memories.get(key)
        now = datetime.now(UTC)
        fingerprint = memory.metadata["semantic"]["content_fingerprint"]
        if existing is not None:
            existing_fingerprint = existing.metadata["semantic"]["content_fingerprint"]
            if existing_fingerprint == fingerprint:
                return SemanticWriteStatus.UNCHANGED
            stored = StoredSemanticMemory(
                id=existing.id,
                tenant_id=memory.tenant_id,
                subject_id=memory.subject_id,
                memory_key=memory.memory_key,
                slot_key=memory.slot_key,
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
                is_active=True,
                derivations=memory.derivations,
                observation_evidence=memory.observation_evidence,
                entities=memory.entities,
                metadata=MappingProxyType(dict(memory.metadata)),
                created_at=existing.created_at,
                updated_at=now,
            )
            self._memories[key] = stored
            return SemanticWriteStatus.UPDATED

        stored = StoredSemanticMemory(
            id=str(uuid4()),
            tenant_id=memory.tenant_id,
            subject_id=memory.subject_id,
            memory_key=memory.memory_key,
            slot_key=memory.slot_key,
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
            is_active=True,
            derivations=memory.derivations,
            observation_evidence=memory.observation_evidence,
            entities=memory.entities,
            metadata=MappingProxyType(dict(memory.metadata)),
            created_at=now,
            updated_at=now,
        )
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
    ) -> list[StoredSemanticMemory]:
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
            results.append(memory)
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
    ) -> int:
        deactivated = 0
        now = datetime.now(UTC)
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
        keys = [key for key in self._memories if key[0] == tenant_id]
        for key in keys:
            del self._memories[key]
