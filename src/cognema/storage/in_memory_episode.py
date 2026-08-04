"""In-memory episode store for tests."""

from __future__ import annotations

from datetime import UTC, datetime
from types import MappingProxyType
from uuid import uuid4

from cognema.models import (
    EpisodeInput,
    EpisodeWriteStatus,
    StoredEpisode,
)
from cognema.storage.base import EpisodeStore


class InMemoryEpisodeStore(EpisodeStore):
    """In-memory episodic memory store."""

    def __init__(self) -> None:
        self._episodes: dict[tuple[str, str], StoredEpisode] = {}

    def _key(self, tenant_id: str, memory_key: str) -> tuple[str, str]:
        return (tenant_id, memory_key)

    async def upsert(self, episode: EpisodeInput) -> EpisodeWriteStatus:
        key = self._key(episode.tenant_id, episode.memory_key)
        existing = self._episodes.get(key)
        now = datetime.now(UTC)
        fingerprint = episode.metadata["episode"]["content_fingerprint"]
        if existing is not None:
            existing_fingerprint = existing.metadata["episode"]["content_fingerprint"]
            if existing_fingerprint == fingerprint:
                return EpisodeWriteStatus.UNCHANGED
            stored = StoredEpisode(
                id=existing.id,
                tenant_id=episode.tenant_id,
                subject_id=episode.subject_id,
                memory_key=episode.memory_key,
                statement=episode.statement,
                started_at=episode.started_at,
                ended_at=episode.ended_at,
                confidence=episode.confidence,
                importance=episode.importance,
                is_active=True,
                evidence=episode.evidence,
                entities=episode.entities,
                metadata=MappingProxyType(dict(episode.metadata)),
                created_at=existing.created_at,
                updated_at=now,
            )
            self._episodes[key] = stored
            return EpisodeWriteStatus.UPDATED

        stored = StoredEpisode(
            id=str(uuid4()),
            tenant_id=episode.tenant_id,
            subject_id=episode.subject_id,
            memory_key=episode.memory_key,
            statement=episode.statement,
            started_at=episode.started_at,
            ended_at=episode.ended_at,
            confidence=episode.confidence,
            importance=episode.importance,
            is_active=True,
            evidence=episode.evidence,
            entities=episode.entities,
            metadata=MappingProxyType(dict(episode.metadata)),
            created_at=now,
            updated_at=now,
        )
        self._episodes[key] = stored
        return EpisodeWriteStatus.CREATED

    async def list(
        self,
        *,
        tenant_id: str,
        subject_id: str | None = None,
        include_inactive: bool = False,
        limit: int | None = None,
    ) -> list[StoredEpisode]:
        results: list[StoredEpisode] = []
        for episode in self._episodes.values():
            if episode.tenant_id != tenant_id:
                continue
            if subject_id is not None and episode.subject_id != subject_id:
                continue
            if not include_inactive and not episode.is_active:
                continue
            results.append(episode)
        results.sort(key=lambda item: (item.started_at, item.id))
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
        for key, episode in list(self._episodes.items()):
            if episode.tenant_id != tenant_id:
                continue
            if subject_id is not None and episode.subject_id != subject_id:
                continue
            if not episode.is_active:
                continue
            if episode.memory_key in active_memory_keys:
                continue
            self._episodes[key] = StoredEpisode(
                id=episode.id,
                tenant_id=episode.tenant_id,
                subject_id=episode.subject_id,
                memory_key=episode.memory_key,
                statement=episode.statement,
                started_at=episode.started_at,
                ended_at=episode.ended_at,
                confidence=episode.confidence,
                importance=episode.importance,
                is_active=False,
                evidence=episode.evidence,
                entities=episode.entities,
                metadata=episode.metadata,
                created_at=episode.created_at,
                updated_at=now,
            )
            deactivated += 1
        return deactivated

    async def clear(self, *, tenant_id: str) -> None:
        keys = [key for key in self._episodes if key[0] == tenant_id]
        for key in keys:
            del self._episodes[key]
