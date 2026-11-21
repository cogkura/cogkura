"""In-memory episode store for tests."""

from __future__ import annotations

from datetime import UTC, datetime
from types import MappingProxyType
from uuid import uuid4

from cogkura.models import (
    EpisodeEvidenceInput,
    EpisodeInput,
    EpisodeWriteStatus,
    StoredEpisode,
    memory_context_signatures_equal,
)
from cogkura.storage.base import EpisodeStore
from cogkura.storage.historical_snapshots import (
    encode_episode_snapshot_or_none,
    select_episode_snapshot,
)
from cogkura.storage.history_order import HistoryOrderStatus, InMemoryHistoryOrder


class InMemoryEpisodeStore(EpisodeStore):
    """In-memory episodic memory store."""

    def __init__(self, *, history_order: InMemoryHistoryOrder | None = None) -> None:
        self._history_order = history_order or InMemoryHistoryOrder()
        self._episodes: dict[tuple[str, str], StoredEpisode] = {}
        self._episode_history: dict[tuple[str, str], list[tuple[datetime, str | None]]] = {}

    def _record_history(self, episode: StoredEpisode, now: datetime) -> None:
        self._history_order.record(episode.tenant_id, now)
        self._episode_history.setdefault((episode.tenant_id, episode.id), []).append(
            (now, encode_episode_snapshot_or_none(episode))
        )

    async def historical_order_status(self, *, tenant_id: str) -> HistoryOrderStatus:
        return self._history_order.get(tenant_id)

    async def clear_history_order(self, *, tenant_id: str) -> None:
        self._history_order.clear(tenant_id)

    async def historical_episodes(
        self, *, tenant_id: str, episode_ids: set[str], as_of: datetime
    ) -> dict[str, StoredEpisode]:
        results = {}
        for episode_id in episode_ids:
            versions = self._episode_history.get((tenant_id, episode_id), [])
            episode = select_episode_snapshot(versions, as_of=as_of)
            if episode is not None:
                results[episode_id] = episode
        return results

    def _key(self, tenant_id: str, memory_key: str) -> tuple[str, str]:
        return (tenant_id, memory_key)

    async def upsert(
        self,
        episode: EpisodeInput,
        *,
        as_of: datetime | None = None,
    ) -> EpisodeWriteStatus:
        key = self._key(episode.tenant_id, episode.memory_key)
        existing = self._episodes.get(key)
        now = as_of.astimezone(UTC) if as_of is not None else datetime.now(UTC)
        fingerprint = episode.metadata["episode"]["content_fingerprint"]
        if existing is not None:
            existing_fingerprint = existing.metadata["episode"]["content_fingerprint"]
            if existing_fingerprint == fingerprint and memory_context_signatures_equal(
                existing.encoding_context,
                episode.encoding_context,
            ):
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
                encoding_context=episode.encoding_context,
                created_at=existing.created_at,
                updated_at=now,
            )
            self._episodes[key] = stored
            self._record_history(stored, now)
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
            encoding_context=episode.encoding_context,
            created_at=now,
            updated_at=now,
        )
        self._episodes[key] = stored
        self._record_history(stored, now)
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

    async def get_evidence_many(
        self,
        *,
        tenant_id: str,
        episode_ids: set[str],
    ) -> dict[str, tuple[EpisodeEvidenceInput, ...]]:
        if not episode_ids:
            return {}
        return {
            episode.id: episode.evidence
            for episode in self._episodes.values()
            if episode.tenant_id == tenant_id and episode.id in episode_ids
        }

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
                encoding_context=episode.encoding_context,
                created_at=episode.created_at,
                updated_at=now,
            )
            self._record_history(self._episodes[key], now)
            deactivated += 1
        return deactivated

    async def clear(self, *, tenant_id: str) -> None:
        keys = [key for key in self._episodes if key[0] == tenant_id]
        for key in keys:
            del self._episodes[key]
        for key in [key for key in self._episode_history if key[0] == tenant_id]:
            del self._episode_history[key]
