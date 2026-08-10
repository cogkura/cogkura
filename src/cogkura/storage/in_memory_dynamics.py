"""In-memory memory dynamics store for tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from cogkura.models import MemoryIdentity, MemoryRetentionState, StoredMemoryDynamics
from cogkura.storage.base import MemoryDynamicsStore


class InMemoryMemoryDynamicsStore(MemoryDynamicsStore):
    """In-memory store for cognitive forgetting lifecycle state."""

    def __init__(self) -> None:
        self._dynamics: dict[tuple[str, str, str], StoredMemoryDynamics] = {}

    def _key(self, dynamics: StoredMemoryDynamics) -> tuple[str, str, str]:
        return (
            dynamics.tenant_id,
            dynamics.memory_kind.value,
            dynamics.memory_key,
        )

    async def get_many(
        self,
        *,
        tenant_id: str,
        identities: Sequence[MemoryIdentity],
    ) -> Mapping[MemoryIdentity, StoredMemoryDynamics]:
        result: dict[MemoryIdentity, StoredMemoryDynamics] = {}
        for identity in identities:
            key = (tenant_id, identity.memory_kind.value, identity.memory_key)
            dynamics = self._dynamics.get(key)
            if dynamics is not None:
                result[identity] = dynamics
        return result

    async def upsert_many(self, dynamics: Sequence[StoredMemoryDynamics]) -> None:
        for record in dynamics:
            self._dynamics[self._key(record)] = record

    async def reactivate(
        self,
        *,
        tenant_id: str,
        identities: Sequence[MemoryIdentity],
        at: datetime,
    ) -> None:
        timestamp = at.astimezone(UTC)
        for identity in identities:
            key = (tenant_id, identity.memory_kind.value, identity.memory_key)
            existing = self._dynamics.get(key)
            if existing is None:
                continue
            self._dynamics[key] = StoredMemoryDynamics(
                tenant_id=existing.tenant_id,
                memory_kind=existing.memory_kind,
                memory_key=existing.memory_key,
                retention_state=MemoryRetentionState.ACTIVE,
                last_base_level=existing.last_base_level,
                last_retention_score=existing.last_retention_score,
                below_threshold_since=None,
                forgotten_at=None,
                evaluated_at=timestamp,
                updated_at=timestamp,
            )

    async def clear(self, *, tenant_id: str) -> None:
        self._dynamics = {
            key: value for key, value in self._dynamics.items() if key[0] != tenant_id
        }
