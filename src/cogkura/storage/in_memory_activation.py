"""In-memory activation reference store for tests."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from cogkura.models import MemoryIdentity, MemoryReference
from cogkura.storage.base import ActivationStore


class InMemoryActivationStore(ActivationStore):
    """In-memory store for memory access references."""

    def __init__(self) -> None:
        self._references: list[MemoryReference] = []

    async def append_references(self, references: Sequence[MemoryReference]) -> None:
        for reference in references:
            if reference.request_id is not None and self._has_request(reference):
                continue
            self._references.append(reference)

    def _has_request(self, reference: MemoryReference) -> bool:
        for existing in self._references:
            if existing.tenant_id != reference.tenant_id:
                continue
            if existing.request_id != reference.request_id:
                continue
            if existing.memory_kind != reference.memory_kind:
                continue
            if existing.memory_key != reference.memory_key:
                continue
            if existing.reference_kind != reference.reference_kind:
                continue
            return True
        return False

    async def list_reference_times(
        self,
        *,
        tenant_id: str,
        identities: Sequence[MemoryIdentity],
        before_or_at: datetime,
    ) -> Mapping[MemoryIdentity, tuple[datetime, ...]]:
        identity_set = set(identities)
        grouped: dict[MemoryIdentity, list[datetime]] = defaultdict(list)
        cutoff = before_or_at.astimezone(UTC)
        for reference in self._references:
            if reference.tenant_id != tenant_id:
                continue
            if reference.identity not in identity_set:
                continue
            referenced_at = reference.referenced_at.astimezone(UTC)
            if referenced_at > cutoff:
                continue
            grouped[reference.identity].append(referenced_at)
        return {identity: tuple(sorted(times)) for identity, times in grouped.items()}

    async def clear(self, *, tenant_id: str) -> None:
        self._references = [
            reference for reference in self._references if reference.tenant_id != tenant_id
        ]
