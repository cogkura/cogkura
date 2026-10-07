"""In-memory inhibitory trace store."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from cogkura.models import InhibitoryTrace, MemoryIdentity
from cogkura.storage.base import InhibitionStore


class InMemoryInhibitionStore(InhibitionStore):
    """Tenant-scoped in-memory inhibitory traces."""

    def __init__(self) -> None:
        self._traces: list[InhibitoryTrace] = []

    async def append_traces(self, traces: Sequence[InhibitoryTrace]) -> None:
        for trace in traces:
            if trace.request_id is not None and self._has_request(trace):
                continue
            self._traces.append(trace)

    def _has_request(self, trace: InhibitoryTrace) -> bool:
        for existing in self._traces:
            if existing.request_id is None:
                continue
            if existing.tenant_id != trace.tenant_id:
                continue
            if existing.request_id != trace.request_id:
                continue
            if existing.selected_identity != trace.selected_identity:
                continue
            if existing.inhibited_identity != trace.inhibited_identity:
                continue
            if existing.scope.scope_key != trace.scope.scope_key:
                continue
            return True
        return False

    async def list_for_memory(
        self,
        *,
        tenant_id: str,
        identity: MemoryIdentity,
        as_of: datetime | None = None,
    ) -> Sequence[InhibitoryTrace]:
        cutoff = as_of.astimezone(UTC) if as_of is not None else None
        matched = [
            trace
            for trace in self._traces
            if trace.tenant_id == tenant_id
            and trace.inhibited_identity == identity
            and (cutoff is None or trace.induced_at <= cutoff)
        ]
        return tuple(
            sorted(matched, key=lambda trace: (trace.induced_at, trace.scope.scope_key, trace.id))
        )

    async def list_for_memories(
        self,
        *,
        tenant_id: str,
        identities: Sequence[MemoryIdentity],
        after: datetime | None = None,
        before_or_at: datetime,
        limit_per_memory: int | None = None,
    ) -> Mapping[MemoryIdentity, tuple[InhibitoryTrace, ...]]:
        if not identities:
            return {}
        identity_set = set(identities)
        cutoff = before_or_at.astimezone(UTC)
        lower = after.astimezone(UTC) if after is not None else None
        grouped: dict[MemoryIdentity, list[InhibitoryTrace]] = {
            identity: [] for identity in identities
        }
        for trace in self._traces:
            if trace.tenant_id != tenant_id or trace.inhibited_identity not in identity_set:
                continue
            if trace.induced_at > cutoff:
                continue
            if lower is not None and trace.induced_at <= lower:
                continue
            grouped[trace.inhibited_identity].append(trace)
        limited: dict[MemoryIdentity, tuple[InhibitoryTrace, ...]] = {}
        for identity, traces in grouped.items():
            ordered = sorted(traces, key=lambda item: (-item.induced_at.timestamp(), item.id))
            if limit_per_memory is not None:
                ordered = ordered[:limit_per_memory]
            if ordered:
                limited[identity] = tuple(ordered)
        return limited

    async def clear(self, *, tenant_id: str) -> None:
        self._traces = [trace for trace in self._traces if trace.tenant_id != tenant_id]
