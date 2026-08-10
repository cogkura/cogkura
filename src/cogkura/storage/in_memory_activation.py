"""In-memory activation reference store for tests."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta

from cogkura.models import (
    ActivationReferenceTrace,
    MemoryIdentity,
    MemoryReference,
    ReferenceCompactionResult,
)
from cogkura.storage.activation_compaction import compaction_representative_time
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

    async def list_reference_traces(
        self,
        *,
        tenant_id: str,
        identities: Sequence[MemoryIdentity],
        before_or_at: datetime,
    ) -> Mapping[MemoryIdentity, tuple[ActivationReferenceTrace, ...]]:
        identity_set = set(identities)
        grouped: dict[MemoryIdentity, list[ActivationReferenceTrace]] = defaultdict(list)
        cutoff = before_or_at.astimezone(UTC)
        for reference in self._references:
            if reference.tenant_id != tenant_id:
                continue
            if reference.identity not in identity_set:
                continue
            referenced_at = reference.referenced_at.astimezone(UTC)
            if referenced_at > cutoff:
                continue
            grouped[reference.identity].append(
                ActivationReferenceTrace(
                    referenced_at=referenced_at,
                    weight=reference.weight,
                )
            )
        return {
            identity: tuple(sorted(traces, key=lambda trace: trace.referenced_at))
            for identity, traces in grouped.items()
        }

    async def compact_references(
        self,
        *,
        tenant_id: str,
        before: datetime,
        bucket_seconds: float,
    ) -> ReferenceCompactionResult:
        cutoff = before.astimezone(UTC)
        retained: list[MemoryReference] = []
        to_compact: list[MemoryReference] = []
        for reference in self._references:
            if reference.tenant_id != tenant_id:
                retained.append(reference)
                continue
            if reference.referenced_at.astimezone(UTC) >= cutoff:
                retained.append(reference)
                continue
            to_compact.append(reference)

        if not to_compact:
            return ReferenceCompactionResult(references_compacted=0)

        buckets: dict[
            tuple[str, str, str, str, datetime],
            list[MemoryReference],
        ] = defaultdict(list)
        for reference in to_compact:
            referenced_at = reference.referenced_at.astimezone(UTC)
            bucket_start = _bucket_start(referenced_at, bucket_seconds)
            key = (
                reference.tenant_id,
                reference.memory_kind.value,
                reference.memory_key,
                reference.reference_kind.value,
                bucket_start,
            )
            buckets[key].append(reference)

        compacted: list[MemoryReference] = []
        compacted_count = 0
        for key, references in buckets.items():
            if len(references) == 1 and references[0].weight == 1:
                compacted.append(references[0])
                continue
            total_weight = sum(reference.weight for reference in references)
            bucket_start = key[4]
            representative_at = compaction_representative_time(
                references,
                bucket_start=bucket_start,
                as_of=cutoff,
            )
            compacted.append(
                MemoryReference(
                    tenant_id=references[0].tenant_id,
                    memory_kind=references[0].memory_kind,
                    memory_key=references[0].memory_key,
                    reference_kind=references[0].reference_kind,
                    referenced_at=representative_at,
                    request_id=None,
                    weight=total_weight,
                    metadata=references[0].metadata,
                )
            )
            compacted_count += len(references)

        self._references = retained + compacted
        return ReferenceCompactionResult(references_compacted=compacted_count)

    async def clear(self, *, tenant_id: str) -> None:
        self._references = [
            reference for reference in self._references if reference.tenant_id != tenant_id
        ]


def _bucket_start(referenced_at: datetime, bucket_seconds: float) -> datetime:
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    elapsed = (referenced_at - epoch).total_seconds()
    bucket_index = int(elapsed // bucket_seconds)
    return epoch + timedelta(seconds=bucket_index * bucket_seconds)
