"""Storage interfaces for observations, checkpoints, and episodes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from cognema.models import (
        EpisodeInput,
        EpisodeWriteStatus,
        MemoryIdentity,
        MemoryReference,
        SemanticMemoryInput,
        SemanticMemoryStatus,
        SemanticWriteStatus,
        StoredEpisode,
        StoredSemanticMemory,
    )

from cognema.observations.models import IngestStatus, ObservationInput, StoredObservation
from cognema.observations.retention import RetainedObservation


class ObservationStore(Protocol):
    """Persists normalized observations with revision history."""

    async def ingest(
        self,
        observation: ObservationInput,
        *,
        retained: RetainedObservation,
    ) -> IngestStatus:
        """Store or update an observation. Returns ingest status."""

    async def get_by_source(
        self,
        *,
        tenant_id: str,
        source_namespace: str,
        source_record_id: str,
    ) -> StoredObservation | None:
        """Fetch the current observation for a source record."""

    async def list(
        self,
        *,
        tenant_id: str,
        subject_id: str | None = None,
        include_deleted: bool = False,
    ) -> list[StoredObservation]:
        """List observations for a tenant, optionally filtered by subject."""

    async def get_many(
        self,
        *,
        tenant_id: str,
        observation_ids: set[str],
    ) -> Sequence[StoredObservation]:
        """Load tenant-scoped observations by Cognema observation ID."""

    async def clear(self, *, tenant_id: str) -> None:
        """Remove all observations for a tenant."""


class CheckpointStore(Protocol):
    """Stores connector ingestion checkpoints per tenant."""

    async def get(
        self,
        *,
        tenant_id: str,
        connector_id: str,
    ) -> dict[str, Any] | None:
        """Load checkpoint for a connector."""

    async def set(
        self,
        *,
        tenant_id: str,
        connector_id: str,
        checkpoint: dict[str, Any],
    ) -> None:
        """Persist checkpoint for a connector."""


class EpisodeStore(Protocol):
    """Persists derived episodic memories and their evidence."""

    async def upsert(self, episode: EpisodeInput) -> EpisodeWriteStatus:
        """Create, update, or preserve an episodic memory."""

    async def list(
        self,
        *,
        tenant_id: str,
        subject_id: str | None = None,
        include_inactive: bool = False,
        limit: int | None = None,
    ) -> list[StoredEpisode]:
        """List tenant-scoped episodes."""

    async def deactivate_missing(
        self,
        *,
        tenant_id: str,
        subject_id: str | None,
        active_memory_keys: set[str],
    ) -> int:
        """Deactivate episodes no longer produced by current observations."""

    async def clear(self, *, tenant_id: str) -> None:
        """Remove episodic memories for a tenant."""


class SemanticMemoryStore(Protocol):
    """Persists derived semantic memories and their provenance."""

    async def upsert(self, memory: SemanticMemoryInput) -> SemanticWriteStatus:
        """Create, update, or preserve a semantic memory."""

    async def list(
        self,
        *,
        tenant_id: str,
        subject_id: str | None = None,
        include_inactive: bool = False,
        status: SemanticMemoryStatus | None = None,
        limit: int | None = None,
    ) -> list[StoredSemanticMemory]:
        """List tenant-scoped semantic memories."""

    async def deactivate_missing(
        self,
        *,
        tenant_id: str,
        subject_id: str | None,
        active_memory_keys: set[str],
    ) -> int:
        """Deactivate semantic memories no longer produced."""

    async def clear(self, *, tenant_id: str) -> None:
        """Remove semantic memories for a tenant."""


class ActivationStore(Protocol):
    """Persists memory access references for base-level activation."""

    async def append_references(self, references: Sequence[MemoryReference]) -> None:
        """Append access references for durable memories."""

    async def list_reference_times(
        self,
        *,
        tenant_id: str,
        identities: Sequence[MemoryIdentity],
        before_or_at: datetime,
    ) -> Mapping[MemoryIdentity, tuple[datetime, ...]]:
        """Load reference timestamps for the given memory identities."""

    async def clear(self, *, tenant_id: str) -> None:
        """Remove activation references for a tenant."""
