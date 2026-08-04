"""Storage interface for memory events and observations."""

from __future__ import annotations

from typing import Any, Protocol

from cognema.event import MemoryEvent
from cognema.observations.models import IngestStatus, ObservationInput, StoredObservation
from cognema.observations.retention import RetainedObservation


class MemoryStorage(Protocol):
    """Minimal storage protocol used by transitional recall."""

    def store(self, event: MemoryEvent) -> None:
        """Persist a memory event."""

    def get(self, event_id: str) -> MemoryEvent | None:
        """Fetch a single event by id."""

    def list(self) -> list[MemoryEvent]:
        """Return all stored events."""

    def clear(self) -> None:
        """Remove all stored events."""


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
