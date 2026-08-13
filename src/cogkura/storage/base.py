"""Storage interfaces for observations, checkpoints, and episodes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from cogkura.models import (
    ActivationReferenceTrace,
    LearningPlan,
    LearningWriteResult,
    MemoryIdentity,
    MemoryReference,
    ReferenceCompactionResult,
    SemanticReconciliationPlan,
    SemanticReconciliationWriteResult,
    StoredMemoryAssociation,
    StoredMemoryDynamics,
    StoredMemoryLearningState,
    StoredSemanticRevision,
)

if TYPE_CHECKING:
    from cogkura.models import (
        EpisodeInput,
        EpisodeWriteStatus,
        SemanticMemoryInput,
        SemanticMemoryStatus,
        SemanticWriteStatus,
        StoredEpisode,
        StoredSemanticMemory,
    )
from cogkura.observations.models import IngestStatus, ObservationInput, StoredObservation
from cogkura.observations.retention import RetainedObservation


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
        """Load tenant-scoped observations by Cogkura observation ID."""

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

    async def upsert(
        self,
        episode: EpisodeInput,
        *,
        as_of: datetime | None = None,
    ) -> EpisodeWriteStatus:
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
        as_of: datetime | None = None,
    ) -> int:
        """Deactivate episodes no longer produced by current observations."""

    async def clear(self, *, tenant_id: str) -> None:
        """Remove episodic memories for a tenant."""


class SemanticMemoryStore(Protocol):
    """Persists derived semantic memories and their provenance."""

    async def upsert(
        self,
        memory: SemanticMemoryInput,
        *,
        as_of: datetime | None = None,
    ) -> SemanticWriteStatus:
        """Create, update, or preserve a semantic memory."""

    async def list(
        self,
        *,
        tenant_id: str,
        subject_id: str | None = None,
        include_inactive: bool = False,
        status: SemanticMemoryStatus | None = None,
        limit: int | None = None,
        valid_at: datetime | None = None,
    ) -> Sequence[StoredSemanticMemory]:
        """List tenant-scoped semantic memories."""

    async def list_revisions(
        self,
        *,
        tenant_id: str,
        memory_key: str | None = None,
        subject_id: str | None = None,
        valid_at: datetime | None = None,
        limit: int | None = None,
    ) -> Sequence[StoredSemanticRevision]:
        """List semantic revisions for a tenant."""

    async def apply_reconciliation(
        self,
        plan: SemanticReconciliationPlan,
        *,
        as_of: datetime | None = None,
    ) -> SemanticReconciliationWriteResult:
        """Apply a reconciliation plan atomically."""

    async def deactivate_missing(
        self,
        *,
        tenant_id: str,
        subject_id: str | None,
        active_memory_keys: set[str],
        as_of: datetime | None = None,
    ) -> int:
        """Deactivate semantic memories no longer produced."""

    async def clear(self, *, tenant_id: str) -> None:
        """Remove semantic memories for a tenant."""


class ActivationStore(Protocol):
    """Persists memory access references for base-level activation."""

    async def append_references(self, references: Sequence[MemoryReference]) -> None:
        """Append access references for durable memories."""

    async def list_reference_traces(
        self,
        *,
        tenant_id: str,
        identities: Sequence[MemoryIdentity],
        before_or_at: datetime,
    ) -> Mapping[MemoryIdentity, tuple[ActivationReferenceTrace, ...]]:
        """Load weighted reference traces for the given memory identities."""

    async def compact_references(
        self,
        *,
        tenant_id: str,
        before: datetime,
        bucket_seconds: float,
    ) -> ReferenceCompactionResult:
        """Compact old references into weighted bucket traces."""

    async def clear(self, *, tenant_id: str) -> None:
        """Remove activation references for a tenant."""


class MemoryDynamicsStore(Protocol):
    """Persists cognitive forgetting lifecycle state for durable memories."""

    async def get_many(
        self,
        *,
        tenant_id: str,
        identities: Sequence[MemoryIdentity],
    ) -> Mapping[MemoryIdentity, StoredMemoryDynamics]:
        """Load dynamics records for the given memory identities."""

    async def upsert_many(self, dynamics: Sequence[StoredMemoryDynamics]) -> None:
        """Insert or update dynamics records."""

    async def reactivate(
        self,
        *,
        tenant_id: str,
        identities: Sequence[MemoryIdentity],
        at: datetime,
    ) -> None:
        """Clear forgotten/fading state after explicit reinforcement."""

    async def clear(self, *, tenant_id: str) -> None:
        """Remove dynamics records for a tenant."""


class LearningStore(Protocol):
    """Persists learning feedback, utility counts, and learned associations."""

    async def apply(self, plan: LearningPlan) -> LearningWriteResult:
        """Apply a learning plan atomically and idempotently."""

    async def list_states(
        self,
        *,
        tenant_id: str,
        identities: Sequence[MemoryIdentity],
        context_keys: Sequence[str],
    ) -> Sequence[StoredMemoryLearningState]:
        """Load learning state for identities in the given contexts."""

    async def list_associations(
        self,
        *,
        tenant_id: str,
        identities: Sequence[MemoryIdentity],
    ) -> Sequence[StoredMemoryAssociation]:
        """Load learned associations whose endpoints are in the identity set."""

    async def list_reinforcement_traces(
        self,
        *,
        tenant_id: str,
        identities: Sequence[MemoryIdentity],
        before_or_at: datetime,
    ) -> Mapping[MemoryIdentity, tuple[ActivationReferenceTrace, ...]]:
        """Load HELPFUL learning reinforcement traces for base-level activation."""

    async def clear(self, *, tenant_id: str) -> None:
        """Remove learning data for a tenant."""
