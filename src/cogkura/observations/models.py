"""Observation data models."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from cogkura.exceptions import ValidationError


def _validate_attention_score(score: float) -> None:
    if not math.isfinite(score):
        raise ValidationError("attention_score must be finite.")
    if not 0.0 <= score <= 1.0:
        raise ValidationError("attention_score must be between 0.0 and 1.0.")


class IngestStatus(StrEnum):
    """Outcome of ingesting a single observation."""

    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    UNCHANGED = "unchanged"
    RESTORED = "restored"


@dataclass(frozen=True, slots=True)
class ObservationInput:
    """Normalized observation from a customer source record."""

    source_namespace: str
    source_record_id: str
    content: str
    observed_at: datetime
    tenant_id: str
    subject_id: str | None = None
    actor_id: str | None = None
    source_type: str = "application"
    source_version: str | None = None
    source_created_at: datetime | None = None
    source_updated_at: datetime | None = None
    event_type: str = "document"
    metadata: dict[str, Any] = field(default_factory=dict)
    is_deleted: bool = False

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValidationError("tenant_id must not be empty.")
        if not self.source_namespace.strip():
            raise ValidationError("source_namespace must not be empty.")
        if not self.source_record_id.strip():
            raise ValidationError("source_record_id must not be empty.")
        if not self.is_deleted and not self.content.strip():
            raise ValidationError("content must not be empty for non-deleted observations.")
        if self.observed_at.tzinfo is None:
            raise ValidationError("observed_at must be timezone-aware.")
        object.__setattr__(self, "observed_at", self.observed_at.astimezone(UTC))
        for label, ts in (
            ("source_created_at", self.source_created_at),
            ("source_updated_at", self.source_updated_at),
        ):
            if ts is not None and ts.tzinfo is None:
                raise ValidationError(f"{label} must be timezone-aware.")
        metadata_dict = dict(self.metadata)
        object.__setattr__(self, "metadata", MappingProxyType(metadata_dict))


@dataclass(frozen=True, slots=True)
class StoredObservation:
    """Persisted observation with revision metadata."""

    id: str
    tenant_id: str
    subject_id: str | None
    actor_id: str | None
    source_type: str
    source_namespace: str
    source_record_id: str
    source_version: str | None
    event_type: str
    content: str | None
    content_hash: str
    metadata: Mapping[str, Any]
    source_created_at: datetime | None
    source_updated_at: datetime | None
    observed_at: datetime
    current_revision: int
    is_deleted: bool
    attention_score: float = 0.5
    retention_class: str = "full"
    policy_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_attention_score(self.attention_score)
        if not self.retention_class.strip():
            raise ValidationError("retention_class must not be empty.")


@dataclass(frozen=True, slots=True)
class ObservationDecision:
    """Policy evaluation result for an observation."""

    accept: bool
    attention_score: float
    retention_class: str
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_attention_score(self.attention_score)
        if not self.retention_class.strip():
            raise ValidationError("retention_class must not be empty.")


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Aggregated outcome of a source ingestion run."""

    discovered: int = 0
    created: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0
    restored: int = 0
    rejected: int = 0
    failed: int = 0

    def record(self, status: IngestStatus) -> IngestionResult:
        """Return a new result with one counter incremented."""
        return IngestionResult(
            discovered=self.discovered,
            created=self.created + (1 if status is IngestStatus.CREATED else 0),
            updated=self.updated + (1 if status is IngestStatus.UPDATED else 0),
            deleted=self.deleted + (1 if status is IngestStatus.DELETED else 0),
            unchanged=self.unchanged + (1 if status is IngestStatus.UNCHANGED else 0),
            restored=self.restored + (1 if status is IngestStatus.RESTORED else 0),
            rejected=self.rejected,
            failed=self.failed,
        )
