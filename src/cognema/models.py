"""Additional package data models."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from cognema.exceptions import ValidationError
from cognema.observations.models import StoredObservation


@dataclass(frozen=True, slots=True)
class RecallResult:
    """A scored recall match for a query."""

    observation: StoredObservation
    score: float
    reason: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.score):
            raise ValidationError("Recall score must be finite.")
        if not 0.0 <= self.score <= 1.0:
            raise ValidationError("Recall score must be between 0.0 and 1.0.")


class EpisodeWriteStatus(StrEnum):
    """Outcome of upserting a single episodic memory."""

    CREATED = "created"
    UPDATED = "updated"
    UNCHANGED = "unchanged"


@dataclass(frozen=True, slots=True)
class EpisodeEntity:
    """An entity linked to an episodic memory."""

    entity_id: str
    role: str

    def __post_init__(self) -> None:
        if not self.entity_id.strip():
            raise ValidationError("entity_id must not be empty.")
        if not self.role.strip():
            raise ValidationError("entity role must not be empty.")


@dataclass(frozen=True, slots=True)
class EpisodeEvidenceInput:
    """Evidence linking an episode to a source observation revision."""

    observation_id: str
    observation_revision: int
    sequence_number: int
    contribution_score: float = 1.0

    def __post_init__(self) -> None:
        if not self.observation_id.strip():
            raise ValidationError("observation_id must not be empty.")
        if self.observation_revision <= 0:
            raise ValidationError("observation_revision must be greater than zero.")
        if self.sequence_number < 0:
            raise ValidationError("sequence_number must not be negative.")
        if not math.isfinite(self.contribution_score):
            raise ValidationError("contribution_score must be finite.")
        if not 0.0 <= self.contribution_score <= 1.0:
            raise ValidationError("contribution_score must be between 0.0 and 1.0.")


@dataclass(frozen=True, slots=True)
class EpisodeInput:
    """Candidate episodic memory ready for persistence."""

    tenant_id: str
    subject_id: str | None
    memory_key: str
    statement: str
    started_at: datetime
    ended_at: datetime
    confidence: float
    importance: float
    evidence: tuple[EpisodeEvidenceInput, ...]
    entities: tuple[EpisodeEntity, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValidationError("tenant_id must not be empty.")
        if not self.memory_key.strip():
            raise ValidationError("memory_key must not be empty.")
        if not self.statement.strip():
            raise ValidationError("statement must not be empty.")
        if self.started_at.tzinfo is None:
            raise ValidationError("started_at must be timezone-aware.")
        if self.ended_at.tzinfo is None:
            raise ValidationError("ended_at must be timezone-aware.")
        started = self.started_at.astimezone(UTC)
        ended = self.ended_at.astimezone(UTC)
        object.__setattr__(self, "started_at", started)
        object.__setattr__(self, "ended_at", ended)
        if ended < started:
            raise ValidationError("ended_at must not be before started_at.")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValidationError("confidence must be between 0.0 and 1.0.")
        if not math.isfinite(self.importance) or not 0.0 <= self.importance <= 1.0:
            raise ValidationError("importance must be between 0.0 and 1.0.")
        if not self.evidence:
            raise ValidationError("evidence must contain at least one item.")
        observation_ids = {item.observation_id for item in self.evidence}
        if len(observation_ids) != len(self.evidence):
            raise ValidationError("evidence observation IDs must be unique.")
        sequence_numbers = {item.sequence_number for item in self.evidence}
        if len(sequence_numbers) != len(self.evidence):
            raise ValidationError("evidence sequence numbers must be unique.")
        metadata_dict = dict(self.metadata)
        object.__setattr__(self, "metadata", MappingProxyType(metadata_dict))


@dataclass(frozen=True, slots=True)
class StoredEpisode:
    """Persisted episodic memory with evidence and entity links."""

    id: str
    tenant_id: str
    subject_id: str | None
    memory_key: str
    statement: str
    started_at: datetime
    ended_at: datetime
    confidence: float
    importance: float
    is_active: bool
    evidence: tuple[EpisodeEvidenceInput, ...]
    entities: tuple[EpisodeEntity, ...]
    metadata: Mapping[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class EpisodeEncodingResult:
    """Aggregated outcome of encoding episodes from observations."""

    observations: int = 0
    candidates: int = 0
    created: int = 0
    updated: int = 0
    unchanged: int = 0
    deactivated: int = 0

    def record(self, status: EpisodeWriteStatus) -> EpisodeEncodingResult:
        """Return a new result with one upsert counter incremented."""
        return EpisodeEncodingResult(
            observations=self.observations,
            candidates=self.candidates,
            created=self.created + (1 if status is EpisodeWriteStatus.CREATED else 0),
            updated=self.updated + (1 if status is EpisodeWriteStatus.UPDATED else 0),
            unchanged=self.unchanged + (1 if status is EpisodeWriteStatus.UNCHANGED else 0),
            deactivated=self.deactivated,
        )
