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


class SemanticPolarity(StrEnum):
    """Whether a semantic claim affirms or denies a proposition."""

    AFFIRM = "affirm"
    DENY = "deny"


class SemanticCardinality(StrEnum):
    """Whether a slot allows one or many object values."""

    ONE = "one"
    MANY = "many"


class SemanticMemoryStatus(StrEnum):
    """Lifecycle status of a semantic memory."""

    ACTIVE = "active"
    CONTESTED = "contested"
    SUPERSEDED = "superseded"


class SemanticDerivationRelation(StrEnum):
    """How an episode relates to a semantic memory."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"


class SemanticWriteStatus(StrEnum):
    """Outcome of upserting a single semantic memory."""

    CREATED = "created"
    UPDATED = "updated"
    UNCHANGED = "unchanged"


def _validate_qualifiers(qualifiers: Mapping[str, Any]) -> None:
    for key in qualifiers:
        if not str(key).strip():
            raise ValidationError("qualifier keys must not be empty.")


@dataclass(frozen=True, slots=True)
class SemanticFactCandidate:
    """Atomic proposition extracted from one episode."""

    tenant_id: str
    source_episode_id: str
    subject_entity_id: str | None
    predicate: str
    object_value: str
    object_entity_id: str | None
    polarity: SemanticPolarity
    cardinality: SemanticCardinality
    confidence: float
    observed_at: datetime
    qualifiers: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValidationError("tenant_id must not be empty.")
        if not self.source_episode_id.strip():
            raise ValidationError("source_episode_id must not be empty.")
        if not self.predicate.strip():
            raise ValidationError("predicate must not be empty.")
        if not self.object_value.strip():
            raise ValidationError("object_value must not be empty.")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValidationError("confidence must be between 0.0 and 1.0.")
        if self.observed_at.tzinfo is None:
            raise ValidationError("observed_at must be timezone-aware.")
        object.__setattr__(self, "observed_at", self.observed_at.astimezone(UTC))
        _validate_qualifiers(self.qualifiers)
        object.__setattr__(self, "qualifiers", MappingProxyType(dict(self.qualifiers)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class SemanticDerivationInput:
    """Episode-level support or contradiction link for a semantic memory."""

    episode_id: str
    relation: SemanticDerivationRelation
    contribution_score: float

    def __post_init__(self) -> None:
        if not self.episode_id.strip():
            raise ValidationError("episode_id must not be empty.")
        if not math.isfinite(self.contribution_score):
            raise ValidationError("contribution_score must be finite.")
        if not 0.0 <= self.contribution_score <= 1.0:
            raise ValidationError("contribution_score must be between 0.0 and 1.0.")


@dataclass(frozen=True, slots=True)
class SemanticMemoryInput:
    """Candidate semantic memory ready for persistence."""

    tenant_id: str
    subject_id: str | None
    memory_key: str
    slot_key: str
    statement: str
    subject_entity_id: str | None
    predicate: str
    object_value: str
    object_entity_id: str | None
    polarity: SemanticPolarity
    cardinality: SemanticCardinality
    qualifiers: Mapping[str, Any]
    confidence: float
    importance: float
    status: SemanticMemoryStatus
    support_count: int
    contradiction_count: int
    first_supported_at: datetime
    last_supported_at: datetime
    derivations: tuple[SemanticDerivationInput, ...]
    observation_evidence: tuple[EpisodeEvidenceInput, ...]
    entities: tuple[EpisodeEntity, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValidationError("tenant_id must not be empty.")
        if not self.memory_key.strip():
            raise ValidationError("memory_key must not be empty.")
        if not self.slot_key.strip():
            raise ValidationError("slot_key must not be empty.")
        if not self.statement.strip():
            raise ValidationError("statement must not be empty.")
        if not self.predicate.strip():
            raise ValidationError("predicate must not be empty.")
        if not self.object_value.strip():
            raise ValidationError("object_value must not be empty.")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValidationError("confidence must be between 0.0 and 1.0.")
        if not math.isfinite(self.importance) or not 0.0 <= self.importance <= 1.0:
            raise ValidationError("importance must be between 0.0 and 1.0.")
        if self.support_count < 0:
            raise ValidationError("support_count must not be negative.")
        if self.contradiction_count < 0:
            raise ValidationError("contradiction_count must not be negative.")
        for label, ts in (
            ("first_supported_at", self.first_supported_at),
            ("last_supported_at", self.last_supported_at),
        ):
            if ts.tzinfo is None:
                raise ValidationError(f"{label} must be timezone-aware.")
        first = self.first_supported_at.astimezone(UTC)
        last = self.last_supported_at.astimezone(UTC)
        object.__setattr__(self, "first_supported_at", first)
        object.__setattr__(self, "last_supported_at", last)
        if last < first:
            raise ValidationError("last_supported_at must not be before first_supported_at.")
        _validate_qualifiers(self.qualifiers)
        object.__setattr__(self, "qualifiers", MappingProxyType(dict(self.qualifiers)))
        derivation_keys = {(d.episode_id, d.relation) for d in self.derivations}
        if len(derivation_keys) != len(self.derivations):
            raise ValidationError("derivation episode/relation pairs must be unique.")
        evidence_keys = {
            (e.observation_id, e.observation_revision) for e in self.observation_evidence
        }
        if len(evidence_keys) != len(self.observation_evidence):
            raise ValidationError("observation evidence revision pairs must be unique.")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class StoredSemanticMemory:
    """Persisted semantic memory with derivations and evidence."""

    id: str
    tenant_id: str
    subject_id: str | None
    memory_key: str
    slot_key: str
    statement: str
    subject_entity_id: str | None
    predicate: str
    object_value: str
    object_entity_id: str | None
    polarity: SemanticPolarity
    cardinality: SemanticCardinality
    qualifiers: Mapping[str, Any]
    confidence: float
    importance: float
    status: SemanticMemoryStatus
    support_count: int
    contradiction_count: int
    first_supported_at: datetime
    last_supported_at: datetime
    is_active: bool
    derivations: tuple[SemanticDerivationInput, ...]
    observation_evidence: tuple[EpisodeEvidenceInput, ...]
    entities: tuple[EpisodeEntity, ...]
    metadata: Mapping[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class SemanticExtractionResult:
    """Outcome of extracting semantic fact candidates from episodes."""

    candidates: tuple[SemanticFactCandidate, ...]
    failed: int = 0


@dataclass(frozen=True, slots=True)
class SemanticConsolidationResult:
    """Aggregated outcome of semantic consolidation."""

    episodes: int = 0
    extracted_candidates: int = 0
    extracted_failures: int = 0
    canonical_claims: int = 0
    promoted: int = 0
    created: int = 0
    updated: int = 0
    unchanged: int = 0
    contested: int = 0
    deactivated: int = 0

    def record(
        self,
        write_status: SemanticWriteStatus,
        memory_status: SemanticMemoryStatus,
    ) -> SemanticConsolidationResult:
        """Return a new result with upsert and status counters incremented."""
        return SemanticConsolidationResult(
            episodes=self.episodes,
            extracted_candidates=self.extracted_candidates,
            extracted_failures=self.extracted_failures,
            canonical_claims=self.canonical_claims,
            promoted=self.promoted + 1,
            created=self.created + (1 if write_status is SemanticWriteStatus.CREATED else 0),
            updated=self.updated + (1 if write_status is SemanticWriteStatus.UPDATED else 0),
            unchanged=self.unchanged + (1 if write_status is SemanticWriteStatus.UNCHANGED else 0),
            contested=self.contested
            + (1 if memory_status is SemanticMemoryStatus.CONTESTED else 0),
            deactivated=self.deactivated,
        )
