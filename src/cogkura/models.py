"""Additional package data models."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from cogkura.exceptions import ValidationError


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


class SemanticUpdateRelation(StrEnum):
    """How new semantic evidence relates to existing revision state."""

    REINFORCES = "reinforces"
    COEXISTS = "coexists"
    SUPERSEDES = "supersedes"
    CONFLICTS = "conflicts"


class SemanticWriteStatus(StrEnum):
    """Outcome of upserting a single semantic memory."""

    CREATED = "created"
    UPDATED = "updated"
    UNCHANGED = "unchanged"


def _validate_qualifiers(qualifiers: Mapping[str, Any]) -> None:
    for key in qualifiers:
        if not str(key).strip():
            raise ValidationError("qualifier keys must not be empty.")


def _validate_temporal_validity(
    *,
    valid_from: datetime | None,
    valid_until: datetime | None,
) -> tuple[datetime | None, datetime | None]:
    normalized_from = None
    normalized_until = None
    if valid_from is not None:
        if valid_from.tzinfo is None:
            raise ValidationError("valid_from must be timezone-aware.")
        normalized_from = valid_from.astimezone(UTC)
    if valid_until is not None:
        if valid_until.tzinfo is None:
            raise ValidationError("valid_until must be timezone-aware.")
        normalized_until = valid_until.astimezone(UTC)
    if normalized_from is not None and normalized_until is not None:
        if normalized_until <= normalized_from:
            raise ValidationError("valid_until must be after valid_from.")
    return normalized_from, normalized_until


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
    valid_from: datetime | None = None
    valid_until: datetime | None = None
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
        normalized_from, normalized_until = _validate_temporal_validity(
            valid_from=self.valid_from,
            valid_until=self.valid_until,
        )
        object.__setattr__(self, "valid_from", normalized_from)
        object.__setattr__(self, "valid_until", normalized_until)
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
class SemanticRevisionCandidate:
    """Revision-aware semantic candidate produced by consolidation."""

    tenant_id: str
    memory_key: str
    slot_key: str
    revision_key: str
    statement: str
    subject_id: str | None
    subject_entity_id: str | None
    predicate: str
    object_value: str
    object_entity_id: str | None
    polarity: SemanticPolarity
    cardinality: SemanticCardinality
    qualifiers: Mapping[str, Any]
    valid_from: datetime | None
    valid_until: datetime | None
    support_count: int
    first_supported_at: datetime
    last_supported_at: datetime
    support_confidence: float
    importance: float
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
        if not self.revision_key.strip():
            raise ValidationError("revision_key must not be empty.")
        if not self.statement.strip():
            raise ValidationError("statement must not be empty.")
        if self.support_count < 0:
            raise ValidationError("support_count must not be negative.")
        normalized_from, normalized_until = _validate_temporal_validity(
            valid_from=self.valid_from,
            valid_until=self.valid_until,
        )
        object.__setattr__(self, "valid_from", normalized_from)
        object.__setattr__(self, "valid_until", normalized_until)
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
        if not math.isfinite(self.support_confidence) or not 0.0 <= self.support_confidence <= 1.0:
            raise ValidationError("support_confidence must be between 0.0 and 1.0.")
        if not math.isfinite(self.importance) or not 0.0 <= self.importance <= 1.0:
            raise ValidationError("importance must be between 0.0 and 1.0.")
        _validate_qualifiers(self.qualifiers)
        object.__setattr__(self, "qualifiers", MappingProxyType(dict(self.qualifiers)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class SemanticRevisionInput:
    """Desired semantic revision state from reconciliation."""

    tenant_id: str
    memory_key: str
    revision_key: str
    revision_number: int
    status: SemanticMemoryStatus
    valid_from: datetime | None
    valid_until: datetime | None
    confidence: float
    importance: float
    support_count: int
    contradiction_count: int
    first_supported_at: datetime
    last_supported_at: datetime
    derivations: tuple[SemanticDerivationInput, ...]
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValidationError("tenant_id must not be empty.")
        if not self.memory_key.strip():
            raise ValidationError("memory_key must not be empty.")
        if not self.revision_key.strip():
            raise ValidationError("revision_key must not be empty.")
        if self.revision_number <= 0:
            raise ValidationError("revision_number must be greater than zero.")
        normalized_from, normalized_until = _validate_temporal_validity(
            valid_from=self.valid_from,
            valid_until=self.valid_until,
        )
        object.__setattr__(self, "valid_from", normalized_from)
        object.__setattr__(self, "valid_until", normalized_until)
        for label, value in (
            ("confidence", self.confidence),
            ("importance", self.importance),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValidationError(f"{label} must be between 0.0 and 1.0.")
        if self.support_count < 0 or self.contradiction_count < 0:
            raise ValidationError("support and contradiction counts must not be negative.")
        for label, ts in (
            ("first_supported_at", self.first_supported_at),
            ("last_supported_at", self.last_supported_at),
        ):
            if ts.tzinfo is None:
                raise ValidationError(f"{label} must be timezone-aware.")
        object.__setattr__(self, "first_supported_at", self.first_supported_at.astimezone(UTC))
        object.__setattr__(self, "last_supported_at", self.last_supported_at.astimezone(UTC))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class StoredSemanticRevision:
    """Persisted semantic revision with temporal validity."""

    revision_key: str
    memory_key: str
    tenant_id: str
    revision_number: int
    status: SemanticMemoryStatus
    valid_from: datetime | None
    valid_until: datetime | None
    confidence: float
    importance: float
    support_count: int
    contradiction_count: int
    first_supported_at: datetime
    last_supported_at: datetime
    derivations: tuple[SemanticDerivationInput, ...]
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        if not self.revision_key.strip():
            raise ValidationError("revision_key must not be empty.")
        if not self.memory_key.strip():
            raise ValidationError("memory_key must not be empty.")
        if not self.tenant_id.strip():
            raise ValidationError("tenant_id must not be empty.")
        if self.revision_number <= 0:
            raise ValidationError("revision_number must be greater than zero.")
        normalized_from, normalized_until = _validate_temporal_validity(
            valid_from=self.valid_from,
            valid_until=self.valid_until,
        )
        object.__setattr__(self, "valid_from", normalized_from)
        object.__setattr__(self, "valid_until", normalized_until)
        for label, value in (
            ("confidence", self.confidence),
            ("importance", self.importance),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValidationError(f"{label} must be between 0.0 and 1.0.")
        if self.support_count < 0 or self.contradiction_count < 0:
            raise ValidationError("support and contradiction counts must not be negative.")
        for label, ts in (
            ("first_supported_at", self.first_supported_at),
            ("last_supported_at", self.last_supported_at),
            ("created_at", self.created_at),
            ("updated_at", self.updated_at),
        ):
            if ts.tzinfo is None:
                raise ValidationError(f"{label} must be timezone-aware.")
            object.__setattr__(self, label, ts.astimezone(UTC))


@dataclass(frozen=True, slots=True)
class SemanticRevisionRelation:
    """Persisted relationship between semantic revisions."""

    tenant_id: str
    left_revision_key: str
    right_revision_key: str
    relation: SemanticUpdateRelation
    effective_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValidationError("tenant_id must not be empty.")
        if not self.left_revision_key.strip() or not self.right_revision_key.strip():
            raise ValidationError("revision relation keys must not be empty.")
        if self.relation not in (
            SemanticUpdateRelation.SUPERSEDES,
            SemanticUpdateRelation.CONFLICTS,
        ):
            raise ValidationError("Only SUPERSEDES and CONFLICTS relations are persisted.")
        if self.effective_at is not None:
            if self.effective_at.tzinfo is None:
                raise ValidationError("effective_at must be timezone-aware.")
            object.__setattr__(self, "effective_at", self.effective_at.astimezone(UTC))


@dataclass(frozen=True, slots=True)
class SemanticReconciliationPlan:
    """Desired semantic state after reconciliation."""

    current_memories: tuple[SemanticMemoryInput, ...]
    revisions: tuple[SemanticRevisionInput, ...]
    relations: tuple[SemanticRevisionRelation, ...]
    reinforced_count: int = 0
    coexist_count: int = 0
    conflict_count: int = 0
    superseded_count: int = 0
    revisions_created: int = 0
    revisions_updated: int = 0


@dataclass(frozen=True, slots=True)
class SemanticReconciliationWriteResult:
    """Outcome of applying a reconciliation plan."""

    created: int = 0
    updated: int = 0
    unchanged: int = 0
    revisions_created: int = 0
    revisions_updated: int = 0
    relations_written: int = 0


@dataclass(frozen=True, slots=True)
class SemanticMemoryInput:
    """Candidate semantic memory ready for persistence."""

    tenant_id: str
    subject_id: str | None
    memory_key: str
    slot_key: str
    revision_key: str
    revision_number: int
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
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    entities: tuple[EpisodeEntity, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValidationError("tenant_id must not be empty.")
        if not self.memory_key.strip():
            raise ValidationError("memory_key must not be empty.")
        if not self.slot_key.strip():
            raise ValidationError("slot_key must not be empty.")
        if not self.revision_key.strip():
            raise ValidationError("revision_key must not be empty.")
        if self.revision_number <= 0:
            raise ValidationError("revision_number must be greater than zero.")
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
        normalized_from, normalized_until = _validate_temporal_validity(
            valid_from=self.valid_from,
            valid_until=self.valid_until,
        )
        object.__setattr__(self, "valid_from", normalized_from)
        object.__setattr__(self, "valid_until", normalized_until)
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
    revision_key: str
    revision_number: int
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
    valid_from: datetime | None
    valid_until: datetime | None
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
    reinforced: int = 0
    coexisting: int = 0
    conflicts: int = 0
    superseded: int = 0
    revisions_created: int = 0
    revisions_updated: int = 0

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
            reinforced=self.reinforced,
            coexisting=self.coexisting,
            conflicts=self.conflicts,
            superseded=self.superseded,
            revisions_created=self.revisions_created,
            revisions_updated=self.revisions_updated,
        )

    def with_reconciliation(
        self,
        *,
        reinforced: int,
        coexisting: int,
        conflicts: int,
        superseded: int,
        revisions_created: int,
        revisions_updated: int,
    ) -> SemanticConsolidationResult:
        return SemanticConsolidationResult(
            episodes=self.episodes,
            extracted_candidates=self.extracted_candidates,
            extracted_failures=self.extracted_failures,
            canonical_claims=self.canonical_claims,
            promoted=self.promoted,
            created=self.created,
            updated=self.updated,
            unchanged=self.unchanged,
            contested=self.contested,
            deactivated=self.deactivated,
            reinforced=reinforced,
            coexisting=coexisting,
            conflicts=conflicts,
            superseded=superseded,
            revisions_created=revisions_created,
            revisions_updated=revisions_updated,
        )


class MemoryKind(StrEnum):
    """Kind of durable memory returned by declarative recall."""

    EPISODE = "episode"
    SEMANTIC = "semantic"


class ActivationReferenceKind(StrEnum):
    """How a memory access reference was produced."""

    RETRIEVED = "retrieved"
    REHEARSED = "rehearsed"


class CognitiveTraceOrigin(StrEnum):
    """How a cognitive activation trace was produced."""

    ENCODED = "encoded"
    SUPPORTED = "supported"
    RECORDED_ACCESS = "recorded_access"
    LEARNING_REINFORCEMENT = "learning_reinforcement"
    AGGREGATE_SUPPORT_FALLBACK = "aggregate_support_fallback"


@dataclass(frozen=True, slots=True)
class CognitiveReferenceTrace:
    """Derived or persisted cognitive evidence trace for activation."""

    origin: CognitiveTraceOrigin
    referenced_at: datetime
    weight: int = 1
    episode_id: str | None = None
    revision_key: str | None = None

    def __post_init__(self) -> None:
        if self.referenced_at.tzinfo is None:
            raise ValidationError("referenced_at must be timezone-aware.")
        object.__setattr__(self, "referenced_at", self.referenced_at.astimezone(UTC))
        if self.weight <= 0:
            raise ValidationError("weight must be greater than zero.")
        if self.episode_id is not None and not self.episode_id.strip():
            raise ValidationError("episode_id must not be empty when provided.")
        if self.revision_key is not None and not self.revision_key.strip():
            raise ValidationError("revision_key must not be empty when provided.")

    def to_activation_trace(self) -> ActivationReferenceTrace:
        return ActivationReferenceTrace(referenced_at=self.referenced_at, weight=self.weight)


@dataclass(frozen=True, slots=True)
class MemoryIdentity:
    """Stable identifier for activation history lookup."""

    memory_kind: MemoryKind
    memory_key: str

    def __post_init__(self) -> None:
        if not self.memory_key.strip():
            raise ValidationError("memory_key must not be empty.")


@dataclass(frozen=True, slots=True)
class MemoryReference:
    """Recorded access to a durable memory for base-level activation."""

    tenant_id: str
    memory_kind: MemoryKind
    memory_key: str
    reference_kind: ActivationReferenceKind
    referenced_at: datetime
    request_id: str | None = None
    weight: int = 1
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValidationError("tenant_id must not be empty.")
        if not self.memory_key.strip():
            raise ValidationError("memory_key must not be empty.")
        if self.referenced_at.tzinfo is None:
            raise ValidationError("referenced_at must be timezone-aware.")
        if self.weight <= 0:
            raise ValidationError("weight must be greater than zero.")
        object.__setattr__(self, "referenced_at", self.referenced_at.astimezone(UTC))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def identity(self) -> MemoryIdentity:
        return MemoryIdentity(memory_kind=self.memory_kind, memory_key=self.memory_key)


@dataclass(frozen=True, slots=True)
class RetrievalCue:
    """Structured retrieval cue for declarative activation."""

    text: str | None = None
    subject_id: str | None = None
    entity_ids: tuple[str, ...] = ()
    predicate: str | None = None
    object_value: str | None = None
    qualifiers: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "qualifiers", MappingProxyType(dict(self.qualifiers)))
        if not any(
            (
                self.text and self.text.strip(),
                self.subject_id and self.subject_id.strip(),
                self.entity_ids,
                self.predicate and self.predicate.strip(),
                self.object_value and self.object_value.strip(),
                self.qualifiers,
            )
        ):
            raise ValidationError("Retrieval cue must contain at least one field.")


@dataclass(frozen=True, slots=True)
class ActivationConfig:
    """Configuration for ACT-R declarative activation."""

    decay: float = 0.5
    base_level_constant: float = 0.0
    source_activation: float = 1.0
    maximum_associative_strength: float = 1.0
    mismatch_penalty: float = 1.0
    retrieval_threshold: float = -3.0
    latency_factor: float = 1.0
    latency_exponent: float = 1.0
    time_unit_seconds: float = 3600.0
    minimum_elapsed_seconds: float = 1.0
    enable_spreading_activation: bool = True
    spreading_decay: float = 0.5
    spreading_max_hops: int = 2
    spreading_min_activation: float = 0.01
    enable_partial_matching: bool = True
    enable_noise: bool = False
    max_candidates: int = 10_000
    learned_association_scale: float = 0.25
    enable_candidate_idf: bool = True
    enable_duplicate_collapse: bool = True
    duplicate_jaccard_threshold: float = 0.75
    current_state_weight: float = 0.5
    current_state_cue_tokens: frozenset[str] = frozenset(
        {"currently", "current", "now", "live", "today"}
    )
    enable_text_entity_seeding: bool = True
    seeded_entity_partial_match_weight: float = 0.75
    enable_semantic_slot_admission: bool = True
    enable_entity_slot_admission: bool = True
    collapse_normalize_numeric_tokens: bool = True
    access_minimum_score: float | None = None
    access_burst_limit: int | None = None
    access_burst_window_seconds: float = 3600.0
    slot_admission_requires_current_state_or_predicate: bool = True
    force_slot_admission: bool = False
    enable_multi_entity_conjunction: bool = True
    conjunction_weight: float = 0.5
    distinctive_token_idf_scale: float = 1.5
    incident_cue_tokens: frozenset[str] = frozenset(
        {"incident", "on", "call", "alert", "woke", "3am", "am", "overnight"}
    )
    enable_incident_tag_seeding: bool = True
    enable_text_precision_matching: bool = True
    exclude_superseded_support_on_current_state: bool = True
    collapse_same_slot_support: bool = True
    enable_lexical_slot_matching: bool = True
    lexical_slot_min_overlap: int = 1
    semantic_soft_admission_floor: float = -4.0
    max_soft_admitted_semantics: int = 8
    enable_semantic_evidence_linking: bool = True
    max_evidence_link_derivations: int = 3
    semantic_current_min_relevance: float = 0.06

    def __post_init__(self) -> None:
        if not 0.0 < self.decay <= 1.0:
            raise ValidationError("decay must be greater than zero and at most 1.0.")
        if self.source_activation < 0:
            raise ValidationError("source_activation must not be negative.")
        if self.maximum_associative_strength < 0:
            raise ValidationError("maximum_associative_strength must not be negative.")
        if not 0.0 < self.spreading_decay <= 1.0:
            raise ValidationError("spreading_decay must be greater than zero and at most 1.0.")
        if self.spreading_max_hops < 1:
            raise ValidationError("spreading_max_hops must be at least 1.")
        if self.spreading_min_activation < 0:
            raise ValidationError("spreading_min_activation must not be negative.")
        if self.time_unit_seconds <= 0:
            raise ValidationError("time_unit_seconds must be greater than zero.")
        if self.minimum_elapsed_seconds <= 0:
            raise ValidationError("minimum_elapsed_seconds must be greater than zero.")
        if self.mismatch_penalty < 0:
            raise ValidationError("mismatch_penalty must not be negative.")
        if self.latency_factor <= 0:
            raise ValidationError("latency_factor must be greater than zero.")
        if self.latency_exponent <= 0:
            raise ValidationError("latency_exponent must be greater than zero.")
        if self.max_candidates <= 0:
            raise ValidationError("max_candidates must be greater than zero.")
        if self.enable_noise:
            raise ValidationError("enable_noise is not supported in this release.")
        if not 0.0 <= self.learned_association_scale <= 1.0:
            raise ValidationError("learned_association_scale must be between 0.0 and 1.0.")
        if not 0.0 <= self.duplicate_jaccard_threshold <= 1.0:
            raise ValidationError("duplicate_jaccard_threshold must be between 0.0 and 1.0.")
        if self.current_state_weight < 0:
            raise ValidationError("current_state_weight must not be negative.")
        if self.seeded_entity_partial_match_weight < 0:
            raise ValidationError("seeded_entity_partial_match_weight must not be negative.")
        if self.access_minimum_score is not None and not (0.0 <= self.access_minimum_score <= 1.0):
            raise ValidationError("access_minimum_score must be between 0.0 and 1.0.")
        if self.access_burst_limit is not None and self.access_burst_limit <= 0:
            raise ValidationError("access_burst_limit must be greater than zero.")
        if self.access_burst_window_seconds <= 0:
            raise ValidationError("access_burst_window_seconds must be greater than zero.")
        if self.conjunction_weight < 0:
            raise ValidationError("conjunction_weight must not be negative.")
        if self.distinctive_token_idf_scale <= 0:
            raise ValidationError("distinctive_token_idf_scale must be greater than zero.")
        if self.lexical_slot_min_overlap < 1:
            raise ValidationError("lexical_slot_min_overlap must be at least 1.")
        if not math.isfinite(self.semantic_soft_admission_floor):
            raise ValidationError("semantic_soft_admission_floor must be finite.")
        if self.max_soft_admitted_semantics < 1:
            raise ValidationError("max_soft_admitted_semantics must be at least 1.")
        if self.max_evidence_link_derivations < 1:
            raise ValidationError("max_evidence_link_derivations must be at least 1.")
        if not 0.0 <= self.semantic_current_min_relevance <= 1.0:
            raise ValidationError("semantic_current_min_relevance must be between 0.0 and 1.0.")


@dataclass(frozen=True, slots=True)
class ActivationComponents:
    """Decomposed activation values for a recalled memory."""

    base_level: float
    spreading: float
    partial_match: float
    noise: float
    total: float
    current_state: float = 0.0

    def __post_init__(self) -> None:
        for label, value in (
            ("base_level", self.base_level),
            ("spreading", self.spreading),
            ("partial_match", self.partial_match),
            ("noise", self.noise),
            ("total", self.total),
            ("current_state", self.current_state),
        ):
            if not math.isfinite(value):
                raise ValidationError(f"{label} must be finite.")


class RetrievalEligibility(StrEnum):
    """How a candidate became eligible for returned recall results."""

    THRESHOLD = "threshold"
    SEMANTIC_SLOT_ADMISSION = "semantic_slot_admission"
    ENTITY_SLOT_ADMISSION = "entity_slot_admission"
    HISTORICAL_SLOT_ADMISSION = "historical_slot_admission"
    SEMANTIC_CURRENT_ADMISSION = "semantic_current_admission"


class SlotFitSource(StrEnum):
    """Source of structured slot-fit evidence for ranking diagnostics."""

    SEMANTIC = "semantic"
    SUPPORT = "support"


class RelevanceTier(StrEnum):
    """Specificity tier for semantic relevance-based competition."""

    DIRECT_VALUE = "direct_value"
    DIRECT_SEMANTIC = "direct_semantic"
    ENTITY_ASSOCIATION = "entity_association"
    EVIDENCE_ASSOCIATION = "evidence_association"
    CONTEXTUAL = "contextual"


@dataclass(frozen=True, slots=True)
class SupportProvenance:
    """Derivation-backed semantic provenance for a SUPPORT episode."""

    semantic_memory_key: str
    semantic_revision_key: str
    semantic_slot_key: str
    semantic_status: SemanticMemoryStatus
    contribution_score: float
    slot_fit: float | None = None
    semantic_valid_from: datetime | None = None
    semantic_valid_until: datetime | None = None

    def __post_init__(self) -> None:
        if not self.semantic_memory_key.strip():
            raise ValidationError("semantic_memory_key must not be empty.")
        if not self.semantic_revision_key.strip():
            raise ValidationError("semantic_revision_key must not be empty.")
        if not self.semantic_slot_key.strip():
            raise ValidationError("semantic_slot_key must not be empty.")
        if not math.isfinite(self.contribution_score):
            raise ValidationError("contribution_score must be finite.")
        if not 0.0 <= self.contribution_score <= 1.0:
            raise ValidationError("contribution_score must be between 0.0 and 1.0.")
        if self.slot_fit is not None:
            if not math.isfinite(self.slot_fit):
                raise ValidationError("slot_fit must be finite when provided.")
            if not 0.0 <= self.slot_fit <= 1.0:
                raise ValidationError("slot_fit must be between 0.0 and 1.0 when provided.")
        if self.semantic_valid_from is not None:
            if self.semantic_valid_from.tzinfo is None:
                raise ValidationError("semantic_valid_from must be timezone-aware.")
            object.__setattr__(
                self,
                "semantic_valid_from",
                self.semantic_valid_from.astimezone(UTC),
            )
        if self.semantic_valid_until is not None:
            if self.semantic_valid_until.tzinfo is None:
                raise ValidationError("semantic_valid_until must be timezone-aware.")
            object.__setattr__(
                self,
                "semantic_valid_until",
                self.semantic_valid_until.astimezone(UTC),
            )
        if (
            self.semantic_valid_from is not None
            and self.semantic_valid_until is not None
            and self.semantic_valid_until <= self.semantic_valid_from
        ):
            raise ValidationError("semantic_valid_until must be after semantic_valid_from.")


@dataclass(frozen=True, slots=True)
class AssociationPath:
    """Winning deterministic association path for inspectable recall."""

    seed_episode_id: str
    matched_features: tuple[str, ...]
    bridge_entity_id: str | None = None
    related_episode_id: str | None = None
    hop_kind: str = "entity"
    weight: float = 0.0

    def __post_init__(self) -> None:
        if not self.seed_episode_id.strip():
            raise ValidationError("seed_episode_id must not be empty.")
        if not self.hop_kind.strip():
            raise ValidationError("hop_kind must not be empty.")
        if not math.isfinite(self.weight) or not 0.0 <= self.weight <= 1.0:
            raise ValidationError("weight must be between 0.0 and 1.0.")


@dataclass(frozen=True, slots=True)
class RetrievalDiagnostics:
    """Structured recall diagnostics for ranking and provenance analysis."""

    rank_activation: float
    accessibility_partial: float
    ranking_partial: float
    conjunction: float
    text_coverage: float
    text_cue_fit: float
    temporal_mode: str
    direct_cue_fit: float = 0.0
    direct_value_fit: float = 0.0
    direct_predicate_fit: float = 0.0
    evidence_linked_fit: float = 0.0
    associative_fit: float = 0.0
    semantic_relevance: float = 0.0
    relevance_tier: str = RelevanceTier.CONTEXTUAL.value
    matched_direct_features: tuple[str, ...] = ()
    matched_evidence_features: tuple[str, ...] = ()
    association_path: AssociationPath | None = None
    eligibility: RetrievalEligibility = RetrievalEligibility.THRESHOLD
    base_level: float = 0.0
    spreading: float = 0.0
    current_state: float = 0.0
    reference_count: int = 0
    matched_entities: int = 0
    cue_entity_count: int = 0
    spread_hop: int | None = None
    spread_sources: tuple[str, ...] = ()
    learned_edges: int = 0
    slot_fit: float | None = None
    structured_adjustment: float = 0.0
    slot_fit_source: SlotFitSource | None = None
    admission_reason: str | None = None
    soft_admitted: bool = False
    semantic_slot_key: str | None = None
    semantic_status: SemanticMemoryStatus | None = None
    observation_evidence_ids: tuple[str, ...] = ()
    support_provenance: tuple[SupportProvenance, ...] = ()
    selected_support_revision_key: str | None = None
    collapse_key: str | None = None
    collapse_reason: str | None = None
    collapsed_into: str | None = None
    canonical_object_value: str | None = None
    cardinality: str | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("rank_activation", self.rank_activation),
            ("accessibility_partial", self.accessibility_partial),
            ("ranking_partial", self.ranking_partial),
            ("conjunction", self.conjunction),
            ("text_coverage", self.text_coverage),
            ("text_cue_fit", self.text_cue_fit),
            ("base_level", self.base_level),
            ("spreading", self.spreading),
            ("current_state", self.current_state),
            ("structured_adjustment", self.structured_adjustment),
        ):
            if not math.isfinite(value):
                raise ValidationError(f"{label} must be finite.")
        for label, value in (
            ("text_coverage", self.text_coverage),
            ("text_cue_fit", self.text_cue_fit),
            ("direct_cue_fit", self.direct_cue_fit),
            ("direct_value_fit", self.direct_value_fit),
            ("direct_predicate_fit", self.direct_predicate_fit),
            ("evidence_linked_fit", self.evidence_linked_fit),
            ("associative_fit", self.associative_fit),
            ("semantic_relevance", self.semantic_relevance),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValidationError(f"{label} must be between 0.0 and 1.0.")
        if self.reference_count < 0:
            raise ValidationError("reference_count must not be negative.")
        if self.matched_entities < 0:
            raise ValidationError("matched_entities must not be negative.")
        if self.cue_entity_count < 0:
            raise ValidationError("cue_entity_count must not be negative.")
        if self.matched_entities > self.cue_entity_count and self.cue_entity_count > 0:
            raise ValidationError("matched_entities must not exceed cue_entity_count.")
        if self.learned_edges < 0:
            raise ValidationError("learned_edges must not be negative.")
        if self.slot_fit is not None:
            if not math.isfinite(self.slot_fit):
                raise ValidationError("slot_fit must be finite when provided.")
            if not 0.0 <= self.slot_fit <= 1.0:
                raise ValidationError("slot_fit must be between 0.0 and 1.0 when provided.")
        if self.slot_fit_source is not None and self.slot_fit is None:
            raise ValidationError("slot_fit_source requires slot_fit.")
        if self.admission_reason is not None and not self.admission_reason.strip():
            raise ValidationError("admission_reason must not be empty when provided.")
        if (
            self.selected_support_revision_key is not None
            and not self.selected_support_revision_key.strip()
        ):
            raise ValidationError("selected_support_revision_key must not be empty when provided.")
        for observation_id in self.observation_evidence_ids:
            if not observation_id.strip():
                raise ValidationError("observation_evidence_ids must not contain empty values.")
        if self.spread_hop is not None and self.spread_hop < 0:
            raise ValidationError("spread_hop must not be negative when provided.")
        if self.collapse_reason is not None and not self.collapse_reason.strip():
            raise ValidationError("collapse_reason must not be empty when provided.")
        if self.collapsed_into is not None and not self.collapsed_into.strip():
            raise ValidationError("collapsed_into must not be empty when provided.")
        if self.collapse_key is not None and not self.collapse_key.strip():
            raise ValidationError("collapse_key must not be empty when provided.")
        if self.canonical_object_value is not None and not self.canonical_object_value.strip():
            raise ValidationError("canonical_object_value must not be empty when provided.")
        if self.cardinality is not None and not self.cardinality.strip():
            raise ValidationError("cardinality must not be empty when provided.")


@dataclass(frozen=True, slots=True)
class ActivationCandidate:
    """Common representation for episodic and semantic activation ranking."""

    memory_kind: MemoryKind
    memory_key: str
    created_at: datetime
    text: str
    subject_id: str | None
    entity_ids: tuple[str, ...]
    predicate: str | None
    object_value: str | None
    qualifiers: Mapping[str, Any]
    memory: StoredEpisode | StoredSemanticMemory
    importance: float = 0.5
    slot_key: str | None = None
    semantic_status: SemanticMemoryStatus | None = None
    last_supported_at: datetime | None = None
    support_provenance: tuple[SupportProvenance, ...] = ()
    cognitive_traces: tuple[CognitiveReferenceTrace, ...] = ()

    def __post_init__(self) -> None:
        if not self.memory_key.strip():
            raise ValidationError("memory_key must not be empty.")
        if not self.text.strip():
            raise ValidationError("text must not be empty.")
        if self.created_at.tzinfo is None:
            raise ValidationError("created_at must be timezone-aware.")
        object.__setattr__(self, "created_at", self.created_at.astimezone(UTC))
        object.__setattr__(self, "qualifiers", MappingProxyType(dict(self.qualifiers)))
        if not math.isfinite(self.importance) or not 0.0 <= self.importance <= 1.0:
            raise ValidationError("importance must be between 0.0 and 1.0.")
        if self.last_supported_at is not None:
            if self.last_supported_at.tzinfo is None:
                raise ValidationError("last_supported_at must be timezone-aware.")
            object.__setattr__(
                self,
                "last_supported_at",
                self.last_supported_at.astimezone(UTC),
            )
        for trace in self.cognitive_traces:
            if trace.referenced_at.tzinfo is None:
                raise ValidationError("cognitive_traces must be timezone-aware.")

    @property
    def identity(self) -> MemoryIdentity:
        return MemoryIdentity(memory_kind=self.memory_kind, memory_key=self.memory_key)


@dataclass(frozen=True, slots=True)
class RecallResult:
    """A declaratively activated memory match for a retrieval cue."""

    memory_kind: MemoryKind
    memory: StoredEpisode | StoredSemanticMemory
    activation: float
    score: float
    latency_seconds: float
    components: ActivationComponents
    reason: str
    diagnostics: RetrievalDiagnostics | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.activation):
            raise ValidationError("activation must be finite.")
        if not math.isfinite(self.score):
            raise ValidationError("score must be finite.")
        if not 0.0 <= self.score <= 1.0:
            raise ValidationError("score must be between 0.0 and 1.0.")
        if not math.isfinite(self.latency_seconds) or self.latency_seconds < 0:
            raise ValidationError("latency_seconds must be finite and non-negative.")


class RecallInspectionDisposition(StrEnum):
    """Terminal disposition for a recall inspection candidate."""

    RETURNED = "returned"
    BELOW_THRESHOLD = "below_threshold"
    FILTERED_FORGOTTEN = "filtered_forgotten"
    FILTERED_VALID_TIME = "filtered_valid_time"
    FILTERED_SEMANTIC_STATUS = "filtered_semantic_status"
    FILTERED_SUPERSEDED_SUPPORT = "filtered_superseded_support"
    FILTERED_BELOW_SOFT_FLOOR = "filtered_below_soft_floor"
    FILTERED_INSUFFICIENT_RELEVANCE = "filtered_insufficient_relevance"
    COLLAPSED = "collapsed"
    LIMITED = "limited"


@dataclass(frozen=True, slots=True)
class RecallInspectionCandidate:
    """One evaluated recall candidate with activation diagnostics."""

    memory_kind: MemoryKind
    memory: StoredEpisode | StoredSemanticMemory
    disposition: RecallInspectionDisposition
    activation: float
    score: float
    retrieval_threshold: float
    passed_threshold: bool
    soft_admitted: bool
    components: ActivationComponents
    cognitive_traces: tuple[CognitiveReferenceTrace, ...]
    stored_traces: tuple[ActivationReferenceTrace, ...]
    retention_state: MemoryRetentionState | None = None
    rank: int | None = None
    diagnostics: RetrievalDiagnostics | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.activation):
            raise ValidationError("activation must be finite.")
        if not math.isfinite(self.score):
            raise ValidationError("score must be finite.")
        if not 0.0 <= self.score <= 1.0:
            raise ValidationError("score must be between 0.0 and 1.0.")
        if not math.isfinite(self.retrieval_threshold):
            raise ValidationError("retrieval_threshold must be finite.")
        if self.rank is not None and self.rank <= 0:
            raise ValidationError("rank must be greater than zero when provided.")


@dataclass(frozen=True, slots=True)
class RecallInspectionResult:
    """Bounded recall inspection with accepted and rejected candidates."""

    tenant_id: str
    subject_id: str | None
    inspected_at: datetime
    retrieval_threshold: float
    returned: tuple[RecallInspectionCandidate, ...]
    rejected: tuple[RecallInspectionCandidate, ...]
    truncated: bool = False
    considered_count: int = 0
    canonical_query_features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValidationError("tenant_id must not be empty.")
        if self.inspected_at.tzinfo is None:
            raise ValidationError("inspected_at must be timezone-aware.")
        object.__setattr__(self, "inspected_at", self.inspected_at.astimezone(UTC))
        if not math.isfinite(self.retrieval_threshold):
            raise ValidationError("retrieval_threshold must be finite.")
        if self.considered_count < 0:
            raise ValidationError("considered_count must not be negative.")


class MemoryRetentionState(StrEnum):
    """Cognitive accessibility state for a durable memory."""

    ACTIVE = "active"
    FADING = "fading"
    FORGOTTEN = "forgotten"


@dataclass(frozen=True, slots=True)
class ActivationReferenceTrace:
    """Weighted activation history trace for base-level calculation."""

    referenced_at: datetime
    weight: int = 1

    def __post_init__(self) -> None:
        if self.referenced_at.tzinfo is None:
            raise ValidationError("referenced_at must be timezone-aware.")
        object.__setattr__(self, "referenced_at", self.referenced_at.astimezone(UTC))
        if self.weight <= 0:
            raise ValidationError("weight must be greater than zero.")


@dataclass(frozen=True, slots=True)
class ForgettingConfig:
    """Configuration for cognitive forgetting and reference compaction."""

    enabled: bool = True
    fading_retention_threshold: float = 0.25
    forgotten_retention_threshold: float = 0.05
    grace_period_seconds: float = 604_800.0
    exclude_forgotten_from_recall: bool = True
    enable_reference_compaction: bool = True
    compact_after_seconds: float = 2_592_000.0
    compaction_bucket_seconds: float = 86_400.0
    enable_importance_scaling: bool = True
    importance_retention_floor: float = 0.15
    protect_semantic_support: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.forgotten_retention_threshold < self.fading_retention_threshold <= 1.0:
            raise ValidationError("Retention thresholds must satisfy 0 <= forgotten < fading <= 1.")
        if self.grace_period_seconds <= 0:
            raise ValidationError("grace_period_seconds must be greater than zero.")
        if self.compact_after_seconds <= 0:
            raise ValidationError("compact_after_seconds must be greater than zero.")
        if self.compaction_bucket_seconds <= 0:
            raise ValidationError("compaction_bucket_seconds must be greater than zero.")
        if not 0.0 <= self.importance_retention_floor <= 1.0:
            raise ValidationError("importance_retention_floor must be between 0.0 and 1.0.")


@dataclass(frozen=True, slots=True)
class StoredMemoryDynamics:
    """Persisted forgetting lifecycle state for a durable memory."""

    tenant_id: str
    memory_kind: MemoryKind
    memory_key: str
    retention_state: MemoryRetentionState
    last_base_level: float
    last_retention_score: float
    below_threshold_since: datetime | None
    forgotten_at: datetime | None
    evaluated_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValidationError("tenant_id must not be empty.")
        if not self.memory_key.strip():
            raise ValidationError("memory_key must not be empty.")
        for label, value in (
            ("last_base_level", self.last_base_level),
            ("last_retention_score", self.last_retention_score),
        ):
            if not math.isfinite(value):
                raise ValidationError(f"{label} must be finite.")
        if not 0.0 <= self.last_retention_score <= 1.0:
            raise ValidationError("last_retention_score must be between 0.0 and 1.0.")
        for label, timestamp in (
            ("evaluated_at", self.evaluated_at),
            ("updated_at", self.updated_at),
        ):
            if timestamp.tzinfo is None:
                raise ValidationError(f"{label} must be timezone-aware.")
            object.__setattr__(self, label, timestamp.astimezone(UTC))
        if self.below_threshold_since is not None:
            if self.below_threshold_since.tzinfo is None:
                raise ValidationError("below_threshold_since must be timezone-aware.")
            object.__setattr__(
                self,
                "below_threshold_since",
                self.below_threshold_since.astimezone(UTC),
            )
        if self.forgotten_at is not None:
            if self.forgotten_at.tzinfo is None:
                raise ValidationError("forgotten_at must be timezone-aware.")
            object.__setattr__(self, "forgotten_at", self.forgotten_at.astimezone(UTC))

    @property
    def identity(self) -> MemoryIdentity:
        return MemoryIdentity(memory_kind=self.memory_kind, memory_key=self.memory_key)


@dataclass(frozen=True, slots=True)
class ForgettingDecision:
    """Outcome of evaluating one memory's forgetting state."""

    dynamics: StoredMemoryDynamics
    previous_state: MemoryRetentionState | None
    reactivated: bool = False


@dataclass(frozen=True, slots=True)
class ForgettingResult:
    """Aggregated outcome of forgetting maintenance."""

    evaluated: int = 0
    active: int = 0
    fading: int = 0
    forgotten: int = 0
    reactivated: int = 0
    references_compacted: int = 0


@dataclass(frozen=True, slots=True)
class ReferenceCompactionResult:
    """Outcome of activation reference compaction."""

    references_compacted: int = 0


@dataclass(frozen=True, slots=True)
class WorkingMemoryConfig:
    """Configuration for bounded working-memory selection."""

    candidate_pool_size: int = 50
    max_items: int = 8
    max_prompt_tokens: int = 2_048
    activation_weight: float = 0.45
    goal_relevance_weight: float = 0.35
    importance_weight: float = 0.15
    carryover_weight: float = 0.05
    minimum_goal_relevance: float = 0.0
    minimum_selection_score: float = 0.0
    inhibition_strength: float = 0.30
    redundancy_threshold: float = 0.70
    decay_half_life_seconds: float = 300.0
    learned_utility_weight: float = 0.10
    collapse_same_slot_support: bool = True
    stale_goal_penalty: float = 0.35

    def __post_init__(self) -> None:
        if self.candidate_pool_size <= 0:
            raise ValidationError("candidate_pool_size must be greater than zero.")
        if self.max_items <= 0:
            raise ValidationError("max_items must be greater than zero.")
        if self.max_prompt_tokens <= 0:
            raise ValidationError("max_prompt_tokens must be greater than zero.")
        ranking_weights = (
            self.activation_weight,
            self.goal_relevance_weight,
            self.importance_weight,
            self.carryover_weight,
        )
        for label, weight in (
            ("activation_weight", self.activation_weight),
            ("goal_relevance_weight", self.goal_relevance_weight),
            ("importance_weight", self.importance_weight),
            ("carryover_weight", self.carryover_weight),
        ):
            if not math.isfinite(weight) or weight < 0:
                raise ValidationError(f"{label} must be finite and non-negative.")
        if sum(weight for weight in ranking_weights if weight > 0) <= 0:
            raise ValidationError("At least one ranking weight must be positive.")
        if not 0.0 <= self.minimum_goal_relevance <= 1.0:
            raise ValidationError("minimum_goal_relevance must be between 0.0 and 1.0.")
        if not 0.0 <= self.minimum_selection_score <= 1.0:
            raise ValidationError("minimum_selection_score must be between 0.0 and 1.0.")
        if not 0.0 <= self.inhibition_strength <= 1.0:
            raise ValidationError("inhibition_strength must be between 0.0 and 1.0.")
        if not 0.0 <= self.redundancy_threshold <= 1.0:
            raise ValidationError("redundancy_threshold must be between 0.0 and 1.0.")
        if self.decay_half_life_seconds <= 0:
            raise ValidationError("decay_half_life_seconds must be greater than zero.")
        if not 0.0 <= self.learned_utility_weight <= 1.0:
            raise ValidationError("learned_utility_weight must be between 0.0 and 1.0.")
        if not math.isfinite(self.stale_goal_penalty) or not 0.0 <= self.stale_goal_penalty <= 1.0:
            raise ValidationError("stale_goal_penalty must be finite and between 0.0 and 1.0.")


@dataclass(frozen=True, slots=True)
class WorkingMemoryComponents:
    """Decomposed working-memory selection scores."""

    activation: float
    goal_relevance: float
    importance: float
    carryover: float
    base_priority: float
    learned_utility: float
    utility_adjustment: float
    adjusted_priority: float
    inhibition: float
    final_score: float

    def __post_init__(self) -> None:
        for label, value in (
            ("activation", self.activation),
            ("goal_relevance", self.goal_relevance),
            ("importance", self.importance),
            ("carryover", self.carryover),
            ("base_priority", self.base_priority),
            ("learned_utility", self.learned_utility),
            ("utility_adjustment", self.utility_adjustment),
            ("adjusted_priority", self.adjusted_priority),
            ("inhibition", self.inhibition),
            ("final_score", self.final_score),
        ):
            if not math.isfinite(value):
                raise ValidationError(f"{label} must be finite.")
        for label, value in (
            ("activation", self.activation),
            ("goal_relevance", self.goal_relevance),
            ("importance", self.importance),
            ("carryover", self.carryover),
            ("inhibition", self.inhibition),
            ("final_score", self.final_score),
            ("learned_utility", self.learned_utility),
            ("adjusted_priority", self.adjusted_priority),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValidationError(f"{label} must be between 0.0 and 1.0.")
        if not 0.0 <= self.base_priority <= 1.0:
            raise ValidationError("base_priority must be between 0.0 and 1.0.")
        if not -1.0 <= self.utility_adjustment <= 1.0:
            raise ValidationError("utility_adjustment must be between -1.0 and 1.0.")


@dataclass(frozen=True, slots=True)
class WorkingMemoryItem:
    """One memory selected into the current working-memory workspace."""

    recall: RecallResult
    estimated_tokens: int
    transient_strength: float
    components: WorkingMemoryComponents
    rank: int
    reason: str

    def __post_init__(self) -> None:
        if self.estimated_tokens < 0:
            raise ValidationError("estimated_tokens must not be negative.")
        if not math.isfinite(self.transient_strength):
            raise ValidationError("transient_strength must be finite.")
        if not 0.0 <= self.transient_strength <= 1.0:
            raise ValidationError("transient_strength must be between 0.0 and 1.0.")
        if self.rank <= 0:
            raise ValidationError("rank must be greater than zero.")

    @property
    def memory_kind(self) -> MemoryKind:
        return self.recall.memory_kind

    @property
    def memory(self) -> StoredEpisode | StoredSemanticMemory:
        return self.recall.memory

    @property
    def identity(self) -> MemoryIdentity:
        memory = self.recall.memory
        if isinstance(memory, StoredEpisode):
            return MemoryIdentity(
                memory_kind=MemoryKind.EPISODE,
                memory_key=memory.memory_key,
            )
        return MemoryIdentity(
            memory_kind=MemoryKind.SEMANTIC,
            memory_key=memory.memory_key,
        )

    @property
    def goal_relevance(self) -> float:
        return self.components.goal_relevance

    @property
    def selection_score(self) -> float:
        return self.components.final_score

    @property
    def inhibition_penalty(self) -> float:
        return self.components.inhibition


@dataclass(frozen=True, slots=True)
class WorkingMemorySnapshot:
    """Immutable working-memory workspace returned to the caller."""

    tenant_id: str
    subject_id: str | None
    goal: RetrievalCue
    items: tuple[WorkingMemoryItem, ...]
    created_at: datetime
    candidate_count: int
    selected_count: int
    estimated_prompt_tokens: int
    prompt_budget_tokens: int
    goal_filtered_count: int
    inhibited_count: int
    budget_skipped_count: int

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValidationError("tenant_id must not be empty.")
        if self.created_at.tzinfo is None:
            raise ValidationError("created_at must be timezone-aware.")
        object.__setattr__(self, "created_at", self.created_at.astimezone(UTC))
        if self.candidate_count < 0:
            raise ValidationError("candidate_count must not be negative.")
        if self.selected_count < 0:
            raise ValidationError("selected_count must not be negative.")
        if self.estimated_prompt_tokens < 0:
            raise ValidationError("estimated_prompt_tokens must not be negative.")
        if self.prompt_budget_tokens <= 0:
            raise ValidationError("prompt_budget_tokens must be greater than zero.")
        if self.goal_filtered_count < 0:
            raise ValidationError("goal_filtered_count must not be negative.")
        if self.inhibited_count < 0:
            raise ValidationError("inhibited_count must not be negative.")
        if self.budget_skipped_count < 0:
            raise ValidationError("budget_skipped_count must not be negative.")

    @property
    def recall_results(self) -> tuple[RecallResult, ...]:
        return tuple(item.recall for item in self.items)


class LearningOutcome(StrEnum):
    """Outcome label for memory-use feedback."""

    HELPFUL = "helpful"
    UNHELPFUL = "unhelpful"
    INCORRECT = "incorrect"


@dataclass(frozen=True, slots=True)
class MemoryFeedback:
    """Feedback for one memory identity within a learning event."""

    identity: MemoryIdentity
    outcome: LearningOutcome
    revision_key: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class LearningFeedback:
    """Application-supplied learning feedback for one task or evaluation."""

    tenant_id: str
    feedback_id: str
    items: tuple[MemoryFeedback, ...]
    occurred_at: datetime
    subject_id: str | None = None
    goal: RetrievalCue | None = None
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValidationError("tenant_id must not be empty.")
        if not self.feedback_id.strip():
            raise ValidationError("feedback_id must not be empty.")
        if not self.items:
            raise ValidationError("items must not be empty.")
        if self.occurred_at.tzinfo is None:
            raise ValidationError("occurred_at must be timezone-aware.")
        object.__setattr__(self, "occurred_at", self.occurred_at.astimezone(UTC))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        seen: set[MemoryIdentity] = set()
        for item in self.items:
            if item.identity in seen:
                raise ValidationError("items must not contain duplicate memory identities.")
            seen.add(item.identity)


@dataclass(frozen=True, slots=True)
class LearningConfig:
    """Configuration for learning, utility, and association behaviour."""

    enabled: bool = True
    utility_prior_positive: float = 1.0
    utility_prior_negative: float = 1.0
    incorrect_utility_weight: float = 1.0
    association_tau: float = 3.0
    minimum_association_coactivations: int = 2
    max_feedback_items: int = 64
    max_association_items_per_feedback: int = 8

    def __post_init__(self) -> None:
        if self.utility_prior_positive <= 0:
            raise ValidationError("utility_prior_positive must be greater than zero.")
        if self.utility_prior_negative <= 0:
            raise ValidationError("utility_prior_negative must be greater than zero.")
        if self.incorrect_utility_weight < 0:
            raise ValidationError("incorrect_utility_weight must not be negative.")
        if self.association_tau <= 0:
            raise ValidationError("association_tau must be greater than zero.")
        if self.minimum_association_coactivations <= 0:
            raise ValidationError("minimum_association_coactivations must be greater than zero.")
        if self.max_feedback_items <= 0:
            raise ValidationError("max_feedback_items must be greater than zero.")
        if self.max_association_items_per_feedback < 2:
            raise ValidationError("max_association_items_per_feedback must be at least 2.")
        if self.max_association_items_per_feedback > self.max_feedback_items:
            raise ValidationError(
                "max_association_items_per_feedback must not exceed max_feedback_items."
            )


@dataclass(frozen=True, slots=True)
class StoredMemoryLearningState:
    """Persisted learning counts for one memory in one context."""

    tenant_id: str
    context_key: str
    memory_kind: MemoryKind
    memory_key: str
    helpful_count: int
    unhelpful_count: int
    incorrect_count: int
    first_feedback_at: datetime
    last_feedback_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValidationError("tenant_id must not be empty.")
        if not self.context_key.strip():
            raise ValidationError("context_key must not be empty.")
        if not self.memory_key.strip():
            raise ValidationError("memory_key must not be empty.")
        for label, count in (
            ("helpful_count", self.helpful_count),
            ("unhelpful_count", self.unhelpful_count),
            ("incorrect_count", self.incorrect_count),
        ):
            if count < 0:
                raise ValidationError(f"{label} must not be negative.")
        for label, timestamp in (
            ("first_feedback_at", self.first_feedback_at),
            ("last_feedback_at", self.last_feedback_at),
            ("updated_at", self.updated_at),
        ):
            if timestamp.tzinfo is None:
                raise ValidationError(f"{label} must be timezone-aware.")
        object.__setattr__(self, "first_feedback_at", self.first_feedback_at.astimezone(UTC))
        object.__setattr__(self, "last_feedback_at", self.last_feedback_at.astimezone(UTC))
        object.__setattr__(self, "updated_at", self.updated_at.astimezone(UTC))

    @property
    def identity(self) -> MemoryIdentity:
        return MemoryIdentity(memory_kind=self.memory_kind, memory_key=self.memory_key)


@dataclass(frozen=True, slots=True)
class StoredMemoryAssociation:
    """Persisted learned association between two memories."""

    tenant_id: str
    left: MemoryIdentity
    right: MemoryIdentity
    coactivation_count: int
    first_reinforced_at: datetime
    last_reinforced_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValidationError("tenant_id must not be empty.")
        if self.coactivation_count <= 0:
            raise ValidationError("coactivation_count must be greater than zero.")
        if self.left == self.right:
            raise ValidationError("left and right identities must differ.")
        for label, value in (
            ("first_reinforced_at", self.first_reinforced_at),
            ("last_reinforced_at", self.last_reinforced_at),
            ("updated_at", self.updated_at),
        ):
            if value.tzinfo is None:
                raise ValidationError(f"{label} must be timezone-aware.")
        object.__setattr__(self, "first_reinforced_at", self.first_reinforced_at.astimezone(UTC))
        object.__setattr__(self, "last_reinforced_at", self.last_reinforced_at.astimezone(UTC))
        object.__setattr__(self, "updated_at", self.updated_at.astimezone(UTC))


@dataclass(frozen=True, slots=True)
class LearnedAssociation:
    """Algorithm-facing learned association with derived strength."""

    left: MemoryIdentity
    right: MemoryIdentity
    strength: float
    coactivation_count: int

    def __post_init__(self) -> None:
        if self.left == self.right:
            raise ValidationError("left and right identities must differ.")
        if not 0.0 <= self.strength <= 1.0:
            raise ValidationError("strength must be between 0.0 and 1.0.")
        if self.coactivation_count <= 0:
            raise ValidationError("coactivation_count must be greater than zero.")


@dataclass(frozen=True, slots=True)
class LearningPlan:
    """Immutable plan produced by a learning processor."""

    feedback_id: str
    feedback_fingerprint: str
    tenant_id: str
    subject_id: str | None
    context_key: str
    occurred_at: datetime
    items: tuple[MemoryFeedback, ...]
    association_pairs: tuple[tuple[MemoryIdentity, MemoryIdentity], ...]
    association_items_skipped: int
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.feedback_id.strip():
            raise ValidationError("feedback_id must not be empty.")
        if not self.feedback_fingerprint.strip():
            raise ValidationError("feedback_fingerprint must not be empty.")
        if not self.tenant_id.strip():
            raise ValidationError("tenant_id must not be empty.")
        if not self.context_key.strip():
            raise ValidationError("context_key must not be empty.")
        if self.occurred_at.tzinfo is None:
            raise ValidationError("occurred_at must be timezone-aware.")
        if not self.items:
            raise ValidationError("items must not be empty.")
        if self.association_items_skipped < 0:
            raise ValidationError("association_items_skipped must not be negative.")
        object.__setattr__(self, "occurred_at", self.occurred_at.astimezone(UTC))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class LearningWriteResult:
    """Outcome of applying a learning plan to storage."""

    created: bool
    unchanged: bool
    helpful: int
    unhelpful: int
    incorrect: int
    associations_reinforced: int


@dataclass(frozen=True, slots=True)
class LearningResult:
    """Public outcome of Memory.learn()."""

    created: bool = False
    unchanged: bool = False
    helpful: int = 0
    unhelpful: int = 0
    incorrect: int = 0
    memories_reinforced: int = 0
    associations_reinforced: int = 0
    association_items_skipped: int = 0
    reactivated: int = 0


@dataclass(frozen=True, slots=True)
class MetamemoryConfig:
    """Configuration for read-only metamemory assessment."""

    enabled: bool = True
    candidate_pool_size: int = 50
    max_report_items: int = 8
    low_cue_coverage_threshold: float = 0.40
    low_retrieval_strength_threshold: float = 0.40
    low_evidence_confidence_threshold: float = 0.50
    semantic_conflict_threshold: float = 0.25
    provenance_diversity_tau: float = 3.0
    low_provenance_diversity_threshold: float = 0.30
    forgetting_pressure_threshold: float = 0.70
    low_learned_utility_threshold: float = 0.35
    freshness_half_life_seconds: float | None = None
    stale_evidence_threshold: float = 0.25
    missing_knowledge_coverage_threshold: float = 0.35
    missing_knowledge_strength_threshold: float = 0.45

    def __post_init__(self) -> None:
        if self.candidate_pool_size <= 0:
            raise ValidationError("candidate_pool_size must be greater than zero.")
        if self.max_report_items <= 0:
            raise ValidationError("max_report_items must be greater than zero.")
        if self.max_report_items > self.candidate_pool_size:
            raise ValidationError("max_report_items must not exceed candidate_pool_size.")
        for label, threshold in (
            ("low_cue_coverage_threshold", self.low_cue_coverage_threshold),
            ("low_retrieval_strength_threshold", self.low_retrieval_strength_threshold),
            ("low_evidence_confidence_threshold", self.low_evidence_confidence_threshold),
            ("semantic_conflict_threshold", self.semantic_conflict_threshold),
            ("low_provenance_diversity_threshold", self.low_provenance_diversity_threshold),
            ("forgetting_pressure_threshold", self.forgetting_pressure_threshold),
            ("low_learned_utility_threshold", self.low_learned_utility_threshold),
            ("stale_evidence_threshold", self.stale_evidence_threshold),
            ("missing_knowledge_coverage_threshold", self.missing_knowledge_coverage_threshold),
            ("missing_knowledge_strength_threshold", self.missing_knowledge_strength_threshold),
        ):
            if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                raise ValidationError(f"{label} must be finite and between 0.0 and 1.0.")
        if not math.isfinite(self.provenance_diversity_tau) or self.provenance_diversity_tau <= 0:
            raise ValidationError("provenance_diversity_tau must be finite and greater than zero.")
        if self.freshness_half_life_seconds is not None:
            if (
                not math.isfinite(self.freshness_half_life_seconds)
                or self.freshness_half_life_seconds <= 0
            ):
                raise ValidationError(
                    "freshness_half_life_seconds must be finite and greater than zero."
                )


class MemoryAssessmentFlag(StrEnum):
    """Deterministic diagnostic flags for memory assessment."""

    NO_RETRIEVED_MEMORY = "no_retrieved_memory"
    LOW_CUE_COVERAGE = "low_cue_coverage"
    LOW_RETRIEVAL_STRENGTH = "low_retrieval_strength"
    LOW_EVIDENCE_CONFIDENCE = "low_evidence_confidence"
    CONFLICTING_SEMANTIC_MEMORY = "conflicting_semantic_memory"
    LOW_PROVENANCE_DIVERSITY = "low_provenance_diversity"
    HIGH_FORGETTING_PRESSURE = "high_forgetting_pressure"
    LOW_LEARNED_UTILITY = "low_learned_utility"
    STALE_EVIDENCE = "stale_evidence"
    MISSING_KNOWLEDGE = "missing_knowledge"


@dataclass(frozen=True, slots=True)
class MetamemorySignals:
    """Aggregate metamemory signals for a retrieval assessment."""

    cue_coverage: float
    top_retrieval_strength: float
    mean_retrieval_strength: float
    evidence_confidence: float | None
    semantic_conflict: float
    provenance_diversity: float
    forgetting_pressure: float | None
    learned_utility: float | None
    freshness: float | None

    def __post_init__(self) -> None:
        for label, value in (
            ("cue_coverage", self.cue_coverage),
            ("top_retrieval_strength", self.top_retrieval_strength),
            ("mean_retrieval_strength", self.mean_retrieval_strength),
            ("semantic_conflict", self.semantic_conflict),
            ("provenance_diversity", self.provenance_diversity),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValidationError(f"{label} must be finite and between 0.0 and 1.0.")
        for label, optional_value in (
            ("evidence_confidence", self.evidence_confidence),
            ("forgetting_pressure", self.forgetting_pressure),
            ("learned_utility", self.learned_utility),
            ("freshness", self.freshness),
        ):
            if optional_value is not None and (
                not math.isfinite(optional_value) or not 0.0 <= optional_value <= 1.0
            ):
                raise ValidationError(f"{label} must be None or finite between 0.0 and 1.0.")


@dataclass(frozen=True, slots=True)
class MetamemoryItem:
    """Per-memory metamemory diagnostics for one recalled candidate."""

    recall: RecallResult
    cue_relevance: float
    evidence_confidence: float
    semantic_conflict: float
    observation_count: int
    evidence_at: datetime
    evidence_age_seconds: float
    retention_score: float
    forgetting_pressure: float
    learned_utility: float | None

    def __post_init__(self) -> None:
        if not math.isfinite(self.cue_relevance) or not 0.0 <= self.cue_relevance <= 1.0:
            raise ValidationError("cue_relevance must be finite and between 0.0 and 1.0.")
        if (
            not math.isfinite(self.evidence_confidence)
            or not 0.0 <= self.evidence_confidence <= 1.0
        ):
            raise ValidationError("evidence_confidence must be finite and between 0.0 and 1.0.")
        if not math.isfinite(self.semantic_conflict) or not 0.0 <= self.semantic_conflict <= 1.0:
            raise ValidationError("semantic_conflict must be finite and between 0.0 and 1.0.")
        if self.observation_count < 0:
            raise ValidationError("observation_count must not be negative.")
        if self.evidence_at.tzinfo is None:
            raise ValidationError("evidence_at must be timezone-aware.")
        if not math.isfinite(self.evidence_age_seconds) or self.evidence_age_seconds < 0:
            raise ValidationError("evidence_age_seconds must be finite and non-negative.")
        if not math.isfinite(self.retention_score) or not 0.0 <= self.retention_score <= 1.0:
            raise ValidationError("retention_score must be finite and between 0.0 and 1.0.")
        if (
            not math.isfinite(self.forgetting_pressure)
            or not 0.0 <= self.forgetting_pressure <= 1.0
        ):
            raise ValidationError("forgetting_pressure must be finite and between 0.0 and 1.0.")
        if self.learned_utility is not None and (
            not math.isfinite(self.learned_utility) or not 0.0 <= self.learned_utility <= 1.0
        ):
            raise ValidationError("learned_utility must be None or finite between 0.0 and 1.0.")

    @property
    def identity(self) -> MemoryIdentity:
        memory = self.recall.memory
        if isinstance(memory, StoredEpisode):
            return MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key=memory.memory_key)
        return MemoryIdentity(memory_kind=MemoryKind.SEMANTIC, memory_key=memory.memory_key)

    @property
    def memory(self) -> StoredEpisode | StoredSemanticMemory:
        return self.recall.memory


@dataclass(frozen=True, slots=True)
class MemoryAssessment:
    """Immutable read-only assessment of retrieved memory state."""

    tenant_id: str
    subject_id: str | None
    query: RetrievalCue
    goal: RetrievalCue
    assessed_at: datetime
    valid_at: datetime | None
    signals: MetamemorySignals
    flags: tuple[MemoryAssessmentFlag, ...]
    items: tuple[MetamemoryItem, ...]
    retrieved_count: int
    episode_count: int
    semantic_count: int
    contested_count: int
    historical_revision_count: int
    distinct_observation_count: int
    helpful_feedback_count: int
    unhelpful_feedback_count: int
    incorrect_feedback_count: int
    newest_evidence_at: datetime | None
    oldest_evidence_at: datetime | None

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValidationError("tenant_id must not be empty.")
        if self.assessed_at.tzinfo is None:
            raise ValidationError("assessed_at must be timezone-aware.")
        if self.valid_at is not None and self.valid_at.tzinfo is None:
            raise ValidationError("valid_at must be timezone-aware.")
        if self.retrieved_count < 0:
            raise ValidationError("retrieved_count must not be negative.")
        if self.episode_count < 0:
            raise ValidationError("episode_count must not be negative.")
        if self.semantic_count < 0:
            raise ValidationError("semantic_count must not be negative.")
        if self.contested_count < 0:
            raise ValidationError("contested_count must not be negative.")
        if self.historical_revision_count < 0:
            raise ValidationError("historical_revision_count must not be negative.")
        if self.distinct_observation_count < 0:
            raise ValidationError("distinct_observation_count must not be negative.")
        for label, count in (
            ("helpful_feedback_count", self.helpful_feedback_count),
            ("unhelpful_feedback_count", self.unhelpful_feedback_count),
            ("incorrect_feedback_count", self.incorrect_feedback_count),
        ):
            if count < 0:
                raise ValidationError(f"{label} must not be negative.")


@dataclass(frozen=True, slots=True)
class MemoryProcessingResult:
    """Outcome of explicit observation-to-memory processing."""

    tenant_id: str
    subject_id: str | None
    processed_at: datetime
    episodes: EpisodeEncodingResult
    semantics: SemanticConsolidationResult

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValidationError("tenant_id must not be empty.")
        if self.processed_at.tzinfo is None:
            raise ValidationError("processed_at must be timezone-aware.")
        object.__setattr__(self, "processed_at", self.processed_at.astimezone(UTC))


@dataclass(frozen=True, slots=True)
class MemoryContext:
    """Provider-neutral memory context prepared for external reasoning."""

    tenant_id: str
    subject_id: str | None
    query: RetrievalCue
    goal: RetrievalCue
    prepared_at: datetime
    valid_at: datetime | None
    working_memory: WorkingMemorySnapshot
    assessment: MemoryAssessment

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValidationError("tenant_id must not be empty.")
        if self.prepared_at.tzinfo is None:
            raise ValidationError("prepared_at must be timezone-aware.")
        object.__setattr__(self, "prepared_at", self.prepared_at.astimezone(UTC))
        if self.valid_at is not None and self.valid_at.tzinfo is None:
            raise ValidationError("valid_at must be timezone-aware.")
        if self.working_memory.tenant_id != self.tenant_id:
            raise ValidationError("working_memory.tenant_id must match tenant_id.")
        if self.assessment.tenant_id != self.tenant_id:
            raise ValidationError("assessment.tenant_id must match tenant_id.")
        if self.working_memory.subject_id != self.subject_id:
            raise ValidationError("working_memory.subject_id must match subject_id.")
        if self.assessment.subject_id != self.subject_id:
            raise ValidationError("assessment.subject_id must match subject_id.")
        if self.assessment.valid_at != self.valid_at:
            raise ValidationError("assessment.valid_at must match valid_at.")
        if self.working_memory.goal != self.goal:
            raise ValidationError("working_memory.goal must match goal.")
        if self.assessment.query != self.query:
            raise ValidationError("assessment.query must match query.")
        if self.assessment.goal != self.goal:
            raise ValidationError("assessment.goal must match goal.")
        if self.working_memory.created_at != self.prepared_at:
            raise ValidationError("working_memory.created_at must match prepared_at.")
        if self.assessment.assessed_at != self.prepared_at:
            raise ValidationError("assessment.assessed_at must match prepared_at.")

    @property
    def items(self) -> tuple[WorkingMemoryItem, ...]:
        return self.working_memory.items

    @property
    def recall_results(self) -> tuple[RecallResult, ...]:
        return self.working_memory.recall_results

    @property
    def estimated_tokens(self) -> int:
        return self.working_memory.estimated_prompt_tokens

    @property
    def prompt_budget_tokens(self) -> int:
        return self.working_memory.prompt_budget_tokens

    def render(self) -> str:
        """Return a deterministic plain-text rendering of selected working-memory items."""
        if not self.items:
            return ""
        lines = ["Relevant memory:", ""]
        for item in self.items:
            statement = item.memory.statement.replace("\n", "\n  ")
            lines.append(f"- {statement}")
        return "\n".join(lines)
