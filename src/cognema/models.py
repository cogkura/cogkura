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


class MemoryKind(StrEnum):
    """Kind of durable memory returned by declarative recall."""

    EPISODE = "episode"
    SEMANTIC = "semantic"


class ActivationReferenceKind(StrEnum):
    """How a memory access reference was produced."""

    RETRIEVED = "retrieved"
    REHEARSED = "rehearsed"


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
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValidationError("tenant_id must not be empty.")
        if not self.memory_key.strip():
            raise ValidationError("memory_key must not be empty.")
        if self.referenced_at.tzinfo is None:
            raise ValidationError("referenced_at must be timezone-aware.")
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
    enable_spreading_activation: bool = False
    enable_partial_matching: bool = True
    enable_noise: bool = False
    max_candidates: int = 10_000

    def __post_init__(self) -> None:
        if not 0.0 < self.decay <= 1.0:
            raise ValidationError("decay must be greater than zero and at most 1.0.")
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


@dataclass(frozen=True, slots=True)
class ActivationComponents:
    """Decomposed activation values for a recalled memory."""

    base_level: float
    spreading: float
    partial_match: float
    noise: float
    total: float

    def __post_init__(self) -> None:
        for label, value in (
            ("base_level", self.base_level),
            ("spreading", self.spreading),
            ("partial_match", self.partial_match),
            ("noise", self.noise),
            ("total", self.total),
        ):
            if not math.isfinite(value):
                raise ValidationError(f"{label} must be finite.")


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

    def __post_init__(self) -> None:
        if not self.memory_key.strip():
            raise ValidationError("memory_key must not be empty.")
        if not self.text.strip():
            raise ValidationError("text must not be empty.")
        if self.created_at.tzinfo is None:
            raise ValidationError("created_at must be timezone-aware.")
        object.__setattr__(self, "created_at", self.created_at.astimezone(UTC))
        object.__setattr__(self, "qualifiers", MappingProxyType(dict(self.qualifiers)))

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

    def __post_init__(self) -> None:
        if not math.isfinite(self.activation):
            raise ValidationError("activation must be finite.")
        if not math.isfinite(self.score):
            raise ValidationError("score must be finite.")
        if not 0.0 <= self.score <= 1.0:
            raise ValidationError("score must be between 0.0 and 1.0.")
        if not math.isfinite(self.latency_seconds) or self.latency_seconds < 0:
            raise ValidationError("latency_seconds must be finite and non-negative.")
