"""Deterministic semantic reconsolidation and temporal reconciliation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol

from cogkura.algorithms.semantic import _calculate_confidence
from cogkura.exceptions import ValidationError
from cogkura.models import (
    SemanticCardinality,
    SemanticDerivationInput,
    SemanticMemoryInput,
    SemanticMemoryStatus,
    SemanticReconciliationPlan,
    SemanticRevisionCandidate,
    SemanticRevisionInput,
    SemanticRevisionRelation,
    SemanticUpdateRelation,
    StoredSemanticMemory,
    StoredSemanticRevision,
)

_RECONCILIATION_VERSION = "reconsolidation-v1"


class _TemporalRelation(StrEnum):
    BEFORE = "before"
    AFTER = "after"
    OVERLAPS = "overlaps"
    UNKNOWN = "unknown"


class SemanticReconciler(Protocol):
    """Protocol for semantic revision reconciliation."""

    def reconcile(
        self,
        *,
        candidates: Sequence[SemanticRevisionCandidate],
        existing_memories: Sequence[StoredSemanticMemory],
        existing_revisions: Sequence[StoredSemanticRevision],
        as_of: datetime,
    ) -> SemanticReconciliationPlan:
        """Reconcile revision candidates against stored semantic history."""
        ...


class DeterministicSemanticReconciler:
    """Deterministic temporal semantic reconciliation."""

    def __init__(
        self,
        *,
        recurrence_tau: float = 1.5,
        source_diversity_target: int = 2,
        contested_ratio_threshold: float = 0.25,
    ) -> None:
        self._recurrence_tau = recurrence_tau
        self._source_diversity_target = source_diversity_target
        self._contested_ratio_threshold = contested_ratio_threshold

    def reconcile(
        self,
        *,
        candidates: Sequence[SemanticRevisionCandidate],
        existing_memories: Sequence[StoredSemanticMemory],
        existing_revisions: Sequence[StoredSemanticRevision],
        as_of: datetime,
    ) -> SemanticReconciliationPlan:
        if as_of.tzinfo is None:
            raise ValidationError("as_of must be timezone-aware.")
        evaluation_time = as_of.astimezone(UTC)

        builder = _ReconciliationBuilder(
            existing_memories=existing_memories,
            existing_revisions=existing_revisions,
            recurrence_tau=self._recurrence_tau,
            source_diversity_target=self._source_diversity_target,
            contested_ratio_threshold=self._contested_ratio_threshold,
            as_of=evaluation_time,
        )
        for candidate in _sorted_candidates(candidates):
            builder.apply_candidate(candidate)
        return builder.build_plan()


def compare_temporal_validity(
    left_from: datetime | None,
    left_until: datetime | None,
    right_from: datetime | None,
    right_until: datetime | None,
) -> _TemporalRelation:
    if left_from is None and left_until is None and right_from is None and right_until is None:
        return _TemporalRelation.UNKNOWN
    if _intervals_overlap(left_from, left_until, right_from, right_until):
        return _TemporalRelation.OVERLAPS
    left_end = left_until
    right_end = right_until
    if left_end is not None and right_from is not None and left_end <= right_from:
        return _TemporalRelation.BEFORE
    if right_end is not None and left_from is not None and right_end <= left_from:
        return _TemporalRelation.AFTER
    return _TemporalRelation.UNKNOWN


def classify_update_relation(
    *,
    existing: SemanticRevisionInput | SemanticRevisionCandidate | StoredSemanticRevision,
    incoming: SemanticRevisionCandidate,
) -> SemanticUpdateRelation:
    existing_slot = _slot_key_from_existing(existing, incoming.slot_key)
    if existing.memory_key != incoming.memory_key:
        if existing_slot != incoming.slot_key:
            return SemanticUpdateRelation.COEXISTS
        if incoming.cardinality is SemanticCardinality.MANY:
            return SemanticUpdateRelation.COEXISTS
        if _cardinality_mismatch(existing, incoming):
            return SemanticUpdateRelation.CONFLICTS
    else:
        if _same_proposition(existing, incoming):
            if _validity_compatible(existing, incoming):
                return SemanticUpdateRelation.REINFORCES
        if incoming.cardinality is SemanticCardinality.MANY:
            return SemanticUpdateRelation.COEXISTS

    temporal = compare_temporal_validity(
        existing.valid_from,
        existing.valid_until,
        incoming.valid_from,
        incoming.valid_until,
    )
    if _cardinality_mismatch(existing, incoming):
        return SemanticUpdateRelation.CONFLICTS
    if _opposite_polarity(existing, incoming):
        if temporal is _TemporalRelation.OVERLAPS or temporal is _TemporalRelation.UNKNOWN:
            return SemanticUpdateRelation.CONFLICTS
        if temporal in (_TemporalRelation.BEFORE, _TemporalRelation.AFTER):
            return SemanticUpdateRelation.SUPERSEDES
    if existing_slot != incoming.slot_key:
        return SemanticUpdateRelation.COEXISTS
    if temporal is _TemporalRelation.OVERLAPS or temporal is _TemporalRelation.UNKNOWN:
        return SemanticUpdateRelation.CONFLICTS
    if temporal in (_TemporalRelation.BEFORE, _TemporalRelation.AFTER):
        return SemanticUpdateRelation.SUPERSEDES
    return SemanticUpdateRelation.CONFLICTS


def revision_valid_at(
    revision: SemanticRevisionInput | StoredSemanticRevision,
    valid_at: datetime,
) -> bool:
    timestamp = valid_at.astimezone(UTC)
    if revision.valid_from is not None and timestamp < revision.valid_from:
        return False
    if revision.valid_until is not None and timestamp >= revision.valid_until:
        return False
    return True


class _ReconciliationBuilder:
    def __init__(
        self,
        *,
        existing_memories: Sequence[StoredSemanticMemory],
        existing_revisions: Sequence[StoredSemanticRevision],
        recurrence_tau: float,
        source_diversity_target: int,
        contested_ratio_threshold: float,
        as_of: datetime,
    ) -> None:
        self._recurrence_tau = recurrence_tau
        self._source_diversity_target = source_diversity_target
        self._contested_ratio_threshold = contested_ratio_threshold
        self._as_of = as_of
        self._revisions: dict[str, SemanticRevisionInput] = {}
        self._relations: dict[tuple[str, str, str], SemanticRevisionRelation] = {}
        self._candidate_by_key: dict[str, SemanticRevisionCandidate] = {}
        self._memory_projection: dict[str, SemanticMemoryInput] = {}
        self._revision_numbers: dict[str, int] = {}
        self._reinforced = 0
        self._coexist = 0
        self._conflicts = 0
        self._superseded = 0
        self._revisions_created = 0
        self._revisions_updated = 0

        for memory in existing_memories:
            self._memory_projection[memory.memory_key] = _memory_input_from_stored(memory)
        for revision in existing_revisions:
            self._revisions[revision.revision_key] = _revision_input_from_stored(revision)
            self._revision_numbers[revision.memory_key] = max(
                self._revision_numbers.get(revision.memory_key, 0),
                revision.revision_number,
            )

    def apply_candidate(self, candidate: SemanticRevisionCandidate) -> None:
        self._candidate_by_key[candidate.revision_key] = candidate
        existing = self._revisions.get(candidate.revision_key)
        if existing is not None:
            merged = _merge_revision(existing, candidate)
            self._revisions[candidate.revision_key] = merged
            self._revisions_updated += 1
            self._reinforced += 1
            return

        competitors = self._current_competitors(candidate)
        if not competitors:
            self._create_revision(candidate, status=SemanticMemoryStatus.ACTIVE)
            return

        relations = [
            (competitor, classify_update_relation(existing=competitor, incoming=candidate))
            for competitor in competitors
        ]
        if all(relation is SemanticUpdateRelation.REINFORCES for _, relation in relations):
            merged = _merge_revision(competitors[0], candidate)
            self._revisions[competitors[0].revision_key] = merged
            self._revisions_updated += 1
            self._reinforced += 1
            return

        if all(
            relation in (SemanticUpdateRelation.COEXISTS, SemanticUpdateRelation.REINFORCES)
            for _, relation in relations
        ):
            self._create_revision(candidate, status=SemanticMemoryStatus.ACTIVE)
            self._coexist += 1
            return

        if any(relation is SemanticUpdateRelation.SUPERSEDES for _, relation in relations):
            self._apply_supersession(candidate, relations)
            return

        self._apply_conflict(candidate, relations)

    def build_plan(self) -> SemanticReconciliationPlan:
        self._recompute_confidence()
        self._choose_current_projections()
        return SemanticReconciliationPlan(
            current_memories=tuple(
                self._memory_projection[key] for key in sorted(self._memory_projection)
            ),
            revisions=tuple(self._revisions[key] for key in sorted(self._revisions)),
            relations=tuple(self._relations[key] for key in sorted(self._relations)),
            reinforced_count=self._reinforced,
            coexist_count=self._coexist,
            conflict_count=self._conflicts,
            superseded_count=self._superseded,
            revisions_created=self._revisions_created,
            revisions_updated=self._revisions_updated,
        )

    def _current_competitors(
        self,
        candidate: SemanticRevisionCandidate,
    ) -> list[SemanticRevisionInput]:
        competitors: list[SemanticRevisionInput] = []
        for revision in self._revisions.values():
            if revision.tenant_id != candidate.tenant_id:
                continue
            if revision.status is SemanticMemoryStatus.SUPERSEDED:
                continue
            if revision.memory_key == candidate.memory_key:
                continue
            if _slot_key_from_revision(revision) != candidate.slot_key:
                continue
            if candidate.cardinality is SemanticCardinality.ONE:
                competitors.append(revision)
        competitors.sort(key=lambda item: item.revision_key)
        return competitors

    def _create_revision(
        self,
        candidate: SemanticRevisionCandidate,
        *,
        status: SemanticMemoryStatus,
    ) -> SemanticRevisionInput:
        revision_number = self._revision_numbers.get(candidate.memory_key, 0) + 1
        self._revision_numbers[candidate.memory_key] = revision_number
        revision = SemanticRevisionInput(
            tenant_id=candidate.tenant_id,
            memory_key=candidate.memory_key,
            revision_key=candidate.revision_key,
            revision_number=revision_number,
            status=status,
            valid_from=candidate.valid_from,
            valid_until=candidate.valid_until,
            confidence=candidate.support_confidence,
            importance=candidate.importance,
            support_count=candidate.support_count,
            contradiction_count=0,
            first_supported_at=candidate.first_supported_at,
            last_supported_at=candidate.last_supported_at,
            derivations=candidate.derivations,
            metadata=MappingProxyType(
                {
                    **dict(candidate.metadata),
                    "semantic": {
                        **dict(candidate.metadata.get("semantic", {})),
                        "slot_key": candidate.slot_key,
                    },
                }
            ),
        )
        self._revisions[candidate.revision_key] = revision
        self._revisions_created += 1
        return revision

    def _apply_supersession(
        self,
        candidate: SemanticRevisionCandidate,
        relations: list[tuple[SemanticRevisionInput, SemanticUpdateRelation]],
    ) -> None:
        successor = self._create_revision(candidate, status=SemanticMemoryStatus.ACTIVE)
        for predecessor, relation in relations:
            if relation is not SemanticUpdateRelation.SUPERSEDES:
                continue
            updated_predecessor = replace(
                predecessor,
                status=SemanticMemoryStatus.SUPERSEDED,
                valid_until=successor.valid_from or predecessor.valid_until,
            )
            self._revisions[predecessor.revision_key] = updated_predecessor
            self._revisions_updated += 1
            self._superseded += 1
            left_key, right_key = _canonical_relation_keys(
                predecessor.revision_key,
                successor.revision_key,
            )
            self._relations[(predecessor.tenant_id, left_key, right_key)] = (
                SemanticRevisionRelation(
                    tenant_id=predecessor.tenant_id,
                    left_revision_key=left_key,
                    right_revision_key=right_key,
                    relation=SemanticUpdateRelation.SUPERSEDES,
                    effective_at=successor.valid_from,
                )
            )

    def _apply_conflict(
        self,
        candidate: SemanticRevisionCandidate,
        relations: list[tuple[SemanticRevisionInput, SemanticUpdateRelation]],
    ) -> None:
        revision = self._create_revision(candidate, status=SemanticMemoryStatus.CONTESTED)
        self._conflicts += 1
        for competitor, _ in relations:
            contested = replace(competitor, status=SemanticMemoryStatus.CONTESTED)
            self._revisions[competitor.revision_key] = contested
            self._revisions_updated += 1
            left_key, right_key = _canonical_relation_keys(
                competitor.revision_key,
                revision.revision_key,
            )
            self._relations[(candidate.tenant_id, left_key, right_key)] = SemanticRevisionRelation(
                tenant_id=candidate.tenant_id,
                left_revision_key=left_key,
                right_revision_key=right_key,
                relation=SemanticUpdateRelation.CONFLICTS,
                effective_at=None,
            )

    def _recompute_confidence(self) -> None:
        conflict_partners = _conflict_partner_map(self._relations)
        updated: dict[str, SemanticRevisionInput] = {}
        for revision_key, revision in self._revisions.items():
            candidate = self._candidate_by_key.get(revision_key)
            support_mass = float(revision.support_count)
            if candidate is not None:
                support_mass = max(
                    support_mass,
                    float(candidate.metadata.get("semantic", {}).get("support_mass", support_mass)),
                )
            contradiction_mass = float(len(conflict_partners.get(revision_key, ())))
            confidence = _calculate_confidence(
                support_mass=support_mass,
                contradiction_mass=contradiction_mass,
                source_namespaces=frozenset(
                    candidate.metadata.get("semantic", {}).get("source_namespaces", ())
                    if candidate is not None
                    else ()
                ),
                recurrence_tau=self._recurrence_tau,
                source_diversity_target=self._source_diversity_target,
            )
            status = revision.status
            total = support_mass + contradiction_mass
            ratio = contradiction_mass / total if total > 0 else 0.0
            if (
                ratio >= self._contested_ratio_threshold
                and status is not SemanticMemoryStatus.SUPERSEDED
            ):
                status = SemanticMemoryStatus.CONTESTED
            updated[revision_key] = replace(
                revision,
                confidence=confidence,
                contradiction_count=int(contradiction_mass),
                status=status,
            )
        self._revisions = updated

    def _choose_current_projections(self) -> None:
        by_memory_key: dict[str, list[SemanticRevisionInput]] = {}
        for revision in self._revisions.values():
            by_memory_key.setdefault(revision.memory_key, []).append(revision)
        for memory_key, revisions in by_memory_key.items():
            current = _select_current_revision(revisions)
            if current is None:
                continue
            candidate = self._candidate_by_key.get(current.revision_key)
            if candidate is None:
                existing = self._memory_projection.get(memory_key)
                if existing is None:
                    continue
                self._memory_projection[memory_key] = replace(
                    existing,
                    revision_key=current.revision_key,
                    revision_number=current.revision_number,
                    confidence=current.confidence,
                    importance=current.importance,
                    status=current.status,
                    support_count=current.support_count,
                    contradiction_count=current.contradiction_count,
                    first_supported_at=current.first_supported_at,
                    last_supported_at=current.last_supported_at,
                    valid_from=current.valid_from,
                    valid_until=current.valid_until,
                )
                continue
            self._memory_projection[memory_key] = _memory_input_from_candidate(
                candidate,
                revision=current,
            )


def _sorted_candidates(
    candidates: Sequence[SemanticRevisionCandidate],
) -> list[SemanticRevisionCandidate]:
    return sorted(
        candidates,
        key=lambda item: (
            item.slot_key,
            item.valid_from.isoformat() if item.valid_from else "",
            item.memory_key,
            item.revision_key,
        ),
    )


def _intervals_overlap(
    left_from: datetime | None,
    left_until: datetime | None,
    right_from: datetime | None,
    right_until: datetime | None,
) -> bool:
    if left_from is None and left_until is None:
        return True
    if right_from is None and right_until is None:
        return True
    start_left = left_from
    end_left = left_until
    start_right = right_from
    end_right = right_until
    if end_left is not None and start_right is not None and end_left <= start_right:
        return False
    if end_right is not None and start_left is not None and end_right <= start_left:
        return False
    return True


def _same_proposition(
    left: SemanticRevisionInput | SemanticRevisionCandidate | StoredSemanticRevision,
    right: SemanticRevisionCandidate,
) -> bool:
    return left.memory_key == right.memory_key


def _validity_compatible(
    left: SemanticRevisionInput | SemanticRevisionCandidate | StoredSemanticRevision,
    right: SemanticRevisionCandidate,
) -> bool:
    if left.valid_from == right.valid_from and left.valid_until == right.valid_until:
        return True
    temporal = compare_temporal_validity(
        left.valid_from,
        left.valid_until,
        right.valid_from,
        right.valid_until,
    )
    return temporal is _TemporalRelation.OVERLAPS


def _cardinality_mismatch(
    left: SemanticRevisionInput | SemanticRevisionCandidate | StoredSemanticRevision,
    right: SemanticRevisionCandidate,
) -> bool:
    left_cardinality = getattr(left, "cardinality", None)
    if left_cardinality is None:
        return False
    return bool(left_cardinality != right.cardinality)


def _opposite_polarity(
    left: SemanticRevisionInput | SemanticRevisionCandidate | StoredSemanticRevision,
    right: SemanticRevisionCandidate,
) -> bool:
    left_polarity = getattr(left, "polarity", None)
    if left_polarity is None:
        return False
    left_object = getattr(left, "object_entity_id", None) or getattr(left, "object_value", None)
    right_object = right.object_entity_id or right.object_value
    return left_object == right_object and left_polarity != right.polarity


def _slot_key_from_existing(
    existing: SemanticRevisionInput | SemanticRevisionCandidate | StoredSemanticRevision,
    default: str,
) -> str:
    slot_key = getattr(existing, "slot_key", None)
    if isinstance(slot_key, str) and slot_key:
        return slot_key
    return (
        _slot_key_from_revision(existing)
        if isinstance(existing, SemanticRevisionInput)
        else default
    )


def _slot_key_from_revision(
    revision: SemanticRevisionInput | SemanticRevisionCandidate | StoredSemanticRevision,
) -> str:
    metadata = getattr(revision, "metadata", None)
    if isinstance(metadata, Mapping):
        slot_key = metadata.get("semantic", {}).get("slot_key")
        if isinstance(slot_key, str) and slot_key:
            return slot_key
    return revision.memory_key


def _canonical_relation_keys(left: str, right: str) -> tuple[str, str]:
    if left <= right:
        return left, right
    return right, left


def _conflict_partner_map(
    relations: Mapping[tuple[str, str, str], SemanticRevisionRelation],
) -> dict[str, tuple[str, ...]]:
    partners: dict[str, list[str]] = {}
    for relation in relations.values():
        if relation.relation is not SemanticUpdateRelation.CONFLICTS:
            continue
        partners.setdefault(relation.left_revision_key, []).append(relation.right_revision_key)
        partners.setdefault(relation.right_revision_key, []).append(relation.left_revision_key)
    return {key: tuple(sorted(set(values))) for key, values in partners.items()}


def _merge_revision(
    existing: SemanticRevisionInput,
    candidate: SemanticRevisionCandidate,
) -> SemanticRevisionInput:
    merged_derivations = _merge_derivations(existing.derivations, candidate.derivations)
    return replace(
        existing,
        support_count=max(existing.support_count, candidate.support_count),
        first_supported_at=min(existing.first_supported_at, candidate.first_supported_at),
        last_supported_at=max(existing.last_supported_at, candidate.last_supported_at),
        derivations=merged_derivations,
        importance=candidate.importance,
    )


def _merge_derivations(
    left: tuple[SemanticDerivationInput, ...],
    right: tuple[SemanticDerivationInput, ...],
) -> tuple[SemanticDerivationInput, ...]:
    merged = {(item.episode_id, item.relation): item for item in left}
    for item in right:
        key = (item.episode_id, item.relation)
        current = merged.get(key)
        if current is None or item.contribution_score > current.contribution_score:
            merged[key] = item
    return tuple(sorted(merged.values(), key=lambda item: (item.episode_id, item.relation.value)))


def _select_current_revision(
    revisions: Sequence[SemanticRevisionInput],
) -> SemanticRevisionInput | None:
    active = [revision for revision in revisions if revision.status is SemanticMemoryStatus.ACTIVE]
    if len(active) == 1:
        return active[0]
    if active:
        return sorted(active, key=lambda item: item.revision_number)[-1]
    contested = [
        revision for revision in revisions if revision.status is SemanticMemoryStatus.CONTESTED
    ]
    if contested:
        return sorted(contested, key=lambda item: item.revision_number)[-1]
    superseded = [
        revision for revision in revisions if revision.status is SemanticMemoryStatus.SUPERSEDED
    ]
    if superseded:
        return sorted(superseded, key=lambda item: item.revision_number)[-1]
    return None


def _revision_input_from_stored(revision: StoredSemanticRevision) -> SemanticRevisionInput:
    return SemanticRevisionInput(
        tenant_id=revision.tenant_id,
        memory_key=revision.memory_key,
        revision_key=revision.revision_key,
        revision_number=revision.revision_number,
        status=revision.status,
        valid_from=revision.valid_from,
        valid_until=revision.valid_until,
        confidence=revision.confidence,
        importance=revision.importance,
        support_count=revision.support_count,
        contradiction_count=revision.contradiction_count,
        first_supported_at=revision.first_supported_at,
        last_supported_at=revision.last_supported_at,
        derivations=revision.derivations,
        metadata=MappingProxyType({}),
    )


def _memory_input_from_stored(memory: StoredSemanticMemory) -> SemanticMemoryInput:
    return SemanticMemoryInput(
        tenant_id=memory.tenant_id,
        subject_id=memory.subject_id,
        memory_key=memory.memory_key,
        slot_key=memory.slot_key,
        revision_key=memory.revision_key,
        revision_number=memory.revision_number,
        statement=memory.statement,
        subject_entity_id=memory.subject_entity_id,
        predicate=memory.predicate,
        object_value=memory.object_value,
        object_entity_id=memory.object_entity_id,
        polarity=memory.polarity,
        cardinality=memory.cardinality,
        qualifiers=memory.qualifiers,
        confidence=memory.confidence,
        importance=memory.importance,
        status=memory.status,
        support_count=memory.support_count,
        contradiction_count=memory.contradiction_count,
        first_supported_at=memory.first_supported_at,
        last_supported_at=memory.last_supported_at,
        valid_from=memory.valid_from,
        valid_until=memory.valid_until,
        derivations=memory.derivations,
        observation_evidence=memory.observation_evidence,
        entities=memory.entities,
        metadata=memory.metadata,
    )


def _memory_input_from_candidate(
    candidate: SemanticRevisionCandidate,
    *,
    revision: SemanticRevisionInput,
) -> SemanticMemoryInput:
    return SemanticMemoryInput(
        tenant_id=candidate.tenant_id,
        subject_id=candidate.subject_id,
        memory_key=candidate.memory_key,
        slot_key=candidate.slot_key,
        revision_key=revision.revision_key,
        revision_number=revision.revision_number,
        statement=candidate.statement,
        subject_entity_id=candidate.subject_entity_id,
        predicate=candidate.predicate,
        object_value=candidate.object_value,
        object_entity_id=candidate.object_entity_id,
        polarity=candidate.polarity,
        cardinality=candidate.cardinality,
        qualifiers=candidate.qualifiers,
        confidence=revision.confidence,
        importance=revision.importance,
        status=revision.status,
        support_count=revision.support_count,
        contradiction_count=revision.contradiction_count,
        first_supported_at=revision.first_supported_at,
        last_supported_at=revision.last_supported_at,
        valid_from=revision.valid_from,
        valid_until=revision.valid_until,
        derivations=revision.derivations,
        observation_evidence=candidate.observation_evidence,
        entities=candidate.entities,
        metadata=candidate.metadata,
    )
