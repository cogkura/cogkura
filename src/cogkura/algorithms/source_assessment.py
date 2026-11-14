"""Read-only source coverage and conservative semantic inventory assessment."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime

from cogkura.algorithms.corrections import (
    _boundary_effective_at,
    _knowledge_at,
    _predecessor_key,
    memory_from_revision_snapshot,
)
from cogkura.algorithms.reconsolidation import revision_valid_at
from cogkura.models import (
    ContextProvenance,
    EvidenceQuality,
    MemoryContext,
    MemoryIdentity,
    MemoryKind,
    SemanticCardinality,
    SemanticKnowledgeState,
    SemanticMemoryStatus,
    SemanticRevisionAssessment,
    SemanticRevisionViewState,
    SemanticSlotAssessment,
    SemanticSlotRecords,
    SemanticTransitionAssessment,
    SemanticUpdateRelation,
    SourceAssessmentReason,
    SourceConflictAssessment,
    SourceConflictState,
    SourceCurrentness,
    SourceEvidence,
    SourceEvidenceRelation,
    SourceEvidenceResolutionStatus,
    SourceInspectionLimits,
    SourceMemoryAssessment,
    SourceMemoryItem,
    StoredSemanticMemory,
    StoredSemanticRevision,
    semantic_promotion_from_metadata,
)
from cogkura.observations.models import ObservationRevisionIdentity
from cogkura.provenance import (
    _context_memories,
    _load_evidence_rows,
    _memory_evidence,
    _resolve_source_evidence,
)
from cogkura.storage.base import EpisodeStore, ObservationStore

_RESOLVED = {
    SourceEvidenceResolutionStatus.RESOLVED,
    SourceEvidenceResolutionStatus.SOURCE_REF_MISSING,
}
_SEMANTIC_RELATIONS = {
    SourceEvidenceRelation.SEMANTIC_SUPPORT,
    SourceEvidenceRelation.SEMANTIC_CONTRADICTION,
}


def evidence_quality(
    sources: Sequence[SourceEvidence],
    *,
    referenced_count: int,
    semantic_references: int,
) -> EvidenceQuality:
    resolved = [s for s in sources if s.resolution_status in _RESOLVED]
    return EvidenceQuality(
        referenced_count=referenced_count,
        returned_count=len(sources),
        revision_resolved_count=len(resolved),
        locator_present_count=sum(s.source_ref is not None for s in resolved),
        relation_known_count=sum(s.relation in _SEMANTIC_RELATIONS for s in sources),
        relation_applicable_count=semantic_references,
        relation_not_applicable_count=referenced_count - semantic_references,
        distinct_source_count=len({(s.source_namespace, s.source_record_id) for s in resolved}),
    )


def source_currentness(sources: Sequence[SourceEvidence]) -> SourceCurrentness:
    return SourceCurrentness(
        latest_count=sum(s.is_latest_known_revision is True for s in sources),
        older_revision_count=sum(s.is_latest_known_revision is False for s in sources),
        unknown_count=sum(s.is_latest_known_revision is None for s in sources),
    )


def quality_reasons(
    sources: Sequence[SourceEvidence], quality: EvidenceQuality
) -> set[SourceAssessmentReason]:
    reasons: set[SourceAssessmentReason] = set()
    if not quality.has_evidence:
        reasons.add(SourceAssessmentReason.NO_EVIDENCE)
    for source in sources:
        if source.resolution_status is SourceEvidenceResolutionStatus.SOURCE_UNAVAILABLE:
            reasons.add(SourceAssessmentReason.SOURCE_UNAVAILABLE)
        if source.resolution_status is SourceEvidenceResolutionStatus.REVISION_UNAVAILABLE:
            reasons.add(SourceAssessmentReason.REVISION_UNAVAILABLE)
        if source.resolution_status is SourceEvidenceResolutionStatus.SOURCE_REF_MISSING:
            reasons.add(SourceAssessmentReason.LOCATOR_MISSING)
        if source.is_latest_known_revision is False:
            reasons.add(SourceAssessmentReason.SOURCE_REVISION_SUPERSEDED)
        if source.is_latest_known_revision is None:
            reasons.add(SourceAssessmentReason.SOURCE_CURRENTNESS_UNKNOWN)
    if quality.is_truncated:
        reasons.add(SourceAssessmentReason.EVIDENCE_TRUNCATED)
    if quality.relation_coverage_complete is False:
        reasons.add(SourceAssessmentReason.RELATION_UNAVAILABLE)
    return reasons


def ordered_reasons(reasons: set[SourceAssessmentReason]) -> tuple[SourceAssessmentReason, ...]:
    return tuple(r for r in SourceAssessmentReason if r in reasons)


def assess_context_sources(
    context: MemoryContext, provenance: ContextProvenance
) -> SourceMemoryAssessment:
    memories = {m.identity: m.memory for m in _context_memories(context)}
    items: list[SourceMemoryItem] = []
    all_sources: list[SourceEvidence] = []
    references = semantic_references = 0
    for item in provenance.items:
        memory = memories[item.memory_identity]
        semantic = item.memory_identity.memory_kind is MemoryKind.SEMANTIC
        count = item.referenced_evidence_count
        quality = evidence_quality(
            item.sources, referenced_count=count, semantic_references=count if semantic else 0
        )
        items.append(
            SourceMemoryItem(
                memory_identity=item.memory_identity,
                representation=item.representation,
                revision_key=memory.revision_key
                if isinstance(memory, StoredSemanticMemory)
                else None,
                quality=quality,
                currentness=source_currentness(item.sources),
                promotion=item.promotion,
            )
        )
        references += count
        semantic_references += count if semantic else 0
        all_sources.extend(item.sources)
    quality = evidence_quality(
        all_sources, referenced_count=references, semantic_references=semantic_references
    )
    return SourceMemoryAssessment(
        tenant_id=context.tenant_id,
        subject_id=context.subject_id,
        assessed_at=context.prepared_at,
        valid_at=context.valid_at,
        provenance=provenance,
        items=tuple(items),
        quality=quality,
        reasons=ordered_reasons(quality_reasons(all_sources, quality)),
    )


@dataclass(frozen=True, slots=True)
class _ProjectedRevision:
    revision: StoredSemanticRevision
    memory: StoredSemanticMemory
    state: SemanticRevisionViewState
    history_complete: bool
    transitions: tuple[SemanticTransitionAssessment, ...]


def _project_revisions(
    records: SemanticSlotRecords,
    *,
    as_of: datetime,
    valid_at: datetime | None,
) -> tuple[list[_ProjectedRevision], set[SourceAssessmentReason]]:
    boundary = valid_at or as_of
    reasons: set[SourceAssessmentReason] = set()
    bases = {m.memory_key: m for m in records.memories}
    by_key = {r.revision_key: r for r in records.revisions}
    unknown_successors: set[str] = set()
    reopened: set[str] = set()
    transitions: dict[str, list[SemanticTransitionAssessment]] = {}
    for revision in records.revisions:
        predecessor = _predecessor_key(revision.snapshot)
        if predecessor is None:
            continue
        knowledge = _knowledge_at(revision.snapshot)
        effective = _boundary_effective_at(revision.snapshot)
        if knowledge is None or effective is None:
            reasons.add(SourceAssessmentReason.TEMPORAL_HISTORY_UNAVAILABLE)
            unknown_successors.add(revision.revision_key)
            continue
        if knowledge > as_of:
            unknown_successors.add(revision.revision_key)
            reopened.add(predecessor)
            continue
        if effective > boundary:
            reopened.add(predecessor)
        if predecessor not in by_key:
            reasons.add(SourceAssessmentReason.TRANSITION_ATTRIBUTION_UNAVAILABLE)
            continue
        withdrawal = revision.snapshot.get("withdrawal")
        evidence = None
        if isinstance(withdrawal, dict):
            obs_id, obs_revision = (
                withdrawal.get("observation_id"),
                withdrawal.get("observation_revision"),
            )
            if isinstance(obs_id, str) and isinstance(obs_revision, int) and obs_revision > 0:
                evidence = ObservationRevisionIdentity(obs_id, obs_revision)
        else:
            promotion = semantic_promotion_from_metadata(revision.snapshot.get("metadata", {}))
            if (
                promotion is not None
                and promotion.observation_id is not None
                and promotion.observation_revision is not None
            ):
                evidence = ObservationRevisionIdentity(
                    promotion.observation_id, promotion.observation_revision
                )
        transition = SemanticTransitionAssessment(
            relation=SemanticUpdateRelation.RETRACTS
            if withdrawal
            else SemanticUpdateRelation.SUPERSEDES,
            predecessor_revision_key=predecessor,
            successor_revision_key=revision.revision_key,
            effective_at=effective,
            knowledge_at=knowledge,
            asserting_evidence=evidence,
        )
        transitions.setdefault(revision.revision_key, []).append(transition)
        transitions.setdefault(predecessor, []).append(transition)
    output: list[_ProjectedRevision] = []
    for revision in records.revisions:
        if revision.revision_key in unknown_successors:
            continue
        explicit = _predecessor_key(revision.snapshot) is not None
        if not explicit and revision.first_supported_at > as_of:
            continue
        base = bases.get(revision.memory_key)
        if base is None:
            reasons.add(SourceAssessmentReason.TEMPORAL_HISTORY_UNAVAILABLE)
            continue
        memory = memory_from_revision_snapshot(base, revision)
        complete = memory is not None
        if memory is None:
            reasons.add(SourceAssessmentReason.LEGACY_HISTORY_INCOMPLETE)
            memory = replace(
                base,
                revision_key=revision.revision_key,
                revision_number=revision.revision_number,
                status=revision.status,
                valid_from=revision.valid_from,
                valid_until=revision.valid_until,
                observation_evidence=(),
                derivations=(),
                metadata={},
            )
        # Revision windows/status are canonical; snapshots may predate reconciliation.
        memory = replace(
            memory,
            status=revision.status,
            valid_from=revision.valid_from,
            valid_until=revision.valid_until,
        )
        if revision.revision_key in reopened:
            memory = replace(memory, status=SemanticMemoryStatus.ACTIVE, valid_until=None)
        if revision.updated_at > as_of and not explicit and not revision.snapshot.get("closed_by"):
            reasons.add(SourceAssessmentReason.TEMPORAL_HISTORY_UNAVAILABLE)
            complete = False
        effective = _boundary_effective_at(revision.snapshot) if explicit else None
        if effective is not None and effective > boundary:
            state = SemanticRevisionViewState.PENDING_EFFECTIVE
            reasons.add(SourceAssessmentReason.PENDING_EFFECTIVE)
        elif revision_valid_at(
            replace(revision, valid_from=memory.valid_from, valid_until=memory.valid_until),
            boundary,
        ):
            if memory.status is SemanticMemoryStatus.RETRACTED:
                state = SemanticRevisionViewState.WITHDRAWN
            elif memory.status is SemanticMemoryStatus.SUPERSEDED:
                # A superseded assertion is still an answer in its historical interval.
                memory = replace(memory, status=SemanticMemoryStatus.ACTIVE)
                state = SemanticRevisionViewState.VISIBLE_CURRENT
            else:
                state = SemanticRevisionViewState.VISIBLE_CURRENT
        elif memory.valid_from is not None and memory.valid_from > boundary:
            # Ordinary facts with future validity are not past history.
            continue
        else:
            state = SemanticRevisionViewState.VISIBLE_HISTORICAL
        if not complete:
            state = SemanticRevisionViewState.INDETERMINATE
        output.append(
            _ProjectedRevision(
                revision, memory, state, complete, tuple(transitions.get(revision.revision_key, ()))
            )
        )
    return output, reasons


async def inspect_slot_sources(
    records: SemanticSlotRecords | None,
    *,
    tenant_id: str,
    subject_id: str | None,
    slot_key: str,
    as_of: datetime,
    valid_at: datetime | None,
    historical_currentness: bool,
    limits: SourceInspectionLimits,
    observation_store: ObservationStore,
    episode_store: EpisodeStore,
) -> SemanticSlotAssessment:
    reasons = {SourceAssessmentReason.DECISION_HISTORY_UNAVAILABLE}
    if records is None:
        reasons.add(SourceAssessmentReason.INSPECTION_CAPABILITY_UNAVAILABLE)
        projected: list[_ProjectedRevision] = []
        complete = False
    else:
        projected, projection_reasons = _project_revisions(records, as_of=as_of, valid_at=valid_at)
        reasons.update(projection_reasons)
        complete = records.is_complete
        if not complete:
            reasons.add(SourceAssessmentReason.INVENTORY_TRUNCATED)
    episode_ids = {d.episode_id for p in projected for d in p.memory.derivations}
    episode_evidence = (
        await episode_store.get_evidence_many(tenant_id=tenant_id, episode_ids=episode_ids)
        if episode_ids
        else {}
    )
    populations = [_memory_evidence(p.memory, episode_evidence) for p in projected]
    # Output evidence cap is global. Exact revision lookups are deduplicated.
    remaining = limits.max_evidence
    selected = []
    for population in populations:
        selected.append(population[:remaining])
        remaining -= len(selected[-1])
    references = [ref.identity for population in selected for ref in population]
    current, revisions = (
        await _load_evidence_rows(observation_store, tenant_id=tenant_id, references=references)
        if references
        else ({}, {})
    )
    items: list[SemanticRevisionAssessment] = []
    all_sources: list[SourceEvidence] = []
    referenced_count = 0
    for projected_item, population, returned in zip(projected, populations, selected, strict=True):
        sources: list[SourceEvidence] = []
        hidden_count = 0
        history_complete = projected_item.history_complete
        for reference in returned:
            revision = revisions.get(reference.identity)
            if revision is not None and revision.observed_at > as_of:
                hidden_count += 1
                history_complete = False
                reasons.add(SourceAssessmentReason.TEMPORAL_HISTORY_UNAVAILABLE)
                continue
            source = _resolve_source_evidence(
                reference, current=current.get(reference.identity.observation_id), revision=revision
            )
            if historical_currentness:
                source = replace(source, is_latest_known_revision=None)
            sources.append(source)
        count = len(population) - hidden_count
        quality = evidence_quality(sources, referenced_count=count, semantic_references=count)
        item_state = (
            projected_item.state if history_complete else SemanticRevisionViewState.INDETERMINATE
        )
        items.append(
            SemanticRevisionAssessment(
                memory_identity=MemoryIdentity(
                    MemoryKind.SEMANTIC, projected_item.memory.memory_key
                ),
                revision_key=projected_item.revision.revision_key,
                projected_status=projected_item.memory.status,
                view_state=item_state,
                valid_from=projected_item.memory.valid_from,
                valid_until=projected_item.memory.valid_until,
                sources=tuple(sources),
                quality=quality,
                currentness=source_currentness(sources),
                promotion=semantic_promotion_from_metadata(projected_item.memory.metadata),
                transitions=projected_item.transitions,
                historical_attribution_complete=history_complete,
            )
        )
        all_sources.extend(sources)
        referenced_count += count
    quality = evidence_quality(
        all_sources, referenced_count=referenced_count, semantic_references=referenced_count
    )
    reasons.update(quality_reasons(all_sources, quality))
    conflict = _conflict_assessment(records, items, complete)
    if conflict.state is SourceConflictState.SEMANTIC_CONFLICT:
        reasons.add(SourceAssessmentReason.SEMANTIC_CONTESTED)
    if conflict.cross_source_disagreement:
        reasons.add(SourceAssessmentReason.CROSS_SOURCE_DISAGREEMENT)
    active = [i for i in items if i.view_state is SemanticRevisionViewState.VISIBLE_CURRENT]
    if (
        not complete
        or any(i.view_state is SemanticRevisionViewState.INDETERMINATE for i in items)
        or SourceAssessmentReason.TEMPORAL_HISTORY_UNAVAILABLE in reasons
    ):
        state = SemanticKnowledgeState.INDETERMINATE
    elif conflict.state is SourceConflictState.SEMANTIC_CONFLICT:
        state = SemanticKnowledgeState.CONTESTED
    elif active:
        one = any(
            p.memory.cardinality is SemanticCardinality.ONE
            and p.revision.revision_key in {i.revision_key for i in active}
            for p in projected
        )
        state = (
            SemanticKnowledgeState.INDETERMINATE
            if one and len(active) > 1
            else SemanticKnowledgeState.KNOWN
        )
    elif any(i.view_state is SemanticRevisionViewState.WITHDRAWN for i in items):
        state = SemanticKnowledgeState.WITHDRAWN
        reasons.add(SourceAssessmentReason.FACT_WITHDRAWN)
    elif any(i.view_state is SemanticRevisionViewState.VISIBLE_HISTORICAL for i in items):
        state = SemanticKnowledgeState.HISTORICAL_ONLY
        reasons.add(SourceAssessmentReason.HISTORICAL_ONLY)
    else:
        state = SemanticKnowledgeState.UNKNOWN
    return SemanticSlotAssessment(
        tenant_id=tenant_id,
        subject_id=subject_id,
        slot_key=slot_key,
        assessed_at=as_of,
        valid_at=valid_at,
        state=state,
        conflict=conflict,
        revisions=tuple(items),
        quality=quality,
        inventory_complete=complete,
        reasons=ordered_reasons(reasons),
    )


def _source_ids(
    item: SemanticRevisionAssessment, relation: SourceEvidenceRelation
) -> set[tuple[str, str]]:
    return {
        (s.source_namespace, s.source_record_id)
        for s in item.sources
        if s.relation is relation
        and s.resolution_status in _RESOLVED
        and s.source_namespace is not None
        and s.source_record_id is not None
    }


def _conflict_assessment(
    records: SemanticSlotRecords | None,
    items: Sequence[SemanticRevisionAssessment],
    complete: bool,
) -> SourceConflictAssessment:
    visible = {
        i.revision_key: i
        for i in items
        if i.view_state is SemanticRevisionViewState.VISIBLE_CURRENT
    }
    contested = {
        k for k, i in visible.items() if i.projected_status is SemanticMemoryStatus.CONTESTED
    }
    pairs = (
        [
            (r.left_revision_key, r.right_revision_key)
            for r in records.relations
            if r.relation is SemanticUpdateRelation.CONFLICTS
            and r.left_revision_key in visible
            and r.right_revision_key in visible
        ]
        if records
        else []
    )
    cross = False
    supporters: set[tuple[str, str]] = set()
    contradictors: set[tuple[str, str]] = set()
    for item in visible.values():
        supporters.update(_source_ids(item, SourceEvidenceRelation.SEMANTIC_SUPPORT))
        contradictors.update(_source_ids(item, SourceEvidenceRelation.SEMANTIC_CONTRADICTION))
    for left, right in pairs:
        contested.update((left, right))
        left_sources = _source_ids(visible[left], SourceEvidenceRelation.SEMANTIC_SUPPORT)
        right_sources = _source_ids(visible[right], SourceEvidenceRelation.SEMANTIC_SUPPORT)
        cross |= bool(left_sources and right_sources and len(left_sources | right_sources) > 1)
    attribution_complete = complete and all(
        i.quality.has_evidence
        and i.quality.revision_coverage_complete
        and i.quality.relation_coverage_complete is True
        for i in visible.values()
    )
    return SourceConflictAssessment(
        state=SourceConflictState.SEMANTIC_CONFLICT
        if contested
        else (SourceConflictState.NONE_OBSERVED if complete else SourceConflictState.INDETERMINATE),
        conflicting_revision_keys=tuple(sorted(contested)),
        distinct_supporting_sources=len(supporters),
        distinct_contradicting_sources=len(contradictors),
        cross_source_disagreement=True if cross else False if attribution_complete else None,
        is_complete=attribution_complete,
    )
