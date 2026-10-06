"""Retrieval-induced inhibition scope and trace induction."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from uuid import uuid4

from cogkura.exceptions import ValidationError
from cogkura.models import (
    BehavioralCompetitionEligibility,
    BehavioralStructuralAnchor,
    InhibitionConfig,
    InhibitionScopeSignature,
    InhibitoryTrace,
    MemoryIdentity,
    RecallResult,
    RetrievalCompetitionSnapshot,
)


def build_inhibition_scope(
    eligibility: BehavioralCompetitionEligibility,
    *,
    subject_entity_id: str | None,
    predicate: str | None,
    semantic_slot_key: str | None,
) -> InhibitionScopeSignature:
    """Freeze the structural scope that made a pair behaviourally credible."""
    anchor = eligibility.structural_anchor
    if anchor is None:
        raise ValidationError("inhibition scope requires a structural anchor.")
    if anchor is BehavioralStructuralAnchor.SEMANTIC_SLOT:
        return InhibitionScopeSignature(
            structural_anchor=anchor,
            subject_id=subject_entity_id,
            predicate=predicate,
            semantic_slot_key=semantic_slot_key,
        )
    if anchor is BehavioralStructuralAnchor.SUBJECT_PREDICATE:
        entity_ids = eligibility.shared_query_entity_ids
        if not entity_ids and subject_entity_id:
            entity_ids = (subject_entity_id,)
        return InhibitionScopeSignature(
            structural_anchor=anchor,
            subject_id=subject_entity_id,
            predicate=predicate,
            entity_ids=entity_ids,
        )
    return InhibitionScopeSignature(
        structural_anchor=anchor,
        entity_ids=eligibility.shared_query_entity_ids,
        feature_ids=eligibility.shared_query_features,
    )


def _result_identity(result: RecallResult) -> MemoryIdentity:
    return MemoryIdentity(memory_kind=result.memory_kind, memory_key=result.memory.memory_key)


def build_inhibitory_traces(
    consumed: Sequence[RecallResult],
    *,
    tenant_id: str,
    config: InhibitionConfig,
    induced_at: datetime,
    request_id: str | None,
) -> tuple[InhibitoryTrace, ...]:
    """Select bounded inhibitory traces for one explicit use event."""
    consumed_ids = {_result_identity(result) for result in consumed}
    pending: list[tuple[float, RetrievalCompetitionSnapshot, MemoryIdentity]] = []
    for selected in consumed:
        if selected.memory.tenant_id != tenant_id:
            raise ValidationError("recall result tenant_id must match record_access tenant_id.")
        diagnostics = selected.diagnostics
        if diagnostics is None:
            continue
        selected_identity = _result_identity(selected)
        for snapshot in diagnostics.inhibition_candidates:
            if not snapshot.scope_eligible:
                continue
            if snapshot.competitor_identity in consumed_ids:
                continue
            if (
                snapshot.candidate_lineage_group is not None
                and snapshot.candidate_lineage_group == snapshot.competitor_lineage_group
            ):
                continue
            pressure = snapshot.competition_strength * snapshot.competitor_accessibility
            if pressure < config.minimum_induction_pressure:
                continue
            pending.append((min(pressure, 1.0), snapshot, selected_identity))

    pending.sort(
        key=lambda item: (
            -item[0],
            -item[1].competition_strength,
            item[2].memory_kind.value,
            item[2].memory_key,
            item[1].competitor_identity.memory_kind.value,
            item[1].competitor_identity.memory_key,
            item[1].scope.scope_key,
        )
    )
    traces: list[InhibitoryTrace] = []
    for pressure, snapshot, selected_identity in pending[: config.max_traces_per_use]:
        traces.append(
            InhibitoryTrace(
                id=str(uuid4()),
                tenant_id=tenant_id,
                inhibited_identity=snapshot.competitor_identity,
                selected_identity=selected_identity,
                direction=snapshot.direction,
                scope=snapshot.scope,
                competition_strength=snapshot.competition_strength,
                competitor_accessibility=snapshot.competitor_accessibility,
                induction_pressure=pressure,
                retrieval_evaluated_at=snapshot.retrieval_evaluated_at,
                induced_at=induced_at,
                request_id=request_id,
            )
        )
    return tuple(traces)
