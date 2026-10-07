"""Apply stored inhibitory traces during declarative retrieval."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Protocol

from cogkura.algorithms.retrieval_features import (
    canonical_content_features,
    predicate_content_features,
)
from cogkura.models import (
    BehavioralQueryScope,
    BehavioralStructuralAnchor,
    InhibitionConfig,
    InhibitionScopeMatch,
    InhibitionScopeMatchReason,
    InhibitionScopeSignature,
    InhibitoryTrace,
    MemoryIdentity,
    PersistentInhibitionContribution,
    PersistentInhibitionDiagnostics,
    RecallInspectionCandidate,
    RecallResult,
    RetrievalDiagnostics,
    StoredEpisode,
    StoredSemanticMemory,
)


@dataclass(frozen=True, slots=True)
class InhibitionCandidateProfile:
    """Retrieval-local memory facts used to match an inhibition scope."""

    identity: MemoryIdentity
    subject_id: str | None
    subject_entity_id: str | None
    semantic_slot_key: str | None
    predicate: str | None
    entity_ids: tuple[str, ...]
    retrieval_features: tuple[str, ...]


class InhibitionScopeMatcher(Protocol):
    """Decide whether a stored inhibition scope applies to the current retrieval."""

    def matches(
        self,
        *,
        trace_scope: InhibitionScopeSignature,
        candidate: InhibitionCandidateProfile,
        query_scope: BehavioralQueryScope,
    ) -> InhibitionScopeMatch:
        """Return a fail-closed scope match."""


def remaining_strength(
    induction_pressure: float,
    *,
    induced_at: datetime,
    evaluated_at: datetime,
    half_life_seconds: float,
) -> tuple[float, float]:
    """Return remaining strength and elapsed seconds since induction."""
    elapsed = max(0.0, (evaluated_at - induced_at).total_seconds())
    recovered = induction_pressure * (2.0 ** (-elapsed / half_life_seconds))
    return min(max(recovered, 0.0), 1.0), elapsed


def recovery_lookback_seconds(config: InhibitionConfig) -> float:
    """Oldest age at which a full-strength trace can still exceed the cutoff."""
    return config.recovery_half_life_seconds * math.log2(1.0 / config.minimum_remaining_strength)


def noisy_or(values: Sequence[float]) -> float:
    """Bounded accumulation of independent pressures in ``[0, 1]``."""
    product = 1.0
    for value in values:
        product *= 1.0 - min(max(value, 0.0), 1.0)
    return 1.0 - product


def _features(*parts: str | None) -> frozenset[str]:
    combined: set[str] = set()
    for part in parts:
        if part and part.strip():
            combined.update(canonical_content_features(part))
    return frozenset(combined)


def _query_features(query_scope: BehavioralQueryScope) -> frozenset[str]:
    return frozenset(query_scope.features)


def _predicate_features(predicate: str | None) -> frozenset[str]:
    return predicate_content_features(predicate)


def _match(
    *,
    matched: bool,
    reason: InhibitionScopeMatchReason,
    anchor: BehavioralStructuralAnchor,
    entity_ids: tuple[str, ...] = (),
    features: tuple[str, ...] = (),
) -> InhibitionScopeMatch:
    return InhibitionScopeMatch(
        matched=matched,
        reason=reason,
        structural_anchor=anchor,
        matched_entity_ids=entity_ids,
        matched_features=features,
    )


def _query_supports_subject_and_predicate(
    query_scope: BehavioralQueryScope,
    *,
    subject_id: str | None,
    predicate: str | None,
) -> InhibitionScopeMatchReason | None:
    """Return a mismatch reason, or None when the query addresses this scope."""
    query_features = _query_features(query_scope)
    if query_scope.predicate is not None and query_scope.predicate != predicate:
        return InhibitionScopeMatchReason.QUERY_PREDICATE_MISMATCH
    if query_scope.entity_ids:
        anchors = {entity_id for entity_id in (subject_id,) if entity_id}
        if not anchors.intersection(query_scope.entity_ids):
            return InhibitionScopeMatchReason.QUERY_ENTITY_MISMATCH
    if query_scope.predicate is None and not query_scope.entity_ids:
        subject_tokens = _features(subject_id)
        predicate_tokens = _predicate_features(predicate)
        if subject_tokens and not subject_tokens.issubset(query_features):
            return InhibitionScopeMatchReason.QUERY_ENTITY_MISMATCH
        if predicate_tokens and not predicate_tokens.issubset(query_features):
            return InhibitionScopeMatchReason.QUERY_PREDICATE_MISMATCH
        if subject_tokens and not predicate_tokens:
            return InhibitionScopeMatchReason.QUERY_PREDICATE_MISMATCH
        if not subject_tokens and not predicate_tokens:
            return InhibitionScopeMatchReason.QUERY_FEATURE_MISMATCH
    elif query_scope.predicate is None:
        predicate_tokens = _predicate_features(predicate)
        if predicate_tokens and not predicate_tokens.issubset(query_features):
            return InhibitionScopeMatchReason.QUERY_PREDICATE_MISMATCH
    elif not query_scope.entity_ids:
        subject_tokens = _features(subject_id)
        if subject_tokens and not subject_tokens.issubset(query_features):
            return InhibitionScopeMatchReason.QUERY_ENTITY_MISMATCH
    return None


@dataclass(frozen=True, slots=True)
class DeterministicInhibitionScopeMatcher:
    """Conservative scope match using existing canonical retrieval features."""

    def matches(
        self,
        *,
        trace_scope: InhibitionScopeSignature,
        candidate: InhibitionCandidateProfile,
        query_scope: BehavioralQueryScope,
    ) -> InhibitionScopeMatch:
        anchor = trace_scope.structural_anchor
        if anchor is BehavioralStructuralAnchor.SEMANTIC_SLOT:
            return self._semantic_slot(trace_scope, candidate, query_scope)
        if anchor is BehavioralStructuralAnchor.SUBJECT_PREDICATE:
            return self._subject_predicate(trace_scope, candidate, query_scope)
        return self._query_scope(trace_scope, candidate, query_scope)

    def _semantic_slot(
        self,
        trace_scope: InhibitionScopeSignature,
        candidate: InhibitionCandidateProfile,
        query_scope: BehavioralQueryScope,
    ) -> InhibitionScopeMatch:
        if (
            not trace_scope.semantic_slot_key
            or candidate.semantic_slot_key != trace_scope.semantic_slot_key
        ):
            return _match(
                matched=False,
                reason=InhibitionScopeMatchReason.MEMORY_SCOPE_MISMATCH,
                anchor=trace_scope.structural_anchor,
            )
        mismatch = _query_supports_subject_and_predicate(
            query_scope,
            subject_id=trace_scope.subject_id or candidate.subject_entity_id,
            predicate=trace_scope.predicate or candidate.predicate,
        )
        if mismatch is not None:
            return _match(matched=False, reason=mismatch, anchor=trace_scope.structural_anchor)
        return _match(
            matched=True,
            reason=InhibitionScopeMatchReason.SEMANTIC_SLOT_MATCH,
            anchor=trace_scope.structural_anchor,
            features=tuple(sorted(_predicate_features(trace_scope.predicate))),
        )

    def _subject_predicate(
        self,
        trace_scope: InhibitionScopeSignature,
        candidate: InhibitionCandidateProfile,
        query_scope: BehavioralQueryScope,
    ) -> InhibitionScopeMatch:
        if (
            not trace_scope.subject_id
            or not trace_scope.predicate
            or candidate.subject_entity_id != trace_scope.subject_id
            or candidate.predicate != trace_scope.predicate
        ):
            return _match(
                matched=False,
                reason=InhibitionScopeMatchReason.MEMORY_SCOPE_MISMATCH,
                anchor=trace_scope.structural_anchor,
            )
        mismatch = _query_supports_subject_and_predicate(
            query_scope,
            subject_id=trace_scope.subject_id,
            predicate=trace_scope.predicate,
        )
        if mismatch is not None:
            return _match(matched=False, reason=mismatch, anchor=trace_scope.structural_anchor)
        return _match(
            matched=True,
            reason=InhibitionScopeMatchReason.SUBJECT_PREDICATE_MATCH,
            anchor=trace_scope.structural_anchor,
            entity_ids=(trace_scope.subject_id,),
            features=tuple(sorted(_predicate_features(trace_scope.predicate))),
        )

    def _query_scope(
        self,
        trace_scope: InhibitionScopeSignature,
        candidate: InhibitionCandidateProfile,
        query_scope: BehavioralQueryScope,
    ) -> InhibitionScopeMatch:
        query_features = _query_features(query_scope)
        anchor_tokens = _features(*trace_scope.entity_ids, trace_scope.subject_id)
        matched_entities: tuple[str, ...] = ()
        if trace_scope.entity_ids:
            if query_scope.entity_ids:
                matched_entities = tuple(
                    sorted(set(trace_scope.entity_ids).intersection(query_scope.entity_ids))
                )
                if not matched_entities:
                    return _match(
                        matched=False,
                        reason=InhibitionScopeMatchReason.QUERY_ENTITY_MISMATCH,
                        anchor=trace_scope.structural_anchor,
                    )
            else:
                covered = [
                    entity_id
                    for entity_id in trace_scope.entity_ids
                    if _features(entity_id) and _features(entity_id).issubset(query_features)
                ]
                if not covered:
                    return _match(
                        matched=False,
                        reason=InhibitionScopeMatchReason.QUERY_ENTITY_MISMATCH,
                        anchor=trace_scope.structural_anchor,
                    )
                matched_entities = tuple(sorted(covered))
            candidate_entities = set(candidate.entity_ids)
            if candidate.subject_entity_id:
                candidate_entities.add(candidate.subject_entity_id)
            if candidate_entities and not candidate_entities.intersection(trace_scope.entity_ids):
                return _match(
                    matched=False,
                    reason=InhibitionScopeMatchReason.MEMORY_SCOPE_MISMATCH,
                    anchor=trace_scope.structural_anchor,
                )
        functional = set(trace_scope.feature_ids) - anchor_tokens
        matched_features = tuple(sorted(functional.intersection(query_features)))
        if trace_scope.feature_ids and not matched_features:
            return _match(
                matched=False,
                reason=InhibitionScopeMatchReason.QUERY_FEATURE_MISMATCH,
                anchor=trace_scope.structural_anchor,
            )
        if not matched_entities and not matched_features:
            return _match(
                matched=False,
                reason=InhibitionScopeMatchReason.QUERY_FEATURE_MISMATCH,
                anchor=trace_scope.structural_anchor,
            )
        return _match(
            matched=True,
            reason=InhibitionScopeMatchReason.QUERY_SCOPE_MATCH,
            anchor=trace_scope.structural_anchor,
            entity_ids=matched_entities,
            features=matched_features,
        )


def candidate_profile_from_result(result: RecallResult) -> InhibitionCandidateProfile | None:
    """Build an application profile from a scored recall result."""
    diagnostics = result.diagnostics
    memory = result.memory
    subject_entity_id: str | None = None
    predicate: str | None = None
    semantic_slot_key = diagnostics.semantic_slot_key if diagnostics is not None else None
    entity_ids: set[str] = set()
    if isinstance(memory, StoredSemanticMemory):
        subject_entity_id = memory.subject_entity_id
        predicate = memory.predicate
        semantic_slot_key = semantic_slot_key or memory.slot_key
        entity_ids.update(entity.entity_id for entity in memory.entities if entity.entity_id)
        if memory.subject_entity_id:
            entity_ids.add(memory.subject_entity_id)
    elif isinstance(memory, StoredEpisode):
        entity_ids.update(entity.entity_id for entity in memory.entities if entity.entity_id)
    else:
        return None
    features: set[str] = set(canonical_content_features(memory.statement))
    if diagnostics is not None:
        features.update(diagnostics.matched_direct_features)
        features.update(diagnostics.matched_evidence_features)
    return InhibitionCandidateProfile(
        identity=MemoryIdentity(memory_kind=result.memory_kind, memory_key=memory.memory_key),
        subject_id=memory.subject_id,
        subject_entity_id=subject_entity_id,
        semantic_slot_key=semantic_slot_key,
        predicate=predicate,
        entity_ids=tuple(sorted(entity_ids)),
        retrieval_features=tuple(sorted(features)),
    )


def _presentation_score(activation: float, threshold: float) -> float:
    return 1.0 / (1.0 + math.exp(-(activation - threshold)))


def evaluate_persistent_inhibition(
    result: RecallResult,
    traces: Sequence[InhibitoryTrace],
    *,
    config: InhibitionConfig,
    query_scope: BehavioralQueryScope,
    matcher: InhibitionScopeMatcher,
    evaluated_at: datetime,
    retrieval_threshold: float,
    latency_factor: float,
    latency_exponent: float,
    rank_activation: float,
) -> RecallResult:
    """Apply scope-matched recovered traces once. Does not mutate stored traces."""
    profile = candidate_profile_from_result(result)
    diagnostics = result.diagnostics
    contributions: list[PersistentInhibitionContribution] = []
    matched_count = 0
    if profile is not None:
        for trace in traces:
            if trace.induced_at > evaluated_at:
                continue
            match = matcher.matches(
                trace_scope=trace.scope,
                candidate=profile,
                query_scope=query_scope,
            )
            if not match.matched:
                continue
            matched_count += 1
            strength, elapsed = remaining_strength(
                trace.induction_pressure,
                induced_at=trace.induced_at,
                evaluated_at=evaluated_at,
                half_life_seconds=config.recovery_half_life_seconds,
            )
            if strength < config.minimum_remaining_strength:
                continue
            contributions.append(
                PersistentInhibitionContribution(
                    trace_id=trace.id,
                    selected_identity=trace.selected_identity,
                    direction=trace.direction,
                    scope_key=trace.scope.scope_key,
                    induction_pressure=trace.induction_pressure,
                    age_seconds=elapsed,
                    remaining_strength=strength,
                )
            )
    contributions.sort(key=lambda item: (-item.remaining_strength, item.trace_id))
    pressure = noisy_or([item.remaining_strength for item in contributions])
    penalty = -config.inhibition_weight * pressure
    activation_before = result.activation
    activation_after = activation_before + penalty
    rank_before = rank_activation
    rank_after = rank_before + penalty
    crossed = activation_before >= retrieval_threshold and activation_after < retrieval_threshold
    inhibition = PersistentInhibitionDiagnostics(
        stored_trace_count=len(traces),
        evaluated_trace_count=len(traces),
        matched_trace_count=matched_count,
        effective_trace_count=len(contributions),
        inhibition_pressure=pressure,
        penalty=penalty,
        contributions=tuple(contributions),
    )
    updated_diagnostics: RetrievalDiagnostics | None = diagnostics
    if diagnostics is not None:
        updated_diagnostics = replace(
            diagnostics,
            persistent_inhibition=inhibition,
            activation_before_inhibition=activation_before,
            rank_activation_before_inhibition=rank_before,
            crossed_activation_threshold_due_to_inhibition=crossed,
            rank_activation=rank_after,
        )
    components = replace(
        result.components,
        inhibition=penalty,
        total=activation_after,
    )
    return replace(
        result,
        activation=activation_after,
        score=_presentation_score(activation_after, retrieval_threshold),
        latency_seconds=latency_factor * math.exp(-latency_exponent * activation_after),
        components=components,
        diagnostics=updated_diagnostics,
    )


def apply_persistent_inhibition(
    results: Sequence[RecallResult],
    rank_by_identity: Mapping[MemoryIdentity, float],
    traces_by_identity: Mapping[MemoryIdentity, Sequence[InhibitoryTrace]],
    *,
    config: InhibitionConfig,
    query_scope: BehavioralQueryScope,
    matcher: InhibitionScopeMatcher,
    evaluated_at: datetime,
    retrieval_threshold: float,
    latency_factor: float,
    latency_exponent: float,
) -> tuple[list[RecallResult], dict[MemoryIdentity, float]]:
    """Apply persistent inhibition before transient competition."""
    if not config.apply_to_recall:
        return list(results), dict(rank_by_identity)
    updated_rank = dict(rank_by_identity)
    updated: list[RecallResult] = []
    for result in results:
        identity = MemoryIdentity(
            memory_kind=result.memory_kind,
            memory_key=result.memory.memory_key,
        )
        rank_activation = updated_rank.get(identity, result.activation)
        revised = evaluate_persistent_inhibition(
            result,
            traces_by_identity.get(identity, ()),
            config=config,
            query_scope=query_scope,
            matcher=matcher,
            evaluated_at=evaluated_at,
            retrieval_threshold=retrieval_threshold,
            latency_factor=latency_factor,
            latency_exponent=latency_exponent,
            rank_activation=rank_activation,
        )
        penalty = revised.components.inhibition
        updated_rank[identity] = rank_activation + penalty
        updated.append(revised)
    return updated, updated_rank


def assign_inhibition_ranks(
    candidates: Sequence[RecallInspectionCandidate],
) -> tuple[RecallInspectionCandidate, ...]:
    """Assign pre/post persistent-inhibition ranks without changing interference ranks."""
    eligible = [
        candidate
        for candidate in candidates
        if candidate.diagnostics is not None
        and candidate.diagnostics.activation_before_inhibition is not None
    ]
    if not eligible:
        return tuple(candidates)

    def _before(candidate: RecallInspectionCandidate) -> float:
        diagnostics = candidate.diagnostics
        assert diagnostics is not None
        if diagnostics.rank_activation_before_inhibition is not None:
            return diagnostics.rank_activation_before_inhibition
        assert diagnostics.activation_before_inhibition is not None
        return diagnostics.activation_before_inhibition

    def _after(candidate: RecallInspectionCandidate) -> float:
        diagnostics = candidate.diagnostics
        assert diagnostics is not None
        if diagnostics.rank_activation_before_interference is not None:
            return diagnostics.rank_activation_before_interference
        return diagnostics.rank_activation

    before_sorted = sorted(
        eligible,
        key=lambda item: (-_before(item), item.memory_kind.value, item.memory.memory_key),
    )
    after_sorted = sorted(
        eligible,
        key=lambda item: (-_after(item), item.memory_kind.value, item.memory.memory_key),
    )
    before_rank = {
        candidate.memory.memory_key: index + 1 for index, candidate in enumerate(before_sorted)
    }
    after_rank = {
        candidate.memory.memory_key: index + 1 for index, candidate in enumerate(after_sorted)
    }
    updated: list[RecallInspectionCandidate] = []
    for candidate in candidates:
        if candidate.memory.memory_key not in before_rank:
            updated.append(candidate)
            continue
        before = before_rank[candidate.memory.memory_key]
        after = after_rank[candidate.memory.memory_key]
        updated.append(
            replace(
                candidate,
                rank_before_inhibition=before,
                rank_after_inhibition=after,
                inhibition_rank_delta=after - before,
            )
        )
    return tuple(updated)
