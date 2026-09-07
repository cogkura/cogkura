"""Unit tests for retrieval-level context observability."""

from __future__ import annotations

from datetime import UTC, datetime
from types import MappingProxyType

import pytest

from cogkura.algorithms.context_observability import (
    DeterministicRetrievalContextPolicy,
    apply_inspection_context_attribution,
    assign_context_ranks,
    candidate_has_partial_context_conflict,
    context_strength_from_diagnostics,
    cue_specificity,
    discrimination_set,
    provided_dimension_count,
    with_threshold_crossing,
)
from cogkura.models import (
    ActivationComponents,
    ContextDimensionMatch,
    ContextMatch,
    ContextMatchState,
    ContextObservabilityReason,
    ContextReinstatement,
    ContextReinstatementReason,
    EpisodeEntity,
    EpisodeEvidenceInput,
    MemoryKind,
    RecallInspectionCandidate,
    RecallInspectionDisposition,
    RetrievalContextState,
    RetrievalDiagnostics,
    StoredEpisode,
)
from cogkura.observations.encoding_context import RetrievalContext

_T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
_POLICY = DeterministicRetrievalContextPolicy()


def _diagnostics(
    *,
    activation_before_context: float | None = None,
    activation: float = 0.0,
    context_reinstatement: ContextReinstatement | None = None,
    context_match: ContextMatch | None = None,
    crossed: bool = False,
) -> RetrievalDiagnostics:
    return RetrievalDiagnostics(
        rank_activation=activation,
        accessibility_partial=0.5,
        ranking_partial=0.5,
        conjunction=0.5,
        text_coverage=0.5,
        text_cue_fit=0.5,
        temporal_mode="neutral",
        context_match=context_match,
        context_reinstatement=context_reinstatement,
        activation_before_context=activation_before_context,
        crossed_activation_threshold_due_to_context=crossed,
    )


def _reinstatement(
    *,
    strength: float,
    activation_contribution: float,
    applied: bool,
    reason: ContextReinstatementReason,
    match_score: float | None = None,
) -> ContextReinstatement:
    return ContextReinstatement(
        match_score=strength if match_score is None else match_score,
        cue_coverage=1.0,
        strength=strength,
        weight=max(1.0, activation_contribution),
        activation_contribution=activation_contribution,
        applied=applied,
        reason=reason,
    )


def _components() -> ActivationComponents:
    return ActivationComponents(
        base_level=0.0,
        spreading=0.0,
        partial_match=0.0,
        noise=0.0,
        total=0.0,
        current_state=0.0,
        context_reinstatement=0.0,
    )


def _episode(*, memory_key: str) -> StoredEpisode:
    return StoredEpisode(
        id=f"id-{memory_key}",
        tenant_id="tenant",
        subject_id="subject",
        memory_key=memory_key,
        statement=f"Episode {memory_key}.",
        started_at=_T0,
        ended_at=_T0,
        confidence=0.9,
        importance=0.7,
        is_active=True,
        evidence=(
            EpisodeEvidenceInput(
                observation_id=f"obs-{memory_key}",
                observation_revision=1,
                sequence_number=0,
            ),
        ),
        entities=(EpisodeEntity(entity_id="redis", role="mention"),),
        metadata=MappingProxyType({}),
        created_at=_T0,
        updated_at=_T0,
    )


def _candidate(
    *,
    memory_key: str,
    disposition: RecallInspectionDisposition,
    activation: float,
    diagnostics: RetrievalDiagnostics | None,
    rank: int | None = None,
) -> RecallInspectionCandidate:
    return RecallInspectionCandidate(
        memory_kind=MemoryKind.EPISODE,
        memory=_episode(memory_key=memory_key),
        disposition=disposition,
        activation=activation,
        score=0.8,
        retrieval_threshold=-3.0,
        passed_threshold=disposition is RecallInspectionDisposition.RETURNED,
        soft_admitted=False,
        components=_components(),
        cognitive_traces=(),
        stored_traces=(),
        rank=rank,
        diagnostics=diagnostics,
    )


def test_provided_dimension_count_and_cue_specificity() -> None:
    context = RetrievalContext(
        conversation_id="arch-42",
        activity="architecture-decision",
        domain="payments-api",
        temporal_context=("queue-redesign",),
    )
    assert provided_dimension_count(context) == 4
    assert cue_specificity(context) == pytest.approx(0.5)
    assert provided_dimension_count(None) == 0
    assert cue_specificity(RetrievalContext()) is None


def test_context_strength_from_reinstatement_or_support() -> None:
    reinstatement = _reinstatement(
        strength=0.75,
        activation_contribution=0.25,
        applied=True,
        reason=ContextReinstatementReason.APPLIED,
    )
    diagnostics = _diagnostics(context_reinstatement=reinstatement)
    assert context_strength_from_diagnostics(diagnostics) == pytest.approx(0.75)
    assert context_strength_from_diagnostics(None) == 0.0


def test_discrimination_set_excludes_relevance_filtered_candidates() -> None:
    returned = _candidate(
        memory_key="returned",
        disposition=RecallInspectionDisposition.RETURNED,
        activation=1.0,
        diagnostics=_diagnostics(activation=1.0),
        rank=1,
    )
    filtered = _candidate(
        memory_key="filtered",
        disposition=RecallInspectionDisposition.FILTERED_INSUFFICIENT_RELEVANCE,
        activation=0.5,
        diagnostics=_diagnostics(activation=0.5),
    )
    eligible = discrimination_set((returned, filtered))
    assert [item.memory.memory_key for item in eligible] == ["returned"]


def test_assign_context_ranks_pre_post_delta() -> None:
    reinstatement_a = _reinstatement(
        strength=1.0,
        activation_contribution=0.5,
        applied=True,
        reason=ContextReinstatementReason.APPLIED,
    )
    reinstatement_b = _reinstatement(
        strength=0.0,
        activation_contribution=0.0,
        applied=False,
        reason=ContextReinstatementReason.ZERO_MATCH,
        match_score=0.0,
    )
    candidate_a = _candidate(
        memory_key="candidate-a",
        disposition=RecallInspectionDisposition.RETURNED,
        activation=0.2,
        diagnostics=_diagnostics(
            activation_before_context=-1.0,
            activation=0.2,
            context_reinstatement=reinstatement_a,
            context_match=_match(score=1.0),
        ),
        rank=1,
    )
    candidate_b = _candidate(
        memory_key="candidate-b",
        disposition=RecallInspectionDisposition.RETURNED,
        activation=0.1,
        diagnostics=_diagnostics(
            activation_before_context=0.5,
            activation=0.1,
            context_reinstatement=reinstatement_b,
            context_match=_match(score=0.0),
        ),
        rank=2,
    )
    ranks = assign_context_ranks((candidate_a, candidate_b))
    assert ranks["candidate-a"].rank_before_context == 2
    assert ranks["candidate-a"].rank_after_context == 1
    assert ranks["candidate-a"].context_rank_delta == 1
    assert ranks["candidate-b"].context_rank_delta == -1


def _match(*, score: float | None) -> ContextMatch:
    return ContextMatch(
        dimensions=(),
        attribute_dimensions=(),
        score=score,
        comparable_count=1 if score is not None else 0,
        cue_dimension_count=1,
        cue_coverage=1.0,
    )


def test_with_threshold_crossing_only_when_before_below_and_after_above() -> None:
    diagnostics = _diagnostics(activation_before_context=-2.0, activation=-1.0)
    crossed = with_threshold_crossing(
        diagnostics,
        retrieval_threshold=-1.5,
        activation_after_context=-1.0,
    )
    assert crossed.crossed_activation_threshold_due_to_context is True

    already_above = _diagnostics(activation_before_context=0.0, activation=0.5)
    not_crossed = with_threshold_crossing(
        already_above,
        retrieval_threshold=-1.5,
        activation_after_context=0.5,
    )
    assert not_crossed.crossed_activation_threshold_due_to_context is False


def test_apply_inspection_context_attribution_sets_ranks_and_crossing() -> None:
    reinstatement = _reinstatement(
        strength=1.0,
        activation_contribution=0.6,
        applied=True,
        reason=ContextReinstatementReason.APPLIED,
    )
    candidate = _candidate(
        memory_key="borderline",
        disposition=RecallInspectionDisposition.RETURNED,
        activation=-2.4,
        diagnostics=_diagnostics(
            activation_before_context=-3.1,
            activation=-2.4,
            context_reinstatement=reinstatement,
            context_match=_match(score=1.0),
        ),
        rank=1,
    )
    attributed = apply_inspection_context_attribution(
        (candidate,),
        retrieval_threshold=-3.0,
    )
    assert attributed[0].diagnostics is not None
    assert attributed[0].diagnostics.crossed_activation_threshold_due_to_context is True
    assert attributed[0].rank_before_context == 1
    assert attributed[0].rank_after_context == 1
    assert attributed[0].context_rank_delta == 0


def test_evaluate_not_provided_and_unavailable() -> None:
    not_provided = _POLICY.evaluate(
        retrieval_context=None,
        candidates=(),
        underspecified_margin=0.0,
    )
    assert not_provided.state is RetrievalContextState.CONTEXT_NOT_PROVIDED
    assert ContextObservabilityReason.NO_RETRIEVAL_CONTEXT in not_provided.reasons

    unavailable = _POLICY.evaluate(
        retrieval_context=RetrievalContext(domain="payments-api"),
        candidates=(
            _candidate(
                memory_key="empty-encoding",
                disposition=RecallInspectionDisposition.RETURNED,
                activation=1.0,
                diagnostics=_diagnostics(
                    activation=1.0,
                    context_match=_match(score=None),
                ),
                rank=1,
            ),
        ),
        underspecified_margin=0.0,
    )
    assert unavailable.state is RetrievalContextState.CONTEXT_UNAVAILABLE
    assert ContextObservabilityReason.NO_COMPARABLE_CONTEXT in unavailable.reasons


def test_evaluate_underspecified_tie_and_sufficient_margin() -> None:
    reinstatement = _reinstatement(
        strength=1.0,
        activation_contribution=0.5,
        applied=True,
        reason=ContextReinstatementReason.APPLIED,
    )
    tie_candidates = (
        _candidate(
            memory_key="ep-a",
            disposition=RecallInspectionDisposition.RETURNED,
            activation=1.0,
            diagnostics=_diagnostics(
                activation=1.0,
                context_reinstatement=reinstatement,
                context_match=_match(score=1.0),
            ),
            rank=1,
        ),
        _candidate(
            memory_key="ep-b",
            disposition=RecallInspectionDisposition.RETURNED,
            activation=0.9,
            diagnostics=_diagnostics(
                activation=0.9,
                context_reinstatement=reinstatement,
                context_match=_match(score=1.0),
            ),
            rank=2,
        ),
    )
    underspecified = _POLICY.evaluate(
        retrieval_context=RetrievalContext(activity="architecture-decision"),
        candidates=tie_candidates,
        underspecified_margin=0.0,
    )
    assert underspecified.state is RetrievalContextState.CONTEXT_UNDERSPECIFIED
    assert underspecified.top_context_strength == pytest.approx(1.0)
    assert underspecified.second_context_strength == pytest.approx(1.0)
    assert underspecified.context_margin == pytest.approx(0.0)
    assert ContextObservabilityReason.MULTIPLE_TOP_CONTEXT_MATCHES in underspecified.reasons

    sufficient = _POLICY.evaluate(
        retrieval_context=RetrievalContext(
            activity="architecture-decision",
            domain="payments-api",
        ),
        candidates=(
            _candidate(
                memory_key="payments",
                disposition=RecallInspectionDisposition.RETURNED,
                activation=1.0,
                diagnostics=_diagnostics(
                    activation=1.0,
                    context_reinstatement=reinstatement,
                    context_match=_match(score=1.0),
                ),
                rank=1,
            ),
            _candidate(
                memory_key="analytics",
                disposition=RecallInspectionDisposition.RETURNED,
                activation=0.5,
                diagnostics=_diagnostics(
                    activation=0.5,
                    context_reinstatement=_reinstatement(
                        strength=0.0,
                        activation_contribution=0.0,
                        applied=False,
                        reason=ContextReinstatementReason.ZERO_MATCH,
                        match_score=0.0,
                    ),
                    context_match=_match(score=0.0),
                ),
                rank=2,
            ),
        ),
        underspecified_margin=0.0,
    )
    assert sufficient.state is RetrievalContextState.CONTEXT_SUFFICIENT
    assert sufficient.context_margin == pytest.approx(1.0)


def test_candidate_has_partial_context_conflict() -> None:
    mixed = ContextMatch(
        dimensions=(
            ContextDimensionMatch(
                dimension="domain",
                state=ContextMatchState.MATCH,
                score=1.0,
            ),
            ContextDimensionMatch(
                dimension="activity",
                state=ContextMatchState.MISMATCH,
                score=0.0,
            ),
        ),
        attribute_dimensions=(),
        score=0.5,
        comparable_count=2,
        cue_dimension_count=2,
        cue_coverage=1.0,
    )
    diagnostics = _diagnostics(context_match=mixed)
    assert candidate_has_partial_context_conflict(diagnostics) is True
