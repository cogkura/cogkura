"""Context observability and retrieval-level contextual metamemory."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from cogkura.models import (
    ContextObservabilityReason,
    RecallInspectionCandidate,
    RecallInspectionDisposition,
    RecallResult,
    RetrievalContextDiagnostics,
    RetrievalContextState,
    RetrievalDiagnostics,
)
from cogkura.observations.encoding_context import RetrievalContext

_STANDARD_CONTEXT_DIMENSION_COUNT = 8

_DISCRIMINATION_DISPOSITIONS = frozenset(
    {
        RecallInspectionDisposition.RETURNED,
        RecallInspectionDisposition.BELOW_THRESHOLD,
        RecallInspectionDisposition.LIMITED,
    }
)


def context_strength_from_diagnostics(diagnostics: RetrievalDiagnostics | None) -> float:
    """Return contextual strength for episodic or semantic candidates."""
    if diagnostics is None:
        return 0.0
    if diagnostics.context_reinstatement is not None:
        return diagnostics.context_reinstatement.strength
    if diagnostics.support_context is not None:
        return diagnostics.support_context.strength
    return 0.0


def candidate_has_comparable_context(diagnostics: RetrievalDiagnostics | None) -> bool:
    """Return True when contextual evidence is comparable for this candidate."""
    if diagnostics is None:
        return False
    if diagnostics.context_match is not None and diagnostics.context_match.score is not None:
        return True
    if diagnostics.support_context is not None:
        return diagnostics.support_context.comparable_support_count > 0
    return False


def candidate_has_partial_context_conflict(diagnostics: RetrievalDiagnostics | None) -> bool:
    """Return True when comparable context has mixed match/mismatch dimensions."""
    if diagnostics is None:
        return False
    support = diagnostics.support_context
    if support is not None:
        if support.matching_support_count > 0 and support.conflicting_support_count > 0:
            return True
    if diagnostics.context_match is None:
        return False
    match = diagnostics.context_match
    if match.score is None:
        return False
    dimensions = (*match.dimensions, *match.attribute_dimensions)
    has_match = any(
        dimension.state.value in {"match", "partial_match"} and dimension.score not in (None, 0.0)
        for dimension in dimensions
    )
    has_mismatch = any(
        dimension.score == 0.0 or dimension.state.value == "mismatch" for dimension in dimensions
    )
    if has_match and has_mismatch:
        return True
    return match.score is not None and 0.0 < match.score < 1.0


def provided_dimension_count(retrieval_context: RetrievalContext | None) -> int:
    """Count populated standard retrieval-context dimensions."""
    if retrieval_context is None or retrieval_context.is_empty():
        return 0
    count = 0
    for value in (
        retrieval_context.conversation_id,
        retrieval_context.thread_id,
        retrieval_context.session_id,
        retrieval_context.goal,
        retrieval_context.activity,
        retrieval_context.domain,
        retrieval_context.location,
    ):
        if value is not None:
            count += 1
    if retrieval_context.temporal_context:
        count += 1
    return count


def cue_specificity(retrieval_context: RetrievalContext | None) -> float | None:
    """Descriptive ratio of provided standard dimensions (not used in activation)."""
    if retrieval_context is None or retrieval_context.is_empty():
        return None
    return provided_dimension_count(retrieval_context) / _STANDARD_CONTEXT_DIMENSION_COUNT


def discrimination_set(
    candidates: Sequence[RecallInspectionCandidate],
) -> tuple[RecallInspectionCandidate, ...]:
    """Return inspect candidates eligible for retrieval-level context discrimination."""
    return tuple(
        candidate
        for candidate in candidates
        if candidate.disposition in _DISCRIMINATION_DISPOSITIONS
        and candidate.diagnostics is not None
    )


def discrimination_set_from_recall(
    candidates: Sequence[RecallResult],
) -> tuple[RecallResult, ...]:
    """Return recall results for metamemory contextual assessment (all threshold-qualified)."""
    return tuple(candidate for candidate in candidates if candidate.diagnostics is not None)


@dataclass(frozen=True, slots=True)
class _RankedContextCandidate:
    memory_key: str
    rank_before_context: int
    rank_after_context: int
    context_rank_delta: int


def assign_context_ranks(
    candidates: Sequence[RecallInspectionCandidate],
) -> dict[str, _RankedContextCandidate]:
    """Assign pre/post context ranks over the discrimination set."""
    eligible = discrimination_set(candidates)
    if not eligible:
        return {}

    def _memory_key(candidate: RecallInspectionCandidate) -> str:
        return candidate.memory.memory_key

    pre_ordered = sorted(
        eligible,
        key=lambda item: (
            item.diagnostics.activation_before_context
            if item.diagnostics is not None
            and item.diagnostics.activation_before_context is not None
            else float("-inf"),
            _memory_key(item),
        ),
        reverse=True,
    )
    post_ordered = sorted(
        eligible,
        key=lambda item: (item.activation, _memory_key(item)),
        reverse=True,
    )

    pre_rank: dict[str, int] = {}
    post_rank: dict[str, int] = {}
    for index, candidate in enumerate(pre_ordered, start=1):
        pre_rank[_memory_key(candidate)] = index
    for index, candidate in enumerate(post_ordered, start=1):
        post_rank[_memory_key(candidate)] = index

    ranked: dict[str, _RankedContextCandidate] = {}
    for candidate in eligible:
        key = _memory_key(candidate)
        before = pre_rank[key]
        after = post_rank[key]
        ranked[key] = _RankedContextCandidate(
            memory_key=key,
            rank_before_context=before,
            rank_after_context=after,
            context_rank_delta=before - after,
        )
    return ranked


def with_threshold_crossing(
    diagnostics: RetrievalDiagnostics,
    *,
    retrieval_threshold: float,
    activation_after_context: float,
) -> RetrievalDiagnostics:
    """Attach threshold-crossing attribution without changing activation."""
    before = diagnostics.activation_before_context
    crossed = (
        before is not None
        and before < retrieval_threshold
        and activation_after_context >= retrieval_threshold
    )
    if crossed == diagnostics.crossed_activation_threshold_due_to_context:
        return diagnostics
    return replace(diagnostics, crossed_activation_threshold_due_to_context=crossed)


def apply_inspection_context_attribution(
    candidates: Sequence[RecallInspectionCandidate],
    *,
    retrieval_threshold: float,
) -> tuple[RecallInspectionCandidate, ...]:
    """Apply threshold crossing and context rank attribution to inspect candidates."""
    ranks = assign_context_ranks(candidates)
    updated: list[RecallInspectionCandidate] = []
    for candidate in candidates:
        diagnostics = candidate.diagnostics
        if diagnostics is not None:
            diagnostics = with_threshold_crossing(
                diagnostics,
                retrieval_threshold=retrieval_threshold,
                activation_after_context=candidate.activation,
            )
        rank_info = ranks.get(candidate.memory.memory_key)
        updated.append(
            replace(
                candidate,
                diagnostics=diagnostics,
                rank_before_context=rank_info.rank_before_context if rank_info else None,
                rank_after_context=rank_info.rank_after_context if rank_info else None,
                context_rank_delta=rank_info.context_rank_delta if rank_info else None,
            )
        )
    return tuple(updated)


class RetrievalContextPolicy(Protocol):
    """Classify retrieval-level contextual evidence."""

    def evaluate(
        self,
        *,
        retrieval_context: RetrievalContext | None,
        candidates: Sequence[RecallInspectionCandidate],
        underspecified_margin: float,
    ) -> RetrievalContextDiagnostics:
        """Return retrieval-level contextual diagnostics."""
        ...

    def evaluate_recall_pool(
        self,
        *,
        retrieval_context: RetrievalContext | None,
        candidates: Sequence[RecallResult],
        underspecified_margin: float,
    ) -> RetrievalContextDiagnostics:
        """Return contextual diagnostics over a threshold-qualified recall pool."""
        ...


@dataclass(frozen=True, slots=True)
class DeterministicRetrievalContextPolicy:
    """Deterministic retrieval-level contextual metamemory classification."""

    def evaluate(
        self,
        *,
        retrieval_context: RetrievalContext | None,
        candidates: Sequence[RecallInspectionCandidate],
        underspecified_margin: float,
    ) -> RetrievalContextDiagnostics:
        eligible = discrimination_set(candidates)
        comparable = [
            candidate
            for candidate in eligible
            if candidate_has_comparable_context(candidate.diagnostics)
        ]
        strengths = [
            (candidate.memory.memory_key, context_strength_from_diagnostics(candidate.diagnostics))
            for candidate in comparable
        ]
        matching_count = sum(1 for _, strength in strengths if strength > 0.0)
        return _build_diagnostics(
            retrieval_context=retrieval_context,
            strengths=strengths,
            comparable_count=len(comparable),
            matching_count=matching_count,
            underspecified_margin=underspecified_margin,
            candidates=eligible,
        )

    def evaluate_recall_pool(
        self,
        *,
        retrieval_context: RetrievalContext | None,
        candidates: Sequence[RecallResult],
        underspecified_margin: float,
    ) -> RetrievalContextDiagnostics:
        eligible = discrimination_set_from_recall(candidates)
        comparable = [
            result for result in eligible if candidate_has_comparable_context(result.diagnostics)
        ]
        strengths = [
            (result.memory.memory_key, context_strength_from_diagnostics(result.diagnostics))
            for result in comparable
        ]
        matching_count = sum(1 for _, strength in strengths if strength > 0.0)
        return _build_diagnostics(
            retrieval_context=retrieval_context,
            strengths=strengths,
            comparable_count=len(comparable),
            matching_count=matching_count,
            underspecified_margin=underspecified_margin,
            candidates=(),
        )


def _build_diagnostics(
    *,
    retrieval_context: RetrievalContext | None,
    strengths: list[tuple[str, float]],
    comparable_count: int,
    matching_count: int,
    underspecified_margin: float,
    candidates: Sequence[RecallInspectionCandidate],
) -> RetrievalContextDiagnostics:
    reasons: list[ContextObservabilityReason] = []
    if retrieval_context is None or retrieval_context.is_empty():
        return RetrievalContextDiagnostics(
            state=RetrievalContextState.CONTEXT_NOT_PROVIDED,
            provided_dimension_count=0,
            cue_specificity=None,
            comparable_candidate_count=0,
            matching_candidate_count=0,
            top_context_strength=None,
            second_context_strength=None,
            context_margin=None,
            top_candidate_ids=(),
            reasons=(ContextObservabilityReason.NO_RETRIEVAL_CONTEXT,),
            underspecified_margin=underspecified_margin,
        )

    provided = provided_dimension_count(retrieval_context)
    specificity = cue_specificity(retrieval_context)

    if comparable_count == 0:
        return RetrievalContextDiagnostics(
            state=RetrievalContextState.CONTEXT_UNAVAILABLE,
            provided_dimension_count=provided,
            cue_specificity=specificity,
            comparable_candidate_count=0,
            matching_candidate_count=0,
            top_context_strength=None,
            second_context_strength=None,
            context_margin=None,
            top_candidate_ids=(),
            reasons=(ContextObservabilityReason.NO_COMPARABLE_CONTEXT,),
            underspecified_margin=underspecified_margin,
        )

    ordered_strengths = sorted(strengths, key=lambda item: (-item[1], item[0]))
    top_strength = ordered_strengths[0][1]
    second_strength = ordered_strengths[1][1] if len(ordered_strengths) > 1 else None
    margin = top_strength - second_strength if second_strength is not None else None
    top_ids = tuple(key for key, strength in ordered_strengths if strength == top_strength)

    for candidate in candidates:
        if candidate_has_partial_context_conflict(candidate.diagnostics):
            reasons.append(ContextObservabilityReason.PARTIAL_CONTEXT_CONFLICT)
            break

    if matching_count == 0:
        reasons.append(ContextObservabilityReason.NO_CONTEXTUAL_MATCH)
        return RetrievalContextDiagnostics(
            state=RetrievalContextState.CONTEXT_CONFLICT,
            provided_dimension_count=provided,
            cue_specificity=specificity,
            comparable_candidate_count=comparable_count,
            matching_candidate_count=0,
            top_context_strength=top_strength,
            second_context_strength=second_strength,
            context_margin=margin,
            top_candidate_ids=top_ids,
            reasons=tuple(dict.fromkeys(reasons)),
            underspecified_margin=underspecified_margin,
        )

    if len(ordered_strengths) >= 2 and margin is not None and margin <= underspecified_margin:
        state = RetrievalContextState.CONTEXT_UNDERSPECIFIED
        if second_strength is not None and top_strength == second_strength:
            reasons.append(ContextObservabilityReason.MULTIPLE_TOP_CONTEXT_MATCHES)
        else:
            reasons.append(ContextObservabilityReason.LOW_CONTEXT_MARGIN)
    else:
        state = RetrievalContextState.CONTEXT_SUFFICIENT

    for candidate in candidates:
        diagnostics = candidate.diagnostics
        if diagnostics is not None and diagnostics.crossed_activation_threshold_due_to_context:
            reasons.append(ContextObservabilityReason.CONTEXT_RESTORED_THRESHOLD)
            break

    for candidate in candidates:
        if candidate.context_rank_delta not in (None, 0):
            reasons.append(ContextObservabilityReason.CONTEXT_CHANGED_RANK)
            break

    return RetrievalContextDiagnostics(
        state=state,
        provided_dimension_count=provided,
        cue_specificity=specificity,
        comparable_candidate_count=comparable_count,
        matching_candidate_count=matching_count,
        top_context_strength=top_strength,
        second_context_strength=second_strength,
        context_margin=margin,
        top_candidate_ids=top_ids,
        reasons=tuple(dict.fromkeys(reasons)),
        underspecified_margin=underspecified_margin,
    )
