"""Query-level interference and inhibition observability.

Consumes diagnostics already produced by retrieval. Does not rescore candidates,
recompute competition, or read stores.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from cogkura.algorithms.competition import CompetitionEvaluation
from cogkura.models import (
    CompetitionConfig,
    CompetitionDirection,
    InhibitionConfig,
    MemoryIdentity,
    RecallResult,
    RetrievalInterferenceObservability,
    RetrievalInterferenceState,
    StrongestCompetitionDiagnostics,
)


@dataclass(frozen=True, slots=True)
class _ScopeEligiblePair:
    strength: float
    candidate: MemoryIdentity
    competitor: MemoryIdentity
    direction: CompetitionDirection


def summarize_retrieval_interference(
    scored: Sequence[RecallResult],
    evaluation: CompetitionEvaluation | None,
    *,
    competition_config: CompetitionConfig,
    inhibition_config: InhibitionConfig,
) -> RetrievalInterferenceObservability:
    """Summarize interference over the pre-threshold discrimination set."""
    competition_evaluated = competition_config.enabled
    transient_evaluated = competition_evaluated and competition_config.apply_interference
    persistent_evaluated = inhibition_config.apply_to_recall

    diagnostic_count = 0
    scope_eligible_count = 0
    transient_eligible_count = 0
    competing_candidate_count = 0
    max_competition_strength: float | None = 0.0 if competition_evaluated else None
    pairs: list[_ScopeEligiblePair] = []

    if competition_evaluated and evaluation is not None:
        for identity in sorted(evaluation.by_identity, key=_identity_key):
            diagnostics = evaluation.by_identity[identity]
            candidate_competes = False
            for evidence in diagnostics.competitors:
                diagnostic_count += 1
                eligibility = evidence.behavioral_eligibility
                if eligibility is None:
                    continue
                if eligibility.eligible:
                    transient_eligible_count += 1
                if not eligibility.scope_eligible:
                    continue
                scope_eligible_count += 1
                candidate_competes = True
                strength = eligibility.competition_strength
                if max_competition_strength is None or strength > max_competition_strength:
                    max_competition_strength = strength
                pairs.append(
                    _ScopeEligiblePair(
                        strength=strength,
                        candidate=identity,
                        competitor=evidence.competitor_identity,
                        direction=evidence.direction,
                    )
                )
            if candidate_competes:
                competing_candidate_count += 1

    transiently_affected = 0
    threshold_suppressed_by_interference = 0
    max_transient_pressure: float | None = None
    max_transient_penalty_magnitude: float | None = None
    if transient_evaluated:
        max_transient_pressure = 0.0
        max_transient_penalty_magnitude = 0.0
        if evaluation is not None:
            for identity in sorted(evaluation.by_identity, key=_identity_key):
                interference = evaluation.by_identity[identity].interference
                if interference is None:
                    continue
                max_transient_pressure = max(
                    max_transient_pressure,
                    interference.proactive_pressure,
                    interference.retroactive_pressure,
                )
                max_transient_penalty_magnitude = max(
                    max_transient_penalty_magnitude,
                    abs(interference.total_penalty),
                )
        for result in scored:
            if result.components.interference < 0.0:
                transiently_affected += 1
            retrieval_diagnostics = result.diagnostics
            if (
                retrieval_diagnostics is not None
                and retrieval_diagnostics.crossed_activation_threshold_due_to_interference
            ):
                threshold_suppressed_by_interference += 1

    persistently_inhibited = 0
    threshold_suppressed_by_inhibition = 0
    effective_traces = 0
    inactive_traces = 0
    max_persistent_pressure: float | None = None
    max_persistent_penalty_magnitude: float | None = None
    if persistent_evaluated:
        max_persistent_pressure = 0.0
        max_persistent_penalty_magnitude = 0.0
        for result in scored:
            retrieval_diagnostics = result.diagnostics
            if (
                retrieval_diagnostics is not None
                and retrieval_diagnostics.crossed_activation_threshold_due_to_inhibition
            ):
                threshold_suppressed_by_inhibition += 1
            inhibition = (
                retrieval_diagnostics.persistent_inhibition
                if retrieval_diagnostics is not None
                else None
            )
            if inhibition is None:
                continue
            if inhibition.penalty < 0.0:
                persistently_inhibited += 1
            max_persistent_pressure = max(max_persistent_pressure, inhibition.inhibition_pressure)
            max_persistent_penalty_magnitude = max(
                max_persistent_penalty_magnitude,
                abs(inhibition.penalty),
            )
            effective_traces += inhibition.effective_trace_count
            inactive_traces += max(
                0,
                inhibition.matched_trace_count - inhibition.effective_trace_count,
            )

    transient_negative = transiently_affected > 0
    persistent_negative = persistently_inhibited > 0
    if not competition_evaluated and not transient_evaluated and not persistent_evaluated:
        state = RetrievalInterferenceState.NOT_EVALUATED
    elif transient_negative and persistent_negative:
        state = RetrievalInterferenceState.COMBINED
    elif transient_negative:
        state = RetrievalInterferenceState.TRANSIENT_INTERFERENCE
    elif persistent_negative:
        state = RetrievalInterferenceState.PERSISTENT_INHIBITION
    elif scope_eligible_count > 0:
        state = RetrievalInterferenceState.COMPETING
    else:
        state = RetrievalInterferenceState.CLEAR

    return RetrievalInterferenceObservability(
        state=state,
        competition_evaluated=competition_evaluated,
        transient_interference_evaluated=transient_evaluated,
        persistent_inhibition_evaluated=persistent_evaluated,
        diagnostic_relationship_count=diagnostic_count,
        scope_eligible_relationship_count=scope_eligible_count,
        transient_eligible_relationship_count=transient_eligible_count,
        competing_candidate_count=competing_candidate_count,
        transiently_affected_candidate_count=transiently_affected,
        persistently_inhibited_candidate_count=persistently_inhibited,
        threshold_suppressed_by_interference_count=threshold_suppressed_by_interference,
        threshold_suppressed_by_inhibition_count=threshold_suppressed_by_inhibition,
        effective_inhibition_trace_count=effective_traces,
        inactive_matched_inhibition_trace_count=inactive_traces,
        max_competition_strength=max_competition_strength,
        max_transient_pressure=max_transient_pressure,
        max_transient_penalty_magnitude=max_transient_penalty_magnitude,
        max_persistent_inhibition_pressure=max_persistent_pressure,
        max_persistent_inhibition_penalty_magnitude=max_persistent_penalty_magnitude,
        strongest_competition=_strongest_pair(pairs, scored),
    )


def _strongest_pair(
    pairs: Sequence[_ScopeEligiblePair],
    scored: Sequence[RecallResult],
) -> StrongestCompetitionDiagnostics | None:
    if not pairs:
        return None
    selected = min(
        pairs,
        key=lambda pair: (
            -pair.strength,
            _identity_key(pair.candidate),
            _identity_key(pair.competitor),
        ),
    )
    return StrongestCompetitionDiagnostics(
        candidate_identity=selected.candidate,
        competitor_identity=selected.competitor,
        direction=selected.direction,
        competition_strength=selected.strength,
        activation_margin=_activation_margin(selected, scored),
    )


def _activation_margin(pair: _ScopeEligiblePair, scored: Sequence[RecallResult]) -> float | None:
    ranks = {
        _result_identity(result): (
            result.diagnostics.rank_activation_before_interference
            if result.diagnostics is not None
            else None
        )
        for result in scored
    }
    candidate_rank = ranks.get(pair.candidate)
    competitor_rank = ranks.get(pair.competitor)
    if candidate_rank is None or competitor_rank is None:
        return None
    return abs(candidate_rank - competitor_rank)


def _result_identity(result: RecallResult) -> MemoryIdentity:
    return MemoryIdentity(memory_kind=result.memory_kind, memory_key=result.memory.memory_key)


def _identity_key(identity: MemoryIdentity) -> tuple[str, str]:
    return (identity.memory_kind.value, identity.memory_key)
