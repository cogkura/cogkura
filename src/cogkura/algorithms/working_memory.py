"""Bounded working-memory selection from declarative recall candidates."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from cogkura.algorithms.relevance import calculate_cue_relevance
from cogkura.algorithms.relevance import tokenize as _tokenize
from cogkura.exceptions import ValidationError
from cogkura.models import (
    MemoryIdentity,
    MemoryKind,
    RecallResult,
    RetrievalCue,
    StoredEpisode,
    WorkingMemoryComponents,
    WorkingMemoryConfig,
    WorkingMemoryItem,
    WorkingMemorySnapshot,
)


class TokenEstimator(Protocol):
    """Protocol for estimating token counts from memory statement text."""

    def estimate(self, text: str) -> int:
        """Return a deterministic token estimate for the supplied text."""
        ...


class ApproximateTokenEstimator:
    """Dependency-free UTF-8 byte-length token approximation."""

    def estimate(self, text: str) -> int:
        if not text:
            return 0
        byte_length = len(text.encode("utf-8"))
        return max(1, math.ceil(byte_length / 4))


class WorkingMemorySelector(Protocol):
    """Protocol for bounded working-memory selection."""

    def select(
        self,
        *,
        candidates: Sequence[RecallResult],
        goal: RetrievalCue,
        tenant_id: str,
        subject_id: str | None,
        previous: WorkingMemorySnapshot | None,
        as_of: datetime,
        config: WorkingMemoryConfig,
        token_estimator: TokenEstimator,
        prompt_budget_tokens: int | None = None,
        learning_utilities: Mapping[MemoryIdentity, float] | None = None,
    ) -> WorkingMemorySnapshot:
        """Select a bounded working-memory set from recall candidates."""
        ...


@dataclass(slots=True)
class _ScoredCandidate:
    recall: RecallResult
    goal_relevance: float
    importance: float
    carryover: float
    base_priority: float
    learned_utility: float
    utility_adjustment: float
    adjusted_priority: float
    estimated_tokens: int
    final_score: float = 0.0
    inhibition: float = 0.0


class DeterministicWorkingMemorySelector:
    """Deterministic goal-aware working-memory selection with inhibition."""

    def select(
        self,
        *,
        candidates: Sequence[RecallResult],
        goal: RetrievalCue,
        tenant_id: str,
        subject_id: str | None,
        previous: WorkingMemorySnapshot | None,
        as_of: datetime,
        config: WorkingMemoryConfig,
        token_estimator: TokenEstimator,
        prompt_budget_tokens: int | None = None,
        learning_utilities: Mapping[MemoryIdentity, float] | None = None,
    ) -> WorkingMemorySnapshot:
        if not tenant_id.strip():
            raise ValidationError("tenant_id must not be empty.")
        if as_of.tzinfo is None:
            raise ValidationError("as_of must be timezone-aware.")
        evaluation_time = as_of.astimezone(UTC)

        budget = (
            prompt_budget_tokens if prompt_budget_tokens is not None else config.max_prompt_tokens
        )
        if budget <= 0:
            raise ValidationError("prompt_budget_tokens must be greater than zero.")

        if previous is not None:
            _validate_previous_scope(
                previous=previous,
                tenant_id=tenant_id,
                subject_id=subject_id,
            )

        previous_strengths = _previous_strengths(previous)
        weight_a, weight_g, weight_i, weight_d = _normalised_ranking_weights(config)

        scored: list[_ScoredCandidate] = []
        goal_filtered_count = 0

        for recall in candidates:
            goal_relevance = calculate_goal_relevance(recall, goal)
            if goal_relevance < config.minimum_goal_relevance:
                goal_filtered_count += 1
                continue

            carryover = _decayed_carryover(
                recall=recall,
                previous_strengths=previous_strengths,
                previous=previous,
                as_of=evaluation_time,
                half_life_seconds=config.decay_half_life_seconds,
            )
            activation = recall.score
            importance = _memory_importance(recall)
            base_priority = (
                weight_a * activation
                + weight_g * goal_relevance
                + weight_i * importance
                + weight_d * carryover
            )
            identity = _identity_from_recall(recall)
            learned_utility = 0.5
            if learning_utilities is not None:
                learned_utility = learning_utilities.get(identity, 0.5)
            utility_signal = 2.0 * learned_utility - 1.0
            utility_adjustment = config.learned_utility_weight * utility_signal
            adjusted_priority = _clamp(
                base_priority + utility_adjustment,
                0.0,
                1.0,
            )
            estimated_tokens = token_estimator.estimate(recall.memory.statement)
            scored.append(
                _ScoredCandidate(
                    recall=recall,
                    goal_relevance=goal_relevance,
                    importance=importance,
                    carryover=carryover,
                    base_priority=base_priority,
                    learned_utility=learned_utility,
                    utility_adjustment=utility_adjustment,
                    adjusted_priority=adjusted_priority,
                    estimated_tokens=estimated_tokens,
                )
            )

        selected: list[_ScoredCandidate] = []
        selected_statements: list[str] = []
        used_tokens = 0
        inhibited_identities: set[MemoryIdentity] = set()
        budget_skipped_count = 0

        while scored and len(selected) < config.max_items:
            for candidate in scored:
                redundancy = _maximum_statement_similarity(
                    candidate.recall.memory.statement,
                    selected_statements,
                )
                inhibition = _calculate_inhibition(redundancy, config)
                candidate.inhibition = inhibition
                if inhibition > 0:
                    inhibited_identities.add(_identity_from_recall(candidate.recall))
                candidate.final_score = _clamp(
                    candidate.adjusted_priority - inhibition,
                    0.0,
                    1.0,
                )

            best = _deterministic_best(scored)
            if best.final_score < config.minimum_selection_score:
                break

            if used_tokens + best.estimated_tokens > budget:
                budget_skipped_count += 1
                scored.remove(best)
                continue

            selected.append(best)
            selected_statements.append(best.recall.memory.statement)
            used_tokens += best.estimated_tokens
            scored.remove(best)

        items = tuple(
            WorkingMemoryItem(
                recall=candidate.recall,
                estimated_tokens=candidate.estimated_tokens,
                transient_strength=candidate.final_score,
                components=WorkingMemoryComponents(
                    activation=candidate.recall.score,
                    goal_relevance=candidate.goal_relevance,
                    importance=candidate.importance,
                    carryover=candidate.carryover,
                    base_priority=candidate.base_priority,
                    learned_utility=candidate.learned_utility,
                    utility_adjustment=candidate.utility_adjustment,
                    adjusted_priority=candidate.adjusted_priority,
                    inhibition=candidate.inhibition,
                    final_score=candidate.final_score,
                ),
                rank=index + 1,
                reason=_selection_reason(candidate),
            )
            for index, candidate in enumerate(selected)
        )

        return WorkingMemorySnapshot(
            tenant_id=tenant_id,
            subject_id=subject_id,
            goal=goal,
            items=items,
            created_at=evaluation_time,
            candidate_count=len(candidates),
            selected_count=len(items),
            estimated_prompt_tokens=used_tokens,
            prompt_budget_tokens=budget,
            goal_filtered_count=goal_filtered_count,
            inhibited_count=len(inhibited_identities),
            budget_skipped_count=budget_skipped_count,
        )


calculate_goal_relevance = calculate_cue_relevance


def _validate_previous_scope(
    *,
    previous: WorkingMemorySnapshot,
    tenant_id: str,
    subject_id: str | None,
) -> None:
    if previous.tenant_id != tenant_id:
        raise ValidationError("previous working-memory tenant_id must match tenant_id.")
    if previous.subject_id != subject_id:
        raise ValidationError("previous working-memory subject_id must match subject_id.")


def _previous_strengths(
    previous: WorkingMemorySnapshot | None,
) -> dict[MemoryIdentity, float]:
    if previous is None:
        return {}
    return {item.identity: item.transient_strength for item in previous.items}


def _decayed_carryover(
    *,
    recall: RecallResult,
    previous_strengths: Mapping[MemoryIdentity, float],
    previous: WorkingMemorySnapshot | None,
    as_of: datetime,
    half_life_seconds: float,
) -> float:
    if previous is None:
        return 0.0
    identity = _identity_from_recall(recall)
    previous_strength = previous_strengths.get(identity)
    if previous_strength is None:
        return 0.0
    elapsed_seconds = (as_of - previous.created_at).total_seconds()
    if elapsed_seconds < 0:
        elapsed_seconds = 0.0
    decay_factor = math.pow(2.0, -elapsed_seconds / half_life_seconds)
    return _clamp(previous_strength * decay_factor, 0.0, 1.0)


def _normalised_ranking_weights(config: WorkingMemoryConfig) -> tuple[float, float, float, float]:
    weights = (
        config.activation_weight,
        config.goal_relevance_weight,
        config.importance_weight,
        config.carryover_weight,
    )
    positive_total = sum(weight for weight in weights if weight > 0)
    normalised = tuple(weight / positive_total if weight > 0 else 0.0 for weight in weights)
    return normalised[0], normalised[1], normalised[2], normalised[3]


def _memory_importance(recall: RecallResult) -> float:
    return recall.memory.importance


def _identity_from_recall(recall: RecallResult) -> MemoryIdentity:
    memory = recall.memory
    if isinstance(memory, StoredEpisode):
        return MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key=memory.memory_key)
    return MemoryIdentity(memory_kind=MemoryKind.SEMANTIC, memory_key=memory.memory_key)


def _memory_key_from_recall(recall: RecallResult) -> str:
    memory = recall.memory
    if isinstance(memory, StoredEpisode):
        return memory.memory_key
    return memory.memory_key


def _maximum_statement_similarity(statement: str, selected_statements: Sequence[str]) -> float:
    if not selected_statements:
        return 0.0
    return max(_statement_similarity(statement, selected) for selected in selected_statements)


def _statement_similarity(left: str, right: str) -> float:
    left_tokens = _tokenize(left)
    right_tokens = _tokenize(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = len(left_tokens.intersection(right_tokens))
    union = len(left_tokens.union(right_tokens))
    return intersection / union


def _calculate_inhibition(redundancy: float, config: WorkingMemoryConfig) -> float:
    if redundancy < config.redundancy_threshold:
        return 0.0
    return config.inhibition_strength * redundancy


def _deterministic_best(scored: Sequence[_ScoredCandidate]) -> _ScoredCandidate:
    return max(scored, key=_candidate_sort_key)


def _candidate_sort_key(candidate: _ScoredCandidate) -> tuple[float, float, float, str, str]:
    return (
        candidate.final_score,
        candidate.adjusted_priority,
        candidate.recall.activation,
        candidate.recall.memory_kind.value,
        _memory_key_from_recall(candidate.recall),
    )


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _selection_reason(candidate: _ScoredCandidate) -> str:
    return (
        f"selected with final_score={candidate.final_score:.4f} "
        f"(activation={candidate.recall.score:.4f}, "
        f"goal_relevance={candidate.goal_relevance:.4f}, "
        f"importance={candidate.importance:.4f}, "
        f"carryover={candidate.carryover:.4f}, "
        f"inhibition={candidate.inhibition:.4f})"
    )
