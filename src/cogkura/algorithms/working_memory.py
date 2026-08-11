"""Bounded working-memory selection from declarative recall candidates."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from cogkura.exceptions import ValidationError
from cogkura.models import (
    MemoryIdentity,
    MemoryKind,
    RecallResult,
    RetrievalCue,
    StoredEpisode,
    StoredSemanticMemory,
    WorkingMemoryComponents,
    WorkingMemoryConfig,
    WorkingMemoryItem,
    WorkingMemorySnapshot,
)

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
_WHITESPACE_PATTERN = re.compile(r"\s+")


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
            estimated_tokens = token_estimator.estimate(recall.memory.statement)
            scored.append(
                _ScoredCandidate(
                    recall=recall,
                    goal_relevance=goal_relevance,
                    importance=importance,
                    carryover=carryover,
                    base_priority=base_priority,
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
                    candidate.base_priority - inhibition,
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


def calculate_goal_relevance(recall: RecallResult, goal: RetrievalCue) -> float:
    """Compute mean goal-relevance across supplied goal cue fields."""
    memory = recall.memory
    text, subject_id, entity_ids, predicate, object_value, qualifiers = _memory_cue_fields(memory)
    components: list[float] = []

    if goal.text and goal.text.strip():
        components.append(_text_coverage(goal.text, text))

    if goal.subject_id and goal.subject_id.strip():
        goal_subject = goal.subject_id.strip()
        candidate_subject = subject_id.strip() if subject_id else None
        components.append(1.0 if goal_subject == candidate_subject else 0.0)

    if goal.entity_ids:
        goal_entities = set(goal.entity_ids)
        candidate_entities = set(entity_ids)
        matched = len(goal_entities.intersection(candidate_entities))
        components.append(matched / len(goal_entities))

    if goal.predicate and goal.predicate.strip():
        goal_predicate = _normalise_text(goal.predicate)
        candidate_predicate = _normalise_text(predicate) if predicate else ""
        components.append(1.0 if goal_predicate == candidate_predicate else 0.0)

    if goal.object_value and goal.object_value.strip():
        goal_object = goal.object_value
        if object_value and _normalise_text(goal_object) == _normalise_text(object_value):
            components.append(1.0)
        else:
            components.append(_text_coverage(goal_object, text))

    if goal.qualifiers:
        components.append(_qualifier_coverage(goal.qualifiers, qualifiers))

    if not components:
        return 1.0

    return sum(components) / len(components)


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


def _memory_cue_fields(
    memory: StoredEpisode | StoredSemanticMemory,
) -> tuple[str, str | None, tuple[str, ...], str | None, str | None, Mapping[str, object]]:
    if isinstance(memory, StoredEpisode):
        entity_ids = tuple(sorted({entity.entity_id for entity in memory.entities}))
        return (
            memory.statement,
            memory.subject_id,
            entity_ids,
            None,
            None,
            memory.metadata,
        )
    entity_id_set = {entity.entity_id for entity in memory.entities}
    if memory.subject_entity_id:
        entity_id_set.add(memory.subject_entity_id)
    if memory.object_entity_id:
        entity_id_set.add(memory.object_entity_id)
    return (
        memory.statement,
        memory.subject_id,
        tuple(sorted(entity_id_set)),
        memory.predicate,
        memory.object_value,
        memory.qualifiers,
    )


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
        candidate.base_priority,
        candidate.recall.activation,
        candidate.recall.memory_kind.value,
        _memory_key_from_recall(candidate.recall),
    )


def _text_coverage(goal_text: str, candidate_text: str) -> float:
    goal_tokens = _tokenize(goal_text)
    if not goal_tokens:
        return 0.0
    candidate_tokens = _tokenize(candidate_text)
    if not candidate_tokens:
        return 0.0
    matched = goal_tokens.intersection(candidate_tokens)
    return len(matched) / len(goal_tokens)


def _qualifier_coverage(
    goal_qualifiers: Mapping[str, object],
    candidate_qualifiers: Mapping[str, object],
) -> float:
    goal_pairs = {_qualifier_pair(key, value) for key, value in goal_qualifiers.items()}
    if not goal_pairs:
        return 1.0
    candidate_pairs = {_qualifier_pair(key, value) for key, value in candidate_qualifiers.items()}
    matched = len(goal_pairs.intersection(candidate_pairs))
    return matched / len(goal_pairs)


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_PATTERN.findall(text)}


def _normalise_text(value: str) -> str:
    normalised = unicodedata.normalize("NFKC", value)
    normalised = _WHITESPACE_PATTERN.sub(" ", normalised).strip()
    return normalised.casefold()


def _qualifier_pair(key: object, value: object) -> tuple[str, str]:
    return (_normalise_text(str(key)), _normalise_text(str(value)))


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
