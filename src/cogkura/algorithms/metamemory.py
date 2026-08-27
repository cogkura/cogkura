"""Deterministic metamemory assessment over recalled memories."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol

from cogkura.algorithms.activation import (
    TemporalRetrievalMode,
    _cue_requests_current_state,
    _seed_entity_ids_from_text,
    _stored_semantic_matches_cue,
    _temporal_retrieval_mode,
    _tokenize,
    activation_candidate_from_episode,
    activation_candidate_from_semantic,
)
from cogkura.algorithms.forgetting import retention_score_from_base_level
from cogkura.algorithms.learning import learning_context_key, learning_counts_by_identity
from cogkura.algorithms.relevance import calculate_cue_coverage, calculate_cue_relevance
from cogkura.models import (
    ActivationConfig,
    MemoryAssessment,
    MemoryAssessmentFlag,
    MemoryIdentity,
    MemoryKind,
    MetamemoryConfig,
    MetamemoryItem,
    MetamemorySignals,
    RecallResult,
    RetrievalCue,
    SemanticMemoryStatus,
    StoredEpisode,
    StoredMemoryLearningState,
    StoredSemanticMemory,
)

_FLAG_ORDER: tuple[MemoryAssessmentFlag, ...] = (
    MemoryAssessmentFlag.NO_RETRIEVED_MEMORY,
    MemoryAssessmentFlag.MISSING_KNOWLEDGE,
    MemoryAssessmentFlag.LOW_CUE_COVERAGE,
    MemoryAssessmentFlag.LOW_RETRIEVAL_STRENGTH,
    MemoryAssessmentFlag.LOW_EVIDENCE_CONFIDENCE,
    MemoryAssessmentFlag.CONFLICTING_SEMANTIC_MEMORY,
    MemoryAssessmentFlag.LOW_PROVENANCE_DIVERSITY,
    MemoryAssessmentFlag.HIGH_FORGETTING_PRESSURE,
    MemoryAssessmentFlag.LOW_LEARNED_UTILITY,
    MemoryAssessmentFlag.STALE_EVIDENCE,
)


class MemoryAnswerability(StrEnum):
    """Whether recalled memory contains a resolving semantic assertion."""

    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    NOT_APPLICABLE = "not_applicable"


class MemoryMonitor(Protocol):
    """Protocol for deterministic read-only memory assessment."""

    def assess(
        self,
        *,
        candidates: Sequence[RecallResult],
        query: RetrievalCue,
        goal: RetrievalCue,
        tenant_id: str,
        subject_id: str | None,
        as_of: datetime,
        valid_at: datetime | None,
        config: MetamemoryConfig,
        activation_config: ActivationConfig,
        learning_utilities: Mapping[MemoryIdentity, float] | None = None,
        learning_states: Sequence[StoredMemoryLearningState] = (),
    ) -> MemoryAssessment:
        """Assess retrieved memory state without storage I/O."""
        ...


@dataclass(slots=True)
class DeterministicMemoryMonitor:
    """Pure deterministic metamemory monitor over recall candidates."""

    def assess(
        self,
        *,
        candidates: Sequence[RecallResult],
        query: RetrievalCue,
        goal: RetrievalCue,
        tenant_id: str,
        subject_id: str | None,
        as_of: datetime,
        valid_at: datetime | None,
        config: MetamemoryConfig,
        activation_config: ActivationConfig,
        learning_utilities: Mapping[MemoryIdentity, float] | None = None,
        learning_states: Sequence[StoredMemoryLearningState] = (),
    ) -> MemoryAssessment:
        evaluation_time = as_of.astimezone(UTC)
        context_key = learning_context_key(goal)

        if not candidates:
            return MemoryAssessment(
                tenant_id=tenant_id,
                subject_id=subject_id,
                query=query,
                goal=goal,
                assessed_at=evaluation_time,
                valid_at=valid_at,
                signals=MetamemorySignals(
                    cue_coverage=0.0,
                    top_retrieval_strength=0.0,
                    mean_retrieval_strength=0.0,
                    evidence_confidence=None,
                    semantic_conflict=0.0,
                    provenance_diversity=0.0,
                    forgetting_pressure=None,
                    learned_utility=None,
                    freshness=None,
                ),
                flags=(MemoryAssessmentFlag.NO_RETRIEVED_MEMORY,),
                items=(),
                retrieved_count=0,
                episode_count=0,
                semantic_count=0,
                contested_count=0,
                historical_revision_count=0,
                distinct_observation_count=0,
                helpful_feedback_count=0,
                unhelpful_feedback_count=0,
                incorrect_feedback_count=0,
                newest_evidence_at=None,
                oldest_evidence_at=None,
            )

        item_diagnostics = [
            _build_item_diagnostics(
                recall=recall,
                goal=goal,
                as_of=evaluation_time,
                activation_config=activation_config,
                learning_utilities=learning_utilities,
            )
            for recall in candidates
        ]

        scores = [recall.score for recall in candidates]
        top_retrieval_strength = max(scores)
        mean_retrieval_strength = sum(scores) / len(scores)
        cue_coverage = calculate_cue_coverage(candidates, query)

        evidence_confidence = _retrieval_weighted_mean(
            [
                (recall.score, item.evidence_confidence)
                for recall, item in zip(candidates, item_diagnostics, strict=True)
            ]
        )

        semantic_conflict = 0.0
        for recall, item in zip(candidates, item_diagnostics, strict=True):
            if recall.memory_kind is MemoryKind.SEMANTIC:
                semantic_conflict = max(semantic_conflict, recall.score * item.semantic_conflict)

        observation_ids: set[str] = set()
        for recall in candidates:
            observation_ids.update(_observation_ids(recall.memory))
        distinct_observation_count = len(observation_ids)
        provenance_diversity = _provenance_diversity_score(
            distinct_observation_count,
            tau=config.provenance_diversity_tau,
        )

        forgetting_pressure = max(
            recall.score * item.forgetting_pressure
            for recall, item in zip(candidates, item_diagnostics, strict=True)
        )

        learned_utility: float | None = None
        if learning_utilities is not None:
            utility_pairs = [
                (recall.score, learning_utilities.get(_identity_from_recall(recall), 0.5))
                for recall in candidates
            ]
            learned_utility = _retrieval_weighted_mean(utility_pairs)

        freshness: float | None = None
        if config.freshness_half_life_seconds is not None:
            freshness_pairs = [
                (
                    recall.score,
                    _freshness_score(item.evidence_age_seconds, config.freshness_half_life_seconds),
                )
                for recall, item in zip(candidates, item_diagnostics, strict=True)
            ]
            freshness = _retrieval_weighted_mean(freshness_pairs)

        signals = MetamemorySignals(
            cue_coverage=cue_coverage,
            top_retrieval_strength=top_retrieval_strength,
            mean_retrieval_strength=mean_retrieval_strength,
            evidence_confidence=evidence_confidence,
            semantic_conflict=semantic_conflict,
            provenance_diversity=provenance_diversity,
            forgetting_pressure=forgetting_pressure,
            learned_utility=learned_utility,
            freshness=freshness,
        )

        flags = _build_flags(
            signals=signals,
            config=config,
            has_candidates=True,
            candidates=candidates,
            query=query,
            activation_config=activation_config,
            valid_at=valid_at,
        )

        report_limit = min(config.max_report_items, len(candidates))
        items = tuple(item_diagnostics[:report_limit])

        episode_count = sum(1 for recall in candidates if recall.memory_kind is MemoryKind.EPISODE)
        semantic_count = len(candidates) - episode_count
        contested_count = sum(
            1
            for recall in candidates
            if recall.memory_kind is MemoryKind.SEMANTIC
            and isinstance(recall.memory, StoredSemanticMemory)
            and recall.memory.status is SemanticMemoryStatus.CONTESTED
        )
        historical_revision_count = sum(
            1
            for recall in candidates
            if recall.memory_kind is MemoryKind.SEMANTIC
            and isinstance(recall.memory, StoredSemanticMemory)
            and recall.memory.status is SemanticMemoryStatus.SUPERSEDED
        )

        identities = [_identity_from_recall(recall) for recall in candidates]
        counts_by_identity = learning_counts_by_identity(
            identities=identities,
            states=learning_states,
            context_key=context_key,
        )
        helpful_feedback_count = 0
        unhelpful_feedback_count = 0
        incorrect_feedback_count = 0
        for identity in identities:
            counts = counts_by_identity[identity]
            helpful_feedback_count += counts.helpful
            unhelpful_feedback_count += counts.unhelpful
            incorrect_feedback_count += counts.incorrect

        evidence_times = [item.evidence_at for item in item_diagnostics]
        newest_evidence_at = max(evidence_times)
        oldest_evidence_at = min(evidence_times)

        return MemoryAssessment(
            tenant_id=tenant_id,
            subject_id=subject_id,
            query=query,
            goal=goal,
            assessed_at=evaluation_time,
            valid_at=valid_at,
            signals=signals,
            flags=flags,
            items=items,
            retrieved_count=len(candidates),
            episode_count=episode_count,
            semantic_count=semantic_count,
            contested_count=contested_count,
            historical_revision_count=historical_revision_count,
            distinct_observation_count=distinct_observation_count,
            helpful_feedback_count=helpful_feedback_count,
            unhelpful_feedback_count=unhelpful_feedback_count,
            incorrect_feedback_count=incorrect_feedback_count,
            newest_evidence_at=newest_evidence_at,
            oldest_evidence_at=oldest_evidence_at,
        )


def _build_item_diagnostics(
    *,
    recall: RecallResult,
    goal: RetrievalCue,
    as_of: datetime,
    activation_config: ActivationConfig,
    learning_utilities: Mapping[MemoryIdentity, float] | None,
) -> MetamemoryItem:
    memory = recall.memory
    cue_relevance = calculate_cue_relevance(recall, goal)
    evidence_confidence = memory.confidence
    semantic_conflict = _item_semantic_conflict(memory)
    observation_ids = _observation_ids(memory)
    evidence_at = _evidence_at(memory)
    evidence_age_seconds = max(0.0, (as_of - evidence_at).total_seconds())
    retention_score = retention_score_from_base_level(
        recall.components.base_level,
        retrieval_threshold=activation_config.retrieval_threshold,
    )
    forgetting_pressure = 1.0 - retention_score

    learned_utility: float | None = None
    if learning_utilities is not None:
        learned_utility = learning_utilities.get(_identity_from_recall(recall), 0.5)

    return MetamemoryItem(
        recall=recall,
        cue_relevance=cue_relevance,
        evidence_confidence=evidence_confidence,
        semantic_conflict=semantic_conflict,
        observation_count=len(observation_ids),
        evidence_at=evidence_at,
        evidence_age_seconds=evidence_age_seconds,
        retention_score=retention_score,
        forgetting_pressure=forgetting_pressure,
        learned_utility=learned_utility,
    )


def _item_semantic_conflict(memory: StoredEpisode | StoredSemanticMemory) -> float:
    if isinstance(memory, StoredEpisode):
        return 0.0
    if memory.status is SemanticMemoryStatus.CONTESTED:
        return 1.0
    if memory.status is SemanticMemoryStatus.SUPERSEDED:
        return 0.0
    total = memory.support_count + memory.contradiction_count
    if total == 0:
        return 0.0
    return memory.contradiction_count / total


def _observation_ids(memory: StoredEpisode | StoredSemanticMemory) -> set[str]:
    if isinstance(memory, StoredEpisode):
        return {evidence.observation_id for evidence in memory.evidence}
    return {evidence.observation_id for evidence in memory.observation_evidence}


def _evidence_at(memory: StoredEpisode | StoredSemanticMemory) -> datetime:
    if isinstance(memory, StoredEpisode):
        return memory.ended_at.astimezone(UTC)
    return memory.last_supported_at.astimezone(UTC)


def _provenance_diversity_score(distinct_count: int, *, tau: float) -> float:
    return 1.0 - math.exp(-max(0, distinct_count - 1) / tau)


def _freshness_score(age_seconds: float, half_life_seconds: float) -> float:
    return math.pow(2.0, -age_seconds / half_life_seconds)


def _retrieval_weighted_mean(pairs: Sequence[tuple[float, float]]) -> float:
    if not pairs:
        return 0.0
    total_weight = sum(score for score, _ in pairs)
    if total_weight > 0:
        return sum(score * value for score, value in pairs) / total_weight
    return sum(value for _, value in pairs) / len(pairs)


def _build_flags(
    *,
    signals: MetamemorySignals,
    config: MetamemoryConfig,
    has_candidates: bool,
    candidates: Sequence[RecallResult] = (),
    query: RetrievalCue | None = None,
    activation_config: ActivationConfig | None = None,
    valid_at: datetime | None = None,
) -> tuple[MemoryAssessmentFlag, ...]:
    if not has_candidates:
        return (MemoryAssessmentFlag.NO_RETRIEVED_MEMORY,)

    active: set[MemoryAssessmentFlag] = set()
    if query is not None and activation_config is not None:
        answerability = _assess_answerability(
            candidates,
            query,
            valid_at=valid_at,
            activation_config=activation_config,
        )
        if answerability is MemoryAnswerability.UNRESOLVED:
            active.add(MemoryAssessmentFlag.MISSING_KNOWLEDGE)
        elif answerability is MemoryAnswerability.NOT_APPLICABLE and (
            not _has_active_semantic_slot_match(
                candidates,
                query,
                activation_config,
            )
            and (
                signals.cue_coverage < config.missing_knowledge_coverage_threshold
                or signals.top_retrieval_strength < config.missing_knowledge_strength_threshold
            )
        ):
            active.add(MemoryAssessmentFlag.MISSING_KNOWLEDGE)
    if signals.cue_coverage < config.low_cue_coverage_threshold:
        active.add(MemoryAssessmentFlag.LOW_CUE_COVERAGE)
    if signals.top_retrieval_strength < config.low_retrieval_strength_threshold:
        active.add(MemoryAssessmentFlag.LOW_RETRIEVAL_STRENGTH)
    if signals.evidence_confidence is not None and (
        signals.evidence_confidence < config.low_evidence_confidence_threshold
    ):
        active.add(MemoryAssessmentFlag.LOW_EVIDENCE_CONFIDENCE)
    if signals.semantic_conflict >= config.semantic_conflict_threshold:
        active.add(MemoryAssessmentFlag.CONFLICTING_SEMANTIC_MEMORY)
    if signals.provenance_diversity < config.low_provenance_diversity_threshold:
        active.add(MemoryAssessmentFlag.LOW_PROVENANCE_DIVERSITY)
    if signals.forgetting_pressure is not None and (
        signals.forgetting_pressure >= config.forgetting_pressure_threshold
    ):
        active.add(MemoryAssessmentFlag.HIGH_FORGETTING_PRESSURE)
    if signals.learned_utility is not None and (
        signals.learned_utility < config.low_learned_utility_threshold
    ):
        active.add(MemoryAssessmentFlag.LOW_LEARNED_UTILITY)
    if signals.freshness is not None and signals.freshness < config.stale_evidence_threshold:
        active.add(MemoryAssessmentFlag.STALE_EVIDENCE)

    return tuple(flag for flag in _FLAG_ORDER if flag in active)


def _assess_answerability(
    candidates: Sequence[RecallResult],
    query: RetrievalCue,
    *,
    valid_at: datetime | None,
    activation_config: ActivationConfig,
) -> MemoryAnswerability:
    temporal_mode = _temporal_retrieval_mode(
        query,
        valid_at=valid_at,
        config=activation_config,
    )
    slot_like = (
        query.predicate is not None
        or query.object_value is not None
        or temporal_mode is TemporalRetrievalMode.CURRENT
        or temporal_mode is TemporalRetrievalMode.HISTORICAL
    )
    if not slot_like:
        return MemoryAnswerability.NOT_APPLICABLE
    if _has_resolving_semantic_assertion(
        candidates,
        query,
        activation_config=activation_config,
        temporal_mode=temporal_mode,
    ):
        return MemoryAnswerability.RESOLVED
    return MemoryAnswerability.UNRESOLVED


def _has_resolving_semantic_assertion(
    candidates: Sequence[RecallResult],
    query: RetrievalCue,
    *,
    activation_config: ActivationConfig,
    temporal_mode: TemporalRetrievalMode,
) -> bool:
    effective_sources, current_state_cue, distinctive_tokens = _metamemory_match_context(
        candidates,
        query,
        activation_config=activation_config,
    )
    for recall in candidates:
        if recall.memory_kind is not MemoryKind.SEMANTIC:
            continue
        memory = recall.memory
        if not isinstance(memory, StoredSemanticMemory):
            continue
        if not (memory.object_value and memory.object_value.strip()):
            continue
        if temporal_mode is TemporalRetrievalMode.CURRENT:
            if memory.status is not SemanticMemoryStatus.ACTIVE:
                continue
        if _stored_semantic_matches_cue(
            memory,
            query,
            effective_sources=effective_sources,
            current_state_cue=current_state_cue,
            distinctive_tokens=distinctive_tokens,
            config=activation_config,
        ):
            return True
    return False


def _has_active_semantic_slot_match(
    candidates: Sequence[RecallResult],
    query: RetrievalCue,
    activation_config: ActivationConfig,
) -> bool:
    effective_sources, current_state_cue, distinctive_tokens = _metamemory_match_context(
        candidates,
        query,
        activation_config=activation_config,
    )
    for recall in candidates:
        if recall.memory_kind is not MemoryKind.SEMANTIC:
            continue
        memory = recall.memory
        if not isinstance(memory, StoredSemanticMemory):
            continue
        if memory.status is not SemanticMemoryStatus.ACTIVE:
            continue
        if _stored_semantic_matches_cue(
            memory,
            query,
            effective_sources=effective_sources,
            current_state_cue=current_state_cue,
            distinctive_tokens=distinctive_tokens,
            config=activation_config,
        ):
            return True
    return False


def _metamemory_match_context(
    candidates: Sequence[RecallResult],
    query: RetrievalCue,
    *,
    activation_config: ActivationConfig,
) -> tuple[tuple[str, ...], bool, set[str]]:
    activation_candidates = []
    for recall in candidates:
        memory = recall.memory
        if isinstance(memory, StoredEpisode):
            activation_candidates.append(activation_candidate_from_episode(memory))
        elif isinstance(memory, StoredSemanticMemory):
            activation_candidates.append(activation_candidate_from_semantic(memory))
    seeded_entity_ids = _seed_entity_ids_from_text(query, activation_candidates, activation_config)
    effective_sources = query.entity_ids if query.entity_ids else seeded_entity_ids
    current_state_cue = _cue_requests_current_state(query, activation_config)
    cue_tokens = _tokenize(query.text) if query.text else set()
    distinctive_tokens = cue_tokens - activation_config.current_state_cue_tokens
    return effective_sources, current_state_cue, distinctive_tokens


def _identity_from_recall(recall: RecallResult) -> MemoryIdentity:
    memory = recall.memory
    if isinstance(memory, StoredEpisode):
        return MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key=memory.memory_key)
    return MemoryIdentity(memory_kind=MemoryKind.SEMANTIC, memory_key=memory.memory_key)
