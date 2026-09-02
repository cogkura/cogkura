"""Bounded working-memory selection from declarative recall candidates."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Protocol

from cogkura.algorithms.activation import (
    _collapse_results,
    build_episode_slot_index_from_results,
)
from cogkura.algorithms.relevance import calculate_cue_relevance
from cogkura.algorithms.relevance import tokenize as _tokenize
from cogkura.exceptions import ValidationError
from cogkura.models import (
    ActivationConfig,
    MemoryIdentity,
    MemoryKind,
    RecallResult,
    RelevanceTier,
    RetrievalCue,
    SemanticCardinality,
    SemanticDerivationRelation,
    SemanticMemoryStatus,
    StoredEpisode,
    StoredSemanticMemory,
    WorkingMemoryChunk,
    WorkingMemoryChunkType,
    WorkingMemoryComponents,
    WorkingMemoryConfig,
    WorkingMemoryItem,
    WorkingMemoryRejectionReason,
    WorkingMemorySnapshot,
)

_RELEVANCE_TIER_RANK = {
    RelevanceTier.DIRECT_VALUE: 6,
    RelevanceTier.DIRECT_SEMANTIC: 5,
    RelevanceTier.ENTITY_ASSOCIATION: 4,
    RelevanceTier.STRUCTURED_RELATION: 3,
    RelevanceTier.EVIDENCE_ASSOCIATION: 2,
    RelevanceTier.CONTEXTUAL: 1,
}


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
        activation_config: ActivationConfig | None = None,
        episode_slot_index: Mapping[str, str] | None = None,
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


@dataclass(slots=True)
class _ChunkCandidate:
    chunk_type: WorkingMemoryChunkType
    coverage_key: str
    members: tuple[_ScoredCandidate, ...]
    primary: _ScoredCandidate
    relevance_tier: str
    activation: float
    goal_relevance: float
    importance: float
    carryover: float
    base_priority: float
    learned_utility: float
    utility_adjustment: float
    adjusted_priority: float
    serialized_text: str
    estimated_tokens: int
    included_members: tuple[_ScoredCandidate, ...]
    members_omitted: int = 0
    final_score: float = 0.0
    inhibition: float = 0.0
    novelty: float = 1.0
    rejection_reason: str | None = None


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
        activation_config: ActivationConfig | None = None,
        episode_slot_index: Mapping[str, str] | None = None,
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

        pool = _prepare_pool(
            candidates,
            config=config,
            activation_config=activation_config,
            episode_slot_index=episode_slot_index,
        )
        previous_strengths = _previous_strengths(previous)
        scored, goal_filtered_count = _score_candidates(
            pool,
            goal=goal,
            previous=previous,
            previous_strengths=previous_strengths,
            as_of=evaluation_time,
            config=config,
            token_estimator=token_estimator,
            learning_utilities=learning_utilities,
        )

        if config.enable_chunking:
            return self._select_chunks(
                scored=scored,
                goal=goal,
                tenant_id=tenant_id,
                subject_id=subject_id,
                evaluation_time=evaluation_time,
                config=config,
                token_estimator=token_estimator,
                budget=budget,
                candidate_count=len(candidates),
                goal_filtered_count=goal_filtered_count,
            )

        return self._select_items(
            scored=scored,
            goal=goal,
            tenant_id=tenant_id,
            subject_id=subject_id,
            evaluation_time=evaluation_time,
            config=config,
            budget=budget,
            candidate_count=len(candidates),
            goal_filtered_count=goal_filtered_count,
        )

    def _select_items(
        self,
        *,
        scored: list[_ScoredCandidate],
        goal: RetrievalCue,
        tenant_id: str,
        subject_id: str | None,
        evaluation_time: datetime,
        config: WorkingMemoryConfig,
        budget: int,
        candidate_count: int,
        goal_filtered_count: int,
    ) -> WorkingMemorySnapshot:
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
                components=_components_from_candidate(candidate),
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
            candidate_count=candidate_count,
            selected_count=len(items),
            estimated_prompt_tokens=used_tokens,
            prompt_budget_tokens=budget,
            goal_filtered_count=goal_filtered_count,
            inhibited_count=len(inhibited_identities),
            budget_skipped_count=budget_skipped_count,
        )

    def _select_chunks(
        self,
        *,
        scored: list[_ScoredCandidate],
        goal: RetrievalCue,
        tenant_id: str,
        subject_id: str | None,
        evaluation_time: datetime,
        config: WorkingMemoryConfig,
        token_estimator: TokenEstimator,
        budget: int,
        candidate_count: int,
        goal_filtered_count: int,
    ) -> WorkingMemorySnapshot:
        chunk_candidates = _form_chunks(scored, token_estimator=token_estimator)
        selected_chunks: list[_ChunkCandidate] = []
        selected_texts: list[str] = []
        selected_coverage: set[str] = set()
        used_tokens = 0
        inhibited_identities: set[MemoryIdentity] = set()
        budget_skipped_count = 0

        pending = list(chunk_candidates)
        while pending and len(selected_chunks) < config.max_items:
            for chunk in pending:
                novelty = 0.0 if chunk.coverage_key in selected_coverage else 1.0
                chunk.novelty = novelty
                redundancy = _maximum_statement_similarity(
                    chunk.serialized_text,
                    selected_texts,
                )
                inhibition = _calculate_inhibition(redundancy, config)
                chunk.inhibition = inhibition
                if inhibition > 0:
                    for member in chunk.included_members:
                        inhibited_identities.add(_identity_from_recall(member.recall))
                coverage_penalty = (
                    config.inhibition_strength if chunk.coverage_key in selected_coverage else 0.0
                )
                chunk.final_score = _clamp(
                    chunk.adjusted_priority + (novelty * 0.05) - inhibition - coverage_penalty,
                    0.0,
                    1.0,
                )

            best = _deterministic_best_chunk(pending)
            if best.final_score < config.minimum_selection_score:
                for chunk in pending:
                    if chunk.rejection_reason is None:
                        chunk.rejection_reason = WorkingMemoryRejectionReason.LOW_UTILITY.value
                break

            fit_chunk, fit_tokens = _trim_chunk_for_budget(
                best,
                remaining_tokens=budget - used_tokens,
                token_estimator=token_estimator,
            )
            if fit_chunk is None:
                best.rejection_reason = WorkingMemoryRejectionReason.TOKEN_BUDGET.value
                budget_skipped_count += 1
                pending.remove(best)
                continue

            selected_chunks.append(fit_chunk)
            selected_texts.append(fit_chunk.serialized_text)
            selected_coverage.add(fit_chunk.coverage_key)
            used_tokens += fit_tokens
            pending.remove(best)

        for chunk in pending:
            if chunk.rejection_reason is None:
                chunk.rejection_reason = WorkingMemoryRejectionReason.CHUNK_CAPACITY.value

        all_chunks = tuple(
            _chunk_from_candidate(chunk, selected=False) for chunk in chunk_candidates
        )
        selected_chunk_models: list[WorkingMemoryChunk] = []
        items: list[WorkingMemoryItem] = []

        for index, chunk in enumerate(selected_chunks):
            chunk_model = _chunk_from_candidate(chunk, selected=True)
            selected_chunk_models.append(chunk_model)
            primary = chunk.primary
            items.append(
                WorkingMemoryItem(
                    recall=primary.recall,
                    estimated_tokens=chunk.estimated_tokens,
                    transient_strength=chunk.final_score,
                    components=WorkingMemoryComponents(
                        activation=chunk.activation,
                        goal_relevance=chunk.goal_relevance,
                        importance=chunk.importance,
                        carryover=chunk.carryover,
                        base_priority=chunk.base_priority,
                        learned_utility=chunk.learned_utility,
                        utility_adjustment=chunk.utility_adjustment,
                        adjusted_priority=chunk.adjusted_priority,
                        inhibition=chunk.inhibition,
                        final_score=chunk.final_score,
                    ),
                    rank=index + 1,
                    reason=_chunk_selection_reason(chunk),
                    chunk=chunk_model,
                    member_recalls=tuple(member.recall for member in chunk.included_members),
                )
            )

        final_chunks = _merge_chunk_states(all_chunks, selected_chunk_models)
        return WorkingMemorySnapshot(
            tenant_id=tenant_id,
            subject_id=subject_id,
            goal=goal,
            items=tuple(items),
            created_at=evaluation_time,
            candidate_count=candidate_count,
            selected_count=len(items),
            estimated_prompt_tokens=used_tokens,
            prompt_budget_tokens=budget,
            goal_filtered_count=goal_filtered_count,
            inhibited_count=len(inhibited_identities),
            budget_skipped_count=budget_skipped_count,
            candidate_chunk_count=len(chunk_candidates),
            selected_chunk_count=len(items),
            chunks=final_chunks,
        )


calculate_goal_relevance = calculate_cue_relevance


def _prepare_pool(
    candidates: Sequence[RecallResult],
    *,
    config: WorkingMemoryConfig,
    activation_config: ActivationConfig | None,
    episode_slot_index: Mapping[str, str] | None,
) -> list[RecallResult]:
    pool = list(candidates)
    if config.collapse_same_slot_support:
        base_config = activation_config or ActivationConfig()
        collapse_config = replace(
            base_config,
            enable_duplicate_collapse=False,
            collapse_same_slot_support=True,
        )
        slot_index = episode_slot_index or build_episode_slot_index_from_results(pool)
        pool = _collapse_results(
            sorted(
                pool,
                key=lambda item: (
                    -item.activation,
                    item.memory_kind.value,
                    _memory_key_from_recall(item),
                ),
            ),
            limit=len(pool),
            config=collapse_config,
            support_index={},
            slot_index=slot_index,
        )
    return pool


def _score_candidates(
    pool: Sequence[RecallResult],
    *,
    goal: RetrievalCue,
    previous: WorkingMemorySnapshot | None,
    previous_strengths: Mapping[MemoryIdentity, float],
    as_of: datetime,
    config: WorkingMemoryConfig,
    token_estimator: TokenEstimator,
    learning_utilities: Mapping[MemoryIdentity, float] | None,
) -> tuple[list[_ScoredCandidate], int]:
    weight_a, weight_g, weight_i, weight_d = _normalised_ranking_weights(config)
    scored: list[_ScoredCandidate] = []
    goal_filtered_count = 0

    for recall in pool:
        goal_relevance = calculate_goal_relevance(recall, goal)
        if goal_relevance < config.minimum_goal_relevance:
            goal_filtered_count += 1
            continue

        carryover = _decayed_carryover(
            recall=recall,
            previous_strengths=previous_strengths,
            previous=previous,
            as_of=as_of,
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
        stale_penalty = _stale_goal_penalty(recall, goal, config)
        adjusted_priority = _clamp(
            base_priority + utility_adjustment - stale_penalty,
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
    return scored, goal_filtered_count


def _form_chunks(
    scored: Sequence[_ScoredCandidate],
    *,
    token_estimator: TokenEstimator,
) -> list[_ChunkCandidate]:
    episode_by_id = {
        candidate.recall.memory.id: candidate
        for candidate in scored
        if candidate.recall.memory_kind is MemoryKind.EPISODE
        and isinstance(candidate.recall.memory, StoredEpisode)
    }
    assigned: set[MemoryIdentity] = set()
    chunks: list[_ChunkCandidate] = []

    semantics = sorted(
        (
            candidate
            for candidate in scored
            if candidate.recall.memory_kind is MemoryKind.SEMANTIC
            and isinstance(candidate.recall.memory, StoredSemanticMemory)
        ),
        key=lambda item: _memory_key_from_recall(item.recall),
    )

    buckets: dict[tuple[str, SemanticMemoryStatus, str], list[_ScoredCandidate]] = {}
    for candidate in semantics:
        memory = candidate.recall.memory
        assert isinstance(memory, StoredSemanticMemory)
        if memory.cardinality is not SemanticCardinality.MANY:
            continue
        bucket_key = (memory.slot_key, memory.status, _relevance_tier(candidate.recall))
        buckets.setdefault(bucket_key, []).append(candidate)

    for bucket_candidates in buckets.values():
        clusters = _cluster_by_shared_provenance(bucket_candidates)
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            if not _cluster_has_shared_provenance(cluster):
                continue
            memory = cluster[0].recall.memory
            assert isinstance(memory, StoredSemanticMemory)
            members = _order_members(cluster)
            for member in members:
                assigned.add(_identity_from_recall(member.recall))
            chunks.append(
                _build_semantic_chunk(
                    chunk_type=WorkingMemoryChunkType.SEMANTIC_COLLECTION,
                    coverage_key=f"slot:{memory.slot_key}",
                    members=members,
                    episode_by_id=episode_by_id,
                    token_estimator=token_estimator,
                )
            )

    for candidate in semantics:
        identity = _identity_from_recall(candidate.recall)
        if identity in assigned:
            continue
        memory = candidate.recall.memory
        assert isinstance(memory, StoredSemanticMemory)
        members = _attach_support_episodes(
            (candidate,),
            memory=memory,
            episode_by_id=episode_by_id,
            assigned=assigned,
        )
        for member in members:
            assigned.add(_identity_from_recall(member.recall))
        semantic_member_count = sum(
            1 for member in members if member.recall.memory_kind is MemoryKind.SEMANTIC
        )
        grouped_collection = (
            memory.cardinality is SemanticCardinality.MANY and semantic_member_count > 1
        )
        chunk_type = (
            WorkingMemoryChunkType.SEMANTIC_WITH_SUPPORT
            if len(members) > 1 and not grouped_collection
            else (
                WorkingMemoryChunkType.SEMANTIC_COLLECTION
                if grouped_collection
                else WorkingMemoryChunkType.SEMANTIC_SLOT
            )
        )
        coverage_key = _coverage_key_for_semantic(
            memory,
            collection=grouped_collection,
        )
        chunks.append(
            _build_semantic_chunk(
                chunk_type=chunk_type,
                coverage_key=coverage_key,
                members=members,
                episode_by_id=episode_by_id,
                token_estimator=token_estimator,
            )
        )

    for candidate in scored:
        identity = _identity_from_recall(candidate.recall)
        if identity in assigned:
            continue
        if candidate.recall.memory_kind is not MemoryKind.EPISODE:
            continue
        memory = candidate.recall.memory
        assert isinstance(memory, StoredEpisode)
        if _episode_supports_assigned_semantic(memory.id, semantics, assigned):
            continue
        assigned.add(identity)
        chunks.append(
            _build_episodic_chunk(
                candidate,
                token_estimator=token_estimator,
            )
        )

    return sorted(chunks, key=_chunk_sort_key)


def _build_semantic_chunk(
    *,
    chunk_type: WorkingMemoryChunkType,
    coverage_key: str,
    members: Sequence[_ScoredCandidate],
    episode_by_id: Mapping[str, _ScoredCandidate],
    token_estimator: TokenEstimator,
) -> _ChunkCandidate:
    ordered = _order_members(members)
    primary = ordered[0]
    serialized = _serialize_semantic_chunk(chunk_type, ordered)
    estimated_tokens = token_estimator.estimate(serialized)
    return _chunk_candidate_from_members(
        chunk_type=chunk_type,
        coverage_key=coverage_key,
        members=ordered,
        primary=primary,
        serialized_text=serialized,
        estimated_tokens=estimated_tokens,
        included_members=ordered,
    )


def _build_episodic_chunk(
    candidate: _ScoredCandidate,
    *,
    token_estimator: TokenEstimator,
) -> _ChunkCandidate:
    memory = candidate.recall.memory
    assert isinstance(memory, StoredEpisode)
    serialized = memory.statement
    estimated_tokens = token_estimator.estimate(serialized)
    return _chunk_candidate_from_members(
        chunk_type=WorkingMemoryChunkType.EPISODIC,
        coverage_key=f"episode:{memory.memory_key}",
        members=(candidate,),
        primary=candidate,
        serialized_text=serialized,
        estimated_tokens=estimated_tokens,
        included_members=(candidate,),
    )


def _chunk_candidate_from_members(
    *,
    chunk_type: WorkingMemoryChunkType,
    coverage_key: str,
    members: Sequence[_ScoredCandidate],
    primary: _ScoredCandidate,
    serialized_text: str,
    estimated_tokens: int,
    included_members: Sequence[_ScoredCandidate],
    members_omitted: int = 0,
) -> _ChunkCandidate:
    activation = max(member.recall.score for member in members)
    goal_relevance = max(member.goal_relevance for member in members)
    importance = max(member.importance for member in members)
    carryover = max(member.carryover for member in members)
    base_priority = max(member.base_priority for member in members)
    learned_utility = max(member.learned_utility for member in members)
    utility_adjustment = max(member.utility_adjustment for member in members)
    adjusted_priority = max(member.adjusted_priority for member in members)
    relevance_tier = max(
        (_relevance_tier(member.recall) for member in members),
        key=lambda tier: _RELEVANCE_TIER_RANK.get(RelevanceTier(tier), 0),
    )
    return _ChunkCandidate(
        chunk_type=chunk_type,
        coverage_key=coverage_key,
        members=tuple(members),
        primary=primary,
        relevance_tier=relevance_tier,
        activation=activation,
        goal_relevance=goal_relevance,
        importance=importance,
        carryover=carryover,
        base_priority=base_priority,
        learned_utility=learned_utility,
        utility_adjustment=utility_adjustment,
        adjusted_priority=adjusted_priority,
        serialized_text=serialized_text,
        estimated_tokens=estimated_tokens,
        included_members=tuple(included_members),
        members_omitted=members_omitted,
    )


def _episode_supports_assigned_semantic(
    episode_id: str,
    semantics: Sequence[_ScoredCandidate],
    assigned: set[MemoryIdentity],
) -> bool:
    for candidate in semantics:
        if _identity_from_recall(candidate.recall) not in assigned:
            continue
        memory = candidate.recall.memory
        if not isinstance(memory, StoredSemanticMemory):
            continue
        for derivation in memory.derivations:
            if (
                derivation.relation is SemanticDerivationRelation.SUPPORTS
                and derivation.episode_id == episode_id
            ):
                return True
    return False


def _attach_support_episodes(
    members: Sequence[_ScoredCandidate],
    *,
    memory: StoredSemanticMemory,
    episode_by_id: Mapping[str, _ScoredCandidate],
    assigned: set[MemoryIdentity],
) -> tuple[_ScoredCandidate, ...]:
    attached = list(members)
    for derivation in memory.derivations:
        if derivation.relation is not SemanticDerivationRelation.SUPPORTS:
            continue
        episode_candidate = episode_by_id.get(derivation.episode_id)
        if episode_candidate is None:
            continue
        identity = _identity_from_recall(episode_candidate.recall)
        if identity in assigned:
            continue
        attached.append(episode_candidate)
        break
    return tuple(attached)


def _cluster_by_shared_provenance(
    candidates: Sequence[_ScoredCandidate],
) -> list[list[_ScoredCandidate]]:
    if not candidates:
        return []
    parent = list(range(len(candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    provenance_sets = [_provenance_ids(candidate.recall.memory) for candidate in candidates]
    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            if provenance_sets[left].intersection(provenance_sets[right]):
                union(left, right)

    clusters: dict[int, list[_ScoredCandidate]] = {}
    for index, candidate in enumerate(candidates):
        root = find(index)
        clusters.setdefault(root, []).append(candidate)
    return [
        sorted(cluster, key=lambda item: _memory_key_from_recall(item.recall))
        for cluster in clusters.values()
    ]


def _cluster_has_shared_provenance(cluster: Sequence[_ScoredCandidate]) -> bool:
    if len(cluster) < 2:
        return False
    intersection = _provenance_ids(cluster[0].recall.memory)
    for candidate in cluster[1:]:
        intersection = intersection.intersection(_provenance_ids(candidate.recall.memory))
        if not intersection:
            return False
    return True


def _provenance_ids(memory: StoredEpisode | StoredSemanticMemory) -> frozenset[str]:
    if isinstance(memory, StoredEpisode):
        return frozenset(f"obs:{evidence.observation_id}" for evidence in memory.evidence)
    semantic = memory
    provenance: set[str] = set()
    for derivation in semantic.derivations:
        if derivation.relation is SemanticDerivationRelation.SUPPORTS:
            provenance.add(f"episode:{derivation.episode_id}")
    for evidence in semantic.observation_evidence:
        provenance.add(f"obs:{evidence.observation_id}")
    return frozenset(provenance)


def _coverage_key_for_semantic(memory: StoredSemanticMemory, *, collection: bool) -> str:
    if collection:
        return f"slot:{memory.slot_key}"
    if memory.object_entity_id:
        return f"slot:{memory.slot_key}:entity:{memory.object_entity_id}"
    if ":" in memory.object_value:
        return f"slot:{memory.slot_key}:entity:{memory.object_value.split(':', 1)[0]}"
    return f"slot:{memory.slot_key}:object:{memory.object_value}"


def _serialize_semantic_chunk(
    chunk_type: WorkingMemoryChunkType,
    members: Sequence[_ScoredCandidate],
) -> str:
    primary = members[0]
    memory = primary.recall.memory
    assert isinstance(memory, StoredSemanticMemory)
    semantic_members = [
        candidate for candidate in members if candidate.recall.memory_kind is MemoryKind.SEMANTIC
    ]
    episode_members = [
        candidate for candidate in members if candidate.recall.memory_kind is MemoryKind.EPISODE
    ]

    if chunk_type is WorkingMemoryChunkType.SEMANTIC_COLLECTION and len(semantic_members) > 1:
        objects = tuple(
            candidate.recall.memory.object_value
            for candidate in semantic_members
            if isinstance(candidate.recall.memory, StoredSemanticMemory)
        )
        subject = memory.subject_entity_id or memory.subject_id or "subject"
        return f"{subject} {memory.predicate} {_oxford_join(objects)}."

    statement = memory.statement
    if not episode_members:
        return statement

    episode = episode_members[0].recall.memory
    assert isinstance(episode, StoredEpisode)
    episode_tokens = _tokenize(episode.statement)
    semantic_tokens = _tokenize(statement)
    novel_tokens = episode_tokens.difference(semantic_tokens)
    if not novel_tokens:
        return statement
    return f"{statement} {episode.statement}"


def _trim_chunk_for_budget(
    chunk: _ChunkCandidate,
    *,
    remaining_tokens: int,
    token_estimator: TokenEstimator,
) -> tuple[_ChunkCandidate | None, int]:
    if chunk.estimated_tokens <= remaining_tokens:
        return chunk, chunk.estimated_tokens

    if len(chunk.members) == 1:
        return None, 0

    included = list(chunk.included_members)
    omitted = 0
    while len(included) > 1:
        included.pop()
        omitted += 1
        serialized = _serialize_semantic_chunk(chunk.chunk_type, included)
        estimated = token_estimator.estimate(serialized)
        if estimated <= remaining_tokens:
            trimmed = _chunk_candidate_from_members(
                chunk_type=chunk.chunk_type,
                coverage_key=chunk.coverage_key,
                members=chunk.members,
                primary=chunk.primary,
                serialized_text=serialized,
                estimated_tokens=estimated,
                included_members=tuple(included),
                members_omitted=omitted,
            )
            trimmed.final_score = chunk.final_score
            trimmed.inhibition = chunk.inhibition
            trimmed.novelty = chunk.novelty
            return trimmed, estimated

    primary_only = _serialize_semantic_chunk(chunk.chunk_type, (chunk.primary,))
    estimated = token_estimator.estimate(primary_only)
    if estimated > remaining_tokens:
        return None, 0
    trimmed = _chunk_candidate_from_members(
        chunk_type=chunk.chunk_type,
        coverage_key=chunk.coverage_key,
        members=chunk.members,
        primary=chunk.primary,
        serialized_text=primary_only,
        estimated_tokens=estimated,
        included_members=(chunk.primary,),
        members_omitted=len(chunk.members) - 1,
    )
    trimmed.final_score = chunk.final_score
    trimmed.inhibition = chunk.inhibition
    trimmed.novelty = chunk.novelty
    return trimmed, estimated


def _chunk_from_candidate(
    chunk: _ChunkCandidate,
    *,
    selected: bool,
) -> WorkingMemoryChunk:
    member_identities = tuple(_identity_from_recall(member.recall) for member in chunk.members)
    chunk_id = _chunk_id(
        chunk.chunk_type.value,
        chunk.coverage_key,
        member_identities,
    )
    return WorkingMemoryChunk(
        chunk_id=chunk_id,
        chunk_type=chunk.chunk_type,
        coverage_key=chunk.coverage_key,
        member_identities=member_identities,
        primary_identity=_identity_from_recall(chunk.primary.recall),
        serialized_text=chunk.serialized_text,
        estimated_tokens=chunk.estimated_tokens,
        relevance_tier=chunk.relevance_tier,
        activation=chunk.activation,
        novelty=chunk.novelty,
        selected=selected,
        rejection_reason=chunk.rejection_reason,
        members_total=len(chunk.members),
        members_included=len(chunk.included_members),
        members_omitted=chunk.members_omitted,
    )


def _merge_chunk_states(
    all_chunks: tuple[WorkingMemoryChunk, ...],
    selected_chunks: Sequence[WorkingMemoryChunk],
) -> tuple[WorkingMemoryChunk, ...]:
    selected_by_id = {chunk.chunk_id: chunk for chunk in selected_chunks}
    merged: list[WorkingMemoryChunk] = []
    for chunk in all_chunks:
        merged.append(selected_by_id.get(chunk.chunk_id, chunk))
    return tuple(merged)


def _chunk_id(
    chunk_type: str,
    coverage_key: str,
    member_identities: Sequence[MemoryIdentity],
) -> str:
    canonical = "\x1f".join(
        (
            chunk_type,
            coverage_key,
            *sorted(
                f"{identity.memory_kind.value}:{identity.memory_key}"
                for identity in member_identities
            ),
        )
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _order_members(members: Sequence[_ScoredCandidate]) -> tuple[_ScoredCandidate, ...]:
    return tuple(
        sorted(
            members,
            key=lambda item: (
                -_RELEVANCE_TIER_RANK.get(RelevanceTier(_relevance_tier(item.recall)), 0),
                -item.recall.score,
                item.recall.memory.created_at,
                _memory_key_from_recall(item.recall),
            ),
        )
    )


def _relevance_tier(recall: RecallResult) -> str:
    diagnostics = recall.diagnostics
    if diagnostics is not None and diagnostics.relevance_tier:
        return diagnostics.relevance_tier
    return RelevanceTier.CONTEXTUAL.value


def _oxford_join(items: Sequence[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _components_from_candidate(candidate: _ScoredCandidate) -> WorkingMemoryComponents:
    return WorkingMemoryComponents(
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
    )


def _stale_goal_penalty(
    recall: RecallResult,
    goal: RetrievalCue,
    config: WorkingMemoryConfig,
) -> float:
    goal_text = (goal.text or "").casefold()
    if "stale" not in goal_text:
        return 0.0
    penalty = 0.0
    memory = recall.memory
    if (
        isinstance(memory, StoredSemanticMemory)
        and memory.status is SemanticMemoryStatus.SUPERSEDED
    ):
        penalty += config.stale_goal_penalty
    for tag in _memory_tags(memory):
        if tag == "stale":
            penalty += config.stale_goal_penalty
            break
    return min(penalty, 1.0)


def _memory_tags(memory: StoredEpisode | StoredSemanticMemory) -> set[str]:
    metadata = memory.metadata
    tags: set[str] = set()
    raw_tags = metadata.get("tags")
    if isinstance(raw_tags, (list, tuple, set, frozenset)):
        tags.update(str(tag).casefold() for tag in raw_tags)
    episode_meta = metadata.get("episode")
    if isinstance(episode_meta, Mapping):
        episode_tags = episode_meta.get("tags")
        if isinstance(episode_tags, (list, tuple, set, frozenset)):
            tags.update(str(tag).casefold() for tag in episode_tags)
    return tags


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


def _deterministic_best_chunk(chunks: Sequence[_ChunkCandidate]) -> _ChunkCandidate:
    return max(chunks, key=_chunk_candidate_sort_key)


def _candidate_sort_key(candidate: _ScoredCandidate) -> tuple[float, float, float, str, str]:
    return (
        candidate.final_score,
        candidate.adjusted_priority,
        candidate.recall.activation,
        candidate.recall.memory_kind.value,
        _memory_key_from_recall(candidate.recall),
    )


def _chunk_candidate_sort_key(chunk: _ChunkCandidate) -> tuple[float, int, float, float, str]:
    return (
        chunk.final_score,
        _RELEVANCE_TIER_RANK.get(RelevanceTier(chunk.relevance_tier), 0),
        chunk.novelty,
        chunk.adjusted_priority,
        chunk.coverage_key,
    )


def _chunk_sort_key(chunk: _ChunkCandidate) -> tuple[str, str]:
    return (chunk.coverage_key, _memory_key_from_recall(chunk.primary.recall))


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


def _chunk_selection_reason(chunk: _ChunkCandidate) -> str:
    return (
        f"selected chunk with final_score={chunk.final_score:.4f} "
        f"(coverage={chunk.coverage_key}, type={chunk.chunk_type.value}, "
        f"members={len(chunk.included_members)}/{len(chunk.members)}, "
        f"novelty={chunk.novelty:.4f}, inhibition={chunk.inhibition:.4f})"
    )
