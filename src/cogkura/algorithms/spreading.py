"""Bounded spreading activation over entity-memory associations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from cogkura.models import (
    ActivationCandidate,
    ActivationConfig,
    MemoryIdentity,
    RetrievalCue,
)


@dataclass(frozen=True, slots=True)
class SpreadingMetadata:
    """Lightweight explainability metadata for associative retrieval."""

    hop: int
    sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SpreadingResult:
    """Spreading activation scores and metadata keyed by memory identity."""

    scores: Mapping[MemoryIdentity, float]
    metadata: Mapping[MemoryIdentity, SpreadingMetadata]


class SpreadingActivator(Protocol):
    """Computes associative spreading activation over a candidate set."""

    def calculate(
        self,
        *,
        candidates: Sequence[ActivationCandidate],
        cue: RetrievalCue,
        config: ActivationConfig,
    ) -> SpreadingResult:
        """Return bounded spreading scores for each activated memory."""


def _unique_sorted_entity_ids(entity_ids: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for entity_id in sorted(entity_ids):
        if entity_id in seen:
            continue
        seen.add(entity_id)
        ordered.append(entity_id)
    return tuple(ordered)


def _unique_cue_sources(cue: RetrievalCue) -> tuple[str, ...]:
    return _unique_sorted_entity_ids(cue.entity_ids)


def _spreading_entity_ids(candidate: ActivationCandidate) -> tuple[str, ...]:
    """Entity concepts for spreading; subject scope is excluded from association."""
    excluded = {candidate.subject_id} if candidate.subject_id else set()
    return _unique_sorted_entity_ids(
        tuple(entity_id for entity_id in candidate.entity_ids if entity_id not in excluded)
    )


def _build_entity_to_memories(
    candidates: Sequence[ActivationCandidate],
) -> dict[str, tuple[MemoryIdentity, ...]]:
    index: dict[str, list[MemoryIdentity]] = defaultdict(list)
    for candidate in candidates:
        identity = candidate.identity
        for entity_id in _spreading_entity_ids(candidate):
            index[entity_id].append(identity)
    return {
        entity_id: tuple(sorted(identities, key=_identity_sort_key))
        for entity_id, identities in sorted(index.items())
    }


def _identity_sort_key(identity: MemoryIdentity) -> tuple[str, str]:
    return (identity.memory_kind.value, identity.memory_key)


def _build_candidate_index(
    candidates: Sequence[ActivationCandidate],
) -> dict[MemoryIdentity, ActivationCandidate]:
    return {candidate.identity: candidate for candidate in candidates}


def calculate_spreading_activation(
    *,
    candidates: Sequence[ActivationCandidate],
    cue: RetrievalCue,
    config: ActivationConfig,
) -> SpreadingResult:
    """Compute deterministic spreading activation over the candidate graph."""
    sources = _unique_cue_sources(cue)
    if not sources:
        return SpreadingResult(scores={}, metadata={})

    entity_to_memories = _build_entity_to_memories(candidates)
    candidate_by_identity = _build_candidate_index(candidates)

    entity_frontier = {entity_id: config.source_activation / len(sources) for entity_id in sources}
    memory_scores: dict[MemoryIdentity, float] = defaultdict(float)
    memory_hops: dict[MemoryIdentity, int] = {}
    expanded_memories: set[MemoryIdentity] = set()

    for hop in range(1, config.spreading_max_hops + 1):
        memory_frontier: dict[MemoryIdentity, float] = defaultdict(float)

        for entity_id in sorted(entity_frontier):
            entity_activation = entity_frontier[entity_id]
            if entity_activation < config.spreading_min_activation:
                continue

            memories = entity_to_memories.get(entity_id, ())
            fan = len(memories)
            if fan == 0:
                continue

            association = config.maximum_associative_strength / fan
            for identity in memories:
                contribution = entity_activation * association
                if contribution < config.spreading_min_activation:
                    continue
                memory_frontier[identity] += contribution

        for identity in sorted(memory_frontier, key=_identity_sort_key):
            contribution = memory_frontier[identity]
            memory_scores[identity] = min(
                config.source_activation,
                memory_scores[identity] + contribution,
            )
            if identity not in memory_hops:
                memory_hops[identity] = hop

        if hop >= config.spreading_max_hops:
            break

        next_entity_frontier: dict[str, float] = defaultdict(float)
        incoming_entities = set(entity_frontier)

        for identity in sorted(memory_frontier, key=_identity_sort_key):
            if identity in expanded_memories:
                continue
            expanded_memories.add(identity)

            candidate = candidate_by_identity[identity]
            neighbours = _spreading_entity_ids(candidate)
            remaining_degree = max(1, len(neighbours) - 1)
            memory_activation = memory_frontier[identity]

            for entity_id in neighbours:
                if entity_id in incoming_entities:
                    continue

                contribution = memory_activation * config.spreading_decay / remaining_degree
                if contribution < config.spreading_min_activation:
                    continue
                next_entity_frontier[entity_id] += contribution

        entity_frontier = dict(next_entity_frontier)
        if not entity_frontier:
            break

    metadata = {
        identity: SpreadingMetadata(hop=memory_hops[identity], sources=sources)
        for identity in memory_scores
        if memory_scores[identity] > 0.0
    }
    return SpreadingResult(scores=dict(memory_scores), metadata=metadata)


class DeterministicSpreadingActivator:
    """Collins & Loftus-inspired bounded spreading activation."""

    def calculate(
        self,
        *,
        candidates: Sequence[ActivationCandidate],
        cue: RetrievalCue,
        config: ActivationConfig,
    ) -> SpreadingResult:
        return calculate_spreading_activation(
            candidates=candidates,
            cue=cue,
            config=config,
        )
