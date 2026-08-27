"""Derived cognitive activation references from durable memory evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from cogkura.exceptions import ValidationError
from cogkura.models import (
    ActivationCandidate,
    ActivationReferenceTrace,
    CognitiveReferenceTrace,
    CognitiveTraceOrigin,
    SemanticDerivationRelation,
    StoredEpisode,
    StoredSemanticMemory,
    SupportProvenance,
)


def derive_episode_cognitive_traces(episode: StoredEpisode) -> tuple[CognitiveReferenceTrace, ...]:
    """Return encoding traces from episode evidence chronology."""
    return (
        CognitiveReferenceTrace(
            origin=CognitiveTraceOrigin.ENCODED,
            referenced_at=episode.ended_at,
            weight=1,
        ),
    )


def derive_semantic_cognitive_traces(
    memory: StoredSemanticMemory,
    episode_by_id: Mapping[str, StoredEpisode],
) -> tuple[CognitiveReferenceTrace, ...]:
    """Return support traces from revision-scoped supporting episode evidence."""
    traces: list[CognitiveReferenceTrace] = []
    for derivation in memory.derivations:
        if derivation.relation is not SemanticDerivationRelation.SUPPORTS:
            continue
        episode = episode_by_id.get(derivation.episode_id)
        if episode is not None:
            referenced_at = episode.ended_at
        else:
            referenced_at = memory.first_supported_at
        traces.append(
            CognitiveReferenceTrace(
                origin=CognitiveTraceOrigin.SUPPORTED,
                referenced_at=referenced_at,
                weight=1,
                episode_id=derivation.episode_id,
                revision_key=memory.revision_key,
            )
        )
    if traces:
        return tuple(sorted(traces, key=lambda trace: trace.referenced_at))
    return (
        CognitiveReferenceTrace(
            origin=CognitiveTraceOrigin.AGGREGATE_SUPPORT_FALLBACK,
            referenced_at=memory.first_supported_at,
            weight=1,
            revision_key=memory.revision_key,
        ),
    )


def activation_candidate_from_episode_with_traces(
    episode: StoredEpisode,
    *,
    support_provenance: Sequence[SupportProvenance] = (),
) -> ActivationCandidate:
    """Adapt a stored episode with derived encoding traces."""
    from cogkura.algorithms.activation import activation_candidate_from_episode

    candidate = activation_candidate_from_episode(
        episode,
        support_provenance=support_provenance,
    )
    return _with_cognitive_traces(candidate, derive_episode_cognitive_traces(episode))


def activation_candidate_from_semantic_with_traces(
    memory: StoredSemanticMemory,
    episode_by_id: Mapping[str, StoredEpisode],
) -> ActivationCandidate:
    """Adapt a stored semantic memory with derived support traces."""
    from cogkura.algorithms.activation import activation_candidate_from_semantic

    candidate = activation_candidate_from_semantic(memory)
    traces = derive_semantic_cognitive_traces(memory, episode_by_id)
    return _with_cognitive_traces(candidate, traces)


def build_activation_candidates(
    episodes: Sequence[StoredEpisode],
    semantics: Sequence[StoredSemanticMemory],
    *,
    episode_support_provenance_index: Mapping[str, Sequence[SupportProvenance]] | None = None,
) -> list[ActivationCandidate]:
    """Build activation candidates with evidence-time cognitive traces."""
    episode_by_id = {episode.id: episode for episode in episodes}
    provenance_index = episode_support_provenance_index or {}
    return [
        activation_candidate_from_episode_with_traces(
            episode,
            support_provenance=provenance_index.get(episode.id, ()),
        )
        for episode in episodes
    ] + [
        activation_candidate_from_semantic_with_traces(memory, episode_by_id)
        for memory in semantics
    ]


def _with_cognitive_traces(
    candidate: ActivationCandidate,
    traces: tuple[CognitiveReferenceTrace, ...],
) -> ActivationCandidate:
    if not traces:
        raise ValidationError("Activation candidates require at least one cognitive trace.")
    return ActivationCandidate(
        memory_kind=candidate.memory_kind,
        memory_key=candidate.memory_key,
        created_at=candidate.created_at,
        text=candidate.text,
        subject_id=candidate.subject_id,
        entity_ids=candidate.entity_ids,
        predicate=candidate.predicate,
        object_value=candidate.object_value,
        qualifiers=candidate.qualifiers,
        memory=candidate.memory,
        importance=candidate.importance,
        slot_key=candidate.slot_key,
        semantic_status=candidate.semantic_status,
        last_supported_at=candidate.last_supported_at,
        support_provenance=candidate.support_provenance,
        cognitive_traces=traces,
    )


def activation_reference_traces_for_candidate(
    candidate: ActivationCandidate,
    stored_traces: Sequence[ActivationReferenceTrace],
) -> tuple[ActivationReferenceTrace, ...]:
    """Merge derived cognitive traces with persisted access and learning traces."""
    cognitive = tuple(trace.to_activation_trace() for trace in candidate.cognitive_traces)
    combined = (*cognitive, *stored_traces)
    return tuple(sorted(combined, key=lambda trace: trace.referenced_at))
