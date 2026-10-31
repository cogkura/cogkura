"""Read-only, revision-aware source provenance for prepared memory context."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from cogkura.exceptions import ValidationError
from cogkura.models import (
    ContextProvenance,
    EpisodeEvidenceInput,
    MemoryContext,
    MemoryIdentity,
    MemoryKind,
    MemoryProvenance,
    MemoryProvenanceRepresentation,
    SemanticDerivationRelation,
    SourceEvidence,
    SourceEvidenceRelation,
    SourceEvidenceResolutionStatus,
    StoredEpisode,
    StoredSemanticMemory,
    semantic_promotion_from_metadata,
)
from cogkura.observations.models import (
    ObservationRevisionIdentity,
    StoredObservation,
    StoredObservationRevision,
)
from cogkura.storage.base import EpisodeStore, ObservationStore

DEFAULT_MAX_PROVENANCE_EVIDENCE = 256


@dataclass(frozen=True, slots=True)
class _EvidenceReference:
    identity: ObservationRevisionIdentity
    contribution_score: float
    relation: SourceEvidenceRelation | None


@dataclass(frozen=True, slots=True)
class _ContextMemory:
    identity: MemoryIdentity
    memory: StoredEpisode | StoredSemanticMemory
    representation: MemoryProvenanceRepresentation


class SourceProvenanceResolver:
    """Resolve selected-context evidence without touching cognitive state."""

    def __init__(
        self,
        *,
        observation_store: ObservationStore,
        episode_store: EpisodeStore,
    ) -> None:
        self._observation_store = observation_store
        self._episode_store = episode_store

    async def resolve(
        self,
        context: MemoryContext,
        *,
        max_evidence: int = DEFAULT_MAX_PROVENANCE_EVIDENCE,
    ) -> ContextProvenance:
        if max_evidence <= 0:
            raise ValidationError("max_evidence must be greater than zero.")

        context_memories = _context_memories(context)
        episode_evidence = await self._episode_evidence_for_semantics(
            tenant_id=context.tenant_id,
            memories=context_memories,
        )
        raw_by_memory = {
            item.identity: _memory_evidence(item.memory, episode_evidence)
            for item in context_memories
        }

        returned_by_memory: dict[MemoryIdentity, tuple[_EvidenceReference, ...]] = {}
        truncated_memories: set[MemoryIdentity] = set()
        remaining = max_evidence
        for item in context_memories:
            references = raw_by_memory[item.identity]
            returned = tuple(references[:remaining])
            returned_by_memory[item.identity] = returned
            if len(returned) < len(references):
                truncated_memories.add(item.identity)
            remaining -= len(returned)

        requested = tuple(
            sorted(
                {
                    reference.identity
                    for references in returned_by_memory.values()
                    for reference in references
                },
                key=lambda item: (item.observation_id, item.observation_revision),
            )
        )
        current_observations, revisions = await asyncio.gather(
            self._observation_store.get_many(
                tenant_id=context.tenant_id,
                observation_ids={item.observation_id for item in requested},
            ),
            self._observation_store.get_revisions_many(
                tenant_id=context.tenant_id,
                references=requested,
            ),
        )
        current_by_id = {item.id: item for item in current_observations}

        items: list[MemoryProvenance] = []
        all_sources: set[tuple[str, str]] = set()
        unresolved_count = 0
        context_warnings: list[str] = []
        for item in context_memories:
            raw = raw_by_memory[item.identity]
            returned = returned_by_memory[item.identity]
            sources = tuple(
                _resolve_source_evidence(
                    reference,
                    current=current_by_id.get(reference.identity.observation_id),
                    revision=revisions.get(reference.identity),
                )
                for reference in returned
            )
            for source in sources:
                if source.source_namespace is not None and source.source_record_id is not None:
                    all_sources.add((source.source_namespace, source.source_record_id))
            unresolved_count += sum(
                source.resolution_status is not SourceEvidenceResolutionStatus.RESOLVED
                for source in sources
            )
            warnings = _warnings_for_sources(sources)
            is_truncated = item.identity in truncated_memories
            if is_truncated:
                warnings = (*warnings, "Evidence results were truncated by max_evidence.")
                context_warnings.append("Evidence results were truncated by max_evidence.")
            is_complete = not is_truncated and all(
                source.resolution_status is SourceEvidenceResolutionStatus.RESOLVED
                for source in sources
            )
            items.append(
                MemoryProvenance(
                    memory_identity=item.identity,
                    representation=item.representation,
                    sources=sources,
                    referenced_evidence_count=len(raw),
                    resolved_evidence_count=sum(
                        source.resolution_status is SourceEvidenceResolutionStatus.RESOLVED
                        for source in sources
                    ),
                    distinct_source_count=len(
                        {
                            (source.source_namespace, source.source_record_id)
                            for source in sources
                            if source.source_namespace is not None
                            and source.source_record_id is not None
                        }
                    ),
                    is_complete=is_complete,
                    is_truncated=is_truncated,
                    warnings=warnings,
                    promotion=(
                        semantic_promotion_from_metadata(item.memory.metadata)
                        if isinstance(item.memory, StoredSemanticMemory)
                        else None
                    ),
                )
            )

        context_warnings.extend(
            warning
            for item in items
            for warning in item.warnings
            if warning not in context_warnings
        )
        return ContextProvenance(
            items=tuple(items),
            source_count=len(all_sources),
            unresolved_evidence_count=unresolved_count,
            is_complete=all(item.is_complete for item in items),
            is_truncated=bool(truncated_memories),
            warnings=tuple(context_warnings),
        )

    async def _episode_evidence_for_semantics(
        self,
        *,
        tenant_id: str,
        memories: Sequence[_ContextMemory],
    ) -> Mapping[str, tuple[EpisodeEvidenceInput, ...]]:
        episode_ids = {
            derivation.episode_id
            for item in memories
            if isinstance(item.memory, StoredSemanticMemory)
            for derivation in item.memory.derivations
        }
        return await self._episode_store.get_evidence_many(
            tenant_id=tenant_id,
            episode_ids=episode_ids,
        )


def _context_memories(context: MemoryContext) -> tuple[_ContextMemory, ...]:
    """List actual selected representations, not derivation-only support."""
    ordered: list[_ContextMemory] = []
    seen: set[MemoryIdentity] = set()
    for item in context.items:
        primary = _context_memory(
            item.recall.memory,
            representation=MemoryProvenanceRepresentation.SELECTED_RENDERED,
        )
        if primary.identity not in seen:
            ordered.append(primary)
            seen.add(primary.identity)
        for recall in item.member_recalls:
            member = _context_memory(
                recall.memory,
                representation=MemoryProvenanceRepresentation.CHUNK_MEMBER_NOT_INDEPENDENTLY_RENDERED,
            )
            if member.identity not in seen:
                ordered.append(member)
                seen.add(member.identity)
    return tuple(ordered)


def _context_memory(
    memory: StoredEpisode | StoredSemanticMemory,
    *,
    representation: MemoryProvenanceRepresentation,
) -> _ContextMemory:
    identity = MemoryIdentity(
        memory_kind=(
            MemoryKind.EPISODE if isinstance(memory, StoredEpisode) else MemoryKind.SEMANTIC
        ),
        memory_key=memory.memory_key,
    )
    return _ContextMemory(identity=identity, memory=memory, representation=representation)


def _memory_evidence(
    memory: StoredEpisode | StoredSemanticMemory,
    episode_evidence: Mapping[str, tuple[EpisodeEvidenceInput, ...]],
) -> tuple[_EvidenceReference, ...]:
    if isinstance(memory, StoredEpisode):
        return tuple(
            _EvidenceReference(
                identity=ObservationRevisionIdentity(
                    observation_id=item.observation_id,
                    observation_revision=item.observation_revision,
                ),
                contribution_score=item.contribution_score,
                relation=SourceEvidenceRelation.DIRECT_EPISODIC_EVIDENCE,
            )
            for item in sorted(
                memory.evidence,
                key=lambda item: (
                    item.sequence_number,
                    item.observation_id,
                    item.observation_revision,
                ),
            )
        )

    derivations_by_evidence: dict[
        ObservationRevisionIdentity, list[tuple[SemanticDerivationRelation, float]]
    ] = defaultdict(list)
    for derivation in memory.derivations:
        for evidence in episode_evidence.get(derivation.episode_id, ()):
            identity = ObservationRevisionIdentity(
                observation_id=evidence.observation_id,
                observation_revision=evidence.observation_revision,
            )
            derivations_by_evidence[identity].append(
                (derivation.relation, derivation.contribution_score)
            )

    references: list[_EvidenceReference] = []
    for evidence in sorted(
        memory.observation_evidence,
        key=lambda item: (item.sequence_number, item.observation_id, item.observation_revision),
    ):
        identity = ObservationRevisionIdentity(
            observation_id=evidence.observation_id,
            observation_revision=evidence.observation_revision,
        )
        relations = derivations_by_evidence.get(identity, [])
        distinct_relations = {relation for relation, _score in relations}
        relation: SourceEvidenceRelation | None = None
        contribution_score = evidence.contribution_score
        if len(distinct_relations) == 1:
            derivation_relation, derivation_score = sorted(
                relations,
                key=lambda item: (item[0].value, -item[1]),
            )[0]
            relation = _source_relation(derivation_relation)
            contribution_score = derivation_score
        references.append(
            _EvidenceReference(
                identity=identity,
                contribution_score=contribution_score,
                relation=relation,
            )
        )
    return tuple(references)


def _source_relation(relation: SemanticDerivationRelation) -> SourceEvidenceRelation:
    if relation is SemanticDerivationRelation.SUPPORTS:
        return SourceEvidenceRelation.SEMANTIC_SUPPORT
    return SourceEvidenceRelation.SEMANTIC_CONTRADICTION


def _resolve_source_evidence(
    reference: _EvidenceReference,
    *,
    current: StoredObservation | None,
    revision: StoredObservationRevision | None,
) -> SourceEvidence:
    if revision is None:
        if current is None:
            return SourceEvidence(
                observation_id=reference.identity.observation_id,
                observation_revision=reference.identity.observation_revision,
                source_namespace=None,
                source_record_id=None,
                source_version=None,
                source_ref=None,
                observed_at=None,
                contribution_score=reference.contribution_score,
                relation=reference.relation,
                resolution_status=SourceEvidenceResolutionStatus.SOURCE_UNAVAILABLE,
                is_latest_known_revision=None,
            )
        return SourceEvidence(
            observation_id=reference.identity.observation_id,
            observation_revision=reference.identity.observation_revision,
            source_namespace=current.source_namespace,
            source_record_id=current.source_record_id,
            source_version=None,
            source_ref=None,
            observed_at=None,
            contribution_score=reference.contribution_score,
            relation=reference.relation,
            resolution_status=SourceEvidenceResolutionStatus.REVISION_UNAVAILABLE,
            is_latest_known_revision=None,
        )

    source_ref_value = revision.metadata.get("source_ref")
    source_ref = source_ref_value.strip() if isinstance(source_ref_value, str) else None
    status = (
        SourceEvidenceResolutionStatus.RESOLVED
        if source_ref
        else SourceEvidenceResolutionStatus.SOURCE_REF_MISSING
    )
    return SourceEvidence(
        observation_id=reference.identity.observation_id,
        observation_revision=reference.identity.observation_revision,
        source_namespace=revision.source_namespace,
        source_record_id=revision.source_record_id,
        source_version=revision.source_version,
        source_ref=source_ref or None,
        observed_at=revision.observed_at,
        contribution_score=reference.contribution_score,
        relation=reference.relation,
        resolution_status=status,
        is_latest_known_revision=(
            reference.identity.observation_revision == revision.current_revision
        ),
    )


def _warnings_for_sources(sources: Sequence[SourceEvidence]) -> tuple[str, ...]:
    statuses = {source.resolution_status for source in sources}
    warnings: list[str] = []
    if SourceEvidenceResolutionStatus.SOURCE_UNAVAILABLE in statuses:
        warnings.append("One or more source observations are unavailable.")
    if SourceEvidenceResolutionStatus.REVISION_UNAVAILABLE in statuses:
        warnings.append("One or more referenced source revisions are unavailable.")
    if SourceEvidenceResolutionStatus.SOURCE_REF_MISSING in statuses:
        warnings.append("One or more resolved source revisions have no source_ref metadata.")
    return tuple(warnings)
