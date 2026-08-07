"""Public memory API."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from cognema.algorithms.activation import (
    ACTRDeclarativeActivator,
    DeclarativeActivator,
    activation_candidate_from_episode,
    activation_candidate_from_semantic,
)
from cognema.algorithms.episodic import DeterministicEpisodicEncoder, EpisodicEncoder
from cognema.algorithms.semantic import (
    ComplementaryLearningSemanticConsolidator,
    MetadataSemanticExtractor,
    SemanticConsolidator,
    SemanticExtractor,
)
from cognema.exceptions import CandidateSetTooLargeError, ValidationError
from cognema.mappers.base import ObservationMapper
from cognema.models import (
    ActivationConfig,
    ActivationReferenceKind,
    EpisodeEncodingResult,
    MemoryReference,
    RecallResult,
    RetrievalCue,
    SemanticConsolidationResult,
    SemanticMemoryStatus,
    StoredEpisode,
    StoredSemanticMemory,
)
from cognema.observations.models import IngestionResult, IngestStatus, ObservationInput
from cognema.observations.pipeline import ObservationPipeline
from cognema.observations.policies import DefaultObservationPolicy, ObservationPolicy
from cognema.observations.retention import ObservationRetentionMode
from cognema.sources.base import SourceConnector
from cognema.storage import (
    ActivationStore,
    CheckpointStore,
    EpisodeStore,
    ObservationStore,
    SemanticMemoryStore,
)
from cognema.storage.in_memory_activation import InMemoryActivationStore
from cognema.storage.in_memory_episode import InMemoryEpisodeStore
from cognema.storage.in_memory_observation import InMemoryCheckpointStore, InMemoryObservationStore
from cognema.storage.in_memory_semantic import InMemorySemanticMemoryStore

_DEFAULT_BATCH_SIZE = 500


class Memory:
    """Cognitive memory facade for observation ingestion and recall."""

    def __init__(
        self,
        *,
        observation_store: ObservationStore | None = None,
        checkpoint_store: CheckpointStore | None = None,
        episode_store: EpisodeStore | None = None,
        semantic_store: SemanticMemoryStore | None = None,
        activation_store: ActivationStore | None = None,
        episodic_encoder: EpisodicEncoder | None = None,
        semantic_extractor: SemanticExtractor | None = None,
        semantic_consolidator: SemanticConsolidator | None = None,
        declarative_activator: DeclarativeActivator | None = None,
        activation_config: ActivationConfig | None = None,
        policy: ObservationPolicy | None = None,
        retention_mode: ObservationRetentionMode = ObservationRetentionMode.FULL,
    ) -> None:
        self._observation_store = (
            observation_store if observation_store is not None else InMemoryObservationStore()
        )
        self._checkpoint_store = (
            checkpoint_store if checkpoint_store is not None else InMemoryCheckpointStore()
        )
        self._episode_store = episode_store if episode_store is not None else InMemoryEpisodeStore()
        self._semantic_store = (
            semantic_store if semantic_store is not None else InMemorySemanticMemoryStore()
        )
        self._activation_store = (
            activation_store if activation_store is not None else InMemoryActivationStore()
        )
        self._episodic_encoder = (
            episodic_encoder if episodic_encoder is not None else DeterministicEpisodicEncoder()
        )
        self._semantic_extractor = (
            semantic_extractor if semantic_extractor is not None else MetadataSemanticExtractor()
        )
        self._semantic_consolidator = (
            semantic_consolidator
            if semantic_consolidator is not None
            else ComplementaryLearningSemanticConsolidator()
        )
        self._declarative_activator = (
            declarative_activator
            if declarative_activator is not None
            else ACTRDeclarativeActivator()
        )
        self._activation_config = (
            activation_config if activation_config is not None else ActivationConfig()
        )
        self._policy = policy if policy is not None else DefaultObservationPolicy()
        self._retention_mode = retention_mode
        self._pipeline = ObservationPipeline(
            self._observation_store,
            policy=self._policy,
            retention_mode=retention_mode,
        )

    async def observe(self, observation: ObservationInput) -> IngestStatus:
        """Ingest a single normalized observation."""
        status = await self._pipeline.ingest(observation)
        if status is None:
            raise ValidationError("Observation was rejected by policy.")
        return status

    async def ingest(
        self,
        *,
        source: SourceConnector,
        mapper: ObservationMapper,
        tenant_id: str,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> IngestionResult:
        """Ingest observations from a source connector with checkpointing."""
        if not tenant_id.strip():
            raise ValidationError("tenant_id must not be empty.")

        result = IngestionResult()
        checkpoint = await self._checkpoint_store.get(
            tenant_id=tenant_id,
            connector_id=source.connector_id,
        )
        batch: list[Any] = []
        last_checkpoint: dict[str, Any] | None = checkpoint

        async for record in source.records(checkpoint):
            result = IngestionResult(
                discovered=result.discovered + 1,
                created=result.created,
                updated=result.updated,
                deleted=result.deleted,
                unchanged=result.unchanged,
                restored=result.restored,
                rejected=result.rejected,
                failed=result.failed,
            )
            batch.append(record)
            if len(batch) >= batch_size:
                result, last_checkpoint = await self._process_batch(
                    mapper,
                    tenant_id,
                    source,
                    batch,
                    result,
                    last_checkpoint,
                )
                batch = []

        if batch:
            result, last_checkpoint = await self._process_batch(
                mapper,
                tenant_id,
                source,
                batch,
                result,
                last_checkpoint,
            )

        return result

    async def _process_batch(
        self,
        mapper: ObservationMapper,
        tenant_id: str,
        source: SourceConnector,
        batch: list[Any],
        result: IngestionResult,
        last_checkpoint: dict[str, Any] | None,
    ) -> tuple[IngestionResult, dict[str, Any] | None]:
        batch_checkpoint = last_checkpoint
        for record in batch:
            try:
                observation = mapper.map(record)
                status = await self._pipeline.ingest(
                    observation,
                    expected_tenant_id=tenant_id,
                )
                if status is None:
                    result = IngestionResult(
                        discovered=result.discovered,
                        created=result.created,
                        updated=result.updated,
                        deleted=result.deleted,
                        unchanged=result.unchanged,
                        restored=result.restored,
                        rejected=result.rejected + 1,
                        failed=result.failed,
                    )
                else:
                    result = result.record(status)
                batch_checkpoint = source.checkpoint_for(record)
            except Exception:
                result = IngestionResult(
                    discovered=result.discovered,
                    created=result.created,
                    updated=result.updated,
                    deleted=result.deleted,
                    unchanged=result.unchanged,
                    restored=result.restored,
                    rejected=result.rejected,
                    failed=result.failed + 1,
                )
                return result, last_checkpoint

        if batch_checkpoint is not None:
            await self._checkpoint_store.set(
                tenant_id=tenant_id,
                connector_id=source.connector_id,
                checkpoint=batch_checkpoint,
            )
        return result, batch_checkpoint

    async def recall(
        self,
        query: str | RetrievalCue,
        *,
        tenant_id: str,
        subject_id: str | None = None,
        limit: int = 5,
        as_of: datetime | None = None,
        semantic_statuses: frozenset[SemanticMemoryStatus] | None = None,
    ) -> list[RecallResult]:
        """Recall episodic and semantic memories by declarative activation."""
        if not tenant_id.strip():
            raise ValidationError("tenant_id must not be empty.")
        if limit <= 0:
            raise ValidationError("Limit must be greater than zero.")

        cue = _normalise_cue(query, subject_id=subject_id)
        if as_of is not None:
            if as_of.tzinfo is None:
                raise ValidationError("as_of must be timezone-aware.")
            evaluation_time = as_of.astimezone(UTC)
        else:
            evaluation_time = datetime.now(UTC)

        episodes, semantic_memories = await asyncio.gather(
            self._episode_store.list(
                tenant_id=tenant_id,
                subject_id=subject_id,
                include_inactive=False,
            ),
            self._semantic_store.list(
                tenant_id=tenant_id,
                subject_id=subject_id,
                include_inactive=False,
            ),
        )
        eligible_semantics = [
            memory
            for memory in semantic_memories
            if memory.status is not SemanticMemoryStatus.SUPERSEDED
            and (semantic_statuses is None or memory.status in semantic_statuses)
        ]
        candidates = [activation_candidate_from_episode(episode) for episode in episodes] + [
            activation_candidate_from_semantic(memory) for memory in eligible_semantics
        ]
        if len(candidates) > self._activation_config.max_candidates:
            raise CandidateSetTooLargeError(
                f"Candidate set size {len(candidates)} exceeds max_candidates "
                f"{self._activation_config.max_candidates}."
            )

        identities = [candidate.identity for candidate in candidates]
        references = await self._activation_store.list_reference_times(
            tenant_id=tenant_id,
            identities=identities,
            before_or_at=evaluation_time,
        )
        return self._declarative_activator.rank(
            candidates=candidates,
            cue=cue,
            references=references,
            as_of=evaluation_time,
            config=self._activation_config,
            limit=limit,
        )

    async def record_access(
        self,
        results: Sequence[RecallResult],
        *,
        tenant_id: str,
        referenced_at: datetime | None = None,
        reference_kind: ActivationReferenceKind = ActivationReferenceKind.RETRIEVED,
        request_id: str | None = None,
    ) -> None:
        """Record explicit access to recalled memories for base-level activation."""
        if not tenant_id.strip():
            raise ValidationError("tenant_id must not be empty.")
        if referenced_at is not None:
            if referenced_at.tzinfo is None:
                raise ValidationError("referenced_at must be timezone-aware.")
            timestamp = referenced_at.astimezone(UTC)
        else:
            timestamp = datetime.now(UTC)
        if not results:
            return
        references = [
            MemoryReference(
                tenant_id=tenant_id,
                memory_kind=result.memory_kind,
                memory_key=_memory_key_from_result(result),
                reference_kind=reference_kind,
                referenced_at=timestamp,
                request_id=request_id,
            )
            for result in results
        ]
        await self._activation_store.append_references(references)

    def sleep(self) -> None:
        """Run deferred memory maintenance (no-op in this release)."""

    async def encode_episodes(
        self,
        *,
        tenant_id: str,
        subject_id: str | None = None,
    ) -> EpisodeEncodingResult:
        """Build current episodic memories from stored observations."""
        if not tenant_id.strip():
            raise ValidationError("tenant_id must not be empty.")

        observations = await self._observation_store.list(
            tenant_id=tenant_id,
            subject_id=subject_id,
        )
        candidates = self._episodic_encoder.encode(observations)
        result = EpisodeEncodingResult(
            observations=len(observations),
            candidates=len(candidates),
        )
        active_keys: set[str] = set()
        for candidate in candidates:
            active_keys.add(candidate.memory_key)
            status = await self._episode_store.upsert(candidate)
            result = result.record(status)

        deactivated = await self._episode_store.deactivate_missing(
            tenant_id=tenant_id,
            subject_id=subject_id,
            active_memory_keys=active_keys,
        )
        return replace(result, deactivated=deactivated)

    async def list_episodes(
        self,
        *,
        tenant_id: str,
        subject_id: str | None = None,
        include_inactive: bool = False,
        limit: int | None = None,
    ) -> list[StoredEpisode]:
        """List encoded episodes for a tenant."""
        if not tenant_id.strip():
            raise ValidationError("tenant_id must not be empty.")
        if limit is not None and limit <= 0:
            raise ValidationError("Limit must be greater than zero.")
        return await self._episode_store.list(
            tenant_id=tenant_id,
            subject_id=subject_id,
            include_inactive=include_inactive,
            limit=limit,
        )

    async def consolidate_semantics(
        self,
        *,
        tenant_id: str,
        subject_id: str | None = None,
    ) -> SemanticConsolidationResult:
        """Build semantic memories from active episodic memories."""
        if not tenant_id.strip():
            raise ValidationError("tenant_id must not be empty.")

        episodes = await self._episode_store.list(
            tenant_id=tenant_id,
            subject_id=subject_id,
            include_inactive=False,
        )
        observation_ids = {
            evidence.observation_id for episode in episodes for evidence in episode.evidence
        }
        observations = await self._observation_store.get_many(
            tenant_id=tenant_id,
            observation_ids=observation_ids,
        )
        observations_by_id = {observation.id: observation for observation in observations}
        extraction = await self._semantic_extractor.extract(
            episodes,
            observations=observations_by_id,
        )
        semantic_memories = self._semantic_consolidator.consolidate(
            episodes,
            extraction.candidates,
        )
        result = SemanticConsolidationResult(
            episodes=len(episodes),
            extracted_candidates=len(extraction.candidates),
            extracted_failures=extraction.failed,
            canonical_claims=len(semantic_memories),
        )
        active_keys: set[str] = set()
        for semantic_memory in semantic_memories:
            active_keys.add(semantic_memory.memory_key)
            status = await self._semantic_store.upsert(semantic_memory)
            result = result.record(status, semantic_memory.status)

        deactivated = await self._semantic_store.deactivate_missing(
            tenant_id=tenant_id,
            subject_id=subject_id,
            active_memory_keys=active_keys,
        )
        return replace(result, deactivated=deactivated)

    async def list_semantic_memories(
        self,
        *,
        tenant_id: str,
        subject_id: str | None = None,
        include_inactive: bool = False,
        status: SemanticMemoryStatus | None = None,
        limit: int | None = None,
    ) -> list[StoredSemanticMemory]:
        """List consolidated semantic memories for a tenant."""
        if not tenant_id.strip():
            raise ValidationError("tenant_id must not be empty.")
        if limit is not None and limit <= 0:
            raise ValidationError("Limit must be greater than zero.")
        return await self._semantic_store.list(
            tenant_id=tenant_id,
            subject_id=subject_id,
            include_inactive=include_inactive,
            status=status,
            limit=limit,
        )

    async def clear(self, *, tenant_id: str) -> None:
        """Clear activation references, semantic memories, episodes, and observations."""
        if not tenant_id.strip():
            raise ValidationError("tenant_id must not be empty.")
        await self._activation_store.clear(tenant_id=tenant_id)
        await self._semantic_store.clear(tenant_id=tenant_id)
        await self._episode_store.clear(tenant_id=tenant_id)
        await self._observation_store.clear(tenant_id=tenant_id)


def _normalise_cue(query: str | RetrievalCue, *, subject_id: str | None) -> RetrievalCue:
    if isinstance(query, RetrievalCue):
        if subject_id is not None and query.subject_id is None:
            return RetrievalCue(
                text=query.text,
                subject_id=subject_id,
                entity_ids=query.entity_ids,
                predicate=query.predicate,
                object_value=query.object_value,
                qualifiers=query.qualifiers,
            )
        return query
    stripped = query.strip()
    if not stripped and not (subject_id and subject_id.strip()):
        raise ValidationError("Query must not be empty.")
    return RetrievalCue(text=stripped or None, subject_id=subject_id)


def _memory_key_from_result(result: RecallResult) -> str:
    memory = result.memory
    if isinstance(memory, StoredEpisode):
        return memory.memory_key
    if isinstance(memory, StoredSemanticMemory):
        return memory.memory_key
    raise ValidationError("Unsupported memory type for access recording.")
