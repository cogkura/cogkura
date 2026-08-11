"""Public memory API."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

from cogkura.algorithms.activation import (
    ACTRDeclarativeActivator,
    DeclarativeActivator,
    activation_candidate_from_episode,
    activation_candidate_from_semantic,
)
from cogkura.algorithms.episodic import DeterministicEpisodicEncoder, EpisodicEncoder
from cogkura.algorithms.forgetting import EbbinghausForgettingEvaluator, ForgettingEvaluator
from cogkura.algorithms.learning import (
    DeterministicLearningProcessor,
    LearningProcessor,
    build_learning_utilities,
    calculate_association_strength,
    learning_context_key,
)
from cogkura.algorithms.reconsolidation import (
    DeterministicSemanticReconciler,
    SemanticReconciler,
)
from cogkura.algorithms.semantic import (
    ComplementaryLearningSemanticConsolidator,
    MetadataSemanticExtractor,
    SemanticConsolidator,
    SemanticExtractor,
)
from cogkura.algorithms.working_memory import (
    ApproximateTokenEstimator,
    DeterministicWorkingMemorySelector,
    TokenEstimator,
    WorkingMemorySelector,
)
from cogkura.exceptions import CandidateSetTooLargeError, ValidationError
from cogkura.mappers.base import ObservationMapper
from cogkura.models import (
    ActivationCandidate,
    ActivationConfig,
    ActivationReferenceKind,
    ActivationReferenceTrace,
    EpisodeEncodingResult,
    ForgettingConfig,
    ForgettingResult,
    LearnedAssociation,
    LearningConfig,
    LearningFeedback,
    LearningOutcome,
    LearningResult,
    MemoryIdentity,
    MemoryKind,
    MemoryReference,
    MemoryRetentionState,
    RecallResult,
    RetrievalCue,
    SemanticConsolidationResult,
    SemanticMemoryStatus,
    StoredEpisode,
    StoredMemoryLearningState,
    StoredSemanticMemory,
    StoredSemanticRevision,
    WorkingMemoryConfig,
    WorkingMemorySnapshot,
)
from cogkura.observations.models import IngestionResult, IngestStatus, ObservationInput
from cogkura.observations.pipeline import ObservationPipeline
from cogkura.observations.policies import DefaultObservationPolicy, ObservationPolicy
from cogkura.observations.retention import ObservationRetentionMode
from cogkura.sources.base import SourceConnector
from cogkura.storage import (
    ActivationStore,
    CheckpointStore,
    EpisodeStore,
    LearningStore,
    MemoryDynamicsStore,
    ObservationStore,
    SemanticMemoryStore,
)
from cogkura.storage.in_memory_activation import InMemoryActivationStore
from cogkura.storage.in_memory_dynamics import InMemoryMemoryDynamicsStore
from cogkura.storage.in_memory_episode import InMemoryEpisodeStore
from cogkura.storage.in_memory_learning import InMemoryLearningStore
from cogkura.storage.in_memory_observation import InMemoryCheckpointStore, InMemoryObservationStore
from cogkura.storage.in_memory_semantic import InMemorySemanticMemoryStore

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
        dynamics_store: MemoryDynamicsStore | None = None,
        learning_store: LearningStore | None = None,
        learning_processor: LearningProcessor | None = None,
        learning_config: LearningConfig | None = None,
        episodic_encoder: EpisodicEncoder | None = None,
        semantic_extractor: SemanticExtractor | None = None,
        semantic_consolidator: SemanticConsolidator | None = None,
        semantic_reconciler: SemanticReconciler | None = None,
        declarative_activator: DeclarativeActivator | None = None,
        forgetting_evaluator: ForgettingEvaluator | None = None,
        activation_config: ActivationConfig | None = None,
        forgetting_config: ForgettingConfig | None = None,
        working_memory_selector: WorkingMemorySelector | None = None,
        working_memory_config: WorkingMemoryConfig | None = None,
        token_estimator: TokenEstimator | None = None,
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
        self._dynamics_store = (
            dynamics_store if dynamics_store is not None else InMemoryMemoryDynamicsStore()
        )
        self._learning_store = (
            learning_store if learning_store is not None else InMemoryLearningStore()
        )
        self._learning_processor = (
            learning_processor
            if learning_processor is not None
            else DeterministicLearningProcessor()
        )
        self._learning_config = learning_config if learning_config is not None else LearningConfig()
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
        self._semantic_reconciler = (
            semantic_reconciler
            if semantic_reconciler is not None
            else DeterministicSemanticReconciler()
        )
        self._declarative_activator = (
            declarative_activator
            if declarative_activator is not None
            else ACTRDeclarativeActivator()
        )
        self._forgetting_evaluator = (
            forgetting_evaluator
            if forgetting_evaluator is not None
            else EbbinghausForgettingEvaluator()
        )
        self._activation_config = (
            activation_config if activation_config is not None else ActivationConfig()
        )
        self._forgetting_config = (
            forgetting_config if forgetting_config is not None else ForgettingConfig()
        )
        self._working_memory_selector = (
            working_memory_selector
            if working_memory_selector is not None
            else DeterministicWorkingMemorySelector()
        )
        self._working_memory_config = (
            working_memory_config if working_memory_config is not None else WorkingMemoryConfig()
        )
        self._token_estimator = (
            token_estimator if token_estimator is not None else ApproximateTokenEstimator()
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
        valid_at: datetime | None = None,
        semantic_statuses: frozenset[SemanticMemoryStatus] | None = None,
        include_forgotten: bool = False,
    ) -> list[RecallResult]:
        """Recall episodic and semantic memories by declarative activation."""
        if not tenant_id.strip():
            raise ValidationError("tenant_id must not be empty.")
        if limit <= 0:
            raise ValidationError("Limit must be greater than zero.")
        if valid_at is not None and valid_at.tzinfo is None:
            raise ValidationError("valid_at must be timezone-aware.")

        cue = _normalise_cue(query, subject_id=subject_id)
        evaluation_time = _evaluation_time(as_of)

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
                valid_at=valid_at,
            ),
        )
        if valid_at is None:
            eligible_semantics = [
                memory
                for memory in semantic_memories
                if memory.status is not SemanticMemoryStatus.SUPERSEDED
                and (semantic_statuses is None or memory.status in semantic_statuses)
            ]
        else:
            eligible_semantics = [
                memory
                for memory in semantic_memories
                if semantic_statuses is None or memory.status in semantic_statuses
            ]
        candidates = [activation_candidate_from_episode(episode) for episode in episodes] + [
            activation_candidate_from_semantic(memory) for memory in eligible_semantics
        ]
        candidates = await self._filter_recallable_candidates(
            candidates=candidates,
            tenant_id=tenant_id,
            include_forgotten=include_forgotten,
        )
        if len(candidates) > self._activation_config.max_candidates:
            raise CandidateSetTooLargeError(
                f"Candidate set size {len(candidates)} exceeds max_candidates "
                f"{self._activation_config.max_candidates}."
            )

        identities = [candidate.identity for candidate in candidates]
        references, learned_associations = await asyncio.gather(
            self._list_activation_traces(
                tenant_id=tenant_id,
                identities=identities,
                before_or_at=evaluation_time,
            ),
            self._load_learned_associations(
                tenant_id=tenant_id,
                identities=identities,
            ),
        )
        return self._declarative_activator.rank(
            candidates=candidates,
            cue=cue,
            references=references,
            as_of=evaluation_time,
            config=self._activation_config,
            limit=limit,
            learned_associations=learned_associations,
        )

    async def select_working_memory(
        self,
        query: str | RetrievalCue,
        *,
        tenant_id: str,
        subject_id: str | None = None,
        goal: str | RetrievalCue | None = None,
        previous: WorkingMemorySnapshot | None = None,
        prompt_budget_tokens: int | None = None,
        as_of: datetime | None = None,
        valid_at: datetime | None = None,
        semantic_statuses: frozenset[SemanticMemoryStatus] | None = None,
        include_forgotten: bool = False,
    ) -> WorkingMemorySnapshot:
        """Select a bounded working-memory set from declarative recall candidates."""
        if not tenant_id.strip():
            raise ValidationError("tenant_id must not be empty.")
        if prompt_budget_tokens is not None and prompt_budget_tokens <= 0:
            raise ValidationError("prompt_budget_tokens must be greater than zero.")

        query_cue = _normalise_cue(query, subject_id=subject_id)
        if goal is None:
            goal_cue = query_cue
        elif isinstance(goal, RetrievalCue):
            goal_cue = goal
        else:
            goal_cue = RetrievalCue(text=goal)

        evaluation_time = _evaluation_time(as_of)
        config = self._working_memory_config

        results = await self.recall(
            query,
            tenant_id=tenant_id,
            subject_id=subject_id,
            limit=config.candidate_pool_size,
            as_of=evaluation_time,
            valid_at=valid_at,
            semantic_statuses=semantic_statuses,
            include_forgotten=include_forgotten,
        )

        learning_utilities = None
        if self._learning_config.enabled and results:
            context_key = learning_context_key(goal_cue)
            identities = [_identity_from_recall(result) for result in results]
            states = await self._learning_store.list_states(
                tenant_id=tenant_id,
                identities=identities,
                context_keys=("global", context_key),
            )
            learning_utilities = build_learning_utilities(
                identities=identities,
                states=states,
                context_key=context_key,
                config=self._learning_config,
            )

        return self._working_memory_selector.select(
            candidates=results,
            goal=goal_cue,
            tenant_id=tenant_id,
            subject_id=subject_id,
            previous=previous,
            as_of=evaluation_time,
            config=config,
            token_estimator=self._token_estimator,
            prompt_budget_tokens=prompt_budget_tokens,
            learning_utilities=learning_utilities,
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
        await self._dynamics_store.reactivate(
            tenant_id=tenant_id,
            identities=[
                MemoryIdentity(
                    memory_kind=result.memory_kind,
                    memory_key=_memory_key_from_result(result),
                )
                for result in results
            ],
            at=timestamp,
        )

    async def apply_forgetting(
        self,
        *,
        tenant_id: str,
        subject_id: str | None = None,
        as_of: datetime | None = None,
    ) -> ForgettingResult:
        """Evaluate forgetting lifecycle state for durable memories."""
        if not tenant_id.strip():
            raise ValidationError("tenant_id must not be empty.")
        if not self._forgetting_config.enabled:
            return ForgettingResult()

        evaluation_time = _evaluation_time(as_of)
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
        candidates = [activation_candidate_from_episode(episode) for episode in episodes] + [
            activation_candidate_from_semantic(memory)
            for memory in semantic_memories
            if memory.status is not SemanticMemoryStatus.SUPERSEDED
        ]
        if not candidates:
            return ForgettingResult()

        identities = [candidate.identity for candidate in candidates]
        dynamics, references = await asyncio.gather(
            self._dynamics_store.get_many(tenant_id=tenant_id, identities=identities),
            self._list_activation_traces(
                tenant_id=tenant_id,
                identities=identities,
                before_or_at=evaluation_time,
            ),
        )

        decisions = [
            self._forgetting_evaluator.evaluate(
                candidate=candidate,
                references=references.get(candidate.identity, ()),
                previous=dynamics.get(candidate.identity),
                as_of=evaluation_time,
                activation_config=self._activation_config,
                forgetting_config=self._forgetting_config,
                tenant_id=tenant_id,
            )
            for candidate in candidates
        ]
        await self._dynamics_store.upsert_many([decision.dynamics for decision in decisions])

        compaction_result = None
        if self._forgetting_config.enable_reference_compaction:
            compact_before = evaluation_time - timedelta(
                seconds=self._forgetting_config.compact_after_seconds
            )
            compaction_result = await self._activation_store.compact_references(
                tenant_id=tenant_id,
                before=compact_before,
                bucket_seconds=self._forgetting_config.compaction_bucket_seconds,
            )

        active = fading = forgotten = reactivated = 0
        for decision in decisions:
            if decision.dynamics.retention_state is MemoryRetentionState.ACTIVE:
                active += 1
            elif decision.dynamics.retention_state is MemoryRetentionState.FADING:
                fading += 1
            else:
                forgotten += 1
            if decision.reactivated:
                reactivated += 1

        return ForgettingResult(
            evaluated=len(decisions),
            active=active,
            fading=fading,
            forgotten=forgotten,
            reactivated=reactivated,
            references_compacted=(
                compaction_result.references_compacted if compaction_result else 0
            ),
        )

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
        revision_candidates = self._semantic_consolidator.consolidate(
            episodes,
            extraction.candidates,
        )
        existing_memories = await self._semantic_store.list(
            tenant_id=tenant_id,
            subject_id=subject_id,
            include_inactive=True,
        )
        existing_revisions = await self._semantic_store.list_revisions(
            tenant_id=tenant_id,
            subject_id=subject_id,
        )
        plan = self._semantic_reconciler.reconcile(
            candidates=revision_candidates,
            existing_memories=existing_memories,
            existing_revisions=existing_revisions,
            as_of=datetime.now(UTC),
        )
        write_result = await self._semantic_store.apply_reconciliation(plan)
        result = SemanticConsolidationResult(
            episodes=len(episodes),
            extracted_candidates=len(extraction.candidates),
            extracted_failures=extraction.failed,
            canonical_claims=len(revision_candidates),
            promoted=len(plan.current_memories),
            created=write_result.created,
            updated=write_result.updated,
            unchanged=write_result.unchanged,
            contested=sum(
                1
                for memory in plan.current_memories
                if memory.status is SemanticMemoryStatus.CONTESTED
            ),
        ).with_reconciliation(
            reinforced=plan.reinforced_count,
            coexisting=plan.coexist_count,
            conflicts=plan.conflict_count,
            superseded=plan.superseded_count,
            revisions_created=plan.revisions_created,
            revisions_updated=plan.revisions_updated,
        )
        return result

    async def list_semantic_revisions(
        self,
        *,
        tenant_id: str,
        memory_key: str | None = None,
        subject_id: str | None = None,
        valid_at: datetime | None = None,
        limit: int | None = None,
    ) -> list[StoredSemanticRevision]:
        """List semantic revision history for a tenant."""
        if not tenant_id.strip():
            raise ValidationError("tenant_id must not be empty.")
        if limit is not None and limit <= 0:
            raise ValidationError("Limit must be greater than zero.")
        if valid_at is not None and valid_at.tzinfo is None:
            raise ValidationError("valid_at must be timezone-aware.")
        return list(
            await self._semantic_store.list_revisions(
                tenant_id=tenant_id,
                memory_key=memory_key,
                subject_id=subject_id,
                valid_at=valid_at,
                limit=limit,
            )
        )

    async def list_semantic_memories(
        self,
        *,
        tenant_id: str,
        subject_id: str | None = None,
        include_inactive: bool = False,
        status: SemanticMemoryStatus | None = None,
        limit: int | None = None,
        valid_at: datetime | None = None,
    ) -> list[StoredSemanticMemory]:
        """List consolidated semantic memories for a tenant."""
        if not tenant_id.strip():
            raise ValidationError("tenant_id must not be empty.")
        if limit is not None and limit <= 0:
            raise ValidationError("Limit must be greater than zero.")
        if valid_at is not None and valid_at.tzinfo is None:
            raise ValidationError("valid_at must be timezone-aware.")
        return list(
            await self._semantic_store.list(
                tenant_id=tenant_id,
                subject_id=subject_id,
                include_inactive=include_inactive,
                status=status,
                limit=limit,
                valid_at=valid_at,
            )
        )

    async def learn(self, feedback: LearningFeedback) -> LearningResult:
        """Apply outcome-driven learning from application feedback."""
        if not self._learning_config.enabled:
            return LearningResult()
        await self._validate_learning_targets(feedback)
        plan = self._learning_processor.plan(
            feedback=feedback,
            config=self._learning_config,
        )
        write_result = await self._learning_store.apply(plan)
        if write_result.unchanged:
            return LearningResult(
                created=False,
                unchanged=True,
                association_items_skipped=plan.association_items_skipped,
            )

        helpful_identities = [
            item.identity for item in plan.items if item.outcome is LearningOutcome.HELPFUL
        ]
        reactivated = 0
        if helpful_identities:
            dynamics_before = await self._dynamics_store.get_many(
                tenant_id=plan.tenant_id,
                identities=helpful_identities,
            )
            await self._dynamics_store.reactivate(
                tenant_id=plan.tenant_id,
                identities=helpful_identities,
                at=plan.occurred_at,
            )
            reactivated = sum(
                1
                for identity in helpful_identities
                if dynamics_before.get(identity) is not None
                and dynamics_before[identity].retention_state
                in {MemoryRetentionState.FADING, MemoryRetentionState.FORGOTTEN}
            )

        return LearningResult(
            created=write_result.created,
            unchanged=write_result.unchanged,
            helpful=write_result.helpful,
            unhelpful=write_result.unhelpful,
            incorrect=write_result.incorrect,
            memories_reinforced=write_result.helpful,
            associations_reinforced=write_result.associations_reinforced,
            association_items_skipped=plan.association_items_skipped,
            reactivated=reactivated,
        )

    async def list_learning_state(
        self,
        *,
        tenant_id: str,
        identities: Sequence[MemoryIdentity] | None = None,
        goal: RetrievalCue | None = None,
    ) -> list[StoredMemoryLearningState]:
        """List persisted learning counts for inspection and debugging."""
        if not tenant_id.strip():
            raise ValidationError("tenant_id must not be empty.")

        target_identities = list(identities or ())
        if not target_identities:
            episodes, semantic_memories = await asyncio.gather(
                self._episode_store.list(
                    tenant_id=tenant_id,
                    include_inactive=False,
                ),
                self._semantic_store.list(
                    tenant_id=tenant_id,
                    include_inactive=False,
                ),
            )
            target_identities = [
                MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key=episode.memory_key)
                for episode in episodes
            ] + [
                MemoryIdentity(
                    memory_kind=MemoryKind.SEMANTIC,
                    memory_key=memory.memory_key,
                )
                for memory in semantic_memories
            ]

        context_key = learning_context_key(goal)
        context_keys = ("global",) if context_key == "global" else ("global", context_key)
        return list(
            await self._learning_store.list_states(
                tenant_id=tenant_id,
                identities=target_identities,
                context_keys=context_keys,
            )
        )

    async def clear(self, *, tenant_id: str) -> None:
        """Clear learning, activation, dynamics, semantic memories, episodes, and observations."""
        if not tenant_id.strip():
            raise ValidationError("tenant_id must not be empty.")
        await self._learning_store.clear(tenant_id=tenant_id)
        await self._activation_store.clear(tenant_id=tenant_id)
        await self._dynamics_store.clear(tenant_id=tenant_id)
        await self._semantic_store.clear(tenant_id=tenant_id)
        await self._episode_store.clear(tenant_id=tenant_id)
        await self._observation_store.clear(tenant_id=tenant_id)

    async def _filter_recallable_candidates(
        self,
        *,
        candidates: list[ActivationCandidate],
        tenant_id: str,
        include_forgotten: bool,
    ) -> list[ActivationCandidate]:
        if (
            not self._forgetting_config.enabled
            or include_forgotten
            or not self._forgetting_config.exclude_forgotten_from_recall
        ):
            return candidates

        identities = [candidate.identity for candidate in candidates]
        dynamics = await self._dynamics_store.get_many(
            tenant_id=tenant_id,
            identities=identities,
        )
        return [
            candidate
            for candidate in candidates
            if dynamics.get(candidate.identity) is None
            or dynamics[candidate.identity].retention_state is not MemoryRetentionState.FORGOTTEN
        ]

    async def _list_activation_traces(
        self,
        *,
        tenant_id: str,
        identities: Sequence[MemoryIdentity],
        before_or_at: datetime,
    ) -> Mapping[MemoryIdentity, tuple[ActivationReferenceTrace, ...]]:
        access_traces, learning_traces = await asyncio.gather(
            self._activation_store.list_reference_traces(
                tenant_id=tenant_id,
                identities=identities,
                before_or_at=before_or_at,
            ),
            self._learning_store.list_reinforcement_traces(
                tenant_id=tenant_id,
                identities=identities,
                before_or_at=before_or_at,
            ),
        )
        merged: dict[MemoryIdentity, list[ActivationReferenceTrace]] = {}
        for source in (access_traces, learning_traces):
            for identity, traces in source.items():
                merged.setdefault(identity, []).extend(traces)
        return {
            identity: tuple(sorted(traces, key=lambda trace: trace.referenced_at))
            for identity, traces in merged.items()
        }

    async def _load_learned_associations(
        self,
        *,
        tenant_id: str,
        identities: Sequence[MemoryIdentity],
    ) -> tuple[LearnedAssociation, ...]:
        if not self._learning_config.enabled or not identities:
            return ()
        stored = await self._learning_store.list_associations(
            tenant_id=tenant_id,
            identities=identities,
        )
        learned: list[LearnedAssociation] = []
        for association in stored:
            strength = calculate_association_strength(
                association.coactivation_count,
                config=self._learning_config,
            )
            if strength <= 0.0:
                continue
            learned.append(
                LearnedAssociation(
                    left=association.left,
                    right=association.right,
                    strength=strength,
                    coactivation_count=association.coactivation_count,
                )
            )
        return tuple(learned)

    async def _validate_learning_targets(self, feedback: LearningFeedback) -> None:
        episodes, semantic_memories = await asyncio.gather(
            self._episode_store.list(
                tenant_id=feedback.tenant_id,
                subject_id=feedback.subject_id,
                include_inactive=False,
            ),
            self._semantic_store.list(
                tenant_id=feedback.tenant_id,
                subject_id=feedback.subject_id,
                include_inactive=False,
            ),
        )
        episodes_by_key = {episode.memory_key: episode for episode in episodes}
        semantics_by_key = {memory.memory_key: memory for memory in semantic_memories}
        revision_keys: set[str] = {
            item.revision_key
            for item in feedback.items
            if item.revision_key is not None and item.revision_key.strip()
        }
        revisions_by_key: dict[str, StoredSemanticRevision] = {}
        if revision_keys:
            revisions = await self._semantic_store.list_revisions(
                tenant_id=feedback.tenant_id,
                subject_id=feedback.subject_id,
            )
            revisions_by_key = {revision.revision_key: revision for revision in revisions}

        for item in feedback.items:
            identity = item.identity
            if identity.memory_kind is MemoryKind.EPISODE:
                episode = episodes_by_key.get(identity.memory_key)
                if episode is None:
                    raise ValidationError(
                        f"Unknown episode memory_key {identity.memory_key!r} for tenant."
                    )
                if feedback.subject_id is not None and episode.subject_id != feedback.subject_id:
                    raise ValidationError(
                        f"Episode {identity.memory_key!r} subject_id does not match feedback."
                    )
                if item.revision_key is not None:
                    raise ValidationError("revision_key is only valid for semantic memories.")
                continue

            memory = semantics_by_key.get(identity.memory_key)
            if memory is None:
                raise ValidationError(
                    f"Unknown semantic memory_key {identity.memory_key!r} for tenant."
                )
            if feedback.subject_id is not None and memory.subject_id != feedback.subject_id:
                raise ValidationError(
                    f"Semantic memory {identity.memory_key!r} subject_id does not match feedback."
                )
            if item.revision_key is not None:
                revision = revisions_by_key.get(item.revision_key)
                if revision is None:
                    raise ValidationError(f"Unknown revision_key {item.revision_key!r} for tenant.")
                if revision.memory_key != identity.memory_key:
                    raise ValidationError(
                        f"revision_key {item.revision_key!r} does not belong to "
                        f"memory_key {identity.memory_key!r}."
                    )


def _identity_from_recall(result: RecallResult) -> MemoryIdentity:
    memory = result.memory
    if isinstance(memory, StoredEpisode):
        return MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key=memory.memory_key)
    return MemoryIdentity(memory_kind=MemoryKind.SEMANTIC, memory_key=memory.memory_key)


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


def _evaluation_time(as_of: datetime | None) -> datetime:
    if as_of is not None:
        if as_of.tzinfo is None:
            raise ValidationError("as_of must be timezone-aware.")
        return as_of.astimezone(UTC)
    return datetime.now(UTC)
