"""Public memory API."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from cognema.algorithms.episodic import DeterministicEpisodicEncoder, EpisodicEncoder
from cognema.exceptions import ValidationError
from cognema.mappers.base import ObservationMapper
from cognema.models import EpisodeEncodingResult, RecallResult, StoredEpisode
from cognema.observations.models import IngestionResult, IngestStatus, ObservationInput
from cognema.observations.pipeline import ObservationPipeline
from cognema.observations.policies import DefaultObservationPolicy, ObservationPolicy
from cognema.observations.retention import ObservationRetentionMode
from cognema.sources.base import SourceConnector
from cognema.storage import CheckpointStore, EpisodeStore, ObservationStore
from cognema.storage.in_memory_episode import InMemoryEpisodeStore
from cognema.storage.in_memory_observation import InMemoryCheckpointStore, InMemoryObservationStore

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
_DEFAULT_BATCH_SIZE = 500


class Memory:
    """Cognitive memory facade for observation ingestion and recall."""

    def __init__(
        self,
        *,
        observation_store: ObservationStore | None = None,
        checkpoint_store: CheckpointStore | None = None,
        episode_store: EpisodeStore | None = None,
        episodic_encoder: EpisodicEncoder | None = None,
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
        self._episodic_encoder = (
            episodic_encoder if episodic_encoder is not None else DeterministicEpisodicEncoder()
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
        query: str,
        *,
        tenant_id: str,
        subject_id: str | None = None,
        limit: int = 5,
    ) -> list[RecallResult]:
        """Recall observations by tenant-scoped token-overlap scoring.

        Placeholder retrieval until cognitive recall lands. Always requires a tenant.
        """
        if not tenant_id.strip():
            raise ValidationError("tenant_id must not be empty.")
        normalized_query = query.strip()
        if not normalized_query:
            raise ValidationError("Query must not be empty.")
        if limit <= 0:
            raise ValidationError("Limit must be greater than zero.")

        query_tokens = _tokenize(normalized_query)
        if not query_tokens:
            raise ValidationError("Query must contain at least one alphanumeric token.")

        results: list[RecallResult] = []
        for observation in await self._observation_store.list(
            tenant_id=tenant_id,
            subject_id=subject_id,
        ):
            content = observation.content or ""
            observation_tokens = _tokenize(content)
            score, matched_tokens = _score_overlap(query_tokens, observation_tokens)
            if score <= 0.0:
                continue
            reason = f"Matched tokens: {', '.join(matched_tokens)}" if matched_tokens else None
            results.append(RecallResult(observation=observation, score=score, reason=reason))

        results.sort(key=lambda item: (-item.score, item.observation.id))
        return results[:limit]

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

    async def clear(self, *, tenant_id: str) -> None:
        """Clear episodes and observations for a tenant."""
        if not tenant_id.strip():
            raise ValidationError("tenant_id must not be empty.")
        await self._episode_store.clear(tenant_id=tenant_id)
        await self._observation_store.clear(tenant_id=tenant_id)


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_PATTERN.findall(text)}


def _score_overlap(
    query_tokens: set[str],
    observation_tokens: set[str],
) -> tuple[float, tuple[str, ...]]:
    matched_tokens = tuple(sorted(query_tokens.intersection(observation_tokens)))
    if not matched_tokens:
        return 0.0, matched_tokens
    score = len(matched_tokens) / len(query_tokens)
    return score, matched_tokens
