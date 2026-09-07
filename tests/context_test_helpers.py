"""Shared fixtures for contextual-memory behaviour tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import MappingProxyType

from cogkura import Memory, ObservationInput
from cogkura.models import (
    ActivationConfig,
    ActivationReferenceTrace,
    ContextMatch,
    EpisodeEntity,
    EpisodeEvidenceInput,
    MemoryContextSignature,
    MemoryKind,
    SemanticCardinality,
    SemanticDerivationInput,
    SemanticMemoryStatus,
    SemanticPolarity,
    StoredEpisode,
    StoredSemanticMemory,
)
from cogkura.observations.encoding_context import ObservationContext

T = datetime(2026, 8, 4, 10, 0, tzinfo=UTC)
T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
TENANT = "context-observability"
SUBJECT = "developer-1"
QUERY = "Why did we decide not to use Redis?"


def diagnostic_observations() -> list[ObservationInput]:
    return [
        ObservationInput(
            tenant_id=TENANT,
            subject_id=SUBJECT,
            source_namespace="github",
            source_record_id="1",
            source_type="discussion",
            content=(
                "We decided not to use Redis because another stateful dependency "
                "would increase operational complexity."
            ),
            observed_at=T,
            metadata={
                "conversation_id": "arch-42",
                "entity_ids": ("redis", "payments-api"),
            },
            context=ObservationContext(
                conversation_id="arch-42",
                thread_id="queue-selection",
                goal="reduce-operational-complexity",
                activity="architecture-decision",
                domain="payments-api",
                temporal_context=("queue-redesign",),
            ),
        ),
        ObservationInput(
            tenant_id=TENANT,
            subject_id=SUBJECT,
            source_namespace="github",
            source_record_id="2",
            source_type="discussion",
            content="Analytics-api stores metrics without Redis coordination.",
            observed_at=T + timedelta(hours=2),
            metadata={
                "conversation_id": "arch-44",
                "entity_ids": ("redis", "analytics-api"),
            },
            context=ObservationContext(
                conversation_id="arch-44",
                thread_id="queue-selection",
                goal="reduce-operational-complexity",
                activity="architecture-decision",
                domain="analytics-api",
                temporal_context=("queue-redesign",),
            ),
        ),
        ObservationInput(
            tenant_id=TENANT,
            subject_id=SUBJECT,
            source_namespace="github",
            source_record_id="3",
            source_type="discussion",
            content="Auth service uses Redis for session caching.",
            observed_at=T + timedelta(hours=4),
            metadata={
                "conversation_id": "arch-43",
                "entity_ids": ("redis", "auth-api"),
            },
            context=ObservationContext(
                conversation_id="arch-43",
                thread_id="auth-cache",
                goal="improve-session-latency",
                activity="architecture-decision",
                domain="auth-api",
                temporal_context=("session-cache",),
            ),
        ),
    ]


async def build_store(
    observations: list[ObservationInput] | None = None,
    *,
    activation_config: ActivationConfig | None = None,
) -> Memory:
    memory = Memory(activation_config=activation_config)
    for observation in observations or diagnostic_observations():
        await memory.observe(observation)
    await memory.process(tenant_id=TENANT, subject_id=SUBJECT, as_of=T.replace(hour=11))
    return memory


def episode_candidates(inspection) -> list:
    return [
        candidate
        for candidate in (*inspection.returned, *inspection.rejected)
        if candidate.memory_kind is MemoryKind.EPISODE
    ]


def episode(
    *,
    episode_id: str,
    memory_key: str,
    statement: str,
    encoding_context: MemoryContextSignature | None = None,
    entity_ids: tuple[str, ...] = ("redis",),
) -> StoredEpisode:
    return StoredEpisode(
        id=episode_id,
        tenant_id=TENANT,
        subject_id=SUBJECT,
        memory_key=memory_key,
        statement=statement,
        started_at=T0,
        ended_at=T0,
        confidence=0.9,
        importance=0.7,
        is_active=True,
        evidence=(
            EpisodeEvidenceInput(
                observation_id=f"obs-{memory_key}",
                observation_revision=1,
                sequence_number=0,
            ),
        ),
        entities=tuple(EpisodeEntity(entity_id=eid, role="mention") for eid in entity_ids),
        metadata=MappingProxyType({}),
        created_at=T0,
        updated_at=T0,
        encoding_context=encoding_context or MemoryContextSignature(),
    )


def reference(*, weight: int = 1) -> ActivationReferenceTrace:
    return ActivationReferenceTrace(referenced_at=T0, weight=weight)


def semantic(
    *,
    derivations: tuple[SemanticDerivationInput, ...],
    memory_key: str = "service-redis",
) -> StoredSemanticMemory:
    return StoredSemanticMemory(
        id="sem-1",
        tenant_id=TENANT,
        subject_id=SUBJECT,
        memory_key=memory_key,
        slot_key=f"slot:{memory_key}",
        revision_key=f"rev:{memory_key}",
        revision_number=1,
        statement="The service avoids Redis for coordination.",
        subject_entity_id="service",
        predicate="cache",
        object_value="redis",
        object_entity_id=None,
        polarity=SemanticPolarity.AFFIRM,
        cardinality=SemanticCardinality.ONE,
        qualifiers=MappingProxyType({}),
        confidence=0.9,
        importance=0.7,
        status=SemanticMemoryStatus.ACTIVE,
        support_count=len(derivations),
        contradiction_count=0,
        first_supported_at=T0,
        last_supported_at=T0,
        valid_from=None,
        valid_until=None,
        is_active=True,
        derivations=derivations,
        observation_evidence=(),
        entities=(EpisodeEntity(entity_id="redis", role="mention"),),
        metadata=MappingProxyType({}),
        created_at=T0,
        updated_at=T0,
    )


def support_episode(
    *,
    episode_id: str,
    encoding_context: MemoryContextSignature,
) -> StoredEpisode:
    return episode(
        episode_id=episode_id,
        memory_key=f"support-{episode_id}",
        statement=f"Support episode {episode_id}.",
        encoding_context=encoding_context,
    )


def context_match(*, score: float | None, coverage: float = 1.0) -> ContextMatch:
    return ContextMatch(
        dimensions=(),
        attribute_dimensions=(),
        score=score,
        comparable_count=1 if score is not None else 0,
        cue_dimension_count=1,
        cue_coverage=coverage,
    )


@dataclass(slots=True)
class StubMatchCache:
    strengths: dict[str, tuple[ContextMatch | None, float, bool]]

    def match_strength(self, episode_id: str) -> tuple[ContextMatch | None, float, bool]:
        return self.strengths.get(episode_id, (None, 0.0, True))
