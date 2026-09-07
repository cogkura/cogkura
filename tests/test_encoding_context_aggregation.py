"""Tests for encoding-context aggregation during episodic encoding."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import MappingProxyType

from cogkura.algorithms.episodic import DeterministicEpisodicEncoder
from cogkura.observations.encoding_context import ObservationContext
from cogkura.observations.models import StoredObservation


def _stored(
    *,
    obs_id: str,
    content: str = "Observation content.",
    metadata: dict | None = None,
    context: ObservationContext | None = None,
    subject_id: str = "developer-1",
    source_namespace: str = "github",
    source_type: str = "discussion",
    observed_at: datetime | None = None,
) -> StoredObservation:
    return StoredObservation(
        id=obs_id,
        tenant_id="acme",
        subject_id=subject_id,
        actor_id=None,
        source_type=source_type,
        source_namespace=source_namespace,
        source_record_id=f"record-{obs_id}",
        source_version="v1",
        event_type="message",
        content=content,
        content_hash=f"hash-{obs_id}",
        metadata=MappingProxyType(metadata or {}),
        source_created_at=None,
        source_updated_at=None,
        observed_at=observed_at or datetime(2026, 8, 4, 10, 0, tzinfo=UTC),
        current_revision=1,
        is_deleted=False,
        context=context or ObservationContext(),
    )


def test_single_observation_populates_encoding_context() -> None:
    encoder = DeterministicEpisodicEncoder()
    episodes = encoder.encode(
        [
            _stored(
                obs_id="obs-1",
                metadata={"entity_ids": ("redis", "payments-api")},
                context=ObservationContext(
                    conversation_id="arch-42",
                    thread_id="queue-selection",
                    goal="reduce-operational-complexity",
                    activity="architecture-decision",
                    domain="payments-api",
                    temporal_context=("queue-redesign",),
                ),
            )
        ]
    )
    signature = episodes[0].encoding_context
    assert signature.subject_ids == ("developer-1",)
    assert signature.conversation_ids == ("arch-42",)
    assert signature.thread_ids == ("queue-selection",)
    assert signature.goals == ("reduce-operational-complexity",)
    assert signature.activities == ("architecture-decision",)
    assert signature.domains == ("payments-api",)
    assert signature.source_namespaces == ("github",)
    assert signature.source_types == ("discussion",)
    assert signature.entity_ids == ("payments-api", "redis")
    assert signature.temporal_contexts == ("queue-redesign",)
    assert signature.concept_ids == ()


def test_multiple_observations_union_goals() -> None:
    encoder = DeterministicEpisodicEncoder()
    base = datetime(2026, 8, 4, 10, 0, tzinfo=UTC)
    episodes = encoder.encode(
        [
            _stored(
                obs_id="obs-1",
                observed_at=base,
                metadata={"conversation_id": "arch-42"},
                context=ObservationContext(
                    domain="payments-api",
                    activity="architecture",
                    goal="evaluate-queue-technology",
                ),
            ),
            _stored(
                obs_id="obs-2",
                observed_at=base + timedelta(minutes=5),
                metadata={"conversation_id": "arch-42"},
                context=ObservationContext(
                    domain="payments-api",
                    activity="architecture",
                    goal="reduce-operational-complexity",
                ),
            ),
        ]
    )
    assert len(episodes) == 1
    signature = episodes[0].encoding_context
    assert signature.conversation_ids == ("arch-42",)
    assert signature.domains == ("payments-api",)
    assert signature.activities == ("architecture",)
    assert signature.goals == (
        "evaluate-queue-technology",
        "reduce-operational-complexity",
    )


def test_missing_context_keeps_first_class_fields_only() -> None:
    encoder = DeterministicEpisodicEncoder()
    episodes = encoder.encode([_stored(obs_id="obs-1", metadata={"entity_ids": ("redis",)})])
    signature = episodes[0].encoding_context
    assert signature.goals == ()
    assert signature.domains == ()
    assert signature.subject_ids == ("developer-1",)
    assert signature.entity_ids == ("redis",)
    assert signature.source_namespaces == ("github",)


def test_prose_does_not_infer_context() -> None:
    encoder = DeterministicEpisodicEncoder()
    episodes = encoder.encode(
        [
            _stored(
                obs_id="obs-1",
                content=(
                    "While working on payments-api we decided not to use Redis "
                    "because we wanted fewer operational dependencies."
                ),
            )
        ]
    )
    signature = episodes[0].encoding_context
    assert signature.goals == ()
    assert signature.domains == ()
    assert signature.activities == ()


def test_metadata_conversation_id_derived_into_signature() -> None:
    encoder = DeterministicEpisodicEncoder()
    episodes = encoder.encode(
        [_stored(obs_id="obs-1", metadata={"conversation_id": "arch-legacy"})]
    )
    assert episodes[0].encoding_context.conversation_ids == ("arch-legacy",)


def test_context_only_conversation_id_does_not_change_grouping() -> None:
    encoder = DeterministicEpisodicEncoder()
    base = datetime(2026, 8, 4, 10, 0, tzinfo=UTC)
    without_context = encoder.encode(
        [
            _stored(obs_id="obs-1", observed_at=base),
            _stored(obs_id="obs-2", observed_at=base + timedelta(minutes=5)),
        ]
    )
    with_context = encoder.encode(
        [
            _stored(
                obs_id="obs-1",
                observed_at=base,
                context=ObservationContext(conversation_id="arch-42"),
            ),
            _stored(
                obs_id="obs-2",
                observed_at=base + timedelta(minutes=5),
                context=ObservationContext(conversation_id="arch-42"),
            ),
        ]
    )
    assert len(without_context) == len(with_context)
    assert [len(episode.evidence) for episode in without_context] == [
        len(episode.evidence) for episode in with_context
    ]
