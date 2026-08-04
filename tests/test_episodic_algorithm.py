"""Unit tests for deterministic episodic encoding."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import MappingProxyType

import pytest

from cognema.algorithms.episodic import DeterministicEpisodicEncoder
from cognema.exceptions import ValidationError
from cognema.observations.models import StoredObservation


def _stored(
    *,
    obs_id: str,
    tenant_id: str = "company_123",
    subject_id: str | None = "user_1",
    actor_id: str | None = "user_1",
    source_namespace: str = "public.messages",
    content: str = "Hello world.",
    observed_at: datetime | None = None,
    metadata: dict | None = None,
    is_deleted: bool = False,
    attention_score: float = 0.5,
    current_revision: int = 1,
) -> StoredObservation:
    return StoredObservation(
        id=obs_id,
        tenant_id=tenant_id,
        subject_id=subject_id,
        actor_id=actor_id,
        source_type="postgres",
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
        current_revision=current_revision,
        is_deleted=is_deleted,
        attention_score=attention_score,
    )


def test_encoder_constructor_validation() -> None:
    with pytest.raises(ValidationError, match="maximum_gap_seconds"):
        DeterministicEpisodicEncoder(maximum_gap_seconds=0)
    with pytest.raises(ValidationError, match="grouping_metadata_keys"):
        DeterministicEpisodicEncoder(grouping_metadata_keys=())
    with pytest.raises(ValidationError, match="maximum_statement_length"):
        DeterministicEpisodicEncoder(maximum_statement_length=0)
    with pytest.raises(ValidationError, match="encoding_version"):
        DeterministicEpisodicEncoder(encoding_version=" ")


def test_single_observation_creates_one_episode() -> None:
    encoder = DeterministicEpisodicEncoder()
    episodes = encoder.encode([_stored(obs_id="obs-1")])
    assert len(episodes) == 1
    assert len(episodes[0].evidence) == 1


def test_conversation_id_groups_observations() -> None:
    encoder = DeterministicEpisodicEncoder()
    base = datetime(2026, 8, 4, 10, 0, tzinfo=UTC)
    episodes = encoder.encode(
        [
            _stored(
                obs_id="obs-1",
                observed_at=base,
                metadata={"conversation_id": "conv-1"},
            ),
            _stored(
                obs_id="obs-2",
                observed_at=base + timedelta(hours=2),
                metadata={"conversation_id": "conv-1"},
            ),
        ]
    )
    assert len(episodes) == 1
    assert len(episodes[0].evidence) == 2


def test_different_conversations_do_not_merge() -> None:
    encoder = DeterministicEpisodicEncoder()
    episodes = encoder.encode(
        [
            _stored(obs_id="obs-1", metadata={"conversation_id": "conv-1"}),
            _stored(obs_id="obs-2", metadata={"conversation_id": "conv-2"}),
        ]
    )
    assert len(episodes) == 2


def test_time_gap_creates_boundary() -> None:
    encoder = DeterministicEpisodicEncoder(maximum_gap_seconds=1800)
    base = datetime(2026, 8, 4, 10, 0, tzinfo=UTC)
    episodes = encoder.encode(
        [
            _stored(obs_id="obs-1", observed_at=base),
            _stored(obs_id="obs-2", observed_at=base + timedelta(minutes=31)),
        ]
    )
    assert len(episodes) == 2


def test_deleted_observations_are_ignored() -> None:
    encoder = DeterministicEpisodicEncoder()
    episodes = encoder.encode(
        [
            _stored(obs_id="obs-1"),
            _stored(obs_id="obs-2", is_deleted=True),
        ]
    )
    assert len(episodes) == 1
    assert len(episodes[0].evidence) == 1


def test_episode_boundary_and_terminal_event() -> None:
    encoder = DeterministicEpisodicEncoder()
    episodes = encoder.encode(
        [
            _stored(obs_id="obs-1", metadata={"conversation_id": "conv-1"}),
            _stored(
                obs_id="obs-2",
                metadata={"conversation_id": "conv-1", "terminal_event": True},
            ),
            _stored(
                obs_id="obs-3",
                metadata={"conversation_id": "conv-1", "episode_boundary": True},
            ),
        ]
    )
    assert len(episodes) == 2


def test_statement_is_chronological_and_deduplicates_adjacent() -> None:
    encoder = DeterministicEpisodicEncoder()
    episodes = encoder.encode(
        [
            _stored(obs_id="obs-1", content="First line."),
            _stored(obs_id="obs-2", content="First line."),
            _stored(obs_id="obs-3", content="Second line."),
        ]
    )
    assert episodes[0].statement == "First line.\nSecond line."


def test_structural_statement_when_no_content() -> None:
    encoder = DeterministicEpisodicEncoder()
    observation = _stored(obs_id="obs-1", content="")
    object.__setattr__(observation, "content", None)
    episodes = encoder.encode([observation])
    assert "Episode containing 1 message observation" in episodes[0].statement


def test_salience_uses_attention_and_terminal_event() -> None:
    encoder = DeterministicEpisodicEncoder()
    episodes = encoder.encode(
        [
            _stored(obs_id="obs-1", attention_score=0.8),
            _stored(
                obs_id="obs-2",
                attention_score=0.6,
                metadata={"terminal_event": True, "importance": 0.5},
            ),
        ]
    )
    salience = episodes[0].metadata["salience"]
    assert salience["version"] == "salience-v1"
    assert episodes[0].importance == pytest.approx(salience["score"])


def test_entities_include_subject_actor_and_metadata() -> None:
    encoder = DeterministicEpisodicEncoder()
    episodes = encoder.encode(
        [
            _stored(
                obs_id="obs-1",
                metadata={"entity_ids": ["customer-42", "redis"]},
            )
        ]
    )
    roles = {entity.role for entity in episodes[0].entities}
    assert roles == {"subject", "actor", "metadata"}


def test_encoding_is_deterministic() -> None:
    encoder = DeterministicEpisodicEncoder()
    observations = [
        _stored(obs_id="obs-1", metadata={"conversation_id": "conv-1"}),
        _stored(obs_id="obs-2", metadata={"conversation_id": "conv-1"}),
    ]
    first = encoder.encode(observations)
    second = encoder.encode(list(reversed(observations)))
    assert first[0].memory_key == second[0].memory_key
    assert first[0].statement == second[0].statement


def test_exact_gap_threshold_stays_in_same_episode() -> None:
    encoder = DeterministicEpisodicEncoder(maximum_gap_seconds=1800)
    base = datetime(2026, 8, 4, 10, 0, tzinfo=UTC)
    episodes = encoder.encode(
        [
            _stored(obs_id="obs-1", observed_at=base),
            _stored(obs_id="obs-2", observed_at=base + timedelta(seconds=1800)),
        ]
    )
    assert len(episodes) == 1


def test_different_tenants_subjects_and_namespaces_do_not_merge() -> None:
    encoder = DeterministicEpisodicEncoder()
    base = datetime(2026, 8, 4, 10, 0, tzinfo=UTC)
    episodes = encoder.encode(
        [
            _stored(obs_id="obs-1", tenant_id="t1", observed_at=base),
            _stored(obs_id="obs-2", tenant_id="t2", observed_at=base),
            _stored(obs_id="obs-3", subject_id="user_a", observed_at=base),
            _stored(obs_id="obs-4", subject_id="user_b", observed_at=base),
            _stored(
                obs_id="obs-5",
                source_namespace="public.messages",
                observed_at=base,
            ),
            _stored(
                obs_id="obs-6",
                source_namespace="public.tickets",
                observed_at=base,
            ),
        ]
    )
    assert len(episodes) == 6


def test_custom_grouping_key_spans_time_gap() -> None:
    encoder = DeterministicEpisodicEncoder(
        grouping_metadata_keys=("meeting_id",),
        maximum_gap_seconds=60,
    )
    base = datetime(2026, 8, 4, 10, 0, tzinfo=UTC)
    episodes = encoder.encode(
        [
            _stored(
                obs_id="obs-1",
                observed_at=base,
                metadata={"meeting_id": "m-1"},
            ),
            _stored(
                obs_id="obs-2",
                observed_at=base + timedelta(hours=3),
                metadata={"meeting_id": "m-1"},
            ),
        ]
    )
    assert len(episodes) == 1
    assert episodes[0].metadata["episode"]["segmentation_key"] == "meeting_id"


def test_statement_truncation_is_deterministic() -> None:
    encoder = DeterministicEpisodicEncoder(maximum_statement_length=20)
    episodes = encoder.encode([_stored(obs_id="obs-1", content="ABCDEFGHIJKLMNOPQRSTUVWXYZ")])
    assert episodes[0].statement == "ABCDEFGHIJKLMNOPQRST"
    assert len(episodes[0].statement) == 20


def test_invalid_metadata_importance_is_ignored() -> None:
    encoder = DeterministicEpisodicEncoder()
    episodes = encoder.encode(
        [
            _stored(
                obs_id="obs-1",
                attention_score=0.5,
                metadata={"importance": 2.0},
            )
        ]
    )
    salience = episodes[0].metadata["salience"]
    assert salience["explicit_importance"] == 0.0
    expected = 0.50 * 0.5 + 0.30 * 0.5
    assert episodes[0].importance == pytest.approx(expected)
