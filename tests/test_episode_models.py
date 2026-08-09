"""Unit tests for episode models."""

from __future__ import annotations

from datetime import UTC, datetime
from types import MappingProxyType

import pytest

from cogkura.exceptions import ValidationError
from cogkura.models import (
    EpisodeEntity,
    EpisodeEvidenceInput,
    EpisodeInput,
    EpisodeWriteStatus,
)


def _evidence(
    *,
    observation_id: str = "obs-1",
    observation_revision: int = 1,
    sequence_number: int = 0,
    contribution_score: float = 1.0,
) -> EpisodeEvidenceInput:
    return EpisodeEvidenceInput(
        observation_id=observation_id,
        observation_revision=observation_revision,
        sequence_number=sequence_number,
        contribution_score=contribution_score,
    )


def _episode_input(**overrides: object) -> EpisodeInput:
    defaults = {
        "tenant_id": "company_123",
        "subject_id": "user_1",
        "memory_key": "key-1",
        "statement": "Episode statement.",
        "started_at": datetime(2026, 8, 4, 10, 0, tzinfo=UTC),
        "ended_at": datetime(2026, 8, 4, 10, 30, tzinfo=UTC),
        "confidence": 0.9,
        "importance": 0.7,
        "evidence": (_evidence(),),
    }
    defaults.update(overrides)
    return EpisodeInput(**defaults)  # type: ignore[arg-type]


def test_episode_write_status_values() -> None:
    assert EpisodeWriteStatus.CREATED == "created"
    assert EpisodeWriteStatus.UPDATED == "updated"
    assert EpisodeWriteStatus.UNCHANGED == "unchanged"


def test_episode_entity_validation() -> None:
    with pytest.raises(ValidationError, match="entity_id"):
        EpisodeEntity(entity_id=" ", role="subject")
    with pytest.raises(ValidationError, match="role"):
        EpisodeEntity(entity_id="user_1", role=" ")


def test_episode_evidence_validation() -> None:
    with pytest.raises(ValidationError, match="observation_id"):
        _evidence(observation_id=" ")
    with pytest.raises(ValidationError, match="observation_revision"):
        _evidence(observation_revision=0)
    with pytest.raises(ValidationError, match="sequence_number"):
        _evidence(sequence_number=-1)
    with pytest.raises(ValidationError, match="contribution_score"):
        _evidence(contribution_score=1.5)


def test_episode_input_validation() -> None:
    with pytest.raises(ValidationError, match="tenant_id"):
        _episode_input(tenant_id=" ")
    with pytest.raises(ValidationError, match="memory_key"):
        _episode_input(memory_key=" ")
    with pytest.raises(ValidationError, match="statement"):
        _episode_input(statement=" ")
    with pytest.raises(ValidationError, match="timezone-aware"):
        _episode_input(started_at=datetime(2026, 8, 4, 10, 0))
    with pytest.raises(ValidationError, match="ended_at"):
        _episode_input(
            started_at=datetime(2026, 8, 4, 11, 0, tzinfo=UTC),
            ended_at=datetime(2026, 8, 4, 10, 0, tzinfo=UTC),
        )
    with pytest.raises(ValidationError, match="confidence"):
        _episode_input(confidence=1.5)
    with pytest.raises(ValidationError, match="importance"):
        _episode_input(importance=-0.1)
    with pytest.raises(ValidationError, match="evidence"):
        _episode_input(evidence=())
    with pytest.raises(ValidationError, match="observation IDs"):
        _episode_input(evidence=(_evidence(), _evidence()))
    with pytest.raises(ValidationError, match="sequence numbers"):
        _episode_input(
            evidence=(
                _evidence(observation_id="obs-1", sequence_number=0),
                _evidence(observation_id="obs-2", sequence_number=0),
            )
        )


def test_episode_input_metadata_is_immutable() -> None:
    episode = _episode_input(metadata=MappingProxyType({"episode": {"count": 1}}))
    assert isinstance(episode.metadata, MappingProxyType)
