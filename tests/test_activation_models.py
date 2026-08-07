"""Unit tests for activation models."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cognema.exceptions import ValidationError
from cognema.models import (
    ActivationConfig,
    ActivationReferenceKind,
    MemoryIdentity,
    MemoryKind,
    MemoryReference,
    RetrievalCue,
)


def test_retrieval_cue_requires_field() -> None:
    with pytest.raises(ValidationError, match="at least one field"):
        RetrievalCue()


def test_retrieval_cue_accepts_text() -> None:
    cue = RetrievalCue(text="payment incident")
    assert cue.text == "payment incident"


def test_activation_config_defaults() -> None:
    config = ActivationConfig()
    assert config.decay == 0.5
    assert config.enable_spreading_activation is False


def test_activation_config_rejects_noise() -> None:
    with pytest.raises(ValidationError, match="enable_noise"):
        ActivationConfig(enable_noise=True)


def test_memory_reference_identity() -> None:
    reference = MemoryReference(
        tenant_id="company_123",
        memory_kind=MemoryKind.EPISODE,
        memory_key="episode-key",
        reference_kind=ActivationReferenceKind.RETRIEVED,
        referenced_at=datetime(2026, 8, 7, 10, 0, tzinfo=UTC),
    )
    assert reference.identity == MemoryIdentity(
        memory_kind=MemoryKind.EPISODE,
        memory_key="episode-key",
    )
