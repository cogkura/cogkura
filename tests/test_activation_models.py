"""Unit tests for activation models."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cogkura.exceptions import ValidationError
from cogkura.models import (
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
    assert config.enable_spreading_activation is True
    assert config.spreading_decay == 0.5
    assert config.spreading_max_hops == 2
    assert config.spreading_min_activation == 0.01
    assert config.enable_entity_slot_admission is True
    assert config.enable_text_precision_matching is True


def test_activation_config_validates_spreading_fields() -> None:
    with pytest.raises(ValidationError, match="spreading_decay"):
        ActivationConfig(spreading_decay=0.0)
    with pytest.raises(ValidationError, match="spreading_max_hops"):
        ActivationConfig(spreading_max_hops=0)
    with pytest.raises(ValidationError, match="source_activation"):
        ActivationConfig(source_activation=-1.0)


def test_activation_config_rejects_noise() -> None:
    with pytest.raises(ValidationError, match="enable_noise"):
        ActivationConfig(enable_noise=True)


def test_retrieval_diagnostics_relevance_tier_defaults() -> None:
    from cogkura.models import RelevanceTier, RetrievalDiagnostics

    diagnostics = RetrievalDiagnostics(
        rank_activation=1.0,
        accessibility_partial=0.5,
        ranking_partial=0.5,
        conjunction=0.5,
        text_coverage=0.5,
        text_cue_fit=0.5,
        temporal_mode="current",
    )
    assert diagnostics.relevance_tier == RelevanceTier.CONTEXTUAL.value
    assert diagnostics.direct_value_fit == 0.0
    assert diagnostics.direct_predicate_fit == 0.0


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
