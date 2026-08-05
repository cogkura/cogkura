"""Unit tests for semantic memory models."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cognema.exceptions import ValidationError
from cognema.models import (
    SemanticCardinality,
    SemanticDerivationInput,
    SemanticDerivationRelation,
    SemanticFactCandidate,
    SemanticMemoryStatus,
    SemanticPolarity,
    SemanticWriteStatus,
)


def _candidate(**overrides: object) -> SemanticFactCandidate:
    defaults = {
        "tenant_id": "company_123",
        "source_episode_id": "ep-1",
        "subject_entity_id": "customer_42",
        "predicate": "preferred_database",
        "object_value": "postgresql",
        "object_entity_id": "postgresql",
        "polarity": SemanticPolarity.AFFIRM,
        "cardinality": SemanticCardinality.ONE,
        "confidence": 0.9,
        "observed_at": datetime(2026, 8, 5, 10, 0, tzinfo=UTC),
    }
    defaults.update(overrides)
    return SemanticFactCandidate(**defaults)  # type: ignore[arg-type]


def test_semantic_fact_candidate_validation() -> None:
    with pytest.raises(ValidationError, match="tenant_id"):
        _candidate(tenant_id=" ")
    with pytest.raises(ValidationError, match="predicate"):
        _candidate(predicate=" ")
    with pytest.raises(ValidationError, match="object_value"):
        _candidate(object_value=" ")
    with pytest.raises(ValidationError, match="confidence"):
        _candidate(confidence=1.5)
    with pytest.raises(ValidationError, match="timezone-aware"):
        _candidate(observed_at=datetime(2026, 8, 5, 10, 0))


def test_semantic_derivation_validation() -> None:
    with pytest.raises(ValidationError, match="episode_id"):
        SemanticDerivationInput(
            episode_id=" ",
            relation=SemanticDerivationRelation.SUPPORTS,
            contribution_score=0.5,
        )


def test_semantic_write_status_values() -> None:
    assert SemanticWriteStatus.CREATED == "created"
    assert SemanticMemoryStatus.CONTESTED == "contested"
