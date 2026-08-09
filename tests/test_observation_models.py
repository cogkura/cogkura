"""Unit tests for observation models."""

from datetime import UTC, datetime
from types import MappingProxyType

import pytest

from cogkura.exceptions import ValidationError
from cogkura.observations.models import (
    IngestionResult,
    IngestStatus,
    ObservationDecision,
    ObservationInput,
    StoredObservation,
)


def test_observation_input_requires_tenant() -> None:
    with pytest.raises(ValidationError, match="tenant_id"):
        ObservationInput(
            source_namespace="public.messages",
            source_record_id="1",
            content="hello",
            observed_at=datetime.now(UTC),
            tenant_id="",
        )


def test_ingestion_result_record_increments_counters() -> None:
    result = IngestionResult(discovered=1)
    result = result.record(IngestStatus.CREATED)
    assert result.created == 1
    result = result.record(IngestStatus.RESTORED)
    assert result.restored == 1


def test_observation_decision_rejects_invalid_attention() -> None:
    with pytest.raises(ValidationError, match="attention_score"):
        ObservationDecision(accept=True, attention_score=1.5, retention_class="full")
    with pytest.raises(ValidationError, match="attention_score"):
        ObservationDecision(accept=True, attention_score=float("nan"), retention_class="full")


def test_stored_observation_rejects_invalid_attention() -> None:
    with pytest.raises(ValidationError, match="attention_score"):
        StoredObservation(
            id="obs-1",
            tenant_id="company_123",
            subject_id=None,
            actor_id=None,
            source_type="application",
            source_namespace="direct",
            source_record_id="1",
            source_version=None,
            event_type="message",
            content="hello",
            content_hash="hash",
            metadata=MappingProxyType({}),
            source_created_at=None,
            source_updated_at=None,
            observed_at=datetime.now(UTC),
            current_revision=1,
            is_deleted=False,
            attention_score=-0.1,
        )
