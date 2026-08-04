"""Unit tests for observation models."""

from datetime import UTC, datetime

import pytest

from cognema.exceptions import ValidationError
from cognema.observations.models import IngestionResult, IngestStatus, ObservationInput


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
