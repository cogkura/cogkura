"""Unit tests for observation retention."""

import pytest

from cognema.exceptions import ValidationError
from cognema.observations.models import ObservationInput
from cognema.observations.retention import ObservationRetentionMode, apply_retention


def _observation(content: str = "George prefers PostgreSQL.") -> ObservationInput:
    from datetime import UTC, datetime

    return ObservationInput(
        tenant_id="company_123",
        source_namespace="public.messages",
        source_record_id="msg-1",
        content=content,
        observed_at=datetime.now(UTC),
    )


def test_full_retention_keeps_content() -> None:
    retained = apply_retention(_observation(), mode=ObservationRetentionMode.FULL)
    assert retained.content == "George prefers PostgreSQL."
    assert retained.content_hash


def test_hash_only_retention_drops_content() -> None:
    retained = apply_retention(_observation(), mode=ObservationRetentionMode.HASH_ONLY)
    assert retained.content is None
    assert retained.content_hash


def test_reference_only_not_implemented() -> None:
    with pytest.raises(ValidationError, match="REFERENCE_ONLY"):
        apply_retention(_observation(), mode=ObservationRetentionMode.REFERENCE_ONLY)
