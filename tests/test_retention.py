"""Unit tests for observation retention."""

import pytest

from cogkura.exceptions import ValidationError
from cogkura.observations.models import ObservationInput
from cogkura.observations.retention import ObservationRetentionMode, apply_retention


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


def test_retention_preserves_policy_fields() -> None:
    from cogkura.observations.models import ObservationDecision

    decision = ObservationDecision(
        accept=True,
        attention_score=0.8,
        retention_class="hash_only",
        reasons=("high_signal",),
    )
    retained = apply_retention(
        _observation(),
        mode=ObservationRetentionMode.FULL,
        decision=decision,
    )
    assert retained.attention_score == 0.8
    assert retained.retention_class == "hash_only"
    assert retained.policy_reasons == ("high_signal",)


def test_reference_only_not_implemented() -> None:
    with pytest.raises(ValidationError, match="REFERENCE_ONLY"):
        apply_retention(_observation(), mode=ObservationRetentionMode.REFERENCE_ONLY)
