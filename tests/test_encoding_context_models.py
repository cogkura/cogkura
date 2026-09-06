"""Unit tests for encoding-context models."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from cogkura.exceptions import ValidationError
from cogkura.models import MemoryContextSignature
from cogkura.observations.encoding_context import ObservationContext


def test_empty_observation_context() -> None:
    context = ObservationContext()
    assert context.conversation_id is None
    assert context.temporal_context == ()
    assert context.attributes == MappingProxyType({})
    assert context.to_canonical_dict()["attributes"] == {}


def test_observation_context_trims_and_deduplicates() -> None:
    context = ObservationContext(
        goal=" reduce operational complexity ",
        temporal_context=("queue-redesign", "queue-redesign", "  "),
        attributes={" repository ": " payments-api ", "branch": "redis-evaluation"},
    )
    assert context.goal == "reduce operational complexity"
    assert context.temporal_context == ("queue-redesign",)
    assert dict(context.attributes) == {"branch": "redis-evaluation", "repository": "payments-api"}


def test_observation_context_rejects_invalid_attributes() -> None:
    with pytest.raises(ValidationError, match="keys must be strings"):
        ObservationContext(attributes={1: "value"})  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="values must be strings"):
        ObservationContext(attributes={"key": 42})  # type: ignore[arg-type]


def test_observation_context_round_trip() -> None:
    original = ObservationContext(
        conversation_id="arch-42",
        thread_id="queue-selection",
        goal="reduce-operational-complexity",
        activity="architecture-decision",
        domain="payments-api",
        temporal_context=("queue-redesign",),
    )
    restored = ObservationContext.from_canonical_dict(original.to_canonical_dict())
    assert restored.to_canonical_dict() == original.to_canonical_dict()


def test_memory_context_signature_sorts_and_deduplicates() -> None:
    signature = MemoryContextSignature(
        goals=(
            "reduce-operational-complexity",
            "evaluate-queue-technology",
            "reduce-operational-complexity",
        ),
        entity_ids=("redis", "payments-api", "redis"),
    )
    assert signature.goals == (
        "evaluate-queue-technology",
        "reduce-operational-complexity",
    )
    assert signature.entity_ids == ("payments-api", "redis")


def test_memory_context_signature_attributes_union() -> None:
    signature = MemoryContextSignature(
        attributes={
            "repository": ("payments-api", "analytics-api"),
            "branch": ("redis-evaluation",),
        }
    )
    assert signature.attributes["repository"] == ("analytics-api", "payments-api")
    assert signature.attributes["branch"] == ("redis-evaluation",)


def test_memory_context_signature_is_empty() -> None:
    assert MemoryContextSignature().is_empty()
    assert not MemoryContextSignature(goals=("plan-release",)).is_empty()


def test_memory_context_signature_round_trip() -> None:
    original = MemoryContextSignature(
        subject_ids=("developer-1",),
        conversation_ids=("arch-42",),
        goals=("reduce-operational-complexity",),
        source_namespaces=("github",),
        source_types=("discussion",),
        entity_ids=("payments-api", "redis"),
        temporal_contexts=("queue-redesign",),
    )
    restored = MemoryContextSignature.from_canonical_dict(original.to_canonical_dict())
    assert restored.to_canonical_dict() == original.to_canonical_dict()
