"""Unit tests for retrieval-context models and cue plumbing."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from cogkura.exceptions import ValidationError
from cogkura.models import RetrievalCue
from cogkura.observations.encoding_context import ObservationContext, RetrievalContext


def test_empty_retrieval_context() -> None:
    context = RetrievalContext()
    assert context.conversation_id is None
    assert context.temporal_context == ()
    assert context.attributes == MappingProxyType({})
    assert context.is_empty()


def test_retrieval_context_trims_and_deduplicates() -> None:
    context = RetrievalContext(
        goal=" reduce operational complexity ",
        temporal_context=("queue-redesign", "queue-redesign", "  "),
        attributes={" repository ": " payments-api ", "branch": "redis-evaluation"},
    )
    assert context.goal == "reduce operational complexity"
    assert context.temporal_context == ("queue-redesign",)
    assert dict(context.attributes) == {"branch": "redis-evaluation", "repository": "payments-api"}


def test_retrieval_context_rejects_invalid_attributes() -> None:
    with pytest.raises(ValidationError, match="keys must be strings"):
        RetrievalContext(attributes={1: "value"})  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="values must be strings"):
        RetrievalContext(attributes={"key": 42})  # type: ignore[arg-type]


def test_retrieval_context_round_trip() -> None:
    original = RetrievalContext(
        conversation_id="arch-42",
        thread_id="queue-selection",
        goal="reduce-operational-complexity",
        activity="architecture-decision",
        domain="payments-api",
        temporal_context=("queue-redesign",),
    )
    restored = RetrievalContext.from_canonical_dict(original.to_canonical_dict())
    assert restored.to_canonical_dict() == original.to_canonical_dict()


def test_retrieval_context_matches_observation_context_shape() -> None:
    observation = ObservationContext(
        conversation_id="arch-42",
        goal="reduce-operational-complexity",
        temporal_context=("queue-redesign",),
    )
    retrieval = RetrievalContext(
        conversation_id="arch-42",
        goal="reduce-operational-complexity",
        temporal_context=("queue-redesign",),
    )
    assert observation.to_canonical_dict() == retrieval.to_canonical_dict()


def test_retrieval_cue_accepts_retrieval_context_field() -> None:
    cue = RetrievalCue(
        text="Why Redis?",
        retrieval_context=RetrievalContext(goal="reduce-operational-complexity"),
    )
    assert cue.retrieval_context is not None
    assert cue.retrieval_context.goal == "reduce-operational-complexity"


def test_retrieval_cue_allows_context_only_when_populated() -> None:
    cue = RetrievalCue(retrieval_context=RetrievalContext(goal="plan-release"))
    assert cue.retrieval_context is not None


def test_retrieval_cue_rejects_empty_context_only() -> None:
    with pytest.raises(ValidationError, match="must contain at least one field"):
        RetrievalCue(retrieval_context=RetrievalContext())
