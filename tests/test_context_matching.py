"""Contract tests for deterministic context matching."""

from __future__ import annotations

import pytest

from cogkura.algorithms.context_matching import DeterministicContextMatcher
from cogkura.models import ContextMatchState, MemoryContextSignature, MemoryKind
from cogkura.observations.encoding_context import RetrievalContext


@pytest.fixture
def matcher() -> DeterministicContextMatcher:
    return DeterministicContextMatcher()


def test_empty_retrieval_context_returns_none_score(matcher: DeterministicContextMatcher) -> None:
    match = matcher.match(None, MemoryContextSignature(goals=("plan-release",)))
    assert match.score is None
    assert match.cue_coverage == 0.0
    assert match.cue_dimension_count == 0


def test_exact_singular_match(matcher: DeterministicContextMatcher) -> None:
    match = matcher.match(
        RetrievalContext(domain="payments-api"),
        MemoryContextSignature(domains=("payments-api", "analytics-api")),
    )
    assert match.score == 1.0
    assert match.cue_coverage == 1.0
    assert match.dimensions[0].state is ContextMatchState.MATCH


def test_singular_mismatch(matcher: DeterministicContextMatcher) -> None:
    match = matcher.match(
        RetrievalContext(domain="analytics-api"),
        MemoryContextSignature(domains=("payments-api",)),
    )
    assert match.score == 0.0
    assert match.dimensions[0].state is ContextMatchState.MISMATCH


def test_missing_encoding_is_unavailable_not_mismatch(matcher: DeterministicContextMatcher) -> None:
    match = matcher.match(
        RetrievalContext(domain="payments-api"),
        MemoryContextSignature(),
    )
    assert match.score is None
    assert match.cue_coverage == 0.0
    assert match.dimensions[0].state is ContextMatchState.UNAVAILABLE


def test_absent_cue_dimension_does_not_participate(matcher: DeterministicContextMatcher) -> None:
    match = matcher.match(
        RetrievalContext(goal="reduce-operational-complexity"),
        MemoryContextSignature(domains=("payments-api",)),
    )
    assert len(match.dimensions) == 1
    assert match.dimensions[0].dimension == "goal"


def test_temporal_partial_match(matcher: DeterministicContextMatcher) -> None:
    match = matcher.match(
        RetrievalContext(temporal_context=("queue-redesign", "redis-evaluation")),
        MemoryContextSignature(temporal_contexts=("queue-redesign", "release-planning")),
    )
    assert match.score == 0.5
    assert match.dimensions[0].state is ContextMatchState.PARTIAL_MATCH


def test_temporal_empty_intersection_is_mismatch(matcher: DeterministicContextMatcher) -> None:
    match = matcher.match(
        RetrievalContext(temporal_context=("queue-redesign", "redis-evaluation")),
        MemoryContextSignature(temporal_contexts=("release-planning",)),
    )
    assert match.score == 0.0
    assert match.dimensions[0].state is ContextMatchState.MISMATCH


def test_extra_memory_values_do_not_penalise(matcher: DeterministicContextMatcher) -> None:
    match = matcher.match(
        RetrievalContext(goal="reduce-operational-complexity"),
        MemoryContextSignature(
            goals=("reduce-operational-complexity", "evaluate-queue-technology"),
        ),
    )
    assert match.score == 1.0


def test_aggregation_three_match_one_mismatch(matcher: DeterministicContextMatcher) -> None:
    match = matcher.match(
        RetrievalContext(
            conversation_id="arch-42",
            thread_id="queue-selection",
            goal="reduce-operational-complexity",
            domain="analytics-api",
        ),
        MemoryContextSignature(
            conversation_ids=("arch-42",),
            thread_ids=("queue-selection",),
            goals=("reduce-operational-complexity",),
            domains=("payments-api",),
        ),
    )
    assert match.score == 0.75
    assert match.cue_coverage == 1.0


def test_aggregation_two_match_two_unavailable(matcher: DeterministicContextMatcher) -> None:
    match = matcher.match(
        RetrievalContext(
            conversation_id="arch-42",
            thread_id="queue-selection",
            goal="reduce-operational-complexity",
            domain="payments-api",
        ),
        MemoryContextSignature(
            conversation_ids=("arch-42",),
            goals=("reduce-operational-complexity",),
        ),
    )
    assert match.score == 1.0
    assert match.cue_coverage == 0.5
    assert match.cue_dimension_count == 4


def test_attributes_are_diagnostic_only(matcher: DeterministicContextMatcher) -> None:
    match = matcher.match(
        RetrievalContext(
            goal="reduce-operational-complexity",
            attributes={"repository": "payments-api"},
        ),
        MemoryContextSignature(
            goals=("reduce-operational-complexity",),
            attributes={"repository": ("analytics-api",)},
        ),
    )
    assert match.score == 1.0
    assert len(match.attribute_dimensions) == 1
    assert match.attribute_dimensions[0].state is ContextMatchState.MISMATCH


def test_legacy_empty_signature_with_populated_cue(matcher: DeterministicContextMatcher) -> None:
    match = matcher.match(
        RetrievalContext(
            conversation_id="arch-42",
            goal="reduce-operational-complexity",
        ),
        MemoryContextSignature(),
    )
    assert match.score is None
    assert match.cue_coverage == 0.0
    assert all(item.state is ContextMatchState.UNAVAILABLE for item in match.dimensions)


def test_semantic_results_keep_context_match_none() -> None:
    from cogkura.algorithms.context_matching import DeterministicContextMatcher
    from cogkura.memory import _context_match_for_result
    from cogkura.models import (
        ActivationComponents,
        RecallResult,
        RetrievalDiagnostics,
    )

    class _SemanticMemory:
        memory_key = "sem-1"

    result = RecallResult(
        memory_kind=MemoryKind.SEMANTIC,
        memory=_SemanticMemory(),  # type: ignore[arg-type]
        activation=0.5,
        score=0.5,
        latency_seconds=0.1,
        components=ActivationComponents(
            base_level=0.1,
            spreading=0.1,
            partial_match=0.1,
            noise=0.0,
            total=0.5,
        ),
        reason="test",
        diagnostics=RetrievalDiagnostics(
            rank_activation=0.5,
            accessibility_partial=0.5,
            ranking_partial=0.5,
            conjunction=0.0,
            text_coverage=0.5,
            text_cue_fit=0.5,
            temporal_mode="current",
        ),
    )
    updated = _context_match_for_result(
        context_matcher=DeterministicContextMatcher(),
        retrieval_context=RetrievalContext(domain="payments-api"),
        result=result,
    )
    assert updated.diagnostics is not None
    assert updated.diagnostics.context_match is None
