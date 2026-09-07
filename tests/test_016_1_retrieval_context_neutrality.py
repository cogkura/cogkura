"""Diagnostic and contract tests for retrieval context (0.16.1 matching semantics)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from cogkura import Memory, ObservationInput, RetrievalContext
from cogkura.models import MemoryKind, RecallInspectionDisposition
from cogkura.observations.encoding_context import ObservationContext

_T = datetime(2026, 8, 4, 10, 0, tzinfo=UTC)
_TENANT = "retrieval-neutral"
_SUBJECT = "developer-1"
_QUERY = "Why did we decide not to use Redis?"
_GOAL = "reduce-operational-complexity"
_RETRIEVAL_CONTEXT = RetrievalContext(
    conversation_id="arch-42",
    thread_id="queue-selection",
    goal="reduce-operational-complexity",
    activity="architecture-decision",
    domain="payments-api",
    temporal_context=("queue-redesign",),
)


def _neutral_observations() -> list[ObservationInput]:
    """Two observations that merge into one recallable episode (0.16.0 parity)."""
    return [
        ObservationInput(
            tenant_id=_TENANT,
            subject_id=_SUBJECT,
            source_namespace="github",
            source_record_id="1",
            source_type="discussion",
            content=(
                "We decided not to use Redis because another stateful dependency "
                "would increase operational complexity."
            ),
            observed_at=_T,
            metadata={
                "conversation_id": "arch-42",
                "entity_ids": ("redis", "payments-api"),
            },
            context=ObservationContext(
                conversation_id="arch-42",
                thread_id="queue-selection",
                goal="reduce-operational-complexity",
                activity="architecture-decision",
                domain="payments-api",
                temporal_context=("queue-redesign",),
            ),
        ),
        ObservationInput(
            tenant_id=_TENANT,
            subject_id=_SUBJECT,
            source_namespace="github",
            source_record_id="2",
            source_type="discussion",
            content="PostgreSQL remains the coordination store for payments-api.",
            observed_at=_T + timedelta(minutes=5),
            metadata={
                "conversation_id": "arch-42",
                "entity_ids": ("postgresql", "payments-api"),
            },
            context=ObservationContext(
                conversation_id="arch-42",
                thread_id="queue-selection",
                goal="reduce-operational-complexity",
                activity="architecture-decision",
                domain="payments-api",
                temporal_context=("queue-redesign",),
            ),
        ),
    ]


def _diagnostic_observations() -> list[ObservationInput]:
    """Three separate episodes for observational context-match diagnostics."""
    return [
        ObservationInput(
            tenant_id=_TENANT,
            subject_id=_SUBJECT,
            source_namespace="github",
            source_record_id="1",
            source_type="discussion",
            content=(
                "We decided not to use Redis because another stateful dependency "
                "would increase operational complexity."
            ),
            observed_at=_T,
            metadata={
                "conversation_id": "arch-42",
                "entity_ids": ("redis", "payments-api"),
            },
            context=ObservationContext(
                conversation_id="arch-42",
                thread_id="queue-selection",
                goal="reduce-operational-complexity",
                activity="architecture-decision",
                domain="payments-api",
                temporal_context=("queue-redesign",),
            ),
        ),
        ObservationInput(
            tenant_id=_TENANT,
            subject_id=_SUBJECT,
            source_namespace="github",
            source_record_id="2",
            source_type="discussion",
            content="Analytics-api stores metrics without Redis coordination.",
            observed_at=_T + timedelta(hours=2),
            metadata={
                "conversation_id": "arch-44",
                "entity_ids": ("redis", "analytics-api"),
            },
            context=ObservationContext(
                conversation_id="arch-44",
                thread_id="queue-selection",
                goal="reduce-operational-complexity",
                activity="architecture-decision",
                domain="analytics-api",
                temporal_context=("queue-redesign",),
            ),
        ),
        ObservationInput(
            tenant_id=_TENANT,
            subject_id=_SUBJECT,
            source_namespace="github",
            source_record_id="3",
            source_type="discussion",
            content="Auth service uses Redis for session caching.",
            observed_at=_T + timedelta(hours=4),
            metadata={
                "conversation_id": "arch-43",
                "entity_ids": ("redis", "auth-api"),
            },
            context=ObservationContext(
                conversation_id="arch-43",
                thread_id="auth-cache",
                goal="improve-session-latency",
                activity="architecture-decision",
                domain="auth-api",
                temporal_context=("session-cache",),
            ),
        ),
    ]


async def _build_store(observations: list[ObservationInput] | None = None) -> Memory:
    memory = Memory()
    for observation in observations or _neutral_observations():
        await memory.observe(observation)
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=_T.replace(hour=11))
    return memory


def _episode_candidates(inspection) -> list:
    return [
        candidate
        for candidate in (*inspection.returned, *inspection.rejected)
        if candidate.memory_kind is MemoryKind.EPISODE
    ]


def _recall_snapshot(results: list) -> list[tuple[str, MemoryKind, float]]:
    return [
        (result.memory.memory_key, result.memory_kind, round(result.score, 6)) for result in results
    ]


@pytest.mark.asyncio
async def test_empty_retrieval_context_equivalent_to_none() -> None:
    memory = await _build_store()

    none_results = await memory.recall(_QUERY, tenant_id=_TENANT, retrieval_context=None)
    empty_results = await memory.recall(
        _QUERY,
        tenant_id=_TENANT,
        retrieval_context=RetrievalContext(),
    )
    assert _recall_snapshot(none_results) == _recall_snapshot(empty_results)


@pytest.mark.asyncio
async def test_reinstatement_boosts_activation_separately_from_match_score() -> None:
    memory = await _build_store(_diagnostic_observations())
    inspection = await memory.inspect_recall(
        _QUERY,
        tenant_id=_TENANT,
        retrieval_context=_RETRIEVAL_CONTEXT,
        limit=10,
    )
    episode_candidates = _episode_candidates(inspection)
    boosted = [
        candidate
        for candidate in episode_candidates
        if candidate.diagnostics
        and candidate.diagnostics.context_reinstatement
        and candidate.diagnostics.context_reinstatement.applied
    ]
    assert boosted
    for candidate in boosted:
        assert candidate.diagnostics is not None
        assert candidate.diagnostics.activation_before_context is not None
        assert candidate.activation > candidate.diagnostics.activation_before_context
        assert candidate.diagnostics.context_match is not None
        assert candidate.diagnostics.context_match.score is not None


@pytest.mark.asyncio
async def test_perfect_mismatch_remains_admissible() -> None:
    memory = await _build_store(_diagnostic_observations())
    mismatch_context = RetrievalContext(
        conversation_id="arch-99",
        goal="unrelated-goal",
        domain="unknown-domain",
    )
    inspection = await memory.inspect_recall(
        _QUERY,
        tenant_id=_TENANT,
        retrieval_context=mismatch_context,
        limit=10,
    )
    episode_candidates = _episode_candidates(inspection)
    assert episode_candidates
    for candidate in episode_candidates:
        assert candidate.diagnostics is not None
        assert candidate.diagnostics.context_match is not None
        assert candidate.diagnostics.context_match.score == 0.0 or (
            candidate.diagnostics.context_match.score is None
            and candidate.diagnostics.context_match.cue_coverage == 0.0
        )
        assert candidate.disposition is not RecallInspectionDisposition.FILTERED_FORGOTTEN


@pytest.mark.asyncio
async def test_redis_payments_observational_context_ranking() -> None:
    memory = await _build_store(_diagnostic_observations())
    inspection = await memory.inspect_recall(
        _QUERY,
        tenant_id=_TENANT,
        retrieval_context=_RETRIEVAL_CONTEXT,
        limit=10,
    )
    assert inspection.retrieval_context == _RETRIEVAL_CONTEXT

    episode_candidates = _episode_candidates(inspection)
    assert len(episode_candidates) >= 3

    def domain_dimension_score(candidate) -> float | None:
        if candidate.diagnostics is None or candidate.diagnostics.context_match is None:
            return None
        for dimension in candidate.diagnostics.context_match.dimensions:
            if dimension.dimension == "domain":
                return dimension.score
        return None

    episode_rows = [
        (
            candidate.memory.encoding_context.domains[0]
            if candidate.memory.encoding_context.domains
            else "",
            candidate.diagnostics.context_match.score if candidate.diagnostics else None,
            domain_dimension_score(candidate),
        )
        for candidate in episode_candidates
    ]

    overall_by_domain = {domain: overall for domain, overall, _ in episode_rows if domain}
    domain_dimension_by_domain = {
        domain: domain_score for domain, _, domain_score in episode_rows if domain
    }
    assert overall_by_domain["payments-api"] == 1.0
    assert domain_dimension_by_domain["payments-api"] == 1.0
    assert domain_dimension_by_domain["analytics-api"] == 0.0
    assert overall_by_domain["auth-api"] == pytest.approx(0.17, abs=0.05)
