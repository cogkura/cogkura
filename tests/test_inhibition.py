"""Unit tests for inhibitory trace induction and the in-memory store."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cogkura.algorithms.inhibition import build_inhibitory_traces
from cogkura.models import (
    ActivationComponents,
    BehavioralStructuralAnchor,
    CompetitionDirection,
    InhibitionConfig,
    InhibitionScopeSignature,
    MemoryIdentity,
    MemoryKind,
    RecallResult,
    RetrievalCompetitionSnapshot,
    RetrievalDiagnostics,
    SemanticCardinality,
    SemanticMemoryStatus,
    SemanticPolarity,
    StoredSemanticMemory,
)
from cogkura.storage.in_memory_inhibition import InMemoryInhibitionStore

_T = datetime(2026, 8, 1, tzinfo=UTC)


def _semantic(memory_key: str) -> StoredSemanticMemory:
    return StoredSemanticMemory(
        id=memory_key,
        tenant_id="tenant",
        subject_id="operator",
        memory_key=memory_key,
        slot_key="slot",
        revision_key="rev",
        revision_number=1,
        statement=memory_key,
        subject_entity_id="payments-api",
        predicate="deployment_system",
        object_value=memory_key,
        object_entity_id=None,
        polarity=SemanticPolarity.AFFIRM,
        cardinality=SemanticCardinality.MANY,
        qualifiers={},
        confidence=0.9,
        importance=0.5,
        status=SemanticMemoryStatus.ACTIVE,
        support_count=1,
        contradiction_count=0,
        first_supported_at=_T,
        last_supported_at=_T,
        valid_from=_T,
        valid_until=None,
        is_active=True,
        derivations=(),
        observation_evidence=(),
        entities=(),
        metadata={},
        created_at=_T,
        updated_at=_T,
    )


def _result(memory_key: str, snapshots: tuple[RetrievalCompetitionSnapshot, ...]) -> RecallResult:
    diagnostics = RetrievalDiagnostics(
        rank_activation=0.0,
        accessibility_partial=0.0,
        ranking_partial=0.0,
        conjunction=0.0,
        text_coverage=0.5,
        text_cue_fit=0.5,
        temporal_mode="current",
        inhibition_candidates=snapshots,
    )
    return RecallResult(
        memory_kind=MemoryKind.SEMANTIC,
        memory=_semantic(memory_key),
        activation=1.0,
        score=0.8,
        latency_seconds=0.1,
        components=ActivationComponents(0.0, 0.0, 0.0, 0.0, 1.0),
        reason="test",
        diagnostics=diagnostics,
    )


def _snapshot(
    competitor_key: str, strength: float, accessibility: float
) -> RetrievalCompetitionSnapshot:
    return RetrievalCompetitionSnapshot(
        competitor_identity=MemoryIdentity(
            memory_kind=MemoryKind.SEMANTIC,
            memory_key=competitor_key,
        ),
        direction=CompetitionDirection.PROACTIVE,
        competition_strength=strength,
        competitor_accessibility=accessibility,
        scope_eligible=True,
        scope=InhibitionScopeSignature(
            structural_anchor=BehavioralStructuralAnchor.SEMANTIC_SLOT,
            subject_id="payments-api",
            predicate="deployment_system",
            semantic_slot_key=f"slot-{competitor_key}",
        ),
        retrieval_evaluated_at=_T,
    )


def test_trace_selection_is_bounded_and_deterministic() -> None:
    snapshots = tuple(
        _snapshot(f"c-{index}", strength=0.9 - (index * 0.01), accessibility=0.8)
        for index in range(5)
    )
    selected = _result("selected", snapshots)
    traces = build_inhibitory_traces(
        [selected],
        tenant_id="tenant",
        config=InhibitionConfig(enabled=True, max_traces_per_use=2, minimum_induction_pressure=0.2),
        induced_at=_T,
        request_id=None,
    )
    assert len(traces) == 2
    assert traces[0].induction_pressure >= traces[1].induction_pressure
    assert traces[0].inhibited_identity.memory_key == "c-0"


@pytest.mark.asyncio
async def test_in_memory_request_id_is_idempotent() -> None:
    store = InMemoryInhibitionStore()
    selected = _result("selected", (_snapshot("loser", 0.8, 0.8),))
    traces = build_inhibitory_traces(
        [selected],
        tenant_id="tenant",
        config=InhibitionConfig(enabled=True),
        induced_at=_T,
        request_id="req-1",
    )
    await store.append_traces(traces)
    await store.append_traces(traces)
    listed = await store.list_for_memory(
        tenant_id="tenant",
        identity=MemoryIdentity(memory_kind=MemoryKind.SEMANTIC, memory_key="loser"),
    )
    assert len(listed) == 1
