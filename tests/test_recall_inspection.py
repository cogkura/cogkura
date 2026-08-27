"""Tests for recall inspection API."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cogkura import Memory
from cogkura.exceptions import RecallInspectionUnsupportedError
from cogkura.models import (
    ActivationCandidate,
    ActivationConfig,
    ActivationReferenceTrace,
    LearnedAssociation,
    MemoryIdentity,
    RecallInspectionDisposition,
    RecallResult,
    RetrievalCue,
)
from cogkura.observations.models import ObservationInput

_T1 = datetime(2026, 1, 15, tzinfo=UTC)
_T2 = datetime(2026, 8, 1, tzinfo=UTC)


class _RankOnlyActivator:
    def rank(
        self,
        *,
        candidates: list[ActivationCandidate],
        cue: RetrievalCue,
        references: dict[MemoryIdentity, tuple[ActivationReferenceTrace, ...]],
        as_of: datetime,
        config: ActivationConfig,
        limit: int,
        learned_associations: tuple[LearnedAssociation, ...] = (),
        episode_support_index: dict | None = None,
        valid_at: datetime | None = None,
        episode_slot_index: dict | None = None,
    ) -> list[RecallResult]:
        return []


@pytest.mark.asyncio
async def test_inspect_recall_unsupported_for_custom_activator() -> None:
    memory = Memory(declarative_activator=_RankOnlyActivator())  # type: ignore[arg-type]
    with pytest.raises(RecallInspectionUnsupportedError):
        await memory.inspect_recall("query", tenant_id="company_123")


@pytest.mark.asyncio
async def test_inspect_recall_reports_below_threshold_candidates() -> None:
    memory = Memory()
    await memory.observe(
        ObservationInput(
            tenant_id="company_123",
            subject_id="team",
            source_namespace="chat.messages",
            source_record_id="early",
            content="Project Atlas selected Redis for job coordination.",
            observed_at=_T1,
        )
    )
    await memory.process(tenant_id="company_123", as_of=_T1)
    inspection = await memory.inspect_recall(
        "unrelated finance topic",
        tenant_id="company_123",
        subject_id="team",
        as_of=_T2,
        limit=5,
    )
    assert inspection.considered_count >= 1
    if inspection.rejected:
        assert any(
            item.disposition is RecallInspectionDisposition.BELOW_THRESHOLD
            for item in inspection.rejected
        )
