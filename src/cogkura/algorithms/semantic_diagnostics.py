"""Bounded observers for actual semantic decisions, with no decision authority."""

from __future__ import annotations

from datetime import datetime

from cogkura.models import (
    SemanticDecision,
    SemanticDecisionOperation,
    SemanticDecisionOutcome,
    SemanticDecisionReason,
    SemanticDecisionReport,
    SemanticRevisionCandidate,
    semantic_promotion_from_metadata,
)
from cogkura.observations.models import ObservationRevisionIdentity


class SemanticDecisionCollector:
    """Pass-local recorder; never reads storage or evaluates an eligibility rule."""

    def __init__(self, max_items: int) -> None:
        self.max_items = max_items
        self._seen: set[tuple[str, ...]] = set()
        self._items: dict[tuple[str, ...], SemanticDecision] = {}
        self.complete = True
        self.source_identities: dict[str, tuple[str, str]] = {}

    def add(self, decision: SemanticDecision) -> None:
        identity = decision.asserting_evidence
        source = (
            self.source_identities.get(identity.observation_id, ("", "")) if identity else ("", "")
        )
        key = (
            decision.operation.value,
            decision.slot_key or "",
            decision.claim_key or "",
            decision.target_revision_key or "",
            decision.effective_at.isoformat() if decision.effective_at else "",
            *source,
            str(identity.observation_revision) if identity else "",
            identity.observation_id if identity else "",
        )
        if key in self._seen:
            return
        self._seen.add(key)
        if len(self._items) < self.max_items:
            self._items[key] = decision
        elif key < max(self._items):
            del self._items[max(self._items)]
            self._items[key] = decision

    def candidate(
        self,
        candidate: SemanticRevisionCandidate,
        *,
        outcome: SemanticDecisionOutcome,
        reason: SemanticDecisionReason,
    ) -> None:
        promotion = semantic_promotion_from_metadata(candidate.metadata)
        identity = None
        if (
            promotion
            and promotion.observation_id is not None
            and promotion.observation_revision is not None
        ):
            identity = ObservationRevisionIdentity(
                promotion.observation_id, promotion.observation_revision
            )
        self.add(
            SemanticDecision(
                operation=SemanticDecisionOperation.ADMIT,
                outcome=outcome,
                reasons=(reason,),
                slot_key=candidate.slot_key,
                claim_key=candidate.memory_key,
                resulting_revision_key=candidate.revision_key
                if outcome
                in (SemanticDecisionOutcome.APPLIED, SemanticDecisionOutcome.ALREADY_APPLIED)
                else None,
                asserting_evidence=identity,
            )
        )

    def report(
        self,
        *,
        tenant_id: str,
        subject_id: str | None,
        evaluated_at: datetime,
        authority_evaluated: bool,
        corrections_evaluated: bool,
    ) -> SemanticDecisionReport:
        return SemanticDecisionReport(
            tenant_id=tenant_id,
            subject_id=subject_id,
            evaluated_at=evaluated_at,
            authority_evaluated=authority_evaluated,
            corrections_evaluated=corrections_evaluated,
            items=tuple(self._items[k] for k in sorted(self._items)),
            total_decisions=len(self._seen),
            is_truncated=len(self._seen) > len(self._items),
            diagnostics_complete=self.complete,
        )


class EditDecisionObserver:
    """Identity-bound observer passed down the existing qualification branches."""

    def __init__(
        self,
        collector: SemanticDecisionCollector | None,
        *,
        operation: SemanticDecisionOperation,
        observation_id: str,
        observation_revision: int,
        target_key: str | None,
        effective_at: datetime | None,
        slot_key: str | None = None,
        claim_key: str | None = None,
        resulting_revision_key: str | None = None,
    ) -> None:
        self.collector = collector
        self.operation = operation
        self.identity = ObservationRevisionIdentity(observation_id, observation_revision)
        self.target_key = target_key
        self.effective_at = effective_at
        self.slot_key = slot_key
        self.claim_key = claim_key
        self.resulting_revision_key = resulting_revision_key

    def decide(self, outcome: str, reason: SemanticDecisionReason) -> str:
        if self.collector is not None and outcome != "ok":
            mapped = {
                "invalid": SemanticDecisionOutcome.INVALID,
                "deferred": SemanticDecisionOutcome.DEFERRED,
                "applied": SemanticDecisionOutcome.ALREADY_APPLIED,
                "committed": SemanticDecisionOutcome.APPLIED,
            }[outcome]
            self.collector.add(
                SemanticDecision(
                    operation=self.operation,
                    outcome=mapped,
                    reasons=(reason,),
                    slot_key=self.slot_key,
                    claim_key=self.claim_key,
                    target_revision_key=self.target_key,
                    resulting_revision_key=self.resulting_revision_key
                    if outcome in ("applied", "committed")
                    else None,
                    asserting_evidence=self.identity,
                    effective_at=self.effective_at,
                )
            )
        return outcome
