"""Observation acceptance policies."""

from __future__ import annotations

from typing import Protocol

from cogkura.observations.models import ObservationDecision, ObservationInput


class ObservationPolicy(Protocol):
    """Evaluates whether an observation should be stored."""

    async def evaluate(self, observation: ObservationInput) -> ObservationDecision:
        """Return accept/reject decision with attention metadata."""


class DefaultObservationPolicy:
    """Accept all non-empty observations with baseline attention."""

    async def evaluate(self, observation: ObservationInput) -> ObservationDecision:
        if observation.is_deleted:
            return ObservationDecision(
                accept=True,
                attention_score=0.3,
                retention_class="full",
                reasons=("deletion_event",),
            )
        content = observation.content.strip()
        if len(content) < 3:
            return ObservationDecision(
                accept=False,
                attention_score=0.0,
                retention_class="full",
                reasons=("content_too_short",),
            )
        return ObservationDecision(
            accept=True,
            attention_score=0.5,
            retention_class="full",
        )
