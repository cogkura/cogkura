"""Ebbinghaus-inspired forgetting lifecycle over ACT-R base-level activation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from cogkura.algorithms.activation import calculate_base_level
from cogkura.models import (
    ActivationCandidate,
    ActivationConfig,
    ActivationReferenceTrace,
    ForgettingConfig,
    ForgettingDecision,
    MemoryRetentionState,
    StoredMemoryDynamics,
)


def retention_score_from_base_level(
    base_level: float,
    *,
    retrieval_threshold: float,
) -> float:
    """Map ACT-R base-level activation to a bounded retention score."""
    return 1.0 / (1.0 + math.exp(-(base_level - retrieval_threshold)))


class ForgettingEvaluator(Protocol):
    """Evaluates cognitive forgetting state for durable memories."""

    def evaluate(
        self,
        *,
        candidate: ActivationCandidate,
        references: Sequence[ActivationReferenceTrace],
        previous: StoredMemoryDynamics | None,
        as_of: datetime,
        activation_config: ActivationConfig,
        forgetting_config: ForgettingConfig,
        tenant_id: str,
    ) -> ForgettingDecision:
        """Return the forgetting decision for one memory candidate."""


class EbbinghausForgettingEvaluator:
    """Deterministic forgetting lifecycle driven by ACT-R base-level activation."""

    def evaluate(
        self,
        *,
        candidate: ActivationCandidate,
        references: Sequence[ActivationReferenceTrace],
        previous: StoredMemoryDynamics | None,
        as_of: datetime,
        activation_config: ActivationConfig,
        forgetting_config: ForgettingConfig,
        tenant_id: str,
    ) -> ForgettingDecision:
        creation_trace = ActivationReferenceTrace(
            referenced_at=candidate.created_at,
            weight=1,
        )
        reference_history = (creation_trace, *references)
        base_level = calculate_base_level(
            reference_history,
            as_of=as_of,
            decay=activation_config.decay,
            constant=activation_config.base_level_constant,
            time_unit_seconds=activation_config.time_unit_seconds,
            minimum_elapsed_seconds=activation_config.minimum_elapsed_seconds,
        )
        retention_score = retention_score_from_base_level(
            base_level,
            retrieval_threshold=activation_config.retrieval_threshold,
        )
        previous_state = previous.retention_state if previous is not None else None

        if retention_score >= forgetting_config.fading_retention_threshold:
            dynamics = _build_dynamics(
                tenant_id=tenant_id,
                candidate=candidate,
                retention_state=MemoryRetentionState.ACTIVE,
                base_level=base_level,
                retention_score=retention_score,
                below_threshold_since=None,
                forgotten_at=None,
                as_of=as_of,
            )
            reactivated = (
                previous_state is MemoryRetentionState.FORGOTTEN
                or previous_state is MemoryRetentionState.FADING
            )
            return ForgettingDecision(
                dynamics=dynamics,
                previous_state=previous_state,
                reactivated=reactivated,
            )

        below_since = (
            previous.below_threshold_since
            if previous is not None and previous.below_threshold_since is not None
            else as_of
        )

        if retention_score > forgetting_config.forgotten_retention_threshold:
            dynamics = _build_dynamics(
                tenant_id=tenant_id,
                candidate=candidate,
                retention_state=MemoryRetentionState.FADING,
                base_level=base_level,
                retention_score=retention_score,
                below_threshold_since=below_since,
                forgotten_at=None,
                as_of=as_of,
            )
            return ForgettingDecision(
                dynamics=dynamics,
                previous_state=previous_state,
                reactivated=False,
            )

        grace_elapsed = (as_of - below_since).total_seconds()
        if grace_elapsed < forgetting_config.grace_period_seconds:
            dynamics = _build_dynamics(
                tenant_id=tenant_id,
                candidate=candidate,
                retention_state=MemoryRetentionState.FADING,
                base_level=base_level,
                retention_score=retention_score,
                below_threshold_since=below_since,
                forgotten_at=None,
                as_of=as_of,
            )
            return ForgettingDecision(
                dynamics=dynamics,
                previous_state=previous_state,
                reactivated=False,
            )

        forgotten_at = (
            previous.forgotten_at
            if previous is not None and previous.forgotten_at is not None
            else as_of
        )
        dynamics = _build_dynamics(
            tenant_id=tenant_id,
            candidate=candidate,
            retention_state=MemoryRetentionState.FORGOTTEN,
            base_level=base_level,
            retention_score=retention_score,
            below_threshold_since=below_since,
            forgotten_at=forgotten_at,
            as_of=as_of,
        )
        return ForgettingDecision(
            dynamics=dynamics,
            previous_state=previous_state,
            reactivated=False,
        )


def _build_dynamics(
    *,
    tenant_id: str,
    candidate: ActivationCandidate,
    retention_state: MemoryRetentionState,
    base_level: float,
    retention_score: float,
    below_threshold_since: datetime | None,
    forgotten_at: datetime | None,
    as_of: datetime,
) -> StoredMemoryDynamics:
    return StoredMemoryDynamics(
        tenant_id=tenant_id,
        memory_kind=candidate.memory_kind,
        memory_key=candidate.memory_key,
        retention_state=retention_state,
        last_base_level=base_level,
        last_retention_score=retention_score,
        below_threshold_since=below_threshold_since,
        forgotten_at=forgotten_at,
        evaluated_at=as_of,
        updated_at=as_of,
    )
