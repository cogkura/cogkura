"""Unit tests for forgetting evaluation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import MappingProxyType

import pytest

from cogkura.algorithms.activation import activation_candidate_from_episode
from cogkura.algorithms.forgetting import (
    EbbinghausForgettingEvaluator,
    retention_score_from_base_level,
)
from cogkura.models import (
    ActivationConfig,
    ActivationReferenceTrace,
    EpisodeEvidenceInput,
    ForgettingConfig,
    MemoryRetentionState,
    StoredEpisode,
)

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _episode(*, created_at: datetime, memory_key: str = "episode-key") -> StoredEpisode:
    return StoredEpisode(
        id=f"id-{memory_key}",
        tenant_id="company_123",
        subject_id="customer_42",
        memory_key=memory_key,
        statement="Episode statement.",
        started_at=created_at,
        ended_at=created_at,
        confidence=0.9,
        importance=0.7,
        is_active=True,
        evidence=(
            EpisodeEvidenceInput(
                observation_id="obs-1",
                observation_revision=1,
                sequence_number=0,
            ),
        ),
        entities=(),
        metadata=MappingProxyType({}),
        created_at=created_at,
        updated_at=created_at,
    )


def test_retention_score_is_bounded() -> None:
    assert retention_score_from_base_level(10.0, retrieval_threshold=-3.0) > 0.99
    assert retention_score_from_base_level(-3.0, retrieval_threshold=-3.0) == pytest.approx(0.5)
    assert retention_score_from_base_level(-10.0, retrieval_threshold=-3.0) < 0.01


def test_retention_decays_without_access() -> None:
    evaluator = EbbinghausForgettingEvaluator()
    candidate = activation_candidate_from_episode(_episode(created_at=_T0))
    activation_config = ActivationConfig(retrieval_threshold=-3.0, time_unit_seconds=1.0)
    forgetting_config = ForgettingConfig()

    early = evaluator.evaluate(
        candidate=candidate,
        references=(),
        previous=None,
        as_of=_T0 + timedelta(seconds=10),
        activation_config=activation_config,
        forgetting_config=forgetting_config,
        tenant_id="company_123",
    )
    late = evaluator.evaluate(
        candidate=candidate,
        references=(),
        previous=early.dynamics,
        as_of=_T0 + timedelta(days=365),
        activation_config=activation_config,
        forgetting_config=forgetting_config,
        tenant_id="company_123",
    )
    assert late.dynamics.last_retention_score < early.dynamics.last_retention_score


def test_reinforcement_increases_retention() -> None:
    evaluator = EbbinghausForgettingEvaluator()
    candidate = activation_candidate_from_episode(_episode(created_at=_T0))
    activation_config = ActivationConfig(retrieval_threshold=-3.0, time_unit_seconds=1.0)
    forgetting_config = ForgettingConfig()
    as_of = _T0 + timedelta(days=30)

    without_refs = evaluator.evaluate(
        candidate=candidate,
        references=(),
        previous=None,
        as_of=as_of,
        activation_config=activation_config,
        forgetting_config=forgetting_config,
        tenant_id="company_123",
    )
    with_refs = evaluator.evaluate(
        candidate=candidate,
        references=(
            ActivationReferenceTrace(
                referenced_at=_T0 + timedelta(days=29),
                weight=1,
            ),
            ActivationReferenceTrace(
                referenced_at=_T0 + timedelta(days=29, hours=12),
                weight=1,
            ),
        ),
        previous=None,
        as_of=as_of,
        activation_config=activation_config,
        forgetting_config=forgetting_config,
        tenant_id="company_123",
    )
    assert with_refs.dynamics.last_retention_score > without_refs.dynamics.last_retention_score


def test_fading_transition() -> None:
    evaluator = EbbinghausForgettingEvaluator()
    candidate = activation_candidate_from_episode(_episode(created_at=_T0))
    activation_config = ActivationConfig(retrieval_threshold=-3.0, time_unit_seconds=1.0)
    forgetting_config = ForgettingConfig(
        fading_retention_threshold=0.9,
        forgotten_retention_threshold=0.05,
    )
    decision = evaluator.evaluate(
        candidate=candidate,
        references=(),
        previous=None,
        as_of=_T0 + timedelta(days=60),
        activation_config=activation_config,
        forgetting_config=forgetting_config,
        tenant_id="company_123",
    )
    assert decision.dynamics.retention_state is MemoryRetentionState.FADING
    assert decision.dynamics.below_threshold_since is not None


def test_grace_period_prevents_immediate_forgotten() -> None:
    evaluator = EbbinghausForgettingEvaluator()
    candidate = activation_candidate_from_episode(_episode(created_at=_T0))
    activation_config = ActivationConfig(retrieval_threshold=-3.0, time_unit_seconds=1.0)
    forgetting_config = ForgettingConfig(
        fading_retention_threshold=0.9,
        forgotten_retention_threshold=0.8,
        grace_period_seconds=86_400.0,
    )
    previous = evaluator.evaluate(
        candidate=candidate,
        references=(),
        previous=None,
        as_of=_T0 + timedelta(days=60),
        activation_config=activation_config,
        forgetting_config=forgetting_config,
        tenant_id="company_123",
    )
    decision = evaluator.evaluate(
        candidate=candidate,
        references=(),
        previous=previous.dynamics,
        as_of=_T0 + timedelta(days=60, hours=12),
        activation_config=activation_config,
        forgetting_config=forgetting_config,
        tenant_id="company_123",
    )
    assert decision.dynamics.retention_state is MemoryRetentionState.FADING


def test_forgotten_after_grace_period() -> None:
    evaluator = EbbinghausForgettingEvaluator()
    candidate = activation_candidate_from_episode(_episode(created_at=_T0))
    activation_config = ActivationConfig(retrieval_threshold=-3.0, time_unit_seconds=1.0)
    forgetting_config = ForgettingConfig(
        fading_retention_threshold=0.9,
        forgotten_retention_threshold=0.8,
        grace_period_seconds=86_400.0,
    )
    previous = evaluator.evaluate(
        candidate=candidate,
        references=(),
        previous=None,
        as_of=_T0 + timedelta(days=60),
        activation_config=activation_config,
        forgetting_config=forgetting_config,
        tenant_id="company_123",
    )
    decision = evaluator.evaluate(
        candidate=candidate,
        references=(),
        previous=previous.dynamics,
        as_of=_T0 + timedelta(days=62),
        activation_config=activation_config,
        forgetting_config=forgetting_config,
        tenant_id="company_123",
    )
    assert decision.dynamics.retention_state is MemoryRetentionState.FORGOTTEN
    assert decision.dynamics.forgotten_at is not None
