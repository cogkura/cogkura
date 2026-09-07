"""Unit tests for deterministic context reinstatement policy."""

from __future__ import annotations

import pytest

from cogkura.algorithms.context_reinstatement import DeterministicContextReinstatementPolicy
from cogkura.models import ContextMatch, ContextReinstatementReason


@pytest.fixture
def policy() -> DeterministicContextReinstatementPolicy:
    return DeterministicContextReinstatementPolicy()


def _match(*, score: float | None, coverage: float = 1.0) -> ContextMatch:
    return ContextMatch(
        dimensions=(),
        attribute_dimensions=(),
        score=score,
        comparable_count=1 if score is not None else 0,
        cue_dimension_count=1,
        cue_coverage=coverage,
    )


def test_not_episodic_returns_zero(policy: DeterministicContextReinstatementPolicy) -> None:
    result = policy.evaluate(
        match=_match(score=1.0),
        weight=0.5,
        episodic=False,
        has_retrieval_context=True,
    )
    assert result.activation_contribution == 0.0
    assert result.reason is ContextReinstatementReason.NOT_EPISODIC


def test_no_retrieval_context_returns_zero(policy: DeterministicContextReinstatementPolicy) -> None:
    result = policy.evaluate(
        match=_match(score=1.0),
        weight=0.5,
        episodic=True,
        has_retrieval_context=False,
    )
    assert result.activation_contribution == 0.0
    assert result.reason is ContextReinstatementReason.NO_RETRIEVAL_CONTEXT


def test_no_comparable_context_returns_zero(
    policy: DeterministicContextReinstatementPolicy,
) -> None:
    result = policy.evaluate(
        match=_match(score=None),
        weight=0.5,
        episodic=True,
        has_retrieval_context=True,
    )
    assert result.strength == 0.0
    assert result.activation_contribution == 0.0
    assert result.reason is ContextReinstatementReason.NO_COMPARABLE_CONTEXT


def test_zero_match_returns_zero(policy: DeterministicContextReinstatementPolicy) -> None:
    result = policy.evaluate(
        match=_match(score=0.0),
        weight=0.5,
        episodic=True,
        has_retrieval_context=True,
    )
    assert result.strength == 0.0
    assert result.activation_contribution == 0.0
    assert result.reason is ContextReinstatementReason.ZERO_MATCH


def test_disabled_weight_preserves_strength(
    policy: DeterministicContextReinstatementPolicy,
) -> None:
    result = policy.evaluate(
        match=_match(score=1.0, coverage=1.0),
        weight=0.0,
        episodic=True,
        has_retrieval_context=True,
    )
    assert result.strength == pytest.approx(1.0)
    assert result.activation_contribution == 0.0
    assert result.reason is ContextReinstatementReason.DISABLED


def test_applied_strength_is_match_times_coverage(
    policy: DeterministicContextReinstatementPolicy,
) -> None:
    result = policy.evaluate(
        match=_match(score=0.5, coverage=1.0),
        weight=0.5,
        episodic=True,
        has_retrieval_context=True,
    )
    assert result.strength == pytest.approx(0.5)
    assert result.activation_contribution == pytest.approx(0.25)
    assert result.applied is True
    assert result.reason is ContextReinstatementReason.APPLIED


def test_coverage_attenuates_contribution(policy: DeterministicContextReinstatementPolicy) -> None:
    full = policy.evaluate(
        match=_match(score=1.0, coverage=1.0),
        weight=0.5,
        episodic=True,
        has_retrieval_context=True,
    )
    partial = policy.evaluate(
        match=_match(score=1.0, coverage=0.25),
        weight=0.5,
        episodic=True,
        has_retrieval_context=True,
    )
    assert partial.activation_contribution == pytest.approx(full.activation_contribution / 4.0)


def test_none_match_with_context_uses_empty_match(
    policy: DeterministicContextReinstatementPolicy,
) -> None:
    result = policy.evaluate(
        match=None,
        weight=0.5,
        episodic=True,
        has_retrieval_context=True,
    )
    assert result.reason is ContextReinstatementReason.NO_COMPARABLE_CONTEXT
