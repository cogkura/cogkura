"""Unit tests for semantic support-context aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType

import pytest

from cogkura.algorithms.semantic_support_context import (
    DeterministicSemanticSupportContextPolicy,
    reinstatement_strength_from_match,
    unique_support_episode_ids,
)
from cogkura.models import (
    ContextMatch,
    SemanticCardinality,
    SemanticDerivationInput,
    SemanticDerivationRelation,
    SemanticMemoryStatus,
    SemanticPolarity,
    SemanticSupportContextReason,
    StoredSemanticMemory,
)
from cogkura.observations.encoding_context import RetrievalContext

_T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def _match(*, score: float | None, coverage: float = 1.0) -> ContextMatch:
    return ContextMatch(
        dimensions=(),
        attribute_dimensions=(),
        score=score,
        comparable_count=1 if score is not None else 0,
        cue_dimension_count=1,
        cue_coverage=coverage,
    )


def _semantic(*, derivations: tuple[SemanticDerivationInput, ...]) -> StoredSemanticMemory:
    return StoredSemanticMemory(
        id="sem-1",
        tenant_id="tenant",
        subject_id="subject",
        memory_key="service-redis",
        slot_key="slot:service-redis",
        revision_key="rev:service-redis",
        revision_number=1,
        statement="The service uses Redis.",
        subject_entity_id="service",
        predicate="cache",
        object_value="redis",
        object_entity_id=None,
        polarity=SemanticPolarity.AFFIRM,
        cardinality=SemanticCardinality.ONE,
        qualifiers=MappingProxyType({}),
        confidence=0.9,
        importance=0.7,
        status=SemanticMemoryStatus.ACTIVE,
        support_count=len(derivations),
        contradiction_count=0,
        first_supported_at=_T0,
        last_supported_at=_T0,
        valid_from=None,
        valid_until=None,
        is_active=True,
        derivations=derivations,
        observation_evidence=(),
        entities=(),
        metadata=MappingProxyType({}),
        created_at=_T0,
        updated_at=_T0,
    )


@dataclass(slots=True)
class _StubMatchCache:
    strengths: dict[str, tuple[ContextMatch | None, float, bool]]

    def match_strength(self, episode_id: str) -> tuple[ContextMatch | None, float, bool]:
        return self.strengths.get(episode_id, (None, 0.0, True))


@pytest.fixture
def policy() -> DeterministicSemanticSupportContextPolicy:
    return DeterministicSemanticSupportContextPolicy()


def test_reinstatement_strength_from_match() -> None:
    assert reinstatement_strength_from_match(None) == 0.0
    assert reinstatement_strength_from_match(_match(score=None)) == 0.0
    assert reinstatement_strength_from_match(_match(score=0.0)) == 0.0
    assert reinstatement_strength_from_match(_match(score=1.0, coverage=1.0)) == 1.0
    assert reinstatement_strength_from_match(_match(score=0.5, coverage=1.0)) == 0.5
    assert reinstatement_strength_from_match(_match(score=1.0, coverage=0.5)) == 0.5


def test_unique_support_episode_ids_deduplicates_and_ignores_contradicts() -> None:
    memory = _semantic(
        derivations=(
            SemanticDerivationInput(
                episode_id="ep-2",
                relation=SemanticDerivationRelation.SUPPORTS,
                contribution_score=0.9,
            ),
            SemanticDerivationInput(
                episode_id="ep-1",
                relation=SemanticDerivationRelation.SUPPORTS,
                contribution_score=0.8,
            ),
            SemanticDerivationInput(
                episode_id="ep-1",
                relation=SemanticDerivationRelation.SUPPORTS,
                contribution_score=0.7,
            ),
            SemanticDerivationInput(
                episode_id="ep-3",
                relation=SemanticDerivationRelation.CONTRADICTS,
                contribution_score=0.6,
            ),
        )
    )
    assert unique_support_episode_ids(memory) == ("ep-1", "ep-2")


def test_canonical_mixed_fixture_rs_equals_0_375(
    policy: DeterministicSemanticSupportContextPolicy,
) -> None:
    memory = _semantic(
        derivations=tuple(
            SemanticDerivationInput(
                episode_id=episode_id,
                relation=SemanticDerivationRelation.SUPPORTS,
                contribution_score=0.9,
            )
            for episode_id in ("ep-1", "ep-2", "ep-3", "ep-4")
        )
    )
    cache = _StubMatchCache(
        {
            "ep-1": (_match(score=1.0), 1.0, False),
            "ep-2": (_match(score=0.5), 0.5, False),
            "ep-3": (_match(score=0.0), 0.0, False),
            "ep-4": (None, 0.0, True),
        }
    )
    evidence = policy.evaluate(
        memory=memory,
        retrieval_context=RetrievalContext(domain="payments-api"),
        weight=0.25,
        match_cache=cache,
    )
    assert evidence.support_count == 4
    assert evidence.comparable_support_count == 3
    assert evidence.unavailable_support_count == 1
    assert evidence.matching_support_count == 2
    assert evidence.conflicting_support_count == 1
    assert evidence.support_coverage == pytest.approx(0.75)
    assert evidence.mean_reinstatement_strength == pytest.approx(0.5)
    assert evidence.strength == pytest.approx(0.375)
    assert evidence.activation_contribution == pytest.approx(0.09375)
    assert evidence.reason is SemanticSupportContextReason.APPLIED


def test_no_retrieval_context_short_circuits(
    policy: DeterministicSemanticSupportContextPolicy,
) -> None:
    memory = _semantic(
        derivations=(
            SemanticDerivationInput(
                episode_id="ep-1",
                relation=SemanticDerivationRelation.SUPPORTS,
                contribution_score=0.9,
            ),
        )
    )
    evidence = policy.evaluate(
        memory=memory,
        retrieval_context=None,
        weight=0.25,
        match_cache=None,
    )
    assert evidence.reason is SemanticSupportContextReason.NO_RETRIEVAL_CONTEXT
    assert evidence.comparable_support_count == 0
    assert evidence.unavailable_support_count == 0
    assert evidence.activation_contribution == 0.0
    assert evidence.supports == ()


def test_missing_match_cache_is_not_evaluated(
    policy: DeterministicSemanticSupportContextPolicy,
) -> None:
    memory = _semantic(
        derivations=(
            SemanticDerivationInput(
                episode_id="ep-1",
                relation=SemanticDerivationRelation.SUPPORTS,
                contribution_score=0.9,
            ),
        )
    )
    evidence = policy.evaluate(
        memory=memory,
        retrieval_context=RetrievalContext(domain="payments-api"),
        weight=0.25,
        match_cache=None,
    )
    assert evidence.reason is SemanticSupportContextReason.NOT_EVALUATED
    assert evidence.comparable_support_count == 0
    assert evidence.unavailable_support_count == 0


def test_all_unavailable_supports_are_no_comparable_context(
    policy: DeterministicSemanticSupportContextPolicy,
) -> None:
    memory = _semantic(
        derivations=(
            SemanticDerivationInput(
                episode_id="ep-1",
                relation=SemanticDerivationRelation.SUPPORTS,
                contribution_score=0.9,
            ),
            SemanticDerivationInput(
                episode_id="ep-2",
                relation=SemanticDerivationRelation.SUPPORTS,
                contribution_score=0.8,
            ),
        )
    )
    cache = _StubMatchCache(
        {
            "ep-1": (None, 0.0, True),
            "ep-2": (None, 0.0, True),
        }
    )
    evidence = policy.evaluate(
        memory=memory,
        retrieval_context=RetrievalContext(domain="payments-api"),
        weight=0.25,
        match_cache=cache,
    )
    assert evidence.reason is SemanticSupportContextReason.NO_COMPARABLE_CONTEXT
    assert evidence.comparable_support_count == 0
    assert evidence.unavailable_support_count == 2
    assert evidence.strength == 0.0


def test_disabled_weight_keeps_diagnostics(
    policy: DeterministicSemanticSupportContextPolicy,
) -> None:
    memory = _semantic(
        derivations=(
            SemanticDerivationInput(
                episode_id="ep-1",
                relation=SemanticDerivationRelation.SUPPORTS,
                contribution_score=0.9,
            ),
        )
    )
    cache = _StubMatchCache({"ep-1": (_match(score=1.0), 1.0, False)})
    evidence = policy.evaluate(
        memory=memory,
        retrieval_context=RetrievalContext(domain="payments-api"),
        weight=0.0,
        match_cache=cache,
    )
    assert evidence.strength == pytest.approx(1.0)
    assert evidence.activation_contribution == 0.0
    assert evidence.reason is SemanticSupportContextReason.DISABLED
