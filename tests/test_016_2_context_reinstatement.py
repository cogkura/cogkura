"""Behaviour and regression tests for 0.16.2 context reinstatement."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import MappingProxyType

import pytest

from cogkura import Memory, ObservationInput, RetrievalContext
from cogkura.algorithms.activation import (
    ACTRDeclarativeActivator,
    activation_candidate_from_episode,
    activation_candidate_from_semantic,
    build_episode_support_index,
)
from cogkura.algorithms.context_reinstatement import DeterministicContextReinstatementPolicy
from cogkura.models import (
    ActivationConfig,
    ActivationReferenceTrace,
    ContextMatch,
    ContextReinstatementReason,
    EpisodeEntity,
    EpisodeEvidenceInput,
    MemoryContextSignature,
    MemoryKind,
    RetrievalCue,
    SemanticCardinality,
    SemanticMemoryStatus,
    SemanticPolarity,
    SemanticSupportContextReason,
    StoredEpisode,
    StoredSemanticMemory,
)
from cogkura.observations.encoding_context import ObservationContext

_T = datetime(2026, 8, 4, 10, 0, tzinfo=UTC)
_TENANT = "retrieval-neutral"
_SUBJECT = "developer-1"
_QUERY = "Why did we decide not to use Redis?"
_RETRIEVAL_CONTEXT = RetrievalContext(
    conversation_id="arch-42",
    thread_id="queue-selection",
    goal="reduce-operational-complexity",
    activity="architecture-decision",
    domain="payments-api",
    temporal_context=("queue-redesign",),
)
_T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def _diagnostic_observations() -> list[ObservationInput]:
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


def _neutral_observations() -> list[ObservationInput]:
    return _diagnostic_observations()[:1]


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


def _episode(
    *,
    episode_id: str,
    memory_key: str,
    statement: str,
    encoding_context: MemoryContextSignature | None = None,
    entity_ids: tuple[str, ...] = ("redis",),
) -> StoredEpisode:
    return StoredEpisode(
        id=episode_id,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        memory_key=memory_key,
        statement=statement,
        started_at=_T0,
        ended_at=_T0,
        confidence=0.9,
        importance=0.7,
        is_active=True,
        evidence=(
            EpisodeEvidenceInput(
                observation_id=f"obs-{memory_key}",
                observation_revision=1,
                sequence_number=0,
            ),
        ),
        entities=tuple(EpisodeEntity(entity_id=eid, role="mention") for eid in entity_ids),
        metadata=MappingProxyType({}),
        created_at=_T0,
        updated_at=_T0,
        encoding_context=encoding_context or MemoryContextSignature(),
    )


def _reference(
    *, referenced_at: datetime | None = None, weight: int = 1
) -> ActivationReferenceTrace:
    return ActivationReferenceTrace(
        referenced_at=referenced_at or _T0,
        weight=weight,
    )


@pytest.mark.asyncio
async def test_no_retrieval_context_matches_baseline_recall() -> None:
    memory = await _build_store()

    baseline = await memory.recall(_QUERY, tenant_id=_TENANT)
    none_context = await memory.recall(_QUERY, tenant_id=_TENANT, retrieval_context=None)
    assert _recall_snapshot(baseline) == _recall_snapshot(none_context)


@pytest.mark.asyncio
async def test_weight_zero_with_context_matches_no_context_recall() -> None:
    memory = Memory(
        activation_config=ActivationConfig(context_reinstatement_weight=0.0),
    )
    for observation in _diagnostic_observations():
        await memory.observe(observation)
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=_T.replace(hour=11))

    baseline = await memory.recall(_QUERY, tenant_id=_TENANT)
    contextual = await memory.recall(
        _QUERY,
        tenant_id=_TENANT,
        retrieval_context=_RETRIEVAL_CONTEXT,
    )
    assert _recall_snapshot(baseline) == _recall_snapshot(contextual)

    inspection = await memory.inspect_recall(
        _QUERY,
        tenant_id=_TENANT,
        retrieval_context=_RETRIEVAL_CONTEXT,
        limit=10,
    )
    for candidate in _episode_candidates(inspection):
        assert candidate.diagnostics is not None
        assert candidate.diagnostics.context_reinstatement is not None
        assert (
            candidate.diagnostics.context_reinstatement.reason
            is ContextReinstatementReason.DISABLED
        )


@pytest.mark.asyncio
async def test_weight_zero_prepare_context_render_unchanged() -> None:
    memory = Memory(
        activation_config=ActivationConfig(context_reinstatement_weight=0.0),
    )
    for observation in _diagnostic_observations():
        await memory.observe(observation)
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=_T.replace(hour=11))

    baseline = await memory.prepare_context(_QUERY, tenant_id=_TENANT)
    contextual = await memory.prepare_context(
        _QUERY,
        tenant_id=_TENANT,
        retrieval_context=_RETRIEVAL_CONTEXT,
    )
    assert baseline.render() == contextual.render()


def test_rank_reorder_from_reinstatement() -> None:
    payments_context = MemoryContextSignature(domains=("payments-api",))
    weaker = _episode(
        episode_id="ep-weak",
        memory_key="weak-match",
        statement="Redis coordination discussed for payments-api.",
        encoding_context=payments_context,
        entity_ids=("redis", "payments-api"),
    )
    stronger = _episode(
        episode_id="ep-strong",
        memory_key="strong-no-match",
        statement="Redis coordination discussed for payments-api with more detail.",
        encoding_context=MemoryContextSignature(domains=("analytics-api",)),
        entity_ids=("redis", "payments-api"),
    )
    activator = ACTRDeclarativeActivator()
    weaker_candidate = activation_candidate_from_episode(weaker)
    stronger_candidate = activation_candidate_from_episode(stronger)
    references = {
        weaker_candidate.identity: (_reference(weight=1),),
        stronger_candidate.identity: (_reference(weight=2),),
    }
    cue = RetrievalCue(
        text="Why did we decide not to use Redis?",
        entity_ids=("redis",),
        retrieval_context=RetrievalContext(domain="payments-api"),
    )
    config = ActivationConfig(
        retrieval_threshold=-10.0,
        context_reinstatement_weight=1.0,
        enable_spreading_activation=False,
        enable_duplicate_collapse=False,
        enable_text_entity_seeding=False,
        enable_semantic_slot_admission=False,
    )
    results = activator.rank(
        candidates=[weaker_candidate, stronger_candidate],
        cue=cue,
        references=references,
        as_of=_T0 + timedelta(hours=1),
        config=config,
        limit=5,
        episode_support_index=build_episode_support_index([]),
    )
    assert results[0].memory.memory_key == weaker.memory_key
    assert results[0].diagnostics is not None
    assert results[0].diagnostics.context_reinstatement is not None
    assert results[0].diagnostics.context_reinstatement.applied is True


def test_threshold_crossing_from_reinstatement() -> None:
    episode = _episode(
        episode_id="ep-borderline",
        memory_key="borderline",
        statement="Redis decision for payments-api.",
        encoding_context=MemoryContextSignature(domains=("payments-api",)),
    )
    activator = ACTRDeclarativeActivator()
    candidate = activation_candidate_from_episode(episode)
    as_of = _T0 + timedelta(seconds=601)
    references: dict = {}
    cue_with_context = RetrievalCue(
        text="redis",
        entity_ids=("redis",),
        retrieval_context=RetrievalContext(domain="payments-api"),
    )
    cue_without_context = RetrievalCue(text="redis", entity_ids=("redis",))
    config = ActivationConfig(
        retrieval_threshold=-3.0,
        context_reinstatement_weight=0.6,
        time_unit_seconds=1.0,
        minimum_elapsed_seconds=1.0,
        enable_spreading_activation=False,
        enable_duplicate_collapse=False,
        enable_text_entity_seeding=False,
        enable_semantic_slot_admission=False,
        enable_partial_matching=False,
    )

    without = activator.inspect(
        candidates=[candidate],
        cue=cue_without_context,
        references=references,
        as_of=as_of,
        config=config,
        limit=5,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        episode_support_index=build_episode_support_index([]),
    )
    with_context = activator.inspect(
        candidates=[candidate],
        cue=cue_with_context,
        references=references,
        as_of=as_of,
        config=config,
        limit=5,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        episode_support_index=build_episode_support_index([]),
    )
    assert without.rejected
    assert with_context.returned
    returned = with_context.returned[0]
    assert returned.diagnostics is not None
    reinstatement = returned.diagnostics.context_reinstatement
    assert reinstatement is not None
    assert reinstatement.applied is True
    assert reinstatement.activation_contribution == pytest.approx(0.6, abs=0.01)
    assert returned.activation >= config.retrieval_threshold
    assert without.rejected[0].activation < config.retrieval_threshold


def test_coverage_attenuates_contribution() -> None:
    policy = DeterministicContextReinstatementPolicy()

    full = policy.evaluate(
        match=ContextMatch(
            dimensions=(),
            attribute_dimensions=(),
            score=1.0,
            comparable_count=1,
            cue_dimension_count=1,
            cue_coverage=1.0,
        ),
        weight=0.5,
        episodic=True,
        has_retrieval_context=True,
    )
    partial = policy.evaluate(
        match=ContextMatch(
            dimensions=(),
            attribute_dimensions=(),
            score=1.0,
            comparable_count=1,
            cue_dimension_count=4,
            cue_coverage=0.25,
        ),
        weight=0.5,
        episodic=True,
        has_retrieval_context=True,
    )
    assert partial.activation_contribution == pytest.approx(full.activation_contribution / 4.0)


def test_partial_match_halves_contribution() -> None:
    policy = DeterministicContextReinstatementPolicy()

    perfect = policy.evaluate(
        match=ContextMatch(
            dimensions=(),
            attribute_dimensions=(),
            score=1.0,
            comparable_count=1,
            cue_dimension_count=1,
            cue_coverage=1.0,
        ),
        weight=0.5,
        episodic=True,
        has_retrieval_context=True,
    )
    partial = policy.evaluate(
        match=ContextMatch(
            dimensions=(),
            attribute_dimensions=(),
            score=0.5,
            comparable_count=1,
            cue_dimension_count=1,
            cue_coverage=1.0,
        ),
        weight=0.5,
        episodic=True,
        has_retrieval_context=True,
    )
    assert partial.activation_contribution == pytest.approx(perfect.activation_contribution / 2.0)


def test_none_and_zero_match_never_penalise() -> None:
    policy = DeterministicContextReinstatementPolicy()

    none_result = policy.evaluate(
        match=ContextMatch(
            dimensions=(),
            attribute_dimensions=(),
            score=None,
            comparable_count=0,
            cue_dimension_count=1,
            cue_coverage=0.0,
        ),
        weight=0.5,
        episodic=True,
        has_retrieval_context=True,
    )
    zero_result = policy.evaluate(
        match=ContextMatch(
            dimensions=(),
            attribute_dimensions=(),
            score=0.0,
            comparable_count=1,
            cue_dimension_count=1,
            cue_coverage=1.0,
        ),
        weight=0.5,
        episodic=True,
        has_retrieval_context=True,
    )
    assert none_result.activation_contribution == 0.0
    assert zero_result.activation_contribution == 0.0


def test_semantic_candidate_has_no_context_contribution() -> None:
    semantic = StoredSemanticMemory(
        id="sem-1",
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
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
        support_count=1,
        contradiction_count=0,
        first_supported_at=_T0,
        last_supported_at=_T0,
        valid_from=None,
        valid_until=None,
        is_active=True,
        derivations=(),
        observation_evidence=(
            EpisodeEvidenceInput(
                observation_id="obs-sem",
                observation_revision=1,
                sequence_number=0,
            ),
        ),
        entities=(EpisodeEntity(entity_id="redis", role="mention"),),
        metadata=MappingProxyType({}),
        created_at=_T0,
        updated_at=_T0,
    )
    activator = ACTRDeclarativeActivator()
    cue = RetrievalCue(
        text="service redis cache",
        entity_ids=("redis",),
        retrieval_context=RetrievalContext(domain="payments-api"),
    )
    results = activator.rank(
        candidates=[activation_candidate_from_semantic(semantic)],
        cue=cue,
        references={},
        as_of=_T0 + timedelta(hours=1),
        config=ActivationConfig(
            retrieval_threshold=-10.0,
            context_reinstatement_weight=1.0,
            enable_spreading_activation=False,
            enable_duplicate_collapse=False,
        ),
        limit=5,
        episode_support_index=build_episode_support_index([semantic]),
    )
    diagnostics = results[0].diagnostics
    assert diagnostics is not None
    assert diagnostics.context_match is None
    assert diagnostics.support_context is not None
    assert diagnostics.support_context.reason is SemanticSupportContextReason.NO_SUPPORTS
    assert diagnostics.context_reinstatement is not None
    assert diagnostics.context_reinstatement.activation_contribution == 0.0


@pytest.mark.asyncio
async def test_adversarial_context_match_does_not_bypass_relevance() -> None:
    memory = await _build_store(
        [
            ObservationInput(
                tenant_id=_TENANT,
                subject_id=_SUBJECT,
                source_namespace="github",
                source_record_id="unrelated",
                source_type="discussion",
                content="Weekly lunch menu options for the cafeteria were updated.",
                observed_at=_T,
                metadata={"entity_ids": ("cafeteria",)},
                context=ObservationContext(
                    conversation_id="arch-42",
                    domain="payments-api",
                    activity="architecture-decision",
                ),
            )
        ]
    )
    inspection = await memory.inspect_recall(
        "Why did we decide not to use Redis?",
        tenant_id=_TENANT,
        retrieval_context=_RETRIEVAL_CONTEXT,
        limit=10,
    )
    unrelated = [
        candidate
        for candidate in inspection.returned
        if candidate.memory_kind is MemoryKind.EPISODE
        and "cafeteria" in candidate.memory.statement.lower()
    ]
    assert not unrelated


@pytest.mark.asyncio
async def test_sparse_shared_activity_cue_gives_equal_weak_boost() -> None:
    sparse_context = RetrievalContext(activity="architecture-decision")
    memory = await _build_store(_diagnostic_observations())
    inspection = await memory.inspect_recall(
        _QUERY,
        tenant_id=_TENANT,
        retrieval_context=sparse_context,
        limit=10,
    )
    episodes = _episode_candidates(inspection)
    contributions = [
        candidate.diagnostics.context_reinstatement.activation_contribution
        for candidate in episodes
        if candidate.diagnostics
        and candidate.diagnostics.context_reinstatement
        and candidate.diagnostics.context_reinstatement.applied
    ]
    assert len(contributions) >= 2
    assert max(contributions) - min(contributions) <= 0.05


@pytest.mark.asyncio
async def test_working_memory_render_format_unchanged_without_context() -> None:
    memory = await _build_store(_diagnostic_observations())
    first = await memory.prepare_context(_QUERY, tenant_id=_TENANT, prompt_budget_tokens=512)
    second = await memory.prepare_context(_QUERY, tenant_id=_TENANT, prompt_budget_tokens=512)
    assert first.render() == second.render()


@pytest.mark.asyncio
async def test_redis_payments_rank_improves_with_default_weight() -> None:
    memory_default = Memory()
    memory_disabled = Memory(activation_config=ActivationConfig(context_reinstatement_weight=0.0))
    for observation in _diagnostic_observations():
        await memory_default.observe(observation)
        await memory_disabled.observe(observation)
    as_of = _T.replace(hour=11)
    await memory_default.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=as_of)
    await memory_disabled.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=as_of)

    enabled_inspection = await memory_default.inspect_recall(
        _QUERY,
        tenant_id=_TENANT,
        retrieval_context=_RETRIEVAL_CONTEXT,
        limit=10,
    )
    disabled_inspection = await memory_disabled.inspect_recall(
        _QUERY,
        tenant_id=_TENANT,
        retrieval_context=_RETRIEVAL_CONTEXT,
        limit=10,
    )

    def payments_activation_rank(inspection) -> int:
        episodes = sorted(
            _episode_candidates(inspection),
            key=lambda item: item.activation,
            reverse=True,
        )
        for index, item in enumerate(episodes):
            memory_obj = item.memory
            if isinstance(memory_obj, StoredEpisode) and memory_obj.encoding_context.domains == (
                "payments-api",
            ):
                return index
        raise AssertionError("payments-api episode not found")

    assert payments_activation_rank(enabled_inspection) <= payments_activation_rank(
        disabled_inspection
    )

    payments_candidate = next(
        candidate
        for candidate in _episode_candidates(enabled_inspection)
        if isinstance(candidate.memory, StoredEpisode)
        and candidate.memory.encoding_context.domains == ("payments-api",)
    )
    assert payments_candidate.diagnostics is not None
    assert payments_candidate.diagnostics.context_reinstatement is not None
    assert payments_candidate.diagnostics.context_reinstatement.applied is True


@pytest.mark.parametrize("weight", [0.25, 0.50, 0.75, 1.00])
@pytest.mark.asyncio
async def test_calibration_sweep_restores_payments_without_unrelated_admission(
    weight: float,
) -> None:
    memory = Memory(activation_config=ActivationConfig(context_reinstatement_weight=weight))
    for observation in _diagnostic_observations():
        await memory.observe(observation)
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=_T.replace(hour=11))

    inspection = await memory.inspect_recall(
        _QUERY,
        tenant_id=_TENANT,
        retrieval_context=_RETRIEVAL_CONTEXT,
        limit=10,
    )
    episode_candidates = _episode_candidates(inspection)
    payments_candidate = next(
        candidate
        for candidate in episode_candidates
        if isinstance(candidate.memory, StoredEpisode)
        and candidate.memory.encoding_context.domains == ("payments-api",)
    )
    assert payments_candidate.diagnostics is not None
    assert payments_candidate.diagnostics.context_reinstatement is not None
    assert payments_candidate.diagnostics.context_reinstatement.strength == pytest.approx(1.0)
    if weight > 0.0:
        assert payments_candidate.diagnostics.context_reinstatement.applied is True
    assert len(episode_candidates) >= 3
