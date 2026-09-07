"""Behaviour and regression tests for 0.16.4 context observability."""

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
from cogkura.algorithms.context_observability import discrimination_set
from cogkura.algorithms.semantic import ComplementaryLearningSemanticConsolidator
from cogkura.models import (
    ActivationConfig,
    ActivationReferenceTrace,
    ContextObservabilityReason,
    EpisodeEntity,
    EpisodeEvidenceInput,
    MemoryAssessmentFlag,
    MemoryContextSignature,
    MemoryKind,
    RecallInspectionDisposition,
    RetrievalContextState,
    RetrievalCue,
    SemanticCardinality,
    SemanticDerivationInput,
    SemanticDerivationRelation,
    SemanticMemoryStatus,
    SemanticPolarity,
    StoredEpisode,
    StoredSemanticMemory,
)
from cogkura.observations.encoding_context import ObservationContext

_T = datetime(2026, 8, 4, 10, 0, tzinfo=UTC)
_T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
_TENANT = "context-observability"
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


def _minimal_observations_without_encoding_context() -> list[ObservationInput]:
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
            metadata={"entity_ids": ("redis", "payments-api")},
        ),
        ObservationInput(
            tenant_id=_TENANT,
            subject_id=_SUBJECT,
            source_namespace="github",
            source_record_id="2",
            source_type="discussion",
            content="Analytics-api stores metrics without Redis coordination.",
            observed_at=_T + timedelta(hours=2),
            metadata={"entity_ids": ("redis", "analytics-api")},
        ),
    ]


async def _build_store(
    observations: list[ObservationInput] | None = None,
    *,
    activation_config: ActivationConfig | None = None,
) -> Memory:
    memory = Memory(activation_config=activation_config)
    for observation in observations or _diagnostic_observations():
        await memory.observe(observation)
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=_T.replace(hour=11))
    return memory


def _recall_snapshot(results: list) -> list[tuple[str, MemoryKind, float]]:
    return [
        (result.memory.memory_key, result.memory_kind, round(result.score, 6)) for result in results
    ]


def _inspection_snapshot(inspection) -> list[tuple[str, str, float]]:
    rows: list[tuple[str, str, float]] = []
    for candidate in (*inspection.returned, *inspection.rejected):
        rows.append(
            (
                candidate.memory.memory_key,
                candidate.disposition.value,
                round(candidate.activation, 6),
            )
        )
    return rows


def _episode_candidates(inspection) -> list:
    return [
        candidate
        for candidate in (*inspection.returned, *inspection.rejected)
        if candidate.memory_kind is MemoryKind.EPISODE
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


def _reference(*, weight: int = 1) -> ActivationReferenceTrace:
    return ActivationReferenceTrace(referenced_at=_T0, weight=weight)


def _semantic(
    *,
    derivations: tuple[SemanticDerivationInput, ...],
    memory_key: str = "service-redis",
) -> StoredSemanticMemory:
    return StoredSemanticMemory(
        id="sem-1",
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        memory_key=memory_key,
        slot_key=f"slot:{memory_key}",
        revision_key=f"rev:{memory_key}",
        revision_number=1,
        statement="The service avoids Redis for coordination.",
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
        entities=(EpisodeEntity(entity_id="redis", role="mention"),),
        metadata=MappingProxyType({}),
        created_at=_T0,
        updated_at=_T0,
    )


def _support_episode(
    *,
    episode_id: str,
    encoding_context: MemoryContextSignature,
) -> StoredEpisode:
    return _episode(
        episode_id=episode_id,
        memory_key=f"support-{episode_id}",
        statement=f"Support episode {episode_id}.",
        encoding_context=encoding_context,
    )


@pytest.mark.asyncio
async def test_no_retrieval_context_neutral_recall_and_inspect_state() -> None:
    memory = await _build_store(
        activation_config=ActivationConfig(
            context_reinstatement_weight=0.0,
            semantic_context_reinstatement_weight=0.0,
        )
    )
    baseline = await memory.recall(_QUERY, tenant_id=_TENANT)
    none_context = await memory.recall(_QUERY, tenant_id=_TENANT, retrieval_context=None)
    assert _recall_snapshot(baseline) == _recall_snapshot(none_context)

    wm = await memory.prepare_context(_QUERY, tenant_id=_TENANT, prompt_budget_tokens=512)
    wm_again = await memory.prepare_context(_QUERY, tenant_id=_TENANT, prompt_budget_tokens=512)
    assert wm.render() == wm_again.render()

    inspection = await memory.inspect_recall(_QUERY, tenant_id=_TENANT, limit=10)
    assert inspection.context is not None
    assert inspection.context.state is RetrievalContextState.CONTEXT_NOT_PROVIDED
    assert ContextObservabilityReason.NO_RETRIEVAL_CONTEXT in inspection.context.reasons


@pytest.mark.asyncio
async def test_populated_context_recall_matches_inspect_scoring() -> None:
    memory = await _build_store()
    recall = await memory.recall(
        _QUERY,
        tenant_id=_TENANT,
        retrieval_context=_RETRIEVAL_CONTEXT,
    )
    inspection = await memory.inspect_recall(
        _QUERY,
        tenant_id=_TENANT,
        retrieval_context=_RETRIEVAL_CONTEXT,
        limit=10,
    )
    recall_rows = {
        (result.memory.memory_key, result.memory_kind): round(result.score, 6) for result in recall
    }
    returned_rows = {
        (candidate.memory.memory_key, candidate.memory_kind): round(candidate.score, 6)
        for candidate in inspection.returned
    }
    assert recall_rows == returned_rows
    assert inspection.context is not None
    assert inspection.context.state is RetrievalContextState.CONTEXT_SUFFICIENT


@pytest.mark.asyncio
async def test_underspecified_sparse_activity_cue() -> None:
    memory = await _build_store()
    sparse_context = RetrievalContext(activity="architecture-decision")
    inspection = await memory.inspect_recall(
        _QUERY,
        tenant_id=_TENANT,
        retrieval_context=sparse_context,
        limit=10,
    )
    assert inspection.context is not None
    assert inspection.context.state is RetrievalContextState.CONTEXT_UNDERSPECIFIED
    assert inspection.context.top_context_strength == pytest.approx(1.0)
    assert inspection.context.second_context_strength == pytest.approx(1.0)
    assert inspection.context.context_margin == pytest.approx(0.0)
    assert ContextObservabilityReason.MULTIPLE_TOP_CONTEXT_MATCHES in inspection.context.reasons


@pytest.mark.asyncio
async def test_sufficient_with_domain_qualifier() -> None:
    memory = await _build_store()
    qualified_context = RetrievalContext(
        activity="architecture-decision",
        domain="payments-api",
    )
    inspection = await memory.inspect_recall(
        _QUERY,
        tenant_id=_TENANT,
        retrieval_context=qualified_context,
        limit=10,
    )
    assert inspection.context is not None
    assert inspection.context.state is RetrievalContextState.CONTEXT_SUFFICIENT
    assert inspection.context.context_margin is not None
    assert inspection.context.context_margin > 0.0


@pytest.mark.asyncio
async def test_context_unavailable_when_encoding_context_empty() -> None:
    memory = await _build_store(_minimal_observations_without_encoding_context())
    inspection = await memory.inspect_recall(
        _QUERY,
        tenant_id=_TENANT,
        retrieval_context=_RETRIEVAL_CONTEXT,
        limit=10,
    )
    assert inspection.context is not None
    assert inspection.context.state is RetrievalContextState.CONTEXT_UNAVAILABLE
    assert ContextObservabilityReason.NO_COMPARABLE_CONTEXT in inspection.context.reasons


@pytest.mark.asyncio
async def test_mixed_candidate_match_can_still_be_sufficient() -> None:
    memory = await _build_store()
    inspection = await memory.inspect_recall(
        _QUERY,
        tenant_id=_TENANT,
        retrieval_context=_RETRIEVAL_CONTEXT,
        limit=10,
    )
    auth_candidates = [
        candidate
        for candidate in _episode_candidates(inspection)
        if candidate.memory.encoding_context.domains == ("auth-api",)
    ]
    assert auth_candidates
    auth = auth_candidates[0]
    assert auth.diagnostics is not None
    assert auth.diagnostics.context_match is not None
    assert 0.0 < auth.diagnostics.context_match.score < 1.0
    assert inspection.context is not None
    assert inspection.context.state is RetrievalContextState.CONTEXT_SUFFICIENT


def test_threshold_crossing_flag_on_inspect() -> None:
    episode = _episode(
        episode_id="ep-borderline",
        memory_key="borderline",
        statement="Redis decision for payments-api.",
        encoding_context=MemoryContextSignature(domains=("payments-api",)),
    )
    activator = ACTRDeclarativeActivator()
    candidate = activation_candidate_from_episode(episode)
    as_of = _T0 + timedelta(seconds=601)
    cue_with_context = RetrievalCue(
        text="redis",
        entity_ids=("redis",),
        retrieval_context=RetrievalContext(domain="payments-api"),
    )
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
    inspection = activator.inspect(
        candidates=[candidate],
        cue=cue_with_context,
        references={},
        as_of=as_of,
        config=config,
        limit=5,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        episode_support_index=build_episode_support_index([]),
    )
    assert inspection.returned
    returned = inspection.returned[0]
    assert returned.diagnostics is not None
    assert returned.diagnostics.crossed_activation_threshold_due_to_context is True
    assert inspection.context is not None
    assert ContextObservabilityReason.CONTEXT_RESTORED_THRESHOLD in inspection.context.reasons


def test_rank_delta_on_context_reorder() -> None:
    weaker = _episode(
        episode_id="ep-weak",
        memory_key="candidate-a",
        statement="Redis coordination discussed for payments-api.",
        encoding_context=MemoryContextSignature(domains=("payments-api",)),
    )
    stronger = _episode(
        episode_id="ep-strong",
        memory_key="candidate-b",
        statement="Redis coordination discussed for analytics-api with more detail.",
        encoding_context=MemoryContextSignature(domains=("analytics-api",)),
        entity_ids=("redis", "analytics-api"),
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
    inspection = activator.inspect(
        candidates=[weaker_candidate, stronger_candidate],
        cue=cue,
        references=references,
        as_of=_T0 + timedelta(hours=1),
        config=config,
        limit=5,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        episode_support_index=build_episode_support_index([]),
    )
    by_key = {
        candidate.memory.memory_key: candidate
        for candidate in (*inspection.returned, *inspection.rejected)
    }
    assert by_key["candidate-a"].context_rank_delta == 1
    assert by_key["candidate-b"].context_rank_delta == -1
    assert inspection.context is not None
    assert ContextObservabilityReason.CONTEXT_CHANGED_RANK in inspection.context.reasons


def test_semantic_support_strength_0_375_on_inspect() -> None:
    derivations = tuple(
        SemanticDerivationInput(
            episode_id=episode_id,
            relation=SemanticDerivationRelation.SUPPORTS,
            contribution_score=0.9,
        )
        for episode_id in ("ep-1", "ep-2", "ep-3", "ep-4")
    )
    semantic = _semantic(derivations=derivations)
    episodes = [
        _support_episode(
            episode_id="ep-1",
            encoding_context=MemoryContextSignature(
                domains=("payments-api",),
                goals=("reduce-complexity",),
            ),
        ),
        _support_episode(
            episode_id="ep-2",
            encoding_context=MemoryContextSignature(
                domains=("payments-api",),
                goals=("other",),
            ),
        ),
        _support_episode(
            episode_id="ep-3",
            encoding_context=MemoryContextSignature(domains=("analytics-api",)),
        ),
        _support_episode(episode_id="ep-4", encoding_context=MemoryContextSignature()),
    ]
    activator = ACTRDeclarativeActivator()
    cue = RetrievalCue(
        text="Why did we decide not to use Redis?",
        entity_ids=("redis",),
        retrieval_context=RetrievalContext(
            domain="payments-api",
            goal="reduce-complexity",
        ),
    )
    candidates = [activation_candidate_from_semantic(semantic)]
    candidates.extend(activation_candidate_from_episode(episode) for episode in episodes)
    references = {candidate.identity: (_reference(),) for candidate in candidates}
    config = ActivationConfig(
        retrieval_threshold=-10.0,
        semantic_context_reinstatement_weight=0.25,
        enable_spreading_activation=False,
        enable_duplicate_collapse=False,
        enable_text_entity_seeding=False,
        enable_semantic_slot_admission=False,
    )
    inspection = activator.inspect(
        candidates=candidates,
        cue=cue,
        references=references,
        as_of=_T0 + timedelta(hours=1),
        config=config,
        limit=10,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        episode_support_index=build_episode_support_index([semantic]),
        episode_by_id={episode.id: episode for episode in episodes},
    )
    semantic_candidates = [
        candidate
        for candidate in (*inspection.returned, *inspection.rejected)
        if candidate.memory_kind is MemoryKind.SEMANTIC
    ]
    assert semantic_candidates
    semantic_candidate = semantic_candidates[0]
    assert semantic_candidate.diagnostics is not None
    assert semantic_candidate.diagnostics.support_context is not None
    assert semantic_candidate.diagnostics.support_context.strength == pytest.approx(0.375)
    assert len(semantic_candidate.diagnostics.support_context.supports) == 4


@pytest.mark.asyncio
async def test_relevance_failed_candidate_excluded_from_discrimination_set() -> None:
    memory = Memory(
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            minimum_supporting_episodes=1,
        ),
    )
    evidence_time = _T - timedelta(days=30)
    await memory.observe(
        ObservationInput(
            tenant_id="shop",
            subject_id="customer_42",
            actor_id="customer_42",
            source_namespace="chat.messages",
            source_record_id="grey-clothing",
            event_type="message",
            content="Customer prefers grey colours for formal clothing.",
            observed_at=evidence_time,
            metadata={
                "conversation_id": "arch-42",
                "entity_ids": ["customer_42"],
                "semantic_facts": [
                    {
                        "predicate": "colour_preference",
                        "object_value": "grey",
                        "cardinality": "many",
                        "polarity": "affirm",
                        "qualifiers": {},
                    }
                ],
            },
            context=ObservationContext(
                conversation_id="arch-42",
                domain="payments-api",
                activity="architecture-decision",
            ),
        )
    )
    await memory.process(tenant_id="shop", subject_id="customer_42", as_of=evidence_time)
    inspection = await memory.inspect_recall(
        "Could you recommend a backpack for me?",
        tenant_id="shop",
        subject_id="customer_42",
        as_of=_T,
        valid_at=_T,
        retrieval_context=_RETRIEVAL_CONTEXT,
        limit=10,
    )
    filtered = [
        candidate
        for candidate in inspection.rejected
        if candidate.disposition is RecallInspectionDisposition.FILTERED_INSUFFICIENT_RELEVANCE
        and candidate.memory_kind is MemoryKind.SEMANTIC
    ]
    assert filtered
    filtered_candidate = filtered[0]
    assert filtered_candidate.diagnostics is not None
    assert filtered_candidate.diagnostics.support_context is not None
    assert filtered_candidate.diagnostics.support_context.supports
    assert filtered_candidate not in discrimination_set(
        (*inspection.returned, *inspection.rejected)
    )


@pytest.mark.asyncio
async def test_prepare_context_render_and_record_context_use_unchanged() -> None:
    memory = await _build_store()
    first = await memory.prepare_context(_QUERY, tenant_id=_TENANT, prompt_budget_tokens=512)
    second = await memory.prepare_context(_QUERY, tenant_id=_TENANT, prompt_budget_tokens=512)
    assert first.render() == second.render()
    standalone = await memory.assess_memory(_QUERY, tenant_id=_TENANT)
    assert first.assessment.flags == standalone.flags
    assert first.assessment.signals == standalone.signals
    await memory.record_context_use(first)


@pytest.mark.asyncio
async def test_metamemory_flags_unchanged_without_retrieval_context() -> None:
    memory = await _build_store()
    assessment = await memory.assess_memory(_QUERY, tenant_id=_TENANT)
    assert assessment.context is not None
    assert assessment.context.state is RetrievalContextState.CONTEXT_NOT_PROVIDED
    assert MemoryAssessmentFlag.MISSING_KNOWLEDGE not in assessment.flags
