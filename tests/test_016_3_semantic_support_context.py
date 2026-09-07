"""Behaviour and regression tests for 0.16.3 semantic support-context propagation."""

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
from cogkura.models import (
    ActivationConfig,
    ActivationReferenceTrace,
    EpisodeEntity,
    EpisodeEvidenceInput,
    MemoryContextSignature,
    MemoryKind,
    RetrievalCue,
    SemanticCardinality,
    SemanticDerivationInput,
    SemanticDerivationRelation,
    SemanticMemoryStatus,
    SemanticPolarity,
    SemanticSupportContextReason,
    StoredEpisode,
    StoredSemanticMemory,
)
from cogkura.observations.encoding_context import ObservationContext

_T = datetime(2026, 8, 4, 10, 0, tzinfo=UTC)
_T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
_TENANT = "semantic-support"
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


def _observations() -> list[ObservationInput]:
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
    ]


async def _build_store(
    *,
    semantic_weight: float | None = None,
    episodic_weight: float | None = None,
) -> Memory:
    config_kwargs: dict[str, float] = {}
    if semantic_weight is not None:
        config_kwargs["semantic_context_reinstatement_weight"] = semantic_weight
    if episodic_weight is not None:
        config_kwargs["context_reinstatement_weight"] = episodic_weight
    memory = Memory(
        activation_config=ActivationConfig(**config_kwargs) if config_kwargs else None,
    )
    for observation in _observations():
        await memory.observe(observation)
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=_T.replace(hour=11))
    return memory


def _recall_snapshot(results: list) -> list[tuple[str, MemoryKind, float]]:
    return [
        (result.memory.memory_key, result.memory_kind, round(result.score, 6)) for result in results
    ]


def _semantic_results(results: list):
    return [result for result in results if result.memory_kind is MemoryKind.SEMANTIC]


def _episode(
    *,
    episode_id: str,
    memory_key: str,
    encoding_context: MemoryContextSignature | None = None,
) -> StoredEpisode:
    return StoredEpisode(
        id=episode_id,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        memory_key=memory_key,
        statement=f"Support episode {memory_key}.",
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
        entities=(EpisodeEntity(entity_id="redis", role="mention"),),
        metadata=MappingProxyType({}),
        created_at=_T0,
        updated_at=_T0,
        encoding_context=encoding_context or MemoryContextSignature(),
    )


def _semantic(
    *,
    semantic_id: str = "sem-1",
    memory_key: str = "service-redis",
    derivations: tuple[SemanticDerivationInput, ...],
    statement: str = "The service avoids Redis for coordination.",
) -> StoredSemanticMemory:
    return StoredSemanticMemory(
        id=semantic_id,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        memory_key=memory_key,
        slot_key=f"slot:{memory_key}",
        revision_key=f"rev:{memory_key}",
        revision_number=1,
        statement=statement,
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


def _reference(*, weight: int = 1) -> ActivationReferenceTrace:
    return ActivationReferenceTrace(referenced_at=_T0, weight=weight)


def _rank_semantic(
    *,
    semantic: StoredSemanticMemory,
    episodes: list[StoredEpisode],
    retrieval_context: RetrievalContext | None,
    semantic_weight: float = 0.25,
    episodic_weight: float = 0.5,
) -> list:
    activator = ACTRDeclarativeActivator()
    cue = RetrievalCue(
        text="Why did we decide not to use Redis?",
        entity_ids=("redis",),
        retrieval_context=retrieval_context,
    )
    candidates = [activation_candidate_from_semantic(semantic)]
    candidates.extend(activation_candidate_from_episode(episode) for episode in episodes)
    return activator.rank(
        candidates=candidates,
        cue=cue,
        references={candidate.identity: (_reference(),) for candidate in candidates},
        as_of=_T0 + timedelta(hours=1),
        config=ActivationConfig(
            retrieval_threshold=-10.0,
            context_reinstatement_weight=episodic_weight,
            semantic_context_reinstatement_weight=semantic_weight,
            enable_spreading_activation=False,
            enable_duplicate_collapse=False,
            enable_text_entity_seeding=False,
            enable_semantic_slot_admission=False,
        ),
        limit=10,
        episode_support_index=build_episode_support_index([semantic]),
        episode_by_id={episode.id: episode for episode in episodes},
    )


def _semantic_diagnostics(results: list, memory_key: str):
    semantic_result = next(
        result
        for result in results
        if result.memory.memory_key == memory_key  # type: ignore[union-attr]
    )
    assert semantic_result.diagnostics is not None
    return semantic_result.diagnostics


@pytest.mark.asyncio
async def test_no_retrieval_context_matches_baseline_semantic_recall() -> None:
    memory = await _build_store()
    baseline = await memory.recall(_QUERY, tenant_id=_TENANT)
    none_context = await memory.recall(_QUERY, tenant_id=_TENANT, retrieval_context=None)
    baseline_sem = _recall_snapshot(_semantic_results(baseline))
    none_sem = _recall_snapshot(_semantic_results(none_context))
    assert baseline_sem == none_sem


@pytest.mark.asyncio
async def test_semantic_weight_zero_with_context_matches_no_context_recall() -> None:
    memory = await _build_store(semantic_weight=0.0)
    baseline = await memory.recall(_QUERY, tenant_id=_TENANT)
    contextual = await memory.recall(
        _QUERY,
        tenant_id=_TENANT,
        retrieval_context=_RETRIEVAL_CONTEXT,
    )
    assert _recall_snapshot(_semantic_results(baseline)) == _recall_snapshot(
        _semantic_results(contextual)
    )
    inspection = await memory.inspect_recall(
        _QUERY,
        tenant_id=_TENANT,
        retrieval_context=_RETRIEVAL_CONTEXT,
        limit=10,
    )
    for candidate in inspection.returned:
        if candidate.memory_kind is not MemoryKind.SEMANTIC:
            continue
        assert candidate.diagnostics is not None
        assert candidate.diagnostics.support_context is not None
        assert candidate.diagnostics.support_context.reason is SemanticSupportContextReason.DISABLED


def test_duplicate_support_links_do_not_change_rs() -> None:
    derivations = tuple(
        SemanticDerivationInput(
            episode_id="ep-1",
            relation=SemanticDerivationRelation.SUPPORTS,
            contribution_score=0.9,
        )
        for _ in range(3)
    )
    semantic = _semantic(derivations=derivations)
    episode = _episode(
        episode_id="ep-1",
        memory_key="support-1",
        encoding_context=MemoryContextSignature(
            conversation_ids=("arch-42",),
            thread_ids=("queue-selection",),
            goals=("reduce-operational-complexity",),
            activities=("architecture-decision",),
            domains=("payments-api",),
            temporal_contexts=("queue-redesign",),
        ),
    )
    results = _rank_semantic(
        semantic=semantic,
        episodes=[episode],
        retrieval_context=_RETRIEVAL_CONTEXT,
        semantic_weight=1.0,
    )
    diagnostics = _semantic_diagnostics(results, semantic.memory_key)
    assert diagnostics.support_context is not None
    assert diagnostics.support_context.support_count == 1
    assert diagnostics.support_context.strength == pytest.approx(1.0)


def test_cross_context_four_domains_one_perfect_match() -> None:
    domains = ("payments-api", "analytics-api", "auth-api", "billing-api")
    derivations = tuple(
        SemanticDerivationInput(
            episode_id=f"ep-{domain}",
            relation=SemanticDerivationRelation.SUPPORTS,
            contribution_score=0.9,
        )
        for domain in domains
    )
    semantic = _semantic(derivations=derivations)
    perfect_context = MemoryContextSignature(
        conversation_ids=("arch-42",),
        thread_ids=("queue-selection",),
        goals=("reduce-operational-complexity",),
        activities=("architecture-decision",),
        domains=("payments-api",),
        temporal_contexts=("queue-redesign",),
    )
    episodes = [
        _episode(
            episode_id="ep-payments-api",
            memory_key="support-payments-api",
            encoding_context=perfect_context,
        ),
        *[
            _episode(
                episode_id=f"ep-{domain}",
                memory_key=f"support-{domain}",
                encoding_context=MemoryContextSignature(domains=(domain,)),
            )
            for domain in ("analytics-api", "auth-api", "billing-api")
        ],
    ]
    results = _rank_semantic(
        semantic=semantic,
        episodes=episodes,
        retrieval_context=_RETRIEVAL_CONTEXT,
        semantic_weight=1.0,
    )
    diagnostics = _semantic_diagnostics(results, semantic.memory_key)
    assert diagnostics.support_context is not None
    assert diagnostics.support_context.strength == pytest.approx(0.25)


def test_sparse_legacy_two_perfect_eight_unavailable() -> None:
    derivations = tuple(
        SemanticDerivationInput(
            episode_id=f"ep-{index}",
            relation=SemanticDerivationRelation.SUPPORTS,
            contribution_score=0.9,
        )
        for index in range(10)
    )
    semantic = _semantic(derivations=derivations)
    episodes = [
        _episode(
            episode_id=f"ep-{index}",
            memory_key=f"support-{index}",
            encoding_context=MemoryContextSignature(
                conversation_ids=("arch-42",),
                thread_ids=("queue-selection",),
                goals=("reduce-operational-complexity",),
                activities=("architecture-decision",),
                domains=("payments-api",),
                temporal_contexts=("queue-redesign",),
            )
            if index < 2
            else MemoryContextSignature(domains=(f"legacy-{index}",)),
        )
        for index in range(2)
    ]
    results = _rank_semantic(
        semantic=semantic,
        episodes=episodes,
        retrieval_context=_RETRIEVAL_CONTEXT,
        semantic_weight=1.0,
    )
    diagnostics = _semantic_diagnostics(results, semantic.memory_key)
    assert diagnostics.support_context is not None
    assert diagnostics.support_context.strength == pytest.approx(0.2)


def test_all_mismatch_gives_zero_contribution() -> None:
    derivations = tuple(
        SemanticDerivationInput(
            episode_id=f"ep-{index}",
            relation=SemanticDerivationRelation.SUPPORTS,
            contribution_score=0.9,
        )
        for index in range(3)
    )
    semantic = _semantic(derivations=derivations)
    episodes = [
        _episode(
            episode_id=f"ep-{index}",
            memory_key=f"support-{index}",
            encoding_context=MemoryContextSignature(domains=("analytics-api",)),
        )
        for index in range(3)
    ]
    results = _rank_semantic(
        semantic=semantic,
        episodes=episodes,
        retrieval_context=_RETRIEVAL_CONTEXT,
        semantic_weight=0.5,
    )
    diagnostics = _semantic_diagnostics(results, semantic.memory_key)
    assert diagnostics.support_context is not None
    assert diagnostics.support_context.strength == pytest.approx(0.0)
    assert diagnostics.support_context.activation_contribution == pytest.approx(0.0)
    assert diagnostics.context_match is None


def test_context_specific_semantic_rank_improves_vs_weight_zero() -> None:
    derivations = (
        SemanticDerivationInput(
            episode_id="ep-1",
            relation=SemanticDerivationRelation.SUPPORTS,
            contribution_score=0.9,
        ),
    )
    semantic = _semantic(derivations=derivations)
    episode = _episode(
        episode_id="ep-1",
        memory_key="support-1",
        encoding_context=MemoryContextSignature(
            conversation_ids=("arch-42",),
            thread_ids=("queue-selection",),
            goals=("reduce-operational-complexity",),
            activities=("architecture-decision",),
            domains=("payments-api",),
            temporal_contexts=("queue-redesign",),
        ),
    )
    disabled = _rank_semantic(
        semantic=semantic,
        episodes=[episode],
        retrieval_context=_RETRIEVAL_CONTEXT,
        semantic_weight=0.0,
    )
    enabled = _rank_semantic(
        semantic=semantic,
        episodes=[episode],
        retrieval_context=_RETRIEVAL_CONTEXT,
        semantic_weight=0.25,
    )
    disabled_score = next(
        result for result in disabled if result.memory_kind is MemoryKind.SEMANTIC
    ).activation
    enabled_score = next(
        result for result in enabled if result.memory_kind is MemoryKind.SEMANTIC
    ).activation
    assert enabled_score > disabled_score


def test_episodic_reinstatement_unchanged_with_semantic_support_context() -> None:
    episode = _episode(
        episode_id="ep-match",
        memory_key="ep-match",
        encoding_context=MemoryContextSignature(domains=("payments-api",)),
    )
    semantic = _semantic(
        derivations=(
            SemanticDerivationInput(
                episode_id="ep-support",
                relation=SemanticDerivationRelation.SUPPORTS,
                contribution_score=0.9,
            ),
        )
    )
    support_episode = _episode(
        episode_id="ep-support",
        memory_key="ep-support",
        encoding_context=MemoryContextSignature(domains=("payments-api",)),
    )
    activator = ACTRDeclarativeActivator()
    cue = RetrievalCue(
        text="redis payments",
        entity_ids=("redis",),
        retrieval_context=_RETRIEVAL_CONTEXT,
    )
    candidates = [
        activation_candidate_from_episode(episode),
        activation_candidate_from_semantic(semantic),
    ]
    references = {candidate.identity: (_reference(),) for candidate in candidates}
    config = ActivationConfig(
        retrieval_threshold=-10.0,
        context_reinstatement_weight=0.5,
        semantic_context_reinstatement_weight=0.25,
        enable_spreading_activation=False,
        enable_duplicate_collapse=False,
    )
    without_semantic_support = activator.rank(
        candidates=candidates,
        cue=cue,
        references=references,
        as_of=_T0 + timedelta(hours=1),
        config=config,
        limit=5,
        episode_support_index=build_episode_support_index([semantic]),
        episode_by_id={support_episode.id: support_episode},
    )
    episodic = next(
        result for result in without_semantic_support if result.memory_kind is MemoryKind.EPISODE
    )
    assert episodic.diagnostics is not None
    assert episodic.diagnostics.context_reinstatement is not None
    assert episodic.diagnostics.context_reinstatement.applied is True
    assert episodic.diagnostics.support_context is None


def test_max_support_forbidden_one_perfect_nineteen_mismatch() -> None:
    derivations = tuple(
        SemanticDerivationInput(
            episode_id=f"ep-{index}",
            relation=SemanticDerivationRelation.SUPPORTS,
            contribution_score=0.9,
        )
        for index in range(20)
    )
    semantic = _semantic(derivations=derivations)
    episodes = [
        _episode(
            episode_id="ep-0",
            memory_key="support-perfect",
            encoding_context=MemoryContextSignature(
                conversation_ids=("arch-42",),
                thread_ids=("queue-selection",),
                goals=("reduce-operational-complexity",),
                activities=("architecture-decision",),
                domains=("payments-api",),
                temporal_contexts=("queue-redesign",),
            ),
        ),
    ]
    episodes.extend(
        _episode(
            episode_id=f"ep-{index}",
            memory_key=f"support-{index}",
            encoding_context=MemoryContextSignature(domains=("analytics-api",)),
        )
        for index in range(1, 20)
    )
    results = _rank_semantic(
        semantic=semantic,
        episodes=episodes,
        retrieval_context=_RETRIEVAL_CONTEXT,
        semantic_weight=1.0,
    )
    diagnostics = _semantic_diagnostics(results, semantic.memory_key)
    assert diagnostics.support_context is not None
    assert diagnostics.support_context.strength == pytest.approx(0.05)


@pytest.mark.asyncio
async def test_adversarial_context_matching_semantic_does_not_bypass_relevance() -> None:
    memory = Memory()
    await memory.observe(
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
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=_T.replace(hour=11))
    inspection = await memory.inspect_recall(
        "Why did we decide not to use Redis?",
        tenant_id=_TENANT,
        retrieval_context=_RETRIEVAL_CONTEXT,
        limit=10,
    )
    unrelated = [
        candidate
        for candidate in inspection.returned
        if candidate.memory_kind is MemoryKind.SEMANTIC
        and "cafeteria" in candidate.memory.statement.lower()
    ]
    assert not unrelated


@pytest.mark.asyncio
async def test_working_memory_render_unchanged_without_context() -> None:
    memory = await _build_store()
    first = await memory.prepare_context(_QUERY, tenant_id=_TENANT, prompt_budget_tokens=512)
    second = await memory.prepare_context(_QUERY, tenant_id=_TENANT, prompt_budget_tokens=512)
    assert first.render() == second.render()


@pytest.mark.asyncio
async def test_working_memory_items_expose_no_support_context_field() -> None:
    memory = await _build_store()
    context = await memory.prepare_context(_QUERY, tenant_id=_TENANT)
    for item in context.items:
        assert not hasattr(item, "support_context")
    await memory.record_context_use(context)
