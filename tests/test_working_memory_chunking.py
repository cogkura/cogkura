"""Tests for 0.15.9 working-memory chunking and coverage-aware selection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import MappingProxyType

import pytest

from cogkura import Memory, ObservationInput
from cogkura.algorithms.semantic import ComplementaryLearningSemanticConsolidator
from cogkura.algorithms.working_memory import (
    ApproximateTokenEstimator,
    DeterministicWorkingMemorySelector,
)
from cogkura.models import (
    ActivationComponents,
    EpisodeEntity,
    EpisodeEvidenceInput,
    MemoryKind,
    RecallResult,
    RelevanceTier,
    RetrievalCue,
    RetrievalDiagnostics,
    SemanticCardinality,
    SemanticDerivationInput,
    SemanticDerivationRelation,
    SemanticMemoryStatus,
    SemanticPolarity,
    StoredEpisode,
    StoredSemanticMemory,
    WorkingMemoryChunkType,
    WorkingMemoryConfig,
    WorkingMemoryRejectionReason,
)

_TENANT = "shop"
_SUBJECT = "customer_42"
_T = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
_T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
_JACKET_QUERY = "Recommend a waterproof hiking jacket."
_SELECTOR = DeterministicWorkingMemorySelector()
_ESTIMATOR = ApproximateTokenEstimator()


def _memory() -> Memory:
    return Memory(
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            minimum_supporting_episodes=1,
        ),
    )


def _semantic_observation(
    *,
    source_record_id: str,
    conversation_id: str,
    semantic_fact: dict,
    observed_at: datetime,
    content: str,
    entity_ids: list[str] | None = None,
    semantic_facts: list[dict] | None = None,
) -> ObservationInput:
    metadata: dict[str, object] = {
        "conversation_id": conversation_id,
        "entity_ids": entity_ids or [_SUBJECT],
        "semantic_facts": semantic_facts or [semantic_fact],
    }
    return ObservationInput(
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        actor_id=_SUBJECT,
        source_namespace="chat.messages",
        source_record_id=source_record_id,
        event_type="message",
        content=content,
        observed_at=observed_at,
        metadata=metadata,
    )


def _episode_observation(
    *,
    source_record_id: str,
    conversation_id: str,
    observed_at: datetime,
    content: str,
    entity_ids: list[str] | None = None,
) -> ObservationInput:
    return ObservationInput(
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        actor_id=_SUBJECT,
        source_namespace="commerce.events",
        source_record_id=source_record_id,
        event_type="browse",
        content=content,
        observed_at=observed_at,
        metadata={
            "conversation_id": conversation_id,
            "entity_ids": entity_ids or [_SUBJECT],
        },
    )


def _episode(
    *,
    episode_id: str,
    memory_key: str,
    statement: str,
    observation_id: str,
    entity_ids: tuple[str, ...] = (),
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
                observation_id=observation_id,
                observation_revision=1,
                sequence_number=0,
            ),
        ),
        entities=tuple(EpisodeEntity(entity_id=eid, role="mention") for eid in entity_ids),
        metadata=MappingProxyType({}),
        created_at=_T0,
        updated_at=_T0,
    )


def _semantic(
    *,
    semantic_id: str,
    memory_key: str,
    statement: str,
    predicate: str,
    object_value: str,
    slot_key: str,
    cardinality: SemanticCardinality = SemanticCardinality.MANY,
    status: SemanticMemoryStatus = SemanticMemoryStatus.ACTIVE,
    derivations: tuple[SemanticDerivationInput, ...] = (),
    observation_ids: tuple[str, ...] = (),
) -> StoredSemanticMemory:
    observation_evidence = tuple(
        EpisodeEvidenceInput(
            observation_id=observation_id,
            observation_revision=1,
            sequence_number=index,
        )
        for index, observation_id in enumerate(observation_ids)
    )
    return StoredSemanticMemory(
        id=semantic_id,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        memory_key=memory_key,
        slot_key=slot_key,
        revision_key=f"legacy:{memory_key}",
        revision_number=1,
        statement=statement,
        subject_entity_id=_SUBJECT,
        predicate=predicate,
        object_value=object_value,
        object_entity_id=None,
        polarity=SemanticPolarity.AFFIRM,
        cardinality=cardinality,
        qualifiers=MappingProxyType({}),
        status=status,
        confidence=0.9,
        importance=0.7,
        support_count=1,
        contradiction_count=0,
        first_supported_at=_T0,
        last_supported_at=_T0,
        valid_from=_T0,
        valid_until=None,
        is_active=True,
        derivations=derivations,
        observation_evidence=observation_evidence,
        entities=(),
        metadata=MappingProxyType({}),
        created_at=_T0,
        updated_at=_T0,
    )


def _recall(
    memory: StoredEpisode | StoredSemanticMemory,
    *,
    score: float,
    activation: float = 1.0,
    relevance_tier: RelevanceTier = RelevanceTier.CONTEXTUAL,
) -> RecallResult:
    kind = MemoryKind.EPISODE if isinstance(memory, StoredEpisode) else MemoryKind.SEMANTIC
    return RecallResult(
        memory_kind=kind,
        memory=memory,
        activation=activation,
        score=score,
        latency_seconds=0.1,
        components=ActivationComponents(
            base_level=activation,
            spreading=0.0,
            partial_match=0.0,
            noise=0.0,
            total=activation,
        ),
        reason="test recall",
        diagnostics=RetrievalDiagnostics(
            rank_activation=score,
            accessibility_partial=score,
            ranking_partial=score,
            conjunction=score,
            text_coverage=score,
            text_cue_fit=score,
            temporal_mode="current",
            semantic_relevance=score,
            relevance_tier=relevance_tier.value,
        ),
    )


def _select(
    candidates: list[RecallResult],
    *,
    goal: str,
    config: WorkingMemoryConfig | None = None,
    prompt_budget_tokens: int | None = None,
) -> object:
    return _SELECTOR.select(
        candidates=candidates,
        goal=RetrievalCue(text=goal),
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        previous=None,
        as_of=_T,
        config=config or WorkingMemoryConfig(),
        token_estimator=_ESTIMATOR,
        prompt_budget_tokens=prompt_budget_tokens,
    )


async def _load_colour_collection_fixture(memory: Memory) -> None:
    evidence_time = _T - timedelta(days=60)
    await memory.observe(
        ObservationInput(
            tenant_id=_TENANT,
            subject_id=_SUBJECT,
            actor_id=_SUBJECT,
            source_namespace="chat.messages",
            source_record_id="colour",
            event_type="message",
            content="Customer prefers black, navy and grey waterproof jacket colours.",
            observed_at=evidence_time,
            metadata={
                "conversation_id": "conv-colour",
                "entity_ids": [_SUBJECT],
                "semantic_facts": [
                    {
                        "predicate": "colour_preference",
                        "object_value": colour,
                        "cardinality": "many",
                        "polarity": "affirm",
                        "qualifiers": {},
                    }
                    for colour in ("black", "navy", "grey")
                ],
            },
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=evidence_time)


async def _load_hiking_skiing_fixture(memory: Memory) -> None:
    t_hike = _T - timedelta(days=120)
    t_ski_ep = _T - timedelta(days=30)
    t_ski_sem = _T - timedelta(days=25)
    await memory.observe(
        _semantic_observation(
            source_record_id="hike",
            conversation_id="conv-hike",
            semantic_fact={
                "predicate": "activity_interest",
                "object_value": "hiking",
                "cardinality": "many",
                "polarity": "affirm",
                "qualifiers": {},
            },
            observed_at=t_hike,
            content="Customer enjoys hiking in the highlands.",
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=t_hike)
    await memory.observe(
        _episode_observation(
            source_record_id="ski-jacket",
            conversation_id="conv-ski",
            observed_at=t_ski_ep,
            content="Customer browsed waterproof ski jackets online.",
            entity_ids=[_SUBJECT, "ski-jacket-pro"],
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=t_ski_ep)
    await memory.observe(
        _semantic_observation(
            source_record_id="ski",
            conversation_id="conv-ski-sem",
            semantic_fact={
                "predicate": "activity_interest",
                "object_value": "skiing",
                "cardinality": "many",
                "polarity": "affirm",
                "qualifiers": {},
            },
            observed_at=t_ski_sem,
            content="Customer enjoys skiing and ski jackets.",
            entity_ids=[_SUBJECT, "ski-jacket-pro"],
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=t_ski_sem)


def test_collection_groups_shared_provenance_colours() -> None:
    shared_episode = "ep-colour"
    slot = "slot:colour_preference"
    colour_specs = (
        ("black", "Customer prefers black jackets."),
        ("navy", "Customer prefers navy jackets."),
        ("grey", "Customer prefers grey jackets."),
    )
    candidates = [
        _recall(
            _semantic(
                semantic_id=f"id-{colour}",
                memory_key=colour,
                statement=statement,
                predicate="colour_preference",
                object_value=colour,
                slot_key=slot,
                derivations=(
                    SemanticDerivationInput(
                        episode_id=shared_episode,
                        relation=SemanticDerivationRelation.SUPPORTS,
                        contribution_score=0.9,
                    ),
                ),
                observation_ids=("obs-colour",),
            ),
            score=0.7,
            relevance_tier=RelevanceTier.DIRECT_SEMANTIC,
        )
        for colour, statement in colour_specs
    ]
    snapshot = _select(candidates, goal="waterproof jacket colours")
    assert snapshot.selected_count == 1
    item = snapshot.items[0]
    assert item.chunk is not None
    assert item.chunk.chunk_type is WorkingMemoryChunkType.SEMANTIC_COLLECTION
    assert item.chunk.members_total == 3
    assert item.chunk.members_included == 3
    assert len(item.member_recalls) == 3
    assert "black" in item.chunk.serialized_text
    assert "navy" in item.chunk.serialized_text
    assert "grey" in item.chunk.serialized_text


def test_independent_many_hiking_and_skiing_are_separate_chunks() -> None:
    slot = "slot:activity_interest"
    hiking = _semantic(
        semantic_id="id-hike",
        memory_key="hike",
        statement="Customer enjoys hiking.",
        predicate="activity_interest",
        object_value="hiking",
        slot_key=slot,
        derivations=(
            SemanticDerivationInput(
                episode_id="ep-hike",
                relation=SemanticDerivationRelation.SUPPORTS,
                contribution_score=0.9,
            ),
        ),
        observation_ids=("obs-hike",),
    )
    skiing = _semantic(
        semantic_id="id-ski",
        memory_key="ski",
        statement="Customer enjoys skiing.",
        predicate="activity_interest",
        object_value="skiing",
        slot_key=slot,
        derivations=(
            SemanticDerivationInput(
                episode_id="ep-ski",
                relation=SemanticDerivationRelation.SUPPORTS,
                contribution_score=0.9,
            ),
        ),
        observation_ids=("obs-ski",),
    )
    snapshot = _select(
        [
            _recall(hiking, score=0.8, relevance_tier=RelevanceTier.DIRECT_VALUE),
            _recall(skiing, score=0.6, relevance_tier=RelevanceTier.CONTEXTUAL),
        ],
        goal="hiking equipment",
        config=WorkingMemoryConfig(max_items=4),
    )
    assert snapshot.selected_chunk_count == 2
    coverage_keys = {item.chunk.coverage_key for item in snapshot.items if item.chunk}
    assert f"slot:{slot}:object:hiking" in coverage_keys
    assert f"slot:{slot}:object:skiing" in coverage_keys


def test_semantic_with_support_attaches_episode() -> None:
    semantic = _semantic(
        semantic_id="id-light",
        memory_key="lightweight",
        statement="Customer prefers lightweight outerwear.",
        predicate="outerwear_weight_preference",
        object_value="lightweight",
        slot_key="slot:outerwear_weight_preference",
        cardinality=SemanticCardinality.ONE,
        derivations=(
            SemanticDerivationInput(
                episode_id="ep-light",
                relation=SemanticDerivationRelation.SUPPORTS,
                contribution_score=0.9,
            ),
        ),
    )
    support = _episode(
        episode_id="ep-light",
        memory_key="light-support",
        statement="Customer praised the lightweight waterproof shell purchase.",
        observation_id="obs-light",
    )
    subject_only = _episode(
        episode_id="ep-subject",
        memory_key="subject-only",
        statement="Customer visited the store.",
        observation_id="obs-subject",
    )
    snapshot = _select(
        [
            _recall(semantic, score=0.85, relevance_tier=RelevanceTier.DIRECT_SEMANTIC),
            _recall(support, score=0.7, relevance_tier=RelevanceTier.EVIDENCE_ASSOCIATION),
            _recall(subject_only, score=0.65),
        ],
        goal="lightweight hiking jacket",
        config=WorkingMemoryConfig(max_items=4),
    )
    chunk_types = {item.chunk.chunk_type for item in snapshot.items if item.chunk}
    assert WorkingMemoryChunkType.SEMANTIC_WITH_SUPPORT in chunk_types
    assert WorkingMemoryChunkType.EPISODIC in chunk_types
    support_chunk = next(
        item
        for item in snapshot.items
        if item.chunk and item.chunk.chunk_type is WorkingMemoryChunkType.SEMANTIC_WITH_SUPPORT
    )
    assert len(support_chunk.member_recalls) == 2


@pytest.mark.asyncio
async def test_tight_capacity_prefers_collection_over_extra_colours() -> None:
    memory = _memory()
    memory._working_memory_config = WorkingMemoryConfig(max_items=2, candidate_pool_size=20)
    await _load_colour_collection_fixture(memory)
    t_shell = _T - timedelta(days=30)
    await memory.observe(
        _episode_observation(
            source_record_id="shell-browse",
            conversation_id="conv-shell",
            observed_at=t_shell,
            content="Customer compared waterproof hiking shell jacket options online.",
            entity_ids=[_SUBJECT, "northpeak-alpine-shell"],
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=t_shell)
    await memory.observe(
        _semantic_observation(
            source_record_id="return",
            conversation_id="conv-return",
            semantic_fact={
                "predicate": "product_fit_issue",
                "object_value": "northpeak-alpine-shell:sleeves_too_short",
                "cardinality": "many",
                "polarity": "affirm",
                "qualifiers": {},
            },
            observed_at=_T - timedelta(days=20),
            content="Return processed for northpeak-alpine-shell due to sleeve length.",
            entity_ids=[_SUBJECT, "northpeak-alpine-shell"],
        )
    )
    await memory.process(tenant_id=_TENANT, subject_id=_SUBJECT, as_of=_T - timedelta(days=20))

    snapshot = await memory.select_working_memory(
        _JACKET_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
    )
    assert snapshot.selected_count <= 2
    colour_items = [
        item
        for item in snapshot.items
        if item.chunk and item.chunk.chunk_type is WorkingMemoryChunkType.SEMANTIC_COLLECTION
    ]
    assert colour_items
    assert colour_items[0].chunk.members_included == 3


@pytest.mark.asyncio
async def test_hiking_direct_value_stays_selected() -> None:
    memory = _memory()
    memory._working_memory_config = WorkingMemoryConfig(max_items=4, candidate_pool_size=20)
    await _load_hiking_skiing_fixture(memory)
    snapshot = await memory.select_working_memory(
        _JACKET_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
    )
    hiking_selected = any(
        recall.memory_kind is MemoryKind.SEMANTIC and recall.memory.object_value == "hiking"
        for recall in snapshot.recall_results
    )
    assert hiking_selected


@pytest.mark.asyncio
async def test_enable_chunking_false_matches_item_level_capacity() -> None:
    memory = _memory()
    await _load_colour_collection_fixture(memory)
    memory._working_memory_config = WorkingMemoryConfig(
        max_items=4,
        candidate_pool_size=20,
        enable_chunking=True,
    )
    chunked = await memory.select_working_memory(
        _JACKET_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
    )
    memory._working_memory_config = WorkingMemoryConfig(
        max_items=4,
        candidate_pool_size=20,
        enable_chunking=False,
    )
    raw = await memory.select_working_memory(
        _JACKET_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
    )
    assert chunked.selected_count < raw.selected_count
    assert all(item.chunk is not None for item in chunked.items)
    assert all(item.chunk is None for item in raw.items)


@pytest.mark.asyncio
async def test_prepare_context_render_uses_chunk_serialized_text() -> None:
    memory = _memory()
    memory._working_memory_config = WorkingMemoryConfig(max_items=4, candidate_pool_size=20)
    await _load_colour_collection_fixture(memory)
    context = await memory.prepare_context(
        _JACKET_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
    )
    colour_item = next(
        item
        for item in context.items
        if item.chunk and item.chunk.chunk_type is WorkingMemoryChunkType.SEMANTIC_COLLECTION
    )
    assert colour_item.chunk is not None
    assert colour_item.chunk.serialized_text in context.render()


@pytest.mark.asyncio
async def test_record_context_use_flattens_chunk_members() -> None:
    memory = _memory()
    memory._working_memory_config = WorkingMemoryConfig(max_items=4, candidate_pool_size=20)
    await _load_colour_collection_fixture(memory)
    context = await memory.prepare_context(
        _JACKET_QUERY,
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        as_of=_T,
    )
    colour_chunk = next(
        item
        for item in context.items
        if item.chunk and item.chunk.chunk_type is WorkingMemoryChunkType.SEMANTIC_COLLECTION
    )
    member_keys = {
        recall.memory.memory_key
        for recall in colour_chunk.member_recalls
        if recall.memory_kind is MemoryKind.SEMANTIC
    }
    flattened_keys = {
        recall.memory.memory_key
        for recall in context.recall_results
        if recall.memory_kind is MemoryKind.SEMANTIC
        and recall.memory.predicate == "colour_preference"
    }
    assert member_keys.issubset(flattened_keys)
    assert len(member_keys) == 3


def test_token_budget_trims_collection_members_deterministically() -> None:
    shared_episode = "ep-colour"
    slot = "slot:colour_preference"
    candidates = []
    for colour, score in (("black", 0.9), ("navy", 0.7), ("grey", 0.5)):
        candidates.append(
            _recall(
                _semantic(
                    semantic_id=f"id-{colour}",
                    memory_key=colour,
                    statement=f"Customer prefers {colour} jackets.",
                    predicate="colour_preference",
                    object_value=colour,
                    slot_key=slot,
                    derivations=(
                        SemanticDerivationInput(
                            episode_id=shared_episode,
                            relation=SemanticDerivationRelation.SUPPORTS,
                            contribution_score=0.9,
                        ),
                    ),
                    observation_ids=("obs-colour",),
                ),
                score=score,
                relevance_tier=RelevanceTier.DIRECT_SEMANTIC,
            )
        )

    class TightEstimator:
        def estimate(self, text: str) -> int:
            if "and" in text and text.count(",") >= 1:
                return 20
            return 4

    snapshot = _SELECTOR.select(
        candidates=candidates,
        goal=RetrievalCue(text="jacket colours"),
        tenant_id=_TENANT,
        subject_id=_SUBJECT,
        previous=None,
        as_of=_T,
        config=WorkingMemoryConfig(max_items=4),
        token_estimator=TightEstimator(),
        prompt_budget_tokens=10,
    )
    item = snapshot.items[0]
    assert item.chunk is not None
    assert item.chunk.members_included < item.chunk.members_total
    assert item.chunk.members_omitted > 0


def test_chunk_ids_and_render_are_deterministic() -> None:
    shared_episode = "ep-colour"
    slot = "slot:colour_preference"
    candidates = [
        _recall(
            _semantic(
                semantic_id=f"id-{colour}",
                memory_key=colour,
                statement=f"Customer prefers {colour} jackets.",
                predicate="colour_preference",
                object_value=colour,
                slot_key=slot,
                derivations=(
                    SemanticDerivationInput(
                        episode_id=shared_episode,
                        relation=SemanticDerivationRelation.SUPPORTS,
                        contribution_score=0.9,
                    ),
                ),
                observation_ids=("obs-colour",),
            ),
            score=0.7,
            relevance_tier=RelevanceTier.DIRECT_SEMANTIC,
        )
        for colour in ("black", "navy", "grey")
    ]
    first = _select(candidates, goal="waterproof jacket colours")
    second = _select(candidates, goal="waterproof jacket colours")
    assert first.items[0].chunk is not None
    assert second.items[0].chunk is not None
    assert first.items[0].chunk.chunk_id == second.items[0].chunk.chunk_id
    assert first.items[0].chunk.serialized_text == second.items[0].chunk.serialized_text


def test_rejected_chunks_expose_capacity_reason() -> None:
    shared_episode = "ep-colour"
    slot = "slot:colour_preference"
    candidates = []
    for index, colour in enumerate(("black", "navy", "grey", "olive")):
        candidates.append(
            _recall(
                _semantic(
                    semantic_id=f"id-{colour}",
                    memory_key=colour,
                    statement=f"Customer prefers {colour} jackets.",
                    predicate="colour_preference",
                    object_value=colour,
                    slot_key=slot,
                    derivations=(
                        SemanticDerivationInput(
                            episode_id=shared_episode,
                            relation=SemanticDerivationRelation.SUPPORTS,
                            contribution_score=0.9,
                        ),
                    ),
                    observation_ids=("obs-colour",),
                ),
                score=0.9 - (index * 0.05),
                relevance_tier=RelevanceTier.DIRECT_SEMANTIC,
            )
        )
    candidates.append(
        _recall(
            _semantic(
                semantic_id="id-hike",
                memory_key="hike",
                statement="Customer enjoys hiking.",
                predicate="activity_interest",
                object_value="hiking",
                slot_key="slot:activity_interest",
                derivations=(
                    SemanticDerivationInput(
                        episode_id="ep-hike",
                        relation=SemanticDerivationRelation.SUPPORTS,
                        contribution_score=0.9,
                    ),
                ),
                observation_ids=("obs-hike",),
            ),
            score=0.8,
            relevance_tier=RelevanceTier.DIRECT_VALUE,
        )
    )
    snapshot = _select(
        candidates,
        goal="hiking jacket colours",
        config=WorkingMemoryConfig(max_items=1),
    )
    assert snapshot.selected_count == 1
    rejected = [
        chunk
        for chunk in snapshot.chunks
        if not chunk.selected and chunk.rejection_reason is not None
    ]
    assert rejected
    assert any(
        reason == WorkingMemoryRejectionReason.CHUNK_CAPACITY.value
        for reason in (chunk.rejection_reason for chunk in rejected)
    )
