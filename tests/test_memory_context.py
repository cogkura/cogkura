"""Tests for MemoryContext, prepare_context, and record_context_use."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import MappingProxyType

import pytest

from cogkura import Memory, ObservationInput
from cogkura.algorithms.semantic import ComplementaryLearningSemanticConsolidator
from cogkura.exceptions import ValidationError
from cogkura.models import (
    ActivationConfig,
    EpisodeEntity,
    EpisodeEvidenceInput,
    MemoryAssessmentFlag,
    MemoryContext,
    MemoryIdentity,
    MemoryKind,
    MemoryRetentionState,
    MetamemoryConfig,
    RetrievalCue,
    StoredEpisode,
    StoredMemoryDynamics,
    WorkingMemoryConfig,
)
from cogkura.storage.in_memory_activation import InMemoryActivationStore
from cogkura.storage.in_memory_dynamics import InMemoryMemoryDynamicsStore
from cogkura.storage.in_memory_episode import InMemoryEpisodeStore
from cogkura.storage.in_memory_learning import InMemoryLearningStore
from cogkura.storage.in_memory_semantic import InMemorySemanticMemoryStore

_T0 = datetime(2026, 1, 1, tzinfo=UTC)
_T1 = datetime(2026, 6, 1, tzinfo=UTC)
_T2 = datetime(2027, 1, 1, tzinfo=UTC)
_T_QUERY = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)
_LOW_THRESHOLD = ActivationConfig(retrieval_threshold=-10.0)


class FixedTokenEstimator:
    def __init__(self, mapping: dict[str, int]) -> None:
        self._mapping = mapping

    def estimate(self, text: str) -> int:
        return self._mapping.get(text, 10)


def _episode(
    *,
    tenant_id: str = "company_123",
    subject_id: str | None = "customer_42",
    memory_key: str = "episode-a",
    statement: str = "PostgreSQL reduces operational complexity.",
    started_at: datetime = _T0,
) -> StoredEpisode:
    return StoredEpisode(
        id=f"id-{memory_key}",
        tenant_id=tenant_id,
        subject_id=subject_id,
        memory_key=memory_key,
        statement=statement,
        started_at=started_at,
        ended_at=started_at,
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
        entities=(),
        metadata=MappingProxyType({"episode": {"content_fingerprint": memory_key}}),
        created_at=started_at,
        updated_at=started_at,
    )


async def _memory_with_episodes(*episodes: StoredEpisode) -> Memory:
    episode_store = InMemoryEpisodeStore()
    for episode in episodes:
        episode_store._episodes[(episode.tenant_id, episode.memory_key)] = episode
    return Memory(
        episode_store=episode_store,
        activation_config=_LOW_THRESHOLD,
    )


def _semantic_observation(
    *,
    source_record_id: str,
    conversation_id: str,
    semantic_fact: dict,
    content: str,
    observed_at: datetime,
) -> ObservationInput:
    return ObservationInput(
        tenant_id="company_123",
        subject_id="customer_42",
        actor_id="customer_42",
        source_namespace="chat.messages",
        source_record_id=source_record_id,
        event_type="message",
        content=content,
        observed_at=observed_at,
        metadata={
            "conversation_id": conversation_id,
            "entity_ids": ["customer_42"],
            "semantic_facts": [semantic_fact],
        },
    )


@pytest.mark.asyncio
async def test_prepare_context_equivalent_working_memory() -> None:
    memory = await _memory_with_episodes(
        _episode(memory_key="wm-eq", statement="operational complexity database")
    )
    kwargs = {
        "query": "database operational complexity",
        "tenant_id": "company_123",
        "subject_id": "customer_42",
        "goal": "minimise operational complexity",
        "as_of": _T_QUERY,
    }
    existing = await memory.select_working_memory(**kwargs)
    context = await memory.prepare_context(**kwargs)
    assert context.working_memory == existing


@pytest.mark.asyncio
async def test_prepare_context_equivalent_assessment() -> None:
    memory = await _memory_with_episodes(
        _episode(memory_key="mm-eq", statement="PostgreSQL production workload")
    )
    kwargs = {
        "query": "PostgreSQL production",
        "tenant_id": "company_123",
        "subject_id": "customer_42",
        "as_of": _T_QUERY,
    }
    existing = await memory.assess_memory(**kwargs)
    context = await memory.prepare_context(**kwargs)
    assert context.assessment == existing


@pytest.mark.asyncio
async def test_select_working_memory_rejects_naive_valid_at() -> None:
    memory = await _memory_with_episodes(_episode(memory_key="naive-wm"))
    with pytest.raises(ValidationError, match="timezone-aware"):
        await memory.select_working_memory(
            "operational complexity",
            tenant_id="company_123",
            subject_id="customer_42",
            valid_at=datetime(2026, 1, 1),
        )


@pytest.mark.asyncio
async def test_assess_memory_rejects_naive_valid_at() -> None:
    memory = await _memory_with_episodes(_episode(memory_key="naive-mm"))
    with pytest.raises(ValidationError, match="timezone-aware"):
        await memory.assess_memory(
            "operational complexity",
            tenant_id="company_123",
            subject_id="customer_42",
            valid_at=datetime(2026, 1, 1),
        )


@pytest.mark.asyncio
async def test_prepare_context_rejects_naive_valid_at() -> None:
    memory = Memory(activation_config=_LOW_THRESHOLD)
    with pytest.raises(ValidationError, match="timezone-aware"):
        await memory.prepare_context(
            "query",
            tenant_id="company_123",
            valid_at=datetime(2026, 1, 1),
        )


@pytest.mark.asyncio
async def test_prepare_context_single_declarative_rank() -> None:
    memory = await _memory_with_episodes(
        _episode(memory_key="rank-once", statement="operational complexity")
    )
    rank_calls = 0
    original_rank = memory._declarative_activator.rank

    def counting_rank(*args: object, **kwargs: object) -> list:
        nonlocal rank_calls
        rank_calls += 1
        return original_rank(*args, **kwargs)

    memory._declarative_activator.rank = counting_rank  # type: ignore[method-assign]
    await memory.prepare_context(
        "operational complexity",
        tenant_id="company_123",
        subject_id="customer_42",
        as_of=_T_QUERY,
    )
    assert rank_calls == 1


@pytest.mark.asyncio
async def test_prepare_context_single_learning_state_load() -> None:
    memory = await _memory_with_episodes(
        _episode(memory_key="learn-once", statement="operational complexity")
    )
    list_states_calls = 0
    original_list_states = memory._learning_store.list_states

    async def counting_list_states(*args: object, **kwargs: object) -> list:
        nonlocal list_states_calls
        list_states_calls += 1
        return await original_list_states(*args, **kwargs)

    memory._learning_store.list_states = counting_list_states  # type: ignore[method-assign]
    await memory.prepare_context(
        "operational complexity",
        tenant_id="company_123",
        subject_id="customer_42",
        as_of=_T_QUERY,
    )
    assert list_states_calls == 1


@pytest.mark.asyncio
async def test_prepare_context_disabled_metamemory_raises() -> None:
    memory = Memory(metamemory_config=MetamemoryConfig(enabled=False))
    with pytest.raises(ValidationError, match="disabled"):
        await memory.prepare_context("query", tenant_id="company_123")


@pytest.mark.asyncio
async def test_prepare_context_no_activation_writes() -> None:
    activation_store = InMemoryActivationStore()
    memory = await _memory_with_episodes(
        _episode(memory_key="no-write", statement="operational complexity")
    )
    memory._activation_store = activation_store
    identity = MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="no-write")
    refs_before = await activation_store.list_reference_traces(
        tenant_id="company_123",
        identities=[identity],
        before_or_at=_T_QUERY,
    )
    await memory.prepare_context(
        "operational complexity",
        tenant_id="company_123",
        subject_id="customer_42",
        as_of=_T_QUERY,
    )
    refs_after = await activation_store.list_reference_traces(
        tenant_id="company_123",
        identities=[identity],
        before_or_at=_T_QUERY,
    )
    assert refs_before == refs_after


@pytest.mark.asyncio
async def test_prepare_context_no_dynamics_or_learning_writes() -> None:
    dynamics_store = InMemoryMemoryDynamicsStore()
    learning_store = InMemoryLearningStore()
    memory = await _memory_with_episodes(
        _episode(memory_key="no-durable-write", statement="operational complexity")
    )
    memory._dynamics_store = dynamics_store
    memory._learning_store = learning_store
    dynamics_before = len(dynamics_store._dynamics)
    learning_before = len(learning_store._states)
    await memory.prepare_context(
        "operational complexity",
        tenant_id="company_123",
        subject_id="customer_42",
        as_of=_T_QUERY,
    )
    assert len(dynamics_store._dynamics) == dynamics_before
    assert len(learning_store._states) == learning_before


@pytest.mark.asyncio
async def test_render_preserves_rank_and_excludes_unselected() -> None:
    first = "First operational complexity database fact."
    second = "Second operational complexity database mention."
    noise = "Manchester Christmas party unrelated."
    memory = await _memory_with_episodes(
        _episode(memory_key="rank-1", statement=first),
        _episode(memory_key="rank-2", statement=second),
        _episode(memory_key="noise", statement=noise),
    )
    memory._working_memory_config = WorkingMemoryConfig(
        candidate_pool_size=10,
        max_items=2,
    )
    context = await memory.prepare_context(
        "operational complexity database",
        tenant_id="company_123",
        subject_id="customer_42",
        goal="operational complexity database",
        as_of=_T_QUERY,
    )
    selected_keys = tuple(item.memory.memory_key for item in context.items)
    assert selected_keys == ("rank-2", "rank-1")
    assert context.render() == (
        "Relevant memory:\n\n"
        "- Second operational complexity database mention.\n"
        "- First operational complexity database fact."
    )
    assert noise not in context.render()


@pytest.mark.asyncio
async def test_render_multiline_statement_indentation() -> None:
    memory = await _memory_with_episodes(
        _episode(
            memory_key="multiline",
            statement="Line one.\nLine two.",
        )
    )
    context = await memory.prepare_context(
        "Line one",
        tenant_id="company_123",
        subject_id="customer_42",
        as_of=_T_QUERY,
    )
    assert context.render() == "Relevant memory:\n\n- Line one.\n  Line two."


@pytest.mark.asyncio
async def test_render_empty_context() -> None:
    memory = Memory(activation_config=_LOW_THRESHOLD)
    context = await memory.prepare_context(
        "nonexistent query",
        tenant_id="company_123",
        as_of=_T_QUERY,
    )
    assert context.render() == ""
    assert context.assessment.flags == (MemoryAssessmentFlag.NO_RETRIEVED_MEMORY,)


@pytest.mark.asyncio
async def test_unequal_candidate_pools() -> None:
    episodes = [
        _episode(
            memory_key=f"pool-{index}",
            statement=f"operational complexity item {index}",
        )
        for index in range(5)
    ]
    memory = await _memory_with_episodes(*episodes)
    memory._working_memory_config = WorkingMemoryConfig(candidate_pool_size=2)
    memory._metamemory_config = MetamemoryConfig(candidate_pool_size=4, max_report_items=4)
    wm_kwargs = {
        "query": "operational complexity",
        "tenant_id": "company_123",
        "subject_id": "customer_42",
        "as_of": _T_QUERY,
    }
    standalone_wm = await memory.select_working_memory(**wm_kwargs)
    standalone_mm = await memory.assess_memory(**wm_kwargs)
    context = await memory.prepare_context(**wm_kwargs)
    assert context.working_memory == standalone_wm
    assert context.assessment == standalone_mm


@pytest.mark.asyncio
async def test_prepare_context_propagates_valid_at() -> None:
    memory = await _memory_with_episodes(
        _episode(memory_key="valid-at", statement="operational complexity")
    )
    context = await memory.prepare_context(
        "operational complexity",
        tenant_id="company_123",
        subject_id="customer_42",
        valid_at=_T0,
        as_of=_T_QUERY,
    )
    assert context.valid_at == _T0
    assert context.assessment.valid_at == _T0


@pytest.mark.asyncio
async def test_memory_context_rejects_subject_mismatch() -> None:
    memory = await _memory_with_episodes(_episode(memory_key="ctx-subject"))
    context = await memory.prepare_context(
        "operational complexity",
        tenant_id="company_123",
        subject_id="customer_42",
        as_of=_T_QUERY,
    )
    bad_working_memory = replace(context.working_memory, subject_id="other_subject")
    with pytest.raises(ValidationError, match="working_memory.subject_id"):
        MemoryContext(
            tenant_id=context.tenant_id,
            subject_id=context.subject_id,
            query=context.query,
            goal=context.goal,
            prepared_at=context.prepared_at,
            valid_at=context.valid_at,
            working_memory=bad_working_memory,
            assessment=context.assessment,
        )


@pytest.mark.asyncio
async def test_memory_context_rejects_assessment_valid_at_mismatch() -> None:
    memory = await _memory_with_episodes(_episode(memory_key="ctx-valid-at"))
    context = await memory.prepare_context(
        "operational complexity",
        tenant_id="company_123",
        subject_id="customer_42",
        valid_at=_T0,
        as_of=_T_QUERY,
    )
    bad_assessment = replace(context.assessment, valid_at=None)
    with pytest.raises(ValidationError, match="assessment.valid_at"):
        MemoryContext(
            tenant_id=context.tenant_id,
            subject_id=context.subject_id,
            query=context.query,
            goal=context.goal,
            prepared_at=context.prepared_at,
            valid_at=context.valid_at,
            working_memory=context.working_memory,
            assessment=bad_assessment,
        )


@pytest.mark.asyncio
async def test_memory_context_rejects_none_subject_with_nested_subject() -> None:
    memory = await _memory_with_episodes(_episode(memory_key="ctx-none-subject"))
    context = await memory.prepare_context(
        "operational complexity",
        tenant_id="company_123",
        subject_id="customer_42",
        as_of=_T_QUERY,
    )
    with pytest.raises(ValidationError, match="working_memory.subject_id"):
        MemoryContext(
            tenant_id=context.tenant_id,
            subject_id=None,
            query=context.query,
            goal=context.goal,
            prepared_at=context.prepared_at,
            valid_at=context.valid_at,
            working_memory=context.working_memory,
            assessment=context.assessment,
        )


@pytest.mark.asyncio
async def test_prompt_budget_limits_selected_context() -> None:
    statements = ("alpha", "beta", "gamma", "delta")
    token_map = {
        "alpha": 300,
        "beta": 250,
        "gamma": 700,
        "delta": 200,
    }
    episodes = [_episode(memory_key=key, statement=key) for key in statements]
    memory = await _memory_with_episodes(*episodes)
    memory._token_estimator = FixedTokenEstimator(token_map)
    memory._working_memory_config = WorkingMemoryConfig(max_items=8)
    context = await memory.prepare_context(
        "alpha beta gamma delta",
        tenant_id="company_123",
        subject_id="customer_42",
        goal="alpha beta gamma delta",
        prompt_budget_tokens=750,
        as_of=_T_QUERY,
    )
    assert context.estimated_tokens <= 750
    assert len(context.items) == 1
    assert context.working_memory.budget_skipped_count >= 1


@pytest.mark.asyncio
async def test_goal_changes_working_memory_selection() -> None:
    operational = _episode(
        memory_key="goal-operational",
        statement="PostgreSQL reduces operational complexity for production.",
    )
    party = _episode(
        memory_key="goal-party",
        statement="The company Christmas party was held in Manchester.",
    )
    memory = await _memory_with_episodes(operational, party)
    memory._working_memory_config = WorkingMemoryConfig(max_items=1)
    operational_context = await memory.prepare_context(
        "PostgreSQL production",
        tenant_id="company_123",
        subject_id="customer_42",
        goal="minimise operational complexity",
        as_of=_T_QUERY,
    )
    party_context = await memory.prepare_context(
        "PostgreSQL production",
        tenant_id="company_123",
        subject_id="customer_42",
        goal="company Christmas party Manchester",
        as_of=_T_QUERY,
    )
    operational_keys = {item.memory.memory_key for item in operational_context.items}
    party_keys = {item.memory.memory_key for item in party_context.items}
    assert operational_keys == {"goal-operational"}
    assert party_keys == {"goal-party"}


@pytest.mark.asyncio
async def test_missing_knowledge_for_unresolved_slot_query() -> None:
    memory = await _memory_with_episodes(
        _episode(
            memory_key="region-discussion",
            statement="The team discussed eu-west but made no region decision.",
        )
    )
    context = await memory.prepare_context(
        RetrievalCue(
            text="what is the current primary region?",
            entity_ids=("billing-region",),
            predicate="primary",
        ),
        tenant_id="company_123",
        subject_id="customer_42",
        goal="current primary region",
        as_of=_T_QUERY,
    )
    assert MemoryAssessmentFlag.MISSING_KNOWLEDGE in context.assessment.flags


@pytest.mark.asyncio
async def test_prepare_context_excludes_episodes_after_valid_at() -> None:
    memory = Memory()
    tenant_id = "company_123"
    await memory.observe(
        ObservationInput(
            tenant_id=tenant_id,
            subject_id="team",
            source_namespace="chat.messages",
            source_record_id="early",
            content="Project Atlas selected Redis for job coordination.",
            observed_at=_T1,
        )
    )
    await memory.observe(
        ObservationInput(
            tenant_id=tenant_id,
            subject_id="team",
            source_namespace="chat.messages",
            source_record_id="late",
            content="Project Atlas team reviewed customer support escalations.",
            observed_at=_T2,
        )
    )
    await memory.process(tenant_id=tenant_id, as_of=_T2)
    episodes = await memory.list_episodes(tenant_id=tenant_id, subject_id="team")
    late_keys = {episode.memory_key for episode in episodes if episode.started_at > _T1}
    early_keys = {episode.memory_key for episode in episodes if episode.started_at <= _T1}
    assert late_keys
    assert early_keys
    context = await memory.prepare_context(
        "Project Atlas job coordination",
        tenant_id=tenant_id,
        subject_id="team",
        valid_at=_T1,
        as_of=_T2,
    )
    episode_keys = {
        item.memory.memory_key for item in context.items if item.memory_kind is MemoryKind.EPISODE
    }
    assert episode_keys & early_keys
    assert episode_keys.isdisjoint(late_keys)
    for item in context.items:
        if isinstance(item.memory, StoredEpisode):
            assert item.memory.started_at <= _T1


@pytest.mark.asyncio
async def test_prepare_context_historical_semantic_revision() -> None:
    memory = Memory(
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            minimum_supporting_episodes=1,
        )
    )
    await memory.observe(
        _semantic_observation(
            source_record_id="message_1",
            conversation_id="conv-1",
            semantic_fact={
                "predicate": "preferred_vendor",
                "object_value": "Acme",
                "object_entity_id": "acme",
                "cardinality": "one",
                "polarity": "affirm",
                "valid_from": _T0.isoformat(),
                "valid_until": _T1.isoformat(),
            },
            content="Preferred Acme.",
            observed_at=_T0,
        )
    )
    await memory.observe(
        _semantic_observation(
            source_record_id="message_2",
            conversation_id="conv-2",
            semantic_fact={
                "predicate": "preferred_vendor",
                "object_value": "Beta",
                "object_entity_id": "beta",
                "cardinality": "one",
                "polarity": "affirm",
                "valid_from": _T1.isoformat(),
                "valid_until": _T2.isoformat(),
            },
            content="Preferred Beta.",
            observed_at=_T1,
        )
    )
    await memory.process(tenant_id="company_123", as_of=_T2)
    context = await memory.prepare_context(
        RetrievalCue(predicate="preferred_vendor"),
        tenant_id="company_123",
        subject_id="customer_42",
        valid_at=_T0,
        as_of=_T2,
    )
    values = {
        item.memory.object_value.lower()
        for item in context.items
        if item.memory_kind is MemoryKind.SEMANTIC
    }
    assert "acme" in values


@pytest.mark.asyncio
async def test_prepare_context_excludes_forgotten_by_default() -> None:
    dynamics_store = InMemoryMemoryDynamicsStore()
    memory = await _memory_with_episodes(
        _episode(memory_key="forgotten-ep", statement="forgotten statement memory")
    )
    memory._dynamics_store = dynamics_store
    await dynamics_store.upsert_many(
        [
            StoredMemoryDynamics(
                tenant_id="company_123",
                memory_kind=MemoryKind.EPISODE,
                memory_key="forgotten-ep",
                retention_state=MemoryRetentionState.FORGOTTEN,
                last_base_level=-1.0,
                last_retention_score=0.01,
                below_threshold_since=_T_QUERY - timedelta(days=3),
                forgotten_at=_T_QUERY - timedelta(days=1),
                evaluated_at=_T_QUERY,
                updated_at=_T_QUERY,
            )
        ]
    )
    context = await memory.prepare_context(
        "forgotten statement",
        tenant_id="company_123",
        subject_id="customer_42",
        as_of=_T_QUERY,
    )
    assert context.items == ()
    included = await memory.prepare_context(
        "forgotten statement",
        tenant_id="company_123",
        subject_id="customer_42",
        as_of=_T_QUERY,
        include_forgotten=True,
    )
    assert {item.memory.memory_key for item in included.items} == {"forgotten-ep"}


@pytest.mark.asyncio
async def test_record_context_use_only_selected_memory_keys() -> None:
    activation_store = InMemoryActivationStore()
    episodes = [
        _episode(memory_key="selected", statement="operational complexity selected"),
        _episode(memory_key="distractor", statement="unrelated party Manchester noise"),
    ]
    memory = await _memory_with_episodes(*episodes)
    memory._activation_store = activation_store
    memory._working_memory_config = WorkingMemoryConfig(candidate_pool_size=10, max_items=1)
    context = await memory.prepare_context(
        "operational complexity",
        tenant_id="company_123",
        subject_id="customer_42",
        as_of=_T_QUERY,
    )
    selected_keys = {result.memory.memory_key for result in context.recall_results}
    assert selected_keys == {"selected"}
    assert "distractor" not in selected_keys
    await memory.record_context_use(context, referenced_at=_T_QUERY)
    for memory_key in ("selected", "distractor"):
        identity = MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key=memory_key)
        refs = await activation_store.list_reference_traces(
            tenant_id="company_123",
            identities=[identity],
            before_or_at=_T_QUERY + timedelta(seconds=1),
        )
        if memory_key in selected_keys:
            traces = refs.get(identity, ())
            assert len(traces) >= 1
            assert all(trace.referenced_at == _T_QUERY for trace in traces)
        else:
            assert refs.get(identity, ()) == ()


@pytest.mark.asyncio
async def test_record_context_use_forwards_min_score() -> None:
    episode_store = InMemoryEpisodeStore()
    activation_store = InMemoryActivationStore()
    distractor = _episode(
        memory_key="distractor",
        statement="Generic filler about charges.",
        subject_id="team",
    )
    gold = StoredEpisode(
        id="id-gold",
        tenant_id="company_123",
        subject_id="team",
        memory_key="gold",
        statement="PostgreSQL stores finalized charges for finance.",
        started_at=_T0,
        ended_at=_T0,
        confidence=0.9,
        importance=0.7,
        is_active=True,
        evidence=(
            EpisodeEvidenceInput(
                observation_id="obs-gold",
                observation_revision=1,
                sequence_number=0,
            ),
        ),
        entities=tuple(
            EpisodeEntity(entity_id=eid, role="mention") for eid in ("postgresql", "finance")
        ),
        metadata=MappingProxyType({"episode": {"content_fingerprint": "gold"}}),
        created_at=_T0,
        updated_at=_T0,
    )
    episode_store._episodes[(distractor.tenant_id, distractor.memory_key)] = distractor
    episode_store._episodes[(gold.tenant_id, gold.memory_key)] = gold
    memory = Memory(
        episode_store=episode_store,
        activation_store=activation_store,
        activation_config=ActivationConfig(retrieval_threshold=-10.0),
        working_memory_config=WorkingMemoryConfig(max_items=10, candidate_pool_size=10),
    )
    context = await memory.prepare_context(
        "PostgreSQL finalized charges",
        tenant_id="company_123",
        subject_id="team",
        as_of=_T_QUERY,
    )
    assert len(context.recall_results) >= 2
    await memory.record_context_use(
        context,
        referenced_at=_T_QUERY,
        min_score=0.997,
    )
    traces = await activation_store.list_reference_traces(
        tenant_id="company_123",
        identities=[
            MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="distractor"),
            MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="gold"),
        ],
        before_or_at=_T_QUERY + timedelta(seconds=1),
    )
    assert (
        len(traces.get(MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="distractor"), ()))
        == 0
    )
    assert (
        len(traces.get(MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="gold"), ())) >= 1
    )


@pytest.mark.asyncio
async def test_record_context_use_empty_context_writes_nothing() -> None:
    activation_store = InMemoryActivationStore()
    memory = Memory(activation_store=activation_store, activation_config=_LOW_THRESHOLD)
    context = await memory.prepare_context(
        "nonexistent query",
        tenant_id="company_123",
        as_of=_T_QUERY,
    )
    await memory.record_context_use(context, referenced_at=_T_QUERY)
    assert len(activation_store._references) == 0


@pytest.mark.asyncio
async def test_application_integration_flow() -> None:
    memory = Memory()
    await memory.observe(
        ObservationInput(
            tenant_id="shop",
            subject_id="customer_42",
            source_namespace="orders",
            source_record_id="order_123",
            event_type="purchase",
            content="Customer purchased running shoes in UK size 11.",
            observed_at=_T0,
            metadata={
                "semantic_facts": [
                    {
                        "predicate": "shoe_size",
                        "object_value": "UK 11",
                        "cardinality": "one",
                        "polarity": "affirm",
                    }
                ],
            },
        )
    )
    process_result = await memory.process(
        tenant_id="shop",
        subject_id="customer_42",
        as_of=_T0,
    )
    assert process_result.processed_at == _T0
    context = await memory.prepare_context(
        "I'd like another pair, but something lighter.",
        tenant_id="shop",
        subject_id="customer_42",
        goal="Help the customer choose suitable running shoes.",
        prompt_budget_tokens=1500,
        as_of=_T0,
    )
    rendered = context.render()
    assert rendered.startswith("Relevant memory:")
    assert context.estimated_tokens > 0
    assert context.recall_results
    activation_store = memory._activation_store
    await memory.record_context_use(context, referenced_at=_T0, request_id="req-1")
    identity = MemoryIdentity(
        memory_kind=context.recall_results[0].memory_kind,
        memory_key=context.recall_results[0].memory.memory_key,
    )
    refs = await activation_store.list_reference_traces(
        tenant_id="shop",
        identities=[identity],
        before_or_at=_T0 + timedelta(seconds=1),
    )
    traces = refs.get(identity, ())
    assert len(traces) >= 1
    assert all(trace.referenced_at == _T0 for trace in traces)


@pytest.mark.asyncio
async def test_prepare_context_excludes_live_superseded_semantic() -> None:
    memory = Memory(
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            minimum_supporting_episodes=1,
        )
    )
    await memory.observe(
        _semantic_observation(
            source_record_id="message_1",
            conversation_id="conv-1",
            semantic_fact={
                "predicate": "preferred_vendor",
                "object_value": "Acme",
                "object_entity_id": "acme",
                "cardinality": "one",
                "polarity": "affirm",
                "valid_from": _T0.isoformat(),
                "valid_until": _T1.isoformat(),
            },
            content="Preferred Acme.",
            observed_at=_T0,
        )
    )
    await memory.observe(
        _semantic_observation(
            source_record_id="message_2",
            conversation_id="conv-2",
            semantic_fact={
                "predicate": "preferred_vendor",
                "object_value": "Beta",
                "object_entity_id": "beta",
                "cardinality": "one",
                "polarity": "affirm",
                "valid_from": _T1.isoformat(),
                "valid_until": _T2.isoformat(),
            },
            content="Preferred Beta.",
            observed_at=_T1,
        )
    )
    await memory.process(tenant_id="company_123", as_of=_T2)
    context = await memory.prepare_context(
        RetrievalCue(predicate="preferred_vendor"),
        tenant_id="company_123",
        subject_id="customer_42",
        as_of=_T2,
    )
    semantic_values = {
        item.memory.object_value.lower()
        for item in context.items
        if item.memory_kind is MemoryKind.SEMANTIC
    }
    assert semantic_values == {"beta"}


@pytest.mark.asyncio
async def test_prepare_context_no_semantic_writes() -> None:
    semantic_store = InMemorySemanticMemoryStore()
    memory = await _memory_with_episodes(
        _episode(memory_key="semantic-read", statement="operational complexity")
    )
    memory._semantic_store = semantic_store
    memories_before = len(semantic_store._memories)
    revisions_before = len(semantic_store._revisions)
    await memory.prepare_context(
        "operational complexity",
        tenant_id="company_123",
        subject_id="customer_42",
        as_of=_T_QUERY,
    )
    assert len(semantic_store._memories) == memories_before
    assert len(semantic_store._revisions) == revisions_before


@pytest.mark.asyncio
async def test_record_context_use_forwards_request_id() -> None:
    activation_store = InMemoryActivationStore()
    memory = await _memory_with_episodes(
        _episode(memory_key="request-id", statement="operational complexity")
    )
    memory._activation_store = activation_store
    context = await memory.prepare_context(
        "operational complexity",
        tenant_id="company_123",
        subject_id="customer_42",
        as_of=_T_QUERY,
    )
    await memory.record_context_use(
        context,
        referenced_at=_T_QUERY,
        request_id="req-forward",
    )
    matching = [ref for ref in activation_store._references if ref.request_id == "req-forward"]
    assert matching
    assert all(ref.referenced_at == _T_QUERY for ref in matching)


@pytest.mark.asyncio
async def test_record_context_use_burst_limit() -> None:
    episode_store = InMemoryEpisodeStore()
    activation_store = InMemoryActivationStore()
    episode = _episode(
        memory_key="burst-target",
        statement="Target memory for burst limiting.",
        subject_id="team",
    )
    episode_store._episodes[(episode.tenant_id, episode.memory_key)] = episode
    memory = Memory(
        episode_store=episode_store,
        activation_store=activation_store,
        activation_config=ActivationConfig(
            retrieval_threshold=-10.0,
            access_burst_limit=2,
            access_burst_window_seconds=3600.0,
        ),
    )
    context = await memory.prepare_context(
        "target memory",
        tenant_id="company_123",
        subject_id="team",
        as_of=_T_QUERY,
    )
    identity = MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="burst-target")
    for offset in (0, 600, 1200, 1800):
        await memory.record_context_use(
            context,
            referenced_at=_T_QUERY + timedelta(seconds=offset),
        )
    traces = await activation_store.list_reference_traces(
        tenant_id="company_123",
        identities=[identity],
        before_or_at=_T_QUERY + timedelta(hours=2),
    )
    assert len(traces[identity]) == 2
