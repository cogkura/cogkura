"""Tests for 0.12 string cues, slot admission, and current-state behaviour."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import MappingProxyType

import pytest

from cogkura import Memory
from cogkura.algorithms.activation import (
    ACTRDeclarativeActivator,
    activation_candidate_from_episode,
    activation_candidate_from_semantic,
    build_episode_support_index,
)
from cogkura.models import (
    ActivationConfig,
    EpisodeEntity,
    EpisodeEvidenceInput,
    MemoryIdentity,
    MemoryKind,
    RetrievalCue,
    SemanticCardinality,
    SemanticDerivationInput,
    SemanticDerivationRelation,
    SemanticMemoryStatus,
    SemanticPolarity,
    StoredEpisode,
    StoredSemanticMemory,
)
from cogkura.storage.in_memory_activation import InMemoryActivationStore
from cogkura.storage.in_memory_episode import InMemoryEpisodeStore

_T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
_T1 = datetime(2026, 1, 4, 12, 0, tzinfo=UTC)


def _episode(
    *,
    episode_id: str,
    memory_key: str,
    statement: str,
    entity_ids: tuple[str, ...] = (),
    started_at: datetime | None = None,
) -> StoredEpisode:
    started = started_at or _T0
    return StoredEpisode(
        id=episode_id,
        tenant_id="company_123",
        subject_id="team",
        memory_key=memory_key,
        statement=statement,
        started_at=started,
        ended_at=started,
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
        created_at=started,
        updated_at=started,
    )


def _semantic(
    *,
    semantic_id: str,
    memory_key: str,
    statement: str,
    status: SemanticMemoryStatus,
    entity_ids: tuple[str, ...] = (),
    object_value: str = "value",
    derivations: tuple[SemanticDerivationInput, ...] = (),
) -> StoredSemanticMemory:
    return StoredSemanticMemory(
        id=semantic_id,
        tenant_id="company_123",
        subject_id="team",
        memory_key=memory_key,
        slot_key=f"slot:{memory_key}",
        revision_key=f"legacy:{memory_key}",
        revision_number=1,
        statement=statement,
        subject_entity_id=None,
        predicate="backing-store",
        object_value=object_value,
        object_entity_id=None,
        polarity=SemanticPolarity.AFFIRM,
        cardinality=SemanticCardinality.ONE,
        qualifiers=MappingProxyType({}),
        confidence=0.9,
        importance=0.7,
        status=status,
        support_count=1,
        contradiction_count=0,
        first_supported_at=_T0,
        last_supported_at=_T1,
        valid_from=None,
        valid_until=None,
        is_active=True,
        derivations=derivations,
        observation_evidence=(),
        entities=tuple(EpisodeEntity(entity_id=eid, role="mention") for eid in entity_ids),
        metadata=MappingProxyType({}),
        created_at=_T0,
        updated_at=_T1,
    )


def test_string_cue_seeding_enables_spreading() -> None:
    activator = ACTRDeclarativeActivator()
    finance_episode = _episode(
        episode_id="ep-finance",
        memory_key="finance-pain",
        statement="Finance reported operational pain from manual ledger reconciliation.",
        entity_ids=("finance",),
    )
    ledger_episode = _episode(
        episode_id="ep-ledger",
        memory_key="ledger-change",
        statement="The charge-ledger workflow was redesigned after finance escalation.",
        entity_ids=("charge-ledger",),
    )
    candidates = [
        activation_candidate_from_episode(finance_episode),
        activation_candidate_from_episode(ledger_episode),
    ]
    config = ActivationConfig(retrieval_threshold=-10.0)
    results = activator.rank(
        candidates=candidates,
        cue=RetrievalCue(
            text="What operational pain from finance drove the ledger change?",
        ),
        references={},
        as_of=_T1,
        config=config,
        limit=5,
    )
    ledger_result = next(
        result
        for result in results
        if result.memory.memory_key == "ledger-change"  # type: ignore[union-attr]
    )
    assert ledger_result.components.spreading > 0.0
    assert "finance" in ledger_result.reason or "charge-ledger" in ledger_result.reason


def test_explicit_entity_cue_matches_seeding_disabled() -> None:
    activator = ACTRDeclarativeActivator()
    episode = _episode(
        episode_id="ep-finance",
        memory_key="finance-pain",
        statement="Finance reported operational pain.",
        entity_ids=("finance",),
    )
    candidate = activation_candidate_from_episode(episode)
    config = ActivationConfig(retrieval_threshold=-10.0, enable_text_entity_seeding=False)
    explicit = activator.rank(
        candidates=[candidate],
        cue=RetrievalCue(text="finance pain", entity_ids=("finance",)),
        references={},
        as_of=_T1,
        config=config,
        limit=5,
    )
    seeded_off = activator.rank(
        candidates=[candidate],
        cue=RetrievalCue(text="finance pain", entity_ids=("finance",)),
        references={},
        as_of=_T1,
        config=config,
        limit=5,
    )
    assert explicit[0].activation == seeded_off[0].activation


def test_string_cue_without_entity_metadata_has_no_spreading() -> None:
    activator = ACTRDeclarativeActivator()
    episode = _episode(
        episode_id="ep-plain",
        memory_key="plain",
        statement="A generic operational update with no entity tags.",
        entity_ids=(),
    )
    candidate = activation_candidate_from_episode(episode)
    results = activator.rank(
        candidates=[candidate],
        cue=RetrievalCue(text="operational update"),
        references={},
        as_of=_T1,
        config=ActivationConfig(retrieval_threshold=-10.0),
        limit=5,
    )
    assert results[0].components.spreading == 0.0


def test_semantic_slot_admission_keeps_active_fact_below_threshold() -> None:
    activator = ACTRDeclarativeActivator()
    postgres_episode = _episode(
        episode_id="ep-postgres",
        memory_key="postgres-support",
        statement="Engineering migrated charge persistence to PostgreSQL last quarter.",
        entity_ids=("postgresql", "charges"),
    )
    dynamo_episode = _episode(
        episode_id="ep-dynamo",
        memory_key="dynamo-support",
        statement="Charges were stored in DynamoDB before migration.",
        entity_ids=("dynamodb", "charges"),
    )
    active_semantic = _semantic(
        semantic_id="sem-postgres",
        memory_key="backing-postgres",
        statement="Finalized charges persist in PostgreSQL now.",
        status=SemanticMemoryStatus.ACTIVE,
        entity_ids=("postgresql", "charges"),
        object_value="postgresql",
        derivations=(
            SemanticDerivationInput(
                episode_id="ep-postgres",
                relation=SemanticDerivationRelation.SUPPORTS,
                contribution_score=0.9,
            ),
        ),
    )
    superseded_semantic = _semantic(
        semantic_id="sem-dynamo",
        memory_key="backing-dynamo",
        statement="Charges persisted in DynamoDB.",
        status=SemanticMemoryStatus.SUPERSEDED,
        entity_ids=("dynamodb", "charges"),
        object_value="dynamodb",
        derivations=(
            SemanticDerivationInput(
                episode_id="ep-dynamo",
                relation=SemanticDerivationRelation.SUPPORTS,
                contribution_score=0.9,
            ),
        ),
    )
    filler = [
        activation_candidate_from_episode(
            _episode(
                episode_id=f"ep-fill-{index}",
                memory_key=f"fill-{index}",
                statement=f"Unrelated filler about charges scenario {index}.",
                entity_ids=("charges",),
            )
        )
        for index in range(12)
    ]
    candidates = filler + [
        activation_candidate_from_episode(dynamo_episode),
        activation_candidate_from_episode(postgres_episode),
        activation_candidate_from_semantic(active_semantic),
    ]
    support_index = build_episode_support_index([active_semantic, superseded_semantic])
    config = ActivationConfig(retrieval_threshold=0.0, enable_text_entity_seeding=False)
    results = activator.rank(
        candidates=candidates,
        cue=RetrievalCue(text="Where do finalized charges persist now?"),
        references={},
        as_of=_T1,
        config=config,
        limit=5,
        episode_support_index=support_index,
    )
    keys = {result.memory.memory_key for result in results}  # type: ignore[union-attr]
    assert "backing-postgres" in keys
    assert "postgres-support" in keys


def test_current_state_penalises_superseded_support_episode() -> None:
    activator = ACTRDeclarativeActivator()
    dynamo_episode = _episode(
        episode_id="ep-dynamo",
        memory_key="dynamo-support",
        statement="Charges were stored in DynamoDB.",
        entity_ids=("dynamodb",),
    )
    postgres_episode = _episode(
        episode_id="ep-postgres",
        memory_key="postgres-support",
        statement="Charges now persist in PostgreSQL.",
        entity_ids=("postgresql",),
    )
    active_semantic = _semantic(
        semantic_id="sem-postgres",
        memory_key="backing-postgres",
        statement="PostgreSQL is the current backing store.",
        status=SemanticMemoryStatus.ACTIVE,
        entity_ids=("postgresql",),
        derivations=(
            SemanticDerivationInput(
                episode_id="ep-postgres",
                relation=SemanticDerivationRelation.SUPPORTS,
                contribution_score=0.9,
            ),
        ),
    )
    superseded_semantic = _semantic(
        semantic_id="sem-dynamo",
        memory_key="backing-dynamo",
        statement="DynamoDB was the backing store.",
        status=SemanticMemoryStatus.SUPERSEDED,
        entity_ids=("dynamodb",),
        derivations=(
            SemanticDerivationInput(
                episode_id="ep-dynamo",
                relation=SemanticDerivationRelation.SUPPORTS,
                contribution_score=0.9,
            ),
        ),
    )
    candidates = [
        activation_candidate_from_episode(dynamo_episode),
        activation_candidate_from_episode(postgres_episode),
        activation_candidate_from_semantic(active_semantic),
    ]
    support_index = build_episode_support_index([active_semantic, superseded_semantic])
    config = ActivationConfig(retrieval_threshold=-10.0, enable_text_entity_seeding=False)
    results = activator.rank(
        candidates=candidates,
        cue=RetrievalCue(text="Where do charges persist currently now?"),
        references={},
        as_of=_T1,
        config=config,
        limit=5,
        episode_support_index=support_index,
    )
    episode_results = [result for result in results if result.memory_kind is MemoryKind.EPISODE]
    assert episode_results[0].memory.memory_key == "postgres-support"  # type: ignore[union-attr]
    assert episode_results[0].components.current_state > episode_results[1].components.current_state


def test_without_current_state_tokens_superseded_episode_stays_retrievable() -> None:
    activator = ACTRDeclarativeActivator()
    dynamo_episode = _episode(
        episode_id="ep-dynamo",
        memory_key="dynamo-support",
        statement="Historical DynamoDB charge storage details.",
        entity_ids=("dynamodb",),
    )
    postgres_episode = _episode(
        episode_id="ep-postgres",
        memory_key="postgres-support",
        statement="PostgreSQL charge storage details.",
        entity_ids=("postgresql",),
    )
    superseded_semantic = _semantic(
        semantic_id="sem-dynamo",
        memory_key="backing-dynamo",
        statement="DynamoDB backing store.",
        status=SemanticMemoryStatus.SUPERSEDED,
        entity_ids=("dynamodb",),
        derivations=(
            SemanticDerivationInput(
                episode_id="ep-dynamo",
                relation=SemanticDerivationRelation.SUPPORTS,
                contribution_score=0.9,
            ),
        ),
    )
    candidates = [
        activation_candidate_from_episode(dynamo_episode),
        activation_candidate_from_episode(postgres_episode),
    ]
    support_index = build_episode_support_index([superseded_semantic])
    results = activator.rank(
        candidates=candidates,
        cue=RetrievalCue(text="DynamoDB charge storage history"),
        references={},
        as_of=_T1,
        config=ActivationConfig(retrieval_threshold=-10.0),
        limit=5,
        episode_support_index=support_index,
    )
    keys = {result.memory.memory_key for result in results}  # type: ignore[union-attr]
    assert "dynamo-support" in keys


def test_numeric_token_collapse_frees_slots_for_distinct_gold() -> None:
    activator = ACTRDeclarativeActivator()
    paraphrases = [
        activation_candidate_from_episode(
            _episode(
                episode_id=f"ep-scenario-{index}",
                memory_key=f"scenario-{index}",
                statement=f"Team reviewed operational scenario {index} for charge handling.",
            )
        )
        for index in range(1, 9)
    ]
    gold = activation_candidate_from_episode(
        _episode(
            episode_id="ep-gold",
            memory_key="gold",
            statement="PostgreSQL advisory locks coordinate charge batch jobs.",
        )
    )
    config = ActivationConfig(
        retrieval_threshold=-10.0,
        duplicate_jaccard_threshold=0.75,
        collapse_normalize_numeric_tokens=True,
    )
    results = activator.rank(
        candidates=[*paraphrases, gold],
        cue=RetrievalCue(text="PostgreSQL charge coordination"),
        references={},
        as_of=_T1,
        config=config,
        limit=5,
    )
    scenario_count = sum(
        1
        for result in results
        if result.memory.memory_key.startswith("scenario-")  # type: ignore[union-attr]
    )
    assert scenario_count <= 1
    assert any(result.memory.memory_key == "gold" for result in results)  # type: ignore[union-attr]


def test_dynamo_and_postgres_statements_do_not_collapse() -> None:
    activator = ACTRDeclarativeActivator()
    dynamo = activation_candidate_from_episode(
        _episode(
            episode_id="ep-dynamo",
            memory_key="dynamo",
            statement="Charges persisted in DynamoDB before migration.",
        )
    )
    postgres = activation_candidate_from_episode(
        _episode(
            episode_id="ep-postgres",
            memory_key="postgres",
            statement="Charges persist in PostgreSQL after migration.",
        )
    )
    config = ActivationConfig(retrieval_threshold=-10.0)
    results = activator.rank(
        candidates=[dynamo, postgres],
        cue=RetrievalCue(text="charges persist"),
        references={},
        as_of=_T1,
        config=config,
        limit=5,
    )
    assert len(results) == 2


@pytest.mark.asyncio
async def test_recall_does_not_append_activation_references() -> None:
    episode_store = InMemoryEpisodeStore()
    activation_store = InMemoryActivationStore()
    episode = _episode(
        episode_id="ep-1",
        memory_key="durability",
        statement="Operational complexity from charge handling.",
    )
    episode_store._episodes[(episode.tenant_id, episode.memory_key)] = episode
    memory = Memory(
        episode_store=episode_store,
        activation_store=activation_store,
        activation_config=ActivationConfig(retrieval_threshold=-10.0),
    )
    identity = MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="durability")
    refs_before = await activation_store.list_reference_traces(
        tenant_id="company_123",
        identities=[identity],
        before_or_at=_T1,
    )
    await memory.recall(
        "operational complexity",
        tenant_id="company_123",
        subject_id="team",
    )
    refs_after = await activation_store.list_reference_traces(
        tenant_id="company_123",
        identities=[identity],
        before_or_at=_T1,
    )
    assert refs_before == refs_after


@pytest.mark.asyncio
async def test_record_access_min_score_skips_weak_rows() -> None:
    episode_store = InMemoryEpisodeStore()
    activation_store = InMemoryActivationStore()
    distractor = _episode(
        episode_id="ep-distractor",
        memory_key="distractor",
        statement="Generic filler about charges.",
    )
    gold = _episode(
        episode_id="ep-gold",
        memory_key="gold",
        statement="PostgreSQL stores finalized charges for finance.",
        entity_ids=("postgresql", "finance"),
    )
    episode_store._episodes[(distractor.tenant_id, distractor.memory_key)] = distractor
    episode_store._episodes[(gold.tenant_id, gold.memory_key)] = gold
    memory = Memory(
        episode_store=episode_store,
        activation_store=activation_store,
        activation_config=ActivationConfig(retrieval_threshold=-10.0),
    )
    results = await memory.recall(
        "PostgreSQL finalized charges",
        tenant_id="company_123",
        subject_id="team",
        limit=2,
    )
    assert len(results) >= 2
    await memory.record_access(
        results,
        tenant_id="company_123",
        referenced_at=_T1,
        min_score=0.997,
    )
    traces = await activation_store.list_reference_traces(
        tenant_id="company_123",
        identities=[
            MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="distractor"),
            MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="gold"),
        ],
        before_or_at=_T1 + timedelta(seconds=1),
    )
    assert (
        len(traces.get(MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="distractor"), ()))
        == 0
    )
    assert (
        len(traces.get(MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="gold"), ())) >= 1
    )


@pytest.mark.asyncio
async def test_record_access_burst_limit() -> None:
    episode_store = InMemoryEpisodeStore()
    activation_store = InMemoryActivationStore()
    episode = _episode(
        episode_id="ep-1",
        memory_key="target",
        statement="Target memory for burst limiting.",
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
    results = await memory.recall(
        "target memory",
        tenant_id="company_123",
        subject_id="team",
        limit=1,
    )
    identity = MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="target")
    for offset in (0, 600, 1200, 1800):
        await memory.record_access(
            results,
            tenant_id="company_123",
            referenced_at=_T1 + timedelta(seconds=offset),
        )
    traces = await activation_store.list_reference_traces(
        tenant_id="company_123",
        identities=[identity],
        before_or_at=_T1 + timedelta(hours=2),
    )
    assert len(traces[identity]) == 2
