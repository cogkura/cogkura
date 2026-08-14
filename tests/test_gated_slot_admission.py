"""Tests for 0.13 gated slot admission, association, metamemory, and WM precision."""

from __future__ import annotations

from datetime import UTC, datetime
from types import MappingProxyType

from cogkura.algorithms.activation import (
    ACTRDeclarativeActivator,
    activation_candidate_from_episode,
    activation_candidate_from_semantic,
    build_episode_slot_index,
    build_episode_support_index,
)
from cogkura.algorithms.metamemory import DeterministicMemoryMonitor
from cogkura.algorithms.working_memory import DeterministicWorkingMemorySelector
from cogkura.models import (
    ActivationComponents,
    ActivationConfig,
    EpisodeEntity,
    EpisodeEvidenceInput,
    MemoryAssessmentFlag,
    MemoryKind,
    MetamemoryConfig,
    RecallResult,
    RetrievalCue,
    SemanticCardinality,
    SemanticDerivationInput,
    SemanticDerivationRelation,
    SemanticMemoryStatus,
    SemanticPolarity,
    StoredEpisode,
    StoredSemanticMemory,
    WorkingMemoryConfig,
)

_T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
_T1 = datetime(2026, 1, 4, 12, 0, tzinfo=UTC)


def _episode(
    *,
    episode_id: str,
    memory_key: str,
    statement: str,
    entity_ids: tuple[str, ...] = (),
    metadata: MappingProxyType[str, object] | None = None,
) -> StoredEpisode:
    return StoredEpisode(
        id=episode_id,
        tenant_id="company_123",
        subject_id="team",
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
        metadata=metadata or MappingProxyType({}),
        created_at=_T0,
        updated_at=_T0,
    )


def _semantic(
    *,
    semantic_id: str,
    memory_key: str,
    statement: str,
    status: SemanticMemoryStatus,
    entity_ids: tuple[str, ...] = (),
    object_value: str = "value",
    predicate: str = "backing-store",
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
        predicate=predicate,
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


def _recall(
    memory: StoredEpisode | StoredSemanticMemory,
    *,
    activation: float = 1.0,
    score: float = 0.5,
) -> RecallResult:
    memory_kind = MemoryKind.EPISODE if isinstance(memory, StoredEpisode) else MemoryKind.SEMANTIC
    return RecallResult(
        memory_kind=memory_kind,
        memory=memory,
        activation=activation,
        score=score,
        latency_seconds=0.1,
        components=ActivationComponents(
            base_level=0.0,
            spreading=0.0,
            partial_match=0.0,
            noise=0.0,
            total=activation,
        ),
        reason="test",
    )


def test_entity_ids_alone_do_not_trigger_slot_admission() -> None:
    activator = ACTRDeclarativeActivator()
    ledger_support = _episode(
        episode_id="ep-ledger",
        memory_key="ledger-support",
        statement="Charge ledger uses PostgreSQL for finalized charges.",
        entity_ids=("charge-ledger",),
    )
    finance_episode = _episode(
        episode_id="ep-finance",
        memory_key="finance-assoc",
        statement="Finance and charge-ledger teams coordinated the migration.",
        entity_ids=("finance", "charge-ledger"),
    )
    active_semantic = _semantic(
        semantic_id="sem-active",
        memory_key="backing-postgres",
        statement="Finalized charges persist in PostgreSQL.",
        status=SemanticMemoryStatus.ACTIVE,
        entity_ids=("charge-ledger", "postgresql"),
        object_value="postgresql",
        derivations=(
            SemanticDerivationInput(
                episode_id="ep-ledger",
                relation=SemanticDerivationRelation.SUPPORTS,
                contribution_score=0.9,
            ),
        ),
    )
    candidates = [
        activation_candidate_from_episode(finance_episode),
        activation_candidate_from_episode(ledger_support),
        activation_candidate_from_semantic(active_semantic),
    ]
    support_index = build_episode_support_index([active_semantic])
    slot_index = build_episode_slot_index([active_semantic])
    config = ActivationConfig(retrieval_threshold=-10.0, enable_text_entity_seeding=False)
    results = activator.rank(
        candidates=candidates,
        cue=RetrievalCue(
            text="finance charge-ledger association", entity_ids=("finance", "charge-ledger")
        ),
        references={},
        as_of=_T1,
        config=config,
        limit=5,
        episode_support_index=support_index,
        episode_slot_index=slot_index,
    )
    keys = [result.memory.memory_key for result in results]  # type: ignore[union-attr]
    assert keys[0] == "finance-assoc"


def test_current_state_with_entity_id_still_admits_active_slot() -> None:
    activator = ACTRDeclarativeActivator()
    ledger_support = _episode(
        episode_id="ep-ledger",
        memory_key="ledger-support",
        statement="PostgreSQL stores finalized charges.",
        entity_ids=("charge-ledger", "postgresql"),
    )
    active_semantic = _semantic(
        semantic_id="sem-active",
        memory_key="backing-postgres",
        statement="Finalized charges persist in PostgreSQL now.",
        status=SemanticMemoryStatus.ACTIVE,
        entity_ids=("charge-ledger", "postgresql"),
        object_value="postgresql",
        derivations=(
            SemanticDerivationInput(
                episode_id="ep-ledger",
                relation=SemanticDerivationRelation.SUPPORTS,
                contribution_score=0.9,
            ),
        ),
    )
    candidates = [
        activation_candidate_from_episode(ledger_support),
        activation_candidate_from_semantic(active_semantic),
    ]
    semantics = [active_semantic]
    config = ActivationConfig(retrieval_threshold=-10.0, enable_text_entity_seeding=False)
    results = activator.rank(
        candidates=candidates,
        cue=RetrievalCue(
            text="Where do finalized charges persist currently now?",
            entity_ids=("charge-ledger",),
        ),
        references={},
        as_of=_T1,
        config=config,
        limit=5,
        episode_support_index=build_episode_support_index(semantics),
        episode_slot_index=build_episode_slot_index(semantics),
    )
    keys = {result.memory.memory_key for result in results}  # type: ignore[union-attr]
    assert "backing-postgres" in keys
    assert "ledger-support" in keys


def test_multi_entity_conjunction_outranks_single_entity_sibling() -> None:
    activator = ACTRDeclarativeActivator()
    two_entity = _episode(
        episode_id="ep-two",
        memory_key="two-entity",
        statement="Finance and charge-ledger coordinated operational changes.",
        entity_ids=("finance", "charge-ledger"),
    )
    one_entity = _episode(
        episode_id="ep-one",
        memory_key="one-entity",
        statement="Finance reported operational pain from manual reconciliation.",
        entity_ids=("finance",),
    )
    candidates = [
        activation_candidate_from_episode(one_entity),
        activation_candidate_from_episode(two_entity),
    ]
    config = ActivationConfig(retrieval_threshold=-10.0, enable_text_entity_seeding=False)
    results = activator.rank(
        candidates=candidates,
        cue=RetrievalCue(
            text="finance charge-ledger coordination", entity_ids=("finance", "charge-ledger")
        ),
        references={},
        as_of=_T1,
        config=config,
        limit=2,
    )
    assert results[0].memory.memory_key == "two-entity"  # type: ignore[union-attr]
    assert "conjunction=" in results[0].reason


def test_incident_cue_ranks_tagged_episode() -> None:
    activator = ACTRDeclarativeActivator()
    incident = _episode(
        episode_id="ep-incident",
        memory_key="incident",
        statement="On-call engineer woke at 3am for billing platform incident.",
        entity_ids=("billing-platform",),
        metadata=MappingProxyType({"tags": ("on-call", "incident")}),
    )
    distractors = [
        _episode(
            episode_id=f"ep-dist-{index}",
            memory_key=f"billing-{index}",
            statement=f"Billing platform routine update {index} during month-end close.",
            entity_ids=("billing-platform",),
        )
        for index in range(6)
    ]
    candidates = [activation_candidate_from_episode(incident)] + [
        activation_candidate_from_episode(episode) for episode in distractors
    ]
    config = ActivationConfig(retrieval_threshold=-10.0)
    results = activator.rank(
        candidates=candidates,
        cue=RetrievalCue(text="What on-call incident woke the team overnight?"),
        references={},
        as_of=_T1,
        config=config,
        limit=5,
    )
    keys = [result.memory.memory_key for result in results]  # type: ignore[union-attr]
    assert "incident" in keys[:3]


def test_superseded_only_support_excluded_on_current_state() -> None:
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
    semantics = [active_semantic, superseded_semantic]
    config = ActivationConfig(retrieval_threshold=-10.0, enable_text_entity_seeding=False)
    results = activator.rank(
        candidates=candidates,
        cue=RetrievalCue(text="Where do charges persist currently now?"),
        references={},
        as_of=_T1,
        config=config,
        limit=5,
        valid_at=None,
        episode_support_index=build_episode_support_index(semantics),
        episode_slot_index=build_episode_slot_index(semantics),
    )
    keys = {result.memory.memory_key for result in results}  # type: ignore[union-attr]
    assert "postgres-support" in keys
    assert "dynamo-support" not in keys


def test_missing_knowledge_flag_when_pool_full_but_no_active_slot() -> None:
    monitor = DeterministicMemoryMonitor()
    fillers = [
        _recall(
            _episode(
                episode_id=f"ep-{index}",
                memory_key=f"fill-{index}",
                statement=f"Unrelated AWS region deployment note {index}.",
            ),
            score=0.4,
        )
        for index in range(5)
    ]
    assessment = monitor.assess(
        candidates=fillers,
        query=RetrievalCue(text="What AWS region hosts the live charge ledger currently now?"),
        goal=RetrievalCue(text="live charge ledger"),
        tenant_id="company_123",
        subject_id="team",
        as_of=_T1,
        valid_at=None,
        config=MetamemoryConfig(
            missing_knowledge_coverage_threshold=0.5,
            missing_knowledge_strength_threshold=0.6,
        ),
        activation_config=ActivationConfig(),
    )
    assert assessment.retrieved_count == 5
    assert MemoryAssessmentFlag.MISSING_KNOWLEDGE in assessment.flags
    assert MemoryAssessmentFlag.NO_RETRIEVED_MEMORY not in assessment.flags


def test_active_current_state_cue_has_no_missing_knowledge_flag() -> None:
    monitor = DeterministicMemoryMonitor()
    active = _semantic(
        semantic_id="sem-active",
        memory_key="backing-postgres",
        statement="Finalized charges persist in PostgreSQL now.",
        status=SemanticMemoryStatus.ACTIVE,
        object_value="postgresql",
    )
    assessment = monitor.assess(
        candidates=[_recall(active, score=0.9)],
        query=RetrievalCue(text="Where do finalized charges persist currently now?"),
        goal=RetrievalCue(text="current backing store"),
        tenant_id="company_123",
        subject_id="team",
        as_of=_T1,
        valid_at=None,
        config=MetamemoryConfig(),
        activation_config=ActivationConfig(),
    )
    assert MemoryAssessmentFlag.MISSING_KNOWLEDGE not in assessment.flags


def test_working_memory_collapses_same_slot_support() -> None:
    selector = DeterministicWorkingMemorySelector()
    active_semantic = _semantic(
        semantic_id="sem-active",
        memory_key="backing-postgres",
        statement="PostgreSQL is the current backing store.",
        status=SemanticMemoryStatus.ACTIVE,
        object_value="postgresql",
        derivations=(
            SemanticDerivationInput(
                episode_id="ep-postgres",
                relation=SemanticDerivationRelation.SUPPORTS,
                contribution_score=0.9,
            ),
            SemanticDerivationInput(
                episode_id="ep-dynamo",
                relation=SemanticDerivationRelation.SUPPORTS,
                contribution_score=0.2,
            ),
        ),
    )
    postgres_support = _episode(
        episode_id="ep-postgres",
        memory_key="postgres-support",
        statement="PostgreSQL stores finalized charges.",
        entity_ids=("postgresql",),
    )
    dynamo_support = _episode(
        episode_id="ep-dynamo",
        memory_key="dynamo-support",
        statement="DynamoDB previously stored charges.",
        entity_ids=("dynamodb",),
    )
    candidates = [
        _recall(postgres_support, activation=2.0, score=0.9),
        _recall(dynamo_support, activation=1.9, score=0.85),
        _recall(active_semantic, activation=2.5, score=0.95),
    ]
    snapshot = selector.select(
        candidates=candidates,
        goal=RetrievalCue(text="current backing store snapshot"),
        tenant_id="company_123",
        subject_id="team",
        previous=None,
        as_of=_T1,
        config=WorkingMemoryConfig(max_items=3),
        token_estimator=__import__(
            "cogkura.algorithms.working_memory", fromlist=["ApproximateTokenEstimator"]
        ).ApproximateTokenEstimator(),
        activation_config=ActivationConfig(),
    )
    keys = {item.recall.memory.memory_key for item in snapshot.items}  # type: ignore[union-attr]
    assert "backing-postgres" in keys
    support_keys = keys.intersection({"postgres-support", "dynamo-support"})
    assert len(support_keys) <= 1


def test_stale_goal_penalises_superseded_and_tagged_rows() -> None:
    selector = DeterministicWorkingMemorySelector()
    stale_wiki = _episode(
        episode_id="ep-stale",
        memory_key="stale-wiki",
        statement="Legacy wiki page about charge storage.",
        metadata=MappingProxyType({"tags": ("stale",)}),
    )
    current = _episode(
        episode_id="ep-current",
        memory_key="current-store",
        statement="PostgreSQL stores finalized charges.",
        entity_ids=("postgresql",),
    )
    candidates = [
        _recall(stale_wiki, activation=2.0, score=0.9),
        _recall(current, activation=1.5, score=0.7),
    ]
    snapshot = selector.select(
        candidates=candidates,
        goal=RetrievalCue(text="ignore stale wiki; show live ledger store"),
        tenant_id="company_123",
        subject_id="team",
        previous=None,
        as_of=_T1,
        config=WorkingMemoryConfig(max_items=2, stale_goal_penalty=0.5),
        token_estimator=__import__(
            "cogkura.algorithms.working_memory", fromlist=["ApproximateTokenEstimator"]
        ).ApproximateTokenEstimator(),
    )
    keys = [item.recall.memory.memory_key for item in snapshot.items]  # type: ignore[union-attr]
    assert keys[0] == "current-store"
