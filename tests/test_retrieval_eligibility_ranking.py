"""Synthetic fixtures for 0.14 retrieval eligibility, ranking, and temporal relevance."""

from __future__ import annotations

from datetime import UTC, datetime
from types import MappingProxyType

from cogkura.algorithms.activation import (
    ACTRDeclarativeActivator,
    _text_similarity,
    activation_candidate_from_episode,
    activation_candidate_from_semantic,
    build_episode_slot_index,
    build_episode_support_index,
)
from cogkura.algorithms.relevance import calculate_cue_relevance
from cogkura.algorithms.working_memory import (
    ApproximateTokenEstimator,
    DeterministicWorkingMemorySelector,
)
from cogkura.models import (
    ActivationComponents,
    ActivationConfig,
    EpisodeEntity,
    EpisodeEvidenceInput,
    MemoryKind,
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
_T2 = datetime(2026, 1, 8, 12, 0, tzinfo=UTC)


def _episode(
    *,
    episode_id: str,
    memory_key: str,
    statement: str,
    entity_ids: tuple[str, ...] = (),
    started_at: datetime | None = None,
    metadata: MappingProxyType[str, object] | None = None,
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
        metadata=metadata or MappingProxyType({}),
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


def test_admitted_below_threshold_remains_eligible() -> None:
    activator = ACTRDeclarativeActivator()
    postgres_episode = _episode(
        episode_id="ep-postgres",
        memory_key="postgres-support",
        statement="Engineering migrated charge persistence to PostgreSQL.",
        entity_ids=("postgresql",),
    )
    active_semantic = _semantic(
        semantic_id="sem-postgres",
        memory_key="backing-postgres",
        statement="Finalized charges persist in PostgreSQL now.",
        status=SemanticMemoryStatus.ACTIVE,
        entity_ids=("postgresql",),
        object_value="postgresql",
        derivations=(
            SemanticDerivationInput(
                episode_id="ep-postgres",
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
        activation_candidate_from_episode(postgres_episode),
        activation_candidate_from_semantic(active_semantic),
    ]
    semantics = [active_semantic]
    config = ActivationConfig(retrieval_threshold=0.0, enable_text_entity_seeding=False)
    results = activator.rank(
        candidates=candidates,
        cue=RetrievalCue(text="Where do finalized charges persist now?"),
        references={},
        as_of=_T1,
        config=config,
        limit=5,
        episode_support_index=build_episode_support_index(semantics),
        episode_slot_index=build_episode_slot_index(semantics),
    )
    keys = {result.memory.memory_key for result in results}  # type: ignore[union-attr]
    assert "backing-postgres" in keys


def test_admission_does_not_override_stronger_ranking() -> None:
    activator = ACTRDeclarativeActivator()
    high_episode = _episode(
        episode_id="ep-high",
        memory_key="high-activation",
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
        derivations=(),
    )
    candidates = [
        activation_candidate_from_episode(high_episode),
        activation_candidate_from_semantic(active_semantic),
    ]
    semantics = [active_semantic]
    config = ActivationConfig(
        retrieval_threshold=-2.0,
        enable_text_entity_seeding=False,
        enable_duplicate_collapse=False,
    )
    results = activator.rank(
        candidates=candidates,
        cue=RetrievalCue(
            text="finance charge-ledger association",
            entity_ids=("charge-ledger",),
        ),
        references={},
        as_of=_T1,
        config=config,
        limit=5,
        episode_support_index=build_episode_support_index(semantics),
        episode_slot_index=build_episode_slot_index(semantics),
    )
    keys = [result.memory.memory_key for result in results]  # type: ignore[union-attr]
    assert keys[0] == "high-activation"
    assert "backing-postgres" in keys


def test_explicit_entity_id_admits_active_semantic_below_threshold() -> None:
    activator = ACTRDeclarativeActivator()
    support_episode = _episode(
        episode_id="ep-helios",
        memory_key="helios-support",
        statement="Helios uses Dynamo for primary storage.",
        entity_ids=("helios", "dynamodb"),
    )
    active_semantic = _semantic(
        semantic_id="sem-helios",
        memory_key="helios-backing",
        statement="Helios primary storage is Dynamo.",
        status=SemanticMemoryStatus.ACTIVE,
        entity_ids=("helios", "dynamodb"),
        object_value="dynamodb",
        derivations=(
            SemanticDerivationInput(
                episode_id="ep-helios",
                relation=SemanticDerivationRelation.SUPPORTS,
                contribution_score=0.9,
            ),
        ),
    )
    unrelated_semantic = _semantic(
        semantic_id="sem-other",
        memory_key="other-backing",
        statement="Another system uses Redis.",
        status=SemanticMemoryStatus.ACTIVE,
        entity_ids=("redis",),
        object_value="redis",
    )
    candidates = [
        activation_candidate_from_episode(support_episode),
        activation_candidate_from_semantic(active_semantic),
        activation_candidate_from_semantic(unrelated_semantic),
    ]
    semantics = [active_semantic, unrelated_semantic]
    config = ActivationConfig(retrieval_threshold=5.0, enable_text_entity_seeding=False)
    results = activator.rank(
        candidates=candidates,
        cue=RetrievalCue(entity_ids=("helios",)),
        references={},
        as_of=_T1,
        config=config,
        limit=5,
        episode_support_index=build_episode_support_index(semantics),
        episode_slot_index=build_episode_slot_index(semantics),
    )
    keys = {result.memory.memory_key for result in results}  # type: ignore[union-attr]
    assert "helios-backing" in keys
    assert "helios-support" in keys
    assert "other-backing" not in keys


def test_historical_valid_at_avoids_present_lifecycle_penalty() -> None:
    activator = ACTRDeclarativeActivator()
    dynamo_episode = _episode(
        episode_id="ep-dynamo",
        memory_key="dynamo-support",
        statement="The system used Dynamo for storage.",
        entity_ids=("dynamodb",),
        started_at=_T0,
    )
    postgres_episode = _episode(
        episode_id="ep-postgres",
        memory_key="postgres-support",
        statement="The system switched to Postgres for storage.",
        entity_ids=("postgresql",),
        started_at=_T2,
    )
    dynamo_semantic = _semantic(
        semantic_id="sem-dynamo",
        memory_key="backing-dynamo",
        statement="Primary database was Dynamo.",
        status=SemanticMemoryStatus.SUPERSEDED,
        entity_ids=("dynamodb",),
        object_value="dynamodb",
        derivations=(
            SemanticDerivationInput(
                episode_id="ep-dynamo",
                relation=SemanticDerivationRelation.SUPPORTS,
                contribution_score=0.9,
            ),
        ),
    )
    postgres_semantic = _semantic(
        semantic_id="sem-postgres",
        memory_key="backing-postgres",
        statement="Primary database is Postgres.",
        status=SemanticMemoryStatus.ACTIVE,
        entity_ids=("postgresql",),
        object_value="postgresql",
        derivations=(
            SemanticDerivationInput(
                episode_id="ep-postgres",
                relation=SemanticDerivationRelation.SUPPORTS,
                contribution_score=0.9,
            ),
        ),
    )
    candidates = [
        activation_candidate_from_episode(dynamo_episode),
        activation_candidate_from_episode(postgres_episode),
        activation_candidate_from_semantic(dynamo_semantic),
        activation_candidate_from_semantic(postgres_semantic),
    ]
    semantics = [dynamo_semantic, postgres_semantic]
    config = ActivationConfig(
        retrieval_threshold=-10.0,
        enable_text_entity_seeding=False,
        enable_duplicate_collapse=False,
    )
    results = activator.rank(
        candidates=candidates,
        cue=RetrievalCue(text="primary database"),
        references={},
        as_of=_T2,
        config=config,
        limit=5,
        valid_at=_T1,
        episode_support_index=build_episode_support_index(semantics),
        episode_slot_index=build_episode_slot_index(semantics),
    )
    semantic_results = [r for r in results if r.memory_kind is MemoryKind.SEMANTIC]
    assert semantic_results
    assert semantic_results[0].memory.memory_key == "backing-dynamo"  # type: ignore[union-attr]
    assert semantic_results[0].components.current_state == 0.0


def test_generic_entity_query_no_lifecycle_bonus() -> None:
    activator = ACTRDeclarativeActivator()
    active_semantic = _semantic(
        semantic_id="sem-active",
        memory_key="backing-active",
        statement="Helios uses Postgres.",
        status=SemanticMemoryStatus.ACTIVE,
        entity_ids=("helios", "postgresql"),
        object_value="postgresql",
    )
    superseded_semantic = _semantic(
        semantic_id="sem-old",
        memory_key="backing-old",
        statement="Helios used Dynamo.",
        status=SemanticMemoryStatus.SUPERSEDED,
        entity_ids=("helios", "dynamodb"),
        object_value="dynamodb",
    )
    candidates = [
        activation_candidate_from_semantic(active_semantic),
        activation_candidate_from_semantic(superseded_semantic),
    ]
    config = ActivationConfig(
        retrieval_threshold=-10.0,
        enable_text_entity_seeding=False,
        enable_duplicate_collapse=False,
    )
    results = activator.rank(
        candidates=candidates,
        cue=RetrievalCue(entity_ids=("helios",)),
        references={},
        as_of=_T1,
        config=config,
        limit=5,
        episode_support_index=build_episode_support_index([active_semantic, superseded_semantic]),
        episode_slot_index=build_episode_slot_index([active_semantic, superseded_semantic]),
    )
    for result in results:
        assert result.components.current_state == 0.0


def test_precision_matching_prefers_concise_candidate() -> None:
    config = ActivationConfig(enable_text_precision_matching=True)
    query = "charge ledger overnight incident"
    concise = "charge ledger overnight incident triage"
    verbose = (
        "charge ledger overnight incident service project deployment database "
        "architecture migration customer configuration"
    )
    concise_score = _text_similarity(query, concise, config=config)
    verbose_score = _text_similarity(query, verbose, config=config)
    assert concise_score > verbose_score


def test_coverage_matching_when_precision_disabled() -> None:
    config = ActivationConfig(enable_text_precision_matching=False)
    query = "charge ledger overnight incident"
    concise = "charge ledger overnight incident triage"
    verbose = (
        "charge ledger overnight incident service project deployment database "
        "architecture migration customer configuration"
    )
    concise_score = _text_similarity(query, concise, config=config)
    verbose_score = _text_similarity(query, verbose, config=config)
    assert concise_score == verbose_score == 1.0


def test_working_memory_goal_specific_beats_generic() -> None:
    selector = DeterministicWorkingMemorySelector()
    estimator = ApproximateTokenEstimator()
    goal_specific = _episode(
        episode_id="ep-specific",
        memory_key="incident-specific",
        statement="charge ledger overnight incident triage completed",
    )
    generic = _episode(
        episode_id="ep-generic",
        memory_key="generic-project",
        statement=(
            "charge ledger overnight incident service project deployment database "
            "architecture migration customer configuration"
        ),
    )
    pool = [
        _recall(generic, activation=2.0, score=0.9),
        _recall(goal_specific, activation=1.0, score=0.7),
    ]
    snapshot = selector.select(
        candidates=pool,
        goal=RetrievalCue(text="charge ledger overnight incident"),
        tenant_id="company_123",
        subject_id="team",
        previous=None,
        as_of=_T1,
        config=WorkingMemoryConfig(max_items=1),
        token_estimator=estimator,
    )
    assert snapshot.items[0].memory.memory_key == "incident-specific"


def test_entity_admission_without_current_state_exclusion() -> None:
    activator = ACTRDeclarativeActivator()
    dynamo_episode = _episode(
        episode_id="ep-dynamo",
        memory_key="dynamo-support",
        statement="Helios stored data in Dynamo.",
        entity_ids=("helios", "dynamodb"),
    )
    postgres_episode = _episode(
        episode_id="ep-postgres",
        memory_key="postgres-support",
        statement="Helios now stores data in Postgres.",
        entity_ids=("helios", "postgresql"),
    )
    active_semantic = _semantic(
        semantic_id="sem-postgres",
        memory_key="backing-postgres",
        statement="Helios uses Postgres.",
        status=SemanticMemoryStatus.ACTIVE,
        entity_ids=("helios", "postgresql"),
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
        statement="Helios used Dynamo.",
        status=SemanticMemoryStatus.SUPERSEDED,
        entity_ids=("helios", "dynamodb"),
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
        cue=RetrievalCue(entity_ids=("helios",)),
        references={},
        as_of=_T2,
        config=config,
        limit=5,
        valid_at=None,
        episode_support_index=build_episode_support_index(semantics),
        episode_slot_index=build_episode_slot_index(semantics),
    )
    keys = {result.memory.memory_key for result in results}  # type: ignore[union-attr]
    assert "dynamo-support" in keys


def test_goal_relevance_precision_beats_coverage_equivalent() -> None:
    concise = _episode(
        episode_id="ep-concise",
        memory_key="concise",
        statement="operational complexity database",
    )
    verbose = _episode(
        episode_id="ep-verbose",
        memory_key="verbose",
        statement="operational complexity database migration service project deployment",
    )
    concise_relevance = calculate_cue_relevance(
        _recall(concise),
        RetrievalCue(text="operational complexity database"),
    )
    verbose_relevance = calculate_cue_relevance(
        _recall(verbose),
        RetrievalCue(text="operational complexity database"),
    )
    assert concise_relevance > verbose_relevance
