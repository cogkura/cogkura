"""Synthetic fixtures for temporal mode, conjunctive slot matching, and ranking."""

from __future__ import annotations

from datetime import UTC, datetime
from types import MappingProxyType

from cogkura.algorithms.activation import (
    ACTRDeclarativeActivator,
    TemporalRetrievalMode,
    _semantic_slot_fit,
    _semantic_slot_matches_cue,
    _temporal_retrieval_mode,
    activation_candidate_from_episode,
    activation_candidate_from_semantic,
    build_episode_slot_index,
    build_episode_support_index,
)
from cogkura.algorithms.metamemory import (
    DeterministicMemoryMonitor,
    MemoryAnswerability,
    _assess_answerability,
)
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
    RetrievalEligibility,
    SemanticCardinality,
    SemanticDerivationInput,
    SemanticDerivationRelation,
    SemanticMemoryStatus,
    SemanticPolarity,
    SlotFitSource,
    StoredEpisode,
    StoredSemanticMemory,
)

_T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
_T1 = datetime(2026, 1, 4, 12, 0, tzinfo=UTC)
_T2 = datetime(2026, 1, 8, 12, 0, tzinfo=UTC)
_CONFIG = ActivationConfig()
_MONITOR = DeterministicMemoryMonitor()


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
    subject_entity_id: str | None = None,
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
        subject_entity_id=subject_entity_id,
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
    score: float = 0.9,
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


def test_valid_at_overrides_lexical_current_to_historical() -> None:
    mode = _temporal_retrieval_mode(
        RetrievalCue(text="what is the current live backing store now?"),
        valid_at=_T1,
        config=_CONFIG,
    )
    assert mode is TemporalRetrievalMode.HISTORICAL


def test_live_now_query_is_current_mode() -> None:
    mode = _temporal_retrieval_mode(
        RetrievalCue(text="what is live now?"),
        valid_at=None,
        config=_CONFIG,
    )
    assert mode is TemporalRetrievalMode.CURRENT


def test_entity_only_query_is_neutral() -> None:
    mode = _temporal_retrieval_mode(
        RetrievalCue(entity_ids=("ledger",)),
        valid_at=None,
        config=_CONFIG,
    )
    assert mode is TemporalRetrievalMode.NEUTRAL


def test_historical_text_only_admits_visible_superseded_slot() -> None:
    activator = ACTRDeclarativeActivator()
    historical = _semantic(
        semantic_id="sem-dynamo",
        memory_key="backing-dynamo",
        statement="The charge-ledger backing store was DynamoDB.",
        status=SemanticMemoryStatus.SUPERSEDED,
        entity_ids=("charge-ledger", "dynamodb"),
        subject_entity_id="charge-ledger",
        object_value="dynamodb",
    )
    candidates = [activation_candidate_from_semantic(historical)]
    results = activator.rank(
        candidates=candidates,
        cue=RetrievalCue(text="What backing store held the charge ledger during the first week?"),
        references={},
        as_of=_T2,
        config=ActivationConfig(
            retrieval_threshold=5.0,
            enable_entity_slot_admission=False,
            enable_duplicate_collapse=False,
        ),
        limit=5,
        valid_at=_T1,
        episode_support_index=build_episode_support_index([historical]),
        episode_slot_index=build_episode_slot_index([historical]),
    )
    keys = {result.memory.memory_key for result in results}  # type: ignore[union-attr]
    assert "backing-dynamo" in keys
    assert "temporal_mode=historical" in results[0].reason
    assert "soft_admitted=true" in results[0].reason
    assert results[0].components.current_state == 0.0


def test_historical_mode_does_not_admit_unrelated_semantic() -> None:
    activator = ACTRDeclarativeActivator()
    historical = _semantic(
        semantic_id="sem-dynamo",
        memory_key="backing-dynamo",
        statement="The charge-ledger backing store was DynamoDB.",
        status=SemanticMemoryStatus.SUPERSEDED,
        entity_ids=("charge-ledger", "dynamodb"),
        subject_entity_id="charge-ledger",
        object_value="dynamodb",
    )
    unrelated = _semantic(
        semantic_id="sem-region",
        memory_key="billing-region",
        statement="Finance discussed a billing region during planning.",
        status=SemanticMemoryStatus.SUPERSEDED,
        entity_ids=("billing-region",),
        subject_entity_id="billing-region",
        predicate="primary",
        object_value="eu-west-1",
    )
    candidates = [
        activation_candidate_from_semantic(historical),
        activation_candidate_from_semantic(unrelated),
    ]
    results = activator.rank(
        candidates=candidates,
        cue=RetrievalCue(text="What backing store held the charge ledger during the first week?"),
        references={},
        as_of=_T2,
        config=ActivationConfig(
            retrieval_threshold=5.0,
            enable_entity_slot_admission=False,
            enable_duplicate_collapse=False,
        ),
        limit=5,
        valid_at=_T1,
        episode_support_index=build_episode_support_index([historical, unrelated]),
        episode_slot_index=build_episode_slot_index([historical, unrelated]),
    )
    keys = {result.memory.memory_key for result in results}  # type: ignore[union-attr]
    assert "backing-dynamo" in keys
    assert "billing-region" not in keys


def test_exact_slot_has_strongest_structured_fit() -> None:
    activator = ACTRDeclarativeActivator()
    matching = _semantic(
        semantic_id="sem-db",
        memory_key="service-database-postgres",
        statement="The service database is postgres.",
        status=SemanticMemoryStatus.ACTIVE,
        entity_ids=("service",),
        subject_entity_id="service",
        predicate="database",
        object_value="postgres",
    )
    other_slot = _semantic(
        semantic_id="sem-region",
        memory_key="service-region-eu",
        statement="The service region is eu-west.",
        status=SemanticMemoryStatus.ACTIVE,
        entity_ids=("service",),
        subject_entity_id="service",
        predicate="region",
        object_value="eu-west",
    )
    other_service = _semantic(
        semantic_id="sem-other",
        memory_key="other-database-mysql",
        statement="The other-service database is mysql.",
        status=SemanticMemoryStatus.ACTIVE,
        entity_ids=("other-service",),
        subject_entity_id="other-service",
        predicate="database",
        object_value="mysql",
    )
    semantics = [matching, other_slot, other_service]
    cue = RetrievalCue(
        text="what is the current database?",
        entity_ids=("service",),
        predicate="database",
    )
    results = activator.rank(
        candidates=[activation_candidate_from_semantic(item) for item in semantics],
        cue=cue,
        references={},
        as_of=_T2,
        config=ActivationConfig(
            retrieval_threshold=-10.0,
            enable_text_entity_seeding=False,
            enable_duplicate_collapse=False,
        ),
        limit=5,
        episode_support_index=build_episode_support_index(semantics),
        episode_slot_index=build_episode_slot_index(semantics),
    )
    keys = [result.memory.memory_key for result in results]  # type: ignore[union-attr]
    assert keys[0] == "service-database-postgres"
    matched = results[0]
    other_service_result = next(
        item for item in results if item.memory.memory_key == "other-database-mysql"
    )
    assert "slot_fit=1.00" in matched.reason
    assert "structured_adjustment=+0.50" in matched.reason
    assert "slot_fit=0.00" in other_service_result.reason
    assert "structured_adjustment=-0.50" in other_service_result.reason


def test_support_episode_inherits_only_supported_slot_fit() -> None:
    activator = ACTRDeclarativeActivator()
    matching = _semantic(
        semantic_id="sem-db",
        memory_key="service-database-postgres",
        statement="The service database is postgres.",
        status=SemanticMemoryStatus.ACTIVE,
        entity_ids=("service",),
        subject_entity_id="service",
        predicate="database",
        object_value="postgres",
        derivations=(
            SemanticDerivationInput(
                episode_id="ep-db",
                relation=SemanticDerivationRelation.SUPPORTS,
                contribution_score=0.9,
            ),
        ),
    )
    other_slot = _semantic(
        semantic_id="sem-region",
        memory_key="service-region-eu",
        statement="The service region is eu-west.",
        status=SemanticMemoryStatus.ACTIVE,
        entity_ids=("service",),
        subject_entity_id="service",
        predicate="region",
        object_value="eu-west",
        derivations=(
            SemanticDerivationInput(
                episode_id="ep-region",
                relation=SemanticDerivationRelation.SUPPORTS,
                contribution_score=0.9,
            ),
        ),
    )
    db_episode = _episode(
        episode_id="ep-db",
        memory_key="db-support",
        statement="Engineering confirmed postgres as the service database.",
        entity_ids=("service",),
    )
    region_episode = _episode(
        episode_id="ep-region",
        memory_key="region-support",
        statement="Engineering confirmed eu-west as the service region.",
        entity_ids=("service",),
    )
    semantics = [matching, other_slot]
    cue = RetrievalCue(
        text="what is the current database?",
        entity_ids=("service",),
        predicate="database",
    )
    results = activator.rank(
        candidates=[
            activation_candidate_from_semantic(matching),
            activation_candidate_from_semantic(other_slot),
            activation_candidate_from_episode(db_episode),
            activation_candidate_from_episode(region_episode),
        ],
        cue=cue,
        references={},
        as_of=_T2,
        config=ActivationConfig(
            retrieval_threshold=-10.0,
            enable_text_entity_seeding=False,
            enable_duplicate_collapse=False,
            collapse_same_slot_support=False,
        ),
        limit=5,
        episode_support_index=build_episode_support_index(semantics),
        episode_slot_index=build_episode_slot_index(semantics),
    )
    by_key = {result.memory.memory_key: result for result in results}  # type: ignore[union-attr]
    assert "slot_fit=1.00" in by_key["db-support"].reason
    assert "slot_fit_source=support" in by_key["db-support"].reason
    assert "slot_fit=1.00" not in by_key["region-support"].reason
    assert "structured_adjustment=-0.50" in by_key["region-support"].reason


def test_historical_superseded_revision_has_full_temporal_fit() -> None:
    cue = RetrievalCue(
        text="what database was in use?",
        entity_ids=("service",),
        predicate="database",
    )
    historical = activation_candidate_from_semantic(
        _semantic(
            semantic_id="sem-dynamo",
            memory_key="service-database-dynamo",
            statement="The service database was dynamo.",
            status=SemanticMemoryStatus.SUPERSEDED,
            entity_ids=("service",),
            subject_entity_id="service",
            predicate="database",
            object_value="dynamo",
        )
    )
    fit = _semantic_slot_fit(
        historical,
        cue,
        temporal_mode=TemporalRetrievalMode.HISTORICAL,
        effective_entities=("service",),
    )
    assert fit == 1.0


def test_neutral_associative_query_skips_slot_fit() -> None:
    activator = ACTRDeclarativeActivator()
    associated = _episode(
        episode_id="ep-assoc",
        memory_key="alpha-beta-link",
        statement="Alpha and Beta coordinated the operational incident.",
        entity_ids=("alpha", "beta"),
    )
    semantic = _semantic(
        semantic_id="sem-alpha",
        memory_key="alpha-status",
        statement="Alpha is active in production.",
        status=SemanticMemoryStatus.ACTIVE,
        entity_ids=("alpha",),
        subject_entity_id="alpha",
        predicate="status",
        object_value="active",
    )
    cue = RetrievalCue(text="alpha beta operational incident", entity_ids=("alpha", "beta"))
    results = activator.rank(
        candidates=[
            activation_candidate_from_episode(associated),
            activation_candidate_from_semantic(semantic),
        ],
        cue=cue,
        references={},
        as_of=_T1,
        config=ActivationConfig(
            retrieval_threshold=-10.0,
            enable_text_entity_seeding=False,
            enable_duplicate_collapse=False,
        ),
        limit=5,
        episode_support_index=build_episode_support_index([semantic]),
        episode_slot_index=build_episode_slot_index([semantic]),
    )
    assert (
        _temporal_retrieval_mode(cue, valid_at=None, config=_CONFIG)
        is TemporalRetrievalMode.NEUTRAL
    )
    for result in results:
        assert "slot_fit=" not in result.reason
        assert "structured_adjustment=" not in result.reason
    keys = [result.memory.memory_key for result in results]  # type: ignore[union-attr]
    assert keys[0] == "alpha-beta-link"


def test_resolved_current_fact_has_no_missing_knowledge() -> None:
    active = _semantic(
        semantic_id="sem-region",
        memory_key="billing-primary",
        statement="The billing-region primary is eu-west-1.",
        status=SemanticMemoryStatus.ACTIVE,
        entity_ids=("billing-region",),
        subject_entity_id="billing-region",
        predicate="primary",
        object_value="eu-west-1",
    )
    query = RetrievalCue(
        text="what is the current primary region?",
        entity_ids=("billing-region",),
        predicate="primary",
    )
    candidates = [_recall(active, score=0.9)]
    assert (
        _assess_answerability(
            candidates,
            query,
            valid_at=None,
            activation_config=_CONFIG,
        )
        is MemoryAnswerability.RESOLVED
    )
    assessment = _MONITOR.assess(
        candidates=candidates,
        query=query,
        goal=RetrievalCue(text="current primary region"),
        tenant_id="company_123",
        subject_id="team",
        as_of=_T2,
        valid_at=None,
        config=MetamemoryConfig(),
        activation_config=_CONFIG,
    )
    assert MemoryAssessmentFlag.MISSING_KNOWLEDGE not in assessment.flags


def test_related_but_unresolved_slot_is_missing_knowledge() -> None:
    discussion = _episode(
        episode_id="ep-discuss",
        memory_key="region-discussion",
        statement="The team discussed eu-west but made no region decision.",
        entity_ids=("billing-region",),
    )
    query = RetrievalCue(
        text="what is the current primary region?",
        entity_ids=("billing-region",),
        predicate="primary",
    )
    candidates = [_recall(discussion, score=0.92)]
    assert (
        _assess_answerability(
            candidates,
            query,
            valid_at=None,
            activation_config=_CONFIG,
        )
        is MemoryAnswerability.UNRESOLVED
    )
    assessment = _MONITOR.assess(
        candidates=candidates,
        query=query,
        goal=RetrievalCue(text="current primary region"),
        tenant_id="company_123",
        subject_id="team",
        as_of=_T2,
        valid_at=None,
        config=MetamemoryConfig(),
        activation_config=_CONFIG,
    )
    assert MemoryAssessmentFlag.MISSING_KNOWLEDGE in assessment.flags
    assert assessment.signals.top_retrieval_strength >= 0.9


def test_historical_superseded_fact_is_resolved() -> None:
    historical = _semantic(
        semantic_id="sem-region",
        memory_key="billing-primary-t1",
        statement="The billing-region primary was eu-west-1.",
        status=SemanticMemoryStatus.SUPERSEDED,
        entity_ids=("billing-region",),
        subject_entity_id="billing-region",
        predicate="primary",
        object_value="eu-west-1",
    )
    query = RetrievalCue(
        text="what was the primary region?",
        entity_ids=("billing-region",),
        predicate="primary",
    )
    candidates = [_recall(historical, score=0.8)]
    assert (
        _assess_answerability(
            candidates,
            query,
            valid_at=_T1,
            activation_config=_CONFIG,
        )
        is MemoryAnswerability.RESOLVED
    )
    assessment = _MONITOR.assess(
        candidates=candidates,
        query=query,
        goal=RetrievalCue(text="historical primary region"),
        tenant_id="company_123",
        subject_id="team",
        as_of=_T2,
        valid_at=_T1,
        config=MetamemoryConfig(),
        activation_config=_CONFIG,
    )
    assert MemoryAssessmentFlag.MISSING_KNOWLEDGE not in assessment.flags


def test_exploratory_episodic_query_is_not_applicable() -> None:
    episode = _episode(
        episode_id="ep-incident",
        memory_key="deployment-incident",
        statement="The deployment incident caused a brief outage.",
        entity_ids=("deployment",),
    )
    query = RetrievalCue(text="What happened during the deployment incident?")
    candidates = [_recall(episode, score=0.8)]
    assert (
        _assess_answerability(
            candidates,
            query,
            valid_at=None,
            activation_config=_CONFIG,
        )
        is MemoryAnswerability.NOT_APPLICABLE
    )
    assessment = _MONITOR.assess(
        candidates=candidates,
        query=query,
        goal=RetrievalCue(text="deployment incident"),
        tenant_id="company_123",
        subject_id="team",
        as_of=_T2,
        valid_at=None,
        config=MetamemoryConfig(),
        activation_config=_CONFIG,
    )
    assert assessment.signals.top_retrieval_strength == 0.8


def _slot_match(
    memory: StoredSemanticMemory,
    cue: RetrievalCue,
    *,
    seeded: tuple[str, ...] = (),
    current_state_cue: bool = False,
    distinctive_tokens: set[str] | None = None,
) -> bool:
    return _semantic_slot_matches_cue(
        activation_candidate_from_semantic(memory),
        cue,
        effective_sources=seeded,
        current_state_cue=current_state_cue,
        distinctive_tokens=distinctive_tokens or set(),
    )


def test_conjunctive_entity_and_predicate_match() -> None:
    matching = _semantic(
        semantic_id="sem-a-db",
        memory_key="service-a-database-postgres",
        statement="The service-a database is postgres.",
        status=SemanticMemoryStatus.ACTIVE,
        entity_ids=("service-a",),
        subject_entity_id="service-a",
        predicate="database",
        object_value="postgres",
    )
    other_slot = _semantic(
        semantic_id="sem-a-region",
        memory_key="service-a-region",
        statement="The service-a region is eu-west.",
        status=SemanticMemoryStatus.ACTIVE,
        entity_ids=("service-a",),
        subject_entity_id="service-a",
        predicate="region",
        object_value="eu-west",
    )
    other_service = _semantic(
        semantic_id="sem-b-db",
        memory_key="service-b-database",
        statement="The service-b database is mysql.",
        status=SemanticMemoryStatus.ACTIVE,
        entity_ids=("service-b",),
        subject_entity_id="service-b",
        predicate="database",
        object_value="mysql",
    )
    cue = RetrievalCue(entity_ids=("service-a",), predicate="database")
    assert _slot_match(matching, cue) is True
    assert _slot_match(other_slot, cue) is False
    assert _slot_match(other_service, cue) is False


def test_conjunctive_entity_predicate_and_object_match() -> None:
    matching = _semantic(
        semantic_id="sem-pg",
        memory_key="billing-database-postgres",
        statement="The billing database is postgres.",
        status=SemanticMemoryStatus.ACTIVE,
        entity_ids=("billing",),
        subject_entity_id="billing",
        predicate="database",
        object_value="postgres",
    )
    mysql = _semantic(
        semantic_id="sem-mysql",
        memory_key="billing-database-mysql",
        statement="The billing database is mysql.",
        status=SemanticMemoryStatus.ACTIVE,
        entity_ids=("billing",),
        subject_entity_id="billing",
        predicate="database",
        object_value="mysql",
    )
    cue = RetrievalCue(
        entity_ids=("billing",),
        predicate="database",
        object_value="postgres",
    )
    assert _slot_match(matching, cue) is True
    assert _slot_match(mysql, cue) is False


def test_predicate_only_matches_any_candidate_with_that_predicate() -> None:
    first = _semantic(
        semantic_id="sem-a",
        memory_key="service-a-database",
        statement="The service-a database is postgres.",
        status=SemanticMemoryStatus.ACTIVE,
        entity_ids=("service-a",),
        predicate="database",
        object_value="postgres",
    )
    second = _semantic(
        semantic_id="sem-b",
        memory_key="service-b-database",
        statement="The service-b database is mysql.",
        status=SemanticMemoryStatus.ACTIVE,
        entity_ids=("service-b",),
        predicate="database",
        object_value="mysql",
    )
    region = _semantic(
        semantic_id="sem-region",
        memory_key="service-a-region",
        statement="The service-a region is eu-west.",
        status=SemanticMemoryStatus.ACTIVE,
        entity_ids=("service-a",),
        predicate="region",
        object_value="eu-west",
    )
    cue = RetrievalCue(predicate="database")
    assert _slot_match(first, cue) is True
    assert _slot_match(second, cue) is True
    assert _slot_match(region, cue) is False


def test_explicit_mismatch_is_not_rescued_by_lexical_overlap() -> None:
    candidate = _semantic(
        semantic_id="sem-b",
        memory_key="service-b-database",
        statement="service-a asked whether service-b should use postgres.",
        status=SemanticMemoryStatus.ACTIVE,
        entity_ids=("service-b",),
        subject_entity_id="service-b",
        predicate="database",
        object_value="postgres",
    )
    cue = RetrievalCue(entity_ids=("service-a",), predicate="database")
    assert (
        _slot_match(
            candidate,
            cue,
            current_state_cue=True,
            distinctive_tokens={"service", "a", "postgres"},
        )
        is False
    )


def test_seeded_entity_locator_constrains_structured_match() -> None:
    service_a = _semantic(
        semantic_id="sem-a",
        memory_key="service-a-database",
        statement="The service-a database is postgres.",
        status=SemanticMemoryStatus.ACTIVE,
        entity_ids=("service-a",),
        subject_entity_id="service-a",
        predicate="database",
        object_value="postgres",
    )
    service_b = _semantic(
        semantic_id="sem-b",
        memory_key="service-b-database",
        statement="The service-b database is mysql.",
        status=SemanticMemoryStatus.ACTIVE,
        entity_ids=("service-b",),
        subject_entity_id="service-b",
        predicate="database",
        object_value="mysql",
    )
    cue = RetrievalCue(predicate="database")
    assert _slot_match(service_a, cue, seeded=("service-a",)) is True
    assert _slot_match(service_b, cue, seeded=("service-a",)) is False


def test_perfect_slot_fit_outranks_unstructured_episode() -> None:
    activator = ACTRDeclarativeActivator()
    exact = _semantic(
        semantic_id="sem-exact",
        memory_key="billing-database-postgres",
        statement="The billing database is postgres.",
        status=SemanticMemoryStatus.ACTIVE,
        entity_ids=("billing",),
        subject_entity_id="billing",
        predicate="database",
        object_value="postgres",
    )
    generic = _episode(
        episode_id="ep-generic",
        memory_key="generic-note",
        statement="Unrelated operational note about weather.",
        entity_ids=("billing",),
        started_at=_T0,
    )
    cue = RetrievalCue(
        text="what is the current database?",
        entity_ids=("billing",),
        predicate="database",
    )
    results = activator.rank(
        candidates=[
            activation_candidate_from_semantic(exact),
            activation_candidate_from_episode(generic),
        ],
        cue=cue,
        references={},
        as_of=_T2,
        config=ActivationConfig(
            retrieval_threshold=-10.0,
            enable_spreading_activation=False,
            enable_text_entity_seeding=False,
            enable_duplicate_collapse=False,
        ),
        limit=5,
        episode_support_index=build_episode_support_index([exact]),
        episode_slot_index=build_episode_slot_index([exact]),
    )
    keys = [result.memory.memory_key for result in results]  # type: ignore[union-attr]
    by_key = {result.memory.memory_key: result for result in results}  # type: ignore[union-attr]
    assert keys[0] == "billing-database-postgres"
    assert "slot_fit=1.00" in by_key["billing-database-postgres"].reason
    assert "structured_adjustment=+0.50" in by_key["billing-database-postgres"].reason
    assert "slot_fit=" not in by_key["generic-note"].reason
    assert "structured_adjustment=" not in by_key["generic-note"].reason


def test_positive_structured_adjustment_can_beat_neutral_with_lower_activation() -> None:
    activator = ACTRDeclarativeActivator()
    exact = _semantic(
        semantic_id="sem-exact-low",
        memory_key="billing-database-postgres-low",
        statement="The billing database is postgres.",
        status=SemanticMemoryStatus.ACTIVE,
        entity_ids=("billing",),
        subject_entity_id="billing",
        predicate="database",
        object_value="postgres",
    )
    generic = _episode(
        episode_id="ep-neutral-high",
        memory_key="generic-neutral-high",
        statement="Unrelated operational note.",
        entity_ids=("billing",),
        started_at=_T1,
    )
    cue = RetrievalCue(entity_ids=("billing",), predicate="database")
    results = activator.rank(
        candidates=[
            activation_candidate_from_semantic(exact),
            activation_candidate_from_episode(generic),
        ],
        cue=cue,
        references={},
        as_of=_T2,
        config=ActivationConfig(
            retrieval_threshold=-10.0,
            enable_spreading_activation=False,
            enable_partial_matching=False,
            current_state_weight=0.0,
            enable_text_entity_seeding=False,
            enable_duplicate_collapse=False,
        ),
        limit=5,
        episode_support_index=build_episode_support_index([exact]),
        episode_slot_index=build_episode_slot_index([exact]),
    )
    by_key = {result.memory.memory_key: result for result in results}  # type: ignore[union-attr]
    semantic_result = by_key["billing-database-postgres-low"]
    neutral_result = by_key["generic-neutral-high"]
    assert semantic_result.activation < neutral_result.activation
    assert results[0].memory.memory_key == "billing-database-postgres-low"  # type: ignore[union-attr]
    assert "structured_adjustment=+0.50" in semantic_result.reason
    assert "structured_adjustment=" not in neutral_result.reason


def test_structured_mismatch_receives_negative_adjustment() -> None:
    activator = ACTRDeclarativeActivator()
    exact = _semantic(
        semantic_id="sem-exact",
        memory_key="billing-database-postgres",
        statement="The billing database is postgres.",
        status=SemanticMemoryStatus.ACTIVE,
        entity_ids=("billing",),
        subject_entity_id="billing",
        predicate="database",
        object_value="postgres",
    )
    mismatch = _semantic(
        semantic_id="sem-orders",
        memory_key="orders-database-mysql",
        statement="The orders database is mysql.",
        status=SemanticMemoryStatus.ACTIVE,
        entity_ids=("orders",),
        subject_entity_id="orders",
        predicate="database",
        object_value="mysql",
    )
    cue = RetrievalCue(
        text="what is the current database?",
        entity_ids=("billing",),
        predicate="database",
    )
    results = activator.rank(
        candidates=[
            activation_candidate_from_semantic(exact),
            activation_candidate_from_semantic(mismatch),
        ],
        cue=cue,
        references={},
        as_of=_T2,
        config=ActivationConfig(
            retrieval_threshold=-10.0,
            enable_spreading_activation=False,
            enable_text_entity_seeding=False,
            enable_duplicate_collapse=False,
        ),
        limit=5,
        episode_support_index=build_episode_support_index([exact, mismatch]),
        episode_slot_index=build_episode_slot_index([exact, mismatch]),
    )
    by_key = {result.memory.memory_key: result for result in results}  # type: ignore[union-attr]
    assert results[0].memory.memory_key == "billing-database-postgres"  # type: ignore[union-attr]
    assert "structured_adjustment=-0.50" in by_key["orders-database-mysql"].reason


def test_current_active_slot_outranks_superseded_mismatch() -> None:
    activator = ACTRDeclarativeActivator()
    postgres = _semantic(
        semantic_id="sem-pg",
        memory_key="billing-database-postgres",
        statement="The billing database is postgres.",
        status=SemanticMemoryStatus.ACTIVE,
        entity_ids=("billing",),
        subject_entity_id="billing",
        predicate="database",
        object_value="postgres",
    )
    dynamo = _semantic(
        semantic_id="sem-dynamo",
        memory_key="billing-database-dynamo",
        statement="The billing database was dynamo.",
        status=SemanticMemoryStatus.SUPERSEDED,
        entity_ids=("billing",),
        subject_entity_id="billing",
        predicate="database",
        object_value="dynamo",
    )
    cue = RetrievalCue(
        text="what is the current database?",
        entity_ids=("billing",),
        predicate="database",
    )
    results = activator.rank(
        candidates=[
            activation_candidate_from_semantic(postgres),
            activation_candidate_from_semantic(dynamo),
        ],
        cue=cue,
        references={},
        as_of=_T2,
        config=ActivationConfig(
            retrieval_threshold=-10.0,
            enable_spreading_activation=False,
            enable_text_entity_seeding=False,
            enable_duplicate_collapse=False,
        ),
        limit=5,
        episode_support_index=build_episode_support_index([postgres, dynamo]),
        episode_slot_index=build_episode_slot_index([postgres, dynamo]),
    )
    by_key = {result.memory.memory_key: result for result in results}  # type: ignore[union-attr]
    assert results[0].memory.memory_key == "billing-database-postgres"  # type: ignore[union-attr]
    assert "slot_fit=1.00" in by_key["billing-database-postgres"].reason
    dynamo_result = by_key.get("billing-database-dynamo")
    if dynamo_result is not None:
        assert "slot_fit=0.00" in dynamo_result.reason


def test_current_support_diagnostics_require_matching_active_slot() -> None:
    activator = ACTRDeclarativeActivator()
    postgres_support = _episode(
        episode_id="ep-postgres",
        memory_key="postgres-support",
        statement="Production confirms postgres hosts the live charge ledger.",
        entity_ids=("service", "postgres"),
        started_at=_T2,
    )
    region_support = _episode(
        episode_id="ep-region",
        memory_key="region-support",
        statement="Production confirms service region is eu-west.",
        entity_ids=("service", "eu-west"),
        started_at=_T2,
    )
    postgres = _semantic(
        semantic_id="sem-postgres",
        memory_key="service-database-postgres",
        statement="The service database is postgres.",
        status=SemanticMemoryStatus.ACTIVE,
        entity_ids=("service",),
        subject_entity_id="service",
        predicate="database",
        object_value="postgres",
        derivations=(
            SemanticDerivationInput(
                episode_id=postgres_support.id,
                relation=SemanticDerivationRelation.SUPPORTS,
                contribution_score=0.9,
            ),
        ),
    )
    region = _semantic(
        semantic_id="sem-region",
        memory_key="service-region-eu-west",
        statement="The service region is eu-west.",
        status=SemanticMemoryStatus.ACTIVE,
        entity_ids=("service",),
        subject_entity_id="service",
        predicate="region",
        object_value="eu-west",
        derivations=(
            SemanticDerivationInput(
                episode_id=region_support.id,
                relation=SemanticDerivationRelation.SUPPORTS,
                contribution_score=0.9,
            ),
        ),
    )
    cue = RetrievalCue(text="where does service data persist now", entity_ids=("service",))
    results = activator.rank(
        candidates=[
            activation_candidate_from_semantic(postgres),
            activation_candidate_from_semantic(region),
            activation_candidate_from_episode(postgres_support),
            activation_candidate_from_episode(region_support),
        ],
        cue=cue,
        references={},
        as_of=_T2,
        config=ActivationConfig(
            retrieval_threshold=-10.0,
            enable_spreading_activation=False,
            enable_duplicate_collapse=False,
            enable_text_entity_seeding=False,
        ),
        limit=5,
        episode_support_index=build_episode_support_index([postgres, region]),
        episode_slot_index=build_episode_slot_index([postgres, region]),
    )
    by_key = {result.memory.memory_key: result for result in results}  # type: ignore[union-attr]
    postgres_diag = by_key["postgres-support"].diagnostics
    region_diag = by_key["region-support"].diagnostics
    assert postgres_diag is not None
    assert region_diag is not None
    assert postgres_diag.slot_fit == 1.0
    assert postgres_diag.slot_fit_source is SlotFitSource.SUPPORT
    assert postgres_diag.selected_support_revision_key == postgres.revision_key
    assert region_diag.slot_fit == 1.0
    assert region_diag.slot_fit_source is SlotFitSource.SUPPORT
    assert region_diag.selected_support_revision_key == region.revision_key


def test_historical_support_diagnostics_follow_valid_revision() -> None:
    activator = ACTRDeclarativeActivator()
    historical_support = _episode(
        episode_id="ep-dynamo",
        memory_key="dynamo-support",
        statement="Production used dynamo at that time.",
        entity_ids=("service", "dynamo"),
        started_at=_T1,
    )
    historical = _semantic(
        semantic_id="sem-dynamo",
        memory_key="service-database-dynamo",
        statement="The service database was dynamo.",
        status=SemanticMemoryStatus.SUPERSEDED,
        entity_ids=("service",),
        subject_entity_id="service",
        predicate="database",
        object_value="dynamo",
        derivations=(
            SemanticDerivationInput(
                episode_id=historical_support.id,
                relation=SemanticDerivationRelation.SUPPORTS,
                contribution_score=0.8,
            ),
        ),
    )
    cue = RetrievalCue(
        text="what database was in use", entity_ids=("service",), predicate="database"
    )
    results = activator.rank(
        candidates=[
            activation_candidate_from_semantic(historical),
            activation_candidate_from_episode(historical_support),
        ],
        cue=cue,
        references={},
        as_of=_T2,
        config=ActivationConfig(
            retrieval_threshold=-10.0,
            enable_spreading_activation=False,
            enable_duplicate_collapse=False,
            enable_text_entity_seeding=False,
        ),
        limit=5,
        valid_at=_T1,
        episode_support_index=build_episode_support_index([historical]),
        episode_slot_index=build_episode_slot_index([historical]),
    )
    support_result = next(item for item in results if item.memory_kind is MemoryKind.EPISODE)
    diagnostics = support_result.diagnostics
    assert diagnostics is not None
    assert diagnostics.temporal_mode == "historical"
    assert diagnostics.eligibility is RetrievalEligibility.HISTORICAL_SLOT_ADMISSION
    assert diagnostics.slot_fit == 1.0
    assert diagnostics.selected_support_revision_key == historical.revision_key
