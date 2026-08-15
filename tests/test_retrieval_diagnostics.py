"""Regression tests for 0.14.4 retrieval diagnostics and provenance."""

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
from cogkura.models import (
    ActivationComponents,
    ActivationConfig,
    EpisodeEntity,
    EpisodeEvidenceInput,
    MemoryKind,
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


def _episode(
    *,
    episode_id: str,
    memory_key: str,
    statement: str,
    entity_ids: tuple[str, ...] = (),
) -> StoredEpisode:
    return StoredEpisode(
        id=episode_id,
        tenant_id="company_123",
        subject_id="team",
        memory_key=memory_key,
        statement=statement,
        started_at=_T1,
        ended_at=_T1,
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
        created_at=_T1,
        updated_at=_T1,
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
        revision_key=f"rev:{memory_key}",
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
        observation_evidence=(
            EpisodeEvidenceInput(
                observation_id=f"obs-{memory_key}",
                observation_revision=1,
                sequence_number=0,
            ),
        ),
        entities=tuple(EpisodeEntity(entity_id=eid, role="mention") for eid in entity_ids),
        metadata=MappingProxyType({}),
        created_at=_T0,
        updated_at=_T1,
    )


def test_threshold_eligibility_has_structured_diagnostics() -> None:
    activator = ACTRDeclarativeActivator()
    semantic = _semantic(
        semantic_id="sem-region",
        memory_key="service-region-eu",
        statement="The service region is eu-west.",
        status=SemanticMemoryStatus.ACTIVE,
        entity_ids=("service",),
        subject_entity_id="service",
        predicate="region",
        object_value="eu-west",
    )
    results = activator.rank(
        candidates=[activation_candidate_from_semantic(semantic)],
        cue=RetrievalCue(
            text="service region eu-west", entity_ids=("service",), predicate="region"
        ),
        references={},
        as_of=_T2,
        config=ActivationConfig(
            retrieval_threshold=-10.0,
            enable_spreading_activation=False,
            enable_duplicate_collapse=False,
            enable_text_entity_seeding=False,
            enable_semantic_slot_admission=False,
        ),
        limit=5,
        episode_support_index=build_episode_support_index([semantic]),
        episode_slot_index=build_episode_slot_index([semantic]),
    )
    result = results[0]
    diagnostics = result.diagnostics
    assert diagnostics is not None
    assert diagnostics.eligibility is RetrievalEligibility.THRESHOLD
    assert diagnostics.rank_activation >= result.activation
    assert diagnostics.temporal_mode == "current"
    assert diagnostics.slot_fit == 1.0
    assert diagnostics.slot_fit_source is SlotFitSource.SEMANTIC
    assert diagnostics.semantic_slot_key == semantic.slot_key
    assert diagnostics.observation_evidence_ids == (f"obs-{semantic.memory_key}",)
    assert "eligibility=threshold" in result.reason


def test_entity_slot_admission_marks_support_with_provenance() -> None:
    activator = ACTRDeclarativeActivator()
    support = _episode(
        episode_id="ep-support",
        memory_key="postgres-support",
        statement="Production config confirms postgres for service storage.",
        entity_ids=("service", "postgres"),
    )
    semantic = _semantic(
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
                episode_id=support.id,
                relation=SemanticDerivationRelation.SUPPORTS,
                contribution_score=0.9,
            ),
        ),
    )
    results = activator.rank(
        candidates=[
            activation_candidate_from_episode(support),
            activation_candidate_from_semantic(semantic),
        ],
        cue=RetrievalCue(text="what is service database now", entity_ids=("service",)),
        references={},
        as_of=_T2,
        config=ActivationConfig(
            retrieval_threshold=0.0,
            enable_spreading_activation=False,
            enable_duplicate_collapse=False,
            enable_text_entity_seeding=False,
        ),
        limit=5,
        episode_support_index=build_episode_support_index([semantic]),
        episode_slot_index=build_episode_slot_index([semantic]),
    )
    support_result = next(item for item in results if item.memory_kind is MemoryKind.EPISODE)
    diagnostics = support_result.diagnostics
    assert diagnostics is not None
    assert diagnostics.eligibility is RetrievalEligibility.ENTITY_SLOT_ADMISSION
    assert diagnostics.soft_admitted is True
    assert diagnostics.slot_fit_source is SlotFitSource.SUPPORT
    assert diagnostics.slot_fit == 1.0
    assert diagnostics.selected_support_revision_key == semantic.revision_key
    assert diagnostics.support_provenance
    assert diagnostics.support_provenance[0].semantic_revision_key == semantic.revision_key
    assert "soft_admitted=true" in support_result.reason


def test_historical_slot_admission_sets_historical_eligibility() -> None:
    activator = ACTRDeclarativeActivator()
    historical = _semantic(
        semantic_id="sem-dynamo",
        memory_key="service-database-dynamo",
        statement="The service database was dynamo.",
        status=SemanticMemoryStatus.SUPERSEDED,
        entity_ids=("service",),
        subject_entity_id="service",
        predicate="database",
        object_value="dynamo",
    )
    results = activator.rank(
        candidates=[activation_candidate_from_semantic(historical)],
        cue=RetrievalCue(
            text="what database was in use", entity_ids=("service",), predicate="database"
        ),
        references={},
        as_of=_T2,
        config=ActivationConfig(
            retrieval_threshold=0.0,
            enable_spreading_activation=False,
            enable_duplicate_collapse=False,
            enable_text_entity_seeding=False,
        ),
        limit=5,
        valid_at=_T0,
        episode_support_index=build_episode_support_index([historical]),
        episode_slot_index=build_episode_slot_index([historical]),
    )
    result = results[0]
    diagnostics = result.diagnostics
    assert diagnostics is not None
    assert diagnostics.eligibility is RetrievalEligibility.HISTORICAL_SLOT_ADMISSION
    assert diagnostics.temporal_mode == "historical"


def test_recall_result_default_diagnostics_remains_optional() -> None:
    episode = _episode(
        episode_id="ep-compat",
        memory_key="compat-note",
        statement="Compatibility check entry.",
        entity_ids=("service",),
    )
    result = RecallResult(
        memory_kind=MemoryKind.EPISODE,
        memory=episode,
        activation=1.0,
        score=0.5,
        latency_seconds=0.1,
        components=ActivationComponents(
            base_level=0.0,
            spreading=0.0,
            partial_match=0.0,
            noise=0.0,
            total=1.0,
        ),
        reason="compat",
    )
    assert result.diagnostics is None
