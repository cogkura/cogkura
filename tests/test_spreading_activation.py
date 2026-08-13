"""Unit tests for spreading activation."""

from __future__ import annotations

import random
from datetime import UTC, datetime
from types import MappingProxyType

from cogkura.algorithms.activation import (
    ACTRDeclarativeActivator,
    activation_candidate_from_episode,
    activation_candidate_from_semantic,
)
from cogkura.algorithms.spreading import (
    DeterministicSpreadingActivator,
    calculate_spreading_activation,
)
from cogkura.models import (
    ActivationConfig,
    EpisodeEntity,
    EpisodeEvidenceInput,
    RetrievalCue,
    SemanticCardinality,
    SemanticMemoryStatus,
    SemanticPolarity,
    StoredEpisode,
    StoredSemanticMemory,
)

_AS_OF = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)
_CREATED = datetime(2026, 8, 7, 10, 0, tzinfo=UTC)


def _episode(
    *,
    memory_key: str,
    statement: str,
    entity_ids: tuple[str, ...] = (),
) -> StoredEpisode:
    return StoredEpisode(
        id=f"id-{memory_key}",
        tenant_id="company_123",
        subject_id="customer_42",
        memory_key=memory_key,
        statement=statement,
        started_at=_CREATED,
        ended_at=_CREATED,
        confidence=0.9,
        importance=0.7,
        is_active=True,
        evidence=(
            EpisodeEvidenceInput(
                observation_id="obs-1",
                observation_revision=1,
                sequence_number=0,
            ),
        ),
        entities=tuple(EpisodeEntity(entity_id=eid, role="mention") for eid in entity_ids),
        metadata=MappingProxyType({}),
        created_at=_CREATED,
        updated_at=_CREATED,
    )


def _semantic(
    *,
    memory_key: str,
    statement: str,
    subject_entity_id: str | None = None,
    object_entity_id: str | None = None,
    entity_ids: tuple[str, ...] = (),
) -> StoredSemanticMemory:
    entities = tuple(EpisodeEntity(entity_id=eid, role="mention") for eid in entity_ids)
    return StoredSemanticMemory(
        id=f"id-{memory_key}",
        tenant_id="company_123",
        subject_id="customer_42",
        memory_key=memory_key,
        slot_key="slot",
        revision_key=f"legacy:{memory_key}",
        revision_number=1,
        statement=statement,
        subject_entity_id=subject_entity_id,
        predicate="related_to",
        object_value="value",
        object_entity_id=object_entity_id,
        polarity=SemanticPolarity.AFFIRM,
        cardinality=SemanticCardinality.ONE,
        qualifiers=MappingProxyType({}),
        confidence=0.9,
        importance=0.7,
        status=SemanticMemoryStatus.ACTIVE,
        support_count=1,
        contradiction_count=0,
        first_supported_at=_CREATED,
        last_supported_at=_CREATED,
        valid_from=None,
        valid_until=None,
        is_active=True,
        derivations=(),
        observation_evidence=(),
        entities=entities,
        metadata=MappingProxyType({}),
        created_at=_CREATED,
        updated_at=_CREATED,
    )


def _rank_spreading(
    candidates: list,
    *,
    entity_ids: tuple[str, ...],
    config: ActivationConfig | None = None,
) -> list:
    activator = ACTRDeclarativeActivator()
    return activator.rank(
        candidates=candidates,
        cue=RetrievalCue(entity_ids=entity_ids),
        references={},
        as_of=_AS_OF,
        config=config or ActivationConfig(retrieval_threshold=-10.0),
        limit=10,
    )


def test_spreading_enabled_by_default() -> None:
    config = ActivationConfig()
    assert config.enable_spreading_activation is True
    assert config.spreading_decay == 0.5
    assert config.spreading_max_hops == 2
    assert config.spreading_min_activation == 0.01


def test_direct_association() -> None:
    candidates = [
        activation_candidate_from_episode(
            _episode(memory_key="a", statement="Alice memory", entity_ids=("alice",))
        ),
        activation_candidate_from_episode(
            _episode(memory_key="b", statement="Bob memory", entity_ids=("bob",))
        ),
    ]
    result = calculate_spreading_activation(
        candidates=candidates,
        cue=RetrievalCue(entity_ids=("alice",)),
        config=ActivationConfig(),
    )
    identity_a = candidates[0].identity
    identity_b = candidates[1].identity
    assert result.scores.get(identity_a, 0.0) > 0.0
    assert result.scores.get(identity_b, 0.0) == 0.0


def test_two_hop_association() -> None:
    candidates = [
        activation_candidate_from_episode(
            _episode(
                memory_key="a",
                statement="Alice proposed Project Kura",
                entity_ids=("alice", "project-kura"),
            )
        ),
        activation_candidate_from_semantic(
            _semantic(
                memory_key="b",
                statement="Project Kura uses PostgreSQL",
                subject_entity_id="project-kura",
                object_entity_id="postgresql",
            )
        ),
    ]
    result = calculate_spreading_activation(
        candidates=candidates,
        cue=RetrievalCue(entity_ids=("alice",)),
        config=ActivationConfig(),
    )
    spread_a = result.scores[candidates[0].identity]
    spread_b = result.scores[candidates[1].identity]
    assert spread_a > spread_b > 0.0
    assert result.metadata[candidates[0].identity].hop == 1
    assert result.metadata[candidates[1].identity].hop == 2


def test_unrelated_memory_receives_no_spreading() -> None:
    candidates = [
        activation_candidate_from_episode(
            _episode(
                memory_key="a",
                statement="Alice memory",
                entity_ids=("alice",),
            )
        ),
        activation_candidate_from_episode(
            _episode(
                memory_key="c",
                statement="Holiday in Turkey",
                entity_ids=("holiday",),
            )
        ),
    ]
    result = calculate_spreading_activation(
        candidates=candidates,
        cue=RetrievalCue(entity_ids=("alice",)),
        config=ActivationConfig(),
    )
    assert result.scores.get(candidates[1].identity, 0.0) == 0.0


def test_fan_sensitivity() -> None:
    rare = activation_candidate_from_episode(
        _episode(memory_key="rare", statement="Rare entity", entity_ids=("rare-entity",))
    )
    common_memories = [
        activation_candidate_from_episode(
            _episode(
                memory_key=f"common-{index}",
                statement=f"Common memory {index}",
                entity_ids=("common-entity",),
            )
        )
        for index in range(10)
    ]
    candidates = [rare, *common_memories]
    config = ActivationConfig()

    rare_result = calculate_spreading_activation(
        candidates=candidates,
        cue=RetrievalCue(entity_ids=("rare-entity",)),
        config=config,
    )
    common_result = calculate_spreading_activation(
        candidates=candidates,
        cue=RetrievalCue(entity_ids=("common-entity",)),
        config=config,
    )
    rare_contribution = rare_result.scores[rare.identity]
    common_contribution = common_result.scores[common_memories[0].identity]
    assert rare_contribution > common_contribution


def test_distance_decay_with_three_hops() -> None:
    candidates = [
        activation_candidate_from_episode(
            _episode(
                memory_key="m1",
                statement="Hop 1",
                entity_ids=("alice", "e1"),
            )
        ),
        activation_candidate_from_episode(
            _episode(
                memory_key="m2",
                statement="Hop 2",
                entity_ids=("e1", "e2"),
            )
        ),
        activation_candidate_from_episode(
            _episode(
                memory_key="m3",
                statement="Hop 3",
                entity_ids=("e2", "e3"),
            )
        ),
    ]
    config = ActivationConfig(spreading_max_hops=3)
    result = calculate_spreading_activation(
        candidates=candidates,
        cue=RetrievalCue(entity_ids=("alice",)),
        config=config,
    )
    hop1 = result.scores[candidates[0].identity]
    hop2 = result.scores[candidates[1].identity]
    hop3 = result.scores[candidates[2].identity]
    assert hop1 > hop2 > hop3 > 0.0


def test_converging_paths() -> None:
    candidates = [
        activation_candidate_from_episode(
            _episode(
                memory_key="m1",
                statement="Alice to project",
                entity_ids=("alice", "project"),
            )
        ),
        activation_candidate_from_episode(
            _episode(
                memory_key="m2",
                statement="Postgres to project",
                entity_ids=("postgres", "project"),
            )
        ),
        activation_candidate_from_episode(
            _episode(
                memory_key="m3",
                statement="Project hub",
                entity_ids=("project", "postgres"),
            )
        ),
        activation_candidate_from_episode(
            _episode(
                memory_key="m4",
                statement="Project only",
                entity_ids=("project", "other"),
            )
        ),
    ]
    config = ActivationConfig(spreading_max_hops=2)
    result = calculate_spreading_activation(
        candidates=candidates,
        cue=RetrievalCue(entity_ids=("alice", "postgres")),
        config=config,
    )
    converging = result.scores[candidates[2].identity]
    single_path = result.scores[candidates[3].identity]
    assert converging > single_path > 0.0


def test_cycle_termination() -> None:
    candidates = [
        activation_candidate_from_episode(
            _episode(memory_key="m1", statement="A to B", entity_ids=("a", "b"))
        ),
        activation_candidate_from_episode(
            _episode(memory_key="m2", statement="B to A", entity_ids=("b", "a"))
        ),
    ]
    config = ActivationConfig(spreading_max_hops=5)
    result = calculate_spreading_activation(
        candidates=candidates,
        cue=RetrievalCue(entity_ids=("a",)),
        config=config,
    )
    assert len(result.scores) == 2
    assert all(score <= config.source_activation for score in result.scores.values())


def test_propagation_threshold() -> None:
    candidates = [
        activation_candidate_from_episode(
            _episode(memory_key="m1", statement="Start", entity_ids=("alice", "e1"))
        ),
        activation_candidate_from_episode(
            _episode(memory_key="m2", statement="Far", entity_ids=("e1", "e2"))
        ),
        activation_candidate_from_episode(
            _episode(memory_key="m3", statement="Farther", entity_ids=("e2", "e3"))
        ),
    ]
    strict = ActivationConfig(
        spreading_max_hops=3,
        spreading_min_activation=0.5,
        spreading_decay=0.1,
    )
    result = calculate_spreading_activation(
        candidates=candidates,
        cue=RetrievalCue(entity_ids=("alice",)),
        config=strict,
    )
    assert candidates[0].identity in result.scores
    assert candidates[2].identity not in result.scores


def test_spreading_disabled() -> None:
    candidates = [
        activation_candidate_from_episode(
            _episode(memory_key="a", statement="Alice", entity_ids=("alice",))
        ),
    ]
    results = _rank_spreading(
        candidates,
        entity_ids=("alice",),
        config=ActivationConfig(
            enable_spreading_activation=False,
            retrieval_threshold=-10.0,
        ),
    )
    assert results[0].components.spreading == 0.0


def test_text_only_cue_produces_no_spreading_without_seeding() -> None:
    candidates = [
        activation_candidate_from_episode(
            _episode(memory_key="a", statement="Alice memory", entity_ids=("alice",))
        ),
    ]
    activator = ACTRDeclarativeActivator()
    results = activator.rank(
        candidates=candidates,
        cue=RetrievalCue(text="Alice memory"),
        references={},
        as_of=_AS_OF,
        config=ActivationConfig(
            retrieval_threshold=-10.0,
            enable_text_entity_seeding=False,
        ),
        limit=5,
    )
    assert results[0].components.spreading == 0.0
    assert results[0].components.base_level != 0.0 or results[0].components.partial_match != 0.0


def test_text_only_cue_seeds_spreading_from_candidate_entities() -> None:
    candidates = [
        activation_candidate_from_episode(
            _episode(memory_key="a", statement="Alice memory", entity_ids=("alice",))
        ),
    ]
    activator = ACTRDeclarativeActivator()
    results = activator.rank(
        candidates=candidates,
        cue=RetrievalCue(text="Alice memory"),
        references={},
        as_of=_AS_OF,
        config=ActivationConfig(retrieval_threshold=-10.0),
        limit=5,
    )
    assert results[0].components.spreading > 0.0


def test_candidate_order_independence() -> None:
    candidates = [
        activation_candidate_from_episode(
            _episode(
                memory_key="a",
                statement="Alice to project",
                entity_ids=("alice", "project-kura"),
            )
        ),
        activation_candidate_from_semantic(
            _semantic(
                memory_key="b",
                statement="Project Kura uses PostgreSQL",
                subject_entity_id="project-kura",
                object_entity_id="postgresql",
            )
        ),
        activation_candidate_from_episode(
            _episode(memory_key="c", statement="Unrelated", entity_ids=("holiday",))
        ),
    ]
    cue = RetrievalCue(entity_ids=("alice",))
    config = ActivationConfig(retrieval_threshold=-10.0)
    activator = ACTRDeclarativeActivator()

    baseline = activator.rank(
        candidates=candidates,
        cue=cue,
        references={},
        as_of=_AS_OF,
        config=config,
        limit=10,
    )
    shuffled = list(candidates)
    random.Random(42).shuffle(shuffled)
    shuffled_results = activator.rank(
        candidates=shuffled,
        cue=cue,
        references={},
        as_of=_AS_OF,
        config=config,
        limit=10,
    )
    assert [item.activation for item in baseline] == [item.activation for item in shuffled_results]
    assert [_memory_key(item) for item in baseline] == [
        _memory_key(item) for item in shuffled_results
    ]


def test_tenant_isolation_via_candidate_scope() -> None:
    tenant_a = activation_candidate_from_episode(
        _episode(memory_key="a", statement="Alice tenant A", entity_ids=("alice",))
    )
    tenant_b = activation_candidate_from_episode(
        StoredEpisode(
            id="id-b",
            tenant_id="other_tenant",
            subject_id="other",
            memory_key="b",
            statement="Alice tenant B",
            started_at=_CREATED,
            ended_at=_CREATED,
            confidence=0.9,
            importance=0.7,
            is_active=True,
            evidence=(
                EpisodeEvidenceInput(
                    observation_id="obs-2",
                    observation_revision=1,
                    sequence_number=0,
                ),
            ),
            entities=(EpisodeEntity(entity_id="alice", role="mention"),),
            metadata=MappingProxyType({}),
            created_at=_CREATED,
            updated_at=_CREATED,
        )
    )
    result = calculate_spreading_activation(
        candidates=[tenant_a],
        cue=RetrievalCue(entity_ids=("alice",)),
        config=ActivationConfig(),
    )
    assert tenant_a.identity in result.scores
    cross_tenant = calculate_spreading_activation(
        candidates=[tenant_a, tenant_b],
        cue=RetrievalCue(entity_ids=("alice",)),
        config=ActivationConfig(),
    )
    assert cross_tenant.scores[tenant_b.identity] > 0.0
    scoped = calculate_spreading_activation(
        candidates=[tenant_a],
        cue=RetrievalCue(entity_ids=("alice",)),
        config=ActivationConfig(),
    )
    assert tenant_b.identity not in scoped.scores


def test_deterministic_spreading_activator_protocol() -> None:
    activator = DeterministicSpreadingActivator()
    candidates = [
        activation_candidate_from_episode(
            _episode(memory_key="a", statement="Alice", entity_ids=("alice",))
        ),
    ]
    result = activator.calculate(
        candidates=candidates,
        cue=RetrievalCue(entity_ids=("alice",)),
        config=ActivationConfig(),
    )
    assert result.scores[candidates[0].identity] > 0.0


def _memory_key(result: object) -> str:
    from cogkura.models import RecallResult

    assert isinstance(result, RecallResult)
    memory = result.memory
    assert hasattr(memory, "memory_key")
    return memory.memory_key
