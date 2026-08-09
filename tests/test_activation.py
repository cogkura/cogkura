"""Unit tests for ACT-R declarative activation."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from types import MappingProxyType

from cogkura.algorithms.activation import (
    ACTRDeclarativeActivator,
    activation_candidate_from_episode,
    calculate_base_level,
)
from cogkura.models import (
    ActivationConfig,
    EpisodeEvidenceInput,
    MemoryIdentity,
    MemoryKind,
    RetrievalCue,
    StoredEpisode,
)


def _episode(
    *,
    episode_id: str = "ep-1",
    memory_key: str = "episode-key",
    statement: str = "PostgreSQL incident resolved.",
    created_at: datetime | None = None,
) -> StoredEpisode:
    created = created_at or datetime(2026, 8, 7, 10, 0, tzinfo=UTC)
    return StoredEpisode(
        id=episode_id,
        tenant_id="company_123",
        subject_id="customer_42",
        memory_key=memory_key,
        statement=statement,
        started_at=created,
        ended_at=created,
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
        entities=(),
        metadata=MappingProxyType({}),
        created_at=created,
        updated_at=created,
    )


def test_base_level_reference_vectors() -> None:
    as_of = datetime(2026, 8, 7, 11, 0, tzinfo=UTC)
    config_kwargs = {
        "as_of": as_of,
        "decay": 0.5,
        "constant": 0.0,
        "time_unit_seconds": 1.0,
        "minimum_elapsed_seconds": 1.0,
    }
    one_unit_ago = as_of - timedelta(seconds=1)
    assert math.isclose(
        calculate_base_level((one_unit_ago,), **config_kwargs),
        0.0,
        rel_tol=1e-9,
    )
    sixteen_units_ago = as_of - timedelta(seconds=16)
    assert math.isclose(
        calculate_base_level((sixteen_units_ago,), **config_kwargs),
        math.log(16**-0.5),
        rel_tol=1e-9,
    )
    four_units_ago = as_of - timedelta(seconds=4)
    assert math.isclose(
        calculate_base_level(
            (four_units_ago, four_units_ago, four_units_ago, four_units_ago), **config_kwargs
        ),
        math.log(2.0),
        rel_tol=1e-9,
    )


def test_partial_match_ranks_text_overlap() -> None:
    activator = ACTRDeclarativeActivator()
    as_of = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)
    created = datetime(2026, 8, 7, 10, 0, tzinfo=UTC)
    candidates = [
        activation_candidate_from_episode(
            _episode(
                memory_key="match",
                statement="Payment incident resolved with PostgreSQL.",
                created_at=created,
            )
        ),
        activation_candidate_from_episode(
            _episode(
                episode_id="ep-2",
                memory_key="miss",
                statement="Unrelated weather report.",
                created_at=created,
            )
        ),
    ]
    config = ActivationConfig(retrieval_threshold=-10.0)
    results = activator.rank(
        candidates=candidates,
        cue=RetrievalCue(text="payment PostgreSQL"),
        references={},
        as_of=as_of,
        config=config,
        limit=5,
    )
    assert len(results) == 2
    assert results[0].memory_kind is MemoryKind.EPISODE
    assert _episode_key(results[0]) == "match"
    assert results[0].activation > results[1].activation


def test_threshold_excludes_low_activation() -> None:
    activator = ACTRDeclarativeActivator()
    as_of = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)
    old = datetime(2020, 1, 1, tzinfo=UTC)
    candidate = activation_candidate_from_episode(_episode(created_at=old))
    config = ActivationConfig(retrieval_threshold=0.0)
    results = activator.rank(
        candidates=[candidate],
        cue=RetrievalCue(text="unrelated topic"),
        references={},
        as_of=as_of,
        config=config,
        limit=5,
    )
    assert results == []


def test_reinforcement_increases_activation() -> None:
    activator = ACTRDeclarativeActivator()
    as_of = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)
    created = datetime(2026, 8, 7, 10, 0, tzinfo=UTC)
    candidate = activation_candidate_from_episode(_episode(created_at=created))
    identity = MemoryIdentity(memory_kind=MemoryKind.EPISODE, memory_key="episode-key")
    config = ActivationConfig(retrieval_threshold=-10.0)
    cue = RetrievalCue(text="PostgreSQL incident")
    without_refs = activator.rank(
        candidates=[candidate],
        cue=cue,
        references={},
        as_of=as_of,
        config=config,
        limit=1,
    )[0]
    reinforced = activator.rank(
        candidates=[candidate],
        cue=cue,
        references={identity: (datetime(2026, 8, 7, 11, 30, tzinfo=UTC),)},
        as_of=as_of,
        config=config,
        limit=1,
    )[0]
    assert reinforced.activation > without_refs.activation


def test_higher_activation_lower_latency() -> None:
    activator = ACTRDeclarativeActivator()
    as_of = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)
    created = datetime(2026, 8, 7, 11, 0, tzinfo=UTC)
    strong = activation_candidate_from_episode(
        _episode(memory_key="strong", statement="payment incident", created_at=created)
    )
    weak = activation_candidate_from_episode(
        _episode(
            episode_id="ep-2",
            memory_key="weak",
            statement="unrelated",
            created_at=datetime(2020, 1, 1, tzinfo=UTC),
        )
    )
    config = ActivationConfig(retrieval_threshold=-10.0)
    results = activator.rank(
        candidates=[weak, strong],
        cue=RetrievalCue(text="payment incident"),
        references={},
        as_of=as_of,
        config=config,
        limit=2,
    )
    assert results[0].latency_seconds < results[1].latency_seconds


def _episode_key(result: object) -> str:
    from cogkura.models import RecallResult

    assert isinstance(result, RecallResult)
    memory = result.memory
    assert hasattr(memory, "memory_key")
    return memory.memory_key
