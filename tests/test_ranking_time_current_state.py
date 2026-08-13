"""Tests for 0.11 ranking, simulated time, and current-state behaviour."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import MappingProxyType

from cogkura import Memory, ObservationInput
from cogkura.algorithms.activation import (
    ACTRDeclarativeActivator,
    _text_similarity,
    activation_candidate_from_episode,
)
from cogkura.algorithms.forgetting import EbbinghausForgettingEvaluator
from cogkura.models import (
    ActivationConfig,
    EpisodeEvidenceInput,
    ForgettingConfig,
    MemoryKind,
    MemoryRetentionState,
    RetrievalCue,
    StoredEpisode,
)

_T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
_T1 = datetime(2026, 1, 4, 12, 0, tzinfo=UTC)
_T2 = datetime(2026, 1, 6, 12, 0, tzinfo=UTC)
_T3 = datetime(2026, 1, 16, 12, 0, tzinfo=UTC)


def _episode(
    *,
    memory_key: str,
    statement: str,
    started_at: datetime,
    created_at: datetime | None = None,
    importance: float = 0.7,
) -> StoredEpisode:
    created = created_at or started_at
    return StoredEpisode(
        id=f"id-{memory_key}",
        tenant_id="company_123",
        subject_id="team",
        memory_key=memory_key,
        statement=statement,
        started_at=started_at,
        ended_at=started_at,
        confidence=0.9,
        importance=importance,
        is_active=True,
        evidence=(
            EpisodeEvidenceInput(
                observation_id=f"obs-{memory_key}",
                observation_revision=1,
                sequence_number=0,
            ),
        ),
        entities=(),
        metadata=MappingProxyType({}),
        created_at=created,
        updated_at=created,
    )


async def test_encode_and_recall_with_as_of_do_not_raise() -> None:
    memory = Memory()
    tenant_id = "company_123"

    await memory.observe(
        ObservationInput(
            tenant_id=tenant_id,
            subject_id="team",
            source_namespace="chat.messages",
            source_record_id="msg-1",
            content="Project Atlas will use FastAPI for all public HTTP endpoints.",
            observed_at=_T0,
        )
    )
    await memory.encode_episodes(tenant_id=tenant_id, as_of=_T0)
    await memory.consolidate_semantics(tenant_id=tenant_id, as_of=_T0)

    results = await memory.recall(
        "Which API framework was selected for Project Atlas?",
        tenant_id=tenant_id,
        as_of=_T1,
    )
    assert results


async def test_recall_excludes_episodes_after_valid_at() -> None:
    memory = Memory()
    tenant_id = "company_123"

    await memory.observe(
        ObservationInput(
            tenant_id=tenant_id,
            subject_id="team",
            source_namespace="chat.messages",
            source_record_id="early",
            content="Project Atlas selected Redis for job coordination.",
            observed_at=_T2,
        )
    )
    await memory.observe(
        ObservationInput(
            tenant_id=tenant_id,
            subject_id="team",
            source_namespace="chat.messages",
            source_record_id="late",
            content="Project Atlas team reviewed customer support escalations.",
            observed_at=_T3,
        )
    )
    await memory.encode_episodes(tenant_id=tenant_id, as_of=_T3)

    results = await memory.recall(
        "Project Atlas job coordination",
        tenant_id=tenant_id,
        valid_at=_T2,
        limit=10,
        as_of=_T3,
    )
    keys = {
        result.memory.memory_key for result in results if result.memory_kind is MemoryKind.EPISODE
    }
    assert all(
        episode.started_at <= _T2
        for result in results
        if isinstance(result.memory, StoredEpisode)
        for episode in [result.memory]
    )
    assert keys


def test_text_similarity_idf_prefers_distinctive_token() -> None:
    candidates = [
        activation_candidate_from_episode(
            _episode(
                memory_key="clone-1",
                statement="Project Atlas team reviewed pagination.",
                started_at=_T1,
            )
        ),
        activation_candidate_from_episode(
            _episode(
                memory_key="clone-2",
                statement="Project Atlas team reviewed support escalations.",
                started_at=_T1,
            )
        ),
        activation_candidate_from_episode(
            _episode(
                memory_key="fact",
                statement="Project Atlas will use FastAPI for all public HTTP endpoints.",
                started_at=_T1,
            )
        ),
    ]
    idf = {
        token: 1.0
        for token in {"project", "atlas", "team", "reviewed", "pagination", "support", "fastapi"}
    }
    idf["project"] = 1.1
    idf["atlas"] = 1.1
    idf["fastapi"] = 3.0

    clone_score = _text_similarity(
        "Which API framework was selected for Project Atlas?",
        candidates[0].text,
        idf_weights=idf,
    )
    fact_score = _text_similarity(
        "Which API framework was selected for Project Atlas?",
        candidates[2].text,
        idf_weights=idf,
    )
    assert fact_score > clone_score


def test_duplicate_collapse_keeps_one_paraphrase() -> None:
    activator = ACTRDeclarativeActivator()
    created = _T1
    clone_statement = "Project Atlas team reviewed customer support escalations."
    candidates = [
        activation_candidate_from_episode(
            _episode(
                memory_key=f"clone-{index}",
                statement=clone_statement.replace("escalations", suffix),
                started_at=created,
            )
        )
        for index, suffix in enumerate(
            ("escalations", "tickets", "backlog", "metrics"),
            start=1,
        )
    ] + [
        activation_candidate_from_episode(
            _episode(
                memory_key="decision",
                statement="Project Atlas will use FastAPI for all public HTTP endpoints.",
                started_at=created,
            )
        )
    ]
    config = ActivationConfig(retrieval_threshold=-10.0, duplicate_jaccard_threshold=0.75)
    results = activator.rank(
        candidates=candidates,
        cue=RetrievalCue(text="FastAPI Project Atlas"),
        references={},
        as_of=_T1 + timedelta(hours=1),
        config=config,
        limit=5,
    )
    clone_count = sum(
        1
        for result in results
        if "team reviewed customer support" in result.memory.statement  # type: ignore[union-attr]
    )
    assert clone_count <= 1
    assert any("FastAPI" in result.memory.statement for result in results)  # type: ignore[union-attr]


def test_importance_scaling_fades_low_importance_faster() -> None:
    evaluator = EbbinghausForgettingEvaluator()
    activation_config = ActivationConfig(retrieval_threshold=-3.0, time_unit_seconds=1.0)
    forgetting_config = ForgettingConfig(
        grace_period_seconds=1.0,
        enable_importance_scaling=True,
        protect_semantic_support=False,
    )
    high = evaluator.evaluate(
        candidate=activation_candidate_from_episode(
            _episode(
                memory_key="high",
                statement="Important decision.",
                started_at=_T0,
                importance=0.9,
            )
        ),
        references=(),
        previous=None,
        as_of=_T0 + timedelta(days=30),
        activation_config=activation_config,
        forgetting_config=forgetting_config,
        tenant_id="company_123",
    )
    low = evaluator.evaluate(
        candidate=activation_candidate_from_episode(
            _episode(
                memory_key="low",
                statement="Project Atlas team filler chatter.",
                started_at=_T0,
                importance=0.1,
            )
        ),
        references=(),
        previous=None,
        as_of=_T0 + timedelta(days=30),
        activation_config=activation_config,
        forgetting_config=forgetting_config,
        tenant_id="company_123",
    )
    assert high.dynamics.last_retention_score > low.dynamics.last_retention_score


def test_protected_supporting_episode_cannot_be_forgotten() -> None:
    evaluator = EbbinghausForgettingEvaluator()
    candidate = activation_candidate_from_episode(
        _episode(
            memory_key="supporting",
            statement="PostgreSQL advisory locks coordinate jobs.",
            started_at=_T0,
            importance=0.2,
        )
    )
    decision = evaluator.evaluate(
        candidate=candidate,
        references=(),
        previous=None,
        as_of=_T0 + timedelta(days=365),
        activation_config=ActivationConfig(retrieval_threshold=-3.0, time_unit_seconds=1.0),
        forgetting_config=ForgettingConfig(
            grace_period_seconds=1.0,
            enable_importance_scaling=True,
            protect_semantic_support=True,
        ),
        tenant_id="company_123",
        protected_identities=frozenset({candidate.identity}),
    )
    assert decision.dynamics.retention_state is MemoryRetentionState.FADING
