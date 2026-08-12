"""Unit tests for cue relevance and coverage helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from types import MappingProxyType

from cogkura.algorithms.relevance import calculate_cue_coverage, calculate_cue_relevance
from cogkura.algorithms.working_memory import calculate_goal_relevance
from cogkura.models import (
    ActivationComponents,
    EpisodeEvidenceInput,
    MemoryKind,
    RecallResult,
    RetrievalCue,
    StoredEpisode,
)

_NOW = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)


def _episode(
    *,
    memory_key: str,
    statement: str,
    subject_id: str | None = "customer_42",
) -> StoredEpisode:
    return StoredEpisode(
        id=f"id-{memory_key}",
        tenant_id="company_123",
        subject_id=subject_id,
        memory_key=memory_key,
        statement=statement,
        started_at=_NOW,
        ended_at=_NOW,
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
        metadata=MappingProxyType({}),
        created_at=_NOW,
        updated_at=_NOW,
    )


def _recall(episode: StoredEpisode, *, score: float = 0.8) -> RecallResult:
    return RecallResult(
        memory_kind=MemoryKind.EPISODE,
        memory=episode,
        activation=score,
        score=score,
        latency_seconds=0.1,
        components=ActivationComponents(
            base_level=score,
            spreading=0.0,
            partial_match=0.0,
            noise=0.0,
            total=score,
        ),
        reason="test",
    )


def test_goal_relevance_alias_matches_cue_relevance() -> None:
    episode = _episode(memory_key="a", statement="Alice works at Globex.")
    recall = _recall(episode)
    cue = RetrievalCue(text="Alice Globex")
    assert calculate_goal_relevance(recall, cue) == calculate_cue_relevance(recall, cue)


def test_union_text_coverage_across_candidates() -> None:
    episode_a = _episode(memory_key="a", statement="Alice works at company.")
    episode_b = _episode(memory_key="b", statement="Globex headquarters in London.")
    candidates = [_recall(episode_a), _recall(episode_b)]
    cue = RetrievalCue(text="Alice works_at Globex")
    coverage = calculate_cue_coverage(candidates, cue)
    single_best = max(calculate_cue_relevance(recall, cue) for recall in candidates)
    assert coverage > single_best


def test_empty_candidates_coverage_is_zero() -> None:
    assert calculate_cue_coverage([], RetrievalCue(text="Alice")) == 0.0
