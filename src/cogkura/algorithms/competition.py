"""Observational cue-competition diagnostics for recall inspection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Protocol

from cogkura.algorithms.context_observability import discrimination_set
from cogkura.algorithms.retrieval_features import canonical_content_features
from cogkura.models import (
    CompetitionConfig,
    CompetitionDiagnostics,
    CompetitionDirection,
    CompetitionEvidence,
    CompetitionRunDiagnostics,
    MemoryIdentity,
    MemoryKind,
    RecallInspectionCandidate,
    RecallInspectionDisposition,
    RetrievalDiagnostics,
    StoredEpisode,
    StoredSemanticMemory,
)

_DISCRIMINATION_DISPOSITIONS = frozenset(
    {
        RecallInspectionDisposition.RETURNED,
        RecallInspectionDisposition.BELOW_THRESHOLD,
        RecallInspectionDisposition.LIMITED,
    }
)


@dataclass(frozen=True, slots=True)
class CompetitionProfile:
    """Ephemeral internal contract for competition matching."""

    identity: MemoryIdentity
    memory_kind: MemoryKind
    effective_time: datetime
    subject_id: str | None
    subject_entity_id: str | None
    semantic_slot_key: str | None
    predicate: str | None
    entity_ids: tuple[str, ...]
    retrieval_features: tuple[str, ...]
    cue_fit: float
    effective_cue_fit: float
    lineage_group: str | None


def effective_memory_time(
    memory: StoredEpisode | StoredSemanticMemory,
    *,
    memory_kind: MemoryKind,
) -> datetime:
    """Return the deterministic temporal point used for competition direction.

    Precedence:
    - Episode: ``started_at`` else ``created_at``
    - Semantic: ``valid_from`` else ``last_supported_at`` else ``created_at``
    """
    if memory_kind is MemoryKind.EPISODE:
        if not isinstance(memory, StoredEpisode):
            raise TypeError("Episode effective time requires StoredEpisode.")
        return memory.started_at
    if not isinstance(memory, StoredSemanticMemory):
        raise TypeError("Semantic effective time requires StoredSemanticMemory.")
    if memory.valid_from is not None:
        return memory.valid_from
    return memory.last_supported_at


def competition_direction(
    candidate_time: datetime,
    competitor_time: datetime,
) -> CompetitionDirection:
    """Classify competitor temporally relative to the candidate."""
    if competitor_time < candidate_time:
        return CompetitionDirection.PROACTIVE
    if competitor_time > candidate_time:
        return CompetitionDirection.RETROACTIVE
    return CompetitionDirection.CO_TEMPORAL


def _base_cue_fit(diagnostics: RetrievalDiagnostics) -> float:
    return max(diagnostics.semantic_relevance, diagnostics.text_cue_fit)


def _context_correspondence(
    diagnostics: RetrievalDiagnostics,
    *,
    retrieval_context_provided: bool,
) -> float:
    if not retrieval_context_provided:
        return 1.0
    if diagnostics.context_match is not None:
        match = diagnostics.context_match
        if match.score is not None:
            return match.score * match.cue_coverage
        if match.comparable_count > 0:
            return 0.0
        return 1.0
    if diagnostics.support_context is not None:
        return diagnostics.support_context.strength
    return 1.0


def _fact_subject(profile: CompetitionProfile) -> str | None:
    """Return the structured fact subject used for competition matching."""
    return profile.subject_entity_id or profile.subject_id


def subjects_compatible(
    left: CompetitionProfile,
    right: CompetitionProfile,
    *,
    cue_subject_id: str | None,
) -> bool:
    """Return True when memories share a compatible subject dimension."""
    left_subject = _fact_subject(left)
    right_subject = _fact_subject(right)
    if left_subject is not None and right_subject is not None and left_subject == right_subject:
        return True
    if cue_subject_id:
        return left_subject == cue_subject_id and right_subject == cue_subject_id
    return False


def _lineage_group(
    memory: StoredEpisode | StoredSemanticMemory,
    *,
    memory_kind: MemoryKind,
    diagnostics: RetrievalDiagnostics | None,
    episode_slot_index: Mapping[str, str],
) -> str | None:
    if memory_kind is MemoryKind.SEMANTIC:
        assert isinstance(memory, StoredSemanticMemory)
        return f"semantic:{memory.memory_key}"
    assert isinstance(memory, StoredEpisode)
    if diagnostics is not None and diagnostics.support_provenance:
        primary = diagnostics.support_provenance[0]
        return f"semantic:{primary.semantic_memory_key}"
    slot_key = episode_slot_index.get(memory.id)
    if slot_key is not None:
        return f"slot:{slot_key}"
    return None


def _profile_from_candidate(
    candidate: RecallInspectionCandidate,
    *,
    retrieval_context_provided: bool,
    episode_slot_index: Mapping[str, str],
) -> CompetitionProfile | None:
    diagnostics = candidate.diagnostics
    if diagnostics is None:
        return None
    memory = candidate.memory
    subject_entity_id: str | None = None
    predicate: str | None = None
    semantic_slot_key = diagnostics.semantic_slot_key
    entity_ids: tuple[str, ...] = ()
    if isinstance(memory, StoredSemanticMemory):
        subject_entity_id = memory.subject_entity_id
        predicate = memory.predicate
        semantic_slot_key = semantic_slot_key or memory.slot_key
        entity_ids = tuple(
            sorted({entity.entity_id for entity in memory.entities if entity.entity_id})
        )
    elif isinstance(memory, StoredEpisode):
        entity_ids = tuple(
            sorted({entity.entity_id for entity in memory.entities if entity.entity_id})
        )
    retrieval_features = tuple(
        sorted(
            set(diagnostics.matched_direct_features)
            | set(diagnostics.matched_evidence_features)
            | set(canonical_content_features(memory.statement))
        )
    )
    cue_fit = _base_cue_fit(diagnostics)
    context_factor = _context_correspondence(
        diagnostics,
        retrieval_context_provided=retrieval_context_provided,
    )
    return CompetitionProfile(
        identity=MemoryIdentity(
            memory_kind=candidate.memory_kind,
            memory_key=memory.memory_key,
        ),
        memory_kind=candidate.memory_kind,
        effective_time=effective_memory_time(memory, memory_kind=candidate.memory_kind),
        subject_id=memory.subject_id,
        subject_entity_id=subject_entity_id,
        semantic_slot_key=semantic_slot_key,
        predicate=predicate,
        entity_ids=entity_ids,
        retrieval_features=retrieval_features,
        cue_fit=cue_fit,
        effective_cue_fit=cue_fit * context_factor,
        lineage_group=_lineage_group(
            memory,
            memory_kind=candidate.memory_kind,
            diagnostics=diagnostics,
            episode_slot_index=episode_slot_index,
        ),
    )


def _same_lineage(left: CompetitionProfile, right: CompetitionProfile) -> bool:
    if left.lineage_group is None or right.lineage_group is None:
        return False
    return left.lineage_group == right.lineage_group


class CompetitionMatcher(Protocol):
    """Compare two competition profiles for cue-dependent rivalry."""

    def compare(
        self,
        candidate: CompetitionProfile,
        other: CompetitionProfile,
        *,
        config: CompetitionConfig,
        cue_subject_id: str | None,
    ) -> CompetitionEvidence | None:
        """Return competition evidence when profiles plausibly compete."""
        ...


@dataclass(frozen=True, slots=True)
class DeterministicCompetitionMatcher:
    """Deterministic, side-effect-free competition matcher."""

    def compare(
        self,
        candidate: CompetitionProfile,
        other: CompetitionProfile,
        *,
        config: CompetitionConfig,
        cue_subject_id: str | None,
    ) -> CompetitionEvidence | None:
        if candidate.identity == other.identity:
            return None
        if _same_lineage(candidate, other):
            return None

        same_semantic_slot = (
            candidate.semantic_slot_key is not None
            and candidate.semantic_slot_key == other.semantic_slot_key
        )
        same_predicate = (
            candidate.predicate is not None
            and other.predicate is not None
            and candidate.predicate == other.predicate
        )
        same_subject = subjects_compatible(candidate, other, cue_subject_id=cue_subject_id)
        shared_entity_ids = tuple(sorted(set(candidate.entity_ids).intersection(other.entity_ids)))
        shared_features = tuple(
            sorted(set(candidate.retrieval_features).intersection(other.retrieval_features))
        )

        relationship_strength = _relationship_strength(
            same_semantic_slot=same_semantic_slot,
            same_predicate=same_predicate,
            same_subject=same_subject,
            shared_entity_ids=shared_entity_ids,
            shared_features=shared_features,
            config=config,
        )
        if relationship_strength <= 0.0:
            return None

        joint_cue_fit = min(candidate.effective_cue_fit, other.effective_cue_fit)
        strength = relationship_strength * joint_cue_fit
        if strength < config.minimum_strength:
            return None

        direction = competition_direction(candidate.effective_time, other.effective_time)
        return CompetitionEvidence(
            competitor_identity=other.identity,
            direction=direction,
            strength=strength,
            candidate_cue_fit=candidate.effective_cue_fit,
            competitor_cue_fit=other.effective_cue_fit,
            same_subject=same_subject,
            same_semantic_slot=same_semantic_slot,
            same_predicate=same_predicate,
            shared_entity_ids=shared_entity_ids,
            shared_features=shared_features,
            relationship_strength=relationship_strength,
            joint_cue_fit=joint_cue_fit,
        )


def _relationship_strength(
    *,
    same_semantic_slot: bool,
    same_predicate: bool,
    same_subject: bool,
    shared_entity_ids: tuple[str, ...],
    shared_features: tuple[str, ...],
    config: CompetitionConfig,
) -> float:
    if same_semantic_slot:
        return config.same_slot_strength
    if same_subject and same_predicate:
        return config.same_predicate_strength
    if not same_subject:
        return 0.0
    entity_component = config.entity_overlap_weight if shared_entity_ids else 0.0
    feature_component = config.feature_overlap_weight if shared_features else 0.0
    overlap = min(1.0, entity_component + feature_component)
    if overlap <= 0.0:
        return 0.0
    return overlap


@dataclass(frozen=True, slots=True)
class _CompetitionIndex:
    by_slot: Mapping[str, tuple[MemoryIdentity, ...]]
    by_subject_predicate: Mapping[tuple[str, str], tuple[MemoryIdentity, ...]]
    by_subject: Mapping[str, tuple[MemoryIdentity, ...]]

    def possible_competitors(self, profile: CompetitionProfile) -> tuple[MemoryIdentity, ...]:
        candidates: set[MemoryIdentity] = set()
        if profile.semantic_slot_key is not None:
            candidates.update(self.by_slot.get(profile.semantic_slot_key, ()))
        subject_key = profile.subject_entity_id or profile.subject_id
        if subject_key is not None and profile.predicate is not None:
            candidates.update(self.by_subject_predicate.get((subject_key, profile.predicate), ()))
        if subject_key is not None:
            candidates.update(self.by_subject.get(subject_key, ()))
        candidates.discard(profile.identity)
        return tuple(
            sorted(
                candidates,
                key=lambda identity: (identity.memory_kind.value, identity.memory_key),
            )
        )


def _build_index(profiles: Sequence[CompetitionProfile]) -> _CompetitionIndex:
    by_slot: dict[str, list[MemoryIdentity]] = {}
    by_subject_predicate: dict[tuple[str, str], list[MemoryIdentity]] = {}
    by_subject: dict[str, list[MemoryIdentity]] = {}
    for profile in profiles:
        identity = profile.identity
        if profile.semantic_slot_key is not None:
            by_slot.setdefault(profile.semantic_slot_key, []).append(identity)
        subject_key = profile.subject_entity_id or profile.subject_id
        if subject_key is not None and profile.predicate is not None:
            by_subject_predicate.setdefault((subject_key, profile.predicate), []).append(identity)
        if subject_key is not None:
            by_subject.setdefault(subject_key, []).append(identity)
    return _CompetitionIndex(
        by_slot={
            key: tuple(
                sorted(identities, key=lambda item: (item.memory_kind.value, item.memory_key))
            )
            for key, identities in by_slot.items()
        },
        by_subject_predicate={
            key: tuple(
                sorted(identities, key=lambda item: (item.memory_kind.value, item.memory_key))
            )
            for key, identities in by_subject_predicate.items()
        },
        by_subject={
            key: tuple(
                sorted(identities, key=lambda item: (item.memory_kind.value, item.memory_key))
            )
            for key, identities in by_subject.items()
        },
    )


def _bounded_competitors(
    evidence: Sequence[CompetitionEvidence],
    *,
    max_competitors: int,
) -> tuple[CompetitionEvidence, ...]:
    ordered = sorted(
        evidence,
        key=lambda item: (
            -item.strength,
            item.competitor_identity.memory_kind.value,
            item.competitor_identity.memory_key,
        ),
    )
    return tuple(ordered[:max_competitors])


def _summarize_competition(
    competitors: Sequence[CompetitionEvidence],
) -> CompetitionDiagnostics:
    proactive = sum(1 for item in competitors if item.direction is CompetitionDirection.PROACTIVE)
    retroactive = sum(
        1 for item in competitors if item.direction is CompetitionDirection.RETROACTIVE
    )
    co_temporal = sum(
        1 for item in competitors if item.direction is CompetitionDirection.CO_TEMPORAL
    )
    strongest = max((item.strength for item in competitors), default=0.0)
    return CompetitionDiagnostics(
        competitor_count=len(competitors),
        proactive_count=proactive,
        retroactive_count=retroactive,
        co_temporal_count=co_temporal,
        strongest_competition=strongest,
        competitors=tuple(competitors),
    )


def apply_inspection_competition(
    candidates: Sequence[RecallInspectionCandidate],
    *,
    config: CompetitionConfig,
    matcher: CompetitionMatcher,
    episode_slot_index: Mapping[str, str],
    cue_subject_id: str | None,
    retrieval_context_provided: bool,
) -> tuple[tuple[RecallInspectionCandidate, ...], CompetitionRunDiagnostics]:
    """Attach competition diagnostics to inspect candidates without changing scores."""
    if not config.enabled:
        return tuple(candidates), CompetitionRunDiagnostics(
            candidate_count=0,
            potential_competitor_pairs=0,
            evaluated_competitor_pairs=0,
            accepted_competition_pairs=0,
            maximum_competitors_for_candidate=0,
        )
    eligible = discrimination_set(candidates)
    profiles: list[CompetitionProfile] = []
    profile_by_identity: dict[MemoryIdentity, CompetitionProfile] = {}
    for candidate in eligible:
        profile = _profile_from_candidate(
            candidate,
            retrieval_context_provided=retrieval_context_provided,
            episode_slot_index=episode_slot_index,
        )
        if profile is None:
            continue
        profiles.append(profile)
        profile_by_identity[profile.identity] = profile

    index = _build_index(profiles)
    competition_by_identity: dict[MemoryIdentity, CompetitionDiagnostics] = {}
    potential_pairs = 0
    evaluated_pairs = 0
    accepted_pairs = 0
    maximum_competitors = 0

    for profile in profiles:
        possible = index.possible_competitors(profile)
        potential_pairs += len(possible)
        evidence: list[CompetitionEvidence] = []
        for other_identity in possible:
            other = profile_by_identity.get(other_identity)
            if other is None:
                continue
            evaluated_pairs += 1
            result = matcher.compare(
                profile,
                other,
                config=config,
                cue_subject_id=cue_subject_id,
            )
            if result is None:
                continue
            accepted_pairs += 1
            evidence.append(result)
        bounded = _bounded_competitors(
            evidence,
            max_competitors=config.max_competitors_per_candidate,
        )
        maximum_competitors = max(maximum_competitors, len(bounded))
        competition_by_identity[profile.identity] = _summarize_competition(bounded)

    updated: list[RecallInspectionCandidate] = []
    for candidate in candidates:
        if candidate.disposition not in _DISCRIMINATION_DISPOSITIONS:
            updated.append(candidate)
            continue
        identity = MemoryIdentity(
            memory_kind=candidate.memory_kind,
            memory_key=candidate.memory.memory_key,
        )
        diagnostics = competition_by_identity.get(identity)
        updated.append(replace(candidate, competition=diagnostics))

    run_diagnostics = CompetitionRunDiagnostics(
        candidate_count=len(profiles),
        potential_competitor_pairs=potential_pairs,
        evaluated_competitor_pairs=evaluated_pairs,
        accepted_competition_pairs=accepted_pairs,
        maximum_competitors_for_candidate=maximum_competitors,
    )
    return tuple(updated), run_diagnostics
