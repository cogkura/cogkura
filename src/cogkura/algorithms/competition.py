"""Cue-competition diagnostics and transient interference for declarative recall."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Protocol

from cogkura.algorithms.behavioral_competition import (
    BehavioralCompetitionPolicy,
)
from cogkura.algorithms.inhibition import build_inhibition_scope
from cogkura.algorithms.retrieval_features import canonical_content_features
from cogkura.models import (
    BehavioralQueryScope,
    CompetitionConfig,
    CompetitionDiagnostics,
    CompetitionDirection,
    CompetitionEvidence,
    CompetitionRunDiagnostics,
    InterferenceContribution,
    MemoryIdentity,
    MemoryKind,
    RecallInspectionCandidate,
    RecallInspectionDisposition,
    RecallResult,
    RetrievalCompetitionSnapshot,
    RetrievalDiagnostics,
    SemanticDerivationRelation,
    SemanticMemoryStatus,
    StoredEpisode,
    StoredSemanticMemory,
    TransientInterferenceDiagnostics,
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
    supported_semantic_keys: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CompetitionEvaluation:
    """Shared competition evaluation over recall results."""

    by_identity: Mapping[MemoryIdentity, CompetitionDiagnostics]
    run: CompetitionRunDiagnostics
    inhibition_candidates: Mapping[MemoryIdentity, tuple[RetrievalCompetitionSnapshot, ...]] = (
        field(default_factory=dict)
    )


def effective_memory_time(
    memory: StoredEpisode | StoredSemanticMemory,
    *,
    memory_kind: MemoryKind,
    episode_by_id: Mapping[str, StoredEpisode] | None = None,
) -> datetime:
    """Return the deterministic temporal point used for competition direction.

    Precedence:
    - Episode: ``started_at``
    - Semantic: ``valid_from`` else latest visible SUPPORT episode ``started_at``
      else ``last_supported_at`` else ``created_at``
    """
    if memory_kind is MemoryKind.EPISODE:
        if not isinstance(memory, StoredEpisode):
            raise TypeError("Episode effective time requires StoredEpisode.")
        return memory.started_at
    if not isinstance(memory, StoredSemanticMemory):
        raise TypeError("Semantic effective time requires StoredSemanticMemory.")
    if memory.valid_from is not None:
        return memory.valid_from
    if episode_by_id:
        support_times = [
            episode_by_id[derivation.episode_id].started_at
            for derivation in memory.derivations
            if derivation.relation is SemanticDerivationRelation.SUPPORTS
            and derivation.episode_id in episode_by_id
        ]
        if support_times:
            return max(support_times)
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


def subjects_compatible(
    left: CompetitionProfile,
    right: CompetitionProfile,
    *,
    cue_subject_id: str | None,
    cue_entity_ids: tuple[str, ...],
) -> bool:
    """Return True when memories share a compatible subject dimension."""
    if (
        left.subject_entity_id is not None
        and right.subject_entity_id is not None
        and left.subject_entity_id == right.subject_entity_id
    ):
        return True

    shared_query_anchor = tuple(
        sorted(set(left.entity_ids).intersection(right.entity_ids).intersection(cue_entity_ids))
    )
    if shared_query_anchor:
        return True

    if cue_subject_id:
        left_fact = left.subject_entity_id
        right_fact = right.subject_entity_id
        if left_fact is None or right_fact is None:
            return False
        return left_fact == cue_subject_id and right_fact == cue_subject_id

    if left.subject_entity_id is not None or right.subject_entity_id is not None:
        return False

    if (
        left.subject_id is not None
        and right.subject_id is not None
        and left.subject_id == right.subject_id
    ):
        return True
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


def _active_support_slot(
    diagnostics: RetrievalDiagnostics,
) -> tuple[str | None, tuple[str, ...]]:
    """Return the shared active support slot and the semantic keys that provide it."""
    active = [
        item
        for item in diagnostics.support_provenance
        if item.semantic_status is SemanticMemoryStatus.ACTIVE
    ]
    slots = {item.semantic_slot_key for item in active}
    if len(slots) != 1:
        return None, ()
    slot = next(iter(slots))
    keys = tuple(sorted({item.semantic_memory_key for item in active}))
    return slot, keys


def _lift_episode_cue_fit(
    profiles: list[CompetitionProfile],
) -> list[CompetitionProfile]:
    """Use supported semantic retrieval fit when an episode's own fit is weaker."""
    semantic_cue_fit = {
        profile.identity.memory_key: profile.cue_fit
        for profile in profiles
        if profile.memory_kind is MemoryKind.SEMANTIC
    }
    lifted: list[CompetitionProfile] = []
    for profile in profiles:
        if profile.memory_kind is not MemoryKind.EPISODE or not profile.supported_semantic_keys:
            lifted.append(profile)
            continue
        supported = [
            semantic_cue_fit[key]
            for key in profile.supported_semantic_keys
            if key in semantic_cue_fit
        ]
        if not supported:
            lifted.append(profile)
            continue
        lifted_fit = max(profile.cue_fit, max(supported))
        if lifted_fit == profile.cue_fit:
            lifted.append(profile)
            continue
        factor = 1.0 if profile.cue_fit == 0.0 else profile.effective_cue_fit / profile.cue_fit
        lifted.append(
            replace(
                profile,
                cue_fit=lifted_fit,
                effective_cue_fit=lifted_fit * factor,
            )
        )
    return lifted


def _profile_from_result(
    result: RecallResult,
    *,
    retrieval_context_provided: bool,
    episode_slot_index: Mapping[str, str],
    episode_by_id: Mapping[str, StoredEpisode] | None,
) -> CompetitionProfile | None:
    diagnostics = result.diagnostics
    if diagnostics is None:
        return None
    memory = result.memory
    subject_entity_id: str | None = None
    predicate: str | None = None
    semantic_slot_key = diagnostics.semantic_slot_key
    supported_semantic_keys: tuple[str, ...] = ()
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
        support_slot, supported_semantic_keys = _active_support_slot(diagnostics)
        if semantic_slot_key is None:
            semantic_slot_key = support_slot
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
            memory_kind=result.memory_kind,
            memory_key=memory.memory_key,
        ),
        memory_kind=result.memory_kind,
        effective_time=effective_memory_time(
            memory,
            memory_kind=result.memory_kind,
            episode_by_id=episode_by_id,
        ),
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
            memory_kind=result.memory_kind,
            diagnostics=diagnostics,
            episode_slot_index=episode_slot_index,
        ),
        supported_semantic_keys=supported_semantic_keys,
    )


def _profile_from_candidate(
    candidate: RecallInspectionCandidate,
    *,
    retrieval_context_provided: bool,
    episode_slot_index: Mapping[str, str],
    episode_by_id: Mapping[str, StoredEpisode] | None,
) -> CompetitionProfile | None:
    diagnostics = candidate.diagnostics
    if diagnostics is None:
        return None
    memory = candidate.memory
    subject_entity_id: str | None = None
    predicate: str | None = None
    semantic_slot_key = diagnostics.semantic_slot_key
    supported_semantic_keys: tuple[str, ...] = ()
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
        support_slot, supported_semantic_keys = _active_support_slot(diagnostics)
        if semantic_slot_key is None:
            semantic_slot_key = support_slot
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
        effective_time=effective_memory_time(
            memory,
            memory_kind=candidate.memory_kind,
            episode_by_id=episode_by_id,
        ),
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
        supported_semantic_keys=supported_semantic_keys,
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
        cue_entity_ids: tuple[str, ...],
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
        cue_entity_ids: tuple[str, ...],
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
        same_subject = subjects_compatible(
            candidate,
            other,
            cue_subject_id=cue_subject_id,
            cue_entity_ids=cue_entity_ids,
        )
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
    *,
    interference: TransientInterferenceDiagnostics | None = None,
) -> CompetitionDiagnostics:
    proactive = sum(1 for item in competitors if item.direction is CompetitionDirection.PROACTIVE)
    retroactive = sum(
        1 for item in competitors if item.direction is CompetitionDirection.RETROACTIVE
    )
    co_temporal = sum(
        1 for item in competitors if item.direction is CompetitionDirection.CO_TEMPORAL
    )
    strongest = max((item.strength for item in competitors), default=0.0)
    eligible_count = sum(
        1
        for item in competitors
        if item.behavioral_eligibility is not None and item.behavioral_eligibility.eligible
    )
    rejected_count = sum(
        1
        for item in competitors
        if item.behavioral_eligibility is not None and not item.behavioral_eligibility.eligible
    )
    return CompetitionDiagnostics(
        competitor_count=len(competitors),
        proactive_count=proactive,
        retroactive_count=retroactive,
        co_temporal_count=co_temporal,
        strongest_competition=strongest,
        competitors=tuple(competitors),
        interference=interference,
        behaviorally_eligible_competitor_count=eligible_count,
        behaviorally_rejected_competitor_count=rejected_count,
    )


def _presentation_score(activation: float, threshold: float) -> float:
    return 1.0 / (1.0 + math.exp(-(activation - threshold)))


def _noisy_or_pressure(pressures: Sequence[float]) -> float:
    product = 1.0
    for pressure in pressures:
        product *= 1.0 - pressure
    return 1.0 - product


def _compute_interference(
    profile: CompetitionProfile,
    competitors: Sequence[CompetitionEvidence],
    *,
    config: CompetitionConfig,
    accessibility_by_identity: Mapping[MemoryIdentity, float],
    retrieval_threshold: float,
) -> TransientInterferenceDiagnostics:
    contributions: list[InterferenceContribution] = []
    proactive_pressures: list[float] = []
    retroactive_pressures: list[float] = []

    for evidence in competitors:
        eligibility = evidence.behavioral_eligibility
        if eligibility is None or not eligibility.eligible:
            continue
        competitor_accessibility = accessibility_by_identity.get(
            evidence.competitor_identity,
            0.0,
        )
        score = _presentation_score(competitor_accessibility, retrieval_threshold)
        pressure = evidence.strength * score
        contributions.append(
            InterferenceContribution(
                competitor_identity=evidence.competitor_identity,
                direction=evidence.direction,
                strength=evidence.strength,
                competitor_accessibility=score,
                pressure=pressure,
            )
        )
        if evidence.direction is CompetitionDirection.PROACTIVE:
            proactive_pressures.append(pressure)
        elif evidence.direction is CompetitionDirection.RETROACTIVE:
            retroactive_pressures.append(pressure)

    proactive_pressure = _noisy_or_pressure(proactive_pressures)
    retroactive_pressure = _noisy_or_pressure(retroactive_pressures)
    proactive_penalty = -config.proactive_weight * proactive_pressure
    retroactive_penalty = -config.retroactive_weight * retroactive_pressure
    total_penalty = proactive_penalty + retroactive_penalty
    return TransientInterferenceDiagnostics(
        proactive_pressure=proactive_pressure,
        retroactive_pressure=retroactive_pressure,
        proactive_penalty=proactive_penalty,
        retroactive_penalty=retroactive_penalty,
        total_penalty=total_penalty,
        contributions=tuple(contributions),
    )


def _attach_behavioral_eligibility(
    evidence: CompetitionEvidence,
    *,
    candidate: CompetitionProfile,
    competitor: CompetitionProfile,
    policy: BehavioralCompetitionPolicy,
    query_scope: BehavioralQueryScope,
    config: CompetitionConfig,
) -> CompetitionEvidence:
    eligibility = policy.evaluate(
        candidate=candidate,
        competitor=competitor,
        evidence=evidence,
        query_scope=query_scope,
        config=config,
    )
    return replace(evidence, behavioral_eligibility=eligibility)


def evaluate_competition(
    results: Sequence[RecallResult],
    *,
    config: CompetitionConfig,
    matcher: CompetitionMatcher,
    behavioral_policy: BehavioralCompetitionPolicy,
    query_scope: BehavioralQueryScope,
    cue_subject_id: str | None,
    cue_entity_ids: tuple[str, ...],
    episode_slot_index: Mapping[str, str],
    episode_by_id: Mapping[str, StoredEpisode] | None,
    retrieval_context_provided: bool,
    retrieval_threshold: float,
    capture_inhibition: bool = False,
    retrieval_evaluated_at: datetime | None = None,
) -> CompetitionEvaluation:
    """Evaluate pairwise competition over recall results."""
    if not config.enabled:
        return CompetitionEvaluation(
            by_identity={},
            run=CompetitionRunDiagnostics(
                candidate_count=0,
                potential_competitor_pairs=0,
                evaluated_competitor_pairs=0,
                accepted_competition_pairs=0,
                maximum_competitors_for_candidate=0,
            ),
        )

    profiles: list[CompetitionProfile] = []
    profile_by_identity: dict[MemoryIdentity, CompetitionProfile] = {}
    accessibility_by_identity: dict[MemoryIdentity, float] = {}
    for result in results:
        profile = _profile_from_result(
            result,
            retrieval_context_provided=retrieval_context_provided,
            episode_slot_index=episode_slot_index,
            episode_by_id=episode_by_id,
        )
        if profile is None:
            continue
        profiles.append(profile)
        profile_by_identity[profile.identity] = profile
        accessibility_by_identity[profile.identity] = result.activation

    profiles = _lift_episode_cue_fit(profiles)
    profile_by_identity = {profile.identity: profile for profile in profiles}

    index = _build_index(profiles)
    competition_by_identity: dict[MemoryIdentity, CompetitionDiagnostics] = {}
    potential_pairs = 0
    evaluated_pairs = 0
    accepted_pairs = 0
    maximum_competitors = 0
    behaviorally_eligible_pairs = 0
    behaviorally_rejected_pairs = 0
    rejected_reason_counts: dict[str, int] = {}
    inhibition_by_identity: dict[MemoryIdentity, tuple[RetrievalCompetitionSnapshot, ...]] = {}

    for profile in profiles:
        possible = index.possible_competitors(profile)
        potential_pairs += len(possible)
        evidence: list[CompetitionEvidence] = []
        for other_identity in possible:
            other = profile_by_identity.get(other_identity)
            if other is None:
                continue
            evaluated_pairs += 1
            comparison = matcher.compare(
                profile,
                other,
                config=config,
                cue_subject_id=cue_subject_id,
                cue_entity_ids=cue_entity_ids,
            )
            if comparison is None:
                continue
            accepted_pairs += 1
            enriched = _attach_behavioral_eligibility(
                comparison,
                candidate=profile,
                competitor=other,
                policy=behavioral_policy,
                query_scope=query_scope,
                config=config,
            )
            eligibility = enriched.behavioral_eligibility
            if eligibility is not None:
                if eligibility.eligible:
                    behaviorally_eligible_pairs += 1
                else:
                    behaviorally_rejected_pairs += 1
                    reason_key = eligibility.reason.value
                    rejected_reason_counts[reason_key] = (
                        rejected_reason_counts.get(reason_key, 0) + 1
                    )
            evidence.append(enriched)
        bounded = _bounded_competitors(
            evidence,
            max_competitors=config.max_competitors_per_candidate,
        )
        maximum_competitors = max(maximum_competitors, len(bounded))
        interference = None
        if config.apply_interference:
            interference = _compute_interference(
                profile,
                bounded,
                config=config,
                accessibility_by_identity=accessibility_by_identity,
                retrieval_threshold=retrieval_threshold,
            )
        competition_by_identity[profile.identity] = _summarize_competition(
            bounded,
            interference=interference,
        )
        if capture_inhibition and retrieval_evaluated_at is not None:
            inhibition_by_identity[profile.identity] = _inhibition_snapshots(
                profile,
                bounded,
                profile_by_identity=profile_by_identity,
                accessibility_by_identity=accessibility_by_identity,
                retrieval_threshold=retrieval_threshold,
                retrieval_evaluated_at=retrieval_evaluated_at,
            )

    return CompetitionEvaluation(
        by_identity=competition_by_identity,
        run=CompetitionRunDiagnostics(
            candidate_count=len(profiles),
            potential_competitor_pairs=potential_pairs,
            evaluated_competitor_pairs=evaluated_pairs,
            accepted_competition_pairs=accepted_pairs,
            maximum_competitors_for_candidate=maximum_competitors,
            behaviorally_eligible_pairs=behaviorally_eligible_pairs,
            behaviorally_rejected_pairs=behaviorally_rejected_pairs,
            rejected_by_reason=dict(sorted(rejected_reason_counts.items())),
        ),
        inhibition_candidates=inhibition_by_identity,
    )


def _inhibition_snapshots(
    profile: CompetitionProfile,
    competitors: Sequence[CompetitionEvidence],
    *,
    profile_by_identity: Mapping[MemoryIdentity, CompetitionProfile],
    accessibility_by_identity: Mapping[MemoryIdentity, float],
    retrieval_threshold: float,
    retrieval_evaluated_at: datetime,
) -> tuple[RetrievalCompetitionSnapshot, ...]:
    snapshots: list[RetrievalCompetitionSnapshot] = []
    for evidence in competitors:
        eligibility = evidence.behavioral_eligibility
        if eligibility is None or not eligibility.scope_eligible:
            continue
        if eligibility.structural_anchor is None:
            continue
        competitor = profile_by_identity.get(evidence.competitor_identity)
        if competitor is None:
            continue
        activation = accessibility_by_identity.get(evidence.competitor_identity, 0.0)
        snapshots.append(
            RetrievalCompetitionSnapshot(
                competitor_identity=evidence.competitor_identity,
                direction=evidence.direction,
                competition_strength=evidence.strength,
                competitor_accessibility=_presentation_score(activation, retrieval_threshold),
                scope_eligible=True,
                scope=build_inhibition_scope(
                    eligibility,
                    subject_entity_id=profile.subject_entity_id,
                    predicate=profile.predicate,
                    semantic_slot_key=profile.semantic_slot_key,
                ),
                retrieval_evaluated_at=retrieval_evaluated_at,
                candidate_lineage_group=profile.lineage_group,
                competitor_lineage_group=competitor.lineage_group,
            )
        )
    return tuple(snapshots)


def _attach_inhibition_snapshots(
    results: Sequence[RecallResult],
    snapshots_by_identity: Mapping[MemoryIdentity, tuple[RetrievalCompetitionSnapshot, ...]],
) -> list[RecallResult]:
    if not snapshots_by_identity:
        return list(results)
    updated: list[RecallResult] = []
    for result in results:
        identity = _result_identity(result)
        snapshots = snapshots_by_identity.get(identity)
        if not snapshots or result.diagnostics is None:
            updated.append(result)
            continue
        updated.append(
            replace(
                result,
                diagnostics=replace(result.diagnostics, inhibition_candidates=snapshots),
            )
        )
    return updated


def _result_identity(result: RecallResult) -> MemoryIdentity:
    return MemoryIdentity(memory_kind=result.memory_kind, memory_key=result.memory.memory_key)


def _freeze_pre_interference(
    result: RecallResult,
) -> RecallResult:
    diagnostics = result.diagnostics
    if diagnostics is None:
        return result
    frozen = replace(
        diagnostics,
        activation_before_interference=result.activation,
        rank_activation_before_interference=diagnostics.rank_activation,
    )
    return replace(result, diagnostics=frozen)


def _apply_interference_to_result(
    result: RecallResult,
    *,
    penalty: float,
    retrieval_threshold: float,
    latency_factor: float,
    latency_exponent: float,
) -> RecallResult:
    activation_before = result.activation
    diagnostics = result.diagnostics
    rank_before = diagnostics.rank_activation if diagnostics is not None else activation_before
    activation_after = activation_before + penalty
    rank_after = rank_before + penalty
    score = _presentation_score(activation_after, retrieval_threshold)
    latency_seconds = latency_factor * math.exp(-latency_exponent * activation_after)
    components = replace(
        result.components,
        interference=penalty,
        total=activation_after,
    )
    updated_diagnostics = diagnostics
    if diagnostics is not None:
        crossed = (
            activation_before >= retrieval_threshold and activation_after < retrieval_threshold
        )
        updated_diagnostics = replace(
            diagnostics,
            rank_activation=rank_after,
            crossed_activation_threshold_due_to_interference=crossed,
        )
    return replace(
        result,
        activation=activation_after,
        score=score,
        latency_seconds=latency_seconds,
        components=components,
        diagnostics=updated_diagnostics,
    )


def apply_competition_pipeline(
    scored: Sequence[RecallResult],
    rank_by_identity: Mapping[MemoryIdentity, float],
    *,
    config: CompetitionConfig,
    matcher: CompetitionMatcher,
    behavioral_policy: BehavioralCompetitionPolicy,
    query_scope: BehavioralQueryScope,
    cue_subject_id: str | None,
    cue_entity_ids: tuple[str, ...],
    episode_slot_index: Mapping[str, str],
    episode_by_id: Mapping[str, StoredEpisode] | None,
    retrieval_context_provided: bool,
    retrieval_threshold: float,
    latency_factor: float,
    latency_exponent: float,
    capture_inhibition: bool = False,
    retrieval_evaluated_at: datetime | None = None,
) -> tuple[list[RecallResult], dict[MemoryIdentity, float], CompetitionEvaluation | None]:
    """Freeze pre-interference state, evaluate competition, and optionally apply penalties."""
    if not config.enabled:
        return list(scored), dict(rank_by_identity), None

    frozen_scored = [_freeze_pre_interference(result) for result in scored]
    evaluation = evaluate_competition(
        frozen_scored,
        config=config,
        matcher=matcher,
        behavioral_policy=behavioral_policy,
        query_scope=query_scope,
        cue_subject_id=cue_subject_id,
        cue_entity_ids=cue_entity_ids,
        episode_slot_index=episode_slot_index,
        episode_by_id=episode_by_id,
        retrieval_context_provided=retrieval_context_provided,
        retrieval_threshold=retrieval_threshold,
        capture_inhibition=capture_inhibition,
        retrieval_evaluated_at=retrieval_evaluated_at,
    )
    frozen_scored = _attach_inhibition_snapshots(
        frozen_scored,
        evaluation.inhibition_candidates,
    )

    if not config.apply_interference:
        return frozen_scored, dict(rank_by_identity), evaluation

    updated_rank = dict(rank_by_identity)
    updated_scored: list[RecallResult] = []
    for result in frozen_scored:
        identity = _result_identity(result)
        competition = evaluation.by_identity.get(identity)
        penalty = 0.0
        if competition is not None and competition.interference is not None:
            penalty = competition.interference.total_penalty
        if penalty != 0.0:
            updated = _apply_interference_to_result(
                result,
                penalty=penalty,
                retrieval_threshold=retrieval_threshold,
                latency_factor=latency_factor,
                latency_exponent=latency_exponent,
            )
            updated_rank[identity] = updated_rank[identity] + penalty
            updated_scored.append(updated)
        else:
            updated_scored.append(result)
    return updated_scored, updated_rank, evaluation


@dataclass(frozen=True, slots=True)
class _InterferenceRankInfo:
    rank_before_interference: int
    rank_after_interference: int
    interference_rank_delta: int


def assign_interference_ranks(
    candidates: Sequence[RecallInspectionCandidate],
) -> dict[str, _InterferenceRankInfo]:
    """Assign pre/post interference ranks for inspect candidates."""
    eligible = [
        candidate
        for candidate in candidates
        if candidate.disposition in _DISCRIMINATION_DISPOSITIONS
    ]

    def _rank_before_interference(candidate: RecallInspectionCandidate) -> float:
        diagnostics = candidate.diagnostics
        if diagnostics is not None and diagnostics.rank_activation_before_interference is not None:
            return diagnostics.rank_activation_before_interference
        return candidate.activation

    before_sorted = sorted(
        eligible,
        key=lambda item: (
            -_rank_before_interference(item),
            item.memory_kind.value,
            item.memory.memory_key,
        ),
    )
    after_sorted = sorted(
        eligible,
        key=lambda item: (
            -(item.diagnostics.rank_activation if item.diagnostics else item.activation),
            item.memory_kind.value,
            item.memory.memory_key,
        ),
    )
    before_rank = {
        candidate.memory.memory_key: index + 1 for index, candidate in enumerate(before_sorted)
    }
    after_rank = {
        candidate.memory.memory_key: index + 1 for index, candidate in enumerate(after_sorted)
    }
    ranks: dict[str, _InterferenceRankInfo] = {}
    for candidate in eligible:
        key = candidate.memory.memory_key
        before = before_rank[key]
        after = after_rank[key]
        ranks[key] = _InterferenceRankInfo(
            rank_before_interference=before,
            rank_after_interference=after,
            interference_rank_delta=after - before,
        )
    return ranks


def apply_inspection_interference_attribution(
    candidates: Sequence[RecallInspectionCandidate],
) -> tuple[RecallInspectionCandidate, ...]:
    """Attach interference rank deltas to inspect candidates."""
    ranks = assign_interference_ranks(candidates)
    updated: list[RecallInspectionCandidate] = []
    for candidate in candidates:
        rank_info = ranks.get(candidate.memory.memory_key)
        if rank_info is None:
            updated.append(candidate)
            continue
        updated.append(
            replace(
                candidate,
                rank_before_interference=rank_info.rank_before_interference,
                rank_after_interference=rank_info.rank_after_interference,
                interference_rank_delta=rank_info.interference_rank_delta,
            )
        )
    return tuple(updated)


def apply_inspection_competition(
    candidates: Sequence[RecallInspectionCandidate],
    *,
    config: CompetitionConfig,
    matcher: CompetitionMatcher,
    behavioral_policy: BehavioralCompetitionPolicy,
    query_scope: BehavioralQueryScope,
    episode_slot_index: Mapping[str, str],
    cue_subject_id: str | None,
    cue_entity_ids: tuple[str, ...],
    retrieval_context_provided: bool,
    episode_by_id: Mapping[str, StoredEpisode] | None = None,
    evaluation: CompetitionEvaluation | None = None,
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

    if evaluation is None:
        recall_results = [
            RecallResult(
                memory_kind=candidate.memory_kind,
                memory=candidate.memory,
                activation=candidate.activation,
                score=candidate.score,
                latency_seconds=0.0,
                components=candidate.components,
                reason=candidate.reason or "",
                diagnostics=candidate.diagnostics,
            )
            for candidate in candidates
            if candidate.disposition in _DISCRIMINATION_DISPOSITIONS
        ]
        evaluation = evaluate_competition(
            recall_results,
            config=config,
            matcher=matcher,
            behavioral_policy=behavioral_policy,
            query_scope=query_scope,
            cue_subject_id=cue_subject_id,
            cue_entity_ids=cue_entity_ids,
            episode_slot_index=episode_slot_index,
            episode_by_id=episode_by_id,
            retrieval_context_provided=retrieval_context_provided,
            retrieval_threshold=candidates[0].retrieval_threshold if candidates else -3.0,
        )

    updated: list[RecallInspectionCandidate] = []
    for candidate in candidates:
        if candidate.disposition not in _DISCRIMINATION_DISPOSITIONS:
            updated.append(candidate)
            continue
        identity = MemoryIdentity(
            memory_kind=candidate.memory_kind,
            memory_key=candidate.memory.memory_key,
        )
        diagnostics = evaluation.by_identity.get(identity)
        updated.append(replace(candidate, competition=diagnostics))

    return tuple(updated), evaluation.run
