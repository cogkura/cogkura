"""Behavioural eligibility policy for cue-competition interference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from cogkura.algorithms.retrieval_features import (
    canonical_content_features,
    distinctive_content_features,
    predicate_content_features,
)
from cogkura.models import (
    BehavioralCompetitionEligibility,
    BehavioralEligibilityReason,
    BehavioralQueryScope,
    BehavioralStructuralAnchor,
    CompetitionConfig,
    CompetitionDirection,
    CompetitionEvidence,
    RetrievalCue,
)

if TYPE_CHECKING:
    from cogkura.algorithms.competition import CompetitionProfile


def build_behavioral_query_scope(
    cue: RetrievalCue,
    *,
    exclude_tokens: frozenset[str] = frozenset(),
) -> BehavioralQueryScope:
    """Build retrieval-local query scope from an existing cue."""
    features: set[str] = set()
    if cue.text and cue.text.strip():
        features.update(
            distinctive_content_features(cue.text, exclude_tokens=exclude_tokens),
        )
    if cue.predicate is not None and cue.predicate.strip():
        features.update(predicate_content_features(cue.predicate))
    return BehavioralQueryScope(
        subject_id=cue.subject_id,
        entity_ids=tuple(sorted(set(cue.entity_ids))),
        features=tuple(sorted(features)),
        predicate=cue.predicate,
    )


def _same_lineage(left: CompetitionProfile, right: CompetitionProfile) -> bool:
    if left.lineage_group is None or right.lineage_group is None:
        return False
    return left.lineage_group == right.lineage_group


def _same_fact_subject(left: CompetitionProfile, right: CompetitionProfile) -> bool:
    return (
        left.subject_entity_id is not None
        and right.subject_entity_id is not None
        and left.subject_entity_id == right.subject_entity_id
    )


def _anchor_tokens(
    candidate: CompetitionProfile,
    competitor: CompetitionProfile,
    query_scope: BehavioralQueryScope,
) -> frozenset[str]:
    anchors: set[str] = set()
    for entity_id in (
        *query_scope.entity_ids,
        *candidate.entity_ids,
        *competitor.entity_ids,
    ):
        anchors.update(canonical_content_features(entity_id))
    for subject in (
        candidate.subject_entity_id,
        competitor.subject_entity_id,
        query_scope.subject_id,
    ):
        if subject is not None and subject.strip():
            anchors.update(canonical_content_features(subject))
    return frozenset(anchors)


def _fact_subjects_match_cue(
    candidate: CompetitionProfile,
    competitor: CompetitionProfile,
    cue_subject_id: str,
) -> bool:
    left = candidate.subject_entity_id
    right = competitor.subject_entity_id
    if left is None or right is None:
        return False
    return left == cue_subject_id and right == cue_subject_id


def _query_scope_anchor(
    candidate: CompetitionProfile,
    competitor: CompetitionProfile,
    query_scope: BehavioralQueryScope,
) -> tuple[bool, tuple[str, ...]]:
    if query_scope.entity_ids:
        shared = tuple(
            sorted(
                set(candidate.entity_ids)
                .intersection(competitor.entity_ids)
                .intersection(query_scope.entity_ids)
            )
        )
        if not shared:
            return False, ()
        if query_scope.subject_id is not None:
            if not _fact_subjects_match_cue(candidate, competitor, query_scope.subject_id):
                return False, ()
        return True, shared

    cue_features = frozenset(query_scope.features)

    if query_scope.subject_id is not None:
        if not _fact_subjects_match_cue(candidate, competitor, query_scope.subject_id):
            return False, ()
        subject = candidate.subject_entity_id
        if subject is not None and canonical_content_features(subject).issubset(cue_features):
            return True, (subject,)

    shared_entities = sorted(set(candidate.entity_ids).intersection(competitor.entity_ids))
    for entity_id in shared_entities:
        entity_features = canonical_content_features(entity_id)
        if entity_features and entity_features.issubset(cue_features):
            return True, (entity_id,)

    if _same_fact_subject(candidate, competitor):
        subject = candidate.subject_entity_id
        assert subject is not None
        if canonical_content_features(subject).issubset(cue_features):
            return True, (subject,)

    return False, ()


class BehavioralCompetitionPolicy(Protocol):
    """Evaluate behavioural eligibility for accepted competition evidence."""

    def evaluate(
        self,
        *,
        candidate: CompetitionProfile,
        competitor: CompetitionProfile,
        evidence: CompetitionEvidence,
        query_scope: BehavioralQueryScope,
        config: CompetitionConfig,
    ) -> BehavioralCompetitionEligibility:
        """Return behavioural eligibility for a directed competition pair."""
        ...


@dataclass(frozen=True, slots=True)
class DeterministicBehavioralCompetitionPolicy:
    """Deterministic behavioural eligibility for transient interference."""

    def evaluate(
        self,
        *,
        candidate: CompetitionProfile,
        competitor: CompetitionProfile,
        evidence: CompetitionEvidence,
        query_scope: BehavioralQueryScope,
        config: CompetitionConfig,
    ) -> BehavioralCompetitionEligibility:
        same_fact_subject = _same_fact_subject(candidate, competitor)
        anchor_tokens = _anchor_tokens(candidate, competitor, query_scope)
        has_anchor, anchor_entities = _query_scope_anchor(candidate, competitor, query_scope)
        shared_query_entity_ids = anchor_entities
        shared_query_features = tuple(
            sorted(
                set(candidate.retrieval_features)
                .intersection(competitor.retrieval_features)
                .intersection(query_scope.features)
                - anchor_tokens
            )
        )
        memory_shared_entities = tuple(
            sorted(set(candidate.entity_ids).intersection(competitor.entity_ids))
        )

        def _result(
            *,
            scope_eligible: bool,
            reason: BehavioralEligibilityReason,
            structural_anchor: BehavioralStructuralAnchor | None,
            entity_ids: tuple[str, ...] = shared_query_entity_ids,
            features: tuple[str, ...] = shared_query_features,
        ) -> BehavioralCompetitionEligibility:
            temporal = evidence.direction is not CompetitionDirection.CO_TEMPORAL
            eligible = scope_eligible and temporal
            if scope_eligible and not temporal:
                reason = BehavioralEligibilityReason.NON_BEHAVIORAL_DIRECTION
            return BehavioralCompetitionEligibility(
                eligible=eligible,
                reason=reason,
                competition_strength=evidence.strength,
                candidate_cue_fit=evidence.candidate_cue_fit,
                competitor_cue_fit=evidence.competitor_cue_fit,
                same_semantic_slot=evidence.same_semantic_slot,
                same_fact_subject=same_fact_subject,
                same_predicate=evidence.same_predicate,
                shared_query_entity_ids=entity_ids,
                shared_query_features=features,
                structural_anchor=structural_anchor,
                scope_eligible=scope_eligible,
            )

        if _same_lineage(candidate, competitor):
            return _result(
                scope_eligible=False,
                reason=BehavioralEligibilityReason.SAME_LINEAGE,
                structural_anchor=None,
            )

        structured = evidence.same_semantic_slot or (same_fact_subject and evidence.same_predicate)
        gated_strength = evidence.relationship_strength if structured else evidence.strength
        if gated_strength < config.minimum_behavioral_strength:
            return _result(
                scope_eligible=False,
                reason=BehavioralEligibilityReason.COMPETITION_TOO_WEAK,
                structural_anchor=None,
            )

        if evidence.candidate_cue_fit < config.minimum_behavioral_cue_fit:
            return _result(
                scope_eligible=False,
                reason=BehavioralEligibilityReason.CANDIDATE_CUE_FIT_TOO_WEAK,
                structural_anchor=None,
            )

        if evidence.competitor_cue_fit < config.minimum_behavioral_cue_fit:
            return _result(
                scope_eligible=False,
                reason=BehavioralEligibilityReason.COMPETITOR_CUE_FIT_TOO_WEAK,
                structural_anchor=None,
            )

        if evidence.same_semantic_slot:
            return _result(
                scope_eligible=True,
                reason=BehavioralEligibilityReason.SAME_SEMANTIC_SLOT,
                structural_anchor=BehavioralStructuralAnchor.SEMANTIC_SLOT,
            )

        if same_fact_subject and evidence.same_predicate:
            return _result(
                scope_eligible=True,
                reason=BehavioralEligibilityReason.SAME_SUBJECT_PREDICATE,
                structural_anchor=BehavioralStructuralAnchor.SUBJECT_PREDICATE,
            )

        if has_anchor and shared_query_features:
            return _result(
                scope_eligible=True,
                reason=BehavioralEligibilityReason.QUERY_ANCHORED_COMPETITION,
                structural_anchor=BehavioralStructuralAnchor.QUERY_SCOPE,
            )

        if query_scope.entity_ids and not has_anchor:
            return _result(
                scope_eligible=False,
                reason=BehavioralEligibilityReason.NO_QUERY_SCOPE_ANCHOR,
                structural_anchor=None,
            )

        if has_anchor and not shared_query_features:
            return _result(
                scope_eligible=False,
                reason=BehavioralEligibilityReason.NO_SHARED_QUERY_FEATURE,
                structural_anchor=None,
            )

        if memory_shared_entities:
            return _result(
                scope_eligible=False,
                reason=BehavioralEligibilityReason.ENTITY_OVERLAP_ONLY,
                structural_anchor=None,
            )

        shared_memory_features = tuple(
            sorted(
                set(candidate.retrieval_features).intersection(competitor.retrieval_features)
                - anchor_tokens
            )
        )
        if shared_memory_features:
            return _result(
                scope_eligible=False,
                reason=BehavioralEligibilityReason.FEATURE_OVERLAP_ONLY,
                structural_anchor=None,
                features=shared_query_features,
            )

        if evidence.same_subject and not same_fact_subject:
            return _result(
                scope_eligible=False,
                reason=BehavioralEligibilityReason.BROAD_SUBJECT_ONLY,
                structural_anchor=None,
            )

        return _result(
            scope_eligible=False,
            reason=BehavioralEligibilityReason.NO_QUERY_SCOPE_ANCHOR,
            structural_anchor=None,
        )
