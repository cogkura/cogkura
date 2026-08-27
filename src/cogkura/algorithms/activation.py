"""ACT-R declarative activation for episodic and semantic memories."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from cogkura.algorithms.cognitive_traces import (
    activation_reference_traces_for_candidate,
    derive_episode_cognitive_traces,
    derive_semantic_cognitive_traces,
)
from cogkura.algorithms.spreading import (
    DeterministicSpreadingActivator,
    SpreadingActivator,
    SpreadingMetadata,
)
from cogkura.exceptions import ValidationError
from cogkura.models import (
    ActivationCandidate,
    ActivationComponents,
    ActivationConfig,
    ActivationReferenceTrace,
    LearnedAssociation,
    MemoryIdentity,
    MemoryKind,
    RecallInspectionCandidate,
    RecallInspectionDisposition,
    RecallInspectionResult,
    RecallResult,
    RetrievalCue,
    RetrievalDiagnostics,
    RetrievalEligibility,
    SemanticDerivationRelation,
    SemanticMemoryStatus,
    SlotFitSource,
    StoredEpisode,
    StoredSemanticMemory,
    SupportProvenance,
)

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
_WHITESPACE_PATTERN = re.compile(r"\s+")

_PARTIAL_MATCH_WEIGHTS = {
    "subject": 1.5,
    "entities": 1.5,
    "predicate": 1.0,
    "object": 1.0,
    "qualifiers": 1.0,
    "text": 1.0,
}


@dataclass(frozen=True, slots=True)
class _TextMatchMetrics:
    coverage: float
    cue_fit: float


@dataclass(frozen=True, slots=True)
class _SlotFitSelection:
    slot_fit: float | None
    support_provenance: tuple[SupportProvenance, ...] = ()
    selected_support_revision_key: str | None = None


@dataclass(frozen=True, slots=True)
class _SemanticAdmissionOutcome:
    admitted: frozenset[MemoryIdentity]
    matched_semantics: frozenset[MemoryIdentity]
    below_soft_floor: frozenset[MemoryIdentity]
    capped_out: frozenset[MemoryIdentity]


class TemporalRetrievalMode(StrEnum):
    """Internal temporal frame for retrieval policy, admission, and ranking."""

    NEUTRAL = "neutral"
    CURRENT = "current"
    HISTORICAL = "historical"


class DeclarativeActivator(Protocol):
    """Ranks durable memories by declarative activation."""

    def rank(
        self,
        *,
        candidates: Sequence[ActivationCandidate],
        cue: RetrievalCue,
        references: Mapping[MemoryIdentity, Sequence[ActivationReferenceTrace]],
        as_of: datetime,
        config: ActivationConfig,
        limit: int,
        learned_associations: Sequence[LearnedAssociation] = (),
        episode_support_index: Mapping[str, frozenset[SemanticMemoryStatus]] | None = None,
        valid_at: datetime | None = None,
        episode_slot_index: Mapping[str, str] | None = None,
    ) -> list[RecallResult]:
        """Rank candidates by activation and return those above threshold."""


class InspectableDeclarativeActivator(DeclarativeActivator, Protocol):
    """Declarative activator that can evaluate all candidates for inspection."""

    def inspect(
        self,
        *,
        candidates: Sequence[ActivationCandidate],
        cue: RetrievalCue,
        references: Mapping[MemoryIdentity, Sequence[ActivationReferenceTrace]],
        as_of: datetime,
        config: ActivationConfig,
        limit: int,
        learned_associations: Sequence[LearnedAssociation] = (),
        episode_support_index: Mapping[str, frozenset[SemanticMemoryStatus]] | None = None,
        valid_at: datetime | None = None,
        episode_slot_index: Mapping[str, str] | None = None,
    ) -> RecallInspectionResult:
        """Evaluate all candidates and return inspection dispositions."""


def logsumexp(values: Sequence[float]) -> float:
    """Numerically stable log-sum-exp."""
    if not values:
        raise ValidationError("logsumexp requires at least one value.")
    max_value = max(values)
    if math.isinf(max_value) and max_value < 0:
        return max_value
    total = sum(math.exp(value - max_value) for value in values)
    return max_value + math.log(total)


def calculate_base_level(
    reference_traces: Sequence[ActivationReferenceTrace],
    *,
    as_of: datetime,
    decay: float,
    constant: float,
    time_unit_seconds: float,
    minimum_elapsed_seconds: float,
) -> float:
    """Compute ACT-R base-level activation from weighted reference traces."""
    terms: list[float] = []
    for trace in reference_traces:
        elapsed_seconds = (as_of - trace.referenced_at).total_seconds()
        if elapsed_seconds < 0:
            raise ValidationError("Memory reference cannot occur after the activation timestamp.")
        elapsed_units = max(elapsed_seconds, minimum_elapsed_seconds) / time_unit_seconds
        terms.append(math.log(trace.weight) - decay * math.log(elapsed_units))
    return logsumexp(terms) + constant


def activation_candidate_from_episode(
    episode: StoredEpisode,
    *,
    support_provenance: Sequence[SupportProvenance] = (),
) -> ActivationCandidate:
    """Adapt a stored episode for declarative activation."""
    entity_ids = tuple(sorted({entity.entity_id for entity in episode.entities}))
    return ActivationCandidate(
        memory_kind=MemoryKind.EPISODE,
        memory_key=episode.memory_key,
        created_at=episode.created_at,
        text=episode.statement,
        subject_id=episode.subject_id,
        entity_ids=entity_ids,
        predicate=None,
        object_value=None,
        qualifiers={},
        memory=episode,
        importance=episode.importance,
        support_provenance=tuple(support_provenance),
        cognitive_traces=derive_episode_cognitive_traces(episode),
    )


def activation_candidate_from_semantic(
    memory: StoredSemanticMemory,
    *,
    episode_by_id: Mapping[str, StoredEpisode] | None = None,
) -> ActivationCandidate:
    """Adapt a stored semantic memory for declarative activation."""
    entity_ids = {entity.entity_id for entity in memory.entities}
    if memory.subject_entity_id:
        entity_ids.add(memory.subject_entity_id)
    if memory.object_entity_id:
        entity_ids.add(memory.object_entity_id)
    episodes = episode_by_id or {}
    return ActivationCandidate(
        memory_kind=MemoryKind.SEMANTIC,
        memory_key=memory.memory_key,
        created_at=memory.created_at,
        text=memory.statement,
        subject_id=memory.subject_id,
        entity_ids=tuple(sorted(entity_ids)),
        predicate=memory.predicate,
        object_value=memory.object_value,
        qualifiers=memory.qualifiers,
        memory=memory,
        importance=memory.importance,
        slot_key=memory.slot_key,
        semantic_status=memory.status,
        last_supported_at=memory.last_supported_at,
        cognitive_traces=derive_semantic_cognitive_traces(memory, episodes),
    )


def build_episode_support_index(
    semantics: Sequence[StoredSemanticMemory],
) -> dict[str, frozenset[SemanticMemoryStatus]]:
    """Map episode ids to semantic slot statuses they support."""
    index: dict[str, set[SemanticMemoryStatus]] = defaultdict(set)
    for memory in semantics:
        for derivation in memory.derivations:
            if derivation.relation is SemanticDerivationRelation.SUPPORTS:
                index[derivation.episode_id].add(memory.status)
    return {episode_id: frozenset(statuses) for episode_id, statuses in index.items()}


def build_episode_slot_index(
    semantics: Sequence[StoredSemanticMemory],
) -> dict[str, str]:
    """Map episode ids to semantic slot keys they support."""
    index: dict[str, str] = {}
    for memory in semantics:
        for derivation in memory.derivations:
            if derivation.relation is SemanticDerivationRelation.SUPPORTS:
                index[derivation.episode_id] = memory.slot_key
    return index


def build_episode_support_provenance_index(
    semantics: Sequence[StoredSemanticMemory],
) -> dict[str, tuple[SupportProvenance, ...]]:
    """Map episode ids to derivation-backed semantic SUPPORT provenance."""
    by_episode: dict[str, list[SupportProvenance]] = defaultdict(list)
    for memory in semantics:
        for derivation in memory.derivations:
            if derivation.relation is not SemanticDerivationRelation.SUPPORTS:
                continue
            by_episode[derivation.episode_id].append(
                SupportProvenance(
                    semantic_memory_key=memory.memory_key,
                    semantic_revision_key=memory.revision_key,
                    semantic_slot_key=memory.slot_key,
                    semantic_status=memory.status,
                    contribution_score=derivation.contribution_score,
                    semantic_valid_from=memory.valid_from,
                    semantic_valid_until=memory.valid_until,
                )
            )
    return {
        episode_id: tuple(
            sorted(
                provenance,
                key=lambda item: (
                    item.semantic_slot_key,
                    item.semantic_memory_key,
                    item.semantic_revision_key,
                ),
            )
        )
        for episode_id, provenance in by_episode.items()
    }


def build_episode_slot_index_from_results(
    results: Sequence[RecallResult],
) -> dict[str, str]:
    """Map episode ids to slot keys using semantic memories in recall results."""
    index: dict[str, str] = {}
    for result in results:
        memory = result.memory
        if not isinstance(memory, StoredSemanticMemory):
            continue
        for derivation in memory.derivations:
            if derivation.relation is SemanticDerivationRelation.SUPPORTS:
                index[derivation.episode_id] = memory.slot_key
    return index


class ACTRDeclarativeActivator:
    """Deterministic ACT-R declarative activation (base-level + partial matching)."""

    def __init__(self, spreading_activator: SpreadingActivator | None = None) -> None:
        self._spreading_activator = spreading_activator or DeterministicSpreadingActivator()

    def rank(
        self,
        *,
        candidates: Sequence[ActivationCandidate],
        cue: RetrievalCue,
        references: Mapping[MemoryIdentity, Sequence[ActivationReferenceTrace]],
        as_of: datetime,
        config: ActivationConfig,
        limit: int,
        learned_associations: Sequence[LearnedAssociation] = (),
        episode_support_index: Mapping[str, frozenset[SemanticMemoryStatus]] | None = None,
        valid_at: datetime | None = None,
        episode_slot_index: Mapping[str, str] | None = None,
    ) -> list[RecallResult]:
        seeded_entity_ids = _seed_entity_ids_from_text(cue, candidates, config)
        tag_seed_ids = _seed_tag_tokens_from_text(cue, candidates, config)
        effective_entity_ids = (
            cue.entity_ids if cue.entity_ids else _merge_seed_ids(seeded_entity_ids, tag_seed_ids)
        )
        spread_sources: tuple[str, ...] | None = None
        if not cue.entity_ids and effective_entity_ids:
            spread_sources = effective_entity_ids

        spreading_result = (
            self._spreading_activator.calculate(
                candidates=candidates,
                cue=cue,
                config=config,
                learned_associations=learned_associations,
                spread_sources=spread_sources,
            )
            if config.enable_spreading_activation
            else None
        )
        spreading_by_identity = spreading_result.scores if spreading_result is not None else {}
        spreading_metadata = spreading_result.metadata if spreading_result is not None else {}
        idf_weights = _scaled_idf_weights(candidates, cue, config)
        current_state_cue = _cue_requests_current_state(cue, config)
        temporal_mode = _temporal_retrieval_mode(
            cue,
            valid_at=valid_at,
            config=config,
            current_state_cue=current_state_cue,
        )
        support_index = episode_support_index or {}
        slot_index = episode_slot_index or {}
        current_state_policy_active = temporal_mode is TemporalRetrievalMode.CURRENT
        matched_slot_identities, matched_support_episode_ids = _matching_semantic_slot_identities(
            candidates,
            cue,
            seeded_entity_ids=seeded_entity_ids,
            current_state_cue=current_state_cue,
            config=config,
        )
        semantic_slot_fit_by_identity, episode_support_fits = _semantic_slot_fit_indexes(
            candidates,
            cue,
            temporal_mode=temporal_mode,
            effective_entities=effective_entity_ids,
        )

        scored: list[RecallResult] = []
        rank_by_identity: dict[MemoryIdentity, float] = {}
        for candidate in candidates:
            slot_fit_selection = _candidate_slot_fit(
                candidate,
                semantic_slot_fit_by_identity=semantic_slot_fit_by_identity,
                episode_support_fits=episode_support_fits,
            )
            result, rank_activation = _score_candidate(
                candidate,
                cue=cue,
                references=references,
                as_of=as_of,
                config=config,
                spreading=spreading_by_identity.get(candidate.identity, 0.0),
                spreading_metadata=spreading_metadata.get(candidate.identity),
                idf_weights=idf_weights,
                current_state_cue=current_state_cue,
                effective_entity_ids=effective_entity_ids,
                episode_support_index=support_index,
                current_state_policy_active=current_state_policy_active,
                matched_slot_identities=matched_slot_identities,
                matched_support_episode_ids=matched_support_episode_ids,
                temporal_mode=temporal_mode,
                slot_fit=slot_fit_selection.slot_fit,
                support_provenance=slot_fit_selection.support_provenance,
                selected_support_revision_key=slot_fit_selection.selected_support_revision_key,
            )
            scored.append(result)
            rank_by_identity[_result_identity(result)] = rank_activation

        admitted_outcome = _semantic_slot_admission_identities(
            candidates,
            cue,
            config=config,
            seeded_entity_ids=seeded_entity_ids,
            current_state_cue=current_state_cue,
            valid_at=valid_at,
            temporal_mode=temporal_mode,
            scored=scored,
            rank_by_identity=rank_by_identity,
            as_of=as_of,
        )
        admitted_identities = set(admitted_outcome.admitted)
        admission_eligibility = _admission_eligibility_by_identity(
            cue=cue,
            seeded_entity_ids=seeded_entity_ids,
            temporal_mode=temporal_mode,
            config=config,
            admitted_identities=admitted_identities,
        )
        scored = [
            _annotate_candidate_eligibility(
                result,
                admitted_identities=admitted_identities,
                admission_eligibility=admission_eligibility,
            )
            for result in scored
        ]

        eligible = [
            result
            for result in scored
            if (
                result.activation >= config.retrieval_threshold
                or _result_identity(result) in admitted_identities
            )
        ]
        ordered = sorted(
            eligible,
            key=lambda item: (
                -rank_by_identity[_result_identity(item)],
                item.memory_kind.value,
                _memory_key(item),
            ),
        )

        if config.exclude_superseded_support_on_current_state and current_state_policy_active:
            if valid_at is None:
                ordered = [
                    result
                    for result in ordered
                    if not _is_superseded_only_support_episode(result, support_index)
                ]

        if config.enable_duplicate_collapse:
            return _collapse_results(
                ordered,
                limit=limit,
                config=config,
                support_index=support_index,
                slot_index=slot_index,
            )
        return ordered[:limit]

    def inspect(
        self,
        *,
        candidates: Sequence[ActivationCandidate],
        cue: RetrievalCue,
        references: Mapping[MemoryIdentity, Sequence[ActivationReferenceTrace]],
        as_of: datetime,
        config: ActivationConfig,
        limit: int,
        tenant_id: str,
        subject_id: str | None,
        learned_associations: Sequence[LearnedAssociation] = (),
        episode_support_index: Mapping[str, frozenset[SemanticMemoryStatus]] | None = None,
        valid_at: datetime | None = None,
        episode_slot_index: Mapping[str, str] | None = None,
    ) -> RecallInspectionResult:
        """Evaluate all candidates and return terminal recall dispositions."""
        candidate_by_identity = {candidate.identity: candidate for candidate in candidates}
        seeded_entity_ids = _seed_entity_ids_from_text(cue, candidates, config)
        tag_seed_ids = _seed_tag_tokens_from_text(cue, candidates, config)
        effective_entity_ids = (
            cue.entity_ids if cue.entity_ids else _merge_seed_ids(seeded_entity_ids, tag_seed_ids)
        )
        spread_sources: tuple[str, ...] | None = None
        if not cue.entity_ids and effective_entity_ids:
            spread_sources = effective_entity_ids

        spreading_result = (
            self._spreading_activator.calculate(
                candidates=candidates,
                cue=cue,
                config=config,
                learned_associations=learned_associations,
                spread_sources=spread_sources,
            )
            if config.enable_spreading_activation
            else None
        )
        spreading_by_identity = spreading_result.scores if spreading_result is not None else {}
        spreading_metadata = spreading_result.metadata if spreading_result is not None else {}
        idf_weights = _scaled_idf_weights(candidates, cue, config)
        current_state_cue = _cue_requests_current_state(cue, config)
        temporal_mode = _temporal_retrieval_mode(
            cue,
            valid_at=valid_at,
            config=config,
            current_state_cue=current_state_cue,
        )
        support_index = episode_support_index or {}
        slot_index = episode_slot_index or {}
        current_state_policy_active = temporal_mode is TemporalRetrievalMode.CURRENT
        matched_slot_identities, matched_support_episode_ids = _matching_semantic_slot_identities(
            candidates,
            cue,
            seeded_entity_ids=seeded_entity_ids,
            current_state_cue=current_state_cue,
            config=config,
        )
        semantic_slot_fit_by_identity, episode_support_fits = _semantic_slot_fit_indexes(
            candidates,
            cue,
            temporal_mode=temporal_mode,
            effective_entities=effective_entity_ids,
        )

        scored: list[RecallResult] = []
        rank_by_identity: dict[MemoryIdentity, float] = {}
        for candidate in candidates:
            slot_fit_selection = _candidate_slot_fit(
                candidate,
                semantic_slot_fit_by_identity=semantic_slot_fit_by_identity,
                episode_support_fits=episode_support_fits,
            )
            result, rank_activation = _score_candidate(
                candidate,
                cue=cue,
                references=references,
                as_of=as_of,
                config=config,
                spreading=spreading_by_identity.get(candidate.identity, 0.0),
                spreading_metadata=spreading_metadata.get(candidate.identity),
                idf_weights=idf_weights,
                current_state_cue=current_state_cue,
                effective_entity_ids=effective_entity_ids,
                episode_support_index=support_index,
                current_state_policy_active=current_state_policy_active,
                matched_slot_identities=matched_slot_identities,
                matched_support_episode_ids=matched_support_episode_ids,
                temporal_mode=temporal_mode,
                slot_fit=slot_fit_selection.slot_fit,
                support_provenance=slot_fit_selection.support_provenance,
                selected_support_revision_key=slot_fit_selection.selected_support_revision_key,
            )
            scored.append(result)
            rank_by_identity[_result_identity(result)] = rank_activation

        admitted_outcome = _semantic_slot_admission_identities(
            candidates,
            cue,
            config=config,
            seeded_entity_ids=seeded_entity_ids,
            current_state_cue=current_state_cue,
            valid_at=valid_at,
            temporal_mode=temporal_mode,
            scored=scored,
            rank_by_identity=rank_by_identity,
            as_of=as_of,
        )
        admitted_identities = set(admitted_outcome.admitted)
        admission_eligibility = _admission_eligibility_by_identity(
            cue=cue,
            seeded_entity_ids=seeded_entity_ids,
            temporal_mode=temporal_mode,
            config=config,
            admitted_identities=admitted_identities,
        )
        scored = [
            _annotate_candidate_eligibility(
                result,
                admitted_identities=admitted_identities,
                admission_eligibility=admission_eligibility,
            )
            for result in scored
        ]

        disposition_by_identity: dict[MemoryIdentity, RecallInspectionDisposition] = {}
        eligible_identities = {
            _result_identity(result)
            for result in scored
            if (
                result.activation >= config.retrieval_threshold
                or _result_identity(result) in admitted_identities
            )
        }
        for result in scored:
            identity = _result_identity(result)
            if identity in eligible_identities:
                continue
            disposition_by_identity[identity] = _inspection_disposition_for_rejected(
                result,
                admission_outcome=admitted_outcome,
            )
        ordered = sorted(
            [result for result in scored if _result_identity(result) in eligible_identities],
            key=lambda item: (
                -rank_by_identity[_result_identity(item)],
                item.memory_kind.value,
                _memory_key(item),
            ),
        )

        if config.exclude_superseded_support_on_current_state and current_state_policy_active:
            if valid_at is None:
                filtered_ordered: list[RecallResult] = []
                for result in ordered:
                    identity = _result_identity(result)
                    if _is_superseded_only_support_episode(result, support_index):
                        disposition_by_identity[identity] = (
                            RecallInspectionDisposition.FILTERED_SUPERSEDED_SUPPORT
                        )
                    else:
                        filtered_ordered.append(result)
                ordered = filtered_ordered

        returned_results: list[RecallResult] = []
        if config.enable_duplicate_collapse:
            for result in ordered:
                identity = _result_identity(result)
                if len(returned_results) >= limit:
                    disposition_by_identity[identity] = RecallInspectionDisposition.LIMITED
                    continue
                if any(
                    _should_collapse(
                        existing,
                        result,
                        config=config,
                        support_index=support_index,
                        slot_index=slot_index,
                    )
                    for existing in returned_results
                ):
                    disposition_by_identity[identity] = RecallInspectionDisposition.COLLAPSED
                    continue
                disposition_by_identity[identity] = RecallInspectionDisposition.RETURNED
                returned_results.append(result)
        else:
            for index, result in enumerate(ordered):
                identity = _result_identity(result)
                if index < limit:
                    disposition_by_identity[identity] = RecallInspectionDisposition.RETURNED
                    returned_results.append(result)
                else:
                    disposition_by_identity[identity] = RecallInspectionDisposition.LIMITED

        returned_candidates: list[RecallInspectionCandidate] = []
        rejected_candidates: list[RecallInspectionCandidate] = []
        rank = 0
        for result in scored:
            identity = _result_identity(result)
            candidate = candidate_by_identity[identity]
            disposition = disposition_by_identity[identity]
            soft_admitted = identity in admitted_identities
            passed_threshold = result.activation >= config.retrieval_threshold
            stored = tuple(references.get(identity, ()))
            inspection = RecallInspectionCandidate(
                memory_kind=result.memory_kind,
                memory=result.memory,
                disposition=disposition,
                activation=result.activation,
                score=result.score,
                retrieval_threshold=config.retrieval_threshold,
                passed_threshold=passed_threshold,
                soft_admitted=soft_admitted,
                components=result.components,
                cognitive_traces=candidate.cognitive_traces,
                stored_traces=stored,
                diagnostics=result.diagnostics,
                reason=result.reason,
            )
            if disposition is RecallInspectionDisposition.RETURNED:
                rank += 1
                returned_candidates.append(
                    RecallInspectionCandidate(
                        memory_kind=inspection.memory_kind,
                        memory=inspection.memory,
                        disposition=inspection.disposition,
                        activation=inspection.activation,
                        score=inspection.score,
                        retrieval_threshold=inspection.retrieval_threshold,
                        passed_threshold=inspection.passed_threshold,
                        soft_admitted=inspection.soft_admitted,
                        components=inspection.components,
                        cognitive_traces=inspection.cognitive_traces,
                        stored_traces=inspection.stored_traces,
                        rank=rank,
                        diagnostics=inspection.diagnostics,
                        reason=inspection.reason,
                    )
                )
            else:
                rejected_candidates.append(inspection)

        return RecallInspectionResult(
            tenant_id=tenant_id,
            subject_id=subject_id,
            inspected_at=as_of,
            retrieval_threshold=config.retrieval_threshold,
            returned=tuple(returned_candidates),
            rejected=tuple(rejected_candidates),
            truncated=False,
            considered_count=len(candidates),
        )


def _merge_seed_ids(*groups: Sequence[str]) -> tuple[str, ...]:
    merged: set[str] = set()
    for group in groups:
        merged.update(group)
    return tuple(sorted(merged))


def _slot_admission_active(
    cue: RetrievalCue,
    *,
    config: ActivationConfig,
    current_state_cue: bool,
    seeded_entity_ids: tuple[str, ...],
    temporal_mode: TemporalRetrievalMode | None = None,
) -> bool:
    if not config.enable_semantic_slot_admission:
        return False
    if config.force_slot_admission:
        return True
    if not config.slot_admission_requires_current_state_or_predicate:
        return True
    if temporal_mode is TemporalRetrievalMode.HISTORICAL:
        return True
    if temporal_mode is TemporalRetrievalMode.NEUTRAL and cue.text and cue.text.strip():
        return True
    effective_entity_ids = cue.entity_ids if cue.entity_ids else seeded_entity_ids
    if config.enable_entity_slot_admission and effective_entity_ids:
        return True
    return current_state_cue or cue.predicate is not None


def _temporal_retrieval_mode(
    cue: RetrievalCue,
    *,
    valid_at: datetime | None,
    config: ActivationConfig | None = None,
    current_state_cue: bool | None = None,
) -> TemporalRetrievalMode:
    if valid_at is not None:
        return TemporalRetrievalMode.HISTORICAL
    if current_state_cue is None:
        if config is None:
            raise ValidationError("config is required when current_state_cue is omitted.")
        current_state_cue = _cue_requests_current_state(cue, config)
    if current_state_cue or cue.predicate is not None or cue.object_value is not None:
        return TemporalRetrievalMode.CURRENT
    return TemporalRetrievalMode.NEUTRAL


def _current_state_policy_active(
    cue: RetrievalCue,
    *,
    valid_at: datetime | None,
    current_state_cue: bool,
) -> bool:
    return (
        _temporal_retrieval_mode(
            cue,
            valid_at=valid_at,
            current_state_cue=current_state_cue,
        )
        is TemporalRetrievalMode.CURRENT
    )


def _scaled_idf_weights(
    candidates: Sequence[ActivationCandidate],
    cue: RetrievalCue,
    config: ActivationConfig,
) -> dict[str, float] | None:
    if not config.enable_candidate_idf:
        return None
    base = _candidate_idf_weights(candidates)
    if not cue.text or not cue.text.strip():
        return base
    cue_tokens = _tokenize(cue.text)
    if not cue_tokens.intersection(config.incident_cue_tokens):
        return base
    scale = config.distinctive_token_idf_scale
    return {
        token: weight * (scale if token in cue_tokens else 1.0) for token, weight in base.items()
    }


def _score_candidate(
    candidate: ActivationCandidate,
    *,
    cue: RetrievalCue,
    references: Mapping[MemoryIdentity, Sequence[ActivationReferenceTrace]],
    as_of: datetime,
    config: ActivationConfig,
    spreading: float,
    spreading_metadata: SpreadingMetadata | None,
    idf_weights: Mapping[str, float] | None,
    current_state_cue: bool,
    effective_entity_ids: Sequence[str],
    episode_support_index: Mapping[str, frozenset[SemanticMemoryStatus]],
    current_state_policy_active: bool,
    matched_slot_identities: set[MemoryIdentity],
    matched_support_episode_ids: set[str],
    temporal_mode: TemporalRetrievalMode,
    slot_fit: float | None,
    support_provenance: Sequence[SupportProvenance],
    selected_support_revision_key: str | None,
) -> tuple[RecallResult, float]:
    identity = candidate.identity
    stored_traces = references.get(identity, ())
    reference_traces = activation_reference_traces_for_candidate(candidate, stored_traces)
    if not reference_traces:
        raise ValidationError("Activation requires at least one reference trace.")
    base_level = calculate_base_level(
        reference_traces,
        as_of=as_of,
        decay=config.decay,
        constant=config.base_level_constant,
        time_unit_seconds=config.time_unit_seconds,
        minimum_elapsed_seconds=config.minimum_elapsed_seconds,
    )
    text_match = _text_match_metrics(
        cue.text,
        candidate.text,
        idf_weights=idf_weights,
        precision_enabled=config.enable_text_precision_matching,
    )
    accessibility_partial = (
        _calculate_partial_match(
            candidate,
            cue,
            config,
            effective_entity_ids=effective_entity_ids,
            text_similarity=lambda _query, _text: text_match.coverage,
        )
        if config.enable_partial_matching
        else 0.0
    )
    ranking_partial = accessibility_partial
    if config.enable_partial_matching and config.enable_text_precision_matching:
        ranking_partial = _calculate_partial_match(
            candidate,
            cue,
            config,
            effective_entity_ids=effective_entity_ids,
            text_similarity=lambda _query, _text: text_match.cue_fit,
        )
    current_state = _current_state_activation(
        candidate,
        current_state_cue=current_state_cue,
        config=config,
        episode_support_index=episode_support_index,
        current_state_policy_active=current_state_policy_active,
        matched_slot_identities=matched_slot_identities,
        matched_support_episode_ids=matched_support_episode_ids,
    )
    conjunction = _multi_entity_conjunction_bonus(
        candidate,
        effective_entity_ids=effective_entity_ids,
        config=config,
    )
    noise = 0.0
    activation = (
        base_level + spreading + accessibility_partial + current_state + conjunction + noise
    )
    structured_adjustment = _structured_rank_adjustment(
        slot_fit, mismatch_penalty=config.mismatch_penalty
    )
    rank_activation = activation - accessibility_partial + ranking_partial + structured_adjustment
    latency_seconds = config.latency_factor * math.exp(-config.latency_exponent * activation)
    score = _presentation_score(activation, config.retrieval_threshold)
    components = ActivationComponents(
        base_level=base_level,
        spreading=spreading,
        partial_match=accessibility_partial + conjunction,
        noise=noise,
        total=activation,
        current_state=current_state,
    )
    matched_entities = len(set(effective_entity_ids).intersection(candidate.entity_ids))
    text_coverage = text_match.coverage
    text_cue_fit = text_match.cue_fit
    slot_fit_source = _slot_fit_source(candidate, slot_fit)
    observation_evidence_ids = _observation_evidence_ids(candidate)
    spread_hop = (
        spreading_metadata.hop if spreading_metadata is not None and spreading > 0.0 else None
    )
    spread_sources = (
        tuple(spreading_metadata.sources)
        if spreading_metadata is not None and spreading > 0.0
        else ()
    )
    learned_edges = (
        spreading_metadata.learned_association_count
        if spreading_metadata is not None and spreading > 0.0
        else 0
    )
    diagnostics = RetrievalDiagnostics(
        rank_activation=rank_activation,
        accessibility_partial=accessibility_partial,
        ranking_partial=ranking_partial,
        conjunction=conjunction,
        text_coverage=text_coverage,
        text_cue_fit=text_cue_fit,
        temporal_mode=temporal_mode.value,
        base_level=base_level,
        spreading=spreading,
        current_state=current_state,
        reference_count=len(reference_traces),
        matched_entities=matched_entities,
        cue_entity_count=len(effective_entity_ids),
        spread_hop=spread_hop,
        spread_sources=spread_sources,
        learned_edges=learned_edges,
        slot_fit=slot_fit,
        structured_adjustment=structured_adjustment,
        slot_fit_source=slot_fit_source,
        semantic_slot_key=candidate.slot_key,
        semantic_status=candidate.semantic_status,
        observation_evidence_ids=observation_evidence_ids,
        support_provenance=tuple(support_provenance),
        selected_support_revision_key=selected_support_revision_key,
    )
    reason = _build_reason(
        activation=activation,
        base_level=base_level,
        spreading=spreading,
        partial_match=accessibility_partial,
        reference_count=len(reference_traces),
        matched_entities=matched_entities,
        cue_entity_count=len(effective_entity_ids),
        spreading_metadata=spreading_metadata,
        current_state=current_state,
        conjunction=conjunction,
        rank_activation=rank_activation,
        text_coverage=text_coverage,
        text_cue_fit=text_cue_fit,
        temporal_mode=temporal_mode,
        slot_fit=slot_fit,
        structured_adjustment=structured_adjustment,
        slot_fit_source=slot_fit_source.value if slot_fit_source is not None else None,
        eligibility=diagnostics.eligibility.value,
        admission_reason=diagnostics.admission_reason,
        semantic_slot_key=diagnostics.semantic_slot_key,
        semantic_status=diagnostics.semantic_status,
        selected_support_revision_key=diagnostics.selected_support_revision_key,
        spread_hop=diagnostics.spread_hop,
        spread_sources=diagnostics.spread_sources,
        learned_edges=diagnostics.learned_edges,
    )
    return (
        RecallResult(
            memory_kind=candidate.memory_kind,
            memory=candidate.memory,
            activation=activation,
            score=score,
            latency_seconds=latency_seconds,
            components=components,
            reason=reason,
            diagnostics=diagnostics,
        ),
        rank_activation,
    )


def _multi_entity_conjunction_bonus(
    candidate: ActivationCandidate,
    *,
    effective_entity_ids: Sequence[str],
    config: ActivationConfig,
) -> float:
    if not config.enable_multi_entity_conjunction:
        return 0.0
    if len(effective_entity_ids) < 2:
        return 0.0
    overlap = set(effective_entity_ids).intersection(candidate.entity_ids)
    if len(overlap) < 2:
        return 0.0
    return config.conjunction_weight


def _is_superseded_only_support_episode(
    result: RecallResult,
    support_index: Mapping[str, frozenset[SemanticMemoryStatus]],
) -> bool:
    if result.memory_kind is not MemoryKind.EPISODE:
        return False
    memory = result.memory
    if not isinstance(memory, StoredEpisode):
        return False
    statuses = support_index.get(memory.id, frozenset())
    return (
        SemanticMemoryStatus.SUPERSEDED in statuses and SemanticMemoryStatus.ACTIVE not in statuses
    )


def _result_identity(result: RecallResult) -> MemoryIdentity:
    return MemoryIdentity(memory_kind=result.memory_kind, memory_key=_memory_key(result))


def _memory_key(result: RecallResult) -> str:
    memory = result.memory
    if isinstance(memory, StoredEpisode):
        return memory.memory_key
    return memory.memory_key


def _presentation_score(activation: float, threshold: float) -> float:
    return 1.0 / (1.0 + math.exp(-(activation - threshold)))


def _build_reason(
    *,
    activation: float,
    base_level: float,
    spreading: float,
    partial_match: float,
    reference_count: int,
    matched_entities: int,
    cue_entity_count: int,
    spreading_metadata: SpreadingMetadata | None,
    current_state: float = 0.0,
    conjunction: float = 0.0,
    rank_activation: float | None = None,
    text_coverage: float | None = None,
    text_cue_fit: float | None = None,
    temporal_mode: TemporalRetrievalMode | None = None,
    slot_fit: float | None = None,
    structured_adjustment: float = 0.0,
    slot_fit_source: str | None = None,
    eligibility: str | None = None,
    admission_reason: str | None = None,
    semantic_slot_key: str | None = None,
    semantic_status: SemanticMemoryStatus | None = None,
    selected_support_revision_key: str | None = None,
    spread_hop: int | None = None,
    spread_sources: Sequence[str] = (),
    learned_edges: int = 0,
) -> str:
    reason = (
        f"activation={activation:.3f}; base={base_level:.3f}; "
        f"spread={spreading:.3f}; partial={partial_match:.3f}; "
        f"references={reference_count}; "
        f"matched_entities={matched_entities}/{cue_entity_count}"
    )
    if temporal_mode is not None:
        reason += f"; temporal_mode={temporal_mode.value}"
    if eligibility is not None:
        reason += f"; eligibility={eligibility}"
    if admission_reason is not None:
        reason += f"; admission_reason={admission_reason}"
    if rank_activation is not None and rank_activation != activation:
        reason += f"; rank_activation={rank_activation:.3f}"
    if text_coverage is not None and text_coverage > 0.0:
        reason += f"; text_coverage={text_coverage:.3f}"
    if text_cue_fit is not None and text_coverage is not None and text_cue_fit != text_coverage:
        reason += f"; text_cue_fit={text_cue_fit:.3f}"
    if slot_fit is not None:
        reason += f"; slot_fit={slot_fit:.2f}"
        if slot_fit_source is not None:
            reason += f"; slot_fit_source={slot_fit_source}"
    if semantic_slot_key is not None:
        reason += f"; semantic_slot={semantic_slot_key}"
    if semantic_status is not None:
        reason += f"; semantic_status={semantic_status.value}"
    if selected_support_revision_key is not None:
        reason += f"; supports_revision={selected_support_revision_key}"
    if structured_adjustment != 0.0:
        reason += f"; structured_adjustment={structured_adjustment:+.2f}"
    if current_state != 0.0:
        reason += f"; current_state={current_state:.3f}"
    if conjunction != 0.0:
        reason += f"; conjunction={conjunction:.3f}"
    if spreading > 0.0:
        if spread_hop is None and spreading_metadata is not None:
            spread_hop = spreading_metadata.hop
        if not spread_sources and spreading_metadata is not None:
            spread_sources = spreading_metadata.sources
        if learned_edges == 0 and spreading_metadata is not None:
            learned_edges = spreading_metadata.learned_association_count
        if spread_hop is not None:
            reason += f"; spread_hop={spread_hop}"
        if spread_sources:
            reason += f"; spread_sources={','.join(spread_sources)}"
        if learned_edges > 0:
            reason += f"; learned_edges={learned_edges}"
    return reason


def _text_match_metrics(
    cue_text: str | None,
    candidate_text: str,
    *,
    idf_weights: Mapping[str, float] | None,
    precision_enabled: bool,
) -> _TextMatchMetrics:
    if cue_text is None or not cue_text.strip():
        return _TextMatchMetrics(coverage=0.0, cue_fit=0.0)
    coverage = _text_query_coverage(cue_text, candidate_text, idf_weights=idf_weights)
    cue_fit = (
        _text_cue_fit(cue_text, candidate_text, idf_weights=idf_weights)
        if precision_enabled
        else coverage
    )
    return _TextMatchMetrics(coverage=coverage, cue_fit=cue_fit)


def _slot_fit_source(
    candidate: ActivationCandidate, slot_fit: float | None
) -> SlotFitSource | None:
    if slot_fit is None:
        return None
    if candidate.memory_kind is MemoryKind.EPISODE:
        return SlotFitSource.SUPPORT
    if candidate.memory_kind is MemoryKind.SEMANTIC:
        return SlotFitSource.SEMANTIC
    return None


def _observation_evidence_ids(candidate: ActivationCandidate) -> tuple[str, ...]:
    memory = candidate.memory
    if isinstance(memory, StoredEpisode):
        return tuple(sorted({evidence.observation_id for evidence in memory.evidence}))
    if isinstance(memory, StoredSemanticMemory):
        return tuple(sorted({evidence.observation_id for evidence in memory.observation_evidence}))
    return ()


def _admission_eligibility_by_identity(
    *,
    cue: RetrievalCue,
    seeded_entity_ids: Sequence[str],
    temporal_mode: TemporalRetrievalMode,
    config: ActivationConfig,
    admitted_identities: set[MemoryIdentity],
) -> dict[MemoryIdentity, RetrievalEligibility]:
    if not admitted_identities:
        return {}
    if temporal_mode is TemporalRetrievalMode.HISTORICAL:
        eligibility = RetrievalEligibility.HISTORICAL_SLOT_ADMISSION
    elif cue.entity_ids or (config.enable_entity_slot_admission and seeded_entity_ids):
        eligibility = RetrievalEligibility.ENTITY_SLOT_ADMISSION
    else:
        eligibility = RetrievalEligibility.SEMANTIC_SLOT_ADMISSION
    return {identity: eligibility for identity in admitted_identities}


def _annotate_candidate_eligibility(
    result: RecallResult,
    *,
    admitted_identities: set[MemoryIdentity],
    admission_eligibility: Mapping[MemoryIdentity, RetrievalEligibility],
) -> RecallResult:
    identity = _result_identity(result)
    diagnostics = result.diagnostics
    if diagnostics is None:
        return result
    if identity not in admitted_identities:
        return result
    eligibility = admission_eligibility.get(identity, RetrievalEligibility.SEMANTIC_SLOT_ADMISSION)
    updated_diagnostics = replace(
        diagnostics,
        eligibility=eligibility,
        admission_reason=eligibility.value,
        soft_admitted=True,
    )
    return replace(
        result,
        diagnostics=updated_diagnostics,
        reason=f"{_build_reason_from_diagnostics(result, updated_diagnostics)}; soft_admitted=true",
    )


def _build_reason_from_diagnostics(
    result: RecallResult,
    diagnostics: RetrievalDiagnostics,
) -> str:
    return _build_reason(
        activation=result.activation,
        base_level=diagnostics.base_level,
        spreading=diagnostics.spreading,
        partial_match=diagnostics.accessibility_partial,
        reference_count=diagnostics.reference_count,
        matched_entities=diagnostics.matched_entities,
        cue_entity_count=diagnostics.cue_entity_count,
        spreading_metadata=None,
        current_state=diagnostics.current_state,
        conjunction=diagnostics.conjunction,
        rank_activation=diagnostics.rank_activation,
        text_coverage=diagnostics.text_coverage,
        text_cue_fit=diagnostics.text_cue_fit,
        temporal_mode=TemporalRetrievalMode(diagnostics.temporal_mode),
        slot_fit=diagnostics.slot_fit,
        structured_adjustment=diagnostics.structured_adjustment,
        slot_fit_source=(
            diagnostics.slot_fit_source.value if diagnostics.slot_fit_source is not None else None
        ),
        eligibility=diagnostics.eligibility.value,
        admission_reason=diagnostics.admission_reason,
        semantic_slot_key=diagnostics.semantic_slot_key,
        semantic_status=diagnostics.semantic_status,
        selected_support_revision_key=diagnostics.selected_support_revision_key,
    )


def _seed_entity_ids_from_text(
    cue: RetrievalCue,
    candidates: Sequence[ActivationCandidate],
    config: ActivationConfig,
) -> tuple[str, ...]:
    if not config.enable_text_entity_seeding:
        return ()
    if cue.entity_ids:
        return ()
    if not cue.text or not cue.text.strip():
        return ()
    cue_tokens = _tokenize(cue.text)
    if not cue_tokens:
        return ()
    seeded: set[str] = set()
    for candidate in candidates:
        candidate_tokens = _tokenize(candidate.text)
        for entity_id in candidate.entity_ids:
            if entity_id in seeded:
                continue
            entity_tokens = _tokenize(entity_id)
            if entity_tokens.intersection(cue_tokens):
                seeded.add(entity_id)
                continue
            if entity_id.lower() in cue_tokens:
                seeded.add(entity_id)
                continue
            if entity_id in candidate_tokens and entity_id.lower() in cue_tokens:
                seeded.add(entity_id)
    return tuple(sorted(seeded))


def _seed_tag_tokens_from_text(
    cue: RetrievalCue,
    candidates: Sequence[ActivationCandidate],
    config: ActivationConfig,
) -> tuple[str, ...]:
    if not config.enable_incident_tag_seeding:
        return ()
    if not cue.text or not cue.text.strip():
        return ()
    cue_tokens = _tokenize(cue.text)
    if not cue_tokens:
        return ()
    seeded: set[str] = set()
    for candidate in candidates:
        for tag in _candidate_tag_tokens(candidate):
            if tag in cue_tokens:
                seeded.add(tag)
    return tuple(sorted(seeded))


def _candidate_tag_tokens(candidate: ActivationCandidate) -> set[str]:
    memory = candidate.memory
    if not isinstance(memory, StoredEpisode):
        return set()
    metadata = memory.metadata
    tags: set[str] = set()
    raw_tags = metadata.get("tags")
    if isinstance(raw_tags, (list, tuple, set, frozenset)):
        tags.update(_normalise_tag(token) for token in raw_tags if isinstance(token, str))
    episode_meta = metadata.get("episode")
    if isinstance(episode_meta, Mapping):
        episode_tags = episode_meta.get("tags")
        if isinstance(episode_tags, (list, tuple, set, frozenset)):
            tags.update(_normalise_tag(token) for token in episode_tags if isinstance(token, str))
    return {tag for tag in tags if tag}


def _normalise_tag(value: str) -> str:
    return _normalise_text(value).replace(" ", "-")


def _semantic_slot_admission_identities(
    candidates: Sequence[ActivationCandidate],
    cue: RetrievalCue,
    *,
    config: ActivationConfig,
    seeded_entity_ids: tuple[str, ...],
    current_state_cue: bool,
    valid_at: datetime | None,
    temporal_mode: TemporalRetrievalMode | None = None,
    scored: Sequence[RecallResult] | None = None,
    rank_by_identity: Mapping[MemoryIdentity, float] | None = None,
    as_of: datetime | None = None,
) -> _SemanticAdmissionOutcome:
    empty = _SemanticAdmissionOutcome(
        admitted=frozenset(),
        matched_semantics=frozenset(),
        below_soft_floor=frozenset(),
        capped_out=frozenset(),
    )
    if not _slot_admission_active(
        cue,
        config=config,
        current_state_cue=current_state_cue,
        seeded_entity_ids=seeded_entity_ids,
        temporal_mode=temporal_mode,
    ):
        return empty
    if scored is None or rank_by_identity is None or as_of is None:
        raise ValidationError(
            "Bounded semantic admission requires scored results, rank activations, and as_of."
        )

    effective_sources = cue.entity_ids if cue.entity_ids else seeded_entity_ids
    cue_tokens = _tokenize(cue.text) if cue.text else set()
    distinctive_tokens = cue_tokens - config.current_state_cue_tokens

    activation_by_identity = {_result_identity(result): result.activation for result in scored}
    matched_semantics: list[tuple[MemoryIdentity, float]] = []
    structured_cue = (
        cue.predicate is not None or cue.object_value is not None or bool(cue.entity_ids)
    )
    for candidate in candidates:
        if candidate.memory_kind is not MemoryKind.SEMANTIC:
            continue
        if not _semantic_eligible_for_soft_admission(
            candidate,
            valid_at=valid_at,
            as_of=as_of,
        ):
            continue
        if not _semantic_slot_matches_cue(
            candidate,
            cue,
            effective_sources=effective_sources,
            current_state_cue=current_state_cue,
            distinctive_tokens=distinctive_tokens,
            config=config,
        ):
            continue
        activation = activation_by_identity.get(candidate.identity)
        if activation is None:
            continue
        matched_semantics.append((candidate.identity, activation))

    matched_identity_set = frozenset(identity for identity, _ in matched_semantics)
    below_soft_floor: set[MemoryIdentity] = set()
    above_floor: list[tuple[MemoryIdentity, float]] = []
    for identity, activation in matched_semantics:
        if not structured_cue and activation < config.semantic_soft_admission_floor:
            below_soft_floor.add(identity)
        else:
            above_floor.append((identity, rank_by_identity.get(identity, activation)))

    above_floor.sort(key=lambda item: -item[1])
    admitted_semantics: set[MemoryIdentity] = set()
    capped_out: set[MemoryIdentity] = set()
    for index, (identity, _) in enumerate(above_floor):
        if index >= config.max_soft_admitted_semantics:
            capped_out.add(identity)
        else:
            admitted_semantics.add(identity)

    episode_by_id: dict[str, ActivationCandidate] = {}
    for candidate in candidates:
        if candidate.memory_kind is MemoryKind.EPISODE and isinstance(
            candidate.memory, StoredEpisode
        ):
            episode_by_id[candidate.memory.id] = candidate

    admitted: set[MemoryIdentity] = set(admitted_semantics)
    for candidate in candidates:
        if candidate.identity not in admitted_semantics:
            continue
        memory = candidate.memory
        if not isinstance(memory, StoredSemanticMemory):
            continue
        for derivation in memory.derivations:
            if derivation.relation is not SemanticDerivationRelation.SUPPORTS:
                continue
            episode_candidate = episode_by_id.get(derivation.episode_id)
            if episode_candidate is not None:
                admitted.add(episode_candidate.identity)

    return _SemanticAdmissionOutcome(
        admitted=frozenset(admitted),
        matched_semantics=matched_identity_set,
        below_soft_floor=frozenset(below_soft_floor),
        capped_out=frozenset(capped_out),
    )


def _matching_semantic_slot_identities(
    candidates: Sequence[ActivationCandidate],
    cue: RetrievalCue,
    *,
    seeded_entity_ids: tuple[str, ...],
    current_state_cue: bool,
    config: ActivationConfig,
) -> tuple[set[MemoryIdentity], set[str]]:
    effective_sources = cue.entity_ids if cue.entity_ids else seeded_entity_ids
    cue_tokens = _tokenize(cue.text) if cue.text else set()
    distinctive_tokens = cue_tokens - config.current_state_cue_tokens
    matched: set[MemoryIdentity] = set()
    matched_support_episode_ids: set[str] = set()
    for candidate in candidates:
        if candidate.memory_kind is not MemoryKind.SEMANTIC:
            continue
        if not _semantic_slot_matches_cue(
            candidate,
            cue,
            effective_sources=effective_sources,
            current_state_cue=current_state_cue,
            distinctive_tokens=distinctive_tokens,
            config=config,
        ):
            continue
        matched.add(candidate.identity)
        memory = candidate.memory
        if not isinstance(memory, StoredSemanticMemory):
            continue
        for derivation in memory.derivations:
            if derivation.relation is SemanticDerivationRelation.SUPPORTS:
                matched_support_episode_ids.add(derivation.episode_id)
    return matched, matched_support_episode_ids


def _semantic_slot_matches_cue(
    candidate: ActivationCandidate,
    cue: RetrievalCue,
    *,
    effective_sources: Sequence[str],
    current_state_cue: bool,
    distinctive_tokens: set[str],
    config: ActivationConfig,
) -> bool:
    return _semantic_fields_match_cue(
        predicate=candidate.predicate,
        entity_ids=candidate.entity_ids,
        text=candidate.text,
        object_value=candidate.object_value,
        cue=cue,
        effective_sources=effective_sources,
        current_state_cue=current_state_cue,
        distinctive_tokens=distinctive_tokens,
        config=config,
    )


def _stored_semantic_matches_cue(
    memory: StoredSemanticMemory,
    cue: RetrievalCue,
    *,
    effective_sources: Sequence[str],
    current_state_cue: bool,
    distinctive_tokens: set[str],
    config: ActivationConfig,
) -> bool:
    entity_ids = {entity.entity_id for entity in memory.entities}
    if memory.subject_entity_id:
        entity_ids.add(memory.subject_entity_id)
    if memory.object_entity_id:
        entity_ids.add(memory.object_entity_id)
    return _semantic_fields_match_cue(
        predicate=memory.predicate,
        entity_ids=tuple(sorted(entity_ids)),
        text=memory.statement,
        object_value=memory.object_value,
        cue=cue,
        effective_sources=effective_sources,
        current_state_cue=current_state_cue,
        distinctive_tokens=distinctive_tokens,
        config=config,
    )


def _semantic_fields_match_cue(
    *,
    predicate: str | None,
    entity_ids: Sequence[str],
    text: str,
    object_value: str | None,
    cue: RetrievalCue,
    effective_sources: Sequence[str],
    current_state_cue: bool,
    distinctive_tokens: set[str],
    config: ActivationConfig,
) -> bool:
    has_explicit_constraint = False
    if cue.predicate is not None:
        has_explicit_constraint = True
        if _normalised_equality(cue.predicate, predicate) < 1.0:
            return False
    if cue.object_value is not None:
        has_explicit_constraint = True
        if _normalised_equality(cue.object_value, object_value) < 1.0:
            return False
    if cue.entity_ids:
        has_explicit_constraint = True
        if not set(cue.entity_ids).intersection(entity_ids):
            return False
    if (
        not cue.entity_ids
        and effective_sources
        and (cue.predicate is not None or cue.object_value is not None)
        and not set(effective_sources).intersection(entity_ids)
    ):
        return False
    if has_explicit_constraint:
        return True
    if effective_sources:
        return bool(set(effective_sources).intersection(entity_ids))
    if current_state_cue and distinctive_tokens:
        statement_tokens = _tokenize(text)
        object_tokens = _tokenize(object_value or "")
        if distinctive_tokens.intersection(statement_tokens.union(object_tokens)):
            return True
    if config.enable_lexical_slot_matching and distinctive_tokens:
        semantic_tokens = _semantic_lexical_tokens(predicate, object_value, text)
        overlap = distinctive_tokens.intersection(semantic_tokens)
        if len(overlap) >= config.lexical_slot_min_overlap:
            return True
    return False


def _semantic_slot_fit_indexes(
    candidates: Sequence[ActivationCandidate],
    cue: RetrievalCue,
    *,
    temporal_mode: TemporalRetrievalMode,
    effective_entities: Sequence[str],
) -> tuple[dict[MemoryIdentity, float | None], dict[str, list[SupportProvenance]]]:
    semantic_slot_fit_by_identity: dict[MemoryIdentity, float | None] = {}
    episode_support_fits: dict[str, list[SupportProvenance]] = defaultdict(list)
    for candidate in candidates:
        if candidate.memory_kind is not MemoryKind.SEMANTIC:
            continue
        fit = _semantic_slot_fit(
            candidate,
            cue,
            temporal_mode=temporal_mode,
            effective_entities=effective_entities,
        )
        semantic_slot_fit_by_identity[candidate.identity] = fit
        memory = candidate.memory
        if not isinstance(memory, StoredSemanticMemory):
            continue
        for derivation in memory.derivations:
            if derivation.relation is SemanticDerivationRelation.SUPPORTS:
                episode_support_fits[derivation.episode_id].append(
                    SupportProvenance(
                        semantic_memory_key=memory.memory_key,
                        semantic_revision_key=memory.revision_key,
                        semantic_slot_key=memory.slot_key,
                        semantic_status=memory.status,
                        contribution_score=derivation.contribution_score,
                        slot_fit=fit,
                        semantic_valid_from=memory.valid_from,
                        semantic_valid_until=memory.valid_until,
                    )
                )
    return semantic_slot_fit_by_identity, episode_support_fits


def _candidate_slot_fit(
    candidate: ActivationCandidate,
    *,
    semantic_slot_fit_by_identity: Mapping[MemoryIdentity, float | None],
    episode_support_fits: Mapping[str, Sequence[SupportProvenance]],
) -> _SlotFitSelection:
    if candidate.memory_kind is MemoryKind.SEMANTIC:
        return _SlotFitSelection(slot_fit=semantic_slot_fit_by_identity.get(candidate.identity))
    if candidate.memory_kind is MemoryKind.EPISODE and isinstance(candidate.memory, StoredEpisode):
        provenance_by_revision = {
            item.semantic_revision_key: item for item in candidate.support_provenance
        }
        for fit_item in episode_support_fits.get(candidate.memory.id, ()):
            provenance_by_revision[fit_item.semantic_revision_key] = fit_item
        supported = tuple(provenance_by_revision.values())
        return _support_slot_fit(supported)
    return _SlotFitSelection(slot_fit=None)


def _support_slot_fit(supported_fits: Sequence[SupportProvenance]) -> _SlotFitSelection:
    ordered_provenance = tuple(
        sorted(
            supported_fits,
            key=lambda fit: (
                fit.semantic_slot_key,
                fit.semantic_memory_key,
                fit.semantic_revision_key,
            ),
        )
    )
    fits = [fit for fit in supported_fits if fit.slot_fit is not None]
    if not fits:
        return _SlotFitSelection(slot_fit=None, support_provenance=ordered_provenance)
    selected = max(
        fits,
        key=lambda fit: (
            fit.slot_fit if fit.slot_fit is not None else -1.0,
            fit.contribution_score,
            fit.semantic_revision_key,
        ),
    )
    return _SlotFitSelection(
        slot_fit=selected.slot_fit,
        support_provenance=ordered_provenance,
        selected_support_revision_key=selected.semantic_revision_key,
    )


def _semantic_slot_fit(
    candidate: ActivationCandidate,
    cue: RetrievalCue,
    *,
    temporal_mode: TemporalRetrievalMode,
    effective_entities: Sequence[str],
) -> float | None:
    if candidate.memory_kind is not MemoryKind.SEMANTIC:
        return None
    if not _slot_fit_applicable(cue, temporal_mode, effective_entities):
        return None
    if not _structured_constraints_match(
        candidate,
        cue,
        temporal_mode=temporal_mode,
        effective_entities=effective_entities,
    ):
        return 0.0
    if temporal_mode is TemporalRetrievalMode.NEUTRAL:
        return 1.0
    if _temporal_slot_compatible(candidate, temporal_mode):
        return 1.0
    return 0.0


def _slot_fit_applicable(
    cue: RetrievalCue,
    temporal_mode: TemporalRetrievalMode,
    effective_entities: Sequence[str],
) -> bool:
    if cue.predicate is not None or cue.object_value is not None:
        return True
    if temporal_mode is TemporalRetrievalMode.NEUTRAL:
        return False
    return bool(cue.entity_ids or effective_entities)


def _structured_constraints_match(
    candidate: ActivationCandidate,
    cue: RetrievalCue,
    *,
    temporal_mode: TemporalRetrievalMode,
    effective_entities: Sequence[str],
) -> bool:
    if cue.predicate is not None:
        if _normalised_equality(cue.predicate, candidate.predicate) < 1.0:
            return False
    if cue.object_value is not None:
        if _normalised_equality(cue.object_value, candidate.object_value) < 1.0:
            return False
    required_entities: Sequence[str]
    if cue.entity_ids:
        required_entities = cue.entity_ids
    elif effective_entities and (cue.predicate is not None or cue.object_value is not None):
        required_entities = effective_entities
    elif temporal_mode is not TemporalRetrievalMode.NEUTRAL:
        required_entities = effective_entities
    else:
        required_entities = ()
    if required_entities and not set(required_entities).intersection(candidate.entity_ids):
        return False
    return True


def _temporal_slot_compatible(
    candidate: ActivationCandidate,
    temporal_mode: TemporalRetrievalMode,
) -> bool:
    if temporal_mode is TemporalRetrievalMode.CURRENT:
        return candidate.semantic_status is SemanticMemoryStatus.ACTIVE
    if temporal_mode is TemporalRetrievalMode.HISTORICAL:
        return True
    return False


def _structured_rank_adjustment(
    slot_fit: float | None,
    *,
    mismatch_penalty: float,
) -> float:
    if slot_fit is None:
        return 0.0
    return mismatch_penalty * (slot_fit - 0.5)


def _calculate_partial_match(
    candidate: ActivationCandidate,
    cue: RetrievalCue,
    config: ActivationConfig,
    *,
    effective_entity_ids: Sequence[str] = (),
    text_similarity: Callable[[str, str], float],
) -> float:
    mismatches: list[float] = []
    weights: list[float] = []

    if cue.text and cue.text.strip():
        similarity = text_similarity(cue.text, candidate.text)
        mismatches.append(similarity - 1.0)
        weights.append(_PARTIAL_MATCH_WEIGHTS["text"])

    if cue.subject_id is not None:
        similarity = float(cue.subject_id == candidate.subject_id)
        mismatches.append(similarity - 1.0)
        weights.append(_PARTIAL_MATCH_WEIGHTS["subject"])

    entity_ids_for_match = cue.entity_ids if cue.entity_ids else effective_entity_ids
    if entity_ids_for_match:
        similarity = _jaccard_similarity(entity_ids_for_match, candidate.entity_ids)
        mismatches.append(similarity - 1.0)
        entity_weight = _PARTIAL_MATCH_WEIGHTS["entities"]
        if not cue.entity_ids and effective_entity_ids:
            entity_weight = config.seeded_entity_partial_match_weight
        weights.append(entity_weight)

    if cue.predicate is not None:
        similarity = _normalised_equality(cue.predicate, candidate.predicate)
        mismatches.append(similarity - 1.0)
        weights.append(_PARTIAL_MATCH_WEIGHTS["predicate"])

    if cue.object_value is not None:
        similarity = _normalised_equality(cue.object_value, candidate.object_value)
        mismatches.append(similarity - 1.0)
        weights.append(_PARTIAL_MATCH_WEIGHTS["object"])

    if cue.qualifiers:
        similarity = _qualifier_similarity(cue.qualifiers, candidate.qualifiers)
        mismatches.append(similarity - 1.0)
        weights.append(_PARTIAL_MATCH_WEIGHTS["qualifiers"])

    if not mismatches:
        return 0.0

    average_mismatch = sum(m * w for m, w in zip(mismatches, weights, strict=True)) / sum(weights)
    return config.mismatch_penalty * average_mismatch


def _text_query_coverage(
    query: str,
    candidate_text: str,
    *,
    idf_weights: Mapping[str, float] | None = None,
) -> float:
    query_tokens, candidate_tokens, matched = _token_overlap(query, candidate_text)
    if not query_tokens or not candidate_tokens or not matched:
        return 0.0
    if idf_weights is None:
        return len(matched) / len(query_tokens)
    weighted_match = sum(idf_weights.get(token, 1.0) for token in matched)
    weighted_total = sum(idf_weights.get(token, 1.0) for token in query_tokens)
    if weighted_total <= 0.0:
        return 0.0
    return weighted_match / weighted_total


def _text_cue_fit(
    query: str,
    candidate_text: str,
    *,
    idf_weights: Mapping[str, float] | None = None,
) -> float:
    query_tokens, candidate_tokens, matched = _token_overlap(query, candidate_text)
    if not query_tokens or not candidate_tokens or not matched:
        return 0.0
    if idf_weights is None:
        query_weight = float(len(query_tokens))
        candidate_weight = float(len(candidate_tokens))
        matched_weight = float(len(matched))
    else:
        query_weight = sum(idf_weights.get(token, 1.0) for token in query_tokens)
        candidate_weight = sum(idf_weights.get(token, 1.0) for token in candidate_tokens)
        matched_weight = sum(idf_weights.get(token, 1.0) for token in matched)
    if query_weight <= 0.0 or candidate_weight <= 0.0:
        return 0.0
    recall = matched_weight / query_weight
    precision = matched_weight / candidate_weight
    denominator = precision + recall
    if denominator <= 0.0:
        return 0.0
    return 2.0 * precision * recall / denominator


def _token_overlap(
    query: str,
    candidate_text: str,
) -> tuple[set[str], set[str], set[str]]:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return set(), set(), set()
    candidate_tokens = _tokenize(candidate_text)
    if not candidate_tokens:
        return query_tokens, set(), set()
    return query_tokens, candidate_tokens, query_tokens.intersection(candidate_tokens)


def _candidate_idf_weights(
    candidates: Sequence[ActivationCandidate],
) -> dict[str, float]:
    document_count = len(candidates)
    if document_count == 0:
        return {}
    frequencies: dict[str, int] = {}
    for candidate in candidates:
        for token in _tokenize(candidate.text):
            frequencies[token] = frequencies.get(token, 0) + 1
    return {
        token: math.log((document_count + 1) / (count + 1)) + 1.0
        for token, count in frequencies.items()
    }


def _cue_requests_current_state(cue: RetrievalCue, config: ActivationConfig) -> bool:
    if not cue.text or not cue.text.strip():
        return False
    cue_tokens = _tokenize(cue.text)
    return bool(cue_tokens.intersection(config.current_state_cue_tokens))


def _current_state_activation(
    candidate: ActivationCandidate,
    *,
    current_state_cue: bool,
    config: ActivationConfig,
    episode_support_index: Mapping[str, frozenset[SemanticMemoryStatus]],
    current_state_policy_active: bool,
    matched_slot_identities: set[MemoryIdentity],
    matched_support_episode_ids: set[str],
) -> float:
    if not current_state_policy_active:
        return 0.0
    if candidate.memory_kind is MemoryKind.SEMANTIC:
        if candidate.identity not in matched_slot_identities:
            return 0.0
    elif candidate.memory_kind is MemoryKind.EPISODE and isinstance(
        candidate.memory, StoredEpisode
    ):
        if candidate.memory.id not in matched_support_episode_ids:
            return 0.0
    else:
        return 0.0
    if candidate.semantic_status is SemanticMemoryStatus.SUPERSEDED:
        return -config.current_state_weight
    bonus = 0.0
    if candidate.semantic_status is SemanticMemoryStatus.ACTIVE:
        bonus += config.current_state_weight * 0.5
    if candidate.last_supported_at is not None and candidate.slot_key is not None:
        bonus += config.current_state_weight * 0.25
    if current_state_cue and candidate.semantic_status is SemanticMemoryStatus.ACTIVE:
        bonus += config.current_state_weight
    if current_state_cue and candidate.slot_key is not None and candidate.semantic_status is None:
        bonus -= config.current_state_weight * 0.25

    if (
        current_state_cue
        and candidate.memory_kind is MemoryKind.EPISODE
        and isinstance(candidate.memory, StoredEpisode)
    ):
        statuses = episode_support_index.get(candidate.memory.id, frozenset())
        if SemanticMemoryStatus.ACTIVE in statuses:
            bonus += config.current_state_weight
        if SemanticMemoryStatus.SUPERSEDED in statuses:
            bonus -= config.current_state_weight

    return bonus


def _collapse_results(
    results: Sequence[RecallResult],
    *,
    limit: int,
    config: ActivationConfig,
    support_index: Mapping[str, frozenset[SemanticMemoryStatus]],
    slot_index: Mapping[str, str],
) -> list[RecallResult]:
    selected: list[RecallResult] = []
    for result in results:
        if len(selected) >= limit:
            break
        if any(
            _should_collapse(
                existing,
                result,
                config=config,
                support_index=support_index,
                slot_index=slot_index,
            )
            for existing in selected
        ):
            continue
        selected.append(result)
    return selected


def _should_collapse(
    left: RecallResult,
    right: RecallResult,
    *,
    config: ActivationConfig,
    support_index: Mapping[str, frozenset[SemanticMemoryStatus]],
    slot_index: Mapping[str, str],
) -> bool:
    if config.enable_duplicate_collapse and _results_near_duplicate(left, right, config=config):
        return True
    if config.collapse_same_slot_support and _results_same_slot_support(
        left, right, slot_index=slot_index
    ):
        return True
    return False


def _results_near_duplicate(
    left: RecallResult,
    right: RecallResult,
    *,
    config: ActivationConfig,
) -> bool:
    left_fingerprint = _content_fingerprint_from_result(left)
    right_fingerprint = _content_fingerprint_from_result(right)
    if left_fingerprint and left_fingerprint == right_fingerprint:
        return True
    left_tokens = _collapse_jaccard_tokens(_tokenize(_result_text(left)), config=config)
    right_tokens = _collapse_jaccard_tokens(_tokenize(_result_text(right)), config=config)
    if not left_tokens or not right_tokens:
        return False
    intersection = len(left_tokens.intersection(right_tokens))
    union = len(left_tokens.union(right_tokens))
    if union == 0:
        return False
    return (intersection / union) >= config.duplicate_jaccard_threshold


def _results_same_slot_support(
    left: RecallResult,
    right: RecallResult,
    *,
    slot_index: Mapping[str, str],
) -> bool:
    left_key = _slot_support_group(left, slot_index)
    right_key = _slot_support_group(right, slot_index)
    if left_key is None or right_key is None or left_key != right_key:
        return False
    left_active_semantic = _is_active_semantic(left)
    right_active_semantic = _is_active_semantic(right)
    left_support_episode = _is_slot_support_episode(left, slot_index)
    right_support_episode = _is_slot_support_episode(right, slot_index)
    if left_active_semantic and right_support_episode:
        return False
    if right_active_semantic and left_support_episode:
        return False
    preferred = _preferred_same_slot_result(left, right)
    return preferred is left


def _is_slot_support_episode(
    result: RecallResult,
    slot_index: Mapping[str, str],
) -> bool:
    memory = result.memory
    if not isinstance(memory, StoredEpisode):
        return False
    return memory.id in slot_index


def _slot_support_group(
    result: RecallResult,
    slot_index: Mapping[str, str],
) -> str | None:
    memory = result.memory
    if isinstance(memory, StoredSemanticMemory):
        return memory.slot_key
    if isinstance(memory, StoredEpisode):
        return slot_index.get(memory.id)
    return None


def _preferred_same_slot_result(left: RecallResult, right: RecallResult) -> RecallResult:
    left_semantic_active = _is_active_semantic(left)
    right_semantic_active = _is_active_semantic(right)
    if left_semantic_active and not right_semantic_active:
        return left
    if right_semantic_active and not left_semantic_active:
        return right
    if left.activation >= right.activation:
        return left
    return right


def _is_active_semantic(result: RecallResult) -> bool:
    memory = result.memory
    return isinstance(memory, StoredSemanticMemory) and memory.status is SemanticMemoryStatus.ACTIVE


def _collapse_near_duplicates(
    results: Sequence[RecallResult],
    *,
    limit: int,
    config: ActivationConfig,
) -> list[RecallResult]:
    return _collapse_results(
        results,
        limit=limit,
        config=config,
        support_index={},
        slot_index={},
    )


def _collapse_jaccard_tokens(tokens: set[str], *, config: ActivationConfig) -> set[str]:
    if not config.collapse_normalize_numeric_tokens:
        return tokens
    return {token for token in tokens if not token.isdigit()}


def _content_fingerprint_from_result(result: RecallResult) -> str | None:
    memory = result.memory
    metadata = memory.metadata
    if isinstance(memory, StoredEpisode):
        fingerprint = metadata.get("episode", {}).get("content_fingerprint")
    else:
        fingerprint = metadata.get("semantic", {}).get("content_fingerprint")
    return fingerprint if isinstance(fingerprint, str) and fingerprint else None


def _result_text(result: RecallResult) -> str:
    memory = result.memory
    if isinstance(memory, StoredEpisode):
        return memory.statement
    return memory.statement


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_PATTERN.findall(text)}


def _predicate_tokens(predicate: str | None) -> set[str]:
    if predicate is None:
        return set()
    normalised = _normalise_text(predicate).replace("_", " ")
    return _tokenize(normalised)


def _semantic_lexical_tokens(
    predicate: str | None,
    object_value: str | None,
    text: str,
) -> set[str]:
    tokens: set[str] = set()
    tokens.update(_predicate_tokens(predicate))
    tokens.update(_tokenize(object_value or ""))
    tokens.update(_tokenize(text))
    return tokens


def _semantic_memory_valid_at(memory: StoredSemanticMemory, at: datetime) -> bool:
    if memory.valid_from is not None and at < memory.valid_from:
        return False
    if memory.valid_until is not None and at >= memory.valid_until:
        return False
    return True


def _semantic_eligible_for_soft_admission(
    candidate: ActivationCandidate,
    *,
    valid_at: datetime | None,
    as_of: datetime,
) -> bool:
    if candidate.memory_kind is not MemoryKind.SEMANTIC:
        return False
    memory = candidate.memory
    if not isinstance(memory, StoredSemanticMemory):
        return False
    evaluation = valid_at if valid_at is not None else as_of
    if valid_at is None:
        return candidate.semantic_status is SemanticMemoryStatus.ACTIVE
    if candidate.semantic_status is SemanticMemoryStatus.CONTESTED:
        return False
    return _semantic_memory_valid_at(memory, evaluation)


def _inspection_disposition_for_rejected(
    result: RecallResult,
    *,
    admission_outcome: _SemanticAdmissionOutcome,
) -> RecallInspectionDisposition:
    identity = _result_identity(result)
    if identity in admission_outcome.below_soft_floor:
        return RecallInspectionDisposition.FILTERED_BELOW_SOFT_FLOOR
    if identity in admission_outcome.capped_out:
        return RecallInspectionDisposition.BELOW_THRESHOLD
    if (
        result.memory_kind is MemoryKind.SEMANTIC
        and identity not in admission_outcome.matched_semantics
    ):
        return RecallInspectionDisposition.FILTERED_INSUFFICIENT_RELEVANCE
    return RecallInspectionDisposition.BELOW_THRESHOLD


def _jaccard_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    intersection = len(left_set.intersection(right_set))
    union = len(left_set.union(right_set))
    return intersection / union


def _normalised_equality(left: str | None, right: str | None) -> float:
    if left is None and right is None:
        return 1.0
    if left is None or right is None:
        return 0.0
    return float(_normalise_text(left) == _normalise_text(right))


def _normalise_text(value: str) -> str:
    normalised = unicodedata.normalize("NFKC", value)
    normalised = _WHITESPACE_PATTERN.sub(" ", normalised).strip()
    return normalised.casefold()


def _qualifier_similarity(
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> float:
    left_pairs = {_qualifier_pair(key, value) for key, value in left.items()}
    right_pairs = {_qualifier_pair(key, value) for key, value in right.items()}
    if not left_pairs and not right_pairs:
        return 1.0
    if not left_pairs or not right_pairs:
        return 0.0
    intersection = len(left_pairs.intersection(right_pairs))
    union = len(left_pairs.union(right_pairs))
    return intersection / union


def _qualifier_pair(key: object, value: object) -> tuple[str, str]:
    return (_normalise_text(str(key)), _normalise_text(str(value)))
