"""ACT-R declarative activation for episodic and semantic memories."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Protocol

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
    RecallResult,
    RetrievalCue,
    SemanticDerivationRelation,
    SemanticMemoryStatus,
    StoredEpisode,
    StoredSemanticMemory,
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


def activation_candidate_from_episode(episode: StoredEpisode) -> ActivationCandidate:
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
    )


def activation_candidate_from_semantic(memory: StoredSemanticMemory) -> ActivationCandidate:
    """Adapt a stored semantic memory for declarative activation."""
    entity_ids = {entity.entity_id for entity in memory.entities}
    if memory.subject_entity_id:
        entity_ids.add(memory.subject_entity_id)
    if memory.object_entity_id:
        entity_ids.add(memory.object_entity_id)
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
        support_index = episode_support_index or {}
        slot_index = episode_slot_index or {}
        slot_admission_active = _slot_admission_active(
            cue,
            config=config,
            current_state_cue=current_state_cue,
        )

        scored: list[RecallResult] = []
        for candidate in candidates:
            scored.append(
                _score_candidate(
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
                    slot_admission_active=slot_admission_active,
                )
            )

        admitted_identities = _semantic_slot_admission_identities(
            candidates,
            cue,
            config=config,
            seeded_entity_ids=seeded_entity_ids,
            current_state_cue=current_state_cue,
        )

        scored_by_identity = {_result_identity(result): result for result in scored}
        ordered: list[RecallResult] = []
        seen_identities: set[MemoryIdentity] = set()

        for result in sorted(
            (
                scored_by_identity[identity]
                for identity in admitted_identities
                if identity in scored_by_identity
            ),
            key=lambda item: (
                -item.activation,
                item.memory_kind.value,
                _memory_key(item),
            ),
        ):
            ordered.append(result)
            seen_identities.add(_result_identity(result))

        for result in sorted(
            scored,
            key=lambda item: (
                -item.activation,
                item.memory_kind.value,
                _memory_key(item),
            ),
        ):
            identity = _result_identity(result)
            if identity in seen_identities:
                continue
            if result.activation >= config.retrieval_threshold:
                ordered.append(result)
                seen_identities.add(identity)

        if config.exclude_superseded_support_on_current_state and slot_admission_active:
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
) -> bool:
    if not config.enable_semantic_slot_admission:
        return False
    if config.force_slot_admission:
        return True
    if not config.slot_admission_requires_current_state_or_predicate:
        return True
    return current_state_cue or cue.predicate is not None


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
    slot_admission_active: bool,
) -> RecallResult:
    identity = candidate.identity
    stored_traces = references.get(identity, ())
    creation_trace = ActivationReferenceTrace(
        referenced_at=candidate.created_at,
        weight=1,
    )
    reference_traces = (creation_trace, *stored_traces)
    base_level = calculate_base_level(
        reference_traces,
        as_of=as_of,
        decay=config.decay,
        constant=config.base_level_constant,
        time_unit_seconds=config.time_unit_seconds,
        minimum_elapsed_seconds=config.minimum_elapsed_seconds,
    )
    partial_match = (
        _calculate_partial_match(
            candidate,
            cue,
            config,
            idf_weights=idf_weights,
            effective_entity_ids=effective_entity_ids,
        )
        if config.enable_partial_matching
        else 0.0
    )
    current_state = _current_state_activation(
        candidate,
        current_state_cue=current_state_cue,
        config=config,
        episode_support_index=episode_support_index,
        slot_admission_active=slot_admission_active,
    )
    conjunction = _multi_entity_conjunction_bonus(
        candidate,
        effective_entity_ids=effective_entity_ids,
        config=config,
    )
    noise = 0.0
    activation = base_level + spreading + partial_match + current_state + conjunction + noise
    latency_seconds = config.latency_factor * math.exp(-config.latency_exponent * activation)
    score = _presentation_score(activation, config.retrieval_threshold)
    components = ActivationComponents(
        base_level=base_level,
        spreading=spreading,
        partial_match=partial_match + conjunction,
        noise=noise,
        total=activation,
        current_state=current_state,
    )
    matched_entities = len(set(effective_entity_ids).intersection(candidate.entity_ids))
    reason = _build_reason(
        activation=activation,
        base_level=base_level,
        spreading=spreading,
        partial_match=partial_match,
        reference_count=len(reference_traces),
        matched_entities=matched_entities,
        cue_entity_count=len(effective_entity_ids),
        spreading_metadata=spreading_metadata,
        current_state=current_state,
        conjunction=conjunction,
    )
    return RecallResult(
        memory_kind=candidate.memory_kind,
        memory=candidate.memory,
        activation=activation,
        score=score,
        latency_seconds=latency_seconds,
        components=components,
        reason=reason,
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
) -> str:
    reason = (
        f"activation={activation:.3f}; base={base_level:.3f}; "
        f"spread={spreading:.3f}; partial={partial_match:.3f}; "
        f"references={reference_count}; "
        f"matched_entities={matched_entities}/{cue_entity_count}"
    )
    if current_state != 0.0:
        reason += f"; current_state={current_state:.3f}"
    if conjunction != 0.0:
        reason += f"; conjunction={conjunction:.3f}"
    if spreading > 0.0 and spreading_metadata is not None:
        reason += (
            f"; spread_hop={spreading_metadata.hop}; "
            f"spread_sources={','.join(spreading_metadata.sources)}"
        )
        if spreading_metadata.learned_association_count > 0:
            reason += f"; learned_edges={spreading_metadata.learned_association_count}"
    return reason


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
) -> set[MemoryIdentity]:
    if not _slot_admission_active(cue, config=config, current_state_cue=current_state_cue):
        return set()

    effective_sources = cue.entity_ids if cue.entity_ids else seeded_entity_ids
    cue_tokens = _tokenize(cue.text) if cue.text else set()
    distinctive_tokens = cue_tokens - config.current_state_cue_tokens

    episode_by_id: dict[str, ActivationCandidate] = {}
    for candidate in candidates:
        if candidate.memory_kind is MemoryKind.EPISODE and isinstance(
            candidate.memory, StoredEpisode
        ):
            episode_by_id[candidate.memory.id] = candidate

    admitted: set[MemoryIdentity] = set()
    for candidate in candidates:
        if candidate.semantic_status is not SemanticMemoryStatus.ACTIVE:
            continue
        if cue.predicate is not None:
            if _normalised_equality(cue.predicate, candidate.predicate) < 1.0:
                continue
        elif effective_sources:
            if not set(effective_sources).intersection(candidate.entity_ids):
                continue
        elif current_state_cue and distinctive_tokens:
            statement_tokens = _tokenize(candidate.text)
            object_tokens = _tokenize(candidate.object_value or "")
            if not distinctive_tokens.intersection(statement_tokens.union(object_tokens)):
                continue
        else:
            continue
        admitted.add(candidate.identity)

    for candidate in candidates:
        if candidate.semantic_status is not SemanticMemoryStatus.ACTIVE:
            continue
        memory = candidate.memory
        if not isinstance(memory, StoredSemanticMemory):
            continue
        if candidate.identity not in admitted:
            continue
        for derivation in memory.derivations:
            if derivation.relation is not SemanticDerivationRelation.SUPPORTS:
                continue
            episode_candidate = episode_by_id.get(derivation.episode_id)
            if episode_candidate is not None:
                admitted.add(episode_candidate.identity)

    return admitted


def _calculate_partial_match(
    candidate: ActivationCandidate,
    cue: RetrievalCue,
    config: ActivationConfig,
    *,
    idf_weights: Mapping[str, float] | None = None,
    effective_entity_ids: Sequence[str] = (),
) -> float:
    mismatches: list[float] = []
    weights: list[float] = []

    if cue.text and cue.text.strip():
        similarity = _text_similarity(cue.text, candidate.text, idf_weights=idf_weights)
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


def _text_similarity(
    query: str,
    candidate_text: str,
    *,
    idf_weights: Mapping[str, float] | None = None,
) -> float:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0
    candidate_tokens = _tokenize(candidate_text)
    if not candidate_tokens:
        return 0.0
    matched = query_tokens.intersection(candidate_tokens)
    if not matched:
        return 0.0
    if idf_weights is None:
        return len(matched) / len(query_tokens)
    weighted_match = sum(idf_weights.get(token, 1.0) for token in matched)
    weighted_total = sum(idf_weights.get(token, 1.0) for token in query_tokens)
    if weighted_total <= 0.0:
        return 0.0
    return weighted_match / weighted_total


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
    if cue.predicate is not None or cue.object_value is not None:
        return False
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
    slot_admission_active: bool,
) -> float:
    if candidate.semantic_status is SemanticMemoryStatus.SUPERSEDED:
        return -config.current_state_weight
    bonus = 0.0
    if candidate.semantic_status is SemanticMemoryStatus.ACTIVE:
        bonus += config.current_state_weight * 0.5
    if candidate.last_supported_at is not None and candidate.slot_key is not None:
        bonus += config.current_state_weight * 0.25
    if current_state_cue and candidate.semantic_status is SemanticMemoryStatus.ACTIVE:
        bonus += config.current_state_weight
    if (
        current_state_cue
        and candidate.slot_key is not None
        and candidate.semantic_status is None
        and slot_admission_active
    ):
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
