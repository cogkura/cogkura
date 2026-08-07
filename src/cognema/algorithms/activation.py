"""ACT-R declarative activation for episodic and semantic memories."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Protocol

from cognema.exceptions import ValidationError
from cognema.models import (
    ActivationCandidate,
    ActivationComponents,
    ActivationConfig,
    MemoryIdentity,
    MemoryKind,
    RecallResult,
    RetrievalCue,
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
        references: Mapping[MemoryIdentity, Sequence[datetime]],
        as_of: datetime,
        config: ActivationConfig,
        limit: int,
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
    reference_times: Sequence[datetime],
    *,
    as_of: datetime,
    decay: float,
    constant: float,
    time_unit_seconds: float,
    minimum_elapsed_seconds: float,
) -> float:
    """Compute ACT-R base-level activation from reference timestamps."""
    terms: list[float] = []
    for referenced_at in reference_times:
        elapsed_seconds = (as_of - referenced_at).total_seconds()
        if elapsed_seconds < 0:
            raise ValidationError("Memory reference cannot occur after the activation timestamp.")
        elapsed_units = max(elapsed_seconds, minimum_elapsed_seconds) / time_unit_seconds
        terms.append(-decay * math.log(elapsed_units))
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
    )


class ACTRDeclarativeActivator:
    """Deterministic ACT-R declarative activation (base-level + partial matching)."""

    def rank(
        self,
        *,
        candidates: Sequence[ActivationCandidate],
        cue: RetrievalCue,
        references: Mapping[MemoryIdentity, Sequence[datetime]],
        as_of: datetime,
        config: ActivationConfig,
        limit: int,
    ) -> list[RecallResult]:
        fan_by_entity = _build_entity_fan(candidates)
        context_sources = _context_sources(cue)
        results: list[RecallResult] = []

        for candidate in candidates:
            identity = candidate.identity
            stored_refs = references.get(identity, ())
            reference_times = (candidate.created_at, *stored_refs)
            base_level = calculate_base_level(
                reference_times,
                as_of=as_of,
                decay=config.decay,
                constant=config.base_level_constant,
                time_unit_seconds=config.time_unit_seconds,
                minimum_elapsed_seconds=config.minimum_elapsed_seconds,
            )
            spreading = (
                _calculate_spreading(candidate, context_sources, fan_by_entity, config)
                if config.enable_spreading_activation
                else 0.0
            )
            partial_match = (
                _calculate_partial_match(candidate, cue, config)
                if config.enable_partial_matching
                else 0.0
            )
            noise = 0.0
            activation = base_level + spreading + partial_match + noise
            if activation < config.retrieval_threshold:
                continue

            latency_seconds = config.latency_factor * math.exp(
                -config.latency_exponent * activation
            )
            score = _presentation_score(activation, config.retrieval_threshold)
            components = ActivationComponents(
                base_level=base_level,
                spreading=spreading,
                partial_match=partial_match,
                noise=noise,
                total=activation,
            )
            matched_entities = len(set(cue.entity_ids).intersection(candidate.entity_ids))
            reason = (
                f"activation={activation:.3f}; base={base_level:.3f}; "
                f"spread={spreading:.3f}; partial={partial_match:.3f}; "
                f"references={len(reference_times)}; "
                f"matched_entities={matched_entities}/{len(cue.entity_ids)}"
            )
            results.append(
                RecallResult(
                    memory_kind=candidate.memory_kind,
                    memory=candidate.memory,
                    activation=activation,
                    score=score,
                    latency_seconds=latency_seconds,
                    components=components,
                    reason=reason,
                )
            )

        results.sort(
            key=lambda item: (
                -item.activation,
                item.memory_kind.value,
                _memory_key(item),
            )
        )
        return results[:limit]


def _memory_key(result: RecallResult) -> str:
    memory = result.memory
    if isinstance(memory, StoredEpisode):
        return memory.memory_key
    return memory.memory_key


def _presentation_score(activation: float, threshold: float) -> float:
    return 1.0 / (1.0 + math.exp(-(activation - threshold)))


def _build_entity_fan(candidates: Sequence[ActivationCandidate]) -> dict[str, int]:
    fan: dict[str, int] = {}
    for candidate in candidates:
        for entity_id in candidate.entity_ids:
            fan[entity_id] = fan.get(entity_id, 0) + 1
        if candidate.subject_id:
            fan[candidate.subject_id] = fan.get(candidate.subject_id, 0) + 1
    return fan


def _context_sources(cue: RetrievalCue) -> tuple[str, ...]:
    sources: list[str] = []
    seen: set[str] = set()
    for entity_id in cue.entity_ids:
        if entity_id in seen:
            continue
        seen.add(entity_id)
        sources.append(entity_id)
    if cue.subject_id and cue.subject_id not in seen:
        sources.append(cue.subject_id)
    return tuple(sources)


def _calculate_spreading(
    candidate: ActivationCandidate,
    context_sources: tuple[str, ...],
    fan_by_entity: Mapping[str, int],
    config: ActivationConfig,
) -> float:
    if not context_sources:
        return 0.0
    source_weight = config.source_activation / len(context_sources)
    total = 0.0
    candidate_entities = set(candidate.entity_ids)
    if candidate.subject_id:
        candidate_entities.add(candidate.subject_id)
    for source_id in context_sources:
        if source_id not in candidate_entities:
            continue
        fan = fan_by_entity.get(source_id, 1)
        association_strength = config.maximum_associative_strength - math.log(fan)
        total += source_weight * association_strength
    return total


def _calculate_partial_match(
    candidate: ActivationCandidate,
    cue: RetrievalCue,
    config: ActivationConfig,
) -> float:
    mismatches: list[float] = []
    weights: list[float] = []

    if cue.text and cue.text.strip():
        similarity = _text_similarity(cue.text, candidate.text)
        mismatches.append(similarity - 1.0)
        weights.append(_PARTIAL_MATCH_WEIGHTS["text"])

    if cue.subject_id is not None:
        similarity = float(cue.subject_id == candidate.subject_id)
        mismatches.append(similarity - 1.0)
        weights.append(_PARTIAL_MATCH_WEIGHTS["subject"])

    if cue.entity_ids:
        similarity = _jaccard_similarity(cue.entity_ids, candidate.entity_ids)
        mismatches.append(similarity - 1.0)
        weights.append(_PARTIAL_MATCH_WEIGHTS["entities"])

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


def _text_similarity(query: str, candidate_text: str) -> float:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0
    candidate_tokens = _tokenize(candidate_text)
    if not candidate_tokens:
        return 0.0
    matched = query_tokens.intersection(candidate_tokens)
    return len(matched) / len(query_tokens)


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
