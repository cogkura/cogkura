"""Deterministic cue relevance and coverage for retrieval monitoring."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence

from cogkura.models import (
    RecallResult,
    RetrievalCue,
    StoredEpisode,
    StoredSemanticMemory,
)

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def calculate_cue_relevance(recall: RecallResult, cue: RetrievalCue) -> float:
    """Compute mean cue-relevance across supplied cue fields for one memory."""
    memory = recall.memory
    text, subject_id, entity_ids, predicate, object_value, qualifiers = _memory_cue_fields(memory)
    components: list[float] = []

    if cue.text and cue.text.strip():
        components.append(_text_coverage(cue.text, text))

    if cue.subject_id and cue.subject_id.strip():
        goal_subject = cue.subject_id.strip()
        candidate_subject = subject_id.strip() if subject_id else None
        components.append(1.0 if goal_subject == candidate_subject else 0.0)

    if cue.entity_ids:
        goal_entities = set(cue.entity_ids)
        candidate_entities = set(entity_ids)
        matched = len(goal_entities.intersection(candidate_entities))
        components.append(matched / len(goal_entities))

    if cue.predicate and cue.predicate.strip():
        goal_predicate = _normalise_text(cue.predicate)
        candidate_predicate = _normalise_text(predicate) if predicate else ""
        components.append(1.0 if goal_predicate == candidate_predicate else 0.0)

    if cue.object_value and cue.object_value.strip():
        goal_object = cue.object_value
        if object_value and _normalise_text(goal_object) == _normalise_text(object_value):
            components.append(1.0)
        else:
            components.append(_text_coverage(goal_object, text))

    if cue.qualifiers:
        components.append(_qualifier_coverage(cue.qualifiers, qualifiers))

    if not components:
        return 1.0

    return sum(components) / len(components)


def calculate_cue_coverage(
    candidates: Sequence[RecallResult],
    cue: RetrievalCue,
) -> float:
    """Compute assessment-wide cue coverage across retrieved memories."""
    if not candidates:
        return 0.0

    components: list[float] = []

    if cue.text and cue.text.strip():
        goal_tokens = _tokenize(cue.text)
        if not goal_tokens:
            components.append(0.0)
        else:
            covered: set[str] = set()
            for recall in candidates:
                text, _, _, _, _, _ = _memory_cue_fields(recall.memory)
                covered.update(_tokenize(text))
            components.append(len(goal_tokens.intersection(covered)) / len(goal_tokens))

    if cue.subject_id and cue.subject_id.strip():
        goal_subject = cue.subject_id.strip()
        best = 0.0
        for recall in candidates:
            _, subject_id, _, _, _, _ = _memory_cue_fields(recall.memory)
            candidate_subject = subject_id.strip() if subject_id else None
            if goal_subject == candidate_subject:
                best = 1.0
                break
        components.append(best)

    if cue.entity_ids:
        goal_entities = set(cue.entity_ids)
        best = 0.0
        for recall in candidates:
            _, _, entity_ids, _, _, _ = _memory_cue_fields(recall.memory)
            matched = len(goal_entities.intersection(set(entity_ids)))
            best = max(best, matched / len(goal_entities))
        components.append(best)

    if cue.predicate and cue.predicate.strip():
        goal_predicate = _normalise_text(cue.predicate)
        best = 0.0
        for recall in candidates:
            _, _, _, predicate, _, _ = _memory_cue_fields(recall.memory)
            candidate_predicate = _normalise_text(predicate) if predicate else ""
            if goal_predicate == candidate_predicate:
                best = 1.0
                break
        components.append(best)

    if cue.object_value and cue.object_value.strip():
        goal_object = cue.object_value
        best = 0.0
        for recall in candidates:
            text, _, _, _, object_value, _ = _memory_cue_fields(recall.memory)
            if object_value and _normalise_text(goal_object) == _normalise_text(object_value):
                best = 1.0
            else:
                best = max(best, _text_coverage(goal_object, text))
        components.append(best)

    if cue.qualifiers:
        best = 0.0
        for recall in candidates:
            _, _, _, _, _, qualifiers = _memory_cue_fields(recall.memory)
            best = max(best, _qualifier_coverage(cue.qualifiers, qualifiers))
        components.append(best)

    if not components:
        return 1.0

    return sum(components) / len(components)


def _memory_cue_fields(
    memory: StoredEpisode | StoredSemanticMemory,
) -> tuple[str, str | None, tuple[str, ...], str | None, str | None, Mapping[str, object]]:
    if isinstance(memory, StoredEpisode):
        entity_ids = tuple(sorted({entity.entity_id for entity in memory.entities}))
        return (
            memory.statement,
            memory.subject_id,
            entity_ids,
            None,
            None,
            memory.metadata,
        )
    entity_id_set = {entity.entity_id for entity in memory.entities}
    if memory.subject_entity_id:
        entity_id_set.add(memory.subject_entity_id)
    if memory.object_entity_id:
        entity_id_set.add(memory.object_entity_id)
    return (
        memory.statement,
        memory.subject_id,
        tuple(sorted(entity_id_set)),
        memory.predicate,
        memory.object_value,
        memory.qualifiers,
    )


def _text_coverage(goal_text: str, candidate_text: str) -> float:
    goal_tokens = _tokenize(goal_text)
    if not goal_tokens:
        return 0.0
    candidate_tokens = _tokenize(candidate_text)
    if not candidate_tokens:
        return 0.0
    matched = goal_tokens.intersection(candidate_tokens)
    return len(matched) / len(goal_tokens)


def _qualifier_coverage(
    goal_qualifiers: Mapping[str, object],
    candidate_qualifiers: Mapping[str, object],
) -> float:
    goal_pairs = {_qualifier_pair(key, value) for key, value in goal_qualifiers.items()}
    if not goal_pairs:
        return 1.0
    candidate_pairs = {_qualifier_pair(key, value) for key, value in candidate_qualifiers.items()}
    matched = len(goal_pairs.intersection(candidate_pairs))
    return matched / len(goal_pairs)


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_PATTERN.findall(text)}


tokenize = _tokenize


def _normalise_text(value: str) -> str:
    normalised = unicodedata.normalize("NFKC", value)
    normalised = _WHITESPACE_PATTERN.sub(" ", normalised).strip()
    return normalised.casefold()


def _qualifier_pair(key: object, value: object) -> tuple[str, str]:
    return (_normalise_text(str(key)), _normalise_text(str(value)))
