"""Semantic consolidation and metadata extraction."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, Protocol

from cogkura.exceptions import ValidationError
from cogkura.models import (
    EpisodeEntity,
    EpisodeEvidenceInput,
    SemanticCardinality,
    SemanticDerivationInput,
    SemanticDerivationRelation,
    SemanticExtractionResult,
    SemanticFactCandidate,
    SemanticMemoryInput,
    SemanticMemoryStatus,
    SemanticPolarity,
    StoredEpisode,
)
from cogkura.observations.models import StoredObservation

_CONSOLIDATION_VERSION = "cls-deterministic-v1"
_EXTRACTOR_VERSION = "metadata-v1"
_SUBJECT_PLACEHOLDER = "__none__"
_WHITESPACE_PATTERN = re.compile(r"\s+")


class SemanticConsolidator(Protocol):
    """Consolidates extracted facts into semantic memory candidates."""

    def consolidate(
        self,
        episodes: Sequence[StoredEpisode],
        candidates: Sequence[SemanticFactCandidate],
    ) -> list[SemanticMemoryInput]:
        """Consolidate extracted facts into semantic memory candidates."""


class SemanticExtractor(Protocol):
    """Extracts atomic semantic candidates from episodic evidence."""

    async def extract(
        self,
        episodes: Sequence[StoredEpisode],
        *,
        observations: Mapping[str, StoredObservation],
    ) -> SemanticExtractionResult:
        """Extract atomic semantic candidates from episodic evidence."""


@dataclass(frozen=True, slots=True)
class _CanonicalClaim:
    tenant_id: str
    subject_entity_id: str
    predicate: str
    object_value: str
    object_entity_id: str | None
    polarity: SemanticPolarity
    cardinality: SemanticCardinality
    qualifiers: Mapping[str, Any]
    slot_key: str
    claim_key: str


class ComplementaryLearningSemanticConsolidator:
    """Recurrence-based episodic-to-semantic consolidation."""

    def __init__(
        self,
        *,
        minimum_supporting_episodes: int = 2,
        recurrence_tau: float = 1.5,
        source_diversity_target: int = 2,
        contested_ratio_threshold: float = 0.25,
        consolidation_version: str = _CONSOLIDATION_VERSION,
        extractor_version: str = _EXTRACTOR_VERSION,
    ) -> None:
        if minimum_supporting_episodes < 1:
            raise ValidationError("minimum_supporting_episodes must be at least 1.")
        if recurrence_tau <= 0:
            raise ValidationError("recurrence_tau must be greater than zero.")
        if source_diversity_target < 1:
            raise ValidationError("source_diversity_target must be at least 1.")
        if not 0.0 <= contested_ratio_threshold <= 1.0:
            raise ValidationError("contested_ratio_threshold must be between 0.0 and 1.0.")
        self._minimum_supporting_episodes = minimum_supporting_episodes
        self._recurrence_tau = recurrence_tau
        self._source_diversity_target = source_diversity_target
        self._contested_ratio_threshold = contested_ratio_threshold
        self._consolidation_version = consolidation_version
        self._extractor_version = extractor_version

    def consolidate(
        self,
        episodes: Sequence[StoredEpisode],
        candidates: Sequence[SemanticFactCandidate],
    ) -> list[SemanticMemoryInput]:
        episode_by_id = {episode.id: episode for episode in episodes}
        prepared = _prepare_candidates(candidates, episode_by_id)
        slot_groups: dict[str, dict[str, list[tuple[SemanticFactCandidate, StoredEpisode]]]] = {}
        for candidate, episode in prepared:
            canonical = _canonicalize_candidate(candidate)
            slot_groups.setdefault(canonical.slot_key, {}).setdefault(
                canonical.claim_key, []
            ).append((candidate, episode))

        results: list[SemanticMemoryInput] = []
        for slot_key in sorted(slot_groups):
            claim_map = slot_groups[slot_key]
            claim_entries = {
                claim_key: _build_claim_aggregate(claim_key, items)
                for claim_key, items in claim_map.items()
            }
            for claim_key in sorted(claim_entries):
                aggregate = claim_entries[claim_key]
                if len(aggregate.supporting_episodes) < self._minimum_supporting_episodes:
                    continue
                contradicting = _find_contradictions(claim_key, aggregate, claim_entries)
                memory = self._build_memory(
                    aggregate,
                    contradicting,
                    episode_by_id,
                )
                results.append(memory)
        return results

    def _build_memory(
        self,
        aggregate: _ClaimAggregate,
        contradicting: list[_EpisodeSupport],
        episode_by_id: Mapping[str, StoredEpisode],
    ) -> SemanticMemoryInput:
        support_mass = sum(item.weight for item in aggregate.supporting_episodes)
        contradiction_mass = sum(item.weight for item in contradicting)
        confidence = _calculate_confidence(
            support_mass=support_mass,
            contradiction_mass=contradiction_mass,
            source_namespaces=aggregate.source_namespaces,
            recurrence_tau=self._recurrence_tau,
            source_diversity_target=self._source_diversity_target,
        )
        importance = _calculate_importance(aggregate.supporting_episodes)
        contradiction_ratio = (
            contradiction_mass / (support_mass + contradiction_mass)
            if (support_mass + contradiction_mass) > 0
            else 0.0
        )
        status = (
            SemanticMemoryStatus.CONTESTED
            if contradiction_ratio >= self._contested_ratio_threshold
            else SemanticMemoryStatus.ACTIVE
        )
        supporting_derivations = tuple(
            SemanticDerivationInput(
                episode_id=item.episode_id,
                relation=SemanticDerivationRelation.SUPPORTS,
                contribution_score=min(1.0, item.weight),
            )
            for item in aggregate.supporting_episodes
        )
        contradicting_derivations = tuple(
            SemanticDerivationInput(
                episode_id=item.episode_id,
                relation=SemanticDerivationRelation.CONTRADICTS,
                contribution_score=min(1.0, item.weight),
            )
            for item in contradicting
        )
        derivations = supporting_derivations + contradicting_derivations
        observation_evidence = _flatten_observation_evidence(
            (*aggregate.supporting_episodes, *contradicting),
            episode_by_id,
        )
        entities = _build_entities(aggregate.canonical)
        first_supported_at = min(item.observed_at for item in aggregate.supporting_episodes)
        last_supported_at = max(item.observed_at for item in aggregate.supporting_episodes)
        statement = _build_statement(aggregate.canonical)
        fingerprint = _content_fingerprint(
            claim_key=aggregate.canonical.claim_key,
            status=status,
            confidence=confidence,
            importance=importance,
            supporting=aggregate.supporting_episodes,
            contradicting=contradicting,
            observation_evidence=observation_evidence,
            extractor_version=self._extractor_version,
            consolidation_version=self._consolidation_version,
        )
        metadata = MappingProxyType(
            {
                "semantic": {
                    "content_fingerprint": fingerprint,
                    "consolidation_version": self._consolidation_version,
                    "extractor_version": self._extractor_version,
                    "support_mass": support_mass,
                    "contradiction_mass": contradiction_mass,
                    "contradiction_ratio": contradiction_ratio,
                    "confidence": {
                        "recurrence_tau": self._recurrence_tau,
                        "source_diversity_target": self._source_diversity_target,
                        "score": confidence,
                    },
                }
            }
        )
        subject_id = aggregate.supporting_episodes[0].episode.subject_id
        return SemanticMemoryInput(
            tenant_id=aggregate.canonical.tenant_id,
            subject_id=subject_id,
            memory_key=aggregate.canonical.claim_key,
            slot_key=aggregate.canonical.slot_key,
            statement=statement,
            subject_entity_id=aggregate.canonical.subject_entity_id,
            predicate=aggregate.canonical.predicate,
            object_value=aggregate.canonical.object_value,
            object_entity_id=aggregate.canonical.object_entity_id,
            polarity=aggregate.canonical.polarity,
            cardinality=aggregate.canonical.cardinality,
            qualifiers=aggregate.canonical.qualifiers,
            confidence=confidence,
            importance=importance,
            status=status,
            support_count=len(aggregate.supporting_episodes),
            contradiction_count=len(contradicting),
            first_supported_at=first_supported_at,
            last_supported_at=last_supported_at,
            derivations=derivations,
            observation_evidence=observation_evidence,
            entities=entities,
            metadata=metadata,
        )


class MetadataSemanticExtractor:
    """Extract explicitly structured facts from observation metadata."""

    def __init__(self, *, metadata_key: str = "semantic_facts") -> None:
        if not metadata_key.strip():
            raise ValidationError("metadata_key must not be empty.")
        self._metadata_key = metadata_key

    async def extract(
        self,
        episodes: Sequence[StoredEpisode],
        *,
        observations: Mapping[str, StoredObservation],
    ) -> SemanticExtractionResult:
        candidates: list[SemanticFactCandidate] = []
        failed = 0
        for episode in episodes:
            for evidence in episode.evidence:
                observation = observations.get(evidence.observation_id)
                if observation is None:
                    failed += 1
                    continue
                if observation.tenant_id != episode.tenant_id:
                    failed += 1
                    continue
                raw_facts = observation.metadata.get(self._metadata_key, ())
                if raw_facts is None:
                    continue
                if isinstance(raw_facts, Mapping):
                    raw_facts = (raw_facts,)
                if not isinstance(raw_facts, Sequence) or isinstance(raw_facts, (str, bytes)):
                    failed += 1
                    continue
                for raw_fact in raw_facts:
                    try:
                        candidates.append(
                            _candidate_from_metadata(
                                episode=episode,
                                observation=observation,
                                raw_fact=raw_fact,
                            )
                        )
                    except ValidationError:
                        failed += 1
        return SemanticExtractionResult(candidates=tuple(candidates), failed=failed)


@dataclass(frozen=True, slots=True)
class _EpisodeSupport:
    episode_id: str
    episode: StoredEpisode
    candidate: SemanticFactCandidate
    weight: float
    observed_at: datetime


@dataclass(frozen=True, slots=True)
class _ClaimAggregate:
    canonical: _CanonicalClaim
    supporting_episodes: tuple[_EpisodeSupport, ...]
    source_namespaces: frozenset[str]


def _prepare_candidates(
    candidates: Sequence[SemanticFactCandidate],
    episode_by_id: Mapping[str, StoredEpisode],
) -> list[tuple[SemanticFactCandidate, StoredEpisode]]:
    per_episode: dict[tuple[str, str], SemanticFactCandidate] = {}
    for candidate in candidates:
        episode = episode_by_id.get(candidate.source_episode_id)
        if episode is None:
            raise ValidationError(
                f"Candidate references unknown episode {candidate.source_episode_id!r}."
            )
        if candidate.tenant_id != episode.tenant_id:
            raise ValidationError("Candidate tenant_id does not match source episode.")
        canonical = _canonicalize_candidate(candidate)
        key = (candidate.source_episode_id, canonical.claim_key)
        existing = per_episode.get(key)
        if existing is None or candidate.confidence > existing.confidence:
            per_episode[key] = candidate
    prepared: list[tuple[SemanticFactCandidate, StoredEpisode]] = []
    for candidate in per_episode.values():
        prepared.append((candidate, episode_by_id[candidate.source_episode_id]))
    prepared.sort(
        key=lambda item: (item[0].source_episode_id, _canonicalize_candidate(item[0]).claim_key)
    )
    return prepared


def _build_claim_aggregate(
    claim_key: str,
    items: list[tuple[SemanticFactCandidate, StoredEpisode]],
) -> _ClaimAggregate:
    canonical = _canonicalize_candidate(items[0][0])
    supports: list[_EpisodeSupport] = []
    namespaces: set[str] = set()
    for candidate, episode in items:
        weight = _episode_weight(candidate, episode)
        supports.append(
            _EpisodeSupport(
                episode_id=episode.id,
                episode=episode,
                candidate=candidate,
                weight=weight,
                observed_at=candidate.observed_at,
            )
        )
        episode_meta = episode.metadata.get("episode", {})
        if isinstance(episode_meta, Mapping):
            for namespace in episode_meta.get("source_namespaces", ()):
                namespaces.add(str(namespace))
    return _ClaimAggregate(
        canonical=canonical,
        supporting_episodes=tuple(sorted(supports, key=lambda s: (s.observed_at, s.episode_id))),
        source_namespaces=frozenset(namespaces),
    )


def _find_contradictions(
    claim_key: str,
    aggregate: _ClaimAggregate,
    claim_entries: Mapping[str, _ClaimAggregate],
) -> list[_EpisodeSupport]:
    contradicting: list[_EpisodeSupport] = []
    for other_key, other in claim_entries.items():
        if other_key == claim_key:
            continue
        if not _claims_contradict(aggregate.canonical, other.canonical):
            continue
        for support in other.supporting_episodes:
            contradicting.append(support)
    contradicting.sort(key=lambda item: (item.observed_at, item.episode_id))
    return contradicting


def _claims_contradict(left: _CanonicalClaim, right: _CanonicalClaim) -> bool:
    if left.slot_key != right.slot_key:
        return False
    left_object = left.object_entity_id or left.object_value
    right_object = right.object_entity_id or right.object_value
    if left_object == right_object and left.polarity != right.polarity:
        return True
    if (
        left.cardinality is SemanticCardinality.ONE
        and right.cardinality is SemanticCardinality.ONE
        and left.polarity is SemanticPolarity.AFFIRM
        and right.polarity is SemanticPolarity.AFFIRM
        and left_object != right_object
    ):
        return True
    return False


def _episode_weight(candidate: SemanticFactCandidate, episode: StoredEpisode) -> float:
    return candidate.confidence * episode.confidence * (0.5 + 0.5 * episode.importance)


def _calculate_confidence(
    *,
    support_mass: float,
    contradiction_mass: float,
    source_namespaces: frozenset[str],
    recurrence_tau: float,
    source_diversity_target: int,
) -> float:
    recurrence_strength = 1.0 - math.exp(-support_mass / recurrence_tau)
    total = support_mass + contradiction_mass
    agreement = support_mass / total if total > 0 else 0.0
    source_diversity = min(1.0, len(source_namespaces) / source_diversity_target)
    confidence = recurrence_strength * agreement * (0.8 + 0.2 * source_diversity)
    return min(1.0, max(0.0, confidence))


def _calculate_importance(supports: Sequence[_EpisodeSupport]) -> float:
    importances = [item.episode.importance for item in supports]
    maximum = max(importances)
    mean = sum(importances) / len(importances)
    return 0.6 * maximum + 0.4 * mean


def _flatten_observation_evidence(
    episodes: Sequence[_EpisodeSupport],
    episode_by_id: Mapping[str, StoredEpisode],
) -> tuple[EpisodeEvidenceInput, ...]:
    evidence_map: dict[tuple[str, int], float] = {}
    for support in episodes:
        episode = episode_by_id[support.episode_id]
        for item in episode.evidence:
            key = (item.observation_id, item.observation_revision)
            contribution = min(1.0, support.weight * item.contribution_score)
            evidence_map[key] = max(evidence_map.get(key, 0.0), contribution)
    sorted_keys = sorted(evidence_map)
    return tuple(
        EpisodeEvidenceInput(
            observation_id=key[0],
            observation_revision=key[1],
            sequence_number=index,
            contribution_score=evidence_map[key],
        )
        for index, key in enumerate(sorted_keys)
    )


def _build_entities(canonical: _CanonicalClaim) -> tuple[EpisodeEntity, ...]:
    entities: dict[tuple[str, str], EpisodeEntity] = {}
    if canonical.subject_entity_id:
        entities[(canonical.subject_entity_id, "subject")] = EpisodeEntity(
            entity_id=canonical.subject_entity_id,
            role="subject",
        )
    if canonical.object_entity_id:
        entities[(canonical.object_entity_id, "object")] = EpisodeEntity(
            entity_id=canonical.object_entity_id,
            role="object",
        )
    return tuple(sorted(entities.values(), key=lambda e: (e.entity_id, e.role)))


def _candidate_from_metadata(
    *,
    episode: StoredEpisode,
    observation: StoredObservation,
    raw_fact: Any,
) -> SemanticFactCandidate:
    if not isinstance(raw_fact, Mapping):
        raise ValidationError("semantic_facts entries must be mappings.")
    predicate = str(raw_fact.get("predicate", "")).strip()
    object_value = str(raw_fact.get("object_value", "")).strip()
    if not predicate or not object_value:
        raise ValidationError("semantic_facts entries require predicate and object_value.")
    polarity_raw = str(raw_fact.get("polarity", SemanticPolarity.AFFIRM.value))
    cardinality_raw = str(raw_fact.get("cardinality", SemanticCardinality.MANY.value))
    polarity = SemanticPolarity(polarity_raw)
    cardinality = SemanticCardinality(cardinality_raw)
    confidence_raw = raw_fact.get("confidence", 1.0)
    if not isinstance(confidence_raw, (int, float)) or not math.isfinite(float(confidence_raw)):
        raise ValidationError("semantic_facts confidence must be numeric.")
    confidence = float(confidence_raw)
    subject_entity_id = raw_fact.get("subject_entity_id", observation.subject_id)
    object_entity_id = raw_fact.get("object_entity_id")
    qualifiers = raw_fact.get("qualifiers", {})
    if qualifiers is None:
        qualifiers = {}
    if not isinstance(qualifiers, Mapping):
        raise ValidationError("semantic_facts qualifiers must be a mapping.")
    return SemanticFactCandidate(
        tenant_id=episode.tenant_id,
        source_episode_id=episode.id,
        subject_entity_id=str(subject_entity_id) if subject_entity_id is not None else None,
        predicate=predicate,
        object_value=object_value,
        object_entity_id=str(object_entity_id) if object_entity_id is not None else None,
        polarity=polarity,
        cardinality=cardinality,
        confidence=confidence,
        observed_at=observation.observed_at,
        qualifiers=MappingProxyType(dict(qualifiers)),
    )


def _canonicalize_candidate(candidate: SemanticFactCandidate) -> _CanonicalClaim:
    subject_entity_id = _canonical_text(candidate.subject_entity_id or _SUBJECT_PLACEHOLDER)
    predicate = _canonical_text(candidate.predicate)
    object_value = _canonical_text(candidate.object_value)
    object_entity_id = (
        _canonical_text(candidate.object_entity_id) if candidate.object_entity_id else None
    )
    qualifiers = _canonical_qualifiers(candidate.qualifiers)
    slot_key = _hash_parts(
        (
            candidate.tenant_id,
            subject_entity_id,
            predicate,
            qualifiers,
            _CONSOLIDATION_VERSION,
        )
    )
    claim_key = _hash_parts(
        (
            slot_key,
            object_entity_id or object_value,
            candidate.polarity.value,
            _CONSOLIDATION_VERSION,
        )
    )
    return _CanonicalClaim(
        tenant_id=candidate.tenant_id,
        subject_entity_id=subject_entity_id,
        predicate=predicate,
        object_value=object_value,
        object_entity_id=object_entity_id,
        polarity=candidate.polarity,
        cardinality=candidate.cardinality,
        qualifiers=MappingProxyType(json.loads(qualifiers)),
        slot_key=slot_key,
        claim_key=claim_key,
    )


def _canonical_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    collapsed = _WHITESPACE_PATTERN.sub(" ", normalized.strip())
    return collapsed.casefold()


def _canonical_qualifiers(qualifiers: Mapping[str, Any]) -> str:
    normalized: dict[str, str] = {}
    for key in sorted(qualifiers):
        normalized[_canonical_text(str(key))] = _canonical_text(str(qualifiers[key]))
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def _hash_parts(parts: Sequence[str]) -> str:
    canonical = "\x1f".join(parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _content_fingerprint(
    *,
    claim_key: str,
    status: SemanticMemoryStatus,
    confidence: float,
    importance: float,
    supporting: Sequence[_EpisodeSupport],
    contradicting: Sequence[_EpisodeSupport],
    observation_evidence: Sequence[EpisodeEvidenceInput],
    extractor_version: str,
    consolidation_version: str,
) -> str:
    support_parts = [
        f"{item.episode_id}\x1f{item.weight:.6f}"
        for item in sorted(supporting, key=lambda s: s.episode_id)
    ]
    contradict_parts = [
        f"{item.episode_id}\x1f{item.weight:.6f}"
        for item in sorted(contradicting, key=lambda s: s.episode_id)
    ]
    evidence_parts = [
        f"{item.observation_id}\x1f{item.observation_revision}\x1f{item.contribution_score:.6f}"
        for item in observation_evidence
    ]
    parts = [
        claim_key,
        status.value,
        f"{confidence:.6f}",
        f"{importance:.6f}",
        "\x1e".join(support_parts),
        "\x1e".join(contradict_parts),
        "\x1e".join(evidence_parts),
        extractor_version,
        consolidation_version,
    ]
    return _hash_parts(parts)


def _build_statement(canonical: _CanonicalClaim) -> str:
    subject = canonical.subject_entity_id
    predicate = canonical.predicate.replace("_", " ")
    obj = canonical.object_value
    qualifier_suffix = ""
    if canonical.qualifiers:
        parts = [f"{key}={value}" for key, value in sorted(canonical.qualifiers.items())]
        qualifier_suffix = f" ({', '.join(parts)})"
    if canonical.polarity is SemanticPolarity.DENY:
        return f"{subject} does not {predicate} {obj}{qualifier_suffix}"
    return f"{subject} {predicate} {obj}{qualifier_suffix}"
