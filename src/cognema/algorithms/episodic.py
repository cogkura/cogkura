"""Deterministic episodic memory encoding."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any, Protocol

from cognema.exceptions import ValidationError
from cognema.models import EpisodeEntity, EpisodeEvidenceInput, EpisodeInput
from cognema.observations.models import StoredObservation

_ENCODING_VERSION = "tulving-deterministic-v1"
_SALIENCE_VERSION = "salience-v1"
_SUBJECT_PLACEHOLDER = "__none__"
_GROUPING_KEYS = ("episode_id", "conversation_id", "thread_id", "session_id")


class EpisodicEncoder(Protocol):
    """Converts stored observations into episodic memory candidates."""

    def encode(self, observations: Sequence[StoredObservation]) -> list[EpisodeInput]:
        """Convert current observations into deterministic episodes."""


@dataclass(frozen=True, slots=True)
class _GroupingKey:
    group_type: str | None
    group_value: str | None


class DeterministicEpisodicEncoder:
    """Rule-based episodic encoder with deterministic segmentation."""

    def __init__(
        self,
        *,
        maximum_gap_seconds: int = 1800,
        grouping_metadata_keys: tuple[str, ...] = _GROUPING_KEYS,
        entity_metadata_key: str = "entity_ids",
        maximum_statement_length: int = 2000,
        encoding_version: str = _ENCODING_VERSION,
    ) -> None:
        if maximum_gap_seconds <= 0:
            raise ValidationError("maximum_gap_seconds must be greater than zero.")
        if not grouping_metadata_keys or any(not key.strip() for key in grouping_metadata_keys):
            raise ValidationError("grouping_metadata_keys must not contain empty values.")
        if maximum_statement_length <= 0:
            raise ValidationError("maximum_statement_length must be greater than zero.")
        if not encoding_version.strip():
            raise ValidationError("encoding_version must not be empty.")
        self._maximum_gap_seconds = maximum_gap_seconds
        self._grouping_metadata_keys = grouping_metadata_keys
        self._entity_metadata_key = entity_metadata_key
        self._maximum_statement_length = maximum_statement_length
        self._encoding_version = encoding_version

    def encode(self, observations: Sequence[StoredObservation]) -> list[EpisodeInput]:
        active = [obs for obs in observations if not obs.is_deleted]
        active.sort(key=lambda obs: (obs.observed_at, obs.id))
        partitions = _partition_observations(active, self._grouping_metadata_keys)
        episodes: list[EpisodeInput] = []
        for partition in partitions:
            episode_groups = _segment_partition(
                partition,
                self._maximum_gap_seconds,
                self._grouping_metadata_keys,
            )
            for group in episode_groups:
                episodes.append(self._build_episode(group))
        return episodes

    def _build_episode(self, observations: Sequence[StoredObservation]) -> EpisodeInput:
        if not observations:
            raise ValidationError("Cannot build an episode from zero observations.")
        first = observations[0]
        grouping = _explicit_grouping(first.metadata, self._grouping_metadata_keys)
        subject_id = first.subject_id
        started_at = min(obs.observed_at for obs in observations)
        ended_at = max(obs.observed_at for obs in observations)
        evidence = tuple(
            EpisodeEvidenceInput(
                observation_id=obs.id,
                observation_revision=obs.current_revision,
                sequence_number=index,
                contribution_score=obs.attention_score,
            )
            for index, obs in enumerate(observations)
        )
        entities = _resolve_entities(observations, self._entity_metadata_key)
        statement = _build_statement(observations, self._maximum_statement_length)
        importance, salience_meta = _calculate_salience(observations)
        confidence = _calculate_confidence(observations)
        fingerprint = _content_fingerprint(evidence)
        memory_key = _episode_memory_key(
            observations,
            grouping=grouping,
            subject_id=subject_id,
            encoding_version=self._encoding_version,
        )
        metadata = _build_metadata(
            observations,
            grouping=grouping,
            fingerprint=fingerprint,
            started_at=started_at,
            ended_at=ended_at,
            salience_meta=salience_meta,
            encoding_version=self._encoding_version,
        )
        return EpisodeInput(
            tenant_id=first.tenant_id,
            subject_id=subject_id,
            memory_key=memory_key,
            statement=statement,
            started_at=started_at,
            ended_at=ended_at,
            confidence=confidence,
            importance=importance,
            evidence=evidence,
            entities=entities,
            metadata=metadata,
        )


def _partition_observations(
    observations: Sequence[StoredObservation],
    grouping_keys: Sequence[str],
) -> list[list[StoredObservation]]:
    partitions: dict[tuple[Any, ...], list[StoredObservation]] = {}
    for observation in observations:
        grouping = _explicit_grouping(observation.metadata, grouping_keys)
        partition_key = (
            observation.tenant_id,
            observation.subject_id or observation.actor_id or _SUBJECT_PLACEHOLDER,
            observation.source_namespace,
            grouping.group_type,
            grouping.group_value,
        )
        partitions.setdefault(partition_key, []).append(observation)
    return [partitions[key] for key in sorted(partitions)]


def _segment_partition(
    observations: Sequence[StoredObservation],
    maximum_gap_seconds: int,
    grouping_keys: Sequence[str],
) -> list[list[StoredObservation]]:
    if not observations:
        return []
    grouping = _explicit_grouping(observations[0].metadata, grouping_keys)
    if grouping.group_type is not None:
        return _segment_with_boundaries(observations)

    groups: list[list[StoredObservation]] = []
    current: list[StoredObservation] = []
    previous: StoredObservation | None = None
    for observation in observations:
        if observation.metadata.get("episode_boundary") is True and current:
            groups.append(current)
            current = []
        if previous is not None:
            gap = (observation.observed_at - previous.observed_at).total_seconds()
            if gap > maximum_gap_seconds and current:
                groups.append(current)
                current = []
        current.append(observation)
        if observation.metadata.get("terminal_event") is True:
            groups.append(current)
            current = []
        previous = observation
    if current:
        groups.append(current)
    return groups


def _segment_with_boundaries(
    observations: Sequence[StoredObservation],
) -> list[list[StoredObservation]]:
    groups: list[list[StoredObservation]] = []
    current: list[StoredObservation] = []
    for observation in observations:
        if observation.metadata.get("episode_boundary") is True and current:
            groups.append(current)
            current = []
        current.append(observation)
        if observation.metadata.get("terminal_event") is True:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _explicit_grouping(
    metadata: Mapping[str, Any],
    grouping_keys: Sequence[str],
) -> _GroupingKey:
    for key in grouping_keys:
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return _GroupingKey(group_type=key, group_value=str(value))
    return _GroupingKey(group_type=None, group_value=None)


def _episode_memory_key(
    observations: Sequence[StoredObservation],
    *,
    grouping: _GroupingKey,
    subject_id: str | None,
    encoding_version: str,
) -> str:
    first = observations[0]
    subject_part = subject_id or first.actor_id or _SUBJECT_PLACEHOLDER
    if grouping.group_type is not None and grouping.group_value is not None:
        parts: tuple[str, ...] = (
            first.tenant_id,
            subject_part,
            first.source_namespace,
            grouping.group_type,
            grouping.group_value,
            encoding_version,
        )
    else:
        parts = (
            first.tenant_id,
            subject_part,
            first.source_namespace,
            first.id,
            encoding_version,
        )
    return _hash_parts(parts)


def _content_fingerprint(evidence: Sequence[EpisodeEvidenceInput]) -> str:
    parts = [
        f"{item.observation_id}\x1f{item.observation_revision}\x1f{item.sequence_number}"
        for item in evidence
    ]
    return _hash_parts(parts)


def _hash_parts(parts: Sequence[str]) -> str:
    canonical = "\x1f".join(parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_statement(
    observations: Sequence[StoredObservation],
    maximum_length: int,
) -> str:
    contents: list[str] = []
    for observation in observations:
        if observation.content and observation.content.strip():
            normalized = observation.content.strip()
            if contents and contents[-1] == normalized:
                continue
            contents.append(normalized)
    if contents:
        statement = "\n".join(contents)
        if len(statement) > maximum_length:
            return statement[:maximum_length]
        return statement

    count = len(observations)
    event_type = observations[0].event_type
    namespace = observations[0].source_namespace
    plural = "observation" if count == 1 else "observations"
    return f"Episode containing {count} {event_type} {plural} from {namespace}."


def _resolve_entities(
    observations: Sequence[StoredObservation],
    entity_metadata_key: str,
) -> tuple[EpisodeEntity, ...]:
    entities: dict[tuple[str, str], EpisodeEntity] = {}
    for observation in observations:
        if observation.subject_id:
            key = (observation.subject_id, "subject")
            entities[key] = EpisodeEntity(entity_id=observation.subject_id, role="subject")
        if observation.actor_id:
            key = (observation.actor_id, "actor")
            entities[key] = EpisodeEntity(entity_id=observation.actor_id, role="actor")
        raw_entities = observation.metadata.get(entity_metadata_key, ())
        if isinstance(raw_entities, str):
            raw_entities = (raw_entities,)
        if isinstance(raw_entities, Sequence) and not isinstance(raw_entities, (str, bytes)):
            for entity_id in raw_entities:
                if entity_id is None:
                    continue
                entity_text = str(entity_id).strip()
                if not entity_text:
                    continue
                key = (entity_text, "metadata")
                entities[key] = EpisodeEntity(entity_id=entity_text, role="metadata")
    return tuple(sorted(entities.values(), key=lambda item: (item.entity_id, item.role)))


def _metadata_importance(metadata: Mapping[str, Any]) -> float | None:
    for key in ("importance", "salience"):
        value = metadata.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value) and 0.0 <= float(value) <= 1.0:
            return float(value)
    return None


def _calculate_salience(
    observations: Sequence[StoredObservation],
) -> tuple[float, dict[str, Any]]:
    attention_scores = [obs.attention_score for obs in observations]
    maximum_attention = max(attention_scores)
    mean_attention = sum(attention_scores) / len(attention_scores)
    explicit_values = [
        value for obs in observations if (value := _metadata_importance(obs.metadata)) is not None
    ]
    explicit_importance = max(explicit_values) if explicit_values else 0.0
    terminal_event = (
        1.0 if any(obs.metadata.get("terminal_event") is True for obs in observations) else 0.0
    )
    score = (
        0.50 * maximum_attention
        + 0.30 * mean_attention
        + 0.15 * explicit_importance
        + 0.05 * terminal_event
    )
    score = min(1.0, max(0.0, score))
    salience_meta = {
        "maximum_attention": maximum_attention,
        "mean_attention": mean_attention,
        "explicit_importance": explicit_importance,
        "terminal_event": terminal_event,
        "score": score,
        "version": _SALIENCE_VERSION,
    }
    return score, salience_meta


def _calculate_confidence(observations: Sequence[StoredObservation]) -> float:
    total = len(observations)
    with_content = sum(1 for obs in observations if obs.content and obs.content.strip())
    coverage = with_content / total if total else 0.0
    return 0.7 + 0.3 * coverage


def _build_metadata(
    observations: Sequence[StoredObservation],
    *,
    grouping: _GroupingKey,
    fingerprint: str,
    started_at: datetime,
    ended_at: datetime,
    salience_meta: Mapping[str, Any],
    encoding_version: str,
) -> Mapping[str, Any]:
    content_observations = sum(1 for obs in observations if obs.content and obs.content.strip())
    missing_content = len(observations) - content_observations
    duration_seconds = int((ended_at - started_at).total_seconds())
    source_namespaces = sorted({obs.source_namespace for obs in observations})
    source_types = sorted({obs.source_type for obs in observations})
    event_types = sorted({obs.event_type for obs in observations})
    actor_ids = sorted({obs.actor_id for obs in observations if obs.actor_id})
    segmentation_type = "metadata" if grouping.group_type is not None else "temporal"
    episode_meta: dict[str, Any] = {
        "encoding_version": encoding_version,
        "segmentation_type": segmentation_type,
        "content_fingerprint": fingerprint,
        "observation_count": len(observations),
        "source_namespaces": source_namespaces,
        "source_types": source_types,
        "event_types": event_types,
        "actor_ids": actor_ids,
        "duration_seconds": duration_seconds,
        "content_observations": content_observations,
        "missing_content_observations": missing_content,
    }
    if grouping.group_type is not None:
        episode_meta["segmentation_key"] = grouping.group_type
        episode_meta["segmentation_value"] = grouping.group_value
    return MappingProxyType(
        {
            "episode": episode_meta,
            "salience": dict(salience_meta),
        }
    )
