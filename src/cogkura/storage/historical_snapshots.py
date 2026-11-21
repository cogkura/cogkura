"""Versioned representations of derived memories and semantic revisions.

Snapshot data can contain retained source content and must follow the store's
retention policy.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Any

from cogkura.exceptions import StorageError, ValidationError
from cogkura.models import (
    EpisodeEntity,
    EpisodeEvidenceInput,
    MemoryContextSignature,
    SemanticCardinality,
    SemanticDerivationInput,
    SemanticDerivationRelation,
    SemanticMemoryStatus,
    SemanticPolarity,
    StoredEpisode,
    StoredSemanticMemory,
    StoredSemanticRevision,
)

_VERSION = 1


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, MemoryContextSignature):
        return value.to_canonical_dict()
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: getattr(value, item.name) for item in fields(value)}
    # Model metadata is exposed as a read-only mapping.
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Unsupported snapshot value: {type(value).__name__}")


def encode_memory_snapshot(memory: StoredEpisode | StoredSemanticMemory) -> str:
    """Capture all stored fields, independent of subsequent metadata mutations."""
    _validate_json_value(memory.metadata)
    if isinstance(memory, StoredSemanticMemory):
        _validate_json_value(memory.qualifiers)
    kind = "episode" if isinstance(memory, StoredEpisode) else "semantic"
    return json.dumps(
        {"version": _VERSION, "kind": kind, "memory": memory},
        default=_json_default,
        sort_keys=True,
        allow_nan=False,
    )


def decode_memory_snapshot(payload: str) -> StoredEpisode | StoredSemanticMemory:
    """Restore a snapshot, rejecting unknown formats instead of guessing history."""
    try:
        data = json.loads(payload)
        if (
            not isinstance(data, dict)
            or type(data.get("version")) is not int
            or data["version"] != _VERSION
        ):
            raise ValueError("Unknown snapshot version")
        value = data["memory"]
        if not isinstance(value, dict):
            raise ValueError("Invalid snapshot memory")
        kind = data["kind"]
        for name in ("created_at", "updated_at"):
            value[name] = _timestamp(value[name])
        value["metadata"] = MappingProxyType(value["metadata"])
        value["entities"] = tuple(EpisodeEntity(**item) for item in value["entities"])
        if kind == "episode":
            for name in ("started_at", "ended_at"):
                value[name] = _timestamp(value[name])
            value["evidence"] = tuple(EpisodeEvidenceInput(**item) for item in value["evidence"])
            value["encoding_context"] = MemoryContextSignature.from_canonical_dict(
                value["encoding_context"]
            )
            return StoredEpisode(**value)
        if kind == "semantic":
            value["qualifiers"] = MappingProxyType(value["qualifiers"])
            for name in ("first_supported_at", "last_supported_at"):
                value[name] = _timestamp(value[name])
            for name in ("valid_from", "valid_until"):
                value[name] = _timestamp(value[name]) if value[name] is not None else None
            value["polarity"] = SemanticPolarity(value["polarity"])
            value["cardinality"] = SemanticCardinality(value["cardinality"])
            value["status"] = SemanticMemoryStatus(value["status"])
            value["derivations"] = tuple(
                SemanticDerivationInput(
                    episode_id=item["episode_id"],
                    relation=SemanticDerivationRelation(item["relation"]),
                    contribution_score=item["contribution_score"],
                )
                for item in value["derivations"]
            )
            value["observation_evidence"] = tuple(
                EpisodeEvidenceInput(**item) for item in value["observation_evidence"]
            )
            return StoredSemanticMemory(**value)
        raise ValueError("Unknown snapshot kind")
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise StorageError("Invalid or unsupported historical memory snapshot") from exc


def select_episode_snapshot(
    versions: list[tuple[datetime, str | None]], *, as_of: datetime
) -> StoredEpisode | None:
    """Select the last known episode, failing closed on reordered or missing writes.

    Versions are supplied in actual write order. Equal timestamps use that order;
    a backdated write invalidates the entire sequence because it cannot prove
    what was known at an earlier point.
    """
    memory = _select_snapshot(versions, as_of=as_of)
    return memory if isinstance(memory, StoredEpisode) else None


def select_semantic_snapshot(
    versions: list[tuple[datetime, str | None]], *, as_of: datetime
) -> StoredSemanticMemory | None:
    """Select a semantic snapshot under the same conservative ordering rules."""
    memory = _select_snapshot(versions, as_of=as_of)
    return memory if isinstance(memory, StoredSemanticMemory) else None


def _select_snapshot(
    versions: list[tuple[datetime, str | None]], *, as_of: datetime
) -> StoredEpisode | StoredSemanticMemory | None:
    if any(earlier[0] > later[0] for earlier, later in zip(versions, versions[1:], strict=False)):
        return None
    matches = [payload for recorded_at, payload in versions if recorded_at <= as_of]
    if not matches or matches[-1] is None:
        return None
    try:
        memory = decode_memory_snapshot(matches[-1])
    except StorageError:
        return None
    return memory


def encode_episode_snapshot_or_none(episode: StoredEpisode) -> str | None:
    """Mark unsupported metadata unavailable without breaking existing writes."""
    try:
        return encode_memory_snapshot(episode)
    except (TypeError, ValueError):
        return None


def encode_semantic_snapshot_or_none(memory: StoredSemanticMemory) -> str | None:
    """Mark unsupported semantic metadata unavailable without breaking writes."""
    try:
        return encode_memory_snapshot(memory)
    except (TypeError, ValueError):
        return None


def encode_revision_snapshot_or_none(revision: StoredSemanticRevision) -> str | None:
    """Capture a full revision, marking unrepresentable state unavailable."""
    try:
        _validate_json_value(revision.snapshot)
        return json.dumps(
            {"version": _VERSION, "kind": "semantic_revision", "revision": revision},
            default=_json_default,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return None


def select_revision_snapshot(
    versions: list[tuple[datetime, str | None]], *, as_of: datetime
) -> StoredSemanticRevision | None:
    """Select the last complete revision, failing closed on ambiguous history."""
    if any(earlier[0] > later[0] for earlier, later in zip(versions, versions[1:], strict=False)):
        return None
    matches = [payload for at, payload in versions if at <= as_of]
    if not matches or matches[-1] is None:
        return None
    try:
        data = json.loads(matches[-1])
        if (
            not isinstance(data, dict)
            or type(data.get("version")) is not int
            or data["version"] != _VERSION
            or data.get("kind") != "semantic_revision"
        ):
            return None
        value = data["revision"]
        for name in ("created_at", "updated_at", "first_supported_at", "last_supported_at"):
            value[name] = _timestamp(value[name])
        for name in ("valid_from", "valid_until"):
            value[name] = _timestamp(value[name]) if value[name] is not None else None
        value["status"] = SemanticMemoryStatus(value["status"])
        value["derivations"] = tuple(
            SemanticDerivationInput(
                episode_id=item["episode_id"],
                relation=SemanticDerivationRelation(item["relation"]),
                contribution_score=item["contribution_score"],
            )
            for item in value["derivations"]
        )
        return StoredSemanticRevision(**value)
    except (KeyError, TypeError, ValueError, ValidationError):
        return None


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("Snapshot timestamp must be timezone-aware")
    return parsed


def _validate_json_value(value: Any) -> None:
    """Reject metadata JSON would silently change, such as tuples or datetimes."""
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float) and math.isfinite(value):
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Snapshot metadata keys must be strings")
            _validate_json_value(item)
        return
    raise TypeError(f"Unsupported snapshot metadata value: {type(value).__name__}")
