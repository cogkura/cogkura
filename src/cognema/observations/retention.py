"""Observation content retention modes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from cognema.exceptions import ValidationError
from cognema.observations.hashing import content_hash, normalize_content
from cognema.observations.models import ObservationInput


class ObservationRetentionMode(StrEnum):
    """How observation content is stored."""

    FULL = "full"
    REDACTED = "redacted"
    HASH_ONLY = "hash_only"
    REFERENCE_ONLY = "reference_only"


RedactionFn = Callable[[str, dict[str, Any]], tuple[str, dict[str, Any]]]


@dataclass(frozen=True, slots=True)
class RetainedObservation:
    """Content and metadata after retention rules are applied."""

    content: str | None
    content_hash: str
    metadata: dict[str, Any]


def apply_retention(
    observation: ObservationInput,
    *,
    mode: ObservationRetentionMode = ObservationRetentionMode.FULL,
    redaction_fn: RedactionFn | None = None,
) -> RetainedObservation:
    """Apply retention mode transforms before persistence."""
    raw_content = observation.content
    raw_metadata = dict(observation.metadata)
    digest = content_hash(raw_content) if raw_content else content_hash("")

    if mode is ObservationRetentionMode.FULL:
        return RetainedObservation(
            content=normalize_content(raw_content) if raw_content else None,
            content_hash=digest,
            metadata=raw_metadata,
        )

    if mode is ObservationRetentionMode.HASH_ONLY:
        return RetainedObservation(content=None, content_hash=digest, metadata=raw_metadata)

    if mode is ObservationRetentionMode.REDACTED:
        if redaction_fn is None:
            raise ValidationError("REDACTED retention requires a redaction_fn.")
        redacted_content, redacted_metadata = redaction_fn(raw_content, raw_metadata)
        return RetainedObservation(
            content=normalize_content(redacted_content) if redacted_content else None,
            content_hash=content_hash(redacted_content) if redacted_content else digest,
            metadata=redacted_metadata,
        )

    if mode is ObservationRetentionMode.REFERENCE_ONLY:
        raise ValidationError("REFERENCE_ONLY retention is not implemented yet.")

    raise ValidationError(f"Unknown retention mode: {mode}")
