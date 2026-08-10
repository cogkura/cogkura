"""Helpers for activation reference compaction."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from cogkura.models import MemoryReference

_DEFAULT_COMPACTION_DECAY = 0.5
_MINIMUM_ELAPSED_SECONDS = 1.0


def compaction_representative_time(
    references: Sequence[MemoryReference],
    *,
    bucket_start: datetime,
    as_of: datetime,
    decay: float = _DEFAULT_COMPACTION_DECAY,
) -> datetime:
    """Choose a representative timestamp for a compaction bucket."""
    normalized = tuple(reference.referenced_at.astimezone(UTC) for reference in references)
    if not normalized:
        return bucket_start.astimezone(UTC)
    if len(set(normalized)) == 1:
        return normalized[0]

    as_of_utc = as_of.astimezone(UTC)
    total_weight = sum(reference.weight for reference in references)
    powered_sum = 0.0
    for reference in references:
        elapsed_seconds = max(
            (as_of_utc - reference.referenced_at.astimezone(UTC)).total_seconds(),
            _MINIMUM_ELAPSED_SECONDS,
        )
        powered_sum += reference.weight * (elapsed_seconds**-decay)
    if powered_sum <= 0.0 or not math.isfinite(powered_sum):
        return max(normalized)

    representative_elapsed = (total_weight / powered_sum) ** (1.0 / decay)
    return as_of_utc - timedelta(seconds=representative_elapsed)
