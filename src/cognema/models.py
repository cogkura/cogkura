"""Additional package data models."""

from __future__ import annotations

import math
from dataclasses import dataclass

from cognema.exceptions import ValidationError
from cognema.observations.models import StoredObservation


@dataclass(frozen=True, slots=True)
class RecallResult:
    """A scored recall match for a query."""

    observation: StoredObservation
    score: float
    reason: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.score):
            raise ValidationError("Recall score must be finite.")
        if not 0.0 <= self.score <= 1.0:
            raise ValidationError("Recall score must be between 0.0 and 1.0.")
