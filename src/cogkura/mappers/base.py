"""Mapper protocol for source records."""

from __future__ import annotations

from typing import Any, Protocol

from cogkura.observations.models import ObservationInput


class ObservationMapper(Protocol):
    """Converts a source record into a normalized observation."""

    def map(self, record: Any) -> ObservationInput:
        """Map one source record to an ObservationInput."""
