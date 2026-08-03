"""Storage interface for memory events."""

from __future__ import annotations

from typing import Protocol

from cognema.event import MemoryEvent


class MemoryStorage(Protocol):
    """Minimal storage protocol used by Memory."""

    def store(self, event: MemoryEvent) -> None:
        """Persist a memory event."""

    def get(self, event_id: str) -> MemoryEvent | None:
        """Fetch a single event by id."""

    def list(self) -> list[MemoryEvent]:
        """Return all stored events."""

    def clear(self) -> None:
        """Remove all stored events."""
