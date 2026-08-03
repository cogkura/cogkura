"""In-memory storage backend."""

from __future__ import annotations

from cognema.event import MemoryEvent
from cognema.storage.base import MemoryStorage


class InMemoryStorage(MemoryStorage):
    """Simple in-memory storage for memory events."""

    def __init__(self) -> None:
        self._events: dict[str, MemoryEvent] = {}

    def store(self, event: MemoryEvent) -> None:
        self._events[event.id] = event

    def get(self, event_id: str) -> MemoryEvent | None:
        return self._events.get(event_id)

    def list(self) -> list[MemoryEvent]:
        return list(self._events.values())

    def clear(self) -> None:
        self._events.clear()
