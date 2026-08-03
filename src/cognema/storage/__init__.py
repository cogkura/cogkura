"""Storage backends and interfaces."""

from cognema.storage.base import MemoryStorage
from cognema.storage.in_memory import InMemoryStorage

__all__ = ["InMemoryStorage", "MemoryStorage"]
