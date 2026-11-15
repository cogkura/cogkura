"""Managed local Cogkura memory backed by one SQLite database."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from cogkura.exceptions import StorageError, ValidationError
from cogkura.memory import Memory
from cogkura.storage.sqlite import SQLiteCheckpointStore, SQLiteObservationStore, SQLiteStorage
from cogkura.storage.sqlite_cognitive import (
    SQLiteActivationStore,
    SQLiteEntityRelationshipStore,
    SQLiteInhibitionStore,
    SQLiteLearningStore,
    SQLiteMemoryDynamicsStore,
)
from cogkura.storage.sqlite_episode import SQLiteEpisodeStore
from cogkura.storage.sqlite_semantic import SQLiteSemanticMemoryStore

_STORE_OPTIONS = frozenset(
    {
        "observation_store",
        "checkpoint_store",
        "episode_store",
        "semantic_store",
        "semantic_inspection_store",
        "activation_store",
        "dynamics_store",
        "learning_store",
        "inhibition_store",
        "entity_relationship_store",
    }
)


class EmbeddedMemory:
    """Own a durable SQLite storage bundle and an existing ``Memory`` facade.

    Use ``async with await EmbeddedMemory.open(path) as memory`` or explicitly
    close the returned owner. Reopen by constructing a new owner.
    """

    def __init__(self, storage: SQLiteStorage, memory: Memory) -> None:
        self._storage = storage
        self._memory = memory
        self._closing = False

    @classmethod
    async def open(
        cls, path: str | Path, *, busy_timeout: float = 5.0, **memory_options: Any
    ) -> EmbeddedMemory:
        """Open or migrate a local database and construct its memory facade.

        Non-storage ``Memory`` constructor options may be supplied by keyword.
        The caller owns the returned instance and must close it.
        """
        if not isinstance(path, (str, Path)) or not str(path).strip() or str(path) == ":memory:":
            raise ValidationError("Embedded memory requires a non-empty database file path.")
        if (
            isinstance(busy_timeout, bool)
            or not isinstance(busy_timeout, (int, float))
            or not math.isfinite(busy_timeout)
            or busy_timeout < 0
        ):
            raise ValidationError("busy_timeout must be a finite non-negative number of seconds.")
        supplied = _STORE_OPTIONS.intersection(memory_options)
        if supplied:
            raise ValidationError(
                "Embedded memory manages its own stores: " + ", ".join(sorted(supplied))
            )
        storage = await SQLiteStorage.open(path, busy_timeout=busy_timeout)
        try:
            semantic = SQLiteSemanticMemoryStore(storage)
            memory = Memory(
                observation_store=SQLiteObservationStore(storage),
                checkpoint_store=SQLiteCheckpointStore(storage),
                episode_store=SQLiteEpisodeStore(storage),
                semantic_store=semantic,
                semantic_inspection_store=semantic,
                activation_store=SQLiteActivationStore(storage),
                dynamics_store=SQLiteMemoryDynamicsStore(storage),
                learning_store=SQLiteLearningStore(storage),
                inhibition_store=SQLiteInhibitionStore(storage),
                entity_relationship_store=SQLiteEntityRelationshipStore(storage),
                **memory_options,
            )
            return cls(storage, memory)
        except BaseException:
            await storage.close()
            raise

    @property
    def memory(self) -> Memory:
        """Return the configured facade while this owner is open."""
        if self._closing:
            raise StorageError("Embedded memory is closed")
        return self._memory

    async def close(self) -> None:
        """Drain accepted storage operations and close the database and worker."""
        self._closing = True
        await self._storage.close()

    async def __aenter__(self) -> Memory:
        return self.memory

    async def __aexit__(self, *args: object) -> None:
        await self.close()
