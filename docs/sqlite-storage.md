# SQLite storage

Cogkura's standard-library SQLite backend persists observations, connector checkpoints, episodes, semantic memories and cognitive state in one local database. It retains observation and semantic revision history, evidence and semantic relationships. Install Core without extra dependencies and inject the stores into the existing `Memory` facade:

```python
from cogkura import Memory
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

storage = await SQLiteStorage.open("cogkura.db")
try:
    memory = Memory(
        observation_store=SQLiteObservationStore(storage),
        checkpoint_store=SQLiteCheckpointStore(storage),
        episode_store=SQLiteEpisodeStore(storage),
        semantic_store=SQLiteSemanticMemoryStore(storage),
        activation_store=SQLiteActivationStore(storage),
        dynamics_store=SQLiteMemoryDynamicsStore(storage),
        learning_store=SQLiteLearningStore(storage),
        inhibition_store=SQLiteInhibitionStore(storage),
        entity_relationship_store=SQLiteEntityRelationshipStore(storage),
    )
    # Use memory.observe(...) and memory.ingest(...).
finally:
    await storage.close()
```

Opening creates a new database or migrates a supported SQLite schema; it rejects unrelated or newer databases. The caller owns the shared storage object and must close it after use. Close drains accepted work. Reopen the same path to recover observations and checkpoints. `Memory.clear()` clears observations for the requested tenant but does not clear connector checkpoints.

Activation references, forgetting state, learning, inhibition and entity relationships survive restart when the SQLite stores shown above are injected. Any stores omitted from the constructor still default to in-memory storage. Working-memory snapshots remain transient. A cohesive embedded runtime is a later proposed milestone.

SQLite work runs on a dedicated worker thread. Each observation and its revision commit together; connector checkpoints commit separately after a successful ingestion batch. A failed or interrupted batch can leave earlier observations committed while its checkpoint remains unchanged; replay is expected. Cancellation of an awaiting write does not establish whether it committed. Retry using the same source identity and version after checking the result.

Semantic reads and writes currently materialize the tenant’s semantic inventory in memory; some cognitive-state operations also scan tenant-scoped records. workloads with large histories may have significant latency and memory costs. Slot inspection limits its inventory queries, but broad performance and workload validation are planned for 0.19.4.

The backend uses WAL, foreign keys, `synchronous=FULL` and a five-second default busy timeout; pass `busy_timeout` in seconds to `open` to change it. Independent connections can read concurrently and writes are serialized by SQLite. Lock timeouts and database failures raise `StorageError`. Use a filesystem that supports SQLite WAL and locking. SQLite may create `-wal` and `-shm` sidecars; use SQLite-supported backup mechanisms rather than copying a live database file alone.

The database and sidecars are not encrypted by Cogkura and may contain retained content, metadata and checkpoints. Protect them with host-device encryption and filesystem permissions appropriate to the application. Retention policies still apply before storage; they are not encryption or secure deletion.
