# SQLite storage

Cogkura's standard-library SQLite backend persists observations, connector checkpoints, episodes, semantic memories and cognitive state in one local database. It retains observation and semantic revision history, evidence and semantic relationships. Install Core without extra dependencies and open an embedded instance:

```python
from cogkura import EmbeddedMemory, ObservationInput

async with await EmbeddedMemory.open("cogkura.db") as memory:
    await memory.observe(ObservationInput(...))
    await memory.process(tenant_id="my-tenant")
    context = await memory.prepare_context("What do I know?", tenant_id="my-tenant")
```

The context manager closes the database on exit. For an explicit lifetime, use `runtime = await EmbeddedMemory.open(path)`, access `runtime.memory`, and call `await runtime.close()` when finished. The parent directory must exist. `open` accepts a finite non-negative `busy_timeout` in seconds and non-storage `Memory` constructor options such as `activation_config`; it rejects storage overrides to keep the bundle fully durable. Reopen with a new runtime after closing. Do not use a previously acquired `Memory` after its owner closes.

For advanced dependency injection, the individual stores can also be constructed explicitly:

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

Activation references, forgetting state, learning, inhibition and entity relationships survive restart when the SQLite stores shown above are injected. Any stores omitted from the constructor still default to in-memory storage. Working-memory snapshots remain transient.

SQLite work runs on a dedicated worker thread. Each observation and its revision commit together; connector checkpoints commit separately after a successful ingestion batch. A failed or interrupted batch can leave earlier observations committed while its checkpoint remains unchanged; replay is expected. Cancellation of an awaiting write does not establish whether it committed. Retry using the same source identity and version after checking the result.

Semantic reads and writes materialize the tenant's semantic inventory in memory; some cognitive-state operations also scan tenant-scoped records. Batched semantic child reads reduce SQL statement counts, but large histories can still have substantial latency and memory costs. Slot inspection limits its inventory queries. See the [0.19.4 workload findings](findings/0.19.4-embedded-reliability.md) for the measured workload and limits.

The backend uses WAL, foreign keys, `synchronous=FULL` and a five-second default busy timeout; pass `busy_timeout` in seconds to `open` to change it. On macOS it additionally enables SQLite's `fullfsync` for commits and checkpoints. This strengthens the requested sync but can substantially increase write latency. See the [measured comparison](findings/0.19.4-embedded-reliability.md). Independent connections can read concurrently and writes are serialized by SQLite. Lock timeouts and database failures raise `StorageError`.

SQLite's WAL-reset corruption bug affects concurrent connections on older SQLite versions. Opening fails before touching the file unless Python's linked SQLite is a known fixed release: 3.51.3 or later except the withdrawn 3.52.0, or the 3.44.6 and 3.50.7 backport lines. Check `sqlite3.sqlite_version` in the Python interpreter actually running the application. A distributor may backport a fix without changing the version number; Cogkura conservatively rejects such an unidentified build. Every other process or tool that opens the same database must also use a fixed SQLite build: Cogkura cannot enforce the version used by external processes. See [SQLite's WAL-reset description](https://www.sqlite.org/wal.html#the_wal_reset_bug).

Use a local filesystem with working SQLite locks and sync operations. SQLite WAL requires participating processes to share the same host and is not supported over network filesystems; avoid network shares and concurrent synchronization tools on a live database. SQLite may create `-wal` and `-shm` sidecars; use SQLite-supported backup mechanisms rather than copying a live database file alone.

The database and sidecars are not encrypted by Cogkura and may contain retained content, metadata and checkpoints. Protect them with host-device encryption and filesystem permissions appropriate to the application. Retention policies still apply before storage; they are not encryption or secure deletion.

## Backup, restore and file access

Use SQLite's online backup API to create a consistent copy, including committed data still in the WAL. For example, this synchronous function can run in `asyncio.to_thread` when called by an asynchronous application:

```python
import sqlite3
from pathlib import Path


def backup_database(source_path: str, new_backup_path: str) -> None:
    source_uri = Path(source_path).resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(source_uri, uri=True) as source:
        Path(new_backup_path).touch(mode=0o600, exist_ok=False)
        with sqlite3.connect(new_backup_path) as backup:
            source.backup(backup)
```

Choose a new destination in a protected directory. To restore, close owners of the destination database first and restore the complete backup as a new database file; open it with `EmbeddedMemory.open`. Do not replace a live database or copy only its main file while WAL is active. A backup is a point-in-time snapshot: writes committed after that snapshot are not included. Protect and test backups as carefully as the live database.

Cogkura does not set a universal file owner or file mode: the host account, parent directory permissions and process umask control database and sidecar creation. Set those before opening the runtime, restrict access to the database directory and backups, and use host-device encryption as appropriate. POSIX restrictive-umask behavior is tested; behavior on other platforms and filesystems follows their SQLite and OS rules. Cogkura does not provide encryption, secure deletion, tamper detection or cross-user isolation within a shared SQLite file. Tenant scoping is an application storage contract, not an operating-system security boundary.

SQLite documents that WAL with `synchronous=FULL` synchronizes the WAL on each commit; `fullfsync` requests the stronger macOS method where supported. These guarantees depend on the OS, filesystem, device and firmware honoring synchronization and locking. A sudden power loss before an operation completes can leave the caller uncertain whether it committed; retry only through the operation's existing idempotency contract. The recovery tests cover process termination and disk-full errors, not physical power cuts or devices that falsely acknowledge flushes. See SQLite's [synchronous and fullfsync documentation](https://www.sqlite.org/pragma.html#pragma_synchronous).


The ongoing [0.19.6 stabilisation findings](findings/0.19.6-embedded-stabilisation.md) record additional Python compatibility, independent restart and workload evidence. They do not change the durability or filesystem boundaries above.
