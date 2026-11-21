"""Dependency-free SQLite observation and checkpoint stores.

A SQLiteStorage instance owns a worker and connection. Call ``await close()``
when finished; cancellation of an awaiting write does not guarantee rollback.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeVar
from uuid import uuid4

from cogkura.exceptions import StorageError
from cogkura.observations.encoding_context import ObservationContext, observation_contexts_equal
from cogkura.observations.models import (
    IngestStatus,
    ObservationInput,
    ObservationRevisionIdentity,
    StoredObservation,
    StoredObservationRevision,
)
from cogkura.observations.retention import RetainedObservation

_T = TypeVar("_T")
_APPLICATION_ID = 0x434F474B


def _has_safe_wal_version(version: tuple[int, int, int]) -> bool:
    """Recognize SQLite releases with the 2026 WAL-reset corruption fix."""
    return (
        (version >= (3, 51, 3) and version != (3, 52, 0))
        or (3, 50, 7) <= version < (3, 51, 0)
        or (3, 44, 6) <= version < (3, 45, 0)
    )


def _migrations() -> list[Path]:
    return sorted((Path(__file__).parent.parent / "migrations" / "sqlite").glob("*.sql"))


def _initialize(conn: sqlite3.Connection) -> None:
    conn.execute("BEGIN IMMEDIATE")
    try:
        app_id = conn.execute("PRAGMA application_id").fetchone()[0]
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        if app_id == 0 and not tables:
            conn.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
            conn.execute("CREATE TABLE schema_migrations (version TEXT PRIMARY KEY)")
        elif app_id != _APPLICATION_ID or "schema_migrations" not in tables:
            raise StorageError("Database is not a compatible Cogkura SQLite database")
        paths = _migrations()
        if not paths:
            raise StorageError("SQLite migration files are missing")
        versions = [
            row[0] for row in conn.execute("SELECT version FROM schema_migrations ORDER BY version")
        ]
        expected = [path.stem for path in paths]
        if versions != expected[: len(versions)]:
            raise StorageError("Unsupported or inconsistent SQLite schema version")
        for path in paths[len(versions) :]:
            sql = path.read_text(encoding="utf-8")
            statement = ""
            for line in sql.splitlines(keepends=True):
                statement += line
                if sqlite3.complete_statement(statement):
                    conn.execute(statement)
                    statement = ""
            if statement.strip():
                raise StorageError(f"Incomplete SQLite migration: {path.name}")
            conn.execute("INSERT INTO schema_migrations (version) VALUES (?)", (path.stem,))
        required = {
            "schema_migrations",
            "observations",
            "observation_revisions",
            "connector_checkpoints",
        }
        if "002_episodes" in expected:
            required.update({"episodes", "episode_evidence", "episode_entities"})
        if "003_semantic" in expected:
            required.update(
                {
                    "semantic_memories",
                    "semantic_revisions",
                    "semantic_derivations",
                    "semantic_evidence",
                    "semantic_entities",
                    "semantic_relations",
                }
            )
        if "004_cognitive_state" in expected:
            required.update(
                {
                    "activation_references",
                    "memory_dynamics",
                    "learning_events",
                    "learning_feedback",
                    "learning_state",
                    "learned_associations",
                    "inhibition_traces",
                    "entity_relationships",
                }
            )
        if "005_episode_history" in expected:
            required.add("episode_history")
        if "006_semantic_history" in expected:
            required.add("semantic_history")
        if "007_revision_history" in expected:
            required.update({"semantic_revision_history", "semantic_relation_history"})
        if "008_history_order" in expected:
            required.add("history_order")
        actual = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if not required <= actual:
            raise StorageError("Incompatible SQLite schema: required tables are missing")
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


class SQLiteStorage:
    """Own one SQLite connection on a dedicated worker; share it between stores."""

    def __init__(self, path: str | Path, *, busy_timeout: float = 5.0) -> None:
        if busy_timeout < 0:
            raise ValueError("busy_timeout must not be negative")
        self._path = str(path)
        self._timeout = busy_timeout
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cogkura-sqlite")
        self._connection: sqlite3.Connection | None = None
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None

    @classmethod
    async def open(cls, path: str | Path, *, busy_timeout: float = 5.0) -> SQLiteStorage:
        """Open a file database, configure it and apply pending migrations."""
        owner = cls(path, busy_timeout=busy_timeout)
        try:
            await owner._run(owner._open)
        except BaseException:
            await owner.close()
            raise
        return owner

    def _open(self) -> None:
        if not _has_safe_wal_version(sqlite3.sqlite_version_info):
            raise StorageError(
                f"SQLite {sqlite3.sqlite_version} may contain the WAL-reset corruption bug; "
                "use a fixed SQLite release (3.51.3+, excluding withdrawn 3.52.0, "
                "or the 3.50.7/3.44.6 backports)."
            )
        conn = sqlite3.connect(self._path, timeout=self._timeout, isolation_level=None)
        self._connection = conn
        conn.row_factory = sqlite3.Row
        # Do not change journal mode on an unrelated populated database.
        conn.execute("BEGIN")
        app_id = conn.execute("PRAGMA application_id").fetchone()[0]
        has_tables = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' LIMIT 1"
        ).fetchone()
        conn.commit()
        if app_id != _APPLICATION_ID and (app_id != 0 or has_tables):
            raise StorageError("Database is not a compatible Cogkura SQLite database")
        conn.execute("PRAGMA foreign_keys = ON")
        if conn.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
            raise StorageError("SQLite foreign keys could not be enabled")
        deadline = time.monotonic() + self._timeout
        while True:
            try:
                mode = conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
                break
            except sqlite3.OperationalError as exc:
                if exc.sqlite_errorcode not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise
                time.sleep(min(0.01, remaining))
        if mode.lower() != "wal":
            raise StorageError("SQLite WAL mode could not be enabled")
        conn.execute("PRAGMA synchronous = FULL")
        if sys.platform == "darwin":
            conn.execute("PRAGMA fullfsync = ON")
            if conn.execute("PRAGMA fullfsync").fetchone()[0] != 1:
                raise StorageError("SQLite fullfsync could not be enabled")
        if conn.execute("PRAGMA synchronous").fetchone()[0] != 2:
            raise StorageError("SQLite FULL synchronous mode could not be enabled")
        _initialize(conn)

    async def _run(self, operation: Callable[[], _T]) -> _T:
        if self._closing:
            raise StorageError("SQLite storage is closed")
        loop = asyncio.get_running_loop()

        def execute() -> _T:
            try:
                return operation()
            except (sqlite3.Error, OSError) as exc:
                raise StorageError(f"SQLite operation failed: {exc}") from exc

        return await loop.run_in_executor(self._executor, execute)

    def _conn(self) -> sqlite3.Connection:
        if self._connection is None:
            raise StorageError("SQLite storage is not open")
        return self._connection

    async def close(self) -> None:
        """Drain accepted operations and release the connection and worker."""
        if self._close_task is None:
            self._closing = True
            self._close_task = asyncio.create_task(self._finish_close())
        await asyncio.shield(self._close_task)

    async def _finish_close(self) -> None:
        def finish() -> None:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

        try:
            await asyncio.get_running_loop().run_in_executor(self._executor, finish)
        finally:
            self._executor.shutdown(wait=True)

    async def __aenter__(self) -> SQLiteStorage:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()


def _time(value: datetime | None) -> str | None:
    return value.astimezone(UTC).isoformat(timespec="microseconds") if value else None


def _datetime(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value is not None else None


def _stored(row: sqlite3.Row) -> StoredObservation:
    observed_at = _datetime(row["last_observed_at"])
    assert observed_at is not None
    return StoredObservation(
        id=row["id"],
        tenant_id=row["tenant_id"],
        subject_id=row["subject_id"],
        actor_id=row["actor_id"],
        source_type=row["source_type"],
        source_namespace=row["source_namespace"],
        source_record_id=row["source_record_id"],
        source_version=row["source_version"],
        event_type=row["event_type"],
        content=row["content"],
        content_hash=row["content_hash"],
        metadata=MappingProxyType(json.loads(row["metadata"])),
        source_created_at=_datetime(row["source_created_at"]),
        source_updated_at=_datetime(row["source_updated_at"]),
        observed_at=observed_at,
        current_revision=row["current_revision"],
        is_deleted=bool(row["is_deleted"]),
        attention_score=row["attention_score"],
        retention_class=row["retention_class"],
        policy_reasons=tuple(json.loads(row["policy_reasons"])),
        context=ObservationContext.from_canonical_dict(json.loads(row["encoding_context"])),
    )


class SQLiteObservationStore:
    """Persist current observations and their exact historical revisions."""

    def __init__(self, storage: SQLiteStorage) -> None:
        self._storage = storage

    async def ingest(
        self, observation: ObservationInput, *, retained: RetainedObservation
    ) -> IngestStatus:
        def operation() -> IngestStatus:
            conn = self._storage._conn()
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT * FROM observations WHERE tenant_id=? "
                    "AND source_namespace=? AND source_record_id=?",
                    (
                        observation.tenant_id,
                        observation.source_namespace,
                        observation.source_record_id,
                    ),
                ).fetchone()
                existing = _stored(row) if row else None
                if existing is None:
                    status, revision, obs_id = IngestStatus.CREATED, 1, str(uuid4())
                elif (
                    existing.source_version == observation.source_version
                    and existing.content_hash == retained.content_hash
                    and existing.is_deleted == observation.is_deleted
                    and observation_contexts_equal(existing.context, retained.context)
                ):
                    conn.commit()
                    return IngestStatus.UNCHANGED
                else:
                    status = (
                        IngestStatus.RESTORED
                        if existing.is_deleted and not observation.is_deleted
                        else IngestStatus.DELETED
                        if observation.is_deleted and not existing.is_deleted
                        else IngestStatus.UPDATED
                    )
                    revision, obs_id = existing.current_revision + 1, existing.id
                metadata = json.dumps(retained.metadata)
                context = json.dumps(retained.context.to_canonical_dict(), sort_keys=True)
                values = (
                    observation.subject_id,
                    observation.actor_id,
                    observation.source_type,
                    observation.source_version,
                    observation.event_type,
                    retained.content,
                    retained.content_hash,
                    metadata,
                    context,
                    _time(observation.source_created_at),
                    _time(observation.source_updated_at),
                    _time(observation.observed_at),
                    revision,
                    int(observation.is_deleted),
                    retained.attention_score,
                    retained.retention_class,
                    json.dumps(list(retained.policy_reasons)),
                )
                if existing is None:
                    conn.execute(
                        """INSERT INTO observations
                        (subject_id,actor_id,source_type,source_version,event_type,content,content_hash,
                         metadata,encoding_context,source_created_at,source_updated_at,last_observed_at,
                         current_revision,is_deleted,attention_score,retention_class,policy_reasons,
                         id,tenant_id,source_namespace,source_record_id,first_observed_at)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            *values,
                            obs_id,
                            observation.tenant_id,
                            observation.source_namespace,
                            observation.source_record_id,
                            _time(observation.observed_at),
                        ),
                    )
                else:
                    conn.execute(
                        """UPDATE observations SET subject_id=?,actor_id=?,source_type=?,
                        source_version=?,event_type=?,content=?,content_hash=?,metadata=?,encoding_context=?,
                        source_created_at=?,source_updated_at=?,last_observed_at=?,current_revision=?,
                        is_deleted=?,attention_score=?,retention_class=?,policy_reasons=?
                        WHERE id=?""",
                        (*values, obs_id),
                    )
                conn.execute(
                    """INSERT INTO observation_revisions
                    (id,observation_id,revision_number,source_version,content,content_hash,metadata,
                     encoding_context,change_type,observed_at) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (
                        str(uuid4()),
                        obs_id,
                        revision,
                        observation.source_version,
                        retained.content,
                        retained.content_hash,
                        metadata,
                        context,
                        status.value,
                        _time(observation.observed_at),
                    ),
                )
                conn.commit()
                return status
            except BaseException:
                conn.rollback()
                raise

        return await self._storage._run(operation)

    async def get_by_source(
        self, *, tenant_id: str, source_namespace: str, source_record_id: str
    ) -> StoredObservation | None:
        def operation() -> StoredObservation | None:
            row = (
                self._storage._conn()
                .execute(
                    "SELECT * FROM observations WHERE tenant_id=? "
                    "AND source_namespace=? AND source_record_id=?",
                    (tenant_id, source_namespace, source_record_id),
                )
                .fetchone()
            )
            return _stored(row) if row else None

        return await self._storage._run(operation)

    async def list(
        self, *, tenant_id: str, subject_id: str | None = None, include_deleted: bool = False
    ) -> list[StoredObservation]:
        def operation() -> list[StoredObservation]:
            sql = "SELECT * FROM observations WHERE tenant_id=?"
            params: list[Any] = [tenant_id]
            if subject_id is not None:
                sql += " AND subject_id=?"
                params.append(subject_id)
            if not include_deleted:
                sql += " AND is_deleted=0"
            sql += " ORDER BY last_observed_at,id"
            return [_stored(row) for row in self._storage._conn().execute(sql, params)]

        return await self._storage._run(operation)

    async def get_many(
        self, *, tenant_id: str, observation_ids: set[str]
    ) -> Sequence[StoredObservation]:
        def operation() -> list[StoredObservation]:
            conn = self._storage._conn()
            ids = sorted(observation_ids)
            rows: list[StoredObservation] = []
            conn.execute("BEGIN")
            try:
                for i in range(0, len(ids), 500):
                    chunk = ids[i : i + 500]
                    marks = ",".join("?" for _ in chunk)
                    rows.extend(
                        _stored(row)
                        for row in conn.execute(
                            f"SELECT * FROM observations WHERE tenant_id=? AND id IN ({marks})",
                            (tenant_id, *chunk),
                        )
                    )
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
            return sorted(rows, key=lambda row: (row.observed_at, row.id))

        return await self._storage._run(operation)

    async def get_revisions_many(
        self, *, tenant_id: str, references: Sequence[ObservationRevisionIdentity]
    ) -> Mapping[ObservationRevisionIdentity, StoredObservationRevision]:
        def operation() -> dict[ObservationRevisionIdentity, StoredObservationRevision]:
            conn = self._storage._conn()
            refs = sorted(set(references), key=lambda r: (r.observation_id, r.observation_revision))
            result: dict[ObservationRevisionIdentity, StoredObservationRevision] = {}
            conn.execute("BEGIN")
            try:
                for i in range(0, len(refs), 400):
                    chunk = refs[i : i + 400]
                    conditions = " OR ".join(
                        "(r.observation_id=? AND r.revision_number=?)" for _ in chunk
                    )
                    params: list[Any] = [tenant_id]
                    for ref in chunk:
                        params.extend((ref.observation_id, ref.observation_revision))
                    for row in conn.execute(
                        f"""SELECT o.id, o.tenant_id, o.source_namespace,
                        o.source_record_id, o.current_revision, o.is_deleted, r.revision_number,
                        r.source_version, r.metadata, r.observed_at FROM observations o
                        JOIN observation_revisions r ON r.observation_id=o.id
                        WHERE o.tenant_id=? AND ({conditions}) ORDER BY o.id,r.revision_number""",
                        params,
                    ):
                        identity = ObservationRevisionIdentity(row["id"], row["revision_number"])
                        observed_at = _datetime(row["observed_at"])
                        assert observed_at is not None
                        result[identity] = StoredObservationRevision(
                            identity=identity,
                            tenant_id=row["tenant_id"],
                            source_namespace=row["source_namespace"],
                            source_record_id=row["source_record_id"],
                            source_version=row["source_version"],
                            metadata=json.loads(row["metadata"]),
                            observed_at=observed_at,
                            current_revision=row["current_revision"],
                            is_deleted=bool(row["is_deleted"]),
                        )
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
            return result

        return await self._storage._run(operation)

    async def clear(self, *, tenant_id: str) -> None:
        def operation() -> None:
            self._storage._conn().execute(
                "DELETE FROM observations WHERE tenant_id=?", (tenant_id,)
            )

        await self._storage._run(operation)


class SQLiteCheckpointStore:
    """Persist connector checkpoints scoped to a tenant."""

    def __init__(self, storage: SQLiteStorage) -> None:
        self._storage = storage

    async def get(self, *, tenant_id: str, connector_id: str) -> dict[str, Any] | None:
        def operation() -> dict[str, Any] | None:
            row = (
                self._storage._conn()
                .execute(
                    "SELECT checkpoint FROM connector_checkpoints "
                    "WHERE tenant_id=? AND connector_id=?",
                    (tenant_id, connector_id),
                )
                .fetchone()
            )
            return dict(json.loads(row[0])) if row else None

        return await self._storage._run(operation)

    async def set(self, *, tenant_id: str, connector_id: str, checkpoint: dict[str, Any]) -> None:
        payload = json.dumps(checkpoint)

        def operation() -> None:
            self._storage._conn().execute(
                """INSERT INTO connector_checkpoints (tenant_id,connector_id,checkpoint)
                VALUES (?,?,?) ON CONFLICT (tenant_id,connector_id)
                DO UPDATE SET checkpoint=excluded.checkpoint""",
                (tenant_id, connector_id, payload),
            )

        await self._storage._run(operation)
