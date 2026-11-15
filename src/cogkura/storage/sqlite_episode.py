"""SQLite episodic memory storage."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from types import MappingProxyType
from uuid import uuid4

from cogkura.exceptions import StorageError
from cogkura.models import (
    EpisodeEntity,
    EpisodeEvidenceInput,
    EpisodeInput,
    EpisodeWriteStatus,
    MemoryContextSignature,
    StoredEpisode,
    memory_context_signatures_equal,
)
from cogkura.storage.sqlite import SQLiteStorage, _datetime, _time


def _episode(conn: sqlite3.Connection, row: sqlite3.Row) -> StoredEpisode:
    evidence = tuple(
        EpisodeEvidenceInput(
            observation_id=item["observation_id"],
            observation_revision=item["observation_revision"],
            sequence_number=item["sequence_number"],
            contribution_score=item["contribution_score"],
        )
        for item in conn.execute(
            "SELECT * FROM episode_evidence WHERE tenant_id=? AND episode_id=? "
            "ORDER BY sequence_number,observation_id",
            (row["tenant_id"], row["id"]),
        )
    )
    entities = tuple(
        EpisodeEntity(entity_id=item["entity_id"], role=item["role"])
        for item in conn.execute(
            "SELECT * FROM episode_entities WHERE tenant_id=? AND episode_id=? "
            "ORDER BY entity_id,role",
            (row["tenant_id"], row["id"]),
        )
    )
    started_at, ended_at = _datetime(row["started_at"]), _datetime(row["ended_at"])
    created_at, updated_at = _datetime(row["created_at"]), _datetime(row["updated_at"])
    assert started_at and ended_at and created_at and updated_at
    return StoredEpisode(
        id=row["id"],
        tenant_id=row["tenant_id"],
        subject_id=row["subject_id"],
        memory_key=row["memory_key"],
        statement=row["statement"],
        started_at=started_at,
        ended_at=ended_at,
        confidence=row["confidence"],
        importance=row["importance"],
        is_active=bool(row["is_active"]),
        evidence=evidence,
        entities=entities,
        metadata=MappingProxyType(json.loads(row["metadata"])),
        encoding_context=MemoryContextSignature.from_canonical_dict(
            json.loads(row["encoding_context"])
        ),
        created_at=created_at,
        updated_at=updated_at,
    )


class SQLiteEpisodeStore:
    """Persist episodes, entity links and observation revision evidence."""

    def __init__(self, storage: SQLiteStorage) -> None:
        self._storage = storage

    async def upsert(
        self, episode: EpisodeInput, *, as_of: datetime | None = None
    ) -> EpisodeWriteStatus:
        def operation() -> EpisodeWriteStatus:
            conn = self._storage._conn()
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT * FROM episodes WHERE tenant_id=? AND memory_key=?",
                    (episode.tenant_id, episode.memory_key),
                ).fetchone()
                existing = _episode(conn, row) if row else None
                if (
                    existing is not None
                    and existing.metadata["episode"]["content_fingerprint"]
                    == episode.metadata["episode"]["content_fingerprint"]
                    and memory_context_signatures_equal(
                        existing.encoding_context, episode.encoding_context
                    )
                ):
                    conn.commit()
                    return EpisodeWriteStatus.UNCHANGED
                now = _time(as_of if as_of is not None else datetime.now(UTC))
                episode_id = existing.id if existing else str(uuid4())
                values = (
                    episode.subject_id,
                    episode.statement,
                    _time(episode.started_at),
                    _time(episode.ended_at),
                    episode.confidence,
                    episode.importance,
                    json.dumps(dict(episode.metadata)),
                    json.dumps(episode.encoding_context.to_canonical_dict(), sort_keys=True),
                    now,
                )
                if existing:
                    conn.execute(
                        """UPDATE episodes SET subject_id=?,statement=?,started_at=?,ended_at=?,
                        confidence=?,importance=?,metadata=?,encoding_context=?,updated_at=?,is_active=1
                        WHERE tenant_id=? AND id=?""",
                        (*values, episode.tenant_id, episode_id),
                    )
                    conn.execute(
                        "DELETE FROM episode_evidence WHERE tenant_id=? AND episode_id=?",
                        (episode.tenant_id, episode_id),
                    )
                    conn.execute(
                        "DELETE FROM episode_entities WHERE tenant_id=? AND episode_id=?",
                        (episode.tenant_id, episode_id),
                    )
                else:
                    conn.execute(
                        """INSERT INTO episodes (subject_id,statement,started_at,ended_at,
                        confidence,importance,metadata,encoding_context,updated_at,tenant_id,id,
                        memory_key,created_at,is_active) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,1)""",
                        (*values, episode.tenant_id, episode_id, episode.memory_key, now),
                    )
                for item in episode.evidence:
                    exists = conn.execute(
                        "SELECT 1 FROM observations WHERE id=? AND tenant_id=?",
                        (item.observation_id, episode.tenant_id),
                    ).fetchone()
                    if exists is None:
                        raise StorageError(
                            "Episode evidence observation is missing or belongs to another tenant"
                        )
                    conn.execute(
                        "INSERT INTO episode_evidence VALUES (?,?,?,?,?,?)",
                        (
                            episode.tenant_id,
                            episode_id,
                            item.observation_id,
                            item.observation_revision,
                            item.sequence_number,
                            item.contribution_score,
                        ),
                    )
                for entity in episode.entities:
                    conn.execute(
                        "INSERT INTO episode_entities VALUES (?,?,?,?)",
                        (episode.tenant_id, episode_id, entity.entity_id, entity.role),
                    )
                conn.commit()
                return EpisodeWriteStatus.UPDATED if existing else EpisodeWriteStatus.CREATED
            except BaseException:
                conn.rollback()
                raise

        return await self._storage._run(operation)

    async def list(
        self,
        *,
        tenant_id: str,
        subject_id: str | None = None,
        include_inactive: bool = False,
        limit: int | None = None,
    ) -> list[StoredEpisode]:
        def operation() -> list[StoredEpisode]:
            conn = self._storage._conn()
            conn.execute("BEGIN")
            try:
                sql = "SELECT * FROM episodes WHERE tenant_id=?"
                params: list[object] = [tenant_id]
                if subject_id is not None:
                    sql += " AND subject_id=?"
                    params.append(subject_id)
                if not include_inactive:
                    sql += " AND is_active=1"
                sql += " ORDER BY started_at,id"
                if limit is not None:
                    sql += " LIMIT ?"
                    params.append(limit)
                result = [_episode(conn, row) for row in conn.execute(sql, params)]
                conn.commit()
                return result
            except BaseException:
                conn.rollback()
                raise

        return await self._storage._run(operation)

    async def get_evidence_many(
        self, *, tenant_id: str, episode_ids: set[str]
    ) -> dict[str, tuple[EpisodeEvidenceInput, ...]]:
        def operation() -> dict[str, tuple[EpisodeEvidenceInput, ...]]:
            conn = self._storage._conn()
            ids = sorted(episode_ids)
            result: dict[str, tuple[EpisodeEvidenceInput, ...]] = {}
            conn.execute("BEGIN")
            try:
                for i in range(0, len(ids), 500):
                    chunk = ids[i : i + 500]
                    placeholders = ",".join("?" for _ in chunk)
                    for row in conn.execute(
                        f"SELECT * FROM episodes WHERE tenant_id=? AND id IN ({placeholders})",
                        (tenant_id, *chunk),
                    ):
                        episode = _episode(conn, row)
                        result[episode.id] = episode.evidence
                conn.commit()
                return result
            except BaseException:
                conn.rollback()
                raise

        return await self._storage._run(operation)

    async def deactivate_missing(
        self,
        *,
        tenant_id: str,
        subject_id: str | None,
        active_memory_keys: set[str],
        as_of: datetime | None = None,
    ) -> int:
        def operation() -> int:
            conn = self._storage._conn()
            conn.execute("BEGIN IMMEDIATE")
            try:
                sql = "SELECT id,memory_key FROM episodes WHERE tenant_id=? AND is_active=1"
                params: list[object] = [tenant_id]
                if subject_id is not None:
                    sql += " AND subject_id=?"
                    params.append(subject_id)
                ids = [
                    row["id"]
                    for row in conn.execute(sql, params)
                    if row["memory_key"] not in active_memory_keys
                ]
                now = _time(as_of if as_of is not None else datetime.now(UTC))
                conn.executemany(
                    "UPDATE episodes SET is_active=0,updated_at=? WHERE tenant_id=? AND id=?",
                    [(now, tenant_id, item) for item in ids],
                )
                conn.commit()
                return len(ids)
            except BaseException:
                conn.rollback()
                raise

        return await self._storage._run(operation)

    async def clear(self, *, tenant_id: str) -> None:
        await self._storage._run(
            lambda: self._storage._conn().execute(
                "DELETE FROM episodes WHERE tenant_id=?", (tenant_id,)
            )
        )
