"""SQLite persistence for semantic claims, revisions and reconciliation."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections import defaultdict
from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import fields, replace
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, TypeVar

from cogkura.exceptions import StorageError
from cogkura.models import (
    EpisodeEntity,
    EpisodeEvidenceInput,
    SemanticCardinality,
    SemanticDerivationInput,
    SemanticDerivationRelation,
    SemanticMemoryInput,
    SemanticMemoryStatus,
    SemanticPolarity,
    SemanticReconciliationPlan,
    SemanticReconciliationWriteResult,
    SemanticRevisionRelation,
    SemanticSlotRecords,
    SemanticUpdateRelation,
    SemanticWriteStatus,
    SourceInspectionLimits,
    StoredSemanticMemory,
    StoredSemanticRevision,
)
from cogkura.storage.historical_snapshots import select_semantic_snapshot
from cogkura.storage.in_memory_semantic import InMemorySemanticMemoryStore
from cogkura.storage.sqlite import SQLiteStorage, _datetime, _time

_T = TypeVar("_T")
_MEM_FIELDS = tuple(
    f.name
    for f in fields(StoredSemanticMemory)
    if f.name not in {"derivations", "observation_evidence", "entities"}
)
_REV_FIELDS = tuple(f.name for f in fields(StoredSemanticRevision) if f.name != "derivations")
_DATES = {
    "first_supported_at",
    "last_supported_at",
    "valid_from",
    "valid_until",
    "created_at",
    "updated_at",
}
_ENUMS: dict[str, type[Enum]] = {
    "status": SemanticMemoryStatus,
    "polarity": SemanticPolarity,
    "cardinality": SemanticCardinality,
}
_JSON = {"qualifiers", "metadata", "snapshot"}


def _decode(row: sqlite3.Row, names: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in names:
        value = row[name]
        if name in _DATES:
            value = _datetime(value)
        elif name in _ENUMS:
            value = _ENUMS[name](value)
        elif name in _JSON:
            value = MappingProxyType(json.loads(value))
        elif name == "is_active":
            value = bool(value)
        result[name] = value
    return result


def _encode(value: object) -> Any:
    if isinstance(value, datetime):
        return _time(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return json.dumps(dict(value))
    if isinstance(value, bool):
        return int(value)
    return value


def _upsert(
    conn: sqlite3.Connection, table: str, names: tuple[str, ...], obj: Any, key: tuple[str, ...]
) -> None:
    columns = ",".join(names)
    placeholders = ",".join("?" for _ in names)
    updates = ",".join(f"{name}=excluded.{name}" for name in names if name not in key)
    conflict = ",".join(key)
    conn.execute(
        f"INSERT INTO {table} ({columns}) VALUES ({placeholders}) "
        f"ON CONFLICT ({conflict}) DO UPDATE SET {updates}",
        tuple(_encode(getattr(obj, name)) for name in names),
    )


def _derivations(
    conn: sqlite3.Connection, tenant: str, memory_key: str, revision_key: str
) -> tuple[SemanticDerivationInput, ...]:
    return tuple(
        SemanticDerivationInput(
            episode_id=row["episode_id"],
            relation=SemanticDerivationRelation(row["relation"]),
            contribution_score=row["contribution_score"],
        )
        for row in conn.execute(
            "SELECT * FROM semantic_derivations WHERE tenant_id=? AND memory_key=? "
            "AND revision_key=? ORDER BY episode_id,relation",
            (tenant, memory_key, revision_key),
        )
    )


def _load_memory(conn: sqlite3.Connection, row: sqlite3.Row) -> StoredSemanticMemory:
    tenant, key = row["tenant_id"], row["memory_key"]
    evidence = tuple(
        EpisodeEvidenceInput(
            observation_id=item["observation_id"],
            observation_revision=item["observation_revision"],
            sequence_number=item["sequence_number"],
            contribution_score=item["contribution_score"],
        )
        for item in conn.execute(
            "SELECT * FROM semantic_evidence WHERE tenant_id=? AND memory_key=? "
            "ORDER BY sequence_number,observation_id",
            (tenant, key),
        )
    )
    entities = tuple(
        EpisodeEntity(entity_id=item["entity_id"], role=item["role"])
        for item in conn.execute(
            "SELECT * FROM semantic_entities WHERE tenant_id=? AND memory_key=? "
            "ORDER BY entity_id,role",
            (tenant, key),
        )
    )
    return StoredSemanticMemory(
        **_decode(row, _MEM_FIELDS),
        derivations=_derivations(conn, tenant, key, ""),
        observation_evidence=evidence,
        entities=entities,
    )


def _load_revision(conn: sqlite3.Connection, row: sqlite3.Row) -> StoredSemanticRevision:
    return StoredSemanticRevision(
        **_decode(row, _REV_FIELDS),
        derivations=_derivations(conn, row["tenant_id"], row["memory_key"], row["revision_key"]),
    )


def _hydrate(
    conn: sqlite3.Connection,
    tenants: set[str],
    *,
    include_memories: bool = True,
    include_revisions: bool = True,
    include_relations: bool = True,
) -> InMemorySemanticMemoryStore:
    """Read each tenant's semantic inventory in bounded query batches.

    Reconstruct the same ordered child tuples as the individual row loaders.
    The caller holds one read or write transaction across the complete snapshot.
    """
    state = InMemorySemanticMemoryStore()
    for tenant in sorted(tenants):
        evidence: dict[str, list[EpisodeEvidenceInput]] = defaultdict(list)
        entities: dict[str, list[EpisodeEntity]] = defaultdict(list)
        if include_memories:
            for row in conn.execute(
                "SELECT * FROM semantic_evidence WHERE tenant_id=? "
                "ORDER BY memory_key,sequence_number,observation_id",
                (tenant,),
            ):
                evidence[row["memory_key"]].append(
                    EpisodeEvidenceInput(
                        observation_id=row["observation_id"],
                        observation_revision=row["observation_revision"],
                        sequence_number=row["sequence_number"],
                        contribution_score=row["contribution_score"],
                    )
                )
            for row in conn.execute(
                "SELECT * FROM semantic_entities WHERE tenant_id=? "
                "ORDER BY memory_key,entity_id,role",
                (tenant,),
            ):
                entities[row["memory_key"]].append(
                    EpisodeEntity(entity_id=row["entity_id"], role=row["role"])
                )
        derivations: dict[tuple[str, str], list[SemanticDerivationInput]] = defaultdict(list)
        for row in conn.execute(
            "SELECT * FROM semantic_derivations WHERE tenant_id=? "
            "ORDER BY memory_key,revision_key,episode_id,relation",
            (tenant,),
        ):
            derivations[(row["memory_key"], row["revision_key"])].append(
                SemanticDerivationInput(
                    episode_id=row["episode_id"],
                    relation=SemanticDerivationRelation(row["relation"]),
                    contribution_score=row["contribution_score"],
                )
            )
        if include_memories:
            for row in conn.execute("SELECT * FROM semantic_memories WHERE tenant_id=?", (tenant,)):
                memory_key = row["memory_key"]
                state._memories[(tenant, memory_key)] = StoredSemanticMemory(
                    **_decode(row, _MEM_FIELDS),
                    derivations=tuple(derivations[(memory_key, "")]),
                    observation_evidence=tuple(evidence[memory_key]),
                    entities=tuple(entities[memory_key]),
                )
        if include_revisions:
            for row in conn.execute(
                "SELECT * FROM semantic_revisions WHERE tenant_id=?", (tenant,)
            ):
                revision_key = row["revision_key"]
                state._revisions[(tenant, revision_key)] = StoredSemanticRevision(
                    **_decode(row, _REV_FIELDS),
                    derivations=tuple(derivations[(row["memory_key"], revision_key)]),
                )
        if include_relations:
            for row in conn.execute(
                "SELECT * FROM semantic_relations WHERE tenant_id=?", (tenant,)
            ):
                key = (tenant, row["left_revision_key"], row["right_revision_key"], row["relation"])
                state._relations[key] = (
                    row["left_revision_key"],
                    row["right_revision_key"],
                    row["relation"],
                    row["effective_at"],
                )
    return state


def _write_derivations(
    conn: sqlite3.Connection,
    tenant: str,
    memory_key: str,
    revision_key: str,
    values: Sequence[SemanticDerivationInput],
) -> None:
    conn.execute(
        "DELETE FROM semantic_derivations WHERE tenant_id=? AND memory_key=? AND revision_key=?",
        (tenant, memory_key, revision_key),
    )
    conn.executemany(
        "INSERT INTO semantic_derivations VALUES (?,?,?,?,?,?)",
        [
            (
                tenant,
                memory_key,
                revision_key,
                item.episode_id,
                item.relation.value,
                item.contribution_score,
            )
            for item in values
        ],
    )


def _flush(
    conn: sqlite3.Connection,
    before: InMemorySemanticMemoryStore,
    after: InMemorySemanticMemoryStore,
) -> None:
    for key, memory in after._memories.items():
        if before._memories.get(key) == memory:
            continue
        tenant, memory_key = key
        _upsert(conn, "semantic_memories", _MEM_FIELDS, memory, ("tenant_id", "memory_key"))
        _write_derivations(conn, tenant, memory_key, "", memory.derivations)
        conn.execute("DELETE FROM semantic_evidence WHERE tenant_id=? AND memory_key=?", key)
        for item in memory.observation_evidence:
            if not conn.execute(
                "SELECT 1 FROM observations WHERE tenant_id=? AND id=?",
                (tenant, item.observation_id),
            ).fetchone():
                raise StorageError(
                    "Semantic evidence observation is missing or belongs to another tenant"
                )
            conn.execute(
                "INSERT INTO semantic_evidence VALUES (?,?,?,?,?,?)",
                (
                    tenant,
                    memory_key,
                    item.observation_id,
                    item.observation_revision,
                    item.sequence_number,
                    item.contribution_score,
                ),
            )
        conn.execute("DELETE FROM semantic_entities WHERE tenant_id=? AND memory_key=?", key)
        conn.executemany(
            "INSERT INTO semantic_entities VALUES (?,?,?,?)",
            [(tenant, memory_key, item.entity_id, item.role) for item in memory.entities],
        )
    for key, revision in after._revisions.items():
        if before._revisions.get(key) == revision:
            continue
        _upsert(conn, "semantic_revisions", _REV_FIELDS, revision, ("tenant_id", "revision_key"))
        _write_derivations(
            conn,
            revision.tenant_id,
            revision.memory_key,
            revision.revision_key,
            revision.derivations,
        )
    for relation_key, relation in after._relations.items():
        if relation_key in before._relations:
            continue
        conn.execute(
            "INSERT INTO semantic_relations VALUES (?,?,?,?,?)", (*relation_key, relation[3])
        )


class SQLiteSemanticMemoryStore:
    """Persist semantic revision state using the shared SQLite worker.

    Current implementation materializes the tenant's semantic inventory per operation.
    """

    def __init__(self, storage: SQLiteStorage) -> None:
        self._storage = storage

    async def _read(
        self,
        tenant: str,
        operation: Callable[[InMemorySemanticMemoryStore], Coroutine[Any, Any, _T]],
        *,
        include_memories: bool = True,
        include_revisions: bool = True,
    ) -> _T:
        def work() -> _T:
            conn = self._storage._conn()
            conn.execute("BEGIN")
            try:
                state = _hydrate(
                    conn,
                    {tenant},
                    include_memories=include_memories,
                    include_revisions=include_revisions,
                    include_relations=False,
                )
                result = asyncio.run(operation(state))
                conn.commit()
                return result
            except BaseException:
                conn.rollback()
                raise

        return await self._storage._run(work)

    async def _write(
        self,
        tenants: set[str],
        operation: Callable[[InMemorySemanticMemoryStore], Coroutine[Any, Any, _T]],
    ) -> _T:
        def work() -> _T:
            conn = self._storage._conn()
            conn.execute("BEGIN IMMEDIATE")
            try:
                before = _hydrate(conn, tenants)
                after = InMemorySemanticMemoryStore()
                after._memories = dict(before._memories)
                after._revisions = dict(before._revisions)
                after._relations = dict(before._relations)
                result = asyncio.run(operation(after))
                _flush(conn, before, after)
                if conn.execute(
                    "SELECT 1 FROM schema_migrations WHERE version='006_semantic_history'"
                ).fetchone():
                    for (tenant, memory_id), versions in after._semantic_history.items():
                        conn.executemany(
                            "INSERT INTO semantic_history "
                            "(tenant_id,memory_id,recorded_at,snapshot) "
                            "VALUES (?,?,?,?)",
                            [(tenant, memory_id, _time(at), payload) for at, payload in versions],
                        )
                conn.commit()
                return result
            except BaseException:
                conn.rollback()
                raise

        return await self._storage._run(work)

    async def historical_semantics(
        self, *, tenant_id: str, memory_ids: set[str], as_of: datetime
    ) -> dict[str, StoredSemanticMemory]:
        def work() -> dict[str, StoredSemanticMemory]:
            conn = self._storage._conn()
            result: dict[str, StoredSemanticMemory] = {}
            conn.execute("BEGIN")
            try:
                exists = conn.execute(
                    "SELECT 1 FROM schema_migrations WHERE version='006_semantic_history'"
                ).fetchone()
                ids = sorted(memory_ids) if exists else []
                for i in range(0, len(ids), 500):
                    chunk = ids[i : i + 500]
                    marks = ",".join("?" for _ in chunk)
                    rows = conn.execute(
                        f"SELECT memory_id,recorded_at,snapshot FROM semantic_history "
                        f"WHERE tenant_id=? AND memory_id IN ({marks}) ORDER BY sequence",
                        (tenant_id, *chunk),
                    )
                    grouped: dict[str, list[tuple[datetime, str | None]]] = {}
                    for row in rows:
                        at = _datetime(row["recorded_at"])
                        assert at is not None
                        grouped.setdefault(row["memory_id"], []).append((at, row["snapshot"]))
                    for memory_id, versions in grouped.items():
                        memory = select_semantic_snapshot(versions, as_of=as_of)
                        if memory is not None:
                            result[memory_id] = memory
                conn.commit()
                return result
            except BaseException:
                conn.rollback()
                raise

        return await self._storage._run(work)

    async def upsert(
        self, memory: SemanticMemoryInput, *, as_of: datetime | None = None
    ) -> SemanticWriteStatus:
        return await self._write(
            {memory.tenant_id}, lambda store: store.upsert(memory, as_of=as_of)
        )

    async def list(
        self,
        *,
        tenant_id: str,
        subject_id: str | None = None,
        include_inactive: bool = False,
        status: SemanticMemoryStatus | None = None,
        limit: int | None = None,
        valid_at: datetime | None = None,
    ) -> Sequence[StoredSemanticMemory]:
        return await self._read(
            tenant_id,
            lambda store: store.list(
                tenant_id=tenant_id,
                subject_id=subject_id,
                include_inactive=include_inactive,
                status=status,
                limit=limit,
                valid_at=valid_at,
            ),
            include_revisions=valid_at is not None,
        )

    async def list_revisions(
        self,
        *,
        tenant_id: str,
        memory_key: str | None = None,
        subject_id: str | None = None,
        valid_at: datetime | None = None,
        limit: int | None = None,
    ) -> Sequence[StoredSemanticRevision]:
        return await self._read(
            tenant_id,
            lambda store: store.list_revisions(
                tenant_id=tenant_id,
                memory_key=memory_key,
                subject_id=subject_id,
                valid_at=valid_at,
                limit=limit,
            ),
            include_memories=subject_id is not None,
        )

    async def inspect_slot(
        self,
        *,
        tenant_id: str,
        subject_id: str | None,
        slot_key: str,
        max_revisions: int,
        max_relations: int,
    ) -> SemanticSlotRecords:
        SourceInspectionLimits(max_revisions=max_revisions, max_relations=max_relations)

        def work() -> SemanticSlotRecords:
            conn = self._storage._conn()
            conn.execute("BEGIN")
            try:
                subject_clause = " AND m.subject_id=?" if subject_id is not None else ""
                params: list[Any] = [tenant_id, slot_key]
                if subject_id is not None:
                    params.append(subject_id)
                predicate = "m.tenant_id=? AND m.slot_key=?" + subject_clause
                memories = list(
                    conn.execute(
                        "SELECT m.* FROM semantic_memories m WHERE "
                        + predicate
                        + " ORDER BY m.memory_key,m.revision_number,m.revision_key LIMIT ?",
                        (*params, max_revisions + 1),
                    )
                )
                revisions = list(
                    conn.execute(
                        "SELECT r.* FROM semantic_revisions r JOIN semantic_memories m "
                        "ON m.tenant_id=r.tenant_id AND m.memory_key=r.memory_key WHERE "
                        + predicate
                        + " ORDER BY r.memory_key,r.revision_number,r.revision_key LIMIT ?",
                        (*params, max_revisions + 1),
                    )
                )
                membership = (
                    "SELECT r.revision_key FROM semantic_revisions r JOIN semantic_memories m "
                )
                membership += (
                    "ON m.tenant_id=r.tenant_id AND m.memory_key=r.memory_key WHERE " + predicate
                )
                relations = list(
                    conn.execute(
                        "SELECT rel.* FROM semantic_relations rel WHERE rel.tenant_id=? "
                        "AND (rel.left_revision_key IN ("
                        + membership
                        + ") OR rel.right_revision_key IN ("
                        + membership
                        + ")) "
                        "ORDER BY rel.left_revision_key,rel.right_revision_key,"
                        "rel.relation LIMIT ?",
                        (tenant_id, *params, *params, max_relations + 1),
                    )
                )
                result = SemanticSlotRecords(
                    memories=tuple(_load_memory(conn, row) for row in memories[:max_revisions]),
                    revisions=tuple(_load_revision(conn, row) for row in revisions[:max_revisions]),
                    relations=tuple(
                        SemanticRevisionRelation(
                            tenant_id=tenant_id,
                            left_revision_key=row["left_revision_key"],
                            right_revision_key=row["right_revision_key"],
                            relation=SemanticUpdateRelation(row["relation"]),
                            effective_at=_datetime(row["effective_at"]),
                        )
                        for row in relations[:max_relations]
                    ),
                    memories_overflow=len(memories) > max_revisions,
                    revisions_overflow=len(revisions) > max_revisions,
                    relations_overflow=len(relations) > max_relations,
                )
                conn.commit()
                return result
            except BaseException:
                conn.rollback()
                raise

        return await self._storage._run(work)

    async def apply_reconciliation(
        self, plan: SemanticReconciliationPlan, *, as_of: datetime | None = None
    ) -> SemanticReconciliationWriteResult:
        tenants = (
            {m.tenant_id for m in plan.current_memories}
            | {r.tenant_id for r in plan.revisions}
            | {r.tenant_id for r in plan.relations}
        )

        async def apply(store: InMemorySemanticMemoryStore) -> SemanticReconciliationWriteResult:
            keys = set(store._memories) | {
                (m.tenant_id, m.memory_key) for m in plan.current_memories
            }
            valid = tuple(r for r in plan.revisions if (r.tenant_id, r.memory_key) in keys)
            return await store.apply_reconciliation(replace(plan, revisions=valid), as_of=as_of)

        return await self._write(tenants, apply)

    async def deactivate_missing(
        self,
        *,
        tenant_id: str,
        subject_id: str | None,
        active_memory_keys: set[str],
        as_of: datetime | None = None,
    ) -> int:
        return await self._write(
            {tenant_id},
            lambda store: store.deactivate_missing(
                tenant_id=tenant_id,
                subject_id=subject_id,
                active_memory_keys=active_memory_keys,
                as_of=as_of,
            ),
        )

    async def clear(self, *, tenant_id: str) -> None:
        await self._storage._run(
            lambda: self._storage._conn().execute(
                "DELETE FROM semantic_memories WHERE tenant_id=?", (tenant_id,)
            )
        )
