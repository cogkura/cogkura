"""SQLite implementations of the persistent cognitive-state protocols."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any, TypeVar

from cogkura.models import (
    ActivationReferenceKind,
    ActivationReferenceTrace,
    BehavioralStructuralAnchor,
    CompetitionDirection,
    InhibitionScopeSignature,
    InhibitoryTrace,
    LearningOutcome,
    LearningPlan,
    LearningWriteResult,
    MemoryIdentity,
    MemoryKind,
    MemoryReference,
    MemoryRetentionState,
    ReferenceCompactionResult,
    StoredEntityRelationship,
    StoredMemoryAssociation,
    StoredMemoryDynamics,
    StoredMemoryLearningState,
)
from cogkura.storage.in_memory_activation import InMemoryActivationStore
from cogkura.storage.in_memory_inhibition import InMemoryInhibitionStore
from cogkura.storage.sqlite import SQLiteStorage, _datetime, _time

_T = TypeVar("_T")


def _dt(value: str) -> datetime:
    result = _datetime(value)
    assert result is not None
    return result


def _id(kind: str, key: str) -> MemoryIdentity:
    return MemoryIdentity(memory_kind=MemoryKind(kind), memory_key=key)


class _Store:
    def __init__(self, storage: SQLiteStorage) -> None:
        self._storage = storage

    async def _transaction(
        self, operation: Callable[[sqlite3.Connection], _T], *, write: bool = False
    ) -> _T:
        def work() -> _T:
            conn = self._storage._conn()
            conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            try:
                result = operation(conn)
                conn.commit()
                return result
            except BaseException:
                conn.rollback()
                raise

        return await self._storage._run(work)


class SQLiteActivationStore(_Store):
    """Store access references and weighted compaction traces."""

    async def append_references(self, references: Sequence[MemoryReference]) -> None:
        def work(conn: sqlite3.Connection) -> None:
            for ref in references:
                conn.execute(
                    """INSERT OR IGNORE INTO activation_references
                    (tenant_id,memory_kind,memory_key,reference_kind,referenced_at,request_id,weight,metadata)
                    VALUES (?,?,?,?,?,?,?,?)""",
                    (
                        ref.tenant_id,
                        ref.memory_kind.value,
                        ref.memory_key,
                        ref.reference_kind.value,
                        _time(ref.referenced_at),
                        ref.request_id,
                        ref.weight,
                        json.dumps(dict(ref.metadata)),
                    ),
                )

        await self._transaction(work, write=True)

    async def list_reference_traces(
        self, *, tenant_id: str, identities: Sequence[MemoryIdentity], before_or_at: datetime
    ) -> Mapping[MemoryIdentity, tuple[ActivationReferenceTrace, ...]]:
        def work(
            conn: sqlite3.Connection,
        ) -> Mapping[MemoryIdentity, tuple[ActivationReferenceTrace, ...]]:
            wanted = set(identities)
            result: dict[MemoryIdentity, list[ActivationReferenceTrace]] = {}
            for row in conn.execute(
                "SELECT * FROM activation_references WHERE tenant_id=? AND referenced_at<=? "
                "ORDER BY referenced_at,id",
                (tenant_id, _time(before_or_at)),
            ):
                identity = _id(row["memory_kind"], row["memory_key"])
                if identity in wanted:
                    result.setdefault(identity, []).append(
                        ActivationReferenceTrace(
                            referenced_at=_dt(row["referenced_at"]), weight=row["weight"]
                        )
                    )
            return {key: tuple(value) for key, value in result.items()}

        return await self._transaction(work)

    async def compact_references(
        self, *, tenant_id: str, before: datetime, bucket_seconds: float
    ) -> ReferenceCompactionResult:
        def work(conn: sqlite3.Connection) -> ReferenceCompactionResult:
            store = InMemoryActivationStore()
            for row in conn.execute(
                "SELECT * FROM activation_references WHERE tenant_id=? ORDER BY id", (tenant_id,)
            ):
                store._references.append(
                    MemoryReference(
                        tenant_id=tenant_id,
                        memory_kind=MemoryKind(row["memory_kind"]),
                        memory_key=row["memory_key"],
                        reference_kind=ActivationReferenceKind(row["reference_kind"]),
                        referenced_at=_dt(row["referenced_at"]),
                        request_id=row["request_id"],
                        weight=row["weight"],
                        metadata=json.loads(row["metadata"]),
                    )
                )
            result = asyncio.run(
                store.compact_references(
                    tenant_id=tenant_id, before=before, bucket_seconds=bucket_seconds
                )
            )
            if result.references_compacted:
                conn.execute("DELETE FROM activation_references WHERE tenant_id=?", (tenant_id,))
                for ref in store._references:
                    conn.execute(
                        """INSERT INTO activation_references (tenant_id,memory_kind,memory_key,
                        reference_kind,referenced_at,request_id,weight,metadata)
                        VALUES (?,?,?,?,?,?,?,?)""",
                        (
                            tenant_id,
                            ref.memory_kind.value,
                            ref.memory_key,
                            ref.reference_kind.value,
                            _time(ref.referenced_at),
                            ref.request_id,
                            ref.weight,
                            json.dumps(dict(ref.metadata)),
                        ),
                    )
            return result

        return await self._transaction(work, write=True)

    async def clear(self, *, tenant_id: str) -> None:
        await self._transaction(
            lambda conn: conn.execute(
                "DELETE FROM activation_references WHERE tenant_id=?", (tenant_id,)
            ),
            write=True,
        )


class SQLiteMemoryDynamicsStore(_Store):
    """Store lifecycle state and reactivation."""

    async def get_many(
        self, *, tenant_id: str, identities: Sequence[MemoryIdentity]
    ) -> Mapping[MemoryIdentity, StoredMemoryDynamics]:
        def work(conn: sqlite3.Connection) -> dict[MemoryIdentity, StoredMemoryDynamics]:
            wanted = set(identities)
            result = {}
            for row in conn.execute(
                "SELECT * FROM memory_dynamics WHERE tenant_id=?", (tenant_id,)
            ):
                identity = _id(row["memory_kind"], row["memory_key"])
                if identity in wanted:
                    result[identity] = StoredMemoryDynamics(
                        tenant_id=tenant_id,
                        memory_kind=identity.memory_kind,
                        memory_key=identity.memory_key,
                        retention_state=MemoryRetentionState(row["retention_state"]),
                        last_base_level=row["last_base_level"],
                        last_retention_score=row["last_retention_score"],
                        below_threshold_since=_datetime(row["below_threshold_since"]),
                        forgotten_at=_datetime(row["forgotten_at"]),
                        evaluated_at=_dt(row["evaluated_at"]),
                        updated_at=_dt(row["updated_at"]),
                    )
            return result

        return await self._transaction(work)

    async def upsert_many(self, dynamics: Sequence[StoredMemoryDynamics]) -> None:
        def work(conn: sqlite3.Connection) -> None:
            for item in dynamics:
                conn.execute(
                    """INSERT INTO memory_dynamics VALUES (?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT (tenant_id,memory_kind,memory_key) DO UPDATE SET
                    retention_state=excluded.retention_state,last_base_level=excluded.last_base_level,
                    last_retention_score=excluded.last_retention_score,below_threshold_since=excluded.below_threshold_since,
                    forgotten_at=excluded.forgotten_at,evaluated_at=excluded.evaluated_at,updated_at=excluded.updated_at""",
                    (
                        item.tenant_id,
                        item.memory_kind.value,
                        item.memory_key,
                        item.retention_state.value,
                        item.last_base_level,
                        item.last_retention_score,
                        _time(item.below_threshold_since),
                        _time(item.forgotten_at),
                        _time(item.evaluated_at),
                        _time(item.updated_at),
                    ),
                )

        await self._transaction(work, write=True)

    async def reactivate(
        self, *, tenant_id: str, identities: Sequence[MemoryIdentity], at: datetime
    ) -> None:
        def work(conn: sqlite3.Connection) -> None:
            conn.executemany(
                """UPDATE memory_dynamics SET retention_state='active',below_threshold_since=NULL,
                forgotten_at=NULL,evaluated_at=?,updated_at=?
                WHERE tenant_id=? AND memory_kind=? AND memory_key=?""",
                [
                    (_time(at), _time(at), tenant_id, item.memory_kind.value, item.memory_key)
                    for item in identities
                ],
            )

        await self._transaction(work, write=True)

    async def clear(self, *, tenant_id: str) -> None:
        await self._transaction(
            lambda conn: conn.execute(
                "DELETE FROM memory_dynamics WHERE tenant_id=?", (tenant_id,)
            ),
            write=True,
        )


class SQLiteEntityRelationshipStore(_Store):
    """Store directed entity relationships."""

    async def upsert_many(self, relationships: Sequence[StoredEntityRelationship]) -> None:
        def work(conn: sqlite3.Connection) -> None:
            for item in relationships:
                conn.execute(
                    """INSERT INTO entity_relationships VALUES (?,?,?,?,?,?,?,?,?)
                    ON CONFLICT (tenant_id,relationship_id) DO UPDATE SET
                    source_entity_id=excluded.source_entity_id,
                    relation_type=excluded.relation_type,target_entity_id=excluded.target_entity_id,
                    provenance=excluded.provenance,source_namespace=excluded.source_namespace,
                    source_record_id=excluded.source_record_id,created_at=excluded.created_at""",
                    (
                        item.tenant_id,
                        item.relationship_id,
                        item.source_entity_id,
                        item.relation_type,
                        item.target_entity_id,
                        item.provenance,
                        item.source_namespace,
                        item.source_record_id,
                        _time(item.created_at),
                    ),
                )

        await self._transaction(work, write=True)

    async def list(
        self, *, tenant_id: str, entity_id: str | None = None
    ) -> Sequence[StoredEntityRelationship]:
        def work(conn: sqlite3.Connection) -> list[StoredEntityRelationship]:
            rows = conn.execute(
                "SELECT * FROM entity_relationships WHERE tenant_id=?", (tenant_id,)
            )
            result = [
                StoredEntityRelationship(
                    relationship_id=r["relationship_id"],
                    tenant_id=tenant_id,
                    source_entity_id=r["source_entity_id"],
                    relation_type=r["relation_type"],
                    target_entity_id=r["target_entity_id"],
                    provenance=r["provenance"],
                    source_namespace=r["source_namespace"],
                    source_record_id=r["source_record_id"],
                    created_at=_dt(r["created_at"]),
                )
                for r in rows
                if entity_id is None or entity_id in (r["source_entity_id"], r["target_entity_id"])
            ]
            return sorted(
                result,
                key=lambda item: (
                    item.source_entity_id,
                    item.relation_type.casefold(),
                    item.target_entity_id,
                    item.relationship_id,
                ),
            )

        return await self._transaction(work)

    async def clear(self, *, tenant_id: str) -> None:
        await self._transaction(
            lambda conn: conn.execute(
                "DELETE FROM entity_relationships WHERE tenant_id=?", (tenant_id,)
            ),
            write=True,
        )


def _trace(row: sqlite3.Row) -> InhibitoryTrace:
    payload = json.loads(row["scope_json"])
    scope = InhibitionScopeSignature(
        structural_anchor=BehavioralStructuralAnchor(payload["structural_anchor"]),
        subject_id=payload.get("subject_id"),
        entity_ids=tuple(payload.get("entity_ids") or ()),
        predicate=payload.get("predicate"),
        semantic_slot_key=payload.get("semantic_slot_key"),
        feature_ids=tuple(payload.get("feature_ids") or ()),
    )
    return InhibitoryTrace(
        id=row["id"],
        tenant_id=row["tenant_id"],
        inhibited_identity=_id(row["inhibited_kind"], row["inhibited_key"]),
        selected_identity=_id(row["selected_kind"], row["selected_key"]),
        direction=CompetitionDirection(row["direction"]),
        scope=scope,
        competition_strength=row["competition_strength"],
        competitor_accessibility=row["competitor_accessibility"],
        induction_pressure=row["induction_pressure"],
        retrieval_evaluated_at=_dt(row["retrieval_evaluated_at"]),
        induced_at=_dt(row["induced_at"]),
        request_id=row["request_id"],
    )


class SQLiteInhibitionStore(_Store):
    """Store scoped inhibitory traces."""

    async def append_traces(self, traces: Sequence[InhibitoryTrace]) -> None:
        def work(conn: sqlite3.Connection) -> None:
            for item in traces:
                conn.execute(
                    "INSERT OR IGNORE INTO inhibition_traces VALUES "
                    "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        item.id,
                        item.tenant_id,
                        item.inhibited_identity.memory_kind.value,
                        item.inhibited_identity.memory_key,
                        item.selected_identity.memory_kind.value,
                        item.selected_identity.memory_key,
                        item.direction.value,
                        item.scope.scope_key,
                        json.dumps(item.scope.to_canonical_dict()),
                        item.competition_strength,
                        item.competitor_accessibility,
                        item.induction_pressure,
                        _time(item.retrieval_evaluated_at),
                        _time(item.induced_at),
                        item.request_id,
                    ),
                )

        await self._transaction(work, write=True)

    async def list_for_memory(
        self, *, tenant_id: str, identity: MemoryIdentity, as_of: datetime | None = None
    ) -> Sequence[InhibitoryTrace]:
        def work(conn: sqlite3.Connection) -> Sequence[InhibitoryTrace]:
            sql = (
                "SELECT * FROM inhibition_traces WHERE tenant_id=? "
                "AND inhibited_kind=? AND inhibited_key=?"
            )
            params: list[Any] = [tenant_id, identity.memory_kind.value, identity.memory_key]
            if as_of is not None:
                sql += " AND induced_at<=?"
                params.append(_time(as_of))
            sql += " ORDER BY induced_at,scope_key,id"
            return tuple(_trace(row) for row in conn.execute(sql, params))

        return await self._transaction(work)

    async def list_for_memories(
        self,
        *,
        tenant_id: str,
        identities: Sequence[MemoryIdentity],
        after: datetime | None = None,
        before_or_at: datetime,
        limit_per_memory: int | None = None,
    ) -> Mapping[MemoryIdentity, tuple[InhibitoryTrace, ...]]:
        def work(conn: sqlite3.Connection) -> Mapping[MemoryIdentity, tuple[InhibitoryTrace, ...]]:
            store = InMemoryInhibitionStore()
            for row in conn.execute(
                "SELECT * FROM inhibition_traces WHERE tenant_id=? AND induced_at<=?",
                (tenant_id, _time(before_or_at)),
            ):
                store._traces.append(_trace(row))
            return asyncio.run(
                store.list_for_memories(
                    tenant_id=tenant_id,
                    identities=identities,
                    after=after,
                    before_or_at=before_or_at,
                    limit_per_memory=limit_per_memory,
                )
            )

        return await self._transaction(work)

    async def clear(self, *, tenant_id: str) -> None:
        await self._transaction(
            lambda conn: conn.execute(
                "DELETE FROM inhibition_traces WHERE tenant_id=?", (tenant_id,)
            ),
            write=True,
        )


class SQLiteLearningStore(_Store):
    """Atomically persist feedback, utility counts, associations and reinforcement."""

    async def apply(self, plan: LearningPlan) -> LearningWriteResult:
        from cogkura.exceptions import StorageError

        def work(conn: sqlite3.Connection) -> LearningWriteResult:
            row = conn.execute(
                "SELECT feedback_fingerprint FROM learning_events "
                "WHERE tenant_id=? AND feedback_id=?",
                (plan.tenant_id, plan.feedback_id),
            ).fetchone()
            if row is not None:
                if row[0] != plan.feedback_fingerprint:
                    raise StorageError(
                        f"Conflicting feedback fingerprint for feedback_id {plan.feedback_id!r}."
                    )
                return LearningWriteResult(
                    created=False,
                    unchanged=True,
                    helpful=0,
                    unhelpful=0,
                    incorrect=0,
                    associations_reinforced=0,
                )
            timestamp = _time(plan.occurred_at)
            conn.execute(
                "INSERT INTO learning_events VALUES (?,?,?,?,?,?,?)",
                (
                    plan.tenant_id,
                    plan.feedback_id,
                    plan.feedback_fingerprint,
                    plan.subject_id,
                    plan.context_key,
                    timestamp,
                    json.dumps(dict(plan.metadata)),
                ),
            )
            counts = {outcome: 0 for outcome in LearningOutcome}
            for item in plan.items:
                counts[item.outcome] += 1
                ident = item.identity
                conn.execute(
                    "INSERT INTO learning_feedback VALUES (?,?,?,?,?,?,?)",
                    (
                        plan.tenant_id,
                        plan.feedback_id,
                        ident.memory_kind.value,
                        ident.memory_key,
                        item.revision_key,
                        item.outcome.value,
                        json.dumps(dict(item.metadata)),
                    ),
                )
                conn.execute(
                    """INSERT INTO learning_state VALUES (?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT (tenant_id,context_key,memory_kind,memory_key) DO UPDATE SET
                    helpful_count=learning_state.helpful_count+excluded.helpful_count,
                    unhelpful_count=learning_state.unhelpful_count+excluded.unhelpful_count,
                    incorrect_count=learning_state.incorrect_count+excluded.incorrect_count,
                    last_feedback_at=excluded.last_feedback_at,updated_at=excluded.updated_at""",
                    (
                        plan.tenant_id,
                        plan.context_key,
                        ident.memory_kind.value,
                        ident.memory_key,
                        int(item.outcome is LearningOutcome.HELPFUL),
                        int(item.outcome is LearningOutcome.UNHELPFUL),
                        int(item.outcome is LearningOutcome.INCORRECT),
                        timestamp,
                        timestamp,
                        timestamp,
                    ),
                )
            for left, right in plan.association_pairs:
                conn.execute(
                    """INSERT INTO learned_associations VALUES (?,?,?,?,?,?,?,?,?)
                    ON CONFLICT (tenant_id,left_kind,left_key,right_kind,right_key) DO UPDATE SET
                    coactivation_count=learned_associations.coactivation_count+1,
                    last_reinforced_at=excluded.last_reinforced_at,updated_at=excluded.updated_at""",
                    (
                        plan.tenant_id,
                        left.memory_kind.value,
                        left.memory_key,
                        right.memory_kind.value,
                        right.memory_key,
                        1,
                        timestamp,
                        timestamp,
                        timestamp,
                    ),
                )
            return LearningWriteResult(
                created=True,
                unchanged=False,
                helpful=counts[LearningOutcome.HELPFUL],
                unhelpful=counts[LearningOutcome.UNHELPFUL],
                incorrect=counts[LearningOutcome.INCORRECT],
                associations_reinforced=len(plan.association_pairs),
            )

        return await self._transaction(work, write=True)

    async def list_states(
        self, *, tenant_id: str, identities: Sequence[MemoryIdentity], context_keys: Sequence[str]
    ) -> Sequence[StoredMemoryLearningState]:
        def work(conn: sqlite3.Connection) -> Sequence[StoredMemoryLearningState]:
            wanted, contexts = set(identities), set(context_keys)
            result = []
            for row in conn.execute("SELECT * FROM learning_state WHERE tenant_id=?", (tenant_id,)):
                identity = _id(row["memory_kind"], row["memory_key"])
                if identity in wanted and row["context_key"] in contexts:
                    result.append(
                        StoredMemoryLearningState(
                            tenant_id=tenant_id,
                            context_key=row["context_key"],
                            memory_kind=identity.memory_kind,
                            memory_key=identity.memory_key,
                            helpful_count=row["helpful_count"],
                            unhelpful_count=row["unhelpful_count"],
                            incorrect_count=row["incorrect_count"],
                            first_feedback_at=_dt(row["first_feedback_at"]),
                            last_feedback_at=_dt(row["last_feedback_at"]),
                            updated_at=_dt(row["updated_at"]),
                        )
                    )
            return tuple(result)

        return await self._transaction(work)

    async def list_associations(
        self, *, tenant_id: str, identities: Sequence[MemoryIdentity]
    ) -> Sequence[StoredMemoryAssociation]:
        def work(conn: sqlite3.Connection) -> Sequence[StoredMemoryAssociation]:
            wanted = set(identities)
            result = []
            for row in conn.execute(
                "SELECT * FROM learned_associations WHERE tenant_id=? "
                "ORDER BY left_kind,left_key,right_kind,right_key",
                (tenant_id,),
            ):
                left, right = (
                    _id(row["left_kind"], row["left_key"]),
                    _id(row["right_kind"], row["right_key"]),
                )
                if left in wanted and right in wanted:
                    result.append(
                        StoredMemoryAssociation(
                            tenant_id=tenant_id,
                            left=left,
                            right=right,
                            coactivation_count=row["coactivation_count"],
                            first_reinforced_at=_dt(row["first_reinforced_at"]),
                            last_reinforced_at=_dt(row["last_reinforced_at"]),
                            updated_at=_dt(row["updated_at"]),
                        )
                    )
            return tuple(result)

        return await self._transaction(work)

    async def list_reinforcement_traces(
        self, *, tenant_id: str, identities: Sequence[MemoryIdentity], before_or_at: datetime
    ) -> Mapping[MemoryIdentity, tuple[ActivationReferenceTrace, ...]]:
        def work(
            conn: sqlite3.Connection,
        ) -> Mapping[MemoryIdentity, tuple[ActivationReferenceTrace, ...]]:
            wanted = set(identities)
            grouped: dict[MemoryIdentity, list[ActivationReferenceTrace]] = {}
            rows = conn.execute(
                """SELECT f.memory_kind,f.memory_key,e.occurred_at FROM learning_feedback f
                JOIN learning_events e ON f.tenant_id=e.tenant_id AND f.feedback_id=e.feedback_id
                WHERE f.tenant_id=? AND f.outcome='helpful' AND e.occurred_at<=?
                ORDER BY e.occurred_at""",
                (tenant_id, _time(before_or_at)),
            )
            for row in rows:
                identity = _id(row["memory_kind"], row["memory_key"])
                if identity in wanted:
                    grouped.setdefault(identity, []).append(
                        ActivationReferenceTrace(referenced_at=_dt(row["occurred_at"]), weight=1)
                    )
            return {key: tuple(value) for key, value in grouped.items()}

        return await self._transaction(work)

    async def clear(self, *, tenant_id: str) -> None:
        def work(conn: sqlite3.Connection) -> None:
            for table in ("learning_events", "learning_state", "learned_associations"):
                conn.execute(f"DELETE FROM {table} WHERE tenant_id=?", (tenant_id,))

        await self._transaction(work, write=True)
