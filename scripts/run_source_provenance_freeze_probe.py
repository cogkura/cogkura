#!/usr/bin/env python3
"""Repeatable observational 0.18 source probe; correctness belongs to pytest.

Use PYTHONPATH to select an archived baseline or candidate source tree. PostgreSQL
requires a disposable COGKURA_POSTGRES_MEMORY_URL and --postgres. No timing SLA.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import statistics
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from cogkura import (
    ActivationConfig,
    ComplementaryLearningSemanticConsolidator,
    Memory,
    ObservationInput,
    SemanticCorrectionConfig,
    SemanticSlotQuery,
    SourceInspectionLimits,
    __version__,
)
from cogkura.storage.in_memory_observation import InMemoryObservationStore

CLOCK = datetime(2026, 11, 1, tzinfo=UTC)


def observation(tenant: str, record: str, *, when: datetime = CLOCK) -> ObservationInput:
    return ObservationInput(
        tenant_id=tenant,
        subject_id="service",
        source_namespace="docs",
        source_record_id=record,
        observed_at=when,
        content=f"Deployment policy is startup ({record}).",
        metadata={
            "conversation_id": record,
            "source_ref": f"unfetchable://{record}",
            "payload": "x" * 1024,
            "semantic_facts": [
                {
                    "subject_entity_id": "service",
                    "predicate": "deployment_policy",
                    "object_value": "startup",
                    "cardinality": "one",
                }
            ],
        },
    )


async def probe(backend: str, mode: str, n: int, engine: Any) -> dict[str, Any]:
    tenant = f"freeze-cost-{backend}-{mode}-{n}"
    store: Any = InMemoryObservationStore()
    kwargs: dict[str, Any] = {}
    if engine is not None:
        from cogkura.storage.postgres import (
            PostgresEpisodeStore,
            PostgresObservationStore,
            PostgresSemanticMemoryStore,
        )

        store = PostgresObservationStore(engine)
        kwargs = {
            "episode_store": PostgresEpisodeStore(engine),
            "semantic_store": PostgresSemanticMemoryStore(engine),
        }
    m = Memory(
        observation_store=store,
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
        semantic_correction_config=SemanticCorrectionConfig(enabled=True),
        activation_config=ActivationConfig(retrieval_threshold=-10),
        **kwargs,
    )
    await m.clear(tenant_id=tenant)
    when = CLOCK
    if mode == "slot":
        for i in range(n):
            await m.observe(observation(tenant, f"record-{i:04d}"))
        await m.process(tenant_id=tenant, as_of=when)
    else:
        first = observation(tenant, "original")
        fact = {
            **first.metadata["semantic_facts"][0],
            "authority": {"basis": "application_asserted"},
        }
        await m.observe(replace(first, metadata={**first.metadata, "semantic_facts": [fact]}))
        await m.process(tenant_id=tenant, as_of=when)
        key = (await m.list_semantic_revisions(tenant_id=tenant))[0].revision_key
        when = CLOCK + timedelta(days=1)
        withdrawal = observation(tenant, "withdraw", when=when)
        await m.observe(
            replace(
                withdrawal,
                content="Deployment policy is withdrawn.",
                metadata={
                    "conversation_id": "withdraw",
                    "semantic_withdrawals": [
                        {
                            "target_revision_key": key,
                            "effective_at": when.isoformat(),
                            "authority": {"basis": "application_asserted"},
                        }
                    ],
                },
            )
        )
        await m.process(tenant_id=tenant, as_of=when)
        when = CLOCK + timedelta(days=2)
        for i in range(n):
            obs = observation(tenant, f"candidate-{i:04d}", when=when)
            await m.observe(
                replace(
                    obs,
                    content="Deployment policy is startup again. Ledger database is postgresql."
                    if i == 0
                    else f"Unrelated maintenance note {i}.",
                    metadata={
                        **obs.metadata,
                        "semantic_facts": obs.metadata["semantic_facts"] if i == 0 else [],
                    },
                )
            )
        await m.process(tenant_id=tenant, as_of=when)
    metrics: dict[str, int] = {}
    originals = store.get_many, store.get_revisions_many

    def count(rows: Any) -> None:
        metrics["observation_rows_returned"] += len(rows)
        for row in rows:
            metrics["metadata_bytes"] += len(
                json.dumps(dict(row.metadata), sort_keys=True).encode()
            )
            metrics["content_bytes"] += len((getattr(row, "content", None) or "").encode())

    async def current(**kw: Any) -> Any:
        metrics["observation_batches"] += 1
        rows = await originals[0](**kw)
        count(rows)
        return rows

    async def revisions(**kw: Any) -> Any:
        metrics["observation_batches"] += 1
        rows = await originals[1](**kw)
        count(rows.values())
        return rows

    store.get_many, store.get_revisions_many = current, revisions
    statements: list[str] = []

    def capture(
        conn: Any, cursor: Any, statement: str, parameters: Any, context: Any, many: bool
    ) -> None:
        statements.append(statement)

    if engine is not None:
        from sqlalchemy import event

        event.listen(engine.sync_engine, "before_cursor_execute", capture)
    runs = []
    try:
        for _ in range(5):
            metrics = dict.fromkeys(
                (
                    "observation_batches",
                    "observation_rows_returned",
                    "metadata_bytes",
                    "content_bytes",
                ),
                0,
            )
            statements.clear()
            start = time.perf_counter()
            if mode == "slot":
                report = await m.inspect_semantic_slot(
                    SemanticSlotQuery("service", "deployment_policy"),
                    tenant_id=tenant,
                    as_of=when,
                    limits=SourceInspectionLimits(max_evidence=1),
                )
                assert report.quality.referenced_count == n and report.quality.returned_count == 1
                leaked = False
            else:
                context = await m.prepare_context(
                    "current deployment policy startup ledger postgresql",
                    tenant_id=tenant,
                    as_of=when,
                )
                leaked = "startup" in context.render()
            elapsed = (time.perf_counter() - start) * 1000
            assert all(
                s.lstrip().upper().startswith(("SELECT", "SET TRANSACTION")) for s in statements
            )
            runs.append(
                {
                    **metrics,
                    "sql_statements": len(statements),
                    "latency_ms": round(elapsed, 3),
                    "withdrawn_prose_present": leaked,
                }
            )
    finally:
        if engine is not None:
            event.remove(engine.sync_engine, "before_cursor_execute", capture)
        await m.clear(tenant_id=tenant)
    return {
        "backend": backend,
        "mode": mode,
        "fixture_sources": n,
        "median_latency_ms": round(statistics.median(r["latency_ms"] for r in runs), 3),
        "runs": runs,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postgres", action="store_true")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    engine = None
    server = None
    if args.postgres:
        url = os.environ.get("COGKURA_POSTGRES_MEMORY_URL")
        if not url:
            parser.error("PostgreSQL requested but COGKURA_POSTGRES_MEMORY_URL unset")
        from sqlalchemy.ext.asyncio import create_async_engine

        from cogkura.migrations import apply_migrations

        engine = create_async_engine(url)
        await apply_migrations(engine)
        async with engine.connect() as conn:
            server = await conn.exec_driver_sql("SHOW server_version")
            server = server.scalar_one()
    result = {
        "label": args.label,
        "core_version": __version__,
        "python": platform.python_version(),
        "postgresql": server,
        "methodology": (
            "Five reads per case; source rows returned and metadata/content bytes only, "
            "excluding SQL row overhead, semantic snapshots, episode evidence and engine pages "
            "scanned. Context fixture has one later structured readmission plus unrelated "
            "candidate episodes. No latency threshold."
        ),
        "results": [],
    }
    try:
        for backend, connection in [("in_memory", None)] + (
            [("postgresql", engine)] if engine is not None else []
        ):
            for mode in ("slot", "context"):
                for n in (2, 16, 64):
                    item = await probe(backend, mode, n, connection)
                    result["results"].append(item)
                    print(backend, mode, n, item["median_latency_ms"], flush=True)
    finally:
        if engine is not None:
            await engine.dispose()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
