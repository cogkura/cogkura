"""Profile recalls in the unchanged 0.19.6 mixed workload.

Run in a fresh process: uv run python scripts/profile_retrieval.py --observations 100
Profiler timings are diagnostic, not uninstrumented latency measurements.
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import pstats
import sqlite3
import subprocess
import tempfile
import time
import tracemalloc
from collections.abc import Mapping
from contextlib import ExitStack, nullcontext
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

from memory_retrieval_state import ForkedState
from retrieval_diagnostics import stage_diagnostics
from run_sqlite_workload import _run
from sqlite_retrieval_state import clone_sqlite_state, compare_sqlite_state

import cogkura
from cogkura import (
    ActivationConfig,
    ComplementaryLearningSemanticConsolidator,
    EmbeddedMemory,
    Memory,
    MemoryIdentity,
    MemoryKind,
    ObservationRevisionIdentity,
    SemanticCorrectionConfig,
)


async def fixture_state(memory: Memory, revisions: int) -> dict[str, Any]:
    """Read the same derived and revision state collected by the mixed fixture."""
    tenant = "benchmark"
    episodes = await memory.list_episodes(tenant_id=tenant)
    semantics = await memory.list_semantic_memories(tenant_id=tenant)
    semantic_revisions = await memory.list_semantic_revisions(tenant_id=tenant)
    observations = await memory._observation_store.list(tenant_id=tenant)
    references = [
        ObservationRevisionIdentity(item.id, number)
        for item in observations
        for number in range(1, revisions + 1)
    ]
    historical = await memory._observation_store.get_revisions_many(
        tenant_id=tenant, references=references
    )
    return {
        "counts": {
            "observations": len(observations),
            "observation_revisions": len(historical),
            "episodes": len(episodes),
            "semantic_memories": len(semantics),
            "semantic_revisions": len(semantic_revisions),
        },
        "semantic_values": sorted(
            (item.subject_id, item.predicate, item.object_value, item.status.value)
            for item in semantics
        ),
        "historical_values": sorted(
            (value.source_record_id, value.identity.observation_revision, value.source_version)
            for value in historical.values()
        ),
    }


async def cognitive_state(memory: Memory, revisions: int) -> dict[str, Any]:
    """Capture current fixture state for an untimed before/after audit.

    Context-specific learning outside the fixture's global context and storage
    history tables are not covered by the store APIs used here.
    """
    tenant = "benchmark"
    episodes = await memory.list_episodes(tenant_id=tenant)
    semantics = await memory.list_semantic_memories(tenant_id=tenant)
    observations = await memory._observation_store.list(tenant_id=tenant)
    identities = [
        *(MemoryIdentity(MemoryKind.EPISODE, item.memory_key) for item in episodes),
        *(MemoryIdentity(MemoryKind.SEMANTIC, item.memory_key) for item in semantics),
    ]
    references = [
        ObservationRevisionIdentity(item.id, number)
        for item in observations
        for number in range(1, revisions + 1)
    ]
    through = datetime(2100, 1, 1, tzinfo=UTC)
    return {
        "episodes": episodes,
        "semantics": semantics,
        "semantic_revisions": await memory.list_semantic_revisions(tenant_id=tenant),
        "observations": observations,
        "observation_revisions": await memory._observation_store.get_revisions_many(
            tenant_id=tenant, references=references
        ),
        "relationships": await memory.list_entity_relationships(tenant_id=tenant),
        "activation": await memory._activation_store.list_reference_traces(
            tenant_id=tenant, identities=identities, before_or_at=through
        ),
        "dynamics": await memory._dynamics_store.get_many(tenant_id=tenant, identities=identities),
        "learning": await memory._learning_store.list_states(
            tenant_id=tenant, identities=identities, context_keys=("global",)
        ),
        "associations": await memory._learning_store.list_associations(
            tenant_id=tenant, identities=identities
        ),
        "learning_reinforcement": await memory._learning_store.list_reinforcement_traces(
            tenant_id=tenant, identities=identities, before_or_at=through
        ),
        "inhibition": await memory._inhibition_store.list_for_memories(
            tenant_id=tenant, identities=identities, before_or_at=through
        ),
    }


def freeze_state(value: Any) -> Any:
    """Detach nested store values, including read-only mapping proxies."""
    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__qualname__,
            tuple((f.name, freeze_state(getattr(value, f.name))) for f in fields(value)),
        )
    if isinstance(value, Mapping):
        return tuple(
            sorted(((freeze_state(k), freeze_state(v)) for k, v in value.items()), key=repr)
        )
    if isinstance(value, (tuple, list)):
        return tuple(freeze_state(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((freeze_state(item) for item in value), key=repr))
    return value


def profile_rows(profile: cProfile.Profile) -> list[dict[str, Any]]:
    stats = pstats.Stats(profile)
    rows = []
    for (filename, line, name), (
        primitive,
        calls,
        exclusive,
        inclusive,
        _callers,
    ) in stats.stats.items():
        rows.append(
            {
                "function": f"{Path(filename).name}:{line}:{name}",
                "primitive_calls": primitive,
                "calls": calls,
                "exclusive_seconds": exclusive,
                "inclusive_seconds": inclusive,
            }
        )
    return sorted(rows, key=lambda row: row["exclusive_seconds"], reverse=True)


async def run(
    count: int,
    revisions: int,
    recalls: int,
    backend: str = "memory",
    *,
    profile: bool = True,
    diagnostic: str | None = None,
    postgres_url: str | None = None,
    context: bool = False,
    cognitive_state_audit: bool = False,
    sqlite_state_clone: bool = False,
    memory_state_fork: bool = False,
) -> dict[str, Any]:
    if sqlite_state_clone and (backend != "sqlite" or context or cognitive_state_audit):
        raise ValueError("SQLite clone state check requires SQLite recall-only mode")
    if memory_state_fork and (backend != "memory" or context or cognitive_state_audit):
        raise ValueError("Memory fork state check requires in-memory recall-only mode")
    profiler_path = Path(__file__)
    diagnostics_path = profiler_path.with_name("retrieval_diagnostics.py")
    activation_path = profiler_path.resolve().parents[1] / "src/cogkura/algorithms/activation.py"
    root = profiler_path.resolve().parents[1]

    def source_identity() -> tuple[str, str]:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        patch_bytes = subprocess.check_output(
            ["git", "-C", str(root), "diff", "--binary", "HEAD", "--", "src/cogkura"]
        )
        untracked = subprocess.check_output(
            ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard", "src/cogkura"]
        )
        if untracked:
            raise RuntimeError("Untracked Core source cannot be identified by the Git diff")
        return commit, hashlib.sha256(patch_bytes).hexdigest()

    identity = source_identity()
    try:
        load_before = os.getloadavg()
    except (AttributeError, OSError):
        load_before = None
    workload_path = profiler_path.with_name("run_sqlite_workload.py")
    clone_path_source = profiler_path.with_name("sqlite_retrieval_state.py")
    fork_path_source = profiler_path.with_name("memory_retrieval_state.py")
    measured_paths = (
        profiler_path,
        diagnostics_path,
        activation_path,
        workload_path,
        clone_path_source,
        fork_path_source,
    )
    source_hashes = tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in measured_paths)
    profiles: list[dict[str, Any]] = []
    context_profiles: list[dict[str, Any]] = []
    before_cognitive_state: dict[str, Any] | None = None
    clone_table_counts: dict[str, int] | None = None
    forked_state: ForkedState | None = None

    async def capture_before(memory: Memory) -> None:
        nonlocal before_cognitive_state
        before_cognitive_state = {
            key: freeze_state(value)
            for key, value in (await cognitive_state(memory, revisions)).items()
        }

    original = Memory.recall
    original_context = Memory.prepare_context

    async def measured_context(self: Memory, *args: Any, **kwargs: Any) -> Any:
        wall_start, cpu_start = time.perf_counter(), time.process_time()
        with stage_diagnostics(self) if diagnostic == "stages" else nullcontext(None) as stages:
            result = await original_context(self, *args, **kwargs)
        context_profiles.append(
            {
                "wall_seconds": time.perf_counter() - wall_start,
                "process_cpu_seconds": time.process_time() - cpu_start,
                "stages": stages,
            }
        )
        return result

    async def measured(self: Memory, *args: Any, **kwargs: Any) -> Any:
        profiler = cProfile.Profile() if profile else None
        statements: list[str] = []
        if diagnostic == "sql":
            if owner is None:
                raise ValueError("SQL tracing requires SQLite")
            await owner._storage._run(
                lambda: owner._storage._conn().set_trace_callback(statements.append)
            )
        before = None
        object_count = None
        if diagnostic == "allocations":
            gc.collect()
            object_count = len(gc.get_objects())
            tracemalloc.start()
            before = tracemalloc.take_snapshot()
            tracemalloc.reset_peak()
        wall_start, cpu_start = time.perf_counter(), time.process_time()
        if profiler is not None:
            profiler.enable()
        try:
            with stage_diagnostics(self) if diagnostic == "stages" else nullcontext(None) as stages:
                return await original(self, *args, **kwargs)
        finally:
            if profiler is not None:
                profiler.disable()
            wall, cpu = time.perf_counter() - wall_start, time.process_time() - cpu_start
            if diagnostic == "sql" and owner is not None:
                await owner._storage._run(lambda: owner._storage._conn().set_trace_callback(None))
            allocation = None
            if before is not None:
                current, peak = tracemalloc.get_traced_memory()
                after = tracemalloc.take_snapshot()
                changes = after.compare_to(before, "lineno")
                allocation = {
                    "current_traced_bytes": current,
                    "peak_traced_bytes": peak,
                    "net_traced_bytes": sum(item.size_diff for item in changes),
                    "net_traced_blocks": sum(item.count_diff for item in changes),
                    "gc_tracked_objects_before": object_count,
                    "gc_tracked_objects_after": len(gc.get_objects()),
                    "largest_net_allocations": [
                        {
                            "location": str(item.traceback),
                            "bytes": item.size_diff,
                            "blocks": item.count_diff,
                        }
                        for item in sorted(changes, key=lambda item: item.size_diff, reverse=True)[
                            :20
                        ]
                    ],
                }
                tracemalloc.stop()
            profiles.append(
                {
                    "wall_seconds": wall,
                    "process_cpu_seconds": cpu,
                    "functions": profile_rows(profiler) if profiler is not None else None,
                    "stages": stages if diagnostic == "stages" else None,
                    "allocation": allocation,
                    "sql_statement_count": len(statements) if diagnostic == "sql" else None,
                    "sql_select_count": sum(
                        s.lstrip().upper().startswith("SELECT") for s in statements
                    )
                    if diagnostic == "sql"
                    else None,
                }
            )

    options = {
        "activation_config": ActivationConfig(retrieval_threshold=-10),
        "semantic_consolidator": ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
        "semantic_correction_config": SemanticCorrectionConfig(enabled=True),
    }
    with tempfile.TemporaryDirectory(prefix="cogkura-retrieval-profile-") as directory:
        database_path = Path(directory) / "memory.db"
        clone_path = Path(directory) / "before-recall.db"

        async def capture_clone(_memory: Memory) -> None:
            clone_sqlite_state(database_path, clone_path)

        async def capture_fork(snapshot_memory: Memory) -> None:
            nonlocal forked_state
            forked_state = ForkedState(snapshot_memory, Path(directory) / "before-memory.json")

        owner = None
        engine = None
        if backend == "sqlite":
            owner = await EmbeddedMemory.open(database_path, **options)
            memory = owner.memory
        elif backend == "postgres":
            if postgres_url is None:
                raise ValueError("PostgreSQL requires a disposable database URL")
            from sqlalchemy.ext.asyncio import create_async_engine

            from cogkura.migrations import apply_migrations
            from cogkura.storage.postgres import (
                PostgresActivationStore,
                PostgresCheckpointStore,
                PostgresEntityRelationshipStore,
                PostgresEpisodeStore,
                PostgresInhibitionStore,
                PostgresLearningStore,
                PostgresMemoryDynamicsStore,
                PostgresObservationStore,
                PostgresSemanticMemoryStore,
            )

            engine = create_async_engine(postgres_url)
            await apply_migrations(engine)
            semantic = PostgresSemanticMemoryStore(engine)
            memory = Memory(
                observation_store=PostgresObservationStore(engine),
                checkpoint_store=PostgresCheckpointStore(engine),
                episode_store=PostgresEpisodeStore(engine),
                semantic_store=semantic,
                semantic_inspection_store=semantic,
                activation_store=PostgresActivationStore(engine),
                dynamics_store=PostgresMemoryDynamicsStore(engine),
                learning_store=PostgresLearningStore(engine),
                inhibition_store=PostgresInhibitionStore(engine),
                entity_relationship_store=PostgresEntityRelationshipStore(engine),
                **options,
            )
            await memory.clear(tenant_id="benchmark")
        else:
            memory = Memory(**options)
        try:
            with ExitStack() as stack:
                stack.enter_context(patch.object(Memory, "recall", measured))
                if context:
                    stack.enter_context(patch.object(Memory, "prepare_context", measured_context))
                result = await _run(
                    memory,
                    count,
                    revisions,
                    recalls,
                    mixed=True,
                    trace_sql=False,
                    extended=context,
                    before_recalls=(
                        capture_before
                        if cognitive_state_audit
                        else capture_clone
                        if sqlite_state_clone
                        else capture_fork
                        if memory_state_fork
                        else None
                    ),
                )
            after_state = await fixture_state(memory, revisions)
            if any(after_state[key] != result[key] for key in after_state):
                raise RuntimeError("Recorded fixture state changed during the workload")
            if sqlite_state_clone:
                changed_tables, clone_table_counts = compare_sqlite_state(database_path, clone_path)
                if changed_tables:
                    raise RuntimeError(f"SQLite logical state changed: {changed_tables}")
            if memory_state_fork:
                if forked_state is None:
                    raise RuntimeError("Memory snapshot was not created")
                changed_stores = forked_state.finish(memory)
                if changed_stores:
                    raise RuntimeError(f"In-memory state changed: {changed_stores}")
            if cognitive_state_audit:
                after_cognitive_state = {
                    key: freeze_state(value)
                    for key, value in (await cognitive_state(memory, revisions)).items()
                }
                if before_cognitive_state != after_cognitive_state:
                    changed = [
                        key
                        for key in after_cognitive_state
                        if before_cognitive_state is None
                        or before_cognitive_state[key] != after_cognitive_state[key]
                    ]
                    raise RuntimeError(f"Cognitive state changed during recall: {changed}")
        finally:
            if forked_state is not None:
                forked_state.abort()
            if owner is not None:
                await owner.close()
            if engine is not None:
                try:
                    await memory.clear(tenant_id="benchmark")
                finally:
                    await engine.dispose()
    if source_hashes != tuple(
        hashlib.sha256(path.read_bytes()).hexdigest() for path in measured_paths
    ):
        raise RuntimeError("Measured source changed during the run; discard this observation")
    if identity != source_identity():
        raise RuntimeError("Core Git source changed during the run; discard this observation")
    try:
        import resource

        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_unit = "bytes" if platform.system() == "Darwin" else "KiB on Linux; OS-dependent"
    except ImportError:
        peak_rss, rss_unit = None, None
    direct_url = importlib.metadata.distribution("cogkura").read_text("direct_url.json")
    return {
        "method": ("cProfile" if profile else diagnostic or "uninstrumented")
        + "; per-recall timing, diagnostic modes include instrumentation overhead",
        "environment": {
            "pid": os.getpid(),
            "host_load_average_before_fixture": load_before,
            "commit": identity[0],
            "core_source_diff_sha256": identity[1],
            "python": platform.python_version(),
            "core_runtime_version": cogkura.__version__,
            "core_distribution_version": importlib.metadata.version("cogkura"),
            "core_module": str(Path(cogkura.__file__).resolve()),
            "core_direct_url": json.loads(direct_url) if direct_url else None,
            "platform": platform.platform(),
            "sqlite": sqlite3.sqlite_version,
            "profiler_source_sha256": source_hashes[0],
            "diagnostics_source_sha256": source_hashes[1],
            "activation_source_sha256": source_hashes[2],
            "workload_source_sha256": source_hashes[3],
            "sqlite_state_clone_source_sha256": source_hashes[4],
            "memory_state_fork_source_sha256": source_hashes[5],
        },
        "workload": {
            "observations": count,
            "revisions": revisions,
            "recalls": recalls,
            "backend": backend,
            "profiled": profile,
            "diagnostic": diagnostic,
            "context": context,
            "cognitive_state_audit": cognitive_state_audit,
            "sqlite_state_clone": sqlite_state_clone,
            "memory_state_fork": memory_state_fork,
        },
        "counts": result["counts"],
        "process_peak_rss": {"value": peak_rss, "unit": rss_unit, "includes_fixture": True},
        "mixed_operations": result["mixed_operations"],
        "semantic_values": result["semantic_values"],
        "historical_values": result["historical_values"],
        "recorded_fixture_state_after_matches": True,
        "cognitive_state_audit_matches": True if cognitive_state_audit else None,
        "sqlite_state_clone_matches": True if sqlite_state_clone else None,
        "sqlite_state_clone_table_counts": clone_table_counts,
        "memory_state_fork_matches": True if memory_state_fork else None,
        "recall_outputs": result["recall_outputs"],
        "profiles": profiles,
        "context_profiles": context_profiles,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=int, default=25)
    parser.add_argument("--revisions", type=int, default=5)
    parser.add_argument("--recalls", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--context", action="store_true", help="also time context preparation")
    parser.add_argument(
        "--cognitive-state-audit",
        action="store_true",
        help="compare cognitive state before and after; timings are not benchmark results",
    )
    parser.add_argument(
        "--sqlite-state-clone",
        action="store_true",
        help="experimentally compare all SQLite logical tables using a pre-recall APFS clone",
    )
    parser.add_argument(
        "--memory-state-fork",
        action="store_true",
        help="experimentally compare in-memory stores using a pre-recall fork",
    )
    parser.add_argument("--backend", choices=("memory", "sqlite", "postgres"), default="memory")
    parser.add_argument(
        "--postgres-url", help="disposable PostgreSQL database; tenant benchmark is cleared"
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--uninstrumented", action="store_true")
    modes.add_argument("--stage-diagnostics", action="store_true")
    modes.add_argument("--allocations", action="store_true")
    modes.add_argument("--sql-trace", action="store_true")
    args = parser.parse_args()
    if min(args.observations, args.revisions, args.recalls) < 1:
        parser.error("workload arguments must be positive")
    if args.sql_trace and args.backend != "sqlite":
        parser.error("--sql-trace requires --backend sqlite")
    if args.backend == "postgres" and not args.postgres_url:
        parser.error("--backend postgres requires --postgres-url for a disposable database")
    if args.postgres_url and args.backend != "postgres":
        parser.error("--postgres-url requires --backend postgres")
    if args.cognitive_state_audit and args.context:
        parser.error("--cognitive-state-audit requires recall-only mode")
    if args.sqlite_state_clone and (
        args.backend != "sqlite" or args.context or args.cognitive_state_audit
    ):
        parser.error("--sqlite-state-clone requires SQLite recall-only mode without audit")
    if args.memory_state_fork and (
        args.backend != "memory" or args.context or args.cognitive_state_audit
    ):
        parser.error("--memory-state-fork requires in-memory recall-only mode without audit")
    diagnostic = (
        "stages"
        if args.stage_diagnostics
        else "allocations"
        if args.allocations
        else "sql"
        if args.sql_trace
        else None
    )
    report = asyncio.run(
        run(
            args.observations,
            args.revisions,
            args.recalls,
            args.backend,
            profile=not (args.uninstrumented or diagnostic),
            diagnostic=diagnostic,
            postgres_url=args.postgres_url,
            context=args.context,
            cognitive_state_audit=args.cognitive_state_audit,
            sqlite_state_clone=args.sqlite_state_clone,
            memory_state_fork=args.memory_state_fork,
        )
    )
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
