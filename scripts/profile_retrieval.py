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
import platform
import pstats
import sqlite3
import subprocess
import tempfile
import time
import tracemalloc
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import Any
from unittest.mock import patch

from retrieval_diagnostics import stage_diagnostics
from run_sqlite_workload import _run

import cogkura
from cogkura import (
    ActivationConfig,
    ComplementaryLearningSemanticConsolidator,
    EmbeddedMemory,
    Memory,
    SemanticCorrectionConfig,
)


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
) -> dict[str, Any]:
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
    workload_path = profiler_path.with_name("run_sqlite_workload.py")
    measured_paths = (profiler_path, diagnostics_path, activation_path, workload_path)
    source_hashes = tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in measured_paths)
    profiles: list[dict[str, Any]] = []
    context_profiles: list[dict[str, Any]] = []
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
        owner = None
        engine = None
        if backend == "sqlite":
            owner = await EmbeddedMemory.open(Path(directory) / "memory.db", **options)
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
                    memory, count, revisions, recalls, mixed=True, trace_sql=False, extended=context
                )
        finally:
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
    return {
        "method": ("cProfile" if profile else diagnostic or "uninstrumented")
        + "; per-recall timing, diagnostic modes include instrumentation overhead",
        "environment": {
            "commit": identity[0],
            "core_source_diff_sha256": identity[1],
            "python": platform.python_version(),
            "core_runtime_version": cogkura.__version__,
            "core_distribution_version": importlib.metadata.version("cogkura"),
            "platform": platform.platform(),
            "sqlite": sqlite3.sqlite_version,
            "profiler_source_sha256": source_hashes[0],
            "diagnostics_source_sha256": source_hashes[1],
            "activation_source_sha256": source_hashes[2],
            "workload_source_sha256": source_hashes[3],
        },
        "workload": {
            "observations": count,
            "revisions": revisions,
            "recalls": recalls,
            "backend": backend,
            "profiled": profile,
            "diagnostic": diagnostic,
            "context": context,
        },
        "counts": result["counts"],
        "process_peak_rss": {"value": peak_rss, "unit": rss_unit, "includes_fixture": True},
        "mixed_operations": result["mixed_operations"],
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
        )
    )
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
