"""Instrument the existing scoped project workload on an empty PostgreSQL database.

Pass a disposable database URL through COGKURA_POSTGRES_MEMORY_URL. The script
adds workload data and applies Core migrations; it never clears existing data.
Instrumentation overlaps and timings are diagnostic, not acceptance samples.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any
from unittest.mock import patch

from assess_project_updates import exercise
from retrieval_diagnostics import stage_diagnostics

from cogkura import (
    ActivationConfig,
    ComplementaryLearningSemanticConsolidator,
    Memory,
    SemanticCorrectionConfig,
)
from cogkura.migrations import apply_migrations


async def profile(url: str, projects: int) -> dict[str, object]:
    from sqlalchemy.ext.asyncio import create_async_engine

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

    engine = create_async_engine(url)
    try:
        await apply_migrations(engine)
        observations = PostgresObservationStore(engine)
        if await observations.list(tenant_id="project-workload"):
            raise RuntimeError("project-workload tenant is not empty; use a disposable database")
        semantic = PostgresSemanticMemoryStore(engine)
        memory = Memory(
            observation_store=observations,
            checkpoint_store=PostgresCheckpointStore(engine),
            episode_store=PostgresEpisodeStore(engine),
            semantic_store=semantic,
            semantic_inspection_store=semantic,
            activation_store=PostgresActivationStore(engine),
            dynamics_store=PostgresMemoryDynamicsStore(engine),
            learning_store=PostgresLearningStore(engine),
            inhibition_store=PostgresInhibitionStore(engine),
            entity_relationship_store=PostgresEntityRelationshipStore(engine),
            activation_config=ActivationConfig(retrieval_threshold=-10),
            semantic_consolidator=ComplementaryLearningSemanticConsolidator(
                allow_authoritative_single_episode=True
            ),
            semantic_correction_config=SemanticCorrectionConfig(enabled=True),
        )
        original = Memory.recall
        stages = []

        async def observed(self: Memory, query: Any, *args: Any, **kwargs: Any) -> Any:
            if query != "worker configuration":
                return await original(self, query, *args, **kwargs)
            with stage_diagnostics(self) as current:
                result = await original(self, query, *args, **kwargs)
            stages.append(current)
            return result

        with patch.object(Memory, "recall", observed):
            workload = await exercise(memory, projects)
        return {
            "backend": "postgresql",
            "projects": projects,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pid": os.getpid(),
            "workload": workload,
            "stages": stages,
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "note": "Instrumented overlapping stages; not benchmark timings.",
        }
    finally:
        await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projects", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    url = os.environ.get("COGKURA_POSTGRES_MEMORY_URL")
    if not url or args.projects < 1:
        parser.error("a database URL and positive project count are required")
    result = asyncio.run(profile(url, args.projects))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
