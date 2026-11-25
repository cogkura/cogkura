"""Build a reproducible SQLite project workload for Agent-scale investigation.

This is fixture construction, not a timing run. Close the database before
copying it to independent measurement processes. See 0.20.3 scaling findings.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import timedelta
from pathlib import Path

from assess_project_updates import exercise
from run_software_project_workload import _T, _TENANT, _observation

from cogkura import (
    ActivationConfig,
    ComplementaryLearningSemanticConsolidator,
    EmbeddedMemory,
    SemanticCorrectionConfig,
    WorkingMemoryConfig,
)


async def build(path: Path, projects: int, observations: int) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    if projects < 1 or observations < 12:
        raise ValueError("Need at least one project and 12 observations per project")
    started = time.perf_counter()
    async with await EmbeddedMemory.open(
        path,
        activation_config=ActivationConfig(retrieval_threshold=-10),
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
        semantic_correction_config=SemanticCorrectionConfig(enabled=True),
        working_memory_config=WorkingMemoryConfig(candidate_pool_size=50),
    ) as memory:
        await exercise(memory, projects)
        for index in range(projects):
            subject = f"project-{index:03d}"
            # The established fixture has nine records per subject plus three
            # successive configuration events for project-000.
            existing = 12 if index == 0 else 9
            for number in range(observations - existing):
                await memory.observe(
                    _observation(
                        index,
                        f"engineering-note-{number:03d}",
                        f"Engineering note {number:03d}: routine documentation review "
                        f"for component {number % 17:02d}.",
                        14,
                        {"source_ref": f"{subject}/notes/{number:03d}.md"},
                    )
                )
            await memory.process(
                tenant_id=_TENANT,
                subject_id=subject,
                as_of=_T + timedelta(days=15, minutes=index),
            )
            print(
                json.dumps({"subject": subject, "seconds": time.perf_counter() - started}),
                flush=True,
            )
    print(
        json.dumps(
            {
                "database": str(path.resolve()),
                "projects": projects,
                "observations_per_project": observations,
                "fixture_seconds": time.perf_counter() - started,
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--projects", type=int, default=50)
    parser.add_argument("--observations-per-project", type=int, default=250)
    args = parser.parse_args()
    asyncio.run(build(args.database, args.projects, args.observations_per_project))


if __name__ == "__main__":
    main()
