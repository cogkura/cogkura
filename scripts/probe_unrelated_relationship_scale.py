"""Measure a shared-entity relationship increase on a fixture copy.

One unrelated observation revision adds edges to an existing shared entity.
Observation count remains fixed; evidence and relationship changes are coupled.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

from measure_agent_scale import measure
from run_software_project_workload import _T, _TENANT, _observation

from cogkura import EmbeddedMemory


async def probe(path: Path, edges: int, samples: int, subject_index: int) -> dict[str, object]:
    if edges < 1 or subject_index < 1:
        raise ValueError("Need at least one edge on an unrelated project")
    with path.open("rb") as source:
        input_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    subject = f"project-{subject_index:03d}"
    async with await EmbeddedMemory.open(path) as memory:
        before = len(await memory.list_entity_relationships(tenant_id=_TENANT))
        observation = _observation(
            subject_index,
            "engineering-note-000",
            "Controlled unrelated shared-service integration relationships.",
            15,
            {
                "relationships": [
                    {
                        "source_entity_id": f"{subject}-probe-client-{number:04d}",
                        "relation_type": "related_to",
                        "target_entity_id": "memory-store",
                        "provenance": "relationship-scaling-probe",
                    }
                    for number in range(edges)
                ]
            },
        )
        await memory.observe(
            replace(
                observation, source_version="2", observed_at=_T + timedelta(days=15, minutes=52)
            )
        )
        after = len(await memory.list_entity_relationships(tenant_id=_TENANT))
        if after - before != edges:
            raise RuntimeError(f"Expected {edges} added relationships, got {after - before}")
    result = await measure(path, samples=samples)
    result.update(
        {
            "input_database_sha256": input_sha256,
            "unrelated_subject": subject,
            "added_relationships": after - before,
            "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--edges", type=int, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--subject-index", type=int, default=49)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(probe(args.database, args.edges, args.samples, args.subject_index))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "edges": result["added_relationships"],
                "median_seconds": result["median_seconds"],
                "evidence": {k: v["found"] for k, v in result["evidence"].items()},
            }
        )
    )


if __name__ == "__main__":
    main()
