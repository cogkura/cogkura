"""Add controlled unrelated semantic claims to a copy of the project fixture.

The observation count stays fixed. Each variant changes one unrelated source
revision and its derived evidence/history as well as semantic inventory, so it
is a diagnostic of a controlled workload change, not perfect causal isolation.
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

from cogkura import (
    ActivationConfig,
    ComplementaryLearningSemanticConsolidator,
    EmbeddedMemory,
    SemanticCorrectionConfig,
)


async def probe(path: Path, facts: int, samples: int, subject_index: int) -> dict[str, object]:
    if facts < 1 or subject_index < 1:
        raise ValueError("Need at least one fact on an unrelated project")
    with path.open("rb") as source:
        input_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    subject = f"project-{subject_index:03d}"
    async with await EmbeddedMemory.open(
        path,
        activation_config=ActivationConfig(retrieval_threshold=-10),
        semantic_consolidator=ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
        semantic_correction_config=SemanticCorrectionConfig(enabled=True),
    ) as memory:
        before = len(await memory.list_semantic_memories(tenant_id=_TENANT, subject_id=subject))
        observation = _observation(
            subject_index,
            "engineering-note-000",
            "Engineering note 000: controlled unrelated configuration inventory.",
            15,
            {
                "source_ref": f"{subject}/notes/000.md",
                "semantic_facts": [
                    {
                        "predicate": f"inventory_attribute_{number:04d}",
                        "object_value": f"setting_{number:04d}",
                        "authority": {"basis": "application_asserted"},
                    }
                    for number in range(facts)
                ],
            },
        )
        await memory.observe(
            replace(
                observation,
                source_version="2",
                observed_at=_T + timedelta(days=15, minutes=52),
            )
        )
        await memory.process(
            tenant_id=_TENANT,
            subject_id=subject,
            as_of=_T + timedelta(days=15, minutes=55),
        )
        after = len(await memory.list_semantic_memories(tenant_id=_TENANT, subject_id=subject))
        if after - before != facts:
            raise RuntimeError(f"Expected {facts} added semantic memories, got {after - before}")
    result = await measure(path, samples=samples)
    result["unrelated_subject"] = subject
    result["added_semantic_memories"] = after - before
    result["input_database_sha256"] = input_sha256
    result["probe_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--facts", type=int, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--subject-index", type=int, default=49)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(probe(args.database, args.facts, args.samples, args.subject_index))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "added": result["added_semantic_memories"],
                "median_seconds": result["median_seconds"],
                "evidence": {k: v["found"] for k, v in result["evidence"].items()},
            }
        )
    )


if __name__ == "__main__":
    main()
