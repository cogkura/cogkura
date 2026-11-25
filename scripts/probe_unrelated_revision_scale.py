"""Add corrections and a withdrawal in an unrelated project's existing notes.

Run on a copy of the Agent-scale fixture. Each variant keeps observation count
fixed but also changes source and snapshot history, so the measurement is a
controlled workload diagnostic rather than perfectly isolated causality.
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


async def probe(
    path: Path, corrections: int, samples: int, subject_index: int
) -> dict[str, object]:
    if corrections < 1 or subject_index < 1:
        raise ValueError("Need corrections on an unrelated project")
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
        predecessor = None
        applied = 0
        for number in range(corrections + 2):
            at = _T + timedelta(days=15, minutes=50, seconds=number)
            common = {
                "subject_entity_id": subject,
                "predicate": "unrelated_revision_probe",
                "object_value": f"setting_{number:04d}",
                "cardinality": "one",
                "authority": {"basis": "application_asserted"},
            }
            if number == corrections + 1:
                metadata = {
                    "semantic_withdrawals": [
                        {
                            "target_revision_key": predecessor,
                            "effective_at": at.isoformat(),
                            "authority": {"basis": "application_asserted"},
                        }
                    ]
                }
            else:
                if predecessor is not None:
                    common["correction"] = {
                        "replaces_revision_key": predecessor,
                        "effective_at": at.isoformat(),
                    }
                metadata = {"semantic_facts": [common]}
            observation = _observation(
                subject_index,
                f"engineering-note-{number:03d}",
                f"Controlled unrelated revision {number}.",
                15,
                metadata,
            )
            await memory.observe(replace(observation, source_version="2", observed_at=at))
            processed = await memory.process(tenant_id=_TENANT, subject_id=subject, as_of=at)
            if 0 < number <= corrections:
                applied += processed.semantics.authoritative_corrections_applied
            if number <= corrections:
                current = await memory.list_semantic_memories(tenant_id=_TENANT, subject_id=subject)
                matching = [
                    item
                    for item in current
                    if item.predicate == "unrelated_revision_probe"
                    and item.object_value == f"setting_{number:04d}"
                ]
                if len(matching) != 1:
                    raise RuntimeError(f"Missing expected claim at update {number}")
                predecessor = matching[0].revision_key
            elif processed.semantics.authoritative_withdrawals_applied != 1:
                raise RuntimeError("Expected the final withdrawal to apply")
        if applied != corrections:
            raise RuntimeError(f"Expected {corrections} corrections, got {applied}")
    result = await measure(path, samples=samples)
    result.update(
        {
            "input_database_sha256": input_sha256,
            "unrelated_subject": subject,
            "corrections_applied": applied,
            "withdrawals_applied": 1,
            "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--corrections", type=int, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--subject-index", type=int, default=49)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(probe(args.database, args.corrections, args.samples, args.subject_index))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "corrections": result["corrections_applied"],
                "median_seconds": result["median_seconds"],
                "evidence": {k: v["found"] for k, v in result["evidence"].items()},
            }
        )
    )


if __name__ == "__main__":
    main()
