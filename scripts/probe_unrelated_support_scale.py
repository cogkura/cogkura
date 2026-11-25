"""Measure increased supporting evidence in an unrelated project on a fixture copy.

Existing observations are revised rather than added, keeping subject counts
fixed. The resulting observation, episode and semantic histories also change.
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

from cogkura import ActivationConfig, EmbeddedMemory


async def probe(path: Path, supports: int, samples: int, subject_index: int) -> dict[str, object]:
    if supports < 2 or supports > 200 or subject_index < 1:
        raise ValueError("Need 2–200 supports on an unrelated project")
    with path.open("rb") as source:
        input_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    subject = f"project-{subject_index:03d}"
    async with await EmbeddedMemory.open(
        path, activation_config=ActivationConfig(retrieval_threshold=-10)
    ) as memory:
        for number in range(supports):
            observation = _observation(
                subject_index,
                f"engineering-note-{number:03d}",
                "The shared build system uses the configured release channel.",
                15,
                {
                    "semantic_facts": [
                        {
                            "subject_entity_id": subject,
                            "predicate": "unrelated_build_channel",
                            "object_value": "release_channel",
                        }
                    ]
                },
            )
            await memory.observe(
                replace(
                    observation,
                    source_version="2",
                    observed_at=_T + timedelta(days=15, minutes=50, seconds=number),
                )
            )
        await memory.process(
            tenant_id=_TENANT, subject_id=subject, as_of=_T + timedelta(days=15, minutes=55)
        )
        current = await memory.list_semantic_memories(tenant_id=_TENANT, subject_id=subject)
        claims = [item for item in current if item.predicate == "unrelated_build_channel"]
        if len(claims) != 1 or claims[0].support_count != supports:
            raise RuntimeError("Expected one semantic claim with all requested supports")
        evidence_count = len(claims[0].observation_evidence)
    result = await measure(path, samples=samples)
    result.update(
        {
            "input_database_sha256": input_sha256,
            "unrelated_subject": subject,
            "support_count": supports,
            "observation_evidence_count": evidence_count,
            "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--supports", type=int, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--subject-index", type=int, default=49)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(probe(args.database, args.supports, args.samples, args.subject_index))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "supports": result["support_count"],
                "median_seconds": result["median_seconds"],
                "evidence": {k: v["found"] for k, v in result["evidence"].items()},
            }
        )
    )


if __name__ == "__main__":
    main()
