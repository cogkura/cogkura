"""Locate named project evidence in full unscoped rankings.

This diagnostic changes the result limit of the existing project workload; its
timings cannot be compared directly with the ten-result baseline.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from run_software_project_workload import run

from cogkura import Memory

_EXPECTED = {
    "current_architecture": "project-000 storage engine sqlite_local_postgres_hosted",
    "historical_architecture": "project-000 storage engine postgres_primary",
    "incident": "project-000: An expired database connection pool",
    "migration": "project-000: Database migrations run in the deployment pipeline",
}


async def measure(backend: str) -> dict[str, Any]:
    original = Memory.recall

    async def deep(self: Memory, query: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs["limit"] = 1000
        return await original(self, query, *args, **kwargs)

    with patch.object(Memory, "recall", deep):
        result = await run(backend, 10, heterogeneous=True, relationships=5)
    for query in result["queries"]:
        expected = _EXPECTED.get(query["name"])
        if expected is not None:
            query["expected_evidence"] = expected
            query["expected_rank"] = next(
                (rank for rank, row in enumerate(query["ranked_outputs"], 1) if expected in row[1]),
                None,
            )
        query["returned_count"] = len(query["ranked_outputs"])
    result["diagnostic"] = {
        "subject_scope": None,
        "result_limit": 1000,
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("memory", "sqlite"), default="memory")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = json.dumps(asyncio.run(measure(args.backend)), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
