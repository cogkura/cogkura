"""Instrument scoped project-update recalls; timings are diagnostic only."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from assess_project_updates import run
from retrieval_diagnostics import stage_diagnostics

from cogkura import Memory


async def profile(backend: str, projects: int) -> dict[str, Any]:
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
        result = await run(backend, projects)
    return {
        "workload": result,
        "stages": stages,
        "note": "Instrumented overlapping stages; not benchmark timings.",
        "profile_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "diagnostics_source_sha256": hashlib.sha256(
            Path(__file__).with_name("retrieval_diagnostics.py").read_bytes()
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("memory", "sqlite"), required=True)
    parser.add_argument("--projects", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.projects < 1:
        parser.error("projects must be positive")
    args.output.write_text(
        json.dumps(asyncio.run(profile(args.backend, args.projects)), indent=2, sort_keys=True)
        + "\n"
    )


if __name__ == "__main__":
    main()
