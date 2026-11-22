"""Repeat the existing project workload while explicitly scoping named queries.

The underlying fixture stays unchanged. This is an exploratory diagnostic,
not a frozen benchmark or an assertion that Agent always has a subject ID.
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


async def measure(backend: str) -> dict[str, Any]:
    original = Memory.recall

    async def scoped(self: Memory, query: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(query, str) and query.startswith("project-000 "):
            kwargs["subject_id"] = "project-000"
        return await original(self, query, *args, **kwargs)

    with patch.object(Memory, "recall", scoped):
        result = await run(backend, 10, heterogeneous=True, relationships=5)
    result["diagnostic"] = {
        "subject_id": "project-000",
        "scope": "queries beginning project-000",
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
