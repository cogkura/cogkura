"""Smoke-test an installed wheel in fresh subprocesses; run from a dependency-free venv."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import importlib.resources
import importlib.util
import json
import sqlite3
import subprocess
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path


def check_distribution(expected_version: str) -> dict[str, str]:
    import cogkura

    distribution = importlib.metadata.distribution("cogkura")
    module = Path(cogkura.__file__).resolve()
    root = Path(distribution.locate_file("")).resolve()
    if root not in module.parents:
        raise RuntimeError(f"Core module is not loaded from the installed wheel: {module}")
    if cogkura.__version__ != expected_version or distribution.version != expected_version:
        raise RuntimeError(
            "Core runtime and distribution version do not match the expected version"
        )
    if importlib.util.find_spec("sqlalchemy") is not None:
        raise RuntimeError("The dependency-free smoke environment unexpectedly contains SQLAlchemy")
    migrations = importlib.resources.files("cogkura").joinpath("migrations", "sqlite")
    expected = {
        "001_initial.sql",
        "002_episodes.sql",
        "003_semantic.sql",
        "004_cognitive_state.sql",
    }
    if not expected.issubset({p.name for p in migrations.iterdir()}):
        raise RuntimeError("The wheel does not contain all expected SQLite migrations")
    return {
        "module": str(module),
        "version": distribution.version,
        "sqlite": sqlite3.sqlite_version,
    }


async def run_phase(path: Path, write: bool) -> list[list[object]]:
    from cogkura import ActivationConfig, EmbeddedMemory, ObservationInput

    now = datetime(2026, 1, 1, tzinfo=UTC)
    async with await EmbeddedMemory.open(
        path, activation_config=ActivationConfig(retrieval_threshold=-10)
    ) as memory:
        if write:
            await memory.observe(
                ObservationInput(
                    tenant_id="artifact-check",
                    source_namespace="wheel",
                    source_record_id="one",
                    content="The deployment service uses a database.",
                    observed_at=now,
                )
            )
            await memory.process(tenant_id="artifact-check", as_of=now)
        results = await memory.recall(
            "deployment service database", tenant_id="artifact-check", as_of=now + timedelta(days=1)
        )
        return [
            [r.memory_kind.value, r.memory.memory_key, r.memory.statement, r.score] for r in results
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--phase", choices=("write", "read"))
    parser.add_argument("--database", type=Path)
    args = parser.parse_args()
    identity = check_distribution(args.expected_version)
    if args.phase:
        if args.database is None:
            parser.error("--database is required for a worker phase")
        print(json.dumps(asyncio.run(run_phase(args.database, args.phase == "write"))))
        return
    with tempfile.TemporaryDirectory(prefix="cogkura-wheel-") as directory:
        database = str(Path(directory) / "memory.db")
        results = []
        for phase in ("write", "read"):
            output = subprocess.check_output(
                [
                    sys.executable,
                    "-I",
                    __file__,
                    "--expected-version",
                    args.expected_version,
                    "--phase",
                    phase,
                    "--database",
                    database,
                ],
                text=True,
            )
            results.append(json.loads(output))
        if not results[0] or results[0] != results[1]:
            raise RuntimeError("Wheel restart did not preserve retrieval identity and score")
    print(json.dumps({"identity": identity, "restart_matches": True}, indent=2))


if __name__ == "__main__":
    main()
