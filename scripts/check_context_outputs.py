"""Check mixed-fixture context outputs on memory and SQLite without timing."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import platform
import sys
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from test_0202_context_outputs import _outputs  # noqa: E402

import cogkura  # noqa: E402
from cogkura import (  # noqa: E402
    ActivationConfig,
    ComplementaryLearningSemanticConsolidator,
    EmbeddedMemory,
    LearningConfig,
    Memory,
    SemanticCorrectionConfig,
    WorkingMemoryConfig,
)


def _json_default(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    raise TypeError(f"Cannot serialize {type(value).__qualname__}")


async def check(observations: int, learning: bool) -> dict[str, Any]:
    options = {
        "activation_config": ActivationConfig(retrieval_threshold=-10),
        "semantic_consolidator": ComplementaryLearningSemanticConsolidator(
            allow_authoritative_single_episode=True
        ),
        "semantic_correction_config": SemanticCorrectionConfig(enabled=True),
        "working_memory_config": WorkingMemoryConfig(candidate_pool_size=20),
        "learning_config": LearningConfig(enabled=learning),
    }
    expected = await _outputs(Memory(**options), observations=observations)
    with tempfile.TemporaryDirectory(prefix="cogkura-context-output-check-") as directory:
        async with await EmbeddedMemory.open(Path(directory) / "context.db", **options) as memory:
            observed = await _outputs(memory, observations=observations)
    if observed != expected:
        raise RuntimeError("Memory and SQLite context outputs differ")
    direct_url = importlib.metadata.distribution("cogkura").read_text("direct_url.json")
    sources = (
        Path(__file__),
        ROOT / "tests/test_0202_context_outputs.py",
        ROOT / "scripts/run_sqlite_workload.py",
    )
    return {
        "observations": observations,
        "learning": learning,
        "python": platform.python_version(),
        "module": str(Path(cogkura.__file__).resolve()),
        "direct_url": json.loads(direct_url) if direct_url else None,
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "outputs": expected,
        "backends_match": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=int, required=True)
    parser.add_argument("--learning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.observations < 7:
        parser.error("at least seven observations are required")
    args.output.write_text(
        json.dumps(
            asyncio.run(check(args.observations, args.learning)),
            default=_json_default,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
