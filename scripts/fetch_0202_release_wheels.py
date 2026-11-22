"""Fetch the two verified prior-release wheels for the PostgreSQL CI gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

_WHEELS = {
    "0.19.5": "29ec696acba22b9752948d8f394551066ad1038989e0a30244162a6ed7460fe0",
    "0.19.6": "862d1a08b3d939bc1eea81d883c03315a30509c6779e12abec6f5fb85d8252d4",
}


def fetch_wheels(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for version, expected in _WHEELS.items():
        filename = f"cogkura-{version}-py3-none-any.whl"
        with urlopen(f"https://pypi.org/pypi/cogkura/{version}/json", timeout=30) as response:
            metadata = json.load(response)
        matches = [item for item in metadata["urls"] if item["filename"] == filename]
        if len(matches) != 1 or matches[0]["digests"]["sha256"] != expected:
            raise RuntimeError(f"Unexpected PyPI metadata for {filename}")
        with urlopen(matches[0]["url"], timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError(f"Downloaded wheel hash differs for {filename}")
        (directory / filename).write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fetch_wheels(args.output)


if __name__ == "__main__":
    main()
