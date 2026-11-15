"""Print the SQLite version, local temporary filesystem and runtime durability settings."""

from __future__ import annotations

import asyncio
import ctypes
import json
import os
import platform
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

from cogkura import EmbeddedMemory


def _filesystem_type(path: str) -> str:
    if sys.platform == "win32":
        root = os.path.splitdrive(os.path.abspath(path))[0] + "\\"
        buffer = ctypes.create_unicode_buffer(256)
        from ctypes import wintypes

        get_info = ctypes.windll.kernel32.GetVolumeInformationW  # type: ignore[attr-defined]
        get_info.argtypes = [
            wintypes.LPCWSTR,
            wintypes.LPWSTR,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(wintypes.DWORD),
            wintypes.LPWSTR,
            wintypes.DWORD,
        ]
        get_info.restype = wintypes.BOOL
        if not get_info(root, None, 0, None, None, None, buffer, len(buffer)):
            return "unknown"
        return buffer.value
    try:
        if sys.platform == "darwin":
            disk = (
                subprocess.check_output(["df", "-P", path], text=True).splitlines()[-1].split()[0]
            )
            mounts = subprocess.check_output(["mount"], text=True).splitlines()
            for line in mounts:
                if line.startswith(disk + " on ") and " (" in line:
                    return line.rsplit(" (", 1)[1].split(",", 1)[0].rstrip(")")
            return "unknown"
        return subprocess.check_output(
            ["stat", "-f", "-c", "%T", path], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError, IndexError):
        return "unknown"


async def _settings(path: Path) -> dict[str, object]:
    owner = await EmbeddedMemory.open(path)
    try:

        def read() -> dict[str, object]:
            conn = owner._storage._conn()
            return {
                name: conn.execute(f"PRAGMA {name}").fetchone()[0]
                for name in ("journal_mode", "synchronous", "foreign_keys", "fullfsync")
            }

        return await owner._storage._run(read)
    finally:
        await owner.close()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="cogkura-ci-") as directory:
        with sqlite3.connect(":memory:") as conn:
            source_id = conn.execute("SELECT sqlite_source_id()").fetchone()[0]
        filesystem = _filesystem_type(directory)
        if filesystem == "unknown":
            raise RuntimeError(
                "Could not identify the temporary filesystem for portability evidence"
            )
        report = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "sqlite": sqlite3.sqlite_version,
            "sqlite_source_id": source_id,
            "filesystem": filesystem,
            "settings": asyncio.run(_settings(Path(directory) / "memory.db")),
        }
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
