"""Compare SQLite logical state with an APFS clone made before timed recalls.

The fixture must be quiescent while the main database and its WAL are cloned.
The clone is not opened or read until the timed recalls have completed.
"""

from __future__ import annotations

import ctypes
import os
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from typing import Any


def clone_sqlite_state(database: Path, snapshot: Path) -> None:
    """Clone database and WAL without reading their contents in the benchmark."""
    if sys.platform != "darwin":
        raise RuntimeError("The metadata-only SQLite snapshot requires macOS clonefile")
    clonefile = ctypes.CDLL(None, use_errno=True).clonefile
    clonefile.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int)
    clonefile.restype = ctypes.c_int
    for source, target in (
        (database, snapshot),
        (Path(f"{database}-wal"), Path(f"{snapshot}-wal")),
    ):
        if source != database and not source.exists():
            continue
        if clonefile(os.fsencode(source), os.fsencode(target), 0) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(source))


def _logical_state(path: Path) -> dict[str, Any]:
    with closing(sqlite3.connect(path)) as connection:
        if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            raise RuntimeError(f"SQLite integrity check failed: {path}")
        schema = connection.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()
        tables = [row[1] for row in schema if row[0] == "table"]
        state: dict[str, Any] = {"schema": schema}
        for table in tables:
            quoted = '"' + table.replace('"', '""') + '"'
            rows = connection.execute(f"SELECT * FROM {quoted}").fetchall()
            state[table] = sorted(rows, key=repr)
        return state


def compare_sqlite_state(database: Path, snapshot: Path) -> tuple[list[str], dict[str, int]]:
    """Report changed logical contents and the number of rows checked per table."""
    before = _logical_state(snapshot)
    after = _logical_state(database)
    changed = sorted(
        key for key in before.keys() | after.keys() if before.get(key) != after.get(key)
    )
    counts = {key: len(rows) for key, rows in before.items() if key != "schema"}
    return changed, counts
