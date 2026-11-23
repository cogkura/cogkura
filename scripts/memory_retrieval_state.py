"""Experimental copy-on-write snapshot of the default in-memory stores.

The single-threaded benchmark forks immediately before recall. The child does
not inspect the copied state until the parent finishes timing. This experimental
method changes the process environment and is not release timing evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from cogkura import Memory

_STORES = (
    "_observation_store",
    "_checkpoint_store",
    "_episode_store",
    "_semantic_store",
    "_activation_store",
    "_dynamics_store",
    "_learning_store",
    "_entity_relationship_store",
    "_inhibition_store",
)


def _freeze(value: Any) -> Any:
    if isinstance(value, Enum):
        return ("enum", type(value).__qualname__, _freeze(value.value))
    if value is None or isinstance(value, (str, int, float, bool, bytes)):
        return value
    if isinstance(value, (datetime, date, timedelta)):
        return (type(value).__name__, str(value))
    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__qualname__,
            tuple((f.name, _freeze(getattr(value, f.name))) for f in fields(value)),
        )
    if isinstance(value, Mapping):
        return tuple(sorted(((_freeze(k), _freeze(v)) for k, v in value.items()), key=repr))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    raise TypeError(f"Unsupported in-memory state value: {type(value).__qualname__}")


def state_hashes(memory: Memory) -> dict[str, str]:
    hashes = {}
    for name in _STORES:
        store = getattr(memory, name)
        if not type(store).__module__.startswith("cogkura.storage.in_memory_"):
            raise TypeError(f"Unexpected store for {name}: {type(store).__qualname__}")
        data = dict(vars(store))
        if "_history_order" in data:
            data["_history_order"] = vars(data["_history_order"])
        hashes[name] = hashlib.sha256(repr(_freeze(data)).encode()).hexdigest()
    return hashes


class ForkedState:
    def __init__(self, memory: Memory, path: Path) -> None:
        if sys.platform != "darwin" or not hasattr(os, "fork"):
            raise RuntimeError("The experimental memory snapshot requires macOS fork")
        if threading.active_count() != 1:
            raise RuntimeError("The experimental memory snapshot requires one Python thread")
        read_fd, write_fd = os.pipe()
        try:
            pid = os.fork()
        except BaseException:
            os.close(read_fd)
            os.close(write_fd)
            raise
        if pid == 0:
            os.close(write_fd)
            try:
                signal = os.read(read_fd, 1)
                os.close(read_fd)
                if signal != b"c":
                    os._exit(2)
                path.write_text(json.dumps(state_hashes(memory), sort_keys=True))
                os._exit(0)
            except BaseException:
                os._exit(1)
        os.close(read_fd)
        self._write_fd = write_fd
        self._pid = pid
        self._path = path

    def finish(self, memory: Memory) -> list[str]:
        os.write(self._write_fd, b"c")
        os.close(self._write_fd)
        self._write_fd = -1
        _, status = os.waitpid(self._pid, 0)
        self._pid = -1
        if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
            raise RuntimeError("Memory snapshot child failed")
        before = json.loads(self._path.read_text())
        after = state_hashes(memory)
        return sorted(
            key for key in before.keys() | after.keys() if before.get(key) != after.get(key)
        )

    def abort(self) -> None:
        if self._write_fd >= 0:
            os.close(self._write_fd)
            self._write_fd = -1
        if self._pid >= 0:
            os.waitpid(self._pid, 0)
            self._pid = -1
