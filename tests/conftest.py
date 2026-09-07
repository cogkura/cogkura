"""Pytest configuration."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "postgres: integration tests requiring PostgreSQL")


@pytest.fixture(scope="session")
def postgres_memory_url() -> str | None:
    return os.environ.get("COGKURA_POSTGRES_MEMORY_URL")


@pytest.fixture(scope="session")
def postgres_source_url() -> str | None:
    return os.environ.get("COGKURA_POSTGRES_SOURCE_URL")
