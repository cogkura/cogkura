"""Pytest configuration."""

from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "postgres: integration tests requiring PostgreSQL")


@pytest.fixture(scope="session")
def postgres_memory_url() -> str | None:
    return os.environ.get("COGKURA_POSTGRES_MEMORY_URL")


@pytest.fixture(scope="session")
def postgres_source_url() -> str | None:
    return os.environ.get("COGKURA_POSTGRES_SOURCE_URL")
