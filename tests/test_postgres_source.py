"""Unit tests for PostgresTableSource configuration."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from cogkura.exceptions import ValidationError
from cogkura.sources.postgres import PostgresTableSource


def _source(**kwargs: Any) -> PostgresTableSource:
    return PostgresTableSource(
        connector_id="messages",
        engine=MagicMock(),
        table="public.messages",
        **kwargs,
    )


def test_soft_delete_column_appended_to_explicit_columns() -> None:
    source = _source(
        columns=("id", "body", "updated_at"),
        soft_delete_column="deleted_at",
    )
    assert source.selected_columns == ["id", "body", "updated_at", "deleted_at"]
    assert source.soft_delete_column == "deleted_at"


def test_soft_delete_column_not_duplicated() -> None:
    source = _source(
        columns=("id", "body", "updated_at", "deleted_at"),
        soft_delete_column="deleted_at",
    )
    assert source.selected_columns.count("deleted_at") == 1


def test_cursor_columns_appended_when_missing() -> None:
    source = _source(columns=("body",), soft_delete_column="deleted_at")
    assert "updated_at" in source.selected_columns
    assert "id" in source.selected_columns
    assert "deleted_at" in source.selected_columns


def test_rejects_unsafe_identifiers() -> None:
    with pytest.raises(ValidationError, match="soft_delete_column"):
        _source(soft_delete_column="deleted_at; drop table")
