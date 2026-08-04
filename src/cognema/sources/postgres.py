"""PostgreSQL table source connector."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from cognema.exceptions import ValidationError

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")


def _validate_identifier(name: str, *, label: str) -> str:
    if not _IDENTIFIER.match(name):
        raise ValidationError(f"{label} must be a simple SQL identifier, got {name!r}.")
    return name


class PostgresTableSource:
    """Read-only incremental source over a PostgreSQL table.

    Uses a compound cursor of (cursor_timestamp_column, cursor_id_column).
    Hard deletes are not detected; use a soft-delete column when available.

    When ``soft_delete_column`` is set and an explicit column list is provided,
    that column is always included in the SELECT so mappers can set
    ``ObservationInput.is_deleted``. Deletion semantics remain mapper-owned.
    """

    def __init__(
        self,
        *,
        connector_id: str,
        engine: AsyncEngine,
        table: str,
        columns: Sequence[str] | None = None,
        cursor_columns: tuple[str, str] = ("updated_at", "id"),
        batch_size: int = 500,
        filters: Mapping[str, Any] | None = None,
        soft_delete_column: str | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValidationError("batch_size must be greater than zero.")
        if len(cursor_columns) != 2:
            raise ValidationError("cursor_columns must contain exactly two column names.")
        for part in table.split("."):
            _validate_identifier(part, label="table")
        ts_col, id_col = cursor_columns
        _validate_identifier(ts_col, label="cursor timestamp column")
        _validate_identifier(id_col, label="cursor id column")
        if soft_delete_column is not None:
            _validate_identifier(soft_delete_column, label="soft_delete_column")
        if columns is not None:
            for column in columns:
                _validate_identifier(column, label="column")
        for key in filters or {}:
            _validate_identifier(key, label="filter column")

        self.connector_id = connector_id
        self._engine = engine
        self._table = table
        self._cursor_ts_col, self._cursor_id_col = cursor_columns
        self._batch_size = batch_size
        self._filters = dict(filters or {})
        self._soft_delete_column = soft_delete_column
        self._columns = self._resolve_columns(columns)

    def _resolve_columns(self, columns: Sequence[str] | None) -> list[str]:
        if columns is None:
            return ["*"]
        resolved = list(columns)
        if self._soft_delete_column and self._soft_delete_column not in resolved:
            resolved.append(self._soft_delete_column)
        for required in (self._cursor_ts_col, self._cursor_id_col):
            if required not in resolved:
                resolved.append(required)
        return resolved

    @property
    def soft_delete_column(self) -> str | None:
        return self._soft_delete_column

    @property
    def selected_columns(self) -> list[str]:
        return list(self._columns)

    async def records(
        self,
        checkpoint: dict[str, Any] | None,
    ) -> AsyncIterator[Any]:
        current_checkpoint = checkpoint
        while True:
            batch = await self._fetch_batch(current_checkpoint)
            if not batch:
                return
            for record in batch:
                yield record
                current_checkpoint = self.checkpoint_for(record)

    def checkpoint_for(self, record: Mapping[str, Any]) -> dict[str, Any]:
        ts_value = record[self._cursor_ts_col]
        id_value = record[self._cursor_id_col]
        if isinstance(ts_value, datetime):
            ts_str = ts_value.isoformat()
        else:
            ts_str = str(ts_value)
        return {self._cursor_ts_col: ts_str, self._cursor_id_col: str(id_value)}

    async def _fetch_batch(
        self,
        checkpoint: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        column_list = ", ".join(self._columns)
        where_parts = ["TRUE"]
        params: dict[str, Any] = {"limit": self._batch_size}

        for index, (key, value) in enumerate(self._filters.items()):
            param_name = f"filter_{index}"
            where_parts.append(f"{key} = :{param_name}")
            params[param_name] = value

        if checkpoint is not None:
            ts_key = self._cursor_ts_col
            id_key = self._cursor_id_col
            ts_value = checkpoint[ts_key]
            id_value = checkpoint[id_key]
            if isinstance(ts_value, str):
                ts_value = datetime.fromisoformat(ts_value)
            params["cursor_ts"] = ts_value
            params["cursor_id"] = id_value
            where_parts.append(
                f"({self._cursor_ts_col} > :cursor_ts OR "
                f"({self._cursor_ts_col} = :cursor_ts AND {self._cursor_id_col} > :cursor_id))"
            )

        query = f"""
            SELECT {column_list}
            FROM {self._table}
            WHERE {" AND ".join(where_parts)}
            ORDER BY {self._cursor_ts_col}, {self._cursor_id_col}
            LIMIT :limit
        """
        async with self._engine.connect() as conn:
            result = await conn.execute(text(query), params)
            rows = result.mappings().all()
        return [dict(row) for row in rows]
