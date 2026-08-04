"""Source connector protocol."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol


class SourceConnector(Protocol):
    """Reads incremental batches from a customer data source."""

    connector_id: str

    def records(
        self,
        checkpoint: dict[str, Any] | None,
    ) -> AsyncIterator[Any]:
        """Yield source records after the given checkpoint."""

    def checkpoint_for(self, record: Any) -> dict[str, Any]:
        """Return the checkpoint value for a successfully processed record."""
