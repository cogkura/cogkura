"""Track whether cognitive write times are ordered across a tenant's records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class HistoryOrderStatus:
    """Coverage and ordering known for a tenant's derived-memory writes."""

    last_recorded_at: datetime | None = None
    ambiguous: bool = False
    complete: bool = True


class InMemoryHistoryOrder:
    """A clock shared by the episode and semantic stores of a memory instance."""

    def __init__(self) -> None:
        self._tenants: dict[str, HistoryOrderStatus] = {}

    def record(self, tenant_id: str, at: datetime) -> None:
        previous = self._tenants.get(tenant_id, HistoryOrderStatus())
        last = previous.last_recorded_at
        self._tenants[tenant_id] = HistoryOrderStatus(
            last_recorded_at=max(last, at) if last is not None else at,
            ambiguous=previous.ambiguous or (last is not None and at < last),
            complete=previous.complete,
        )

    def get(self, tenant_id: str) -> HistoryOrderStatus:
        return self._tenants.get(tenant_id, HistoryOrderStatus())

    def clear(self, tenant_id: str) -> None:
        self._tenants.pop(tenant_id, None)
