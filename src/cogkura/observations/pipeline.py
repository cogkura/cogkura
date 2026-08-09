"""Observation ingestion pipeline."""

from __future__ import annotations

from cogkura.exceptions import ValidationError
from cogkura.observations.models import IngestStatus, ObservationInput
from cogkura.observations.policies import DefaultObservationPolicy, ObservationPolicy
from cogkura.observations.retention import ObservationRetentionMode, apply_retention
from cogkura.storage.base import ObservationStore


class ObservationPipeline:
    """Validates, evaluates policy, and persists observations."""

    def __init__(
        self,
        store: ObservationStore,
        *,
        policy: ObservationPolicy | None = None,
        retention_mode: ObservationRetentionMode = ObservationRetentionMode.FULL,
    ) -> None:
        self._store = store
        self._policy = policy if policy is not None else DefaultObservationPolicy()
        self._retention_mode = retention_mode

    async def ingest(
        self,
        observation: ObservationInput,
        *,
        expected_tenant_id: str | None = None,
    ) -> IngestStatus | None:
        """Ingest one observation. Returns None when rejected by policy."""
        if expected_tenant_id is not None and observation.tenant_id != expected_tenant_id:
            raise ValidationError(
                f"Observation tenant_id {observation.tenant_id!r} does not match "
                f"expected {expected_tenant_id!r}."
            )

        decision = await self._policy.evaluate(observation)
        if not decision.accept:
            return None

        retained = apply_retention(
            observation,
            mode=self._retention_mode,
            decision=decision,
        )
        return await self._store.ingest(observation, retained=retained)
