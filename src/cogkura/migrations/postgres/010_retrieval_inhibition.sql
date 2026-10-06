-- Cogkura retrieval-induced inhibitory traces (version 010)
-- Additive. Existing rows are not backfilled. Traces are not read during recall.

CREATE TABLE IF NOT EXISTS cogkura.memory_inhibition_traces (
    id UUID PRIMARY KEY,
    tenant_id TEXT NOT NULL,

    inhibited_memory_kind TEXT NOT NULL
        CHECK (inhibited_memory_kind IN ('episode', 'semantic')),
    inhibited_memory_key TEXT NOT NULL,

    selected_memory_kind TEXT NOT NULL
        CHECK (selected_memory_kind IN ('episode', 'semantic')),
    selected_memory_key TEXT NOT NULL,

    direction TEXT NOT NULL
        CHECK (direction IN ('proactive', 'retroactive', 'co_temporal')),

    scope_key TEXT NOT NULL,
    scope_json JSONB NOT NULL,

    competition_strength DOUBLE PRECISION NOT NULL
        CHECK (competition_strength >= 0.0 AND competition_strength <= 1.0),
    competitor_accessibility DOUBLE PRECISION NOT NULL
        CHECK (competitor_accessibility >= 0.0 AND competitor_accessibility <= 1.0),
    induction_pressure DOUBLE PRECISION NOT NULL
        CHECK (induction_pressure >= 0.0 AND induction_pressure <= 1.0),

    retrieval_evaluated_at TIMESTAMPTZ NOT NULL,
    induced_at TIMESTAMPTZ NOT NULL,

    request_id TEXT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_inhibition_trace_memory
    ON cogkura.memory_inhibition_traces (
        tenant_id,
        inhibited_memory_kind,
        inhibited_memory_key,
        induced_at DESC
    );

CREATE INDEX IF NOT EXISTS idx_inhibition_trace_scope
    ON cogkura.memory_inhibition_traces (
        tenant_id,
        scope_key,
        induced_at DESC
    );

CREATE UNIQUE INDEX IF NOT EXISTS idx_inhibition_trace_request
    ON cogkura.memory_inhibition_traces (
        tenant_id,
        request_id,
        selected_memory_kind,
        selected_memory_key,
        inhibited_memory_kind,
        inhibited_memory_key,
        scope_key
    )
    WHERE request_id IS NOT NULL;
