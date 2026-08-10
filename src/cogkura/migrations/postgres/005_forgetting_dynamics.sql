-- Cogkura forgetting dynamics schema (version 005)

CREATE TABLE IF NOT EXISTS cogkura.memory_dynamics (
    tenant_id TEXT NOT NULL,
    memory_kind TEXT NOT NULL
        CHECK (memory_kind IN ('episode', 'semantic')),
    memory_key TEXT NOT NULL,

    retention_state TEXT NOT NULL
        CHECK (retention_state IN ('active', 'fading', 'forgotten')),

    last_base_level DOUBLE PRECISION NOT NULL,
    last_retention_score DOUBLE PRECISION NOT NULL
        CHECK (last_retention_score >= 0.0 AND last_retention_score <= 1.0),

    below_threshold_since TIMESTAMPTZ NULL,
    forgotten_at TIMESTAMPTZ NULL,

    evaluated_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (tenant_id, memory_kind, memory_key)
);

CREATE INDEX IF NOT EXISTS idx_memory_dynamics_tenant_state
    ON cogkura.memory_dynamics (tenant_id, retention_state);

CREATE INDEX IF NOT EXISTS idx_memory_dynamics_tenant_forgotten_at
    ON cogkura.memory_dynamics (tenant_id, forgotten_at);

CREATE INDEX IF NOT EXISTS idx_memory_dynamics_tenant_evaluated_at
    ON cogkura.memory_dynamics (tenant_id, evaluated_at);

ALTER TABLE cogkura.memory_activation_references
    ADD COLUMN IF NOT EXISTS weight INTEGER NOT NULL DEFAULT 1
        CHECK (weight > 0);
