-- Cogkura declarative activation schema (version 004)

CREATE TABLE IF NOT EXISTS cogkura.memory_activation_references (
    id UUID PRIMARY KEY,
    tenant_id TEXT NOT NULL,

    memory_kind TEXT NOT NULL
        CHECK (memory_kind IN ('episode', 'semantic')),

    memory_key TEXT NOT NULL,

    reference_kind TEXT NOT NULL
        CHECK (reference_kind IN ('retrieved', 'rehearsed')),

    referenced_at TIMESTAMPTZ NOT NULL,

    request_id TEXT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_activation_reference_lookup
    ON cogkura.memory_activation_references (
        tenant_id,
        memory_kind,
        memory_key,
        referenced_at DESC
    );

CREATE INDEX IF NOT EXISTS idx_activation_reference_tenant_time
    ON cogkura.memory_activation_references (
        tenant_id,
        referenced_at DESC
    );

CREATE UNIQUE INDEX IF NOT EXISTS idx_activation_reference_request
    ON cogkura.memory_activation_references (
        tenant_id,
        request_id,
        memory_kind,
        memory_key,
        reference_kind
    )
    WHERE request_id IS NOT NULL;
