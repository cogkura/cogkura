-- Cogkura learning and reinforcement schema (version 007)

CREATE TABLE IF NOT EXISTS cogkura.memory_learning_events (
    tenant_id TEXT NOT NULL,
    feedback_id TEXT NOT NULL,
    feedback_fingerprint TEXT NOT NULL,
    subject_id TEXT NULL,
    context_key TEXT NOT NULL,
    occurred_at TIMESTAMPTZ NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (tenant_id, feedback_id)
);

CREATE TABLE IF NOT EXISTS cogkura.memory_learning_feedback (
    tenant_id TEXT NOT NULL,
    feedback_id TEXT NOT NULL,
    memory_kind TEXT NOT NULL,
    memory_key TEXT NOT NULL,
    revision_key TEXT NULL,
    outcome TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (
        tenant_id,
        feedback_id,
        memory_kind,
        memory_key
    ),

    FOREIGN KEY (tenant_id, feedback_id)
        REFERENCES cogkura.memory_learning_events (tenant_id, feedback_id)
        ON DELETE CASCADE,

    CONSTRAINT memory_learning_feedback_outcome_check
        CHECK (outcome IN ('helpful', 'unhelpful', 'incorrect'))
);

CREATE TABLE IF NOT EXISTS cogkura.memory_learning_state (
    tenant_id TEXT NOT NULL,
    context_key TEXT NOT NULL,
    memory_kind TEXT NOT NULL,
    memory_key TEXT NOT NULL,
    helpful_count INTEGER NOT NULL,
    unhelpful_count INTEGER NOT NULL,
    incorrect_count INTEGER NOT NULL,
    first_feedback_at TIMESTAMPTZ NOT NULL,
    last_feedback_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (
        tenant_id,
        context_key,
        memory_kind,
        memory_key
    ),

    CONSTRAINT memory_learning_state_helpful_count_check
        CHECK (helpful_count >= 0),

    CONSTRAINT memory_learning_state_unhelpful_count_check
        CHECK (unhelpful_count >= 0),

    CONSTRAINT memory_learning_state_incorrect_count_check
        CHECK (incorrect_count >= 0)
);

CREATE TABLE IF NOT EXISTS cogkura.memory_learned_associations (
    tenant_id TEXT NOT NULL,
    left_memory_kind TEXT NOT NULL,
    left_memory_key TEXT NOT NULL,
    right_memory_kind TEXT NOT NULL,
    right_memory_key TEXT NOT NULL,
    coactivation_count INTEGER NOT NULL,
    first_reinforced_at TIMESTAMPTZ NOT NULL,
    last_reinforced_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (
        tenant_id,
        left_memory_kind,
        left_memory_key,
        right_memory_kind,
        right_memory_key
    ),

    CONSTRAINT memory_learned_associations_coactivation_count_check
        CHECK (coactivation_count > 0),

    CONSTRAINT memory_learned_associations_distinct_identities_check
        CHECK (
            left_memory_kind <> right_memory_kind
            OR left_memory_key <> right_memory_key
        )
);

CREATE INDEX IF NOT EXISTS memory_learning_state_tenant_context_idx
    ON cogkura.memory_learning_state (tenant_id, context_key);

CREATE INDEX IF NOT EXISTS memory_learning_feedback_tenant_feedback_idx
    ON cogkura.memory_learning_feedback (tenant_id, feedback_id);

CREATE INDEX IF NOT EXISTS memory_learned_associations_tenant_left_idx
    ON cogkura.memory_learned_associations (
        tenant_id,
        left_memory_kind,
        left_memory_key
    );

CREATE INDEX IF NOT EXISTS memory_learned_associations_tenant_right_idx
    ON cogkura.memory_learned_associations (
        tenant_id,
        right_memory_kind,
        right_memory_key
    );
