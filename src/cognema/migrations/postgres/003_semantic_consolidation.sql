-- Cognema semantic consolidation schema (version 003)

CREATE UNIQUE INDEX IF NOT EXISTS memories_tenant_id_memory_id_idx
    ON cognema.memories (tenant_id, id);

CREATE TABLE IF NOT EXISTS cognema.semantic_claims (
    tenant_id TEXT NOT NULL,
    memory_id UUID NOT NULL,
    slot_key TEXT NOT NULL,

    subject_entity_id TEXT,
    predicate TEXT NOT NULL,
    object_value TEXT NOT NULL,
    object_entity_id TEXT,

    polarity TEXT NOT NULL,
    cardinality TEXT NOT NULL,
    qualifiers JSONB NOT NULL DEFAULT '{}',

    status TEXT NOT NULL,
    support_count INTEGER NOT NULL,
    contradiction_count INTEGER NOT NULL,

    first_supported_at TIMESTAMPTZ NOT NULL,
    last_supported_at TIMESTAMPTZ NOT NULL,

    PRIMARY KEY (memory_id),

    FOREIGN KEY (memory_id)
        REFERENCES cognema.memories(id)
        ON DELETE CASCADE,

    CONSTRAINT semantic_claims_polarity_check
        CHECK (polarity IN ('affirm', 'deny')),

    CONSTRAINT semantic_claims_cardinality_check
        CHECK (cardinality IN ('one', 'many')),

    CONSTRAINT semantic_claims_status_check
        CHECK (status IN ('active', 'contested', 'superseded')),

    CONSTRAINT semantic_claims_support_count_check
        CHECK (support_count >= 0),

    CONSTRAINT semantic_claims_contradiction_count_check
        CHECK (contradiction_count >= 0)
);

CREATE INDEX IF NOT EXISTS semantic_claims_tenant_slot_idx
    ON cognema.semantic_claims (tenant_id, slot_key);

CREATE INDEX IF NOT EXISTS semantic_claims_tenant_predicate_idx
    ON cognema.semantic_claims (tenant_id, predicate);

CREATE INDEX IF NOT EXISTS semantic_claims_tenant_subject_idx
    ON cognema.semantic_claims (tenant_id, subject_entity_id);

CREATE INDEX IF NOT EXISTS semantic_claims_tenant_object_entity_idx
    ON cognema.semantic_claims (tenant_id, object_entity_id);

CREATE TABLE IF NOT EXISTS cognema.memory_derivations (
    tenant_id TEXT NOT NULL,
    target_memory_id UUID NOT NULL,
    source_memory_id UUID NOT NULL,
    relation TEXT NOT NULL,
    contribution_score DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (
        target_memory_id,
        source_memory_id,
        relation
    ),

    FOREIGN KEY (tenant_id, target_memory_id)
        REFERENCES cognema.memories(tenant_id, id)
        ON DELETE CASCADE,

    FOREIGN KEY (tenant_id, source_memory_id)
        REFERENCES cognema.memories(tenant_id, id)
        ON DELETE CASCADE,

    CONSTRAINT memory_derivations_relation_check
        CHECK (relation IN ('supports', 'contradicts')),

    CONSTRAINT memory_derivations_score_check
        CHECK (
            contribution_score >= 0.0
            AND contribution_score <= 1.0
        ),

    CONSTRAINT memory_derivations_distinct_memory_check
        CHECK (target_memory_id <> source_memory_id)
);

CREATE INDEX IF NOT EXISTS memory_derivations_target_idx
    ON cognema.memory_derivations (target_memory_id);

CREATE INDEX IF NOT EXISTS memory_derivations_source_idx
    ON cognema.memory_derivations (source_memory_id);
