-- Cognema episodic memory schema (version 002)

ALTER TABLE cognema.observations
    ADD COLUMN IF NOT EXISTS attention_score DOUBLE PRECISION NOT NULL DEFAULT 0.5;

ALTER TABLE cognema.observations
    ADD COLUMN IF NOT EXISTS retention_class TEXT NOT NULL DEFAULT 'full';

ALTER TABLE cognema.observations
    ADD COLUMN IF NOT EXISTS policy_reasons JSONB NOT NULL DEFAULT '[]';

ALTER TABLE cognema.observations
    DROP CONSTRAINT IF EXISTS observations_attention_score_check;

ALTER TABLE cognema.observations
    ADD CONSTRAINT observations_attention_score_check
    CHECK (
        attention_score >= 0.0
        AND attention_score <= 1.0
    );

ALTER TABLE cognema.memories
    ADD COLUMN IF NOT EXISTS memory_key TEXT;

CREATE UNIQUE INDEX IF NOT EXISTS memories_tenant_type_key_idx
    ON cognema.memories (
        tenant_id,
        memory_type,
        memory_key
    )
    WHERE memory_key IS NOT NULL;

CREATE INDEX IF NOT EXISTS memories_tenant_type_active_idx
    ON cognema.memories (
        tenant_id,
        memory_type,
        is_active
    );

CREATE INDEX IF NOT EXISTS memories_tenant_subject_type_idx
    ON cognema.memories (
        tenant_id,
        subject_id,
        memory_type
    );

CREATE INDEX IF NOT EXISTS memories_valid_from_idx
    ON cognema.memories (
        valid_from
    );

ALTER TABLE cognema.memory_evidence
    ADD COLUMN IF NOT EXISTS observation_revision INTEGER;

ALTER TABLE cognema.memory_evidence
    ADD COLUMN IF NOT EXISTS sequence_number INTEGER;

CREATE INDEX IF NOT EXISTS memory_evidence_observation_idx
    ON cognema.memory_evidence (
        observation_id
    );

CREATE INDEX IF NOT EXISTS memory_evidence_memory_sequence_idx
    ON cognema.memory_evidence (
        memory_id,
        sequence_number
    );

CREATE TABLE IF NOT EXISTS cognema.memory_entities (
    memory_id UUID NOT NULL
        REFERENCES cognema.memories(id)
        ON DELETE CASCADE,
    entity_id TEXT NOT NULL,
    entity_role TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (
        memory_id,
        entity_id,
        entity_role
    )
);

CREATE INDEX IF NOT EXISTS memory_entities_entity_idx
    ON cognema.memory_entities (
        entity_id
    );
