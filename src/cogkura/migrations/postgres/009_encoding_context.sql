-- Cogkura encoding context columns (version 009)

ALTER TABLE cogkura.observations
    ADD COLUMN IF NOT EXISTS encoding_context JSONB NOT NULL DEFAULT '{}';

ALTER TABLE cogkura.observation_revisions
    ADD COLUMN IF NOT EXISTS encoding_context JSONB NOT NULL DEFAULT '{}';

ALTER TABLE cogkura.memories
    ADD COLUMN IF NOT EXISTS encoding_context JSONB NOT NULL DEFAULT '{}';
