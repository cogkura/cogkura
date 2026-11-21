CREATE TABLE cogkura.history_order (
    tenant_id TEXT PRIMARY KEY,
    last_recorded_at TIMESTAMPTZ,
    ambiguous BOOLEAN NOT NULL DEFAULT FALSE,
    complete BOOLEAN NOT NULL DEFAULT TRUE
);
INSERT INTO cogkura.history_order (tenant_id, complete)
SELECT DISTINCT tenant_id, FALSE FROM cogkura.memories;
CREATE FUNCTION cogkura.record_history_order() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN INSERT INTO cogkura.history_order(tenant_id,last_recorded_at) VALUES (NEW.tenant_id,NEW.recorded_at) ON CONFLICT(tenant_id) DO UPDATE SET ambiguous = cogkura.history_order.ambiguous OR (cogkura.history_order.last_recorded_at IS NOT NULL AND EXCLUDED.last_recorded_at < cogkura.history_order.last_recorded_at), last_recorded_at = GREATEST(cogkura.history_order.last_recorded_at, EXCLUDED.last_recorded_at); RETURN NEW; END; $$;
CREATE TRIGGER episode_history_order AFTER INSERT ON cogkura.episode_history FOR EACH ROW EXECUTE FUNCTION cogkura.record_history_order();
CREATE TRIGGER semantic_history_order AFTER INSERT ON cogkura.semantic_history FOR EACH ROW EXECUTE FUNCTION cogkura.record_history_order();
CREATE TRIGGER semantic_revision_history_order AFTER INSERT ON cogkura.semantic_revision_history FOR EACH ROW EXECUTE FUNCTION cogkura.record_history_order();
CREATE TRIGGER semantic_relation_history_order AFTER INSERT ON cogkura.semantic_relation_history FOR EACH ROW EXECUTE FUNCTION cogkura.record_history_order();
