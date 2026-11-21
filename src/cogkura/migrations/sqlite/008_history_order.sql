CREATE TABLE history_order (
    tenant_id TEXT PRIMARY KEY,
    last_recorded_at TEXT,
    ambiguous INTEGER NOT NULL DEFAULT 0,
    complete INTEGER NOT NULL DEFAULT 1
);
INSERT INTO history_order(tenant_id, complete)
SELECT tenant_id, 0 FROM episodes UNION SELECT tenant_id, 0 FROM semantic_memories;
CREATE TRIGGER episode_history_order AFTER INSERT ON episode_history
BEGIN
    INSERT INTO history_order(tenant_id,last_recorded_at) VALUES (NEW.tenant_id,NEW.recorded_at)
    ON CONFLICT(tenant_id) DO UPDATE SET
      ambiguous = history_order.ambiguous OR (history_order.last_recorded_at IS NOT NULL AND NEW.recorded_at < history_order.last_recorded_at),
      last_recorded_at = CASE WHEN history_order.last_recorded_at IS NULL OR NEW.recorded_at > history_order.last_recorded_at THEN NEW.recorded_at ELSE history_order.last_recorded_at END;
END;
CREATE TRIGGER semantic_history_order AFTER INSERT ON semantic_history
BEGIN
    INSERT INTO history_order(tenant_id,last_recorded_at) VALUES (NEW.tenant_id,NEW.recorded_at)
    ON CONFLICT(tenant_id) DO UPDATE SET
      ambiguous = history_order.ambiguous OR (history_order.last_recorded_at IS NOT NULL AND NEW.recorded_at < history_order.last_recorded_at),
      last_recorded_at = CASE WHEN history_order.last_recorded_at IS NULL OR NEW.recorded_at > history_order.last_recorded_at THEN NEW.recorded_at ELSE history_order.last_recorded_at END;
END;
CREATE TRIGGER semantic_revision_history_order AFTER INSERT ON semantic_revision_history
BEGIN
    INSERT INTO history_order(tenant_id,last_recorded_at) VALUES (NEW.tenant_id,NEW.recorded_at)
    ON CONFLICT(tenant_id) DO UPDATE SET
      ambiguous = history_order.ambiguous OR (history_order.last_recorded_at IS NOT NULL AND NEW.recorded_at < history_order.last_recorded_at),
      last_recorded_at = CASE WHEN history_order.last_recorded_at IS NULL OR NEW.recorded_at > history_order.last_recorded_at THEN NEW.recorded_at ELSE history_order.last_recorded_at END;
END;
CREATE TRIGGER semantic_relation_history_order AFTER INSERT ON semantic_relation_history
BEGIN
    INSERT INTO history_order(tenant_id,last_recorded_at) VALUES (NEW.tenant_id,NEW.recorded_at)
    ON CONFLICT(tenant_id) DO UPDATE SET
      ambiguous = history_order.ambiguous OR (history_order.last_recorded_at IS NOT NULL AND NEW.recorded_at < history_order.last_recorded_at),
      last_recorded_at = CASE WHEN history_order.last_recorded_at IS NULL OR NEW.recorded_at > history_order.last_recorded_at THEN NEW.recorded_at ELSE history_order.last_recorded_at END;
END;
