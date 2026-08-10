# Forgetting and memory dynamics design reference

**Target release:** Cogkura `0.6`  
**Status:** Shipped

## Summary

Cogkura `0.6` adds **Ebbinghaus-inspired memory dynamics** — an `ACTIVE → FADING → FORGOTTEN` lifecycle driven by ACT-R **base-level activation only** (not spreading or partial match).

Cognitive forgetting is separate from physical deletion: episodes and semantic memories remain stored; `FORGOTTEN` memories are excluded from recall by default.

## Public API

```python
from cogkura import Memory

# Explicit maintenance (not wired to sleep())
result = await memory.apply_forgetting(tenant_id="company_123", as_of=None)

# Default recall excludes FORGOTTEN memories
results = await memory.recall("preferred database", tenant_id="company_123")

# Opt in to forgotten candidates (still subject to activation threshold)
results = await memory.recall(
    "preferred database",
    tenant_id="company_123",
    include_forgotten=True,
)

# Reinforcement also reactivates forgotten/fading dynamics
await memory.record_access(results, tenant_id="company_123")
```

`Memory.sleep()` remains a synchronous no-op; call `apply_forgetting()` explicitly.

## Retention model

Retention score maps base-level to \((0, 1)\):

\[
\text{retention} = \sigma(B_i - \theta)
\]

where \(\theta\) is `ActivationConfig.retrieval_threshold`.

Lifecycle rules (`EbbinghausForgettingEvaluator`):

- `retention ≥ fading_retention_threshold` → **ACTIVE** (clears `below_threshold_since`)
- else below fading but `retention > forgotten_retention_threshold` → **FADING**
- else within `grace_period_seconds` of first drop below forgotten threshold → **FADING**
- else → **FORGOTTEN**

`FADING` memories stay in the recall candidate set so spreading activation can still rescue weak memories.

## Configuration defaults

```text
enabled:                       true
fading_retention_threshold:    0.25
forgotten_retention_threshold: 0.05
grace_period_seconds:          604800  (7 days)
exclude_forgotten_from_recall: true
enable_reference_compaction:   true
compact_after_seconds:         2592000 (30 days)
compaction_bucket_seconds:     86400   (1 day)
```

## Weighted reference compaction

`ActivationStore.list_reference_traces()` replaces `list_reference_times` (pre-1.0 break).

Base-level terms use weighted traces:

\[
B_i = \log\sum_j w_j \cdot t_j^{-d}
\]

(`calculate_base_level` uses `log(w_j) - d \log(\text{elapsed}_j)` then `logsumexp`.)

`compact_references()` merges old references into weighted rows per tenant/memory/kind bucket. Same-timestamp merges preserve base level exactly; day-bucket merges target `< 5%` relative base-level drift at compaction time (engineering bound).

## Storage

- `MemoryDynamicsStore` protocol with in-memory and PostgreSQL backends
- Migration `005_forgetting_dynamics.sql`: `memory_dynamics` table; `weight` column on `memory_activation_references`
- `Memory.clear()` clears dynamics after activation references

## PostgreSQL wiring

```python
from cogkura.storage.postgres import (
    PostgresActivationStore,
    PostgresMemoryDynamicsStore,
    # ... other stores
)

memory = Memory(
    activation_store=PostgresActivationStore(engine),
    dynamics_store=PostgresMemoryDynamicsStore(engine),
)
```

## 0.5 parity

Until `apply_forgetting()` transitions state, recall behaviour matches `0.5` (no forgotten filter applied when dynamics are absent).
