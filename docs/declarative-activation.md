# Declarative activation (ACT-R) design reference

**Target release:** Cogkura `0.4`  
**Status:** Shipped (base-level + partial matching; spreading in `0.5`)

## Summary

Cogkura `0.4` replaces observation token-overlap `recall()` with ACT-R-inspired **declarative activation** over durable episodic and semantic memories.

Activation combines:

- **Base-level** frequency/recency from access references
- **Partial matching** deterministic cue similarity (mismatch penalties)
- **Spreading activation** — deferred to `0.5` (`enable_spreading_activation=False` by default)

\[
A_i = B_i + S_i + P_i + \epsilon_i
\]

In `0.4`, \(S_i = 0\) and \(\epsilon_i = 0\) by default.

## Public API

| Method | Purpose |
|--------|---------|
| `recall(query, tenant_id=..., ...)` | Rank episodic + semantic memories by activation |
| `record_access(results, tenant_id=...)` | Explicitly reinforce selected memories |

`recall()` is pure — it does not mutate access history.

### Breaking change

`RecallResult` now references `StoredEpisode | StoredSemanticMemory`, not `StoredObservation`. Raw observations are evidence, not declarative recall candidates.

## Configuration defaults

```text
decay:                  0.5
time_unit_seconds:      3600.0   # one hour per activation unit
retrieval_threshold:   -3.0
enable_spreading_activation: false
enable_partial_matching:     true
```

Presentation score (not ACT-R): \(1 / (1 + e^{-(A-\tau)})\)

## Storage

Migration `004_declarative_activation.sql` adds `cogkura.memory_activation_references`.

Postgres apps must pass `PostgresActivationStore` alongside other Postgres stores.

`Memory.clear()` order: activation → semantic → episodic → observations.

## Roadmap

- `0.5` Spreading activation
- `0.6` Forgetting / memory dynamics
- `0.7` Working memory / goal-aware attention
