# Spreading activation design reference

**Target release:** Cogkura `0.5`  
**Status:** Shipped

## Summary

Cogkura `0.5` adds **spreading activation** — associative retrieval over entity–memory links inspired by Collins & Loftus (1975).

Declarative activation remains:

\[
A_i = B_i + S_i + P_i + \epsilon_i
\]

where \(S_i\) is spreading activation from cue entities through shared concepts in the recall candidate set.

Spreading is computed once per `recall()` over the full candidate set. No persistent graph storage is required.

## Public API

No new top-level recall method. Supply structured context on `RetrievalCue`:

```python
from cogkura import RetrievalCue

results = await memory.recall(
    RetrievalCue(
        text="What database decision followed Alice's proposal?",
        entity_ids=("alice",),
    ),
    tenant_id="company_123",
)
```

Associatively retrieved memories expose `result.components.spreading > 0`.

Plain text cues seed spreading from candidate entity overlap when `enable_text_entity_seeding` is `true` (default). Explicit `RetrievalCue.entity_ids` behaviour is unchanged. Disable seeding to reproduce pre-`0.12` text-only spreading (`spreading == 0` without `entity_ids`).

`0.13` adds optional incident tag seeding (`enable_incident_tag_seeding`) from episode metadata tags. Tag tokens propagate spreading like entity ids and do not trigger slot admission.

## Configuration defaults

```text
enable_spreading_activation: true
source_activation:             1.0
maximum_associative_strength:  1.0
spreading_decay:               0.5
spreading_max_hops:            2
spreading_min_activation:      0.01
```

Reproduce `0.4` behaviour:

```python
ActivationConfig(enable_spreading_activation=False)
```

## Graph model

Each recall builds a transient bipartite graph from `ActivationCandidate.entity_ids` (subject scope is excluded from association). Activation propagates entity → memory → entity with fan-sensitive strength, distance decay, hop limits, and converging-path accumulation capped at `source_activation`.

## Roadmap

- `0.6` Forgetting / memory dynamics
- `0.7` Working memory / goal-aware attention
