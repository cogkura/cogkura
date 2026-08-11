# Learning and reinforcement (0.9)

Cogkura `0.9.0` adds deterministic learning from application-supplied outcomes.

## Public API

```python
result = await memory.learn(LearningFeedback(...))
states = await memory.list_learning_state(tenant_id=..., goal=...)
```

`Memory.learn()` accepts typed outcomes:

- `HELPFUL` — ACT-R reinforcement trace, positive utility, association co-use
- `UNHELPFUL` — contextual utility penalty only
- `INCORRECT` — utility accounting only; semantic corrections stay on the `0.8` observe → consolidate path

Feedback is idempotent by `tenant_id + feedback_id`. Conflicting fingerprints raise `StorageError`.

## Separation of mechanisms

| Mechanism | Source |
|-----------|--------|
| ACT-R base level | `record_access()` + HELPFUL learning traces |
| Contextual utility | persisted counts → working-memory adjustment |
| Spreading associations | HELPFUL co-use pairs only |

`learn()` does not call `record_access()`. Negative feedback never creates negative ACT-R references.

## Working memory

When learning state exists, `select_working_memory()` applies a bounded utility adjustment outside the normalized activation/goal/importance/carry-over weights. With no learning state, ranking matches `0.8`.

## Storage

- In-memory: `InMemoryLearningStore`
- PostgreSQL: migration `007_learning_reinforcement.sql`, `PostgresLearningStore`

See [`examples/learning.py`](../examples/learning.py).
