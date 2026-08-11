# Working memory and inhibition design reference

Target release: 0.7  
Status: Shipped

## Summary

Cogkura `0.7.0` adds a deterministic, bounded **working-memory selection layer** between declarative recall and downstream reasoning. Working memory selects a small subset of activated `RecallResult` candidates using goal relevance, item and token budgets, competitive redundancy inhibition, and short-timescale decay with explicit carry-over via `previous` snapshots.

Working memory is **not persisted**. Selection does not call `record_access()` or modify ACT-R base-level history.

## Research basis

The implementation is research-inspired (Baddeley working memory, executive inhibition, Miyake et al.) rather than a literal neuropsychology simulation. Limited capacity maps to bounded selection; attention maps to goal-sensitive ranking; inhibition maps to competitive suppression of redundant candidates; transient maintenance maps to fast decay and refresh.

## Relationship to declarative activation

`Memory.recall()` answers which durable memories are sufficiently activated by a retrieval cue. `Memory.select_working_memory()` answers which of those activated candidates deserve attention for the current goal.

Selection consumes recall candidates only. It does not re-scan stores or run another spreading-activation pass.

## Relationship to forgetting

Long-term forgetting (`ACTIVE` → `FADING` → `FORGOTTEN`) operates on ACT-R base level over days to months. Working-memory decay operates over seconds to minutes on **previous snapshot strength** only. Working-memory selection never updates `memory_activation_references`, `memory_dynamics`, or stored episodes/semantics.

## Public API

```python
workspace = await memory.select_working_memory(
    "What database should we use?",
    tenant_id="company_123",
    subject_id="customer_42",
    goal="Choose a production database with low operational complexity.",
)

workspace = await memory.select_working_memory(
    "What are the operational risks?",
    tenant_id="company_123",
    subject_id="customer_42",
    goal="Choose a production database with low operational complexity.",
    previous=workspace,
)
```

Inject configuration and estimators via `Memory(working_memory_config=..., token_estimator=...)`.

## Goal relevance

Goal relevance is computed separately from declarative activation as the mean of available component scores from the goal `RetrievalCue`: text token coverage, subject match, entity coverage, predicate match, object exact or token coverage, and qualifier-pair coverage. If `goal` is omitted, the normalised query cue becomes the goal.

## Selection equation

Base priority combines normalised weights over:

- `RecallResult.score` (activation component)
- goal relevance
- `memory.importance`
- decayed carry-over from `previous`

Competitive inhibition subtracts a penalty when lexical Jaccard similarity to an already selected statement exceeds `redundancy_threshold`.

## Competitive inhibition

Greedy selection recalculates inhibition after each pick. Near-duplicate statements compete; distinct facts about the same entity are not penalised by entity overlap alone.

## Working-memory decay

For memories present in `previous` and again in the current candidate pool:

```text
carryover = previous_transient_strength × 2^(-Δt / half_life)
```

Memories in `previous` that are not recalled again drop out. There is no hidden session cache inside `Memory`.

## Prompt budgeting

Selection enforces `max_items` and `max_prompt_tokens` (or per-request `prompt_budget_tokens`). Oversized statements are skipped; statements are never truncated.

`ApproximateTokenEstimator` uses `max(1, ceil(utf8_bytes / 4))` for non-empty text. Inject a model-specific `TokenEstimator` for tighter accounting. Cogkura budgets memory statements only—not full LLM prompts.

## Explicit reinforcement

Selection does not record access. After selection:

```python
await memory.record_access(workspace.recall_results, tenant_id="company_123")
```

## Configuration defaults

See `WorkingMemoryConfig` in `src/cogkura/models.py`. Defaults include `candidate_pool_size=50`, `max_items=8`, `max_prompt_tokens=2048`, and `decay_half_life_seconds=300`.

## Limitations

- No LLM or embedding relevance judges
- No persistent working-memory store or migration
- No automatic `record_access` on selection
- Previous-only items are not reinserted without passing recall again
