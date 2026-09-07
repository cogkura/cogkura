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

`0.13` adds same-slot SUPPORT collapse before selection (`collapse_same_slot_support`, default `true`) and a stale-goal penalty (`stale_goal_penalty`) when goal text contains `stale`, applied to SUPERSEDED semantics and metadata tags tagged `stale`.

## Limitations

- No LLM or embedding relevance judges
- No persistent working-memory store or migration
- No automatic `record_access` on selection
- Previous-only items are not reinserted without passing recall again

## 0.15.9 working-memory chunking and coverage

`0.15.9` groups related recall candidates into deterministic **chunks** before greedy selection. `max_items` counts **chunks**, not raw memories (default remains `8`). Chunking is enabled by default (`WorkingMemoryConfig.enable_chunking=true`); set `enable_chunking=false` to restore item-level `0.15.8` selection on the same pool.

Research basis: Baddeley’s bounded workspace and Miller’s chunking insight—capacity applies to meaningful units, not every surface form.

**Collection vs independent many:** `cardinality=MANY` semantics that share `slot_key`, `status`, `relevance_tier`, and provenance (shared SUPPORT episode or observation evidence) form one `SEMANTIC_COLLECTION` chunk—for example `database = postgres / replicated / encrypted` from one source becomes one bullet. Same predicate with disjoint provenance (hiking vs skiing) stays separate chunks with object-specific coverage keys.

**Coverage-aware selection:** staged greedy pick prefers uncovered `coverage_key` areas, then activation/goal/importance within tier. Jaccard inhibition applies to chunk `serialized_text`, not raw member statements.

**Render and access:** `MemoryContext.render()` emits one bullet per selected chunk using deterministic serialized text (Oxford-comma collections, semantic-only render for structured `SEMANTIC_WITH_SUPPORT`). `WorkingMemorySnapshot.recall_results` and `record_context_use()` flatten **included** chunk members for reinforcement; trimmed members are omitted. Inspect `WorkingMemorySnapshot.chunks` for all formed chunks, rejection reasons, and member include/omit counts. Members and rendered text are intentionally separate: support episodes may be omitted from `serialized_text` while remaining in `member_recalls`.

## 0.15.10 chunk primary correctness

`0.15.10` fixes a correctness bug where `SEMANTIC_WITH_SUPPORT` serialization assumed `members[0]` was always the semantic. Relevance ordering may legitimately place a supporting episode first; structural primary is now resolved at chunk construction and used by the serializer. Member relevance order remains independent and is still used for diagnostics, trimming, and display.

## 0.15.11 support serialization precision

`SEMANTIC_WITH_SUPPORT` chunks with structured predicate and object now serialize the **semantic statement only** by default. Supporting episodes remain chunk members for provenance, inspection, and `record_context_use`; only rendered context text is omitted when the semantic already carries the structured value.

Chunk membership is separate from chunk serialization. Token-budget trimming may still drop members from `included_members`; that path is unrelated to support-text omission.

## 0.15.12 frozen rendering contract

**Frozen for the 0.15 line:** structured `SEMANTIC_WITH_SUPPORT` chunks serialize the semantic primary statement only. Supporting episodes stay attached as chunk members (`member_identities`, `member_recalls`) for inspection and `record_context_use`; omitted support **text** does not omit support **access**. Chunk member count is not the number of rendered statements.

Use `WorkingMemoryChunk.primary_identity`, `member_identities`, `serialized_text`, and `WorkingMemoryItem.member_recalls` to distinguish members from model-facing text. `SEMANTIC_COLLECTION` chunks still Oxford-join object values; episodic chunks still render episode text. Support-detail rendering heuristics remain future work (0.16+).

`inspect_recall()` remains the diagnostic API for admission; `prepare_context()` explains selection. Admitted candidates may be selected while capacity remains because `minimum_goal_relevance` and `minimum_selection_score` default to `0.0`.

## 0.16.0 encoding context vs working memory

**Encoding context** (`ObservationContext`, `MemoryContextSignature`) describes the circumstances present when an observation or episode was encoded (conversation, goal, activity, domain, and related dimensions).

**Retrieval context** (`RetrievalContext`) describes structured cues supplied with a recall request. In `0.16.1`, Cogkura compares retrieval context to stored encoding context and exposes `ContextMatch` on recall diagnostics. In **`0.16.2`**, matching evidence may boost episodic activation upstream of working-memory selection when `context_reinstatement_weight > 0`. In **`0.16.3`**, unique `SUPPORTS` episode matches may boost semantic activation upstream when `semantic_context_reinstatement_weight > 0`. Working-memory chunk construction and render format are unchanged for no-context calls.

**Working memory** (`WorkingMemorySnapshot`, `MemoryContext`) is the bounded set of memories currently selected for model-facing presentation.

These terms all use “context” but refer to different concepts. Encoding context is historical evidence on episodic traces. Retrieval context is a request-time cue overlay that may reinstate episodic accessibility in `0.16.2` and semantic accessibility via support propagation in `0.16.3`. Working memory is transient retrieval output. No-context recall and disabled context weights preserve prior release selection behaviour.

See also [`configuration.md`](configuration.md).
