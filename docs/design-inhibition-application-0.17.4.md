# Persistent inhibition recovery and retrieval application (0.17.4)

Shipped in Cogkura `0.17.4`.

> Stored inhibitory traces affect retrieval only when `InhibitionConfig.apply_to_recall` is true and the current query matches the trace scope. The default is off, and that path does not read `InhibitionStore`.

## Pipeline

```text
A_context
  -> scope match + half-life recovery from induced_at
  -> A_inhibited = A_context + I_persistent
  -> competition and transient interference
  -> threshold, rank, collapse
```

`I_persistent = -inhibition_weight * noisy_or(remaining strengths)` and is never positive. Recovery is `induction_pressure * 2 ** (-elapsed / half_life)`. Traces below `minimum_remaining_strength` are ignored and not deleted.

`as_of` bounds which traces exist and how far they have recovered. `valid_at` selects which facts are valid; it does not rewind inhibition history.

Transient interference then uses post-inhibition activation as competitor accessibility. Explicit semantic admission is decided before this penalty and is not revoked by it.

Recording (`enabled`) and application (`apply_to_recall`) are independent. Recall never writes traces.

## Out of scope

Compaction, deletion, cancellation, learned recovery rates, embeddings, metamemory flags, and working-memory algorithm changes.
