# Semantic support-context propagation (0.16.3)

**Status:** Shipped

## Summary

Cogkura `0.16.3` propagates frozen `0.16.1`/`0.16.2` context-match evidence from unique `SUPPORTS` episode derivations onto semantic declarative activation. Propagation adjusts **accessibility**, not relevance: it does not fabricate semantic encoding context, change matching/admission, or alter episodic reinstatement.

## Formula

\[
R_j = M_j \times V_j \qquad R_s = \frac{\sum_j R_j}{K} \qquad C_s = \lambda_{sem} \times R_s \qquad A_s' = A_s + C_s
\]

| Symbol | Source |
|--------|--------|
| \(K\) | Unique `SUPPORTS` episode ids on the semantic memory |
| \(R_j\) | Per-support `ContextMatch.score × cue_coverage` (`0` when unavailable) |
| \(\lambda_{sem}\) | `ActivationConfig.semantic_context_reinstatement_weight` |
| \(C_s\) | `SemanticSupportContextEvidence.activation_contribution` |

## Policy reasons

| Reason | Contribution |
|--------|--------------|
| `applied` | `weight × strength` |
| `disabled` | `0` (diagnostics retain strength) |
| `no_retrieval_context` | `0` |
| `no_supports` | `0` |

## Invariants

- Semantic memories have no singular encoding context; support set = unique `SUPPORTS` derivations only.
- Reuses `DeterministicContextMatcher` and episodic reinstatement strength semantics.
- Missing/unresolved supports count in \(K\) and contribute `0`.
- Forgotten supports resolve from the pre-forgetting loaded episode list (`episode_by_id`).
- Request-scoped memoization: one match per support episode id per `rank`/`inspect` call.
- Episodic `0.16.2` path unchanged; no double-counting of support episodic bonuses on semantic candidates.
- Positive-only: mismatch/unavailable never penalise.
- No storage migration; support-context evidence is not persisted.
- No-context and `semantic_context_reinstatement_weight=0` reproduce `0.16.2` semantic retrieval behaviour.

## Public types

- `SemanticSupportContextItem`, `SemanticSupportContextEvidence`, `SemanticSupportContextReason`
- `DeterministicSemanticSupportContextPolicy`
- `ActivationConfig.semantic_context_reinstatement_weight`
- `RetrievalDiagnostics.support_context` (semantic candidates keep `context_match=None`)

## Tests

- `tests/test_semantic_support_context.py` — policy unit tests (canonical `Rₛ=0.375`)
- `tests/test_016_3_semantic_support_context.py` — behaviour and regression
- `tests/test_016_2_context_reinstatement.py` — episodic gates preserved

## Calibration

See [`findings/0.16.3-semantic-context-calibration.md`](findings/0.16.3-semantic-context-calibration.md).
