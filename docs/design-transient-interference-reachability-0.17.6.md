# Transient interference reachability (0.17.6)

Shipped in Cogkura `0.17.6`.

> Transient interference is retained and validated. It remains off unless `apply_interference` is true, and it still requires behavioural eligibility.

Episodes that support one shared active semantic slot now carry that slot into competition matching, and may use the supported semantic's existing cue fit. Same-slot and same-subject/same-predicate pairs clear `minimum_behavioral_strength` on relationship strength, while cue fit stays a separate gate. Pair pressure is unchanged: competition strength times competitor accessibility, then noisy-OR.

Proactive and retroactive penalties are reachable for temporally ordered payments-api deployment memories. Different predicates, other services, shared entities, broad subjects, co-temporal pairs, same lineage, superseded semantics, future memories, and unrelated high-accessibility memories do not receive a transient penalty.

Findings: [`0.17.6-transient-interference-reachability.md`](findings/0.17.6-transient-interference-reachability.md).
