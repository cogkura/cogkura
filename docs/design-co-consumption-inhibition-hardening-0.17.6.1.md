# Co-consumption inhibition hardening (0.17.6.1)

Shipped in Cogkura `0.17.6.1`.

> A logical memory consumed in the current use event is not inhibited through another retrieval representation of that same memory.

`build_inhibitory_traces` still skips exact consumed identities and same-lineage pairs. It also skips a competitor whose frozen `competitor_lineage_group` is an active semantic lineage of any consumed result. A semantic result contributes `semantic:<memory_key>`. A support episode contributes that string only for `ACTIVE` support provenance. Semantic slot membership does not count as consumption.

Transient interference, behavioural eligibility, induction pressure, and persistent application are unchanged. Existing traces are not rewritten.

`0.17.6` transient behaviour remains as validated in [`design-transient-interference-reachability-0.17.6.md`](design-transient-interference-reachability-0.17.6.md).
