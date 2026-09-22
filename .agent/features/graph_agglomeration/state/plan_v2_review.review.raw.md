Plan v2 is not yet executable without inventing material linker semantics. Five of the six prior major findings are resolved; the reproducible operating-point finding is not.

## Findings

- [major] The ambiguity score is circular and incomplete. `ambiguity` uses competing `pair_score`s, but the only declared score already subtracts `ambiguity`. The plan must define a separate raw score and specify competitor exclusion, aggregation across both endpoints, no-competitor behavior, clipping/division behavior, and whether mutual-best ranks raw or final score. These choices change unions and NERL.

- [major] S5 requires “tangent/caliber floors,” but no caliber threshold, default, or CLI flag exists. Code must either invent a threshold or ignore the stated condition. Explicitly remove the caliber floor or pin `tau_caliber_min`—possibly `0`.

- [minor] The no-edge metric comparison validates sampled evaluator equivalence but cannot prove collision freedom across all labels because GT is sparse and the compared values are aggregates. Add structural zero-union assertions: one distinct nonzero ID per enumerated node, background fixed at zero, and complete remap lookup coverage.

- [minor] S4 says non-mutual candidates are dropped, while S5/S6 promise rejected and ambiguous edges in the certificate. Retain all generated candidates for certification while allowing only mutual-best candidates into union processing.

- [minor] The marker gate remains qualitative (“close” and “large shortfall”), and the fallback-source wording weakens the pinned-source contract. Define a numeric pass criterion and forbid automatic fallback for this run.

- [minor] The claimed crossing-schema validation is not an explicit hard verification step. Assert all 600 crossing files have the required keys and consistent row lengths before linking.

The total namespace design, exactly-once remapping contract, centralized marker stage, quarantine/transitive firewall semantics, `tau_iou_min = 0`, and single `eps=1e-4` oracle gate are otherwise coherent.

READY: no