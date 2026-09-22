## Findings

- [minor] **Background finding: resolved.** Background and ignored voxels remain excluded; connected components use valid foreground.
- [minor] **Label-pipeline findings 2 and 3: resolved.** The deterministic crop and interior thin-process fixture address both refinements.
- [minor] **AMP finding 4: resolved at plan level.** Calibration is bounded, accuracy uses identical quantized predictions, and the smoke checks skipped-update counts and the final uninterrupted update streak.
- [minor] **Weight-rule finding 5: resolved.** Unweighted losses, four-batch aggregation, invalid-denominator handling, and a fresh smoke at the selected weight are explicit.
- [minor] **Memory/chunking finding 6: resolved.** Gradient parity covers chunk sizes and checkpointing; the memory measurement determines the training setting.
- [minor] **Denominator fixture: resolved.** Nonzero allowed contributions distinguish division by six from division by four.
- [minor] **Connectivity finding: resolved.** Both requested wording corrections are present.
- [minor] **Shape finding: resolved.** Rank-based normalization and singleton-spatial-dimension tests address the bug.
- [minor] **Precision finding: resolved.** Preserving fp64 aligns implementation and oracle precision.
- [major] **Watchdog finding: partially resolved.** Directory pinning, missing-gate deadlines, cancellation confirmation, dry-run behavior, and deterministic tests are specified. However, §5 converts a non-finite gate value to infinity and then permits `GATE PASS` whenever the other gate passes. For example, NaN at step 4999 and a ratio of 1.0 at step 9999 produce PASS. This leaves the previous requirement to reject non-finite values unresolved and can certify invalid training. Non-finite gate data must produce a distinct failure outcome that cannot become PASS; add a mixed finite/non-finite test.

## Questions

- Should a non-finite validation gate trigger immediate cancellation or an immediate failure report without cancellation? This remaining policy decision requires human resolution. The rest of plan_v2 is implementable without further design decisions.

READY: no