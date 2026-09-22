# Plan v2 Review

## Summary

Plan v2 resolves global node identity, candidate-versus-selected reporting, the unlinked oracle threshold, deterministic edge ordering, and the helper-signature mismatch. Material execution gaps remain.

## Findings

- [major] Upstream eligibility still does not establish independent tubes. Two traces can remain disjoint for three sections and converge immediately afterward—the same-axon over-split case this gate must reject. “Follow spine links” is also undefined for branching predecessors. Use deterministic upstream-only components after removing the fusion transition, require distinct components, and define `min_len` by axial depth rather than node count.

- [major] Run termination and ownership remain incorrect or ambiguous. A still-fused unsplit section cannot be assigned to only one side without losing the other identity. Natural separation requires deterministic one-to-one terminal matching. Terminals are neither explicitly owned nor included in `mand_links`, allowing later runs to replace them and leaving terminal-to-seed connectivity unguaranteed. Seed-anchor ownership is likewise inconsistent with the stated exclusive-owner rule. Define handling for every stop cause, ownership of every committed label, rollback/skip behavior, and mandatory terminal links.

- [major] Consequently, constrained-UF initialization is not proven safe. A label can appear in multiple run-side sets, singleton metadata can overwrite prior annotations, and forced unions are merely asserted conflict-free. Accumulate all run annotations per label, reject contradictory membership before linking, and conflict-check metadata during forced unions.

- [major] Detection and propagation are not reproducible as specified. `area_tol` has no fixed default or CLI/reference binding, and “single overlapping next section” has no overlap threshold, selection rule, or tie handling. Additionally, the plan should establish that `IoU >= 0.2` is intentionally the fusion-detector definition; the task describes raw-overlap detection, while `0.2` is explicitly the conservative-link threshold.

- [major] `--zslice` remains incomplete. Instance metrics still require GT cropped to the identical z interval with bounds and shape validation. Because slice mode omits NERL, it cannot use Stage 6’s NERL-based `{a,c}` selection; the plan must specify which artifact becomes `decode_v2_z0-96.h5`, whether a/b/c are also emitted, and how the automatic slice suffix composes with `--tag`. `--full`/`--zslice` exclusivity and default `--tag` behavior should also be explicit.

- [major] Verification has contradictory completion criteria: `c` versus `b` is “reporting-only” in step 3 but `c > b` is mandatory in the final Success definition. Choose one gate consistent with the task’s permitted non-regressing fallback. Extend the self-test to require terminal—not merely split-piece—connectivity and cover a terminal subsequently encountered as a run head.

- [major] The planned literal SLURM option `--mem120G` is invalid. Use `--mem=120G` or `--mem 120G`; otherwise the required full-volume verification may fail before execution.

- [minor] `split_merges.md` is modified in verification but omitted from Scope and Files and Areas. List it and clarify that it is updated after the full run.

Findings tagged [minor]/[major] above.
READY: no