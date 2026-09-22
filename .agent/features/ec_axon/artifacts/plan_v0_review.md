# Plan v0 Review

## Summary
Reviewer (codex, read-only) judges the direction sound but the plan **not executable as written**: several
execution-critical contracts are undefined. The most consequential findings are (a) the DAG has **no defined
seed segmentation** — the reference pipeline starts from `v0_sm0` but the plan feeds only `raw`, hiding an
undefined initialization inside `axon_split`; (b) the **multi-input registration mechanics are wrong** as
described (a native `GraphOp` receives a `inputs` sequence, so `axon_merge(aff, seg, ...)` cannot be
registered directly); (c) the plan **does not actually enforce "port, not redesign"** (it introduces a
polymorphic `aff_or_seg` API and new defaults without a source-to-port mapping); and (d) the **tutorial does
not perform the requested comparison** — two YAMLs plus a README table never invoke NERL/oracle-merge.
Verification is also judged too weak: a ±0.001 metric tolerance can hide voxel-level drift.

## Findings
Major (13):
1. No seed/v0 node — add an explicit v0 op or document+verify the exact seed `axon_split` vendors.
2. Multi-input registration mechanically incorrect — specify `(inputs, **kwargs)` adapters that validate arity
   and unpack `[aff, seg]`; test input order and invalid arity.
3. Registration site contradicts the task — name the exact call site (lazy built-in registration in
   `decoding/registry.py`), no decorators, no import-time side effects.
4. `stats=` / `inplace=` contracts not planned — preserve source signatures, defaults, mutation behavior,
   return identity, dtype; define stats **staleness** after topology-changing mutation.
5. "Port, not redesign" not enforced — keep source-aligned internals/constants verbatim, add graph adapters
   separately, and record every permitted mechanical change.
6. Tutorial performs no evaluation — configs/commands must supply GT, artifact paths, both NERL modes, and
   `merge_threshold: 10` at the real schema path; prove both branches consume the identical affinity artifact.
7. ±0.001 acceptance too weak — require stage-by-stage exact chunked array equality (or relabel-invariant
   partition equality), with pinned params, checksums, channel order, dtype.
8. No CI parity fixtures for the sensitive gates (local-min/min-frag, `host_both=False`, anchor-slice carve,
   lateral + z-isolated completion, mutual/margin rejection, weak-gap projection, `recover=False`).
9. Verification omits v4, completeness, `seg_stats`, `apply_lut` parity (incl. `inplace` both ways, background
   labels, sparse/high IDs, chunk boundaries).
10. Perf verification insufficient — add `cc3d.statistics` call-count instrumentation, allocation/mutation
    tests for LUT application, and a documented full-volume batch benchmark (wall time + peak memory).
11. YAML omits v4 though it is registered — include an opt-in node, keep `output: v3` default.
12. Deleting `axon_tracklet.py` without auditing references risks breaking the reproduction path — scan refs,
    update the superseded tutorial deterministically, smoke-test entry points; make deletion an explicit choice.

Minor (2):
13. Add an explicit dependency-boundary check (no `dev`/`training`/`evaluation` imports from decoding); invoke
    NERL through the runtime/evaluation stage rather than importing it into decoding.
14. Do not update the public API snapshot merely for new registry entries; test cold-start lazy discovery and
    duplicate registration instead.

## Questions
- Q1 (from plan): delete `axon_tracklet.py` vs deprecation shim — reviewer requires the choice be explicit and
  reference-audited, not left as "delete or update".
- Q2 (from plan): register `axon_complete` despite being om-negative — reviewer implicitly accepts registration
  but requires it to appear in the YAML as an opt-in node.

## Verdict
VERDICT: NEEDS_CHANGES
