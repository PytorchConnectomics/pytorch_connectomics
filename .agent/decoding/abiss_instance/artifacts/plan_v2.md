# Plan v2

## Summary

Final plan version. It closes all ten items left open by `plan_v1_review.md` and resolves the
central objection — that plan_v1 measured efficiency without improving it — by **selecting an
optimization now and justifying the selection on grounds that do not depend on the measurement**.

The selected optimization is the **per-unit flood array**. plan_v1 treated it as worth building
only if floods turn out to dominate; that framing was too narrow. Splitting an 11-hour monolith
into 8–9 independently retryable units is a **robustness** improvement on its own terms: today a
failure at flood 8 of 9 discards every completed flood, and the B1 incident already showed this
pipeline losing a 10-hour stage to a 4-second error. It is worth building at 1.1x, and it is worth
building before we know the phase split. Expected speedup is stated honestly below as a range, not
a promise.

Scan sharding is now given a concrete, implementable design and an unambiguous selection rule, and
is deferred to a follow-up run — a deferral that is honest because an optimization does land here.

`nucleus_overlay.py` comes into scope (it must, per finding #1), and the measurement baseline uses
the reviewer's Q-2 route: re-run `nuccomp` alone against an existing completed watershed, in a
fresh namespace, with the references read-only.

## Scope

In scope for `code_v0`:

- **A. Instrumentation** of the serial path (phase timers, per-unit records, defined resource
  semantics, fingerprinted `stage_report.json`).
- **B. Per-unit flood array** — the selected optimization, with the ID rule, fixed-capacity array,
  publication policy and determinism tests below.
- **C. Robustness** — durable zero-repair flag judged by an acceptance step, five fail-closed
  conditions *audited in production code* and then tested, and a B2 test with a real oracle.
- **D. Written deliverables** — `docs/nucleus_competition_review.md`, produced (not committed) in
  the run folder, answering `task.md` §7 including the ordered efficiency plan and the concrete
  `CHUNKMAP_INPUT` repair.

Designed here, deferred to a follow-up run: **scan sharding** (§B.5), selected by the rule in
§B.6. Out of scope, unchanged: watershed reuse as a first-class flag, the `ABISS_NUC_MIN_TAGGED`
sweep, and any change to the competition algorithm or cannot-link.

## Proposed Changes

### A. Instrumentation

As plan_v1 §A: phase timers around mask load, scan, map, classification, each flood, manifest
write; per-flood records (unit, parent, anchors, box extent, voxels visited, territories with
voxel counts, wall seconds); `peak_rss_kib_process` with its unit stated and a `topology` field
(`"serial"` or `"array"`); `schema_version`, run name and `param` fingerprint in the report.

**Publication semantics** (finding #4): `stage_report.json` is written to a temp path and
atomically renamed on success. On failure the partial evidence is preserved as
`stage_report.partial.json` — never under the final name, so a partial report can never be read as
a complete one.

### B. Per-unit flood array (selected optimization)

**B.1 Territory-ID rule.** Scan preallocates a deterministic `>=2^60` id for **every** territory
keyed on `(parent, anchor)`, including the one that will later win. Each flood writes its
territory under its preallocated id and reports its voxel count. Merge selects the winner by
`(largest voxel count, then lowest anchor id)` — the existing tie-break — and records
`parent_id_assigned_to` in the manifest.

**B.2 Overlay translation contract** (closes #1). `nucleus_overlay.py` is **in scope**. The
manifest gains one explicit mapping, `territory_id -> emitted_id`, where exactly one territory per
unit maps to the parent watershed id and the rest map to themselves. `apply_nucleus_competition`
consumes that mapping rather than re-deriving "largest". A missing or non-bijective mapping is a
fail-closed error. This keeps the serial and sharded paths reading one authority.

**B.3 Fixed-capacity array.** `0..NUC_MAX_UNITS-1` (default 64). Task *i* reads `units.json`,
exits 0 immediately when `i >= len(units)`, else floods unit *i*. Statically submittable,
represents zero units naturally, and scan **fails closed** when the unit count exceeds capacity.

**B.4 Publication and fingerprints** (closes #4). One policy, not two:

```text
before execution   any existing manifest.json is MOVED to manifest.json.superseded.<plan_digest>
                   (quarantine, not refusal) so no stale publication can survive a failed merge
during execution   all outputs are written under a run-scoped staging directory
on validation      merge publishes manifest.json by atomic rename, and only then
```

Fingerprints use content or authoritative identities, not path and size: sha256 of the `param`
JSON, of `units.json`, and of the nucleus mask file; the watershed manifest identity; `WS_PATH`;
and the ABISS code version via the existing `_abiss_build_id` helper the gate scripts already use.
Every per-unit record and `terr_*.npz` embeds the `plan_digest`; merge accepts a record only on an
exact digest match and requires **exactly** the expected artifact set.

**B.5 Scan sharding — concrete design, deferred.** Shard the nuclei→watershed mapping by nucleus
id range. Shard *i* takes its slice of the 465 nuclei, converts their 80 nm mask voxels to mip-0
via `NUC_RATIO`, reads the watershed at those coordinates using chunk-grouped block reads (the
same access pattern `wholevol_nerl.py::_read_lut` already uses for 500,845 scattered nodes), and
writes `scanshard_<i>.npz` of `(nucleus_id, watershed_id, count)` triples. Merge sums the
histograms into the identical global mapping; contact/bridge classification stays serial because
it is 465 nuclei and trivially cheap. Determinism is unaffected — summation is order-independent
and the mapping is keyed, not positional.

**B.6 Selection rule** (closes #6; the 40/60 boundary in plan_v1 could select both branches). From
the §A phase table, let `s` be the scan+map fraction and `f` the flood fraction of stage wall:

```text
s >= 0.5            -> build scan sharding (B.5) in the follow-up run
else if f >= 0.5    -> flood array alone is the win; no follow-up needed
else                -> build whichever of s, f is larger, and record the table in the run folder
```

Total and mutually exclusive by construction. Expected speedup for the flood array, stated before
building: `1/(1-f+f/9)` — so **1.1x if f=0.1, 1.6x if f=0.4, 3.9x if f=0.7**. The honest claim is
that this range is wide because the split is unmeasured; the array is justified independently by
per-unit retryability.

### C. Robustness

1. **Zero-repair** (closes #13). The stage emits the warning line and a durable
   `"zero_repairs": true` in manifest and report. `submit_wholevol_sharded.sh` is **not** asked to
   surface a result that does not exist at submit time; instead the acceptance/reporting step
   (`nucleus_acceptance_report.py`, already present in `dev/zebrafinch/`) reads the flag and
   decides whether zero repairs is acceptable for the intended experiment.
2. **Five fail-closed conditions — audit first, then test** (closes #10). For each of: missing
   unanimous seeds; overlapping repair scopes for one parent; generated-ID collision; incompatible
   watershed manifest; invalid nucleus coordinates — first **audit whether the production check
   exists**, add the validation where it does not, then add the test that drives it. Reachability,
   including the case plan_v1 omitted: an *incompatible* watershed manifest is reachable whenever a
   run consumes a `WS_PATH` from a different watershed (exactly the reuse pattern used this week),
   and is distinct from the *missing* manifest the existing test covers.
3. **B2 test with an oracle** (already accepted): construct a cutout with a known competitive
   split, drive the real `apply_nucleus_competition` through both `cut_chunk_agg.py` and
   `cut_chunk_remap.py`, assert the competitive labels are present in the remap output and that
   distinct owners stay distinct. Prefer asserting the complete expected label array.

### D. Written deliverables (closes #7, #9)

`docs/nucleus_competition_review.md`, **produced** in the run folder (not committed — no commits
during a CCC run, finding #14), containing:

1. **R1–R7 answers with planned work, not positions** (closes #9):
   - **R3** — emit `separation_claim: local_only` in the manifest unless global containment is
     demonstrable; containment is checked by comparing each territory's bbox against its parent
     segment's extent beyond the repair box.
   - **R5** — trace the ordering inputs to untagged-object attachment and test invariance across
     shard counts and iteration order.
   - **R6** — test whether local per-chunk scoring can see cross-chunk merges; if it cannot, that
     explains the `z4_y6_x1` null and is recorded; if it can, the null stays an open ship blocker.
   - **R4** — every cross-arm conclusion is qualified, because `ABISS_NUC_MIN_TAGGED` 50 vs 1024
     makes the two runs non-equivalent configurations.
   - R1, R2, R7 as resolved by §C above.
2. **An ordered efficiency plan** with expected speedup **and** determinism risk per change —
   flood array (range above, low risk: ids preallocated, merge ordering sorted), scan sharding
   (bounded by shard count, low risk: keyed summation), watershed reuse (removes ~3.5 h per sweep
   cell, medium risk: touches shared execution).
3. **The `CHUNKMAP_INPUT` repair, concretely**: `abiss_chunk.py:663-664` assigns
   `CHUNKMAP_INPUT` from `CHUNKMAP_OUTPUT` inside a `payload.update()` that overrides
   `param_overrides`. The repair is to default `CHUNKMAP_INPUT` to `CHUNKMAP_OUTPUT` **only when
   unset**, letting `param_overrides` win, so watershed reuse stops needing a 126 MB chunkmap copy.
4. **A ship verdict** for the feature, constrained by R4 and by whichever of §4's acceptance checks
   have landed.

## Files and Areas

| path | change |
|---|---|
| `lib/abiss/scripts/nucleus_competition.py` (747) | timers, per-flood records, report, scan/flood/merge entry points, publication policy, fingerprints, fail-closed audit |
| `lib/abiss/scripts/nucleus_overlay.py` (279) | **now in scope**: consume the explicit `territory_id -> emitted_id` mapping; fail closed if absent or non-bijective |
| `dev/zebrafinch/sbatch_nucleus_competition.sh` (36) | scan/merge wrapper; `set -u` after activation |
| `dev/zebrafinch/sbatch_nuccomp_flood.sh` (new) | fixed-capacity array over units |
| `dev/zebrafinch/submit_wholevol_sharded.sh` (115) | submit the 3-op chain; no zero-repair claim at submit time |
| `dev/zebrafinch/nucleus_acceptance_report.py` (existing) | read and judge `zero_repairs` |
| `tests/unit/test_abiss_nucleus_competition.py` (442) | five fail-closed tests, oracle B2 test, sharded-equals-serial voxel test, shard-count invariance |
| `.agent/decoding/abiss_instance/docs/nucleus_competition_review.md` (new) | §D deliverables |

## Verification Plan

Baseline route per the reviewer's Q-2: re-run `nuccomp` **alone** against an existing completed
watershed, in a **fresh output namespace**, with reference artifacts read-only, one baseline for
**each** parameterization (`ABISS_NUC_MIN_TAGGED` 50 and 1024).

1. `python -m pytest tests/unit/test_abiss_nucleus_competition.py -q` — existing tests unchanged,
   plus the new ones.
2. **Sharded equals serial, voxel-for-voxel** (closes #1): every `terr_*.npz` from the array path
   is byte-identical to the serial path's, and the overlay applied to the same cutout yields an
   identical label array. This is the gate the reviewer said was missing.
3. **Manifest equality modulo documented additive keys** (`zero_repairs`, report pointers,
   `parent_id_assigned_to`, `separation_claim`), per run against its own `param`: 8 contact units
   / 0 bridges on native96; 9 on win144 including `parent 72199226020331523 anchors [611, 651]`.
   Never compared across runs.
4. **Shard-count invariance**: array throttled to 1 versus N yields byte-identical merged output.
5. **Fail-closed**: each of the five conditions aborts, writes no `manifest.json`, and leaves any
   quarantined prior manifest under its `.superseded.<digest>` name.
6. **Efficiency acceptance** (closes #11): measured **end-to-end critical-path** elapsed time for
   the array path versus the instrumented serial baseline on the same input, compared against the
   `1/(1-f+f/9)` prediction computed from the same run's phase table. Summed array-task time is
   not an acceptable metric.
7. **Downstream regression gates** (closes #8), each with named input, output and failure
   criterion:
   - material **outside every repair box** is unchanged versus the reference segmentation;
   - **already-clean nuclei are unchanged** — the half plan_v1 omitted — checked from the
     `nucleus_shell_contamination.py --tol 0.0` reports for nuclei absent from `units.json`;
   - `nuc_cuts.data` **rejected-edge counts per hierarchy level** match the reference; reference
     counts are established by parsing the existing completed runs' `nuc_cuts.data` **before** any
     refactored run, and recorded in the run folder.
8. `ccc-validate.sh` stays green and `git rev-parse HEAD` still equals
   `c705458ae5b907bb9c32a75c63c85c6aad7edec7`.

## Risks and Questions

- **R-a.** The flood array may deliver only ~1.1x if the scan dominates. Accepted deliberately:
  it is justified by per-unit retryability, and §B.6 then selects scan sharding for the follow-up.
- **R-b.** Bringing `nucleus_overlay.py` into scope widens the blast radius to the path that
  applies territories to real cutouts. Mitigated by verification 2 being voxel-level.
- **R-c.** Quarantining a prior manifest mutates run output before execution. It is reversible
  (the file is renamed, not deleted) and is the only policy that prevents a stale publication
  surviving a failed merge.
- **R-d.** `NUC_MAX_UNITS` too low silently caps repairs — hence scan fails closed above capacity,
  tested in verification 5.
- **Q-1.** `code_v0` cannot itself wait ~7–11 h for the baseline run inside a CLI call. The plan
  assumes the coder implements and unit-tests, and the **coordinator or operator** executes the
  cluster baseline and records the phase table. Confirm that division.
- **Q-2.** Is quarantine-before-execution acceptable operationally, or should the staging
  directory be fully separate with publication only at the end (no quarantine of the old file
  until success)? The latter is safer but leaves two manifests visible mid-run.

## Changes Since Previous Plan Version

All ten open items from `plan_v1_review.md` are closed; the four the reviewer marked resolved are
unchanged.

1. **#1 overlay contract** → §B.2 puts `nucleus_overlay.py` in scope with an explicit
   `territory_id -> emitted_id` mapping and fail-closed validation; verification 2 adds the
   voxel-level sharded-equals-serial gate.
2. **#4 publication policy and fingerprints** → §B.4 replaces "removes or refuses" with one policy
   (quarantine → stage → publish on validation) and uses content digests plus the ABISS build id
   rather than path and size; §A gives `stage_report.json` atomic publication and a `.partial`
   failure name.
3. **#6 scan branch and threshold** → §B.5 is a concrete implementable design; §B.6 replaces the
   overlapping 40/60 rule with a total, mutually exclusive one.
4. **#7 §7 deliverable 2 and CHUNKMAP_INPUT** → §D.2 adds the ordered efficiency plan with
   per-change speedup and determinism risk; §D.3 gives the concrete precedence repair.
5. **#8 acceptance gates underspecified** → verification 7 adds the already-clean-nuclei half and
   states how reference rejected-edge counts are established, with named inputs and failure
   criteria.
6. **#9 R3/R5/R6 cosmetic** → §D.1 replaces "a position" with planned work for each, including
   emitting `separation_claim: local_only` unless containment is demonstrable.
7. **#10 abort-condition audit** → §C.2 audits production checks before writing tests and states
   reachability for the incompatible-manifest case plan_v1 omitted.
8. **#11 no efficiency criterion** → an optimization now lands in this increment, and
   verification 6 gives it a measurable acceptance criterion against a stated prediction.
9. **#13 launcher cannot surface a future result** → §C.1 moves the judgement to the acceptance
   step and drops the launcher claim.
10. **#14 "committed as"** → §D says the review is **produced** in the run folder; no commits
    occur during the run.

The central objection — measuring efficiency without improving it — is resolved by selecting the
flood array now on robustness-plus-speed grounds rather than deferring all optimization behind the
measurement.
