# Plan v3

## Summary

Terminal plan version under `p3-c2`. It closes the eight items left open by `plan_v2_review.md`,
corrects two numerical errors the reviewer caught, and reverses one design decision on the
reviewer's advice.

The substantive change from v2 is in the **verification gate**, which v2 had made
self-contradictory. Preallocating an id for the winning territory necessarily changes the id
stored inside `terr_*.npz`, so "byte-identical territory arrays" and "manifest differs only by
additive keys" cannot both hold. The gate therefore moves to the artifact that actually reaches
the segmentation: **the post-overlay emitted label array must be byte-identical**, compared after
applying the `territory_id -> emitted_id` translation. That is a stronger check than the one it
replaces, because it is the thing the volume sees.

Scope and selected optimization are unchanged from v2 — instrumentation, the per-unit flood array,
robustness, and the written deliverables — with the retry workflow now specified and gated, which
is what the flood array's robustness justification was missing.

## Scope

In scope for `code_v0`: **A** instrumentation, **B** per-unit flood array, **C** robustness,
**D** written deliverables. Designed and deferred: scan sharding, now with designs for *both*
sub-phases (§B.6). Out of scope, unchanged: watershed reuse as a first-class flag, the
`ABISS_NUC_MIN_TAGGED` sweep, and any change to the competition algorithm or cannot-link.

## Proposed Changes

### A. Instrumentation

As v2, with the phase list explicitly separating **`scan_geometry`** from **`map_to_watershed`**
(the reviewer's #6 depends on these being distinguishable). `stage_report.json` is written to a
staging path and atomically published on success; on failure the evidence is preserved as
`stage_report.partial.json`, never under the final name.

### B. Per-unit flood array

**B.1 ID rule.** Scan preallocates a deterministic `>=2^60` id for every territory keyed on
`(parent, anchor)`. Each flood writes its territory under that **internal** id and reports its
voxel count. Merge selects the winner by `(largest voxel count, then lowest anchor id)` and records
the mapping.

**B.2 Manifest ID semantics** (closes new-#8/#12). The manifest records **both**, unambiguously:

```text
internal_territory_id   the preallocated >=2^60 id as stored in terr_<unit>.npz
emitted_id              what appears in the volume: the parent watershed id for the winner,
                        the internal id for every other territory
```

Validation, per the reviewer's #1 tightening: the mapping's domain must equal the **exact**
territory-id set; **exactly one** emitted parent id per unit; all remaining emitted ids unique.
Any violation is fail-closed. `nucleus_overlay.py` consumes the mapping and never re-derives
"largest".

**B.3 Fixed-capacity array.** `0..NUC_MAX_UNITS-1` (default 64); task *i* exits 0 when
`i >= len(units)`; scan fails closed above capacity. Zero units is the natural no-op case.

**B.4 Staging and publication** (closes #4; **reverses v2's quarantine** per the reviewer's Q-2).
All outputs are written under a run-scoped staging directory. The previous canonical
`manifest.json` **remains in place as the last successful publication** and is atomically replaced
only after the merge validates. There is never a window without a valid canonical result, and
concurrent readers are never disrupted. Consumers validate the expected `plan_digest` and
completion marker.

**Fingerprints** now cover every input the stage consumes, by content or authoritative identity:

```text
param JSON                sha256
units.json                sha256 (the plan_digest)
nucleus input             sha256 if a single file; if a store, the store's manifest/index
                          digest plus per-member sizes and count
watershed                 the watershed manifest's own digest (an identity, not WS_PATH, which
                          is only a location)
affinity / cost input     sha256 of the chunk-store index.json plus chunk count -- REQUIRED,
                          because the flood consumes affinity and v2 omitted it
code identity             ABISS native build id via the existing _abiss_build_id helper, plus a
                          digest of the Python sources under lib/abiss/scripts/
```

**B.5 Retry workflow** (closes new-#7 — the missing half of the robustness justification):

```text
failed unit        rerun that array index alone; scan is NOT rerun, so units.json and every
                   completed terr_*.npz are reused unchanged
merge dependency   an afterok chain cannot survive a failed element, so merge is submitted
                   separately after the retry rather than being repaired in place; the
                   documented recovery is: sbatch --array=<failed indices> <flood>, then
                   sbatch --dependency=afterok:<retry> <merge>
reuse safety       merge accepts a per-unit record only on exact plan_digest match, so a stale
                   record from an earlier scan can never satisfy fan-in
```

**B.6 Scan sharding — both sub-phases designed, deferred** (closes #6):

- **`map_to_watershed`**: shard by nucleus id range; each shard converts its nuclei's 80 nm mask
  voxels to mip-0 via `NUC_RATIO` and reads the watershed with chunk-grouped block reads (the
  access pattern `wholevol_nerl.py::_read_lut` already uses for 500,845 scattered nodes); emits
  `(nucleus_id, watershed_id, count)` triples. Merge sums keyed histograms — order-independent.
- **`scan_geometry`**: shard by **z-slab** over the 80 nm nucleus mask; each shard accumulates per
  instance a voxel count, coordinate sums and a bbox. Merge combines by summation and bbox union,
  both associative, then derives centroids and equivalent-sphere radii exactly as the serial path
  does. Deterministic because no floating-point reduction order affects integer counts and bounds.

**Selection rule**, using the separated fractions `g` (geometry), `m` (mapping), `f` (flood):

```text
let d = argmax(g, m, f)
d == m and m >= 0.4  -> shard map_to_watershed
d == g and g >= 0.4  -> shard scan_geometry
d == f               -> the flood array already addresses the dominant phase; no follow-up
otherwise            -> no phase dominates; record the table and do not build
```

Total, mutually exclusive, and — unlike v2's rule — it can no longer select an optimization that
addresses the wrong sub-phase.

### C. Robustness

Unchanged from v2: durable `zero_repairs` judged after completion by
`nucleus_acceptance_report.py`; the five fail-closed conditions **audited in production code
first**, then tested, each required to abort and publish **no canonical manifest** (the reviewer's
#10 tightening); and the oracle-bearing B2 test.

### D. Written deliverables

`docs/nucleus_competition_review.md`, produced in the run folder:

1. **R1–R7 with executable work** (closes #9's two remaining gaps):
   - **R5** — owning target `lib/abiss/src/agg/mean_aggl.cpp::nuc_can_merge` and the attachment
     ordering it consumes. Fixture: a synthetic RAG with one untagged object bridging two
     `PROPER` identities. Ownership oracle: the untagged object must attach to the same identity
     under reversed edge-iteration order and under different shard counts; divergence is the
     finding.
   - **R6** — method: score chunk `z4_y6_x1` twice, once with the local per-chunk scorer and once
     with the whole-volume node LUT restricted to that chunk's nodes. Expected result if local
     scoring is blind to cross-chunk merges: the whole-volume-restricted score is *lower*.
     Verification: if the two agree, the `z4_y6_x1` null stays an open ship blocker; if they
     differ as predicted, it is explained and closed.
   - R3 emits `separation_claim: local_only` unless containment is demonstrable; R4 qualifies every
     cross-arm conclusion; R1, R2, R7 resolved by §C.
2. **Ordered efficiency plan** with expected speedup and determinism risk per change — see the
   corrected figures in §Verification 6.
3. **The `CHUNKMAP_INPUT` repair**: default it to `CHUNKMAP_OUTPUT` only when unset, so
   `param_overrides` wins and watershed reuse stops needing a 126 MB chunkmap copy.
4. **A ship verdict**, which may now cite measured per-skeleton evidence: on native96 the nucleus
   arm improves 3 skeletons and regresses 1 at mt=50 (skel28, −16,274 ERL, −11.5% of its GT
   length) and improves 5 with **zero** regressions at mt=0 — so the mean gain is not masking
   broad over-splitting. The verdict remains constrained by R4 and by the acceptance checks still
   in flight.

## Files and Areas

Unchanged from `plan_v2` §Files and Areas, plus: `dev/zebrafinch/sbatch_nuccomp_flood.sh` gains the
documented retry invocation, and `nucleus_acceptance_report.py` gains the `zero_repairs` judgement.

## Verification Plan

Baseline route per Q-2 of `plan_v1_review`: re-run `nuccomp` alone against an existing completed
watershed, fresh output namespace, references read-only, one baseline per parameterization
(`ABISS_NUC_MIN_TAGGED` 50 and 1024).

1. `python -m pytest tests/unit/test_abiss_nucleus_competition.py -q`.
2. **Emitted label array byte-identical** (replaces v2's contradictory gate, closes new-#8):
   apply the overlay to a fixed cutout using the reference run's territories and using the
   refactored run's territories plus its `territory_id -> emitted_id` mapping; the resulting label
   arrays must be **byte-identical**. `terr_*.npz` byte-identity is explicitly *not* claimed,
   because B.1 changes the winner's stored internal id by design.
3. **Manifest semantic equality after translation**: same territory set, same parent assignment,
   same voxel counts, per run against its own `param`. 8 contact units / 0 bridges on native96;
   9 on win144 including `parent 72199226020331523 anchors [611, 651]`.
4. **Shard-count invariance**: array at throttle 1 versus N yields identical merged output.
5. **Fail-closed**: each of the five conditions aborts, publishes no canonical manifest, and leaves
   the previous canonical manifest intact and valid.
6. **Efficiency acceptance** (closes #11), with the corrected model:

```text
T_serial  = T_nonflood + sum(t_i)                         from the instrumented baseline
T_array  ~= T_nonflood + max(t_i) + overhead              units are NOT balanced; native96 has
                                                          8 units, win144 has 9
predicted speedup = T_serial / T_array
```

   Recomputed illustrative figures for a *balanced* 9-unit case, correcting v2's error:
   `f=0.1 -> 1.10x`, `f=0.4 -> 1.55x`, `f=0.7 -> 2.65x` (v2 said 3.9x, which was wrong). The real
   prediction uses measured `t_i`, not this idealization.

```text
PASS  measured critical path <= 1.15 x predicted T_array
FAIL  measured critical path >  T_serial -- the array is a regression; keep the serial path as
      default, record the result, and retain the array only if gate 7 demonstrates the retry
      benefit independently
```

7. **Failure injection** (closes new-#7): fail unit 8 deliberately; rerun only that index; assert
   every other `terr_*.npz` is unchanged on disk; submit merge; assert it completes and the
   manifest matches the uninterrupted run. This is what converts "retryability" from a claim into
   the flood array's justification.
8. **Downstream regression gates**, each with named input, output and failure criterion:
   - material outside every repair box unchanged versus the reference segmentation;
   - **already-clean nuclei unchanged** (closes #8): for every nucleus **absent from
     `units.json`**, compare its `nucleus_shell_contamination.py --tol 0.0` record between
     reference and refactored runs and require **exact equality of `dominant_segment`,
     `total_mask_voxels`, and the full `segments` list** (seg id and voxel count per entry). Report
     equality is the criterion; any difference fails.
   - `nuc_cuts.data` rejected-edge counts per hierarchy level match the reference, with reference
     counts parsed from the existing completed runs **before** any refactored run.
9. `ccc-validate.sh` green; `git rev-parse HEAD` still `c705458ae5b907bb9c32a75c63c85c6aad7edec7`.

**`code_v0` may not claim gate 6 passed** (reviewer Q-1). It implements and unit-tests; the
coordinator or operator executes the cluster baselines and array runs, using exact commands,
frozen fingerprints, fresh namespaces and expected artifacts that `code_v0` must supply.

## Risks and Questions

- **R-a.** The flood array may still deliver ~1.1x. Accepted: gate 7 makes the retry benefit a
  tested property rather than a claim, and gate 6 defines what happens if it is a regression.
- **R-b.** `nucleus_overlay.py` is in scope, widening blast radius to the path that writes real
  cutouts; gate 2 is voxel-level precisely because of that.
- **R-c.** Fingerprinting a chunk-store by `index.json` digest plus chunk count is weaker than
  hashing every chunk. Chosen deliberately — hashing 726 multi-GB chunks per run is not viable —
  but it would miss an in-place edit of a chunk that preserved the index.
- **R-d.** Gate 8's already-clean-nuclei check compares report fields rather than voxels; a change
  that preserved every reported field while altering voxels would pass. Full voxel comparison of a
  58 GB volume is out of proportion here; the limitation is recorded rather than hidden.
- **Q-1.** `NUC_MAX_UNITS=64` is a guess sized ~7x above the observed 8–9. Confirm or set a policy.

## Changes Since Previous Plan Version

All eight open items from `plan_v2_review.md` closed; the four already-closed items unchanged.

1. **#4 fingerprints** → §B.4 adds the affinity/cost input (sha256 of the chunk-store `index.json`
   plus chunk count), replaces `WS_PATH` with the watershed manifest's own digest, handles a
   nucleus *store* as well as a single file, and covers both native and Python code identity.
2. **#6 selection rule** → §A separates `scan_geometry` from `map_to_watershed`; §B.6 designs
   sharding for **both** sub-phases and the rule uses `argmax(g, m, f)`, so it can no longer select
   an optimization aimed at the wrong phase.
3. **#7 efficiency plan** → arithmetic corrected (`f=0.7` is **2.65x**, not 3.9x); §B.6 gives
   scan-sharding an expected speedup derived from measured `T_map`/`T_geometry` and shard count,
   not a bare ceiling.
4. **#8 already-clean nuclei** → gate 8 names the exact compared fields (`dominant_segment`,
   `total_mask_voxels`, full `segments` list) and requires exact equality, with the residual
   limitation recorded in R-d.
5. **#9 R5/R6** → §D.1 gives R5 an owning target, fixture and ownership oracle, and R6 a method,
   input, expected result and verification step with a defined consequence either way.
6. **#11 efficiency acceptance** → gate 6 uses the correct critical-path model
   (`nonflood + max(t_i) + overhead`), notes native96 has 8 units not 9, and defines PASS, FAIL and
   the regression fallback.
7. **new-#7 retry workflow** → §B.5 specifies requeue, the merge-dependency recovery after an
   `afterok` failure, and reuse safety via `plan_digest`; gate 7 tests it by failure injection.
8. **new-#8 ID representation** → §B.2 makes the manifest record `internal_territory_id` and
   `emitted_id` explicitly; gate 2 replaces the contradictory byte-identity claim with byte
   identity of the **post-overlay emitted label array**.

Reviewer questions adopted: **Q-1** — the cluster baselines are the coordinator's or operator's to
run and `code_v0` must not claim gate 6 passed. **Q-2** — staging directory with publication only
after validation; v2's quarantine-before-execution is **reversed**, since it created a window with
no valid canonical result.
