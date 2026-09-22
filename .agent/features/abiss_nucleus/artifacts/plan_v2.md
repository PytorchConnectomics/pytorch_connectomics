# Plan v2

## Summary

Make each supervoxel's nucleus identity a **global** fact instead of a per-chunk one.

`NucExtractor` runs inside one atomic chunk and resolves dominance over the voxels it can see.
A supervoxel spanning chunks therefore gets several *partial* records, each independently
dominance-resolved, so the same supervoxel is `PROPER(275)` in one chunk and `PROPER(319)` in
another. Both are locally correct, both authorise merges, and the contradiction only appears when
the records are reduced -- by which time the merges are committed and all `nuc_join` can do is
stamp the result `CONFLICT`. That is the measured bug: two locally-legal merges that are globally
contradictory.

The fix keeps the counts local and additive and moves only the **dominance resolution** to global
scope. A pre-pass accumulates the complete per-supervoxel nucleus histogram over the whole volume
and resolves each supervoxel's state once; `NucExtractor` then reports that state instead of
deriving its own. Every downstream stage -- `nuc_join`, `nuc_can_merge`, the reducers, `load_nuc`
-- is untouched, and the wire format does not change.

Task-owner decision on `plan_v1_review` question 3: this direction is approved, replacing the
merge-ordering change the task originally mandated.

## Scope

In scope: the global nucleus table, its plumbing, and the verification the two prior reviews asked
for. Out of scope: merge ordering (refuted, plan v1), the `nuc_can_merge` clause-2 change
(refuted, this plan's evidence), the must-link snap, post-hoc splitting, whole-volume re-run.

## Proposed Changes

### 0. Why this is the right target (evidence)

| arm | nucleus mask | segment `72198606672811349` | notes |
|---|---|---|---|
| `nonuc` | none | **640,562,148** vox | the three somata are *inside* it |
| `worst3` | yes | **291,235,662** vox, CONFLICT, 28,576,326 tagged | somata split out at 109M/130M/110M |
| `w2ctl` | yes, on `312bf54` | identical to `worst3` on every number | control reproduces |
| `mint5000` | yes, `MIN_TAGGED=5000` | 320,483,315 vox, **28,566,865** tagged | PROPER records 200 -> 11, tagged mass unchanged |

Two things follow. The constraint already does real work -- it halves the blob and pulls all three
somata out of it. And the residual contamination is **invariant to the extraction threshold**
(`mint5000`), so it is not marginally-tagged supervoxels; it is structural.

The structure is visible in the per-chunk ongoing records. Segment `72198606672811349` exists as
one segment at the **atomic** level, in 70+ atomic chunks, with locally-correct records:

```
0_0_2_0  PROPER 275   337,887      0_1_2_2  PROPER 319  1,681,241
0_1_2_1  PROPER 275 2,963,635      0_1_3_4  PROPER 373    539,187
0_1_3_1  PROPER 275 2,285,810      0_3_3_4  PROPER 373  1,898,663
0_1_1_1  CONFLICT     205,541      ... 70+ chunks, three different ids
```

No merge was ever vetoed (`merge propagation violated`: 0 occurrences) because in each chunk
individually there was nothing to veto.

### 1. Global nucleus table

New stage producing one file: for every supervoxel that touches the mask, the complete histogram
resolved once.

* **Accumulate** -- per chunk of the existing grid, read ws + nucleus mask through the existing
  `open_volume` / `cut_nucleus_data` path and emit a partial histogram
  `(supervoxel, nucleus_id, count)`. This is the same counting `NucExtractor::collectVoxel` already
  does; the new part is that it is written out rather than resolved in place.
* **Reduce** -- sum partials by `(supervoxel, nucleus_id)`, then apply the *existing* rules to the
  **global** counts: `tagged < min_tagged` -> NONE, `nuc_is_dominant(max_count, tagged, ratio)` ->
  PROPER(max_id), else CONFLICT. Same predicates, same env knobs, different scope.
* **Emit** -- one file sorted by supervoxel id, `nuc_wire_t` records, i.e. the format
  `NucExtractor::output` already writes.

Size: whole-volume this is bounded by the supervoxels touching ~1000 nuclei; the run 1 whole-volume
root carried 17,160 PROPER segment records, and the supervoxel-level table is order 1e6 entries at
29 bytes -- tens of MB, loadable in every worker.

### 2. `NucExtractor` reads the table

When `NUC_TABLE` is set, `NucExtractor::output` keeps its local counting but takes the **state and
id** from the table:

```
record = { state:  table[sid].state,
           id:     table[sid].id,
           count:  local count of table[sid].id,     // local, so reduction stays additive
           total:  local tagged total }              // local, ditto
```

Counts stay local and additive, so `nuc_join` still sums to the correct global totals and the
reducers need no change. Only the resolution is global -- which is exactly the defect.

Emitting the *global* count/total in every chunk instead would double-count under
`nuc_join`; this is the one trap in the design and the reason counts stay local.

### 3. Configuration

`NUC_TABLE` in the param JSON, exported by `scripts/set_env.py` in the existing
`if "NUC_PATH" in data:` block. Absent -> `NucExtractor` resolves locally exactly as today ->
default path bit-identical. `print_params` echoes whether the table is in use.

## Files and Areas

| Path | Change |
|---|---|
| `work2/abiss/scripts/build_nucleus_table.py` (new) | accumulate + reduce + emit |
| `work2/abiss/scripts/set_env.py` | export `NUC_TABLE` alongside `NUC_PATH` |
| `work2/abiss/src/seg/NucExtractor.hpp` | optional table lookup replacing local resolution |
| `work2/abiss/src/seg/atomic_chunk_ME.cpp` | pass the table path through |
| `work2/abiss/work/test/` | **copy** from `work/abiss/work/test/` -- untracked, not in `312bf54` |
| `work2/abiss/work/test/run_v2_invariance.sh` | `run_start_ref` -> `312bf54`; replace the `expected_sidecars` block (at this baseline the sidecars exist in baseline too, so the assertion is "no current-only files") |
| `work2/abiss/work/test/make_hierarchy_fixture.py` | the split-supervoxel fixture in V3 |
| `dev/zebrafinch/nucleus_shell_contamination.py`, `nuc_arm_summary.py`, `nuc_z1_y7_x6/run_arm_env.sh` | done |

Untouched: live `lib/abiss` (on `main`, unrelated uncommitted `volume_backends.py` work),
`lib/abiss/build/`, `scripts/nucleus_snap.py`. Use `work2/abiss` (`312bf54`, built) -- `work/abiss`
is at the pre-rebase `8afcd87` and lacks upstream's percent-decoding of `file://` URIs, so an
AFF_PATH containing `%3D` dies on a missing `info`. A new arm also needs its
`precomputed/{ws,seg}/<arm>/info` and `seg/<arm>/size_map/info` pre-created.

## Verification Plan

**V0 -- predict before building (blocking).** Compute the global table for the `worst3` crop
offline in Python and diff it against the per-chunk resolution the run actually used
(`state/nucx/ongoing_nuclei_labels_0_*.data`). Report: how many supervoxels change state, how many
flip PROPER->CONFLICT, and specifically what the representative of `72198606672811349` resolves
to globally. This predicts the outcome to within the agglomeration's own nonlinearity, at zero
implementation cost. **If no supervoxel changes state, the mechanism is wrong and the plan stops
here.**

**V1 -- algebra and equivalence.** `nuc_join`/`nuc_can_merge` are untouched, so run 1's
`test_nuc_algebra.cpp` must pass unmodified. Add: on a single-chunk volume the table path and the
local path produce identical records (with one chunk, local resolution *is* global).

**V2 -- default-path bit-invariance.** `run_v2_invariance.sh` with both corrections above. Run
twice: `NUC_TABLE` unset, and `NUC_TABLE` set but no `NUC_PATH` (must be inert). Both must report
`identical=N differing=0 missing=0`. Non-negotiable; the pipeline reproduces a Seuron provenance
record.

**V3 -- the partition-dependence case (this is the test that matters).** Both prior reviews
flagged that a generic boundary-spanning fixture proves nothing. Build the *adversarial* one:
a supervoxel straddling the chunk boundary whose raw counts are 70% nucleus 1 / 30% nucleus 2,
arranged so chunk A sees mostly nucleus 1 and chunk B mostly nucleus 2. Assert:

* today's path yields `PROPER(1)` in A and `PROPER(2)` in B, and the reduced record is CONFLICT;
* the table path yields `PROPER(1)` in **both**, matching the monolithic result;
* child+parent equals monolithic for the whole fixture with the table on.

**V4 -- the crop A/B**, `run_arm_env.sh` with `WORKER_HOME=work2/abiss`, arms `w2ctl` (control,
already run) and `table`. Criteria fixed now, and rewritten to close the gaps
`plan_v1_review` found in v1's version:

1. **shared mask mass** (replaces the gameable segment count): total mask voxels lying in any
   segment that holds mask voxels of >=2 nuclei, **at any mass, tol=0**. Control **110,244**.
   Target: **<= 27,500** (a 75% reduction). Splitting the shared mass across five sub-tolerance
   fragments no longer passes, which was the hole in v1's criterion 1.
2. **coverage fragmentation**: segments needed to cover 95% of each nucleus's mask. Control is 2
   for 275/319/373 and 1 for the five clean nuclei. Target: **<= 2** for all eight, i.e. no
   nucleus may be shattered to buy criterion 1.
3. **dominance**: >= **0.93** for 275/319/373 (control 0.8957/0.9104/0.8988) and >= **0.999** for
   the five clean nuclei (control 1.0000/0.9998/1.0000/1.0000/1.0000). Both thresholds are stated
   consistently, which v1's criteria 2 and 4 were not.
4. **no runaway, no shatter**: largest segment <= **450,000,000** (control 291,235,662; `nonuc`
   640,562,148) **and** count of segments > 100M voxels <= **5** (control 5). The pair blocks
   v1's "split 291M into two 145M" hole.
5. **root segment count** within **0.9x-1.25x** of the control's 3,613.
6. **control reproduction**: rerunning `w2ctl` with `NUC_TABLE` unset must reproduce `worst3`'s
   `nuc_arm_summary` output exactly -- segment count, top-5 sizes, state histogram -- not merely
   the contamination JSON, which v1's criterion 5 wrongly treated as sufficient.

Report all six for control and treatment side by side. NERL is a whole-volume measure and is not
evaluated on a crop; if V4 passes, a whole-volume run and NERL is the natural follow-up and is
out of scope here.

**Determinism.** Stated precisely rather than overclaimed: the table is a pure function of
(ws, mask, `min_tagged`, ratio) and is computed once, so it is order-independent by construction,
and V1's single-chunk equivalence plus V3's child-vs-monolithic test cover it. Full
permutation-invariance of the final segmentation is **not** claimed -- equal-affinity edges are
ordered by fibonacci-heap insertion order, a pre-existing property this change does not touch.

## Risks and Questions

* **[major] The table cannot split a supervoxel.** Whatever tagged mass sits *inside* a single
  supervoxel that genuinely spans two nuclei stays there; the table can only stop that supervoxel
  from recruiting more. V0 measures how much of the 28.5M is intrinsic and therefore bounds the
  achievable improvement before any code is written. If most of it is intrinsic, criterion 1's
  75% target is unreachable and the honest answer is a smaller target plus a splitting follow-up.
* **[major] Globally-CONFLICT supervoxels become more restricted, which costs splits.** A
  supervoxel that today is locally PROPER(275) and merges freely with 275's soma will, under the
  table, be CONFLICT and merge with nothing but NONE. That is the intended behaviour and it is
  also a split risk; criteria 2, 4 and 5 price it.
* **[major] Cost and staging.** The pre-pass is a full read of ws + mask before agglomeration can
  start, and it serialises: every chunk needs the finished table. On the whole volume that is a
  real addition to the pipeline, not a free lookup. Measure it on the crop and extrapolate.
* **[minor] `nucleus_fusion_audit.py` stays blind** to minority-mass fusion;
  `nucleus_shell_contamination.py` is the metric that sees this class.
* **[minor] `task.md` records a refuted mechanism** and should be corrected after the run.
* **Question:** the table is keyed by supervoxel id, so it must be built against the *same* ws ids
  the agglomeration consumes -- i.e. after `chunkmap` is applied, not before. `nucleus_snap.py`
  broke on exactly this distinction in run 1. The coder should confirm which id space
  `cut_chunk_agg.py` sees and build the table in that space.

## Changes Since Previous Plan Version

Plan v1's fix (tightening `nuc_can_merge` clause 2) is withdrawn. `plan_v1_review`'s lead finding
is correct and I have confirmed it: CONFLICT+NONE preserves `total`, so the 28.5M tagged voxels
were present at conflict birth and clause 2 could not have created the fusion.

* **v1-F1, v1-F2, v1-F3 (clause 2 is not the cause; CONFLICT is a post-hoc symptom; the refutation
  of `task.md` was only partial) -- accepted.** Located: the fusion is complete at the atomic
  level, and `nonuc` reproduces the same segment at 640,562,148 voxels with no mask at all.
* **v1-F4 (the treatment cannot satisfy the primary criterion) -- accepted**, the change is
  withdrawn. The misplaced `conflict_absorbed` diagnostic goes with it.
* **v1-F5 (the predicate sketch broke flag-off behaviour) -- moot**, and it was correct.
* **v1-F6 (the extractor is partition-dependent) -- promoted from a review finding to the fix.**
  The reviewer's own counterexample -- one child emits PROPER-1, another PROPER-2, while the
  combined raw counts are 70/30 and monolithically PROPER-1 -- is now the mechanism this plan
  targets and the fixture V3 asserts on.
* **v1-F7 (`nonuc` is not the discriminator) -- accepted**; it was run anyway and answered
  question 3, but the reasoning is now carried by the per-chunk record trace, not by `nonuc`.
  `mint5000` is reported as what it is: a threshold sweep showing the tagged mass is invariant.
* **v1-F8 (V4 gameable) -- fixed.** Shared mask mass at tol=0 replaces the segment count;
  fragmentation and a >100M-segment count close the split-in-two hole; the two dominance
  thresholds are now consistent; control reproduction compares the full summary.
* **v1-F9 (owner approval required) -- obtained.** The task owner chose this direction over
  post-hoc splitting, accepting the measurement, and re-attempting merge ordering.
* **New:** V0 is a blocking predictive measurement, so the achievable improvement is bounded
  before any code is written.
