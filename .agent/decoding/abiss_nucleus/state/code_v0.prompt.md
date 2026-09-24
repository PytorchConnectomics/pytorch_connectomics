You are the CODER for a CCC run. Implement plan_v2 (included below in full).

You reviewed plan_v0 and plan_v1 and returned READY: no on both. Your findings drove two
withdrawals; plan_v2 is the terminal plan and is authoritative. In particular, your plan_v1_review
finding that the extractor is PARTITION-DEPENDENT (one child emits PROPER-1, another PROPER-2,
while the combined raw counts are 70/30 and monolithically PROPER-1) is no longer a review comment
-- it is the defect this change fixes.

The task owner has directed: make the instance-mask channel CORRECT first; ABISS parameter tuning
comes afterwards. So implement the correctness fix even though the measurement below shows it does
not, on its own, dissolve the reported 291M-voxel blob. That is expected and is not a reason to
deviate.

## Hard constraints

1. WORK ONLY IN THIS DIRECTORY:
       /projects/weilab/weidf/lib/pytorch_connectomics/.agent/features/abiss_nucleus/work2/abiss
   An isolated git clone of lib/abiss detached at 312bf54, already configured and built at
   work2/abiss/build.
   NEVER touch /projects/weilab/weidf/lib/pytorch_connectomics/lib/abiss - it is the live checkout,
   sits on `main`, carries unrelated uncommitted work in scripts/volume_backends.py, and SLURM jobs
   execute binaries from its build/ directory. A previous run broke a running job by rebuilding it.
   Do NOT use .agent/features/abiss_nucleus/work/abiss either: it is the PRE-REBASE commit 8afcd87
   and lacks upstream's percent-decoding of file:// URIs, so any run with this AFF_PATH dies.
2. DO NOT create git commits. Leave all changes in the working tree.
3. DO NOT add runtime or build dependencies.
4. Behaviour with no NUC_TABLE configured must be bit-identical to baseline. This repo reproduces a
   Seuron provenance record. V2 in the plan is the check; failing it is a hard failure.
5. Do not claim a command passed unless you ran it and saw the output. A step you did not run is
   reported as NOT RUN. This matters more than appearing complete.
6. Environment: `set +u; source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc; set -u`
   (the conda activate-binutils hook dereferences unset variables). Build only into
   work2/abiss/build. HDF5 on /projects needs HDF5_USE_FILE_LOCKING=FALSE.

## The one trap that has already broken this feature once

The table is keyed by supervoxel id and MUST be built in the id space the agglomeration consumes.
Measured: the ws precomputed layer's supervoxel ids run 134,233,002 .. 275,146,546,445, while the
agg segment ids in nuclei_labels_*.data are ~7.2e16 -- DIFFERENT SPACES, related by
CHUNKMAP_INPUT/CHUNKMAP_OUTPUT. A prototype of this table accumulated in the wrong one. The same
distinction is why scripts/nucleus_snap.py is known-broken and must stay disabled: chunkmap.data
records a remap ALREADY APPLIED to the watershed volume, so injecting at agglomeration time
strands the snapped-away supervoxels. Confirm which id space cut_chunk_agg.py / atomic_chunk_ME
actually see before you write the accumulator, and say in your artifact how you confirmed it.

## What to produce

Implement the plan's changes 1-3, then run its verification plan V0-V4 as far as it goes. V2
(default-path bit-invariance) and V3 (the adversarial partition-dependence fixture) are the two
that matter most; V4 needs SLURM and may be left for the coordinator if it does not fit -- say so
rather than skipping it silently.

Note on the test harness: work/test/ is UNTRACKED scratch in work/abiss and is NOT in 312bf54.
Copy it into work2/abiss/work/test/ first. run_v2_invariance.sh then needs BOTH corrections named
in the plan: retarget run_start_ref to 312bf54, AND replace the expected_sidecars block (at this
baseline the nucleus sidecars exist in the baseline build too, so the correct assertion is "no
current-only files at all"). Retargeting alone will fail.

Then write this file (exact path):

    /projects/weilab/weidf/lib/pytorch_connectomics/.agent/features/abiss_nucleus/artifacts/code_v0.md

with EXACTLY these level-2 sections, in this order, each present exactly once:

    # Code v0
    ## Overview
    ## What Changed
    ## Implementation Details
    ## Files Changed
    ## Git Baseline
    ## Verification
    ## Review Focus
    ## Risks and Unknowns
    ## Changes Since Previous Code Version

Requirements on that file:
- "## Files Changed" must contain a markdown table whose header is exactly:
    | File | Purpose |
    |---|---|
- "## Git Baseline" must contain these two lines:
    run_start_ref: 312bf54183faa94be1aff8ae246fcc211808f0bb
    current_head: <output of `git -C work2/abiss rev-parse HEAD`>
- "## Changes Since Previous Code Version" must contain exactly: Initial implementation.
- "## Verification" must state, per check, the command you ran and its actual result, or NOT RUN.

=========================== BEGIN artifacts/plan_v2.md ===========================
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
=========================== END plan_v2.md ===========================

=========================== BEGIN state/evidence_conflict.md (the measurements) ===========================
# Evidence: the contamination is carried by NUC_STATE_CONFLICT segments

Collected during run 2 planning. Feeds plan_v2 / the code stage. Not a CCC artifact.

## Whole volume (`wholevol_nuc_arm1ft`, root `5_0_0_0`)

`scratch/seg_nuc_arm1ft/agg/info/{nuclei_labels,seg_size}_5_0_0_0.data`:

```
records 1,388,052        NONE 1,370,881   PROPER 17,160   CONFLICT 11
CONFLICT segments: 11    8,670,570,144 voxels total, 220,278,373 tagged voxels total

  73184112755759756   2,046,998,065 voxels   41,446,632 tagged
  73183563402632211   2,023,809,586 voxels   23,280,271 tagged
  73184456017589004   1,163,658,886 voxels   38,262,495 tagged
  73043376072559282     640,780,578 voxels   19,806,686 tagged
  73184112957069116     565,869,123 voxels   20,067,074 tagged
  72481181429404917     555,437,916 voxels   28,578,569 tagged
largest segment overall  4,918,818,258 voxels
```

**Both segments the user reported are CONFLICT segments.** `72481181429404917` (the one holding
nuclei 275/319/373) and `73184112755759756` (holding 173/213) are entries 6 and 1 above. The
whole reported defect is carried by 11 segments out of 1.39M, and those 11 are already labelled
as such in the run's own output.

## Crop (`worst3`, BBOX `[2772,9324,1260,3780,10584,2520]`, root `3_0_0_0`)

```
records 3,613            NONE 3,412   PROPER 200   CONFLICT 1     distinct PROPER ids 8
  72198606672811349   291,235,662 voxels   28,576,326 tagged   <- largest in the crop
  72128031702968589   130,115,976 voxels   nucleus 319 soma
```

Exactly one CONFLICT segment, and it is the largest object in the crop (~18% of the crop volume,
2.2x the biggest soma). Its tagged mass matches the whole-volume record for `72481181429404917`
(28,576,326 vs 28,578,569) -- the crop reproduces the whole-volume object.

Strict any-mass contamination (`nucleus_shell_contamination.py --tol 0.01`):

```
 nucleus   mask vox  dominance  segments >= 1%
     275     369,823    0.8957  72198469032353291:0.896  72198606672811349:0.101
     319     430,028    0.9104  72128031702968589:0.910  72198606672811349:0.084
     373     379,073    0.8988  72268975416894476:0.899  72198606672811349:0.097
     286      91,366    1.0000  one segment
     293     106,204    0.9998  one segment
     325      29,855    1.0000  one segment
     337      87,873    1.0000  one segment
     377     126,178    1.0000  one segment
STRICT: segments holding >= 1% of more than one nucleus: 1
```

Five of eight nuclei are clean. Not a generic shell effect.

## Why `task.md`'s mechanism is refuted

Extraction/load counters summed over the crop:

```
nuc: conflict_sv                1     nuc: minority_sv       11
nuc: subfloor_sv           13,975     nuc: subfloor_voxels  245,923
nuc: load_conflict_collisions   4     "merge propagation violated": 0
```

`subfloor_voxels` = 245,923 mip0 voxels ~= 960 mask voxels, against 110,244 contaminated mask
voxels. The "partly-tagged supervoxel records NONE and escapes the veto" story is short by two
orders of magnitude.

Note also that `NucExtractor` computes dominance over **tagged**, not over the supervoxel's total
volume (`nuc_is_dominant(max_count, tagged, m_ratio)`, `NucExtractor.hpp:69`). A supervoxel with
60 tagged voxels of one nucleus and five million untagged voxels is PROPER. So the NONE escape
hatch is much narrower than `task.md` assumed, which is consistent with `subfloor_sv` being small.

`load_conflict_collisions 4` (`mean_aggl.cpp:389`) is where CONFLICT is born: a composite chunk
`nuc_join`s two children's records for the same segment id and they disagree.

## Tooling

* `dev/zebrafinch/nucleus_shell_contamination.py` -- strict any-mass metric (the test
  `nucleus_fusion_audit.py` structurally cannot perform; it scored this run 0 fusions).
* `dev/zebrafinch/nuc_arm_summary.py` -- root `seg_size` + `nuclei_labels` summary, seconds, no
  CloudVolume.
* `dev/zebrafinch/nuc_z1_y7_x6/run_arm_env.sh` -- `run_worst3.sh` with `WORKER_HOME` and the
  nucleus env knobs parameterised.

Gotchas found while setting the arms up:

* `work/abiss` is at `8afcd87`, the **pre-rebase** feature commit. It lacks upstream's
  percent-decoding of `file://` URIs in `volume_backends.py`, so an AFF_PATH containing `%3D`
  fails its backend predicate, falls through to CloudVolume and dies on a missing `info`. Use
  `work2/abiss` (`312bf54`, built at `work2/abiss/build`).
* A new arm needs its precomputed `ws/<arm>/info`, `seg/<arm>/info` and `seg/<arm>/size_map/info`
  pre-created (copy from `worst3`); the pipeline does not create them and fails at `ws remap`.

## V0 (plan v2's blocking gate) -- ran, and does NOT cleanly pass

`dev/zebrafinch/nucleus_global_table.py` on the `worst3` crop, 100 chunks of 252^3,
`min_tagged=50`, `dominance=0.6` -- the same rules `NucExtractor.hpp:66-77` applies:

```
supervoxels touching the mask: 168,778     global: NONE 13,166  PROPER 155,437  CONFLICT 175
  resolved inconsistently across chunks :   1,295     5,419,768 tagged voxels
  ... with two or more DIFFERENT PROPER ids:  857     4,312,085
  globally CONFLICT (cannot be split)   :     174       862,319
  total tagged voxels                   :           413,793,068
  (supervoxel 0 = background excluded: 573,524 tagged voxels over 4 nuclei)
```

The partition-dependence plan v2 targets is **real**: 857 supervoxels are resolved as
`PROPER(a)` in one chunk and `PROPER(b)` in another, each locally correct, each authorising
merges that are globally contradictory. Example: `sv 274945071254`, 479,414 voxels of nucleus 286
and 649 of 275 -- in the chunk that sees only those 649, it is `PROPER(275)`.

**But it is not the mechanism behind the reported bug.** Focusing on the three nuclei that share
the contaminating segment:

```
supervoxels mixing >=2 of [275, 319, 373]: 36    11,625,846 tagged voxels
  of which resolved CONSISTENTLY (global == per-chunk, so the table changes nothing): 14
  MINORITY mass carried into the dominant nucleus's segment: 1,976,803 voxels (7,722 mask voxels)

  sv 137506181484  tagged=2,876,134  global=PROPER(275)  per-chunk=[PROPER(275)]
      histogram: 275:2,326,210, 319:549,924
  sv 206225670932  tagged=2,307,590  global=PROPER(275)  per-chunk=[PROPER(275)]
      histogram: 275:2,182,513, 319:125,077
```

These are single watershed supervoxels straddling two nuclei with a clear dominant. Global and
per-chunk resolution **agree** -- both say `PROPER(275)`, correctly by the rule -- so the supervoxel
merges freely with 275's soma and carries 549,924 voxels of nucleus 319 in with it. A global table
changes nothing here. This is run 1's finding I1 exactly: `nuc_record_t` keeps only the dominant id
and discards minority identities.

Scale: that minority mass is 7,722 mask voxels against 110,244 contaminated ones, so straddling
supervoxels explain ~7% directly. The 857 contradictory supervoxels have small mass (4.3M) but
large *leverage* -- each can weld two somata regardless of its own size -- so their contribution is
not readable from mass alone and is not yet attributed.

## V0's other finding: the prototype used the wrong id space

```
ws precomputed layer supervoxel ids : 134,233,002 .. 275,146,546,445
agg segment ids (nuclei_labels)     : ~7.2e16
conflict-segment representative present in the ws table: False
```

The measurement above was accumulated in the **watershed layer's** id space, not the chunk-prefixed
space the agglomeration uses; `CHUNKMAP_INPUT`/`CHUNKMAP_OUTPUT` maps between them. The qualitative
findings hold (they are properties of the watershed partition, which is the same either way), but
no statement attributing mass to a particular agg segment is sound from this run.

This is the caveat recorded in plan v2's Questions, and it is the same distinction that broke
`nucleus_snap.py` in run 1: `chunkmap.data` records a remap already applied to the watershed
volume. `build_nucleus_table.py` must accumulate in the id space `cut_chunk_agg.py` consumes.
=========================== END evidence_conflict.md ===========================

=========================== BEGIN artifacts/plan_v1_review.md (your own findings, for context) ===========================
# Plan v1 Review

## Summary

Reviewer: codex (`gpt-5.6-sol`, `xhigh`), read-only sandbox. Raw transcript:
`state/plan_v1_review.review.raw.md`. Verdict upheld without softening; the lead finding is
correct and I have since confirmed it independently.

Plan v1 was right that `task.md`'s mechanism is inconsistent with `NucExtractor`, and wrong about
what replaces it. It mistook the final CONFLICT state for the cause. The reviewer's arithmetic is
decisive: once a cluster is CONFLICT the only permitted merge is with NONE, whose record has
`total == 0`, so every CONFLICT+NONE join **preserves** `total`. The final `total = 28,576,326`
was therefore already present when CONFLICT was created. Invariant D clause 2 explains growth in
untagged volume; it cannot explain how three nuclei became fused.

Post-review measurement (`state/evidence_conflict.md`) confirms the reviewer and goes further:
the offending SID is already one segment at the **atomic** level, appearing across 70+ atomic
chunks with locally-correct PROPER records -- 275 in some, 319 in others, 373 in others -- and the
`nonuc` arm reproduces the same SID at **640,562,148 voxels** with no nucleus mask at all. The
fusion is pre-existing and predates any nucleus evidence.

## Findings

* **[major] Clause 2 cannot explain the measured tagged contamination.** CONFLICT+NONE preserves
  `total`; `28,576,326 / 256` closely matches the 110-112K contaminated mask voxels, so that mass
  was present at conflict birth. Confirmed.
* **[major] CONFLICT is a post-hoc symptom, not the cause.** The `load_conflict_collisions`
  counter is not keyed by SID and does not establish where this segment became CONFLICT; zero
  "merge propagation violated" messages is expected and says nothing about which merge classes
  occurred. Confirmed -- and located: the fusion is already complete at level 0.
* **[major] The refutation of `task.md` was only partial.** The code refutes "missing the
  dominance ratio produces NONE" (it produces CONFLICT unless below `min_tagged`), and the
  subfloor count rules out below-floor voxels carrying 110K mask voxels; it does not establish
  clause 2 as the replacement cause. Correct.
* **[major] The treatment cannot satisfy V4's primary criterion.** `load_nuc`/`reduce_nuc` create
  CONFLICT without consulting `nuc_can_merge`, and by then the records share a SID; making the
  cluster inert does not split that SID. The proposed `conflict_absorbed` diagnostic is also
  misplaced -- the absorbed operand is NONE, so "tagged voxels absorbed" is always zero; it should
  record the NONE operand's `seg_size`, the conflict's size at birth, hierarchy level and SID.
* **[major] The proposed predicate does not preserve flag-off behaviour.**
  `if (conflict_a || conflict_b) return !nuc_strict_conflict();` admits CONFLICT+PROPER when the
  flag is off. The flag-off path must retain the other operand's NONE check. The plan also
  alternates between an unspecified global and plumbing a parameter to call sites.
* **[major] The reducibility argument is still incomplete.** A veto does have a better local
  argument than a priority key -- under strict mode no permitted in-loop join can create CONFLICT,
  so conflict clusters can be treated as removed from the active graph -- but that closure
  argument must be stated, not just commutativity/associativity. Across levels, the extractor is
  **partition-dependent**: one child emits PROPER-1, another PROPER-2, their reduction is
  CONFLICT, while the combined raw counts may be 70/30 and monolithically PROPER-1. A generic
  boundary-spanning fixture does not exercise this, so V3 would not establish chunk independence.
* **[major] `nonuc` is not the claimed discriminator and its branches are not objectively
  defined.** Removing all nucleus vetoes changes the whole mean-linkage trajectory; "comparable"
  and "much smaller" have no thresholds. The direct discriminator is per-SID size and tagged mass
  at conflict birth versus volume absorbed afterwards. `mint5000` also creates more NONE records,
  so it is not inherently safer.
* **[major] V4 is gameable and does not price the stated NERL risk.** Five shared fragments of
  0.0099 each pass criteria 1/2/4 while retaining 4.95% contamination; splitting 291M into two
  145M false objects passes the size ceiling with one added record; criteria 2 and 4 disagree on
  the clean-nucleus threshold; criterion 2 silently upgrades "does not regress" to 0.95.
* **[major] Not all nine v0 findings are resolved.** F1-F5 are legitimately mooted and F8 is
  fixed; F6 remains (no adversarial load-created-CONFLICT case), F7 is scoped away rather than
  tested, F9 remains through the subjective gate and gameable criteria. Because the plan abandons
  the task's mandated ordering change, implementation requires explicit task-owner approval.

## Questions

1. For the offending SID, what were its `seg_size`, nucleus `total`, and contributing child
   records at the exact collision that first created CONFLICT?
2. How much untagged `seg_size` did that SID absorb through CONFLICT+NONE after that point?
3. Is the task owner authorising replacement of the required ordering change with an earlier
   hierarchy fix or a post-agglomeration split, if the tagged fusion already exists at conflict
   birth?

Questions 1 and 2 are now answered in `state/evidence_conflict.md`: the SID is fused at the atomic
level and exists at 640,562,148 voxels with no nucleus mask at all, so the fusion is pre-existing;
the nucleus constraint already halves it to 291,235,662. Question 3 is the open decision and is
the reason this run is blocked.

## Verdict

VERDICT: NEEDS_CHANGES
=========================== END plan_v1_review.md ===========================

You have full read access to work2/abiss; read whatever you need there.
Relevant starting points: src/seg/NucExtractor.hpp, src/seg/atomic_chunk_ME.cpp,
src/seg/Types.h (nucleus algebra), src/agg/mean_aggl.cpp (load_nuc ~line 370),
scripts/set_env.py, scripts/cut_chunk_agg.py (cut_nucleus_data), scripts/volume_backends.py,
scripts/atomic_chunk_me.sh, scripts/composite_chunk_me.sh.
