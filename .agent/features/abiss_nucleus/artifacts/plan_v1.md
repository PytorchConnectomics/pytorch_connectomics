# Plan v1

## Summary

**The bug in `task.md` is misdiagnosed, and plan v0 was built on that misdiagnosis. Both are
withdrawn.** Measurement on the `worst3` crop -- which reproduces the whole-volume defect exactly
-- shows the contaminating segment is not a shell of `NUC_STATE_NONE` supervoxels losing a race.
It is a single `NUC_STATE_CONFLICT` cluster of **291,235,662 voxels**, the largest segment in the
crop, ~18% of the crop volume and 2.2x larger than the biggest soma.

Nucleus-first merge ordering cannot fix this. Ordering gives priority to `PROPER` endpoints; the
offending cluster is `CONFLICT`, and its growth is not an ordering outcome at all. That is on top
of the two independent objections in `plan_v0_review`: no strict-priority key is reducible (F1),
and `agglomerate_cc` runs per threshold-ladder rung so within-round ordering cannot arbitrate a
race between edges on different rungs (F4).

The actual defect is **Invariant D clause 2**: `nuc_can_merge(CONFLICT, NONE) == true`. Once the
mask has proved a cluster spans two or more nuclei, ABISS lets it keep absorbing unlimited
untagged material. Run 1's `plan_v3_review` named this as finding I1 and it was accepted as a
known limitation; it is now the measured cause of the bug we are asked to fix.

This plan replaces the ordering change with a one-predicate veto tightening, and puts a hard
empirical gate in front of it, because the gate decides whether a veto can fix the bug at all.

## Scope

In scope:

* the diagnostic evidence and the gate measurement that selects the fix branch;
* `nuc_can_merge` clause 2, behind an env flag, default OFF;
* the verification the previous plan review demanded and v0 did not supply;
* the crop A/B, with pass/fail numbers fixed **before** the treatment is run.

Out of scope: nucleus-first merge ordering (withdrawn, with reasons recorded above); the
must-link snap; whole-volume re-run; a strict mode for `nucleus_fusion_audit.py`.

## Proposed Changes

### 0. Evidence (already collected, `worst3` crop, BBOX `[2772,9324,1260,3780,10584,2520]`)

Strict any-mass contamination on the control output, via the new
`dev/zebrafinch/nucleus_shell_contamination.py`:

| nucleus | mask voxels | dominance | segments >=1% |
|---|---|---|---|
| 275 | 369,823 | 0.8957 | `72198469032353291`:0.896  `72198606672811349`:**0.101** |
| 319 | 430,028 | 0.9104 | `72128031702968589`:0.910  `72198606672811349`:**0.084** |
| 373 | 379,073 | 0.8988 | `72268975416894476`:0.899  `72198606672811349`:**0.097** |
| 286, 293, 325, 337, 377 | -- | **1.0000** | one segment each |

Five of the eight nuclei in the crop are perfectly clean. This is not a generic shell effect.

The shared segment's own records:

```
nuclei_labels_3_0_0_0.data   72198606672811349   state=2 (CONFLICT)  id=0  count=0  total=28,576,326
seg_size_3_0_0_0.data        72198606672811349   291,235,662 voxels     <- largest in the crop
                             72128031702968589   130,115,976            <- nucleus 319's soma
```

Extraction and load counters, summed over the crop:

```
nuc: conflict_sv               1          nuc: minority_sv    11
nuc: subfloor_sv          13,975          nuc: subfloor_voxels 245,923
nuc: load_conflict_collisions  4          "merge propagation violated": 0 occurrences
```

Two things follow directly. First, `subfloor_voxels` is 245,923 mip0 voxels ~= 960 mask voxels,
while the contamination is 110,244 mask voxels by the sampled measurement above and 111,626 by the
segment's own `total`/256 -- either way **two orders of magnitude too small**, so the
"partly-tagged supervoxel records NONE and escapes the veto" mechanism asserted in `task.md` is
refuted, not merely unproven. Second, the veto never once detected an inconsistency it had to
abort on; it is internally consistent and simply had no grounds to fire, because every merge it
saw was `CONFLICT`+`NONE`, which clause 2 permits.

`load_conflict_collisions 4` locates where CONFLICT is born: `load_nuc` (`mean_aggl.cpp:389`)
`nuc_join`s two children's records for the same segment id, and four such joins disagreed.

### 1. Gate measurement (blocking; running now)

Arm `nonuc` -- the identical crop and threshold with `NUC_PATH`, `NUC_RATIO` and `NUC_OFFSET`
removed from the param -- answers the one question that selects the fix:

* **Branch A -- the 291M blob exists in `nonuc` too.** The fusion is a pre-existing runaway merge
  in the affinity. The nucleus mask *detects* it (that is what CONFLICT means) but the merge
  predates any nucleus evidence, so tightening clause 2 stops further growth without undoing the
  existing fusion. A veto cannot fix it; splitting is required, which is post-agglomeration work
  (`nucleus_graph_split.py` / the seeded split) and a different task.
* **Branch B -- `nonuc` has no comparable blob, or a much smaller one.** Then the growth happens
  under the nucleus constraint and clause 2 is load-bearing. Proceed with change 2.

Report the number either way. Arm `w2ctl` (same param, built from `work2/abiss` at `312bf54`)
runs alongside so the control is reproduced on the run-2 binary rather than assumed; `mint5000`
(`ABISS_NUC_MIN_TAGGED=5000`) tests whether a stricter tagging floor removes the CONFLICT seed
without any code change.

### 2. Restrictive CONFLICT (branch B only)

`src/seg/Types.h`, `nuc_can_merge`, clause 2 only:

```cpp
if (conflict_a || conflict_b) return !nuc_strict_conflict();   // today: returns b/a == NONE
```

behind `agglomeration_nucleus_heuristic_t { bool strict_conflict = false; }` in
`agglomeration_param_t`, set from `AGG_NUC_STRICT_CONFLICT` through the existing
`apply_env_overrides()` (a new `env_bool`), echoed by `print_params`. Default `false` keeps the
currently-validated behaviour reachable and the default path bit-identical.

Why this shape and not the previous plan's:

* It is a change to the merge **predicate**, not to the ordering key. Reducibility is a property
  of the linkage criterion; a veto removes candidate merges without reordering the survivors, so
  the argument that failed for `(nuc_priority, w)` in `plan_v0_review` F1 does not arise. The
  predicate is a pure function of the two clusters' `nuc_record_t`s, which are `nuc_join` folds --
  commutative and associative, proved in run 1's `test_nuc_algebra.cpp`.
* `nuc_join` is untouched, so every algebraic property already tested still holds.
* It is the same shape as the semantic-mask veto, which is the style the user asked for.

Consequence to state plainly: a CONFLICT cluster becomes inert, so material that would have joined
it stays separate. That is a **split**, and splits cost NERL. The guard measurements below exist
to price it.

### 3. Diagnostics

`nucleus_shell_contamination.py` (written, used for the table above) is the strict any-mass metric
that success criterion 1 requires. Add to `NucExtractor`/`load_nuc` a `nuc: conflict_absorbed`
counter -- tagged voxels and merge count absorbed by a CONFLICT cluster -- so the mechanism is
visible in the log of any future run instead of needing this forensic reconstruction.

## Files and Areas

| Path | Change |
|---|---|
| `work2/abiss/src/seg/Types.h` | `nuc_can_merge` clause 2 behind a flag |
| `work2/abiss/src/agg/mean_aggl.cpp` | `agglomeration_nucleus_heuristic_t`, `env_bool`, `apply_env_overrides`, `print_params`, plumb the flag into `nuc_can_merge`'s call sites; `conflict_absorbed` counter |
| `work2/abiss/work/test/` | **copy** from `work/abiss/work/test/` -- untracked scratch, not in `312bf54` |
| `work2/abiss/work/test/run_v2_invariance.sh` | `run_start_ref` -> `312bf54`, **and** the sidecar assertion (see Verification) |
| `work2/abiss/work/test/test_nuc_algebra.cpp` | CONFLICT-closure cases |
| `dev/zebrafinch/nucleus_shell_contamination.py` | done |
| `dev/zebrafinch/nuc_z1_y7_x6/run_arm_env.sh` | done -- `run_worst3.sh` with `WORKER_HOME` and the nucleus env knobs parameterised |

Untouched: the live `lib/abiss` checkout (on `main`, unrelated uncommitted
`scripts/volume_backends.py` work), `lib/abiss/build/`, `scripts/nucleus_snap.py`.

Note for the coder: `work/abiss` is at `8afcd87`, the **pre-rebase** feature commit, and lacks
upstream's percent-decoding fix in `volume_backends.py`. A `file://` AFF_PATH containing `%3D`
falls through to CloudVolume there and the run dies on a missing `info`. Use `work2/abiss`
(`312bf54`, already built at `work2/abiss/build`) for every arm.

## Verification Plan

Environment: `set +u; source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc; set -u`.
Build only in `work2/abiss/build`.

**V1 -- algebra.** Extend `test_nuc_algebra.cpp` over the full `{NONE, PROPER, CONFLICT}` x
`{id_a, id_b}` product: with the flag on, `nuc_can_merge` returns false whenever either side is
CONFLICT; `nuc_join` is unchanged and still commutative and associative; with the flag off, every
existing case is bit-identical.

**V2 -- default-path bit-invariance.** `run_v2_invariance.sh` with two corrections, both of which
`plan_v0_review` F8 caught and v0 missed: retarget `run_start_ref` to `312bf54`, **and** replace
the `expected_sidecars` block. At the run-1 baseline the nucleus sidecars were current-only; at
`312bf54` they exist in baseline too, so the correct assertion is now "no current-only files at
all". Run twice: flag unset, and `AGG_NUC_STRICT_CONFLICT=1` with no `NUC_PATH` (must be inert).
Both must give `identical=N differing=0 missing=0`. Non-negotiable -- the pipeline reproduces a
Seuron provenance record.

**V3 -- distributed hierarchy.** `run_hierarchy.sh` already builds a two-chunk fixture and a
composite; extend `make_hierarchy_fixture.py` to write a nucleus mask spanning the chunk boundary,
then assert child+parent equals monolithic with the flag on. `plan_v0_review` F6 is right that v0
had no hierarchy test at all, and this is where a veto's chunk-independence must actually be shown
rather than argued. A divergence here is a blocker, not a tuning problem.

**V4 -- the crop A/B**, `run_arm_env.sh` with `WORKER_HOME=work2/abiss`:

| arm | flag | purpose |
|---|---|---|
| `w2ctl` | off | control; must reproduce `worst3` |
| `nonuc` | no `NUC_PATH` | the gate |
| `mint5000` | off, `ABISS_NUC_MIN_TAGGED=5000` | zero-code alternative |
| `strict` | `AGG_NUC_STRICT_CONFLICT=1` | treatment (branch B only) |

Pass/fail fixed now, before the treatment result is seen (`plan_v0_review` F9):

1. **primary** -- `nucleus_shell_contamination.py --tol 0.01` reports **0** segments holding >=1%
   of the mask of more than one nucleus. Control is 1.
2. **dominance** -- per-nucleus dominance for 275/319/373 >= **0.95** (control 0.8957 / 0.9104 /
   0.8988), and for the five clean nuclei >= **0.999** (control 1.0000). A treatment that fixes
   contamination by shattering a nucleus fails here.
3. **no runaway** -- largest segment <= **150,000,000** voxels (control 291,235,662; largest soma
   130,115,976).
4. **no shatter** -- total segment count at the root <= **1.25x** the control's 3,613 records, and
   the 5 clean nuclei keep dominance 1.000.
5. **control reproduction** -- `w2ctl`'s contamination JSON must equal `worst3`'s exactly. If it
   does not, stop: something other than the flag changed between `8afcd87` and `312bf54`.

**Determinism (`plan_v0_review` F7).** v0 promised more than the code can deliver. Stated
precisely: a veto is evaluated at pop time from cluster state, so with identical input the run is
reproducible, and V2/V3 cover that. Full permutation-invariance of the output is **not** claimed
and is not achievable today, because equal-affinity edges are ordered by fibonacci-heap insertion
order -- a pre-existing property of ABISS that this change neither creates nor repairs.

## Risks and Questions

* **[major] Branch A is a real possibility and would mean this task cannot be closed by a veto.**
  The gate runs before any code is written precisely so that is discovered in ten minutes rather
  than after an implementation.
* **[major] Making CONFLICT inert trades a false merge for false splits.** A CONFLICT cluster that
  is 291M voxels is mostly legitimate tissue; freezing it strands that tissue in whatever
  fragments existed when the conflict was recognised. Guards 2-4 price this, and guard 3's
  threshold is deliberately loose (a 150M ceiling still permits a soma-sized object).
* **[major] The conflict is recognised late.** `load_conflict_collisions` fires at composite level,
  i.e. after the children have already merged. Tightening clause 2 stops growth *from that point*.
  If most of the 291M was accumulated before recognition, the fix moves the number very little --
  the same failure mode as branch A, arriving through a different door. The `conflict_absorbed`
  counter in change 3 is what makes this measurable rather than inferred.
* **[minor] `nucleus_fusion_audit.py` remains blind** to minority-mass fusion and will keep
  reporting 0 on runs that have this bug. Every past "no nucleus fusion" claim made with it is a
  claim about dominant segments only.
* **[minor] `task.md` now contains a refuted mechanism.** It should be corrected rather than left
  as the record, but not during a CCC run.
* **Question:** the user's literal request was to re-order merging. This plan does not, because
  the segment at fault is CONFLICT rather than NONE and ordering has no purchase on it. If the
  intent was specifically "the nucleus should claim its surroundings before anything else does",
  that is a different and larger change than either plan, and worth raising before more is built.

## Changes Since Previous Plan Version

Every finding in `plan_v0_review` is addressed; five of them by withdrawing the change they were
about.

* **F1 (reducibility counterexample) -- accepted, and generalised.** The counterexample is not
  specific to the construction: *any* strict-priority key can raise a merged pair's key above the
  max of its constituents', so no nucleus-first ordering is reducible. The ordering change is
  withdrawn rather than patched.
* **F4 (per-rung, not global, ordering) -- accepted, and decisive.** `agglomerate_cc` is invoked
  once per ladder rung (0.9 down to 0.3 by 0.1 on this crop), so within-round ordering cannot
  arbitrate a race between edges on different rungs. Withdrawn with F1.
* **F2 (deferred set never consumed), F3 (inexact liveness test) -- moot**, both were bookkeeping
  in the withdrawn mechanism. Both were correct.
* **F5 (frozen-boundary claim unsupported) -- accepted and retracted.** v0's claim that phase
  ordering changes "only the interior order" was unsupported; no equivalent claim is made here.
* **F6 (no hierarchy verification) -- fixed.** V3 is a real monolithic-vs-child+parent comparison
  on `run_hierarchy.sh`, with a nucleus spanning the chunk boundary.
* **F7 (determinism criterion overclaimed) -- fixed by narrowing honestly**, with the reason
  (pre-existing tie-breaking by heap insertion order) stated rather than the claim quietly dropped.
* **F8 (`run_v2_invariance.sh` sidecar assertion) -- fixed.** The `expected_sidecars` block, not
  just `run_start_ref`, has to change for the new baseline.
* **F9 (gates not objectively specified) -- fixed.** Five numeric criteria, fixed before the
  treatment is run, using the control numbers now measured rather than the ~0.90 guess.
* **Reviewer questions 1 and 2** are answered by the withdrawal; **question 3** is answered by the
  numbers in V4.
* **New, from measurement:** the mechanism in `task.md` is refuted; the cause is a 291M-voxel
  CONFLICT cluster and Invariant D clause 2; a gate measurement now precedes any code.
