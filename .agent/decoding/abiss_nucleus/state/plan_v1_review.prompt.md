You are reviewing plan v1 for a CCC run. You are the CODER: review it for executability.

You reviewed plan v0 and returned READY: no with nine major findings. Plan v1 responds by
WITHDRAWING the merge-ordering change entirely, on the strength of your F1 and F4 plus a new
measurement, and replacing it with a one-predicate veto change gated on an experiment.

RULES:
- Do not edit files.
- Review only the artifacts and text included in this prompt. Do not inspect other repository files.
- Tag each finding as [minor] or [major].
- Ambiguous severity is major.
- End your response with a line of the form: READY: yes   or   READY: no

Judge plan v1 on:
1. Is the new DIAGNOSIS sound? The plan claims the contaminating segment is a 291M-voxel
   NUC_STATE_CONFLICT cluster and that Invariant D clause 2 (nuc_can_merge(CONFLICT, NONE)==true)
   is what let it grow. Check that against the code and counters included below. Is there an
   alternative reading of the same evidence the plan has not considered?
2. Is the proposed fix executable, and is the claim that a VETO (unlike an ordering key) preserves
   reducibility actually true? You refuted the ordering version's reducibility argument; apply the
   same scrutiny here rather than accepting the distinction.
3. Is the gate (arm `nonuc`) the right discriminator, and are branch A / branch B correctly drawn?
4. Are the five numeric pass/fail criteria in V4 the right ones, and are any of them gameable?
5. Did plan v1 genuinely address all nine of your v0 findings, or quietly drop any?
6. NOTE: task.md below states a mechanism that plan v1 argues is REFUTED. Judge which is right
   from the evidence, and say so if you think plan v1's refutation is wrong.

=========================== BEGIN task.md ===========================
# Task

Fix the residual **perinuclear-shell fusion** in the `NUC_PATH` nucleus-constraint feature, by
changing agglomeration merge ORDER so that a nucleus claims its own shell before the shells of
neighbouring nuclei can merge with each other.

The literal user request:

> "one error case to fix is that abiss result doesn't preserve nuc seg. yl_cb_80nm_neuron.h5
> (seg 275) and abiss result seg 72481181429404917 contaminate it"
>
> "another error case to fix is to re-order the merging order. in general by affinity strength,
> but with nuc mask, make sure to merge around nuc mask first until they branch out. currently
> abiss result seg 72481181429404917 merges the shell of nuclei 275, 319, 373"

These are ONE bug with one fix. Do not treat them separately.

## Repository and baseline

`lib/abiss`, worked in the isolated clone `work2/abiss`, detached at
**312bf54** = `feature/nucleus-mask` (the pushed nucleus feature, rebased onto upstream `main`
`eec6d04` which added `apply_env_overrides()` for param-JSON-configurable heuristics).

The live `lib/abiss` checkout must NOT be touched: it is on `main` and carries unrelated
uncommitted work in `scripts/volume_backends.py`. Do not build into `lib/abiss/build/`.

## The measured bug

Whole-volume `wholevol_nuc_arm1ft` (arm1_ft affinity + the 80nm nucleus mask). Per-nucleus
segment coverage over the nucleus MASK voxels:

| nucleus | dominant segment | share | second segment | share |
|---|---|---|---|---|
| 275 | 72481112575749903 | 0.897 | **72481181429404917** | 0.102 |
| 319 | 72481112709945093 | 0.911 | **72481181429404917** | 0.085 |
| 373 | 72481181496677052 | 0.900 | **72481181429404917** | 0.097 |
| 173 | 72973350053406672 | 0.886 | **73184112755759756** | 0.111 |
| 213 | 72973281401214988 | 0.642 | **73184112755759756** | 0.357 |

One segment holds mask voxels from THREE different nuclei (28.4M voxels: 9.7M/9.3M/9.4M), and
another holds two (41.9M). These are the reference run's fused segment ids surviving as a
shared shell.

**It is a shell effect.** Fraction of nucleus 275's mask inside the contaminating segment, by
distance from the nucleus surface:

    80nm  19.88%   |  240nm  9.68%  |  400nm  5.74%
    160nm 13.47%   |  320nm  7.46%  |  560nm  5.31%

Monotonic decay with depth.

**Mechanism.** A supervoxel straddling the nucleus surface is only partly tagged. Its tagged
fraction misses `ABISS_NUC_DOMINANCE` (0.6) or `ABISS_NUC_MIN_TAGGED` (50), so `NucExtractor`
records it `NUC_STATE_NONE`. `nuc_can_merge` lets NONE merge with anything, so the boundary
supervoxels of 275, 319 and 373 merge with EACH OTHER into one shared shell segment. Interior
supervoxels are fully tagged, become PROPER, and are protected -- hence the gradient.

**Why the existing audit reported 0 fusions.** `nucleus_fusion_audit.py` asks "do two nuclei
share their DOMINANT segment". Each nucleus here has a distinct dominant, so it scores zero. The
audit is structurally blind to a segment holding MINORITY mass from several nuclei. This is
exactly finding I1 from the previous run's `plan_v3_review` (see
`run1_nucleus_mask/artifacts/plan_v3_review.md`), which the user accepted as a known limitation
of Invariant D: "Minority identities are not tracked at all, at any contamination level."

## Required fix

Nucleus-first merge ordering in `src/agg/mean_aggl.cpp` `agglomerate_cc`:

1. **Phase 1** -- process only edges with at least one `NUC_STATE_PROPER` endpoint, in descending
   affinity, deferring all others. Each nucleus absorbs its shell outward until no qualifying
   edge remains above threshold ("until they branch out").
2. **Phase 2** -- the deferred edges, in normal affinity order.

## Hard constraints

* **Default-path bit-invariance.** With no `NUC_PATH`, phase 1 is empty and the merge order must
  be EXACTLY today's. `work/test/run_v2_invariance.sh` (carried over in the branch) is the gate.
  This is non-negotiable: the pipeline reproduces a Seuron provenance record.
* **Distributed correctness.** ABISS's chunk-independence guarantee needs the linkage criterion
  monotone (Ran's reducibility condition). A lexicographic key `(has_proper_id, affinity)` is
  only safe because a cluster's PROPER state is monotonically non-decreasing under `nuc_join` --
  once tagged, always tagged. That argument must be stated explicitly and tested, not assumed.
  If it does not hold, say so rather than shipping it.
* Prefer making the phase policy a param via upstream's `apply_env_overrides()` rather than a
  recompile, and default it OFF so the currently-validated behaviour is reachable.
* Do not create git commits. Do not add dependencies. Do not enable `nucleus_snap.py` (it is
  known-broken; `chunkmap.data` is the record of a remap already applied to the watershed
  volume, so agg-time injection strands the snapped-away supervoxels).

## Success criteria

1. On a crop containing nuclei 275/319/373, NO single segment holds mask voxels from more than
   one nucleus above a small tolerance -- i.e. the fix must be verified with the STRICT
   (any-mass) test, not the dominant-segment test that missed this bug.
2. Per-nucleus dominance does not regress (currently ~0.90 on that crop).
3. `run_v2_invariance.sh` passes: no-`NUC_PATH` output byte-identical.
4. The reducibility argument is stated, and a test demonstrates phase ordering is deterministic
   and independent of input edge order.

## Out of scope

The must-link snap; whole-volume re-run (a crop is enough to demonstrate); `nucleus_fusion_audit.py`
gaining a strict mode (useful, but note it in Risks rather than doing it here).
=========================== END task.md ===========================

=========================== BEGIN artifacts/plan_v0_review.md ===========================
# Plan v0 Review

## Summary

Reviewer: codex (`gpt-5.6-sol`, `xhigh`), read-only sandbox. Raw transcript:
`state/plan_v0_review.review.raw.md`.

Not executable safely as written. Nine findings, all tagged major by the reviewer, none softened
here. Two are structural rather than fixable-in-place: the reducibility argument has a concrete
counterexample, and the proposed ordering operates within a single threshold-ladder rung while the
bug's competing edges can sit on different rungs. The rest are bookkeeping, harness and
gate-specification defects that would each have produced a silently wrong result.

## Findings

* **[major] The reducibility argument is false.** With `A` PROPER, `B`/`C` NONE, equal-area links
  `w(A,C)=0.2` and `w(B,C)=0.9`: before merging, the lexicographic maximum is `(true, 0.2)`; after
  the permitted `A+B` merge, `A u B` is PROPER with mean linkage `0.55`, giving `(true, 0.55)` --
  greater than both prior keys. PROPER monotonicity therefore does not establish Ran reducibility.
  The distributed-correctness constraint in `task.md` is unmet.
* **[major] Deferred membership is never consumed.** After promotion an edge stays in the deferred
  set, so a later sweep of the same PROPER survivor can emplace it a second time; the wrappers keep
  only the newest handle and the orphaned heap entry then has a key that `plus` mutates without an
  `update()`, corrupting fibonacci-heap ordering.
* **[major] The proposed liveness test is not exact.** `incident[e->v0].at(e->v1).edge == e`
  neither confirms the reverse wrapper nor that both agree on edge and handle, and `at()` throws
  when the forward slot is absent.
* **[major] The plan implements `(threshold bucket, phase, affinity)`, not the claimed global
  `(has_proper_id, affinity)`.** `agglomerate_cc` is invoked separately at every ladder step, so a
  shell-shell edge eligible at 0.8 merges in phase 2 before a shell-to-own-nucleus edge becomes
  eligible at 0.7. V0's comparison against the final 0.25 threshold cannot detect this.
* **[major] The frozen-boundary claim is unsupported and contradicted by the supplied code.** When
  an edge touches a frozen endpoint both endpoints are marked frozen, so reordering earlier merges
  can change which composite segment becomes frozen and is handed upward. "Only the interior order"
  is invalid.
* **[major] No distributed hierarchy verification.** V1 tests algebra and V3 a single atomic
  fixture; neither compares monolithic execution against child chunks plus parent aggregation, nor
  validates `ongoing_nuc.data` propagation.
* **[major] V3 does not meet the stated determinism criterion.** Two identical runs prove
  repeatability for one insertion order only, and it relies on a phase-tagged log no proposed
  change creates.
* **[major] Retargeting `run_v2_invariance.sh` alone makes its sidecar assertion fail.** At
  `312bf54` the nucleus sidecars exist in the baseline too, so they can no longer be the three
  expected `current_only` files.
* **[major] The empirical gates are not objectively specified.** "Small tolerance", "mostly below
  threshold", dominance "does not regress", size-distribution "collapse" and control reproduction
  have no formulas or pass/fail values; nucleus 275's control dominance is already 0.897, so a
  literal ~0.90 floor is ambiguous.

## Questions

1. Must nucleus priority dominate affinity across the complete threshold ladder, or only within
   each `agglomerate_cc` invocation?
2. Is exact chunk independence mandatory? If so the lexicographic policy needs redesign, because
   the stated reducibility premise has a counterexample.
3. What exact contamination tolerance and permitted per-nucleus dominance delta define success?

## Verdict

VERDICT: NEEDS_CHANGES
=========================== END artifacts/plan_v0_review.md ===========================

=========================== BEGIN artifacts/plan_v1.md ===========================
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
=========================== END artifacts/plan_v1.md ===========================

Sources at the run baseline 312bf54, for judging the diagnosis and the fix.

=========================== BEGIN src/seg/Types.h lines 30-140 (nucleus algebra) ===========================

using semantic_t = uint8_t;
using nuc_t = uint32_t;

enum : uint8_t {
    NUC_STATE_NONE = 0,
    NUC_STATE_PROPER = 1,
    NUC_STATE_CONFLICT = 2,
};

// Reachable records preserve:
//   PROPER => count * ratio.den >= ratio.num * total
//   NONE   => count == 0 && total == 0
//   CONFLICT => count == 0
struct __attribute__((packed)) nuc_record_t
{
    uint8_t state = NUC_STATE_NONE;
    uint32_t id = 0;
    uint64_t count = 0;
    uint64_t total = 0;
};

struct __attribute__((packed)) nuc_wire_t
{
    seg_t sid;
    uint8_t state;
    uint32_t id;
    uint64_t count;
    uint64_t total;
};

static_assert(sizeof(nuc_wire_t) == 29);
static_assert(offsetof(nuc_wire_t, sid) == 0);
static_assert(offsetof(nuc_wire_t, state) == 8);
static_assert(offsetof(nuc_wire_t, id) == 9);
static_assert(offsetof(nuc_wire_t, count) == 13);
static_assert(offsetof(nuc_wire_t, total) == 21);

struct nuc_ratio_t
{
    uint64_t num = 3;
    uint64_t den = 5;
};

inline bool nuc_is_dominant(uint64_t count, uint64_t total, const nuc_ratio_t & ratio)
{
    return static_cast<__uint128_t>(count) * ratio.den
        >= static_cast<__uint128_t>(ratio.num) * total;
}

inline uint64_t nuc_add(uint64_t a, uint64_t b)
{
    if (a > UINT64_MAX - b) {
        std::cerr << "nuc: voxel count overflow: " << a << " + " << b << std::endl;
        std::abort();
    }
    return a + b;
}

inline nuc_record_t nuc_join(const nuc_record_t & a, const nuc_record_t & b)
{
    nuc_record_t result;
    result.total = nuc_add(a.total, b.total);
    if (a.state == NUC_STATE_CONFLICT || b.state == NUC_STATE_CONFLICT) {
        result.state = NUC_STATE_CONFLICT;
    } else if (a.state == NUC_STATE_NONE) {
        result.state = b.state;
        result.id = b.id;
        result.count = b.count;
    } else if (b.state == NUC_STATE_NONE) {
        result.state = a.state;
        result.id = a.id;
        result.count = a.count;
    } else if (a.id == b.id) {
        result.state = NUC_STATE_PROPER;
        result.id = a.id;
        result.count = nuc_add(a.count, b.count);
    } else {
        result.state = NUC_STATE_CONFLICT;
    }
    return result;
}

inline bool nuc_can_merge(const nuc_record_t & a, const nuc_record_t & b)
{
    const bool conflict_a = a.state == NUC_STATE_CONFLICT;
    const bool conflict_b = b.state == NUC_STATE_CONFLICT;
    if (conflict_a && conflict_b) {
        return false; // Invariant D, clause 3.
    }
    if (conflict_a) {
        return b.state == NUC_STATE_NONE; // Invariant D, clause 2.
    }
    if (conflict_b) {
        return a.state == NUC_STATE_NONE; // Invariant D, clause 2.
    }
    if (a.state == NUC_STATE_NONE || b.state == NUC_STATE_NONE) {
        return true;
    }
    return a.id == b.id; // Invariant D, clause 1.
}

inline nuc_record_t nuc_record_from_wire(const nuc_wire_t & wire)
{
    return nuc_record_t{wire.state, wire.id, wire.count, wire.total};
}

inline nuc_wire_t make_nuc_wire(seg_t sid, const nuc_record_t & record)
{
    return nuc_wire_t{sid, record.state, record.id, record.count, record.total};
}
=========================== END ===========================

=========================== BEGIN src/seg/NucExtractor.hpp (full) ===========================
#ifndef NUC_EXTRACTOR_HPP
#define NUC_EXTRACTOR_HPP

#include "Types.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

template<typename Tseg, typename Chunk>
class NucExtractor
{
public:
    NucExtractor(const Chunk * nuc, nuc_ratio_t ratio, uint64_t min_tagged)
        : m_nuc(nuc), m_ratio(ratio), m_min_tagged(min_tagged) {}

    void collectVoxel(Coord c, Tseg segid)
    {
        if (m_nuc == nullptr) {
            return;
        }
        const nuc_t id = (*m_nuc)[c[0]][c[1]][c[2]];
        if (id != 0) {
            auto & count = m_counts[segid][id];
            count = nuc_add(count, 1);
        }
    }

    void collectBoundary(int face, Coord c, Tseg segid) {}
    void collectContactingSurface(int nv, Coord c, Tseg segid1, Tseg segid2) {}

    void output(const MapContainer<Tseg, Tseg> & chunk_map, const std::string & filename)
    {
        MapContainer<Tseg, MapContainer<nuc_t, uint64_t>> remapped_counts;
        for (const auto & [sid, counts] : m_counts) {
            const Tseg target = chunk_map.contains(sid) ? chunk_map.at(sid) : sid;
            for (const auto & [id, count] : counts) {
                auto & target_count = remapped_counts[target][id];
                target_count = nuc_add(target_count, count);
            }
        }

        uint64_t conflict_sv = 0;
        uint64_t minority_sv = 0;
        uint64_t subfloor_sv = 0;
        uint64_t subfloor_voxels = 0;
        std::vector<nuc_wire_t> output;
        output.reserve(remapped_counts.size());

        for (const auto & [sid, counts] : remapped_counts) {
            uint64_t tagged = 0;
            uint64_t max_count = 0;
            nuc_t max_id = 0;
            for (const auto & [id, count] : counts) {
                tagged = nuc_add(tagged, count);
                if (count > max_count || (count == max_count && id < max_id)) {
                    max_count = count;
                    max_id = id;
                }
            }

            nuc_record_t record;
            if (tagged < m_min_tagged) {
                subfloor_sv = nuc_add(subfloor_sv, 1);
                subfloor_voxels = nuc_add(subfloor_voxels, tagged);
            } else if (nuc_is_dominant(max_count, tagged, m_ratio)) {
                record = nuc_record_t{NUC_STATE_PROPER, max_id, max_count, tagged};
                if (tagged > max_count) {
                    minority_sv = nuc_add(minority_sv, 1);
                }
            } else {
                record = nuc_record_t{NUC_STATE_CONFLICT, 0, 0, tagged};
                conflict_sv = nuc_add(conflict_sv, 1);
            }
            output.push_back(make_nuc_wire(sid, record));
        }

        std::sort(output.begin(), output.end(),
                  [](const auto & a, const auto & b) { return a.sid < b.sid; });
        std::ofstream ofs(filename, std::ios_base::binary);
        assert(ofs.is_open());
        for (const auto & wire : output) {
            ofs.write(reinterpret_cast<const char *>(&wire), sizeof(wire));
        }
        assert(!ofs.bad());
        ofs.close();

        if (m_nuc != nullptr) {
            std::cout << "nuc: conflict_sv " << conflict_sv << std::endl;
            std::cout << "nuc: minority_sv " << minority_sv << std::endl;
            std::cout << "nuc: subfloor_sv " << subfloor_sv << std::endl;
            std::cout << "nuc: subfloor_voxels " << subfloor_voxels << std::endl;
        }
    }

private:
    // Nucleus masks are expected to be sparse, so the per-supervoxel inner maps
    // only exist for supervoxels that contain tagged voxels.
    const Chunk * m_nuc;
    nuc_ratio_t m_ratio;
    uint64_t m_min_tagged;
    MapContainer<Tseg, MapContainer<nuc_t, uint64_t>> m_counts;
};

#endif
=========================== END ===========================

=========================== BEGIN src/agg/mean_aggl.cpp lines 370-405 (load_nuc, where CONFLICT is born) ===========================
        return {};
    }

    auto nuc_array = read_array<nuc_wire_t>(nuc_filename);
    std::vector<nuc_record_t> nuc_ids(seg_indices.size());
    std::vector<bool> populated(seg_indices.size(), false);
    uint64_t conflict_collisions = 0;

    for (const auto & wire : nuc_array) {
        const auto it = std::lower_bound(seg_indices.begin(), seg_indices.end(), wire.sid);
        if (it == seg_indices.end() || wire.sid != *it) {
            std::cerr << "nuc: cannot find nucleus sid " << wire.sid << std::endl;
            std::abort();
        }
        const size_t index = std::distance(seg_indices.begin(), it);
        const auto record = nuc_record_from_wire(wire);
        if (populated[index]) {
            const bool was_conflict =
                nuc_ids[index].state == NUC_STATE_CONFLICT;
            nuc_ids[index] = nuc_join(nuc_ids[index], record);
            if (!was_conflict
                && nuc_ids[index].state == NUC_STATE_CONFLICT) {
                conflict_collisions = nuc_add(conflict_collisions, 1);
            }
        } else {
            nuc_ids[index] = record;
            populated[index] = true;
        }
    }
    std::cout << "nuc: load_conflict_collisions " << conflict_collisions << std::endl;
    return nuc_ids;
}

template <class T, class Compare = std::greater<T>, class Plus = std::plus<T>, class Limits = std::numeric_limits<T> >
void merge_edges(agglomeration_data_t<T, Compare> & agg_data, size_t offset)
{
=========================== END ===========================

=========================== BEGIN src/agg/mean_aggl.cpp lines 796-915 (agglomerate_cc merge loop: frozen guard, nuc veto, nuc propagation) ===========================
    while (!heap.empty() && comp(heap.top().edge->w, target_threshold))
    {
        num_of_edges += 1;
        auto e = heap.top();
        auto v0 = e.edge->v0;
        auto v1 = e.edge->v1;

        incident[v0].erase(v1);
        incident[v1].erase(v0);
        heap.pop();


        if (v0 != v1)
        {

            auto s = v0;
#ifdef EXTRA
            if ((is_frozen(supervoxel_counts[v0]) && is_frozen(supervoxel_counts[v1]))
                || (is_frozen(supervoxel_counts[v0]) && (frozen_neighbors(incident[v1], supervoxel_counts, v1) || (!comp(e.edge->w, h_threshold) && (!sem_counts.empty() || seg_size[v1] > size_params.small_voxel_threshold))))
                || (is_frozen(supervoxel_counts[v1]) && (frozen_neighbors(incident[v0], supervoxel_counts, v0) || (!comp(e.edge->w, h_threshold) && (!sem_counts.empty() || seg_size[v0] > size_params.small_voxel_threshold)))))
#else
            if ((is_frozen(supervoxel_counts[v0]) || is_frozen(supervoxel_counts[v1])))
#endif
            {
                supervoxel_counts[v0] |= frozen;
                supervoxel_counts[v1] |= frozen;
                output.res_rg_vector.push_back(*(e.edge));
                e.edge->w = Limits::min();
                continue;
            }

            if (!nuc_ids.empty() && !nuc_can_merge(nuc_ids[v0], nuc_ids[v1])) {
                output.nuc_rg_vector.push_back(*(e.edge));
                e.edge->w = Limits::min();
                continue;
            }

            if (!comp(e.edge->w, sem_aff_threshold)) {
                if (!sem_counts.empty()){
                    if (!sem_can_merge(sem_counts[v0],sem_counts[v1],sem_params)) {
                        output.sem_rg_vector.push_back(*(e.edge));
                        e.edge->w = Limits::min();
                        continue;
                    }
                }
            }

            if (!comp(e.edge->w, size_aff_threshold)) {
                size_t size0 = seg_size[v0];
                size_t size1 = seg_size[v1];
                auto p = std::minmax({size0, size1});
                if (p.first > size_params.small_voxel_threshold and (size0+size1) > size_params.large_voxel_threshold) {
                    output.rej_rg_vector.push_back(*(e.edge));
                    e.edge->w = Limits::min();
                    continue;
                }
            }

            if (!comp(e.edge->w, backbone_threshold)){
                size_t size0 = seg_size[v0];
                size_t size1 = seg_size[v1];
                auto p = std::minmax({size0, size1});
                if ((p.first > twig_params.voxel_threshold) or (e.edge->w.num > twig_params.area_threshold)) {
                    e.edge->w = Limits::min();
                    continue;
                }
                output.twig_rg_vector.push_back(*(e.edge));
            }
#ifdef FINAL
            if (incident[v0].size() < incident[v1].size()) {
                s = v1;
            }
#else
            if (supervoxel_counts[v0] < supervoxel_counts[v1]) {
                s = v1;
            }
#endif
            if (is_frozen(supervoxel_counts[v0])) {
                s = v0;
            } else if (is_frozen(supervoxel_counts[v1])) {
                s = v1;
            }

            //std::cout << "Join " << v0 << " and " << v1 << std::endl;
            supervoxel_counts[v0] += supervoxel_counts[v1];
            supervoxel_counts[v1] = 0;
            std::swap(supervoxel_counts[v0], supervoxel_counts[s]);

            seg_size[v0] += seg_size[v1];
            seg_size[v1] = 0;
            std::swap(seg_size[v0], seg_size[s]);

            if (!sem_counts.empty()) {
                std::transform(sem_counts[v0].begin(), sem_counts[v0].end(), sem_counts[v1].begin(), sem_counts[v0].begin(), std::plus<size_t>());
                sem_counts[v1] = sem_array_t();
                std::swap(sem_counts[v0], sem_counts[s]);
            }

            if (!nuc_ids.empty()) {
                const auto nucleus0 = nuc_ids[v0];
                const auto nucleus1 = nuc_ids[v1];
                if (!nuc_can_merge(nucleus0, nucleus1)) {
                    std::cerr << "nuc: merge propagation violated veto for "
                              << seg_indices[v0] << " (state="
                              << static_cast<unsigned>(nucleus0.state) << ", id="
                              << nucleus0.id << ", count=" << nucleus0.count << ", total="
                              << nucleus0.total << ") and " << seg_indices[v1] << " (state="
                              << static_cast<unsigned>(nucleus1.state) << ", id="
                              << nucleus1.id << ", count=" << nucleus1.count << ", total="
                              << nucleus1.total << ")" << std::endl;
                    std::abort();
                }
                nuc_ids[v0] = nuc_join(nucleus0, nucleus1);
                nuc_ids[v1] = nuc_record_t();
                std::swap(nuc_ids[v0], nuc_ids[s]);
            }

            output.merged_rg_vector.push_back(*(e.edge));
            if (v0 == s) {
                output.remap.emplace_back(v1, s);
=========================== END ===========================

=========================== BEGIN src/agg/mean_aggl.cpp lines 120-225 (params + apply_env_overrides) ===========================

struct agglomeration_size_heuristic_t
{
    aff_t aff_threshold = 0.35;
    size_t small_voxel_threshold = 1'000'000;
    size_t large_voxel_threshold = 10'000'000;
};

struct agglomeration_semantic_heuristic_t
{
    aff_t aff_threshold = 0.5;
    size_t total_signal_threshold = 100'000;
    double dominant_signal_ratio = 0.6;
};

struct agglomeration_twig_heuristic_t
{
    aff_t aff_threshold_delta = 0;
    size_t voxel_threshold = 100'000;
    size_t area_threshold = 50;
};

struct agglomeration_param_t
{
    agglomeration_size_heuristic_t size_params;
    agglomeration_semantic_heuristic_t sem_params;
    agglomeration_twig_heuristic_t twig_params;
    aff_t input_aff_threshold;
    aff_t heuristics_aff_threshold = size_params.aff_threshold > sem_params.aff_threshold ? size_params.aff_threshold : sem_params.aff_threshold;
    aff_t starting_aff_threshold = 0.9;
    aff_t agglomeration_step = 0.1;
    size_t optimal_number_of_partitions = omp_get_num_procs() ;
    size_t minimal_number_of_edges = 100'000;
};

// ---------------------------------------------------------------------------
// Environment overrides for the agglomeration heuristics.
//
// Only `input_aff_threshold` (argv[1] <- AGG_THRESHOLD) used to be reachable from the
// param JSON; every other knob above was a compile-time literal, so tuning one meant
// editing this file and rebuilding. These helpers let each be set from the environment,
// which `scripts/set_env.py` populates from the param JSON, which the YAML feeds.
//
// Every default is the literal it replaces, so a run with none of these set is
// bit-identical to the previous binary. `print_params` echoes the resolved values on
// every run, so a segmentation's parameters are recoverable from its log.
// ---------------------------------------------------------------------------

static bool env_aff(const char * name, aff_t & out)
{
    const char * v = std::getenv(name);
    if (v == nullptr || *v == '\0') return false;
    out = static_cast<aff_t>(atof(v));
    return true;
}

static bool env_double(const char * name, double & out)
{
    const char * v = std::getenv(name);
    if (v == nullptr || *v == '\0') return false;
    out = atof(v);
    return true;
}

static bool env_size(const char * name, size_t & out)
{
    const char * v = std::getenv(name);
    if (v == nullptr || *v == '\0') return false;
    out = static_cast<size_t>(strtoull(v, nullptr, 10));
    return true;
}

static void apply_env_overrides(agglomeration_param_t & p)
{
    env_aff   ("AGG_SIZE_AFF_THRESHOLD",        p.size_params.aff_threshold);
    env_size  ("AGG_SMALL_VOXEL_THRESHOLD",     p.size_params.small_voxel_threshold);
    env_size  ("AGG_LARGE_VOXEL_THRESHOLD",     p.size_params.large_voxel_threshold);

    env_aff   ("AGG_SEM_AFF_THRESHOLD",         p.sem_params.aff_threshold);
    env_size  ("AGG_SEM_TOTAL_SIGNAL_THRESHOLD",p.sem_params.total_signal_threshold);
    env_double("AGG_SEM_DOMINANT_SIGNAL_RATIO", p.sem_params.dominant_signal_ratio);

    env_aff   ("AGG_TWIG_AFF_THRESHOLD_DELTA",  p.twig_params.aff_threshold_delta);
    env_size  ("AGG_TWIG_VOXEL_THRESHOLD",      p.twig_params.voxel_threshold);
    env_size  ("AGG_TWIG_AREA_THRESHOLD",       p.twig_params.area_threshold);

    // Derived from size/sem at construction, so it must be recomputed after those may
    // have changed -- otherwise overriding size_params silently leaves it stale.
    // An explicit AGG_HEURISTICS_AFF_THRESHOLD still wins.
    p.heuristics_aff_threshold = p.size_params.aff_threshold > p.sem_params.aff_threshold
                               ? p.size_params.aff_threshold : p.sem_params.aff_threshold;
    env_aff   ("AGG_HEURISTICS_AFF_THRESHOLD",  p.heuristics_aff_threshold);

    env_aff   ("AGG_STARTING_AFF_THRESHOLD",    p.starting_aff_threshold);
    env_aff   ("AGG_STEP",                      p.agglomeration_step);
    env_size  ("AGG_MIN_EDGES",                 p.minimal_number_of_edges);

    // Defaults to omp_get_num_procs(), i.e. the segmentation depends on how many cores
    // the machine had. Setting this pins it and makes runs portable across nodes.
    env_size  ("AGG_NUM_PARTITIONS",            p.optimal_number_of_partitions);
}

static void print_params(const agglomeration_param_t & p)
{
    std::cout << "agglomeration parameters:"
              << "\n  input_aff_threshold          " << p.input_aff_threshold
=========================== END ===========================

=========================== BEGIN src/seg/reduce_chunk.cpp lines 195-245 (nucleus reducer) ===========================
void reduce_nuc(const std::string & tag, const MapContainer<T, T> & remaps,
                const SetContainer<T> & boundary_sv)
{
    const std::string input =
        str(boost::format("ongoing_nuclei_labels_%1%.data") % tag);
    std::vector<nuc_wire_t> nuc_array;
    if (filesize(input) != 0) {
        nuc_array = read_array<nuc_wire_t>(input);
    }

    MapContainer<T, nuc_record_t> reduced;
    std::vector<nuc_wire_t> boundary_nuc;
    uint64_t conflict_collisions = 0;
    for (const auto & wire : nuc_array) {
        if (boundary_sv.contains(wire.sid)) {
            boundary_nuc.push_back(wire);
            continue;
        }
        const T sid = remaps.contains(wire.sid) ? remaps.at(wire.sid) : wire.sid;
        const auto record = nuc_record_from_wire(wire);
        if (reduced.contains(sid)) {
            const bool was_conflict =
                reduced.at(sid).state == NUC_STATE_CONFLICT;
            reduced[sid] = nuc_join(reduced.at(sid), record);
            if (!was_conflict
                && reduced.at(sid).state == NUC_STATE_CONFLICT) {
                conflict_collisions = nuc_add(conflict_collisions, 1);
            }
        } else {
            reduced[sid] = record;
        }
    }

    std::vector<nuc_wire_t> output;
    output.reserve(reduced.size());
    for (const auto & [sid, record] : reduced) {
        output.push_back(make_nuc_wire(sid, record));
    }
    std::sort(output.begin(), output.end(),
              [](const auto & a, const auto & b) { return a.sid < b.sid; });
    write_vector(str(boost::format("reduced_ongoing_nuclei_labels_%1%.data") % tag),
                 output);
    write_vector(str(boost::format("reduced_boundary_nuclei_labels_%1%.data") % tag),
                 boundary_nuc);
    if (!nuc_array.empty()) {
        std::cout << "nuc: reduce_conflict_collisions "
                  << conflict_collisions << std::endl;
    }
}

template <typename T>
=========================== END ===========================

Now produce your review: a short summary, then findings each tagged [minor] or [major], then
any questions, then the final READY: line.
