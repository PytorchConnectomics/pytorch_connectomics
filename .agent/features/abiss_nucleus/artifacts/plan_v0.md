# Plan v0

## Summary

Give a nucleus first claim on its own perinuclear shell by splitting each agglomeration round
into two phases inside `agglomerate_cc`: **phase 1** processes only edges with at least one
`NUC_STATE_PROPER` endpoint, in the normal descending-affinity order; **phase 2** processes the
edges phase 1 deferred, also in descending-affinity order. The edge *set* per round is unchanged
-- this is a reordering, not an admission change -- and the whole mechanism is behind an
env-configurable flag that defaults OFF, so a run without it is bit-identical to `312bf54`.

The bug being fixed: a supervoxel straddling the nuclear envelope is only partly tagged, misses
`ABISS_NUC_DOMINANCE`/`ABISS_NUC_MIN_TAGGED`, and records `NUC_STATE_NONE`. `nuc_can_merge` lets
NONE merge with anything, so the shells of nuclei 275/319/373 merge with *each other* before
either nucleus claims them, producing one segment holding mask voxels from three nuclei. Under
nucleus-first ordering the interior PROPER cluster reaches those shell supervoxels first; once
absorbed they carry a nucleus id, and the veto then blocks the shell-to-shell merge that is
currently unopposed.

Two things this plan states up front rather than discovering later:

* **This is a heuristic, and its effectiveness is an empirical question, not a proof.** It works
  only if a shell supervoxel's affinity to its own nucleus interior is above the round threshold.
  If the nuclear envelope suppresses that affinity below threshold, the nucleus never reaches the
  shell in phase 1 and the fix moves nothing. The verification plan is built to answer that
  question with a measurement *before* the full crop re-run, and to fail informatively.
* **The complementary lever is the extraction threshold, not the merge order.** Tagging a
  straddling supervoxel PROPER at a lower dominance bar removes the NONE escape hatch directly.
  The task asks for the ordering fix, so that is what this plan builds; the extraction lever is
  recorded in Risks as the fallback if the measurement says ordering cannot reach the shell.

## Scope

In scope:

* `src/agg/mean_aggl.cpp`: two-phase merge ordering, the deferral/promotion bookkeeping, the new
  param and its env override.
* `src/seg/Types.h`: one `nuc_priority()` helper so agg and the tests share a single definition.
* Test harness under `work/test/` (must be *copied into* `work2/abiss`, see Files and Areas --
  `task.md` is wrong that it is carried in the branch; it is untracked scratch in `work/abiss`).
* A strict any-mass contamination metric, because the metric that missed this bug is the one
  currently in the repo.

Out of scope (per `task.md`): must-link snap, whole-volume re-run, adding a strict mode to
`nucleus_fusion_audit.py`.

## Proposed Changes

### 1. `nuc_priority` (`src/seg/Types.h`)

```cpp
inline bool nuc_priority(const nuc_record_t & r) { return r.state == NUC_STATE_PROPER; }
```

`NUC_STATE_CONFLICT` is deliberately *not* phase 1: a conflicted cluster has no single nucleus
identity to defend, so giving it first claim would let it capture shell that a PROPER neighbour
should win.

### 2. Param and env override (`src/agg/mean_aggl.cpp`)

```cpp
struct agglomeration_nucleus_heuristic_t { bool merge_order_nucleus_first = false; };
```

added to `agglomeration_param_t`, plus an `env_bool()` alongside the existing `env_aff`/
`env_double`/`env_size`, wired in `apply_env_overrides()` as `AGG_NUC_MERGE_ORDER` and echoed by
`print_params()` (the existing contract: a segmentation's parameters are recoverable from its
log). Default `false`.

The effective switch inside `agglomerate_cc` is

```cpp
const bool nuc_first = !nuc_ids.empty() && params.nuc_params.merge_order_nucleus_first;
```

so the feature is off both when unconfigured and when there is no nucleus source at all.

### 3. Phase 1 admission (`populate_heap`)

`populate_heap` gains two optional out/in parameters: a `const std::vector<nuc_record_t> *`
(null = no filtering) and a `std::vector<edge_t<T>*> * deferred`. When filtering is active, an
edge that passes `comp(e.w, threshold)` but has `!nuc_priority(nuc[v0]) && !nuc_priority(nuc[v1])`
is **not** emplaced -- it is recorded in `deferred` (in `rg_vector` order) and its `incident`
wrappers get the default (invalid) handle, exactly the representation already used for
below-threshold edges. `incident` itself is still populated for every edge, unchanged.

When the pointer is null the function is byte-for-byte today's behaviour.

### 4. Two-phase loop (`agglomerate_cc`)

Wrap the existing `while` loop:

```cpp
for (int phase = nuc_first ? 0 : 1; phase <= 1; ++phase) {
    if (phase == 1 && nuc_first) admit_deferred(...);
    while (!heap.empty() && comp(heap.top().edge->w, target_threshold)) { /* body unchanged */ }
}
```

With `nuc_first == false` the outer loop runs exactly once, `admit_deferred` is never called, and
`populate_heap` was called with a null filter -- the merge order is today's, instruction for
instruction on the hot path.

`admit_deferred` walks the recorded `deferred` vector in order and emplaces each edge that is
still eligible. Eligibility is three exact conditions, not a heuristic:

* `comp(e->w, target_threshold)` -- rules out edges killed with `Limits::min()`;
* `e->v0 != e->v1` -- rules out self-loops created by merging;
* `incident[e->v0].at(e->v1).edge == e` -- rules out an edge that lost its canonical slot to a
  parallel edge during the `plus` merge, and confirms both wrappers still exist.

On emplace, the handle is written back into **both** `incident[e->v0].at(e->v1)` and
`incident[e->v1].at(e->v0)`; the existing code relies on those two staying in sync.

Restricting phase 2 to the recorded deferred set (rather than rescanning `[startpos, endpos)`)
is what makes this a pure reordering. A rescan would also pick up edges whose weight *rose* above
threshold mid-round via `plus` -- edges today's code leaves out of the round entirely -- which
would be a second, unrequested behaviour change hiding inside this one.

### 5. Promotion

A deferred NONE-NONE edge must re-enter phase 1 the moment one of its endpoints joins a nucleus;
otherwise phase 1 only ever absorbs the first ring of supervoxels and stops.

After the merge bookkeeping completes (i.e. after `incident[v0].clear()`, at which point `v1` is
the surviving representative and all re-pointed edges are already in `incident[v1]`), when
`phase == 0 && nuc_priority(nuc_ids[v1])`, sweep `incident[v1]` and emplace every entry that
satisfies the same three eligibility conditions **and** is in the deferred set. Membership is
tested against a `SetContainer<edge_t<T>*>` built alongside the deferred vector.

Correctness rests on the monotonicity argument in the next section: a cluster's
`nuc_priority` never goes true -> false, so an edge never needs demoting, and each edge is
promoted at most once.

Cost: the sweep is `O(deg(v1))` per merge into a PROPER cluster, against the existing
`O(deg(v0))` loop over the *disappearing* node -- and `v1` is usually the larger side, so this is
not automatically the same order. Start with the simple unconditional sweep; the Verification
Plan measures it, and Risks records the two-part optimisation to fall back on.

### 6. Strict contamination metric

`dev/zebrafinch/nucleus_shell_contamination.py`: for each nucleus id in the mask, the full
distribution of output segments over its mask voxels (not just the dominant one), reported as a
per-segment table plus the headline "segments holding above-tolerance mask mass from >1 nucleus".
This is the test `nucleus_fusion_audit.py` structurally cannot perform, and it is success
criterion 1.

## Files and Areas

| Path | Change |
|---|---|
| `work2/abiss/src/seg/Types.h` | add `nuc_priority()` |
| `work2/abiss/src/agg/mean_aggl.cpp` | `agglomeration_nucleus_heuristic_t`, `env_bool`, `apply_env_overrides`, `print_params`, `populate_heap` filter + deferral, two-phase loop, `admit_deferred`, promotion sweep |
| `work2/abiss/work/test/` | **copy** from `work/abiss/work/test/` first -- it is untracked scratch, not in `312bf54` |
| `work2/abiss/work/test/run_v2_invariance.sh` | retarget `run_start_ref` `3c4f5621` -> `312bf54` |
| `work2/abiss/work/test/test_nuc_algebra.cpp` | add the monotonicity cases |
| `work2/abiss/work/test/run_phase_order.sh` (new) | determinism fixture |
| `dev/zebrafinch/nucleus_shell_contamination.py` (new) | strict any-mass metric |
| `dev/zebrafinch/nuc_z1_y7_x6/run_worst3.sh` | parameterise `WORKER_HOME`/`AGG_NUC_MERGE_ORDER` for the A/B arms |

Untouched, and must stay untouched: the live `lib/abiss` checkout (on `main`, carrying unrelated
uncommitted `scripts/volume_backends.py` work), `lib/abiss/build/` (SLURM jobs execute those
binaries), and `scripts/nucleus_snap.py` (known-broken; `chunkmap.data` records a remap already
applied to the watershed volume, so agg-time injection strands the snapped-away supervoxels).

## Verification Plan

Build in `work2/abiss/build` only. Environment:
`source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc` (wrap in `set +u` / `set -u`;
the conda `activate-binutils` hook dereferences unset variables).

**V0 -- feasibility measurement, run before the crop A/B.** On the existing `worst3` control
output, for shell supervoxels of nuclei 275/319/373 (mask voxels within ~160nm of the surface),
compare the affinity distribution of shell-to-own-interior edges against shell-to-other-shell
edges, both against the 0.25 round threshold. If own-interior affinity is mostly below threshold,
nucleus-first ordering cannot reach the shell and the plan's premise is refuted -- report that
rather than shipping a no-op.

**V1 -- algebra.** Extend `test_nuc_algebra.cpp`: over the full `{NONE, PROPER, CONFLICT}` x
`{id_a, id_b}` product, assert `nuc_can_merge(a,b) => nuc_priority(nuc_join(a,b)) >=
max(nuc_priority(a), nuc_priority(b))`, i.e. `nuc_priority` is monotone non-decreasing across
every join the veto permits. This is the property the phase implementation depends on.

**V2 -- default-path bit-invariance.** `work/test/run_v2_invariance.sh` with `run_start_ref`
retargeted to `312bf54`, run twice: once with `AGG_NUC_MERGE_ORDER` unset, once with
`AGG_NUC_MERGE_ORDER=1` but no `NUC_PATH` (the flag must be inert without a nucleus source).
Both must report `identical=N differing=0 missing=0`. Non-negotiable: the pipeline reproduces a
Seuron provenance record.

**V3 -- phase-order determinism.** Using the `make_fixture.py` / `run_atomic_fixture.sh` harness
with a nucleus source and `AGG_NUC_MERGE_ORDER=1`: two runs on identical input must produce
identical `remap.data`. A third run with `input_rg.data` permuted must produce the same
*partition of merges into phase 1 vs phase 2* (asserted from the phase-tagged log). Full
output identity under permutation is **not** asserted, because affinity ties are broken by heap
insertion order today; claiming otherwise would be claiming a property the code does not have.

**V4 -- the actual fix, on the crop.** `dev/zebrafinch/nuc_z1_y7_x6/run_worst3.sh` already targets
BBOX `[2772,9324,1260,3780,10584,2520]`, the crop holding 275/319/373, with `NUC_PATH` and
`NUC_RATIO [4,8,8]`. Two arms, `WORKER_HOME` pointed at `work2/abiss`:

* control -- `AGG_NUC_MERGE_ORDER=0` (must reproduce the existing `worst3` output; if it does not,
  stop, because something other than the flag changed);
* treatment -- `AGG_NUC_MERGE_ORDER=1`.

Then `nucleus_shell_contamination.py` on both:

1. **primary** -- treatment has no segment holding above-tolerance mask mass from more than one
   nucleus (criterion 1);
2. **guard** -- per-nucleus dominance does not regress below the control's ~0.90 (criterion 2);
3. **guard** -- total segment count and the size distribution do not collapse, which would mean
   phase 1 turned each nucleus into a runaway blob;
4. **cost** -- agg wall-clock, treatment vs control, to size the promotion sweep.

Report the treatment/control numbers side by side. A treatment that fixes contamination by
over-merging is not a fix.

## Risks and Questions

* **[major] The premise may not hold.** If the nuclear envelope drives shell-to-interior affinity
  below threshold, phase 1 never reaches the shell. V0 tests this first. Fallback, in order of
  preference: (a) lower the extraction bar so straddling supervoxels record PROPER
  (`ABISS_NUC_DOMINANCE` / `ABISS_NUC_MIN_TAGGED` are already env-tunable, so this is a parameter
  sweep, not a code change); (b) a phase-1-only threshold relaxation, which is a real semantic
  change and should be its own task.
* **[major] Reordering is not relabeling.** ABISS's size and twig heuristics read `seg_size` at
  merge time, so a different order genuinely rejects a different set of edges. The treatment arm
  is a different segmentation, not a permutation of the control -- hence guard checks 2 and 3.
* **[major] Reducibility.** The linkage key becomes lexicographic `(nuc_priority(A) ||
  nuc_priority(B), w)`. `nuc_priority` is a function of the cluster's `nuc_record_t`, which is a
  `nuc_join`-fold over its members; `nuc_join` is commutative and associative (proved in run 1's
  `test_nuc_algebra.cpp`), so the key is a pure function of the two clusters and the edge -- it
  does not depend on merge history. Monotonicity holds because `nuc_can_merge` forbids
  PROPER+PROPER(different id) and PROPER+CONFLICT, so the only joins a PROPER cluster can make are
  with NONE or same-id PROPER, both yielding PROPER: once PROPER, always PROPER (V1 asserts this).
  The frozen guard sits *above* both the nucleus veto and phase assignment, so phase ordering
  cannot change which segments are frozen and handed to the parent chunk -- only the interior
  order. The parent recomputes phases from the same `nuc_join` folds in `ongoing_nuc.data`. The
  guarantee is therefore preserved to exactly the degree it holds today, which is not absolute:
  the `#ifdef EXTRA` frozen heuristics already break exact chunk-independence, and this change
  neither repairs nor worsens that.
* **[major] Promotion sweep cost.** `O(deg(survivor))` per merge into a PROPER cluster. If V4
  shows agg wall-clock regressing more than ~20%, replace the unconditional sweep with: (a)
  handle re-pointed edges inline in the existing `incident[v0]` loop, which is already
  `O(deg(v0))`, plus (b) one full `incident[v1]` sweep at the single NONE->PROPER transition,
  which monotonicity bounds to once per lineage.
* **[minor] `nucleus_fusion_audit.py` stays blind.** It tests dominant-segment sharing and will
  keep reporting 0 fusions on runs that have this bug. Out of scope here, but every future claim
  of "no nucleus fusion" made with that script is only a claim about dominant segments.
* **[minor] Deferred-edge memory.** One pointer plus one hash-set slot per deferred edge, bounded
  by the round's above-threshold edge count. Negligible against `rg_vector`, but it scales with
  the largest round.
* **Question for the reviewer:** phase 1 currently runs to exhaustion within a round before phase
  2 begins. The alternative -- interleaving at each threshold step of the outer ladder -- would
  bound how far a nucleus can grow before the rest of the graph moves at all. Exhaustion is
  simpler and matches "merge around nuc mask first until they branch out"; flag it if you read the
  requirement differently.

## Changes Since Previous Plan Version

Initial plan.
