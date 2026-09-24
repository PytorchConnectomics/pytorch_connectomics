You are reviewing an implementation plan for a CCC run. You are the CODER for this run: you
will be the one implementing this plan, so review it for executability.

RULES:
- Do not edit files.
- Review only the artifacts and text included in this prompt. Do not inspect other repository files.
- Tag each finding as [minor] or [major].
- Ambiguous severity is major.
- End your response with a line of the form: READY: yes   or   READY: no

Judge the plan on:
1. Is it executable as written? Are the named files, mechanisms and invariants concrete enough
   to implement without re-deriving the design? Pay particular attention to the heap/incident
   bookkeeping: the plan writes handles into two places and relies on an exact liveness test.
2. Is anything missing that would make the implementation silently wrong - especially the
   default-path bit-invariance requirement, the fibonacci-heap invariant (a key that changes
   without heap.update() silently corrupts ordering), and the distributed chunk hierarchy?
3. Is the verification plan real and sufficient, or does it assert rather than prove?
4. Is the scope right - anything included that should be cut, or excluded that must be in?
5. Is the reducibility/monotonicity argument in Risks actually sound, or does it have a hole?

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

=========================== BEGIN artifacts/plan_v0.md ===========================
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
=========================== END artifacts/plan_v0.md ===========================

The following are the CURRENT sources at the run baseline 312bf54, for judging
executability. They are provided in full so you do not need to open the repository.

=========================== BEGIN src/seg/Types.h (nucleus part, lines 30-140) ===========================

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
=========================== END src/seg/Types.h excerpt ===========================

=========================== BEGIN src/agg/mean_aggl.cpp lines 36-260 (heap types, params, env overrides) ===========================
#include "../seg/RemapTable.hpp"

#include "edges.h"

static const size_t frozen = (1UL<<(std::numeric_limits<std::size_t>::digits-2));
static const size_t boundary = (1UL<<(std::numeric_limits<std::size_t>::digits-1))|frozen;

size_t filesize(const std::string & filename)
{
    struct stat stat_buf;
    int rc = stat(filename.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : 0;
}

bool is_frozen(size_t size) {
    return size & frozen;
}

void print_neighbors(auto neighbors, const auto source)
{
    std::cout << "neighbors of " << source << ":";
    for (auto & e : neighbors) {
        std::cout << e.segid(source) << " ";
    }
    std::cout << std::endl;
}

bool frozen_neighbors(const auto & neighbors, const auto & supervoxel_counts)
{
    for (auto & [k, v] : neighbors) {
        if (is_frozen(supervoxel_counts[k])) {
            return true;
        }
    }
    return false;
}

template <class T>
using region_graph = std::vector<edge_t<T>>;

template <class T, class C = std::greater<T>>
struct heapable_edge;

template <class T, class C = std::greater<T>>
struct heapable_edge_compare
{
    bool operator()(heapable_edge<T, C> const & a,
                    heapable_edge<T, C> const & b) const
    {
        C c;
        return c(b.edge->w, a.edge->w);
    }
};

template <class T, class C = std::greater<T>>
using heap_type = boost::heap::fibonacci_heap<
    heapable_edge<T, C>, boost::heap::compare<heapable_edge_compare<T, C>>>;

template <class T, class C = std::greater<T>>
using heap_handle_type = typename heap_type<T, C>::handle_type;

template <class T, class C>
struct __attribute__((packed)) heapable_edge
{
    edge_t<T> * edge;
    explicit constexpr heapable_edge(edge_t<T> * e)
        : edge(e) {};
};

template <class T, class C>
struct handle_wrapper
{
    edge_t<T> * edge;
    heap_handle_type<T, C> handle;

    explicit constexpr handle_wrapper(edge_t<T> * e, heap_handle_type<T, C> & h)
        : handle(h) {
            edge = e;
        };
    [[nodiscard]] bool valid_handle() const { return handle != heap_handle_type<T, C>(); }
    [[nodiscard]] seg_t segid(const seg_t exclude) const {
        return exclude == edge->v0 ? edge->v1 : edge->v0;
    }
};

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
              << "\n  size.aff_threshold           " << p.size_params.aff_threshold
              << "\n  size.small_voxel_threshold   " << p.size_params.small_voxel_threshold
              << "\n  size.large_voxel_threshold   " << p.size_params.large_voxel_threshold
              << "\n  sem.aff_threshold            " << p.sem_params.aff_threshold
              << "\n  sem.total_signal_threshold   " << p.sem_params.total_signal_threshold
              << "\n  sem.dominant_signal_ratio    " << p.sem_params.dominant_signal_ratio
              << "\n  twig.aff_threshold_delta     " << p.twig_params.aff_threshold_delta
              << "\n  twig.voxel_threshold         " << p.twig_params.voxel_threshold
              << "\n  twig.area_threshold          " << p.twig_params.area_threshold
              << "\n  heuristics_aff_threshold     " << p.heuristics_aff_threshold
              << "\n  starting_aff_threshold       " << p.starting_aff_threshold
              << "\n  agglomeration_step           " << p.agglomeration_step
              << "\n  optimal_number_of_partitions " << p.optimal_number_of_partitions
              << "\n  minimal_number_of_edges      " << p.minimal_number_of_edges
              << std::endl;
}

template <class T, class Compare = std::greater<T> >
struct agglomeration_data_t
{
    std::vector<MapContainer<seg_t, handle_wrapper<T, Compare> > > incident;
    std::vector<edge_t<T> > rg_vector;
    std::vector<size_t> supervoxel_counts;
    std::vector<seg_t> seg_indices;
    std::vector<sem_array_t> sem_counts;
    std::vector<nuc_record_t> nuc_ids;
    std::vector<size_t> seg_size;
    agglomeration_param_t params;
};

template <class T>
struct agglomeration_output_t
{
    std::vector<edge_t<T> > res_rg_vector;
    std::vector<edge_t<T> > rej_rg_vector;
=========================== END excerpt ===========================

=========================== BEGIN src/agg/mean_aggl.cpp lines 715-990 (populate_heap and agglomerate_cc) ===========================
template <class T, class Compare = std::greater<T> >
inline heap_type<T, Compare> populate_heap(agglomeration_data_t<T, Compare> & agg_data, size_t startpos, size_t endpos, T const & threshold)
{
    Compare comp;
    heap_type<T, Compare> heap;
    auto & incident = agg_data.incident;
    auto & supervoxel_counts = agg_data.supervoxel_counts;
    auto & rg_vector = agg_data.rg_vector;
    auto & seg_indices = agg_data.seg_indices;

    size_t i = 0;

    for (size_t j = startpos; j != endpos; j++) {
        auto & e = rg_vector[j];
        heap_handle_type<T, Compare> handle;
        if (comp(e.w, threshold)){
            handle = heap.emplace(& e);
        }
        auto v0 = e.v0;
        auto v1 = e.v1;
        incident[e.v0].emplace(v1,handle_wrapper<T, Compare>(&e, handle));
        incident[e.v1].emplace(v0,handle_wrapper<T, Compare>(&e, handle));
        i++;
        if (i % 10'000'000 == 0) {
            std::cout << "reading " << i << "th edge" << std::endl;
        }
    }

    return heap;
}

std::pair<size_t, size_t> sem_label(const sem_array_t & labels)
{
    const auto *label = std::max_element(labels.begin(), labels.end());
    return std::make_pair(std::distance(labels.begin(), label), (*label));
}

bool sem_can_merge(const sem_array_t & labels1, const sem_array_t & labels2, const agglomeration_semantic_heuristic_t & sem_params)
{
    auto max_label1 = std::distance(labels1.begin(), std::max_element(labels1.begin(), labels1.end()));
    auto max_label2 = std::distance(labels2.begin(), std::max_element(labels2.begin(), labels2.end()));
    auto total_label1 = std::accumulate(labels1.begin(), labels1.end(), static_cast<size_t>(0));
    auto total_label2 = std::accumulate(labels2.begin(), labels2.end(), static_cast<size_t>(0));
    if (labels1[max_label1] < sem_params.dominant_signal_ratio * total_label1 || total_label1 < sem_params.total_signal_threshold) { //unsure about the semantic label
        return true;
    }
    if (labels2[max_label2] < sem_params.dominant_signal_ratio * total_label2 || total_label2 < sem_params.total_signal_threshold) { //unsure about the semantic label
        return true;
    }
    if (max_label1 == max_label2) {
        return true;
    }
    return false;
}

template <class T, class Compare = std::greater<T>, class Plus = std::plus<T>,
          class Limits = std::numeric_limits<T> >
inline agglomeration_output_t<T> agglomerate_cc(agglomeration_data_t<T, Compare> & agg_data, size_t startpos, size_t endpos, T const target_threshold)
{
    Compare comp;
    Plus    plus;

    T const h_threshold = T(agg_data.params.heuristics_aff_threshold,1);
    T const size_aff_threshold = T(agg_data.params.size_params.aff_threshold, 1);
    T const sem_aff_threshold = T(agg_data.params.sem_params.aff_threshold, 1);
    T const backbone_threshold = T(agg_data.params.input_aff_threshold, 1);
    const auto twig_params = agg_data.params.twig_params;
    const auto size_params = agg_data.params.size_params;
    const auto sem_params = agg_data.params.sem_params;

    auto & supervoxel_counts = agg_data.supervoxel_counts;
    auto & seg_indices = agg_data.seg_indices;
    auto & sem_counts = agg_data.sem_counts;
    auto & nuc_ids = agg_data.nuc_ids;
    auto & seg_size = agg_data.seg_size;
    auto & rg_vector = agg_data.rg_vector;
    agglomeration_output_t<T> output;
    auto heap= populate_heap<T, Compare>(agg_data, startpos, endpos, target_threshold);
    auto & incident = agg_data.incident;
    auto heap_size = heap.size();
    size_t num_of_edges = 0;
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
            } else if (v1 == s) {
                output.remap.emplace_back(v0, s);
            } else {
                std::cout << "Something is wrong in the MST" << std::endl;
                std::cout << "s: " << s << ", v0: " << v0 << ", v1: " << v1 << std::endl;
                std::abort();
            }

            if (s == v0)
            {
                std::swap(v0, v1);
            }

            // v0 is dissapearing from the graph

            // loop over other edges e0 = {v0,v}
            e.edge->w = Limits::min();
            for (auto p: incident[v0]) {
                auto v = p.first;
                auto e0 = p.second;
                if (v == v0) {
                    std::cerr << "loop in the incident matrix: " << seg_indices[v] << std::endl;
                    std::abort();
                }

                incident[v].erase(v0);

                //auto it = search_neighbors(incident[v1], v1, v);
                //if (it != std::end(incident[v1]) && (*it).segid(v1) == v)
                if (incident[v1].count(v) != 0)
                                                  // {v0,v} and {v1,v} exist, we
                                                  // need to merge them
                {
                    auto & e = incident[v1].at(v);
                    if(e0.edge->v0 == v0) {
                        e0.edge->v0 = v1;
                    }
                    if(e0.edge->v1 == v0) {
                        e0.edge->v1 = v1;
                    }
                    if (!e.valid_handle()) {
                        auto & e_dual = incident[v].at(v1);
                        std::swap(e0.edge, e.edge);
                        std::swap(e0.handle, e.handle);
                        e_dual.edge = e.edge;
                        e_dual.handle = e.handle;
                    }
                    e.edge->w=plus(e.edge->w, e0.edge->w);
                    e0.edge->w = Limits::min();
                    if (e.valid_handle()) {
                        heap.update(e.handle);
                        if (e0.valid_handle()) {
                            heap.erase(e0.handle);
                        }
                    }
                }
                else
                {
                    auto e = e0.edge;
                    if (e->v0 == v0) {
                        e->v0 = v1;
                    }
                    if (e->v1 == v0) {
                        e->v1 = v1;
                    }
                    incident[v].emplace(v1,e0);
                    incident[v1].emplace(v,e0);
                }
            }
            incident[v0].clear();
        }
    }
    return output;
}

=========================== END excerpt ===========================

=========================== BEGIN src/agg/mean_aggl.cpp lines 1144-1262 (threshold ladder driving agglomerate_cc) ===========================
    size_t rg_size = rg_vector.size();

    while (target_th >= th)
    {
        std::cout << "current threshold: " << target_th << std::endl;
        T const target_threshold = T(target_th,1);
        std::vector<std::future<agglomeration_output_t<T> > > fs;

        auto ccids = extract_cc<T, Compare>(agg_data, target_threshold);
        auto cc_queue = sorted_components(ccids);

        auto m = std::max_element(std::execution::par, cc_queue.begin(), cc_queue.end(), [](const auto & a, const auto & b) {
                return a.second < b.second;
        });

        if ((cc_queue.empty() or (*m).second < min_cc_threshold) and target_th != th) {
            std::cout << "No large CC found in " << rg_vector.size() << " edges" << std::endl;
            target_th -= agg_step;
            if (target_th < th) {
                target_th = th;
            }
            continue;
        }

        if (cc_queue.empty()) {
            std::cout << "no connect component, nothing to agglomerate" << std::endl;
            break;
        }

        partition_rg(rg_vector, ccids, cc_queue);
        auto cc_edges = cc_edge_offsets(rg_vector, ccids);
        std::cout << "cc_queue: " << cc_queue.size() << std::endl;
        std::cout << "cc_edges: " << cc_edges.size() << std::endl;

        size_t total_size = cc_edges.back().second;
        std::cout << "number of active edges: " << total_size << std::endl;

        size_t package_size = total_size/ncpus;
        if (package_size < min_edge_threshold) {
            package_size = min_edge_threshold;
        }

        SetContainer<seg_t> processed_cc;
        SetContainer<seg_t> target_cc;
        auto target_size = 0;

        if (seg_indices.size() < min_edge_threshold or cc_edges[0].second < min_edge_threshold) {
            fs.push_back(std::async(std::launch::async, agglomerate_cc<T, Compare, Plus, Limits>, std::ref(agg_data), 0, total_size, target_threshold));
        } else {
            size_t startpos = 0;
            size_t offset = 0;
            for (auto & [k, v]: cc_edges) {
                offset = v;
                if ((offset - startpos) >= package_size) {
                    fs.push_back(std::async(std::launch::async, agglomerate_cc<T, Compare, Plus, Limits>, std::ref(agg_data), startpos, offset, target_threshold));
                    std::cout << "submiting cc package from " << startpos << " to " << offset << std::endl;
                    startpos = offset;
                }
            }
            if (startpos != total_size) {
                std::cout << "submiting remaining cc from " << startpos << " to " << offset << std::endl;
                fs.push_back(std::async(std::launch::async, agglomerate_cc<T, Compare, Plus, Limits>, std::ref(agg_data), startpos, total_size, target_threshold));
            }
        }

        std::vector<std::vector<std::pair<seg_t, seg_t> > >remaps;
        for (auto & f: fs) {
            auto o = f.get();
            for (auto & r: o.remap) {
                of_remap.write(reinterpret_cast<const char *>(&(seg_indices[r.first])), sizeof(seg_t));
                of_remap.write(reinterpret_cast<const char *>(&(seg_indices[r.second])), sizeof(seg_t));
            }
            for (auto & e: o.res_rg_vector) {
                of_res.write(reinterpret_cast<const char *>(&(seg_indices[e.v0])), sizeof(seg_t));
                of_res.write(reinterpret_cast<const char *>(&(seg_indices[e.v1])), sizeof(seg_t));
                write_edge(of_res, e.w);
            }
            for (auto & e: o.rej_rg_vector) {
                of_reject.write(reinterpret_cast<const char *>(&(seg_indices[e.v0])), sizeof(seg_t));
                of_reject.write(reinterpret_cast<const char *>(&(seg_indices[e.v1])), sizeof(seg_t));
                write_edge(of_reject, e.w);
            }
            for (auto & e: o.sem_rg_vector) {
                of_sem_cuts.write(reinterpret_cast<const char *>(&(seg_indices[e.v0])), sizeof(seg_t));
                of_sem_cuts.write(reinterpret_cast<const char *>(&(seg_indices[e.v1])), sizeof(seg_t));
                //write_edge(of_reject, e.w);
            }
            for (auto & e: o.nuc_rg_vector) {
                of_nuc_cuts.write(reinterpret_cast<const char *>(&(seg_indices[e.v0])), sizeof(seg_t));
                of_nuc_cuts.write(reinterpret_cast<const char *>(&(seg_indices[e.v1])), sizeof(seg_t));
            }
            for (auto & e: o.twig_rg_vector) {
                of_twig.write(reinterpret_cast<const char *>(&(seg_indices[e.v0])), sizeof(seg_t));
                of_twig.write(reinterpret_cast<const char *>(&(seg_indices[e.v1])), sizeof(seg_t));
                write_edge(of_twig, e.w);
            }
            remaps.push_back(o.remap);
            mst_size += o.merged_rg_vector.size();
            residue_size += o.res_rg_vector.size();
        }
        std::cout << "finish agglomeration connect components" << std::endl;

        remap_edges<T, Compare>(agg_data, remaps, total_size);
        merge_edges<T, Compare, Plus, Limits>(agg_data, total_size);
        std::for_each(std::execution::par, agg_data.incident.begin(), agg_data.incident.end(), [](auto & d) {d.clear();});
        if (target_th == th) {
            std::cout << "stop agglomeration" << std::endl;
            target_th -= agg_step;
        } else {
            if ((target_th - th) > (agg_step*1.5)) {
                target_th -= agg_step;
            } else {
                target_th = th;
            }
            std::cout << "next agglomeration threshold:" << target_th << std::endl;
        }
    }//while
    assert(!of_mst.bad());
    assert(!of_remap.bad());
=========================== END excerpt ===========================

=========================== BEGIN work/test/run_v2_invariance.sh (currently only in work/abiss, must be copied) ===========================
#!/bin/bash
set -euo pipefail

run_start_ref=3c4f56219488441edb4c51ee7ca4f07cdef2cd8f
repo="$(realpath "$(dirname "$0")/../..")"
v2_root="$(mktemp -d "${TMPDIR:-/tmp}/abiss-v2.XXXXXX")"

cleanup() {
    if [ "${ABISS_KEEP_V2:-0}" = "1" ]; then
        echo "V2 temporary files kept at $v2_root"
    else
        rm -rf -- "$v2_root"
    fi
}
trap cleanup EXIT

if [ "$(git -C "$repo" rev-parse HEAD)" != "$run_start_ref" ]; then
    echo "V2: HEAD does not match run_start_ref $run_start_ref" >&2
    exit 1
fi
git -C "$repo" cat-file -e "$run_start_ref^{commit}"

toolchain_prefix="${ABISS_TOOLCHAIN_PREFIX:-${CONDA_PREFIX:-}}"
cxx_compiler="${ABISS_CXX_COMPILER:-}"
if [ -z "$cxx_compiler" ] && [ -f "$repo/build/CMakeCache.txt" ]; then
    cxx_compiler="$(
        sed -n 's/^CMAKE_CXX_COMPILER:FILEPATH=//p' \
            "$repo/build/CMakeCache.txt" | head -n 1
    )"
fi
if [ -z "$toolchain_prefix" ] && [ -n "$cxx_compiler" ]; then
    toolchain_prefix="$(dirname "$(dirname "$cxx_compiler")")"
fi
if [ -z "$cxx_compiler" ] && [ -n "$toolchain_prefix" ]; then
    cxx_compiler="$toolchain_prefix/bin/x86_64-conda-linux-gnu-c++"
fi
if [ -z "$cxx_compiler" ] || [ ! -x "$cxx_compiler" ]; then
    echo "V2: set ABISS_CXX_COMPILER or ABISS_TOOLCHAIN_PREFIX" >&2
    exit 1
fi

export PATH="$(dirname "$cxx_compiler"):$PATH"
if [ -n "$toolchain_prefix" ]; then
    export CMAKE_PREFIX_PATH="$toolchain_prefix${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
    export LD_LIBRARY_PATH="$toolchain_prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mkdir "$v2_root/baseline_src"
git -C "$repo" archive "$run_start_ref" | tar -x -C "$v2_root/baseline_src"

configure_and_build() {
    local label="$1"
    local source_dir="$2"
    local build_dir="$v2_root/${label}_build"
    cmake -S "$source_dir" -B "$build_dir" \
        -DCMAKE_BUILD_TYPE=Release \
        -DEXTRACT_SIZE=ON \
        -DCMAKE_CXX_COMPILER="$cxx_compiler" \
        > "$v2_root/${label}_configure.log"
    cmake --build "$build_dir" --parallel "${ABISS_BUILD_JOBS:-4}" \
        --target acme agg > "$v2_root/${label}_build.log"
    echo "V2 build $label: PASS"
}

configure_and_build baseline "$v2_root/baseline_src"
configure_and_build current "$repo"

baseline_run="$v2_root/baseline_run"
current_run="$v2_root/current_run"
python3 "$repo/work/test/make_fixture.py" "$baseline_run"
python3 "$repo/work/test/make_fixture.py" "$current_run"

env -u ABISS_NUC_DOMINANCE -u ABISS_NUC_MIN_TAGGED \
    "$repo/work/test/run_atomic_fixture.sh" \
    "$v2_root/baseline_build" "$baseline_run" \
    > "$v2_root/baseline_run.log" 2>&1
env -u ABISS_NUC_DOMINANCE -u ABISS_NUC_MIN_TAGGED \
    "$repo/work/test/run_atomic_fixture.sh" \
    "$v2_root/current_build" "$current_run" \
    > "$v2_root/current_run.log" 2>&1

identical=0
differing=0
missing=0
while IFS= read -r -d '' baseline_file; do
    relative="${baseline_file#"$baseline_run"/}"
    current_file="$current_run/$relative"
    if [ ! -f "$current_file" ]; then
        missing=$((missing + 1))
    elif cmp -s "$baseline_file" "$current_file"; then
        identical=$((identical + 1))
    else
        differing=$((differing + 1))
    fi
done < <(find "$baseline_run" -type f -print0 | sort -z)

echo "V2 baseline comparison: identical=$identical differing=$differing missing=$missing"
if [ "$differing" -ne 0 ] || [ "$missing" -ne 0 ]; then
    exit 1
fi

mapfile -t current_only < <(
    cd "$current_run"
    find . -type f -print | while IFS= read -r relative; do
        if [ ! -f "$baseline_run/$relative" ]; then
            printf '%s\n' "${relative#./}"
        fi
    done | sort
)
expected_sidecars=(done_nuc.data nuc_cuts.data ongoing_nuc.data)
if [ "${current_only[*]}" != "${expected_sidecars[*]}" ]; then
    echo "V2: unexpected current-only files: ${current_only[*]}" >&2
    exit 1
fi
for sidecar in "${expected_sidecars[@]}"; do
    if [ -s "$current_run/$sidecar" ]; then
        echo "V2: current-only sidecar is not empty: $sidecar" >&2
        exit 1
    fi
done
echo "V2 current-only empty sidecars: ${expected_sidecars[*]}"

if grep -q '^nuc:' "$v2_root/current_run.log"; then
    echo "V2: nucleus log output appeared with no nucleus source" >&2
    exit 1
fi
echo "V2 default-path nucleus log lines: 0"

env_run="$v2_root/default_env_run"
python3 "$repo/work/test/make_fixture.py" "$env_run"
(
    cd "$env_run"
    ABISS_NUC_DOMINANCE=garbage \
    ABISS_NUC_MIN_TAGGED=garbage \
        "$v2_root/current_build/acme" param.txt 0_0_0_0
) > "$v2_root/default_env.log" 2>&1
if grep -q '^nuc:' "$v2_root/default_env.log"; then
    echo "V2: nucleus log output appeared in the stale-environment test" >&2
    exit 1
fi
echo "V2 stale nucleus environment without nuc.raw: PASS"
echo "run_v2_invariance: PASS"
=========================== END excerpt ===========================

=========================== BEGIN work/test/run_atomic_fixture.sh ===========================
#!/bin/bash
set -euo pipefail

bin_path="$(realpath "$1")"
run_path="$(realpath "$2")"
tag=0_0_0_0

cd "$run_path"
"$bin_path/acme" param.txt "$tag"
mv edges_"$tag".data input_rg.data
for i in {0..5}; do
    cat boundary_"$i"_"$tag".data >> frozen.data
done
touch ns.data ongoing_semantic_labels.data ongoing_nuclei_labels.data ongoing_seg_size.data
"$bin_path/agg" 0.25 input_rg.data frozen.data ns.data
=========================== END excerpt ===========================

Now produce your review: a short summary, then findings each tagged [minor] or [major], then
any questions, then the final READY: line.
