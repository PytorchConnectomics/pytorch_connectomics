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
