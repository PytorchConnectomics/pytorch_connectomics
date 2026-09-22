# Task

Snap watershed supervoxels onto the nucleus instance mask, and split the ones that overlap more
than one nucleus, so that ABISS's nucleus constraint operates on supervoxels that each belong to
exactly one nucleus.

The literal user request:

> "for the abiss run, just make sure, to merge all ws small supervoxel to overlapped nucleus mask,
> if overlapped with multiple, do a simple seeded watershed with the overlapped region
> (e.g. lib/em_seg)"

asked in answer to "is it better to do it during abiss or better to do another run with
graph-based split?".

## Why the watershed stage, and not a region-graph split

Measured in `.agent/features/abiss_nucleus/state/evidence_conflict.md`: the entire contamination on
the `worst3` crop lives inside **one** watershed supervoxel.

```
globally resolved nucleus table (nuctable arm):
  139,294 records    NONE 5,401   PROPER 133,892   CONFLICT 1
  CONFLICT sv 72198606672811349   tagged 28,576,381
  final CONFLICT segment tagged   28,576,381        -> ratio 1.000
  the object is 243,481,814 vox across 88 atomic chunks before agglomeration starts
```

Region-graph nodes *are* supervoxels, so no graph-based split can separate voxels that share a
node. Only the watershed stage can. Four agglomeration-level levers were tried first and all
measured out at zero effect on this: merge ordering, the `nuc_can_merge` CONFLICT clause, the
global nucleus table, and the frozen-boundary rule (already strict -- `agg` is built with `FINAL`
only, `EXTRA` is commented out in `CMakeLists.txt:131-135`).

The affinity offers no help in placing the cut: min-pooled bottleneck between all three nucleus
pairs is ~0.999, i.e. **any** separating surface must sever affinity >= 0.999, versus a 0.3
agglomeration threshold. So the split is imposed by the mask, not discovered in the image. That is
acceptable here only because the task owner has confirmed the mask is correct and conservative
(smaller than the true nucleus), which makes it ground truth for "these voxels are one neuron".

## Repository and baseline

`lib/abiss`, worked in `work2/abiss` (symlinked here), detached at **312bf54**. The working tree
already carries the completed, unreviewed-for-merge global-nucleus-table change from the
`abiss_nucleus` run 2 (`state/run_start.diff`, 257 lines). That is the baseline for this run --
do not revert it.

Never touch the live `lib/abiss` checkout or `lib/abiss/build/`; SLURM jobs execute those binaries.

## Required behaviour

1. **Snap.** After the atomic watershed, every supervoxel overlapping nucleus `N`'s mask is
   relabelled to a single id derived deterministically from `N` -- **not** a chunk-local id. This
   is `em_seg/seg_pipeline.py:_affinityToSeg2D` ("snap to soma id",
   `ii = np.unique(seg[0][mask_soma==soma_id]); rl[ii] = seg_m + i`).
   Because the id is derived from `N`, ABISS's ws remap (`merge_remaps.py` -> `ws3` ->
   `chunkmap.data`) stitches `N`'s pieces across chunks for free, and a merge between two different
   nuclei becomes inexpressible rather than something that has to be filtered out. That is the
   `waterz.somaBFS` guarantee obtained by construction.
2. **Split.** A supervoxel overlapping two or more nuclei is divided by a seeded watershed
   restricted to that supervoxel's own voxels, seeded by the overlapping nucleus regions. em_seg
   marks this case `# need to split sometimes` and does not handle it; this run must.
3. The nucleus id space must not collide with watershed ids. Reserve or offset explicitly.

## Hard constraints

* **Default-path bit-invariance.** With no `NUC_PATH`, the watershed output must be byte-identical
  to baseline. `work/test/run_v2_invariance.sh` (already retargeted to `312bf54` by run 2) is the
  gate. Non-negotiable: the pipeline reproduces a Seuron provenance record.
* **Do the relabel before `chunkmap` is generated.** `chunkmap.data` records a remap already
  applied to the watershed volume; injecting later strands the snapped-away supervoxels. This is
  exactly how `scripts/nucleus_snap.py` shattered nucleus 173 into 39,439 fragments in run 1. That
  script stays disabled and is not a starting point.
* Chunk-locality: an atomic chunk is 252^3 (~2.3 x 2.3 x 5 um), smaller than a soma, so a chunk
  often sees only part of one nucleus. The snap must still be correct under that, which is why the
  id is derived from the nucleus rather than assigned locally.
* No git commits. No new dependencies.

## Success criteria

Measured on the `worst3` crop (BBOX `[2772,9324,1260,3780,10584,2520]`), against the `w2ctl`
control, with `dev/zebrafinch/nucleus_shell_contamination.py --tol 0.0`:

1. **Primary** -- shared mask mass (mask voxels in any segment holding mask voxels of >= 2 nuclei)
   drops from **110,244** to **0**. With one supervoxel per nucleus this should be exact, not
   approximate; anything above 0 means the snap leaked.
2. **Fragmentation** -- segments needed to cover 95% of each nucleus's mask goes to **1** for all
   eight nuclei (control: 2 for 275/319/373, 1 for the rest), and the whole-volume median of 34
   segments per nucleus is the standing target for the follow-up run.
3. **Dominance** -- >= 0.99 for all eight nuclei (control 0.8957 / 0.9104 / 0.8988 for the three,
   1.000 for the clean five).
4. **No runaway, no shatter** -- largest segment <= 450,000,000 voxels (control 291,235,662,
   `nonuc` 640,562,148); root segment count within 0.9x-1.25x of the control's 3,613.
5. `run_v2_invariance.sh` passes.

Report all five for control and treatment side by side. A treatment that buys criterion 1 by
shattering the somata fails 2-4.

## Out of scope

The two `abiss_nucleus` review_v0 findings (per-worker table memory, stale-table provenance) --
they belong to that run's `code_v1`. Whole-volume re-run. NERL.
