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

## The next lever, and why it closed: the contamination is ONE supervoxel, and the affinity fuses it

**The frozen-boundary relaxation does not exist in these runs.** `CMakeLists.txt:131-135` --
`target_compile_definitions(agg PRIVATE FINAL)`; the three `EXTRA` lines are commented out. With
`OVERLAP=0` both `atomic_chunk_me.sh:44` and `composite_chunk_me.sh:71` call `agg`. So the strict
branch is already active: any edge touching a frozen segment is deferred.

**The globally-resolved table (nuctable arm) localises the entire defect to one object:**

```
table records 139,294    NONE 5,401   PROPER 133,892   CONFLICT 1
  CONFLICT sv 72198606672811349   tagged 28,576,381
  final CONFLICT segment tagged   28,576,381      -> ratio 1.000
  PROPER minority mass discarded       2,124      -> finding I1 is negligible here
  the object is already 243,481,814 vox across 88 atomic chunks (83.5% of the final segment)
```

So the contamination is inside a single watershed supervoxel. Agglomeration cannot split a
supervoxel, which is why ordering, the clause-2 veto, the global table and the frozen rule all
measured out at zero effect.

**And the affinity fuses the three somata at the top of its range**
(`conflict_bottleneck.py`, min-pool factor 4 -- min-pooling preserves bottlenecks):

```
pooled min-affinity inside the segment: p1=0.00010  p5=0.00671  p25=0.99974  p50=0.99993
fraction >= 0.99999 (WS_HIGH_THRESHOLD): 0.0453
joined at 0.3 / 0.9 / 0.99 / 0.999 / 0.9999 : all three pairs
joined at 0.99999 / 0.999999                : none
pairwise bottleneck 275-319, 275-373, 319-373: ~0.9999
```

By widest-path duality the bottleneck is the min over cuts of the max edge crossing, so **any**
surface separating 275 from 319 must sever affinity >= 0.999 -- 3.3x the 0.3 agglomeration
threshold and above the 25th percentile of the segment's own interior. There is no weak surface to
find; an affinity-driven split is ill-posed, not merely hard.

Caveat on the digits: affinity is stored float16 (spacing ~5e-4 near 1.0) and passed through
`AFF_RESTORE_SIGMOID=0.2`, so the 0.9999-vs-0.99999 boundary sits at storage resolution and should
not be over-read. The robust statement is joined at 0.999, separated at 0.99999. Note this does
mean the watershed's own basin threshold (0.99999) would NOT have merged them -- the fusion enters
during the watershed's agglomeration phase (`WS_LOW_THRESHOLD=0.00001`), not at basin formation.

**Conclusion.** The 291M-voxel three-soma segment is an affinity failure, not a decoding failure.
Delivering "no other segment holds nucleus mask voxels" requires the mask to OVERRIDE the image
evidence (a nucleus-seeded geodesic partition of the CONFLICT objects), not to arbitrate against
it. That is imposing structure: no image evidence supports any particular cut location.
