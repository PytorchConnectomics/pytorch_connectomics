# Abiss
Affinity based image segmentation system, implementing algorithm described in https://arxiv.org/abs/2106.10795

# Build
```sh
mkdir build && cd build
cmake ..
make
```

# Nucleus instance guidance

ABISS can use an optional nucleus instance mask to constrain agglomeration. Set
`NUC_PATH` in the pipeline JSON to an instance volume. `NUC_RATIO` gives its
`[z,y,x]` downsample factor relative to `AFF_RESOLUTION`, and `NUC_OFFSET` gives
its `[z,y,x]` low-resolution voxel offset; their defaults are `[1,1,1]` and
`[0,0,0]`. A high-resolution coordinate `q` maps to
`(q + ratio // 2) // ratio + offset` on each axis. ABISS reads those source
voxels and upsamples with nearest-neighbour selection, so instance ids are
never interpolated. The aligned spatial cutout must exactly match the
watershed cutout after upsampling. Input values must be integers in
`[0, 0xFFFFFFFF]`; they are written to `nuc.raw` as `uint32`. Zero is
background and every nonzero value is an instance id.

Upgrade note: do not mix chunks produced by pre-nucleus and nucleus-aware
binaries in one hierarchy. The merge stage expects every child to provide the
new nucleus sidecar, even when nucleus guidance is disabled. Restart the
pipeline or regenerate all child chunks after upgrading.

Two extraction settings are read from the environment:

* `ABISS_NUC_DOMINANCE` is the minimum dominant-id fraction. It must be finite
  and in `(0.5, 1.0]`; the default is `0.6`.
* `ABISS_NUC_MIN_TAGGED` is the minimum number of tagged voxels needed for a
  supervoxel to carry usable nucleus evidence; the default is `50`.

Each cluster carries one of three fixed-width records:

* `NONE`: no usable evidence, with `count == 0` and `total == 0`.
* `PROPER`: a dominant nonzero id, its supporting `count`, and the `total`
  usable tagged voxels.
* `CONFLICT`: tagged evidence has no dominant id, with `count == 0`.

Extraction and every record join preserve Closure:
`PROPER => count * dominance_den >= dominance_num * total`, and
`state != PROPER => count == 0`. Joins use exact integer comparisons and
checked 64-bit addition.

The implemented contract is:

> **Invariant D.** No merge performed by agglomeration (1) joins two clusters
> whose recorded dominant nucleus ids differ, (2) joins a CONFLICT cluster to
> a cluster carrying a recorded dominant id, or (3) joins two CONFLICT
> clusters.

The nucleus veto is applied at every affinity. Clause 3 can over-segment:
adjacent CONFLICT clusters stay separate even when their edge would otherwise
merge. It is retained because joining already-mixed clusters compounds
contamination.

Invariant D is deliberately weaker than tracking every identity. Minority
identities are not retained: `A: 99 id1 + 1 id2 -> PROPER id1` and
`B: 99 id1 + 1 id3 -> PROPER id1` may merge, thereby joining identities 2 and
3. A `60 id1 + 40 id2` supervoxel is PROPER id1 at the default ratio, while
any supervoxel below `ABISS_NUC_MIN_TAGGED` becomes NONE however mixed.

**Bound C applies only to recorded usable evidence.** In a PROPER cluster,
`total - count <= (1 - dominance_ratio) * total`. It does not bound the
cluster's actual minority voxels: sub-floor supervoxels contribute real mass
that the record intentionally omits.

Extraction reports `nuc: conflict_sv`, `nuc: minority_sv`,
`nuc: subfloor_sv`, and `nuc: subfloor_voxels`. Hierarchy stages also report
conflicting record collisions. These counters indicate whether identity-set
tracking or a future pre-watershed nucleus cut is warranted. Rejected edges
are written to `nuc_cuts.data` and archived as
`nuc_rejected_edges_<chunk>.log`.

Two corruption tripwires abort instead of continuing: merge propagation
rechecks the same nucleus predicate enforced by the veto, and nucleus voxel
counts abort on 64-bit overflow.

Raw nucleus interiors often remain separated from cytoplasm by the nuclear
envelope in affinity predictions, so an id confined to the interior may not
constrain the soma. In practice, tag perinuclear cytoplasm (for example, an
instance-specific dilated shell) rather than assuming raw nucleus interiors
will attach to the cell body.

This feature only observes nucleus identity and refuses incompatible merges;
it never forces fragments carrying the same nucleus id to merge. Soma
fragmentation is therefore not addressed, and the large-cluster size veto in
`mean_aggl.cpp` can still prevent a soma from absorbing proximal dendrites. A
follow-up should add a **must-link snap**, as `lib/em_seg` does: before building
the region graph, snap every watershed fragment overlapping one soma onto a
single reserved id.
