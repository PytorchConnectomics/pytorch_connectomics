# Plan v2

## Summary

Plan v2 closes every remaining contract the v1 review found undefined, and drops two things rather
than specifying them badly: **multi-hop absorption** (its precision gate would have fed GT-measured
precision back into selection — a real leak) and **Phase 3 multiscale** (already dropped in v1).

Three corrections are substantive, not editorial:

1. **The caliber formula in v1 was dimensionally wrong.** `voxels · 1620 nm³ / centerline_nm` is an
   area, not a length, so `caliber_nm > 1500` would have quarantined almost every anchor. v2 uses
   the diameter `caliber_nm = 2·sqrt(volume_nm³ / (π · centerline_nm))`.
2. **`endpoint_frac <= 0.5` was tautological** given v1's own declared range. v2 redefines it on
   `[0, 1]` with `1` at a tip and gates `>= 0.5`.
3. **`tangent_cos >= 0` was not sign-invariant** for an unoriented principal axis. v2 uses
   `|cos| >= 0.5`.

The affinity contract is now stated in the two source-indexed forms that are actually used, and is
anchored to the canonical helper: `connectomics/data/processing/affinity.py`'s
`resolve_affinity_offsets_from_kwargs({})` returns `[(1,0,0), (0,1,0), (0,0,1)]` in z-y-x order —
channel `c` carries offset `offsets[c]`, which is exactly the BANIS layout measured against
`_ArrayVolume` for plan v1. Canonical declaration and measured behaviour agree.

Every numeric rule below is predeclared and fixed before any GT is read.

## Scope

Unchanged from v1: everything in `dev/zebrafinch/ec_mid_piece/`; no package code, no dependency, no
commit, no canonical input modified, no new segmentation volume.

Removed in v2: multi-hop absorption (deferred, with reason recorded). Already removed in v1: the
GT-informed scope selector and Phase 3 execution.

Operative repository root is the main checkout `/projects/weilab/weidf/lib/pytorch_connectomics`
(branch `codex/gt-free-tube-analysis`); `dev/` and `.agent/` are gitignored and exist only there, so
tracked-file git diffs stay empty and code review is served the new files directly.

## Proposed Changes

### Firewall ordering (v1 review finding A)

`task.md` contains a real tension: the evaluator-only section says GT may be read only after the
assignment manifest is immutable, while Phase 0 and the first decision gate *require* reproducing
the baseline NERL and L126 inventory before any policy is tested. v2 resolves it structurally
rather than by preference:

- `stage0_reproduce.py` is **evaluator-side**, lives with the `evaluation_gt/` outputs, and emits
  only (a) a pass/fail gate and (b) constants already published in `lessons.md` L126.
- **No `gt_free/` script reads anything under `evaluation_gt/`**, and the firewall test asserts
  this by static inspection of every `gt_free/` script's imports, CLI options, and opened paths.
- The README states the ordering and this justification explicitly, so the reader is not left to
  reconcile the two sentences of `task.md` themselves.

### Freeze and attestation contract (finding A, second half)

Stage 2 writes, per grid point, in this order:

```
gt_free/assignments_<key>.npz          # payload
gt_free/assignments_<key>.frozen.json  # proposal_sha256 = sha256(<key>.npz bytes),
                                       # implementation_sha256, parameters,
                                       # gt_free: true, frozen_before_evaluation: true
```

`<key> = <policy>__anchor<A>_floor<F.FF>_margin<M.MM>` with `A` rendered `1e6` or `3e5`, so the
anchor-floor sensitivity variant cannot collide with the primary. Stage 3 recomputes
`sha256(npz)`, compares against `frozen.json`, verifies both markers, and refuses to run on any
mismatch — the same contract `arm096_evaluate_frozen.py` already enforces.

### Affinity contract, source-indexed (finding D)

Let `A` be the raw h5 dataset `main`, shape `(3, Z, Y, X)`, float16, with
`e_0 = (+1,0,0)`, `e_1 = (0,+1,0)`, `e_2 = (0,0,+1)` in ZYX, and
`R = clip(restore_sigmoid(A, 0.2), 0, 1)` where `restore_sigmoid(a, s) = sigmoid(logit(a)/s)`.

**Form 1 — the only form Stage 1 uses.** For the face-adjacent voxel pair `(p, p + e_c)`, its
affinity is `R[c, p]`. Valid domain: `0 <= p` and `p + e_c < (Z,Y,X)`. This matches the canonical
helper's declared offsets `[(1,0,0),(0,1,0),(0,0,1)]` for channels `0,1,2`.

**Form 2 — used only by the test, to compare against the code that produced arm0_96.** For ABISS
output index `q` in ZYX and ABISS channel `c'` (`0=x, 1=y, 2=z`):

```
abiss[c', q] == R[2 - c', q - e_{2-c'}]        valid when q - e_{2-c'} >= 0
abiss[c', q] == 0                              when q_{2-c'} == 0   (no source; zero-padded)
```

Output index `q` reads input `q - e`, not `q + e`. This equality was verified elementwise on a real
crop of `chunk_z2_y5_x5.h5` with `maxdiff = 0.0` on all three channels, and all four wrong
conventions (no shift, `channel c = c'`, wrong-axis shift, `+1` instead of `-1`) fail.

Store facts: 726 files `chunk_z{Z}_y{Y}_x{X}.h5`, grid `6 x 11 x 11`, each `(3,1008,1008,1008)`
float16 gzip with chunks `(3,64,64,64)`; chunk `(Z,Y,X)` covers `[1008·Z : min(1008·(Z+1), dim)]`
per axis; **no halo inside the file**. Keep-mask `tissue_border_keep_mask_full.zarr`, uint8 ZYX
`(5700,10912,10664)`, masked → affinity 0, applied after restore, matching the decode.

Seam behaviour: `read_slab` requests one extra low-side plane per axis when available so that every
edge `(p, p+e_c)` with both voxels inside the requested region has its source `R[c,p]` present. At
the global volume face where no source exists, the edge does not exist and is not emitted.

### Stage 0 — GT-free full-population inventory

`stage0_inventory.py` streams the segmentation only, Z-slabs of 64 planes with a 1-plane halo, over
all 726 chunk columns, accumulating per segment, in nm units (`voxel = 9 x 9 x 20 nm`, volume
`1620 nm³`):

`voxels`, `bbox_lo/hi`, `sum_p` (3), `sum_ppT` (6 upper-triangular), `border_faces` (6-bit),
`rag_degree` (distinct 6-neighbour labels).

Derived after the pass, from the covariance `C = sum_ppT/n - centroid·centroidᵀ` with eigenvalues
`λ1 >= λ2 >= λ3` and unit first eigenvector `u1`:

```
est_centerline_nm = sqrt(12·λ1)                        # uniform-rod length from its variance
est_endpoint_a/b  = centroid ± 0.5·est_centerline_nm·u1 # ESTIMATED, labelled as such
caliber_nm        = 2·sqrt(voxels·1620 / (π·est_centerline_nm))   # diameter; nm, not nm²
elongated         = (λ1 / max(λ2, 1e-12)) >= 4          # u1 is meaningful only when true
```

Output `gt_free/segment_inventory.npz` over the **entire** segment population. Cost ≈2.4 h at 8-way.

**Predeclared GT-free bands** (fixed before any GT read; reported post-hoc against the oracle
population, never refitted to it):

```
anchor   : voxels >= 1e6           primary ;  3e5 as a declared sensitivity variant
fragment : voxels <  anchor floor, partitioned for reporting into
           F1 [1, 1e3)   F2 [1e3, 1e4)   F3 [1e4, 1e5)   F4 [1e5, anchor_floor)
```

**Predeclared GT-free anchor quarantine** — quarantined if **any** of:
`border_faces` includes two opposite faces; `voxels > 5e8`; `caliber_nm > 1500`.

### Stage 0b — evaluator-side reproduction

`stage0_reproduce.py` asserts `score_lut(..., merge_threshold=50) == 0.4443760423975249` within
`1e-10` via `arm096_endpoint_oracle.{load_graph_and_lut, score_lut}`, and reproduces the L126
partition with `analyze_arm096.py`'s per-`(skeleton, label)` piece definition: `668 / 1887 / 8979`
pieces exactly and `87.9 / 7.65 / 4.11` mass percent within `0.05` pp. (A global per-label count
gives `665 / 1889 / 8854` and `88.21 / 7.68 / 4.10`; the per-skeleton definition is the correct one
and must not be silently substituted.) Aborts naming the mismatching quantity.

### Stage 1 — frozen candidate table (GT-free, full population)

`stage1_candidate_edges.py --shard i --nshard N` (`N <= 8`). Per chunk, stream seg + affinity
Z-slabs with a 1-voxel halo. For every face-adjacent pair whose labels are
`(fragment-band, anchor-band)`, accumulate per `(fragment_id, anchor_id)`:
`contact_voxels`, per-axis contact counts, contact bbox and centroid, and affinity accumulators
(`sum`, `sum²`, `min`, `max`, and a 64-bin fixed histogram on `[0,1]` giving `p10/p50/p90` to
`1/64`). Accumulators, never per-voxel rows.

Chunk-boundary ownership: each contact pair is owned by the chunk containing its **low-side**
voxel, which makes the halo overlap idempotent by construction; the owning chunk id is recorded.

Two outputs, both frozen and hashed into `gt_free/input_manifest.json`:

- `gt_free/candidate_edges.npz` — one row per `(fragment, anchor)` edge, joined against Stage 0 to
  add `caliber_ratio = min(cal_f,cal_a)/max(cal_f,cal_a)`, `tangent_cos = |u1_f · u1_a|` (valid only
  when both are `elongated`), `endpoint_frac` (below), `n_anchor_candidates`, anchor descriptors,
  and one `valid_*` flag per evidence source. `best`/`runner_up`/`margin` are **not** stored.
- `gt_free/fragment_contact_summary.npz` — one row for **every** fragment in the inventory,
  including those with zero anchor contacts, so Stage 2 can enumerate the zero-contact population
  from the frozen artifacts alone (finding F).

```
t             = clip( (c_contact − est_endpoint_a)·u1 / est_centerline_nm , 0, 1 )
endpoint_frac = 2·|t − 0.5|          # 0 at the middle, 1 at either tip
```

### Stage 2 — frozen policy resolution (GT-free; no LUT, no NERL, no GT path)

`stage2_resolve.py`. Candidate set for a fragment = its edges to **non-quarantined** anchors with
all `valid_*` true. Quarantined anchors are removed **before** ranking, so they cannot manufacture
false ambiguity (finding B).

| policy | score | floor applies |
|---|---|---|
| `none` | — | empty assignment baseline |
| `contact_area` | `contact_voxels` | no (declared negative control; floor fixed at 0) |
| `aff_mean` | `aff_mean` | yes |
| `aff_p90` | `aff_p90` | yes |
| `aff_p90_margin` | `aff_p90` | yes |
| `aff_p90_margin_morph` | `aff_p90` | yes |

Grid: `anchor_floor ∈ {1e6, 3e5}` x `floor ∈ {0.5,0.6,0.7,0.8,0.9}` x
`margin ∈ {0.00,0.02,0.05,0.10,0.20}`.

**Margin is relative and therefore dimensionless**, so one grid works for both affinity and
contact-area scores:

```
runner_up = 0.0 when the fragment has exactly one eligible anchor
margin    = (best − runner_up) / max(best, 1e-9)      # singleton -> margin = 1.0
```

**Acceptance** (finding B): accept the top-ranked anchor iff `best >= floor` **and**
— when `margin_threshold > 0` — `margin > margin_threshold`; when `margin_threshold == 0`,
`best > runner_up` **strictly**. Exact ties always abstain, at every grid point.
Ranking order: `score` descending, then `contact_voxels` descending, then `anchor_id`
**ascending**; stable sort throughout.

`aff_p90_margin_morph` additionally requires: `caliber_ratio >= 0.25` **and**
`endpoint_frac >= 0.5` **and** (`tangent_cos >= 0.5` when both segments are `elongated`, else this
term passes).

`reciprocal_best` (f's best anchor is a, and f is a's best fragment) is computed and recorded as a
reported field; it gates nothing in the primary policies.

Invariants, each with a test: one anchor per accepted fragment; an anchor's own label never
changes; **no anchor-anchor union** — every induced component contains exactly one anchor,
asserted, and a violation aborts; an assigned fragment cannot itself be an anchor; quarantined
anchors receive nothing; byte-identical reruns from the cached table.

`gt_free/residual_<key>.npz` assigns each fragment in `fragment_contact_summary` exactly one reason
under this fixed **first-match-wins** order, which keeps every code reachable (finding F):

```
1 no_anchor_contact          zero anchor-band contacts of any kind
2 only_quarantined_anchor    >=1 anchor contact, all quarantined
3 missing_or_invalid_evidence >=1 non-quarantined anchor, but every such edge has a false valid_* flag
4 geometry_conflict          best eligible edge fails the morphology gate (morph policies only)
5 weak_affinity              best eligible score < floor
6 ambiguous_multiple_anchors passes floor but fails the margin / strict-tie rule
7 multiscale_contradiction   never emitted (Phase 3 not run)
```

**Multi-hop is deferred, not specified** (finding E). Stage 1 records only fragment-anchor edges,
so complete-graph depth-2 resolution would need fragment-fragment adjacency that no stage builds;
and triggering a second Stage 2 run from a Stage 3 GT-measured precision gate would feed test
evaluation back into selection. The README records this as a deferral with both reasons. `task.md`
permits it ("*may* follow only if one-hop is precision-safe").

### Stage 3 — post-freeze evaluation (evaluator-only)

Refuses to run unless `sha256(assignments npz)` matches `frozen.json` and both markers are set.
Loads graph+LUT, re-asserts the baseline, applies assignments, scores.

`apply_assignments(lut, assignments)` is a new **directed** helper, not
`arm096_endpoint_oracle.relabel` — `relabel` is union-find over unordered pairs and would happily
union two anchors. It maps fragment label → anchor label, asserts anchors are fixed points, and
asserts one anchor per induced component.

**Ceiling composition rules** (finding G), stated so the coder invents nothing:

- **Mechanism ceiling.** Apply the oracle joins to the LUT **first**
  (`lut_o = relabel(lut, oracle_pairs)`), then apply the **unchanged** frozen assignments, mapping
  each assignment's anchor to its oracle-join root: `target = root_oracle(anchor)`. Assignments are
  never re-derived, so this measures the mechanism on a correct backbone rather than a different
  policy. Reported for both L123's 277 frozen joins from `oracle_gt/gt_nominated_pairs.json` and
  the stronger L126 `>=50-owner-node` rung.
- **Oracle-clean-anchor ceiling.** **Filter** the already-frozen assignments, dropping those whose
  anchor is one of the three known ERL-visible cross-neuron labels (`73465106696008740`,
  `72269800654646543`, `72198881818968095`). Do **not** re-rank onto clean runners-up — that would
  be a different, GT-informed policy. The report states this row is therefore a **lower bound** on
  what clean-anchor knowledge could buy.

Reported per grid point: NERL and delta from its direct baseline; fraction of the `+0.138303` L126
increment recovered; per-skeleton deltas with improved/unchanged/regressed counts and the max
single-skeleton regression; accepted fragments and components; correct-host precision and GT-mid
coverage **with a Wilson 95% score interval and sample count**; wrong-host, cross-neuron-union and
multi-owner-component counts at both `mt=50` materiality and zero tolerance; stratification by
contact class, band, contact type, anchor quality and margin bin; full risk/coverage curves; wall
time and peak memory per stage.

**Honest GT-free baseline = raw arm0_96 `0.4443760423975249`.**
`arm096_error_correction/decoder_gtfree/` holds only `segment_skeleton_graph.h5` and `chunks/`; the
`frozen_endpoint_merges.npz` that `arm096_evaluate_frozen.py` defaults to does **not** exist, so no
frozen autonomous substantial-linker artifact is available. Per `task.md` the run proceeds on raw
arm0_96 and states it measures mid-piece absorption *before* the substantial-linking stage exists.
Only that row sits next to FFN `0.538003`; every ceiling goes in a separately headed table. The
whole grid is reported as a diagnostic sweep and one setting is *nominated* for a future
confirmation run — no test-selected point is called validated.

### Stage 4 — residual taxonomy, funnel, recommendation

Shares of the `+0.138303` L126 increment per reason bucket, and the funnel: candidate coverage →
quarantine → ranking ambiguity → confidence gate → residual no-contact. Predeclared thresholds
(finding F):

```
material bucket      : bucket holds >= 20% of the +0.138303 increment
corridor evidence    : nearest-anchor est_endpoint distance <= 2000 nm
                       AND |u1_f · u1_a| >= 0.5
                       AND median restored affinity sampled along the straight
                           endpoint-to-endpoint corridor >= 0.3
escalate             : no_anchor_contact is material AND >= 50% of that bucket's
                       increment share passes corridor evidence
```

If escalation does not fire, the report states that ownership or anchor quarantine is the binding
constraint instead.

## Files and Areas

| Path | Purpose |
|---|---|
| `dev/zebrafinch/ec_mid_piece/README.md` | commands, environment, wall time, peak memory, resume, deferrals |
| `affinity_io.py` | verified BANIS reader, restore-sigmoid, keep-mask, `edge_affinity` |
| `stage0_inventory.py` | GT-free full-population segment inventory (no GT argument surface) |
| `stage0_reproduce.py` | evaluator-side baseline + L126 reproduction gate |
| `stage1_candidate_edges.py` | sharded GT-free edge extraction + global merge + contact summary |
| `stage2_resolve.py` | frozen policies, assignments, freeze attestation, residual |
| `stage3_evaluate.py` | post-freeze GT evaluation + ceiling composition |
| `stage4_residual.py` | residual taxonomy, funnel, recommendation |
| `visual_probes.py` | crop manifest for correct/wrong/ambiguous/contaminated/no-contact cases |
| `tests/test_*.py` | truth table, canonical-offset agreement, chunk dedup, invariants, determinism, firewall, LUT eval |
| `gt_free/`, `evaluation_gt/` | artifacts named exactly as `task.md` requires |

Read-only and unmodified: `reports/arm096_lut/`, `reports/arm096_{error_structure,breaks_v2}.json`,
`arm096_error_correction/oracle_gt/`, the arm0_96 precomputed segmentation, the affinity chunk
store, `tissue_border_keep_mask_full.zarr`,
`/projects/weilab/dataset/zebrafinch/test_50_skeletons.h5`. Imported unmodified:
`arm096_endpoint_oracle.{load_graph_and_lut, score_lut}`,
`connectomics.metrics.nerl.import_em_erl`,
`connectomics.data.processing.affinity.resolve_affinity_offsets_from_kwargs`.

## Verification Plan

Environment: `source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc`.

1. **Affinity truth table (primary).** Read a real crop of `chunk_z2_y5_x5.h5`; build
   `_ArrayVolume(blk, convention='banis', restore_sigmoid_scale=0.2)`; assert Form 2 elementwise
   with `maxdiff == 0` for all three channels on the interior domain, and assert the zero-pad rule
   at the low face. Assert all four negative controls are **not** equal.
2. **Canonical-offset agreement.** Assert
   `resolve_affinity_offsets_from_kwargs({}) == [(1,0,0),(0,1,0),(0,0,1)]` and that `edge_affinity`
   for channel `c` implements the pair `(p, p + offsets[c])` — grounding Form 1 in the canonical
   helper as `task.md` requires.
3. **Chunk-boundary dedup.** A synthetic object straddling a 1008 boundary yields identical contact
   statistics extracted as one box or as two overlapping shards; plus an explicit assertion that
   low-side ownership makes the halo idempotent.
4. **Invariants and determinism.** Property tests on synthetic graphs including two anchors
   contending for one fragment and an exact score tie (must abstain at every margin including
   `0.00`); rerunning Stage 2 from the cached table reproduces byte-identical assignment files.
5. **Geometry unit tests.** A synthetic rod of known length and radius recovers
   `est_centerline_nm` and `caliber_nm` to within 10%, which would fail under v1's `nm²` formula;
   `endpoint_frac` is `1` at a tip and `0` at the middle; `tangent_cos` is invariant when either
   principal axis is negated.
6. **GT firewall.** Static audit that no `gt_free/` script imports from, takes a CLI option for, or
   opens `test_50_skeletons.h5`, `reports/arm096_lut`, `oracle_gt/`, or anything under
   `evaluation_gt/`; plus a manifest audit that every `assignments_*.npz` matches its
   `frozen.json` hash and carries both markers.
7. **LUT-level evaluation.** `score_lut` reproduces `0.4443760423975249` within `1e-10`;
   `apply_assignments` with an empty map is the identity; a hand-built two-anchor assignment is
   rejected by the invariant rather than silently merged; the mechanism-ceiling composition maps a
   known anchor alias to its oracle root.
8. **Smoke run** on a small crop exercising all three affinity axes and a chunk border, before the
   sharded full pass.
9. **Reproduce** the baseline and L126 inventory (Stage 0b) before any policy is resolved.

Operational: shards capped at 8 concurrent readers and chained rather than co-scheduled with other
affinity jobs — `/projects` has previously saturated under parallel affinity streaming and blocked
unrelated read-heavy jobs in D-state. Every stage resumes from its per-shard cache and prints a
heartbeat; the README records the non-interactive resume command with measured wall time and peak
memory. Expected: Stage 0 ≈2.4 h, Stage 1 ≈3.5 h at 8-way, Stages 2–4 minutes.

## Risks and Questions

1. **Edge-table size remains the main unknown.** The segmentation is dense (99.4% foreground in a
   probe), so fragment-anchor adjacency count is not yet bounded. Only `(fragment-band,
   anchor-band)` pairs are emitted, stored as accumulators. Stage 1 prints running edge counts and
   **stops and reports** if the merged table would exceed ~200 M rows, rather than silently
   sampling.
2. **The band ladder is coarse and predeclared.** `anchor >= 1e6` voxels is a judgement made
   without GT and may not align with the oracle's `10-49 GT node` population. That misalignment is
   a result to report, not a knob to retune; the `3e5` sensitivity variant bounds how much the
   choice mattered.
3. **Endpoints and centerline are estimated from second moments**, not skeletonized. For a curved
   or branched fragment the rod model understates length and misplaces tips, which weakens
   `endpoint_frac` and the corridor test. This is a declared approximation, flagged in the report;
   upgrading to kimimaro skeletons is a follow-up, not part of this task.
4. **Precision bar is brutal.** L123–L125 measured joins needing ~0.95+ precision to pay and tip
   geometry ceilinged at 0.819. Absorption is cheaper than a bad anchor-anchor join, but abstention
   is still expected to dominate, and a near-zero-coverage honest result is a legitimate outcome —
   Stage 0's topology inventory is what makes it interpretable rather than a null.
5. **No frozen substantial-linker baseline exists**, so the honest number measures absorption on an
   unlinked backbone and understates deployable value; the mechanism-ceiling rows quantify by how
   much.
6. **Calibration honesty.** No independent Zebrafinch calibration volume with compatible
   annotations exists, so the full sweep is diagnostic and one setting is nominated for a future
   confirmation run.
7. **Concurrent repository activity.** The main checkout was already dirty at run start (75 status
   lines) and may be in use by another session. All deliverables are new files under a gitignored
   directory, so collision risk is low, but a concurrent tracked-file edit would trip CCC's
   pre/post mutation guard at code review.

## Changes Since Previous Plan Version

- **Fixed three wrong or vacuous geometry rules** (v1 finding C). `caliber_nm` is now the diameter
  `2·sqrt(volume/(π·length))` instead of v1's `nm²` quantity, which would have quarantined nearly
  every anchor; `endpoint_frac` is redefined on `[0,1]` with `1` at a tip and gated `>= 0.5`
  instead of the tautological `<= 0.5`; `tangent_cos` is `|u1_f·u1_a|` gated `>= 0.5`, which is
  invariant to principal-axis sign. Principal axis, centerline and endpoints are now explicitly
  constructed from second-moment accumulators added to Stage 0.
- **Defined every selector rule left open** (finding B): relative dimensionless margin with
  `runner_up = 0` for singletons; exact ties abstain at every grid point including `margin = 0.00`;
  quarantined anchors removed *before* ranking; `contact_area` declared a control with floor `0`;
  ranking order fixed as score desc, contact desc, `anchor_id` **asc**; and the anchor-floor
  variant folded into the artifact key so `1e6` and `3e5` cannot collide.
- **Made the affinity contract source-indexed** (finding D): Form 1 (`R[c,p]` for the pair
  `(p, p+e_c)`) is the only form the extractor uses; Form 2 states that ABISS output index `q`
  reads input `q − e`, with its valid domain and the low-face zero-pad rule. Added a test asserting
  agreement with the canonical helper's declared offsets.
- **Deferred multi-hop entirely** (finding E) rather than half-specify it, recording both reasons:
  Stage 1 has no fragment-fragment adjacency, and a Stage 3 precision gate would feed GT-measured
  precision back into selection.
- **Made the residual and escalation contracts complete** (finding F): Stage 1 emits
  `fragment_contact_summary.npz` covering every fragment including zero-contact ones, so Stage 2
  can enumerate them from frozen artifacts; the precedence order is reordered so
  `no_anchor_contact` is reachable and `missing_or_invalid_evidence` applies only to fragments that
  do have eligible anchors; endpoints now exist (Stage 0); the corridor test gains a tangent term
  and an explicit `>= 50%`-of-bucket escalation fraction.
- **Defined ceiling composition** (finding G): mechanism ceiling applies oracle joins first and
  maps frozen assignment targets through the oracle root without re-deriving assignments;
  oracle-clean-anchor **filters** frozen assignments without re-ranking, and is labelled a lower
  bound.
- **Resolved the firewall-ordering objection structurally** (finding A): `stage0_reproduce.py` is
  evaluator-side, emits only a gate plus already-published constants, and the firewall test asserts
  no `gt_free/` script reads anything under `evaluation_gt/`. The freeze artifact is now named
  (`assignments_<key>.frozen.json`) with `proposal_sha256` over the assignment bytes.
- Added the estimated-geometry risk introduced by the second-moment approach.
