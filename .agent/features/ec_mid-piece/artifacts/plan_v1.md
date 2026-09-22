# Plan v1

## Summary

Plan v0's central compromise — deriving the candidate scope from the GT LUT — is withdrawn. It
rested on a cost claim that I have now measured and found wrong by roughly an order of magnitude
in interpretation, so the honest full-population pass the task actually requires is affordable and
is what this plan specifies.

Two measurements taken since v0 make the plan concrete rather than aspirational:

**1. The affinity truth table is verified, not asserted.** Reading the arm0_96 affinity source
(`chunk_z*_y*_x*.h5`, dataset `main`, `(3, 1008, 1008, 1008)` float16 gzip, no halo in file) and
comparing elementwise against `lib/abiss/scripts/volume_backends.py::_ArrayVolume` with
`convention='banis', restore_sigmoid_scale=0.2` — the exact code path that produced arm0_96 —
gives an exact match on all three channels with `maxdiff = 0.0`, and all four wrong conventions
fail:

```
abiss ch0 (=x) == A[2] shifted by -e2 : True   maxdiff=0.000e+00
abiss ch1 (=y) == A[1] shifted by -e1 : True   maxdiff=0.000e+00
abiss ch2 (=z) == A[0] shifted by -e0 : True   maxdiff=0.000e+00
negative controls (all False): no shift | channel c=c' | wrong-axis shift | +1 instead of -1
```

**The contract, therefore, measured:** for the raw h5 array `A` of shape `(3, Z, Y, X)`,
`A[c, z, y, x]` is the affinity of the edge between array voxel `(z, y, x)` and
`(z, y, x) + e_c`, with `e_0 = +Z`, `e_1 = +Y`, `e_2 = +X`; values require
`restore_sigmoid(a, 0.2) = sigmoid(logit(a)/0.2)` then clip to `[0, 1]`. `aff_absorb.py`'s
`2 - ax` is *wrong for this artifact* — it is the `channel c=c'` negative control, which fails.

**2. The full-population pass costs hours, not weeks.** Measured throughput on this filesystem:
affinity `0.14 GB/s` decompressed (43 s per 1008³ chunk, 726 chunks = 8.7 h single-stream) and
segmentation `0.086 GB/s` via CloudVolume (96 s per chunk region, 19.3 h single-stream). Capped at
8 concurrent readers the combined pass is **≈3.5 h wall**. That is an ordinary non-interactive
sharded job, so there is no reason to restrict the candidate population at all.

The design is therefore a clean two-pass, fully GT-free extractor over all 726 chunks, followed by
one global resolver and a strictly post-freeze evaluator. Phase 3 (multiscale) is declared **not
run** in this task, because the only available larger-context artifact is `r10_minpool` and
`task.md` forbids reusing it without the affinity-TTA contract, which is explicitly out of scope.

## Scope

Unchanged from v0 in extent: everything lives in `dev/zebrafinch/ec_mid_piece/`; no package code,
no dependency, no commit, no canonical input modified, no new segmentation volume written.

Removed from scope in v1: the GT-informed scope selector, `evaluation_gt/scope_segment_ids.txt`,
the scope-invariance argument, and Phase 3 execution.

Operative repository root is the main checkout `/projects/weilab/weidf/lib/pytorch_connectomics`
(branch `codex/gt-free-tube-analysis`), not the bridge worktree, because `dev/` and `.agent/` are
gitignored and exist only there. Every deliverable is gitignored, so tracked-file git diffs stay
empty and code review is served the new files directly.

## Proposed Changes

### `affinity_io.py` — verified BANIS reader

Store: `outputs/nisb_base_banis_plus_zebrafinch_heavy/20260726_114349/test_step=00200000/0/`
`raw_x1_ch0-1-2_chunked-raw_cs1008x1008x1008_halo72x72x72_zebrafinch_chunk_raw_grid1008_halo72.h5.chunks`,
files `chunk_z{Z}_y{Y}_x{X}.h5`, dataset `main`, `(3,1008,1008,1008)` float16, gzip, chunks
`(3,64,64,64)`. Grid is `6 x 11 x 11 = 726` over the `(5700, 10912, 10664)` ZYX volume; chunk
`(Z,Y,X)` covers `[1008*Z : min(1008*(Z+1), dim)]` per axis. **There is no halo inside the file**
— the `halo72` in the name was trimmed at write time. The reader asserts the observed dataset
shape and the 726-file grid, and aborts on mismatch rather than mis-registering.

API: `read_slab(zyx_start, zyx_stop) -> float32 (3, dz, dy, dx)` applying `restore_sigmoid(·,0.2)`
then `clip(0,1)`, then the keep-mask (`dev/zebrafinch/tissue_border_keep_mask_full.zarr`, uint8
ZYX `(5700,10912,10664)`, chunks `(126,252,252)`; masked → affinity 0, matching the decode); and
`edge_affinity(slab, axis) -> slab[axis, ...]` documented with the truth table above. Reads that
cross a chunk boundary are stitched by the reader, with a 1-voxel low-side overlap so no edge is
lost at a seam.

### Stage 0 — full-population GT-free inventory + evaluator-side reproduction

`stage0_inventory.py` (**GT-free**, argument surface has no skeleton/LUT/owner option and it opens
none): streams the segmentation only, in Z-slabs of 64 planes with a 1-plane halo, over all 726
chunk columns, accumulating per segment: voxel count, bbox, centroid (running sums), per-face
volume-border-touch flags, and the six-neighbour RAG degree. Emits
`gt_free/segment_inventory.npz` over the **entire segment population**. Cost ≈2.4 h at 8-way.
Derived GT-free descriptors: `extent = bbox diagonal`, `elongation = max_extent/min_extent`,
`est_centerline_nm ≈ bbox principal-axis length`, `caliber_nm = voxels·1620 / max(est_centerline_nm, 1)`
(voxel volume `9·9·20 = 1620 nm³`), all computed in this script.

`stage0_reproduce.py` (**evaluator-only**): loads graph+LUT via
`arm096_endpoint_oracle.load_graph_and_lut`, asserts
`score_lut(..., merge_threshold=50) == 0.4443760423975249` within `1e-10`, and reproduces the L126
partition using `analyze_arm096.py`'s per-`(skeleton, label)` piece definition — target
`668 / 1887 / 8979` pieces and `87.9 / 7.65 / 4.11` mass percent, asserting counts exactly and
mass within `0.05` pp. (A global per-label count instead gives `665 / 1889 / 8854` and
`88.21 / 7.68 / 4.10`; the per-skeleton definition is the correct one and the script must not
silently use the other.) Aborts naming the mismatching quantity.

**Predeclared, GT-free size bands** — fixed before any GT is read, reported post-hoc against the
oracle population rather than fitted to it:

```
anchor:   voxels >= 1e6            (primary)   ; 3e5 as a declared sensitivity variant
fragment: voxels <  anchor floor, partitioned for reporting into
          F1 [1, 1e3)  F2 [1e3, 1e4)  F3 [1e4, 1e5)  F4 [1e5, anchor_floor)
```

**Predeclared GT-free anchor quarantine** — an anchor is quarantined if it touches the volume
border on two opposite faces, **or** `voxels > 5e8`, **or** `caliber_nm > 1500`. Quarantined
anchors receive nothing in the honest policy.

Stage 0's first report is the topology inventory and the experiment's first decision gate:
fragments partitioned into zero eligible anchor contacts / exactly one / multiple / only
quarantined.

### Stage 1 — frozen candidate edge table (GT-free, full population)

`stage1_candidate_edges.py`, sharded by chunk (`--shard i --nshard N`, `N<=8`). Per chunk, stream
Z-slabs of seg + affinity with a 1-voxel halo, and for every face-adjacent voxel pair whose two
labels are `(fragment-band, anchor-band)` accumulate into a `(fragment_id, anchor_id)` record:
`contact_voxels`, per-axis contact counts, contact bbox and centroid, and affinity accumulators
(`sum`, `sum²`, `min`, `max`, plus a 64-bin fixed histogram on `[0,1]` from which `p10/p50/p90` are
derived exactly enough at 1/64 resolution). Accumulators, not per-voxel rows, so the table stays
bounded by the fragment population.

Chunk-boundary handling: each contact voxel pair is owned by the chunk containing its **low-side**
voxel, which makes the halo overlap idempotent by construction; the owning chunk id is recorded
for provenance and a test asserts one-box and two-box extraction agree.

A global merge step sums accumulators across shards and joins the Stage 0 inventory to add
`caliber_ratio = min(cal_f, cal_a)/max(cal_f, cal_a)`, `tangent_cos` (principal-axis cosine, when
both extents are elongated enough to define one), `endpoint_frac` (contact centroid position along
the fragment's principal axis, `0` = tip, `0.5` = middle), `n_anchor_candidates`, anchor
descriptors, and a `valid_*` flag per evidence source. `best`/`runner_up`/`margin` are **not**
stored — Stage 2 computes them. Output `gt_free/candidate_edges.npz`, sha256 recorded in
`gt_free/input_manifest.json` together with every input path, the affinity convention string, the
verified truth table, `restore_sigmoid=0.2`, the keep-mask path, resolution, and `gt_free: true`.

### Stage 2 — frozen policy resolution (GT-free; reads the candidate table only, never the LUT)

`stage2_resolve.py`. It computes **no NERL and opens no GT artifact**; it turns the frozen table
into frozen assignment files. All scoring against GT happens in Stage 3.

Policies, exactly the six in `task.md`:

| name | score | gate |
|---|---|---|
| `none` | — | baseline, empty assignment |
| `contact_area` | `contact_voxels` | negative control |
| `aff_mean` | `aff_mean` | floor |
| `aff_p90` | `aff_p90` | floor |
| `aff_p90_margin` | `aff_p90` | floor + margin |
| `aff_p90_margin_morph` | `aff_p90` | floor + margin + morphology |

Predeclared grid: floor `{0.5, 0.6, 0.7, 0.8, 0.9}` x margin `{0.00, 0.02, 0.05, 0.10, 0.20}`,
where margin is `best − runner_up` on the policy's own score.

Morphology gate (hard indicator, no learned weights): `caliber_ratio >= 0.25` **and**
`endpoint_frac <= 0.5` **and** (`tangent_cos >= 0.0` when defined, else pass).

Invariants, each with a test: exactly one anchor per accepted fragment; an anchor's own label never
changes; **no anchor-anchor union** — after applying assignments every component contains exactly
one anchor, asserted, and a violation aborts; an assigned fragment cannot itself be an anchor;
quarantined anchors receive nothing; deterministic tie-break by `(score, contact_voxels, anchor_id)`
with a stable sort. `reciprocal_best` (f's best anchor is a, and f is a's best fragment) is
computed and recorded as a field, reported but not gated on in the primary policy.

Multi-hop is conditional on one-hop clearing the precision gate: depth cap 2, relay paths recorded,
resolved on the complete recorded segment graph before any scoring, one anchor per component
enforced, relay chains that would connect two anchors rejected.

Outputs per grid point: `gt_free/policy_manifest_<policy>__floor<F.FF>_margin<M.MM>.json`
(complete rule + parameter values + `implementation_sha256`),
`gt_free/assignments_<policy>__floor<F.FF>_margin<M.MM>.npz` carrying `gt_free: true`,
`frozen_before_evaluation: true`, and a sibling `.json` recording `proposal_sha256` —
mirroring the freeze contract `arm096_evaluate_frozen.py` already enforces.
`gt_free/residual_<...>.npz` carries one reason per unresolved fragment under this fixed
precedence, first match wins:

```
missing_or_invalid_evidence -> no_anchor_contact -> only_quarantined_anchor
-> geometry_conflict -> weak_affinity -> ambiguous_multiple_anchors
-> multiscale_contradiction   (never emitted in this run; Phase 3 is not run)
```

### Stage 3 — post-freeze evaluation (evaluator-only)

`stage3_evaluate.py`, modelled on `arm096_evaluate_frozen.py`: refuses to run unless the
assignment sha256 matches its pre-evaluation report and both freeze markers are set. Then loads
graph+LUT, re-asserts the baseline, applies assignments, scores.

`apply_assignments(lut, assignments)` is a new **directed** helper, not
`arm096_endpoint_oracle.relabel` — `relabel` is union-find over unordered pairs and would happily
union two anchors. It maps fragment label → anchor label, asserts anchors are fixed points, and
asserts one anchor per induced component.

Reported per grid point: NERL and delta from its direct baseline; fraction of the `+0.138303`
L126 increment recovered; per-skeleton deltas with improved/unchanged/regressed counts and the max
single-skeleton regression; accepted fragments and components; correct-host precision and GT-mid
coverage **with a Wilson 95% score interval and the sample count**; wrong-host assignments,
cross-neuron unions and multi-owner components at both the evaluator's `mt=50` materiality and at
zero tolerance; stratification by contact class, band, contact type, anchor quality and margin
bin; full risk/coverage curves; wall time and peak memory per stage.

Baselines, kept in separate tables:

- **Honest GT-free baseline = raw arm0_96 `0.4443760423975249`.**
  `arm096_error_correction/decoder_gtfree/` contains only `segment_skeleton_graph.h5` and
  `chunks/`; the `frozen_endpoint_merges.npz` that `arm096_evaluate_frozen.py` defaults to does
  **not exist**, so no frozen autonomous substantial-linker artifact is available. Per `task.md`
  the run proceeds on raw arm0_96 and states that it measures mid-piece absorption *before* the
  substantial-linking stage exists.
- **Mechanism ceilings (labelled ceilings, never compared to FFN):** L123's 277 frozen oracle
  joins from `oracle_gt/gt_nominated_pairs.json`, *and* the stronger L126 `>=50-owner-node` rung.
  Both are LUT-only and cheap, so v1 reports both rather than choosing.
- **Oracle-clean-anchor ceiling:** quarantine the three known ERL-visible cross-neuron labels
  (`73465106696008740`, `72269800654646543`, `72198881818968095`). Diagnoses lost potential; does
  not alter the honest assignment file.

Only the honest row is placed next to FFN `0.538003`. Every grid point is reported as a
diagnostic sweep; one setting is *nominated* for a future confirmation run and no test-selected
point is called validated, since no independent Zebrafinch calibration volume with compatible
annotations exists.

### Stage 4 — residual taxonomy and funnel

`stage4_residual.py` reports the share of the L126 mid increment in each reason bucket and the
funnel: candidate coverage → quarantine → ranking ambiguity → confidence gate → residual
no-contact. Predeclared thresholds so the recommendation is reproducible: **material share = ≥20%**
of the `+0.138303` increment; a `no_anchor_contact` fragment has **corridor evidence** when its
nearest-anchor endpoint distance is `<= 2000 nm` **and** the median restored affinity sampled along
the straight corridor is `>= 0.3`. Escalation to curve fitting / bidirectional tracking / Hungarian
assignment is recommended only when both hold; otherwise the report states that ownership or
anchor quarantine is the binding constraint.

### Phase 3 — declared not run

The only larger-context artifact on disk is `r10_minpool`, which `task.md` forbids reusing without
`.agent/features/affinity_tta/task.md`, itself explicitly out of scope. Rather than substitute a
home-grown geometry gate, v1 records the gate decision and its reason in
`evaluation_gt/results.md` and leaves the `multiscale_contradiction` reason code unused. No
deleted r10 artifact is recreated.

## Files and Areas

| Path | Purpose |
|---|---|
| `dev/zebrafinch/ec_mid_piece/README.md` | commands, environment, wall time, peak memory, resume |
| `affinity_io.py` | verified BANIS reader, restore-sigmoid, keep-mask, edge accessor |
| `stage0_inventory.py` | GT-free full-population segment inventory (no GT argument surface) |
| `stage0_reproduce.py` | evaluator-side baseline + L126 reproduction |
| `stage1_candidate_edges.py` | sharded GT-free candidate edge extraction + global merge |
| `stage2_resolve.py` | frozen policies, assignments, residual (no LUT access) |
| `stage3_evaluate.py` | post-freeze GT evaluation |
| `stage4_residual.py` | residual taxonomy, funnel, recommendation |
| `visual_probes.py` | crop manifest for correct/wrong/ambiguous/contaminated/no-contact cases |
| `tests/test_*.py` | affinity truth table, chunk dedup, one-anchor invariant, determinism, firewall, LUT eval |
| `gt_free/`, `evaluation_gt/` | artifacts named exactly as `task.md` requires |

Read-only and unmodified: `reports/arm096_lut/`, `reports/arm096_{error_structure,breaks_v2}.json`,
`arm096_error_correction/oracle_gt/`, the arm0_96 precomputed segmentation, the affinity chunk
store, `tissue_border_keep_mask_full.zarr`, `/projects/weilab/dataset/zebrafinch/test_50_skeletons.h5`.
Imported unmodified: `arm096_endpoint_oracle.{load_graph_and_lut, score_lut}`,
`connectomics.metrics.nerl.import_em_erl`.

## Verification Plan

Environment: `source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc`.

1. **Affinity truth table** — port the verification already run for this plan into
   `tests/test_affinity_io.py`: read a real crop from `chunk_z2_y5_x5.h5`, build
   `_ArrayVolume(blk, convention='banis', restore_sigmoid_scale=0.2)`, and assert for each ABISS
   channel `c'` that `abiss_out[..., c']` transposed to ZYX equals `R[2-c']` shifted by `-e_{2-c'}`,
   with `maxdiff == 0`. Assert all four negative controls (no shift, `channel c=c'`, wrong-axis
   shift, `+1` instead of `-1`) are **not** equal. This is the decisive test: it compares against
   the code that produced arm0_96, so any swap, flip or one-voxel error fails elementwise. A
   synthetic asymmetric volume is added for interpretability, not as the primary check.
2. **Chunk-boundary dedup** — synthetic object straddling a 1008 boundary yields identical contact
   statistics extracted as one box or as two overlapping shards; plus an assertion that
   low-side ownership makes the halo idempotent.
3. **One-anchor invariant / determinism** — property tests on synthetic graphs including two
   anchors contending for one fragment and a relay chain that would bridge two anchors; rerunning
   Stage 2 from the cached table reproduces byte-identical assignment files.
4. **GT firewall** — audit test asserting no `gt_free/` script exposes a skeleton/LUT/owner CLI
   option or opens `test_50_skeletons.h5`, `reports/arm096_lut`, or `oracle_gt/`; plus a manifest
   audit that every `assignments_*.npz` carries both freeze markers and matches its recorded hash.
5. **LUT-level evaluation** — `score_lut` reproduces `0.4443760423975249` within `1e-10`;
   `apply_assignments` with an empty map is the identity; a hand-built two-anchor assignment is
   rejected by the invariant rather than silently merged.
6. **Smoke run** on a small crop exercising all three affinity axes and a chunk border, before the
   sharded full pass.
7. **Reproduce** the baseline and L126 inventory (Stage 0) before any policy is resolved.

Operational: shards capped at 8 concurrent readers and chained rather than co-scheduled with other
affinity jobs — `/projects` has previously saturated under parallel affinity streaming and blocked
unrelated read-heavy jobs in D-state. Every stage is resumable from its per-shard cache and prints
a heartbeat; the README records the non-interactive resume command, measured wall time and peak
memory. Expected: Stage 0 ≈2.4 h, Stage 1 ≈3.5 h at 8-way, Stages 2-4 minutes.

## Risks and Questions

1. **Edge-table size is the main remaining unknown.** The segmentation is dense (99.4% foreground
   in a probe), so the number of fragment-anchor adjacencies is not yet bounded. Mitigation: only
   `(fragment-band, anchor-band)` pairs are emitted and they are stored as accumulators, not
   voxel rows. Stage 1 prints running edge counts, and if the merged table would exceed ~200 M
   rows the run stops and reports rather than silently sampling.
2. **The band ladder is coarse and predeclared.** `anchor >= 1e6` voxels is a judgement made
   without GT; it may not align with the oracle's `10-49 GT node` population. That misalignment is
   a *result to report*, not a knob to retune — the post-hoc table states which band captured the
   oracle mid pieces, and the `3e5` sensitivity anchor floor bounds how much the choice mattered.
3. **Precision bar is brutal.** L123-L125 measured that joins need ~0.95+ precision to pay and that
   tip geometry ceilings at 0.819. Absorbing a mid piece into a wrong host is cheaper than a bad
   anchor-anchor join, but abstention is still expected to dominate. A near-zero-coverage honest
   result is a legitimate outcome; Stage 0's topology inventory is what makes it interpretable.
4. **No frozen substantial-linker baseline exists**, so the honest number measures absorption on
   an unlinked backbone and understates deployable value; the mechanism-ceiling rows quantify by
   how much, and the report must not present the honest row as the pipeline's final capability.
5. **Calibration honesty.** No independent Zebrafinch calibration volume with compatible
   annotations exists, so the full sweep is reported as diagnostic and one setting is nominated for
   a future confirmation run.
6. **Concurrent repository activity.** The main checkout was already dirty at run start (75 status
   lines) and may be in use by another session. All deliverables are new files under a gitignored
   directory, so collision risk is low, but a concurrent tracked-file edit would trip CCC's
   pre/post mutation guard at code review and block the run.

## Changes Since Previous Plan Version

- **Removed the GT-informed scope selector entirely** (review findings 1 and 2). Deleted
  `evaluation_gt/scope_segment_ids.txt`, `scope_manifest.json`, and the scope-invariance argument.
  Stage 0 and Stage 1 now run over the **full segment population** across all 726 chunks. The v0
  premise that this was unaffordable was wrong: measured throughput gives ≈2.4 h (Stage 0) and
  ≈3.5 h (Stage 1) at 8-way concurrency.
- **Replaced the affinity verification with a measured truth table** (finding 4). The contract is
  now stated exactly, was verified elementwise against `_ArrayVolume` with `maxdiff = 0` and four
  failing negative controls, and that verification becomes the primary test. The weak
  "boundary lower than interior" probe is dropped. The chunk shape/halo question is settled: the
  file is `(3,1008,1008,1008)` with **no** in-file halo.
- **Specified every previously undefined contract** (finding 3): numerical anchor/fragment bands
  and the `3e5` sensitivity variant; anchor eligibility; the three quarantine rules with values;
  the morphology gate as an explicit hard indicator rather than an invented weighting;
  reciprocal-best defined and reported but not gated; residual-reason precedence as a fixed
  first-match-wins order; and the grid-point assignment file naming scheme.
- **Declared Phase 3 not run** (finding 5), with the reason recorded in the report, rather than
  substituting a home-grown geometry gate for the out-of-scope affinity-TTA contract.
- **Made the reporting rules reproducible** (finding 6): Wilson 95% score interval with sample
  count for precision; material share `>=20%` of `+0.138303`; corridor evidence defined as
  endpoint distance `<= 2000 nm` and median corridor affinity `>= 0.3`.
- **Removed the "pure LUT-level rescore" ambiguity** (finding 7): Stage 2 is renamed
  `stage2_resolve.py`, computes no NERL and opens no GT artifact; all scoring is Stage 3.
- **Settled v0's open question**: both mechanism-ceiling rungs (L123's 277 joins and the L126
  `>=50-owner-node` rung) are reported, since both are LUT-only and cheap.
- Added the edge-table size risk and its stop condition, which the full-population change
  introduced.
