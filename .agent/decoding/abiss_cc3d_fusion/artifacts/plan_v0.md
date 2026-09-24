# Plan v0

## Summary

Build a staged, resumable experiment under `dev/zebrafinch/abiss_cc3d_fusion/` that asks
whether chunk-local CC3D@0.70 over the arm0_96 affinity can improve the canonical arm0_96
ABISS whole-volume segmentation (funlib NERL 0.444376, mt=50) through a conservative
one-anchor, assignment-only ABISS LUT remap.

Reconnaissance already settles two Phase-0 questions that the task flags as blockers, and
they shape the whole plan:

1. **The affinity tag is `arm0_win144`.** `build_tier10_jobs.py` maps `arm0_win144` to
   `…/test_step=00200000/0/<BASE>.h5.chunks` and `arm0_native` to the sibling
   `<BASE>_win48x96x96.h5.chunks`. On disk the first store holds **726** chunk files and the
   second holds **10** (the tier10 chunks only), so a whole-volume decode could only have
   used the first. `wholevol_arm096_fullmask.yaml` independently confirms it: its `aff_path`
   is literally that 726-chunk store. The honest local control is therefore **0.775832**,
   not 0.8544. Phase 0 records this evidence rather than re-deriving a favourable row.

2. **Two different masks are in play, and they are not interchangeable.** The tier10 CC3D
   harness (`sweep_thr_chunk_v2.build_mask`) reads `ffn_tissue_mask_18-18-20.zarr`
   (5700, 5456, 5332), upsamples 2× in XY, and pads short border slices with **KEEP=1**.
   The arm0_96 ABISS run used `AFF_KEEP_MASK=tissue_border_keep_mask_full.zarr`
   (5700, 10912, 10664) — tissue **AND** border, at native XY. Reproducing the tier10 row
   requires the tier10 mask; making CC3D comparable to the ABISS partition requires the
   ABISS keep-mask, and border chunks past the mask extent must pad with **0 (drop)**, not
   KEEP=1, or the fusion invents CC3D foreground in the padded ring where ABISS has none.
   The plan uses each mask in exactly one role and records both in the manifest.

A third fidelity fact is recorded but deliberately not "corrected": the tier10 CC3D row
thresholds the **stored** h5 affinity (`fg = (aff > thr).any(0)`), while ABISS separately
applies `AFF_RESTORE_SIGMOID: 0.2` and `AFF_CONVENTION: banis`. "0.70" therefore means 0.70
on stored values in both the recorded row and this experiment. Silently switching to
restored affinity would break comparability with the number this task is built on.

The work is ordered so the cheap decisive measurement comes first. Phase 1 is an
evaluator-only complementarity audit that only needs CC3D at test-50 skeleton nodes, so it
streams the ~300–400 node-bearing chunks rather than all 726. **Gate A** then decides
whether the expensive full-population passes happen at all. A rigorous negative result at
Gate A is a complete run.

## Scope

In scope:

* Phase 0 input-identity manifest and six fidelity gates.
* Phase 1 whole-skeleton complementarity audit, expected-negative label-algebra controls,
  GT ceilings, and the three directional candidate oracles.
* Gate A decision, recorded either way.
* If Gate A passes: Phase 2 streaming GT-free evidence graph over all 726 chunks, Phase 3
  one-anchor assignment policy with the fixed six-step ablation ladder, Phase 4
  split/cross-chunk diagnostics as reports.
* Evaluator-only NERL scoring of frozen policies, composition with the existing
  `frozen_endpoint_merges` map, and the fusion funnel report.
* CPU tests covering the twelve required cases.

Out of scope (task's list, restated as build constraints): no production decoder, config, or
`connectomics.decoding` change; no dependency additions; no CC3D threshold sweep; no dense
whole-volume output; no retraining; no modification of canonical affinities, masks, LUTs, the
ABISS segmentation, or prior reports; no commits during the CCC run.

Explicit non-goal: this plan does not promise a positive NERL delta. It promises a
measurement whose sign is trustworthy.

## Proposed Changes

### Phase 0 — input identity and fidelity gates

`phase0_input_audit.py` writes `gt_free/input_manifest.json` containing: the affinity
directory, its `…h5.index.json` (`input_shape [5700,12288,12288]`, `chunk_shape [1008]*3`,
`halo 72`, `checkpoint_path …step=00200000.ckpt`, 726 chunk records with `start_zyx`/
`stop_zyx`), per-store file counts (726 vs 10) as the `arm0_win144` resolution evidence, the
`wholevol_arm096_fullmask.yaml` `aff_path`/`AFF_KEEP_MASK`/`AFF_RESTORE_SIGMOID` provenance,
dtype/channel/shape probes on a sample of chunks including the clipped border chunk, stable
file metadata (size + mtime + sha256 of a fixed byte window, full hashes only for the few
probe chunks), the ABISS CloudVolume URI/info/mip/voxel size, and the frozen CC3D recipe
(`threshold 0.70` on stored values, `connectivity 26`, `fg=(aff[:3]>thr).any(0) & keep`,
mask path, `(chunk_id, local_label)` namespacing convention). Skeleton and LUT paths go only
into `evaluation_gt/evaluator_manifest.json`.

Gates, each recorded pass/fail with the measured value:

| # | gate | method |
|---|---|---|
| 1 | ABISS NERL 0.444376 | `score_lut(graph, lut, 50)` on `reports/arm096_lut/s*.npz`, `abs(x-0.4443760423975249) <= 1e-10` |
| 2 | tier10 CC3D@0.70 row | re-aggregate the ten `reports/*_HUMANGT_t10_arm0_win144_*.csv` rows with the recorded aggregation, and recompute ≥2 chunks (including border `z2_y10_x7`) end-to-end with our code, matching `n_labels`, `fg_frac`, `base_nerl_funlib` exactly |
| 3 | affinity fidelity | direct `h5py` reads at asymmetric probe voxels vs the loader; assert channel order and dtype |
| 4 | ABISS/CC3D alignment | known-voxel round trip `start_zyx → CloudVolume[x,y,z]`, plus displaced controls (±1 voxel, axis-swapped) that must fail |
| 5 | mask contract | ABISS keep-mask crop shape/keep-fraction per chunk; border chunks past 10912/10664 pad with 0; assert the tier10 mask is used only in gate 2 |
| 6 | namespacing | identical local IDs from two chunks never compare equal after packing |

If gate 1, 2, 4, or 5 fails the run stops; a gate-2 failure is a blocker, not a licence to
switch tags.

### Phase 1 — evaluator-only complementarity audit (cheap, decisive)

`phase1_audit.py` loads the skeleton graph and the cached arm0_96 node LUT, maps every node
voxel to `(chunk_key, local_zyx)` via the index, and streams **only node-bearing chunks**.
For each: read affinity, apply the ABISS keep-mask crop, compute CC3D@0.70 in memory, sample
labels at that chunk's node voxels, pack to `uint64` as `(chunk_ordinal << 32) | local_label`
(0 stays 0), discard the dense array. Chunk selection is GT-informed, which is legitimate
because this whole stage is an evaluator-only diagnostic written under `evaluation_gt/`.

It scores the seven required rows by importing — not copying — the pure helpers already in
`dev/zebrafinch/cc3d/compare_cc3d_abiss.py` (`factorize_pairs`, `namespace_partitions`,
`namespace_fill`, `sampled_partition_join`, `per_skeleton_nerl`, `funlib_weights`,
`aggregate_nerl`, `score_partition`, `visible_merge_skeletons`, `gt_routed_mixture`) plus
`branch_merge` from `oracle_cc3d_chunked`. Those functions are already parameterised and
side-effect-free; the pretrain defaults in `main()` and its argparse surface stay untouched.

It then computes the three **directional candidate oracles**, each restricted to operations a
deployable policy could express, reporting candidate count, base rate, precision, obtainable
headroom, and the cost of taking every candidate:

* *assign*: for each shared CC3D component with exactly one established ABISS anchor, assign
  the co-occurring smaller ABISS labels to that anchor — GT only decides correctness, never
  candidate membership;
* *face-join*: CC3D face pieces joined only where ABISS/global overlap nominated them;
* *cut*: ABISS labels cut along CC3D boundaries only inside independently flagged suspects.

Output: `evaluation_gt/partition_audit.json` + a concise markdown table.

**Gate A** passes if some directional family shows ≥ +0.01 candidate-restricted NERL oracle
gain, or recovers a known false merge / long-backbone break whose value NERL under-reports.
If the only positive rows are GT decoder routing or unrestricted meet reassembly, the run
stops and reports that as the finding.

### Phase 2 — streaming full-population GT-free evidence (only if Gate A passes)

`phase2_evidence.py` is a resumable shard worker over all 726 chunks: read affinity → CC3D
in memory → read the spatially matching ABISS crop from CloudVolume → accumulate compact
per-component statistics, ABISS×CC3D contingencies, per-chunk ABISS label stats, and
compact face slabs → write one NPZ shard + `.done` record → discard dense arrays. No dense
CC3D chunks are written. `reduce_evidence.py` refuses missing or duplicate chunks, builds the
**global** ABISS segment table (`gt_free/abiss_segment_stats.npz`: voxels, bbox, chunk span,
face-touch, anchor-quality fields), deduplicates face evidence deterministically, and emits
`gt_free/overlap_candidates.npz` with every component classified `zero_anchor` / `one_anchor`
/ `multi_anchor` / `anchor_only` / `invalid_or_boundary` under predeclared size/extent bands
(dust / mid / established), with band sensitivity reported rather than tuned.

Per-component fields as listed in the task, including affinity summaries at each proposed
ABISS transition, persistence at the stricter 0.75, and boundary flags.

### Phase 3 — one-anchor assignment policy

`phase3_policy.py` scores candidates from cached features only (no volume re-reads) and runs
the fixed ablation ladder 1–6 from the task. Global resolution uses union-find over the whole
ABISS label graph with the invariant asserted before acceptance: **no component may contain
two established anchors**, one hop only, deterministic best/runner-up with an explicit
abstain state and reason codes. Conflicting or order-dependent claims abstain. Outputs
`gt_free/policy_<name>.json` and `gt_free/assignments_<name>.npz`, each stamped
`gt_free: true`, `frozen_before_evaluation: true`, plus a sha256 recorded before evaluation —
the same freeze contract `arm096_evaluate_frozen.py` already enforces.

### Phase 4 — diagnostics only

`phase4_diagnostics.py` emits split proposals for independently flagged suspects and
cross-chunk face-continuation diagnostics, reusing the concepts in `build_face_iou_graph.py`.
No dense split is realised unless Gate C passes.

### Evaluation

`evaluate_frozen.py` consumes a frozen assignment file, applies it as a global ABISS
label→label map to the cached node LUT, and reports every quantity in the task's evaluation
contract, in three separate tables (frozen GT-free / expected-negative controls / GT
ceilings). It also scores the composition matrix against the existing
`arm096_error_correction/decoder_gtfree/frozen_endpoint_merges.npz` (340 pairs,
+0.000129, improved [17], regressed [14, 3]): ours alone, theirs alone, both orders, and
label conflicts, so the same fragment is never counted under two task names. Only the frozen
GT-free table may be compared with FFN 0.538003.

## Files and Areas

New, all under `dev/zebrafinch/abiss_cc3d_fusion/`:

| file | role |
|---|---|
| `README.md` | exact commands, env, stage status, conclusions |
| `common.py` | paths, chunk index, both mask readers, CloudVolume ABISS reader, CC3D recipe, `(chunk, label)` packing |
| `phase0_input_audit.py` | manifest + six gates |
| `phase1_audit.py` | evaluator-only audit, controls, ceilings, directional oracles, Gate A |
| `phase2_evidence.py` | resumable per-chunk GT-free shard worker |
| `reduce_evidence.py` | deterministic reduction, global ABISS table, candidate classification |
| `phase3_policy.py` | scoring, global union-find resolution, frozen policy + assignments |
| `phase4_diagnostics.py` | split proposals, cross-chunk face diagnostics |
| `evaluate_frozen.py` | evaluator-only NERL, funnel, composition matrix |
| `tests/test_*.py` | the twelve required cases |
| `sbatch_phase1.sh`, `sbatch_phase2.sh` | cluster drivers |

Read-only dependencies (imported or read, never modified): `cc3d/compare_cc3d_abiss.py`,
`oracle_cc3d_chunked.py`, `arm096_endpoint_oracle.py` (`load_graph_and_lut`, `score_lut`),
`cc3d_chunks_tissue.py` (coordinate/mask logic reference), `build_face_iou_graph.py`,
`sweep_thr_chunk_v2.py` (`build_mask` semantics for gate 2), `reports/arm096_lut/s*.npz`,
`connectomics.metrics.nerl.import_em_erl`.

Environment: `source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc`,
`export HDF5_USE_FILE_LOCKING=FALSE`, run from the repo root with `sys.path` inserts (the
package is not pip-installed).

## Verification Plan

Focused CPU tests, all synthetic or single-chunk, run with
`conda run -n pytc python -m pytest dev/zebrafinch/abiss_cc3d_fusion/tests -q`:

1. `fg=(aff[:3]>0.70).any(0)` semantics and 26-connectivity on a hand-built array whose
   6-, 18-, and 26-connectivity answers differ.
2. Clipped border-chunk mask alignment: a chunk extending past 10912/10664 pads with 0 under
   the ABISS contract and with 1 under the tier10 contract; both asserted.
3. Identical local CC3D IDs in two chunks stay distinct after packing, and unpacking is exact.
4. Overlap counts are identical for 1, 3, and 7 shards, and independent of chunk order.
5. One component with one anchor + fragments emits assignment candidates.
6. One component with two anchors abstains, in every band.
7. A transitive fragment chain that would connect two anchors abstains.
8. One ABISS label split into several CC3D components is unchanged by the primary policy.
9. Face candidates on a synthetic object crossing a chunk boundary are deduplicated and
   identical across shard counts and processing orders.
10. GT-derived paths (skeletons, `arm096_lut`, `oracle_gt/`) are rejected by GT-free stages.
11. LUT remap equals dense relabel on a small synthetic volume.
12. `cc3d/tests/test_compare_cc3d_abiss.py` still passes unchanged.

Data-side verification, in order:

1. Phase 0 gates 1–6, all six recorded.
2. Synthetic + one-border-chunk (`z2_y10_x7`) smoke run.
3. Tier10 CC3D@0.70 reproduction for `arm0_win144` and ABISS 0.444376 reproduction.
4. Phase 1 audit over node-bearing chunks; Gate A decision written either way.
5. If Gate A passes: all 726 chunks with completeness + determinism checks.
6. Policy rescoring from cache reproduces identical assignments (byte-identical NPZ).
7. Freeze (sha256), then one canonical whole-volume test-50 evaluation per policy.
8. Leakage audit: every `gt_free/*` manifest scanned for skeleton/owner/NERL references.
9. Skipped stages and the exact gate that stopped them recorded in `README.md`.

Cluster hygiene: Phase 1/2 shards are capped at ≤16 concurrent and never co-scheduled with
another affinity-streaming job — whole-volume affinity reads saturate `/projects` and put
unrelated read-heavy jobs into D-state. Each SLURM stage prints its output path and a
per-chunk liveness line so a silent death is visible.

## Risks and Questions

* **I/O is the dominant cost, not compute.** Each chunk is 3 × 1008³ float16 ≈ 6.1 GB, so a
  full 726-chunk pass reads ≈ 4.4 TB. Phase 1's node-bearing subset cuts its own pass to
  roughly half that. This is why Gate A runs before Phase 2, and why shard concurrency is
  capped. If Gate A passes, Phase 2 is a multi-hour cluster stage.
* **Phase 1 may reuse its per-chunk evidence for Phase 2.** Content is GT-independent even
  though chunk *order* is GT-informed; test 4 (order/shard independence) is what makes that
  safe, and `reduce_evidence.py` still requires 726/726 before finalising.
* **Expected outcome is a negative or small result.** The prior pretrain fusion controls all
  failed, and the companion `frozen_endpoint_merges` map gained +0.000129. Precision is the
  binding constraint (~8 correct joins per false merge). The plan is built so a null is
  reported cleanly rather than rescued.
* **The ABISS/CC3D domain mismatch is a real confound.** ABISS decoded under
  tissue+border; using a tissue-only mask for CC3D would manufacture `zero_anchor`
  components in the border ring. Handled by using the ABISS keep-mask everywhere except the
  tier10 reproduction, but the two-mask split must be checked in review.
* **CloudVolume XYZ vs affinity ZYX** is the highest-risk silent bug; gate 4's displaced
  controls exist because a wrong-but-plausible mapping would produce a full, meaningless
  candidate graph.
* Open question for the reviewer: if Gate A passes only on the *cut* family (not *assign*),
  the task routes that to Gate C and the nuclei/glia work. This plan treats that as a
  report-only outcome and does not build a dense splitter.

## Changes Since Previous Plan Version

Initial plan.
