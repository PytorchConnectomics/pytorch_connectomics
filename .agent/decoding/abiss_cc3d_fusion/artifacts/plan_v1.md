# Plan v1

## Summary

Same experiment, restructured to satisfy all ten `plan_v0_review` findings. Three changes are
structural rather than editorial.

**1. The dense CC3D store now exists, so the cheap/expensive tradeoff that shaped plan_v0 is
gone.** SLURM array 2855306 is writing chunk-local CC3D@0.70 for all 726 chunks of the arm0_96
affinity into `…grid1008_halo72.cc3d_t070_tissue.chunks` (~50 GB by the pretrain `t066`
precedent, ~3 h). Full-population GT-free evidence extraction therefore reads a ~50 GB label
store plus the ABISS volume instead of re-streaming 4.4 TB of affinity. The node-sampled
shortcut that findings 2 and 3 attacked is no longer needed for any reason, and is deleted.

**2. Provenance domains are separated, and the order changes accordingly.** GT-free identity
and evidence come first and read no skeleton; evaluator-only fidelity, audit rows, oracles,
and NERL come after, and never write into `gt_free/`. Nothing produced under `gt_free/` is
derived from an artifact whose *selection* saw GT.

**3. Every policy number is frozen in this document**, before any evaluator result for this
experiment exists. That is the contamination guard finding 5 asked for, and it makes stage
order irrelevant to contamination. Gate A consequently changes role: it is no longer "should we
build the policy" (the policy is mechanical once the numbers are fixed) but "does the GT-free
candidate population carry deployable headroom", reported in the same evaluator pass.

Execution order becomes:

```text
A  gt_free      identity manifest                      no GT read
B  evaluation   fidelity gates                         reproduces already-published numbers only
C  gt_free      full-population evidence, all 726      dense CC3D store + ABISS
D  gt_free      freeze policies + assignments + sha256  mechanical, numbers from this plan
E  evaluation   audit rows, directional oracles, Gate A, frozen-policy NERL, funnel
F  evaluation   diagnostics, split proposals, composition
```

The two Phase-0 conclusions from plan_v0 are unchanged and still load-bearing: the substrate is
**`arm0_win144`** (726 chunk files vs 10 for `arm0_native`; `wholevol_arm096_fullmask.yaml`
points at it literally), so the honest local control is **0.775832**; and "0.70" is applied to
**stored** h5 values, as in the recorded tier10 row, while ABISS separately applies
`AFF_RESTORE_SIGMOID: 0.2`.

## Scope

Unchanged from plan_v0 except as required by the findings. Added to scope: a ten-chunk
end-to-end tier10 reproduction, a paired ten-chunk fusion-mask bridge control, batch-global
resolver permutation tests, a composition identity gate, and the visual-probe manifests.
Removed from scope: node-bearing-chunk-only streaming, and any reuse of evaluator-stage
artifacts inside `gt_free/`.

Still out of scope: production decoder/config/API, dependency additions, CC3D threshold sweeps,
dense whole-volume output, retraining, modification of canonical inputs, commits during the run.

## Proposed Changes

### Stage A — GT-free identity manifest (finding 1, 7)

`stage_a_identity.py` writes `gt_free/input_manifest.json` and reads no skeleton, no node LUT,
no GT-derived file. Contents:

* affinity store path, its `index.json` verbatim (`input_shape [5700,12288,12288]`,
  `chunk_shape [1008,1008,1008]`, `halo [72,72,72]`, `checkpoint_path …step=00200000.ckpt`,
  726 records with `start_zyx`/`stop_zyx`);
* **all 726 chunk files enumerated individually** with dataset name, `shape`, `dtype`,
  channel count, file size and mtime — read from HDF5 headers only, not full arrays — plus
  sha256 over a fixed 1 MiB window per file and full sha256 for the probe chunks;
* tag resolution evidence: file counts 726 (`<BASE>.h5.chunks`) vs 10
  (`<BASE>_win48x96x96.h5.chunks`), the `_win48x96x96` index holding a single record, and the
  `wholevol_arm096_fullmask.yaml` `aff_path` string; conclusion `arm0_win144`;
* ABISS: CloudVolume URI, its `info` verbatim (shape, dtype, voxel size 9×9×20, offset/origin,
  mip list), `runtime.json`, the decode config's `AFF_CONVENTION`, `AFF_RESTORE_SIGMOID`,
  `AFF_KEEP_MASK`, and `expected_base_nerl: 0.4443760423975249` recorded **as a declared
  constant**, explicitly not as a measurement;
* CC3D recipe: threshold `0.70` on stored values, connectivity `26`,
  `fg=(aff[:3]>0.70).any(0) & keep`, the dense store path and its provenance (array 2855306),
  the mask used to build it, and the `(chunk_ordinal<<32)|local_label` namespacing convention
  with the chunk-ordinal table;
* both mask identities: `ffn_tissue_mask_18-18-20.zarr` (5700, 5456, 5332) used by the dense
  store and the tier10 recipe, and `tissue_border_keep_mask_full.zarr` (5700, 10912, 10664)
  used by the ABISS decode, with the per-chunk flag `touches_ring` for chunks reaching past
  x>10664 or y>10912.

Skeletons, `reports/arm096_lut/`, and `oracle_gt/` appear only in
`evaluation_gt/evaluator_manifest.json`.

### Stage B — evaluator-only fidelity gates (findings 1, 4, 7)

`stage_b_fidelity.py` writes `evaluation_gt/fidelity.json`. Every output is a pass/fail
reproduction of an already-published number and carries no per-candidate information, so
reading it cannot inform the frozen policy.

| gate | assertion |
|---|---|
| B1 ABISS baseline | `score_lut(graph, lut, 50)` equals `0.4443760423975249` within `1e-10` |
| B2 tier10 reproduction | decode **all ten** tier10 chunks end-to-end with our code under the tier10 tissue-mask recipe, score human-GT with the recorded aggregation, and reproduce **0.775832** within the tolerance the tier10 harness itself used; per-chunk `n_labels`, `fg_frac`, `base_nerl_funlib` must match the archived CSV rows exactly |
| B3 fusion-mask bridge | repeat B2 on the same ten chunks under the **ABISS tissue+border keep-mask**, and report score, coverage, component count, and candidate-topology deltas against B2 — so a later fusion change cannot be confused with a mask change |
| B4 affinity fidelity | direct `h5py` reads at asymmetric probe voxels match the loader; channel order and dtype asserted |
| B5 alignment | discriminative probes, below |
| B6 namespacing | identical local IDs in two chunks never compare equal after packing; unpack is exact |

**B5 discriminative alignment probes** (replacing plan_v0's untestable "displaced controls must
fail"). For each of 12 probe chunks (including `z2_y10_x7` and four `touches_ring` chunks):
select voxels whose 6-neighbourhood contains ≥2 distinct nonzero ABISS labels, i.e. label
boundaries, where a one-voxel displacement provably changes the label. Assert (a) our
chunk-crop read equals a direct per-point CloudVolume read at those voxels; (b) the full
label **histogram** of our crop equals the histogram of a direct CloudVolume read of the same
box; (c) the histogram of each of the six ±1-voxel-displaced boxes and of the axis-swapped
(x↔z) box **differs** from (b). Chunks whose interior is uniform are rejected as
non-discriminative and replaced, rather than silently passing.

### Stage C — full-population GT-free evidence, all 726 chunks (findings 2, 3, 10)

`stage_c_evidence.py` (sharded, resumable) reads, per chunk: the dense CC3D labels from the
2855306 store, and the spatially matching ABISS crop. It never reads a skeleton and never
consults a Stage-B or Stage-E output. It writes one NPZ shard plus a `.done` record, each via
`tmp` + atomic rename, containing per CC3D component: globally namespaced `(chunk, cc3d_id)`,
voxel count, bbox, centroid, face touches, `touches_ring`/mask-boundary flags, every
overlapping nonzero ABISS label with voxel counts and fractions, and compact face slabs for
the six chunk faces.

`stage_c_reduce.py` refuses missing or duplicate chunks (requires 726/726), builds the global
ABISS segment table `gt_free/abiss_segment_stats.npz` (voxels, bbox, chunk span, face touch,
ring contact, nucleus count from `yl_cb_80nm_neuron_v2.h5`, multi-nucleus flag), deduplicates
face evidence deterministically, classifies every component as `zero_anchor` / `one_anchor` /
`multi_anchor` / `anchor_only` / `invalid_or_boundary` under the three predeclared band
settings, and writes `gt_free/overlap_candidates.npz`.

A targeted second pass, `stage_c_affinity.py`, reads affinity **only for chunks containing at
least one surviving candidate transition** and extracts only: max-over-channel stored affinity
mean and p10 over the transition surface (fragment voxels within 2 voxels of the anchor under
26-connectivity, and the reverse), and 0.75-persistence (recompute CC3D at 0.75 inside the
candidate bounding box; record whether fragment and anchor stay in one component). This keeps
the affinity read proportional to the candidate population, not to the volume.

Determinism is asserted on canonical decoded arrays plus a stable content hash of a canonical
serialization (sorted keys, fixed dtypes, fixed byte order), never on NPZ container bytes.

### Stage D — frozen policies (finding 5)

All numbers below are fixed by this plan. `stage_d_freeze.py` only materializes them.

**Predeclared bands** (GT-free ABISS observables; all three reported, none selected):

| setting | established anchor | mid fragment | dust |
|---|---|---|---|
| tight | ≥ 3×10⁶ vox and bbox diag ≥ 10 µm | 3×10⁴ – 3×10⁶ vox | < 3×10⁴ vox |
| central | ≥ 1×10⁶ vox and bbox diag ≥ 5 µm | 1×10⁴ – 1×10⁶ vox | < 1×10⁴ vox |
| loose | ≥ 3×10⁵ vox and bbox diag ≥ 2.5 µm | 3×10³ – 3×10⁵ vox | < 3×10³ vox |

Calibration of these bands is physical, not empirical: at 9×9×20 nm a 150 nm-radius neurite is
≈4.4×10⁵ voxels per 10 µm of length, so 10⁶ voxels ≈ 23 µm of thin process and 10⁴ ≈ 0.23 µm.

**Predeclared ablation ladder** (`P1`–`P6`, cumulative):

| step | rule |
|---|---|
| P1 | no fusion (baseline) |
| P2 | accept every `one_anchor` candidate — unsafe diagnostic control |
| P3 | + CC3D support: ≥ 0.50 of fragment voxels inside the shared component, and ≥ 1000 voxels of that component overlapping the anchor |
| P4 | + affinity floor: transition-surface stored-affinity mean ≥ 0.70 **and** p10 ≥ 0.50 |
| P5 | + winner margin ≥ 0.20 in support fraction over the runner-up host, **and** fragment/anchor voxel ratio ≤ 0.25 |
| P6 | + vetoes: abstain if the component or fragment touches a nucleus instance, if the anchor is multi-nucleus, if `touches_ring`, or if the candidate touches a tissue-mask boundary |

**Predeclared robustness neighbours** (a positive result must survive ≥1 of them, and all are
reported): support ∈ {0.40, 0.60}; affinity mean floor ∈ {0.65, 0.75}; margin ∈ {0.15, 0.25}.
Additional predeclared variant: `P6+persist`, which additionally requires 0.75-persistence.

Outputs `gt_free/policy_<name>.json` and `gt_free/assignments_<name>.npz`, each stamped
`gt_free: true`, `frozen_before_evaluation: true`, with `parameters`, `implementation_sha256`,
and a `proposal_sha256` recorded in a companion JSON before any scoring — the freeze contract
`arm096_evaluate_frozen.py` already enforces.

**Batch-global resolver** (finding 6), replacing plan_v0's sequential union-find:

1. Build the candidate edge set `E = {(fragment, anchor, score)}` from Stage C.
2. Per fragment, keep only its best-scoring anchor and only if the margin rule passes;
   otherwise drop all of that fragment's edges with reason `ambiguous_host`.
3. Build graph `G` on **immutable original** ABISS labels from the surviving edges.
4. For each connected component of `G`, count original established anchors. If the count is
   ≠ 1, drop **every edge in that component** with reason `anchor_conflict` — never a partial
   acceptance of the first claim.
5. Assign every fragment in a surviving component to that component's single anchor.

Anchor status is read from the pre-assignment table only, so an assigned fragment can never
become an anchor or a relay. Steps 2–5 are set operations plus connected components, hence
order-independent by construction; permutation tests prove it. One hop only; the bounded
multi-hop ablation is deferred and runs only if one-hop is positive and precision-safe.

### Stage E — evaluator-only scoring and Gate A (findings 2, 8)

`stage_e_evaluate.py` produces `evaluation_gt/partition_audit.json`, `evaluation_gt/results.json`,
and `evaluation_gt/results.md`:

* the seven required partition rows (`abiss_arm096`, `cc3d_t070_unstitched`, `partition_meet`,
  `partition_join`, `abiss_masked_by_cc3d`, `gt_best_decoder_per_skeleton`,
  `gt_meet_merge_oracle`), computed by importing the pure helpers from
  `cc3d/compare_cc3d_abiss.py` — its `main()`, defaults, and argparse surface untouched;
* the three directional candidate oracles, computed over the **Stage-C full-population
  candidate set** with GT used only to decide correctness, never membership;
* **Gate A**: ≥ +0.01 candidate-restricted NERL oracle gain, or recovery of a known false
  merge / long-backbone break that test-50 NERL under-reports;
* frozen-policy NERL for `P1`–`P6`, `P6+persist`, and every robustness neighbour, with the full
  evaluation contract per row: delta from 0.444376, accepted/abstained counts with reason
  codes, topology counts, post-hoc host precision and candidate recall, improved / unchanged /
  regressed skeleton counts, maximum single-skeleton regression, gained and lost
  funlib-weighted ERL, ERL-visible multi-owner components, nuclei/glia/multi-anchor conflicts
  including those invisible to test-50, coverage and background-node changes, per-band and
  per-boundary-status breakdowns, and wall time / peak memory / bytes read / cache size;
* the fusion funnel table and explicit attribution of lost opportunity to no CC3D support,
  zero anchor, multiple anchors, low confidence, morphology veto, mask/chunk boundary, and
  unresolved global identity;
* three separated tables — frozen GT-free, expected-negative controls, GT-only ceilings — with
  only the first comparable to FFN 0.538003.

`results.md` answers primary questions 1–6 **in order**, including on a Gate-A negative path,
and cross-tabs CC3D disagreement against the three ERL-visible arm0_96 false merges.

**Composition identity gate** (finding 9), before any composed score with
`arm096_error_correction/decoder_gtfree/frozen_endpoint_merges.npz` (340 pairs, +0.000129):
assert its label universe is a subset of the raw arm0_96 label universe in
`gt_free/abiss_segment_stats`; assert its recorded baseline `0.4443760423975247` matches ours
within `1e-10`; define conflict semantics for a label claimed by both maps in each order; and
re-run the one-anchor invariant check on the composed result, since an endpoint pair is a
union and can create an anchor–anchor merge that neither map produces alone. Report ours
alone, theirs alone, both orders, and the conflict set.

### Stage F — diagnostics (finding 8)

`stage_f_diagnostics.py` emits split proposals for independently flagged suspects (multi-nucleus
claims, glia/lamellar morphology, incompatible centerlines), cross-chunk face-continuation
diagnostics reusing `build_face_iou_graph.py` concepts, and the required
`evaluation_gt/visual_probes/` manifests for correct, incorrect, multi-anchor, mask-boundary,
chunk-boundary, suspected-split, and no-benefit cases (≥3 representatives each, with global
coordinates and label IDs). No dense split unless Gate C passes.

## Files and Areas

All new files under `dev/zebrafinch/abiss_cc3d_fusion/`: `README.md`, `common.py`,
`stage_a_identity.py`, `stage_b_fidelity.py`, `stage_c_evidence.py`, `stage_c_reduce.py`,
`stage_c_affinity.py`, `stage_d_freeze.py`, `stage_e_evaluate.py`, `stage_f_diagnostics.py`,
`tests/test_*.py`, `sbatch_stage_c.sh`.

Read-only dependencies: the 2855306 dense CC3D store; `cc3d/compare_cc3d_abiss.py` and
`oracle_cc3d_chunked.py` (imported); `arm096_endpoint_oracle.py` (`load_graph_and_lut`,
`score_lut`, freeze contract); `sweep_thr_chunk_v2.py` (`build_mask` semantics for B2);
`cc3d_chunks_tissue.py`; `build_face_iou_graph.py`; `reports/arm096_lut/s*.npz`;
`yl_cb_80nm_neuron_v2.h5`; `connectomics.metrics.nerl`.

Environment: `pytc` conda env, `HDF5_USE_FILE_LOCKING=FALSE`, repo-root cwd with `sys.path`
inserts.

## Verification Plan

CPU tests (`pytest dev/zebrafinch/abiss_cc3d_fusion/tests -q`), the twelve required cases plus
the resolver permutation suite finding 6 demanded:

1. CC3D foreground semantics and 26-connectivity where 6/18/26 answers differ.
2. Clipped border-chunk mask alignment: ring chunks pad 0 under the ABISS contract and 1 under
   the tier10 contract; both asserted.
3. Collision-free global namespacing; exact unpack.
4. Deterministic overlap counts across 1/3/7 shards and shuffled chunk order.
5. One anchor + fragments emits candidates.
6. Two anchors abstain, in all three bands.
7. Transitive fragment chain connecting two anchors abstains — the whole component, not the
   second edge only.
8. One ABISS label split across several CC3D components is unchanged by the policy.
9. Face candidates deduplicated, independent of shard count and order.
10. GT-derived paths rejected by every `gt_free/` stage.
11. LUT remap equals dense relabel on a synthetic volume.
12. `cc3d/tests/test_compare_cc3d_abiss.py` passes unchanged.
13. **Resolver permutations**: one fragment claimed by two anchors; direct anchor bridge;
    transitive anchor bridge; cycles; duplicated edges; inconsistent hosts across chunks;
    assigned fragments never become anchors or relays — each over ≥20 random edge
    permutations with identical output required.
14. Determinism asserted on canonical decoded arrays and content hashes, not NPZ bytes; all
    shard and `.done` writes are tmp + atomic rename.

Data-side verification, in order: Stage A manifest complete for 726/726 files → Stage B gates
B1–B6 including the ten-chunk reproduction and the fusion-mask bridge → Stage C over 726/726
with completeness and determinism checks → Stage D freeze with recorded hashes → Stage E single
evaluator pass → leakage audit scanning every `gt_free/*` artifact for skeleton/owner/NERL
references → `README.md` records skipped stages and the exact gate that stopped them.

Cluster hygiene: Stage C shards capped at ≤16 concurrent and not co-scheduled with another
affinity-streaming array; each stage prints its output path and a per-chunk liveness line.

## Risks and Questions

* **Stage C depends on array 2855306 completing.** 29/726 chunks at ~3.8 min/chunk when this
  plan was written; ~3 h expected. If it fails or is incomplete, `stage_c_reduce.py` refuses to
  finalise rather than producing a partial graph. The store is resumable.
* **The dense store uses the tier10 tissue-only mask, not the ABISS keep-mask.** Interior
  chunks are unaffected; ring-touching chunks keep voxels the ABISS domain drops. B3 measures
  the size of this effect on ten chunks and P6 vetoes `touches_ring` candidates. If B3 shows a
  material difference, ring chunks must be re-decoded under the ABISS mask before any ring
  candidate is trusted — that is a bounded subset, not a full re-run.
* **Expected result remains a small or null gain.** Every prior label-algebra fusion lost, and
  the companion `frozen_endpoint_merges` map gained +0.000129 with 1 improved / 2 regressed
  skeletons. Precision is binding at roughly 8 correct joins per false merge. The design
  reports a null cleanly rather than rescuing it.
* **Freezing thresholds before seeing any data risks an empty accepted set.** That is the
  intended failure mode: an empty P6 with a populated P2 is a legitimate, informative result,
  and the three band settings plus robustness neighbours bound how much of it is threshold
  choice.
* **CloudVolume XYZ vs affinity ZYX remains the highest-risk silent bug**; B5's histogram-based
  discriminative probes exist because a wrong-but-plausible mapping would otherwise yield a
  full, meaningless candidate graph.
* Open question, unchanged: if Gate A passes only on the *cut* family, this plan reports it and
  routes it to Gate C and the nuclei/glia task rather than building a dense splitter.

## Changes Since Previous Plan Version

Every finding accepted; none deferred or argued down.

1. **Finding 1** — Phase 0 split into Stage A (`gt_free/input_manifest.json`, reads no GT) and
   Stage B (`evaluation_gt/fidelity.json`). Expected NERL appears in the manifest only as a
   declared constant; the measured reproduction and the human-GT tier10 result are
   evaluator-only.
2. **Finding 2** — node-sampled candidate generation deleted. Candidates are generated from
   full-population GT-free dense evidence over all 726 chunks (Stage C) before any evaluator
   overlay; skeleton samples now feed only the partition-control rows. Made affordable by the
   dense CC3D store from array 2855306.
3. **Finding 3** — no artifact reuse across the firewall. Stage C reads only GT-free inputs and
   never a Stage-B or Stage-E output; only code is shared.
4. **Finding 4** — B2 now decodes and evaluates **all ten** tier10 chunks end-to-end and must
   reproduce 0.775832, with per-chunk CSV agreement; new gate B3 adds the paired ten-chunk
   fusion-mask bridge control reporting score, coverage, and topology deltas.
5. **Finding 5** — all policy numbers predeclared in this document: three band settings, the
   P1–P6 ladder with explicit thresholds, robustness neighbours, the `P6+persist` variant, and
   the veto data sources. Stage D only materializes them.
6. **Finding 6** — sequential union-find replaced by the five-step batch-global resolver over
   immutable original anchors, rejecting **every** edge in a conflicted component; test 13 adds
   the seven permutation cases the reviewer listed.
7. **Finding 7** — the manifest enumerates shape/dtype/channels for all 726 files, plus ABISS
   shape/dtype/origin/parameters/expected NERL; "displaced controls must fail" replaced by B5's
   label-boundary probes and box-histogram comparisons, with non-discriminative probe chunks
   rejected and replaced.
8. **Finding 8** — `evaluation_gt/results.json`, `results.md`, and
   `evaluation_gt/visual_probes/` are explicit deliverables; `results.md` answers questions 1–6
   in order including on the negative path, and cross-tabs CC3D disagreement against the three
   ERL-visible false merges.
9. **Finding 9** — composition identity and safety gate added: label-universe subset check,
   baseline agreement to 1e-10, per-order conflict semantics, and a re-run of the one-anchor
   invariant after composition.
10. **Finding 10** — determinism asserted on canonical decoded arrays plus stable content
    hashes rather than NPZ bytes; all shard and `.done` writes use tmp + atomic rename.

Also changed, not from a finding: stage order and the role of Gate A, both consequences of the
dense CC3D store existing and of freezing policy numbers before evaluation.
