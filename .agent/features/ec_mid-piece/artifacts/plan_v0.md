# Plan v0

## Summary

Build `dev/zebrafinch/ec_mid_piece/` as a five-stage, cache-first experiment that asks whether
native-scale affinity can safely absorb small arm0_96 fragments into clean neuron anchors, and
that measures exactly where the L126 mid-piece rung (`+0.138303` NERL over 1,887 GT 10-49-node
pieces) is lost when it cannot.

Three findings from the survey drive the design and are the plan's load-bearing claims:

1. **The scoring surface is tiny and already cached.** `reports/arm096_lut/s*.npz` is a node-level
   LUT over 500,845 GT skeleton nodes; 99.67% carry a label, spread over **11,408 GT-carrying
   segments**. Recomputing the L126 partition from that cache (per-`(skeleton, label)` piece, as
   `analyze_arm096.py` does) is seconds of work, and assignment-only absorption is expressible as
   a pure LUT relabel. No new segmentation volume is written, matching the out-of-scope list.
2. **The arm0_96 affinity contract is authoritative and is NOT `aff_absorb.py`'s.** The arm0_96
   decode config `wholevol_arm096_fullmask.yaml` names the source h5 chunk store, plus
   `AFF_CONVENTION: banis`, `AFF_RESTORE_SIGMOID: 0.2`, and keep-mask
   `tissue_border_keep_mask_full.zarr`. `lib/abiss/scripts/volume_backends.py::_read_banis`
   defines that convention precisely: channel `c` is the edge along **array axis `c`**
   (`0→Z, 1→Y, 2→X`), anchored `v→v+1`. `aff_absorb.py`'s `2 - ax` is correct only for the
   deleted XYZ-ordered CloudVolume mirror it read; applied to this ZYX h5 it swaps Z and X.
   The plan makes this contract explicit, tested, and recorded.
3. **The full-volume candidate pass is not affordable, and the honest workaround is a scope
   split, not a hidden shortcut.** The volume is 10664x10912x5700 at 9x9x20 nm (663 Gvox) with a
   2.6 TB / 726-file affinity store. The plan therefore separates a **GT-free feature extractor
   that never opens a GT file** from an explicitly evaluator-side **scope selector** that decides
   which segment IDs the extractor is run on. The decision rule stays GT-free; only the compute
   budget is GT-informed, and that is recorded in the manifest rather than buried.

The experiment is designed to be scientifically complete on a negative result: the Phase 0
topology inventory alone answers whether the mid-piece rung is a contact problem or a gap problem.

## Scope

In scope: `dev/zebrafinch/ec_mid_piece/` only — a README, narrow stage scripts, cached artifacts
under `gt_free/` and `evaluation_gt/`, and CPU tests.

Out of scope, per `task.md`: affinity-TTA implementation, model training, whole-volume
lower-threshold decoding, curve/Hungarian/weak-region connector implementation, dust absorption,
any `connectomics.decoding` or public-API change, and writing a new segmentation volume.

No package code under `connectomics/` is modified. No dependency is added (the stack already has
`numpy`, `h5py`, `zarr`, `cloud-volume`, `scipy`, `cc3d`, and `em_erl` via
`connectomics.metrics.nerl.import_em_erl`). No commits are created.

Canonical inputs are read-only: `reports/arm096_lut/`, `reports/arm096_error_structure.json`,
`reports/arm096_breaks_v2.json`, `arm096_error_correction/oracle_gt/`, the arm0_96 precomputed
segmentation, and the affinity chunk store. Nothing under those paths is written or overwritten.

## Proposed Changes

### Stage 0 — `stage0_inventory.py` (GT-free inventory + evaluator-side reproduction)

Two separable programs so the firewall is structural, not a convention:

`gt_free/` side (`stage0_segment_inventory.py`): takes a **segment-ID list file** and the
segmentation, emits `gt_free/segment_inventory.npz` with per-segment voxel count, bounding box,
centroid, volume-border-touch flags, extent/elongation, estimated centerline length and caliber
(`voxels / max_extent` and an EDT-based caliber on the local box), and predeclared band
membership. Its argument parser has no skeleton/LUT/owner option and it never opens one.

`evaluation_gt/` side (`stage0_reproduce.py`): loads graph+LUT through
`arm096_endpoint_oracle.load_graph_and_lut`, asserts `score_lut(...) == 0.4443760423975249`
within `1e-10` at `merge_threshold=50`, and reproduces the L126 partition using the same
per-`(skeleton, label)` piece definition as `analyze_arm096.py` — target
`668 / 1887 / 8979` pieces and `87.9 / 7.65 / 4.11` mass percent. A global per-label count gives
`665 / 1889 / 8854` and `88.21 / 7.68 / 4.10`, so the per-skeleton definition is the one that must
be used; the script asserts both the piece counts (exact) and mass fractions (`±0.05` pp) and
aborts with an actionable message naming the mismatching quantity.

It also emits `evaluation_gt/scope_segment_ids.txt` — the GT-carrying segment IDs — plus
`evaluation_gt/scope_manifest.json` recording `scope_source: lut_carrying_segments`, the count,
and the invariance proof restated below. This file is the *only* channel from GT to the extractor,
it carries IDs and nothing else, and the manifest labels it a measurement-scope device.

**Scope-invariance claim (asserted, not assumed).** For one-hop assignment-only absorption, a
fragment with no GT node cannot change any node's label, because the LUT maps node → its own
segment and only that segment's reassignment moves it. Stage 3 asserts this directly: it applies
the frozen assignment map to the LUT twice, once with all assignments and once with assignments
restricted to GT-carrying fragments, and requires identical output. This claim is *false* for
multi-hop, which is why multi-hop resolves on the full recorded segment graph before scoring.

### Stage 1 — `stage1_candidate_edges.py` (GT-free frozen candidate table)

Per fragment, read a local box from the segmentation (CloudVolume, mip `[9,9,20]`) and the
matching affinity box from the h5 chunk store, and emit one row per `(fragment_id, anchor_id)`
face-contact edge into `gt_free/candidate_edges.npz`, sharded then merged.

Affinity read helper `affinity_io.py`, written once and tested:

- open `chunk_z{Z}_y{Y}_x{X}.h5` from the store; **verify** the on-disk shape and halo (the name
  says `cs1008x1008x1008 halo72x72x72`, so a 1152³ payload with a 72-voxel margin is expected —
  the helper asserts the observed shape and derives the valid-region offset rather than assuming);
- `_restore_sigmoid(a, 0.2)` = `sigmoid(logit(a)/0.2)`, copied in behaviour from
  `volume_backends.py` and cross-checked against it in a test;
- apply the `tissue_border_keep_mask_full.zarr` keep-mask (masked → affinity 0), matching the
  decode;
- expose `edge_affinity(box, axis)` returning, for array axis `ax`, the affinity of edge
  `(v, v+1)` as `aff[ax, ...]` — **not** `aff[2-ax, ...]`.

Per edge, computed on the boundary voxel-pair set for that `(fragment, anchor)`:

| field | note |
|---|---|
| `contact_voxels` | number of face-adjacent voxel pairs |
| `contact_extent_zyx`, `contact_bbox` | spatial extent of the contact |
| `aff_mean`, `aff_p50`, `aff_p90`, `aff_p10`, `aff_max` | native-scale boundary affinity |
| `contact_axis_hist` | per-axis contact counts (anisotropy check) |
| `endpoint_flank` | contact centroid distance to the fragment's centerline endpoints, normalised by fragment extent |
| `caliber_ratio`, `tangent_cos` | fragment/anchor caliber compatibility; tangent cosine when both centerlines are defined |
| `n_anchor_candidates` | candidate anchors for this fragment |
| `anchor_voxels`, `anchor_border_touch` | anchor-side descriptors |
| `chunk_ids`, `overlap_dedup_key` | provenance for chunk-boundary dedup |
| `valid_*` flags | one per evidence source; missing evidence is a flag, never a silent zero |

`best`, `runner_up` and `margin` are **not** stored — Stage 2 computes them, so a rescore cannot
be contaminated by a baked-in ranking.

Rows are frozen: `gt_free/candidate_edges.npz` gets a `sha256` recorded in
`gt_free/input_manifest.json` along with the segmentation path, affinity store path, keep-mask
path, `restore_sigmoid=0.2`, convention string, resolution, and `gt_free: true`.

Chunk-boundary handling: a fragment box may straddle 1008-grid chunks. The extractor reads with a
halo, tags every contact with its originating chunk, and deduplicates by
`(fragment, anchor, voxel-pair coordinate)` deterministically before aggregation, so the same
physical contact counted in two chunks contributes once.

### Stage 2 — `stage2_policies.py` (native-scale assignment-only ablations)

Pure LUT-level rescore from the frozen table — no volume reads, so the grid is cheap.

Predeclared policies, exactly the six in `task.md`: `none`, `contact_area`, `aff_mean`,
`aff_p90`, `aff_p90_margin`, `aff_p90_margin_morph`. Predeclared grid: affinity floor
`{0.5, 0.6, 0.7, 0.8, 0.9}` x margin `{0.00, 0.02, 0.05, 0.10, 0.20}`.

Invariants enforced in code, each with a test:

- a fragment inherits exactly one anchor's label; an anchor's own label never changes;
- **no anchor-anchor union**: after applying assignments, every resulting component contains
  exactly one anchor — asserted, and a violation aborts rather than being dropped;
- a one-hop-assigned fragment cannot itself be an anchor;
- quarantined/low-quality anchors receive nothing in the honest policy (GT-free quarantine
  signals only: anchor volume-border touch, extreme caliber, extreme RAG degree);
- deterministic tie-breaking by `(score, contact_voxels, anchor_id)` with a stable sort.

Anchor/fragment split is by the predeclared GT-free voxel bands, not by GT node count.

Multi-hop is a **conditional** ablation, run only if one-hop clears the precision gate: depth
cap 2, relay paths recorded, resolution on the complete recorded segment graph before scoring,
one-anchor-per-component enforced, and a relay chain that would connect two anchors is rejected.

Outputs: `gt_free/policy_manifest_<name>.json` (complete frozen rule + parameters),
`gt_free/assignments_<name>.npz` (`gt_free: true`, `frozen_before_evaluation: true`,
`implementation_sha256`, `proposal_sha256` recorded in a sibling `.json`, mirroring the existing
`arm096_evaluate_frozen.py` freeze contract), and `gt_free/residual_<name>.npz`.

### Stage 3 — `stage3_evaluate.py` (evaluator-only, post-freeze)

Modelled directly on `arm096_evaluate_frozen.py`: verify the assignment `sha256` against the
pre-evaluation frozen report, verify the `gt_free` / `frozen_before_evaluation` markers, refuse
to run otherwise. Then load graph+LUT, re-assert the baseline, apply assignments, and score.

`apply_assignments(lut, assignments)` is a new directed-map helper rather than
`arm096_endpoint_oracle.relabel`, because `relabel` is union-find over unordered pairs and would
happily union two anchors. It maps fragment label → anchor root, asserts anchors are fixed points,
and asserts the one-anchor-per-component invariant on the induced graph.

Reported per policy and grid point: NERL and delta from its direct baseline; fraction of the
`+0.138303` L126 increment recovered; per-skeleton deltas with improved/unchanged/regressed
counts and the max single-skeleton regression; accepted fragments/components; correct-host
precision and GT-mid coverage; wrong-host, cross-neuron-union and multi-owner-component counts at
both the evaluator's materiality threshold (`mt=50`) and zero tolerance; stratification by
contact-count class, band, contact type, anchor quality, margin bin; risk/coverage curves; wall
time and peak memory.

Baselines are kept separate and labelled:

- **honest GT-free baseline = raw arm0_96, `0.4443760423975249`.**
  `arm096_error_correction/decoder_gtfree/` contains only `segment_skeleton_graph.h5` and
  `chunks/` — the `frozen_endpoint_merges.npz` that `arm096_evaluate_frozen.py` defaults to does
  **not exist**, so no frozen autonomous substantial-linker artifact is available. Per `task.md`
  the run proceeds on raw arm0_96 and states that it tests mid-piece absorption *before* the
  substantial-linking stage exists.
- **mechanism ceiling** = L123/L126 oracle substantial joins applied first, then absorption.
- **oracle-clean-anchor ceiling** = quarantine the three known ERL-visible cross-neuron labels
  (`73465106696008740`, `72269800654646543`, `72198881818968095`).

The last two are written to a separate table headed as ceilings. Only the honest row is placed
next to FFN `0.538003`.

### Stage 4 — `stage4_residual.py` + `evaluation_gt/results.md`

Frozen residual manifest, one reason per unresolved fragment from the exact seven codes in
`task.md`, assigned by a fixed precedence so each fragment gets one. Evaluator-side, report the
share of the L126 mid increment sitting in each bucket, and the funnel (candidate coverage →
quarantine → ranking ambiguity → confidence gate → residual no-contact).

The escalation recommendation is emitted only if `no_anchor_contact` holds a material share
**and** endpoint geometry supports a bounded corridor; otherwise the report states that ownership
or quarantine is the binding constraint.

### Phase 3 (multiscale) — conditional, gated, likely deferred

`r10_minpool/` exists on disk. `task.md` forbids reusing historically buggy r10 TTA/min outputs
without passing `.agent/features/affinity_tta/task.md`, which is out of scope. So Phase 3 runs
**only** if a larger-context artifact passes an independent geometry validation implemented here
(non-symmetric synthetic oracle for channel order, direction, indexing endpoint, spatial
alignment, scale mapping, valid faces). If it does not pass, the plan records the gate result and
Phase 3 is reported as not run, with reasons. Under no circumstance is a deleted r10 artifact
recreated.

## Files and Areas

All new, all under the gitignored `dev/zebrafinch/ec_mid_piece/` in the main checkout:

| Path | Purpose |
|---|---|
| `README.md` | commands, environment, wall time, peak memory, stage ownership |
| `affinity_io.py` | BANIS chunk-store reader, restore-sigmoid, keep-mask, edge accessor |
| `stage0_segment_inventory.py` | GT-free segment descriptors (no GT argument surface) |
| `stage0_reproduce.py` | evaluator-side baseline + L126 reproduction, scope ID emission |
| `stage1_candidate_edges.py` | sharded GT-free candidate edge extraction |
| `stage2_policies.py` | frozen policies, assignments, residual, LUT-level rescore |
| `stage3_evaluate.py` | post-freeze GT evaluation |
| `stage4_residual.py` | residual taxonomy + funnel + recommendation |
| `visual_probes.py` | crop manifest for correct/wrong/ambiguous/contaminated/no-contact cases |
| `tests/test_*.py` | affinity indexing, chunk dedup, one-anchor invariant, determinism, firewall, LUT eval |
| `gt_free/`, `evaluation_gt/` | artifacts named exactly as `task.md` requires |

Read-only, unmodified: `dev/zebrafinch/reports/arm096_lut/`,
`reports/arm096_{error_structure,breaks_v2}.json`,
`wholevol_arm096_fullmask/.../seg_arm096_fullmask`, the affinity chunk store,
`tissue_border_keep_mask_full.zarr`, `/projects/weilab/dataset/zebrafinch/test_50_skeletons.h5`.

Imported, unmodified: `arm096_endpoint_oracle.{load_graph_and_lut, score_lut}`,
`connectomics.metrics.nerl.import_em_erl`.

**Operative repository root is the main checkout `/projects/weilab/weidf/lib/pytorch_connectomics`
(branch `codex/gt-free-tube-analysis`), not the bridge worktree** — `dev/` and `.agent/` are
gitignored and exist only there. Because every deliverable is gitignored, tracked-file git diffs
are expected to stay empty; the code review is served the new files directly.

## Verification Plan

Environment: `source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc`.

1. **Affinity indexing** (the highest-risk item). Synthetic `(3, Z, Y, X)` volume with a distinct
   constant per channel and a deliberately asymmetric step so a channel swap, a sign/direction
   flip, and a one-voxel anchoring error each produce a *different* wrong answer; assert the
   helper's `edge_affinity` recovers the planted edges exactly. Then cross-check against
   `volume_backends._ArrayVolume` on the same synthetic array: reading through the ABISS path with
   `convention='banis'` must yield channel `2-c` at the shifted position of our channel `c`.
   Finally a real-crop probe on one chunk asserting boundary affinity is materially lower at a
   true segment boundary than in the segment interior.
2. **Chunk-boundary dedup**: synthetic object straddling a 1008 boundary yields identical contact
   statistics whether extracted as one box or two overlapping boxes.
3. **One-anchor invariant / determinism**: property tests on synthetic graphs including an
   adversarial case where two anchors share a fragment; rerunning Stage 2 from the cached table
   must reproduce byte-identical assignment files.
4. **GT firewall**: audit test asserting no `gt_free/` script's CLI exposes a skeleton/LUT/owner
   path and none opens `test_50_skeletons.h5`, `reports/arm096_lut`, or `oracle_gt/`; plus a
   manifest audit that every `assignments_*.npz` carries `gt_free: true` and
   `frozen_before_evaluation: true` and that its hash matches the pre-evaluation report.
5. **LUT-level evaluation**: assert `score_lut` reproduces `0.4443760423975249` within `1e-10`,
   and that `apply_assignments` with an empty map is the identity.
6. **Scope invariance**: the two-way assignment application described in Stage 0.
7. **Smoke run** on a small crop exercising all three affinity axes and a chunk border before the
   sharded whole-scope pass.
8. Reproduce the L126 inventory (Stage 0) before any policy is scored.

Operational: shard concurrency is capped (`<= 8` concurrent readers) and shards are chained rather
than co-scheduled with other affinity jobs — `/projects` has previously saturated under parallel
affinity streaming and blocked unrelated read-heavy jobs in D-state. Every stage is resumable from
its cache and prints a liveness heartbeat; the README records the non-interactive resume command.

## Risks and Questions

1. **Affinity convention is the single highest risk.** `aff_absorb.py`'s in-repo comment asserts
   `2 - ax`, and it is wrong for this artifact. If the tests in item 1 above are weak, every
   downstream number is meaningless. Mitigation: the cross-check against
   `volume_backends._ArrayVolume` makes the assertion against the code that actually produced
   arm0_96, not against a belief.
2. **Chunk halo geometry is verified, not assumed.** The store's name implies 1008³ + 72 halo; if
   the observed shape disagrees, the helper aborts rather than silently mis-registering by 72
   voxels.
3. **Scope selection is GT-informed.** Stated openly above. The decision rule is GT-free; the
   compute budget is not. This is the plan's most reviewable compromise and is recorded in the
   manifest, not hidden. The alternative — a whole-volume 663 Gvox / 2.6 TB pass — is not
   affordable in this task.
4. **No frozen substantial-linker baseline exists**, so the honest number tests mid-piece
   absorption on an unlinked backbone. This *understates* deployable value (the mechanism ceiling
   row quantifies by how much) and must not be presented as the pipeline's final capability.
5. **Precision bar is brutal.** L123-L125 measured that joins need ~0.95+ precision to pay, and a
   tip-geometry linker ceilinged at 0.819. Absorption of a mid piece into a wrong host is
   cheaper than a bad anchor-anchor join, but the plan still expects abstention to dominate. A
   near-zero-coverage honest result is a legitimate outcome, and Phase 0's topology inventory is
   the gate that makes it interpretable rather than a null.
6. **Calibration honesty.** There is no independent Zebrafinch calibration volume with compatible
   annotations. Per `task.md` the complete predeclared risk/coverage sweep is reported as
   diagnostic and one setting is *nominated* for a future confirmation run; no test-selected point
   is called validated.
7. **Open question for review:** should the mechanism-ceiling row apply the L123 277-join oracle,
   or the stronger L126 `>=50-owner-node` rung? The plan uses L123's 277 joins because that set is
   already frozen in `oracle_gt/gt_nominated_pairs.json` and is the one L126 itself uses as the
   `0.567056` reference point. Flagging in case the reviewer prefers the stronger rung.

## Changes Since Previous Plan Version

Initial plan.
