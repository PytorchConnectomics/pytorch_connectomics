# Code v0 Review

Reviewer: claude (planner role). Surface: `.agent/features/ec_axon/`, `connectomics/decoding/`,
`connectomics/metrics/`, `connectomics/data/processing/bbox.py`, `connectomics/evaluation/`,
`connectomics/runtime/{cache_resolver,output_naming}.py`, `tests/unit/`, `tutorials/neuron_axon/`.
Pre-existing unrelated WIP (schema, `inference/chunked.py`, `training/`, `tuning/`,
`metrics/nerl.py`, `decoders/segmentation.py`, NISB tutorials, docs, root README) excluded.

## Summary

The port is faithful and the contract holds. I re-verified the algorithm against the named
research sources line by line and re-ran the reproduction myself through the public entry point
(`scripts/main.py --mode test`) rather than through the coder's own harness. The naive-waterz
baseline reproduces **exactly** (NERL 0.653011 / oracle-merge 0.758010 vs the reference
0.6530/0.7580) and its segmentation is **voxel-identical** to the reference artifact
`decoded_waterz_large_test.h5` (0 differing voxels, 1223 labels). The staged decode
reproduction is recorded below.

Constants and gate order match the research modules: `AFF_LOW=0.01 / AFF_BG=0.66 / RG_ZERO=False /
score aff30_his256_ran255 / thr 0.3 / small=0` (`decode_lib.py:32`, `decode_axon.build_strong_sections`);
conservative 0.2 + best-buddy 0.3 with force-split unreachable (`decode_v2._decode_sections_core`
returns at the `no_force_split` branch); `link_cut_change(recover=1.1)` = local-min + min_frag only
(`run_gated_final.py`); relaxed `host_both=False` confident carve; v2 `AFF_LO/MERGE_IOU/MIN_OV/
MIN_SIZE/ROUNDS/MARGIN = 0.4/0.45/30/2000/4/0.15`; v3 `15/1.6/0.35/0.3/2000/3`, `DIM_TOL=3`,
`MARGIN=0.15`; v4 opt-in behind `prefer_length`.

The parity-critical constraint from L81 is respected: `split.py` contains no completion;
`merge.py` calls `complete_sections` first thing in `merge_sections`. Getting this backwards
silently turns 360 link-cuts into 16 and costs 0.021 of the om ceiling.

## Findings

Fixed during review (each verified output-identical, not just plausible):

1. **`branch_link` did a 9.1 GB read + 3.35 GB allocation for nothing.** `_affxy` computed the
   in-plane affinity mean and used it only for a shape comparison; the research path
   (`decode_v2._decode_sections_core`, `no_force_split=True`) returns before affinity is
   consumed. Replaced with a direct `aff.shape[1:]` check, keeping the CZYX/3-channel
   validation. A/B on a 40-slice crop of the real volume: bit-identical output (1810 tracklets).
   This mattered because the task's efficiency requirements were explicit.
2. **Unused full-volume `bincount`** in `_section_index` (`area` was discarded by the caller).
   Removed.
3. **Six dangling `fn:` pointers in the two CUE LADDERs** — `_stage1_iou`, `_slice_overlaps`,
   `_stage2_best_buddy`, `_bbox_touch_mask`, `_erosion_seed_components`,
   `_assign_parent_from_markers` are all functions of the *deleted* modules. `merge.py`'s header
   also still described the old em_pipeline three-stage algorithm, which is not what the module
   does. Repointed to the real functions; the two cues that are genuinely not vendored
   (2D-component runs, area bump) are now labelled as such instead of pointing at absent code.
4. **`metrics/{oracle,completeness}.py` were unreachable from `connectomics.metrics`.** Added to
   the package `__init__` exports and docstring, matching how every other metric is surfaced.
5. **CLAUDE.md structure map was stale** (still listed `decoders/branch_merge.py`, no
   `decoders/branch/`, no `metrics/{oracle,completeness}.py`, no `tutorials/neuron_axon/`);
   the "Add a decoder" row now also mentions `register_graph_op`/`as_binary_graph_op` for
   multi-input ops.
6. **`experiment_log.py:72-87` logged the DELETED decoders' parameters.** The `branch_split`
   block emitted `seed_affinity_threshold / min_parent_size / min_seed_size / min_seed_fraction /
   max_seed_fraction / max_splits_per_parent` and the `branch_merge` block `iou_threshold /
   best_buddy / one_sided_threshold / one_sided_min_size` — all parameters of the removed modules
   (decode_waterz also just pops and discards the latter four). The new graph branch at lines
   38-47 made this block reachable for graph configs, so every graph run wrote six always-empty
   columns and logged **none** of the vendored ops' gates: two runs differing only in `drop_thr`
   produced byte-identical parameter rows, defeating the log. Now logs `thr` (seg_2d),
   `drop_thr/w/min_size/min_frag/recover/host_both` (branch_split) and `prefer_length`
   (branch_merge). Reproduced the reported scenario before and after: the split gates
   (`drop_thr=0.35, min_frag=12, host_both=True`) now appear and the six dead columns are gone;
   57 tests over the log/pipeline/naming/graph modules pass.
   *Found by an adversarial review workflow (5 dimensions × refute-by-default verification):
   25 raw findings, 24 refuted — most as parity-preserving behavior present identically in the
   research code (e.g. the `conservative_pairs` IndexError on an empty IoU table reproduces
   line-for-line in `decode_lib.py:96`, and the unclipped-affinity concern is note 8 below).*

Accepted with a note (no change):

6. **`EvaluationContext.resolved_output_path` now prefers `decoding.save_path`** over
   `inference.save_path`. This is required for decode-only runs (no inference save path exists),
   but it also moves the metrics file for existing configs that set both. Intentional and
   correct, but it is a behavior change outside the axon pipeline.
7. **Graph cache tags encode the whole node chain** — 159 chars for the 4-node tutorial, so
   `decoded_x1<tag>.h5` is ~175 of the 255-byte filename budget. Fine today; a 6-node graph or
   long node names would exceed it.
8. **Affinity is not clipped to [0,1]** in the ops, whereas the research scripts clip at load.
   No-op for this volume (checked: min 0.0, max 0.99993), and it only feeds the `aff_lo`
   background floor, but it is a difference worth knowing if a non-sigmoid affinity is fed in.
9. **`stats=` reuse is unguarded.** `merge_sections` forwards a caller-supplied `stats` into
   `complete_sections`; it must describe the *input* segmentation. Correct in the graph path
   (always `None`) and in the batch pattern the perf work introduced, but a stale tuple would
   silently mis-complete.
10. **The old `branch_split`/`branch_merge` algorithms are gone** (528 + 258 lines, plus their
    two decoding templates). That is the consolidation the user asked for, and nothing in the
    repo referenced them, but the names now mean something entirely different — recoverable
    only from git history.
11. **`bridge_weak_gaps` projects with `np.roll`, which wraps at the array border** rather than
    shifting out of the volume (`merge.py:394`). A segment within a few voxels of the y/x edge
    with a lateral velocity can therefore be scored against the opposite edge. Verbatim from
    `dev/mit_liconn/v3_weak.py:70`, so it is parity-preserving, not a port defect — but it is
    real, and the same trap applies to any future reuse on smaller tiles.
12. **`merge_sections(iou_lo=, dist_thr=)` are inert** — they parameterised the collinear
    low-IoU rescue, which measured net-negative and was correctly not vendored. Kept for
    signature parity with the research module; now documented as inert so nobody expects them
    to do something. The dead private `_merge_direction` helper (never called, in either the
    port or the research source's vendored surface) was removed.

## Verification (re-run by the reviewer, not inherited)

| Check | Result |
|---|---|
| `waterz_baseline.yaml --mode test` | NERL **0.653011** / om **0.758010** (ref 0.6530/0.7580) |
| waterz voxel parity vs `decoded_waterz_large_test.h5` | **0 differing voxels**, 1223 labels |
| `axon_decode.yaml --mode test` | NERL **0.843386** / om **0.952493** (ref 0.8434/0.9525) |
| staged voxel parity vs `outputs/mit_liconn/DL288B_crop1/v3_weak.h5` | **0 differing voxels**, 22071 labels |
| Focused unit suite (12 modules) | 118 passed |
| `validate_tutorial_configs.py --glob 'tutorials/neuron_axon/*.yaml'` | 17 canonical validated, 3 custom skipped |
| `test_v3_guardrails.py` (incl. new no-`dev`-import guard) | passed |
| Graph early stop | `sections/tracklets/split/merged` prune to exactly the ancestor set, distinct cache tags |
| `branch_link` edit A/B (40-slice crop) | bit-identical |
| black / isort / flake8 on the changed surface | clean |

## Staged reproduction (job 2774728, 41 min wall)

```
Saved HDF5: volume_0_0/decoded_x1_graph-sections-seg_2d-from-raw__tracklets-branch_link-from-raw
            +sections__split-branch_split-from-raw+tracklets__merged-branch_merge-from-raw+split
            __out-merged.h5
[volume_0_0] NERL: 0.843386     Pred ERL 10612.008533 / GT ERL 12582.622983 / 815 skeletons
[volume_0_0] NERL oracle-merge: 0.952493
```

Voxel-diff vs the research artifact `v3_weak.h5`: **0 differing voxels**, 22071 labels on both
sides. So the packaged DAG reproduces the research pipeline exactly, not just to 4 decimals of
NERL.

Caveat closed. Run 2774728 executed the code as it stood *before* finding #1's edit (the module was
already imported when the edit landed), so the staged decode was re-run on the post-edit tree as job
2774741: **NERL 0.843386 / om 0.952493, 22071 labels, 0 differing voxels against both `v3_weak.h5`
and the pre-edit package output.** The reproduction and the tree on disk are now the same thing.

## Verdict

VERDICT: APPROVED

The port is behaviorally exact on the reference volume (voxel-identical at both the baseline and
the final stage), the parity-critical stage order is right, dev-freedom is real and statically
guarded, and the five defects found were fixed and re-verified. Remaining notes (6-10) are
recorded rather than fixed: they are either intentional (6), headroom-limited (7), or
out-of-scope behavior the user explicitly asked for (10).
