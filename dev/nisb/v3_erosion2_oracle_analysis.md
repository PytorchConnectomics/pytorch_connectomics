# v3_erosion2 — Oracle analysis

## Step 1a — missing skeleton-point fill oracle (2026-05-20)

Goal: estimate how much NERL is recoverable if skeleton points that currently
sample prediction label `0` are filled with the correct neuron label, without
re-running the full `just test` Slurm job.

Conclusion: no full rerun is needed for this probe. `lib/em_erl` already exposes
the needed primitive through `compute_segment_lut(...)` plus
`skeleton_assignment_zero_stats(...)`. Once the node LUT is cached, oracle edits
run directly on arrays.

Script added:

```bash
conda run -n pytc python dev/nisb/v3_erosion2_oracle_lut.py
```

Cached artifacts:

- LUT cache:
  `outputs/nisb_base_banis_v3_erosion2/20260508_224029/test_step=00200000/seed101/seg_fusion/oracle_lut/ch0-1-2_cc0.66_node_luts.npz`
- Summary CSV:
  `outputs/nisb_base_banis_v3_erosion2/20260508_224029/test_step=00200000/seed101/seg_fusion/oracle_lut/ch0-1-2_cc0.66_oracle_summary.csv`
- Per-GT CSV:
  `outputs/nisb_base_banis_v3_erosion2/20260508_224029/test_step=00200000/seed101/seg_fusion/oracle_lut/ch0-1-2_cc0.66_oracle_per_gt.csv`
- Report:
  `outputs/nisb_base_banis_v3_erosion2/20260508_224029/test_step=00200000/seed101/seg_fusion/oracle_lut/ch0-1-2_cc0.66_oracle_report.md`

The cache stores the full prediction node LUT, raw GT/skeleton LUT, a
collision-safe GT/skeleton LUT, node positions, and the ERL graph arrays. Future
oracle variants can load this `.npz` and call `compute_erl_score(...)` without
sampling the H5 again.

### Results on ch0-1-2 cc=0.66

| Variant | NERL | Pred ERL | Missing nodes | Omitted edges | Notes |
|---|---:|---:|---:|---:|---|
| baseline | 0.601431 | 14167.671659 | 46297/791035 | 61241 | matches existing eval text |
| zero_to_gt_safe | 0.603007 | 14204.788875 | 0/791035 | 0 | literal GT-ID replacement; conservative |
| zero_to_dominant_pred | **0.684829** | 16132.237004 | 0/791035 | 0 | nearest-label-style fill oracle |

Interpretation:

- The baseline has 46,297 missing skeleton points, covering 5.85% of sampled
  skeleton nodes. 415/416 skeletons have at least one missing point.
- Literal `pred==0 -> collision-safe GT id` only gives +0.0016 NERL. This is
  not the useful fill estimate because it removes omitted edges but creates
  split edges where a filled GT-id node touches an existing predicted segment.
- Raw GT IDs must not be mixed directly into the prediction LUT: low GT IDs
  collide with low predicted segment IDs and produce artificial merges. The raw
  replacement check scored 0.4163 NERL, so all GT-label oracle edits use the
  collision-safe label band saved in the cache.
- `pred==0 -> dominant nonzero predicted label on the same GT skeleton` gives
  +0.0834 NERL, reaching 0.6848. This is the meaningful LUT-level upper bound
  for a nearest-label / watershed fill track.
- Fill alone can theoretically clear the Phase 1.5 NERL target (>=0.65), but
  only under an oracle skeleton-node assignment. Dense voxel implementation
  still needs false-merge guardrails.

Top dominant-fill gains:

| GT | Missing nodes | Baseline nERL | Dominant-fill nERL |
|---:|---:|---:|---:|
| 139 | 129/1507 | 0.7082 | 0.8738 |
| 30 | 218/2103 | 0.5594 | 0.7203 |
| 276 | 149/1530 | 0.6069 | 0.7674 |
| 104 | 129/1357 | 0.6825 | 0.8386 |
| 389 | 134/1499 | 0.6318 | 0.7858 |

GT 226 has 247 missing nodes but no dominant-fill gain; it remains trapped by
the known persistent false merge with GT 341, so fill cannot fix that class.

### Validation

Commands run:

```bash
conda run -n pytc python dev/nisb/v3_erosion2_oracle_lut.py
conda run -n pytc python dev/nisb/v3_erosion2_oracle_lut.py  # cache reload
conda run -n pytc black --target-version py311 --check dev/nisb/v3_erosion2_oracle_lut.py
conda run -n pytc isort --check-only dev/nisb/v3_erosion2_oracle_lut.py
conda run -n pytc python -m py_compile dev/nisb/v3_erosion2_oracle_lut.py
```

The first script run sampled the H5 and wrote the cache. The second run loaded
the cache and reproduced the same oracle metrics.

## Step 1b — branch-merge LUT oracle (2026-05-20)

Goal: estimate the headroom for a perfect false-split / branch-merge pass
without running dense `branch_merge` over the whole H5 volume.

Script updated:

```bash
conda run -n pytc python dev/nisb/oracle_analysis.py --steps lut_branch_merge
```

This path loads the cached skeleton-node LUT from Step 1a and applies a
label-level merge map directly to the LUT:

- A predicted label is eligible only if its sampled skeleton nodes touch
  exactly one GT skeleton.
- For each GT skeleton, all eligible labels are relabeled to that GT's
  dominant eligible predicted label.
- Predicted labels touching multiple sampled GT skeletons are skipped, so this
  is a branch-merge upper bound, not a split oracle.

New cached artifacts:

- Summary CSV:
  `outputs/nisb_base_banis_v3_erosion2/20260508_224029/test_step=00200000/seed101/seg_fusion/oracle_lut/ch0-1-2_cc0.66_branch_merge_oracle_summary.csv`
- Per-GT CSV:
  `outputs/nisb_base_banis_v3_erosion2/20260508_224029/test_step=00200000/seed101/seg_fusion/oracle_lut/ch0-1-2_cc0.66_branch_merge_oracle_per_gt.csv`
- Merge map CSV:
  `outputs/nisb_base_banis_v3_erosion2/20260508_224029/test_step=00200000/seed101/seg_fusion/oracle_lut/ch0-1-2_cc0.66_branch_merge_oracle_map.csv`
- Skipped multi-owner labels:
  `outputs/nisb_base_banis_v3_erosion2/20260508_224029/test_step=00200000/seed101/seg_fusion/oracle_lut/ch0-1-2_cc0.66_branch_merge_oracle_skipped_multi_owner.csv`
- Oracle LUT arrays:
  `outputs/nisb_base_banis_v3_erosion2/20260508_224029/test_step=00200000/seed101/seg_fusion/oracle_lut/ch0-1-2_cc0.66_branch_merge_oracle_luts.npz`
- Report:
  `outputs/nisb_base_banis_v3_erosion2/20260508_224029/test_step=00200000/seed101/seg_fusion/oracle_lut/ch0-1-2_cc0.66_branch_merge_oracle_report.md`

### Results on ch0-1-2 cc=0.66

| Variant | NERL | Pred ERL | Missing nodes | Omitted edges | Notes |
|---|---:|---:|---:|---:|---|
| baseline | 0.601431 | 14167.671659 | 46297/791035 | 61241 | Step 1a baseline |
| branch_merge_oracle | **0.772069** | 18187.308464 | 46297/791035 | 61241 | label-level safe merge only |
| zero_to_gt_safe | 0.603007 | 14204.788875 | 0/791035 | 0 | Step 1a literal fill control |
| zero_to_dominant_pred | 0.684829 | 16132.237004 | 0/791035 | 0 | Step 1a fill oracle |
| branch_merge_oracle_zero_to_dominant_pred | **0.921596** | 21709.662981 | 0/791035 | 0 | merge oracle stacked with fill oracle |

Mapping stats:

- 11,387 nonzero predicted labels have sampled skeleton ownership.
- 11,361 labels are single-owner and eligible.
- 26 labels are multi-owner and skipped.
- 10,946 source labels are merged into dominant per-GT labels.
- 413/416 GT skeletons have at least two eligible labels.

Interpretation:

- False splits are a much larger recoverable component than Step 1a alone
  suggested: the branch-merge LUT oracle adds +0.1706 NERL over baseline
  even though missing skeleton nodes remain omitted.
- Fill and branch-merge are strongly complementary: stacking the two LUT
  oracles reaches 0.9216 NERL, leaving the skipped multi-owner labels and
  dense-realizability gap as the main residuals.
- This is an upper bound for dense branch-merge: the LUT oracle uses GT
  ownership and can merge non-adjacent fragments. A real implementation still
  needs adjacency, affinity, and shape guardrails before applying dense label
  merges.

Top branch-merge gains:

| GT | Baseline nERL | Branch-merge nERL | Gain |
|---:|---:|---:|---:|
| 90 | 0.3156 | 0.7309 | +0.4153 |
| 47 | 0.4195 | 0.7886 | +0.3691 |
| 44 | 0.5130 | 0.8708 | +0.3579 |
| 212 | 0.3703 | 0.7252 | +0.3549 |
| 157 | 0.4729 | 0.8274 | +0.3545 |

### Validation

Commands run:

```bash
conda run -n pytc python dev/nisb/oracle_analysis.py --steps lut_branch_merge
conda run -n pytc black --target-version py311 --check dev/nisb/oracle_analysis.py
conda run -n pytc isort --check-only dev/nisb/oracle_analysis.py
conda run -n pytc flake8 --max-line-length=100 dev/nisb/oracle_analysis.py
conda run -n pytc python -m py_compile dev/nisb/oracle_analysis.py
```

## Next oracle steps

1. Use ch3-4-5 to split false merges in LUT space, starting with the
   known bridge pairs from `v3_erosion2_err_analysis.md`.
2. Convert the branch-merge LUT headroom into dense candidate rules: restrict
   merge proposals to adjacent labels, then score with contact area, z
   continuity, and affinity evidence before any full-volume postprocessing run.
