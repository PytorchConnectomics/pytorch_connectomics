# Plan v6

## Summary
Final API design per the user: **YAML calls a few ready-to-use ops, not a parameter-per-step pipeline.**
Four registered ops, each doing one conceptual thing with sensible built-in defaults:

| op | inputs | does |
|---|---|---|
| `seg_2d` | `[raw]` | 2D per-slice segmentation + **globally unique IDs** |
| `branch_link` | `[raw, sections]` | cross-slice linking → 3D tracklets (the v0 seed) |
| `branch_split` | `[raw, seg]` | **ALL split steps combined** (link-cut change-point → tunnel/confident carve) |
| `branch_merge` | `[raw, seg]` | **ALL merge steps combined** (completion → IoU-primary mutual + margin → weak-gap bridge) |

YAML DAG is then 4 lines with **no tuning knobs**:
```yaml
nodes:
- {name: s,  op: seg_2d, inputs: [raw]}
- {name: v0, op: branch_link,     inputs: [raw, s]}
- {name: v1, op: branch_split,    inputs: [raw, v0]}
- {name: v2, op: branch_merge,    inputs: [raw, v1]}
output: v2
```

**Acceptance = same result on MIT-LiCONN**, not preservation of legacy behavior: the user has confirmed the
current `branch_merge`/`branch_split` "don't work that well" and **may be perturbed**. The byte-identical
regression gate from plan_v5 is therefore **dropped**.

## Scope
- IN: the four ops with defaults baked in; internal stage functions (reusable, but not YAML surface); removal
  of every `dev/mit_liconn` import from `connectomics/`; canonical placement of generic helpers; the
  waterz-vs-axon tutorial; parity verification against the MIT-LiCONN numbers.
- OUT: `dev/mit_liconn/*` stays (reproduction path); no per-step parameters exposed in YAML.

## Proposed Changes

### V1. Combined ops (the user's core requirement)
- `decoders/branch/sections.py::seg_2d(aff, *, thr=0.3)` — per-slice 2D waterz on the strong band with
  `small=0` (no small-segment removal — that removal caused 4.2% coverage loss) and **volume-unique relabel**
  so every 2D section has a global ID.
- `decoders/branch/linking.py::branch_link(aff, sections)` — conservative cross-slice linking (spine IoU +
  mutual best-buddy, **no force-split**) → tracklets. Reproduces `v0_sm0`.
- `decoders/branch/split.py::branch_split(aff, seg)` — **combined**, fixed internal order:
  1. `link_cut` — real-IoU change-point z-cut (local-min + min-frag gates, persistence off)
  2. `tunnel_split` — close-ended tunnel/parallel carve (`host_both` relaxed, anchor-slice skip)
- `decoders/branch/merge.py::branch_merge(aff, seg)` — **combined**, fixed internal order:
  1. `complete_sections` — absorb cross-section fragments (lateral, else best-IoU z-neighbour)
  2. IoU-primary mutual best-buddy + ambiguity margin, iterated to convergence
  3. weak-gap bridge (projected-mask IoU across a gap, mutual + margin)
All tuning constants are module-level defaults with the validated values; they remain function kwargs for
tests/research but the YAML never sets them.

### V2. Existing decoders — refactor freely
`branch_merge`/`branch_split` may be perturbed. Preferred: move their genuinely-useful primitives
(`_slice_overlaps`, `_bbox_touch_mask`, `_erosion_seed_components`, `_assign_parent_from_markers`) into shared
helpers that both they and the axon ops use; the axon ops must not duplicate marker-watershed or overlap-table
code. If a legacy stage conflicts with the validated algorithm, the validated one wins. Update or delete
legacy tests that encode the old behavior, and say so in the report.

### V3. Canonical placement of generic utilities
- `seg_stats`, `apply_lut` → `connectomics/data/processing/bbox.py` (beside `compute_bbox_all`).
- `completeness_report` → `connectomics/metrics/completeness.py`.
- All axon ops use `seg_stats`/`apply_lut`; nothing calls `compute_bbox_all_3d` (28.6s → 7.0s), and relabels
  are chunked in-place (no full-volume temp).

### V4. Dev-freedom
No module under `connectomics/` imports `dev` when done. Delete `decoders/axon_tracklet.py`, its four
registrations in `registry.py::_register_builtins()`, and `tutorials/axon_decoding/tracklet.yaml`
(audited: the only reference sites; `axon_tracklet.py` is untracked, so deletion loses nothing committed).

### V5. Registration
In `registry.py::_register_builtins()` (lazy, function calls only):
`register_decoder("seg_2d", ...)` (1 input) and `register_graph_op` for `branch_link`, `branch_split`,
`branch_merge` (2 inputs each). Adapters validate arity and unpack `[raw, seg]`, extracting `afz = aff[0]` /
`fgmax = aff[:3].max(0)` internally so the YAML never mentions channels.

### V6. Tutorial `tutorials/neuron_axon/`
- `axon_decode.yaml` — the 4-node DAG above, no kwargs.
- `waterz_baseline.yaml` — `decode_waterz` on the **same** `decoding.load_prediction_path`.
- Both set `evaluation.nerl_merge_threshold: 10` and the GT path; run via `scripts/main.py --mode test`.
- `README.md` — commands, identical-input statement, and the comparison table.

## Verification Plan
1. **Numerical acceptance (the goal):** the 4-node DAG on `raw_x1_head-aff_r1.h5` + `gt_label_clean_v3.h5`
   (943 labels, strongmax, `nerl_merge_threshold=10`) must reproduce, within ±0.002:
   `branch_link` ≈ **0.8284/0.9424**, `+branch_split` ≈ **0.7302/0.9631**, `+branch_merge` ≈ **0.8434/0.9525**;
   `waterz_baseline` ≈ **0.6530/0.7580**. Stage outputs also compared to the on-disk research artifacts
   (`decode_tracklet_v0_sm0.h5`, `v1_lcc_confident.h5`, `v3_weak.h5`) by exact equality where labels are
   deterministic, else relabel-invariant partition equality.
2. **Dev-freedom:** AST scan — no `connectomics/` module imports `dev`.
3. **Graph execution:** all four ops resolve and the DAG runs end-to-end on a small volume; arity errors raise.
4. **Unit fixtures** (small, deterministic) for each internal stage: completion (lateral + z-isolated), margin
   rejection, link-cut gates, tunnel carve, weak-gap projection, `seg_stats` vs `compute_bbox_all`,
   `apply_lut` vs `lut[seg]`.
5. **Perf:** `cc3d.statistics` called once per changed stage (not per call); full-volume DAG wall < 20 min.
6. **Config:** `validate_tutorial_configs.py --glob 'tutorials/neuron_axon/*.yaml'`; repo tests still pass
   (legacy branch_* tests may be updated — report which).

## Risks and Questions
- **R1 — combined-op parity.** Fusing stages behind fixed defaults could drift from the staged research runs.
  Mitigation: the internal order above mirrors the validated sequence exactly; verification step 1 is the gate.
- **R2 — `branch_link` vs `branch_merge` overlap.** Linking is conceptually "merge z-adjacent 2D pieces"; if
  `branch_merge` can subsume it, collapse to 3 ops — **only if** parity holds. Keep separate otherwise.
- **R3 — legacy test churn** from perturbing `branch_*`; permitted, must be reported.
- **R4 — closure size** for sections/linking; report if it balloons past ~600 lines.

## Changes Since Previous Plan Version
Redesigned around the user's API directive: YAML now calls **four ready-to-use ops with no per-step
parameters** (plan_v5 exposed stage-level configuration and wrapped extended legacy decoders); added the
explicit **`seg_2d` (2D seg + global ID)** op the user asked for, and **`branch_link`** to preserve the
v0 seed; **`branch_split` and `branch_merge` are now single combined functions** with the validated stage order
baked in. Because the user confirmed the legacy decoders underperform and **may be perturbed**, the
byte-identical regression gate (plan_v5 GATE 1) is **dropped** and replaced by **numerical acceptance on
MIT-LiCONN** as the sole behavioral target; `branch_*` may be refactored and their tests updated. Shared
primitives are now extracted for reuse rather than duplicated (V2).


## AMENDMENT v6.1 — naming (user, 2026-07-25)
Ops are named GENERICALLY, not axon-specific, and the new combined functions **replace** the underperforming
legacy `branch_split`/`branch_merge` (the user approved perturbing them):

| op | inputs | role |
|---|---|---|
| `seg_2d` | `[raw]` | per-slice 2D segmentation + **globally unique IDs** (generic; not axon-specific) |
| `branch_link` | `[raw, seg]` | cross-slice linking -> tracklets |
| `branch_split` | `[raw, seg]` | ALL split steps combined (**replaces** the legacy `branch_split`) |
| `branch_merge` | `[raw, seg]` | ALL merge steps combined (**replaces** the legacy `branch_merge`) |

YAML node names must be INFORMATIVE, not v0/v1/v2:
```yaml
nodes:
- {name: sections,  op: seg_2d,       inputs: [raw]}
- {name: tracklets, op: branch_link,  inputs: [raw, sections]}
- {name: split,     op: branch_split, inputs: [raw, tracklets]}
- {name: merged,    op: branch_merge, inputs: [raw, split]}
output: merged
```
Module layout follows the op names (`decoders/branch/{sections,linking,split,merge}.py`); the tutorial folder
stays `tutorials/neuron_axon/` (it IS the axon example) but the ops it calls are generic.

## AMENDMENT v6.2 — expose key ablation parameters per node (user, 2026-07-25)
Revises v6.1's "no tuning knobs": each node **lists its key parameters explicitly** so an agent can run
ablations without reading the source. Values shown ARE the validated defaults (the op still works if a key is
omitted). `seg_2d` must name its **backend library** explicitly.

```yaml
nodes:
- name: sections            # 2D per-slice segmentation + globally unique IDs
  op: seg_2d
  inputs: [raw]
  kwargs:
    backend: waterz         # waterz | abiss   <- which library does the 2D seg
    thr: 0.3                # foreground/agglomeration threshold
    small: 0                # 0 = keep every supervoxel (removal cost 4.2% coverage)

- name: tracklets           # cross-slice linking -> 3D tracklets
  op: branch_link
  inputs: [raw, sections]
  kwargs:
    iou_spine: 0.2          # conservative first-pass link
    iou_best_buddy: 0.3     # mutual-best link
    force_split: false      # keep off: v0 is intentionally under-linked

- name: split               # ALL split steps (cut false merges; raises the om ceiling)
  op: branch_split
  inputs: [raw, tracklets]
  kwargs:
    cut_drop_thr: 0.25      # IoU change-point depth vs neighbour median
    cut_window: 4           # neighbourhood for the local-min test
    min_frag: 6             # min slices between cuts / from an end
    tunnel_host_both: false # host may END at the seam (relaxed tunnel-merge)

- name: merged              # ALL merge steps (recover base under the raised ceiling)
  op: branch_merge
  inputs: [raw, split]
  kwargs:
    complete_min_size: 2000 # fragment absorption threshold (cross-section completion)
    merge_iou: 0.45         # shape-continuity threshold (IoU is the SELECTOR)
    aff_floor: 0.4          # z-affinity used only to exclude background
    margin: 0.15            # best must beat runner-up ("no other close match")
    weak_max_gap: 15        # weak-region bridge reach, in slices
    weak_min_iou: 0.35
output: merged
```
Rationale for each knob is the CUE LADDER in the module docstrings; ablations should move ONE knob at a time
and report base + oracle-merge at `nerl_merge_threshold=10`.
