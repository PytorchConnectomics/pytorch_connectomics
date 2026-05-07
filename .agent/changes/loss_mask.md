# Affinity Loss-Mask Refactor

## Problem

Affinity targets emitted by `seg_to_affinity` historically conflated three
distinct semantics into a single float32 channel-stacked tensor:

- `0.0` — a real "no affinity" supervision signal.
- `0.0` (again) — voxels touching `seg == 0` (background).
- `0.0` (again) — voxels in the no-pair border for that offset (`affinity_mode="deepem"`).
- `-1.0` — sentinel for unlabeled / out-of-valid-region voxels (`affinity_mode="banis"` only).

The two affinity modes diverged: `banis` honored `seg == -1` via the `-1`
sentinel; `deepem` did not — unlabeled voxels became spurious "no-affinity = 0"
supervision. Neither matched `lib/DeepEM`, which uses an explicit
`affinity_mask` channel and per-edge valid cropping.

The implicit `-1`-sentinel contract was a footgun: every loss/metric author had
to remember to skip `-1` (`(target >= 0)` synthesis in the orchestrator), the
target dtype was forced to float32, and adding an extra "ignore" reason
(foreground mask, class-balanced zero, etc.) required stacking conditions on
top of the sentinel.

## Target Design

Replace the embedded sentinel with a **paired bool tensor**: target values and
loss-mask are explicit, both `bool`, both stacked over channels, both flowing
through the data pipeline as separate dict keys.

The two affinity modes become consistent: identical masking machinery, only the
storage convention (`dst_slice` vs `src_slice`) differs.

## Contract

### Target generation (`connectomics/data/processing/affinity.py`)

```python
@dataclass(frozen=True)
class AffinityTarget:
    values: np.ndarray   # bool, shape (C, A0, A1, A2)
    mask:   np.ndarray   # bool, shape (C, A0, A1, A2)
    affinity_mode: str   # "deepem" | "banis"
```

`seg_to_affinity` returns this paired target. Both modes:
- Compute `values[i] = (seg[src] == seg[dst]) & (seg[storage] > 0)` over the
  storage slice.
- Compute `mask[i]   = labeled[src] & labeled[dst]` over the storage slice
  (false outside the storage slice, i.e. at the no-pair border).

`AffinityTarget.__array__` returns `.values` so backwards-compatible callers
that do `np.asarray(target)` keep working; `.mask` must be accessed
explicitly.

### Pipeline plumbing (`connectomics/data/processing/transforms.py`)

`MultiTaskLabelTransformd` derives `use_mask: bool` at init from the task list
(true iff any task is `"affinity"`). When `use_mask=True`:

- Per-task masks are accumulated alongside per-task targets. Non-affinity
  tasks contribute all-True bool masks of matching shape, so the stacked
  mask has the same channel layout as the stacked target.
- The dict gets two paired keys: `d[key]` (float32 target) and `d[f"{key}_mask"]`
  (bool mask).

When `use_mask=False`:

- No mask allocations, no mask key written. Binary / EDT / SDT / etc.-only
  pipelines keep their old single-tensor footprint.

`SegToAffinityMapd` (the standalone single-task transform) emits the pair
unconditionally.

### Augmentation pipeline (`connectomics/data/augmentation/build.py`)

Helpers `_label_transform_emits_mask(label_cfg)` and `_label_mask_keys(cfg, keys)`
inspect `cfg.data.label_transform.targets` for affinity tasks. When present:

- An additional `ToTensord(keys=mask_keys, dtype=torch.bool, allow_missing_keys=True)`
  is appended after the float32 `ToTensord(keys=keys)`. The mask stays bool
  (4× memory saving vs float32 propagation).
- `LeadingSpatialCropd` (target-context crop) iterates dict keys when
  `keys=None`, so it picks up the mask key automatically without explicit
  plumbing.

When no affinity task is configured, none of the mask plumbing fires.

### Loss orchestrator (`connectomics/training/losses/orchestrator.py`)

`compute_loss_for_scale`, `compute_deep_supervision_loss`, and
`compute_standard_loss` all accept
`target_mask: Optional[torch.Tensor] = None`. When provided, it is sliced via
`term.target_slice`, resized per scale (`_resize_tensor_for_output(..., target_kind="dense")`),
and folded into the existing `combined_mask_tensor` alongside `term_mask_tensor`,
`batch_mask_tensor` (foreground), and `affinity_valid_mask` (per-offset border).

The previous `target_valid_mask = (target >= 0)` synthesis is **removed** —
targets no longer carry sentinels, so the synthesis would yield all-True and
silently disable masking for unlabeled voxels.

### Lightning integration (`connectomics/training/lightning/model.py`)

`training_step` / `validation_step` read `batch.get("label_mask", None)` and
forward it as `target_mask` through `_compute_loss` to the orchestrator.
`batch.get("mask", None)` (foreground ROI mask, single channel) remains a
separate argument and is combined inside the orchestrator.

## Dependency Direction

This refactor preserves the v3 contract:

- `data → utils` (no upward dependency).
- `training → {config, data, models, metrics}` — `model.py` reads bool mask
  from batch, passes to orchestrator. No imports from data internals.
- `evaluation` / `inference` / `decoding` untouched (no affinity-mask
  consumers below the training boundary).

## Bool Throughout

Memory and dtype budget along the pipeline:

| stage                           | target dtype | mask dtype | notes                             |
|---------------------------------|--------------|------------|-----------------------------------|
| `seg_to_affinity` output        | bool         | bool       | (C, A0, A1, A2) each              |
| `MultiTaskLabelTransformd` stack| bool→float32 | bool       | target cast at `_to_tensor`       |
| Augmentation transforms         | float32      | bool       | spatial-only, no interpolation    |
| `LeadingSpatialCropd`           | float32      | bool       | identical slicing applied         |
| Collate                         | float32      | bool       | normal `default_collate`          |
| `training_step` → orchestrator  | float32      | bool→float | mask cast to pred.dtype on entry  |
| Loss compute                    | float32      | float      | multiplied into `combined_mask`   |

The user-facing requirement "all type bool ... after augmentation, cast back
to original bool" is satisfied at the dataloader→GPU hop: the mask key keeps
its bool dtype through `ToTensord`, the collate, and the device transfer.
The orchestrator casts to `pred.dtype` only at the loss-compute boundary.

## Comparison to Reference Libraries

| convention                  | unlabeled | boundary | tensors     | dtype |
|-----------------------------|-----------|----------|-------------|-------|
| `lib/DeepEM`                | `affinity_mask` channel | per-edge valid crop | per-edge | bool / float |
| `lib/banis`                 | `loss_mask`             | masked into loss_mask | stacked | bool |
| pytc pre-refactor (`banis`) | `-1` sentinel           | `-1` sentinel | stacked, single | float32 |
| pytc pre-refactor (`deepem`)| **none**                | zeros (treated as real target) | stacked, single | float32 |
| pytc post-refactor (both)   | explicit `_mask` key    | explicit `_mask` key | stacked, paired | bool target / bool mask |

The post-refactor pytc shape matches `lib/banis` modulo storage-vs-source-vs-
destination convention; it also matches `lib/DeepEM` in spirit (explicit mask
channel) without splitting tensors per-edge.

## Tests

- `tests/unit/test_affinity_processing.py` — target/mask dtype + shape, both
  modes mask the same logical edge set, BANIS reference parity (mask exact;
  values compared only where mask=True since masked-out values are
  don't-care).
- `tests/unit/test_banis_reproduction_transforms.py`:
  - `test_affinity_pipeline_emits_paired_label_and_mask_keys` — full train
    pipeline ends with `label` (float32) + `label_mask` (bool).
  - `test_binary_only_pipeline_skips_label_mask_emission` — `use_mask=False`
    skips mask plumbing entirely.
  - `test_multi_task_label_transform_use_mask_attribute_reflects_config` —
    `use_mask` flips with task config.
- `tests/unit/test_loss_orchestrator.py::test_banis_affinity_loss_skips_via_explicit_target_mask`
  — orchestrator honors the explicit mask passed via `compute_standard_loss(target_mask=...)`.

## Migration Notes

- Tutorial YAMLs are **unchanged** — the new mask key emerges automatically
  when an affinity task is configured.
- Old training checkpoints continue to load (no schema or weight shape
  changes). Resuming training picks up the new mask plumbing transparently
  on the next batch.
- Custom losses that previously relied on `(target >= 0)` synthesis at the
  orchestrator boundary need to read `target_mask` from the orchestrator
  call signature instead. The `mask`, `target_mask`, `affinity_valid_mask`,
  and `term_mask_tensor` are still combined into a single `combined_mask_tensor`
  internally, so most consumers see no API change.
