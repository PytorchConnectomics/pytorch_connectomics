# MedNeXt Multi-Head Multi-Task Plan

This document defines a step-by-step plan to add true multi-head outputs for MedNeXt only.

Target use case:
- shared MedNeXt trunk
- separate task heads at the final resolution
- heterogeneous losses per head, such as `affinity -> BCE/Dice` and `sdt -> MSE/SmoothL1`

Non-goals for v1:
- no RSUNet changes
- no generic all-architectures abstraction first
- no per-head deep supervision in the first implementation
- no label pipeline rewrite

## Why This Change

The current codebase already supports multi-task losses over one flat output tensor by slicing channels. That is enough for simple cases, but it does not give true task-specific heads. If we want MedNeXt to learn `affinity` and `sdt` with different task-specific blocks before projection, we need named heads rather than one shared `10`-channel output layer.

This is a real architecture change, not just a config change.

## Current Constraints

The current pipeline assumes either:
- a plain tensor output, or
- a deep-supervision dict with `"output"` plus `"ds_*"` tensors

Relevant assumptions:
- `connectomics/training/lightning/model.py`
  - `_compute_loss()` only distinguishes plain tensor vs deep supervision dict.
- `connectomics/inference/tta.py`
  - `_sliding_window_predict()` requires `outputs["output"]` to be a tensor.
- `connectomics/training/lightning/callbacks.py`
  - visualization extracts the first tensor it can find.
- `connectomics/training/loss/orchestrator.py`
  - loss terms slice channels from one tensor.
- `connectomics/config/schema/model.py`
  - only one `model.out_channels`.
- `connectomics/config/pipeline/config_io.py`
  - cross-section validation checks selectors against `model.out_channels`.

MedNeXt itself also hard-codes one final output head:
- `lib/MedNeXt/nnunet_mednext/network_architecture/mednextv1/MedNextV1.py`
- `lib/MedNeXt/nnunet_mednext/network_architecture/mednextv1/blocks.py`

## Design Decision

Use real named heads from the model outward.

Recommended v1 output contract:

```python
{
    "output": {
        "affinity": affinity_tensor,
        "sdt": sdt_tensor,
    }
}
```

Do not flatten these heads back into one tensor internally just to preserve old code. That would keep the coupling we are trying to remove.

For v1, keep deep supervision off for multi-head MedNeXt. Add it later only if it proves necessary.

## Config Shape

Add MedNeXt head definitions under `model`.

Example:

```yaml
model:
  arch:
    type: mednext
  in_channels: 1
  primary_head: affinity
  heads:
    affinity:
      out_channels: 9
      num_blocks: 1
    sdt:
      out_channels: 1
      num_blocks: 1
```

Loss terms should route by head name:

```yaml
model:
  loss:
    losses:
      - name: affinity_bce
        function: PerChannelBCEWithLogitsLoss
        weight: 1.0
        pred_head: affinity
        target_slice: "0:9"
      - name: affinity_dice
        function: DiceLoss
        weight: 0.25
        pred_head: affinity
        target_slice: "0:9"
      - name: sdt_regression
        function: SmoothL1Loss
        weight: 1.0
        pred_head: sdt
        target_slice: "9:10"
```

Keep labels stacked exactly as they are today in v1. Predictions become named heads; targets remain one stacked tensor.

## Step-By-Step Implementation

### Step 1: Refactor MedNeXt to expose the final decoder feature map

Goal:
- stop treating `out_0` as the only final exit
- preserve the existing single-head behavior for backward compatibility

Current state:
- `MedNeXt` runs through `dec_block_0`
- then applies `self.out_0`

Required change:
- split the current logic into:
  - shared trunk forward up to the final decoder feature map
  - output projection from that feature map

Suggested shape:

```python
final_features = self.forward_features(x)
logits = self.out_0(final_features)
```

Why first:
- all later multi-head work depends on access to the shared final feature map
- this is the cleanest architectural breakpoint

Acceptance criteria:
- old MedNeXt single-head configs still return the same output tensor
- no change in tensor shape or checkpoint compatibility for old paths

### Step 2: Add a local multi-head MedNeXt wrapper

Files:
- `connectomics/models/arch/mednext_models.py`

Goal:
- keep the vendored MedNeXt trunk minimal
- attach task heads in the local wrapper layer

Recommended implementation:
- add a wrapper that owns:
  - `self.trunk`
  - `self.heads = nn.ModuleDict(...)`
- each head is:
  - `num_blocks` MedNeXt-style blocks at full resolution
  - final `1x1` projection to `out_channels`

For v1, each head can be:

```python
nn.Sequential(
    *[MedNeXtBlock(...) for _ in range(num_blocks)],
    nn.Conv3d(channels, out_channels, kernel_size=1),
)
```

Critical constraint:
- do not put head logic directly into generic training code
- keep it inside the MedNeXt model path

Acceptance criteria:
- forward returns:

```python
{"output": {"affinity": ..., "sdt": ...}}
```

### Step 3: Extend config schema for named heads

Files:
- `connectomics/config/schema/model.py`

Goal:
- express head layout without overloading `model.out_channels`

Add:
- `model.heads`
- `model.primary_head`
- per-head:
  - `out_channels`
  - `num_blocks`
  - optional future fields such as `hidden_channels`

Backward compatibility:
- if `model.heads` is absent, use the existing `model.out_channels` path unchanged

Critical point:
- do not remove `model.out_channels` yet
- many parts of the codebase still depend on it

### Step 4: Add head-aware loss routing

Files:
- `connectomics/training/loss/plan.py`
- `connectomics/training/loss/orchestrator.py`

Goal:
- let each loss term select a prediction head before optional channel slicing

Add fields:
- `pred_head`
- `pred2_head` if needed for prediction-vs-prediction losses

Resolution order for a loss term:
1. choose prediction head
2. choose channel slice within that head if requested
3. choose label slice from the stacked target tensor
4. compute the loss

Example:
- affinity BCE uses `pred_head: affinity`
- SDT regression uses `pred_head: sdt`

Acceptance criteria:
- a mixed `BCE + Dice + SmoothL1` config trains without flattening heads

### Step 5: Normalize output handling in the Lightning module

Files:
- `connectomics/training/lightning/model.py`

Goal:
- make model output handling explicit instead of relying on accidental dict shape

Add a small normalization layer that converts:
- tensor output
- deep supervision output
- named-head output

into a single internal representation.

Recommended internal shape:

```python
{
    "output": tensor_or_head_dict,
    "ds_1": optional_tensor_or_head_dict,
}
```

For v1:
- support named heads only at `"output"`
- reject per-head deep supervision with a clear error

Why be strict:
- silent fallback here will create hard-to-debug training behavior

### Step 6: Update config validation

Files:
- `connectomics/config/pipeline/config_io.py`
- `connectomics/config/hardware/auto_config.py`

Goal:
- stop assuming one global `model.out_channels`

Validation changes:
- if `model.heads` exists:
  - validate each referenced `pred_head`
  - validate slices against that head's `out_channels`
  - compute total output channels when needed for memory estimation

For v1:
- keep existing selector validation for inference paths that still operate on a single chosen head

Critical point:
- do not try to make every old inference selector instantly multi-head aware
- select one head first, then reuse existing per-channel validation on that head

### Step 7: Add inference head selection

Files:
- `connectomics/inference/tta.py`
- `connectomics/inference/manager.py`
- `connectomics/config/schema/inference.py`

Goal:
- make inference deterministic when model returns named heads

Add:
- `inference.head`
- optional future `inference.save_heads`

Behavior:
- if output is multi-head, inference must choose one head before:
  - TTA aggregation
  - decoding
  - thresholding
  - saving

Why:
- the current inference stack expects one tensor
- decoding for affinity and SDT should not be mixed implicitly

For v1:
- support selecting exactly one head for inference
- do not implement multi-head decoding/export in the same patch

### Step 8: Update visualization

Files:
- `connectomics/training/lightning/callbacks.py`
- `connectomics/training/lightning/visualizer.py`

Goal:
- avoid implicit "first tensor found" behavior

Add:
- visualization head selection

Example:

```yaml
train:
  visualization:
    head: affinity
    channel_mode: all
```

For v1:
- visualize one selected head only
- do not try to tile multiple heads in one figure yet

### Step 9: Add tests before enabling real training runs

Files:
- `tests/unit/test_connectomics_module.py`
- `tests/unit/test_loss_orchestrator.py`
- `tests/unit/test_hydra_config.py`
- add MedNeXt-specific tests if needed

Minimum test set:
- MedNeXt multi-head wrapper returns named heads
- loss routing resolves `pred_head` correctly
- Lightning training step accepts named-head output
- inference requires explicit head selection for multi-head models
- validation rejects unknown head names or invalid per-head slices

Do this before running expensive training jobs.

## Recommended Scope For V1

Do:
- MedNeXt trunk refactor
- MedNeXt local multi-head wrapper
- named-head loss routing
- single selected head for inference and visualization
- strict validation and clear errors

Do not do yet:
- per-head deep supervision
- multi-head decoding in one pass
- generic multi-head support for all architectures
- automatic migration of every old config path

## Risks

### Checkpoint compatibility

Refactoring MedNeXt internals can break old state dict loading if key names move carelessly.

Mitigation:
- preserve old single-head module names where possible
- add compatibility loading logic only if required

### Output shape ambiguity

If training, inference, and visualization each interpret dict outputs differently, bugs will be subtle.

Mitigation:
- centralize output normalization
- fail fast on unsupported dict layouts

### Over-scoping deep supervision

Per-head deep supervision will multiply implementation complexity and memory use.

Mitigation:
- ship full-resolution multi-head first

## nnU-Net Background

Official nnU-Net does not provide a generic heterogeneous multi-task head pattern for this use case. Its extension guidance points toward custom architecture and trainer changes for behavior outside the standard segmentation path. That supports the approach here: keep MedNeXt trunk reuse, but implement local multi-head logic in this repo rather than trying to force the existing single-output contracts to do more than they were designed for.

## Suggested Execution Order

1. Refactor vendored MedNeXt to expose final decoder features.
2. Build the local MedNeXt multi-head wrapper.
3. Extend config schema for `model.heads`.
4. Add `pred_head` routing in the loss planner and orchestrator.
5. Normalize Lightning output handling.
6. Add inference head selection.
7. Add visualization head selection.
8. Add tests and only then run training.

## First Concrete Milestone

The first milestone should be small and verifiable:

- MedNeXt can return:

```python
{"output": {"affinity": torch.Tensor, "sdt": torch.Tensor}}
```

- training can compute:
  - BCE or Dice on `affinity`
  - regression loss on `sdt`

- inference is still limited to one explicitly selected head

If that works cleanly, the architecture change is sound.
