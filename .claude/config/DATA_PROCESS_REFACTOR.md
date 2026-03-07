# DATA_PROCESS_REFACTOR.md

Critical review and refactor plan for `connectomics/data/process/`.

**Goals**: 1) Accurate logic, 2) Efficient execution, 3) Minimal and clean codebase.

---

## A. Logic Bugs & Correctness Issues

### A1. `seg_to_instance_bd` boundary detection is redundant for shift-based method
**File**: `target.py:117-164`
The thickness=1 shift-based code checks both `seg[:-1] != seg[1:]` AND `seg[1:] != seg[:-1]` (forward + backward). These are identical boolean arrays written to overlapping slices. The backward pass only adds boundary marks to the already-marked forward neighbor. This is correct but does redundant work -- the backward half can be removed if we simply mark both sides in the forward pass:
```python
diff = seg[:-1] != seg[1:]
bd_temp[:-1] |= diff
bd_temp[1:]  |= diff
```
This halves the number of comparisons for each axis. Same pattern repeats 3x (Z/Y/X) and in 2D mode. **Priority: medium (correctness OK, but 2x wasted work).**

### A2. `seg_to_flows` shadows loop variable `z`
**File**: `target.py:52-54`
```python
z, y, x = masks.shape
for z in range(z):   # shadows the shape variable
```
Works by accident but fragile. Rename to `for zi in range(z)`.

### A3. `seg_to_small_seg` in `target.py` vs `segment.py` -- two different implementations
**File**: `target.py:317-337` and `segment.py:72-97`
`target.py` uses simple cc3d on entire volume, while `segment.py` has a multi-axis projection approach. The `MultiTaskLabelTransformd._TASK_REGISTRY` maps `"small_object"` to `target.seg_to_small_seg`. Meanwhile `segment.seg_to_small_seg` is never called from the transform pipeline. **Decision: pick one, delete the other.** The `target.py` version is simpler and more correct (3D connected components). Delete `segment.seg_to_small_seg`.

### A4. `SegToFlowFieldd` is a no-op
**File**: `monai_transforms.py:270-288`
The transform does `d[key] = d[key]` -- it literally does nothing. Either implement it properly (call `seg_to_flows`) or remove it entirely.

### A5. `edt_semantic` uses hardcoded resolution `(6.0, 1.0, 1.0)` for 3D mode
**File**: `distance.py:43`
This is a magic constant that assumes a specific anisotropy ratio. Should be a parameter. The `seg_to_semantic_edt` wrapper in `target.py` doesn't pass resolution either.

### A6. `edt_semantic` uses `assert` for input validation
**File**: `distance.py:40`
`assert mode in ["2d", "3d"]` -- should be `ValueError`. Same in `edt_instance` line 84.

### A7. `compute_bbox_all_3d` initializes max columns to -1
**File**: `bbox.py:300-303`
```python
out[:, 2] = -1  # z_max
out[:, 4] = -1  # y_max
out[:, 6] = -1  # x_max
```
If an instance ID exists in `uid` but has no pixels found in any slice/row/col scan, its max stays -1 while min stays `sz[dim]`. This would create an invalid bbox. The issue is that `compute_bbox_all_3d` iterates slices/rows/cols (O(D+H+W) unique calls) instead of using `scipy.ndimage.find_objects` like `index2bbox` does. **Should use `find_objects` for correctness and performance.**

### A8. `weight.py` has dead `foo = np.zeros((1), int)` variable
**File**: `weight.py:17`
`foo` is assigned into `out[wid]` when no weight mode matches. This is confusing -- a zero-length or scalar weight is unlikely to be correct downstream. Should raise ValueError for unknown weight options.

---

## B. Efficiency Issues

### B1. `compute_bbox_all_2d` / `_3d` use row/col iteration instead of `find_objects`
**File**: `bbox.py:198-337`
These functions iterate every row/col/slice calling `np.unique`, which is O(D*H*W * n_iterations). `scipy.ndimage.find_objects` computes all bboxes in a single pass in C. Replace both with `find_objects`.

### B2. `seg_to_instance_bd` 2D mode: Python loop over Z slices
**File**: `target.py:181-222`
Each slice re-does the same shift operations. Could use the 3D code path even for 2D mode by simply not checking Z-axis neighbors, avoiding the Python loop entirely.

### B3. `seg_erosion_dilation` slice-by-slice Python loop
**File**: `target.py:465-477`
Uses Python for-loop over Z for 2D morphological ops. `scipy.ndimage` erosion/dilation supports 3D directly and would be faster.

### B4. `seg_to_small_seg` (target.py) uses Python loop over unique labels
**File**: `target.py:333-336`
```python
for label, count in zip(unique_labels, counts):
    if count <= threshold and label > 0:
        small_mask[labeled_seg == label] = 1.0
```
Should use vectorized approach: build a lookup table from unique/counts, then index.

### B5. `energy_quantize` builds bins with Python loop
**File**: `quantize.py:14-17`
Minor, but `np.linspace(-1/levels, 1.0, levels+1)` would be cleaner and faster than the loop.

### B6. `_decode_quant_numpy` / `_decode_quant_torch` hardcode 11 bins
**File**: `quantize.py:40,60`
The `levels` parameter from `energy_quantize` is not carried through to `decode_quantize`, which hardcodes 11 bins. This is an implicit coupling that will break if levels != 10.

---

## C. Dead Code & Unnecessary Complexity

### C1. `segment.py` has multiple unused functions
- `seg_to_small_seg` (shadowed by `target.py` version, not used in transform pipeline)
- `seg_markInvalid` -- not imported anywhere in the codebase
- `seg_erosion` / `seg_dilation` -- these are used nowhere (the MONAI transforms `SegErosiond`/`SegDilationd` inline their own morphology code instead of calling these)

### C2. `SegErosiond` / `SegDilationd` duplicate logic from `seg_erosion_dilation` in target.py
**File**: `monai_transforms.py:384-445`
These two transforms import `erosion`/`dilation` from skimage inside `__call__` (lazy imports in hot path) and re-implement the same slice loop that `target.py:seg_erosion_dilation` already provides. Should call `seg_erosion_dilation` instead.

### C3. `flow.py` has visualization/utility functions unrelated to target generation
- `normalize_to_range` -- not used anywhere in data pipeline
- `dx_to_circ` -- visualization function, belongs in `utils/visualizer.py`

### C4. `misc.py:show_image` -- matplotlib visualization in data processing module
**File**: `misc.py:69-90`
Should be in `utils/visualizer.py` or removed. Not related to data processing.

### C5. `build.py` has over-engineered config coercion
**File**: `build.py:22-63`
`_to_plain`, `_coerce_config`, and the elaborate key detection logic (lines 126-144) exist to handle dict/namespace/OmegaConf interchangeably. Since we're Hydra-only (no backward compat), simplify to accept only OmegaConf DictConfig.

### C6. `build.py` wraps `MultiTaskLabelTransformd` in `Compose([transform])`
**File**: `build.py:207`
A single-element Compose adds overhead for no benefit. Return the transform directly.

### C7. `bbox_processor.py:make_instance_processor` is never called
**File**: `bbox_processor.py:250-290`
Factory pattern that's not used anywhere.

### C8. `bbox_processor.py:process_instances_with_bbox` is never called
**File**: `bbox_processor.py:220-247`
Functional wrapper not used anywhere.

### C9. `target.py` thin wrappers add no value
- `seg_to_instance_edt` just adds resolution and calls `edt_instance`
- `seg_to_semantic_edt` just forwards to `edt_semantic`
- `seg_to_signed_distance_transform` just adds resolution and calls `signed_distance_transform`

These one-liner wrappers add an extra call frame and obscure the actual function signature. The MONAI transforms could call the underlying functions directly.

### C10. `__init__.py` exports individual MONAI transforms that are only used via `MultiTaskLabelTransformd`
Most individual transforms (`SegToBinaryMaskd`, `SegToInstanceEDTd`, etc.) are only consumed through the registry in `MultiTaskLabelTransformd`. They don't need to be in `__all__`.

### C11. `target.py:seg_to_generic_semantic` is not registered in MultiTaskLabelTransformd._TASK_REGISTRY
Not used anywhere in the codebase. Dead code.

---

## D. Structural Issues

### D1. Two-layer architecture is confusing
The current flow is: `build.py` -> `MultiTaskLabelTransformd` (monai_transforms.py) -> `target.py` wrapper -> `distance.py`/`flow.py` actual implementation. The `target.py` layer is mostly pass-through wrappers that obscure the real implementations. Collapse to two layers: MONAI transforms -> core implementations.

### D2. Module boundaries are unclear
- `segment.py` has erosion/dilation functions that duplicate MONAI transforms
- `target.py` imports from `distance.py`, `flow.py`, and `affinity.py` to re-export wrappers
- `bbox.py` and `bbox_processor.py` serve the same domain but split across files

### D3. Inconsistent dimension handling
Some functions expect `[D, H, W]`, others `[1, H, W]` for 2D, and `MultiTaskLabelTransformd.__call__` has complex logic to handle both. Should standardize: all core functions accept `[D, H, W]` (with D=1 for 2D), and the MONAI transform handles channel dim stripping/adding.

---

## E. Proposed Refactored Structure

```
connectomics/data/process/
├── __init__.py              # Lean exports
├── build.py                 # create_label_transform_pipeline (simplified)
├── monai_transforms.py      # MultiTaskLabelTransformd + individual transforms (slimmed)
├── affinity.py              # Affinity computation + DeepEM helpers (keep as-is, well-structured)
├── distance.py              # EDT, skeleton EDT, signed DT (absorb target.py wrappers)
├── boundary.py              # Instance boundary detection (extract from target.py)
├── weight.py                # Weight computation (clean up dead branches)
├── bbox.py                  # Bbox utilities (rewrite _2d/_3d to use find_objects)
├── bbox_processor.py        # BBoxInstanceProcessor (remove unused wrappers)
├── quantize.py              # Quantize/decode (fix levels coupling)
├── blend.py                 # Blending matrices (keep as-is)
└── misc.py                  # get_seg_type, get_padsize, array_unpad only
```

**Deleted files:**
- `target.py` -- functions absorbed into `distance.py`, `boundary.py`, or called directly
- `segment.py` -- used functions (`seg_erosion_instance`, `seg_selection`) moved to appropriate modules
- `flow.py` -- `seg2d_to_flows`/`extend_centers` moved to a utility or kept inline; viz functions removed

**Key changes:**
1. Core functions accept `[D, H, W]` consistently (D=1 for 2D)
2. `MultiTaskLabelTransformd._TASK_REGISTRY` points directly to core functions (no wrapper layer)
3. `compute_bbox_all` rewritten using `scipy.ndimage.find_objects`
4. `edt_semantic` takes `resolution` as parameter (no hardcoded `6.0, 1.0, 1.0`)
5. `decode_quantize` takes `levels` parameter to match `energy_quantize`
6. Remove all dead/unused functions
7. `SegToFlowFieldd` either implemented or removed
8. `build.py` simplified to Hydra-only config handling

---

## F. Priority Order

1. **Fix bugs**: A2 (variable shadow), A4 (no-op transform), A5 (hardcoded resolution), A7 (invalid bbox)
2. **Remove dead code**: C1, C3, C4, C7, C8, C11
3. **Efficiency**: B1 (bbox find_objects), A1+B2 (boundary detection), B4 (small_seg vectorize)
4. **Simplify**: C2 (erosion/dilation dedup), C5 (config coercion), C9 (target.py wrappers), D1 (collapse layers)
5. **Fix implicit coupling**: B6 (quantize levels)
6. **Structural cleanup**: D2 (module boundaries), D3 (dimension convention)
