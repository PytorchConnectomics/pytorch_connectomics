# Inference: `base_banis.yaml` vs `lib/banis`

Comparison of pytc whole-volume affinity inference (`tutorials/neuron_nisb/base_banis.yaml`) against the reference `lib/banis/inference.py` + `lib/banis/BANIS.py`.

## Match

| Aspect | Both |
| --- | --- |
| Window size | 128³ |
| Overlap | 50% (BANIS: `small_size // 2 = 64` shift) |
| Activation | `scale_sigmoid(x) = sigmoid(0.2·x)` on output channels |
| Precision | fp16 autocast |
| Input normalization | divide-255, XYZ layout, no transpose |
| Decoded channels | first 3 (short-range) via `select_channel: [0,1,2]` |
| Decoder | connected components, 6-connectivity, source-stored edges (`edge_offset: 0`) |
| TTA | disabled (BANIS has no flip/rotate TTA) |
| Whole-volume strategy | both load full image then patch |

## Differences

1. **Blending weight**
   - pytc: `blending: gaussian, sigma_scale=0.25` (Gaussian importance map).
   - BANIS: `distance_transform_cdt` of zero-padded ones cube → L1 chamfer distance from surface (zero on faces, max in center). `lib/banis/inference.py:209-210`.

2. **Boundary handling**
   - pytc: `padding_mode: replicate` — MONAI pads the *whole* volume up front so windows align.
   - BANIS: no padding. `get_offsets` always sets the final offset to `big_size - small_size`, so every window fits fully inside the volume. `lib/banis/inference.py:189-191`.

3. **Threshold**
   - pytc: fixed `threshold: 0.5`.
   - BANIS: sweeps over `eval_ranges = sigmoid(0.2 · range(-1, 12))` ≈ `[0.45, 0.55, 0.65, 0.73, 0.80, 0.85, 0.89, 0.91, ...]`, picks val-best by NERL, reuses on test. `lib/banis/BANIS.py:439`, `lib/banis/BANIS.py:209-211`.

4. **Patch grid**
   - pytc: regular MONAI grid at 50% stride.
   - BANIS: base grid + 7 shifted sets (all combinations of `+small_size//2` per axis) unioned and de-duped. `lib/banis/inference.py:154-174`. Slightly more centers near boundaries.

5. **Stored prediction channels**
   - pytc: short-range only (`select_channel: [0,1,2]`).
   - BANIS: all 6 channels written to `pred_aff_*.zarr`; decoding still reads `[:3]`. `lib/banis/BANIS.py:199-200`, `lib/banis/BANIS.py:217`.

## To match BANIS exactly

- Replace `blending: gaussian` with custom L1-distance window (or accept gaussian as a near-equivalent at 50% overlap).
- Drop `padding_mode: replicate` and use BANIS-style snap-to-edge offsets (last offset = `image_size - roi_size`).
- Run a decoding threshold sweep over BANIS' `eval_ranges` and pick best by NERL on val before testing.

Items 1–2 are cosmetic at 50% overlap; #3 is the main accuracy lever.

## Boundary handling in pytc

Two paths matter:

- **Lazy sliding-window path** (`connectomics/inference/lazy.py`, used when `inference.sliding_window.lazy_load=true`). Honors `snap_to_edge: true` (last window at `image_size - roi_size`, no whole-volume padding) and per-window `target_context` (read `roi + 2·context`, predict, central-crop). `base_banis.yaml` uses this path.
- **Eager MONAI path** (`connectomics/inference/sliding.py`, MONAI's `SlidingWindowInferer`). Vanilla MONAI; ignores `snap_to_edge` / `target_context`. For BANIS-flavored boundary context here, just bump `window_size` larger than the training patch — see below.

## The `window_size = roi + extra` hack

Instead of per-window `target_context` oversample + central crop (extra code, extra forwards), set `window_size` larger than the training patch and rely on default gaussian blending to de-emphasize the outer band:

```yaml
sliding_window:
  window_size: [144, 144, 144]   # 128 (training) + 16 context per axis
  blending: gaussian
  sigma_scale: 0.25
  overlap: 0.5
```

- Interior windows naturally pick up real surrounding-volume voxels in the +16 band.
- Default gaussian (`sigma_scale=0.125–0.25`) gives the outer band ~5× less weight than the central edge — soft taper, no hard mask, no boundary coverage hole.
- Must be a multiple of the model's downsample stride. MedNeXt-S has 4 stages → 144 ✓ (128 + 16); 138 (BANIS training oversample) ✗.
- `~2×` per-patch GPU memory at 144 vs 128. Verify it fits with fp16.

This works for both the lazy and eager paths and replaces the need for an inference-time `target_context` config.

## What's still BANIS-specific

`snap_to_edge: true` (in the yaml) only affects the lazy path and is the BANIS-faithful behavior — model never sees padded volume data. The eager path uses MONAI's whole-volume padding, which is functionally close at 50% overlap.
