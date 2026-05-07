# snemi_old: Segmentation Post-Processing Reference

Summary of useful functions from `lib/snemi_old/*.py` for improving segmentation.

## Key Post-Processing Strategies

### 1. Segment Classification (T_pytc_v2.py, T_snemi220416.py)
- **Border-touching**: `bb[:,1::2] == 0` or `bb[:,2] == D-1` etc. Segments touching volume boundary are unreliable for merge analysis.
- **Interior segments**: `num_border == 0` — candidates for orphan merge.
- **Singletons**: `bb[:,1] == bb[:,2]` — single-slice segments. SNEMI test had 1059/1651 singletons. High error rate.
- **Disconnected components**: cc3d.connected_components per segment — remove non-largest component.

### 2. Orphan Detection & Merge (T_snemi220416.py opt=='0.32', T_pytc_v2.py opt=='2.22')
Criteria:
1. Segment touches ≤1 boundary
2. Not connected across z-slices
3. Has single dominant neighbor in z±1
4. Size-based IoU > 0.6 with neighbor

### 3. Oracle Merge Analysis (T_pytc_v2.py opt=='2.211')
- Map each predicted segment to best GT match via max IoU
- Group predicted segments by GT label
- Segments mapped to same GT = should be merged
- Typical result: 190 oracle merges, ARE 0.048 → 0.025

### 4. Morphological Refinement (T_pytc.py)
- `seg_postprocess()`: 2D constrained watershed (mahotas.cwatershed) per slice
- Optional Sobel edge guidance from raw image
- ~0.008-0.015 error reduction

### 5. Multi-Stage Waterz (T_snemi220416.py, T_waterz.py)
Best parameters found:
- `merge_function: aff85_his256`
- `aff_threshold: [0.1, 0.9]`
- `threshold: 0.4-0.7`
- `dust_merge_size: 800 * rr²` (resolution-dependent)
- `dust_merge_affinity: 0.3-0.5`

### 6. Consistency Checking (T_consistency.py)
- Track segment IDs across z-slices
- Count max consecutive occurrences
- Segments with ≤2 consecutive slices = likely noise
- Abrupt size changes = potential errors

### 7. Skeleton Analysis (T_yulun_skel.py, T_skel.py)
- kimimaro TEASAR: `scale=4, const=500, anisotropy=(30,6,6)`
- Cable length filtering: long axons (≥5000µm) vs short fragments (<1000µm)
- ERL (skeleton-based) metric as alternative to pixel-based ARE
- Oracle skeletonization bridges false splits

## Practical Improvement Hierarchy

1. **Remove single-slice dust** — low risk, removes noise
2. **cc3d disconnect removal** — keep largest component per segment
3. **Orphan merge** — segments with bbox fully inside another
4. **IoU-based cross-slice merge** — cautious, only at bbox endpoints
5. **Morphological refinement** — cwatershed per slice
6. **Skeleton-guided merge** — use cable length to validate merges

## Key Files
- `T_pytc_v2.py` — comprehensive pipeline (merge, split, oracle)
- `T_snemi220416.py` — waterz params, orphan detection
- `T_consistency.py` — cross-slice tracking
- `T_yulun_iou.py` — IoU computation, adapted_rand
- `T_yulun_skel.py` — skeleton analysis
