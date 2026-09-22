# review_v0 raw verification transcript

Stage owner is `claude` (planner), so this stage ran in-session rather than through a companion
CLI. This file is the raw evidence: the commands actually executed and their real output.

## 1. Independent re-run of the focused suite

```text
$ source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc \
  && pytest -q dev/zebrafinch/ec_mid_piece/tests
======================= 16 passed, 3 warnings in 17.82s ========================
```

Confirms code_v0.md's claim of 16 passing tests.

## 2. Is the real producer module importable? (code_v0.md claims it is not)

```text
$ python -c "
import sys; sys.path.insert(0,'lib/abiss/scripts')
import volume_backends as vb
print('IMPORT OK:', vb.__file__)
print('has _ArrayVolume:', hasattr(vb,'_ArrayVolume'))
print('has _restore_sigmoid:', hasattr(vb,'_restore_sigmoid'))
"
IMPORT OK: /projects/weilab/weidf/lib/pytorch_connectomics/lib/abiss/scripts/volume_backends.py
has _ArrayVolume: True
has _restore_sigmoid: True
```

The module imports cleanly in `pytc`. code_v0.md's stated reason for substituting a local
reimplementation ("the original producer dependency is not importable in this checkout") is false.

## 3. What the affinity test actually compares

`dev/zebrafinch/ec_mid_piece/tests/test_affinity_io.py:5`

```python
from affinity_io import _ArrayVolume, banis_to_abiss, edge_affinity, restore_sigmoid
```

`dev/zebrafinch/ec_mid_piece/affinity_io.py:61,86`

```python
class _ArrayVolume:
    ...
    def abiss(self) -> np.ndarray:
        return banis_to_abiss(self.restored)
```

`test_affinity_io.py:30-40` then rebuilds `expected` with the same channel-reversal + one-voxel
shift that `banis_to_abiss` implements, and asserts `max|observed - expected| == 0`. Both sides of
the assertion are derived from the same belief, in the same file tree.

## 4. Independent ground-truth check of the convention (run earlier this session)

Against the real `volume_backends._ArrayVolume` on a real crop of `chunk_z2_y5_x5.h5`:

```text
abiss ch0 (=x) == my A[2] shifted by -e2: True   maxdiff=0.000e+00
abiss ch1 (=y) == my A[1] shifted by -e1: True   maxdiff=0.000e+00
abiss ch2 (=z) == my A[0] shifted by -e0: True   maxdiff=0.000e+00

negative controls (all must be False):
  no shift        : False
  channel c=cp    : False
  wrong axis shift: False
  +1 instead of -1: False
```

So the convention the implementation encodes is correct. The defect in finding 1 is that the
committed test does not establish this; it would pass even if the convention were wrong.

## 5. Stage 1 uses Form 1 correctly (the legacy `2 - ax` bug is absent)

`dev/zebrafinch/ec_mid_piece/stage1_candidate_edges.py:104-130`

```python
for axis in range(3):
    source_slice[axis] = slice(0, count)
    destination_slice[axis] = slice(1, count + 1)
    ...
    values = affinity[(axis, *source_slice)][different][candidate]
```

Channel index is `axis`, sampled at the low-side source voxel — exactly Form 1,
`A[c, p]` for the pair `(p, p + e_c)`.

## 6. Per-variant anchor-floor filtering is enforced

`stage1_candidate_edges.py:121-127` emits the **union** of both anchor-floor variants, but
`stage2_resolve.py:189-212` filters rows by `valid_anchor_floor_<key>`, and `:255-257` asserts:

```python
if np.any(inventory["voxels"][inventory_rows_fragment] >= anchor_floor):
    ...
if np.any(inventory["voxels"][inventory_rows_anchor] < anchor_floor):
    ...
```

Extract-once/resolve-twice is therefore safe.

## 7. Freeze-attestation coverage

`tests/test_firewall.py:36-56` writes a fixture npz, hashes it, then asserts the recorded hash
equals `sha256_file` of that same fixture. No tampered payload is exercised and Stage 3's
verification path is not invoked. `tests/test_resolver.py:68-72` does cover write-once immutability
at freeze time (`pytest.raises(ValueError, match="immutable frozen payload")`).

## 8. Input drift

```text
$ ls -la dev/zebrafinch/arm096_error_correction/decoder_gtfree/
-rw-r--r-- 1 weidf weilab      2053 Aug 14 12:47 frozen_endpoint_merges.json
-rw-r--r-- 1 weidf weilab     16285 Aug 14 12:47 frozen_endpoint_merges.npz
-rw-r--r-- 1 weidf weilab 218123055 Aug 14 12:47 endpoint_candidates.npz
```

These did not exist when plan_v2 was written (verified at run start); they were created at 12:47 by
a concurrent session, during this CCC run.

## 9. Mutation guard

```text
$ git rev-parse HEAD
6ad67866c3ca3e1cf1d52437a0cb300b0340c1bd      # equals run_start_ref
unstaged: UNCHANGED
staged:   UNCHANGED
```
