# CuPy Connected Components Review Feedback

The CuPy backend appears aimed at correct affinity partitions, but the new
implementation has avoidable full-volume copies, duplicate compaction, and
initialization temporaries that directly undermine the stated speed and memory
goals at target chunk sizes.

## Review Comments

### P2: Use a change flag for convergence

File: `connectomics/decoding/decoders/segmentation_kernels.py:604-608`

When a large chunk needs multiple outer sweeps, `cp.copyto(seg_prev, seg)` plus
`cp.array_equal` keeps a second 4/8-byte-per-voxel label buffer and performs
extra full-volume passes plus a scalar sync every iteration. For the documented
512^3-1008^3 CuPy path this can erase much of the GPU win and push memory
toward OOM.

Recommendation: have the sweep kernels set a device-side changed/block flag and
test that instead of snapshotting the whole label volume.

### P2: Compact the CuPy result only once

File: `connectomics/decoding/decoders/segmentation_kernels.py:620-625`

In the normal `decode_affinity_cc(..., backend='cupy')` path, the array returned
here is immediately passed through `fastremap.refit`, so this
`cp.unique(..., return_inverse=True)` performs a full sort/remap and
materializes a volume-sized inverse array only for the CPU wrapper to compact
again. On large chunks this can dominate both time and peak GPU memory.

Recommendation: return the converged root labels to the wrapper, or skip the
wrapper refit when no missing foreground was added, so compaction happens once.

### P3: Avoid boolean-mask scatter for initial labels

File: `connectomics/decoding/decoders/segmentation_kernels.py:587-588`

For dense foreground chunks, this boolean assignment has to build `arange(n_fg)`
and mask/scatter temporaries before the sweeps, so peak memory grows by another
label-sized allocation and possibly index-sized scratch. A prefix-sum over
`fg_flat` (`labels_flat = cp.cumsum(...)` then zero backgrounds) produces the
same scan-order labels with one V-sized output and sequential memory access.
