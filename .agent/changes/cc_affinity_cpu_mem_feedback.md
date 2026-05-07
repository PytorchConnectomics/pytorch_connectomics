# Critical feedback on `cc_affinity_cpu_mem.md`

## Executive take

The note correctly identifies the root problem: full-volume affinity CC decode is
host-RAM bound, and the current whole-volume path keeps large prediction aliases
alive while allocating threshold masks and labels. The proposed direction is
reasonable, but several details are too optimistic. In particular, the current
numba-parallel kernel uses `uint64` labels for a 3000x3000x1350 volume, the P1
refcount sketch does not actually free the raw affinity before the CC call, and
the existing chunked config path can still fall back to whole-volume decode.

I would not hand this plan to another agent as-is. It needs a corrected memory
model and a sharper split between "small refcount cleanup" and "real large-volume
decode architecture."

## Major corrections

### 1. The memory estimate misses the `uint64` label path

For a 3000x3000x1350 volume, `N = 12.15B` voxels, which is greater than
`2**32`. In
`connectomics/decoding/decoders/segmentation_kernels.py`,
`connected_components_affinity_3d_numba_parallel()` chooses:

```python
label_dtype = np.uint32 if fg_flat.size < (1 << 32) else np.uint64
labels_flat = np.cumsum(fg_flat, dtype=label_dtype)
```

That means the working segmentation is about 97.2 GB decimal, not 48.6 GB,
until remap/cast. The corrected large buffers are closer to:

- raw 3-channel fp16 affinity: 72.9 GB
- `hard_aff` bool: 36.45 GB
- foreground bool `fg`: 12.15 GB
- seeded labels / `seg` uint64: 97.2 GB
- final compact cast: another full-volume allocation, because `cast2dtype()`
  uses `segm.astype(m_type)` with the default `copy=True`

This changes the priority. P1 may still prevent swap, but the expected
"hard_aff + uint32 seg ~= 100 GB" target is not accurate for this volume.

### 2. The P1 refcount sketch does not release the affinity during decode

The proposed stage change:

```python
preds = predictions
predictions = None
decoded = apply_decode_mode(cfg, preds)
del preds
```

does not reduce peak inside `apply_decode_mode()`. `preds` remains live until
the call returns, so the raw affinity is still pinned through the CC call.

The same issue exists one level lower. `apply_decode_pipeline()` creates
`batched = arr[np.newaxis, ...]` and `sample = batched[batch_idx]`; both are
views that can pin the original base buffer. Setting `data = None` is not enough
if `batched` or `sample` survives while the decoder is running.

If P1 is meant to be effective, the implementation needs true ownership
discipline:

- avoid routing huge single-sample arrays through the generic batched view path,
- clear all upstream aliases before the decoder enters the expensive CC step,
- delete `affinities` and `short_range_aff` inside `decode_affinity_cc()` as
  soon as `hard_aff` is built when `orphan_fill=False`,
- add a large-array fast path or "take ownership" helper rather than relying on
  cosmetic local rebinding across several frames.

Also, `assert hard_aff.base is None` is not a useful safety check. A comparison
result will own its bool buffer even if the original fp16 array is still pinned
elsewhere. Use RSS/`VmSwap` instrumentation or a focused refcount/view audit.

### 3. P2b can make memory worse than P1

"Stream-threshold inside the kernel" removes `hard_aff`, but it keeps the fp16
affinity alive throughout all sweep iterations. For this volume, that means
keeping ~72.9 GB live instead of a ~36.45 GB bool mask. If P1 successfully frees
the raw affinity after thresholding, raw-streaming is not a memory win.

P2b only becomes attractive if the input can be streamed from disk/chunks or if
the kernel consumes a packed/intermediate representation smaller than the fp16
raw array. Also verify numba support and performance for fp16 array comparisons
before assuming this is practical in `nopython` mode.

### 4. P3 is a kernel rewrite, not a small cumsum cleanup

Chunking `np.cumsum()` requires carrying global offsets between chunks. More
importantly, the current label dtype is selected from total voxel count, not
component count. Avoiding the `uint64` seed array probably requires a different
label assignment strategy, such as tile-local labels plus union/remap, not just
lazy foreground generation.

This is worthwhile, but it belongs with the packed/chunked kernel work rather
than as an isolated low-risk optimization.

### 5. P4 conflates two chunked paths

`connectomics/decoding/streamed_chunked.py` supports memory-bounded
`output_mode="decoded"` and decodes each chunk before stitching labels. However
`tutorials/neuron_nisb/base_banis_chunk.yaml` currently uses:

```yaml
inference:
  chunking:
    output_mode: raw_prediction
```

That mode writes chunked raw affinities, then loads the assembled raw prediction
and runs the standard whole-volume decoder. It improves inference tiling, but it
does not solve whole-volume decode RAM.

If the recommendation is "prefer chunked decode for full volumes", it must mean
`output_mode="decoded"` or a new streamed decoder over cached raw H5 chunks. The
plan should say that explicitly and should call out the tradeoff: the current
decoded chunked path skips evaluation because streaming metrics are not
implemented.

### 6. GPU CC is not a near-term answer for this workload

The A10 jobs do not have enough device memory. Even an 80 GB GPU is tight or
impossible for full-volume raw affinity plus labels and scratch. GPU only helps
if the decode is chunked or if the GPU backend also gets a memory-bounded input
representation.

## Cleaner implementation direction

1. Add an explicit byte-budget estimator for affinity CC decode.
   It should include raw affinity, `hard_aff`, foreground mask, label dtype,
   expected output cast copy, and backend-specific scratch. Log the estimate
   before decode and warn/error when the whole-volume path exceeds a configured
   budget.

2. Make full NISB default to decoded chunked inference when the goal is memory.
   Do not present `raw_prediction` chunking as the memory-safe path. If exact
   whole-volume semantics are required, say that they currently require enough
   RAM or a new streamed raw-H5 decoder.

3. Implement a narrow owned-array fast path for single-step `decode_affinity_cc`.
   The generic pipeline is convenient, but it is the wrong abstraction for
   70-200 GB arrays because each wrapper frame creates aliases. Keep the public
   decoder signature, but route large single-sample affinity decode through a
   function whose only responsibility is to consume the raw prediction and free it
   before CC.

4. Make trivial copy reductions while touching this area.
   `cast2dtype(segm)` should use `astype(m_type, copy=False)` unless a real copy
   is required. Audit `fastremap.refit()`, the "ensure background label" copy,
   and postprocessing so they do not silently duplicate the full output volume.

5. Treat packed/streamed affinity as a second phase.
   Prefer packed bool over raw fp16 streaming for CPU memory. Design the packed
   layout around the sweep access pattern; a generic C-order bitset may add too
   much division/modulo work in the hot loops.

6. Treat `uint64` seed labels as a first-class problem.
   For volumes with `N >= 2**32`, whole-volume cumsum labels are intrinsically
   expensive. A robust solution likely needs tile-local labels plus boundary
   union/remap, which aligns with the chunked decode architecture.

## Test and review requirements

- Add small equivalence tests for any optimized `decode_affinity_cc` path against
  the current numba-parallel backend, including components crossing X/Y/Z
  boundaries and both `edge_offset=0` and `edge_offset=1`.
- Add chunked-vs-whole equivalence fixtures for decoded chunking. Include
  components crossing chunk faces and cases where seam affinity exists only on
  one side according to `edge_offset`.
- Add a memory regression test gated by an env var. Use process RSS and
  `/proc/<pid>/status` `VmSwap`; `tracemalloc` will not capture numpy data
  buffers.
- Add a unit-level test for the dtype selection helper once it is factored out:
  `N < 2**32` should use `uint32`, `N >= 2**32` currently uses `uint64`, and any
  future tile-local strategy should document why it avoids that.
- Review `orphan_fill=True` separately. It re-reads `short_range_aff`, so the
  raw affinity cannot be dropped at the same point without changing behavior.

## Suggested rewrite of the plan

The most effective near-term plan is:

1. Correct the memory model and add logging/budget guardrails.
2. Route large full-volume NISB runs to decoded chunked inference, not raw
   chunking plus whole-volume decode.
3. Add the owned-array fast path and local `del` cleanup for `orphan_fill=False`
   as a partial improvement for users who still choose whole-volume decode.
4. Reduce avoidable full-volume copies (`astype(copy=False)`, postprocess audit).
5. Defer packed bool / tile-local label kernels to a follow-up with correctness
   and performance benchmarks.

That order gives an immediate operational path for 3kx3kx1.35k volumes while
keeping the kernel rewrite scoped and testable.
