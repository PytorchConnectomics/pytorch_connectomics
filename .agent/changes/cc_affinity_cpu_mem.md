# Cut peak host RAM for `decode_affinity_cc` on full-volume runs

## Summary

`--mode test` on large NISB volumes (3000×3000×1350) currently peaks at
~180–200 GB RSS during `decode_affinity_cc`, because the loaded fp16
affinity (~73 GB), the bool threshold mask (~36 GB), and the uint32 CC
output (~49 GB) coexist in memory across three call frames. Observed run
on `gdw001` with a 2-CPU / ~256 GB cgroup spilled ~58 GB to swap and
deadlocked the process in `D` state. This note enumerates memory-saving
approaches in the affinity → CC decode path. Nothing here is implemented
yet — it is a design surface for the next coding agent.

Status legend: **[implemented behavior]** = already in tree, cite path.
**[proposal]** = not implemented; needs work.

## Implemented behavior (today)

- `connectomics/training/lightning/test_pipeline.py:511` loads the cached
  affinity into `predictions_np` (rank-0 only, lazy_load + sharded SW
  reduce already in place).
- `_process_decoding_postprocessing` (test_pipeline.py:295-313) calls
  `run_decoding_stage(module.cfg, predictions_np)`. The caller's
  `predictions_np` ref is alive through the whole decode call.
- `run_decoding_stage` (`connectomics/decoding/stage.py:119`) forwards
  `predictions` into `apply_decode_mode` and only releases its parameter
  when it returns.
- `apply_decode_pipeline`
  (`connectomics/decoding/pipeline.py:81-128`) holds `data` (param),
  `batched` (`np.newaxis` view of `data`), and `sample` (further view)
  for the entire loop.
- `decode_affinity_cc`
  (`connectomics/decoding/decoders/segmentation.py:455-598`):
  - `short_range_aff = affinities[:3]` (view, line 535).
  - `hard_aff = short_range_aff > threshold` (new ~36 GB bool, line 536).
  - With `orphan_fill=False` (BANIS default; matches
    `tutorials/neuron_nisb/base_banis.yaml`), `affinities` is **not
    read again** — but no `del` releases it.
  - With `orphan_fill=True` (line 578), `(short_range_aff > 0).any(axis=0)`
    re-reads the affinities; the array must stay alive until then.
- Numba parallel CC kernel (`segmentation_kernels.py:237`,
  `_propagate_affinity_sweep`, `@jit(nopython=True, parallel=True)` with
  `prange` over Y·Z, X·Z, X·Y columns) is bit-exact with the serial
  flood-fill and is what `backend: numba` selects via the parallel
  variant `connected_components_affinity_3d_numba_parallel`
  (`segmentation_kernels.py:324`).
- `_affinity_foreground_mask` (`segmentation_kernels.py:213`) already
  builds the foreground via `hard_aff[:3].any(axis=0) | …`, allocating
  one extra bool volume of `(X,Y,Z)`. This is the input to the
  scan-order `cumsum` label seeding at lines 350-356.
- `cupy` backend (`_connected_components_affinity_3d_cupy`,
  `segmentation_kernels.py:_build_affinity_cc_sweep_kernels` and below)
  exists and is selected by `backend: cupy`. Requires
  `cupy-cudaXX` and ~17×volume bytes of GPU memory. Single-GPU only.

## Proposals — ordered by ROI

### P1. Drop affinity refs across the call chain after threshold [proposal]

Cuts ~73 GB from peak. No algorithmic change, no API break. Smallest
diff that gets the current run off swap.

- `connectomics/decoding/decoders/segmentation.py:534-537` — when
  `orphan_fill=False`, drop the local refs immediately after the bool
  is built:
  ```python
  short_range_aff = affinities[:3]
  hard_aff = short_range_aff > threshold
  if not orphan_fill:
      del short_range_aff
      del affinities  # local ref only; caller still holds it
  ```
- `connectomics/decoding/pipeline.py:81-128` — once the first decoder
  has consumed `sample`, drop `batched` and the `data` param alias so
  the underlying buffer's refcount can fall to 1 inside the decoder.
  Concretely after `_prepare_batched_input`:
  ```python
  batched, batch_size = _prepare_batched_input(data)
  data = None
  ```
  and after `sample = decoder(...)` for the first step, set
  `batched = None` when `batch_size == 1`.
- `connectomics/decoding/stage.py:119-130` — pass `predictions` via a
  local that is `del`'d before the call so the parameter alias does not
  pin the buffer:
  ```python
  preds = predictions
  predictions = None
  decoded = apply_decode_mode(cfg, preds)
  del preds
  ```
- `connectomics/training/lightning/test_pipeline.py:295-313` — same
  pattern: rebind `predictions_np` into a local and `del` the caller's
  reference *before* `run_decoding_stage` runs (currently the `del` at
  line 621 happens *after*).

Expected new peak: hard_aff (36 GB) + seg uint32 (49 GB) + numba scratch
(~10 GB) ≈ **100 GB** instead of ~180 GB.

Risk: refcount fragility. CPython numpy buffers free deterministically
when the last ref drops, but any view we missed will pin the buffer.
Add an assertion in dev mode that, after the threshold step, the bool's
`.base` is `None` (i.e. no zero-copy aliasing to the original fp16
buffer).

### P2. Skip the standalone bool allocation [proposal]

Cuts another ~36 GB from peak by folding the threshold into the CC
kernel input path. Two options, pick one.

**P2a. Pack the bool into one byte per voxel via `np.packbits` per axis.**
3 channels × `ceil(N/8)` bits per channel = ~4.5 GB instead of 36 GB.
Numba kernel needs a bit-unpack at the inner read site — a few extra
shifts per neighbor lookup; on memory-bandwidth-bound workloads this is
usually a net speedup, not a regression.

**P2b. Stream-threshold inside the kernel.** Pass `affinities` (fp16) +
`threshold` to the numba sweep directly; the kernel reads
`affinities[c, ...] > threshold` on demand. Eliminates the bool
allocation entirely, but ties the kernel to the input dtype/layout and
forces the threshold to be known at JIT time (or paid as a runtime
compare). Probably worth it given the volume sizes here.

Constraint: `_affinity_foreground_mask` (segmentation_kernels.py:213)
also needs the bool today for the foreground `.any` pass and the
edge-destination OR. Either reuse the same packed/streamed input, or
fuse the foreground build into the seed-label cumsum.

### P3. Fuse foreground OR + cumsum seeding [proposal]

Cuts ~12 GB transiently. The current code:

```python
fg = _affinity_foreground_mask(hard_aff, edge_offset)        # ~12 GB bool (X,Y,Z)
fg_flat = fg.reshape(-1)                                     # view
labels_flat = np.cumsum(fg_flat, dtype=label_dtype)          # ~49 GB uint32
labels_flat *= fg_flat                                       # in-place
seg = labels_flat.reshape((X, Y, Z))
del fg, fg_flat
```

The `cumsum + multiply` pattern produces `labels_flat` (49 GB) while
`fg` (12 GB) is still live, then frees `fg`. We could:

- run cumsum into `seg` in chunks, masking by `fg` on the fly (avoids
  the temporary copy from `cumsum`'s output), or
- build `fg` lazily as a generator over Z-slabs that `cumsum` consumes,
  never materializing the full bool.

Lower priority than P1/P2; only matters once P1/P2 land.

### P4. Always prefer the streamed/chunked decode for full-volume runs [proposal]

`connectomics/decoding/streamed_chunked.py` already exists and stitches
CC results across chunks (see `decode_affinity_cc` mention in that
file). For a 3000×3000×1350 volume on a single host, a chunked decode
with ~512³ chunks puts the working set under ~30 GB even with the
naive bool, no kernel changes needed.

Action: add a config-level guard rail that selects streamed_chunked when
`predictions.nbytes` exceeds an env-tunable budget (default ~32 GB), and
document the threshold in `tutorials/neuron_nisb/base_banis.yaml`.

Caveat: streamed_chunked needs cross-chunk label stitching. Verify it
still produces bit-exact instance IDs against the in-memory parallel
backend (the audit notes in CLAUDE.md mention the stitching path; the
test surface is `tests/unit/test_chunked_inference.py` and
`tests/unit/test_affinity_processing.py` — neither pins
single-volume-vs-chunked equivalence on >1 GB inputs today).

### P5. Move the threshold + CC to GPU when affinities live on host [proposal]

`backend: cupy` already exists; the friction is host→device transfer of
~73 GB plus the ~17×volume GPU footprint (cupy backend doc at
`segmentation.py:489`). On A10 (23 GB) this does not fit. On an 80 GB
GPU (`g007-g009` per memory) it might, with `keep_input_on_cpu=true`
flipped and the affinity streamed in chunks.

Lower priority; only relevant if the cupy backend gets a streamed
variant.

## Contract / API changes

None proposed yet. P1 is purely internal refcount discipline. P2/P3
change the internal kernel signature but `decode_affinity_cc`'s public
signature stays the same. P4 is config-only.

## Tests

Tests not run in this turn — none of the proposals are implemented.

When P1 lands, add a regression test that asserts peak RSS for a
representative full-volume affinity decode stays under a budget
(skippable on CI if the volume is too large; gate by env). When P2
lands, add bit-exact equivalence tests against the current
`_propagate_affinity_sweep` output on small fixtures.

Existing relevant tests:
- `tests/unit/test_affinity_processing.py` — covers the decoder API.
- `tests/unit/test_inference_stage.py` — decode-from-cache path.
- `tests/unit/test_chunked_inference.py` — streamed_chunked plumbing.

## Review focus

- **Refcount discipline (P1).** Every place that takes a numpy array
  through this pipeline holds at least one alias. The change is correct
  only if *all* aliases drop before `del affinities` inside the
  decoder. A missed alias silently regresses to the old peak with no
  loud failure mode. Audit `apply_decode_pipeline`, `run_decoding_stage`,
  `_process_decoding_postprocessing`, and the cache loader
  `_load_cached_predictions` (test_pipeline.py:511) together.
- **`orphan_fill=True` path.** P1's `del` is gated on `orphan_fill` —
  do not regress the orphan-fill branch which re-reads `short_range_aff`
  at segmentation.py:581.
- **Kernel-input ABI (P2).** Switching from bool to packed/streamed
  input changes hot-path memory access patterns; benchmark on the same
  3000×3000×1350 volume before merging.
- **CPU vs swap interaction.** Today's bottleneck is paging, not
  arithmetic. The "did this help?" signal is `VmSwap` from
  `/proc/<pid>/status`, not wall-clock alone — a run that just barely
  fits and never swaps will look much faster than its CPU profile
  predicts.

## Untested / known gaps

- No regression test pins peak RSS for the full-volume decode path.
- `streamed_chunked` correctness vs. in-memory backend on real-size
  inputs is not asserted in CI.
- Numba thread count is bounded by the cgroup CPU mask, not by
  `NUMBA_NUM_THREADS`. Document that `--cpus-per-task` directly caps
  the parallel CC kernel's effective threads; no code path enforces a
  minimum.

## Migration notes

None for P1–P3 (internal). P4 would change the default decode path for
large volumes; gate behind an opt-in flag for one release, then flip.

## Operational guidance for the current run (gdw001)

For the in-progress
`python scripts/main.py --config tutorials/neuron_nisb/base_banis.yaml --mode test --checkpoint outputs/nisb_base_banis/20260427_095218/checkpoints/step=00050000.ckpt`
job, the immediate unblock is more RAM, not more CPUs:

- `--cpus-per-task=8 --mem=320G --gres=gpu:1` is the recommended SLURM
  shape until P1 lands.
- 2 → 8 CPUs gives ~2–4× wall-clock improvement on the SW + CC stages
  (numba parallel scales with `prange`).
- 8 → 16 CPUs gives a further ~1.3–1.5×; past 16 the gain is flat.
- Memory headroom matters far more than CPU count while the bool +
  fp16 + uint32 buffers all coexist.
