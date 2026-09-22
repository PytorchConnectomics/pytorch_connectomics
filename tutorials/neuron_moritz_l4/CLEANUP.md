# Moritz L4 — what to delete, and what never to delete

The decode's working directory is larger than its output. Mid-run and after, this is what
is safe to remove.

## Never

| path | why |
|---|---|
| `outputs/segem_banis_plus/.../raw_x1_ch0-1-2_chunked-raw_.../` | the 645 GB affinity. It is **not** copied anywhere — `2_abiss.yaml` omits `source_affinity_h5` and points `AFF_PATH` straight at it, so this directory is the only copy. Re-creating it means re-running inference. |
| `dataset/segEM/Moritz_l4_2019/em/keep_mask_z4y8x8.h5` | 4.9 MB, rebuildable in a minute (`scripts/build_moritz_l4_keep_mask.py`) but referenced by every decode. |
| `abiss/scratch/done/` | the per-chunk resume markers. Deleting these makes a resumed run redo every finished chunk. |

## While a run is in flight

`abiss/run/<chunk>/` holds one `aff.raw` per *running* chunk (3.39 GB each at the current
chunk size, 19.5 GB at the old one) and is deleted per chunk on completion by
`run_wrapper.sh`. It plateaus at cpu-count chunks. Do not prune it by hand — you will
delete the working set of a running task.

## After a run

| path | size | safe to delete |
|---|---|---|
| `abiss/run/` | ~0 after a clean finish | yes |
| `abiss/scratch/ws`, `abiss/scratch/seg`, `abiss/scratch/remap` | tens of GB | yes, once `precomputed/seg` is complete and scored |
| `abiss/scratch/done/` | KB | only if you will never resume |
| `abiss/chunkmap/` | small | keep; it maps supervoxels to segments |
| `abiss/precomputed/ws` | ~11 GB | keep while iterating on agglomeration; it is the input to it |

## Superseded runs

Name a run by what makes it different and keep the ones that are still evidence:

| directory | keep? |
|---|---|
| `abiss_agg0.001_percolated/` | the known-bad control the guard is calibrated against |
| `abiss_agg0.80_nomask/` | the no-mask baseline the current run is compared to |
| `abiss_sweep/*` | 14 pre-flight boxes, **40 GB** total. Their numbers are already in `2_abiss.yaml` and `RESOURCE.md`, so these can go |
| `abiss_smoke/` | keep one; it is the pre-flight |

Check what a run actually was before deleting it:

```bash
grep -E '"WS_HIGH_THRESHOLD"|"CHUNK_SIZE"|"AFF_KEEP_MASK"|"NUC_PATH"' <run>/secrets/param
```
