# j0126 resource budget

Planning figures for the whole-volume run. See [README.md](README.md) for the
workflow itself and [CLEANUP.md](CLEANUP.md) for reclaiming space as it runs.

These are planning figures for the complete 5,700×10,913×10,664 voxel volume, not guarantees. They assume the configured 726 affinity chunks, a 48 GB-class GPU, a parallel filesystem, and no queue time. With the [staged cleanup](CLEANUP.md), expect a **6–7 TiB peak** and reserve **8 TiB**. Keeping every intermediate or a second affinity arm can require 10 TiB. After final verification, retaining only the result and audit artifacts should take well below 1 TiB; retaining the source affinity raises that to roughly 3–4 TiB.

| Step | Compute specification | Estimated wall time | Storage while running |
|---|---|---:|---:|
| 1. Affinity | 1 GPU with at least 48 GB per shard; 726 independent shards | ~30 min/shard; ~6 h at 64 GPUs, ~5 h at 80 GPUs | 2.5–3 TiB for the float16 chunk store |
| 2. ABISS | Layer-aware CPU fleet; 16 CPU / 130 GB nodes for atomic chunks, then 1–8 CPU workers with 40–100 GB per composite chunk | 3.75 h measured on 40×16 CPU workers; ~1.9 h projected with the per-layer multi-node layout described in [Step 2](README.md#step-2--conservative-abiss-decode) | input affinity plus ~0.7 TiB resumable scratch, a precomputed affinity layer, and ~45 GB final segmentation; budget 4 TiB additional |
| 3. Error correction | 80 array tasks, 8 CPU workers/task and 64 GB/task; reductions run serially | 6–12 h estimate; this has not yet been benchmarked end-to-end | reuse steps 1–2 inputs; reserve 0.5–1 TiB for skeleton, contact, and output artifacts |

The step-1 timing is measured from the chunked Zarr input path. On a 40 GB GPU, lower `sw_batch_size` from 12 before running; that increases the per-chunk time. Reading tiled PNGs instead can take roughly 19 h **per chunk** and is not a usable production path. ABISS scratch and `work/` artifacts are only needed for resume; retain the final precomputed segmentation, parameter file, and manifests after recording the result, then reclaim scratch space.
