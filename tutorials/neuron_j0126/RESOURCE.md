# j0126 resource budget

Planning figures for the whole-volume run. See [README.md](README.md) for the
workflow itself and [CLEANUP.md](CLEANUP.md) for reclaiming space as it runs.

These are planning figures for the complete 5,700×10,913×10,664 voxel volume, not guarantees. They assume the configured 726 affinity chunks, a 48 GB-class GPU, a parallel filesystem, and no queue time. With the [staged cleanup](CLEANUP.md), expect a **6–7 TiB peak** and reserve **8 TiB**. Keeping every intermediate or a second affinity arm can require 10 TiB. After final verification, retaining only the result and audit artifacts should take well below 1 TiB; retaining the source affinity raises that to roughly 3–4 TiB.

| Step | Compute specification | Estimated wall time | Storage while running |
|---|---|---:|---:|
| 1. Affinity | 1 GPU with at least 48 GB per shard; 726 independent shards | ~30 min/shard; ~6 h at 64 GPUs, ~5 h at 80 GPUs | 2.5–3 TiB for the float16 chunk store |
| 2. ABISS | Shipped driver: one shared-memory job; about 7.8 GB per concurrent 2048×2048×80 atomic task, plus overhead | Reported GPFS single-node watershed alone: ~38 h; historical 3.75 h used 40×16 CPU workers with a separate fleet launcher | input affinity plus scratch and full-resolution remap outputs; no precomputed affinity copy on the current HDF5 path |
| 3. Error correction | 80 array tasks, 8 CPU workers/task and 64 GB/task; reductions run serially | 6–12 h estimate; this has not yet been benchmarked end-to-end | reuse steps 1–2 inputs; reserve 0.5–1 TiB for skeleton, contact, and output artifacts |

The step-1 timing is measured from the chunked Zarr input path. On a 40 GB GPU, lower `sw_batch_size` from 12 before running; that increases the per-chunk time. Reading tiled PNGs instead can take roughly 19 h **per chunk** and is not a usable production path. ABISS scratch and `work/` artifacts are only needed for resume; retain the final precomputed segmentation, parameter file, and manifests after recording the result, then reclaim scratch space.

The default ABISS request of 64 CPUs / 250G does not cover the measured ~500 GB
atomic-task total at that concurrency. Benchmark 8–16 workers or raise memory, and
size walltime for the site's filesystem. More workers can reduce throughput: the
reproduction measured ~170 MB/s for one read stream versus ~65 MB/s total for 64.
The shipped driver does not provide the fleet used for the historical timing.

Budget **inodes as well as bytes**. The report counted about 162,540 files for EM
chunks of 64×256×256, 22,264 for the current keep mask, and 10,890 for tissue,
before affinity, ABISS and correction artifacts. Check the site's file quota first.
Measured EM window reads were 13.4 ms with 64×256×256 chunks, 49.7 ms with
128×512×512 (~20,790 files), and 384 ms with 64×2048×2048 (~3,240 files).
The inode-cheap layout can therefore cost about 29× per read. Download slab/tile
settings control work units; the downloader currently fixes XY storage chunks at
256×256 and caps Z chunks at 64. Alternative storage layouts require a code change
or rechunking. Never change mask chunks unless each dimension divides the 1008-voxel
write cell, to avoid concurrent partial-chunk writes.
