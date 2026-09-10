# Moritz L4 — measured resource figures

Every number here was measured on this volume, on BC, in September 2026. Where a figure
is an extrapolation it says so. Compare with `neuron_j0126/RESOURCE.md`, which is a
4.2x larger volume.

## The volume

| | |
|---|---|
| extent (ZYX, volume frame) | 3306 x 8534 x 5599 = 158 Gvoxel |
| resolution | 11.24 x 11.24 x 28.0 nm (XYZ) |
| physical size | 63 x 96 x 93 um |
| affinity | float16, 3 channels, precomputed, **645 GB** |
| keep mask | 4x8x8 grid, 827 x 1067 x 700, **4.9 MB** |
| nucleus instances | 4x8x8 grid, 101 instances, 34 MB |

## Step 2 — ABISS decode

**One shared-memory job, never an array.** Hierarchy levels have strict barriers and all
workers write the same output layers. ABISS's concurrency *is* the CPU count granted to
the process (`lib/abiss/scripts/init.sh:107` runs `parallel -j $(nproc affinity)`).

### Memory is set by chunk size x cpu count

Every running chunk mmaps its own `aff.raw`, which the C++ binaries read as `aff_t`
(float32) — so a float16 affinity is **doubled** on the way in:

| CHUNK_SIZE | grid | padded voxels | % of the 2^31 cap | `aff.raw` | 64 slots need |
|---|---|---:|---:|---:|---:|
| [5599, 8534, 32] | 1 x 1 x 104 | 1.63e9 | 76% | 19.5 GB | 1.2 TB |
| **[2816, 2944, 32]** | **2 x 3 x 104** | 2.82e8 | 13% | **3.39 GB** | **~173 GB** |
| [2816, 4352, 32] | 2 x 2 x 104 | 4.17e8 | 19% | 5.01 GB | ~256 GB |

The measured bound is **<= 9.6 bytes per padded voxel per slot** (job 2982513 ran 32
slots x 1.63e9 padded voxels inside a 500 GB allocation without OOM). Multiply that by
your chunk's padded voxel count and your cpu count before choosing `--mem`.

`Z = 32` is not tunable upward: ABISS's watershed uses `internal_seg_t = uint32_t` and
asserts `chunk_size < watershed_traits<uint32_t>::high_bit` (`atomic_chunk.cpp:135`), so
a padded chunk must stay under 2^31 = 2.147e9 voxels. z=80 is 183% of that and dies on
the assertion (job 2954910).

### What one chunk costs, and the trap

| configuration | chunk | per-chunk wall time |
|---|---|---:|
| WS_HIGH 0.5, whole volume | 1.53 Gvox | ~28 min |
| WS_HIGH 0.95, whole volume | 1.53 Gvox | **4.1 h** |
| WS_HIGH 0.5, pre-flight box | 33 Mvox | 7:05 total |
| WS_HIGH 0.95, pre-flight box | 33 Mvox | 13:17 total |

Raising WS_HIGH costs 1.9x when a chunk fits in memory and 8.8x when it does not. At
[5599, 8534, 32] the working set is 32 x 19.5 GB = **624 GB against a 500 GB
allocation**, and job 2982513 spent 12 hours on 59 of 104 chunks. That is thrashing, not
computation: per chunk it moves 19.5 GB in and 19.5 GB out over 4.1 h, about 2.6 MB/s.

Cutting the supervoxel count does **not** fix it — `WS_SIZE/WS_DUST` at 2500/1250 gives
3.8x fewer supervoxels and the same runtime (14:04 vs 13:17 on the pre-flight box).
Chunk size is the lever.

### Recommended request

```
#SBATCH --partition=long        # 24 h is not enough head-room; see below
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
```

64 cores + 200 GB fits the 113 nodes in `medium`/`long` with 64 cores and 257 GB. The 18
nodes with 96 cores and 2 TB are the fast option but are heavily contended — a 32-core
request at 1500G sat unschedulable (job 2982512) while the same job at 500G started
immediately.

**Do not use a 24-hour partition.** `scontrol update TimeLimit` is refused once a job is
running ("Job is no longer pending execution"), so a 24 h limit that turns out to be
short cannot be extended. Resume is per chunk (`scratch/done/<task>.txt`, checked by
`run_wrapper.sh`), so an interrupted run continues rather than restarting — but queue a
`--dependency=afterany:` follow-on rather than relying on noticing.

### Disk

| | |
|---|---|
| affinity (input, not copied) | 645 GB |
| `run/` while running | ~580 GB at 32 x 19.5 GB chunks; ~220 GB at 64 x 3.39 GB |
| `scratch/` | ~56 GB |
| final `seg` + `ws` precomputed | ~11 GB each (extrapolated from j0126, 4.2x larger) |

`run/<chunk>/aff.raw` is deleted when its chunk completes (`run_wrapper.sh` does
`rm -rf "$CHUNK"`), so the working directory plateaus at cpu-count chunks rather than
growing. **No copy of the affinity is ever made** — `source_affinity_h5` is deliberately
omitted and `AFF_PATH` is used in place.

## Step 3 — scoring

`projects/2026_moritz_l4/s3_score.py`, 8 cpus / 64 GB, ~5 min. It reads segment ids at
263,938 skeleton nodes grouped by storage chunk (59,460 reads), so it is I/O bound and
does not need the segmentation in memory.
