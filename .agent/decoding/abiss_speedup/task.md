# Task

Look for ways to speed up the computation and reduce the memory footprint of `lib/abiss`.
Read the existing context.

## Repository under change

**`lib/abiss` is a separate git repository** nested inside `pytorch_connectomics`, and
`lib/` is gitignored by the parent (`.gitignore:156`). Changes to `lib/abiss` therefore
appear in NO pytorch_connectomics diff. All CCC git baselines and code-review diffs for
this run use the `lib/abiss` repo:

```
repo root : /projects/weilab/weidf/lib/pytorch_connectomics/lib/abiss
branch    : feature/configurable-edge-score
```

## Existing context to read

Measured facts already established (do not re-derive; verify if load-bearing):

- `dev/zebrafinch/lesson_abiss.md` — L112 (whole-volume replay + build/env bugs),
  **L117** (where wall-clock actually goes), L118 (r1 vs r10 soma experiment).
- `dev/zebrafinch/lesson_efficiency.md` — the full efficiency writeup, including the
  measured per-stage table and the memory/concurrency argument.
- `.agent/features/abiss_h5/` — SUMMARY.md, review_v1/v2, response_v1 for the HDF5/zarr
  backend work that precedes this.

### Key measurements already in hand

Whole-volume run (SLURM `2782527..2782540`), sum of stage maxima, stages are sequential:

| | chunks | shards×cpus | concurrency | min | maxRSS GB |
|---|---:|---:|---:|---:|---:|
| ws_L0 | 10626 | 40×16 | 640 | 29.5 | 130.0 |
| ws_L2 | 216 | 4×8 | 32 | 11.1 | 130.0 |
| ws_L3 | 27 | 2×4 | 8 | 12.4 | 120.0 |
| ws_L4 | 8 | 1×2 | 2 | 17.9 | 160.0 |
| ws_L5 | 1 | 1×1 | 1 | 10.4 | 98.8 |
| me_L0 | 10626 | 40×16 | 640 | 20.9 | 130.0 |
| me_L4 | 8 | 1×2 | 2 | 33.2 | 78.2 |
| me_L5 | 1 | 1×1 | 1 | 26.3 | 98.4 |

TOTAL 225 min. Layer-0 stages 82 min (36%); upper composite layers 143 min (64%).

- **The binaries are single-threaded.** Zero `tbb::parallel`, zero `#pragma omp`, zero
  `std::thread` in `src/` (one `std::async`). libtbb is linked but unused for parallelism.
  One chunk = one core.
- **Per-chunk memory** (maxRSS ÷ concurrency): ~8 GB at L0, ~16 at L2, ~20 at L3,
  ~39 at L4, ~98 at L5. Upper layers hold region graphs, not affinity.
- **Only layer-0 reads affinity.** `aff.raw` is opened in exactly three files:
  `src/seg/atomic_chunk.cpp`, `src/seg/atomic_chunk_ME.cpp`, `src/seg/contact_surface.cpp`.
  Composite (upper-layer) binaries never touch it.
- **Affinity is mmap'd, not heap**: `bio::mapped_file_source` + `ConstChunkRef<aff_t,4>`,
  read-only, never copied. `aff.raw` for a 512×512×256 chunk = 805 MB at float32.
- **`aff_t` is both storage AND arithmetic type** (`src/global_types.h`). Used for
  accumulators in `src/agg/edges.h` (`mean_edge::sum`, `atomic_edge::sum_aff`,
  `mst_edge::sum`) and thresholds in `src/agg/mean_aggl.cpp`. Narrowing it wholesale would
  change segmentation numerics.
- **CPU: Intel Xeon Gold 5222 (Cascade Lake)** — has `f16c` (fp16<->fp32 conversion), does
  NOT have `avx512_fp16` (fp16 arithmetic). fp16 math would be emulated.
- Re-sharded run measured 1.48x overall; `ws_L5` got *slower* (10.4 -> 16.4 min) after its
  memory allocation was cut 300G -> 200G, and `ws_L4` gained only 1.23x at 4x concurrency,
  implying chunk cost within a layer is non-uniform.

## Hard constraints

1. **Segmentation output must not change.** This pipeline reproduces a Seuron provenance
   record (`seuron_provenance.json`). `CHUNK_SIZE [512,512,256]`, `AGG_THRESHOLD 0.3`,
   `WS_HIGH/LOW 0.99999/0.00001` come from that record. Any change that alters the
   resulting segmentation voids the reproduction and is out of scope unless explicitly
   gated behind an opt-in flag that is off by default.
2. Numerical changes to agglomeration accumulation are high-risk for the same reason.
3. The scheduling/sharding layer (`submit_wholevol_sharded.sh`, `sbatch_abiss_shard.sh`)
   lives in pytorch_connectomics `dev/zebrafinch/`, NOT in lib/abiss. Prefer changes
   inside lib/abiss; call out cross-repo work explicitly if it is the better lever.

## Deliverable

A plan identifying concrete, ranked speed and memory opportunities in `lib/abiss`, each
with: the mechanism, an estimated payoff tied to the measurements above, the risk to
segmentation fidelity, and how it would be verified.
