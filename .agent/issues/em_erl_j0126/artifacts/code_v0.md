# Code v0
## Overview
Rewrote `scripts/j0126_workflow.py` as a single-command J0126 ERL evaluator that loads GT skeletons, builds the canonical ERL graph, samples segment ids directly from the public CloudVolume layer, and computes ERL without tile downloads or intermediate LUT files.

## What Changed
- Replaced the four-subcommand tile workflow with one CLI requiring `-g/--gt-skeleton`.
- Added `normalize_seg_url`, lazy `open_seg_cloudvolume`, and chunk-binned `sample_cloudvolume_lut`.
- Updated J0126 README usage to the new CloudVolume command and dependency extra.
- Added offline unit tests with a fake CloudVolume for sampler and URL normalization behavior.

## Implementation Details
- The sampler consumes exactly `graph.get_nodes_position(None)` and returns a `uint64` LUT aligned to `graph.num_nodes`.
- Skeleton `zyx` coordinates are converted to CloudVolume `xyz` indexing as `cv[x, y, z]`.
- Out-of-bounds points are excluded from CloudVolume fetches and remain segment `0`.
- In-bounds points are grouped by `cv.chunk_size`, fetched chunk-by-chunk with `ThreadPoolExecutor`, and progress is printed during sampling.
- ERL scoring reuses `skel_to_erlgraph`, `print_skeleton_assignment_zero_stats`, and `compute_erl_score`.
- Optional `-o/--output-path` writes the `ERLScore` with `em_erl.io.write_pkl`.
- No `-r/--resolution` flag is present.

## Files Changed
| File | Purpose |
|---|---|
| `scripts/j0126_workflow.py` | Full rewrite to single-command CloudVolume-backed J0126 ERL workflow |
| `scripts/README.md` | Replaced old 4-step tile workflow docs with one-command CloudVolume usage |
| `tests/test_j0126_workflow.py` | Added offline sampler and URL normalizer unit tests |

## Git Baseline
run_start_ref: c24e68598b7fc3c0cfaf22faece5f2ea6d5f5949
current_head: c24e68598b7fc3c0cfaf22faece5f2ea6d5f5949

## Verification
Network end-to-end was not run; the sandbox has no GCS access, so the CloudVolume registration sentinel run is deferred to the coordinator/reviewer.

Command:
```bash
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc && python scripts/j0126_workflow.py -h
```

Output:
```text
usage: j0126_workflow.py [-h] -g GT_SKELETON [--seg-url SEG_URL]
                         [-mt MERGE_THRESHOLD] [-w NUM_WORKERS]
                         [-o OUTPUT_PATH]

Compute J0126 ERL from GT skeletons by sampling the public FFN segmentation
CloudVolume directly.

options:
  -h, --help            show this help message and exit
  -g GT_SKELETON, --gt-skeleton GT_SKELETON
                        path to ground truth skeleton HDF5 file
  --seg-url SEG_URL     CloudVolume segmentation URL
  -mt MERGE_THRESHOLD, --merge-threshold MERGE_THRESHOLD
                        threshold number of voxels to be a false merge
  -w NUM_WORKERS, --num-workers NUM_WORKERS
                        number of CloudVolume chunk fetch worker threads
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        optional path for pickled ERLScore output
```

Command:
```bash
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc && python -m pytest tests/test_j0126_workflow.py -q
```

Output:
```text
collected 3 items

tests/test_j0126_workflow.py ...                                         [100%]

========================= 3 passed, 1 warning in 0.35s =========================
```

Command:
```bash
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc && python -m pytest tests/ -q
```

Output:
```text
collected 44 items

tests/test_banis_compat.py .                                             [  2%]
tests/test_j0126_workflow.py ...                                         [  9%]
tests/test_unit.py ........................................              [100%]

======================== 44 passed, 1 warning in 1.71s =========================
```

Also ran:
```bash
git diff --check -- scripts/j0126_workflow.py scripts/README.md tests/test_j0126_workflow.py
```
No output; whitespace check passed.

## Review Focus
- Confirm the sampler’s zyx-to-xyz mapping and chunk clamping match the verified J0126 facts.
- Confirm the CLI surface is intentionally single-command and has no resolution option.
- Confirm `node_segment_lut` order is guaranteed by sampling `graph.get_nodes_position(None)` directly.

## Risks and Unknowns
- The real CloudVolume network run, skeleton `"0"` sentinel, and expected 37 OOB count were not executed in this sandbox.
- The sampler assumes the runtime CloudVolume exposes `volume_size` and `chunk_size` as xyz triples, as specified in the approved plan.

## Changes Since Previous Code Version
Initial implementation.