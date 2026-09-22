# Code v1
## Overview
Implemented the J0126 LUT persistence and reuse workflow. Existing LUT scoring now rebuilds the ERL graph from `-g`, loads the LUT with `read_vol`, casts it to `uint64`, validates node count, and never opens CloudVolume. Missing LUT paths still use the existing occupied-chunk sampling strategy, then save the reusable HDF5 LUT.

## What Changed
- Added `--lut PATH` for download-once LUT generation and zero-GCS reuse.
- Added `--mip INT` with default `0` and documented approximate coarser-mip tradeoff.
- Added `--cache-dir PATH` and passed it to CloudVolume as `cache=PATH`, otherwise `cache=False`.
- Factored LUT loading/saving, graph/LUT validation, and graph scoring into unit-testable functions.
- Updated README J0126 workflow docs and expanded offline tests.

## Implementation Details
- `run_j0126_eval` always loads skeletons and rebuilds the ERL graph from `-g`.
- If `--lut` exists, `load_node_segment_lut()` uses `em_erl.io.read_vol()` and casts to `np.uint64`; the CloudVolume opener is not called.
- If `--lut` is missing, the script samples exactly as before, validates LUT length, writes it with `em_erl.io.write_h5()`, and prints the saved path.
- `validate_node_segment_lut()` raises a clear mismatch error when `len(lut) != graph.num_nodes`.
- `score_graph_with_lut()` and `score_skeletons_with_lut()` allow offline tests to score tiny in-memory skeletons without network access.

## Files Changed
| File | Purpose |
|---|---|
| `scripts/j0126_workflow.py` | Added LUT persistence/reuse, mip/cache plumbing, validation, and reusable scoring helpers. |
| `scripts/README.md` | Documented the download-once LUT workflow, 3.6 GB vs 4 MB reuse, zero-CloudVolume scoring, mip tradeoff, and cache directory. |
| `tests/test_j0126_workflow.py` | Added offline LUT round-trip/reuse, mismatch guard, mip plumbing, and CloudVolume cache constructor tests while keeping existing sampler and URL tests. |

## Git Baseline
run_start_ref: c24e68598b7fc3c0cfaf22faece5f2ea6d5f5949
current_head: c24e68598b7fc3c0cfaf22faece5f2ea6d5f5949

## Verification
Network end-to-end job was not run, per instruction; deferred to coordinator.

```text
$ source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc && python scripts/j0126_workflow.py -h
usage: j0126_workflow.py [-h] -g GT_SKELETON [--seg-url SEG_URL]
                         [-mt MERGE_THRESHOLD] [-w NUM_WORKERS] [--lut LUT]
                         [--mip MIP] [--cache-dir CACHE_DIR] [-o OUTPUT_PATH]

Compute J0126 ERL from GT skeletons by sampling the public FFN segmentation
CloudVolume once or by reusing a saved node LUT.

options:
  -h, --help            show this help message and exit
  -g GT_SKELETON, --gt-skeleton GT_SKELETON
                        path to ground truth skeleton HDF5 file
  --seg-url SEG_URL     CloudVolume segmentation URL
  -mt MERGE_THRESHOLD, --merge-threshold MERGE_THRESHOLD
                        threshold number of voxels to be a false merge
  -w NUM_WORKERS, --num-workers NUM_WORKERS
                        number of CloudVolume chunk fetch worker threads
  --lut LUT             optional node-to-segment LUT HDF5 path; if it exists,
                        load it and score without opening CloudVolume,
                        otherwise sample and save it
  --mip MIP             CloudVolume mip for LUT generation (default 0,
                        faithful). A coarser mip downloads about 4x less data
                        per level and is faster, but changes about 1.5% of
                        sampled node labels (measured mip1==mip0 0.985,
                        mip2==mip0 0.980), so ERL drifts slightly; use only
                        for a cheaper/faster approximation.
  --cache-dir CACHE_DIR
                        optional CloudVolume cache directory for LUT
                        generation; stores raw chunks locally to avoid re-
                        download on a generation re-run, off by default
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        optional path for pickled ERLScore output
```

```text
$ source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc && python -m pytest tests/test_j0126_workflow.py -q
============================= test session starts ==============================
platform linux -- Python 3.11.14, pytest-9.0.1, pluggy-1.6.0
rootdir: /projects/weilab/weidf/lib/pytorch_connectomics
configfile: pyproject.toml
plugins: cov-7.0.0, anyio-4.12.1
collected 7 items

tests/test_j0126_workflow.py .......                                     [100%]

=============================== warnings summary ===============================
../../../miniconda3/envs/pytc/lib/python3.11/site-packages/_pytest/cacheprovider.py:475
  /projects/weilab/weidf/lib/miniconda3/envs/pytc/lib/python3.11/site-packages/_pytest/cacheprovider.py:475: PytestCacheWarning: cache could not write path /projects/weilab/weidf/lib/pytorch_connectomics/.pytest_cache/v/cache/nodeids: [Errno 30] Read-only file system: '/projects/weilab/weidf/lib/pytorch_connectomics/.pytest_cache/v/cache/nodeids'
    config.cache.set("cache/nodeids", sorted(self.cached_nodeids))

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
========================= 7 passed, 1 warning in 0.35s =========================
```

```text
$ source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc && python -m pytest tests/ -q
============================= test session starts ==============================
platform linux -- Python 3.11.14, pytest-9.0.1, pluggy-1.6.0
rootdir: /projects/weilab/weidf/lib/pytorch_connectomics
configfile: pyproject.toml
plugins: cov-7.0.0, anyio-4.12.1
collected 48 items

tests/test_banis_compat.py .                                             [  2%]
tests/test_j0126_workflow.py .......                                     [ 16%]
tests/test_unit.py ........................................              [100%]

=============================== warnings summary ===============================
../../../miniconda3/envs/pytc/lib/python3.11/site-packages/_pytest/cacheprovider.py:475
  /projects/weilab/weidf/lib/miniconda3/envs/pytc/lib/python3.11/site-packages/_pytest/cacheprovider.py:475: PytestCacheWarning: cache could not write path /projects/weilab/weidf/lib/pytorch_connectomics/.pytest_cache/v/cache/nodeids: [Errno 30] Read-only file system: '/projects/weilab/weidf/lib/pytorch_connectomics/.pytest_cache/v/cache/nodeids'
    config.cache.set("cache/nodeids", sorted(self.cached_nodeids))

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
======================== 48 passed, 1 warning in 1.18s =========================
```

## Review Focus
- Confirm the existing-LUT branch cannot call `open_seg_cloudvolume()` and does not require `cloud-volume`.
- Confirm the freshly sampled LUT is validated before it is written.
- Confirm `--mip` and `--cache-dir` only affect LUT generation, not saved-LUT scoring.
- Confirm README wording accurately communicates faithful mip0 vs approximate coarser mips.

## Risks and Unknowns
- Live CloudVolume end-to-end verification was intentionally not run.
- Coarser mip ERL drift is documented as measured input data, but not re-measured here.
- Pytest emitted a cache warning because the parent repo `.pytest_cache` path is read-only; tests still passed.

## Changes Since Previous Code Version
- Addressed `[major] Persist the LUT and reuse it`: added `--lut`, `read_vol` loading with `uint64` cast, `write_h5` saving, graph rebuild from `-g`, length validation, and an offline test proving reuse does not open CloudVolume.
- Addressed `[minor] Optional --mip`: added CLI option, default `0`, opener plumbing, help text, README tradeoff docs, and offline capture test.
- Addressed `[minor] Optional local chunk cache`: added `--cache-dir`, CloudVolume `cache=PATH` / `False` behavior, help text, README docs, and constructor plumbing test.
- Confirmed sampling strategy unchanged: the occupied-chunk grouping/fetching logic remains the same and still fetches each occupied chunk once.