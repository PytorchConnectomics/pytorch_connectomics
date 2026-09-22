# Code v0
## Overview
Implemented the approved behavior-preserving reorganization: reusable J0126 CloudVolume/LUT evaluation primitives now live in the `em_erl` package, and the former `scripts/` tree is now `examples/` with runnable source-checkout bootstraps. No git commit was created.

## What Changed
- Moved reusable helpers from `j0126_workflow.py` into `em_erl.io`, `em_erl.sampling`, and `em_erl.eval`.
- Exported the new public API from `em_erl.__init__`.
- Renamed `scripts/` to `examples/` with `git mv`.
- Rewrote `examples/j0126_workflow.py` as a thin CLI/wrapper over `em_erl.evaluate_skeletons_cloudvolume`.
- Updated J0126 tests to import from `em_erl` and monkeypatch `em_erl.eval.open_seg_cloudvolume`.
- Updated README references from `scripts/` to `examples/`.

## Implementation Details
- `open_seg_cloudvolume` keeps `cloudvolume` as a lazy import.
- `evaluate_skeletons_cloudvolume` requires `seg_url`; the J0126 default URL remains only in `examples/j0126_workflow.py`.
- `evaluate_skeletons_cloudvolume` calls module-level `open_seg_cloudvolume` and `sample_cloudvolume_lut`, so tests can monkeypatch `em_erl.eval.open_seg_cloudvolume`.
- `load_skeletons` imports `h5py` lazily to avoid making base `import em_erl` require the optional HDF5 extra.

## Files Changed
| File | Purpose |
|---|---|
| `em_erl/io.py` | Added segmentation URL normalization, CloudVolume opener, and skeleton HDF5 loader. |
| `em_erl/sampling.py` | Added CloudVolume chunk sampler for node-to-segment LUTs. |
| `em_erl/eval.py` | Added LUT save/load/validate, LUT scoring, and generalized CloudVolume skeleton evaluation. |
| `em_erl/__init__.py` | Exported the new public API names. |
| `examples/j0126_workflow.py` | Thin J0126 CLI plus `run_j0126_eval` wrapper over package API. |
| `examples/volume_eval.py` | Moved example, added source-checkout bootstrap, updated usage comments. |
| `examples/seg_to_graph.py` | Moved example, added source-checkout bootstrap, updated usage comments. |
| `examples/skel_to_graph.py` | Moved example, added source-checkout bootstrap, updated usage comments. |
| `examples/README.md` | Moved docs, updated paths, documented demonstrated `em_erl` APIs. |
| `README.md` | Updated `scripts/` references to `examples/`. |
| `tests/test_j0126_workflow.py` | Switched from importlib script loading to package API imports. |

## Git Baseline
run_start_ref: 3549cad07165e9bcd3949501eb611228c66b628b
current_head: 3549cad07165e9bcd3949501eb611228c66b628b

## Verification
`python -m pytest tests/ -q`
```text
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
======================== 48 passed, 1 warning in 1.32s =========================
```

`python examples/j0126_workflow.py -h ; python examples/volume_eval.py -h ; python examples/seg_to_graph.py -h ; python examples/skel_to_graph.py -h`
```text
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
usage: volume_eval.py [-h] -p PRED_PATH [-g GT_PATH] -r GT_RESOLUTION
                      [-gz GT_ZRANGE] [-m GT_MASK_PATH] [-t MERGE_THRESHOLD]
                      [-i ERL_INTERVALS] [-v VERBOSE] [-o OUTPUT_PATH]
                      [-w NUM_WORKERS] [-c CHUNK_NUM]

ERL evaluation on small volume

options:
  -h, --help            show this help message and exit
  -p PRED_PATH, --pred-path PRED_PATH
                        path to the segmentation prediction
  -g GT_PATH, --gt-path GT_PATH
                        path to ground truth network-lite graph
  -r GT_RESOLUTION, --gt-resolution GT_RESOLUTION
                        resolution of the ground truth skeleton (zyx-order).
                        e.g., 30,32,32
  -gz GT_ZRANGE, --gt-zrange GT_ZRANGE
                        range of z in the gt. Full range of z if unspecified.
                        Example: "0,979" for the first 980 slices
  -m GT_MASK_PATH, --gt-mask-path GT_MASK_PATH
                        path to ground truth mask for false merge
  -t MERGE_THRESHOLD, --merge-threshold MERGE_THRESHOLD
                        number of false merge voxels to classify a gt skel to
                        have the false merge error
  -i ERL_INTERVALS, --erl-intervals ERL_INTERVALS
                        compute erl for each range. e.g., 0,5000,50000,150000
  -v VERBOSE, --verbose VERBOSE
                        store detailed info
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        output pickle file path. e.g., erl_score.pkl
  -w NUM_WORKERS, --num-workers NUM_WORKERS
                        number of worker processes for HDF5/Zarr segmentation
                        sampling
  -c CHUNK_NUM, --chunk-num CHUNK_NUM
                        number of z chunks used for segmentation sampling
usage: seg_to_graph.py [-h] -s SEG_PATH [-r SEG_RESOLUTION] [-o OUTPUT_PATH]
                       [-l LENGTH_THRESHOLD] [-t NUM_THREAD]

Convert gt segmentation to graph of skeleton

options:
  -h, --help            show this help message and exit
  -s SEG_PATH, --seg-path SEG_PATH
                        path to the ground truth segmentation
  -r SEG_RESOLUTION, --seg-resolution SEG_RESOLUTION
                        resolution of the ground truth segmentation (zyx-
                        order). e.g., 30,32,32
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        output npz file path. e.g., gt_graph.npz
  -l LENGTH_THRESHOLD, --length-threshold LENGTH_THRESHOLD
                        throw away skeletons that are shorter than the
                        threshold
  -t NUM_THREAD, --num-thread NUM_THREAD
                        number of threads for skeletonization
usage: skel_to_graph.py [-h] -s SKEL_PATH [-o OUTPUT_PATH]
                        [-l LENGTH_THRESHOLD] [-r SAMPLE_RATIO]

Convert gt segmentation to graph of skeleton

options:
  -h, --help            show this help message and exit
  -s SKEL_PATH, --skel-path SKEL_PATH
                        path to the ground truth skeleton
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        output npz file path. e.g., gt_graph.npz
  -l LENGTH_THRESHOLD, --length-threshold LENGTH_THRESHOLD
                        throw away skeletons that are shorter than the
                        threshold
  -r SAMPLE_RATIO, --sample-ratio SAMPLE_RATIO
                        randomly sample skeletons by the ratio
```

`python examples/volume_eval.py -p tests/data/vol_pred.h5 -g tests/data/gt_graph.npz -r 30,30,30`
```text
Load data
Compute seg lookup table for gt skeletons
Skeleton points with assignment 0: 0/204 (ratio: 0.000000)
Compute erl
all skel
ERL	: 2952.02
gt ERL	: 4166.81
#skel	: 2
-----------------
```

`grep -rn "importlib\|scripts/" tests/ ; grep -rn "scripts/" README.md examples/`
```text
tests/test_banis_compat.py:1:import importlib.util
tests/test_banis_compat.py:19:    spec = importlib.util.spec_from_file_location(
tests/test_banis_compat.py:23:    run_length = importlib.util.module_from_spec(spec)
tests/test_banis_compat.py:43:    spec = importlib.util.spec_from_file_location("banis_metrics_for_erl_test", metrics_path)
tests/test_banis_compat.py:44:    module = importlib.util.module_from_spec(spec)
```
The grep output is only pre-existing BANIS compatibility importlib usage. There are no `scripts/` matches in `tests/`, `README.md`, or `examples/`.

## Review Focus
- Confirm the relocated function bodies preserve the old J0126 behavior.
- Confirm `evaluate_skeletons_cloudvolume` has no J0126 default URL and uses the monkeypatchable `em_erl.eval.open_seg_cloudvolume`.
- Confirm examples run from a source checkout without installed `em_erl`.
- Confirm docs and tests no longer refer to moved `scripts/` paths.

## Risks and Unknowns
- No network CloudVolume sampling was run; verification stayed offline as requested.
- Pytest passed, but emitted a cache warning because the parent `.pytest_cache` path is read-only.
- The requested broad grep still reports unrelated pre-existing `importlib` use in `tests/test_banis_compat.py`; I left that test unchanged per scope.

## Changes Since Previous Code Version
Initial implementation.