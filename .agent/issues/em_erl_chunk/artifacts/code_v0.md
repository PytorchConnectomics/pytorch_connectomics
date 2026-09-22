# Code v0
## Overview
Implemented the generic chunked-volume ERL orchestrator as a new example CLI, with offline coverage for chunk mapping, reduction, scoring, wait behavior, and SLURM script generation. No git commit was made.

## What Changed
- Added `examples/volume_eval_chunk.py` with map -> reduce -> score orchestration over existing `em_erl` primitives.
- Added `examples/volume_eval_chunk.yaml` as a j0126-style tile config.
- Added `tests/test_volume_eval_chunk.py` with synthetic offline tests, including full LUT equality against monolithic sampling.
- Added a `volume_eval_chunk.py` section to `examples/README.md`.

## Implementation Details
- Config loading reads a YAML `volume_eval_chunk:` block and applies `key=value` overrides, including dotted nested keys such as `slurm.mem=4G`.
- Ranges are template keys, parsed from explicit lists or `"start,stop,step"` strings.
- `factor` is a length-3 zyx key-to-voxel-start multiplier.
- `--init-only` writes `gt_vertices.h5` from `graph.get_nodes_position(None)` and `gt_graph.npz`.
- `--chunk-index`, `--chunk-range`, local multiprocessing, SLURM sbatch generation, `--wait`, `--reduce`, and `--score` are implemented as importable helpers plus CLI modes.
- `--wait` reports progress and exits nonzero via `WaitTimeoutError` on stall/timeout with missing chunk names.
- `--reduce --score` can be run together; scoring uses `score_graph_with_lut`.

## Files Changed
| File | Purpose |
|---|---|
| `examples/volume_eval_chunk.py` | Generic chunked-volume LUT orchestrator and CLI |
| `examples/volume_eval_chunk.yaml` | j0126 tile-layout example config |
| `tests/test_volume_eval_chunk.py` | Offline tests for LUT parity, chunk indexing, wait policy, reduce, and sbatch script output |
| `examples/README.md` | Added documentation for the chunked-volume workflow |

## Git Baseline
run_start_ref: 3549cad07165e9bcd3949501eb611228c66b628b
current_head: 3549cad07165e9bcd3949501eb611228c66b628b

## Verification
```text
$ source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
$ python -m pytest tests/test_volume_eval_chunk.py -q
============================= test session starts ==============================
platform linux -- Python 3.11.14, pytest-9.0.1, pluggy-1.6.0
rootdir: /projects/weilab/weidf/lib/pytorch_connectomics
configfile: pyproject.toml
plugins: cov-7.0.0, anyio-4.12.1
collected 7 items

tests/test_volume_eval_chunk.py .......                                  [100%]

=============================== warnings summary ===============================
../../../miniconda3/envs/pytc/lib/python3.11/site-packages/_pytest/cacheprovider.py:475
  PytestCacheWarning: cache could not write path /projects/weilab/weidf/lib/pytorch_connectomics/.pytest_cache/v/cache/nodeids: [Errno 30] Read-only file system

========================= 7 passed, 1 warning in 3.38s =========================
```

```text
$ python -m pytest tests/ -q
============================= test session starts ==============================
platform linux -- Python 3.11.14, pytest-9.0.1, pluggy-1.6.0
rootdir: /projects/weilab/weidf/lib/pytorch_connectomics
configfile: pyproject.toml
plugins: cov-7.0.0, anyio-4.12.1
collected 55 items

tests/test_banis_compat.py .                                             [  1%]
tests/test_j0126_workflow.py .......                                     [ 14%]
tests/test_unit.py ........................................              [ 87%]
tests/test_volume_eval_chunk.py .......                                  [100%]

=============================== warnings summary ===============================
../../../miniconda3/envs/pytc/lib/python3.11/site-packages/_pytest/cacheprovider.py:475
  PytestCacheWarning: cache could not write path /projects/weilab/weidf/lib/pytorch_connectomics/.pytest_cache/v/cache/nodeids: [Errno 30] Read-only file system

======================== 55 passed, 1 warning in 3.71s =========================
```

```text
$ python examples/volume_eval_chunk.py -h
usage: volume_eval_chunk.py [-h] --config CONFIG [--init-only]
                            [--chunk-index CHUNK_INDEX]
                            [--chunk-range CHUNK_RANGE] [--parallel PARALLEL]
                            [--local] [--sbatch] [--wait]
                            [--wait-timeout WAIT_TIMEOUT]
                            [--stall-timeout STALL_TIMEOUT]
                            [--poll-interval POLL_INTERVAL] [--reduce]
                            [--score]
                            [overrides ...]

Map, reduce, and score ERL node LUTs from chunked segmentation files.
```

## Review Focus
- Coordinate semantics: ranges as template keys, `factor` as key-to-voxel offset.
- Node-order alignment through shared `gt_vertices.h5`.
- CLI mode interactions, especially local mapping vs SLURM submission and `--reduce --score`.
- Wait timeout behavior and missing chunk diagnostics.

## Risks and Unknowns
- SLURM submission was not run by design; only the generated sbatch script string is tested.
- Pytest passed with a cache warning because the parent `.pytest_cache` path is read-only.
- The worktree still contains pre-existing approved reorg changes outside this implementation; I did not revert or commit them.

## Changes Since Previous Code Version
Initial implementation.