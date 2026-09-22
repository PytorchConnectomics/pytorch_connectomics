# Task

The `em_erl` repo has a main package `em_erl/` and a `scripts/` directory. Move the
reusable auxiliary functions out of `scripts/` into the `em_erl` package, so that
`scripts/` becomes `examples/` — thin, illustrative programs showing how to use the
library. Improve the repo design and implement it.

Repo: `lib/em_erl` (its own nested git repo, gitignored from the outer pytorch_connectomics
repo; it is the review surface). Baseline HEAD `3549cad`.

## Current layout

Package `em_erl/`: `erl.py` (ERLGraph, skel_to_erlgraph, scores), `eval.py`
(compute_segment_lut, compute_erl_score, LUT tile helpers), `sampling.py` (VolumeSource,
sample_segment_lut), `io.py` (read_vol/write_h5, cloudvolume read, mkdir), `skel.py`
(vol_to_skel, cable_length), `__init__.py` (explicit `__all__`).

`scripts/`:
- `j0126_workflow.py` (396 lines) — holds genuinely reusable primitives currently trapped
  in a script: `normalize_seg_url`, `open_seg_cloudvolume`, `sample_cloudvolume_lut`
  (+ `_squeeze_cloudvolume_block`, `_xyz_array`), `load_skeletons`,
  `save/load/validate_node_segment_lut`, `score_graph_with_lut`, `score_skeletons_with_lut`,
  and `run_j0126_eval` (a CloudVolume+LUT eval recipe with a j0126 default URL) + CLI.
- `volume_eval.py`, `seg_to_graph.py`, `skel_to_graph.py` — already thin compositions of
  existing library functions (`compute_segment_lut`/`compute_erl_score`,
  `vol_to_skel`/`skel_to_erlgraph`).
- `README.md` — usage docs.

## Coupling facts

- `tests/test_j0126_workflow.py` imports the script by importlib path (6 references);
  `tests/test_banis_compat.py`'s importlib usage targets funlib/banis, not our scripts.
- No public-API snapshot test in em_erl (changing `__init__.__all__` is safe).
- `setup.py` uses `find_packages()`, no console_scripts, no `scripts=`; renaming
  `scripts/` → `examples/` needs no setup change.
- Top-level `README.md` references `scripts/volume_eval.py` and `scripts/README.md`.
- `eval.py` already imports from `io` and `sampling` (dependency direction supports moving
  cloudvolume/LUT/scoring helpers into these modules without cycles).

## Constraints

- Behavior-preserving move: relocate functions verbatim; do not change their logic or
  rename existing package functions. Keep the change surgical (no unrelated redesign).
- General, reusable functions go into the package; dataset-specific bits (the j0126 default
  URL) stay in the example.
- All existing tests must still pass (adapted imports); example CLIs must still run.
