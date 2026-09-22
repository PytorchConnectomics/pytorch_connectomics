# review_v0 raw reviewer notes (in-session planner review of reorg code_v0)

Reviewed working tree vs run_start_ref 3549cad (nested lib/em_erl repo). HEAD unchanged
(3549cad — no driver commit). diff --stat: README, __init__ (+28), eval (+129), io (+64),
sampling (+108), examples/* (git mv from scripts/*, history preserved), scripts/j0126 deleted
(-396), tests/test_j0126_workflow rewritten (+/-71).

## Independent verification (re-run, not trusting the paste)

- `python -m pytest tests/ -q` -> 48 passed.
- All four examples run uninstalled from repo root: `examples/{j0126_workflow,volume_eval,
  seg_to_graph,skel_to_graph}.py -h` all OK (no ModuleNotFoundError -> sys.path bootstrap works;
  em_erl confirmed NOT installed).
- Import surface: all 10 new names present as attributes AND in em_erl.__all__:
  normalize_seg_url, open_seg_cloudvolume, load_skeletons, sample_cloudvolume_lut,
  save_node_segment_lut, load_node_segment_lut, validate_node_segment_lut,
  score_graph_with_lut, score_skeletons_with_lut, evaluate_skeletons_cloudvolume.
- (codex-reported, consistent) `examples/volume_eval.py -p tests/data/vol_pred.h5 -g
  tests/data/gt_graph.npz -r 30,30,30` -> ERL 2952.02 / 2 skel; grep shows no leftover
  `scripts/` refs (only pre-existing banis importlib, correctly left).

## Spot-read of critical relocations

- eval.py: `from .io import (..., open_seg_cloudvolume, ...)` and `from .sampling import
  sample_cloudvolume_lut, sample_segment_lut` at module level; `evaluate_skeletons_cloudvolume`
  calls `open_seg_cloudvolume(...)` / `sample_cloudvolume_lut(...)` via the module namespace
  (lines 606-607) -> the reuse test can monkeypatch `em_erl.eval.open_seg_cloudvolume`. Confirmed
  by test_j0126_workflow passing.
- `evaluate_skeletons_cloudvolume(gt_skeleton_path, seg_url, merge_threshold=50, num_workers=16,
  output_path="", lut_path="", mip=0, cache_dir="")` — seg_url is a required positional param;
  no J0126 default URL in the package.
- LUT save/load/validate + score_graph_with_lut relocated to eval.py; sampler to sampling.py;
  opener/url/skeleton-loader to io.py (cloudvolume + h5py lazy imports preserved).
- examples/j0126_workflow.py: sys.path bootstrap; DEFAULT_SEG_URL lives here; thin `run_j0126_eval`
  wrapper over `em_erl.evaluate_skeletons_cloudvolume`; full CLI preserved; `%%` correctly escaped
  in argparse help.

## Conclusion

Behavior-preserving relocation as planned. Dependency direction respected (eval -> {io, sampling}
-> io; no cycle — import works). All plan_v1 items delivered; both prior-round findings (sys.path
bootstrap in all examples; thin run_j0126_eval wrapper) present. No new findings.

READY: yes
