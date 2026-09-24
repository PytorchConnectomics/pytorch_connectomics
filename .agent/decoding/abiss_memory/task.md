# Task: make ABISS more memory efficient

User request (verbatim): "make abiss more memory efficient. e.g. have a choice of uint32
instead of uint64 for segmentation. think of more ways"

Scope resolved by the planner: the single-invocation ABISS watershed+agglomeration binary
(`lib/abiss/src/ws/atomic_chunk.cpp`, built as `build/ws` and `build64/ws64`) and its
Python driver `scripts/run_abiss_volume.py` (whose `_run_abiss_ws` is also imported by the
`tutorials/neuron_liconn_{ist,moe}` sweep scripts). This is the path whose per-chunk RSS
decides node class (measured ~24 B/voxel for `ws`, ~58 B/voxel composite for `ws64`
multi-threshold at 2.08 Gvox; 1012 GB at 16.77 Gvox).

Note: `lib/abiss` is a separate git repository (HEAD 92abc91, branch main) that lives in
the MAIN checkout at /projects/weilab/weidf/lib/pytorch_connectomics/lib/abiss, not inside
this worktree. Its changes must be reported with `git -C <abiss> diff`.
