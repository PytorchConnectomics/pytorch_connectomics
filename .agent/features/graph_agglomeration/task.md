# Task

Read `dev/zebrafinch/agglomeration_design_fable5.md` and
`dev/zebrafinch/agglomeration_design_gpt5-5.md` and implement the global
neuron-graph agglomeration design **one step at a time**, where **each step has a
concrete way to evaluate its quality** (e.g. NERL against `test_50_skeletons` /
per-chunk `chunk_gt_skel`).

## Intent

- The two design docs describe the same stage (assembling the per-chunk
  `decode_v1` over-segmentation into whole neurons across the 600-box zebrafinch
  volume, under the one-soma-per-neuron firewall, optimizing length²-weighted
  NERL). Reconcile them into a single ordered sequence of implementable,
  independently-evaluable steps.
- Implement incrementally. Every step must (a) produce a segmentation or a
  measurable intermediate, and (b) report a quality metric (NERL base + oracle,
  or recall_thick/precision vs FFN, or a per-cue precision) so we know whether it
  helped before moving on.
- Reuse the existing `dev/zebrafinch/` tooling (186 scripts, incl.
  `decode_v1_chunk.py`, `oracle_stitch_decode_v1.py`, `big_branch_extract.py`,
  `global_link.py`, `branch_complete.py`, `soma_recon_wholevol.py`,
  `local_nerl_all_chunks.py`, `thick_gt_nerl.py`, `chunk_gt_skel/`) rather than
  reinventing; the base segmentation (`decode_v1`) is already produced and is not
  in scope to change.

## Evaluation substrate (authoritative)

- Whole-volume `test_50_skeletons` NERL (dedup GT, `--break-threshold-nm 1000
  --length-threshold-nm 1000`) is the headline; per-chunk local oracle is
  triage-only and only vs `chunk_gt_skel`.
- A join is merge-safe iff the oracle does not drop; re-score the oracle after
  every edge set.

## Constraints

- CCC run rooted at the main repo `/projects/weilab/weidf/lib/pytorch_connectomics`.
- The repo has pre-existing unrelated dirty files (24 entries at run start,
  captured in `state/run_start.*`); this run must only add/modify files for the
  agglomeration step and must not touch those.
- No git commits during the run.
