# Plan v0 Review

## Summary

The plan is clear, scoped to the NISB-base benchmark, and grounded in the
existing tutorial/config surface. It correctly identifies that the nearest
high-ROI work is mostly config-only: v1/v2/erosion NISB variants, waterz/abiss
decode tuning, TTA, MedNeXt size/kernel overlays, and multi-head/SDT patterns
already present in nearby tutorials.

I approve the plan for implementation with minor comments. The code stage should
turn this into a concrete ranked experiment artifact rather than immediately
over-indexing on infrastructure.

## Findings

- Minor: The final ranking puts the train-time NERL callback ahead of actual
  model/decoder interventions. That callback is useful measurement
  infrastructure, but the user asked for ideas to improve the model and a ranked
  plan of experiments. In `code_v0`, split the deliverable into two lists:
  primary NERL-improvement experiments and enabling infrastructure, or make
  clear that the callback is not expected to improve NERL directly except by
  checkpoint selection.

- Minor: The SDT auxiliary-head idea may be cheaper than the plan suggests.
  NISB and nearby tutorials already show `skeleton_aware_edt` / `label_aux_type:
  sdt` patterns. Before classifying SDT as a Tier-3 code change, `code_v0`
  should verify whether this is config-only plus data availability, or whether a
  real code path is missing for the base BANIS layout.

- Minor: The Dice+BCE proposal needs implementation-specific guardrails before
  launch. MONAI `DiceLoss` should be configured deliberately for affinity logits
  or probabilities, and the experiment note should state the exact kwargs
  instead of just naming the loss combination.

- Minor: TTA and decoder tuning are valid decode/inference-only ideas, but
  `code_v0` should pin concrete kwargs for reproducibility: TTA axes and ensemble
  mode, waterz `channel_order`, waterz threshold ranges, and whether decode-only
  trials reuse the exact same saved prediction artifact.

## Questions

- What GPU budget is available for 200k-step training runs? If the budget is
  small, the plan should prioritize decode-only experiments and a minimal
  v1/v1_erosion2 training pair before larger MedNeXt runs.

- Were the reported 24%@50k and 32%@200k numbers measured with the untuned
  `decode_affinity_cc` threshold of 0.75 or with the existing tune profile? The
  baseline comparison table should record this explicitly.

## Verdict

VERDICT: APPROVE_WITH_MINOR_COMMENTS
