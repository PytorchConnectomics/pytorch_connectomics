# Plan v0 Review

## Summary

Reviewer: `codex exec --sandbox read-only` (coder role). Raw transcript: `state/plan_v0_review.review.raw.md`. Prompt: `state/plan_v0_review.prompt.md`, 22,218 bytes.

The reviewer returned `READY: no`: 6 major and 2 minor findings. All findings are reproduced below, not softened. After the review, `git status --short` in the repo was empty, so there was no mutation.

## Findings

- **[major] `mask_background=False` with `recompute_ext=True` is ill-defined.**
  - CC treats 0 as background and collapses negative ids to 0.
  - Admitting `ids >= 0` then makes all background voxels one object labelled 0.
  - Required: define the behavior or reject the combination, and test background, negative ids, and both recompute settings.
- **[major] The un-eroded `gt_seg` check is too weak.**
  - "More foreground than after another erosion" also holds for an already-eroded segmentation.
  - Required: test the full label-transform pipeline so that the emitted `gt_seg` equals the post-augmentation label exactly, while the affinity targets and masks match the global-erosion pipeline. Include thin processes and touching instances.
- **[major] Routing the affinity validity mask may remove thin-process supervision.**
  - `all()` over 3 affinity-mask channels ties embedding eligibility to the affinity target.
  - The plan does not show those channels stay valid on foreground that erosion removed.
  - Required: an integration assertion that thin-process voxels stay eligible and receive gradient, or a segmentation-based validity mask.
- **[major] The mixed-precision test covers the forward pass only.**
  - Required: finite gradients for predictions and parameters under fp16 plus gradient scaling, including large inputs and small objects.
  - The GPU smoke should detect skipped optimizer updates and non-finite gradients.
- **[major] The equal-weight launch has no actionable interference gate.**
  - Loss magnitudes say nothing about relative gradient strength.
  - Required: measure each term's gradient contribution to the shared trunk and compare affinity learning against a matched baseline during the smoke.
  - The 5k/10k check needs a concrete stop action and should compare affinity-only validation loss, because the new total is not comparable to the baseline.
- **[major] The N×N×12 push computation has no memory bound.**
  - Required: a deliberately fragmented stress case with peak GPU memory for forward and backward at realistic upper component counts.
  - Use chunked exact accumulation if needed, keeping the denominator and gradients.
- **[minor] The recompute justification overstates what is preserved.**
  - Components of the same GT id are neither pulled together nor pushed apart.
  - With 6-connectivity, GT connections that are diagonal only lose identity supervision, so "harmless" is unsupported.
  - Required: record the limitation and add a deterministic connectivity fixture.
- **[minor] Batch reduction needs more tests:** a batch with one empty and one non-empty sample, a fully empty batch, and a hand-computed case with three components where two share a parent (checks the excluded-pair denominator).

## Questions

- Is the 3-channel affinity mask guaranteed to keep supervision on un-eroded thin foreground, including crop boundaries and ignored regions?
- What concrete result stops the run if affinity learning degrades, and who monitors that gate?
- Will including background with recomputation be supported, or rejected explicitly?

## Verdict

VERDICT: NEEDS_CHANGES
