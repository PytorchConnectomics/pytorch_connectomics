## Findings

- [major] **`mask_background=False` is incorrect with `recompute_ext=True`.** CC construction always treats zero as background and collapses negative ids to zero. Subsequently admitting `ids >= 0` makes all background voxels one object labeled 0, regardless of connectivity. Define the intended background-component behavior and implement it consistently, or explicitly reject this combination. Add tests covering background, negative ids, and both recompute settings.

- [major] **The un-eroded `gt_seg` verification does not establish the required contract.** An already-eroded segmentation can still lose foreground under another erosion. Test the complete label-transform pipeline against a known post-augmentation label: emitted `gt_seg` must equal that label exactly, while affinity targets and masks must match the original global-erosion pipeline. Include thin processes and touching instances.

- [major] **Affinity-mask routing may remove the thin-process supervision this change needs.** Taking the intersection of three affinity-validity channels defines embedding supervision through an eroded affinity target. The plan does not establish that these channels remain valid on foreground erased by erosion. An aggregate “all-true fraction” cannot establish this. Add an integration assertion that thin-process voxels remain eligible for embedding loss and receive gradients; if they do not, provide an appropriate segmentation-validity mask.

- [major] **The mixed-precision test checks finite loss, but not finite gradients.** Casting loss calculations to fp32 prevents forward overflow but does not guarantee finite gradients through fp16 predictions and Lightning gradient scaling. Check backward gradients for predictions and model parameters, including large inputs, small objects, and the actual mixed-precision training path. The GPU smoke should detect skipped optimizer updates and nonfinite gradients.

- [major] **The equal-weight launch lacks an actionable interference gate.** Scalar loss magnitudes do not establish relative gradient strength, and decreasing embedding loss does not establish useful identity learning. Compare affinity learning against the matched baseline during the smoke and measure each term’s gradient contribution to the shared trunk. Specify a stop/review action for the 5k/10k threshold; merely flagging it permits the expensive run to continue. Compare affinity-only validation loss, since the new total validation loss is not comparable to the baseline.

- [major] **The quadratic push calculation has no memory qualification.** Materializing `[N,N,12]` distances can become expensive when connected-component recomputation produces many objects. Two sampled batches and a short smoke may miss fragmented crops. Add a deliberately fragmented stress case and report peak GPU memory with forward/backward at realistic upper component counts. Use chunked exact pair accumulation if necessary, preserving the stated denominator and gradients.

- [minor] **The recompute justification overstates what is preserved.** Same-parent disconnected components are neither pulled together nor pushed apart. Consequently, diagonal-only GT connections split by 6-connectivity can lose identity supervision; calling this “harmless” is unsupported. Record this limitation and include a deterministic connectivity fixture.

- [minor] **Batch reduction needs a mixed-empty test.** Averaging only nonempty samples means `[nonempty, empty]` equals the nonempty sample’s loss. Explicitly test that policy, plus an entirely empty batch. Add a hand-calculated three-component case with two shared parents to independently verify the excluded-pair denominator.

## Questions

- Is the three-channel affinity mask guaranteed to preserve supervision on un-eroded thin foreground, including crop boundaries and ignored regions?
- What concrete result stops the run when affinity learning degrades, and who monitors that gate?
- Will background inclusion be supported with recomputation, or rejected explicitly?

READY: no