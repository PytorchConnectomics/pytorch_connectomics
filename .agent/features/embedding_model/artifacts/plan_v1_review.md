# Plan v1 Review

## Summary

**Reviewer:** `codex exec --sandbox read-only` (coder role).
- Raw transcript: `state/plan_v1_review.review.raw.md`.
- Prompt: `state/plan_v1_review.prompt.md` (53,058 bytes).
- The repo was not mutated: `git diff` was identical before and after, and `git status` was empty.

**Result:** `READY: no`.
- Resolved: plan_v0 finding 1. Plan-level resolution of findings 2, 3 and 6, and of the minor connectivity finding.
- Partially resolved: finding 4 (fp16 gradients), finding 5 (gradient-ratio / weight rule), and the minor denominator fixture.
- New majors: a `squeeze` shape bug, fp32-vs-fp64 parity tolerance, and incomplete watchdog contracts.

All findings are reproduced below without softening.

## Findings

- **[minor] Finding 1 (background) resolved.** Removing `mask_background` makes background handling well defined, and computing CC on valid foreground stops masked/ignored voxels from bridging.
- **[minor] Findings 2 and 3 resolved at plan level.** Two refinements:
  - make the crop deterministic;
  - put the tested thin process inside the valid region of all three offsets, since boundary voxels cannot satisfy a universal mask assertion.
- **[major] Finding 4 only partially resolved.** Three problems:
  - Requiring no overflow on the first backward at scale 65536 with outputs near ±20 can reject normal AMP scale calibration.
  - A final checkpoint scale ≥1024 and within 4× of baseline does not prove updates were not repeatedly skipped.
  - Comparing separate fp16 and fp32 conv outputs to within 1e-4 is unreliable.
  - Needed: a bounded calibration period, then actual counts of skipped updates and non-finite gradients during the smoke; compare loss accuracy on the same quantized predictions.
- **[major] Finding 5 partially resolved.** The weight rule does not say:
  - how to aggregate over the 4 batches;
  - whether `g_emb` includes the configured weight (risk of double-counting `w`);
  - how to handle a zero or non-finite denominator.
  - It also needs to require a fresh matched smoke at the selected weight. `r` should be defined from unweighted losses.
- **[minor] Finding 6 resolved at plan level.** Add gradient parity across chunk sizes and with and without checkpointing; the stress comparison currently checks loss values only.
- **[major] Denominator fixture still broken.** Means 0, 0.5, 5.0 make every allowed push term zero, so dividing by 6 and dividing by 4 give the same result. Use e.g. 0, 0.5, 2.0 (numerator 6.5, `L_ext = 6.5/6`) and isolate the term with alpha=gamma=0.
- **[minor] Connectivity finding resolved.** Two wording fixes:
  - the fixture's "sum of two independent pulls" should be their **mean**;
  - "GT-connected thin processes stay one component" should say "connected within valid foreground".
- **[major] New shape bug in §1.** Unconditionally calling `gt_seg[b].squeeze(0)` drops Z for `[B,Z,Y,X]` input when Z=1. Normalize the channel dimension based on rank, and test singleton spatial dimensions in both layouts.
- **[major] New precision mismatch in test 6.1.** The implementation always casts to fp32 while the oracle runs in fp64 with atol 1e-6, which is not a reliable criterion. Either define absolute and relative tolerances suitable for fp32, or run both on identical fp32 inputs.
- **[major] Watchdog verification incomplete.** Gaps:
  - Training scalars can keep the watchdog fresh even if the gate validation tag or step never appears. It needs a deadline for missing gate data.
  - Reject non-finite values.
  - Pin the directory to the submitted job instead of picking the newest run.
  - Confirm `scancel` succeeded before reporting CANCELLED; dry-run must say "would cancel".
  - Add deterministic tests for cancellation, cancellation failure, missing tags, stale data, early termination, and pass/fail combinations.

## Questions

- How is the four-batch gradient ratio aggregated, and does each selected weight need a fresh smoke?
- What skipped-update threshold after calibration blocks the launch?
- What deadline and action apply when training progresses but a required validation gate never appears?

## Verdict

VERDICT: NEEDS_CHANGES
