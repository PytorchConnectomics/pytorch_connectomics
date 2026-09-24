# Plan v2 Review

## Summary

Reviewer: codex, read-only, exit 0, repository unmutated. Raw transcript at
`state/plan_v2_review.review.raw.md`.

**2 major, 2 minor, `READY: no`.** `p2` is exhausted, so this blocks for a human
decision.

The reviewer was asked to label each finding blocking or not, and did: **both
minors are explicitly non-blocking** ("an implementer can reasonably settle
that"), and both majors are named as blocking with a stated reason. That
discrimination is itself evidence the review is converging on real gaps rather
than generating objections — contrast the canceled run, where every round
returned fresh architectural majors.

Both majors are accepted and both are small. Neither requires a design decision;
each requires naming an artifact that `plan_v2` referred to without defining.

Trajectory across this run: **7 major → 2 major → 2 major**, with the remaining
two roughly three lines of specification each.

## Findings

### Major — blocking

1. **The S1 integrity check has no comparison artifact.** It requires comparing a
   whole-volume `max` decode of `aff_prob` against "the compressed-artifact
   result", but no command or output path produces that compressed `max` decode.
   The published `max` layer cannot serve: it is marked provenance-only, and it
   was decoded at `mt 0.550` from the percentile rule, not at the `0.47` this
   check specifies. So the check names a comparison that does not exist.
2. **The HDF5 twin's consumption schema is underspecified.** Path, dtype and value
   transform are given, but not the dataset name, axis order, or the exact layout
   `run_abiss_volume.py` expects. "Shape, dtype and checksum" does not establish
   that both decoders consume the same tensor *semantics*.

### Minor — explicitly non-blocking

3. `resolve` is declared pure, yet asserts `BBOX` equals the affinity's own
   dimensions, which requires reading an artifact. Moving that assertion into
   `preflight` settles it.
4. The acceptance section says "interior planes only" while the set definitions
   are given over label bounding boxes; the plan should state precisely how the
   sets become masked volumes, but the stated definitions are sufficient to
   implement from.

## Questions

Both blocking findings have obvious closures that need no new decisions, and are
recorded here so a `plan_v3` is mechanical rather than exploratory:

- **Finding 1:** add a third invocation of the same `DECODE=whole` path — on the
  **compressed** affinity, `--ws-merge-function max --ws-merge-thresholds 0.47`,
  output `${RUN_PREFIX}/ref_max_compressed/seg.h5`. The integrity check then
  compares two artifacts that both exist and are both produced by this run. This
  also removes the last dependence on the published layer.
- **Finding 2:** state the twin as HDF5 dataset `main`, shape `(3, Z, Y, X)`,
  float32, which is what `run_abiss_volume.py --input-dataset main` reads and what
  the existing `raw_x1_ch0-1-2.h5` already uses; the `channels 2,1,0` reversal
  stays a decoder flag, not a layout change, so both decoders see identical
  semantics.

The decision for the human is only whether to spend one more plan round on those
two, or to accept them as implementer detail. Given the reviewer labelled them
blocking and they are cheap, one more round is the cheaper error.

## Verdict

VERDICT: NEEDS_CHANGES
