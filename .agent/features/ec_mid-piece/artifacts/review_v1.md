# Review v1

## Summary

Reviewer: claude (planner role), performed in-session. Raw verification transcript:
`state/review_v1.review.raw.md`.

All three `review_v0` findings are fixed, and fixed substantively rather than cosmetically.

The major finding is closed properly. `tests/test_affinity_io.py` now imports the real
`lib/abiss/scripts/volume_backends.py`, drives `producer._ArrayVolume` over a real crop, and
compares its output against the declared truth table computed independently by slicing — so the two
sides of the assertion no longer share an author's belief. It also asserts the local
`restore_sigmoid` matches `producer._restore_sigmoid` elementwise, queries the interior so the
one-voxel shift is a real read rather than the zero-padded low face, and guards against a
same-named module shadowing the producer on `sys.path`.

The one way this fix could have been hollow was the `pytest.skip` fallback silently swallowing the
test. I re-ran the suite with `-rs`, which forces skips to be reported: **23 passed, zero skips**.
The test genuinely executes against the real producer.

Finding 2 is closed by three real rejection tests that call `verify_frozen` (tampered bytes, missing
attestation, false marker parametrized over both), replacing the self-referential fixture hash.
Finding 3 is closed by a genuine AST audit over the arguments of `open` / `np.load` / `h5py.File` /
`zarr.open*`, resolving name bindings, `+` and `/` concatenation and f-strings, with a test proving
it catches a dynamically constructed GT path.

`HEAD` still equals `run_start_ref` and both tracked diffs are byte-identical to the run-start
captures. No tracked file, canonical input, commit, or staged change was produced.

Finding 4 from `review_v0` (the `frozen_endpoint_merges.npz` input drift) was explicitly excluded
from this revision's scope as a maintainer decision, and remains open as such — it is not a code
defect and does not block approval.

No new defects were introduced. Approving.

## Diff Baseline

run_start_ref: 6ad67866c3ca3e1cf1d52437a0cb300b0340c1bd

All deliverables remain untracked files under the gitignored `dev/zebrafinch/ec_mid_piece/`, so
`git diff <run_start_ref>...HEAD` is empty by construction and the review surface is the files
themselves. `git rev-parse HEAD` equals `run_start_ref`; `git diff` and `git diff --cached` hash
identically to `state/run_start.diff` and `state/run_start_cached.diff`.

## Findings

None blocking. Two observations carried forward, neither a defect in `code_v1`:

1. `[minor]` **Input drift remains open (carried from `review_v0` finding 4).**
   `arm096_error_correction/decoder_gtfree/frozen_endpoint_merges.{npz,json}` were created at 12:47
   by a concurrent session during this run, invalidating plan v2's premise that no frozen
   substantial-linker artifact existed. `code_v1` correctly continues to record it as unselected
   input drift and does not consume it.
   *Consequence, not failure:* the eventual honest row will measure absorption on an unlinked
   backbone and understate deployable value, exactly as plan v2 risk 4 predicted. This needs a
   maintainer decision before the whole-volume run, not a code change.

2. `[minor]` **No scientific result exists yet, by instruction.** The ~2.4 h Stage 0 and ~3.5 h
   Stage 1 whole-volume passes were deliberately not run, so `candidate_edges.npz`, the assignment
   manifests, `evaluation_gt/results.json` and `results.md` do not exist. `code_v1.md` states this
   plainly and does not present their absence as success. Every conclusion the task asks for
   remains unmeasured until those passes run.

## Tests to Add

None required for approval. Optional hardening for whoever runs the whole-volume pass:

1. A CI-style guard that fails if `test_form2_real_crop_and_negative_controls` skips, so the
   producer comparison cannot silently degrade on a machine where `lib/abiss` is absent. The skip is
   loud in `-rs` output but invisible in a plain `pytest -q` summary line.
2. An end-to-end tiny-volume fixture that runs Stage 0 → Stage 1 → Stage 2 → Stage 3 on a synthetic
   few-hundred-voxel volume with a planted fragment and two anchors, asserting the resulting NERL
   delta has the hand-computed sign. The current suite validates each stage's contracts in
   isolation; nothing yet exercises the full chain.
3. A memory-ceiling regression on the dictionary-backed global candidate merge, which `code_v0.md`
   itself flagged as the scaling risk near the declared 200-million-row stop.

## Questions

- The maintainer decision on observation 1: keep raw arm0_96 as the approved honest primary
  (recommended, and what is implemented), and optionally add a clearly labelled secondary row on the
  frozen linker once its own manifest and metric checks pass.
- Launching Stage 0 and Stage 1 is the natural next step and is outside this CCC run's scope. The
  README carries the resumable non-interactive commands; concurrency should stay capped at 8
  readers and should not be co-scheduled with other affinity streaming jobs.

## Verdict

VERDICT: APPROVE
