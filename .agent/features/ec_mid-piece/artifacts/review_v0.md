# Review v0

## Summary

Reviewer: claude (planner role), performed in-session. Raw verification transcript:
`state/review_v0.review.raw.md`.

The implementation is faithful to plan v2 in the places that matter most for correctness. I
independently re-ran the suite (`16 passed in 17.82s`), confirmed `HEAD` and both tracked diffs are
byte-identical to the CCC baseline, and confirmed the two things most likely to be silently wrong
are right: **Stage 1 reads affinity with Form 1** (`affinity[(axis, *source_slice)]`, channel =
array axis, sampled at the low-side voxel — the legacy `2 - ax` bug is absent), and **Stage 2
re-derives band membership per anchor-floor variant** with explicit assertions, so the
extract-once/resolve-twice union in Stage 1 is safe.

One major finding blocks approval: **the test that plan v2 designated as the decisive affinity check
is tautological.** It compares the implementation against a local reimplementation plus an inline
`expected` array that encode the same belief, instead of against
`lib/abiss/scripts/volume_backends.py` — the code that actually produced arm0_96. The reason
`code_v0.md` gives for the substitution is factually wrong: that module imports cleanly in `pytc`,
which I verified.

This is a test-validity defect, not a live correctness bug — I independently verified the encoded
convention against the real producer with `maxdiff = 0.0` and all four negative controls failing.
But the whole point of that test is to protect the whole-volume run from a convention error, and as
written it cannot.

`code_v0.md` is otherwise unusually honest: it reports the interrupted eight-file seam probe, the
abandoned formatting check, the unmeasured scientific quantities, and the input drift, rather than
papering over them.

## Diff Baseline

run_start_ref: 6ad67866c3ca3e1cf1d52437a0cb300b0340c1bd

All deliverables are untracked files under the gitignored `dev/zebrafinch/ec_mid_piece/`, so
`git diff <run_start_ref>...HEAD` is empty by construction; the review surface is the new files
themselves. `git rev-parse HEAD` equals `run_start_ref`, and `git diff` / `git diff --cached` are
byte-identical to `state/run_start.diff` / `state/run_start_cached.diff`.

## Findings

1. `[major]` **The decisive affinity truth-table test is tautological.**
   `tests/test_affinity_io.py:5` imports `_ArrayVolume` from `affinity_io`, whose `abiss()` is just
   `banis_to_abiss(self.restored)` (`affinity_io.py:86`). `test_affinity_io.py:30-40` then rebuilds
   `expected` using the same channel-reversal and one-voxel shift and asserts
   `max|observed - expected| == 0`. Both sides come from the same author's same belief, so the test
   passes whether or not the belief is true. Plan v2's Verification item 1 required comparing
   against `lib/abiss/scripts/volume_backends.py::_ArrayVolume`.
   *Failure scenario:* had the BANIS channel order actually been `[x,y,z]` rather than `[z,y,x]`,
   `banis_to_abiss` and `expected` would reverse identically, all 16 tests would still pass, and
   Stage 1 would read the Z channel for X edges across the whole volume — producing a plausible
   affinity table that is wrong on the anisotropic axis.
   *Aggravating:* `code_v0.md:260` states the producer "is not importable in this checkout". It is —
   `sys.path.insert(0,'lib/abiss/scripts'); import volume_backends` succeeds in `pytc` and exposes
   both `_ArrayVolume` and `_restore_sigmoid`.
   *Mitigating:* the convention as implemented is correct; I verified it independently against the
   real producer (transcript §4).

2. `[minor]` **The freeze attestation is never tested against a tampered payload.**
   `tests/test_firewall.py:36-56` writes a fixture, records `sha256_file(fixture)`, then asserts the
   recorded hash equals `sha256_file(fixture)` — self-referential. Stage 3's verification path,
   which is the actual firewall, is not exercised at all.
   *Failure scenario:* a bug that made Stage 3 skip or short-circuit its SHA check (for example
   treating a missing `frozen.json` as a pass) would let post-hoc-edited assignments be scored as
   frozen, and no test would notice. `test_resolver.py:68-72` covers write-once immutability at
   freeze time but not the read-side check.

3. `[minor]` **The GT firewall audit is a literal-substring scan.**
   `tests/test_firewall.py:18-33` greps the GT-free sources for tokens such as
   `test_50_skeletons.h5`, `reports/arm096_lut`, `evaluation_gt/`. It is adequate for the code as
   written, but a path assembled by concatenation, read from an environment variable, or reached via
   an indirect import would pass the audit.
   *Failure scenario:* a later edit that does `EVAL = ROOT / "evaluation" + "_gt"` or accepts a
   `--input` path pointing into `evaluation_gt/` would leak GT into the selector while the audit
   stays green.

4. `[minor]` **Input drift makes the plan's baseline premise stale.**
   `arm096_error_correction/decoder_gtfree/frozen_endpoint_merges.{npz,json}` were created at 12:47
   by a concurrent session, *during* this CCC run; they did not exist when plan v2 fixed the honest
   baseline to raw arm0_96 on the grounds that no frozen substantial-linker artifact existed.
   The implementation's behaviour is correct — it records the drift and does not consume the
   artifact, which is the right call under the approved plan — but the interpretation of the eventual
   result depends on this.
   *Failure scenario:* the whole-volume run is launched, and the honest row is reported as the
   pipeline's capability without noting that a substantial-linker baseline was available and unused,
   understating deployable value exactly as plan v2's risk 4 predicted.

## Tests to Add

1. Replace the reference in `tests/test_affinity_io.py` with the real producer:
   `sys.path.insert(0, REPO/'lib/abiss/scripts'); from volume_backends import _ArrayVolume`, wrap
   the real crop, read through `vol[x0:x1, y0:y1, z0:z1]` (which returns `(X,Y,Z,C)`), transpose to
   ZYX, and assert equality with `R[2-c']` shifted by `-e_{2-c'}` at `maxdiff == 0`, keeping the
   four negative controls. Also assert `restore_sigmoid` agrees with `volume_backends._restore_sigmoid`
   elementwise. If the import is ever genuinely unavailable, the test must **skip loudly**, not
   silently fall back to a local reimplementation.
2. A Stage 3 rejection test: mutate one byte of a frozen `assignments_*.npz` and assert Stage 3
   raises; separately assert it raises when `frozen.json` is missing and when either marker is
   `False`.
3. A firewall test that constructs a GT path dynamically and asserts the audit still catches it, or
   replace the substring scan with an import-graph plus opened-path check (e.g. audit `ast` for
   `open`/`np.load`/`h5py.File` call arguments) so the guarantee survives edits.

## Questions

- Finding 4 is a decision for the maintainer rather than a code change: now that
  `frozen_endpoint_merges.npz` exists, should the honest baseline stay raw arm0_96 (faithful to the
  approved plan, and what `code_v0.md` implemented), or should the whole-volume run also report a
  second honest row on top of the frozen linker? My recommendation is to keep raw arm0_96 as the
  approved primary and add the linker row as a clearly labelled secondary once its own manifest and
  metric checks pass — but this is outside what `code_v1` should change without direction.
- The whole-volume Stage 0/1 passes were deliberately not run, so no scientific claim exists yet.
  Fixing finding 1 before launching those ~6 h of compute is the point of blocking here.

## Verdict

VERDICT: NEEDS_CHANGES
