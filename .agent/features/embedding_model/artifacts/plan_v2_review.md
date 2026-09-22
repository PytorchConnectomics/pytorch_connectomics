# Plan v2 Review

## Summary

**Reviewer:** `codex exec --sandbox read-only` (coder role).
- Raw transcript: `state/plan_v2_review.review.raw.md`.
- Prompt: `state/plan_v2_review.prompt.md` (68,463 bytes).
- No repository mutation: `git diff` identical before and after; `git status` empty.

**Result:** `READY: no`.
- Resolved: 9 of the 10 plan_v1_review findings (background, label pipeline, AMP, weight rule, chunking/memory, denominator fixture, connectivity wording, shape normalization, parity precision). The reviewer confirms the rest of plan_v2 is implementable without further design decisions.
- One major finding remains in the watchdog's non-finite handling.

plan_v2 was the last allowed plan version (p2), so in normal mode the run is blocked for a human decision.

## Findings

- [minor] **Background:** resolved.
- [minor] **Label-pipeline findings 2 and 3:** resolved.
- [minor] **AMP (finding 4):** resolved at plan level.
- [minor] **Weight rule (finding 5):** resolved.
- [minor] **Memory/chunking (finding 6):** resolved.
- [minor] **Denominator fixture:** resolved.
- [minor] **Connectivity wording:** resolved.
- [minor] **Shape normalization:** resolved.
- [minor] **Precision parity:** resolved.
- [major] **Watchdog non-finite handling (partially resolved).**
  - Problem: §5 maps a non-finite gate value to ratio=inf, and `GATE PASS` is allowed when any gate passes. So NaN at step 4999 plus ratio 1.0 at 9999 gives PASS, which certifies invalid training.
  - Required: a non-finite gate value must produce a distinct failure outcome that can never become PASS, plus a mixed finite/non-finite test.

## Questions

- Should a non-finite validation gate value trigger an immediate cancel, or an immediate failure report without cancelling? The reviewer defers this policy choice to a human.

## Verdict

VERDICT: NEEDS_CHANGES
