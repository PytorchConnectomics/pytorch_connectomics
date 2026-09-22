# Plan v3 Review

## Summary

Reviewer: codex (`codex exec --sandbox read-only`), exit 0. Tracked and staged
diffs were identical before and after the call. The raw transcript is at
`state/plan_v3_review.review.raw.md` and the prompt at
`state/plan_v3_review.prompt.txt` (35,431 bytes). The prompt contained the task,
`plan_v2_review.md`, `plan_v3.md`, and excerpts from `run_abiss_volume.py`,
`abiss_chunk.py` and ABISS `cut_chunk_common.py`, so the reviewer could check
plan_v3's claim that the two decoders apply different channel and edge transforms.

**0 major, 2 minor, `READY: yes`.** The reviewer explicitly marked both minors
as non-blocking and wrote "No remaining major findings. Code v0 can start."

Trajectory across this run: 7 major, 2 major, 2 major, then 0 major.

## Findings

1. [minor, non-blocking] The cross-chunk-size comparison
   `VOI_total(C_A, C_B) ≤ 0.01` says "same masking" without saying which mask:
   `W ≠ 0`, `M_B`, `M_I`, or several. Reviewer's suggested default is the common
   `W ≠ 0` mask. **code_v0 should use `W ≠ 0`.**
2. [minor, non-blocking] Non-degeneracy requires `|B| > 0` but not `|I| > 0`. If
   `I` is empty, `VOI_total(B) − VOI_total(I)` is undefined or vacuous. **code_v0
   should also assert `|I| > 0`, and fail rather than pass on an empty mask.**

## Questions

None. Both minors have stated defaults, and code_v0 carries them.

## Verdict

VERDICT: APPROVE_WITH_MINOR_COMMENTS
