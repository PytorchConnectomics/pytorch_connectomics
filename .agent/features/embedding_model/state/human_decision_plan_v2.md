# Human decision: plan_v2 amendment (binding for code_v0)

**Date:** 2026-09-15. **Decided by:** the user, via an AskUserQuestion answer: "Cancel + report (Recommended)".

**Resolves:** the plan_v2_review [major] finding on the watchdog's non-finite gate handling.

## Amendment to plan_v2 §5 (`dev/nisb/scripts/train_gate_watch.py`)

1. **Non-finite value is a hard fail.** A non-finite run value (NaN or ±inf) for RUN_TAG at *any* gate step is a distinct hard failure, `GATE NONFINITE`. It is not converted to ratio=inf and can never become `GATE PASS`.
2. **Evaluated immediately.** As soon as a non-finite value is observed at any gate step, even if other gate steps are not yet evaluated or would pass:
   - **Without `--dry-run`:** call `cancel(job_id)`.
     - If `cancel` returns True (scancel rc 0 and the job left RUNNING/PENDING within 120 s): print `GATE NONFINITE at step s — CANCELLED <id>` to stdout and stderr, exit 8.
     - If `cancel` returns False: print `GATE NONFINITE at step s — CANCEL FAILED <id>`, exit 7.
   - **With `--dry-run`:** print `GATE NONFINITE at step s — WOULD CANCEL <id>`, exit 8, and do not call cancel.
3. **Everything else unchanged.** The finite-ratio rule stays: fail only if every gate ratio > max_ratio, which cancels and exits 3. The missing / stale / early-end / bad-baseline behaviors are also unchanged.
4. **Tests to add.** In `dev/nisb/scripts/test_train_gate_watch.py`:
   - (a) NaN at 4999 and ratio 1.0 at 9999 → NONFINITE, cancel called once, exit 8, never PASS.
   - (b) Ratio 1.0 at 4999 and inf at 9999 → NONFINITE, exit 8.
   - (c) NaN with cancel failing → exit 7.
   - (d) NaN with dry-run → WOULD CANCEL, exit 8, cancel not called.
   - This replaces the plan_v2 test "non-finite run value at both gates → treated as fail".
5. **Coordinator reporting.** In plan_v2 Verification step 11, exit 8 is reported loudly as a diverged-and-cancelled run.
