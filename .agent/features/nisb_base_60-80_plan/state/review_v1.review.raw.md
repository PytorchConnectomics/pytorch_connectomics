# review_v1 — coordinator (claude/planner) in-session review notes

- Mutation guard: HEAD == e8844b3d; tracked diff 22104 B (== baseline); staged 0; no tracked changes. OK.
- Fix verified in source: g1_banis_feasibility.py:421 `handle_v = -chord * tangent_v`; `handle_u = chord * tangent_u` unchanged (line 420). Exactly the review_v0 fix.
- Smoke re-run (origin [475,658,31]): curvature gate 7/9 pass (was 0/9) — confirms the sign was the cause. Raw/capped 7/7, matched 2, covered events 0/2. The 0/2 covered is a 256^3 artifact (only 2 oracle events in the tiny crop; the 7 accepted candidates didn't happen to bridge those 2 specific fragment pairs) — NOT a defect; full-chunk recall is the real test.
- Scope: only g1 changed; g0/common/DESIGN untouched; no commit/stage. Contract sections + Git Baseline + "Changes Since" all present.

Verdict: APPROVE. The [major] curvature bug is fixed and verified; sanities (from code_v0) remain exact; G0 correct. Ready for the coordinator to run the full center-chunk gates.
