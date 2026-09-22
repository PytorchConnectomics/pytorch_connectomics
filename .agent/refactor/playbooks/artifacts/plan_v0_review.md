# Plan v0 Review

## Summary

Codex reviewed `plan_v0.md` for executability as the agent that will implement it, under
`codex exec --sandbox read-only`. It returned `READY: no` with five findings, all tagged
major, and answered the plan's open question. The raw transcript is preserved verbatim at
`state/plan_v0_review.review.raw.md`.

The coordinator's assessment: all five are material and none are softened here. Findings 3
and 4 are the strongest -- they attack the two places where this refactor could regress
silently (a fitted threshold reverting to a shared default, and a verification gate that
cannot see the j0126 features it claims to protect).

## Findings

1. **[major] The plan does not establish one shared cube playbook.** It lifts
   `j0126_full` and `cube_decode` independently, with no composition or selection
   mechanism that makes both datasets run the canonical cube playbook. Adding
   smoke/guard/score to j0126's legacy invocation would also contradict the
   command-equivalence gate.

2. **[major] Excluding mask extraction leaves "dataset #3 needs config only"
   unresolved.** Separate implementations are justified by storage format and sharding,
   but the plan supplies no shared interface consuming mask paths, polarities, ratios,
   frame and border settings. The task excludes numerical changes, not behaviour-preserving
   extraction.

3. **[major] Fitted-value enforcement has gaps.** Validation is specified only for
   `cube_decode`, leaving `j0126_full` unprotected, and the plan checks null values without
   requiring ABSENT keys to fail before `prepare_config` supplies its defaults. Absent and
   null must be tested separately, on both execution routes.

4. **[major] The verification gate does not cover the main execution risks.** Filtering
   live `--dry-run` output does not establish that every step emits a command -- completed
   steps emit none and go untested -- and command strings cannot demonstrate preservation
   of `prepare`, `prepare_fn`, input gating or completion checks. Deterministic fixtures
   covering pending/completed/skipped steps are required, comparing resolved submission
   options as well as command text, across both resource schemas.

5. **[major] Config-only onboarding has no integration gate and an unresolved
   registration question.** No test builds dataset #3 from `params.yaml` plus
   `2_abiss.yaml` alone and resolves every step. Separately,
   `scripts/validate_tutorial_configs.py` requires custom workflow roots to be declared,
   which the plan does not explain how a new tutorial satisfies without Python changes.

## Questions

The plan's open question (root `playbooks/` versus `connectomics/playbooks/`) was
answered: **choose `connectomics/playbooks/`**, because the modules contain executable
logic and therefore benefit from normal package imports and packaging support, with
dependencies kept explicit (playbooks assemble runtime primitives; the engine receives the
selected playbook). The reviewer adds that packaging the Python modules does not package
`tutorials/` or `scripts/` assets, so path resolution for those must be defined.

Coordinator note for plan_v1: this reverses the plan's assumption and also removes the
naming objection that motivated a top-level directory, since `connectomics/playbooks/`
does not collide with the repository's two existing meanings of "workflow".

## Verdict

VERDICT: NEEDS_CHANGES
