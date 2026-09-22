# Plan v1 Review

## Summary

Codex re-reviewed under `codex exec --sandbox read-only` and returned `READY: no` with
three major findings and one minor. It confirms previous finding 3 (fitted-value
enforcement) is fully addressed and that the package-location and asset-resolution
decisions answer the open question. Raw transcript: `state/plan_v1_review.review.raw.md`.

It also rejects the plan's own claim that "none are deferred", correctly: `plan_v1`
Risk 3 offers to fall back to keeping the Moritz builder as-is, which is a deferral of
finding 2. The coordinator accepts that criticism as stated and does not soften it -- the
claim was unsupported.

## Findings

1. **[major] Finding 1 only partly resolved.** The registry shares step implementations,
   but each tutorial still owns the ordered sequence via `params.pipeline.steps`, so
   dataset #3 must retype the cube sequence and nothing requires its guard or score stages.
   The canonical cube sequence should be defined once in `cube_decode.py` with an explicit
   composition mechanism that still preserves j0126's legacy commands.

2. **[major] Finding 2 partly deferred, and the fix is in the wrong place.** The
   declarative mask interface excludes j0126's strategy, and the plan narrows the
   config-only requirement without authorisation. Both strategies should be exposed through
   the one configurable interface, with reusable mask computation in an importable package
   and thin script wrappers -- adding another substantive implementation under `scripts/`
   repeats the exact ownership problem the task is about.

3. **[major] Finding 4 still incomplete.** The fixtures exercise the engine while the
   migrated j0126 step definitions are left to manual diff review, and the live comparison
   is filesystem-dependent and no longer mandatory. So nothing establishes that the
   migrated builders preserve commands, preparation hooks, input paths or completion
   predicates. Capture deterministic baselines from both existing drivers and compare their
   replacements under controlled pending/completed/skipped states, including exact
   submission options.

4. **[minor] Onboarding assertion too weak.** The synthetic third-tutorial test should
   resolve the complete canonical cube sequence, assert the commands use that tutorial's
   supplied paths, and run the actual tutorial validator over both YAML files, rather than
   resolving a self-selected step list and checking registered-root membership.

## Questions

None outstanding. The reviewer confirmed the `connectomics/playbooks/` decision and the
repository-root asset resolution, and accepted the evidence that
`validate_tutorial_configs.py` keys `CUSTOM_WORKFLOW_ROOTS` on the top-level YAML key, so
a new tutorial needs no Python registration.

## Verdict

VERDICT: NEEDS_CHANGES
