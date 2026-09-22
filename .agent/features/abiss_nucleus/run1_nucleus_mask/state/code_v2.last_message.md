Implemented the final `code_v2` revision and wrote the validated artifact: [code_v2.md](/projects/weilab/weidf/lib/pytorch_connectomics/.agent/features/abiss_nucleus/work/abiss/code_v2.md).

- G1: Added `NUC_RATIO`/`NUC_OFFSET`, nearest-neighbour alignment, identity-path byte comparison, and lower-resolution z-offset testing.
- G2: Added deferred boundary nucleus records, `extra_nuc.data` restoration, composite append, and a two-child veto-survival regression.
- G3: Documented the observation-only limitation and must-link snap follow-up.
- No commit created; HEAD remains `3c4f56219488441edb4c51ee7ca4f07cdef2cd8f`.

Mandatory invariance output:

```text
V2 build baseline: PASS
V2 build current: PASS
V2 baseline comparison: identical=35 differing=0 missing=0
V2 current-only empty sidecars: done_nuc.data nuc_cuts.data ongoing_nuc.data
V2 default-path nucleus log lines: 0
V2 stale nucleus environment without nuc.raw: PASS
run_v2_invariance: PASS
```

Focused builds, G1/G2 regressions, hierarchy tests, C++ units, shell syntax, artifact validation, and `git diff --check` also passed.

CCC run: `work/abiss`  
Stage completed: `code_v2`  
Next action: `review_v2`