All six plan_v1 findings are addressed: compact-count validation, phase-bracketed and aggregate memory measurement, unconditional `dend_*` field comparison, failure-path coverage, queue-capacity accounting, and removal of branch creation. No blocking issues remain.

1. [minor] **Memory model, C5–C6:** The target bound omits the x-y plane buffer and the retained boundary observations in percentile mode. State the bound per scoring mode, including plane/face buffers and percentile storage. Otherwise measured memory can exceed the stated model despite a correct implementation.

2. [minor] **Verification step 5:** Unmodified stock binaries lack C0 markers, so their samples cannot receive the same phase attribution as modified binaries. Specify how baseline phases are identified, or report baseline overall peaks with detailed phase attribution only for the modified build. Flush modified-build markers promptly so stdout buffering does not distort brackets.

3. [minor] **Verification step 2:** Define `dend_*` score equality as identical field bytes, excluding padding. Numeric equality alone overlooks signed-zero changes and mishandles NaNs. Compute ABI offsets using byte-pointer differences; the displayed subtraction between differently typed pointers is illustrative, not compilable C++.

4. [minor] **Unit-test build:** Ensure `test_ws_bfs` checks remain active under `-DNDEBUG`. Plain `assert` would silently disable the boundary and accumulator checks in the specified Release builds.

Questions:
- Will baseline phase attribution be separately instrumented or explicitly unavailable?
- Will score-field comparisons and accumulator tests use bitwise floating-point comparisons?

The plan is sufficient to implement, with these refinements incorporated during coding. Review was limited to the supplied artifacts and source excerpts; no files were inspected or edited.

READY: yes