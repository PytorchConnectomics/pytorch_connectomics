Scanned the repo and CLAUDE.md; here’s what stands out and a concrete refactor plan tied to the current code (not the outdated REFACTORING_PLAN.md claims).

connectomics/lightning/lit_model.py is 1,439 lines and still mixes training, inference, cache loading, decoding, metrics, and loss logic. It also keeps unused private inference helpers even though connectomics/lightning/inference.py exists.
connectomics/lightning/inference.py is 918 lines handling sliding window, TTA, decoding, and file I/O in one class; responsibilities overlap with logic that still lives in lit_model.py.
connectomics/lightning/lit_config.py (732 lines) bundles datamodule creation with path resolution and interactive download prompts; hard to test or reuse in non-interactive environments.
connectomics/metrics/metrics_seg.py is 817 lines of numpy/scipy routines with globals and no torchmetrics-friendly wrapper, forcing manual conversions in lit_model.py:test_step.
Docs are out of sync: tests/INTEGRATION_TEST_STATUS.md says integration tests are modern, while tests/TEST_STATUS.md still reports legacy failures; REFACTORING_PLAN.md claims lit_model is 539 lines and split, which no longer matches reality.
Refactor plan (prioritized):

Lightning module cleanup (connectomics/lightning/lit_model.py)
Strip unused private inference helpers and delegate prediction flow entirely to InferenceManager.
Break test_step into small helpers (cache check/load, inference/TTA, decode/postprocess, metric logging, optional loss) to cut size and clarify control flow.
Add focused unit tests for the new helpers (no data dependencies) and reuse integration TTA/sliding-window tests after refactor.
Inference module split (connectomics/lightning/inference.py)
Extract sliding-window setup/ROI overlap resolution, TTA ensemble, and output I/O/decoding into separate modules (e.g., sliding_window.py, tta.py, io.py), leaving InferenceManager as a thin orchestrator.
Normalize config access (single dataclass or typed dict) to avoid repeated getattr/hasattr checks and to make this code testable without a full Hydra config.
Data/config factories (connectomics/lightning/lit_config.py, connectomics/lightning/lit_data.py)
Move path expansion and dataset existence/download logic into a dedicated utility with a non-interactive mode; avoid input() in library code.
Slim create_datamodule to orchestration only; push transform selection and per-mode overrides into small helpers that can be unit tested.
Add validation around required config fields before datamodule creation to fail fast with clear errors.
Metrics modernization (connectomics/metrics/metrics_seg.py, connectomics/lightning/lit_model.py)
Split numpy/scipy metrics into a pure-utils module and add lightweight torchmetrics-style wrappers for online evaluation; eliminate global matching_criteria.
Update test_step to consume the wrappers instead of manual tensor→numpy→tensor conversions; add unit tests for the wrappers and regression tests for adapted rand/VOI outputs.
Documentation and test-status alignment
Reconcile tests/TEST_STATUS.md with tests/integration/INTEGRATION_TEST_STATUS.md; remove contradictions and list real pass/fail status.
Update .claude/refactor/REFACTORING_PLAN.md to reflect the current module sizes/state and the new plan above; drop claims that are no longer true (e.g., lit_model already split).
Add a short maintainer note in .claude/refactor/ pointing to the authoritative status doc to avoid future drift.
Natural next step: confirm priorities/order and whether to start with the Lightning/inference cleanup (items 1–2) or the doc/test-status sync (item 5) so we can sequence the work.
