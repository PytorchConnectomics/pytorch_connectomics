Summary: the plan is mostly scoped correctly and places the reusable helpers in sensible package modules without an obvious import cycle. It covers tests, `__init__`, README path updates, and the monkeypatch point. I would not approve it as written because it has a material example import-path risk.

Findings:

- [major] `examples/j0126_workflow.py` removes the `sys.path` shim, but the verification still runs `python examples/j0126_workflow.py -h` directly. In a source checkout, direct script execution puts `examples/` on `sys.path`, not necessarily the repo root, so sibling `em_erl` imports can fail unless the package is already installed or `PYTHONPATH` is set. That breaks the “example CLIs must still run” requirement and is a broken import path.

- [minor] The plan does not retain an example-level `run_j0126_eval` wrapper with the J0126 default URL. The package-level generalized `evaluate_skeletons_cloudvolume(..., seg_url, ...)` is fine, but keeping a thin wrapper in `examples/j0126_workflow.py` would better preserve the old script’s callable surface while still keeping dataset-specific defaults out of the package.

Questions:

- Should examples be runnable directly from a clean source checkout, or is editable install/PYTHONPATH an explicit prerequisite? The plan’s verification assumes direct source-tree execution.

READY: no