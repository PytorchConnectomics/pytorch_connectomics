# Plan v0 Review

## Summary

Codex (coder) reviewed plan_v0 read-only. The module homes are sensible with no import
cycle, and tests/`__init__`/README/monkeypatch are covered, but it withheld approval on a
material example import-path risk. Raw transcript: `state/plan_v0_review.review.raw.md`.

The `[major]` finding is confirmed empirically: `em_erl` is **not** installed in the target
env (`pip show em_erl` is empty; `import em_erl` fails from an arbitrary cwd). The test
suite passes only because `tests/conftest.py` inserts the repo root on `sys.path`, and the
current `scripts/j0126_workflow.py` runs only because it carries its own `sys.path` shim.
Removing that shim would break `python examples/j0126_workflow.py`. This matches the
environment convention (packages are typically run from a source checkout, not pip-installed).

## Findings

- **[major] Do not drop the `sys.path` bootstrap.** Since `em_erl` is not installed,
  examples must add the repo root to `sys.path` to be runnable directly from a source
  checkout. Resolution: keep a minimal 2-line bootstrap at the top of each example (apply
  it consistently to all four, improving on the current state where only `j0126_workflow.py`
  had one). Verification must run the examples directly (no install) to prove this.
- **[minor] Preserve the old callable surface.** Keep a thin `run_j0126_eval(...)` wrapper
  in `examples/j0126_workflow.py` that calls the generalized
  `em_erl.evaluate_skeletons_cloudvolume(seg_url=DEFAULT_SEG_URL, …)`, so the dataset default
  stays out of the package while the example still exposes the familiar entry point.

## Questions

- Should examples run from a clean source checkout without install? Answer: **yes** —
  `em_erl` is not installed here and the repo convention is source-tree execution, so the
  examples keep a `sys.path` bootstrap (resolved in plan_v1).

## Verdict

VERDICT: NEEDS_CHANGES
