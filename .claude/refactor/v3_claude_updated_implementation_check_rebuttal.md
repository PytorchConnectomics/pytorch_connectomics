# Rebuttal to `v3_claude_updated_implementation_check.md`

The implementation check is useful and mostly accurate, but several points
should be tightened before using it as the next execution plan. The main issue
is that it treats some contract ambiguity as a direct implementation failure.

## Main corrections

### 1. The "two broken tutorials" finding is overstated

The check is right that these two files do not load through
`connectomics.config.load_config()`:

- `tutorials/waterz_decoding_large.yaml`
- `tutorials/waterz_decoding_large_abiss.yaml`

But the repo already has a tutorial validator:

```bash
python scripts/validate_tutorial_configs.py
```

and `.github/workflows/tests.yml` already runs it in CI. That script explicitly
declares:

```python
CUSTOM_WORKFLOW_ROOTS = {"large_decode", "abiss_large"}
```

So the actual issue is not "CI missed two broken tutorials." The issue is:

> `tutorials/` contains both canonical `scripts/main.py --config` configs and
> custom large-volume workflow YAMLs that intentionally bypass the structured
> `Config` schema.

That is still a real problem, but it is a documentation and layout contract
problem, not simply a broken tutorial regression.

Recommended rewrite:

- Replace "Two tutorials shipped broken" with "Tutorial vs custom-workflow
  config contract is ambiguous."
- Keep the `load_config()` failure as evidence.
- Also mention that the official validator currently passes and intentionally
  skips those custom workflow roots.

### 2. Add acceptance criteria to every follow-up

Several recommendations are directionally good but too open-ended. Add clear
done conditions:

- Evaluation extraction is done when:

```bash
rg "module\\._|hasattr\\(module" connectomics/evaluation -g '*.py'
```

returns no matches, and `run_evaluation_stage` has a test that runs from a
plain config plus arrays without constructing `ConnectomicsModule`.

- Chunk-grid extraction is done when `streamed_chunked.py` imports no
underscore-prefixed helpers from `inference.chunked`.

- Tutorial contract cleanup is done when `scripts/validate_tutorial_configs.py`
documents custom workflow roots and reports how many canonical configs vs
custom workflow YAMLs it validated or skipped.

- File splits are done only when each split creates a clear ownership boundary
and targeted tests pass with no behavior changes.

### 3. Move quick isolated fixes before major rewrites

The current follow-up sequence starts with tutorial CI and then jumps to the
large evaluation extraction. I would separate the small cleanup work first:

1. Remove ignored leftovers:
   - `tutorials/mito_betaseg.yaml~`
   - `connectomics/decoding/postprocessing/`
2. Clarify `scripts/validate_tutorial_configs.py` behavior around
   `large_decode` and `abiss_large`.
3. Type `DecodingConfig.tuning` instead of leaving it as `Optional[Dict[str, Any]]`.
4. Extract public chunk-grid helpers.
5. Then do the larger evaluation-context refactor.

This reduces noise before the behavioral refactor.

### 4. Be less LOC-driven about PR 10

The file-split criticism is useful, but line count alone is a weak reason to
split code. The next plan should prioritize ownership boundaries:

- Good split candidates:
  - `inference/lazy.py`, because the lazy sliding-window loop and accessor
    behavior are separable.
  - `data/augmentation/transforms.py`, if grouped by transform family while
    preserving public imports.
  - `decoders/segmentation.py`, because the numba connected-components kernel
    and EDT helper are algorithmic kernels.

- Lower priority:
  - `callbacks.py`
  - `data_factory.py`

Those should wait until an adjacent behavior change makes a clear module
boundary obvious.

### 5. Add an accepted-deviations section

Some deviations from `v3_claude_updated.md` are reasonable:

- A3 product/API decisions were deferred because they need maintainer sign-off.
- Not every large file needs to be split immediately.
- Keeping custom large-volume workflow YAMLs outside structured `Config` may be
  acceptable if it is documented.

The implementation check should distinguish "defect" from "accepted for now."

## Suggested revised follow-up sequence

### PR-12A: Contract and leftover cleanup

- Delete ignored leftovers:
  - `tutorials/mito_betaseg.yaml~`
  - `connectomics/decoding/postprocessing/`
- Update `scripts/validate_tutorial_configs.py` docstring to explain custom
  workflow roots.
- Make the validator report canonical configs validated and custom workflows
  skipped.
- Keep CI running the validator.

Acceptance:

```bash
python scripts/validate_tutorial_configs.py --glob 'tutorials/*.yaml' --glob 'tutorials/**/*.yaml'
git ls-files --others --ignored --exclude-standard 'tutorials/mito_betaseg.yaml~' 'connectomics/decoding/postprocessing/**'
```

The first command passes; the second command shows no stale leftovers.

### PR-12B: Type decoding tuning config

- Replace `DecodingConfig.tuning: Optional[Dict[str, Any]]` with a dataclass.
- Reuse or align with `DecodingParameterSpace` / tuning schema types.
- Run tutorial validation and optuna tuner tests.

Acceptance:

```bash
rg "tuning: Optional\\[Dict\\[str, Any\\]\\]" connectomics/config/schema
```

returns no matches.

### PR-13: Public chunk-grid utilities

- Move chunk-grid helpers into a public module, likely
  `connectomics/inference/chunk_grid.py`.
- Drop underscore prefixes for public helpers.
- Update `inference/chunked.py` and `decoding/streamed_chunked.py`.

Acceptance:

```bash
rg "from \\.\\.inference\\.chunked import \\(" connectomics/decoding/streamed_chunked.py
```

returns no private helper import block.

### PR-14: Evaluation context extraction

- Add `EvaluationContext` and metric/logging sink adapter.
- Make `connectomics.evaluation` depend on the context, not on
  `ConnectomicsModule`.
- Delete defensive `hasattr(module, "_...")` paths.
- Add a pure evaluation unit test with config plus arrays.

Acceptance:

```bash
rg "module\\._|hasattr\\(module" connectomics/evaluation -g '*.py'
```

returns no matches.

### PR-15: Thin `scripts/main.py`

- Move mode dispatch into `connectomics/runtime/dispatch.py`.
- Keep `scripts/main.py` focused on argument parsing, config setup, and one
  dispatch call.

Acceptance:

- `scripts/main.py` has no cache/tune/test branching details.
- Existing `scripts/main.py --help` and runtime mode tests pass.

### PR-16: Targeted file splits

Do only splits with clear ownership:

- `inference/lazy.py`
- `data/augmentation/transforms.py`
- `decoders/segmentation.py`

Acceptance:

- Public imports remain stable unless explicitly removed.
- Targeted tests pass after each split.
- No unrelated behavior changes are mixed into split-only commits.

## Bottom line

The implementation check should remain critical about the real gaps:

- Evaluation is still coupled to `ConnectomicsModule`.
- Chunked decode imports private inference helpers.
- `DecodingConfig.tuning` is still too loose.
- Some ignored stale files remain.

But it should correct the tutorial claim. The repo already has tutorial config
validation in CI; the unresolved problem is that custom large-volume workflow
YAMLs live under `tutorials/` while intentionally bypassing the structured
config schema. That needs an explicit contract, not just a stricter smoke test.
