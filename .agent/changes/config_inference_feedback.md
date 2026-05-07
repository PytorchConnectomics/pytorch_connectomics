# Review: `.claude/refactor/config_inference.md`

Reviewer notes on the inference-config refactor PR summary.

## Verified claims

Spot-checked against the current worktree:

- `InferenceModelConfig` and `SaveInferenceConfig` exist in `connectomics/config/schema/inference.py:22` and `:121` with the documented fields. ✓
- `OutputArrayConfig`, `OutputArrayPartitionConfig`, `OutputArrayStoreConfig` are gone — `grep -rn output_array connectomics/ tutorials/` returns nothing. ✓
- Runtime alias materialization (`sync_inference_runtime_aliases`) populates `inference.head`, `inference.select_channel`, `inference.crop_pad`, `inference.sliding_window`, `inference.save_prediction` from `inference.model`/`inference.window`/`inference.save_inference`. Line 267-336. ✓
- `inference/lazy.py:36`, `inference/tta.py:33`, `inference/sliding.py:131` import `resolve_model_output_dtype`; lazy + TTA accumulators use it. ✓
- `inference/output.py:243` reads `save_inference.dtype` first, falls back to `save_prediction.storage_dtype`. ✓
- Tutorials migrated: `tutorials/mito_betaseg.yaml:95-101` shows `inference.model.{head,channel_activations,crop_pad}` and no `output_array`/`storage_dtype`/`accumulator_dtype` keys remain anywhere under `tutorials/`. ✓
- Activation profiles emitted under `inference.model` in `connectomics/config/profiles/pipeline_profiles.yaml`. ✓

## Issues found (sorted by severity)

### 1. ⚠️ `build_sliding_inferer` does not gate on `lazy_load=true`

Pre-existing bug surfaced by this refactor. The doc's "Review Focus" mentions
the dtype path but not this gate.

`connectomics/inference/sliding.py:313-336` (pre-fix):
```python
def build_sliding_inferer(cfg) -> Optional[SlidingWindowInferer]:
    ...
    runtime = _resolve_sliding_window_runtime(cfg, roi_size)
    if is_distance_transform_blending(runtime["mode"]):
        raise ValueError(
            "inference.window.blending=distance_transform requires lazy_load=true; "
            ...
        )
```

`InferenceManager.__init__` calls this unconditionally for all non-2D configs.
With the new tutorials that set `inference.window.blending: distance_transform`
+ `lazy_load: true` (the BANIS path), the eager builder raises even though the
eager inferer is never used at runtime — the lazy path takes over.

`tutorials/neuron_nisb/base_banis.yaml:84` triggers this on `--mode test`.

**Fix already landed** in this turn: `build_sliding_inferer` now returns `None`
early when `inference.sliding_window.lazy_load=True`, since the eager
`SlidingWindowInferer` is unused in that case. `InferenceManager` and
`TTAPredictor` already handle `sliding_inferer=None` (the existing 2D path).

The PR summary should add this to "Implemented Behavior" or list it as a
follow-up; reviewers re-running test mode against `base_banis.yaml` would have
hit it immediately.

### 2. ⚠️ `weight_accumulator` follows `output_dtype` — fp16 precision risk

The doc explicitly notes "both value and weight accumulators now follow that
dtype" and lists this as a review-focus item. Worth flagging the concrete
risks the reviewer should weigh.

`connectomics/inference/lazy.py:970-972`:
```python
weight_accumulator = torch.zeros(
    (1, 1, *output_size), device=accumulation_device, dtype=output_dtype
)
```
Same in `connectomics/inference/tta.py:861-864`.

When `output_dtype=float16`:

- Gaussian importance-map tail values often fall below fp16's smallest
  positive normal (`6.10e-5`); incremental `weight_accumulator += importance_map`
  truncates those to 0 at the window corners.
- Division at line 963 (`raw_accumulator /= torch.clamp_min(weight_accumulator, 1.0e-6)`)
  uses a clamp of `1e-6` that itself underflows in fp16. Voxels with
  near-zero accumulated weight then divide by 0 (or by a denormal), producing
  visible artifacts at the volume corners.
- `lazy.py::_finalize_sliding_accumulators` (lines 41-51) raises the clamp to
  `torch.finfo(float16).tiny` for the value side, but the weight side
  accumulating in fp16 happens BEFORE the divide and is the root precision
  issue.

**Recommendation**: keep `weight_accumulator` at `float32` regardless of
`inference.model.output_dtype`. Only the `value_accumulator` benefits
materially from fp16 (it's `(1, C, ...)`, the weight is `(1, 1, ...)` and ~1/C
the size). This matches the previous `accumulator_dtype` design that this
PR is consolidating.

If the author intentionally wants weight in fp16 for memory parity, document
the corner-artifact risk in a tutorial note and add a smoke test that
validates a uniformly-saturated volume rounds-trips through the fp16 lazy
path within tolerance.

### 3. ⚠️ Test claims need clarification

The "Testing" section reports five separate `pytest` invocations on subsets
totaling 85 + 62 + 54 = 201 tests. The "Untested Areas" then admits "Full
repository test suite was not run". With the worktree sitting at 415 unit
tests after this turn's other refactors, that's <50% coverage.

The doc also says "the worktree contains additional dirty changes beyond
this inference config refactor, including affinity/loss-mask, decoding/kernel,
visualization, and augmentation changes" — meaning those test runs were
against a tree mixing four refactors. Pinning what *this* refactor touched
versus the others is hard from the test list alone.

**Recommendation**:
- Either run `pytest tests/unit/ -q` once on a clean commit and quote the
  total, or
- List the exact test files that exercise the inference-config changes
  (probably `test_hydra_config.py`, `test_inference_tta_masking.py`,
  `test_lazy_inference.py`, `test_chunked_inference.py`,
  `test_inference_stage.py`, `test_prediction_transform.py`) and explicitly
  state that the rest of the suite was not re-run because this refactor
  doesn't touch their code paths.

### 4. 📝 Section naming: `inference.model` is broader than the name implies

`InferenceModelConfig` carries 6 fields (`head`, `select_channel`,
`output_dtype`, `activation_profile`, `channel_activations`, `crop_pad`).
That's "everything that happens between forward and save", not just
"model" parameters. A reader who sees `inference.model` in YAML may expect
architecture knobs (it's not — those live under top-level `model:`).

**Suggestions** (any of):
- Rename to `inference.post_forward` or `inference.model_output`.
- Keep the name but pin a docstring at the top of `InferenceModelConfig`:
  "Anything that runs after model forward and before save/decode: channel
  selection, activation, crop, accumulator dtype."
- Split: keep `inference.model` for `head`/`select_channel`, hoist
  `channel_activations` + `activation_profile` to `inference.activation`,
  keep `crop_pad` + `output_dtype` under `inference.window` (where they
  semantically belong since they're sliding-window-specific).

The split option is intrusive but separates concerns better. The rename or
docstring is a 1-line fix.

### 5. 📝 `cache_suffix` default has hardcoded `.h5` extension

`SaveInferenceConfig.cache_suffix: str = "_x1_prediction.h5"` (line 127) and
`backend: str = "h5"` (line 125) keep `.h5` baked in. When the user sets
`backend: zarr` the suffix is wrong and runtime code has to special-case it.

**Suggestions**:
- Drop the extension from `cache_suffix` and let the writer append it based
  on `backend` (`.h5`/`.zarr`/`.ts`).
- Or document that `cache_suffix` includes the extension and is the
  caller's responsibility to align with `backend`.

Low-priority: only matters once a user actually flips `backend`.

### 6. 📝 Dual `save_inference` + `save_prediction` defaults

`InferenceConfig` has BOTH `save_inference: SaveInferenceConfig` (line 230,
canonical) and `save_prediction: SavePredictionConfig` (line 238, runtime
alias). `sync_inference_runtime_aliases` mirrors save_inference →
save_prediction.

If a user provides both in YAML (intentionally or via copy-paste from an
old config), behavior is "save_inference wins via sync, save_prediction is
silently overwritten". The strict-config-key validator catches typos but not
this kind of duplication.

**Suggestion**: either reject `save_prediction` at YAML-parse time (it's a
runtime-only field), or add a one-line note in the schema docstring saying
"`save_prediction` is an internal runtime alias; configure
`save_inference` instead".

Same dynamic for `inference.window` vs `inference.sliding_window`,
`inference.model.head` vs `inference.head`, etc. Several runtime-alias
fields are accessible via YAML and would silently lose to the canonical
section's value.

### 7. 📝 Scope mixing — flagged by the doc itself

The "Untested Areas" section already admits the worktree mixes four
unrelated refactors. The summary's "Changed Files" section lists
`tests/unit/test_lit_utils.py`, `test_connectomics_module.py`, etc., which
were also touched by the loss-mask refactor in adjacent turns.

**Recommendation for the PR author**: split this into two PRs before
merging — (a) inference config schema + tutorial migration, (b) loss-mask
refactor. They share no semantic dependency. The current bundle makes
`git bisect` and revert harder if either lands a regression.

If the team prefers a single PR, at least split the commit history so the
inference-config commits don't mix with loss-mask edits.

## Things the doc handled well

- Clear "Implemented Behavior" → "Testing" → "Untested Areas" → "Review
  Focus" → "Changed Files" structure. Easy to triage.
- Honest "Untested Areas" — flagged the partial test runs and the dirty
  worktree without burying them.
- Specific review-focus items, not generic "please review carefully".
- File-by-file changed-files list with one-line summaries.

## Suggested follow-ups before merge

1. Land the lazy-mode gating fix in `build_sliding_inferer` (already done
   in the worktree this turn; confirm it's part of this PR or a separate
   one).
2. Decide on weight-accumulator dtype: fp32 always (recommended) vs follow
   `output_dtype` (current). If the latter, add the fp16 corner-artifact
   note to the tutorial.
3. Re-run the full unit test suite on a clean inference-config-only commit
   and quote the total.
4. Either rename `inference.model` or lock its scope with a docstring.
5. Split or document the dual canonical-vs-runtime-alias fields
   (`save_inference`/`save_prediction`, `window`/`sliding_window`).

## Overall

The architectural split is sound and the migration is mostly clean. The
risks are concentrated in the dtype path (item 2) and the operational
surface (items 1, 3). None of items 4-7 block merge; they're follow-up
hygiene.
