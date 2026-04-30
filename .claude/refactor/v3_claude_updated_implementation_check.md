# V3 Implementation Check (against `v3_claude_updated.md`)

Reviewed against branch `master` as of commit `eeab7cc`. Plan baseline was
`ba0f482`; 13 commits land between then and HEAD, mapping cleanly to PRs 0–11
in the plan:

| PR (plan) | Commit | Subject |
|---|---|---|
| 0 | `ceeaa5d` | Add v3 refactor guardrails |
| 1 | `8eed96b` | Remove orphan v3 refactor code |
| 2 | `304a61b` | Extract runtime output naming |
| 3 | `7c63ea4` | Enforce strict config keys |
| 4 | `5db95e2` | Move test evaluation into evaluation package |
| 5 | `1768d8e` | Move tuning runtime orchestration |
| 6 | `eef2ea5` | Separate chunked decoding from inference |
| 7 | `83425ad` | Extract runtime dispatch helpers |
| 8 | `145a695` / `12c6dab` | Split decoding schema and migrate tutorials / Finalize raw prediction artifact stage |
| 9 | `f6b83d3` | Trim public API exports |
| 10 | `150d359` | Split TTA and prediction crop helpers |
| 11 | `eeab7cc` | Refresh docs for current refactor |

The architectural backbone landed. Most of the boundary work, schema split,
strict config flip, runtime extraction, and naming consolidation are correct
and clean. Two classes of issue stand out: **unenforced contracts** (a
guardrail that wasn't run, two tutorials silently broken) and
**half-extractions** (evaluation moved files but kept Lightning coupling). PR
10 is largely deferred — the bulk of the file-split work is unstarted.

This document grades each PR, calls out concrete issues with file:line
evidence, and proposes a focused follow-up sequence.

---

## Executive scorecard

| PR | Goal | Status | Confidence |
|---|---|---|---|
| 0 | Guardrails | **B**: AST tests, snapshot tests, smoke benchmark all in place. Tutorial-load test missing — and that gap caused real damage in PR 8. | high |
| 1 | A1 dead artifacts | **B+**: `auto_tuning.py` (535 LOC), `blend.py`, bbox orphans, `calculate_inference_grid` all gone. Missed `tutorials/mito_betaseg.yaml~` and the empty `decoding/postprocessing/` dir. | high |
| 2 | Runtime naming extraction | **A**: `lightning/utils.py` 771→77; lazy `__getattr__` re-export through `runtime/__init__.py` is a clean design choice the plan didn't even ask for. | high |
| 3 | Strict config | **B**: warn→raise flipped, two ghost reads removed. But shipped two broken tutorials silently — symptom of the missing PR-0 tutorial-load test. | high |
| 4 | Evaluation extraction | **C**: files moved to `connectomics/evaluation/`, `_compute_test_metrics` removed, `TestContext` introduced. But the design promise — "decode-only and evaluate-only modes runnable without `ConnectomicsModule`" — is not delivered. 27 `module._*` / `hasattr(module, "_…")` accesses remain in `evaluation/` files. | medium |
| 5 | Tune runner extraction | **A**: `run_tuning`, `load_and_apply_best_params`, `temporary_tuning_inference_overrides` correctly in `runtime/tune_runner.py`. `optuna_tuner.py` no longer imports `connectomics.training` and works on saved arrays. | high |
| 6 | Chunked decode boundary | **B-**: `decoding/streamed_chunked.py` exists, `inference/chunked.py` separated, `write_prediction_artifact` populated. But `streamed_chunked.py` imports six **private** helpers (`_build_chunk_grid`, `_resolve_chunk_output_mode`, `_resolve_chunk_shape`, `_resolve_global_prediction_crop`, `_resolve_h5_spatial_chunks`, `_validate_chunked_output_contract`) from `inference.chunked`. Boundary-clean by AST, leaky by design. | medium |
| 7 | Runtime CLI extraction | **B+**: all five runtime modules exist; B5 cross-section validation correctly moved to `runtime/preflight.py`. `scripts/main.py` 1107→394 — better, but 2.6× the plan target of ≈150. | high |
| 8 | Schema split + tutorial migration | **B-**: schema split clean, `nnunet_pretrained` rename clean, `postprocess.py` rename clean, 38/40 tutorials load. But two tutorials raise on load (regression) and `DecodingConfig.tuning` is typed `Optional[Dict[str, Any]]` instead of a dataclass. | medium |
| 9 | Public API trim | **A**: snapshot test passes; A2 removals on target. A3 items deferred (correct per plan). | high |
| 10 | File splits | **D**: only `prediction_crops.py` and `tta_combinations.py` extracted. `lazy.py` (1295), `tta.py` (1299), `data_factory.py` (1168), `callbacks.py` (1001), `data/augmentation/transforms.py` (1346), `data/processing/transforms.py` (979), `decoders/segmentation.py` (815) all untouched. | high |
| 11 | Docs refresh | **A-**: stale `VolumeDataset`/`TileDataset` references rewritten with v2 paths and a removal note. Duplicate `models.rst` removed. | medium (spot-check only) |

Weighted average across the 12 PRs lands at roughly **B**. The high-leverage
boundary work is done. The work that requires *behavioral* change rather than
file movement (PR 4, PR 10) is where the slippage concentrates.

---

## Critical issues (block-on-merge)

### 1. Two tutorials shipped broken — no CI net caught it

**Evidence.** Loading every tutorial via `connectomics.config.load_config`:

```text
Total tutorials (recursive): 40
Failures: 2
  tutorials/waterz_decoding_large.yaml:
    ValueError: Unknown top-level config key 'large_decode'
  tutorials/waterz_decoding_large_abiss.yaml:
    ValueError: Unknown top-level config key 'abiss_large'
```

Both files use top-level keys (`large_decode:` and `abiss_large:`) that don't
exist in the schema. `scripts/decode_large.py` (line 94) reads them via
`cfg.get("large_decode", {})`, suggesting they've always been an out-of-band
ad-hoc config surface for the large-volume orchestration scripts. PR 3's
`_warn_unconsumed_keys → raise ValueError` flip turns that into a load
failure, and PR 8 didn't migrate them.

**Why this matters.** The plan's own PR-0 specification said:

> "Add a tutorial-load test that imports every YAML in `tutorials/` and
> materializes the structured config without errors. Run in CI to prevent
> silent regressions during the migration."

That test does not exist. `tests/unit/test_v3_guardrails.py` covers
public-API snapshots, boundary AST scans, and one ad-hoc strict-config test
on a synthetic YAML — but it never exercises an actual tutorial. The most
important PR-0 guardrail is the one that wasn't built.

**Fix.**

1. Add `tests/unit/test_tutorial_load_smoke.py` that calls `load_config()` on
   every `tutorials/**/*.yaml`. Parametrize so failures point to the file.
2. Decide what to do with `large_decode` / `abiss_large`:
   - Move them out of `tutorials/` (these aren't tutorials in the
     `scripts/main.py --config` sense — they feed
     `scripts/decode_large.py`). `tutorials/large_volume/` or `recipes/`
     would be more honest.
   - Or extend the schema with a sibling `large_decode:` / `abiss_large:`
     section if they're going to remain part of the canonical Config surface.

Either is fine; the wrong path is leaving them broken in place.

---

### 2. Evaluation extraction is a file move, not a coupling break

**Evidence.** `connectomics/evaluation/report.py` line 42:

```python
def is_test_evaluation_enabled(module) -> bool:
    if hasattr(module, "_is_test_evaluation_enabled"):
        return bool(module._is_test_evaluation_enabled())
    evaluation_cfg = get_effective_evaluation_config(module)
    return bool(module_cfg_value(module, evaluation_cfg, "enabled", False))
```

Same shape repeats throughout `evaluation/`:

```text
$ grep -c "module\._\|hasattr(module" connectomics/evaluation/*.py
metrics.py: 7
nerl.py:    3
report.py: 17
                    ───
total:     27
```

Every public function in `connectomics.evaluation` takes `module` as its
first argument. Most contain a defensive `hasattr(module, "_…")` fallback
path. `evaluation/stage.py::run_evaluation_stage(module, decoded_predictions,
labels, …)` continues this pattern.

**Why this matters.**

- The plan was explicit: "Removes 25 `module._*` private-method couplings…
  Decode-only and evaluate-only modes become runnable without instantiating
  `ConnectomicsModule`." The current shape of `compute_test_metrics(module,
  predictions, labels, name)` requires either a Lightning module or a
  Lightning-module-shaped duck-typed object (with `_get_runtime_inference_config`,
  `_is_test_evaluation_enabled`, `test_adapted_rand`, `log()`, etc.). You
  cannot run evaluation on a saved prediction artifact with just a `Config`
  and a numpy array — which was the entire point of D2 + B3.
- The defensive `hasattr` fallbacks are exactly the kind of backward-compat
  shim the v2 contract forbids. They support a hypothetical "module without
  the underscore methods" caller that does not exist.
- `TestContext` (test_pipeline.py:51–110) was added to formalize the
  contract, then ignored by the evaluation package itself. The contract is
  declared at the call boundary (`TestContext.from_module`) but doesn't flow
  into `connectomics.evaluation`.

**Fix.** Re-extract for real. Suggested signature for `run_evaluation_stage`:

```python
@dataclass
class EvaluationContext:
    cfg: Config
    evaluation_cfg: EvaluationConfig
    inference_cfg: InferenceConfig
    output_path: Path | None
    checkpoint_path: Path | None
    metric_sinks: MetricSinks   # adapter object that absorbs torchmetrics calls

def run_evaluation_stage(
    ctx: EvaluationContext,
    decoded_predictions: np.ndarray,
    labels: np.ndarray | None,
    *,
    filenames: Sequence[str],
    batch_idx: int,
) -> EvaluationStageResult: ...
```

Lightning callers build `EvaluationContext` once per epoch from the
`ConnectomicsModule`. CLI evaluate-only mode builds it from a config + a
loaded artifact. The `_save_metrics_to_file` / `module.log()` paths become
behavior on the `metric_sinks` adapter, not direct `module.log` calls inside
`evaluation/`.

This is a real refactor, not a rename. It's the design promise the original
PR 4 was supposed to deliver.

---

### 3. `streamed_chunked.py` re-imports private chunk-grid helpers

**Evidence.** `connectomics/decoding/streamed_chunked.py:14–20`:

```python
from ..inference.chunked import (
    _build_chunk_grid,
    _resolve_chunk_output_mode,
    _resolve_chunk_shape,
    _resolve_global_prediction_crop,
    _resolve_h5_spatial_chunks,
    _validate_chunked_output_contract,
)
```

The static AST boundary test (`test_inference_static_imports_do_not_reference_decoding`)
passes because the *direction* is decoding→inference, not the reverse.
But importing six underscore-prefixed helpers across packages is a leak —
those names were intended to be private to `chunked.py`.

**Fix.** Promote the chunk-grid utilities to a public module:

```text
connectomics/inference/chunk_grid.py   # build_chunk_grid, resolve_chunk_shape, …
```

Then both `inference/chunked.py` and `decoding/streamed_chunked.py` import
from the public surface. Plan PR 10 already wants `chunked.py` split; do the
extraction at the boundary, not as a "split for size" exercise.

---

## Significant issues (should fix soon)

### 4. PR 10 is roughly 30% complete

| File | LOC at baseline | LOC now | Plan target | Done? |
|---|---|---|---|---|
| `training/lightning/test_pipeline.py` | 2005 | 968 | ≈1100 after B3 | Yes (B3 extraction is the real reduction) |
| `training/lightning/utils.py` | 771 | 77 | trimmed by PR 2 | Yes |
| `inference/chunked.py` | — | 521 | n/a | Yes (boundary fix) |
| `inference/lazy.py` | 1295 | 1295 | ≈900 | **No** (untouched) |
| `inference/tta.py` | 1547 | 1299 | tta/ subpackage | Partial (`tta_combinations.py` extracted, no subpackage) |
| `data/augmentation/transforms.py` | 1346 | 1346 | 4-way split | **No** |
| `data/processing/transforms.py` | 979 | 979 | 4-way split | **No** |
| `training/lightning/data_factory.py` | 1168 | 1168 | 4-way split | **No** |
| `training/lightning/callbacks.py` | 1001 | 1001 | per-callback files | **No** |
| `decoding/decoders/segmentation.py` | 815 | 815 | extract `_compute_edt`, numba CC kernel | **No** |
| `config/pipeline/config_io.py` | 981 | 692 | further split | Partial (only the B5 move) |

PR 10 was demoted to "secondary" in the plan, which is correct, and the plan
explicitly cautioned against splitting where ownership is already clean. But
seven of the ten target files have **zero** lines moved. That isn't
"prioritized restraint" — that's the work being unstarted.

The largest live cost is `inference/lazy.py` (1295 lines) and the two
`transforms.py` files (1346 + 979). They still mix unrelated concerns;
review-time and merge-conflict cost continues to compound.

**Fix.** Pick the three highest-leverage splits and ship them in a focused
PR-12: (a) `inference/lazy.py` shared `_run_lazy_sliding` core (the plan's
specific recommendation, since `LazyVolumeAccessor` already moved to
`data/io/`); (b) split `data/augmentation/transforms.py` into the four
suggested groupings; (c) extract the numba CC kernel from
`decoders/segmentation.py`. Skip the rest until the codebase asks for them.

---

### 5. `scripts/main.py` at 394 lines (target ≈150)

Plan F said "Target `scripts/main.py` ≈ 150 lines: parse → dispatch by mode →
call `runtime.{train,test,tune}`." Current shape: 394 lines, 17 imports from
`connectomics.runtime.*`, plus inline dispatch logic for cache-only,
TTA-cache, and tune modes.

**Fix.** Pull the mode-dispatch into a `runtime/dispatch.py::main(cfg, args)`
or three small `run_train` / `run_test` / `run_tune` callables. Keep
`scripts/main.py` to ~80 lines of: `args = parse_args(); cfg =
setup_config(args); runtime.dispatch(cfg, args)`. Several of the imports in
`scripts/main.py` (e.g. `_create_decode_only_datamodule`,
`_has_cached_predictions_in_output_dir`, `_has_tta_prediction_file`) can move
behind that dispatch call.

---

### 6. `DecodingConfig.tuning: Optional[Dict[str, Any]]`

**Evidence.** `connectomics/config/schema/decoding.py:67`:

```python
@dataclass
class DecodingConfig:
    ...
    tuning: Optional[Dict[str, Any]] = None
```

Plan D1 specified:

```python
tuning: Optional[DecodingTuningConfig] = None
```

**Why this matters.** The `tuning` block is non-trivial (search space,
parameter spaces, optimizer settings — see `stages.TuneConfig` and
`DecodingParameterSpace`). Accepting an open `Dict[str, Any]` defeats both
the strict-config guarantee (any nested key is silently accepted) and the
schema typing benefit. It also reintroduces ghost-getattr probes inside
`optuna_tuner.py` whenever it reads `cfg.decoding.tuning.*`.

**Fix.** Promote the existing `connectomics/config/schema/stages.py::DecodingParameterSpace`
or define a sibling `DecodingTuningConfig` dataclass and reference it from
`DecodingConfig.tuning`. Use the strict typing pass (PR 3 already raises on
unknown keys) to catch any tutorial drift.

---

### 7. A1 leftovers

- `tutorials/mito_betaseg.yaml~` — editor backup, plan listed it explicitly,
  still present.
- `connectomics/decoding/postprocessing/` — empty directory containing only
  a stale `__pycache__/` (5 .pyc files, including
  `postprocessing.cpython-311.pyc` which references the old name). Remove
  the directory; ensure `.gitignore` keeps it removed.

These are 5-minute fixes that signal incomplete cleanup.

---

## Minor issues (nice to fix when adjacent)

### 8. Three call sites for `register_torch_safe_globals()`

`config/schema/root.py:94`, `training/lightning/trainer.py:34`, and
`scripts/main.py:78` all call it. The implementation is
idempotent (the function early-returns if the registration would no-op), so
this is harmless at runtime, but the plan called for a single canonical site.

A reasonable resolution: keep the call in `runtime/torch_safe_globals.py`
(via module-import side effect), drop the other two, and document that any
process that uses `connectomics.runtime` registers the safe globals. Not
critical.

### 9. `_validate_chunked_output_contract` was repurposed, not removed

Plan B2 said "remove `_validate_chunked_output_contract` postprocessing
checks (175–182): postprocessing is not reachable from raw-prediction
inference once B2 lands." Codex rewrote the function (chunked.py:125–134)
to validate `save_prediction.output_formats` instead — a different concern.
That's defensible, but the plan's intended deletion didn't happen. The new
function is cheap and correct; flag for awareness.

### 10. A3 product items left undecided

`RandMixupd`, `auto_plan_config`, `WandbConfig` (`use_wandb: bool = False`),
`GANLoss`, `SegToFlowFieldd`/`ComputeBinaryRatioWeightd` etc., and
`TestConfig.output_path / cache_suffix` are all still present and exported.
Plan A3 said these required maintainer sign-off. Codex took the safe path
(no decision = keep), which is defensible. But the plan also said they
should be "tracked through PRs 8/9 with explicit decisions" — no such
decision document exists. Recommendation: add a one-line `A3 deferred:` note
in `v3_claude_updated.md` or a follow-up issue tracker so they aren't
silently accepted as permanent.

### 11. `runtime/checkpoint_dispatch.py:10` top-level import of `training`

```python
from ..training.lightning.runtime import setup_run_directory
```

Plan didn't ban runtime→training (and rightly so — runtime *calls* training).
But the other runtime modules (`sharding.py`, `tune_runner.py`,
`cache_resolver.py`) all use lazy in-function imports. The mixed style is
cosmetic, but worth aligning. Either:

- Move `setup_run_directory` to `runtime/` (it's an output-path helper, not
  a Lightning concern), or
- Keep the import but annotate why it's top-level here (e.g. it's eagerly
  needed by every callable in this module, lazy imports buy nothing).

---

## What landed well (worth calling out)

- **`runtime/__init__.py` lazy `__getattr__` re-export** is cleaner than the
  plan even asked for. Avoids circular-import risk while preserving the
  public surface.
- **Boundary AST tests** in `test_v3_guardrails.py` use AST walking (catches
  imports inside functions) rather than `sys.modules` snapshot — more
  robust than the sketched approach in PR 0.
- **Public API snapshot tests** are concrete and tight (they assert exact
  set equality, so any drift fails loudly).
- **Decoder lazy registration** (`registry.py:_BUILTINS_REGISTERED`) cleanly
  delivers Theme P's lazy registration target. Previously
  `decoding/__init__.py` ran the imports at top-level; now first call to
  `get_decoder` / `list_decoders` triggers it.
- **`PredictionArtifactMetadata`** carries every field the plan asked for
  (`transpose`, `model_architecture`, `model_output_identity`,
  `decode_after_inference`).
- **`optuna_tuner.py` is now genuinely a pure tuner.** No
  `connectomics.training` imports, all naming through
  `connectomics.runtime.output_naming`. Reading the file end-to-end, the
  decoding↛training boundary is real, not just AST-clean.
- **`runtime/preflight.py::validate_runtime_coherence`** correctly absorbed
  `count_stacked_label_transform_channels`, eliminating the
  config→data.processing coupling. Good.
- **Tutorial migration body of work** (38/40 files) was done without a
  schema/key drift — the only failures are the two non-tutorial workflow
  YAMLs.

The boundary fixes (PRs 2, 5, 6, 7, 8 schema-side) are the strongest part of
this implementation. They survive a careful read.

---

## Quantitative summary

Plan estimated:
- ≈ 1.2 KLOC removed in PR 1
- ≈ 800 LOC removed in PR 4 (net)
- ≈ 400 LOC removed in PR 9
- ≈ 0.5 KLOC removed in PR 3
- ≈ 1.5 KLOC moved
- Total: ≈ 2.5 KLOC removed, 1.5 KLOC moved

Shipped (rough):
- `auto_tuning.py`: −535
- `lightning/utils.py`: −694
- `scripts/main.py`: −713
- `optuna_tuner.py`: −470
- `test_pipeline.py`: −1037
- `config/pipeline/config_io.py`: −289
- Plus orphan A1 files (≈ 200)
- Total reduction in "before" files: ≈ 3.9 KLOC

Plus the new `runtime/` (≈ 2 KLOC across 9 files), `evaluation/` (≈ 1.5 KLOC
across 6 files), and PR-1 added tests / new schema files. So ≈ 1.5–2 KLOC
genuinely deleted, ≈ 2 KLOC moved, and ≈ 1.5 KLOC added (new package
infrastructure). That's roughly in line with the plan's "honest reassessment"
section: 2.5 KLOC out, 1.5 KLOC moved, ignoring the A3 deferral.

---

## Recommended follow-up sequence

In dependency order, smallest first:

### PR-12: Tutorial-load CI test + fix the two broken tutorials

- Add `tests/unit/test_tutorial_load_smoke.py` that calls `load_config` on
  every `tutorials/**/*.yaml` and asserts no raise. Parametrize so
  failures point at the file.
- Decide whether `waterz_decoding_large.yaml` and
  `waterz_decoding_large_abiss.yaml` belong in `tutorials/` (no — move
  to `recipes/large_volume/` or extend the schema with a real
  `large_decode:` section).
- Delete `tutorials/mito_betaseg.yaml~`.
- Delete `connectomics/decoding/postprocessing/` directory.

This is a 1-hour PR that closes the most damaging miss (a broken-on-load
config in `master`).

### PR-13: Real evaluation extraction

- Define `EvaluationContext` dataclass + `MetricSinks` adapter in
  `connectomics/evaluation/context.py`.
- Rewrite `compute_test_metrics`, `is_test_evaluation_enabled`,
  `evaluation_metric_requested`, `configured_evaluation_metrics`,
  `save_metrics_to_file`, `log_test_epoch_metrics` to take
  `EvaluationContext` (not `module`).
- Delete the `hasattr(module, "_…")` defensive paths.
- Update `test_pipeline.py` and `model.py` to build the context once and
  pass it through.
- Add a unit test that runs `run_evaluation_stage` against a numpy array
  + a Config without instantiating `ConnectomicsModule`.

This is the design change that PR 4 promised and didn't ship. It's the
second-biggest gap.

### PR-14: Public chunk-grid utilities

- Move `_build_chunk_grid`, `_resolve_chunk_output_mode`,
  `_resolve_chunk_shape`, `_resolve_global_prediction_crop`,
  `_resolve_h5_spatial_chunks` into `connectomics/inference/chunk_grid.py`,
  drop the underscore prefix.
- `inference/chunked.py` and `decoding/streamed_chunked.py` import from
  the public module.
- Drop `_validate_chunked_output_contract` if its current
  output-format check duplicates anything elsewhere; otherwise rename to
  `validate_chunked_output_format` and export it.

### PR-15: `scripts/main.py` thin entrypoint

- `runtime/dispatch.py::dispatch(cfg, args)` absorbs the cache-hit / tune /
  test branching logic.
- `scripts/main.py` becomes ≈ 80 lines.

### PR-16: Targeted file splits

The three highest-value:
- `inference/lazy.py` — shared `_run_lazy_sliding` core (already specified
  in plan PR 10).
- `data/augmentation/transforms.py` — four-way split per plan.
- `decoders/segmentation.py` — extract numba CC kernel + `_compute_edt`.

Skip the rest of PR 10 unless review time on those files becomes a problem.

### PR-17: `DecodingConfig.tuning` typing fix

- Define `DecodingTuningConfig` (or reuse `stages.DecodingParameterSpace`);
  retype `DecodingConfig.tuning`.
- Run the strict-config test to find any tutorial drift; migrate.

### PR-18: A3 product decisions

- One issue per A3 item with maintainer sign-off line. Items that get
  approved for removal land in a small follow-up. Items that stay get
  documented as kept-and-supported.

---

## Open questions for the maintainer

1. **`large_decode` / `abiss_large` configs** — are these meant to live as
   first-class top-level Config sections, or are they outside the
   `scripts/main.py --config` contract entirely? The answer drives PR-12.
2. **Decode-only / evaluate-only as a real CLI mode** — was the intent to
   support `python scripts/main.py --mode evaluate --predictions <path>
   --labels <path>`? If not, the PR-4 promise is over-specified and the
   current half-extraction is fine. If yes, PR-13 is required.
3. **A3 items** — is there a desire for a single triage session, or should
   each be handled as it's touched?

These three answers shape whether PR-12, PR-13, and PR-18 are blockers for
calling v3 done.

---

## Verdict

The boundary-first refactor *worked*. The strongest parts of the plan —
runtime extraction, tune split, decoding↛training cleanup, schema split,
strict config, lazy decoder registration, public API trim — landed cleanly
and pass their guardrail tests.

The weakest parts — evaluation behavioral extraction, tutorial-load CI,
file splits — slipped. Two of those (the missing tutorial-load test and the
half-extracted evaluation) are real problems, not stylistic ones. The
tutorial-load test should land this week; the real evaluation extraction
should land before anyone declares PR 4's design intent satisfied.

Net call: **v3 is 75% done.** The architectural skeleton is correct. The
behavioral cleanup (PR 4) and the operational guardrail (PR 0 tutorial-load)
are the work that remains.
