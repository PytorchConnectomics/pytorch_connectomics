# Plan v2

## Summary

Incremental delta over `plan_v1.md` that fixes the three concrete issues
in `plan_v1_review.md` and answers its two open questions. The overall
design is unchanged: storage/load-prefix naming rule (Section A of v1),
per-volume subdir layout in test/tune (Section F of v1),
deterministic stem resolution (Section D of v1), and `TuneOutputConfig`
deletion (Section B of v1).

This document supersedes `plan_v1.md` only on the points listed in
"Changes Since Previous Plan Version"; everything else carries over
verbatim. The code stage should treat the union as the binding contract.

## Scope

Identical to `plan_v1.md` Section "Scope". Not restated.

## Proposed Changes

### A. Cached-prediction filename: single canonical prefix `raw_`

All cached/saved model-prediction artifacts — whether produced by the
sliding-window inferer, by chunked inference, or by the cached
prediction reuse path in tune mode — use the **same** filename
prefix: `raw_`. The TTA pass count is encoded inside the filename via
`_x{n}` (already in plan_v1 Section E). There is no `tta_` prefix.

| Artifact                                   | Filename                              |
|---|---|
| Saved raw prediction (whole-volume)        | `raw_x{n}{head}{ch}.h5`               |
| Saved raw prediction (chunked)             | `raw_x{n}{head}{ch}_chunked-raw_cs<...>.h5` |
| Tune-mode cached intermediate prediction   | `raw_x{n}{head}{ch}.h5` (under `predictions/<volume_stem>/`) |

This **replaces** the `tta_x1_ch0-1-2.h5` example in plan_v1 Section F's
tune directory layout. The corrected tune layout becomes:

```
<output_base>/tuning_step=<NNN>/
  predictions/
    <volume_stem>/
      raw_x{n}{head}{ch}.h5            # was tta_x1_...; same artifact as test mode
  study.db
  best_params.yaml
  trials/
    trial_NNN/
      <volume_stem>/
        decoded_x{n}{head}{ch}_<decoder>_<kwargs>.h5
```

**Code-side renames** flowing from this:

- `connectomics/runtime/output_naming.py:tta_cache_suffix` →
  `raw_cache_suffix` (the function only ever computed the cached
  prediction suffix; the legacy "tta_" name is misleading after the
  unification). Internal consumers: `intermediate_prediction_cache_suffix`,
  `is_tta_cache_suffix` (rename to `is_raw_cache_suffix`),
  `intermediate_prediction_cache_suffix_candidates`.
- `inference.tta_result_path` → `inference.load_tta_path` from v1 is
  retained because it explicitly loads a TTA-aggregated artifact (the
  *value* sometimes points at a file the user produced via TTA outside
  the pipeline). The stored-on-disk filename uses `raw_` because it's
  the same storage class; the load knob's name keeps `tta_` to signal
  "the file you point at is expected to be a TTA result." If reviewer
  prefers full unification (`inference.load_raw_path`), accept that
  alternative — flagged below as Q.

### B. `data.{val,test}.name` lives on the shared `DataInputConfig`

Plan v1 Section C named non-existent `TestDataConfig` /
`ValDataConfig`. The actual schema in
`connectomics/config/schema/data.py` uses a single
`DataInputConfig` dataclass (line 204), and `DataConfig` declares
`train`, `val`, `test` as separate `DataInputConfig` instances.

The `name` field is therefore added once, on `DataInputConfig`:

```python
@dataclass
class DataInputConfig:
    ...
    skeleton: Any = None
    skeleton_mask: Any = None
    name: Optional[Union[str, List[str]]] = None  # NEW
```

**Train-time rejection rule**: the `name` field is meaningful only for
val/test/tune output paths. Setting `data.train.name` has no effect on
training (train mode does not write per-volume artifacts). Two policy
options:

- **(a) ignore silently**: the runtime simply doesn't read
  `cfg.data.train.name`. Pros: simplest. Cons: user typos go unnoticed.
- **(b) validator-warns**: `scripts/validate_tutorial_configs.py` and
  the strict-key rejector emit a warning (not error) if `name` appears
  under any `data.train`-rooted path. Pros: surfaces typos. Cons: adds
  one warning category.

**Plan v2 picks (b)** — emit a non-fatal warning (matching the
existing validator's warning channel for advisory keys, e.g.
`tutorials/waterz_decoding_large*.yaml` skip notice). Strict-key
load-time error is reserved for unknown keys, not for known keys used
in inappropriate stages. This answers reviewer's second question.

### C. Deleting `inference.chunks` and `inference.write_mode`: checkpoint deserialization audit

Both fields are confirmed dead by source-grep
(`grep -rn "inference.chunks\|inference, \"chunks\""` returns 0 hits;
same for `write_mode`). Concerns about checkpoint/hparams payloads:

- Lightning checkpoints embed `cfg` as a dataclass via
  `connectomics/runtime/torch_safe_globals.register_torch_safe_globals`
  (called from `connectomics/config/schema/root.py`). Deleting fields
  from `InferenceConfig` means **older checkpoints saved before this
  PR will fail to deserialize** the `chunks` / `write_mode` keys when
  unpickled (PyTorch 2.6+ `weights_only=True` rejects unknown attrs).

- **Mitigation**: the project already has `scripts/checkpoint_conversion.py`
  for re-pickling pre-PR-8 checkpoints under updated schema modules.
  The same script (or a successor) will need to be invoked once on any
  legacy checkpoint that included these fields.

- **Realistic blast radius**: the dead fields default to
  `chunks: Optional[List[int]] = None` and `write_mode: str =
  "single_writer"`, neither of which is set by any current tutorial.
  Existing checkpoints in `outputs/` were already migrated through the
  prior `save_inference→save→flatten` sweep; re-checking them with
  `scripts/checkpoint_conversion.py` after this PR is a documented
  one-shot maintenance step, not an automated migration.

- **Implementation note for code stage**: add a one-line comment in
  `connectomics/config/schema/inference.py` deletion site referencing
  `scripts/checkpoint_conversion.py` and a verification step that
  loads a sample legacy checkpoint after deletion (manual, not CI).

This addresses reviewer finding 3.

### D. Carry-over from plan_v1

All of plan_v1's other sections remain in force, specifically:

- Section A: narrowed `save_*` / `load_*` rule.
- Section B: schema before/after tables (with the v2 correction that
  `name` lives on `DataInputConfig`).
- Section D: deterministic volume-stem resolution chain.
- Section E: filename token policy table — but with the v2 correction
  that **all cached/raw artifacts use the `raw_` prefix uniformly**.
- Section F: per-mode directory layouts (with the corrected tune
  predictions filename in section A above).
- Section G: cache-resolver contract (all-or-nothing partial-hit policy).
- Section H: code-change file list.
- Section I: tests.
- Section J: YAML migration.
- Section K: strict-key rejector entries.

The renames in plan_v1 Section B that need to ripple through the v2
filename change are limited to the output-naming helper renames in
section A above (`tta_cache_suffix` → `raw_cache_suffix`,
`is_tta_cache_suffix` → `is_raw_cache_suffix`).

## Files and Areas

Identical to plan_v1 Section "Files and Areas" with two additions:

- `connectomics/config/schema/data.py` already listed; the v2 update
  pins the change to `DataInputConfig` (line 204), not to invented
  split-specific classes.
- `connectomics/runtime/output_naming.py` already listed; the v2
  update adds the rename of `tta_cache_suffix` /
  `is_tta_cache_suffix` to `raw_cache_suffix` /
  `is_raw_cache_suffix`. Public `__all__` updates accordingly.

`scripts/checkpoint_conversion.py` is referenced in the implementation
note (Section C above). Not modified by this PR; mentioned for the
PR description.

## Verification Plan

Identical to plan_v1's seven-step plan, with two additions:

8. **Cached-prediction filename consistency check** (new — addresses
   review finding 1):
   - Test that `raw_cache_suffix(cfg)` returns a string starting with
     `_raw_` (or `raw_` once the leading underscore policy is settled
     during code-stage; the helper signature returns the suffix portion
     with leading separator as needed by callers).
   - Test that the tune-mode cached predictions directory contains a
     file matching `raw_x*.h5`, not `tta_x*.h5`, after running
     `temporary_tuning_inference_overrides`.
9. **`data.train.name` warning probe** (new — addresses reviewer's
   second question):
   - Run a tutorial with `data.train.name: "foo"` set; assert
     `validate_tutorial_configs.py` exits 0 but writes the advisory
     warning string to stdout.

## Risks and Questions

### Risks

Same as plan_v1 (external tooling breakage, public-API removal of
`TuneOutputConfig`, opt-in `data.{val,test}.name`, decoded-output
`_x{n}` retention).

Additional risk introduced by Section A: any existing on-disk caches
in `outputs/<exp>/<ts>/results_step=<N>/` written under the old
`{volume_stem}_tta_x1_..._prediction.h5` flat naming will not be
recognized as cache hits after the rename. The cache-resolver miss is
safe (it forces a re-run) but wastes compute on first invocation per
existing run dir. Document in PR description.

### Open questions (none blocking)

- **Q (carry-over)**: should `inference.tta_result_path` rename to
  `inference.load_tta_path` (plan v1, kept), or to
  `inference.load_raw_path` for full unification with the on-disk
  `raw_*.h5` filename convention? Plan v2 keeps `load_tta_path` because
  the field documents *expected file content* (a TTA-aggregated
  prediction), not the on-disk filename pattern. Lock-in: keep
  `load_tta_path`.

- **Q (carry-over)**: `decoding.save_suffix` short name vs verbose —
  short, locked.

- **Q (carry-over)**: `decoded_step{idx}_*.h5` elision when N=1 —
  elide, locked.

## Changes Since Previous Plan Version

Address all three findings + two questions in `plan_v1_review.md`:

- **Finding 1 (raw vs tta cached-prediction filename contradiction)**:
  unified to `raw_` prefix everywhere. Section A above corrects
  plan_v1 Section F's tune layout, and renames the output-naming
  helpers `tta_cache_suffix` → `raw_cache_suffix` and
  `is_tta_cache_suffix` → `is_raw_cache_suffix`. Decoded outputs
  continue to use `decoded_` (unchanged).

- **Finding 2 (`TestDataConfig`/`ValDataConfig` don't exist)**:
  corrected. The `name` field is added to the existing
  `DataInputConfig` class at
  `connectomics/config/schema/data.py:204`. No new dataclasses are
  introduced.

- **Finding 3 (deleted-field checkpoint fallout)**: documented. Section
  C above gives the realistic blast radius (no tutorial sets these
  fields), notes `scripts/checkpoint_conversion.py` as the one-shot
  remediation path for any pre-existing checkpoints, and flags a
  manual verification step for the code stage.

- **Reviewer's question 1** (`raw_` for all cached predictions): yes.
  Locked in Section A. `_x{n}` carries TTA pass count regardless of
  whether TTA is enabled (`x1` for non-TTA runs).

- **Reviewer's question 2** (validator policy on `data.train.name`):
  warning, not error. Section B above. Existing advisory-warning
  channel in `validate_tutorial_configs.py` is reused.

No other changes from plan_v1; all of its Sections A, B (with the v2
correction in Section B above), D, E (with the v2 correction in
Section A above), F (with v2 correction), G, H, I, J, K remain in
force.
