# Config V2 Plan

## Goal

Make config strict, typed, and minimal. V2 should reject old field names instead
of translating them, and every runtime option should map to one schema field
with one meaning.

Baseline: `config.md` reports a mature stage-aware Hydra/OmegaConf system. V2
keeps that architecture but removes remaining compatibility tolerance.

## Public API

Keep a small public API under `connectomics.config`:

- `Config`
- stage dataclasses: `DefaultConfig`, `TrainConfig`, `TestConfig`, `TuneConfig`
- section dataclasses for `system`, `model`, `data`, `optimization`,
  `monitor`, `inference`, `decoding`, and `evaluation`
- `load_config`
- `save_config`
- `validate_config`
- `resolve_default_profiles`
- `as_plain_dict`
- `cfg_get`

Hardware planning helpers are public only under `connectomics.config.hardware`.
All other helpers are private to `config.pipeline`, `config.schema`, or
`config.hardware`.

## Delete

- Any old `hydra_config` or `hydra_utils` facade if reintroduced.
- Any alias fields kept for old tutorials.
- Any `shared` stage handling.
- Any nested `inference.decoding` or `inference.evaluation` fields.
- Any profile selector accepted outside canonical paths.
- Any runtime fallback that probes a removed field with `getattr`.
- Any scheduler option duplicated both as a top-level field and inside `params`
  unless it is genuinely shared by multiple schedulers.

## Move/Rename

- Top-level stage sections should be canonical:
  - `default.decoding`
  - `test.decoding`
  - `tune.decoding`
  - `default.evaluation`
  - `test.evaluation`
  - `tune.evaluation`
- Keep hardware helpers under `config.hardware`.
- Keep schema-only dataclasses under `config.schema`.
- Keep load, merge, profile, and stage-resolution logic under
  `config.pipeline`.

## Config Contract

Stage allowlists:

| Stage | Allowed sections |
| --- | --- |
| `default` | `system`, `model`, `data`, `optimization`, `monitor`, `inference`, `decoding`, `evaluation` |
| `train` | `system`, `model`, `data`, `optimization`, `monitor` |
| `test` | `system`, `model`, `data`, `inference`, `decoding`, `evaluation` |
| `tune` | `system`, `model`, `data`, `inference`, `decoding`, `evaluation` |

Mode names:
- use `train`, `test`, `tune`;
- if a future CLI exposes `infer`, `decode`, or `evaluate`, they should resolve
  from `test`-compatible sections unless separate dataclasses are needed.

Required v2 fields:
- `inference.decode_after_inference: bool`
- `inference.chunking.output_mode: "decoded" | "raw_prediction"`
- `inference.saved_prediction_path: Optional[str]`
- top-level `decoding` list or decoding stage config
- top-level `evaluation` config

Rejected v1 patterns:
- `inference.decoding`
- `inference.evaluation`
- `shared`
- optimizer step aliases other than `n_steps_per_epoch`
- WandB fields with duplicated `wandb_` prefix
- MONAI arch aliases that duplicate canonical schema fields

## Boundary Rules

- Runtime packages import dataclasses and loader functions, not profile-engine
  internals.
- Config code may validate cross-section consistency, but it must not import
  training, inference, decoding, or evaluation execution code.
- YAML profile application happens before dataclass conversion.
- Stage resolution happens before execution.

## Implementation Order

1. Add or confirm top-level `decoding` and `evaluation` stage schema.
2. Add `inference.decode_after_inference` and
   `inference.chunking.output_mode`.
3. Remove any old aliases and compatibility validators.
4. Update all tutorial YAMLs to v2 fields.
5. Add strict failure tests for removed fields.
6. Run config load tests for every tutorial.
7. Run stage-specific runtime tests.

## Tests

Add or update:
- config rejects `shared`;
- config rejects nested `inference.decoding`;
- config rejects nested `inference.evaluation`;
- config rejects removed optimizer and WandB aliases;
- all tutorials load with v2 schema;
- profile selectors work only at canonical paths;
- stage allowlists reject invalid sections;
- `--debug-config` prints the resolved v2 structure.

## Open Decisions

- Whether `infer`, `decode`, and `evaluate` should become first-class CLI modes
  in v2 or remain subcommands/wrappers around `test` stage config.
- Whether scheduler-specific `ReduceLROnPlateau` fields should move fully into
  `scheduler.params`.
