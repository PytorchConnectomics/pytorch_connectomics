# Task

Redesign the naming convention to be of consistent style in Hydra config,
reflecting the hierarchy of the class design.

## Two coupled concerns

### 1. Hydra config naming consistency

The current schema mixes singular nouns (e.g. `inference`, `decoding`),
verbs/booleans (`save_results`, `save_intermediate`), and overloaded fields
(`output_path` carries different meanings in `inference` vs `decoding`). The
recently-flattened `inference.{save_results, output_path, dtype, backend, ...}`
sit alongside structural sub-blocks like `inference.model`, `inference.window`,
and `inference.execution` — flat scalars and nested dataclasses now share the
same namespace.

Goal: pick a consistent style that mirrors the dataclass hierarchy. Decide,
for each scalar field at section root, whether it stays flat or moves under a
named sub-block. Apply the same rule to all four pipeline stages: `inference`,
`decoding`, `evaluation`, `tune`. Resulting YAML should be predictable —
seeing a leaf in one section tells you where the analogous leaf lives in
another section.

### 2. `output_path` derivation per mode

How `output_path` is set today is mode-dependent and unprincipled:

- **Train mode**: a timestamped run dir is built (`outputs/<experiment>/<timestamp>/`)
  and writers descend into `checkpoints/`, `results/`, etc.
- **Test/tune mode**: `runtime/checkpoint_dispatch.configure_checkpoint_output_paths`
  derives the output base from the supplied `--checkpoint` path (it sits next
  to `checkpoints/`) and creates a sibling `results_step=<N>/` folder.
  The input data's identity (volume/dataset name) is **not** used as a
  subfolder; instead it is concatenated into the output filename
  (e.g. `img_x1_ch0-1-2_ckpt-step=00200000_decoding_affinity_cc_numba-0-0.75.h5`).

Goal:

1. **Train mode**: keep timestamp-rooted output construction. Document its
   exact derivation rule.
2. **Tune / test mode**: derive `output_path` from the input checkpoint
   directory. Build subfolders by checkpoint step **and** by input data
   name, instead of flattening data identity into the filename. Filenames
   should carry only what varies within a single (checkpoint, dataset) run
   (TTA pass index, channel selection, decode params, optional user suffix).
3. The input data name should appear in the path as a directory, not in the
   filename. That makes per-volume artifacts groupable for downstream tooling
   (decoded volume, evaluation metrics file, error analysis subfolder, etc.).

## Deliverables

A single CCC plan + code that:

- Specifies the new naming convention in writing (rules, examples, before/after
  table).
- Migrates the schema dataclasses, runtime path resolvers, and YAML tutorials.
- Keeps all stages aligned (one rule applied uniformly).
- Updates strict-config rejection so legacy keys raise with a clear redirect.
- Passes `tests/unit/test_v3_guardrails.py`, `test_hydra_config.py`,
  `scripts/validate_tutorial_configs.py`, and the full unit suite.

## Constraints

- V3 contract: one canonical owner per concept, no compat shim, strict-key
  raise on unknown YAML keys.
- Do not break the current `inference.save_results / decoding.save_results /
  decoding.save_intermediate` semantics from the previous refactor (just
  audit their names against the new convention).
- Per-checkpoint output directories must remain stable across reruns of the
  same checkpoint so downstream analysis can refer to them by path.
