# Models V2 Plan

## Goal

Keep models and losses as construction libraries. They should expose registered
architectures, model builders, and loss metadata, but they should not preserve
legacy wrapper names just because old tutorials reference them.

Baseline: `models.md` reports most cleanup complete. V2 removes the last
compatibility-driven keeps and tightens the public API.

## Public API

Keep:

- `build_model`
- architecture registry functions
- architecture spec/base classes
- model wrappers that implement real behavior
- loss builder
- loss metadata lookup
- custom loss modules that are actively supported

## Delete

- Architecture aliases kept only for old YAMLs.
- Convenience loss factories with no production use.
- Public methods with no callers or tests.
- Hardcoded availability flags for always-available models.
- Duplicate metadata registries.
- Any model wrapper that is pure delegation and not referenced by v2 tutorials.

## Move/Rename

Canonical responsibilities:

| Concept | V2 owner |
| --- | --- |
| Model factory | `connectomics.models.build` |
| Architecture registry | `connectomics.models.architectures.registry` |
| Architecture base | `connectomics.models.architectures.base` |
| MONAI models | `connectomics.models.architectures.monai_models` |
| MedNeXt models | `connectomics.models.architectures.mednext_models` |
| nnUNet models | `connectomics.models.architectures.nnunet_models` |
| RSUNet | `connectomics.models.architectures.rsunet` |
| Loss factory and metadata | `connectomics.models.losses` |

## Config Contract

Model config should specify one canonical architecture type and one canonical
parameter structure. Old aliases are rejected.

Rules:
- no `*_pretrained` aliases unless they represent behavior not expressible by
  canonical fields;
- model output channels must validate against loss, label transform, decoding,
  activation, and evaluation config;
- deep supervision key contract is `ds_1`, `ds_2`, etc.

## Boundary Rules

- Models may import PyTorch and optional model libraries.
- Models may import config dataclasses or plain config values.
- Models must not import training loops.
- Models must not import inference, decoding, or evaluation.
- Loss metadata should stay independent of Lightning.

## Implementation Order

1. Update v2 tutorials to canonical architecture names.
2. Delete old architecture aliases.
3. Trim `models.__init__` exports to the v2 public API.
4. Add tests that removed aliases fail during config/model build.
5. Keep registry error messages clear and exhaustive.
6. Run model, training, and tutorial load tests.

## Tests

Add or update:
- every v2 architecture profile builds;
- old architecture aliases fail clearly;
- RSUNet deep supervision keys match the loss orchestrator;
- loss metadata covers all supported loss modules;
- unsupported loss signatures are excluded from standard pipeline use;
- no model module imports training, inference, decoding, or evaluation.

## Open Decisions

- Whether `nnunet_2d_pretrained` and `nnunet_3d_pretrained` should be deleted
  in favor of a single canonical nnUNet config with `pretrained: true`.
- Whether loss code should remain under `models.losses` or move to
  `training.losses` with only metadata exposed to config validation.
