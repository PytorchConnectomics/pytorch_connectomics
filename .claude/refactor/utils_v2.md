# Utils V2 Plan

## Goal

Keep `connectomics.utils` small and genuinely cross-domain. Helpers with only
one domain consumer should live beside that consumer.

Baseline: `utils.md` finds boundaries mostly clean, with compatibility shims as
the main issue. V2 enforces the rule by deleting shims and single-domain exports.

## Public API

Keep only utilities used by multiple domains:

- channel slice parsing/helpers
- label overlap helpers
- other future helpers only after at least two domain packages need them

Everything else should move to its owning package.

## Delete

- Any top-level utility with one consumer if that consumer has a natural package
  home.
- Any re-export shim under `data.utils`.
- Any old path utility exposed in two places.
- Any script-only helper that can live in a proper runtime or script-support
  package.

## Move/Rename

Canonical ownership:

| Helper | V2 owner |
| --- | --- |
| Channel slicing | `connectomics.utils.channel_slices` |
| Label overlap matrix | `connectomics.utils.label_overlap` |
| Dataset split/sampling | `connectomics.data.datasets` |
| Lightning path expansion | `connectomics.training.lightning` or config path resolver |
| CLI preflight errors | runtime/CLI package if scripts become importable |
| Data IO helpers | `connectomics.data.io` |
| Decoder instance filtering | `connectomics.decoding` |

## Config Contract

Utilities should not inspect global config trees. Config access helpers are an
exception and should stay under `connectomics.config.pipeline.dict_utils`.

## Boundary Rules

- `connectomics.utils` must not import domain packages.
- Domain packages may import top-level utils.
- Domain-local `utils.py` files are allowed only when all consumers are inside
  that domain.
- Avoid new `core`, `common`, or `misc` packages that recreate a dumping ground.

## Implementation Order

1. Delete `data.utils`.
2. Search for all `utils` imports and classify them by owner.
3. Move single-domain helpers to their owning package.
4. Trim `connectomics.utils.__init__`.
5. Add tests or lint checks for banned import paths.

## Tests

Add or update:
- `connectomics.utils` exports only approved cross-domain helpers;
- `connectomics.data.utils` import path fails;
- no top-level util imports from data, training, inference, decoding, metrics,
  or evaluation;
- channel slicing behavior remains covered;
- label overlap behavior remains covered.

## Open Decisions

- Whether import-boundary checks should remain unit tests or become a small
  static script.
