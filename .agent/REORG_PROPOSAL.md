# `.agent/` reorganization proposal

Survey: 58 markdown files, ~12k lines. Most refactor docs are historical
(v1/v2 plans superseded by v3). External references and guides are mostly
fresh.

This is a proposal — nothing has been deleted yet.

## Recommend remove (21 files, ~5k lines)

### 1. v1 refactor reports — superseded by v2 plans + v3 implementation
All dated 2026-03-07; component-level state-of-the-system reports written
before the v2 clean-break refactor. Now obsolete since the v3 implementation
audit ships final structure.

- `refactor/config.md`
- `refactor/data.md`
- `refactor/decoding.md`
- `refactor/inference.md`
- `refactor/metrics.md`
- `refactor/models.md`
- `refactor/training.md`
- `refactor/utils.md`

### 2. v2 plans — superseded by v3
All dated 2026-04-29; v2 was a clean-break plan that was further refined
into v3. The v3 audit (`v3_claude_updated_implementation_check.md`)
documents what actually landed.

- `refactor/v2_codex.md` (v2 contract overview)
- `refactor/config_v2.md`
- `refactor/data_v2.md`
- `refactor/decoding_v2.md`
- `refactor/evaluation_v2.md`
- `refactor/inference_v2.md`
- `refactor/metrics_v2.md`
- `refactor/models_v2.md`
- `refactor/training_v2.md`
- `refactor/utils_v2.md`

### 3. v3 plan iterations — keep only the final
- `refactor/v3_claude.md` (initial draft, 2026-04-30) — superseded
- `refactor/v3_claude_feedback.md` (review of the initial draft) —
  feedback already folded into `v3_claude_updated.md`

### 4. inference-decoding split design — subsumed by v3
- `refactor/inference-decoding-split.md` (2026-04-29) — the split
  actually landed in v3; the v3 audit covers it.

### Keep from `refactor/` (active references):
- `refactor/v3_claude_updated.md` — current architecture contract,
  cited from `CLAUDE.md`.
- `refactor/v3_claude_updated_implementation_check.md` — implementation
  audit, cited from `CLAUDE.md`.
- `refactor/v3_claude_updated_implementation_check_rebuttal.md` —
  audit corrections, cited from `CLAUDE.md`.
- `refactor/loss_mask.md` (2026-05-06) — recent landed PR summary.
- `refactor/config_inference.md` (2026-05-06) — recent landed PR summary.
- `refactor/config_inference_feedback.md` (2026-05-06) — review of above.

## Recommend merge

### `reference/waterz.md` + `reference/waterz_dw.md`
The DW fork only differs in packaging and region-graph helpers (per
`waterz_dw.md`'s own opening line). Merge into a single
`reference/waterz.md` with a "fork variants" section.

### `reference/zwatershed.md` + `reference/zwatershed_dw.md`
Currently 90% the same content (same opening paragraph). Merge.

### `banis/` directory + `reference/banis.md`
`banis/affinity.md` and `banis/inference.md` are deep-dive comparisons
against `lib/banis`. `reference/banis.md` is the high-level overview.
Move both deep-dives under `reference/banis/` to consolidate BANIS
domain knowledge in one place. (Or keep top-level `banis/` if that's
the established convention — the redundancy cost is small.)

## Stale references in non-doc files

After the `.claude → .agent` rename, three external references point
to files that don't exist (these were stale before the rename too):

- `setup.py:83` — `.agent/MEDNEXT.md` does not exist. The MedNeXt info
  lives at `.agent/integrations/mednext.md`. Update or drop the comment.
- `QUICKSTART.md:166` — `[.agent/CLAUDE.md]` does not exist. The
  developer guide is at the repo root (`./CLAUDE.md`). Fix link.
- `tutorials/neuron_liconn_mit_x2.yaml:58` — `.agent/repos_other/ABISS_USAGE_SUMMARY.md`
  does not exist. The directory `repos_other/` was never created. Drop
  or replace with a real path.
- `connectomics/models/architectures/mednext_models.py` (4 refs to
  `.agent/MEDNEXT.md`) — same as `setup.py`. Update to
  `.agent/integrations/mednext.md`.

## Proposed final structure

Keep most current top-level dirs; drop historical refactor docs.

```
.agent/
├── README.md                     # NEW: index of what's here, brief
├── architecture/                 # NEW: active architecture contracts
│   ├── v3_plan.md                # ← refactor/v3_claude_updated.md
│   ├── v3_audit.md               # ← refactor/v3_claude_updated_implementation_check.md
│   └── v3_audit_rebuttal.md      # ← refactor/v3_claude_updated_implementation_check_rebuttal.md
├── changes/                      # NEW: rolling landed-PR summaries + reviews
│   ├── loss_mask.md
│   ├── config_inference.md
│   ├── config_inference_feedback.md
│   ├── cc_cupy.md
│   ├── cc_cupy_feedback.md
│   ├── cc_affinity_cpu_mem.md
│   ├── cc_affinity_cpu_mem_feedback.md
│   └── mednext_multi_head.md     # ← feature/mednext_multi_head.md (if landed)
├── guides/                       # unchanged
│   ├── augmentation.md
│   ├── finetuning.md
│   ├── inference.md
│   ├── multi_task.md
│   ├── nan_debugging.md
│   ├── train_val_split.md
│   └── training_speed.md
├── integrations/                 # unchanged
│   ├── cellmap.md
│   ├── mednext.md
│   ├── nnunet.md
│   └── optuna.md
├── reference/                    # external library / domain refs
│   ├── banis/                    # ← merged banis/{affinity,inference}.md + reference/banis.md
│   │   ├── overview.md
│   │   ├── affinity.md
│   │   └── inference.md
│   ├── autoresearch.md
│   ├── deepem.md
│   ├── em_pipeline.md
│   ├── em_util.md
│   ├── emvision.md
│   ├── NEURD.md
│   ├── pytc-deploy.md
│   ├── seung-lab.md
│   ├── snemi_old.md              # only if still useful as post-proc reference
│   ├── waterz.md                 # ← merged waterz.md + waterz_dw.md
│   └── zwatershed.md             # ← merged zwatershed.md + zwatershed_dw.md
└── benchmark/
    └── SNEMI.md
```

Total after re-org: **~32 files** (down from 58), **~7k lines** (down from
~12k). Architecture contract is concentrated under `architecture/`; rolling
PR summaries live under `changes/` (matches the `pr-summary-writer` skill's
default output home); reference docs are deduplicated.

## Two execution paths

### A. Conservative (recommended)
Move historical files to `.agent/archive/` instead of deleting:

```
.agent/archive/
├── refactor_v1/         # 8 v1 component reports
├── refactor_v2/         # 10 v2 plans
└── v3_drafts/           # v3_claude.md, v3_claude_feedback.md, inference-decoding-split.md
```

Pros: nothing lost. Easy to revert.
Cons: still in the repo, just out of the way.

### B. Aggressive
`git rm` the 21 historical files. Recover via git history if needed.

Pros: clean tree.
Cons: harder to spot historical context without `git log`.

## What to do next

1. **Decide A vs B.** Default: A (archive).
2. **Confirm `mednext_multi_head.md` status** — has multi-head landed?
   If yes, move under `changes/`. If no, keep under `proposals/` (new
   dir).
3. **Confirm `snemi_old.md` is still actively referenced** — if it's a
   dead pointer to `lib/snemi_old/`, drop it.
4. **Fix the four stale `.agent/MEDNEXT.md` / `.agent/CLAUDE.md` /
   `.agent/repos_other/...` references** in the non-doc files listed
   above.
5. Apply the moves with `git mv` to preserve history.
6. Optional: add `.agent/README.md` as a top-level index pointing to
   `architecture/`, `changes/`, `guides/`, `reference/`.

After confirming choices, the moves can be executed in one batch. No
external code paths read from `.agent/` (verified by grep), so this is
text-and-docs-only.
