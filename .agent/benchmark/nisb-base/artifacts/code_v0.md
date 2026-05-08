# Code v0

## Overview

The task is a brainstorm + ranked plan-of-experiments, not a code
change. The "implementation" for `code_v0` is a single concrete
deliverable file at `.agent/benchmark/nisb-base/experiments.md` that
turns the approved plan into a runnable experiment matrix and resolves
the four minor comments from `plan_v0_review.md`.

No source files under `connectomics/` were edited. No new YAMLs were
authored under `tutorials/` either: every Tier-2 row in the matrix is
expressible as a CLI override on an existing tutorial config, and the
single Tier-1 row that would benefit from a small overlay
(E8 Dice+BCE) is flagged but deliberately not authored — the user's
prompt asked for ideas and a plan, and YAML authoring belongs to the
launch step that picks which experiments to actually run.

## What Changed

- New file `.agent/benchmark/nisb-base/experiments.md` (~190 LOC).
  Splits the plan's mixed ranking into two lists:
  - **List A — NERL-improvement experiments** (10 rows, each with type
    train/decode, expected Δ, cost, prerequisites, exact CLI).
  - **List B — Enabling infrastructure** (3 rows: NERL train-time
    callback, pos_weight cap, results table).
- Each List-A row carries pinned kwargs with file_path:line citations:
  - `decode_waterz`: `merge_function: aff85_his256, channel_order: xyz,
    aff_threshold: [0.001, 0.999], dust_merge_size: 800,
    dust_merge_affinity: 0.3, dust_remove_size: 600`. The
    `channel_order` decision flags one verification step (the only
    error-prone setting in the matrix).
  - TTA: `flip_axes: all, rotation90_axes: [[1,2]], ensemble_mode: mean`.
  - DiceLoss kwargs: `sigmoid: true, smooth_nr: 1e-5, smooth_dr: 1e-5,
    reduction: mean, to_onehot_y: false`.
  - SDT auxiliary head: `name: skeleton_aware_edt, kwargs: {bg_value:
    -1.0, alpha: 0.8}` plus `function: SmoothL1Loss, weight: 10.0,
    kwargs: {tanh: true}` on `target_slice: "6:7"`.
- Verified that SDT for NISB-base is **config-only** (no
  precompute step required). The transform falls back to on-the-fly
  kimimaro skeletonization when `label_aux` is absent.
- Lists three open questions that block first launch:
  baseline-measurement protocol, EMA-in-baseline attribution,
  GPU budget.

## Implementation Details

The matrix is structured for triage under tight GPU budgets:

- A "★ minimum set" (E1 + E2 + E3) covers one anchor + one
  best-guess training win + one decode-only sweep, requiring only
  two training runs.
- Decode-only rows (E3, E4, E5, E10) all reuse the saved prediction
  artifact from E1 or E2; no extra training cost.
- The `code_v0` artifact preserves the user's original framing
  (brainstorm + ranking) — it does not pre-commit to a specific run
  list. The reader can pick rows.

The two scope decisions worth flagging to the reviewer:

1. **No new YAML overlays.** Every config-only row uses CLI overrides
   on the existing `tutorials/neuron_nisb/base_banis*.yaml` files.
   Cleaner than committing 6 near-identical YAMLs that may not all
   land. Only authoring overlays for rows that actually run avoids
   tutorial-config sprawl.
2. **List-B infrastructure (NERL callback, pos_weight cap) is
   recommended but not implemented.** Implementing the callback
   requires a new file in `connectomics/training/lightning/` and a
   schema field in `connectomics/config/schema/monitor.py`; that is a
   real PR's worth of work and would balloon `code_v0` past the
   brainstorm scope. Flagged for the next CCC run if the user agrees.

## Files Changed

| File | Purpose |
|---|---|
| `.agent/benchmark/nisb-base/experiments.md` | New — ranked experiment matrix with pinned kwargs and CLI invocations. |

(`.agent/benchmark/nisb-base/{task.md, run.md, artifacts/plan_v0.md,
artifacts/plan_v0_review.md}` were created by earlier CCC stages, not
by this code stage.)

## Git Baseline

run_start_ref: c82ec629ddac061ffca272eea4c7f702771bb0e9
current_head: c82ec629ddac061ffca272eea4c7f702771bb0e9

The CCC run produced no commits and no edits to tracked files. The
single tracked-file diff at run start (`M connectomics/evaluation/nerl.py`)
is captured in `state/run_start.diff` and is **not** part of this run.
The CCC run folder itself (`.agent/benchmark/nisb-base/`) is untracked
and may be committed by the user separately.

## Verification

What was verified:

- Re-read `connectomics/decoding/decoders/waterz.py:36-180` to confirm
  `channel_order` defaults and the prepare_affinities transpose
  behaviour. Pinned `xyz` for base_banis with a verify-step on the
  first run.
- Re-read `connectomics/models/losses/losses.py:228-310` to confirm
  `PerChannelBCEWithLogitsLoss(auto_pos_weight=True, max_pos_weight=10.0)`
  applies the cap per-channel via `torch.clamp(..., max=...)` and that
  `max_pos_weight: inf` would disable it.
- Confirmed via `connectomics/data/processing/transforms.py:725-944`
  that `skeleton_aware_edt` falls back to kimimaro on-the-fly
  skeletonization when `label_aux` is absent — so the SDT auxiliary
  head requires no zarr precompute for NISB-base.
- Confirmed `tutorials/neuron_snemi/neuron_snemi_sdt_multitask.yaml`
  TTA settings (`flip_axes: all, rotation90_axes: [[1,2]]`) and reused
  them as the canonical TTA recipe for E5.

What was **not** verified (would require a launch):

- That `decode_waterz` runs end-to-end on a base_banis prediction
  artifact at `channel_order: xyz`. Marked as the matrix's primary
  verify-step (E3, first trial).
- That the Hydra CLI override syntax for swapping the `tune` block's
  CC `threshold` parameter into a waterz `thresholds` parameter
  actually parses. The matrix flags this; the fallback is a small
  overlay YAML rather than a heavily-quoted CLI.
- No tests run. No code changed; `tests/unit/test_v3_guardrails.py`
  and friends are unaffected by this CCC run.

## Review Focus

For `a2`'s code review, the most useful focus points are:

1. **Scope call.** Was it correct to author only `experiments.md` and
   no YAML overlays? Or should `code_v0` have authored at least a
   waterz decode-only overlay and an `_dicebce` overlay so that the
   matrix is "click to launch" rather than "click after pasting CLI"?
2. **Pinned kwargs accuracy.** Spot-check the cited file_path:line
   references against the live source. The `channel_order: xyz`
   decision for waterz is the highest-stakes assertion in the
   document.
3. **Ranking sanity.** The "★ minimum set" (E1+E2+E3) is opinionated.
   If a tighter or different minimum set makes more sense (e.g.
   "E1+E5 first because TTA is decode-only and free"), call it out.
4. **Open questions.** Are Q1-Q3 the right blockers, or are there
   other measurement-protocol traps to flag before any run?

## Risks and Unknowns

- The matrix's expected-Δ ranges (e.g. "+4 to +10 for E2") are
  educated guesses, not measurements. The actual deltas may be much
  smaller (or negative) on this specific benchmark; the matrix is a
  prioritization tool, not a forecast.
- The `channel_order` decision for waterz is the single setting most
  likely to silently produce degenerate output. If the reviewer is
  uncertain, drop E3/E4 to a lower priority and verify with a 5-min
  smoke test before scheduling 25 tune trials.
- Train-step time and decode-time numbers in the matrix are
  unmeasured. The "★ minimum set" recommendation assumes E2 fits in a
  similar time budget to E1 (200k steps at batch=2 vs batch=4 — likely
  ~2x slower per step but same step count). Worth verifying with a
  dry-run before committing the full schedule.
- The user has not confirmed Q1-Q3 (baseline protocol, EMA, GPU
  budget). Launching without those answers risks unfair comparisons.

## Changes Since Previous Code Version

Initial implementation.
