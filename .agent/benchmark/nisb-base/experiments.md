# NISB-base Experiment Matrix

Concrete, ranked plan-of-experiments to push NERL above the
`tutorials/neuron_nisb/base_banis.yaml` baseline (24%@50k / 32%@200k).
Addresses minor comments from `plan_v0_review.md`:

- Two lists: **NERL-improvement experiments** vs **enabling
  infrastructure**. The train-time NERL callback is infra; it does not
  itself raise NERL except by selecting better checkpoints.
- SDT auxiliary head reclassified to **config-only** (verified —
  `skeleton_aware_edt` runs on-the-fly from `seg`; no precompute
  required for NISB-base).
- All Dice+BCE / TTA / waterz / abiss kwargs pinned with citations.

## Open Questions for User (block first run until answered)

1. **Q1 — Baseline measurement protocol.** Were the reported NERL
   24%@50k and 32%@200k measured with `decode_affinity_cc threshold=0.75`
   (the YAML default) or with the existing `tune` block's tuned
   threshold (range 0.7–0.95, step 0.05, 4 trials)? All Δ comparisons
   below assume the **same** protocol used for the baseline. If the
   baseline used the tuned threshold, every decode-only Tier-2
   experiment must also tune (otherwise the comparison is unfair).
2. **Q2 — EMA in baseline.** `base_banis.yaml` has `ema.enabled: false`.
   Was the 32%@200k number from the EMA-disabled run, or was EMA
   enabled out-of-band? Affects v1 vs base attribution.
3. **Q3 — GPU budget.** How many independent 200k-step training runs fit
   in the available budget over the next iteration? Decode-only
   experiments cost minutes; training experiments cost ~hours-days. The
   matrix below is ordered to maximize information-per-GPU-hour, but a
   tight budget should pick only rows marked **★ minimum set**.

## Experiment Matrix

Two lists. **All Tier-2 decode-only rows reuse the saved prediction
artifact from a single 200k-step run** (no retraining). Tier-1 and the
selected Tier-2 model-arch rows each cost one training run.

### List A — NERL-improvement experiments (ranked)

| # | ID | Type | What | Expected Δ NERL | Cost | Reuses Pred? |
|---|---|---|---|---|---|---|
| 1 ★ | E1 | train | Reproduce baseline | anchor (target 24/32%) | 200k steps | new |
| 2 ★ | E2 | train | `base_banis_v1_erosion2.yaml` (PerChannelBCE + EMA + erosion=2) | +4 to +10 | 200k steps, batch=2 | new |
| 3 ★ | E3 | decode | E1 predictions + `decode_waterz` (single Optuna trial; waterz batch mode internally sweeps the threshold list) | +2 to +6 vs E1 | one waterz batch decode (~minutes) | E1 |
| 4 | E4 | decode | E2 predictions + `decode_waterz` (single trial, batch threshold sweep) | +2 to +6 vs E2 | same | E2 |
| 5 | E5 | infer | E1 checkpoint + flip+rot90 TTA (mean ensemble) at inference | +1 to +3 vs E1 | re-runs full inference 8× over flip/rot symmetries; produces a new prediction artifact | new (not E1's saved prediction) |
| 6 | E6 | train | `base_banis_v2_erosion2.yaml` (multi-head + erosion) | +1 to +4 vs E2 | 200k steps, batch=2 | new |
| 7 | E7 | train | MedNeXt-B / k=5 overlay (config-only via CLI) | +2 to +5 vs E1 | 200k steps, ~1.5× slower | new |
| 8 | E8 | train | E2 + DiceLoss(weight=0.5) + PerChannelBCE(weight=1.0) | +1 to +3 vs E2 | 200k steps | new |
| 9 | E9 | train | E2 + SDT auxiliary head (`skeleton_aware_edt`, weight=10, tanh) | +2 to +5 vs E2 | 200k steps, slower | new |
| 10 | E10 | decode | E1 + `decode_abiss` (tune) | +2 to +6 vs E1 | minutes; CLI per volume | E1 |

★ = minimum set if GPU budget is tight (E1 + E2 + E3 = one anchor + one
strongest config-only train + one decode-only sweep).

### List B — Enabling infrastructure (does not raise NERL on its own)

| # | ID | Type | What | Why | Cost |
|---|---|---|---|---|---|
| I1 | INF-NERL-CB | code | Train-time NERL Lightning callback (every Nk steps) | Lets `monitor.checkpoint.monitor: val_nerl` pick the actually-best segmentation checkpoint, not lowest-loss; cheap NERL eval on a single small val crop | ~150 LOC, new file under `connectomics/training/lightning/` |
| I2 | INF-PW-CAP | code | Allow `max_pos_weight: inf` / per-channel caps in `PerChannelBCEWithLogitsLoss` | r=10 long-range affinity has very low pos rate; current cap=10 saturates gradient | ~10 LOC change in `connectomics/models/losses/losses.py` |
| I3 | INF-RESULTS | doc | Results table at `.agent/benchmark/nisb-base/results.md` | Track NERL@50k, NERL@200k, decode-time, train-step time per experiment | append-only |

I1 is recommended **before E2/E6/E9** to enable best-checkpoint
selection; otherwise lowest-loss may not coincide with best NERL,
inflating noise on Δ measurements. If GPU budget rules out I1
implementation work this round, run List A using lowest-loss
checkpoints and document that caveat in the results table.

## Pinned kwargs and CLI invocations

### E1 — baseline reproduction

```bash
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
python scripts/main.py --config tutorials/neuron_nisb/base_banis.yaml
```

After train, score with:

```bash
python scripts/main.py --config tutorials/neuron_nisb/base_banis.yaml \
    --mode test --checkpoint outputs/nisb_base_banis/checkpoints/last.ckpt
```

### E2 — v1 + erosion=2

```bash
python scripts/main.py --config tutorials/neuron_nisb/base_banis_v1_erosion2.yaml
```

### E3 — waterz decode tune on E1 predictions

Pinned waterz kwargs (from `connectomics/decoding/decoders/waterz.py:36`
and `connectomics/config/templates/decoding_templates.yaml:11-22`):

- `merge_function: aff85_his256` (85th percentile via 256-bin histogram)
- `aff_threshold: [0.001, 0.999]` (low/high tuple)
- `channel_order: xyz` — base_banis affinity offsets are
  `["1-0-0","0-1-0","0-0-1",...]` over an XYZ-stored zarr; the first 3
  channels are along axis0/axis1/axis2 of the in-memory volume which
  matches waterz's `xyz` mode after its internal transpose. **Verify
  with one trial against `channel_order: zyx` if the first run produces
  obviously degenerate segmentation; this is the most error-prone
  setting in the matrix.**
- `dust_merge: true, dust_merge_size: 800, dust_merge_affinity: 0.3,
  dust_remove_size: 600` (SNEMI canonical values).
- `branch_merge: false` initially (z-anisotropic; revisit if false
  splits across z dominate).

**Important — the Optuna tuner reads the decoder from
`tune.parameter_space.decoding.function_name`, not from
`decoding.steps[0]` (`connectomics/decoding/tuning/optuna_tuner.py:515-521`).
A CLI override of `decoding.steps[0]` alone leaves the tuner running
the base config's `decode_affinity_cc` and tries to pass `thresholds`
into `decode_affinity_cc(threshold, backend, edge_offset, orphan_fill)`
(`connectomics/decoding/decoders/segmentation.py:455-461`), which fails.**

To avoid that trap, E3 uses a small run-scoped overlay that replaces
both `decoding.steps` and the `tune.parameter_space.decoding` block in
one place:

```bash
python scripts/main.py \
    --config .agent/benchmark/nisb-base/configs/base_banis_waterz_tune.yaml \
    --mode tune \
    --checkpoint outputs/nisb_base_banis/checkpoints/last.ckpt
```

The overlay (`.agent/benchmark/nisb-base/configs/base_banis_waterz_tune.yaml`)
sets `tune.parameter_space.decoding.function_name: decode_waterz`,
copies the pinned waterz defaults into `tune.parameter_space.decoding.defaults`,
and adds a single `thresholds` parameter swept over `[0.1, 0.6]` step
0.05. E4 reuses the same overlay but with the E2 checkpoint.

**The overlay sets `tune.n_trials: 1`, not 25.** OmegaConf merges
`parameter_space.{defaults,parameters}` as dicts (no replace
semantics), so the inherited CC `threshold` parameter and
`backend`/`edge_offset` defaults from
`tutorials/neuron_nisb/base_banis.yaml:223-234` survive into the
resolved tune config. `decode_waterz` accepts `**kwargs` and silently
drops unknown args, so trial *correctness* is unaffected — but waterz
batch mode evaluates all thresholds inside a single decode call
(`connectomics/decoding/tuning/optuna_tuner.py:562-583, :1018-1021`),
so additional Optuna trials would only re-sample the inherited
`threshold` and re-run identical waterz batches. A single trial is
sufficient. The chosen best threshold is recorded as the
`best_threshold` user attribute on that trial
(`optuna_tuner.py:1058-1063`); read it from the Optuna study/journal
rather than from `best_params`. Per-threshold ARE values are also
stored as `are_thr_<value>` user attributes — useful for inspecting
the full sweep curve without rerunning.

If at some point the matrix needs to also tune `merge_function` or
`aff_threshold`, that requires disabling the batch shortcut. Out of
scope for this iteration.

The overlay-scope question from `review_v0` is answered: run-scoped
overlays under `.agent/benchmark/nisb-base/configs/` (not
`tutorials/`) are the right form for the highest-risk rows.

### E5 — TTA on E1

Pinned (from `tutorials/neuron_snemi/neuron_snemi_sdt_multitask.yaml`):

- `flip_axes: all` (all spatial axes)
- `rotation90_axes: [[1, 2]]` (XY-plane rot only — z is anisotropic)
- `ensemble_mode: mean` (start with mean; `min` is more conservative
  but can shrink predictions and erase small processes)

```bash
python scripts/main.py --config tutorials/neuron_nisb/base_banis.yaml \
    --mode test \
    --checkpoint outputs/nisb_base_banis/checkpoints/last.ckpt \
    inference.test_time_augmentation.enabled=true \
    inference.test_time_augmentation.flip_axes=all \
    'inference.test_time_augmentation.rotation90_axes=[[1,2]]' \
    inference.test_time_augmentation.ensemble_mode=mean
```

### E7 — MedNeXt-B / k=5

```bash
python scripts/main.py --config tutorials/neuron_nisb/base_banis.yaml \
    default.model.arch.profile=mednext_b \
    default.model.mednext.size=B \
    default.model.mednext.kernel_size=5
```

If VRAM tight at batch=4, drop to batch=2 and double accumulation:

```bash
... default.data.dataloader.batch_size=2 \
    train.optimization.accumulate_grad_batches=2
```

### E8 — Dice + BCE on affinity (config-only)

Pinned MONAI `DiceLoss` kwargs (from `loss_profiles.yaml::loss_binary`):

- `sigmoid: true` (multi-channel binary, not multi-class)
- `to_onehot_y: false` (target already per-channel binary)
- `include_background: true` (MONAI default; affinities have no
  "background" label — every channel is a binary edge)
- `smooth_nr: 1.0e-5, smooth_dr: 1.0e-5`
- `reduction: mean`

Add Dice as a second loss term in `default.model.loss.losses` with
weight 0.5 alongside the existing PerChannelBCE (weight 1.0). When
this row is selected, author a small run-scoped overlay at
`.agent/benchmark/nisb-base/configs/base_banis_v1_erosion2_dicebce.yaml`
(~5-line extension over `_v1_erosion2.yaml`) — not authored here
because E8 is below the minimum set and may not run.

### E9 — SDT auxiliary head on top of E2

Verified config-only (no precompute). Pattern adapted from
`tutorials/neuron_nisb/9nm_base.yaml` (the named-head SDT reference for
NISB; the longer
`9nm_base_mednext_l_sdt_multitask.yaml` cited in the prior version
of this matrix does not exist).

The reviewer's high-severity finding: simply adding an SDT channel to
`base_banis_v1_erosion2.yaml` would let the existing
`PerChannelBCEWithLogitsLoss` (no `pred_slice`) train binary
classification on the SDT regression target. The fix is to slice both
loss terms.

E9 uses the run-scoped overlay
`.agent/benchmark/nisb-base/configs/base_banis_v1_erosion2_sdt.yaml`,
which (relative to `_v1_erosion2`):

- sets `model.out_channels: 7`,
- replaces the loss list with two terms:
  - `PerChannelBCEWithLogitsLoss` with `pred_slice: "0:6"` and
    `target_slice: "0:6"`,
  - `SmoothL1Loss(weight=10.0, kwargs={tanh: true})` with
    `pred_slice: "6:7"` and `target_slice: "6:7"`,
- adds a second `data.label_transform.targets` entry:
  ```yaml
  - name: skeleton_aware_edt
    kwargs: {alpha: 0.8, bg_value: -1.0}
  ```
- keeps inference `select_channel: [0, 1, 2]` so decoding still uses
  only short-range affinities.

Top-level (non-named-head) `pred_slice`/`target_slice` is the same
pattern used in `tutorials/mito_mitolab.yaml`,
`tutorials/fiber_linghu26.yaml`, and
`tutorials/neuron_liconn_mit_x2.yaml`.

The `skeleton_aware_edt` transform falls back to on-the-fly kimimaro
skeletonization when `label_aux` is absent
(`connectomics/data/processing/transforms.py:725-733` and `:926-944`),
so no per-zarr precompute step is required.

```bash
python scripts/main.py \
    --config .agent/benchmark/nisb-base/configs/base_banis_v1_erosion2_sdt.yaml
```

### E10 — abiss decode tune

Same shape as E3, with `decode_abiss` instead of `decode_waterz`. The
same Optuna-tuner caveat applies: set
`tune.parameter_space.decoding.function_name: decode_abiss` (not
`decoding.steps[0].name`) and provide an `ws_merge_threshold` parameter
(`connectomics/decoding/decoders/abiss.py`, kwargs pinned from
`tutorials/neuron_nisb/9nm_base_aff9_head3.yaml`). When this row is
selected, mirror the waterz overlay pattern at
`.agent/benchmark/nisb-base/configs/base_banis_abiss_tune.yaml`. Also
requires the abiss CLI to be on PATH and
`scripts/run_abiss_single.py` to be runnable.

## Recommended sequencing

```text
1.  E1 (baseline reproduction; anchor for all Δ)
2.  E2 (v1_erosion2; strongest config-only train)
3.  E3 (waterz tune on E1) — runs in parallel with #2 once E1 finishes
4.  E4 (waterz tune on E2)
5.  E5 (TTA on E1)
[checkpoint: pause and review NERL table; pick winners]
6.  E6 / E7 / E8 / E9 / E10 selectively, based on what worked in 1-5.
```

**Stop early** if any of E3/E4/E5 already pushes NERL above 40%; that
is enough to declare config-only wins and re-prioritize List B (the
NERL callback) for the next round.

## Acceptance criteria

For each row:
1. Same skeleton scoring on the same test seed
   (`/projects/weilab/dataset/nisb/base/test/seed101/`).
2. Both NERL@50k and NERL@200k recorded (training rows only).
3. Decode-only rows record decoder name + threshold + relevant kwargs
   in `results.md`.
4. Δ NERL reported vs E1 baseline (positive = better).
