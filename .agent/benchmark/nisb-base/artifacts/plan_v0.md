# Plan v0

## Summary

Brainstorm and rank ideas to push NISB baseline NERL above the current
24%@50k / 32%@200k achieved by `tutorials/neuron_nisb/base_banis.yaml`
(MedNeXt-S/k3, 6-ch banis affinity, 128^3 patches, plain BCE,
`decode_affinity_cc` @ thr 0.75, no TTA, no erosion).

A survey of `tutorials/neuron_nisb/`, `tutorials/neuron_snemi/`, the
decoder registry, loss registry, and architecture profiles shows that
**most of the user's listed ideas are already scaffolded as variant
configs** — they have not been benchmarked against this specific 24/32%
baseline yet. The plan therefore splits into:

1. **Tier 1 — config-only ablations of existing variants** (highest ROI; no
   new code).
2. **Tier 2 — config combinations not yet present** (still config-only, but
   require composing existing knobs into new YAML files).
3. **Tier 3 — small code additions** (medium cost) that fill gaps the
   current registry does not cover.
4. **Tier 4 — speculative larger changes** (high cost, uncertain payoff).

Each idea below carries: hypothesis, expected NERL delta (low/med/high), cost,
risks. Final selection of which to actually launch is deferred to `code_v0`.

## Scope

In scope:
- Identify and rank ideas to improve NERL on the NISB-base test set
  (`/projects/weilab/dataset/nisb/base/test/seed101/`, evaluated against
  the 9nm/banis-style skeleton).
- Map each idea to either an existing tutorial config, a small overlay,
  or a code change.

Out of scope:
- Changing the dataset, resolution, or evaluation metric.
- Cross-dataset generalization (mitoEM, vesicle, etc.) — those
  tutorials are read only as inspiration for ideas.
- Distillation, semi-supervised, or self-supervised pretraining.
- Hardware/throughput optimization beyond what naturally falls out of
  config choices (e.g. EMA, bigger patch).

The brainstorm explicitly avoids re-proposing variants that already
exist. Where a variant exists but has not been benchmarked vs the
24/32% target, it is listed as a Tier-1 ablation rather than a "new
idea".

## Proposed Changes

The proposals below are **ideas to evaluate**, not edits to land in this
PR. `code_v0` will narrow this down to a concrete experiment matrix.

### Tier 1 — existing variant configs to benchmark (config-only, no code)

These overlays already exist in `tutorials/neuron_nisb/`. They have to
be run against the same NISB-base test seed and NERL has to be
recorded. None of these costs additional code work.

| ID | Config | Hypothesis | Expected NERL Δ | Cost | Risks |
|---|---|---|---|---|---|
| T1.1 | `base_banis_v1.yaml` | `PerChannelBCE` (auto pos_weight, max 10) + EMA fixes long-range channel imbalance and stabilizes near-end-of-cosine. | +2-5 NERL | ~200k step train | EMA decay 0.999 may need tuning; pos_weight cap may be too low. |
| T1.2 | `base_banis_v2.yaml` | Multi-head split of short-range (ch 0:3) vs long-range (ch 3:6) with uncertainty loss balancing. Long-range head should not dilute short-range gradient. | +1-4 NERL on top of v1 | ~200k step train | Uncertainty balancing converges slowly; primary head only short-range — may lose info from long-range at decode. |
| T1.3 | `base_banis_erosion2.yaml` | Eroding instance boundaries (kernel=2) before affinity generation suppresses the hardest near-boundary voxels and reduces false merges at decode. | +3-8 NERL | ~200k step train, batch_size=2 (slower) | Erosion=2 eats thin processes; could hurt small-object recall. |
| T1.4 | `base_banis_v1_erosion2.yaml` | Combine T1.1 + T1.3. | additive of T1.1 + T1.3 | same as v1 + erosion | same as both. |
| T1.5 | `base_banis_v2_erosion2.yaml` | Combine T1.2 + T1.3. | additive of T1.2 + T1.3 | same as v2 + erosion | same as both. |

Tier-1 picks: at minimum run **v1, v1_erosion2, v2_erosion2** — these
cover (a) loss/EMA, (b) erosion, (c) both + multi-head. v2_erosion2 is
likely the strongest of the existing 6-ch banis variants.

### Tier 2 — new combinations of existing config knobs

These are YAML overlays that the inventory does NOT yet contain but
require zero code work because every primitive is already registered.

| ID | Idea | Hypothesis | Expected Δ | Cost | Risks |
|---|---|---|---|---|---|
| T2.1 | **MedNeXt-B / k=5 base overlay** | Bridge S→L. The 9nm aff9 family uses L; base uses S. B is the typical sweet spot for 200k-step budgets, and k=5 gives larger receptive field for r=10 affinity edges. | +2-5 NERL | 200k step train, ~1.5x slower than S/k3; check VRAM at batch=4 | Bigger model may underfit at 200k; may need to drop batch to 2. |
| T2.2 | **`decode_waterz` + threshold tune** on the existing v1/v2 prediction artifacts | Waterz hierarchical agglomeration is generally stronger than vanilla CC for affinity post-processing. SNEMI configs already use it (`aff85_his256` merge fn). Decode-only stage; no retrain. | +2-6 NERL | minutes per tune trial; reuses cached predictions | waterz needs a high+low threshold tuple; tune block already exists for cc threshold but waterz needs its own search. |
| T2.3 | **`decode_abiss` + tune** on the same predictions | abiss is the decoder of choice in `9nm_base_aff9_head3.yaml` and `common.yaml`. Decode-only. | +2-6 NERL | minutes per trial; runs external CLI | dependency on `scripts/run_abiss_single.py` and abiss binary; per-volume CLI cost. |
| T2.4 | **TTA at inference: flip+rot90** | TTA is a free win when the model has not seen all symmetries. SNEMI sdt_multitask uses `flip_axes: all, rotation90_axes: [[1,2]], ensemble_mode: min` for affinities. Reuse the same recipe. | +1-3 NERL | 8x inference cost (flip×rot) but no retrain | sigmoid then min-ensemble may shrink predictions; mean-ensemble is safer first. |
| T2.5 | **Larger patch: 160^3 or 192^3, batch_size=2** | Bigger receptive field helps long-range r=10 affinity edges and reduces tiling artifacts at decode. | +1-4 NERL | ~1.5x train step; halve batch | VRAM; may need grad accum; affinity boundary stats shift. |
| T2.6 | **DiceLoss + PerChannelBCE combo** | DiceLoss is registered but unused on affinities. Dice complements BCE for sparse positive long-range channels. Use weight 1.0 / 0.5 combo. | +1-3 NERL | none | Dice on per-voxel affinity is not standard; may destabilize early training. |
| T2.7 | **Add NERL early-eval at 50k/100k/150k checkpoints** as part of standard run plan (not a config knob, but a launch-script convention) | The 24/32% numbers are at 50k/200k. Sample mid-train to detect over- or under-fitting, decide cosine cap. | n/a (informs other ideas) | minutes per checkpoint | none. |

Tier-2 picks: **T2.1 (model B/k5), T2.2 (waterz on v1 predictions), T2.4
(TTA), T2.6 (Dice+BCE)**.

### Tier 3 — small code changes that fill registry gaps

These are not in the codebase but are well-scoped extensions.

| ID | Idea | Hypothesis | Expected Δ | Cost | Risks |
|---|---|---|---|---|---|
| T3.1 | **Train-time NERL callback** | Currently checkpoints monitor `val_loss_total`. Adding a periodic (e.g. every 10k steps) NERL eval on a 1-volume val skeleton would let `monitor.checkpoint.monitor: val_nerl` pick the actually-best segmentation checkpoint, not the lowest-loss one. | +1-3 NERL by selecting better checkpoint | new Lightning callback in `connectomics/training/lightning/callbacks.py` (~150 LOC) wrapping `connectomics/evaluation/nerl.py`; need decoder + skeleton scoring on a small val crop | slow eval cost; need to gate behind `monitor.nerl.enabled`. |
| T3.2 | **Per-channel pos_weight cap > 10** in `PerChannelBCEWithLogitsLoss` | r=10 long-range affinity has very low positive rate (<2%); cap of 10 saturates the gradient signal. Allow per-channel cap or cap=None. | +1-2 NERL | small loss kwarg change | over-large pos_weight can blow up loss scale at fp16; needs gradient clip headroom. |
| T3.3 | **Mutex Watershed decoder** (graph-based affinity decoder; SOTA on EM) | `decode_mutex_watershed` (e.g. via `affogato` or in-house) is competitive with waterz/abiss without needing external CLI. | +2-5 NERL | new decoder file + register; ~300 LOC + dep | adds new dependency; slower decode than CC. |
| T3.4 | **MALIS-style structured affinity loss** | MALIS penalizes the highest-error edge in a maximin path; aligns loss with downstream agglomeration objective. | +2-5 NERL | non-trivial; ~400 LOC + numpy/cython kernel | slow per-step (seconds at 128^3); bug-prone. |
| T3.5 | **Distance-transform auxiliary head** on top of existing v2 | SDT head (à la `neuron_snemi_sdt_multitask.yaml`) gives a complementary signal that helps separate touching neurons in the seeded watershed step. | +2-5 NERL | overlay-only IF target generator already supports `name: skeleton_aware_edt` for NISB labels — needs verification. | SDT target may not be defined for non-skeletonized labels. |

Tier-3 picks: **T3.1 (NERL callback) — highest leverage**; T3.2 cheap;
T3.3/T3.4 deferred unless Tier 1/2 fall short.

### Tier 4 — speculative

| ID | Idea | Notes |
|---|---|---|
| T4.1 | Self-supervised pretraining on NISB train zarr (MAE or BYOL on EM patches). | Out of scope but worth flagging if Tier 1-3 plateau below 50%. |
| T4.2 | Cascade: train coarse → refine. | Not supported by current registry. |
| T4.3 | Replace `MedNeXt` with `nnunet_pretrained` baseline. | Cross-arch sanity check; slow to set up. |

### Cross-cutting observation

The single highest-leverage missing piece is **train-time NERL
tracking** (T3.1). Without it, every Tier-1 / Tier-2 ablation has to
rerun full inference + decode + skeleton scoring offline to be
compared, which makes the existing variant zoo
(`v1`, `v2`, `erosion2`, `aff9`, `sdt_multitask`) impractical to
ablate properly. Land T3.1 first; everything else gets cheaper.

### Recommended ranking for `code_v0` to choose from

1. **T3.1** — NERL callback (enables fast comparison).
2. **T1.4** — `base_banis_v1_erosion2.yaml` (per-channel BCE + EMA + erosion).
3. **T2.2** — waterz decode tune on the same predictions (no retrain).
4. **T2.1** — MedNeXt-B/k5 overlay.
5. **T2.4** — TTA flip+rot90 at inference.
6. **T1.5** — `base_banis_v2_erosion2.yaml` (multi-head + erosion).
7. **T2.6** — Dice+BCE on affinity.
8. **T3.5** — SDT auxiliary head (only if T2.6/T1.5 plateau).

## Files and Areas

Read-only references (no edits at planning stage):

- `tutorials/neuron_nisb/base_banis.yaml` — current baseline.
- `tutorials/neuron_nisb/base_banis_v{1,2}.yaml`,
  `base_banis_erosion2.yaml`, `base_banis_v{1,2}_erosion2.yaml` — Tier-1
  variants.
- `tutorials/neuron_nisb/9nm_base_aff9*.yaml`,
  `9nm_base_mednext_l_sdt_multitask.yaml` — inspiration for Tier 2/3.
- `tutorials/neuron_snemi/neuron_snemi_sdt_multitask.yaml` — TTA recipe
  (T2.4) and SDT multi-task pattern (T3.5).
- `connectomics/config/profiles/arch_profiles.yaml` — `mednext_b`, `mednext_l` profile
  names for T2.1.
- `connectomics/decoding/decoders/segmentation.py`, `waterz.py`,
  `abiss.py` — decoder kwargs surface for T2.2 / T2.3.
- `connectomics/models/losses/build.py`, `losses.py` — for
  `PerChannelBCEWithLogitsLoss` knobs (T3.2) and DiceLoss surface (T2.6).
- `connectomics/training/lightning/callbacks.py` — target file for the
  T3.1 NERL callback (do not edit at planning stage).
- `connectomics/evaluation/nerl.py` — current NERL scorer; T3.1 wraps it.

Areas that **would** be edited if a Tier-3 idea is selected at
`code_v0`:

- T3.1: new file `connectomics/training/lightning/callbacks_nerl.py` or
  extension of existing `callbacks.py`; new dataclass field in
  `connectomics/config/schema/monitor.py` (e.g. `monitor.nerl.enabled`,
  `every_n_steps`, `val_skeleton_path`).
- T3.2: small kwarg change in `losses.py::PerChannelBCEWithLogitsLoss`.
- T3.3 / T3.4: new modules under `connectomics/decoding/decoders/` or
  `connectomics/models/losses/`.

## Verification Plan

Plan is itself unverifiable except by review (this is a brainstorm).
The verification steps below apply to each idea **once it is selected
and implemented in `code_v0`**:

1. **Baseline reproduction**: re-run `tutorials/neuron_nisb/base_banis.yaml`
   to 50k and 200k, confirm NERL ≈ 24% / 32% on the test seed (within
   ±1% noise). Without this anchor, downstream Δ measurements are
   meaningless.
2. **Single-axis ablation**: each Tier-1/Tier-2 idea is run as a
   single-config-overlay change vs the freshly-reproduced baseline.
   Record NERL at 50k and at 200k. Two checkpoints prevent confusing
   "trains slower" with "actually worse".
3. **Tier-3 changes**: must come with a unit test (loss-tier and
   callback-tier) and must not regress
   `tests/unit/test_v3_guardrails.py`,
   `tests/unit/test_v2_boundaries.py`,
   `tests/unit/test_public_api_snapshot.py`.
4. **Decode-only ideas (T2.2, T2.3, T2.4)**: re-use cached prediction
   artifact; verify only NERL changes, prediction file unchanged.
5. **Comparison table** in `.agent/benchmark/nisb-base/results.md` (to
   be created in `code_v0`): config name | NERL@50k | NERL@200k |
   train-step time | decode time | notes.

Concrete commands `code_v0` should document:

```bash
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc

# baseline reproduction
python scripts/main.py --config tutorials/neuron_nisb/base_banis.yaml

# tier-1 ablations
python scripts/main.py --config tutorials/neuron_nisb/base_banis_v1_erosion2.yaml

# decode-only tune (after a train run produces a prediction artifact)
python scripts/main.py --config tutorials/neuron_nisb/base_banis.yaml \
    --mode tune \
    decoding.steps[0].name=decode_waterz \
    decoding.steps[0].kwargs.merge_function=aff85_his256
```

## Risks and Questions

- **Q1**: Is the NERL 24/32% baseline the EMA / non-EMA / best-loss
  checkpoint? `base_banis.yaml` has `ema.enabled: false`; `v1` enables
  it. If the baseline used a different evaluation protocol, the
  comparison may be unfair.
- **Q2**: Was the 200k baseline run with the cosine `t_max=200000` (i.e.
  ending at lr=0)? If yes, an extra-long run (e.g. 400k) would need a
  new schedule, not just `--max_steps 400000`.
- **Q3**: Does `decode_affinity_cc` use `affinity_mode: banis`
  (edge_offset=0) at the same threshold for all baseline measurements?
  Threshold tuning is part of the existing `tune` block — was the 24/32%
  reported with thr=0.75 or with the tuned threshold?
- **Q4**: Is `9nm_base_aff9.yaml` actually applicable to this benchmark?
  Its data layout is ZYX 32x256x256 vs the base XYZ 128^3, and the
  resolution metadata is `[20, 9, 9]` vs base's `[9, 9, 20]`. This is
  not directly comparable; treat it as inspiration only.
- **Q5**: Are NISB labels skeletonized in a form that
  `name: skeleton_aware_edt` (used by the SDT configs) can consume?
  Required for T3.5 to be config-only.
- **Q6**: Compute budget — how many independent 200k-step runs fit in
  the available GPU budget? This caps how many of T1.x / T2.x can run
  in parallel and dictates whether T3.1 is a prerequisite or a
  nice-to-have.
- **R1**: Erosion=2 may regress NERL on thin-process recall in NISB
  (which has more axons than mitoEM). Worth measuring NERL split by
  object-size bucket if possible.
- **R2**: Adding TTA + waterz at the same time mixes inference and
  decode changes; results have to be ablated separately to attribute
  NERL gains.
- **R3**: T3.1 (NERL callback) requires running decode + skeleton
  scoring inside the Lightning val loop — this can be slow and may
  starve GPU. Need a cheap-mode (small crop, cc decode, every Nk steps).

## Changes Since Previous Plan Version

Initial plan.
