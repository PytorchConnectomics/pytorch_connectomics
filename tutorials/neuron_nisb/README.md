# NISB tutorials (9 nm)

NISB neuron-segmentation experiments on the BANIS basedata (9 nm EM,
6-channel affinity, 416 GT skeletons in `seed101`).

NERL is the validation-selected (or, for the BANIS reference, default)
score on the `seed101` test split — see `dev/nisb/nisb.md` and
`dev/nisb/nisb_v2.md` for the plan and `dev/nisb/banis+_plan.md`
for per-prediction error analysis and the three-track post-processing
roadmap (`banis+_fill.md` / `banis+_merge.md` / `banis+_split.md`).

| YAML | <div style="width:40px">Deep learning</div> |  Decoding | Error&nbsp;correction | NERL |
|---|---|---|---|---|
| `base_banis.yaml` | <div style="width:40px">[BANIS](https://github.com/jasonkena/banis) reproduction (MedNeXt-L/k3, 6-ch affinity, 50k steps) </div> | cc3d| N/A | 43.0% |
| `base_banis+.yaml` | <div style="width:40px">+ML ops (PerChannelBCE<br/> + EMA, erosion=2, 200k steps) </div>|  cc3d | N/A | [60.1%](https://huggingface.co/pytc/tutorial/tree/main/neuron_nisb) |

## Reproduce `base_banis+` (60.1% NERL)

Run the released checkpoint through the val-tuned decode threshold and score
NERL on the `seed101` test split.

### 1. Download the benchmark data (val + test only)

Inference needs only `val/seed100` (decode-threshold tuning) and
`test/seed101` (evaluation) — the large `train/` split is **not** required, so
fetch just those two subfolders:

```bash
DATA=/local/benchmark/dir/base
EP=https://s3.nexus.mpcdf.mpg.de:443
aws s3 sync --endpoint-url $EP --no-sign-request s3://nisb/base/val/  $DATA/val/
aws s3 sync --endpoint-url $EP --no-sign-request s3://nisb/base/test/ $DATA/test/
```

(Drop the `val/`/`test/` suffix — `s3://nisb/base/` — to also pull `train/` for
training from scratch.)

### 2. Download the pretrained checkpoint

```bash
pip install -U huggingface_hub          # provides the `hf` CLI
hf download pytc/tutorial neuron_nisb/base_banis+_seed42.ckpt --local-dir checkpoints
```

### 3. Tune the decode threshold on val, then evaluate on test

A single `tune-test` run grid-sweeps the cc3d threshold on `val/seed100`
(selects `0.66`), applies it to `test/seed101`, decodes, and scores NERL:

```bash
python scripts/main.py --config tutorials/neuron_nisb/base_banis+.yaml \
    --mode tune-test \
    --checkpoint checkpoints/neuron_nisb/base_banis+_seed42.ckpt \
    tune.data.val.path=$DATA/val/seed100/ \
    tune.data.val.skeleton=$DATA/val/seed100/skeleton.pkl \
    test.data.test.path=$DATA/test/seed101/ \
    test.data.test.skeleton=$DATA/test/seed101/skeleton.pkl
```

Expect a val-selected `threshold: 0.66`, then `[seed101] NERL: ~0.596`. Budget
several hours on one GPU + a high-memory node: `tune-test` runs a full-volume
val inference, a 21-point threshold sweep on val, then a test inference + decode.

**Notes**
- On the lab cluster the data already lives at `/projects/weilab/dataset/nisb/base`,
  so you can drop the four path overrides and just run
  `--mode tune-test --checkpoint <ckpt>`.
- `--mode tune` alone stops after the val sweep (prints the best threshold);
  `--mode test` alone decodes at the config default (`0.75`). Use `tune-test`
  to chain val-selection into the test decode — the per-step decode threshold is
  **not** settable via a CLI override, so re-tuning is how you change it.
- Decoding is **mask-free by default**. An optional border/background affinity
  mask (`test.decoding.affinity_mask_path`, `border=32px` / `bg≤30`) recovers
  ~0.5 NERL point (60.1% vs 59.6%) but is specific to the `seed101` volume; see
  the comment in `base_banis.yaml` to enable it.
