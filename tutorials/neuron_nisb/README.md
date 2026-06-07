# NISB tutorials (9 nm)

NISB neuron-segmentation experiments on the BANIS basedata (9 nm EM,
6-channel affinity, 416 GT skeletons in `seed101`).

NERL is the validation-selected (or, for the BANIS reference, default)
score on the `seed101` test split — see `dev/nisb/nisb.md` and
`dev/nisb/nisb_v2.md` for the plan and `dev/nisb/banis+_plan.md`
for per-prediction error analysis and the three-track post-processing
roadmap (`banis+_fill.md` / `banis+_merge.md` / `banis+_split.md`).

| YAML | <div style="width:280px">Deep learning</div> | Error correction | Decoding | NERL |
|---|---|---|---|---|
| `base_banis.yaml` | [BANIS](https://github.com/jasonkena/banis) reproduction (MedNeXt-S/k3, 6-ch affinity, 50k steps) | N/A | cc3d | 24.4% |
| `base_banis+.yaml` | +ML ops (MedNeXt-L/k3, PerChannelBCE + EMA, erosion=2, 200k steps) | N/A | cc3d | 60.1% |
