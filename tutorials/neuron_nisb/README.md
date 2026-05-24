# NISB tutorials (9 nm)

NISB neuron-segmentation experiments on the BANIS basedata (9 nm EM,
6-channel affinity, 416 GT skeletons in `seed101`).

NERL is the validation-selected (or, for the BANIS reference, default)
score on the `seed101` test split — see `dev/nisb/nisb.md` and
`dev/nisb/nisb_v2.md` for the plan and `dev/nisb/v3_erosion2_err_analysis.md`
for per-prediction error analysis.

| YAML | <div style="width:40px">Deep learning</div> |  Decoding | Error&nbsp;correction | NERL |
|---|---|---|---|---|
| `base_banis.yaml` | <div style="width:40px">[BANIS](https://github.com/jasonkena/banis) reproduction (MedNeXt-S/k3, 6-ch affinity, 50k steps) </div> | cc3d| N/A | 24.4% |
| `base_banis+.yaml` | <div style="width:40px">+ML ops (MedNeXt-L/k3, PerChannelBCE<br/> + EMA, erosion=2, 200k steps) </div>|  cc3d | N/A |60.1% |
