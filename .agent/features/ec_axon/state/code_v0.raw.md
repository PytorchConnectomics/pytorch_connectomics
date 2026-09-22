Implemented Plan v6 with v6.1 naming and the explicit knob-free YAML contract. No commit was created, and excluded WIP was preserved.

### Files

Created:

- Branch ops: [sections.py](/projects/weilab/weidf/lib/pytorch_connectomics/connectomics/decoding/decoders/branch/sections.py), [linking.py](/projects/weilab/weidf/lib/pytorch_connectomics/connectomics/decoding/decoders/branch/linking.py), [split.py](/projects/weilab/weidf/lib/pytorch_connectomics/connectomics/decoding/decoders/branch/split.py), [merge.py](/projects/weilab/weidf/lib/pytorch_connectomics/connectomics/decoding/decoders/branch/merge.py), and package `__init__.py`.
- Metrics: [completeness.py](/projects/weilab/weidf/lib/pytorch_connectomics/connectomics/metrics/completeness.py), [oracle.py](/projects/weilab/weidf/lib/pytorch_connectomics/connectomics/metrics/oracle.py).
- Tutorial: [axon_decode.yaml](/projects/weilab/weidf/lib/pytorch_connectomics/tutorials/neuron_axon/axon_decode.yaml), [waterz_baseline.yaml](/projects/weilab/weidf/lib/pytorch_connectomics/tutorials/neuron_axon/waterz_baseline.yaml), [README.md](/projects/weilab/weidf/lib/pytorch_connectomics/tutorials/neuron_axon/README.md).
- Six focused test modules covering seed, split, merge, utilities, oracle NERL, and experiment logging.

Modified:

- Canonical utilities, registry/lazy exports, decode graph, waterz baseline, experiment logging.
- Evaluation context/NERL/reporting and decode-only label propagation.
- Graph-aware output naming, including output-chain and injective ordered-input encoding.
- Relevant existing unit tests and obsolete branch templates.

Deleted:

- `connectomics/decoding/decoders/branch_merge.py`
- `connectomics/decoding/decoders/branch_split.py`
- `connectomics/decoding/decoders/axon_tracklet.py`
- `tutorials/axon_decoding/tracklet.yaml`
- The four old `axon_tracklet_*` registrations.

The exhaustive file table is in [code_v0.md](/projects/weilab/weidf/lib/pytorch_connectomics/.agent/features/ec_axon/artifacts/code_v0.md).

### Verification

| Output | Artifact parity | Labels | NERL base / oracle |
|---|---:|---:|---:|
| Naive waterz | 0 differing voxels | 1,223 | 0.6530 / 0.7580 |
| Sections | 0 differing voxels | 236,877 | — |
| Tracklets | 0 differing voxels | 27,836 | 0.8284 / 0.9424 |
| Split | 0 differing voxels | 28,147 | 0.7302 / 0.9631 |
| v2 intermediate | 0 differing voxels | 22,085 | 0.8377 / 0.9541 |
| Final merged | 0 differing voxels | 22,071 | 0.8434 / 0.9525 |

- Final focused suite: **151 passed, 2 skipped**.
- Tutorial validator: **17 canonical configs validated**, 3 declared custom workflows skipped.
- Black, isort, flake8, `git diff --check`: passed.
- Mypy: passed on 18 changed production files. `bbox.py` retains three unrelated pre-existing errors.
- Static scan: no `dev` imports under `connectomics/`.
- `test_v3_guardrails.py`: passed. The requested `test_public_api_snapshot.py` does not exist; current snapshot/import guards are in `test_v3_guardrails.py`.
- Full waterz parity runtime: 811.2 seconds.
- Full staged parity runtime: 4097.5 seconds, so the `<20 min` performance target was not met in this environment. No constants or algorithm order were changed to chase that target.

`branch_merge` cannot subsume `branch_link` without changing behavior. Linking consumes globally unique single-slice sections using the distinct 0.2 spine and 0.3 best-buddy gates; merging applies completion, size/affinity gates, ambiguity margins, and weak-gap logic.

No algorithm or numerical deviations were found. The baseline uses a fixed `naive_waterz` wrapper around chunked `decode_waterz` so its YAML remains knob-free. The later v6.2 knob text present in the on-disk plan was not applied because this request explicitly mandates v6.1 and no YAML tuning parameters.

CCC run: .agent/features/ec_axon
Stage completed: code_v0
Next action: review_v0