# Inference V2 Plan

## Goal

Make inference responsible only for model prediction and raw prediction
artifacts. It should not decode, evaluate, or contain duplicate postprocessing
paths.

Baseline: `inference.md` removed dead code and folded output postprocessing
into `output.py`. `inference-decoding-split.md` defines the next structural
step: raw prediction artifacts first, decoding second.

## Public API

Keep a small stage API:

- `build_sliding_inferer`
- `InferenceManager`
- `run_prediction_inference`
- `run_chunked_prediction_inference`
- `write_prediction_artifact`
- `read_prediction_artifact`
- `is_2d_inference_mode`

The exact names can change during implementation, but the public API should
separate prediction from decoding.

## Delete

- Any inference code path that immediately decodes unless it is a clearly named
  compatibility-free convenience wrapper.
- Any streamed chunk decode/stitch path hidden behind generic inference names.
- Any ghost config field reads.
- Any output channel-selection logic based on removed config fields.
- Any second application of transpose or activation outside the declared stage.
- Any debug-only artifact analysis module not used by v2.

## Move/Rename

Canonical responsibilities:

| Concept | V2 owner |
| --- | --- |
| Sliding-window prediction | `connectomics.inference.sliding` |
| TTA prediction | `connectomics.inference.tta` |
| Inference orchestration | `connectomics.inference.manager` |
| Raw artifact IO | `connectomics.inference.artifact` or `connectomics.inference.output` |
| Decode-after-inference wrapper | runtime/CLI layer, not core inference |

If `output.py` grows too broad, split raw artifact IO into `artifact.py` and
leave image-format save helpers in `output.py`.

## Config Contract

Inference config owns:
- checkpoint/model loading options;
- input volume selection;
- sliding-window settings;
- TTA settings;
- raw prediction artifact path/layout;
- `decode_after_inference`;
- `chunking.output_mode`.

Inference config does not own:
- decoder list and thresholds;
- evaluation metric list;
- segmentation artifact metrics;
- tuning search spaces.

Artifact layout:
- one volume prediction uses `(C, Z, Y, X)`;
- channel order is explicit metadata;
- crop, transpose, activation, value scale, checkpoint path, and model output
  identity are stored as HDF5 attrs or sidecar metadata.

## Boundary Rules

- Inference may import config, data IO, models, and utils.
- Inference must not import training Lightning modules except through a model
  loading abstraction if unavoidable.
- Inference must not import decoding pipeline execution.
- Inference must not import evaluation.
- Chunked inference writes raw chunks to a raw artifact before whole-volume
  decoding unless `chunking.output_mode == "decoded"` is explicitly selected.

## Implementation Order

1. Define prediction artifact metadata contract.
2. Add artifact read/write helpers.
3. Add `run_prediction_inference` for non-chunked raw output.
4. Add `run_chunked_prediction_inference` for chunked raw output.
5. Route test runtime through raw prediction output when
   `chunking.output_mode == "raw_prediction"`.
6. Keep streamed decoded chunk mode only under an explicit name.
7. Remove any inference imports of decoding/evaluation execution.
8. Add tests comparing non-chunked and chunked raw artifact layout.

## Tests

Add or update:
- raw artifact shape is `(C, Z, Y, X)`;
- artifact metadata contains required fields;
- `decode_after_inference=false` stops after raw prediction writing;
- `decode_after_inference=true` delegates to decoding stage after writing;
- chunked raw output writes one coherent file-backed artifact;
- streamed decoded mode is selected only with `output_mode=decoded`;
- no ghost config reads remain.

## Open Decisions

- Whether artifact metadata should be HDF5 attrs only or a sidecar JSON/YAML
  file for easier inspection.
- Whether existing `write_outputs` should be renamed to make raw prediction vs
  final artifact semantics unambiguous.
