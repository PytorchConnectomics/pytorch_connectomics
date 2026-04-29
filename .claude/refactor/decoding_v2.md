# Decoding V2 Plan

## Goal

Make decoding a standalone stage that consumes raw prediction artifacts or
arrays and writes segmentation artifacts. It should not construct models, load
checkpoints, run sliding-window inference, or compute final evaluation reports.

Baseline: `decoding.md` reports the decoder registry and implementations are
mostly clean. V2 focuses on stage separation and artifact contracts.

## Public API

Keep:

- `Decoder`
- `DecoderOutput`
- `register_decoder`
- `get_decoder`
- `list_decoders`
- `decode_prediction`
- `run_decoding_stage`
- decoder implementations for affinity, watershed, synapse, and ABISS paths
- decoding parameter tuning helpers if they operate only on saved predictions
  and labels

## Delete

- Any decoder registration call duplicated outside package initialization.
- Any decode path that assumes model inference is happening in the same call.
- Any final metric reporting that belongs to `evaluation`.
- Any postprocessing module names that conflict with inference output
  postprocessing semantics.
- Any print-based operational output.
- Any old config nesting under `inference`.

## Move/Rename

Canonical responsibilities:

| Concept | V2 owner |
| --- | --- |
| Decoder registry | `connectomics.decoding.registry` |
| Decoder base types | `connectomics.decoding.base` |
| Segmentation decoders | `connectomics.decoding.segmentation` |
| Synapse decoder | `connectomics.decoding.synapse` |
| Decoding postprocess | `connectomics.decoding.postprocess` |
| Decoding stage runner | `connectomics.decoding.stage` |
| Decoding tuning | `connectomics.decoding.tuning` or current tuner module |

If `optuna_tuner.py` remains large, split stage-independent parameter search
from CLI/reporting helpers.

## Config Contract

Top-level `decoding` config owns:
- decoder name;
- decoder kwargs;
- postprocessing kwargs;
- output segmentation artifact path;
- optional tuning search space for decoder/postprocess parameters.

It does not own:
- model checkpoint;
- sliding-window parameters;
- raw input image paths, except for metadata or visualization;
- metric selection.

Decode-only mode:

```yaml
inference:
  saved_prediction_path: /path/to/raw_prediction.h5
decoding:
  - name: decode_affinity_cc
    kwargs:
      threshold: 0.7
      backend: numba
```

V2 may rename this so `decode.input_prediction_path` is canonical. If so,
`inference.saved_prediction_path` should be deleted and decode-only mode should
not depend on the inference section.

## Boundary Rules

- Decoding may import data processing helpers and top-level utils.
- Decoding may import config dataclasses.
- Decoding must not import training or inference managers.
- Decoding must not write metrics reports except optional decoder diagnostics.
- Decoding output artifacts must be accepted by evaluation without special
  decoder-specific handling.

## Implementation Order

1. Decide whether prediction input path lives under `decoding` or `inference`.
2. Add `run_decoding_stage` that accepts a prediction artifact path and decoder
   config.
3. Update combined test path to call `run_decoding_stage`.
4. Remove decode logic from inference/test orchestration.
5. Ensure all decoders consume canonical artifact arrays and metadata.
6. Move final metric calls out to evaluation.
7. Add decode-only tests.

## Tests

Add or update:
- decode-only mode works without model construction;
- decoding consumes a raw prediction artifact from inference;
- decoded segmentation artifact metadata records decoder name and params;
- all registered decoders are discoverable;
- invalid decoder names raise clear errors;
- decoding does not import training or inference execution modules;
- tuning can run from saved predictions.

## Open Decisions

- Whether decode input path should be `decoding.input_prediction_path` instead
  of `inference.saved_prediction_path`.
- Whether decoding postprocessing should remain a subpackage or be folded into
  decoder-specific modules where only one decoder uses it.
