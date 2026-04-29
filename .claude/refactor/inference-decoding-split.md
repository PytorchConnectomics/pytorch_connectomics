# Inference / Decoding Split Refactor

## Problem

The current test path mixes three responsibilities:

- deep learning inference,
- prediction artifact storage,
- decoding/postprocessing/evaluation.

This is most visible in chunked inference: `run_chunked_affinity_cc_inference`
predicts a chunk, immediately decodes it, stitches labels, and writes the final
segmentation. That is memory efficient, but it cannot reproduce whole-volume
decoding exactly because connected components are solved per chunk and then
stitched heuristically.

## Target Design

Treat model inference and decoding as separate stages.

1. Model inference writes a raw prediction artifact.
   The artifact is file-backed, chunked, and has a stable layout:
   `(C, Z, Y, X)` for one volume after inference-time crop/channel selection.

2. Decoding consumes a raw prediction artifact and writes a segmentation
   artifact.
   It should not require model construction, checkpoint loading, or GPU setup.

3. The combined test path remains a convenience wrapper.
   It can run inference, then optionally decode the just-written artifact.

4. Evaluation should become its own top-level stage.
   It should not live under `decoding`, because metrics consume decoded
   artifacts and labels regardless of which decoder or cache produced them.
   The config tree now has a dedicated top-level/default/test/tune
   `evaluation` section.

## Config Contract

`inference.decode_after_inference`

- `true`: current convenience behavior; decode after prediction.
- `false`: stop after writing raw predictions.

`inference.chunking.output_mode`

- `decoded`: current streaming chunk decode/stitch behavior.
- `raw_prediction`: stream chunked model predictions into one raw prediction
  HDF5, then optionally run the normal whole-volume decoding path.

Existing decode-only mode remains:

```yaml
inference:
  saved_prediction_path: /path/to/raw_prediction.h5
decoding:
  - name: decode_affinity_cc
    kwargs:
      threshold: 0.7
      backend: numba
      edge_offset: 0
```

## Implementation Plan

1. Add schema fields for `decode_after_inference` and chunked `output_mode`.
2. Split chunked inference code into two entry points:
   `run_chunked_prediction_inference` for raw prediction writing, and
   `run_chunked_affinity_cc_inference` for existing streamed decode/stitch.
3. Route `test_pipeline` chunked mode based on `chunking.output_mode`.
4. For `raw_prediction`, write the raw file first. If
   `decode_after_inference=true`, load that file and reuse the standard
   decode/postprocess/save/evaluate path.
5. Keep decode-only via `inference.saved_prediction_path` as the standalone
   decoding entry for now. A future CLI can expose it as `--mode decode`.

## Implemented

- `decoding` is a top-level/default/test/tune stage section.
- `evaluation` is a top-level/default/test/tune stage section.
- Tutorial YAMLs use `default.decoding`/`test.decoding` and
  `default.evaluation`/`test.evaluation` instead of nested inference sections.

## Follow-Ups

- Store prediction artifact metadata such as channel order, crop, activation,
  checkpoint, and value scale in a small sidecar or HDF5 attrs.
- Add lazy/blockwise decode readers for decoders that can operate without
  materializing the full prediction volume.
