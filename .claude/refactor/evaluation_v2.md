# Evaluation V2 Plan

## Goal

Make evaluation a top-level stage that consumes decoded artifacts and ground
truth labels, then writes metric reports. Evaluation should be independent of
how predictions were produced or decoded.

Baseline: `inference-decoding-split.md` identifies evaluation as a missing
top-level stage. `metrics.md` reports that metric implementations are already
lean enough to support this split.

## Public API

Create or keep:

- `EvaluationConfig`
- `run_evaluation_stage`
- `evaluate_segmentation_artifact`
- `EvaluationResult`
- report writers for text/JSON/CSV as needed

Metric implementations stay under `connectomics.metrics`; evaluation owns
orchestration and artifact IO.

## Delete

- Metric report generation embedded in decoding stage code.
- Metric report generation embedded in inference output code.
- Any evaluation config nested under `inference`.
- Any special-case evaluation path coupled to one decoder.

## Move/Rename

Canonical responsibilities:

| Concept | V2 owner |
| --- | --- |
| Metric algorithms | `connectomics.metrics` |
| Evaluation stage orchestration | `connectomics.evaluation` |
| Evaluation config schema | `connectomics.config.schema` |
| Evaluation report writing | `connectomics.evaluation.report` |

## Config Contract

Top-level `evaluation` config owns:
- decoded prediction path;
- ground truth path;
- label dataset/key selection;
- metric names and metric kwargs;
- thresholds or matching parameters;
- output report path;
- optional visualization/report metadata.

It does not own:
- decoder thresholds unless they affect metric calculation;
- model checkpoint paths;
- sliding-window settings;
- raw prediction artifact paths except for provenance metadata.

## Boundary Rules

- Evaluation may import metrics, data IO, config, and utils.
- Evaluation must not import models.
- Evaluation must not import training.
- Evaluation must not import inference managers.
- Evaluation must not import decoder internals except for shared artifact
  metadata types if those are placed in a neutral module.

## Implementation Order

1. Add `connectomics.evaluation` package if it does not exist.
2. Define `EvaluationResult` and report output contract.
3. Add `run_evaluation_stage(decoded_path, gt_path, cfg)`.
4. Move metric orchestration out of test/decoding paths.
5. Update combined test mode to call evaluation stage after decoding.
6. Add standalone evaluation CLI/runtime hook.
7. Add tests for saved decoded artifacts from multiple decoders.

## Tests

Add:
- evaluation runs without constructing a model;
- evaluation runs on a decoded artifact written by decoding stage;
- report path contains all requested metrics;
- missing labels fail clearly;
- metric config validates channel/label requirements;
- combined test mode and standalone evaluation produce the same report.

## Open Decisions

- Whether reports should default to JSON, text, or both.
- Whether artifact metadata types should live in `evaluation`, `decoding`, or a
  neutral `connectomics.artifacts` package.
