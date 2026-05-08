# Review v2

## Summary

`code_v2` fixes the flat chunked output path construction and removes the
obvious `Path(...).stem` cache-preflight mismatch from `review_v1`. One stem
resolution edge case remains: when the configured split `path` is itself the
volume directory, `resolve_dataset_volume_stems` stops before returning that
directory name and falls back to `volume`. That still makes cache discovery
disagree with the writer for a real NISB tutorial shape.

## Diff Baseline

run_start_ref: c82ec629ddac061ffca272eea4c7f702771bb0e9

## Findings

1. `connectomics/runtime/output_naming.py:72`-`80` breaks out when the parent
   equals the configured `base`, so `resolve_dataset_volume_stems` returns
   `volume` for configs where `data.test.path` is the dataset identity directory
   and `data.test.image` points inside a container.

   This is the exact shape used by `tutorials/neuron_nisb/base_banis.yaml:182`
   `path: .../seed101/` with `tutorials/neuron_nisb/base_banis.yaml:184`
   `image: data.zarr/img`. After config path resolution, the helper sees
   `path=/projects/.../seed101/data.zarr/img` and
   `base=/projects/.../seed101`; it skips `data.zarr`, reaches the base path,
   breaks, and falls back to `volume`.

   I reproduced the current behavior:

   ```text
   _stem no base: seed101
   _stem with base: volume
   resolve_dataset_volume_stems: ['volume']
   ```

   The writer path uses `resolve_output_filenames(...)->_stem_from_image_path`
   without `base`, so it still writes under `seed101`. Meanwhile
   `has_cached_predictions_in_output_dir` uses `resolve_dataset_volume_stems`
   and can look under `<save_path>/volume/<artifact>` instead of
   `<save_path>/seed101/<artifact>`. That keeps one of the cache-discovery paths
   inconsistent with the per-volume writer contract.

## Tests to Add

- Add a regression for `resolve_dataset_volume_stems` with
  `cfg.data.test.path = "/.../seed101"` and
  `cfg.data.test.image = "/.../seed101/data.zarr/img"` after path resolution,
  asserting `["seed101"]`.
- Add the same case for relative YAML-style input before/through
  `resolve_data_paths`, using `path: /.../seed101` and `image: data.zarr/img`.
- Keep the chunked path tests requested in `review_v1` as follow-up coverage:
  the implementation now looks correct, but there is still no direct test that
  the test-step chunked runners receive `<save_path>/<volume>/<artifact>.h5`.

## Questions

- Should `_stem_from_image_path(..., base=...)` allow returning `base.name`
  when every child candidate is structural, or should `resolve_dataset_volume_stems`
  avoid passing `base` for already-resolved absolute image paths?

## Verdict

VERDICT: NEEDS_CHANGES
