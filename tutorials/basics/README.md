# Basic Decoding Tutorials

## Single-volume ABISS

[`decoding_abiss.yaml`](decoding_abiss.yaml) runs ABISS directly on one saved
affinity volume through `scripts/run_abiss_single.py`. It produces one HDF5
segmentation and does not use a chunk hierarchy or cross-chunk stitching.

Run it with:

```bash
python scripts/main.py \
  --config tutorials/basics/decoding_abiss.yaml \
  --mode test
```

Change these fields for a different dataset or experiment:

- `default.decoding.load_prediction_path`: input HDF5 containing a `CZYX` affinity array.
- `default.decoding.save_path`: output directory for the decoded HDF5.
- `steps[0].kwargs.input_dataset`: dataset key inside the input HDF5.
- `steps[0].kwargs.channels`: three nearest-neighbor affinity channels in ABISS `XYZ` order. Reorder them here if the model saved another order.
- `steps[0].kwargs.command`: single-volume ABISS command. Keep the input/output placeholders. Add `--abiss-home /absolute/path/to/abiss` when ABISS is not under this repository's `lib/abiss`.
- `cli_args.ws_high_threshold`: confident watershed seed threshold.
- `cli_args.ws_low_threshold`: lower affinity accepted while growing the watershed. This is the ABISS parameter corresponding to `aff_low`.
- `cli_args.ws_size_threshold`: size threshold used during watershed merging.
- `cli_args.ws_dust_threshold`: fragments below this size are removed.
- `cli_args.ws_merge_threshold`: affinity cutoff for size-based region merging. Increase it to merge more conservatively.
- `test.data.test.name`: stable volume name used in output filenames.
- `test.data.test.resolution`: voxel size in `ZYX` array-axis order.
- `test.data.test.label`: optional ground-truth instance HDF5; leave it `null` for decode-only use.

For volumes that must be divided and stitched across multiple logical chunks,
use the separate multi-chunk ABISS workflow instead.
