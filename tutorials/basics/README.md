# Basic decoding tutorials

`tutorials/basics/decoding_abiss.yaml` decodes a saved CZYX affinity volume
through an external ABISS installation, then optionally smooths label shapes.

Set `default.decoding.load_prediction_path` to the saved prediction and
`default.decoding.steps[0].kwargs.abiss_home` to your external ABISS directory.
That directory must contain the compiled `build/ws` binary. PyTC provides the
single-volume runner as `python -m connectomics.decoding.abiss_runner`; the
default command passes `--abiss-home`, `--input`, `--output`, and the threshold
flags listed in `cli_args`.
The `{abiss_home}`, `{python_exe}`, `{input_h5}`, and `{output_h5}` placeholders
are expanded by the decoder; `PYTC_ABISS_HOME` can provide the directory when
the explicit `abiss_home` key is omitted.

```bash
python scripts/main.py --config tutorials/basics/decoding_abiss.yaml --mode test
```

`channels: [0, 1, 2]` selects the nearest-neighbor XYZ affinity channels.
Set `input_dataset` and `output_dataset` to the HDF5 dataset names expected
by the packaged runner. `test.data.test.resolution` uses ZYX voxel order.

The `shape_smooth` step applies label opening and connected-component relabeling.
`open_plane: 2d` operates independently in each z slice; `3d` includes neighbors
along z. `split` enables the optional cross-z area-outlier split. Tune these
settings to your voxel resolution and inspect the resulting segmentation.
