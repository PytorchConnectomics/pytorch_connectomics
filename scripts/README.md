# General tools

Run these commands from the repository root after installing PyTC. Each tool's
`--help` describes its options, except the simple positional image converter and
loader profiler. Paths below are examples to replace with your own files.

| Script | Purpose | Usage |
|---|---|---|
| `scripts/main.py` | Run training or a stage workflow. | `python scripts/main.py --demo` |
| `scripts/check_install.py` | Report installed packages and accelerator support. | `python scripts/check_install.py` |
| `scripts/validate_tutorial_configs.py` | Validate every tutorial YAML and its paths. | `python scripts/validate_tutorial_configs.py` |
| `scripts/visualize_neuroglancer.py` | Inspect image and segmentation volumes. | `python -i scripts/visualize_neuroglancer.py --image /path/to/image.h5` |
| `scripts/download_data.py` | Download public tutorial datasets. | `python scripts/download_data.py --list` |
| `scripts/images_to_h5.py` | Stack image files into an HDF5 volume. | `python scripts/images_to_h5.py '/path/to/images/*.png' volume.h5 image main` |
| `scripts/h5_to_precomputed.py` | Convert HDF5 to a local precomputed layer. | `python scripts/h5_to_precomputed.py image.h5 precomputed/ --resolution 30 6 6` |
| `scripts/h5_to_precomputed_cloud.py` | Stream HDF5 or Zarr to a cloud precomputed layer. | `python scripts/h5_to_precomputed_cloud.py image.h5 gs://bucket/image --resolution 30 6 6` |
| `scripts/tiles_to_zarr.py` | Reconstruct tiled images as a multiscale OME-Zarr volume. | `python scripts/tiles_to_zarr.py --source tiles.json --output image.zarr --stage init` |
| `scripts/convert_h5_to_uint8.py` | Convert a floating-point HDF5 volume to uint8 in chunks. | `python scripts/convert_h5_to_uint8.py float.h5 uint8.h5` |
| `scripts/downsample_data.py` | Downsample volumes by per-axis factors. | `python scripts/downsample_data.py image.h5 --downsample-ratio-zyx 2 2 2` |
| `scripts/uncrop.py` | Pad a cropped HDF5 volume. | `python scripts/uncrop.py cropped.h5 padded.h5 --k 1` |
| `scripts/apply_volume_function.py` | Apply a named function to a volume. | `python scripts/apply_volume_function.py --help` |
| `scripts/evaluate_prediction.py` | Evaluate a saved prediction against labels or skeletons. | `python scripts/evaluate_prediction.py prediction.h5 labels.h5 adapted_rand` |
| `scripts/evaluate_snemi3d.py` | Score instance predictions using the SNEMI3D convention. | `python scripts/evaluate_snemi3d.py segmentation.h5 --output scores.tsv` |
| `scripts/export_onnx.py` | Export a configured checkpoint as ONNX. | `python scripts/export_onnx.py --config tutorials/neuron_snemi/neuron_snemi.yaml --checkpoint model.ckpt --output model.onnx` |
| `scripts/checkpoint_conversion.py` | Convert checkpoint serialization. | `python scripts/checkpoint_conversion.py model.ckpt --help` |
| `scripts/stitch_chunked_prediction.py` | Rebuild or virtually combine chunked prediction artifacts. | `python scripts/stitch_chunked_prediction.py prediction.h5 --vds` |
| `scripts/profile_dataloader.py` | Measure loading speed for a configured dataset. | `python scripts/profile_dataloader.py tutorials/mito_lucchi++/mito_lucchi++.yaml` |
| `scripts/tools/eval_curvilinear.py` | Evaluate curvilinear masks over a directory. | `python scripts/tools/eval_curvilinear.py --gt-path labels/ --pd-path predictions/` |
| `scripts/download_precompute.py` | Download a precomputed volume into Zarr. | `python scripts/download_precompute.py gs://bucket/image --out image.zarr` |
| `scripts/precompute_skeleton_volumes.py` | Cache skeleton volumes for training labels. | `python scripts/precompute_skeleton_volumes.py --label labels.h5` |
| `scripts/build_contact_graph.py` | Count touching label faces in volume slabs. | `python scripts/build_contact_graph.py --seg segmentation.zarr --slab 0 --out contacts/` |

Cloud tools require the optional `cloud` extra; the viewer requires `viz`.
Other optional packages are described in `INSTALLATION.md`.

The viewer binds to `127.0.0.1` by default. On a remote machine, keep that default
and forward its port with `ssh -L 9999:127.0.0.1:9999 user@server`. Open the printed
viewer URL locally. `--bind-address 0.0.0.0` explicitly exposes an unauthenticated
viewer to other machines and prints a warning.
