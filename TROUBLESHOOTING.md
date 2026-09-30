# Troubleshooting

For dependency and CUDA installation failures, start with
[INSTALLATION.md](INSTALLATION.md#common-install-issues) and run
`python scripts/check_install.py`.

## Config or data errors

Validate a tutorial before starting training:

```bash
python scripts/validate_tutorial_configs.py --glob 'tutorials/mito_lucchi++/mito_lucchi++.yaml'
python scripts/main.py --config tutorials/mito_lucchi++/mito_lucchi++.yaml --fast-dev-run
```

Unknown keys are errors. Tutorial overrides belong under the selected stage
(`train`, `test`, or `tune`). For example:

```yaml
train:
  data:
    train:
      image: datasets/lucchi++/train_im.h5
      label: datasets/lucchi++/train_mito.h5
    dataloader:
      batch_size: 1
      patch_size: [32, 64, 64]
test:
  data:
    test:
      image: /path/to/dataset/test_image.h5
      label: /path/to/dataset/test_label.h5
```

Check that the files exist and their image/label shapes agree. Decrease the patch
size if it exceeds your volume. Test labels are optional for prediction alone.
See [tutorials/README.md](tutorials/README.md) for public-data examples.

## Memory and training speed

For GPU out-of-memory errors, reduce the batch size or patch size. Gradient
accumulation can retain a larger effective batch size:

```yaml
train:
  optimization:
    accumulate_grad_batches: 4
    precision: "16-mixed"
  data:
    dataloader:
      batch_size: 1
```

For a killed data-loader worker, reduce `train.system.num_workers` and consider
`train.data.dataloader.use_preloaded_cache_train=false` to avoid preloading whole
volumes. Increase workers only when memory permits. For NaN losses, check input
arrays for NaN/inf, reduce `train.optimization.optimizer.lr`, try
`train.optimization.precision="32"`, and enable `train.monitor.detect_anomaly`.

## Checkpoint and test errors

Use an existing checkpoint explicitly:

```bash
python scripts/main.py --config tutorials/mito_lucchi++/mito_lucchi++.yaml \
  --mode test --checkpoint /path/to/checkpoint.ckpt
```

Set `test.data.test.image` to the prediction input. Optional decoder dependencies
must be installed in the active environment; see the external-packages table in
[INSTALLATION.md](INSTALLATION.md#optional-external-packages). ABISS needs a
configured `abiss_home` or `PYTC_ABISS_HOME` and an appropriate runner command.

## Cluster and environment errors

If `sbatch` is unavailable, run the same command directly. On Slurm, activate the
intended conda environment before using `just slurm` or `just slurm-cpu`. Request
sufficient time and memory and inspect the job logs after a killed job.

Check the active interpreter with `which python` and `python --version`. The
supported range is Python 3.9–3.12, with 3.9 validation pending CI. The recommended
installation uses Python 3.11.

## Reporting an issue

Include the command, config, full traceback, Python and PyTorch versions, and
`nvidia-smi` output when relevant. Report reproducible failures through
[GitHub Issues](https://github.com/PytorchConnectomics/pytorch_connectomics/issues).
