# Installation

PyTorch Connectomics (PyTC) installs into its own conda environment
(default name `pytc`, Python 3.11). A typical install takes 2–5 minutes
and about 6 GB of disk, most of it the PyTorch CUDA wheel.

```bash
git clone https://github.com/PytorchConnectomics/pytorch_connectomics.git
cd pytorch_connectomics
python install.py                    # creates env "pytc", picks the right PyTorch
conda activate pytc
python scripts/check_install.py      # must end with "PASS"
python scripts/main.py --demo        # ~30 s; ends with "DEMO COMPLETED SUCCESSFULLY"
```

If both checks pass, you are done. The rest of this page covers choices,
other install paths, and fixes.

---

## Before you start

| You need | Check with | Notes |
|---|---|---|
| Linux or macOS | — | Windows: use WSL2. |
| conda (Miniforge recommended) | `conda --version` | Miniforge defaults to conda-forge and has no Anaconda Terms-of-Service prompt. `quickstart.sh` installs it if missing. |
| NVIDIA driver (for GPU) | `nvidia-smi` | The **driver** matters, not a CUDA toolkit. PyTorch wheels bundle their own CUDA runtime, so you do not need `nvcc` or `module load cuda`. |
| ~6 GB free where conda keeps envs | `conda info` (see `envs directories`) | On clusters with a small `$HOME` quota, install conda on a data or scratch filesystem. |

**Newer GPUs.** Blackwell cards (RTX 50xx, RTX PRO 6000, B100/B200)
need PyTorch built with CUDA 12.8 or newer, which needs a driver whose
`nvidia-smi` header shows `CUDA Version: 12.8` or higher. With an older
driver, `install.py` stops with an explicit message instead of
installing a PyTorch that imports fine but cannot run on the GPU.

---

## Choose an install path

| Path | Use when | Command |
|---|---|---|
| **`install.py`** (recommended) | Almost always | `python install.py` |
| `quickstart.sh` | Fresh machine, no conda, no clone yet | `curl -fsSL https://raw.githubusercontent.com/PytorchConnectomics/pytorch_connectomics/master/quickstart.sh \| bash` |
| Coding agent | You'd rather have Claude Code / Codex drive and diagnose | `just install-claude` or `just install-codex` |
| Manual | You need full control, or `install.py` fails on your host | [Manual install](#manual-install) |
| Docker | Reproducible, isolated image | [`docker/README.md`](docker/README.md) |

All paths end the same way: run [the checks](#verify-the-install).

### `install.py`

```bash
python install.py [options]
```

| Flag | Default | Purpose |
|---|---|---|
| `--env-name NAME` | `pytc` | Conda env to create or reuse |
| `--python VER` | `3.11` | Python version (3.9–3.12) |
| `--install-type` | `basic` | `basic`, `dev` (adds pytest, linters), or `full` (adds wandb, optuna, tifffile, neuroglancer, nd2) |
| `--cuda X.Y` | auto | Pretend the driver supports CUDA `X.Y` (see below) |
| `--torch-index-url URL` | — | Install PyTorch from exactly this index and skip detection, e.g. `https://download.pytorch.org/whl/cu130` |
| `--cpu-only` | off | CPU PyTorch (small download, no GPU) |
| `--force-recreate` | off | Delete and recreate an existing env |
| `--interactive` | off | Ask before each step |
| `--pip-options STR` | `""` | Extra args for `pip install -e .` |

What it does, in order: create or reuse the env → install
`numpy h5py cython connected-components-3d` from conda-forge (avoids
compiling on old toolchains) → install PyTorch → `pip install -e .` →
repair known ABI and OpenCV issues → install `just` → run
`scripts/check_install.py`. It exits non-zero if the final check fails.

Rerunning `install.py` on an existing env is safe. It reuses the env,
upgrades what changed, and re-verifies. Use this after pulling a new
version of the repo.

### How the PyTorch build is chosen

`install.py` reads the driver's CUDA version from `nvidia-smi` and
picks the newest PyTorch wheel that the driver can run:

| Driver CUDA (`nvidia-smi` header) | Wheel | PyTorch you get (2026-09) |
|---|---|---|
| ≥ 13.0 | `cu130` | latest |
| 12.8 – 12.9 | `cu128` | 2.11 (last on this index) |
| 12.6 – 12.7 | `cu126` | latest |
| 11.8 – 12.5 | `cu118` | 2.7.1 (last on this index) |
| < 11.8 | — | stops; update the driver or use `--cpu-only` |
| no NVIDIA GPU | CPU (or MPS on Apple silicon) | latest |

Most CUDA 12.x drivers can also run `cu126` wheels (CUDA minor-version
compatibility). If you are on a 12.0–12.5 driver and want a current
PyTorch, try `--cuda 12.6`, then confirm with `check_install.py`.

---

## Verify the install

```bash
conda activate pytc
python scripts/check_install.py
```

This checks that every core dependency imports. It also checks that the
installed PyTorch ships kernels for each visible GPU, and it runs one
tiny GPU operation. Each `[FAIL]` line is followed by a `->` hint. The
script exits 0 on success. `--json` gives machine-readable output, and
`--no-gpu-run` skips the GPU operation.

```bash
python scripts/main.py --demo
```

This trains a small model on synthetic data for about 30 seconds and
prints `DEMO COMPLETED SUCCESSFULLY`.

**For training on shared GPU machines**, request an allocated GPU:

```bash
srun --gres=gpu:1 python scripts/main.py --config tutorials/mito_lucchi++/mito_lucchi++.yaml
# or, on a workstation, pick an idle GPU first:
nvidia-smi
CUDA_VISIBLE_DEVICES=0 python scripts/main.py --config tutorials/mito_lucchi++/mito_lucchi++.yaml
```

`python -m pytest tests/unit -q` is a developer check (needs
`--install-type dev`). Tests needing optional packages, a GPU, or explicitly configured external
resources skip with a reason in a fresh clone.

---

## Manual install

```bash
conda create -n pytc -c conda-forge python=3.11 -y
conda activate pytc

# Pre-built binaries for packages that otherwise compile against numpy.
conda install -c conda-forge numpy h5py cython connected-components-3d just -y

# PyTorch: pick the index from the table above (cu130, cu128, cu126, cu118, cpu).
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

pip install -e .            # or: pip install -e ".[full,dev]"
python scripts/check_install.py
```

## Optional extras

```bash
pip install -e ".[full]"   # wandb, optuna, tifffile, neuroglancer, nd2, gputil
pip install -e ".[dev]"    # pytest, pytest-cov, pytest-benchmark, ruff, mypy
pip install git+https://github.com/PytorchConnectomics/MedNeXt.git
```

`wandb` is part of `[full]`; there is no separate `[wandb]` extra.
Focused extras: `[cloud]` installs cloud-volume, `[viz]` installs neuroglancer,
and `[tune]` installs optuna.

### Optional external packages

Install these separately only for workflows that use them.

| Package | Source / configuration |
|---|---|
| waterz | Install from `https://github.com/funkey/waterz`; imported as `waterz`. |
| affogato | Install from `https://github.com/constantinpape/affogato`; imported as `affogato`. |
| em_erl | Install an `em_erl` distribution into the active environment for NERL evaluation. |
| MedNeXt | `pip install git+https://github.com/PytorchConnectomics/MedNeXt.git` |
| ABISS | Build `https://github.com/seung-lab/abiss`; set decoder `abiss_home` or `PYTC_ABISS_HOME` to the checkout containing `build/ws`. The packaged `connectomics.decoding.abiss_runner` module runs it. |

---

## Install with a coding agent

With an authenticated [Claude Code](https://claude.com/claude-code) or
[Codex CLI](https://github.com/openai/codex), the agent can run the
install for you. It follows [`prompts/INSTALL.md`](prompts/INSTALL.md):
it inspects the machine, runs `install.py`, diagnoses failures using
this page, and reports what it installed. It asks you before anything
irreversible or system-wide.

```bash
just install-claude     # or: claude "$(cat prompts/INSTALL.md)"
just install-codex      # or: codex "$(cat prompts/INSTALL.md)"
```

You approve each shell command. Expect prompts for the conda env,
the PyTorch download, `pip install -e .`, and the checks.


---

## Common install issues

Start with `python scripts/check_install.py`. Its hint usually names
the section below.

### "no kernel image is available for execution on the device"

Also shows up as a warning like *"sm_120 is not compatible with the
current PyTorch installation"*. `torch.cuda.is_available()` is `True`,
but GPU ops fail. The PyTorch wheel was built for older GPUs. Rerun
`install.py`, which now maps the driver to a wheel that supports your
GPU. Or install one yourself:

```bash
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

If `install.py` says your driver is too old for your GPU, update the
NVIDIA driver. A different Python package will not help.

### "CUDA not available" but you have a GPU

Either PyTorch is the CPU build, or its CUDA version is newer than the
driver supports. Compare `python -c "import torch; print(torch.version.cuda)"`
with the `CUDA Version` in `nvidia-smi`. The wheel's version must not
be higher (apart from the 12.x minor-version exception above). Then
reinstall using the table in [How the PyTorch build is
chosen](#how-the-pytorch-build-is-chosen). `module load cuda` does not
fix this; wheels bundle their own runtime.

### "ImportError: libGL.so.1: cannot open shared object file"

The GUI build of OpenCV (`opencv-python`) was installed on a headless
server. PyTC now depends on `opencv-python-headless`, and rerunning
`install.py` swaps it automatically. By hand:

```bash
pip uninstall -y opencv-python
pip install --force-reinstall opencv-python-headless
```

Avoid `sudo apt install libgl1` as the fix. It changes the whole
system to cover a problem inside one env.

### "CondaToSNonInteractiveError: Terms of Service have not been accepted"

Your conda is Miniconda or Anaconda using the `defaults` channel.
Either accept the terms (`conda tos accept`) or use Miniforge, which
uses conda-forge. `install.py` creates envs from conda-forge either way.

### "No module named 'connectomics'" or "No module named 'zarr'"

The env isn't active, or the package was installed before a
dependency was added. Rerun `install.py`, or:

```bash
conda activate pytc
pip install -e .
```

### "NumPy requires GCC >= 9.3"

Common on HPC clusters with old toolchains. Install packages with
native extensions from conda-forge:

```bash
conda install -c conda-forge numpy h5py cython connected-components-3d -y
pip install -e . --no-build-isolation
```

### "ABI mismatch on import cc3d"

`install.py` detects and repairs this. After a manual install:

```bash
pip uninstall -y connected-components-3d
pip install --no-cache-dir connected-components-3d
```

### "AttributeError: module 'numpy' has no attribute 'float'"

Mahotas < 1.4.18 is incompatible with NumPy 2.x:
`pip install --upgrade mahotas numpy`.

### macOS aborts with "libomp.dylib already initialized"

`install.py` prevents this by selecting pthreads OpenBLAS. To repair an
older env: `conda install -n pytc -c conda-forge 'libopenblas=*=*pthreads*'`.

### "ImportError: libcudnn.so.8"

This comes from an old PyTorch build. Current wheels bundle cuDNN.
Reinstall PyTorch using the table above instead of loading a cuDNN
module.

### Install seems stuck

`install.py` streams conda and pip output live. The PyTorch step
downloads 1–3 GB, so give it a few minutes on a slow link. If conda's
solver hangs, make sure you are using a recent conda or Miniforge.

---

## Remote visualization

The Neuroglancer viewer binds to `127.0.0.1` by default. Forward its port over SSH:

```bash
ssh -L 9999:127.0.0.1:9999 user@server
# On the server, in the active environment:
python scripts/visualize_neuroglancer.py --port 9999 --image /path/to/image.h5
```

Open the printed viewer URL locally. Explicit `--bind-address 0.0.0.0` exposes
an unauthenticated viewer to the network and prints a warning.

For Docker, see [`docker/README.md`](docker/README.md).
