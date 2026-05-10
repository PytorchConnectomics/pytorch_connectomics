# Installation

PyTorch Connectomics targets Python 3.11 in a conda environment named
`pytc` (both are configurable). Pick the path that matches your level
of control.

---

## Quickest path

```bash
curl -fsSL https://raw.githubusercontent.com/zudi-lin/pytorch_connectomics/master/quickstart.sh | bash
cd pytorch_connectomics
conda activate pytc
python scripts/main.py --demo
```

The `quickstart.sh` script installs Miniconda if missing, clones the
repo (when not already inside one), and runs `install.py`. The `cd`
step is needed because `curl ... | bash` runs the script in a child
process — your interactive shell stays where it started.

---

## Install with options

For control over the env name, Python version, CUDA wheel, or extras:

```bash
git clone https://github.com/zudi-lin/pytorch_connectomics.git
cd pytorch_connectomics
python install.py --install-type basic --python 3.11 --env-name pytc
```

`install.py` flags:

| Flag | Default | Purpose |
|---|---|---|
| `--env-name NAME` | `pytc` | Conda env name |
| `--python VER` | `3.11` | Python version (3.8–3.12) |
| `--cuda X.Y` | auto | Force a specific CUDA version (e.g. `12.1`) |
| `--cpu-only` | off | CPU-only PyTorch wheel |
| `--install-type` | `basic` | `basic`, `dev`, or `full` |
| `--force-recreate` | off | Wipe and recreate an existing env (default: reuse) |
| `--interactive` | off | Prompt before each step |
| `--pip-options STR` | `""` | Extra args passed to `pip install -e .` |
| `--no-color` | off | Disable ANSI colors |

If the target env already exists, `install.py` reuses it by default
and continues installing on top. Pass `--force-recreate` to wipe and
recreate it.

`install.py` auto-detects CUDA from `nvidia-smi`, `nvcc`, the module
system, and `/usr/local/cuda-*`. Passing `--cuda` or `--cpu-only`
overrides detection.

---

## Manual install

Use this when you want full control or are on a host where
`install.py` does not work cleanly:

```bash
conda create -n pytc python=3.11 -y
conda activate pytc

# Pre-built wheels for the few packages that compile against numpy/h5py.
conda install -c conda-forge numpy h5py cython connected-components-3d -y

# Pick the PyTorch wheel that matches your CUDA. See
# https://pytorch.org/get-started/locally/ for the index URL.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Editable install of pytorch_connectomics.
git clone https://github.com/zudi-lin/pytorch_connectomics.git
cd pytorch_connectomics
pip install -e .

# Optional: just (command runner used by README tutorial commands).
conda install -c conda-forge just -y
```

---

## Install via Claude Code or Codex CLI

If you already have an authenticated [Claude Code](https://claude.com/claude-code)
or [Codex CLI](https://github.com/openai/codex) on this machine, you
can delegate the install to the agent. The agent runs `install.py`
and diagnoses failures using this doc — it does not invent new
install steps.

**Prerequisites:** `claude` or `codex` CLI installed and
authenticated; `just` (used by the recipes — direct invocations
without `just` are shown below).

### Default: interactive (recommended)

The agent reads `INSTALL_PROMPT.md`, runs `install.py`, and asks for
your approval on each shell command before executing it. Use this
when you want a guided install without typing the commands yourself.

```bash
just install-claude     # opens claude with INSTALL_PROMPT.md
# or, without just:
claude "$(cat INSTALL_PROMPT.md)"
```

```bash
just install-codex      # opens codex with INSTALL_PROMPT.md
# or, without just:
codex "$(cat INSTALL_PROMPT.md)"
```

You will see prompts for `conda create -n pytc python=3.11`, the
PyTorch `pip install`, `pip install -e .`, and the demo command.
Approve each. The agent exits when `install.py` reports success.

### Unattended (advanced — trusted machines only)

Skip per-command approval. These flags grant the agent full shell
access; only run them on trusted, isolated machines (throwaway VMs,
a fresh user account, or a sandboxed container). Do not use on
shared workstations or development boxes that hold private data.

```bash
claude -p --permission-mode bypassPermissions --allowedTools "Bash(*)" \
       "$(cat INSTALL_PROMPT.md)"

codex exec --dangerously-bypass-approvals-and-sandbox - < INSTALL_PROMPT.md
```

---

## Optional extras

```bash
pip install -e .[full]   # tifffile, wandb, optuna, gputil, neuroglancer
pip install -e .[dev]    # pytest, pytest-benchmark, ruff, etc.
pip install git+https://github.com/PytorchConnectomics/MedNeXt.git
```

`wandb` is bundled inside the `[full]` extra — there is no separate
`[wandb]` extra in `setup.py` / `pyproject.toml`.

---

## Common install issues

### "No module named 'connectomics'"

Reactivate the env and reinstall in editable mode:

```bash
conda activate pytc
pip install -e . --no-build-isolation
```

### "NumPy requires GCC >= 9.3"

Common on HPC clusters with old toolchains. Use conda for the
packages that build native extensions:

```bash
conda activate pytc
conda install -c conda-forge numpy h5py cython connected-components-3d -y
pip install -e . --no-build-isolation
```

### "AttributeError: module 'numpy' has no attribute 'float'"

Mahotas < 1.4.18 is incompatible with NumPy 2.x:

```bash
pip install --upgrade mahotas numpy
```

### "CUDA not available" but you have a GPU

Either PyTorch was installed CPU-only, or CUDA is not loaded:

```bash
# Reinstall with CUDA wheel (adjust to your CUDA version).
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Or load CUDA module on HPC.
module load cuda/12.1
python -c "import torch; print(torch.cuda.is_available())"
```

### "ImportError: libcudnn.so.8: cannot open shared object file"

```bash
module load cudnn/8.9.0
# or:
conda install -c conda-forge cudnn -y
```

### "ABI mismatch on import cc3d"

`install.py` probes for this and repairs automatically. If you see
it after a manual install:

```bash
pip uninstall -y connected-components-3d
pip install --no-cache-dir connected-components-3d
```

---

## Cluster-specific helpers

If you are on the BC SLURM cluster, `scripts/setup_slurm.sh` (also
exposed as `just setup-slurm`) detects the cluster's CUDA/cuDNN
modules and patches `~/.bashrc` to load them automatically. This is
not part of the canonical install path; use it only if you know your
cluster's module conventions.

For Docker, see [`docker/Dockerfile`](docker/Dockerfile). The
container image is not actively maintained beyond CUDA 11.3.
