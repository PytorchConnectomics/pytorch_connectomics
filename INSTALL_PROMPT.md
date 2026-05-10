# PyTorch Connectomics — Agent Install Prompt

You are installing PyTorch Connectomics on this machine. Follow the
steps below in order. Use the user's shell to run commands; do not
edit any files in the repository.

## Steps

1. Detect whether `conda` is on `PATH`. If not, install Miniconda to
   `$HOME/miniconda3` and re-source the shell.

2. From the repository root, run:

   ```bash
   python install.py --install-type basic --python 3.11 --env-name pytc
   ```

   Do **not** pass `--cuda` or `--cpu-only`. `install.py` already
   auto-detects CUDA from `nvidia-smi`, `nvcc`, the module system,
   and `/usr/local`. Override only if the user explicitly asks for
   a specific CUDA version or a CPU-only install.

3. If `install.py` exits non-zero, read its stderr, look up the
   symptom in `INSTALLATION.md` under "Common install issues", and
   rerun with the documented workaround. Do not patch the source
   tree.

4. Verify:

   ```bash
   conda run -n pytc python scripts/main.py --demo
   ```

   The demo should print `DEMO COMPLETED SUCCESSFULLY`.

## Constraints

- Do not modify files under `connectomics/`, `tutorials/`, `tests/`,
  or `scripts/`. The only intended writes are package installs and
  conda env creation.
- Do not invent new install steps. `install.py` is the single source
  of truth for install behaviour; if a step seems missing, add it
  to `install.py` only after checking with the user.
- If `install.py` is missing or fails twice with the same error,
  stop and report the error verbatim. Do not improvise.
