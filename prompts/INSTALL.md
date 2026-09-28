# PyTorch Connectomics — Agent Install Prompt

You are installing PyTorch Connectomics (PyTC) into a conda environment on
the user's machine. Success means `python scripts/check_install.py` exits 0
inside the target env **and** the demo prints `DEMO COMPLETED SUCCESSFULLY`.
Do not report success on any weaker evidence.

`install.py` is the only install procedure. `INSTALLATION.md` is the only
troubleshooting reference. Your job is to run the procedure, diagnose
failures from those two sources, and report accurately.

## 0. Ground rules

- **Env-local changes only.** Install packages only inside the target conda
  env. Never use `sudo`, `apt`, `yum`, `brew`, or edit `~/.bashrc` or system
  files. Never install into `base`. If a fix seems to need a system change,
  stop and ask. Every known issue in `INSTALLATION.md` has an env-local fix.
- **Do not edit the repository.** No changes under `connectomics/`,
  `scripts/`, `tests/`, `tutorials/`, `install.py`, or dependency files. If
  the procedure itself looks wrong, report it with evidence. Don't patch it.
- **Existing envs are the user's data.** Reusing an env is fine: `install.py`
  upgrades it in place. Never pass `--force-recreate`, `conda env remove`,
  or reinstall into a different existing env without explicit user consent.
- **Shared machines.** If this is a cluster or a multi-user workstation, do
  not start long or heavy GPU work outside the scheduler (step 4).
- **Follow host instructions.** If the machine or user has its own agent
  instructions (for example a `CLAUDE.md` or `AGENTS.md` above this repo, or
  a login message describing GPU policy), those take precedence over this
  prompt's defaults.

## 1. Inspect before changing anything

Run these and keep the results; you need them for decisions and the report.

```bash
uname -sm; hostname
conda --version && conda info --envs          # conda present? which envs exist?
nvidia-smi --query-gpu=index,name,compute_cap,memory.used,utilization.gpu --format=csv
nvidia-smi | grep -o "CUDA Version: [0-9.]*"  # driver's max CUDA (what matters)
command -v sbatch srun squeue                  # Slurm present?
df -h "$HOME" .                                # space for a ~6 GB env
git -C . rev-parse --short HEAD && test -f install.py && test -f scripts/check_install.py
```

Decide from the facts:

| Fact | Action |
|---|---|
| `conda` missing | Ask the user where to install Miniforge (default `~/miniforge3`; on clusters `$HOME` is often too small). Then run `CONDA_PREFIX_DIR=<path> bash quickstart.sh` from the repo root. |
| Env `pytc` already exists | Tell the user it will be reused and upgraded in place, then continue. |
| User named a different env or Python | Pass `--env-name` / `--python`. |
| User wants tests or dev tools | `--install-type dev`. Wants wandb/optuna/neuroglancer: `--install-type full`. Otherwise `basic`. |
| No NVIDIA GPU, not Apple silicon | Expect a CPU install; mention it in the report. |
| GPU compute cap ≥ 10.0 and driver CUDA < 12.8 | `install.py` will refuse. Don't work around it: the driver must be updated, which is the user's call. Report and stop. |
| `install.py` or `scripts/check_install.py` missing | Checkout is too old or not PyTC. Stop and report. |

Do not pass `--cuda`, `--torch-index-url`, or `--cpu-only` on the first run.
Detection maps the driver to a wheel and checks GPU compatibility. Override
only if the user asks, or if `check_install.py` shows detection chose wrong
(then use the override that `INSTALLATION.md` names for that symptom).

## 2. Install

From the repository root:

```bash
python install.py --env-name pytc --python 3.11 --install-type basic
```

Output streams live. The PyTorch step downloads 1–3 GB, so a quiet minute is
not a hang. The last step runs `scripts/check_install.py`, and `install.py`
exits non-zero if that check fails.

## 3. Diagnose failures

1. Find the first `[FAIL]` line or the first error in conda/pip output. Read
   the `->` hint under it.
2. Look up the symptom in `INSTALLATION.md` → "Common install issues" and
   apply that fix, which runs inside the env.
3. Rerun `install.py` with the same arguments. Reruns reuse the env and are
   safe.
4. If the same error appears twice, or the symptom is not in
   `INSTALLATION.md`, stop. Report the exact command, the error verbatim
   (the last ~30 relevant lines), and what you already tried. Do not guess
   further.

Known traps to diagnose, not work around:
- `torch.cuda.is_available()` is True but GPU ops fail ("no kernel image").
  This is the wrong wheel for the GPU, not a code bug.
- `libGL.so.1` missing. Swap OpenCV for the headless build inside the env;
  don't install system libraries.
- Conda Terms-of-Service error. Use conda-forge/Miniforge, per the docs.

## 4. Verify

```bash
conda run -n pytc python scripts/check_install.py --json
```

Require `"ok": true`. Then run the demo, which does about 30 s of GPU training:

- **Slurm present:** `srun --gres=gpu:1 --time=00:10:00 conda run -n pytc python scripts/main.py --demo`
  (ask before submitting if the user's partition or account is unknown).
- **Shared workstation without a scheduler:** choose a GPU with ~0 memory
  used from step 1 and run
  `CUDA_VISIBLE_DEVICES=<idx> conda run -n pytc python scripts/main.py --demo`.
  If every GPU is busy, ask.
- **Personal machine or CPU:** `conda run -n pytc python scripts/main.py --demo`.

Success requires `DEMO COMPLETED SUCCESSFULLY` in the output.

Do not run the full `pytest` suite as an install check. Some tests need data
or scripts that are not in a fresh clone. Run it only if the user asks, and
then report failures grouped by cause.

## 5. Report

End with a short report in this shape:

```text
Result:     PASS | FAIL
Env:        pytc at <prefix>  (created | reused)
Python:     3.11.x
PyTorch:    <version>+<cuXXX|cpu>   GPU(s): <name (sm_XY)> ...  kernels: OK
Extras:     basic | dev | full
Checks:     check_install.py ok=<true|false>; demo <passed|failed|skipped: reason>
Changes:    <env-local fixes applied, if any>
Follow-up:  <anything the user must do: driver update, missing data, etc.>
Activate:   conda activate pytc
            (batch jobs: eval "$(conda shell.bash hook)"; conda activate pytc)
```

List every deviation from the default procedure, however small. Don't
describe an unverified step as done.
