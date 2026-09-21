#!/bin/bash
# CPU-only follow-up to the Pinky test run: oracle-merge ceiling on the shipped
# 0.75 decode, then a decode-threshold sweep over the cached affinities.
#
# Do not use `set -u`: conda's activation scripts read optional unset variables.
#SBATCH --job-name=pinky-analyze
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=6:00:00
#SBATCH --output=/projects/weilab/weidf/lib/pytorch_connectomics/slurm_outputs/pinky-analyze_%j.out
#SBATCH --error=/projects/weilab/weidf/lib/pytorch_connectomics/slurm_outputs/pinky-analyze_%j.err

set -eo pipefail

REPO=/projects/weilab/weidf/lib/pytorch_connectomics
WORKTREE=$REPO/.claude/worktrees/bridge-cse_0181aPnfEwx7QMuk46XqCpeh
RESULTS=$REPO/outputs/neuron_microns_pinky_affinity_lsd/20260821_014234/test_step=00200000

source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc

export HDF5_USE_FILE_LOCKING=FALSE
# connectomics is not pip-installed; a script invoked by absolute path gets its
# own dir on sys.path, not the cwd, so the repo has to be on PYTHONPATH.
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$REPO"
echo "host=$(hostname) job=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK"

echo
echo "================ ORACLE-MERGE CEILING (shipped 0.75 decode) ================"
python -u "$WORKTREE/scripts/pinky_test_ceiling.py" --results-dir "$RESULTS"

echo
echo "================ DECODE THRESHOLD SWEEP ================"
python -u "$WORKTREE/scripts/pinky_decode_sweep.py" --results-dir "$RESULTS" \
    --thresholds 0.05,0.10,0.15,0.20,0.30,0.40,0.50,0.60,0.75,0.85
