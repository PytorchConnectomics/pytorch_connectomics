#!/bin/bash
# Verify the new --edge-storage option, including the YAML cli_args route.
#
# Do not use `set -u`: conda's activation scripts read optional unset variables.
#SBATCH --job-name=abiss-verify
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/projects/weilab/weidf/lib/pytorch_connectomics/slurm_outputs/abiss-verify_%j.out
#SBATCH --error=/projects/weilab/weidf/lib/pytorch_connectomics/slurm_outputs/abiss-verify_%j.err

set -eo pipefail

REPO=/projects/weilab/weidf/lib/pytorch_connectomics
WORKTREE=$REPO/.claude/worktrees/bridge-cse_0181aPnfEwx7QMuk46XqCpeh
RESULTS=$REPO/outputs/neuron_microns_pinky_affinity_lsd/20260821_014234/test_step=00200000

source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc

export HDF5_USE_FILE_LOCKING=FALSE
# cwd is the worktree so the command template's relative
# `scripts/run_abiss_volume.py` resolves to the edited runner; the worktree has
# no lib/, so ABISS home is passed explicitly.
export PYTHONPATH=$WORKTREE${PYTHONPATH:+:$PYTHONPATH}

cd "$WORKTREE"
echo "host=$(hostname) job=$SLURM_JOB_ID cwd=$(pwd)"

python -u scripts/verify_abiss_edge_storage.py \
    --results-dir "$RESULTS" \
    --config tutorials/neuron_microns_pinky_abiss.yaml \
    --abiss-home "$REPO/lib/abiss" \
    --runner scripts/run_abiss_volume.py
