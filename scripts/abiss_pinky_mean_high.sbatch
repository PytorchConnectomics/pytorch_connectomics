#!/bin/bash
# The mean-affinity arm's grid (0.40-0.60) sits entirely below its optimum -- it
# agglomerates harder than max at the same threshold and collapsed vol101 to 9
# fragments at 0.40-0.45. Extend upward so the max-vs-mean comparison is measured
# at each function's own optimum instead of asserted.
#
# Do not use `set -u`: conda's activation scripts read optional unset variables.
#SBATCH --job-name=pinky-abissmean
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=4:00:00
#SBATCH --output=/projects/weilab/weidf/lib/pytorch_connectomics/slurm_outputs/pinky-abissmean_%j.out
#SBATCH --error=/projects/weilab/weidf/lib/pytorch_connectomics/slurm_outputs/pinky-abissmean_%j.err

set -eo pipefail

REPO=/projects/weilab/weidf/lib/pytorch_connectomics
WORKTREE=$REPO/.claude/worktrees/bridge-cse_0181aPnfEwx7QMuk46XqCpeh
RESULTS=$REPO/outputs/neuron_microns_pinky_affinity_lsd/20260821_014234/test_step=00200000

source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc

export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}

echo "host=$(hostname) job=$SLURM_JOB_ID"
cd "$REPO"

python -u "$WORKTREE/scripts/pinky_abiss_decode.py" --results-dir "$RESULTS" \
    --volumes pinky_vol101,pinky_vol102,pinky_vol103,pinky_vol104,pinky_vol503 \
    --channels 2,1,0 --roll 1 --ws-high 0.88 --ws-low 0.25 --ws-size 10000000 --ws-dust 200 \
    --merge-function mean --merge-thresholds 0.62,0.65,0.68,0.72 \
    --tag abiss_mean_high_small
