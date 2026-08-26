#!/bin/bash
# vol501's merge-threshold optimum sat at the low edge of the first sweep (0.54,
# the smallest value tried), so extend the curve downward to find the actual
# minimum. vol501 is 134M of the 165M test voxels, so it sets the weighted mean.
#
# Do not use `set -u`: conda's activation scripts read optional unset variables.
#SBATCH --job-name=pinky-abiss501
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=4:00:00
#SBATCH --output=/projects/weilab/weidf/lib/pytorch_connectomics/slurm_outputs/pinky-abiss501_%j.out
#SBATCH --error=/projects/weilab/weidf/lib/pytorch_connectomics/slurm_outputs/pinky-abiss501_%j.err

set -eo pipefail

REPO=/projects/weilab/weidf/lib/pytorch_connectomics
WORKTREE=$REPO/.claude/worktrees/bridge-cse_0181aPnfEwx7QMuk46XqCpeh
RESULTS=$REPO/outputs/neuron_microns_pinky_affinity_lsd/20260821_014234/test_step=00200000

source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc

export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}

AVAIL=$(df -BG --output=avail /tmp | tail -1 | tr -dc '0-9')
echo "host=$(hostname) job=$SLURM_JOB_ID /tmp avail=${AVAIL}G"
if [ "${AVAIL:-0}" -lt 40 ]; then
    export TMPDIR=$RESULTS/abiss_tmp
    mkdir -p "$TMPDIR"
    echo "using TMPDIR=$TMPDIR"
fi

cd "$REPO"

python -u "$WORKTREE/scripts/pinky_abiss_decode.py" --results-dir "$RESULTS" \
    --volumes pinky_vol501 --channels 2,1,0 --roll 1 \
    --ws-high 0.88 --ws-low 0.25 --ws-size 10000000 --ws-dust 200 \
    --merge-function max --merge-thresholds 0.44,0.48,0.50,0.52 \
    --tag abiss_max_vol501_low
