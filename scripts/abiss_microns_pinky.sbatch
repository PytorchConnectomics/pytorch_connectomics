#!/bin/bash
# ABISS watershed + agglomeration on the cached Pinky test affinities.
#
# Conventions were picked by measurement on vol103 (see pinky_abiss_decode.py):
# channels 2,1,0 (repo stores z,y,x; ws reads x,y,z) and roll 1 (source->
# destination edge storage) give an oracle VOI of 0.18 vs 0.49-0.60 for the
# other three combinations. ws high/low barely matter (0.85-0.91 x 0.15-0.35
# all land within 0.02 VOI); the size threshold does: the stock 400 gates
# agglomeration off entirely on 4 nm voxels, so it is raised out of the way and
# the merge threshold does the work.
#
# Do not use `set -u`: conda's activation scripts read optional unset variables.
#SBATCH --job-name=pinky-abiss
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=8:00:00
#SBATCH --output=/projects/weilab/weidf/lib/pytorch_connectomics/slurm_outputs/pinky-abiss_%j.out
#SBATCH --error=/projects/weilab/weidf/lib/pytorch_connectomics/slurm_outputs/pinky-abiss_%j.err

set -eo pipefail

REPO=/projects/weilab/weidf/lib/pytorch_connectomics
WORKTREE=$REPO/.claude/worktrees/bridge-cse_0181aPnfEwx7QMuk46XqCpeh
RESULTS=$REPO/outputs/neuron_microns_pinky_affinity_lsd/20260821_014234/test_step=00200000
DRIVER=$WORKTREE/scripts/pinky_abiss_decode.py

source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc

export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}

# ABISS streams affinity and per-threshold uint64 segmentations through TMPDIR;
# vol501 alone needs ~10 GB. Fall back to /projects when node-local /tmp is small.
AVAIL=$(df -BG --output=avail /tmp | tail -1 | tr -dc '0-9')
echo "host=$(hostname) job=$SLURM_JOB_ID /tmp avail=${AVAIL}G"
if [ "${AVAIL:-0}" -lt 40 ]; then
    export TMPDIR=$RESULTS/abiss_tmp
    mkdir -p "$TMPDIR"
    echo "using TMPDIR=$TMPDIR"
fi

cd "$REPO"

SMALL=pinky_vol101,pinky_vol102,pinky_vol103,pinky_vol104,pinky_vol503
COMMON="--results-dir $RESULTS --channels 2,1,0 --roll 1 --ws-high 0.88 --ws-low 0.25 --ws-size 10000000 --ws-dust 200"

echo
echo "================ ABISS max-affinity agglomeration, 5 small volumes ================"
python -u "$DRIVER" $COMMON --volumes "$SMALL" --merge-function max \
    --merge-thresholds 0.50,0.54,0.56,0.58,0.60,0.62,0.65,0.70 \
    --save-seg --tag abiss_max_small

echo
echo "================ ABISS max-affinity agglomeration, vol501 ================"
python -u "$DRIVER" $COMMON --volumes pinky_vol501 --merge-function max \
    --merge-thresholds 0.54,0.58,0.60,0.62 \
    --save-seg --tag abiss_max_vol501

echo
echo "================ ABISS mean-affinity agglomeration, 5 small volumes ================"
python -u "$DRIVER" $COMMON --volumes "$SMALL" --merge-function mean \
    --merge-thresholds 0.40,0.45,0.50,0.54,0.58,0.60 \
    --tag abiss_mean_small
