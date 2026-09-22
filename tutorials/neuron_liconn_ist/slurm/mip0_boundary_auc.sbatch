#!/bin/bash
# Decoder-free boundary test: does the mip0 affinity separate GT boundaries
# better than the mip1 affinity, on the 18 nm grid?
#
# This exists to break the ambiguity the card names. If mip0 ties mip1 on VOI,
# that is consistent BOTH with "the sharper input does not help the model" and
# with "the model sees more but the 18 nm label and the decoder cannot express
# it" (resolution-gate lesson §5). VOI cannot separate those; asserting one is
# not evidence. This asks only whether each model's affinity predicts, for two
# voxels ADJACENT ON THE 18 nm GRID, that they share a proofread object -- no
# watershed, no agglomeration, no merge threshold, and scored where the GT is
# exact rather than upsampled.
#
# An 18 nm edge is the MIN of the two 9 nm edges it spans (a path is as
# connected as its weakest link, which is also what the watershed's
# steepest-neighbour rule uses). Verified exact on a synthetic 2x-upsampled
# segmentation before running here.
#SBATCH --job-name=mip0_bauc
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/neuron_liconn_ist/%j_mip0_boundary_auc.out
#SBATCH --error=slurm_logs/neuron_liconn_ist/%j_mip0_boundary_auc.out

set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"

echo "host=$(hostname) start=$(date -Is)"

MIP1="$REPO/outputs/liconn_final_banis_plus_tube/20260728_032436/test_step=00200000/val/raw_x1_ch0-1-2.h5"
MIP0="$REPO/outputs/liconn_final_banis_plus_mip0/20260908_164220/test_step=00200000/val/raw_x1_ch0-1-2.h5"

python tutorials/neuron_liconn_ist/mip0_boundary_auc.py \
  --mip1 "$MIP1" --mip0 "$MIP0" --planes 24 \
  --json "$REPO/outputs/neuron_liconn_ist/mt_sweep/mip0_boundary_auc.json"

echo "DONE end=$(date -Is)"
