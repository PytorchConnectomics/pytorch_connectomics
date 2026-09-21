#!/bin/bash
# Measure the ABISS `ws` binary's own peak RSS vs volume size, to decide whether
# the mip0 val decode (16.65 Gvox) needs the h200 or fits a standard node.
#
# The recorded "147 GB peak at 2.08 Gvox" is the whole python job, whose largest
# terms are copies the wrapper makes and `ws` does not. See ws_sizing.py header.
#SBATCH --job-name=ws_sizing
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/neuron_liconn_ist/%j_ws_sizing.out
#SBATCH --error=slurm_logs/neuron_liconn_ist/%j_ws_sizing.out

set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"

echo "host=$(hostname) start=$(date -Is)"
free -g | head -2

python tutorials/neuron_liconn_ist/slurm/ws_sizing.py \
  --sizes 48x384x384,64x512x512,96x768x768,128x1024x1024,145x1536x1536 \
  --workdir "$REPO/outputs/neuron_liconn_ist/ws_sizing" \
  --json "$REPO/outputs/neuron_liconn_ist/mt_sweep/mip0_ws_sizing.json"
echo "DONE end=$(date -Is)"
