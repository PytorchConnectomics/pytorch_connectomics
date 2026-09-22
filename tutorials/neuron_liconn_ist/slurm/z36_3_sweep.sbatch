#!/bin/bash
# 36 nm step 3: whole-val ABISS merge-threshold sweep of a val_z36 affinity,
# scored against val_z36/seg on its own [36,18,18] grid (oracle floor 0 by
# construction). Card MSIDEPLOY-MODEL-002 Stage 2.
#
#   AFF=<.../val_z36/raw_x1_ch0-1-2.h5> LABEL=eb2_z36 sbatch z36_3_sweep.sbatch
#
# ws_high/ws_low stay 94%/20% (the 0.9129 baseline's settings) so this is not
# confounded with MSIDEPLOY-JOB-001. 0.47 is NOT inherited: the grid brackets it.
# 1.39 Gvox (< the 2.08 Gvox baseline that ran at 480G).
#SBATCH --job-name=z36_sweep
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=10:00:00
#SBATCH --output=slurm_logs/neuron_liconn_ist/%j_z36_3_sweep.out
#SBATCH --error=slurm_logs/neuron_liconn_ist/%j_z36_3_sweep.out
set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"
: "${AFF:?set AFF}" "${LABEL:?set LABEL}"
echo "host=$(hostname) start=$(date -Is) aff=$AFF"; free -g | head -2
python tutorials/neuron_liconn_ist/wholeval_sweep.py \
  --affinity "$AFF" --label "$LABEL" \
  --gt /projects/weilab/dataset/liconn/pytc/final_proofread_z36/val_z36/data.zarr/seg \
  --thresholds "${MTS:-0.38,0.41,0.44,0.47,0.50,0.53,0.56}" \
  --json "$REPO/outputs/neuron_liconn_ist/mt_sweep/${LABEL}_wholeval_sweep.json"
echo "DONE end=$(date -Is)"
