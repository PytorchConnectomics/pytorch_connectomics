#!/bin/bash
# 600 nm gate step 0b: the label-damage pass alone, with z-FRAGMENTATION added.
# Card MSIDEPLOY-MODEL-002. Step 0 (job 3031054) already produced the extent and
# vanish counts; this adds the number that actually bears on supervision damage
# -- how many GT objects gain contiguous z-runs under the mode downsample, i.e.
# have a false split written into the label. Streams one z-plane at a time.
#SBATCH --job-name=z600_damage
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/neuron_liconn_ist/%j_z600_0b_label_damage.out
#SBATCH --error=slurm_logs/neuron_liconn_ist/%j_z600_0b_label_damage.out
set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"
echo "host=$(hostname) start=$(date -Is)"
/usr/bin/time -v python tutorials/neuron_liconn_ist/z600_oracle_floor.py --damage-only \
  --json "$REPO/outputs/neuron_liconn_ist/mt_sweep/z600_label_damage.json"
echo "DONE end=$(date -Is)"
