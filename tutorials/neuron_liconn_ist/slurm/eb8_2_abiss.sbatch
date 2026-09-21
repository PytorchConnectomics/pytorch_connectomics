#!/bin/bash
# eb8 step 2: affinity -> ABISS segmentation -> VOI / adapted_rand / nerl_oracle_merge
# against the proofread FFN GT.
#
# Decode settings are inherited verbatim from 2_abiss.yaml (mt 0.47, ws_high/low
# at the 94th/20th percentile, channels [2,1,0], edge_storage source), so this
# number is directly comparable to the eb2 baseline of whole-val VOI 0.9129.
# That 0.47 was slab-tuned on eb2 and is biased HIGH -- a per-model sweep
# reaching <= 0.45 is still required before either model is declared better.
#
# Sizing measured on the eb2 run: decode 43 min at peak RSS 147 GB, scoring
# ~7 min at ~95 GB. 128 GB OOM'd. Do not lower --mem. CPU only.
#SBATCH --job-name=eb8_val_abiss
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=185G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/neuron_liconn_ist/%j_eb8_2_abiss.out
#SBATCH --error=slurm_logs/neuron_liconn_ist/%j_eb8_2_abiss.out

set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"

echo "host=$(hostname) start=$(date -Is)"
free -g | head -2

python scripts/main.py --config tutorials/neuron_liconn_ist/2_abiss_eb8.yaml \
  --mode test --checkpoint "$REPO/outputs/liconn_final_banis_plus_tube_4gpu/20260910_060545/checkpoints/step=00200000.ckpt"

echo "--- artifacts ---"
find "$REPO/outputs/neuron_liconn_ist_eb8" -maxdepth 3 -type f \( -name '*.json' -o -name '*.h5' \) -printf '%10s  %p\n' | sort -rn | head -20
echo "DONE end=$(date -Is)"
