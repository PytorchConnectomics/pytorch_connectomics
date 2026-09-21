#!/bin/bash
# eb8 step 4: upload the staged precomputed layers to GCS.
#
# Split from the build (eb8_3_precomputed.sbatch) because the build is CPU-bound
# and the upload is network-bound. Runs on a COMPUTE node: verified 2026-09-16
# that compute nodes resolve storage.googleapis.com and the stored
# donglai@mindspan.org credential works there, contradicting the older comment in
# 3_precomputed.sbatch that upload needs an interactive login.
#
# MOE_OUT_ROOT picks both the staging tree and the GCS kind, so this lands in
# <family>/mip1_eb8/ beside the eb2 layers rather than on top of them.
#
#   sbatch --array=0-1 --export=ALL,EB8_VOLUMES="volA volB" eb8_4_upload.sbatch
#SBATCH --job-name=eb8_upload
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/projects/weilab/weidf/lib/pytorch_connectomics/slurm_logs/neuron_liconn_moe/%A_%a_eb8_4_upload.out

set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"

export MOE_CKPT="$REPO/outputs/liconn_final_banis_plus_tube_4gpu/20260910_060545/checkpoints/step=00200000.ckpt"
export MOE_OUT_ROOT="$REPO/outputs/neuron_liconn_moe/eb8"
export MOE_GCS_KIND=mip1_eb8

VOLS=(${EB8_VOLUMES:-ExPID96_2ndgel_S1_40XW001_18x ExPID96_S1_40XW002_18x})
VOL="${VOLS[$SLURM_ARRAY_TASK_ID]}"
[ -z "$VOL" ] && { echo "no volume at index $SLURM_ARRAY_TASK_ID" >&2; exit 2; }

echo "host=$(hostname) volume=$VOL start=$(date -Is)"
python - "$VOL" <<'PY'
import sys
sys.path.insert(0, "tutorials/neuron_liconn_moe")
import volumes as V
print("gcs dest ->", V.gcs_folder(sys.argv[1]))
PY

python tutorials/neuron_liconn_moe/upload_seg_precomputed.py --volume "$VOL" --upload --verify
echo "DONE end=$(date -Is)"
