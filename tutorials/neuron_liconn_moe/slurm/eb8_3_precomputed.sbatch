#!/bin/bash
# eb8: build the neuroglancer precomputed layer + meshes for one eb8 segmentation.
#
# Upload is NOT done here. Compute nodes have no route to storage.googleapis.com
# and the gcloud credential is a user credential; run the --upload --verify pass
# from the login node afterwards (see the moe README).
#
# MOE_OUT_ROOT/MOE_CKPT scope this to the eb8 tree, which also picks the GCS
# destination: volumes.gcs_kind() reads OUT_ROOT's basename, so staging lands in
# eb8/precomputed/ and the target becomes <family>/mip1_eb8/.
#
#   sbatch --array=0-1 --export=ALL,EB8_VOLUMES="volA volB" eb8_3_precomputed.sbatch
#SBATCH --job-name=eb8_moe_ng
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/neuron_liconn_moe/%A_%a_eb8_3_precomputed.out
#SBATCH --error=slurm_logs/neuron_liconn_moe/%A_%a_eb8_3_precomputed.out

set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"

export MOE_CKPT="$REPO/outputs/liconn_final_banis_plus_tube_4gpu/20260910_060545/checkpoints/step=00200000.ckpt"
export MOE_OUT_ROOT="$REPO/outputs/neuron_liconn_moe/eb8"

VOLS=(${EB8_VOLUMES:-ExPID96_2ndgel_S1_40XW001_18x ExPID96_S1_40XW002_18x})
VOL="${VOLS[$SLURM_ARRAY_TASK_ID]}"
if [ -z "$VOL" ]; then
  echo "no volume at array index $SLURM_ARRAY_TASK_ID for list: ${VOLS[*]}" >&2
  exit 2
fi

echo "host=$(hostname) volume=$VOL start=$(date -Is)"
python - "$VOL" <<'PY'
import sys
sys.path.insert(0, "tutorials/neuron_liconn_moe")
import volumes as V
name = sys.argv[1]
print(f"staging -> {V.OUT_ROOT / 'precomputed'}")
print(f"gcs dest -> {V.gcs_folder(name)}")
PY

python tutorials/neuron_liconn_moe/upload_seg_precomputed.py \
  --volume "$VOL" --create --downsample --mesh --parallel 16
echo "DONE end=$(date -Is)"
