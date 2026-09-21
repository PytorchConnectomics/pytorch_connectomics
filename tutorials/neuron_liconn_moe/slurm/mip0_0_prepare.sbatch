#!/bin/bash
# mip0 step 0: stage the 18x volumes for the mip0 checkpoint.
#
# NOT a resample. MOE_GRID=mip0 makes plan() emit `--factor 1 1 1`, so this is a
# native passthrough: XY stays at 9.03 nm (the mip0 model's 9 nm, within 0.3%)
# and Z stays at 22.22 nm. The mip1 (1,2,2) average is deliberately NOT applied
# -- averaging XY away is exactly what mip0 is meant to avoid here. Z is left
# unmatched rather than interpolated 1.852x to 12 nm; that would invent planes,
# and the 32x pair (Z interpolated 1.92x) is the worst-scoring group in the batch.
#
# Drives prepare_volume.py from plan() because run_prepare.py indexes PENDING,
# which excludes the already-published volume.
#
#   sbatch --array=0-3 mip0_0_prepare.sbatch
#SBATCH --job-name=mip0_prep
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/neuron_liconn_moe/%A_%a_mip0_0_prepare.out
#SBATCH --error=slurm_logs/neuron_liconn_moe/%A_%a_mip0_0_prepare.out

set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"

export MOE_GRID=mip0

VOLS=(${MIP0_VOLUMES:-ExPID96_2ndgel_S1_40XW001_18x ExPID96_S1_40XW002_18x ExPID99_18x_2_cerebellum_1Byqvupl ExPID99_18x_2_cerebellum_1m2f9z4J})
VOL="${VOLS[$SLURM_ARRAY_TASK_ID]}"
if [ -z "$VOL" ]; then
  echo "no volume at array index $SLURM_ARRAY_TASK_ID for: ${VOLS[*]}" >&2
  exit 2
fi

echo "host=$(hostname) volume=$VOL start=$(date -Is)"
python - "$VOL" <<'PY'
import subprocess, sys
sys.path.insert(0, "tutorials/neuron_liconn_moe")
import volumes as V
name = sys.argv[1]
p = V.plan(name)
p["output"].parent.mkdir(parents=True, exist_ok=True)
cmd = ["python", "tutorials/neuron_liconn_moe/prepare_volume.py",
       "--input", str(p["source"]), "--output", str(p["output"]),
       *p["prepare_args"], "--overwrite"]
gv = p["shape"][0] * p["shape"][1] * p["shape"][2] / 1e9
print(f"{name}: {p['native_shape']} -> {p['shape']} ({gv:.2f} Gvox) at {[round(x,2) for x in p['spacing_zyx']]} nm ZYX")
print(" ".join(cmd), flush=True)
sys.exit(subprocess.call(cmd))
PY
echo "DONE end=$(date -Is)"
