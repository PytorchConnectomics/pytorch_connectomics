#!/bin/bash
# eb8 decode: ABISS watershed + GT-free merge-threshold sweep on the ExPID96 18x
# volumes, from the 4-GPU (effective batch 8) checkpoint.
#
# WHY MOE_OUT_ROOT IS SET. Results stay under outputs/neuron_liconn_moe/ but in a
# model-scoped subtree. Decoding a second model into the DEFAULT root would
# destroy the first model's artifacts: sweep_merge_threshold.py calls
# final.unlink() before hard-linking the chosen segmentation, writes
# mt_sweep.json at a fixed name, and for ExPID96_2ndgel_S1_40XW001_18x
# layer_name() returns the hardcoded LEGACY string `..._seg_abiss_mt060` -- the
# very file already published to GCS. The result would be an eb2-named artifact
# containing eb8 output. MOE_CKPT and MOE_OUT_ROOT must always move together.
#
# MOE_CKPT also re-points volumes.TEST_OUT, so affinity_h5() reads the eb8
# inference tree automatically. No --affinity override is needed.
#
# There is NO GROUND TRUTH for any moe volume. This produces a segmentation and a
# GT-free threshold choice, not a quality number, and nothing here ranks eb8
# against eb2.
#SBATCH --job-name=eb8_moe_sweep
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/neuron_liconn_moe/%A_%a_eb8_2_sweep.out
#SBATCH --error=slurm_logs/neuron_liconn_moe/%A_%a_eb8_2_sweep.out

set -eo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=FALSE
cd "$REPO"

export MOE_CKPT="$REPO/outputs/liconn_final_banis_plus_tube_4gpu/20260910_060545/checkpoints/step=00200000.ckpt"
export MOE_OUT_ROOT="$REPO/outputs/neuron_liconn_moe/eb8"

# Volume list is a parameter, so this file is not re-edited per batch. Pass it
# and size the array to match:
#   sbatch --array=0-1 --export=ALL,EB8_VOLUMES="volA volB" eb8_2_sweep.sbatch
# eb8 has affinity for four volumes only (the two ExPID96 18x and the two
# ExPID99 18x cerebellum); the other eight moe volumes were never inferred with
# this checkpoint, so asking for one here fails at the affinity read.
VOLS=(${EB8_VOLUMES:-ExPID96_2ndgel_S1_40XW001_18x ExPID96_S1_40XW002_18x})
VOL="${VOLS[$SLURM_ARRAY_TASK_ID]}"
if [ -z "$VOL" ]; then
  echo "no volume at array index $SLURM_ARRAY_TASK_ID for list: ${VOLS[*]}" >&2
  exit 2
fi

echo "host=$(hostname) volume=$VOL start=$(date -Is)"
echo "ckpt=$MOE_CKPT"
echo "out_root=$MOE_OUT_ROOT"
free -g | head -2

python tutorials/neuron_liconn_moe/sweep_merge_threshold.py --volume "$VOL"

# layer_name() ignores the chosen threshold for the LEGACY volume and returns a
# fixed `mt060` string. Under the eb8 root that collides with nothing, but the
# filename can still misstate the threshold -- mt_sweep.json holds the truth.
python - "$MOE_OUT_ROOT/$VOL/mt_sweep.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
c = s["chosen"]
print(f"volume={s['volume']} chosen_mt={c['mt']} selected_by={c['selected_by']} "
      f"labels={c['labels']} covered={c['covered']:.3f} largest_share={c['largest_share']:.4f}")
print(f"segmentation={s['segmentation']}")
if f"mt{c['mt']:.3f}".replace('.', '') not in s['segmentation']:
    print("NOTE: filename threshold does not match the chosen mt (LEGACY fixed layer name).")
PY
echo "DONE end=$(date -Is)"
