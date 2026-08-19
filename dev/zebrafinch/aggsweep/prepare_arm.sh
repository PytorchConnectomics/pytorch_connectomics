#!/bin/bash
# Prepare one AGG-sweep arm and print its submission plan. Does NOT sbatch unless --submit.
#
#   bash dev/zebrafinch/aggsweep/prepare_arm.sh a0_control
#   bash dev/zebrafinch/aggsweep/prepare_arm.sh a0_control --submit
#
# Run a0_control first and ALONE: it is the fidelity gate for watershed reuse (README.md).
set -eo pipefail  # not -u until after conda: activate-binutils reads an unbound ADDR2LINE

ARM=${1:?usage: prepare_arm.sh <arm> [--submit]}
SUBMIT=${2:-}

REPO=/projects/weilab/weidf/lib/pytorch_connectomics
PYTC_PYTHON=/projects/weilab/weidf/lib/miniconda3/envs/pytc/bin/python
SWEEP=$REPO/dev/zebrafinch/aggsweep
CFG=$SWEEP/$ARM.yaml
WV=$SWEEP/$ARM
RUN_NAME=seg_agg_$ARM

[ -f "$CFG" ] || { echo "no such arm config: $CFG" >&2; exit 1; }
cd "$REPO"
export HDF5_USE_FILE_LOCKING=FALSE

# The donor is whatever the arm's own WS_PATH points at, so one driver serves the win144 (a*)
# and native-96 (b*) arms without edits. Derived from the YAML before prepare, and re-checked
# against the prepared param afterwards.
DONOR_WS=$("$PYTC_PYTHON" -c "
import sys, yaml
ws = yaml.safe_load(open(sys.argv[1]))['seuron_replay']['param_overrides']['WS_PATH']
print(ws.replace('file://', ''))" "$CFG")
DONOR_NAME=$(basename "$DONOR_WS")
DONOR_ROOT=$(dirname "$(dirname "$(dirname "$DONOR_WS")")")
DONOR_CHUNKMAP=$DONOR_ROOT/scratch/$DONOR_NAME/ws/chunkmap

# --- donor preconditions ------------------------------------------------------------------
N_CHUNKMAP=$(ls "$DONOR_CHUNKMAP" 2>/dev/null | wc -l)
[ "$N_CHUNKMAP" -eq 10626 ] || {
  echo "donor ws chunkmap has $N_CHUNKMAP files, expected 10626 -- donor watershed incomplete" >&2
  echo "  looked in: $DONOR_CHUNKMAP" >&2
  exit 1; }
echo "[donor] $DONOR_NAME -- ws chunkmap OK: $N_CHUNKMAP files"

# --- prepare ------------------------------------------------------------------------------
# --mode fresh REFUSES a non-empty run root and never deletes (run_seuron_provenance.py:464),
# so this cannot clobber a previous arm by accident.
"$PYTC_PYTHON" dev/zebrafinch/wholevol_prepare.py \
  --config "$CFG" --name "$RUN_NAME" --out-root "$WV" \
  --runtime-json "$WV/runtime.json" --mode fresh

# --- verify the prepared param actually reuses the donor -----------------------------------
RUNTIME_JSON="$WV/runtime.json" DONOR_NAME="$DONOR_NAME" "$PYTC_PYTHON" - <<'PY'
import json, os
from pathlib import Path

runtime = json.loads(Path(os.environ["RUNTIME_JSON"]).read_text())
param = json.loads(Path(runtime["param"]).read_text())

donor = os.environ["DONOR_NAME"]
assert donor in param["WS_PATH"], f"WS_PATH is not the donor watershed: {param['WS_PATH']}"
assert donor in param["NUC_COMPETITION_MANIFEST"], "nucleus manifest is not the donor's"
# The arm must NOT write into the donor: its own scratch/seg/chunkmap stay local.
for key in ("SCRATCH_PATH", "SEG_PATH", "CHUNKMAP_OUTPUT"):
    assert donor not in param[key], f"{key} writes into the donor run: {param[key]}"
assert Path(param["NUC_COMPETITION_MANIFEST"]).is_file(), "donor nucleus manifest missing"

agg = {k: v for k, v in sorted(param.items()) if k.startswith("AGG_")}
print("prepared OK")
print("  WS_PATH   (donor):", param["WS_PATH"])
print("  SEG_PATH  (arm)  :", param["SEG_PATH"])
print("  CHUNKMAP  (arm)  :", param["CHUNKMAP_OUTPUT"])
print("  AGG keys         :", agg)
print("  layers           :", runtime["layers"])
PY

# --- seed this arm's chunkmap from the donor's finished watershed --------------------------
# CHUNKMAP_INPUT cannot be overridden (abiss_chunk.py:663-664 clobbers it), and aiming
# CHUNKMAP_OUTPUT at the donor would send this arm's writes there. So copy -- 126 MB.
ARM_CHUNKMAP=$WV/$RUN_NAME/scratch/$RUN_NAME/ws/chunkmap
mkdir -p "$ARM_CHUNKMAP"
N_HAVE=$(ls "$ARM_CHUNKMAP" 2>/dev/null | wc -l)
if [ "$N_HAVE" -ne 10626 ]; then
  echo "[seed] copying donor ws chunkmap -> $ARM_CHUNKMAP"
  cp -n "$DONOR_CHUNKMAP"/*.zst "$ARM_CHUNKMAP"/
fi
N_HAVE=$(ls "$ARM_CHUNKMAP" | wc -l)
[ "$N_HAVE" -eq 10626 ] || { echo "[seed] FAILED: $N_HAVE/10626" >&2; exit 1; }
echo "[seed] arm chunkmap OK: $N_HAVE files"

# --- submission plan ------------------------------------------------------------------------
cat <<EOF

Submission plan for $ARM
  watershed  : REUSED from $DONOR_NAME (ws_L0..L5, remapws skipped)
  nucleus    : REUSED manifest (nuccomp skipped, >5 h saved)
  stages run : me_L0 me_L1 me_L2 me_L3 me_L4 me_L5 remapagg
  command    :
    START_AT=me_L0 WV="$WV" bash dev/zebrafinch/submit_wholevol_sharded.sh \\
      | tee "$WV/slurm_submission.txt"
  then score (BOTH thresholds -- MANUAL.md evaluation standard):
    sbatch --export=ALL,SEG=$WV/$RUN_NAME/precomputed/seg/$RUN_NAME,OUT=$WV/nerl_funlib_mt50.json,MT=50 \\
      dev/zebrafinch/sbatch_score_eval_matrix.sh
    sbatch --export=ALL,SEG=$WV/$RUN_NAME/precomputed/seg/$RUN_NAME,OUT=$WV/nerl_funlib_mt0.json,MT=0 \\
      dev/zebrafinch/sbatch_score_eval_matrix.sh
EOF

if [ "$SUBMIT" = "--submit" ]; then
  set -o pipefail
  START_AT=me_L0 WV="$WV" bash dev/zebrafinch/submit_wholevol_sharded.sh \
    | tee "$WV/slurm_submission.txt"
  echo "[submitted] transcript -> $WV/slurm_submission.txt"

  # Chain scoring behind the decode. Previously the two sbatch lines above were printed as
  # advice and never run, so every arm silently stopped at remapagg with no NERL -- the arm
  # looked "done" while its result did not exist. Both thresholds per MANUAL.md.
  FINAL_JID=$(grep -oE 'final job: [0-9]+' "$WV/slurm_submission.txt" | tail -1 | grep -oE '[0-9]+')
  if [ -z "$FINAL_JID" ]; then
    echo "[score] WARNING: no 'final job:' id in the transcript; scoring NOT chained." >&2
    echo "[score] submit manually once remapagg completes (commands in the plan above)." >&2
  else
    SEG_LAYER=$WV/$RUN_NAME/precomputed/seg/$RUN_NAME
    for MT in 50 0; do
      jid=$(sbatch --parsable --dependency=afterok:"$FINAL_JID" \
        --export=ALL,SEG="$SEG_LAYER",OUT="$WV/nerl_funlib_mt${MT}.json",MT=$MT \
        dev/zebrafinch/sbatch_score_eval_matrix.sh)
      echo "[score] mt=$MT -> $jid (afterok:$FINAL_JID) -> $WV/nerl_funlib_mt${MT}.json"
    done
  fi
else
  echo
  echo "NOT SUBMITTED. Re-run with --submit to launch."
fi
