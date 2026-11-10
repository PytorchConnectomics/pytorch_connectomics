#!/usr/bin/env bash
# Segment neurons in EM volumes with the EMNeuron BANIS+ model (pretrained or fine-tuned).
#
#   tutorials/neuron_segneuron/predict.sh [CKPT] INPUT_DIR OUTPUT_DIR
#
# CKPT        a checkpoint from finetune.sh; omit it to use the pretrained model (downloaded
#             from huggingface.co/pytc/tutorial when missing).
# INPUT_DIR   <name>_img.h5 files: one 3D dataset each, ZYX, uint8 EM, membranes dark,
#             about 6-9 nm in xy and 8-50 nm in z (resample first if far outside).
# OUTPUT_DIR  writes <name>_seg.h5 (dataset "main", uint64 neuron ids) and <name>_aff.h5
#             (3 x Z x Y x X float16 affinities).
#
# Environment (defaults): TTA=0 (1: 16-view test-time augmentation, as in the reported
#                         results, ~4x slower per GPU)  GPUS=1  MERGE=0.6 (ABISS merge
#                         threshold: raise to ~0.7 if neurons merge, lower if they split)
#                         ABISS_HOME (required: a build of github.com/PytorchConnectomics/ABISS)
set -euo pipefail
if [ $# -eq 2 ]; then CKPT=""; IN=$1; OUT=$2
elif [ $# -eq 3 ]; then CKPT=$1; IN=$2; OUT=$3
else sed -n '2,17p' "$0"; exit 2; fi
: "${ABISS_HOME:?set ABISS_HOME to an ABISS build with build/ws}"
TTA=${TTA:-0}; GPUS=${GPUS:-1}; MERGE=${MERGE:-0.6}
IN=$(realpath "$IN"); OUT=$(realpath -m "$OUT")
cd "$(dirname "$(realpath "$0")")/../.."   # PyTC root
if [ -z "$CKPT" ]; then
  CKPT=checkpoints/neuron_segneuron/banis_emneuron_200k.ckpt
  [ -f "$CKPT" ] || hf download pytc/tutorial neuron_segneuron/banis_emneuron_200k.ckpt --local-dir checkpoints
fi
CKPT=$(realpath "$CKPT")
ls "$IN"/*_img.h5 > /dev/null 2>&1 || { echo "no <name>_img.h5 in $IN"; exit 1; }

# PyTC writes test outputs to <checkpoint dir>/../test_<checkpoint name>/; pin that location.
W=$OUT/.work; mkdir -p "$W/model/checkpoints"; ln -sfn "$CKPT" "$W/model/checkpoints/model.ckpt"
OPTS=()
if [ "$TTA" = 1 ]; then
  OPTS=(default.inference.test_time_augmentation.enabled=true
        default.inference.test_time_augmentation.flip_axes=all
        'default.inference.test_time_augmentation.rotation90_axes=[[1,2]]'
        'default.inference.test_time_augmentation.rotate90_k=[0,1]'
        default.inference.test_time_augmentation.patch_first_local=false
        test.system.num_gpus="$GPUS"); X=16
else X=1; fi

for img in "$IN"/*_img.h5; do
  name=$(basename "$img" _img.h5)
  mkdir -p "$W/in/$name"; ln -sfn "$img" "$W/in/$name/"
  echo "== $name: predict (x$X)"
  python scripts/main.py --config tutorials/neuron_segneuron/banis+.yaml --mode test \
    --checkpoint "$W/model/checkpoints/model.ckpt" test.data.test.path="$W/in/$name" test.data.test.label=null "${OPTS[@]}" \
    > "$W/$name.predict.log" 2>&1 || { tail -20 "$W/$name.predict.log"; exit 1; }
  pred=$W/model/test_model/${name}_img/raw_x${X}_ch0-1-2_weights-ema.h5
  [ -f "$pred" ] || { echo "no prediction at $pred (see $W/$name.predict.log)"; exit 1; }
  # Volumes thinner than the 32-section window come back padded; crop to the input grid.
  python - "$img" "$pred" "$OUT/${name}_aff.h5" <<'EOF'
import h5py, sys
with h5py.File(sys.argv[1], "r") as f:
    shape = f[list(f)[0]].shape
with h5py.File(sys.argv[2], "r") as f:
    d = f[list(f)[0]]
    sl = tuple(slice((p - s) // 2, (p - s) // 2 + s) for p, s in zip(d.shape[1:], shape))
    a = d[(slice(None),) + sl]
with h5py.File(sys.argv[3], "w") as f:
    f.create_dataset("main", data=a, compression="gzip")
print("affinities", a.shape, "for input", shape)
EOF
  echo "== $name: decode (ABISS, merge $MERGE)"
  python -m connectomics.decoding.abiss_runner --abiss-home "$ABISS_HOME" \
    --input "$OUT/${name}_aff.h5" --output "$OUT/${name}_seg.h5" \
    --ws-high-threshold 0.9 --ws-low-threshold 0.05 --ws-size-threshold 1000000000 \
    --ws-dust-threshold 0 --ws-merge-function max --ws-merge-threshold "$MERGE" \
    > "$W/$name.decode.log" 2>&1 || { tail -20 "$W/$name.decode.log"; exit 1; }
  echo "wrote $OUT/${name}_seg.h5"
done
