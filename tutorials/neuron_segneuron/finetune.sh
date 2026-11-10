#!/usr/bin/env bash
# Fine-tune the EMNeuron-pretrained BANIS+ neuron segmentation model on your own labeled EM.
#
#   tutorials/neuron_segneuron/finetune.sh TRAIN_DIR VAL_DIR OUT_DIR [PyTC overrides...]
#
# TRAIN_DIR, VAL_DIR  directories of paired HDF5 files, one dataset each, ZYX:
#                       <name>_img.h5  EM image, uint8, membranes dark
#                       <name>_seg.h5  instance labels (one id per neuron, 0 = background or
#                                      membrane, -1 = ignore)
#                     VAL_DIR may equal TRAIN_DIR when there is no separate validation volume.
# OUT_DIR             run directory; the final checkpoint path is printed at the end.
#
# Environment (defaults): STEPS=20000  LR=0.0002  GPUS=1  BATCH=4 (per GPU)
#                         PRETRAINED=checkpoints/neuron_segneuron/banis_emneuron_200k.ckpt
# The pretrained weights are downloaded from huggingface.co/pytc/tutorial when PRETRAINED is
# missing. The run stops if they do not load completely. Then segment with predict.sh.
set -euo pipefail
[ $# -ge 3 ] || { sed -n '2,19p' "$0"; exit 2; }
TRAIN=$(realpath "$1"); VAL=$(realpath "$2"); OUT=$(realpath -m "$3"); shift 3
STEPS=${STEPS:-20000}; LR=${LR:-0.0002}; GPUS=${GPUS:-1}; BATCH=${BATCH:-4}
cd "$(dirname "$(realpath "$0")")/../.."   # PyTC root
PRETRAINED=$(realpath -m "${PRETRAINED:-checkpoints/neuron_segneuron/banis_emneuron_200k.ckpt}")

ls "$TRAIN"/*_img.h5 > /dev/null 2>&1 || { echo "no <name>_img.h5 in $TRAIN"; exit 1; }
for img in "$TRAIN"/*_img.h5 "$VAL"/*_img.h5; do
  [ -f "${img%_img.h5}_seg.h5" ] || { echo "missing label ${img%_img.h5}_seg.h5"; exit 1; }
done
if [ ! -f "$PRETRAINED" ]; then
  hf download pytc/tutorial neuron_segneuron/banis_emneuron_200k.ckpt --local-dir checkpoints
  PRETRAINED=$(realpath checkpoints/neuron_segneuron/banis_emneuron_200k.ckpt)
fi

mkdir -p "$OUT"
LOG=$OUT/finetune_$(date +%Y%m%d_%H%M%S).log
echo "fine-tuning $STEPS steps from $PRETRAINED; log $LOG"
python scripts/main.py --config tutorials/neuron_segneuron/finetune.yaml --mode train \
  model.external_weights_path="$PRETRAINED" \
  train.data.train.path="$TRAIN" train.data.val.path="$VAL" save_path="$OUT" \
  train.system.num_gpus="$GPUS" train.data.dataloader.batch_size="$BATCH" \
  train.optimization.optimizer.lr="$LR" train.optimization.max_steps="$STEPS" \
  train.optimization.scheduler.params.t_max="$STEPS" \
  train.optimization.val_check_interval="$(( STEPS < 1000 ? STEPS : 1000 ))" \
  train.monitor.checkpoint.save_every_n_steps="$STEPS" "$@" 2>&1 | tee "$LOG"

grep -q "Successfully loaded all weights" "$LOG" \
  || { echo "ERROR: pretrained weights did not load completely (see $LOG)"; exit 1; }
CKPT=$(ls -t "$OUT"/*/checkpoints/step=$(printf %08d "$STEPS").ckpt 2>/dev/null | head -1)
[ -n "$CKPT" ] || { echo "ERROR: no step=$(printf %08d "$STEPS").ckpt under $OUT"; exit 1; }
echo "fine-tuned checkpoint: $CKPT"
echo "segment with: tutorials/neuron_segneuron/predict.sh $CKPT <input dir> <output dir>"
