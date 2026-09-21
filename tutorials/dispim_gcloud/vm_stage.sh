#!/bin/bash
# Runs inside a disposable Google Compute Engine VM as root.
# Usage: bash vm_stage.sh gpu|cpu gs://BUCKET/UNIQUE-RUN-PREFIX
set -Eeuo pipefail
stage=$1
gcs=$2
case "$stage" in gpu|cpu) ;; *) exit 2 ;; esac
mkdir -p /work/job /work/data/diSPIM_SNEMI /work/scratch
exec > >(tee -a /work/job/console.log) 2>&1
sync_pid=
finish() {
  status=$?
  trap - EXIT
  if [ -n "$sync_pid" ]; then kill "$sync_pid" 2>/dev/null || true; fi
  printf '%s\n' "$status" > /work/job/exit-code.txt
  gcloud storage rsync /work/job "$gcs/$stage" --recursive
  # Publish a completion marker only after successful work AND upload.
  if [ "$status" -eq 0 ]; then
    echo complete > /work/COMPLETE
    gcloud storage cp /work/COMPLETE "$gcs/$stage/COMPLETE"
  fi
  echo "STAGE FINISHED stage=$stage exit=$status"
  exit "$status"
}
trap finish EXIT
if ! command -v docker >/dev/null; then
  apt-get update
  apt-get install -y docker.io
fi
if [ "$stage" = gpu ]; then
  nvidia-ctk runtime configure --runtime=docker
fi
systemctl enable --now docker
systemctl restart docker
gcloud storage cp "$gcs/build/image.json" /work/expected-image.json
image_matches() {
  docker image inspect pytc:dispim-abiss > /work/loaded-image.json 2>/dev/null || return 1
  # Docker's classic and containerd stores expose different .Id values for
  # the same saved image. Compare its filesystem layers and runtime config.
  python3 - <<'PY'
import json
expected = json.load(open('/work/expected-image.json'))[0]
actual = json.load(open('/work/loaded-image.json'))[0]
assert actual['RootFS']['Layers'] == expected['RootFS']['Layers']
for key in ('User', 'Env', 'Entrypoint', 'Cmd', 'WorkingDir'):
    assert actual['Config'].get(key) == expected['Config'].get(key), key
PY
}
if ! image_matches; then
  gcloud storage cp "$gcs/build/image.tar.gz" /work/image.tar.gz
  gcloud storage cp "$gcs/build/image.sha256" /work/image.sha256
  (cd /work && sha256sum -c image.sha256)
  gzip -dc /work/image.tar.gz | docker load
fi
image_matches
gcloud storage rsync "$gcs/prepared" /work/data/diSPIM_SNEMI --recursive
gcloud storage cp "$gcs/config.yaml" /work/job/config.yaml
chown -R 1000:1000 /work/job /work/scratch
docker image inspect pytc:dispim-abiss > /work/job/image.json
cp /work/data/diSPIM_SNEMI/provenance.json /work/job/data-provenance.json
config=tutorials/dispim_gcloud/run.yaml
# Keep the baked shared recipe intact: finer-resolution YAMLs inherit it.
mounts=(-v /work/data:/workspace/datasets:ro -v /work/job:/workspace/outputs -v /work/scratch:/scratch -e TMPDIR=/scratch -v /work/job/config.yaml:/workspace/tutorials/dispim_gcloud/run.yaml:ro)
docker run --rm "${mounts[@]}" pytc:dispim-abiss python -c 'import json,hashlib; from pathlib import Path; p=Path("/workspace/datasets/diSPIM_SNEMI"); m=json.loads((p/"provenance.json").read_text()); assert m["status"]=="complete"; assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==h for n,h in m["files"].items()); print("Prepared data checksums verified")'
if [ "$stage" = gpu ]; then
  docker run --rm --gpus all pytc:dispim-abiss python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==4; print(torch.cuda.get_device_name())'
  # A tiny actual-data test catches pairing, target, and memory failures first.
  docker run --rm --name dispim-smoke --gpus all --ipc=host "${mounts[@]}" pytc:dispim-abiss \
    python scripts/main.py --config "$config" --mode train --fast-dev-run 1 \
    save_path=/workspace/outputs/smoke > /work/job/smoke.log 2>&1
  (
    while sleep 900; do
      stamp=$(date -u +%Y%m%d_%H%M%S)
      gcloud storage rsync /work/job "$gcs/snapshots/$stamp" --recursive || true
    done
  ) &
  sync_pid=$!
  docker run --rm --name dispim-train --gpus all --ipc=host "${mounts[@]}" pytc:dispim-abiss \
    python -m connectomics.runtime.dispim_benchmark --stage train --config "$config" --output /workspace/outputs/train
  gcloud storage cp /work/job/train/final.ckpt "$gcs/checkpoints/final.ckpt"
  gcloud storage cp /work/job/train/manifest.json "$gcs/checkpoints/manifest.json"
  # Fresh container re-establishes CUDA visibility after multi-hour training.
  docker run --rm --name dispim-infer --gpus all --ipc=host "${mounts[@]}" pytc:dispim-abiss \
    python -m connectomics.runtime.dispim_benchmark --stage infer --config "$config" \
    --output /workspace/outputs/infer --checkpoint /workspace/outputs/train/final.ckpt
else
  gcloud storage cp "$gcs/gpu/infer/affinities.h5" /work/job/affinities.h5
  gcloud storage cp "$gcs/gpu/infer/manifest.json" /work/job/infer-manifest.json
  docker run --rm --name dispim-score --ipc=host "${mounts[@]}" pytc:dispim-abiss \
    python -m connectomics.runtime.dispim_benchmark --stage score --config "$config" \
    --output /workspace/outputs/score --prediction /workspace/outputs/affinities.h5
fi
