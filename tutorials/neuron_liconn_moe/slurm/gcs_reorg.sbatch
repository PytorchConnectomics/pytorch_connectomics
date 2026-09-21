#!/bin/bash
# Reorganise gs://donglai_public/liconn/moe/ into <family>/<kind>/<name>.
#
# RUNS ON A COMPUTE NODE. The login node sits at load ~75 on 7 cores with ~37
# users and reniced gcloud to 19 under a 5-CPU cap; it managed 0.65 objects/sec.
# A compute node does ~2400/sec. (Also: the claim in 3_precomputed.sbatch that
# upload needs an interactive `gcloud auth login` is wrong -- compute nodes
# resolve storage.googleapis.com and the stored credential works there.)
#
# SCALE: ~500k objects, not 40 GB. One seg layer is 41k-95k chunk objects and
# GCS has no directory rename, so each is an API call.
#
# THIS BUCKET HAS A HIERARCHICAL NAMESPACE, which drives everything below:
#   * folders are real objects, so the destination PARENT must already exist --
#     otherwise `mv` fails with "HTTPError 409: The parent folder does not exist"
#   * and when the destination folder does NOT exist, `mv` treats the path as the
#     new NAME for the source rather than a container, so the layer's contents
#     (info, provenance, mesh_*, and one folder per resolution scale) land loose
#     in the parent. That happened once on 2026-09-16 and is repaired below.
# Both are fixed by creating every destination folder up front and only ever
# moving into an existing container.
#
# `mv` takes NO recursion flag (unlike cp's -r); it recurses implicitly.
#SBATCH --job-name=gcs_reorg
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/projects/weilab/weidf/lib/pytorch_connectomics/slurm_logs/gcs_reorg_%j.out

set -uo pipefail
G=~/google-cloud-sdk/bin/gcloud
SRC=gs://donglai_public/liconn/moe/clip_percentile_1_99
DST=gs://donglai_public/liconn/moe
export CLOUDSDK_STORAGE_PROCESS_COUNT=16
export CLOUDSDK_STORAGE_THREAD_COUNT=16

echo "host=$(hostname) cores=$(nproc) start=$(date -Is)"

echo "=== 0. create every destination folder up front ==="
for f in expid96 expid99; do
  $G storage folders create "$DST/$f/" 2>/dev/null
  for k in image mip1_eb2 mip1_eb8 mip0_eb8; do
    $G storage folders create "$DST/$f/$k/" 2>/dev/null && echo "  created $f/$k/"
  done
done

echo "=== 1. repair the layer whose contents were spilled into mip1_eb2/ ==="
SPILL="$DST/expid96/mip1_eb2"
LAYER="$SPILL/ExPID96_2ndgel_S1_40XW001_18x_seg_abiss_mt060"
if $G storage ls "$SPILL/info" >/dev/null 2>&1; then
  $G storage folders create "$LAYER/" 2>/dev/null
  for item in info provenance mesh_mip_0_err_40 \
              144.444448_144.444448_177.777776 \
              18.055556_18.055556_22.222222 \
              36.111112_36.111112_44.444444 \
              72.222224_72.222224_88.888888; do
    if $G storage ls "$SPILL/$item" >/dev/null 2>&1; then
      printf '  %s -> layer/  ' "$item"
      timeout 3600 $G storage mv "$SPILL/$item" "$LAYER/" >/dev/null 2>&1 \
        && echo OK || echo FAILED
    fi
  done
else
  echo "  nothing spilled; skipping"
fi

echo "=== 2. move the remaining entries ==="
$G storage ls "$SRC/" 2>/dev/null | sed "s|.*/clip_percentile_1_99/||; s|/$||" | grep -v '^$' > /tmp/moe_entries.$$
echo "remaining entries: $(wc -l < /tmp/moe_entries.$$)"

while read -r name; do
  case "$name" in
    ExPID96*) family=expid96 ;;
    ExPID99*) family=expid99 ;;
    *) echo "SKIP unknown family: $name"; continue ;;
  esac
  case "$name" in
    *.zarr)        kind=image ;;
    *_seg_abiss_*) kind=mip1_eb2 ;;
    *) echo "SKIP unknown kind: $name"; continue ;;
  esac
  printf '%s -> %s/%s/  ' "$name" "$family" "$kind"
  t0=$SECONDS
  if timeout 3600 $G storage mv "$SRC/$name" "$DST/$family/$kind/" >/dev/null; then
    echo "OK ($((SECONDS-t0))s)"
  else
    echo "FAILED rc=$? ($((SECONDS-t0))s)"
  fi
done < /tmp/moe_entries.$$
rm -f /tmp/moe_entries.$$

echo "=== final layout ==="
for f in expid96 expid99; do
  for k in image mip1_eb2 mip1_eb8 mip0_eb8; do
    n=$($G storage ls "$DST/$f/$k/" 2>/dev/null | grep -c .)
    [ "$n" -gt 0 ] && echo "  $f/$k: $n entries"
  done
done
echo "=== stray non-layer items directly under any kind/ (should be none) ==="
for f in expid96 expid99; do
  for k in image mip1_eb2; do
    $G storage ls "$DST/$f/$k/" 2>/dev/null | sed "s|.*/$k/||" | grep -vE '^(ExPID|$)' | sed "s|^|  $f/$k/|"
  done
done
echo "=== residue under the old prefix (should be empty) ==="
$G storage ls "$SRC/" 2>&1 | head -5
echo "finished $(date -Is)"
