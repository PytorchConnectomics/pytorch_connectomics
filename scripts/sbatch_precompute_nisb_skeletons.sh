#!/bin/bash
# SLURM array job: precompute kimimaro skeleton volumes for NISB GT seg.
#
# Architecture (post-refactor):
#   - One SLURM array task per volume (5 train + 1 val = 6 tasks).
#   - Each task spawns N local workers via ProcessPoolExecutor.
#   - Workers are fully independent: each reads its own input crop,
#     skeletonizes, rasterizes, and writes its slice into the shared output.
#   - NISB output is a zarr sub-key (seg_skeleton), and zarr stores each
#     chunk as its own file -- so concurrent writes from N workers to
#     disjoint processor chunks are natively safe with no central writer.
#     (For HDF5 outputs the processor falls back to fcntl-locked serial
#     writes; libhdf5 isn't multi-process safe.)
#
# Usage:
#     mkdir -p slurm_outputs
#     sbatch scripts/sbatch_precompute_nisb_skeletons.sh
#
# Add test/seed101 by extending one of the --label globs and bumping
# --array=0-N accordingly.
#
# Resume: just re-run sbatch. The ResumeManifest at <output>.chunks.json
# skips already-completed chunks. Do NOT pass --overwrite unless you want
# to recompute from scratch.

#SBATCH --job-name=nisb_skel
#SBATCH --partition=short
#SBATCH --array=0-5
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_outputs/nisb_skel_%A_%a.out
#SBATCH --error=slurm_outputs/nisb_skel_%A_%a.err

source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
cd /projects/weilab/weidf/lib/pytorch_connectomics
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# Defensive: disable libhdf5's own file lock (returns EAGAIN under
# contention). The processor sets this too; setting it here makes the
# behavior visible. Zarr outputs ignore this; HDF5 outputs rely on it
# plus the processor's own fcntl.flock for write serialization.
export HDF5_USE_FILE_LOCKING=FALSE

python scripts/precompute_skeleton_volumes.py \
    --label '/projects/weilab/dataset/nisb/base/train/seed*/data.zarr/seg' \
    --label '/projects/weilab/dataset/nisb/base/val/seed*/data.zarr/seg'  \
    --resolution 9 9 20 \
    --chunk-shape 512 512 256 \
    --parallel 8 \
    --num-shards "${SLURM_ARRAY_TASK_COUNT}" \
    --shard-index "${SLURM_ARRAY_TASK_ID}"
