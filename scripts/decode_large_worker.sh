#!/bin/bash
#SBATCH --job-name=waterz_worker
#SBATCH --mem=150G
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --output=slurm_outputs/waterz_worker_%A_%a.out
#SBATCH --error=slurm_outputs/waterz_worker_%A_%a.err

# Usage:
#   sbatch --array=0-25 scripts/decode_large_worker.sh tutorials/waterz_decoding_large.yaml
#
# Worker N decodes chunk N directly — no task competition, no race conditions.

CONFIG=${1:-tutorials/waterz_decoding_large.yaml}

source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
cd /projects/weilab/weidf/lib/pytorch_connectomics

export CCACHE_DISABLE=1

# Force NFS cache refresh so all nodes see the latest witty .so files
ls -la ~/.cache/witty/ > /dev/null 2>&1

echo "Worker ${SLURM_ARRAY_TASK_ID} of ${SLURM_ARRAY_TASK_COUNT} on $(hostname)"
echo "Config: ${CONFIG}"
echo "Start: $(date)"

python scripts/decode_large.py \
    --config ${CONFIG} \
    --chunk-index ${SLURM_ARRAY_TASK_ID}

echo "End: $(date)"
