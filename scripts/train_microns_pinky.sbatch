#!/bin/bash
# Train the MICrONS Pinky affinity+LSD model on four L40S/A100 GPUs.
# Submit from the repository root:
#   sbatch scripts/train_microns_pinky.sbatch
#
# Do not use `set -u`: conda's activation scripts read optional unset variables.
#SBATCH --job-name=pinky-aff-lsd
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=60
#SBATCH --mem=200G
#SBATCH --time=5-00:00:00
#SBATCH --exclude=g001,g002,gdw001,gdw002,gb001
#SBATCH --output=/projects/weilab/weidf/lib/pytorch_connectomics/slurm_outputs/pinky-aff-lsd_%j.out
#SBATCH --error=/projects/weilab/weidf/lib/pytorch_connectomics/slurm_outputs/pinky-aff-lsd_%j.err

set -eo pipefail

REPO=/projects/weilab/weidf/lib/pytorch_connectomics
CONFIG=$REPO/tutorials/neuron_microns_pinky.yaml

source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc

export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$REPO/slurm_outputs"
cd "$REPO"

echo "host=$(hostname) job=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Lightning owns the four local DDP workers. An outer one-task srun would make
# Lightning's SLURM environment disagree with the requested device count.
python -u scripts/main.py --config "$CONFIG" --mode train
