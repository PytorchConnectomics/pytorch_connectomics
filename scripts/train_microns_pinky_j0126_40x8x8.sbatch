#!/bin/bash
# Submit from the repository root:
#   sbatch scripts/train_microns_pinky_j0126_40x8x8.sbatch
#SBATCH --job-name=pinky-j0126-40x8
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=60
#SBATCH --mem=200G
#SBATCH --time=5-00:00:00
#SBATCH --exclude=g001,g002,gdw001,gdw002,gb001
#SBATCH --output=/projects/weilab/weidf/lib/pytorch_connectomics/slurm_outputs/pinky-j0126-40x8_%j.out
#SBATCH --error=/projects/weilab/weidf/lib/pytorch_connectomics/slurm_outputs/pinky-j0126-40x8_%j.err

set -eo pipefail

REPO=/projects/weilab/weidf/lib/pytorch_connectomics
CONFIG=$REPO/tutorials/neuron_microns_pinky_j0126_40x8x8.yaml

source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$REPO/slurm_outputs"
cd "$REPO"
python -u scripts/main.py --config "$CONFIG" --mode train
