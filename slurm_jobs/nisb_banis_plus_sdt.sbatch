#!/bin/bash
# Fresh training of nisb_base_banis_plus_sdt: banis+ (MedNeXt-L/k3) backbone
# with the v2 affinity+SDT two-head and label erosion=2.
# Config: tutorials/neuron_nisb/base_banis+_sdt.yaml
#
# Pinned to g004 (a100/vr40g) with 3 GPUs. num_gpus=-1 in the config makes
# Lightning use all visible GPUs (DDP). Memory sized to fit g004's current
# free budget so the job starts immediately rather than pending.

#SBATCH --job-name=nisb_pp_sdt
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --nodelist=g004
#SBATCH --gres=gpu:3
#SBATCH --constraint=vr40g
#SBATCH --cpus-per-task=24
#SBATCH --mem=60G
#SBATCH --time=5-00:00:00
#SBATCH --output=slurm_jobs/logs/nisb_pp_sdt_%j.out
#SBATCH --error=slurm_jobs/logs/nisb_pp_sdt_%j.err

source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
cd /projects/weilab/weidf/lib/pytorch_connectomics
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/main.py \
    --config tutorials/neuron_nisb/base_banis+_sdt.yaml
