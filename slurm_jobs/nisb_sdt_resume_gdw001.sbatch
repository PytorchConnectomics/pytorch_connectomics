#!/bin/bash
# Resume nisb_base_banis_v2_erosion2 (banis+ affinity + skeleton_aware_edt/SDT
# two-head, MedNeXt-S) from last.ckpt on gdw001 (4x a10, weilab partition).
# Resumes full trainer state (optimizer + scheduler + global_step) via
# trainer.fit(ckpt_path=). 4 GPUs match the original run's effective batch 8.

#SBATCH --job-name=nisb_sdt_resdw
#SBATCH --partition=weilab
#SBATCH --nodes=1
#SBATCH --nodelist=gdw001
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=5-00:00:00
#SBATCH --output=slurm_jobs/logs/nisb_sdt_resdw_%j.out
#SBATCH --error=slurm_jobs/logs/nisb_sdt_resdw_%j.err

source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
cd /projects/weilab/weidf/lib/pytorch_connectomics
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# The checkpoint is at global_step=70000 with a cosine schedule of T_max=200000.
# base_banis.yaml caps max_steps/t_max at 50000, so a plain resume exits
# immediately ("max_steps=50000 reached"). Override both to the run's true
# target of 200000 so training continues from 70k.
python scripts/main.py \
    --config tutorials/neuron_nisb/base_banis_v2_erosion2.yaml \
    --checkpoint outputs/nisb_base_banis_v2_erosion2/20260518_003417/checkpoints/last.ckpt \
    optimization.max_steps=200000 \
    optimization.scheduler.params.t_max=200000
