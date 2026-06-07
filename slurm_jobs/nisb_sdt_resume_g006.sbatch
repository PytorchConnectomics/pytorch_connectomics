#!/bin/bash
# Resume nisb_base_banis_v2_erosion2 (banis+ affinity + skeleton_aware_edt/SDT
# two-head) training on g006 with 2 GPUs. Resumes full trainer state
# (optimizer + scheduler + global_step) from last.ckpt via trainer.fit(ckpt_path=).
#
# NOTE: original run used 4 GPUs (effective batch 8); this resumes on 2 GPUs
# (effective batch 4). Optimizer/scheduler state restore cleanly; only the
# effective batch size differs. Pinned to g006 (a100/vr40g). Memory sized to
# fit g006's current free budget so it starts immediately.

#SBATCH --job-name=nisb_sdt_res6
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --nodelist=g006
#SBATCH --gres=gpu:2
#SBATCH --constraint=vr40g
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=5-00:00:00
#SBATCH --output=slurm_jobs/logs/nisb_sdt_res6_%j.out
#SBATCH --error=slurm_jobs/logs/nisb_sdt_res6_%j.err

source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
cd /projects/weilab/weidf/lib/pytorch_connectomics
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/main.py \
    --config tutorials/neuron_nisb/base_banis_v2_erosion2.yaml \
    --checkpoint outputs/nisb_base_banis_v2_erosion2/20260518_003417/checkpoints/last.ckpt
