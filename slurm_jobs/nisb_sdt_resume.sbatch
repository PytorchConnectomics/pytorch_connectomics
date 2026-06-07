#!/bin/bash
# Resume nisb_base_banis_v2_erosion2 (banis+ affinity + skeleton_aware_edt/SDT
# two-head) training. Original run (20260518_003417, 4x L40S/g016) stopped at
# global_step 70000 / 200000 on 2026-05-19. This resumes full trainer state
# (optimizer + scheduler + global_step) from last.ckpt via trainer.fit(ckpt_path=).
#
# Matches the original setup: 4 GPUs, batch 2, patch 128^3, fp16. Constrained to
# L40S (vr40g) or a100 (vr80g) so it never lands on a 16GB v100.

#SBATCH --job-name=nisb_sdt_res
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --constraint=vr40g|vr80g
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=5-00:00:00
#SBATCH --output=slurm_jobs/logs/nisb_sdt_res_%j.out
#SBATCH --error=slurm_jobs/logs/nisb_sdt_res_%j.err

source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
cd /projects/weilab/weidf/lib/pytorch_connectomics
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/main.py \
    --config tutorials/neuron_nisb/base_banis_v2_erosion2.yaml \
    --checkpoint outputs/nisb_base_banis_v2_erosion2/20260518_003417/checkpoints/last.ckpt
