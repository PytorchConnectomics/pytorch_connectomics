#!/bin/bash
# Infer + decode + evaluate the trained betaseg banis_v0 model (MedNeXt-S, 50k
# steps) on the CORRECTED betaSeg test split (high_c2, high_c4, low_c3).
#
# Uses a SLIM checkpoint: the original Lightning checkpoint pickles
# SavePredictionConfig (removed in the schema refactor), so the stale
# hyper_parameters were stripped (state_dict only) by /tmp/clean_ckpt.py.
# The model is rebuilt from the YAML, so weights-only loading is correct.
#
# Usage:  cd /projects/weilab/weidf/lib/pytorch_connectomics
#         sbatch slurm_jobs/betaseg_v0_test.sbatch

#SBATCH --job-name=bseg_v0_test
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_jobs/logs/bseg_v0_test_%j.out
#SBATCH --error=slurm_jobs/logs/bseg_v0_test_%j.err

source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc
cd /projects/weilab/weidf/lib/pytorch_connectomics
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

python scripts/main.py \
    --config tutorials/mito_betaseg_banis_v0.yaml \
    --mode test \
    --checkpoint outputs/mito_betaseg_banis_v0/20260430_154449/checkpoints/epoch=009-slim.ckpt \
    system.num_gpus=1
