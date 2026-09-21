#!/bin/bash
# Evaluate the MICrONS Pinky affinity+LSD checkpoint on the held-out test split:
# sliding-window inference -> affinity CC decode -> adapted-Rand/VOI, then the
# false-merge-free oracle ceiling.
#
# Do not use `set -u`: conda's activation scripts read optional unset variables.
#SBATCH --job-name=pinky-test
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=6:00:00
#SBATCH --exclude=g001,g002,gdw001,gdw002,gb001
#SBATCH --output=/projects/weilab/weidf/lib/pytorch_connectomics/slurm_outputs/pinky-test_%j.out
#SBATCH --error=/projects/weilab/weidf/lib/pytorch_connectomics/slurm_outputs/pinky-test_%j.err

set -eo pipefail

REPO=/projects/weilab/weidf/lib/pytorch_connectomics
WORKTREE=$REPO/.claude/worktrees/bridge-cse_0181aPnfEwx7QMuk46XqCpeh
CONFIG=$REPO/tutorials/neuron_microns_pinky.yaml
RUN=$REPO/outputs/neuron_microns_pinky_affinity_lsd/20260821_014234
CKPT=$RUN/checkpoints/step=00200000.ckpt
RESULTS=$RUN/test_step=00200000

source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc

export HDF5_USE_FILE_LOCKING=FALSE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# connectomics is not pip-installed; a script invoked by absolute path gets its
# own dir on sys.path, not the cwd, so the repo has to be on PYTHONPATH.
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}

cd "$REPO"

echo "host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

python -u scripts/main.py --config "$CONFIG" --mode test --checkpoint "$CKPT"

echo
echo "================ ORACLE-MERGE CEILING ================"
python -u "$WORKTREE/scripts/pinky_test_ceiling.py" --results-dir "$RESULTS"
