# justfile for PyTorch Connectomics
# Run with: just <command>

# Default recipe to display available commands
default:
    @just --list

# ============================================================================
# SLURM/HPC Setup
# ============================================================================

# Setup SLURM environment: detect CUDA/cuDNN and install PyTorch with correct versions
setup-slurm:
    bash connectomics/utils/setup_slurm.sh

# ============================================================================
# Training Commands
# ============================================================================

# Train on Lucchi dataset (use '+' to pass extra args: just train monai lucchi -- data.batch_size=8)
train model dataset *ARGS='':
    python scripts/main.py --config tutorials/{{model}}_{{dataset}}.yaml {{ARGS}}

# Continue training from checkpoint (use '+' for extra args: just resume monai lucchi ckpt.pt -- --reset-optimizer)
resume model dataset checkpoint *ARGS='':
    python scripts/main.py --config tutorials/{{model}}_{{dataset}}.yaml --checkpoint {{checkpoint}} {{ARGS}}

# Test on Lucchi dataset (provide path to checkpoint)
test model dataset checkpoint *ARGS='':
    python scripts/main.py --config tutorials/{{model}}_{{dataset}}.yaml --mode test --checkpoint {{checkpoint}} {{ARGS}}

# ============================================================================
# Monitoring Commands
# ============================================================================

# Launch TensorBoard for a specific experiment (e.g., just tensorboard lucchi_monai_unet)
# Shows all runs (timestamped directories) for comparison
tensorboard experiment:
    tensorboard --logdir outputs/{{experiment}} --port 6006

# Launch TensorBoard for all experiments
tensorboard-all:
    tensorboard --logdir outputs/ --port 6006

# Launch TensorBoard for a specific run (e.g., just tensorboard-run lucchi_monai_unet 20250203_143052)
tensorboard-run experiment timestamp:
    tensorboard --logdir outputs/{{experiment}}/{{timestamp}}/logs --port 6006

# Launch any just command on SLURM (e.g., just slurm weilab 8 4 "train lucchi")
slurm partition num_cpu num_gpu cmd:
    #!/usr/bin/env bash
    sbatch --job-name="pytc_{{cmd}}" \
           --partition={{partition}} \
           --output=slurm_outputs/slurm-%j.out \
           --error=slurm_outputs/slurm-%j.err \
           --nodes=1 \
           --gpus-per-node={{num_gpu}} \
           --cpus-per-task={{num_cpu}} \
           --mem=32G \
           --time=48:00:00 \
           --wrap="source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc && cd $PWD && just {{cmd}}"

# Launch parameter sweep from config (e.g., just sweep tutorials/sweep_example.yaml)
sweep config:
    python scripts/slurm_launcher.py --config {{config}}