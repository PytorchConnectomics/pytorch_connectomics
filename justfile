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

# Train on Lucchi dataset
train dataset:
    python scripts/main.py --config tutorials/monai_{{dataset}}.yaml

# Test on Lucchi dataset (provide path to checkpoint)
test dataset checkpoint:
    python scripts/main.py --config tutorials/mednext_{{dataset}}.yaml --mode test --checkpoint {{checkpoint}}

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
