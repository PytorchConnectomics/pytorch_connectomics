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
