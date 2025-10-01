# justfile for PyTorch Connectomics
# Run with: just <command>

# Default recipe to display available commands
default:
    @just --list

# Train on Lucchi dataset
train:
    python scripts/main.py --config tutorials/mednext_lucchi.yaml

# Test on Lucchi dataset (provide path to checkpoint)
test checkpoint:
    python scripts/main.py --config tutorials/mednext_lucchi.yaml --mode test --checkpoint {{checkpoint}}
