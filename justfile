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
# ============================================================================
# Visualization Commands
# ============================================================================

# Visualize volumes with Neuroglancer from config (e.g., just visualize tutorials/monai_lucchi.yaml test )
visualize config mode *ARGS='':
    python -i scripts/visualize_neuroglancer.py --config {{config}} --mode {{mode}} {{ARGS}}

# Visualize specific image and label files (e.g., just visualize-files datasets/img.tif datasets/label.h5)
visualize-files image label *ARGS='':
    python -i scripts/visualize_neuroglancer.py --image {{image}} --label {{label}} {{ARGS}}

# Visualize multiple volumes with custom names (e.g., just visualize-volumes image:path/img.tif label:path/lbl.h5)
visualize-volumes +volumes:
    python -i scripts/visualize_neuroglancer.py --volumes {{volumes}}

# Visualize on custom port (e.g., just visualize-port 8080 tutorials/monai_lucchi.yaml)
visualize-port port config *ARGS='':
    python -i scripts/visualize_neuroglancer.py --config {{config}} --port {{port}} {{ARGS}}

# Visualize with remote access (use 0.0.0.0 for public IP, e.g., just visualize-remote 8080 tutorials/monai_lucchi.yaml)
visualize-remote port config *ARGS='':
    python -i scripts/visualize_neuroglancer.py --config {{config}} --ip 0.0.0.0 --port {{port}} {{ARGS}}
