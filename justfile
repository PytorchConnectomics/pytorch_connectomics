# justfile for PyTorch Connectomics
# Run with: just <command>

# Default recipe to display available commands
default:
    @just --list

# ============================================================================
# Setup & Data
# ============================================================================

# Setup SLURM environment: detect CUDA/cuDNN and install PyTorch with correct versions
setup-slurm:
    bash scripts/setup_slurm.sh

# Download dataset(s) (e.g., just download lucchi++, just download all)
# Available: lucchi++, snemi, mitoem, cremi
download +datasets:
    python scripts/download_data.py {{datasets}}

# List available datasets
download-list:
    python scripts/download_data.py --list

# ============================================================================
# Training Commands
# ============================================================================

# Train (e.g., just train fiber, just train lucchi++ monai_unet -- system.training.num_gpus=1)
# If arch is omitted, uses default from config. Architecture options: monai_unet, monai_basic_unet3d, rsunet, mednext
train dataset arch='' *ARGS='':
    #!/usr/bin/env bash
    if [ -z "{{arch}}" ]; then
        python scripts/main.py --config tutorials/{{dataset}}.yaml {{ARGS}}
    else
        python scripts/main.py --config tutorials/{{dataset}}.yaml model.architecture={{arch}} {{ARGS}}
    fi

# Resume training (e.g., just resume fiber ckpt.pt, just resume lucchi++ monai_unet ckpt.pt)
resume dataset arch_or_ckpt ckpt_or_args='' *ARGS='':
    #!/usr/bin/env bash
    # Check if second arg looks like a checkpoint path
    if [[ "{{arch_or_ckpt}}" == *.ckpt ]] || [[ "{{arch_or_ckpt}}" == */* ]]; then
        python scripts/main.py --config tutorials/{{dataset}}.yaml --checkpoint {{arch_or_ckpt}} {{ckpt_or_args}} {{ARGS}}
    else
        python scripts/main.py --config tutorials/{{dataset}}.yaml model.architecture={{arch_or_ckpt}} --checkpoint {{ckpt_or_args}} {{ARGS}}
    fi

# Test model (e.g., just test fiber ckpt.pt, just test lucchi++ monai_unet ckpt.pt)
test dataset arch_or_ckpt ckpt_or_args='' *ARGS='':
    #!/usr/bin/env bash
    # Check if second arg looks like a checkpoint path
    if [[ "{{arch_or_ckpt}}" == *.ckpt ]] || [[ "{{arch_or_ckpt}}" == */* ]]; then
        python scripts/main.py --config tutorials/{{dataset}}.yaml --mode test --checkpoint {{arch_or_ckpt}} {{ckpt_or_args}} {{ARGS}}
    else
        python scripts/main.py --config tutorials/{{dataset}}.yaml model.architecture={{arch_or_ckpt}} --mode test --checkpoint {{ckpt_or_args}} {{ARGS}}
    fi

# Tune decoding parameters on validation set (e.g., just tune fiber ckpt.pt)
tune dataset ckpt *ARGS='':
    python scripts/main.py --config tutorials/{{dataset}}.yaml --mode tune --checkpoint {{ckpt}} {{ARGS}}

# Tune parameters then test (recommended for optimal results)
tune-test dataset ckpt *ARGS='':
    python scripts/main.py --config tutorials/{{dataset}}.yaml --mode tune-test --checkpoint {{ckpt}} {{ARGS}}

# Quick tuning with 20 trials (for testing)
tune-quick dataset ckpt *ARGS='':
    python scripts/main.py --config tutorials/{{dataset}}.yaml --mode tune --checkpoint {{ckpt}} --tune-trials 20 {{ARGS}}

# Test with specific parameter file
test-with-params dataset ckpt params *ARGS='':
    python scripts/main.py --config tutorials/{{dataset}}.yaml --mode test --checkpoint {{ckpt}} --params {{params}} {{ARGS}}

# Inference (alias for test, clearer naming)
infer dataset ckpt *ARGS='':
    python scripts/main.py --config tutorials/{{dataset}}.yaml --mode infer --checkpoint {{ckpt}} {{ARGS}}

# ============================================================================
# Monitoring Commands
# ============================================================================

# Launch TensorBoard for a specific experiment (e.g., just tensorboard lucchi_monai_unet)
# Shows all runs (timestamped directories) for comparison
# Usage: just tensorboard experiment [port] (default port: 6006)
tensorboard experiment port='6006':
    tensorboard --logdir outputs/{{experiment}} --port {{port}}

# Launch TensorBoard for all experiments
# Usage: just tensorboard-all [port] (default port: 6006)
tensorboard-all port='6006':
    tensorboard --logdir outputs/ --port {{port}}

# Launch TensorBoard for a specific run (e.g., just tensorboard-run lucchi_monai_unet 20250203_143052)
# Usage: just tensorboard-run experiment timestamp [port] (default port: 6006)
tensorboard-run experiment timestamp port='6006':
    tensorboard --logdir outputs/{{experiment}}/{{timestamp}}/logs --port {{port}}

# Launch any just command on SLURM (e.g., just slurm weilab 8 4 "train lucchi")
# Automatically uses srun for distributed training when num_gpu > 1
slurm partition num_cpu num_gpu cmd constraint='':
    #!/usr/bin/env bash
    # Configure for multi-GPU training with PyTorch Lightning DDP
    # Set ntasks=num_gpu and use srun to launch DDP processes
    # SLURM will set CUDA_VISIBLE_DEVICES for each task automatically

    constraint_flag=""
    if [ -n "{{constraint}}" ]; then
        constraint_flag="--constraint={{constraint}}"
    fi

    sbatch --job-name="pytc_{{cmd}}" \
           --partition={{partition}} \
           --output=slurm_outputs/slurm-%j.out \
           --error=slurm_outputs/slurm-%j.err \
           --nodes=1 \
           --ntasks={{num_gpu}} \
           --gpus-per-node={{num_gpu}} \
           --cpus-per-task={{num_cpu}} \
           --mem=32G \
           --time=48:00:00 \
           $constraint_flag \
           --wrap="mkdir -p \$HOME/.just && export JUST_TEMPDIR=\$HOME/.just TMPDIR=\$HOME/.just && source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc && cd $PWD && srun --ntasks={{num_gpu}} --ntasks-per-node={{num_gpu}} just {{cmd}}"

# Launch parameter sweep from config (e.g., just sweep tutorials/sweep_example.yaml)
sweep config:
    python scripts/slurm_launcher.py --config {{config}}
# ============================================================================
# Visualization Commands
# ============================================================================

# Visualize volumes with Neuroglancer from config (e.g., just visualize tutorials/monai_lucchi.yaml test --volumes prediction:path.h5)
# Port defaults to 9999. Override with: just visualize config mode --port 8080 --volumes ...
# Default selects first file from globs. Use --select to change: --select 1, --select filename, --select all
visualize config mode *ARGS='':
    #!/usr/bin/env bash
    args="--config {{config}} --mode {{mode}}"
    # Check if --port is in ARGS, otherwise add default
    if [[ ! "{{ARGS}}" =~ --port ]]; then
        args="$args --port 9999"
    fi
    python -i scripts/visualize_neuroglancer.py $args {{ARGS}}

# Visualize specific image and label files (e.g., just visualize-files datasets/img.tif datasets/label.h5)
# If image and label are not provided (empty), no volumes will be loaded
visualize-files image='' label='' port='9999' *ARGS='':
    #!/usr/bin/env bash
    args="--port {{port}}"
    [[ -n "{{image}}" ]] && args="$args --image {{image}}"
    [[ -n "{{label}}" ]] && args="$args --label {{label}}"
    python -i scripts/visualize_neuroglancer.py $args {{ARGS}}

# Visualize multiple volumes with custom names (e.g., just visualize-volumes image:path/img.tif label:path/lbl.h5)
visualize-volumes +volumes:
    python -i scripts/visualize_neuroglancer.py --volumes {{volumes}}

# Visualize with remote access (use 0.0.0.0 for public IP, e.g., just visualize-remote 8080 tutorials/monai_lucchi.yaml)
visualize-remote port config *ARGS='':
    python -i scripts/visualize_neuroglancer.py --config {{config}} --ip 0.0.0.0 --port {{port}} {{ARGS}}
