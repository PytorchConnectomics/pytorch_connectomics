# justfile for PyTorch Connectomics
# Run with: just <command>

# Default recipe to display available commands
default:
    @just --list

# ============================================================================
# Setup & Data
# ============================================================================

# Install PyTorch Connectomics into a conda env (default: pytc).
#   just install              # creates/uses env "pytc"
#   just install my_env       # creates/uses env "my_env"
install env_name="pytc":
    python install.py --install-type basic --python 3.11 --env-name "{{env_name}}"

# Check the active env: imports, GPU kernel support, one tiny CUDA op.
#   just verify               # human-readable; exits non-zero on failure
#   just verify --json        # machine-readable
verify *args:
    python scripts/check_install.py {{args}}

# Drive the install through a local Claude Code session (interactive).
# Prerequisite: claude CLI installed and authenticated.
install-claude:
    claude "$(cat prompts/INSTALL.md)"

# Drive the install through a local Codex CLI session (interactive).
# Prerequisite: codex CLI installed and authenticated.
install-codex:
    codex "$(cat prompts/INSTALL.md)"

# Drive "add a dataset" through Claude Code (interactive).
add-dataset-claude:
    claude "$(cat prompts/ADD_DATASET.md)"

# Drive "add a dataset" through Codex CLI (interactive).
add-dataset-codex:
    codex "$(cat prompts/ADD_DATASET.md)"

# Drive "add an architecture" through Claude Code (interactive).
add-arch-claude:
    claude "$(cat prompts/ADD_ARCH.md)"

# Drive "add an architecture" through Codex CLI (interactive).
add-arch-codex:
    codex "$(cat prompts/ADD_ARCH.md)"

# Drive "debug a failing tutorial" through Claude Code (interactive).
debug-tutorial-claude:
    claude "$(cat prompts/DEBUG_TUTORIAL.md)"

# Drive "debug a failing tutorial" through Codex CLI (interactive).
debug-tutorial-codex:
    codex "$(cat prompts/DEBUG_TUTORIAL.md)"

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

# Train (e.g., just train mito_lucchi++/mito_lucchi++, just train mito_lucchi++/mito_lucchi++)
# Uses architecture specified in tutorials/{{dataset}}.yaml
train dataset *ARGS='':
    python scripts/main.py --config tutorials/{{dataset}}.yaml {{ARGS}}

# Resume training (e.g., just resume mito_lucchi++/mito_lucchi++ ckpt.pt)
resume dataset ckpt *ARGS='':
    python scripts/main.py --config tutorials/{{dataset}}.yaml --checkpoint {{ckpt}} {{ARGS}}

# Test model (e.g., just test mito_lucchi++/mito_lucchi++ ckpt.pt)
test dataset ckpt *ARGS='':
    python scripts/main.py --config tutorials/{{dataset}}.yaml --mode test --checkpoint {{ckpt}} {{ARGS}}

# Tune decoding parameters on validation set (e.g., just tune mito_lucchi++/mito_lucchi++ ckpt.pt)
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

# Run prediction, decoding and evaluation using the test stage
infer dataset ckpt *ARGS='':
    python scripts/main.py --config tutorials/{{dataset}}.yaml --mode test --checkpoint {{ckpt}} {{ARGS}}

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
    tensorboard --logdir outputs/{{experiment}}/{{timestamp}} --port {{port}}

# Submit a command using the currently active conda environment.
# Example: just slurm gpu 8 1 "just train mito_lucchi++/mito_lucchi++"
slurm partition num_cpu num_gpu cmd mem='32G':
    #!/usr/bin/env bash
    set -euo pipefail
    : "${CONDA_PREFIX:?Activate a conda environment before submitting}"
    mkdir -p slurm_outputs
    export PYTC_ENV="$CONDA_PREFIX" PYTC_CMD='{{cmd}}'
    sbatch --partition='{{partition}}' --nodes=1 --ntasks=1 \
        --gpus-per-task='{{num_gpu}}' --cpus-per-task='{{num_cpu}}' --mem='{{mem}}' \
        --chdir="$PWD" --output='slurm_outputs/slurm-%j.out' \
        --export=ALL,PYTC_ENV,PYTC_CMD \
        --wrap='export PATH="$PYTC_ENV/bin:$PATH"; srun bash -c "$PYTC_CMD"'

# Submit a CPU command using the active conda environment.
slurm-cpu partition num_tasks cpu_per_task cmd mem='32G':
    #!/usr/bin/env bash
    set -euo pipefail
    : "${CONDA_PREFIX:?Activate a conda environment before submitting}"
    mkdir -p slurm_outputs
    export PYTC_ENV="$CONDA_PREFIX" PYTC_CMD='{{cmd}}'
    sbatch --partition='{{partition}}' --nodes=1 --ntasks='{{num_tasks}}' \
        --cpus-per-task='{{cpu_per_task}}' --mem='{{mem}}' \
        --chdir="$PWD" --output='slurm_outputs/slurm-%j.out' \
        --export=ALL,PYTC_ENV,PYTC_CMD \
        --wrap='export PATH="$PYTC_ENV/bin:$PATH"; srun bash -c "$PYTC_CMD"'

# ============================================================================
# Visualization Commands
# ============================================================================

# Visualize volumes with Neuroglancer from config (e.g., just visualize mito_lucchi++/mito_lucchi++ test --volumes prediction:path.h5)
# Port defaults to 9999. Override with: just visualize config mode --port 8080 --volumes ...
# Default selects first file from globs. Use --select to change: --select 1, --select filename, --select all
# Optional bbox shortcut (auto-expands to --bbox): just visualize config mode 0,0,0,32,256,256
visualize config mode bbox='' *ARGS='':
    #!/usr/bin/env bash
    args="--config tutorials/{{config}}.yaml --mode {{mode}}"
    extra_args="{{bbox}} {{ARGS}}"
    # Check if --port is in ARGS, otherwise add default
    if [[ ! "$extra_args" =~ --port ]]; then
        args="$args --port 9999"
    fi
    if [ -n "{{bbox}}" ]; then
        if [[ "{{bbox}}" == --* ]]; then
            # Backward compatibility: first extra CLI flag may be captured in bbox slot
            args="$args {{bbox}}"
        else
            args="$args --bbox {{bbox}}"
        fi
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
# Override the default port with `port=8080` or `--port 8080`.
visualize-volumes +volumes:
    #!/usr/bin/env bash
    set -euo pipefail
    port=9999
    args=({{volumes}})
    volume_args=()

    i=0
    while [ $i -lt ${#args[@]} ]; do
        arg="${args[$i]}"
        case "$arg" in
            port=*)
                port="${arg#port=}"
                ;;
            --port=*)
                port="${arg#--port=}"
                ;;
            --port)
                i=$((i + 1))
                if [ $i -ge ${#args[@]} ]; then
                    echo "ERROR: --port requires a value" >&2
                    exit 1
                fi
                port="${args[$i]}"
                ;;
            *)
                volume_args+=("$arg")
                ;;
        esac
        i=$((i + 1))
    done

    if [ ${#volume_args[@]} -eq 0 ]; then
        echo "ERROR: At least one volume spec is required" >&2
        exit 1
    fi

    python -i scripts/visualize_neuroglancer.py --port "$port" --volumes "${volume_args[@]}"

# Remote viewer via SSH: ssh -L 9999:127.0.0.1:9999 user@server
# Example: just visualize-remote 9999 mito_lucchi++/mito_lucchi++
visualize-remote port config *ARGS='':
    python -i scripts/visualize_neuroglancer.py --config tutorials/{{config}}.yaml --bind-address 127.0.0.1 --port {{port}} {{ARGS}}
