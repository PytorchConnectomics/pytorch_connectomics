# justfile for PyTorch Connectomics
# Run with: just <command>

# Default recipe to display available commands
default:
    @just --list

# Resolve SLURM time limit for a partition (fallback to sensible defaults).
_slurm-time-limit partition:
    #!/usr/bin/env bash
    set -euo pipefail
    time_limit=$(sinfo -p {{partition}} -h -o "%l" | head -1)
    if [ -z "$time_limit" ] || [ "$time_limit" = "infinite" ]; then
        case "{{partition}}" in
            short|interactive)
                time_limit="12:00:00"
                ;;
            medium)
                time_limit="2-00:00:00"
                ;;
            long)
                time_limit="5-00:00:00"
                ;;
            *)
                time_limit="7-00:00:00"
                ;;
        esac
    fi
    echo "$time_limit"

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

# Train (e.g., just train fiber, just train lucchi++ -- system.num_gpus=1)
# Uses architecture specified in tutorials/{{dataset}}.yaml
train dataset *ARGS='':
    python scripts/main.py --config tutorials/{{dataset}}.yaml {{ARGS}}

# Resume training (e.g., just resume fiber ckpt.pt)
resume dataset ckpt *ARGS='':
    python scripts/main.py --config tutorials/{{dataset}}.yaml --checkpoint {{ckpt}} {{ARGS}}

# Test model (e.g., just test fiber ckpt.pt)
test dataset ckpt *ARGS='':
    python scripts/main.py --config tutorials/{{dataset}}.yaml --mode test --checkpoint {{ckpt}} {{ARGS}}

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

# Train CellMap models (e.g., just train-cellmap cos7)
train-cellmap dataset *ARGS='':
    python scripts/cellmap/train_cellmap.py --config tutorials/cellmap_{{dataset}}.yaml {{ARGS}}

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

# Launch a command on SLURM (command is run exactly as provided).
# Examples:
#   just slurm long 8 4 "just train mitoEM/H" vr40g
#   just slurm short 8 4 "python scripts/main.py --config tutorials/lucchi.yaml"
#   just slurm short 8 4 "just train lucchi++" "" "64G"    # override memory
#   just slurm medium 8 2 "just train vesicle_xm" vr144g 128G gb001  # pin node
#   just slurm medium 12 3 "just train neuron_snemi" "" 100G g006 weidf_effort4
# Time limits: short=12h, medium=2d, long=5d
# CPU-only convenience wrapper for single-task jobs.
#   just slurm short 8 0 "python scripts/downsample_nisb.py --splits train"
slurm partition num_cpu num_gpu cmd constraint='' mem='32G' nodelist='' reservation='' exclude='':
    #!/usr/bin/env bash
    constraint_flag=""
    if [ -n "{{constraint}}" ]; then
        constraint_flag="--constraint={{constraint}}"
    fi

    nodelist_flag=""
    if [ -n "{{nodelist}}" ]; then
        nodelist_flag="--nodelist={{nodelist}}"
    fi

    reservation_flag=""
    if [ -n "{{reservation}}" ]; then
        reservation_flag="--reservation={{reservation}}"
    fi

    exclude_flag=""
    if [ -n "{{exclude}}" ]; then
        exclude_flag="--exclude={{exclude}}"
    fi

    # Resolve partition time limit (with fallback defaults)
    time_limit=$(just _slurm-time-limit {{partition}})

    # Sanitize job name: take first 2 words, replace non-alphanumeric with _
    job_name=$(echo "pytc_{{cmd}}" | awk '{print $1"_"$2"_"$3}' | tr -c '[:alnum:]_.\n' '_' | head -c 64)

    # Pass command via env var to avoid quoting issues with long --wrap strings.
    export PYTC_CMD="{{cmd}}"
    sbatch --job-name="$job_name" \
           --partition={{partition}} \
           --output=slurm_outputs/slurm-%j.out \
           --error=slurm_outputs/slurm-%j.err \
           --nodes=1 \
           --ntasks=1 \
           --gpus-per-task={{num_gpu}} \
           --cpus-per-task={{num_cpu}} \
           --mem={{mem}} \
           --time=$time_limit \
           $constraint_flag \
           $nodelist_flag \
           $reservation_flag \
           $exclude_flag \
           --export=ALL,PYTC_CMD \
           --wrap='mkdir -p $HOME/.just && export JUST_TEMPDIR=$HOME/.just TMPDIR=$HOME/.just NCCL_SOCKET_FAMILY=AF_INET && unset SLURM_CPU_BIND && source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc && cd '"$PWD"' && srun --cpu-bind=none --ntasks=1 --gpus-per-task={{num_gpu}} --cpus-per-task={{num_cpu}} $PYTC_CMD'

# Generic CPU-only multi-task launcher (single node, no GPU).
# Example:
slurm-cpu partition num_tasks='7' cpu_per_task='4' cmd='' constraint='' mem='64G':
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p slurm_outputs
    cmd_value='{{cmd}}'
    if [ -z "$cmd_value" ]; then
        echo "Error: cmd must be provided. Usage:"
        echo "  just slurm-cpu <partition> <num_tasks> <cpu_per_task> \"<command>\" [constraint] [mem]"
        exit 2
    fi

    constraint_value='{{constraint}}'
    constraint_flag=""
    if [ -n "$constraint_value" ]; then
        constraint_flag="--constraint=$constraint_value"
    fi

    # Resolve partition time limit (with fallback defaults)
    time_limit=$(just _slurm-time-limit {{partition}})

    sbatch --job-name="pytc_cpu_{{num_tasks}}t" \
           --partition={{partition}} \
           --output=slurm_outputs/slurm-%j.out \
           --error=slurm_outputs/slurm-%j.err \
           --nodes=1 \
           --ntasks={{num_tasks}} \
           --gpus-per-task=0 \
           --cpus-per-task={{cpu_per_task}} \
           --mem={{mem}} \
           --time=$time_limit \
           $constraint_flag \
           --wrap="mkdir -p \$HOME/.just && export JUST_TEMPDIR=\$HOME/.just TMPDIR=\$HOME/.just && unset SLURM_CPU_BIND && source /projects/weilab/weidf/lib/miniconda3/bin/activate pytc && cd $PWD && srun --cpu-bind=none --ntasks={{num_tasks}} --gpus-per-task=0 --cpus-per-task={{cpu_per_task}} bash -c '$cmd_value'"

# Generic CPU-only multi-task launcher for sharded scripts.
# Automatically appends:
#   --num-shards $SLURM_NTASKS --shard-index $SLURM_PROCID
# Example:
#   just slurm-cpu-sharded short 7 1 "python scripts/downsample_nisb.py"
slurm-cpu-sharded partition num_tasks='7' cpu_per_task='4' cmd='' constraint='' mem='64G':
    #!/usr/bin/env bash
    set -euo pipefail
    cmd_value='{{cmd}}'
    if [ -z "$cmd_value" ]; then
        echo "Error: cmd must be provided. Usage:"
        echo "  just slurm-cpu-sharded <partition> <num_tasks> <cpu_per_task> \"<command>\" [constraint] [mem]"
        exit 2
    fi
    just slurm-cpu {{partition}} {{num_tasks}} {{cpu_per_task}} "{{cmd}} --num-shards \$SLURM_NTASKS --shard-index \$SLURM_PROCID" "{{constraint}}" "{{mem}}"

# GPU-sharded test: launch N separate SLURM GPU jobs, one per shard.
# Each job gets --shard-id <i> --num-shards <N> appended to the command.
# Example:
#   just slurm-sharded short 8 1 4 "just test neuron_snemi ckpt.pt" vr40g 64G
slurm-sharded partition num_cpu num_gpu num_shards cmd constraint='' mem='32G' nodelist='' reservation='' exclude='':
    #!/usr/bin/env bash
    set -euo pipefail
    cmd_value='{{cmd}}'
    if [ -z "$cmd_value" ]; then
        echo "Error: cmd must be provided. Usage:"
        echo "  just slurm-sharded <partition> <num_cpu> <num_gpu> <num_shards> \"<command>\" [constraint] [mem] [nodelist] [reservation] [exclude]"
        exit 2
    fi
    echo "Launching {{num_shards}} GPU-sharded jobs..."
    for i in $(seq 0 $(({{num_shards}} - 1))); do
        echo "  Shard $i/{{num_shards}}"
        just slurm {{partition}} {{num_cpu}} {{num_gpu}} "$cmd_value --shard-id $i --num-shards {{num_shards}}" "{{constraint}}" "{{mem}}" "{{nodelist}}" "{{reservation}}" "{{exclude}}"
    done
    echo "All {{num_shards}} shard jobs submitted."

# Launch parameter sweep from config (e.g., just sweep tutorials/sweep_example.yaml)
sweep config:
    python scripts/slurm_launcher.py --config {{config}}
# ============================================================================
# Visualization Commands
# ============================================================================

# Visualize volumes with Neuroglancer from config (e.g., just visualize tutorials/monai_lucchi.yaml test --volumes prediction:path.h5)
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

# Visualize with remote access (use 0.0.0.0 for public IP, e.g., just visualize-remote 8080 tutorials/monai_lucchi.yaml)
visualize-remote port config *ARGS='':
    python -i scripts/visualize_neuroglancer.py --config tutorials/{{config}}.yaml --ip 0.0.0.0 --port {{port}} {{ARGS}}
