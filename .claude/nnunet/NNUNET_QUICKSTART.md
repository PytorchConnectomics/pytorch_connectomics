# nnUNet Integration Quick Start Guide

**Goal**: Use pre-trained nnUNet models for large-scale inference in PyTorch Connectomics

---

## 📦 Prerequisites

```bash
# 1. Install PyTorch Connectomics with all dependencies
pip install -e .[full]

# 2. Install nnUNet v2
pip install nnunetv2

# 3. Verify installation
python -c "import nnunetv2; print('nnUNet v2 installed successfully')"
```

---

## ⚡ Quick Start (3 Steps)

### Step 1: Prepare Your Model

You need 3 files from your trained nnUNet model:
```
model_directory/
├── checkpoint.pth       # Model weights (required)
├── plans.json          # Architecture config (auto-detected)
└── dataset.json        # Dataset metadata (auto-detected)
```

**Example**: `/projects/weilab/liupeng/mito_2d_semantic_model/checkpoints/`

---

### Step 2: Create Config File

Create `my_inference.yaml`:

```yaml
# Minimal config for nnUNet inference
system:
  num_gpus: 1
  num_cpus: 4

model:
  architecture: nnunet
  nnunet_checkpoint: /path/to/checkpoint.pth
  # plans.json and dataset.json auto-detected in same directory

data:
  do_2d: true  # Set to false for 3D models

inference:
  # Volume-by-volume processing
  volume_mode:
    enabled: true
    file_pattern: "/path/to/data/*.h5"  # Or use file_list: path/to/filelist.txt
    output_suffix: prediction

  # Where to save results
  data:
    output_path: /path/to/output/

  # Sliding window (for large images)
  sliding_window:
    window_size: [384, 512]  # From nnUNet plans
    overlap: 0.5

  # Test-time augmentation (optional, improves accuracy)
  test_time_augmentation:
    flip_axes: [[1], [2], [1, 2]]  # H, V, both
    act: softmax
    select_channel: 1  # Foreground class

  # Post-processing (optional)
  postprocessing:
    intensity_scale: 255.0
    intensity_dtype: uint8
```

---

### Step 3: Run Inference

```bash
# Single GPU inference
python scripts/main.py \
  --config my_inference.yaml \
  --mode infer-volume

# Multi-GPU inference (4 GPUs)
for GPU_ID in {0..3}; do
  python scripts/main.py \
    --config my_inference.yaml \
    --mode infer-volume \
    inference.volume_mode.start_index=$GPU_ID \
    inference.volume_mode.step=4 \
    system.device=cuda:$GPU_ID &
done
wait
```

**Output**:
```
Processing 1000 files...
Skipped 427 existing files
Progress: [████████░░] 573/1000 (57.3%)
ETA: 2.5 hours

✓ Saved: /output/volume_001_prediction.h5
✓ Saved: /output/volume_002_prediction.h5
...
```

---

## 📋 Common Use Cases

### 1. Basic 2D Inference (No TTA)

```yaml
inference:
  volume_mode:
    enabled: true
    file_pattern: "/data/*.tiff"

  sliding_window:
    window_size: [512, 512]
    overlap: 0.5

  test_time_augmentation:
    flip_axes: null  # Disable TTA (faster)
```

### 2. High-Accuracy Inference (With TTA)

```yaml
inference:
  test_time_augmentation:
    flip_axes: "all"  # All 8 flip combinations (slower, more accurate)
    ensemble_mode: mean
    act: softmax
```

### 3. Instance Segmentation

```yaml
inference:
  postprocessing:
    binary:
      enabled: true
      threshold: 0.5
      remove_small_objects: 100

  decoding:
    - name: decode_binary_watershed
      kwargs:
        min_seed_size: 32
```

### 4. Resume Interrupted Job

```bash
# Just rerun the same command - it automatically skips existing files
python scripts/main.py --config my_inference.yaml --mode infer-volume
# Output: "Skipped 427/1000 existing files"
```

---

## 🎛️ Configuration Reference

### Required Fields

| Field | Description | Example |
|-------|-------------|---------|
| `model.architecture` | Must be "nnunet" | `nnunet` |
| `model.nnunet_checkpoint` | Path to .pth file | `/path/to/checkpoint.pth` |
| `inference.volume_mode.enabled` | Enable volume mode | `true` |
| `inference.volume_mode.file_pattern` | Input files (glob) | `/data/*.h5` |
| `inference.data.output_path` | Output directory | `/output/` |

### Optional Fields

| Field | Default | Description |
|-------|---------|-------------|
| `model.nnunet_plans` | Auto-detect | Path to plans.json |
| `model.nnunet_dataset` | Auto-detect | Path to dataset.json |
| `inference.volume_mode.start_index` | 0 | Starting file index |
| `inference.volume_mode.step` | 1 | Process every Nth file |
| `inference.sliding_window.window_size` | From plans | Tile size |
| `inference.sliding_window.overlap` | 0.5 | Tile overlap ratio |
| `inference.test_time_augmentation.flip_axes` | `null` | TTA augmentations |

---

## 🚀 Performance Tips

### 1. Speed vs Accuracy Tradeoff

| Configuration | Speed | Accuracy | Memory |
|---------------|-------|----------|--------|
| No TTA, no post-processing | ⚡⚡⚡ | ⭐⭐ | 💾 |
| TTA (4×), no post-processing | ⚡⚡ | ⭐⭐⭐ | 💾💾 |
| TTA (8×) + post-processing | ⚡ | ⭐⭐⭐⭐ | 💾💾💾 |

### 2. Memory Optimization

```yaml
# For large images, reduce sw_batch_size
inference:
  sliding_window:
    sw_batch_size: 2  # Default: 4
    overlap: 0.5
```

### 3. Distributed Processing

```bash
# SLURM array job
sbatch <<EOF
#!/bin/bash
#SBATCH --array=0-3
#SBATCH --gpus-per-task=1

python scripts/main.py \
  --config my_inference.yaml \
  --mode infer-volume \
  inference.volume_mode.start_index=\$SLURM_ARRAY_TASK_ID \
  inference.volume_mode.step=4
EOF
```

---

## 🐛 Troubleshooting

### Issue: "Cannot find checkpoint file"

```bash
# Check file exists
ls -lh /path/to/checkpoint.pth

# Check config path
cat my_inference.yaml | grep nnunet_checkpoint
```

### Issue: "Out of memory"

```yaml
# Solution 1: Reduce batch size
inference:
  sliding_window:
    sw_batch_size: 1

# Solution 2: Disable TTA
inference:
  test_time_augmentation:
    flip_axes: null
```

### Issue: "No files found"

```bash
# Verify file pattern
ls /path/to/data/*.h5

# Or use absolute paths
inference:
  volume_mode:
    file_pattern: "/absolute/path/to/data/*.h5"
```

### Issue: "Module 'nnunetv2' not found"

```bash
# Install nnUNet v2
pip install nnunetv2

# Verify installation
python -c "import nnunetv2; print(nnunetv2.__version__)"
```

---

## 📊 Example Workflow

### Scenario: Process 1000 2D slices on 4 GPUs

**Input**:
- 1000 TIFF images in `/data/slices/`
- nnUNet 2D model trained on mitochondria
- 4 A100 GPUs available

**Config** (`mito_inference.yaml`):
```yaml
system:
  num_gpus: 1  # Per process

model:
  architecture: nnunet
  nnunet_checkpoint: /models/mito_2d.pth

data:
  do_2d: true

inference:
  volume_mode:
    enabled: true
    file_pattern: "/data/slices/*.tiff"

  data:
    output_path: /results/

  sliding_window:
    window_size: [384, 512]
    overlap: 0.5

  test_time_augmentation:
    flip_axes: [[1], [2]]  # H and V flips
    act: softmax
    select_channel: 1

  postprocessing:
    intensity_scale: 255.0
    intensity_dtype: uint8
```

**Run**:
```bash
# Launch 4 parallel processes
for GPU in {0..3}; do
  python scripts/main.py \
    --config mito_inference.yaml \
    --mode infer-volume \
    inference.volume_mode.start_index=$GPU \
    inference.volume_mode.step=4 \
    system.device=cuda:$GPU &
done
wait

echo "All GPUs finished!"
```

**Expected Output**:
```
GPU 0: Processing files 0, 4, 8, 12, ... (250 files)
GPU 1: Processing files 1, 5, 9, 13, ... (250 files)
GPU 2: Processing files 2, 6, 10, 14, ... (250 files)
GPU 3: Processing files 3, 7, 11, 15, ... (250 files)

Total time: ~30 minutes
Throughput: ~33 slices/min per GPU
```

---

## 📚 Full Documentation

- **Design Document**: [NNUNET_INTEGRATION_DESIGN.md](NNUNET_INTEGRATION_DESIGN.md)
- **Summary**: [NNUNET_INTEGRATION_SUMMARY.md](NNUNET_INTEGRATION_SUMMARY.md)
- **Architecture Diagram**: [docs/nnunet_architecture_diagram.txt](docs/nnunet_architecture_diagram.txt)
- **Example Configs**: [tutorials/nnunet_mito_inference.yaml](tutorials/nnunet_mito_inference.yaml)
- **PyTC Documentation**: [CLAUDE.md](CLAUDE.md)

---

## ✅ Checklist

Before running inference, verify:

- [ ] nnUNet v2 installed (`pip install nnunetv2`)
- [ ] Checkpoint file exists and is readable
- [ ] plans.json and dataset.json in same directory as checkpoint
- [ ] Input files exist (test with `ls /path/to/data/*.h5`)
- [ ] Output directory is writable
- [ ] GPU is available (`nvidia-smi`)
- [ ] Config file is valid YAML (test with `python -c "from connectomics.config import load_config; load_config('my_inference.yaml')"`)

---

**Status**: 🚧 Design Phase - Implementation Pending
**Last Updated**: 2025-11-26
**Questions?** See TROUBLESHOOTING_NNUNET.md or consult CLAUDE.md
