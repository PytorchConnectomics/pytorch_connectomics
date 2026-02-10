# BANIS: Baseline for Affinity-based Neuron Instance Segmentation

## Project Overview

BANIS is a baseline implementation for the **Neuron Instance Segmentation Benchmark (NISB)**, providing an easily adaptable framework for neuron instance segmentation in electron microscopy (EM) images. The project combines affinity prediction with modern deep learning architectures (MedNeXt) and simple connected components for post-processing.

**Repository Location**: `/projects/weilab/weidf/lib/banis`

**Key Features**:
- Affinity-based segmentation approach with short and long-range affinities
- MedNeXt architecture integration (predefined sizes: S, B, M, L)
- PyTorch Lightning-based training pipeline
- MONAI transforms for data augmentation
- Comprehensive evaluation metrics (VOI, NERL, ERL)
- Slurm cluster support with auto-resubmission
- Connected components post-processing (Numba-accelerated)

## Architecture

### Core Components

```
banis/
├── BANIS.py                    # Main Lightning module and training script
├── data.py                     # Dataset classes and data loading
├── inference.py                # Patched inference and connected components
├── metrics.py                  # Evaluation metrics (VOI, ERL, NERL)
├── util.py                     # Utility functions (bbox, IoU, H5 I/O)
├── config.yaml                 # Hyperparameter grid for experiments
├── environment.yaml            # Conda environment specification
│
├── slurm_job_scheduler.py      # Slurm job submission
├── aff_train.sh                # Slurm training script
├── validation_watcher.sh       # External validation script
├── andromeda_launcher.py       # Job launcher utility
├── justfile                    # Just commands for training
│
├── generate_all_affs.py        # Generate affinity predictions
├── show_data.py                # Data visualization
└── error_analysis.py           # Error analysis utilities
```

## Methodology

### 1. Affinity Prediction

BANIS predicts **6 affinity channels** (+ optional SDT):
- **3 short-range affinities**: Adjacent voxel connections (x, y, z directions)
- **3 long-range affinities**: Long-distance connections (configurable offset, default 10 voxels)
- **1 optional SDT channel**: Skeleton-aware distance transform

**Affinity Computation**:
```python
# Short-range: Neighbors are same segment ID
affinities[0, :-1] = seg[:-1] == seg[1:]        # x-direction
affinities[1, :, :-1] = seg[:, :-1] == seg[:, 1:]  # y-direction
affinities[2, :, :, :-1] = seg[:, :, :-1] == seg[:, :, 1:]  # z-direction

# Long-range: Long-distance connections (offset=10)
affinities[3, :-10] = seg[:-10] == seg[10:]     # x-direction
affinities[4, :, :-10] = seg[:, :-10] == seg[:, 10:]  # y-direction
affinities[5, :, :, :-10] = seg[:, :, :-10] == seg[:, :, 10:]  # z-direction
```

**Loss Function**:
- Primary: Binary cross-entropy with logits on affinities
- Optional: MSE loss on tanh(SDT) predictions
- Combined loss: `loss = aff_loss + sdt_loss_weight * sdt_loss`

### 2. Model Architecture

Uses **MedNeXt** (ConvNeXt-based 3D U-Net from nnU-Net):
- **Predefined sizes**: S (5.6M), B (10.5M), M (17.6M), L (61.8M)
- **Kernel sizes**: 3, 5, or 7
- **Features**: Outside block checkpointing for memory efficiency
- **Optional**: torch.compile for speed

```python
from nnunet_mednext import create_mednext_v1

model = create_mednext_v1(
    num_input_channels=1,  # or multi-channel
    num_classes=6,         # 3 short + 3 long affinities
    model_id="S",          # S, B, M, or L
    kernel_size=3,         # 3, 5, or 7
)
```

### 3. Data Pipeline

**Dataset Class**: `AffinityDataset`
- Loads large EM volumes (HDF5, Zarr)
- Random patch sampling (default 128³)
- On-the-fly affinity computation
- Connected component relabeling (disconnected segments)
- Unlabeled voxel handling (label = -1)

**Data Sources**:
1. **Synthetic data**: NISB benchmark datasets (multiple settings)
2. **Real data**: Zebrafinch dataset (optional mixing)

**Augmentations** (MONAI + custom):
- Geometric: Rotation (90°), flipping, axis permutation
- Affine: Rotation, scaling, shearing
- Intensity: Multiplicative/additive noise, random scaling
- Slice-level: Drop slices, shift slices
- Noise: Gaussian noise

**WeightedConcatDataset**: Mix synthetic and real data with configurable ratio

### 4. Training

**PyTorch Lightning Module**: `BANIS`
- AdamW optimizer (default lr=1e-3, weight_decay=1e-2)
- CosineAnnealingLR scheduler (optional)
- Mixed precision training (16-bit)
- Gradient clipping (norm=1.0)
- DDP multi-GPU support
- Automatic checkpointing

**Training Workflow**:
```bash
python BANIS.py \
  --seed 0 \
  --batch_size 8 \
  --n_steps 50000 \
  --data_setting base \
  --base_data_path /path/to/nisb/ \
  --save_path /path/to/logs/ \
  --model_id S \
  --kernel_size 3 \
  --long_range 10
```

**Validation**:
- Periodic validation (default every 5000 steps)
- Full-cube inference on validation set
- Threshold sweep for optimal segmentation
- External validation via Slurm job (optional)

### 5. Inference

**Patched Inference** (`patched_inference`):
- Non-overlapping patches (fast) or overlapping with distance-weighted blending
- Distance-transform-based weighting (higher confidence at patch centers)
- Automatic patch coordinate generation
- Mixed precision (FP16)

```python
aff_pred = patched_inference(
    img_data,
    model=model,
    small_size=128,
    do_overlap=True,
    prediction_channels=6,
    divide=255,
)
```

**Connected Components** (`compute_connected_component_segmentation`):
- Numba JIT-compiled for speed
- 6-connectivity (faces only)
- Simple flood-fill algorithm
- Uses **only short-range affinities** (first 3 channels)
- Thresholding: Affinities → binary → connected components

### 6. Evaluation Metrics

**Metrics Computed** (from `funlib.evaluate`):

1. **VOI (Variation of Information)**:
   - `voi_split`: Under-segmentation error
   - `voi_merge`: Over-segmentation error
   - `voi_sum`: Total error

2. **ERL (Expected Run Length)**:
   - Skeleton-based metric
   - Measures correct segment length along neurons
   - `nerl = erl / max_erl`: Normalized ERL (0-1, higher is better)

3. **Merger/Split Statistics**:
   - `n_mergers`: Number of merge errors
   - `n_non0_mergers`: Non-background mergers
   - `n_splits`: Number of split errors

4. **Adapted NERL**:
   - Ignores small mergers (5, 20, 100 nodes)
   - More robust to minor errors

**Threshold Sweep**:
- Evaluate multiple thresholds on short-range affinities
- Select threshold maximizing NERL
- Store best threshold for test set

## Configuration

### Hyperparameter Grid (`config.yaml`)

```yaml
params:
  learning_rate: [1e-3]
  weight_decay: [1e-2]
  seed: [0, 1, 2, 3, 4]
  long_range: [10]
  batch_size: [1]
  scheduler: [true]
  model_id: ["L"]
  kernel_size: [5]
  synthetic: [1.0]  # 1.0 = 100% synthetic, 0.0 = 100% real
  drop_slice_prob: [0.05]
  shift_slice_prob: [0.05]
  intensity_aug: [true]
  noise_scale: [0.5]
  affine: [0.5]
  n_steps: [1_000_000]
  small_size: [256]
  data_setting: ["train_100"]
  base_data_path: ["/cajal/nvmescratch/projects/NISB/"]
  save_path: ["/cajal/scratch/projects/misc/zuzur/xl_banis"]
  auto_resubmit: [True]
  distributed: [True]
  validate_extern: [True]
```

### NISB Data Settings

- `base`: Standard NISB dataset
- `liconn`: Different resolution (9×9×12 nm)
- `multichannel`: Multi-channel input
- `neg_guidance`: Negative guidance labels
- `pos_guidance`: Positive guidance labels
- `no_touch_thick`: No touching neurons (thick)
- `touching_thin`: Touching neurons (thin)
- `slice_perturbed`: Slice-level perturbations
- `train_100`: 100 training samples

## Usage

### Installation

```bash
# Create environment
mamba env create -f environment.yaml
mamba activate nisb

# Or manual install
mamba create -n nisb -c conda-forge python=3.11 -y
mamba activate nisb
pip install torch torchvision pytorch-lightning zarr monai scipy
pip install connected-components-3d numba tensorboard
pip install git+https://github.com/MIC-DKFZ/MedNeXt.git#egg=mednextv1
pip install git+https://github.com/funkelab/funlib.evaluate.git
```

### Training

**Single job**:
```bash
python BANIS.py \
  --seed 0 \
  --batch_size 8 \
  --n_steps 50000 \
  --data_setting base \
  --base_data_path /local/dataset/dir/ \
  --save_path /local/logging/dir/
```

**Large model (BANIS-L)**:
```bash
python BANIS.py --model_id L --kernel_size 5 --batch_size 4 --n_steps 100000
```

**With SDT prediction**:
```bash
python BANIS.py --sdt --sdt_loss_weight 1.0
```

**Slurm cluster** (multiple jobs):
```bash
# Edit config.yaml
python slurm_job_scheduler.py
```

**Using justfile**:
```bash
just train_base
just train_liconn
just train_base_sdt
```

### Inference

```bash
# Generate affinities
python generate_all_affs.py --checkpoint_path /path/to/checkpoint.ckpt

# Compute metrics
python metrics.py \
  --pred_seg /path/to/predictions.zarr \
  --skel_path /path/to/skeleton.pkl \
  --load_to_memory
```

### Visualization

```bash
python show_data.py --base_path /local/benchmark/dir/
```

## Key Implementation Details

### 1. Memory Efficiency

- **Outside block checkpointing**: Reduces GPU memory (MedNeXt feature)
- **FP16 mixed precision**: ~2x memory reduction
- **Zarr lazy loading**: Avoids loading full volumes
- **Patch-based training**: 128³ patches (adjustable)
- **Distance-weighted overlapping inference**: Better boundary predictions

### 2. Distributed Training

- **PyTorch Lightning DDP**: Multi-GPU and multi-node
- **Slurm integration**: Automatic job submission and resubmission
- **SLURM_NNODES**: Auto-detect number of nodes
- **Find unused parameters**: Disabled for efficiency

### 3. Connected Components

- **Numba JIT compilation**: ~10-100x speedup vs pure Python
- **Simple flood-fill**: Stack-based, no recursion
- **6-connectivity**: Face neighbors only (not edges/corners)
- **Hard affinities**: Threshold → binary → CC

### 4. Augmentation Strategy

- **Isotropic augmentation**: x, y, z treated equally (axis permutation)
- **Rotation**: 90° rotations + axis swaps
- **Slice-level**: Simulates EM artifacts (dropped/shifted slices)
- **Affine**: MONAI RandAffined (rotation, scale, shear)
- **Intensity**: Multiplicative/additive + Gaussian noise

### 5. Evaluation Pipeline

1. **Patched inference**: Generate affinity predictions
2. **Threshold sweep**: Try multiple thresholds (sigmoid space)
3. **Connected components**: Convert affinities → segmentation
4. **Skeleton metrics**: VOI, ERL, NERL on ground truth skeletons
5. **Best threshold selection**: Maximize NERL on validation
6. **Test evaluation**: Use best validation threshold

## Performance Considerations

### Hardware Requirements

- **GPU**: 1 NVIDIA A40 (48 GB) for batch_size=8
- **RAM**: 500 GB (for full-cube inference)
- **Storage**: Fast storage (NVMe) recommended for Zarr data

**Reducing Memory**:
- Decrease `batch_size` (adjust `n_steps` and `learning_rate`)
- Use smaller patch size (`small_size=96`)
- Disable overlapping inference (`do_overlap=False`)

### Training Speed

- **BANIS-S**: ~2-3 hours for 50K steps (1 GPU)
- **BANIS-L**: ~8-10 hours for 50K steps (1 GPU)
- **Full training**: 200K-1M steps for best results
- **Validation**: Expensive (full-cube inference every 5K steps)

### Model Comparison

| Model | Parameters | Kernel | Memory | NERL (approx) |
|-------|-----------|--------|--------|---------------|
| BANIS-S | 5.6M | 3×3×3 | ~12 GB | 0.70-0.75 |
| BANIS-L | 61.8M | 5×5×5 | ~40 GB | 0.75-0.80 |

## Differences from PyTorch Connectomics

### Similarities
1. **PyTorch Lightning**: Both use Lightning for training
2. **MedNeXt support**: Both integrate MedNeXt architectures
3. **Affinity-based**: Both support affinity prediction
4. **Zarr/HDF5**: Both support efficient data formats

### Key Differences

| Feature | PyTorch Connectomics | BANIS |
|---------|---------------------|-------|
| **Domain** | General connectomics | NISB benchmark specific |
| **Scope** | Full framework (many tasks) | Focused baseline (affinity + CC) |
| **Configuration** | Hydra/OmegaConf (complex) | Argparse (simple) |
| **Post-processing** | Multiple methods (watershed, mutex, etc.) | Simple connected components |
| **Evaluation** | General metrics | NISB-specific (NERL, skeleton-based) |
| **Data** | Multi-dataset support | NISB + Zebrafinch |
| **Architecture** | Many models (UNet, UNETR, Swin, etc.) | MedNeXt only |
| **Lightning Module** | ConnectomicsModule (general) | BANIS (task-specific) |
| **Augmentation** | MONAI-based (general) | Custom + MONAI (EM-specific) |
| **Slurm** | Not included | Built-in support |
| **External validation** | Not applicable | Slurm-based external validation |

### What PyTC Can Learn from BANIS

1. **Numba-accelerated CC**: Fast connected components implementation
2. **Slice-level augmentation**: EM-specific (drop/shift slices)
3. **Weighted concat dataset**: Easy mixing of multiple datasets
4. **External validation**: Long training with separate validation process
5. **Threshold sweep utilities**: Systematic threshold optimization
6. **NERL evaluation**: Skeleton-based metrics for neuron segmentation

### What BANIS Can Learn from PyTC

1. **Hydra configs**: More structured configuration management
2. **Multi-task support**: Semantic + instance + affinity
3. **Advanced post-processing**: Watershed, mutex, GASP
4. **Model variety**: Support for more architectures
5. **Deep supervision**: Multi-scale loss (already in PyTC + MedNeXt)
6. **Modular design**: Cleaner separation of concerns

## Integration Opportunities

### Integrating BANIS into PyTC

1. **Add slice augmentation transforms**:
   - `DropSlice`: Random slice dropout
   - `ShiftSlice`: Random slice shifting
   - Location: `connectomics/data/augment/`

2. **Add Numba CC implementation**:
   - Fast connected components
   - Location: `connectomics/model/utils/`

3. **Add NERL metrics**:
   - Skeleton-based evaluation
   - Location: `connectomics/engine/metrics/`

4. **Add WeightedConcatDataset**:
   - Mix multiple datasets with weights
   - Location: `connectomics/data/dataset/`

5. **Add NISB dataset loader**:
   - Support NISB benchmark format
   - Location: `connectomics/data/dataset/`

### Using PyTC Features in BANIS

1. **Replace argparse with Hydra**: More maintainable configs
2. **Use PyTC's deep supervision**: Already compatible with MedNeXt
3. **Add semantic segmentation**: Predict foreground/background
4. **Use PyTC's data pipeline**: More robust and tested
5. **Add more architectures**: UNet, UNETR for comparison

## Files Overview

### Core Python Files

- **BANIS.py** (450 lines): Lightning module, training loop, validation, full-cube inference
- **data.py** (471 lines): Dataset classes, augmentation, affinity computation
- **inference.py** (213 lines): Patched inference, connected components (Numba)
- **metrics.py** (392 lines): VOI, ERL, NERL computation, adapted metrics
- **util.py** (341 lines): Bounding box, IoU, H5 I/O utilities

### Utility Scripts

- **slurm_job_scheduler.py**: Grid search job submission
- **validation_watcher.py**: External validation monitoring
- **generate_all_affs.py**: Batch affinity generation
- **show_data.py**: Data visualization
- **error_analysis.py**: Error analysis utilities
- **andromeda_launcher.py**: Cluster job launcher

### Configuration

- **config.yaml**: Hyperparameter grid for experiments
- **environment.yaml**: Conda environment (Python 3.11, PyTorch 2.4.1)
- **justfile**: Just commands for common tasks

### Slurm Scripts

- **aff_train.sh**: Training script (multi-node, auto-resubmit)
- **validation_watcher.sh**: Validation script

## References

1. **NISB Benchmark**: https://structuralneurobiologylab.github.io/nisb/
2. **Affinity Prediction**: Funke et al., "Large Scale Image Segmentation with Structured Loss based Deep Learning for Connectome Reconstruction" (2018)
3. **MedNeXt**: Roy et al., "MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation" (MICCAI 2023)
4. **ERL/NERL Metrics**: https://github.com/funkelab/funlib.evaluate
5. **PyTorch Lightning**: https://lightning.ai/docs/pytorch/stable/
6. **MONAI**: https://docs.monai.io/

## Future Improvements

1. **Memory optimization**: Reduce RAM requirements for full-cube inference
2. **Advanced post-processing**: Watershed, mutex, GASP alternatives
3. **Multi-scale inference**: Different patch sizes for context
4. **Attention mechanisms**: Add attention to MedNeXt
5. **Self-supervised pre-training**: Pre-train on unlabeled EM data
6. **Active learning**: Uncertainty-guided annotation
7. **Test-time augmentation**: Ensemble predictions
8. **Model distillation**: Compress large models

## Contact & Citation

**Harvard Visual Computing Group**

If using this code, cite the NISB benchmark paper and MedNeXt:
```bibtex
@inproceedings{mednext2023,
  title={MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation},
  author={Roy, Saikat and Koehler, Gregor and Ulrich, Constantin and Baumgartner, Michael and Petersen, Jens and Isensee, Fabian and Jaeger, Paul F and Maier-Hein, Klaus},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2023}
}
```
