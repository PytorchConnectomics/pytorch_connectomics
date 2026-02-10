# DeepEM Codebase Summary

**Location:** `/projects/weilab/weidf/lib/seg/DeepEM`

**Purpose:** Deep Learning framework for EM (Electron Microscopy) Connectomics, focused on reconstructing neural connections from EM images.

## Overview

DeepEM is a PyTorch-based framework developed by Kisuk Lee and colleagues (Harvard/Princeton) for superhuman accuracy in connectomics segmentation tasks. The codebase implements the methods described in two key papers:

1. **Lee et al. 2017** - "Superhuman Accuracy on the SNEMI3D Connectomics Challenge"
2. **Dorkenwald et al. 2019** - "Binary and analog variation of synapses between cortical pyramidal neurons"

## Repository Structure

```
DeepEM/
└── deepem/
    ├── data/           # Data loading, augmentation, and preprocessing
    │   ├── augment/    # EM-specific augmentations
    │   ├── dataset/    # Dataset definitions (FlyEM, Kasthuri, Pinky, etc.)
    │   ├── modifier/   # Data modifiers
    │   └── sampler/    # Data sampling strategies
    ├── loss/           # Loss functions (BCE, MSE, Affinity)
    ├── models/         # Model architectures (RSUNet, UpDown)
    ├── test/           # Inference and evaluation
    ├── train/          # Training loop and utilities
    └── utils/          # Utility functions
```

**Total:** 76 Python files

## Core Architecture

### 1. Models (`deepem/models/`)

#### **RSUNet (Residual Symmetric U-Net)**
- **File:** `rsunet.py` (56 lines)
- **Description:** Primary architecture - residual version of symmetric U-Net
- **Key Features:**
  - Residual connections for better gradient flow
  - Symmetric encoder-decoder structure
  - Supports both batch normalization and group normalization
  - Configurable depth and width
  - Built on top of `emvision` library models

```python
# Model creation
core = emvision.models.RSUNet(width=[16,32,64,128,256][:depth])
# or with group normalization:
core = emvision.models.rsunet_gn(width=width[:depth], group=opt.group)
```

#### **UpDown Architecture**
- **File:** `updown.py`
- **Description:** Alternative encoder-decoder architecture
- Simplified structure compared to RSUNet

#### **Custom Layers** (`layers.py`)
- `Conv`: 3D convolution with Kaiming initialization
- `Scale`: Learnable scaling layer
- `Crop`: Center cropping for predictions

**Architecture Parameters:**
- `depth`: Number of downsampling levels (default: 4)
- `width`: Channel sizes per level (e.g., [16,32,64,128,256,512])
- `group`: Group normalization parameter (0 = batch/instance norm)
- `fov`: Field of view (default: (20,256,256))

### 2. Training System (`deepem/train/`)

#### **Training Loop** (`run.py` - 119 lines)
Simple, efficient training loop:

```python
def train(opt):
    model = load_model(opt)
    optimizer = load_optimizer(opt, model.parameters())
    train_loader, val_loader = load_data(opt)

    for i in range(opt.chkpt_num, opt.max_iter):
        sample = train_loader()
        optimizer.zero_grad()
        losses, nmasks, preds = forward(model, sample, opt)
        total_loss = sum([w*losses[k] for k, w in opt.loss_weight.items()])
        total_loss.backward()
        optimizer.step()

        # Logging, evaluation, checkpointing...
```

**Key Features:**
- Iteration-based (not epoch-based) training
- Integrated evaluation loop
- Checkpoint saving every N iterations
- TensorBoard logging via custom `Logger` class

#### **Configuration** (`option.py` - 218 lines)
Comprehensive argparse-based configuration system:

**Required Arguments:**
- `--exp_name`: Experiment name
- `--data_dir`: Data directory path
- `--data`: Dataset module name
- `--model`: Model type (rsunet, updown)
- `--sampler`: Sampling strategy
- `--augment`: Augmentation module (optional)

**Training Parameters:**
- `--max_iter`: Maximum iterations (default: 1,000,000)
- `--batch_size`: Batch size (default: 1)
- `--lr`: Learning rate (default: 0.001)
- `--optim`: Optimizer (Adam, SGD)
- `--eval_intv`: Evaluation interval (default: 1000 iterations)
- `--chkpt_intv`: Checkpoint interval (default: 10000 iterations)

**Augmentation Options:**
- `--flip`: Random flips
- `--grayscale`: Grayscale augmentation
- `--warping`: Elastic warping
- `--misalign`: Misalignment augmentation level
- `--missing`: Missing section augmentation level
- `--blur`: Motion blur augmentation level

**Multi-Task Options:**
- `--aff`: Affinity (neuron boundary)
- `--syn`: Synapse detection
- `--psd`: Post-synaptic density
- `--mit`: Mitochondria segmentation
- `--mye`: Myelin detection
- `--blv`: Blood vessel detection
- `--glia`: Glia detection

### 3. Data Pipeline (`deepem/data/`)

#### **Augmentation** (`data/augment/`)

**Core Augmentations** (`grayscale_warping.py`):
```python
def get_augmentation(is_train, recompute=False, grayscale=False, warping=False):
    augs = []

    # Recompute connected components
    if recompute:
        augs.append(Label())

    # Brightness & contrast perturbation
    if is_train and grayscale:
        augs.append(MixedGrayscale2D(
            contrast_factor=0.5,
            brightness_factor=0.5,
            prob=1, skip=0.3))

    # Elastic warping
    if is_train and warping:
        augs.append(Warp(skip=0.3, do_twist=False, rot_max=45.0))

    # Basic transforms
    augs.append(FlipRotate())

    return Compose(augs)
```

**Dataset-Specific Augmentations:**
- `flyem/`: FlyEM dataset augmentations (CREMI challenges)
- `kasthuri11/`: Kasthuri connectomics data
- `pinky_basil/`: Pinky/Basil cortical datasets

**EM-Specific Augmentations:**
- **Misalignment**: Simulates section alignment errors
- **Missing Sections**: Removes random z-slices
- **Motion Blur**: Adds directional blur artifacts
- **Grayscale Warping**: Elastic deformation + intensity changes

#### **Datasets** (`data/dataset/`)

**Supported Datasets:**
- **FlyEM**: Drosophila brain imaging (CREMI challenges)
- **Kasthuri11**: Mouse cortex connectomics
- **Pinky/Basil**: Mouse visual cortex
- **Minnie**: Large-scale cortical dataset

**Dataset Characteristics:**
- Highly dataset-specific implementations
- Each dataset has multiple variants (MIP levels, padding, etc.)
- Largest file: `minnie/pinky_basil_minnie.py` (471 lines)

### 4. Loss Functions (`deepem/loss/`)

#### **BCELoss** (`loss.py`)
Binary cross-entropy loss with advanced features:

```python
class BCELoss(nn.Module):
    def __init__(self, size_average=True, margin0=0, margin1=0, inverse=True):
        # margin0: negative class margin
        # margin1: positive class margin
        # inverse: apply margin to targets vs. masking predictions

    def forward(self, input, target, mask):
        # Compute loss only on valid (masked) voxels
        loss = F.binary_cross_entropy_with_logits(input, target, weight=mask)
        return loss, nmsk  # Returns loss and number of valid voxels
```

**Key Features:**
- **Margin-based learning**: Ignore easy examples near decision boundary
- **Mask support**: Handle partially labeled data
- **Size averaging**: Normalize by number of valid voxels
- **Inverse mode**: Apply margins to targets instead of masking

#### **MSELoss**
Mean squared error with similar margin and masking features.

#### **Affinity Loss** (`affinity.py`)
Specialized loss for long-range affinity graphs used in neuron segmentation.

**Loss Options:**
- `--loss`: Loss function name (BCELoss, MSELoss)
- `--margin0`: Margin for negative class (0-1)
- `--margin1`: Margin for positive class (0-1)
- `--inverse`: Apply margins inversely
- `--size_average`: Average loss over valid voxels
- `--class_balancing`: Balance positive/negative classes

### 5. Testing/Inference (`deepem/test/`)

**Inference Pipeline** (`forward.py` - 128 lines):
- Sliding window inference
- Handles large volumes that don't fit in GPU memory
- Supports stitching overlapping predictions
- Mask generation for partially labeled regions

**Testing Options** (`option.py` - 210 lines):
- Similar to training options
- Includes output specification
- Mask generation for evaluation
- Batch processing for large volumes

## Key Design Patterns

### 1. **Module-Based Configuration**
Instead of monolithic config files, DeepEM uses Python modules for configuration:

```bash
python train.py \
    --data deepem.data.dataset.flyem.cremi_b \
    --augment deepem.data.augment.flyem.aug_mip1 \
    --model rsunet
```

This allows:
- Dataset-specific logic in Python code
- Easy versioning (aug_v0, aug_v1, aug_v2)
- Type safety and IDE support

### 2. **Iteration-Based Training**
Unlike epoch-based training:
- Fixed number of iterations (e.g., 1M)
- Sample randomly from dataset
- Allows weighted sampling across multiple datasets
- Easier to compare experiments

### 3. **Multi-Task Learning**
Flexible multi-output architecture:

```python
out_spec = {
    'affinity': (1, 3, 128, 128, 128),  # 3-channel affinity
    'mito': (1, 1, 128, 128, 128),      # Mitochondria
    'synapse': (1, 2, 128, 128, 128),   # Pre/post synapse
}
```

Each task has its own:
- Output head
- Loss function
- Weight in total loss

### 4. **Data Loader Design**
Custom data loading with:
- On-the-fly augmentation
- Sample caching
- Weighted sampling across datasets
- Partial label support (masking)

## Dependencies

**External Libraries:**
- `torch`: PyTorch deep learning framework
- `emvision`: Custom vision library for EM data (companion library)
- `augmentor`: Data augmentation library (likely custom)

**System Requirements:**
- CUDA-capable GPU
- cuDNN for auto-tuning (`torch.backends.cudnn.benchmark`)

## Workflow Example

### Training a Model

```bash
# 1. Create experiment config (as text file or command line)
python deepem/train/run.py \
    --exp_name my_experiment \
    --data_dir /path/to/data \
    --data deepem.data.dataset.flyem.cremi_b \
    --model rsunet \
    --sampler deepem.data.sampler.basic \
    --augment deepem.data.augment.grayscale_warping \
    --train_ids train_vol1 train_vol2 \
    --val_ids val_vol1 \
    --depth 4 \
    --width 16 32 64 128 256 \
    --grayscale \
    --warping \
    --lr 0.001 \
    --max_iter 1000000 \
    --batch_size 1 \
    --gpu_ids 0
```

### Multi-Task Training

```bash
python deepem/train/run.py \
    --exp_name multitask_experiment \
    --aff 1.0 \      # Affinity task, weight 1.0
    --mit 0.5 \      # Mitochondria task, weight 0.5
    --syn 0.5 \      # Synapse task, weight 0.5
    # ... other options
```

### Inference

```bash
python deepem/test/forward.py \
    --exp_name my_experiment \
    --chkpt_num 1000000 \
    --data_dir /path/to/test/data \
    --test_ids test_vol1 test_vol2
```

## Comparison with PyTorch Connectomics

| Aspect | DeepEM | PyTorch Connectomics (PyTC) |
|--------|--------|----------------------------|
| **Framework** | Pure PyTorch | PyTorch Lightning + MONAI |
| **Configuration** | Python modules + argparse | YAML + dataclasses (Hydra/OmegaConf) |
| **Training Loop** | Custom iteration-based | Lightning Trainer (epoch-based) |
| **Data Loading** | Custom loaders | MONAI transforms + CacheDataset |
| **Models** | RSUNet, UpDown (custom) | MONAI models + MedNeXt + RSUNet |
| **Augmentation** | Custom augmentor library | MONAI transforms + custom EM transforms |
| **Loss Functions** | Custom with margin/masking | MONAI losses + custom |
| **Multi-Task** | Built-in multi-output | Supported via Lightning |
| **Logging** | Custom Logger (TensorBoard) | Lightning loggers |
| **Distributed Training** | Manual DataParallel | Lightning DDP/FSDP |
| **Checkpointing** | Manual | Lightning callbacks |
| **Maturity** | Research code (2017-2019) | Production framework (2024) |

## Lessons for PyTC Integration

### 1. **EM-Specific Augmentations**
DeepEM has well-tested EM augmentations:
- Misalignment (translation + rotation)
- Missing sections
- Motion blur
- Grayscale warping

**Status in PyTC:** ✅ **Implemented** in `connectomics/data/augment/monai_transforms.py` as MONAI dict transforms.

### 2. **Multi-Task Learning Architecture**
DeepEM's `OutputBlock` design is elegant:

```python
class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_spec):
        for k, v in out_spec.items():
            self.add_module(k, Conv(in_channels, v[-4], kernel_size))

    def forward(self, x):
        return {k: m(x) for k, m in self.named_children()}
```

**Status in PyTC:** ⚠️ **Partial** - Multi-task supported but could adopt this pattern.

### 3. **Margin-Based Loss**
DeepEM's margin-based BCE loss helps focus on hard examples:

```python
# Ignore predictions within margin of decision boundary
if margin0 > 0 or margin1 > 0:
    m_int = torch.ge(activ, 1 - margin1) * torch.eq(target, 1)
    m_ext = torch.le(activ, margin0) * torch.eq(target, 0)
    mask *= 1 - (m_int + m_ext)
```

**Status in PyTC:** ❌ **Not implemented** - Could add as advanced loss option.

### 4. **Iteration-Based Training**
Pros:
- Easier to compare experiments
- Better for weighted multi-dataset training
- No epoch boundary artifacts

Cons:
- Less intuitive than epochs
- Harder to track "passes through data"

**Status in PyTC:** Uses epochs (Lightning standard). Could add iteration mode.

### 5. **Module-Based Config**
DeepEM's Python module approach is flexible but:
- Harder to version control configs
- No schema validation
- Requires code changes for new experiments

**PyTC approach (YAML + dataclasses) is superior** for:
- Reproducibility
- Configuration as data
- Type safety
- Easy experimentation

## Code Quality Assessment

**Strengths:**
- ✅ Clean, readable code
- ✅ Well-organized module structure
- ✅ Proven research results
- ✅ Comprehensive EM augmentations
- ✅ Flexible multi-task learning

**Weaknesses:**
- ❌ Minimal documentation/comments
- ❌ No type hints
- ❌ No unit tests
- ❌ Dataset-specific code duplication
- ❌ Manual training loop management
- ❌ Limited to single-node training

## Integration Recommendations

### What to Adopt from DeepEM:

1. **EM Augmentation Strategies** ✅ DONE
   - Misalignment parameters
   - Missing section handling
   - Motion blur techniques

2. **Margin-Based Loss** 🔄 TODO
   - Add as optional loss parameter
   - Useful for handling label noise

3. **Multi-Task Output Design** 🔄 CONSIDER
   - Cleaner than separate heads
   - Better for dynamic task sets

4. **Long-Range Affinity** 🔄 RESEARCH
   - Useful for neuron segmentation
   - Could be added as optional task

### What NOT to Adopt:

1. **Custom Training Loop** - Lightning is superior
2. **Module-Based Config** - YAML is better for reproducibility
3. **Custom Data Loaders** - MONAI CacheDataset is more efficient
4. **Manual Distributed Training** - Lightning handles this

## Conclusion

DeepEM is a **mature research codebase** (2017-2019) that demonstrated superhuman accuracy in connectomics. Its key contributions are:

1. **Residual Symmetric U-Net** - Clean, effective architecture
2. **EM-Specific Augmentations** - Well-designed for connectomics artifacts
3. **Multi-Task Learning** - Flexible framework for multiple segmentation tasks
4. **Margin-Based Loss** - Focus learning on hard examples

**PyTorch Connectomics has successfully adopted** the best ideas from DeepEM while modernizing the infrastructure with:
- PyTorch Lightning for training orchestration
- MONAI for medical imaging tools
- Modern configuration management (Hydra)
- Production-ready features (distributed training, mixed precision, callbacks)

The main value of DeepEM today is as a **reference implementation** for EM-specific techniques and augmentations, which have been successfully integrated into PyTC's MONAI-based augmentation pipeline.

---

**Related Documentation:**
- [EM_AUGMENTATION_GUIDE.md](.claude/EM_AUGMENTATION_GUIDE.md) - Complete EM augmentation reference
- [BANIS_SUMMARY.md](.claude/BANIS_SUMMARY.md) - Another EM segmentation baseline
- [MEDNEXT_SUMMARY.md](.claude/MEDNEXT_SUMMARY.md) - Modern architecture comparison

**External References:**
- Lee et al. 2017: https://arxiv.org/abs/1706.00120
- Dorkenwald et al. 2019: https://www.biorxiv.org/content/10.1101/2019.12.29.890319v1
