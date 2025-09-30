# PyTorch Connectomics Refactoring Plan

**Based on:**
- DESIGN.md (Lightning + MONAI architecture principles)
- MEDNEXT.md (MedNeXt integration guide)
- CLAUDE.md (Current codebase summary)

**Goal:** Integrate MedNeXt architecture while maintaining clean separation: Lightning (orchestration) + MONAI/MedNeXt (domain tools).

---

## 0. Implementation Status

**✅ COMPLETED:**
- **Phase 1:** Architecture Organization (registry system, base model interface, MONAI wrappers)
- **Phase 2:** MedNeXt Integration (MedNeXt wrappers with deep supervision)
- **Phase 3:** Deep Supervision (Lightning module multi-scale loss support)
- **Phase 4:** Configuration (MedNeXt parameters in hydra_config.py)
- **Phase 5:** Examples & Tests (mednext_lucchi.yaml, mednext_custom.yaml)

**🔄 REMAINING:**
- Final integration tests with actual MedNeXt installation
- Performance profiling and benchmarking
- Documentation updates

**KEY FILES CREATED/MODIFIED:**
- `connectomics/models/architectures/` (NEW directory with registry, base, MONAI, MedNeXt)
- `connectomics/models/build.py` (REFACTORED to use registry)
- `connectomics/lightning/lit_model.py` (UPDATED with deep supervision)
- `connectomics/config/hydra_config.py` (UPDATED with MedNeXt parameters)
- `tutorials/mednext_lucchi.yaml` (NEW example config)
- `tutorials/mednext_custom.yaml` (NEW advanced config)
- `tests/test_architecture_registry.py` (NEW tests)
- `tests/test_registry_basic.py` (NEW basic tests - PASSED ✓)

---

## 1. Current State Analysis

### ✅ Strengths
- Modern Hydra/OmegaConf configuration (`config/hydra_config.py`)
- Clean Lightning integration (`lightning/lit_model.py`, `lit_data.py`, `lit_trainer.py`)
- MONAI native models via `models/build.py`
- Unified loss system (`models/loss/`)
- Working entry point (`scripts/main.py`)

### ⚠️ Gaps
1. **Missing Architecture Organization**: No `models/architectures/` structure
2. **Incorrect MedNeXt Import**: Uses `mednextv1.mednext` instead of `nnunet_mednext`
3. **No Deep Supervision**: Lightning module doesn't handle multi-scale outputs (critical for MedNeXt)
4. **No Architecture Registry**: Hard to validate/list available models
5. **Incomplete MedNeXt Config**: Missing MedNeXt-specific parameters in `hydra_config.py`

---

## 2. Proposed Structure

Following DESIGN.md: Lightning = outer shell, MONAI/MedNeXt = inner toolbox.

```
connectomics/
├── config/
│   ├── hydra_config.py          # ✅ Modern (ADD MedNeXt params)
│   └── hydra_utils.py            # ✅ Modern
│
├── models/
│   ├── build.py                  # 🔧 REFACTOR (use registry)
│   │
│   ├── architectures/            # 📁 NEW
│   │   ├── __init__.py
│   │   ├── base.py               # Base model interface
│   │   ├── registry.py           # Architecture registry
│   │   ├── monai_models.py       # MONAI wrappers
│   │   └── mednext_models.py     # MedNeXt wrappers
│   │
│   ├── loss/                     # ✅ Clean
│   └── solver/                   # ✅ Clean
│
├── lightning/
│   ├── lit_data.py               # ✅ Clean
│   ├── lit_model.py              # 🔧 ADD deep supervision
│   └── lit_trainer.py            # ✅ Clean
│
├── data/                         # ✅ Clean
└── utils/                        # ✅ Clean
```

---

## 3. Refactoring Tasks

### Phase 1: Architecture Organization (Week 1)

#### Task 1.1: Create Architecture Registry
**File:** `connectomics/models/architectures/registry.py` (NEW)

```python
"""Architecture registry for model management."""

from typing import Dict, Callable, List

_ARCHITECTURE_REGISTRY: Dict[str, Callable] = {}

def register_architecture(name: str):
    """Decorator to register architecture builders."""
    def decorator(builder_fn: Callable) -> Callable:
        _ARCHITECTURE_REGISTRY[name] = builder_fn
        return builder_fn
    return decorator

def get_architecture_builder(name: str) -> Callable:
    """Get builder for architecture."""
    if name not in _ARCHITECTURE_REGISTRY:
        raise ValueError(
            f"Architecture '{name}' not found. "
            f"Available: {list_architectures()}"
        )
    return _ARCHITECTURE_REGISTRY[name]

def list_architectures() -> List[str]:
    """List all registered architectures."""
    return sorted(_ARCHITECTURE_REGISTRY.keys())
```

**Purpose:** Centralized model registration, easy extensibility

---

#### Task 1.2: Create Base Model Interface
**File:** `connectomics/models/architectures/base.py` (NEW)

```python
"""Base model interface for all architectures."""

import torch
import torch.nn as nn
from typing import Dict, Any, Union

class ConnectomicsModel(nn.Module):
    """Base class for all models with deep supervision support."""

    def __init__(self):
        super().__init__()
        self.supports_deep_supervision = False
        self.output_scales = 1

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.

        Returns:
            - torch.Tensor: Single scale output
            - Dict: Multi-scale outputs {'output': main, 'ds_1': scale1, ...}
        """
        raise NotImplementedError

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'name': self.__class__.__name__,
            'deep_supervision': self.supports_deep_supervision,
            'output_scales': self.output_scales,
            'parameters': sum(p.numel() for p in self.parameters()),
        }
```

**Purpose:** Standardized interface, explicit deep supervision contract

---

#### Task 1.3: Wrap MONAI Models
**File:** `connectomics/models/architectures/monai_models.py` (NEW)

```python
"""MONAI model wrappers with standard interface."""

from monai.networks.nets import BasicUNet, UNet, UNETR, SwinUNETR
from .base import ConnectomicsModel
from .registry import register_architecture

class MONAIModelWrapper(ConnectomicsModel):
    """Wrapper for MONAI models."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.supports_deep_supervision = False
        self.output_scales = 1

    def forward(self, x):
        return self.model(x)


@register_architecture('monai_basic_unet3d')
def build_basic_unet(cfg) -> ConnectomicsModel:
    """Build MONAI BasicUNet."""
    in_channels = cfg.model.in_channels
    out_channels = cfg.model.out_channels
    base_features = list(cfg.model.filters) if hasattr(cfg.model, 'filters') else [32, 64, 128, 256, 512]
    while len(base_features) < 6:
        base_features.append(base_features[-1] * 2)
    features = tuple(base_features[:6])

    model = BasicUNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        features=features,
        dropout=getattr(cfg.model, 'dropout', 0.0),
        act=getattr(cfg.model, 'activation', 'relu'),
        norm=getattr(cfg.model, 'norm', 'batch'),
    )
    return MONAIModelWrapper(model)


@register_architecture('monai_unet')
def build_monai_unet(cfg) -> ConnectomicsModel:
    """Build MONAI UNet with residual units."""
    features = list(cfg.model.filters) if hasattr(cfg.model, 'filters') else [32, 64, 128, 256, 512]
    channels = features[:5]
    strides = [2] * (len(channels) - 1)

    model = UNet(
        spatial_dims=3,
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        channels=channels,
        strides=strides,
        num_res_units=getattr(cfg.model, 'num_res_units', 2),
        kernel_size=getattr(cfg.model, 'kernel_size', 3),
        norm=getattr(cfg.model, 'norm', 'batch'),
        dropout=getattr(cfg.model, 'dropout', 0.0),
    )
    return MONAIModelWrapper(model)


@register_architecture('monai_unetr')
def build_unetr(cfg) -> ConnectomicsModel:
    """Build MONAI UNETR (transformer-based)."""
    model = UNETR(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        img_size=cfg.model.input_size,
        feature_size=getattr(cfg.model, 'feature_size', 16),
        hidden_size=getattr(cfg.model, 'hidden_size', 768),
        mlp_dim=getattr(cfg.model, 'mlp_dim', 3072),
        num_heads=getattr(cfg.model, 'num_heads', 12),
        pos_embed=getattr(cfg.model, 'pos_embed', 'perceptron'),
        norm_name=getattr(cfg.model, 'norm', 'instance'),
        dropout_rate=getattr(cfg.model, 'dropout', 0.0),
    )
    return MONAIModelWrapper(model)


@register_architecture('monai_swin_unetr')
def build_swin_unetr(cfg) -> ConnectomicsModel:
    """Build MONAI Swin UNETR."""
    model = SwinUNETR(
        img_size=cfg.model.input_size,
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        feature_size=getattr(cfg.model, 'feature_size', 48),
        use_checkpoint=getattr(cfg.model, 'use_checkpoint', False),
        drop_rate=getattr(cfg.model, 'dropout', 0.0),
        attn_drop_rate=getattr(cfg.model, 'attn_drop_rate', 0.0),
        dropout_path_rate=getattr(cfg.model, 'dropout_path_rate', 0.0),
    )
    return MONAIModelWrapper(model)
```

**Purpose:** Clean MONAI integration with registry (Hydra configs only)

---

### Phase 2: MedNeXt Integration (Week 2-3)

#### Task 2.1: Create MedNeXt Wrapper
**File:** `connectomics/models/architectures/mednext_models.py` (NEW)

```python
"""
MedNeXt model wrappers with deep supervision.

Reference: /projects/weilab/weidf/lib/MedNeXt/
See: MEDNEXT.md for detailed documentation
"""

import torch
import torch.nn as nn
from typing import Union, Dict
from .base import ConnectomicsModel
from .registry import register_architecture


class MedNeXtWrapper(ConnectomicsModel):
    """
    Wrapper for MedNeXt with deep supervision support.

    MedNeXt outputs 5 scales when deep_supervision=True:
    - Output 0: Full resolution
    - Output 1: 1/2 resolution
    - Output 2: 1/4 resolution
    - Output 3: 1/8 resolution
    - Output 4: 1/16 resolution (bottleneck)
    """

    def __init__(self, model: nn.Module, deep_supervision: bool = False):
        super().__init__()
        self.model = model
        self.supports_deep_supervision = deep_supervision
        self.output_scales = 5 if deep_supervision else 1

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward with optional deep supervision."""
        outputs = self.model(x)

        if self.supports_deep_supervision and isinstance(outputs, list):
            # Convert list to dict for Lightning
            return {
                'output': outputs[0],
                'ds_1': outputs[1],
                'ds_2': outputs[2],
                'ds_3': outputs[3],
                'ds_4': outputs[4],
            }
        return outputs


@register_architecture('mednext')
def build_mednext(cfg) -> ConnectomicsModel:
    """
    Build MedNeXt model.

    Sizes: S, B, M, L
    Kernels: 3, 5, 7
    Deep supervision: recommended for best performance
    """
    try:
        from nnunet_mednext import create_mednext_v1
    except ImportError:
        raise ImportError(
            "MedNeXt not found. Ensure /projects/weilab/weidf/lib/MedNeXt is in PYTHONPATH.\n"
            "Or install: pip install -e /projects/weilab/weidf/lib/MedNeXt"
        )

    # Extract config (Hydra only)
    in_channels = cfg.model.in_channels
    out_channels = cfg.model.out_channels
    model_size = getattr(cfg.model, 'mednext_size', 'S')
    kernel_size = getattr(cfg.model, 'kernel_size', 3)
    deep_supervision = getattr(cfg.model, 'deep_supervision', False)

    # Validate
    if model_size not in ['S', 'B', 'M', 'L']:
        raise ValueError(f"MedNeXt model_size must be S/B/M/L, got: {model_size}")
    if kernel_size not in [3, 5, 7]:
        raise ValueError(f"MedNeXt kernel_size must be 3/5/7, got: {kernel_size}")

    # Build using factory
    model = create_mednext_v1(
        num_channels=in_channels,
        num_classes=out_channels,
        model_id=model_size,
        kernel_size=kernel_size,
        deep_supervision=deep_supervision,
    )

    return MedNeXtWrapper(model, deep_supervision=deep_supervision)


@register_architecture('mednext_custom')
def build_mednext_custom(cfg) -> ConnectomicsModel:
    """
    Build MedNeXt with custom parameters.

    For advanced users needing full control.
    See MEDNEXT.md for all parameters.
    """
    try:
        from nnunet_mednext.mednextv1 import MedNeXt
    except ImportError:
        raise ImportError("MedNeXt not found.")

    # Extract all custom parameters (Hydra only)
    params = {
        'in_channels': cfg.model.in_channels,
        'n_channels': getattr(cfg.model, 'mednext_base_channels', 32),
        'n_classes': cfg.model.out_channels,
        'exp_r': getattr(cfg.model, 'mednext_exp_r', 4),
        'kernel_size': getattr(cfg.model, 'kernel_size', 7),
        'deep_supervision': getattr(cfg.model, 'deep_supervision', False),
        'do_res': getattr(cfg.model, 'mednext_do_res', True),
        'do_res_up_down': getattr(cfg.model, 'mednext_do_res_up_down', True),
        'block_counts': getattr(cfg.model, 'mednext_block_counts', [2,2,2,2,2,2,2,2,2]),
        'checkpoint_style': getattr(cfg.model, 'mednext_checkpoint', None),
        'norm_type': getattr(cfg.model, 'mednext_norm', 'group'),
        'dim': getattr(cfg.model, 'mednext_dim', '3d'),
        'grn': getattr(cfg.model, 'mednext_grn', False),
    }

    model = MedNeXt(**params)
    return MedNeXtWrapper(model, deep_supervision=params['deep_supervision'])
```

**Purpose:** Proper MedNeXt integration with correct imports and deep supervision

---

#### Task 2.2: Update Model Builder
**File:** `connectomics/models/build.py` (REFACTOR)

```python
"""Modern model builder using architecture registry."""

import torch
import torch.nn as nn
from typing import Optional

from .architectures.registry import get_architecture_builder, list_architectures


def build_model(cfg, device=None, rank=None):
    """
    Build model from configuration using architecture registry.

    Args:
        cfg: Hydra config object
        device: torch.device (optional, auto-detected if None)
        rank: rank for DDP (optional)

    Returns:
        Model ready for training

    Available architectures: call list_architectures()
    """
    model_arch = cfg.model.architecture

    # Get builder from registry
    builder = get_architecture_builder(model_arch)

    # Build model
    model = builder(cfg)

    print(f'Model: {model.__class__.__name__} ({model_arch})')
    if hasattr(model, 'get_model_info'):
        info = model.get_model_info()
        print(f'  Parameters: {info["parameters"]:,}')
        print(f'  Deep Supervision: {info["deep_supervision"]}')

    # Move to device (Lightning handles DDP, so no manual parallelization)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    return model
```

**Purpose:** Registry-based building, better error messages

---

### Phase 3: Deep Supervision (Week 3)

#### Task 3.1: Update Lightning Module
**File:** `connectomics/lightning/lit_model.py` (REFACTOR)

Add deep supervision support to `training_step`, `validation_step`, `test_step`:

```python
def training_step(self, batch, batch_idx):
    """Training step with deep supervision support."""
    images = batch['image']
    labels = batch['label']

    # Forward pass
    outputs = self(images)

    # Check for deep supervision
    is_deep_supervision = isinstance(outputs, dict) and any(k.startswith('ds_') for k in outputs.keys())

    # Compute loss
    total_loss = 0.0
    loss_dict = {}

    if is_deep_supervision:
        # Multi-scale loss
        main_output = outputs['output']
        ds_outputs = [outputs[f'ds_{i}'] for i in range(1, 5) if f'ds_{i}' in outputs]

        # Weights: [1.0, 0.5, 0.25, 0.125, 0.0625]
        ds_weights = [1.0] + [0.5 ** i for i in range(1, len(ds_outputs) + 1)]
        all_outputs = [main_output] + ds_outputs

        for scale_idx, (output, ds_weight) in enumerate(zip(all_outputs, ds_weights)):
            # Match target to output size
            target = self._match_target_to_output(labels, output)

            # Compute loss for this scale
            scale_loss = 0.0
            for loss_fn, weight in zip(self.loss_functions, self.loss_weights):
                loss = loss_fn(output, target)
                scale_loss += loss * weight

            total_loss += scale_loss * ds_weight
            loss_dict[f'train_loss_scale_{scale_idx}'] = scale_loss.item()

        loss_dict['train_loss_total'] = total_loss.item()

    else:
        # Standard single-scale loss
        for i, (loss_fn, weight) in enumerate(zip(self.loss_functions, self.loss_weights)):
            loss = loss_fn(outputs, labels)
            total_loss += loss * weight
            loss_dict[f'train_loss_{i}'] = loss.item()

        loss_dict['train_loss_total'] = total_loss.item()

    self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    return total_loss


def _match_target_to_output(self, target, output):
    """Match target size to output size for deep supervision."""
    if target.shape == output.shape:
        return target

    # Downsample target to match output
    return nn.functional.interpolate(
        target.float(),
        size=output.shape[2:],
        mode='nearest' if target.dtype in [torch.long, torch.int] else 'trilinear',
    )
```

**Purpose:** Enable MedNeXt's multi-scale deep supervision

---

### Phase 4: Configuration (Week 4)

#### Task 4.1: Add MedNeXt Config Options
**File:** `connectomics/config/hydra_config.py` (UPDATE)

```python
from dataclasses import dataclass, field
from typing import List, Union, Optional

@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Architecture
    architecture: str = 'monai_basic_unet3d'

    # I/O
    input_size: List[int] = field(default_factory=lambda: [128, 128, 128])
    in_channels: int = 1
    out_channels: int = 1

    # General
    filters: tuple = (32, 64, 128, 256, 512)
    kernel_size: int = 3
    dropout: float = 0.0
    norm: str = "batch"
    activation: str = "relu"

    # Deep supervision (MedNeXt, nnUNet)
    deep_supervision: bool = False

    # MedNeXt-specific
    mednext_size: str = "S"                              # S, B, M, L
    mednext_base_channels: int = 32
    mednext_exp_r: Union[int, List[int]] = 4
    mednext_do_res: bool = True
    mednext_do_res_up_down: bool = True
    mednext_block_counts: List[int] = field(default_factory=lambda: [2,2,2,2,2,2,2,2,2])
    mednext_checkpoint: Optional[str] = None             # 'outside_block' for M/L
    mednext_norm: str = "group"
    mednext_dim: str = "3d"
    mednext_grn: bool = False

    # Loss
    loss_functions: List[str] = field(default_factory=lambda: ["DiceLoss"])
    loss_weights: List[float] = field(default_factory=lambda: [1.0])
```

**Purpose:** Full MedNeXt config support

---

### Phase 5: Examples & Tests (Week 5)

#### Task 5.1: Create MedNeXt Example Config
**File:** `tutorials/mednext_lucchi.yaml` (NEW)

```yaml
# MedNeXt training for Lucchi dataset
# Based on MEDNEXT.md recommendations

system:
  num_gpus: 1
  num_cpus: 4
  seed: 42

model:
  architecture: mednext
  in_channels: 1
  out_channels: 2
  input_size: [128, 128, 128]

  # MedNeXt config
  mednext_size: S                      # Start with Small
  kernel_size: 3                       # 3x3x3 kernels
  deep_supervision: true               # Critical for performance
  mednext_do_res: true
  mednext_do_res_up_down: true
  mednext_norm: group

  # Loss
  loss_functions:
    - DiceLoss
    - CrossEntropyLoss
  loss_weights: [1.0, 1.0]

data:
  train_image: "datasets/lucchi/train_image.h5"
  train_label: "datasets/lucchi/train_label.h5"
  val_image: "datasets/lucchi/val_image.h5"
  val_label: "datasets/lucchi/val_label.h5"

  # MedNeXt prefers 1mm isotropic spacing
  patch_size: [128, 128, 128]
  batch_size: 2
  num_workers: 4

optimizer:
  name: AdamW                          # MedNeXt default
  lr: 1e-3                             # MedNeXt default
  weight_decay: 1e-4

scheduler:
  name: none                           # MedNeXt uses constant LR

training:
  max_epochs: 100
  precision: "16-mixed"
  gradient_clip_val: 1.0

checkpoint:
  monitor: "val/loss"
  mode: "min"
  save_top_k: 3
```

**Purpose:** Ready-to-use MedNeXt configuration

---

#### Task 5.2: Add Tests
**File:** `tests/test_mednext.py` (NEW)

```python
"""Tests for MedNeXt integration."""

import pytest
import torch
from omegaconf import OmegaConf
from connectomics.models.architectures.registry import is_architecture_available
from connectomics.models import build_model


def test_mednext_registered():
    """Test MedNeXt is registered."""
    assert is_architecture_available('mednext')


@pytest.mark.parametrize('model_size', ['S', 'B', 'M', 'L'])
def test_mednext_sizes(model_size):
    """Test all MedNeXt sizes."""
    cfg = OmegaConf.create({
        'model': {
            'architecture': 'mednext',
            'in_channels': 1,
            'out_channels': 2,
            'mednext_size': model_size,
            'kernel_size': 3,
            'deep_supervision': False,
        }
    })
    model = build_model(cfg)
    assert model is not None


def test_mednext_forward():
    """Test forward pass."""
    cfg = OmegaConf.create({
        'model': {
            'architecture': 'mednext',
            'in_channels': 1,
            'out_channels': 2,
            'mednext_size': 'S',
            'kernel_size': 3,
            'deep_supervision': False,
        }
    })
    model = build_model(cfg)
    x = torch.randn(1, 1, 64, 64, 64)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (1, 2, 64, 64, 64)


def test_mednext_deep_supervision():
    """Test deep supervision outputs."""
    cfg = OmegaConf.create({
        'model': {
            'architecture': 'mednext',
            'in_channels': 1,
            'out_channels': 2,
            'mednext_size': 'S',
            'kernel_size': 3,
            'deep_supervision': True,
        }
    })
    model = build_model(cfg)
    x = torch.randn(1, 1, 128, 128, 128)

    with torch.no_grad():
        outputs = model(x)

    assert isinstance(outputs, dict)
    assert 'output' in outputs
    assert 'ds_1' in outputs
    assert outputs['output'].shape[1] == 2
```

---

## 4. Migration Timeline

### Week 1: Foundation
- Create `models/architectures/` directory
- Implement `registry.py`
- Implement `base.py`
- Write registry tests

### Week 2: MONAI Refactoring
- Create `monai_models.py` with registry
- Update `build.py` to use registry (remove YACS support)
- Remove `make_parallel` (Lightning handles DDP)
- Test all MONAI models with Lightning
- Update docs

### Week 3: MedNeXt Integration
- Implement `mednext_models.py`
- Fix import path (`nnunet_mednext`)
- Test MedNeXt builds and runs
- Implement deep supervision in Lightning

### Week 4: Configuration
- Add MedNeXt params to `hydra_config.py`
- Create example configs
- Test config validation

### Week 5: Testing & Docs
- Write integration tests
- Update CLAUDE.md
- Create migration guide
- Performance profiling

---

## 5. Breaking Changes

### What Changes
- **YACS configs removed**: Only Hydra/OmegaConf configs supported
- **Legacy trainer removed**: Only PyTorch Lightning trainer
- **Old entry points removed**: Only `scripts/main.py` (Lightning-based)
- **Manual parallelization removed**: Lightning handles DDP/DP automatically

### Migration Path
Users need to:
1. Convert YACS configs (`config/defaults.py`) to Hydra YAML (`tutorials/*.yaml`)
2. Use `scripts/main.py` instead of legacy scripts
3. Update training commands to Lightning CLI
4. Remove custom distributed training code (Lightning handles it)

**Benefits of Clean Break:**
- Simpler codebase (no dual config support)
- Easier maintenance (one way to do things)
- Better testing (fewer code paths)
- Modern best practices only

---

## 6. Benefits

### For Users
✅ Easy to add new architectures (register + done)
✅ Better error messages (registry validation)
✅ State-of-art MedNeXt performance
✅ Deep supervision for better accuracy

### For Developers
✅ Clean architecture (Lightning + MONAI/MedNeXt)
✅ Registry pattern = extensible
✅ Type-safe configs (Hydra dataclasses)
✅ Modular, testable code
✅ No legacy code maintenance
✅ Single source of truth

### For Research
✅ Quick architecture comparisons
✅ MedNeXt state-of-art segmentation
✅ Multi-scale learning (deep supervision)
✅ Reproducible configs

---

## 7. References

- **DESIGN.md**: Lightning + MONAI principles
- **MEDNEXT.md**: MedNeXt integration details
- **CLAUDE.md**: Codebase overview
- **MedNeXt Paper**: https://arxiv.org/abs/2303.09975
- **MedNeXt Repo**: /projects/weilab/weidf/lib/MedNeXt/

---

## 8. Key Decisions

### Q: Remove YACS config support entirely?
**A:** YES. Clean break. Hydra only. Simpler codebase.

### Q: Remove legacy trainer (`engine/trainer.py`)?
**A:** YES. Lightning only. No custom training loops.

### Q: Remove manual DDP/DP code?
**A:** YES. Lightning handles all parallelization.

### Q: Where to add UpKern weight loading?
**A:** `mednext_models.py` as utility function.

### Q: Handle MedNeXt's 1mm isotropic preference?
**A:** Document in config, let users configure preprocessing.

### Q: Add nnUNet too?
**A:** Phase 2, after MedNeXt is stable.

### Q: Migration support?
**A:** Provide conversion script from YACS → Hydra. Document migration steps.

---

**Next Steps:**
1. Review plan
2. Get approval
3. Start Phase 1 (Foundation)