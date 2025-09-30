# PyTorch Connectomics Refactoring Plan

## Executive Summary

This plan outlines a comprehensive refactoring strategy to modernize PyTorch Connectomics by leveraging established frameworks like MONAI, nnUNet, and PyTorch Lightning, while learning from the BANIS codebase architecture. The goal is to reduce custom code, improve maintainability, and align with community standards.

## Current Architecture Analysis

### Strengths
- Comprehensive 3D/volumetric data handling
- Flexible configuration system (YACS)
- Multi-task learning capabilities
- Extensive augmentation pipeline
- Support for various EM-specific tasks

### Pain Points
- **Custom Training Loop**: 800+ lines in `trainer.py` with manual optimization, mixed precision, distributed training
- **Reinvented Components**: Custom model architectures when MONAI/nnUNet equivalents exist
- **Data Pipeline Complexity**: Custom dataset classes that could leverage MONAI transforms
- **Configuration Overhead**: YACS system more complex than modern alternatives
- **Testing Infrastructure**: Limited integration testing, manual validation workflows

## Reference Architecture: BANIS Lessons

### Key Insights from BANIS
1. **PyTorch Lightning**: Clean, modular training with automatic distributed training, mixed precision, checkpointing
2. **Modern Model Integration**: Uses nnUNet/MedNeXt instead of custom architectures
3. **Simplified Configuration**: YAML-based hyperparameter management with automatic grid search
4. **Lean Codebase**: ~500 lines vs 10k+ lines for similar functionality
5. **MONAI Integration**: Leverages MONAI transforms for medical image processing

## Refactoring Strategy: 3-Phase Approach

### Phase 1: Foundation Migration (Months 1-2)

#### 1.1 PyTorch Lightning Integration
**Priority: HIGH**

```python
# Target Architecture
class ConnectomicsModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.model = self.build_model()
        self.criterion = self.build_criterion()

    def configure_optimizers(self):
        # Replace custom solver with Lightning optimizers
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        # Replace 100+ line training loop with clean step
        return self.shared_step(batch, 'train')
```

**Benefits:**
- Automatic distributed training, mixed precision, checkpointing
- Built-in logging, profiling, device management
- Eliminate ~500 lines of custom training code

**Migration Path:**
1. Create `ConnectomicsModule` inheriting from `LightningModule`
2. Move model building logic to Lightning module
3. Replace custom training loop with Lightning callbacks
4. Migrate configuration to Lightning's hyperparameter system

#### 1.2 MONAI Data Pipeline Integration
**Priority: HIGH**

**Current Issues:**
- Custom augmentation classes (`connectomics/data/augmentation/`)
- Manual data loading and preprocessing
- Inconsistent tensor handling

**Target Architecture:**
```python
# Replace custom augmentations with MONAI
from monai.transforms import (
    Compose, RandRotate90d, RandFlipd, RandAffined,
    RandGaussianNoised, RandIntensityShiftd
)

transforms = Compose([
    RandRotate90d(keys=['image', 'label'], prob=0.5),
    RandFlipd(keys=['image', 'label'], prob=0.5),
    RandAffined(keys=['image', 'label'], prob=0.5),
    # EM-specific transforms can be custom additions
])
```

**Migration Steps:**
1. Map existing augmentations to MONAI equivalents
2. Create custom MONAI transforms for EM-specific operations
3. Replace `VolumeDataset` with MONAI `Dataset` + `DataLoader`
4. Integrate with Lightning's `LightningDataModule`

#### 1.3 Configuration Modernization
**Priority: MEDIUM**

**Replace YACS with Hydra/Lightning:**
```yaml
# New config structure (inspired by BANIS)
model:
  architecture: "unet_3d"
  backbone: "resnet"
  filters: [28, 36, 48, 64, 80]

data:
  batch_size: 8
  num_workers: 4
  transforms:
    rotation: true
    flip: true

training:
  max_epochs: 100
  learning_rate: 1e-3
  weight_decay: 1e-4
```

### Phase 2: Model Architecture Modernization (Months 2-3)

#### 2.1 nnUNet Integration
**Priority: HIGH**

**Current State:** Custom UNet implementations in `connectomics/models/arch/`
**Target:** Leverage nnUNet's proven architectures

```python
# Replace custom models with nnUNet
from nnunet_mednext import create_mednext_v1

class ConnectomicsUNet(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = create_mednext_v1(
            num_input_channels=kwargs['in_channels'],
            num_classes=kwargs['out_channels'],
            model_id=kwargs.get('model_id', 'S'),
            kernel_size=kwargs.get('kernel_size', 3)
        )
```

**Migration Strategy:**
1. **Keep Core Architectures:** Maintain UNet3D, FPN3D for backward compatibility
2. **Add nnUNet Options:** Introduce `nnunet_*` model types in `MODEL_MAP`
3. **Gradual Migration:** New experiments use nnUNet, existing configs unchanged
4. **Performance Validation:** Benchmark nnUNet vs custom models on key datasets

#### 2.2 MONAI Model Integration
**Priority: MEDIUM**

**Target Models:**
- MONAI SwinUNETR (already partially integrated)
- MONAI UNETR (already partially integrated)
- MONAI SegResNet
- MONAI DiNTS

**Benefits:**
- Pre-trained weights for transfer learning
- Optimized implementations
- Regular updates and bug fixes

### Phase 3: Advanced Features & Optimization (Months 3-4)

#### 3.1 Advanced Training Features
**Priority: MEDIUM**

**Current Gaps:**
- Manual mixed precision implementation
- Custom distributed training logic
- Limited experiment tracking

**Target Features:**
```python
# Lightning callbacks for advanced features
callbacks = [
    ModelCheckpoint(monitor='val_loss'),
    EarlyStopping(monitor='val_loss', patience=10),
    LearningRateMonitor(),
    StochasticWeightAveraging(),  # Replace custom SWA
]

# Automatic mixed precision
trainer = pl.Trainer(
    precision=16,  # Replaces custom GradScaler
    strategy='ddp',  # Replaces custom distributed training
    callbacks=callbacks
)
```

#### 3.2 Experiment Management Integration
**Priority: LOW**

**Options:**
- Weights & Biases integration (like BANIS)
- MLflow integration
- TensorBoard (current, but improved)

#### 3.3 Testing Infrastructure Overhaul
**Priority: MEDIUM**

**Current Issues:** Limited integration tests, manual validation
**Target:** Automated testing pipeline

```python
# Pytest-based testing
class TestConnectomicsModule:
    def test_forward_pass(self):
        model = ConnectomicsModule(cfg)
        x = torch.randn(1, 1, 64, 64, 64)
        y = model(x)
        assert y.shape == expected_shape

    def test_training_step(self):
        # Test training step functionality
        pass
```

## Implementation Roadmap

### Week 1-2: Lightning Migration
- [ ] Create `ConnectomicsModule` base class
- [ ] Migrate optimizer and scheduler configuration
- [ ] Implement training/validation steps
- [ ] Test with existing datasets

### Week 3-4: Data Pipeline
- [ ] Map existing augmentations to MONAI
- [ ] Create `ConnectomicsDataModule`
- [ ] Integrate with Lightning data loading
- [ ] Validate data consistency

### Week 5-6: Model Integration
- [ ] Add nnUNet model options
- [ ] Integrate MONAI models
- [ ] Create model factory pattern
- [ ] Benchmark performance

### Week 7-8: Configuration & Testing
- [ ] Migrate to Hydra/Lightning config
- [ ] Create comprehensive test suite
- [ ] Documentation updates
- [ ] Performance optimization

## Benefits Analysis

### Immediate Benefits (Phase 1)
- **Code Reduction:** ~50% reduction in core training code
- **Maintenance:** Leverage community-maintained libraries
- **Features:** Automatic distributed training, mixed precision, checkpointing
- **Debugging:** Better error handling and logging

### Medium-term Benefits (Phase 2-3)
- **Performance:** Optimized model implementations
- **Reproducibility:** Standardized experiment management
- **Collaboration:** Easier for new contributors
- **Innovation:** Focus on EM-specific problems vs infrastructure

### Quantitative Metrics
- **Lines of Code:** Target 50% reduction (10k → 5k lines)
- **Dependencies:** Reduce custom code, increase standard libraries
- **Test Coverage:** 80% code coverage (currently ~30%)
- **Training Speed:** 10-20% improvement from optimized implementations

## Risk Mitigation

### Backward Compatibility
- **Dual API:** Maintain old API during transition
- **Configuration Migration:** Automatic config translation
- **Model Weights:** Ensure checkpoint compatibility

### Performance Regression
- **Benchmarking:** Continuous performance monitoring
- **Fallback Options:** Keep critical custom implementations
- **Gradual Migration:** Opt-in basis for new features

### Validation Strategy
- **Reference Datasets:** Reproduce existing results on key datasets
- **A/B Testing:** Compare old vs new implementations
- **Community Feedback:** Beta testing with existing users

## Success Metrics

### Technical Metrics
- [ ] 50% reduction in custom training code
- [ ] 80% test coverage
- [ ] 100% backward compatibility for existing configs
- [ ] <5% performance regression on benchmark datasets

### Community Metrics
- [ ] Faster onboarding for new contributors
- [ ] Increased code reuse across projects
- [ ] Better integration with broader medical imaging ecosystem

## Conclusion

This refactoring plan positions PyTorch Connectomics as a modern, maintainable framework that leverages the best practices from the medical imaging and deep learning communities. By adopting Lightning, MONAI, and nnUNet, we can focus on connectomics-specific innovations rather than reimplementing common infrastructure.

The phased approach ensures minimal disruption to existing users while providing a clear path to a more robust and scalable codebase.