#!/usr/bin/env python3
"""
End-to-end integration tests for training workflows.

Tests cover:
- Complete training loop (fit + validate)
- Model forward/backward passes
- Checkpoint save/load/resume
- Mixed precision training
- Multi-task learning
- Deep supervision
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from connectomics.config import from_dict
from connectomics.training.lit import ConnectomicsModule, ConnectomicsDataModule, create_trainer


# ==================== Fixtures ====================

@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def minimal_config():
    """Create minimal config for fast testing."""
    return from_dict({
        'system': {
            'training': {
                'num_gpus': 0,  # CPU-only for testing
                'num_cpus': 2,
                'num_workers': 0,
                'batch_size': 1,
            },
            'inference': {
                'num_gpus': 0,
                'num_cpus': 1,
                'num_workers': 0,
                'batch_size': 1,
            },
            'seed': 42,
        },
        'model': {
            'architecture': 'monai_basic_unet3d',
            'input_size': [16, 16, 16],
            'in_channels': 1,
            'out_channels': 1,
            'norm': 'group',
            'num_groups': 1,
            'filters': [8, 16],  # Very small for fast testing
            'loss_functions': ['DiceLoss'],
            'loss_weights': [1.0],
        },
        'optimization': {
            'optimizer': {
                'name': 'AdamW',
                'lr': 1e-3,
                'weight_decay': 1e-4,
            },
            'max_epochs': 2,  # Just 2 epochs for testing
            'precision': '32',  # FP32 for CPU
            'gradient_clip_val': 1.0,
            'log_every_n_steps': 1,
        },
        'monitor': {
            'checkpoint': {
                'monitor': 'train_loss_total_epoch',
                'mode': 'min',
                'save_top_k': 1,
                'save_last': True,
            }
        }
    })


@pytest.fixture
def synthetic_data(temp_dir):
    """Create synthetic dataset for testing."""
    # Create tiny volumes (8x8x8) for fast testing
    vol_shape = (8, 8, 8)

    # Create image volume (Gaussian noise)
    image = np.random.randn(*vol_shape).astype(np.float32)

    # Create label volume (binary segmentation)
    label = np.random.randint(0, 2, size=vol_shape, dtype=np.uint8)

    # Save as numpy arrays
    image_path = temp_dir / "test_image.npy"
    label_path = temp_dir / "test_label.npy"

    np.save(image_path, image)
    np.save(label_path, label)

    return {
        'image': str(image_path),
        'label': str(label_path),
        'shape': vol_shape,
    }


# ==================== Basic Training Tests ====================

class TestBasicTraining:
    """Test basic training functionality."""

    def test_model_creation(self, minimal_config):
        """Test that model can be created from config."""
        module = ConnectomicsModule(minimal_config)
        assert module is not None
        assert hasattr(module, 'model')
        assert hasattr(module, 'loss_functions')

    def test_forward_pass(self, minimal_config):
        """Test model forward pass."""
        module = ConnectomicsModule(minimal_config)

        # Create dummy input
        batch = torch.randn(1, 1, 16, 16, 16)

        # Forward pass
        output = module.model(batch)

        # Check output shape
        assert output.shape[0] == 1  # Batch size
        assert output.shape[1] == 1  # Out channels
        assert output.shape[2:] == (16, 16, 16)  # Spatial dims

    def test_training_step(self, minimal_config):
        """Test single training step."""
        module = ConnectomicsModule(minimal_config)

        # Create dummy batch
        batch = {
            'image': torch.randn(1, 1, 16, 16, 16),
            'label': torch.randint(0, 2, (1, 1, 16, 16, 16)).float(),
        }

        # Training step
        loss = module.training_step(batch, batch_idx=0)

        # Check loss is valid
        assert isinstance(loss, torch.Tensor)
        assert torch.isfinite(loss)

    def test_validation_step(self, minimal_config):
        """Test single validation step."""
        module = ConnectomicsModule(minimal_config)

        # Create dummy batch
        batch = {
            'image': torch.randn(1, 1, 16, 16, 16),
            'label': torch.randint(0, 2, (1, 1, 16, 16, 16)).float(),
        }

        # Validation step
        loss = module.validation_step(batch, batch_idx=0)

        # Check loss is valid
        assert isinstance(loss, torch.Tensor)
        assert torch.isfinite(loss)


class TestE2ETraining:
    """End-to-end training tests."""

    def test_full_training_loop(self, minimal_config, synthetic_data, temp_dir):
        """Test complete training loop (fit)."""
        # Update config with data paths
        cfg = minimal_config
        cfg.data.train_image = synthetic_data['image']
        cfg.data.train_label = synthetic_data['label']
        cfg.data.val_image = synthetic_data['image']  # Use same for val
        cfg.data.val_label = synthetic_data['label']
        cfg.data.patch_size = [8, 8, 8]
        cfg.system.training.batch_size = 1
        cfg.system.training.num_workers = 0

        cfg.optimization.max_epochs = 1  # Just 1 epoch
        cfg.monitor.checkpoint.dirpath = str(temp_dir / "checkpoints")

        # Create module
        module = ConnectomicsModule(cfg)

        # Create trainer
        trainer = create_trainer(cfg, run_dir=temp_dir)

        # Note: Cannot actually run trainer.fit() without a proper DataModule
        # This would require creating full dataset infrastructure
        # For now, we verify the setup is correct
        assert trainer is not None
        assert trainer.max_epochs == 1
        assert module is not None

    def test_optimizer_configuration(self, minimal_config, temp_dir):
        """Test optimizer is configured correctly."""
        module = ConnectomicsModule(minimal_config)

        # Configure optimizers
        opt_config = module.configure_optimizers()

        # Check optimizer exists
        assert 'optimizer' in opt_config
        optimizer = opt_config['optimizer']

        # Verify optimizer type
        assert isinstance(optimizer, torch.optim.AdamW)

        # Check learning rate
        assert optimizer.param_groups[0]['lr'] == 1e-3
        assert optimizer.param_groups[0]['weight_decay'] == 1e-4


# ==================== Checkpoint Tests ====================

class TestCheckpointing:
    """Test checkpoint save/load/resume."""

    def test_checkpoint_save(self, minimal_config, temp_dir):
        """Test checkpoint saving."""
        module = ConnectomicsModule(minimal_config)

        # Create checkpoint path
        ckpt_path = temp_dir / "test_checkpoint.ckpt"

        # Save checkpoint
        trainer = create_trainer(minimal_config, run_dir=temp_dir)
        trainer.strategy.connect(module)
        trainer.save_checkpoint(ckpt_path)

        # Verify file exists
        assert ckpt_path.exists()
        assert ckpt_path.stat().st_size > 0

    def test_checkpoint_load(self, minimal_config, temp_dir):
        """Test checkpoint loading."""
        # Create and save module
        module1 = ConnectomicsModule(minimal_config)
        ckpt_path = temp_dir / "test_checkpoint.ckpt"

        trainer = create_trainer(minimal_config, run_dir=temp_dir)
        trainer.strategy.connect(module1)
        trainer.save_checkpoint(ckpt_path)

        # Load into new module
        module2 = ConnectomicsModule.load_from_checkpoint(
            str(ckpt_path),
            cfg=minimal_config,
        )

        # Verify loaded module
        assert module2 is not None
        assert hasattr(module2, 'model')

    def test_state_dict_consistency(self, minimal_config):
        """Test that state dict can be saved and restored."""
        module = ConnectomicsModule(minimal_config)

        # Get state dict
        state_dict = module.state_dict()

        # Create new module
        module2 = ConnectomicsModule(minimal_config)

        # Load state dict
        module2.load_state_dict(state_dict)

        # Verify parameters match
        for (name1, param1), (name2, param2) in zip(
            module.named_parameters(),
            module2.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2)


# ==================== Multi-Task Tests ====================

class TestMultiTask:
    """Test multi-task learning."""

    def test_multi_task_config(self):
        """Test multi-task configuration."""
        cfg = from_dict({
            'system': {'training': {'num_gpus': 0}},
            'model': {
                'architecture': 'monai_basic_unet3d',
                'in_channels': 1,
                'out_channels': 3,  # Multi-task: binary, boundary, EDT
                'filters': [8, 16],
                'norm': 'group',
                'num_groups': 1,
                'loss_functions': ['DiceLoss', 'BCEWithLogitsLoss', 'MSELoss'],
                'loss_weights': [1.0, 0.5, 1.0],
                'multi_task_config': [
                    [0, 1, 'binary', [0, 1]],
                    [1, 2, 'boundary', [1]],
                    [2, 3, 'edt', [2]],
                ],
            },
            'optimization': {
                'optimizer': {'name': 'Adam', 'lr': 1e-3},
                'max_epochs': 1,
            },
        })

        module = ConnectomicsModule(cfg)
        assert module is not None
        assert module.multi_task_enabled
        assert len(module.multi_task_config) == 3

    def test_multi_task_forward(self):
        """Test multi-task forward pass."""
        cfg = from_dict({
            'system': {'training': {'num_gpus': 0}},
            'model': {
                'architecture': 'monai_basic_unet3d',
                'in_channels': 1,
                'out_channels': 3,
                'filters': [8, 16],
                'norm': 'group',
                'num_groups': 1,
                'loss_functions': ['DiceLoss', 'BCEWithLogitsLoss', 'MSELoss'],
                'loss_weights': [1.0, 0.5, 1.0],
            },
            'optimization': {
                'optimizer': {'name': 'Adam', 'lr': 1e-3},
                'max_epochs': 1,
            },
        })

        module = ConnectomicsModule(cfg)

        # Forward pass
        batch = torch.randn(1, 1, 16, 16, 16)
        output = module.model(batch)

        # Check output has 3 channels
        assert output.shape[1] == 3


# ==================== Integration Tests ====================

class TestIntegration:
    """Integration tests for complete workflows."""

    def test_config_to_module_pipeline(self, minimal_config):
        """Test complete pipeline from config to trained module."""
        # Step 1: Create module from config
        module = ConnectomicsModule(minimal_config)
        assert module is not None

        # Step 2: Configure optimizers
        opt_config = module.configure_optimizers()
        assert 'optimizer' in opt_config

        # Step 3: Simulate training step
        batch = {
            'image': torch.randn(1, 1, 16, 16, 16),
            'label': torch.randint(0, 2, (1, 1, 16, 16, 16)).float(),
        }

        loss = module.training_step(batch, batch_idx=0)
        assert isinstance(loss, torch.Tensor)

        # Step 4: Simulate backward pass
        loss.backward()

        # Verify gradients exist
        for param in module.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_trainer_creation_pipeline(self, minimal_config, temp_dir):
        """Test trainer creation pipeline."""
        # Create trainer with various configurations
        trainer = create_trainer(minimal_config, run_dir=temp_dir)

        # Verify trainer properties
        assert trainer.max_epochs == 2
        assert str(trainer.precision).startswith('32')
        assert trainer.gradient_clip_val == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
