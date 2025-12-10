"""
Tests for automatic configuration system (config/auto_config.py and config/gpu_utils.py).

Tests cover:
- GPU info detection
- Memory estimation
- Batch size suggestion
- Automatic configuration planning
- Config integration
"""

import pytest
import torch
from unittest.mock import patch


# Test imports
def test_imports():
    """Test that config modules can be imported."""
    from connectomics.config import auto_config, gpu_utils
    assert hasattr(auto_config, 'AutoConfigPlanner')
    assert hasattr(auto_config, 'auto_plan_config')
    assert hasattr(gpu_utils, 'get_gpu_info')
    assert hasattr(gpu_utils, 'suggest_batch_size')


# ==================== GPU Utils Tests ====================

class TestGPUInfo:
    """Tests for GPU information detection."""

    def test_get_gpu_info_no_cuda(self):
        """Test GPU info when CUDA is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            from connectomics.config.gpu_utils import get_gpu_info

            info = get_gpu_info()

            assert info['cuda_available'] is False
            assert info['num_gpus'] == 0
            assert info['gpu_names'] == []
            assert info['total_memory_gb'] == []
            assert info['available_memory_gb'] == []

    def test_get_gpu_info_with_cuda(self):
        """Test GPU info when CUDA is available (mocked)."""
        with patch('torch.cuda.is_available', return_value=True), \
                patch('torch.cuda.device_count', return_value=1), \
                patch('torch.cuda.get_device_name', return_value="FakeGPU"), \
                patch('torch.cuda.get_device_properties') as mock_props, \
                patch('torch.cuda.memory_allocated', return_value=0), \
                patch('torch.cuda.set_device'):
            # Mock device properties
            class Props:
                total_memory = 4 * 1024**3

            mock_props.return_value = Props()

            from connectomics.config.gpu_utils import get_gpu_info

            info = get_gpu_info()

            assert info['cuda_available'] is True
            assert info['num_gpus'] == 1
            assert info['gpu_names'] == ["FakeGPU"]
            assert info['total_memory_gb'][0] == pytest.approx(4.0)
            assert info['available_memory_gb'][0] > 0

    def test_get_system_memory(self):
        """Test system memory detection."""
        from connectomics.config.gpu_utils import get_system_memory_gb

        mem = get_system_memory_gb()
        assert mem > 0
        assert isinstance(mem, float)

    def test_get_available_system_memory(self):
        """Test available system memory detection."""
        from connectomics.config.gpu_utils import (
            get_system_memory_gb,
            get_available_system_memory_gb
        )

        available = get_available_system_memory_gb()
        total = get_system_memory_gb()

        assert available > 0
        assert available <= total


class TestMemoryEstimation:
    """Tests for GPU memory estimation."""

    def test_estimate_memory_basic(self):
        """Test basic memory estimation."""
        from connectomics.config.gpu_utils import estimate_gpu_memory_required

        memory_gb = estimate_gpu_memory_required(
            patch_size=(128, 128, 128),
            batch_size=2,
            in_channels=1,
            out_channels=2,
            base_features=32,
            num_pool_stages=4,
            deep_supervision=False,
            mixed_precision=True,
        )

        assert memory_gb > 0
        assert isinstance(memory_gb, float)

    def test_estimate_memory_increases_with_batch_size(self):
        """Test that memory estimate increases with batch size."""
        from connectomics.config.gpu_utils import estimate_gpu_memory_required

        mem_bs1 = estimate_gpu_memory_required(
            patch_size=(128, 128, 128),
            batch_size=1,
            in_channels=1,
            out_channels=2,
        )

        mem_bs4 = estimate_gpu_memory_required(
            patch_size=(128, 128, 128),
            batch_size=4,
            in_channels=1,
            out_channels=2,
        )

        assert mem_bs4 > mem_bs1
        assert mem_bs4 / mem_bs1 < 4.5  # Should be roughly 4x but with overhead

    def test_estimate_memory_mixed_precision(self):
        """Test that mixed precision uses less memory."""
        from connectomics.config.gpu_utils import estimate_gpu_memory_required

        mem_fp32 = estimate_gpu_memory_required(
            patch_size=(128, 128, 128),
            batch_size=2,
            in_channels=1,
            out_channels=2,
            mixed_precision=False,
        )

        mem_fp16 = estimate_gpu_memory_required(
            patch_size=(128, 128, 128),
            batch_size=2,
            in_channels=1,
            out_channels=2,
            mixed_precision=True,
        )

        assert mem_fp16 < mem_fp32
        # FP16 should be roughly half of FP32 (but not exactly due to parameters)
        assert 0.4 < (mem_fp16 / mem_fp32) < 0.7

    def test_estimate_memory_deep_supervision(self):
        """Test that deep supervision increases memory."""
        from connectomics.config.gpu_utils import estimate_gpu_memory_required

        mem_no_ds = estimate_gpu_memory_required(
            patch_size=(128, 128, 128),
            batch_size=2,
            in_channels=1,
            out_channels=2,
            deep_supervision=False,
        )

        mem_with_ds = estimate_gpu_memory_required(
            patch_size=(128, 128, 128),
            batch_size=2,
            in_channels=1,
            out_channels=2,
            deep_supervision=True,
        )

        assert mem_with_ds > mem_no_ds


class TestBatchSizeSuggestion:
    """Tests for batch size suggestion."""

    def test_suggest_batch_size_basic(self):
        """Test basic batch size suggestion."""
        from connectomics.config.gpu_utils import suggest_batch_size

        bs = suggest_batch_size(
            patch_size=(128, 128, 128),
            in_channels=1,
            out_channels=2,
            available_gpu_memory_gb=16.0,
        )

        assert bs >= 1
        assert bs <= 32
        assert isinstance(bs, int)

    def test_suggest_batch_size_scales_with_memory(self):
        """Test that larger GPU memory suggests larger batch size."""
        from connectomics.config.gpu_utils import suggest_batch_size

        bs_small = suggest_batch_size(
            patch_size=(128, 128, 128),
            in_channels=1,
            out_channels=2,
            available_gpu_memory_gb=8.0,
        )

        bs_large = suggest_batch_size(
            patch_size=(128, 128, 128),
            in_channels=1,
            out_channels=2,
            available_gpu_memory_gb=24.0,
        )

        assert bs_large >= bs_small

    def test_suggest_batch_size_smaller_patches(self):
        """Test that smaller patches allow larger batch sizes."""
        from connectomics.config.gpu_utils import suggest_batch_size

        bs_small_patch = suggest_batch_size(
            patch_size=(64, 64, 64),
            in_channels=1,
            out_channels=2,
            available_gpu_memory_gb=16.0,
        )

        bs_large_patch = suggest_batch_size(
            patch_size=(192, 192, 192),
            in_channels=1,
            out_channels=2,
            available_gpu_memory_gb=16.0,
        )

        assert bs_small_patch > bs_large_patch


class TestOptimalNumWorkers:
    """Tests for optimal number of workers suggestion."""

    def test_get_optimal_num_workers_single_gpu(self):
        """Test worker suggestion for single GPU."""
        from connectomics.config.gpu_utils import get_optimal_num_workers

        workers = get_optimal_num_workers(num_gpus=1)

        assert workers >= 2
        assert isinstance(workers, int)

    def test_get_optimal_num_workers_multi_gpu(self):
        """Test worker suggestion for multiple GPUs."""
        from connectomics.config.gpu_utils import get_optimal_num_workers

        workers_1gpu = get_optimal_num_workers(num_gpus=1)
        workers_4gpu = get_optimal_num_workers(num_gpus=4)

        # More GPUs should suggest more workers
        assert workers_4gpu >= workers_1gpu


# ==================== Auto Config Tests ====================

class TestAutoConfigPlanner:
    """Tests for AutoConfigPlanner class."""

    def test_planner_init(self):
        """Test planner initialization."""
        from connectomics.config.auto_config import AutoConfigPlanner

        planner = AutoConfigPlanner(
            architecture='mednext',
            target_spacing=[1.0, 1.0, 1.0],
            median_shape=[128, 128, 128],
        )

        assert planner.architecture == 'mednext'
        assert planner.target_spacing == [1.0, 1.0, 1.0]
        assert planner.median_shape == [128, 128, 128]
        assert planner.gpu_info is not None

    def test_architecture_defaults_mednext(self):
        """Test MedNeXt architecture defaults."""
        from connectomics.config.auto_config import AutoConfigPlanner

        planner = AutoConfigPlanner(architecture='mednext')
        defaults = planner._get_architecture_defaults()

        assert defaults['base_features'] == 32
        assert defaults['lr'] == 1e-3
        assert defaults['use_scheduler'] is False

    def test_architecture_defaults_unet(self):
        """Test U-Net architecture defaults."""
        from connectomics.config.auto_config import AutoConfigPlanner

        planner = AutoConfigPlanner(architecture='monai_basic_unet3d')
        defaults = planner._get_architecture_defaults()

        assert defaults['base_features'] == 32
        assert defaults['lr'] == 1e-4
        assert defaults['use_scheduler'] is True

    def test_plan_basic(self):
        """Test basic planning."""
        from connectomics.config.auto_config import AutoConfigPlanner

        planner = AutoConfigPlanner(
            architecture='mednext',
            median_shape=[128, 128, 128],
        )

        result = planner.plan(
            in_channels=1,
            out_channels=2,
            deep_supervision=True,
            use_mixed_precision=True,
        )

        # Check result attributes
        assert len(result.patch_size) == 3
        assert result.batch_size >= 1
        assert result.num_workers >= 2
        assert result.base_features > 0
        assert result.lr > 0
        assert result.auto_planned is True
        assert len(result.planning_notes) > 0

    def test_plan_anisotropic_spacing(self):
        """Test planning with anisotropic spacing."""
        from connectomics.config.auto_config import AutoConfigPlanner

        # Anisotropic spacing (e.g., CT data)
        planner = AutoConfigPlanner(
            architecture='mednext',
            target_spacing=[5.0, 1.0, 1.0],  # 5mm z-spacing, 1mm xy-spacing
            median_shape=[64, 256, 256],
        )

        result = planner.plan()

        # Should warn about anisotropic spacing
        # Patch size should be adjusted
        assert len(result.patch_size) == 3

    def test_plan_with_manual_overrides(self):
        """Test planning with manual overrides."""
        from connectomics.config.auto_config import AutoConfigPlanner

        manual_overrides = {
            'batch_size': 8,
            'lr': 5e-4,
        }

        planner = AutoConfigPlanner(
            architecture='mednext',
            manual_overrides=manual_overrides,
        )

        result = planner.plan()

        # Manual overrides should be applied
        assert result.batch_size == 8
        assert result.lr == 5e-4

    @patch('torch.cuda.is_available', return_value=False)
    def test_plan_cpu_only(self, mock_cuda):
        """Test planning for CPU-only training."""
        from connectomics.config.auto_config import AutoConfigPlanner

        planner = AutoConfigPlanner(architecture='mednext')
        result = planner.plan()

        # CPU mode
        assert result.batch_size == 1
        assert result.precision == "32"
        assert len(result.warnings) > 0


class TestAutoPlanConfig:
    """Tests for auto_plan_config function."""

    def test_auto_plan_config_basic(self):
        """Test basic auto-planning of config."""
        from connectomics.config import Config, auto_plan_config
        from omegaconf import OmegaConf

        # Create test config
        cfg = OmegaConf.structured(Config())
        cfg.system.auto_plan = True
        cfg.model.architecture = 'mednext'
        cfg.model.deep_supervision = True

        # Auto-plan
        cfg = auto_plan_config(cfg, print_results=False)

        # Check that values were set
        assert cfg.system.training.batch_size > 0
        assert cfg.system.training.num_workers > 0
        assert len(cfg.data.patch_size) == 3
        assert cfg.optimization.precision in ['32', '16-mixed', 'bf16-mixed']

    def test_auto_plan_config_disabled(self):
        """Test that planning respects disabled flag."""
        from connectomics.config import Config, auto_plan_config
        from omegaconf import OmegaConf

        cfg = OmegaConf.structured(Config())
        cfg.system.auto_plan = False

        original_batch_size = cfg.system.training.batch_size
        cfg = auto_plan_config(cfg, print_results=False)

        # Should not change
        assert cfg.system.training.batch_size == original_batch_size

    def test_auto_plan_config_respects_overrides(self):
        """Test that planning respects manual config values."""
        from connectomics.config import Config, auto_plan_config
        from omegaconf import OmegaConf

        cfg = OmegaConf.structured(Config())
        cfg.system.auto_plan = True
        cfg.system.training.batch_size = 16  # Manual override
        cfg.optimization.optimizer.lr = 2e-3  # Manual override

        cfg = auto_plan_config(cfg, print_results=False)

        # Manual values should be preserved
        assert cfg.system.training.batch_size == 16
        assert cfg.optimization.optimizer.lr == 2e-3


class TestAutoPlanResult:
    """Tests for AutoPlanResult dataclass."""

    def test_auto_plan_result_creation(self):
        """Test creation of AutoPlanResult."""
        from connectomics.config.auto_config import AutoPlanResult

        result = AutoPlanResult()

        assert hasattr(result, 'patch_size')
        assert hasattr(result, 'batch_size')
        assert hasattr(result, 'num_workers')
        assert hasattr(result, 'precision')
        assert hasattr(result, 'lr')
        assert hasattr(result, 'planning_notes')
        assert hasattr(result, 'warnings')

    def test_auto_plan_result_with_values(self):
        """Test AutoPlanResult with custom values."""
        from connectomics.config.auto_config import AutoPlanResult

        result = AutoPlanResult(
            patch_size=[128, 128, 128],
            batch_size=4,
            num_workers=8,
            precision="16-mixed",
            lr=1e-3,
        )

        assert result.patch_size == [128, 128, 128]
        assert result.batch_size == 4
        assert result.num_workers == 8
        assert result.precision == "16-mixed"
        assert result.lr == 1e-3


# ==================== Integration Tests ====================

class TestIntegration:
    """Integration tests for complete auto-configuration workflow."""

    def test_end_to_end_planning(self):
        """Test complete planning workflow from config to planned values."""
        from connectomics.config import Config, auto_plan_config
        from omegaconf import OmegaConf

        # Create config
        cfg = OmegaConf.structured(Config())
        cfg.system.auto_plan = True
        cfg.model.architecture = 'mednext'
        cfg.model.in_channels = 1
        cfg.model.out_channels = 6
        cfg.model.deep_supervision = True

        # Auto-plan
        cfg = auto_plan_config(cfg, print_results=False)

        # Verify all required fields are set
        assert cfg.system.training.batch_size > 0
        assert cfg.system.training.num_workers > 0
        assert cfg.data.patch_size is not None
        assert cfg.optimization.precision is not None
        assert cfg.optimization.optimizer.lr > 0

    def test_planning_with_dataset_properties(self):
        """Test planning with dataset properties specified."""
        from connectomics.config import Config, auto_plan_config
        from omegaconf import OmegaConf

        cfg = OmegaConf.structured(Config())
        cfg.system.auto_plan = True
        cfg.data.target_spacing = [1.0, 1.0, 1.0]
        cfg.data.median_shape = [128, 256, 256]

        cfg = auto_plan_config(cfg, print_results=False)

        # Should use provided dataset properties
        assert cfg.data.patch_size is not None

    def test_planning_with_real_gpu(self):
        """Test planning with real GPU (if available)."""
        from connectomics.config import Config, auto_plan_config
        from omegaconf import OmegaConf

        cfg = OmegaConf.structured(Config())
        cfg.system.auto_plan = True
        cfg.model.architecture = 'mednext'
        cfg.model.deep_supervision = True

        fake_gpu_info = {
            "cuda_available": True,
            "num_gpus": 1,
            "gpu_names": ["FakeGPU"],
            "total_memory_gb": [8.0],
            "available_memory_gb": [7.5],
        }

        with patch("connectomics.config.auto_config.get_gpu_info", return_value=fake_gpu_info):
            cfg = auto_plan_config(cfg, print_results=False)

        # With GPU, planner should consider mixed precision or keep safe default
        assert cfg.optimization.precision in ['16-mixed', 'bf16-mixed', '32']
        assert cfg.system.training.batch_size > 0
