"""
nnUNet model wrappers for loading pretrained models.

Provides wrappers for pretrained nnUNet v2 models that conform to the
ConnectomicsModel interface. Supports loading pretrained checkpoints
with automatic configuration from plans.json and dataset.json.

Uses Hydra/OmegaConf configuration.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn

try:
    import nnunetv2
    from batchgenerators.utilities.file_and_folder_operations import load_json
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    NNUNET_AVAILABLE = True
except ImportError:
    NNUNET_AVAILABLE = False

from .base import ConnectomicsModel
from .registry import register_architecture

logger = logging.getLogger(__name__)


def _check_nnunet_available():
    """Check if nnUNet v2 is installed."""
    if not NNUNET_AVAILABLE:
        raise ImportError(
            "nnUNet v2 is not installed. Install with: pip install nnunetv2\n"
            "For more information, visit: https://github.com/MIC-DKFZ/nnUNet"
        )


class nnUNetWrapper(ConnectomicsModel):
    """
    Wrapper for pretrained nnUNet v2 models.

    Loads a pretrained nnUNet model from checkpoint and provides the
    ConnectomicsModel interface for integration with PyTorch Connectomics.

    Args:
        network: The loaded nnUNet network module
        plans_manager: nnUNet PlansManager for configuration
        configuration_manager: Configuration manager from plans
        dataset_json: Dataset configuration dictionary
        spatial_dims: Spatial dimensions (2 or 3)
        supports_deep_supervision: Whether model was trained with deep supervision

    Attributes:
        network: The nnUNet network
        plans_manager: Plans configuration manager
        configuration_manager: Architecture configuration
        dataset_json: Dataset metadata
        spatial_dims: 2D or 3D model
    """

    def __init__(
        self,
        network: nn.Module,
        plans_manager: Any,
        configuration_manager: Any,
        dataset_json: Dict[str, Any],
        spatial_dims: int = 3,
        supports_deep_supervision: bool = False,
    ):
        super().__init__()
        self.network = network
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.dataset_json = dataset_json
        self.spatial_dims = spatial_dims
        self.supports_deep_supervision = supports_deep_supervision
        self.output_scales = 1  # nnUNet in inference mode returns single output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through nnUNet model.

        Args:
            x: Input tensor of shape (B, C, D, H, W) for 3D or (B, C, H, W) for 2D

        Returns:
            Output tensor with segmentation logits
        """
        was_5d = x.dim() == 5

        # For 2D models with 3D input (e.g., from sliding window), squeeze depth
        if self.spatial_dims == 2 and was_5d and x.size(2) == 1:
            x = x.squeeze(2)  # [B, C, 1, H, W] -> [B, C, H, W]

        # Forward through network
        output = self.network(x)

        # For 2D models, add back depth dimension if input was 5D
        if self.spatial_dims == 2 and output.dim() == 4 and was_5d:
            output = output.unsqueeze(2)  # [B, C, H, W] -> [B, C, 1, H, W]

        return output

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and statistics.

        Returns:
            Dictionary with model metadata including nnUNet-specific info
        """
        info = super().get_model_info()

        # Add nnUNet-specific information
        info.update(
            {
                "framework": "nnUNetv2",
                "spatial_dims": self.spatial_dims,
                "configuration": (
                    self.configuration_manager.configuration_name
                    if hasattr(self.configuration_manager, "configuration_name")
                    else "unknown"
                ),
                "network_class": (
                    self.configuration_manager.network_arch_class_name
                    if hasattr(self.configuration_manager, "network_arch_class_name")
                    else "unknown"
                ),
            }
        )

        return info


@register_architecture("nnunet")
def build_nnunet(cfg) -> ConnectomicsModel:
    """
    Build nnUNet model from pretrained checkpoint.

    Loads a pretrained nnUNet v2 model checkpoint with automatic configuration
    from plans.json and dataset.json files. The model is loaded in inference mode
    (deep supervision disabled) and can be used for prediction or fine-tuning.

    Config parameters:
        Required:
            - model.nnunet.checkpoint: Path to .pth checkpoint file
            - model.nnunet.plans: Path to plans.json file
            - model.nnunet.dataset: Path to dataset.json file

        Optional:
            - model.nnunet.device: Device to load model on ('cuda' or 'cpu', default: 'cuda')
            - model.in_channels: Override input channels (default: from dataset.json)
            - model.out_channels: Override output channels (default: from plans)

    Example config:
        model:
          architecture: nnunet
          nnunet:
            checkpoint: /path/to/checkpoint.pth
            plans: /path/to/plans.json
            dataset: /path/to/dataset.json
            device: cuda

    Args:
        cfg: Hydra config object

    Returns:
        nnUNetWrapper containing the pretrained model

    Raises:
        ImportError: If nnUNet v2 is not installed
        FileNotFoundError: If checkpoint or config files are not found
        RuntimeError: If trainer class cannot be found or loaded
    """
    _check_nnunet_available()

    # Extract paths from config
    checkpoint_raw = getattr(cfg.model.nnunet, "checkpoint", None)
    plans_raw = getattr(cfg.model.nnunet, "plans", None)
    dataset_raw = getattr(cfg.model.nnunet, "dataset", None)
    if checkpoint_raw is None or plans_raw is None or dataset_raw is None:
        raise ValueError(
            "Missing nnUNet model paths. Set model.nnunet.checkpoint, "
            "model.nnunet.plans, and model.nnunet.dataset."
        )
    checkpoint_path = Path(checkpoint_raw)
    plans_path = Path(plans_raw)
    dataset_path = Path(dataset_raw)

    # Validate paths
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    if not plans_path.exists():
        raise FileNotFoundError(f"Plans file not found: {plans_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    # Get device
    device_str = getattr(cfg.model.nnunet, "device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    logger.info("Loading nnUNet pretrained model...")
    logger.info("  Checkpoint: %s", checkpoint_path)
    logger.info("  Plans: %s", plans_path)
    logger.info("  Dataset: %s", dataset_path)
    logger.info("  Device: %s", device)

    # Load configuration files
    plans = load_json(str(plans_path))
    dataset_json = load_json(str(dataset_path))

    # Create plans manager
    plans_manager = PlansManager(plans)

    # Load checkpoint
    logger.info("Loading checkpoint weights...")
    checkpoint = torch.load(
        str(checkpoint_path), map_location=torch.device("cpu"), weights_only=False
    )

    # Extract trainer information
    trainer_name = checkpoint["trainer_name"]
    configuration_name = checkpoint["init_args"]["configuration"]

    # Get configuration manager
    configuration_manager = plans_manager.get_configuration(configuration_name)

    # Determine number of input channels
    num_input_channels = determine_num_input_channels(
        plans_manager, configuration_manager, dataset_json
    )

    # Override if specified in config
    if hasattr(cfg.model, "in_channels") and cfg.model.in_channels is not None:
        num_input_channels = cfg.model.in_channels

    # Find trainer class
    logger.info("Building network architecture...")
    trainer_class = recursive_find_python_class(
        os.path.join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
        trainer_name,
        "nnunetv2.training.nnUNetTrainer",
    )

    if trainer_class is None:
        raise RuntimeError(
            f"Cannot find trainer class: {trainer_name}\n"
            f"This checkpoint may require a custom trainer implementation."
        )

    # Build network
    network = trainer_class.build_network_architecture(
        configuration_manager.network_arch_class_name,
        configuration_manager.network_arch_init_kwargs,
        configuration_manager.network_arch_init_kwargs_req_import,
        num_input_channels,
        plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
        enable_deep_supervision=False,  # Disable for inference
    )

    # Load weights
    logger.info("Loading model weights...")
    network.load_state_dict(checkpoint["network_weights"])

    # Move to device
    network = network.to(device)

    spatial_dims = getattr(cfg.model.nnunet, "spatial_dims", None)
    if spatial_dims is None:
        patch_size = getattr(configuration_manager, "patch_size", None)
        if patch_size is not None:
            spatial_dims = len(tuple(patch_size))
    if spatial_dims not in (2, 3):
        raise ValueError(
            "Could not determine nnUNet spatial dimensions. "
            "Set model.nnunet.spatial_dims to 2 or 3."
        )

    # Check if model was trained with deep supervision
    # (we disable it for inference, but track for metadata)
    trained_with_deep_supervision = checkpoint.get("deep_supervision", False)

    num_heads = plans_manager.get_label_manager(dataset_json).num_segmentation_heads
    logger.info("Model loaded successfully!")
    logger.info("  Trainer: %s", trainer_name)
    logger.info("  Configuration: %s", configuration_name)
    logger.info("  Spatial dimensions: %sD", spatial_dims)
    logger.info("  Input channels: %s", num_input_channels)
    logger.info("  Output heads: %s", num_heads)
    logger.info("  Trained with deep supervision: %s", trained_with_deep_supervision)

    # Create wrapper
    wrapper = nnUNetWrapper(
        network=network,
        plans_manager=plans_manager,
        configuration_manager=configuration_manager,
        dataset_json=dataset_json,
        spatial_dims=spatial_dims,
        supports_deep_supervision=trained_with_deep_supervision,
    )

    return wrapper


__all__ = [
    "nnUNetWrapper",
    "build_nnunet",
]
