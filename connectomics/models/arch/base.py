"""
Base model interface for all architectures.

Provides a standard interface that all models should implement,
with explicit support for deep supervision.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Union

import torch
import torch.nn as nn


class ConnectomicsModel(nn.Module, ABC):
    """
    Base class for all connectomics models.

    Provides common interface for:
    - Forward pass (single or multi-scale outputs)
    - Deep supervision support
    - Model information

    All models in the architectures module should inherit from this class
    or at least implement the same interface.
    """

    def __init__(self):
        super().__init__()
        self.supports_deep_supervision = False
        self.output_scales = 1

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (B, C, D, H, W) for 3D or (B, C, H, W) for 2D

        Returns:
            For single-scale models:
                torch.Tensor: Output tensor of shape (B, num_classes, D, H, W)

            For deep supervision models:
                Dict[str, torch.Tensor]: Dictionary with keys:
                    - 'output': Main output (full resolution)
                    - 'ds_1': Deep supervision output at scale 1 (1/2 resolution)
                    - 'ds_2': Deep supervision output at scale 2 (1/4 resolution)
                    - 'ds_3': Deep supervision output at scale 3 (1/8 resolution)
                    - 'ds_4': Deep supervision output at scale 4 (1/16 resolution)
        """
        raise NotImplementedError

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and statistics.

        Returns:
            Dictionary with model metadata:
                - name: Model class name
                - deep_supervision: Whether model supports deep supervision
                - output_scales: Number of output scales
                - parameters: Total trainable parameters
                - trainable_parameters: Number of trainable parameters
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "name": self.__class__.__name__,
            "deep_supervision": self.supports_deep_supervision,
            "output_scales": self.output_scales,
            "parameters": total_params,
            "trainable_parameters": trainable_params,
        }

    def __repr__(self) -> str:
        """String representation of the model."""
        info = self.get_model_info()
        return (
            f"{info['name']}("
            f"parameters={info['parameters']:,}, "
            f"deep_supervision={info['deep_supervision']})"
        )


__all__ = ["ConnectomicsModel"]
