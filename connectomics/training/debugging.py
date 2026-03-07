"""
Debugging utilities for PyTorch Connectomics.

This module implements:
- NaN/Inf detection in activations
- NaN/Inf detection in parameters and gradients
- Forward hook management for intermediate layer inspection
- Debug statistics collection and reporting
"""

from __future__ import annotations
import logging
import pdb
from typing import Dict, List, Any, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class NaNDetectionHook:
    """
    Forward hook that checks layer outputs for NaN/Inf values.

    Attaches to a layer and checks its output after each forward pass.
    If NaN/Inf detected, prints diagnostics and optionally enters debugger.

    Args:
        layer_name: Name of the layer this hook is attached to
        debug_on_nan: If True, call pdb.set_trace() when NaN detected
        verbose: If True, print statistics for every forward pass
        collect_stats: If True, collect activation statistics
    """

    def __init__(
        self,
        layer_name: str,
        debug_on_nan: bool = True,
        verbose: bool = False,
        collect_stats: bool = True,
    ):
        self.layer_name = layer_name
        self.debug_on_nan = debug_on_nan
        self.verbose = verbose
        self.collect_stats = collect_stats

        # Statistics storage
        self.stats: Dict[str, Any] = {
            "forward_count": 0,
            "nan_count": 0,
            "inf_count": 0,
            "last_min": None,
            "last_max": None,
            "last_mean": None,
            "last_std": None,
        }

    def __call__(
        self,
        module: nn.Module,
        inputs: Tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ):
        """Hook function called after layer forward pass."""
        self.stats["forward_count"] += 1

        # Handle different output types
        if isinstance(output, dict):
            tensors_to_check = [v for v in output.values() if isinstance(v, torch.Tensor)]
        elif isinstance(output, (list, tuple)):
            tensors_to_check = [t for t in output if isinstance(t, torch.Tensor)]
        else:
            tensors_to_check = [output]

        # Check each output tensor
        for i, tensor in enumerate(tensors_to_check):
            if not isinstance(tensor, torch.Tensor):
                continue

            has_nan = torch.isnan(tensor).any().item()
            has_inf = torch.isinf(tensor).any().item()

            if has_nan:
                self.stats["nan_count"] += 1
            if has_inf:
                self.stats["inf_count"] += 1

            if self.collect_stats:
                with torch.no_grad():
                    self.stats["last_min"] = tensor.min().item()
                    self.stats["last_max"] = tensor.max().item()
                    self.stats["last_mean"] = tensor.mean().item()
                    self.stats["last_std"] = tensor.std().item()

            if self.verbose and not (has_nan or has_inf):
                suffix = f"[{i}]" if len(tensors_to_check) > 1 else ""
                print(
                    f"  [OK] {self.layer_name}{suffix}: "
                    f"shape={tuple(tensor.shape)}, "
                    f"min={self.stats['last_min']:.4f}, "
                    f"max={self.stats['last_max']:.4f}, "
                    f"mean={self.stats['last_mean']:.4f}"
                )

            if has_nan or has_inf:
                issue_type = "NaN" if has_nan else "Inf"
                suffix = f"[{i}]" if len(tensors_to_check) > 1 else ""

                print(f"\n{'=' * 80}")
                print(f"WARNING: {issue_type} DETECTED IN LAYER OUTPUT!")
                print(f"{'=' * 80}")
                print(f"Layer: {self.layer_name}{suffix}")
                print(f"Module type: {module.__class__.__name__}")
                print(f"Output shape: {tuple(tensor.shape)}")
                print(f"Forward pass count: {self.stats['forward_count']}")

                print("\nOutput Statistics:")
                print(f"   Min: {self.stats['last_min']}")
                print(f"   Max: {self.stats['last_max']}")
                print(f"   Mean: {self.stats['last_mean']}")
                print(f"   Std: {self.stats['last_std']}")
                print(f"   NaN count: {torch.isnan(tensor).sum().item()} / {tensor.numel()}")
                print(f"   Inf count: {torch.isinf(tensor).sum().item()} / {tensor.numel()}")

                print("\nInput Statistics:")
                for idx, inp in enumerate(inputs):
                    if isinstance(inp, torch.Tensor):
                        in_suffix = f"[{idx}]" if len(inputs) > 1 else ""
                        print(f"   Input{in_suffix} shape: {tuple(inp.shape)}")
                        input_range = f"[{inp.min().item():.4f}, {inp.max().item():.4f}]"
                        print(f"   Input{in_suffix} range: {input_range}")
                        print(f"   Input{in_suffix} has NaN: {torch.isnan(inp).any().item()}")
                        print(f"   Input{in_suffix} has Inf: {torch.isinf(inp).any().item()}")

                print(f"\n{'=' * 80}")

                if self.debug_on_nan:
                    print("\nEntering debugger...")
                    print("Available variables:")
                    print("  - module: The layer that produced NaN")
                    print("  - inputs: Layer inputs (tuple)")
                    print("  - output: Layer output (NaN detected here)")
                    print("  - tensor: The specific output tensor with NaN")
                    print("  - self: Hook object with statistics")
                    print("\nUse 'c' to continue, 'q' to quit, 'up' to go up stack\n")
                    pdb.set_trace()

                break


class NaNDetectionHookManager:
    """
    Manager for attaching NaN detection hooks to all layers in a model.

    Args:
        model: PyTorch model to attach hooks to
        debug_on_nan: If True, enter pdb when NaN detected
        verbose: If True, print statistics for every layer
        collect_stats: If True, collect activation statistics
        layer_types: Tuple of layer types to hook (default: all common layers)

    Example:
        >>> manager = NaNDetectionHookManager(model, debug_on_nan=True)
        >>> output = model(input)  # Will stop at first NaN-producing layer
        >>> stats = manager.get_stats()
        >>> manager.remove_hooks()
    """

    def __init__(
        self,
        model: nn.Module,
        debug_on_nan: bool = True,
        verbose: bool = False,
        collect_stats: bool = True,
        layer_types: Optional[Tuple] = None,
    ):
        self.model = model
        self.debug_on_nan = debug_on_nan
        self.verbose = verbose
        self.collect_stats = collect_stats

        if layer_types is None:
            self.layer_types = (
                nn.Conv1d, nn.Conv2d, nn.Conv3d,
                nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
                nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                nn.GroupNorm, nn.LayerNorm, nn.Linear,
                nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.ELU, nn.GELU,
                nn.Sigmoid, nn.Tanh, nn.Softmax,
                nn.Dropout, nn.Dropout2d, nn.Dropout3d,
                nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
                nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
                nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d,
                nn.Upsample,
            )
        else:
            self.layer_types = layer_types

        self.hooks: Dict[str, NaNDetectionHook] = {}
        self.hook_handles: List[Any] = []

        self._attach_hooks()

    def _attach_hooks(self):
        """Attach hooks to all matching layers in the model."""
        print("Attaching NaN detection hooks...")

        for name, module in self.model.named_modules():
            if name == "":
                continue
            if isinstance(module, self.layer_types):
                hook = NaNDetectionHook(
                    layer_name=name,
                    debug_on_nan=self.debug_on_nan,
                    verbose=self.verbose,
                    collect_stats=self.collect_stats,
                )
                handle = module.register_forward_hook(hook)
                self.hooks[name] = hook
                self.hook_handles.append(handle)

        print(f"   Attached hooks to {len(self.hooks)} layers")
        if self.verbose:
            preview = ", ".join(list(self.hooks.keys())[:5])
            suffix = "..." if len(self.hooks) > 5 else ""
            print(f"   Monitoring: {preview}{suffix}")

    def remove_hooks(self):
        """Remove all hooks from the model."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        print(f"Removed {len(self.hooks)} hooks")

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics from all hooks."""
        return {name: hook.stats for name, hook in self.hooks.items()}

    def print_summary(self):
        """Print summary of hook statistics."""
        print(f"\n{'=' * 80}")
        print("NaN Detection Hook Summary")
        print(f"{'=' * 80}")

        total_nan = sum(hook.stats["nan_count"] for hook in self.hooks.values())
        total_inf = sum(hook.stats["inf_count"] for hook in self.hooks.values())

        print(f"Total layers monitored: {len(self.hooks)}")
        print(f"Total NaN detections: {total_nan}")
        print(f"Total Inf detections: {total_inf}")

        if total_nan > 0 or total_inf > 0:
            print("\nWARNING: Layers with NaN/Inf:")
            for name, hook in self.hooks.items():
                if hook.stats["nan_count"] > 0 or hook.stats["inf_count"] > 0:
                    print(
                        f"   {name}: NaN={hook.stats['nan_count']}, "
                        f"Inf={hook.stats['inf_count']}"
                    )
        else:
            print("\nNo NaN/Inf detected in any layer")

        print(f"{'=' * 80}\n")

    def reset_stats(self):
        """Reset statistics for all hooks."""
        for hook in self.hooks.values():
            hook.stats = {
                "forward_count": 0,
                "nan_count": 0,
                "inf_count": 0,
                "last_min": None,
                "last_max": None,
                "last_mean": None,
                "last_std": None,
            }

    def __del__(self):
        """Cleanup hooks on deletion."""
        if self.hook_handles:
            self.remove_hooks()


class DebugManager:
    """
    Manager for debugging operations including NaN detection.

    This class handles:
    - Forward hooks for NaN/Inf detection in layer outputs
    - Parameter and gradient inspection
    - Statistics collection and reporting
    - Interactive debugging support (pdb integration)

    Args:
        model: PyTorch model to debug (nn.Module)
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._hook_manager: Optional[NaNDetectionHookManager] = None

    def enable_nan_hooks(
        self,
        debug_on_nan: bool = True,
        verbose: bool = False,
        layer_types: Optional[Tuple] = None,
    ) -> NaNDetectionHookManager:
        """
        Enable forward hooks to detect NaN in intermediate layer outputs.

        This attaches hooks to all layers in the model that will check for NaN/Inf
        in layer outputs during forward pass. When NaN is detected, it will print
        diagnostics and optionally enter the debugger.

        Useful for debugging in pdb:
            (Pdb) pl_module.enable_nan_hooks()
            (Pdb) outputs = pl_module(batch['image'])
            # Will stop at first layer producing NaN

        Args:
            debug_on_nan: If True, enter pdb when NaN detected (default: True)
            verbose: If True, print stats for every layer (slow, default: False)
            layer_types: Tuple of layer types to hook (default: all common layers)

        Returns:
            NaNDetectionHookManager instance
        """
        if self._hook_manager is not None:
            logger.warning("Hooks already enabled. Call disable_nan_hooks() first.")
            return self._hook_manager

        self._hook_manager = NaNDetectionHookManager(
            model=self.model,
            debug_on_nan=debug_on_nan,
            verbose=verbose,
            collect_stats=True,
            layer_types=layer_types,
        )

        return self._hook_manager

    def disable_nan_hooks(self):
        """
        Disable forward hooks for NaN detection.

        Removes all hooks that were attached by enable_nan_hooks().
        """
        if self._hook_manager is not None:
            self._hook_manager.remove_hooks()
            self._hook_manager = None
        else:
            logger.warning("No hooks to remove.")

    def get_hook_stats(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Get statistics from NaN detection hooks.

        Returns:
            Dictionary mapping layer names to their statistics, or None if hooks not enabled
        """
        if self._hook_manager is not None:
            return self._hook_manager.get_stats()
        else:
            logger.warning("Hooks not enabled. Call enable_nan_hooks() first.")
            return None

    def print_hook_summary(self):
        """
        Print summary of NaN detection hook statistics.

        Shows which layers detected NaN/Inf and how many times.
        """
        if self._hook_manager is not None:
            self._hook_manager.print_summary()
        else:
            logger.warning("Hooks not enabled. Call enable_nan_hooks() first.")

    def check_for_nan(self, check_grads: bool = True, verbose: bool = True) -> dict:
        """
        Debug utility to check for NaN/Inf in model parameters and gradients.

        Useful when debugging in pdb. Call as: pl_module.check_for_nan()

        Args:
            check_grads: Also check gradients
            verbose: Print detailed information

        Returns:
            Dictionary with NaN/Inf information
        """
        nan_params = []
        inf_params = []
        nan_grads = []
        inf_grads = []

        for name, param in self.model.named_parameters():
            # Check parameters
            if torch.isnan(param).any():
                nan_params.append((name, param.shape))
                if verbose:
                    logger.warning("NaN in parameter: %s, shape=%s", name, param.shape)
            if torch.isinf(param).any():
                inf_params.append((name, param.shape))
                if verbose:
                    logger.warning("Inf in parameter: %s, shape=%s", name, param.shape)

            # Check gradients
            if check_grads and param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_grads.append((name, param.grad.shape))
                    if verbose:
                        logger.warning("NaN in gradient: %s, shape=%s", name, param.grad.shape)
                if torch.isinf(param.grad).any():
                    inf_grads.append((name, param.grad.shape))
                    if verbose:
                        logger.warning("Inf in gradient: %s, shape=%s", name, param.grad.shape)

        result = {
            "nan_params": nan_params,
            "inf_params": inf_params,
            "nan_grads": nan_grads,
            "inf_grads": inf_grads,
            "has_nan": len(nan_params) > 0 or len(nan_grads) > 0,
            "has_inf": len(inf_params) > 0 or len(inf_grads) > 0,
        }

        if verbose:
            if not result["has_nan"] and not result["has_inf"]:
                logger.info("No NaN/Inf found in parameters or gradients")
            else:
                logger.warning(
                    "Summary: NaN parameters: %d, Inf parameters: %d, "
                    "NaN gradients: %d, Inf gradients: %d",
                    len(nan_params), len(inf_params),
                    len(nan_grads), len(inf_grads),
                )

        return result
