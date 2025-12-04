"""
PyTorch forward hooks for debugging NaN/Inf in intermediate layer outputs.

This module provides tools to attach hooks to model layers that check for
NaN/Inf values during forward pass, helping identify exactly which layer
first produces invalid values.

Usage:
    # In pdb when NaN detected:
    hook_manager = NaNDetectionHookManager(model, debug_on_nan=True)
    outputs = model(inputs)  # Will stop at first NaN-producing layer

    # Or from LightningModule:
    pl_module.enable_nan_hooks()
    outputs = pl_module(batch['image'])
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import pdb
import torch
import torch.nn as nn


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
            # For models with dictionary outputs (e.g., deep supervision)
            tensors_to_check = [v for v in output.values() if isinstance(v, torch.Tensor)]
        elif isinstance(output, (list, tuple)):
            # For models with multiple outputs
            tensors_to_check = [t for t in output if isinstance(t, torch.Tensor)]
        else:
            # Single tensor output
            tensors_to_check = [output]

        # Check each output tensor
        for i, tensor in enumerate(tensors_to_check):
            if not isinstance(tensor, torch.Tensor):
                continue

            # Check for NaN/Inf
            has_nan = torch.isnan(tensor).any().item()
            has_inf = torch.isinf(tensor).any().item()

            if has_nan:
                self.stats["nan_count"] += 1
            if has_inf:
                self.stats["inf_count"] += 1

            # Collect statistics
            if self.collect_stats:
                with torch.no_grad():
                    self.stats["last_min"] = tensor.min().item()
                    self.stats["last_max"] = tensor.max().item()
                    self.stats["last_mean"] = tensor.mean().item()
                    self.stats["last_std"] = tensor.std().item()

            # Print verbose output
            if self.verbose and not (has_nan or has_inf):
                suffix = f"[{i}]" if len(tensors_to_check) > 1 else ""
                print(
                    f"  ✓ {self.layer_name}{suffix}: "
                    f"shape={tuple(tensor.shape)}, "
                    f"min={self.stats['last_min']:.4f}, "
                    f"max={self.stats['last_max']:.4f}, "
                    f"mean={self.stats['last_mean']:.4f}"
                )

            # Handle NaN/Inf detection
            if has_nan or has_inf:
                issue_type = "NaN" if has_nan else "Inf"
                suffix = f"[{i}]" if len(tensors_to_check) > 1 else ""

                print(f"\n{'=' * 80}")
                print(f"⚠️  {issue_type} DETECTED IN LAYER OUTPUT!")
                print(f"{'=' * 80}")
                print(f"Layer: {self.layer_name}{suffix}")
                print(f"Module type: {module.__class__.__name__}")
                print(f"Output shape: {tuple(tensor.shape)}")
                print(f"Forward pass count: {self.stats['forward_count']}")

                # Print statistics
                print("\n📊 Output Statistics:")
                print(f"   Min: {self.stats['last_min']}")
                print(f"   Max: {self.stats['last_max']}")
                print(f"   Mean: {self.stats['last_mean']}")
                print(f"   Std: {self.stats['last_std']}")
                print(f"   NaN count: {torch.isnan(tensor).sum().item()} / {tensor.numel()}")
                print(f"   Inf count: {torch.isinf(tensor).sum().item()} / {tensor.numel()}")

                # Check inputs
                print("\n🔍 Input Statistics:")
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
                    print("\n🐛 Entering debugger...")
                    print("Available variables:")
                    print("  - module: The layer that produced NaN")
                    print("  - inputs: Layer inputs (tuple)")
                    print("  - output: Layer output (NaN detected here)")
                    print("  - tensor: The specific output tensor with NaN")
                    print("  - self: Hook object with statistics")
                    print("\nUse 'c' to continue, 'q' to quit, 'up' to go up stack\n")
                    pdb.set_trace()

                # Only stop at first NaN-producing layer
                break


class NaNDetectionHookManager:
    """
    Manager for attaching NaN detection hooks to all layers in a model.

    This class makes it easy to enable/disable NaN checking across an entire
    model hierarchy.

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

        # Default layer types to hook
        if layer_types is None:
            self.layer_types = (
                nn.Conv1d,
                nn.Conv2d,
                nn.Conv3d,
                nn.ConvTranspose1d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
                nn.InstanceNorm1d,
                nn.InstanceNorm2d,
                nn.InstanceNorm3d,
                nn.GroupNorm,
                nn.LayerNorm,
                nn.Linear,
                nn.ReLU,
                nn.LeakyReLU,
                nn.PReLU,
                nn.ELU,
                nn.GELU,
                nn.Sigmoid,
                nn.Tanh,
                nn.Softmax,
                nn.Dropout,
                nn.Dropout2d,
                nn.Dropout3d,
                nn.MaxPool1d,
                nn.MaxPool2d,
                nn.MaxPool3d,
                nn.AvgPool1d,
                nn.AvgPool2d,
                nn.AvgPool3d,
                nn.AdaptiveAvgPool1d,
                nn.AdaptiveAvgPool2d,
                nn.AdaptiveAvgPool3d,
                nn.Upsample,
            )
        else:
            self.layer_types = layer_types

        # Storage for hooks and their handles
        self.hooks: Dict[str, NaNDetectionHook] = {}
        self.hook_handles: List[Any] = []

        # Attach hooks
        self._attach_hooks()

    def _attach_hooks(self):
        """Attach hooks to all matching layers in the model."""
        print("🔗 Attaching NaN detection hooks...")

        for name, module in self.model.named_modules():
            # Skip the root module
            if name == "":
                continue

            # Only hook specified layer types
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
        print(f"🔓 Removed {len(self.hooks)} hooks")

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics from all hooks.

        Returns:
            Dictionary mapping layer names to their statistics
        """
        return {name: hook.stats for name, hook in self.hooks.items()}

    def print_summary(self):
        """Print summary of hook statistics."""
        print(f"\n{'=' * 80}")
        print("📊 NaN Detection Hook Summary")
        print(f"{'=' * 80}")

        total_nan = sum(hook.stats["nan_count"] for hook in self.hooks.values())
        total_inf = sum(hook.stats["inf_count"] for hook in self.hooks.values())

        print(f"Total layers monitored: {len(self.hooks)}")
        print(f"Total NaN detections: {total_nan}")
        print(f"Total Inf detections: {total_inf}")

        if total_nan > 0 or total_inf > 0:
            print("\n⚠️  Layers with NaN/Inf:")
            for name, hook in self.hooks.items():
                if hook.stats["nan_count"] > 0 or hook.stats["inf_count"] > 0:
                    print(
                        f"   {name}: NaN={hook.stats['nan_count']}, Inf={hook.stats['inf_count']}"
                    )
        else:
            print("\n✅ No NaN/Inf detected in any layer")

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


__all__ = [
    "NaNDetectionHook",
    "NaNDetectionHookManager",
]
