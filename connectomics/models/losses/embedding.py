"""Dense instance embeddings with the DeepEM L1 mean loss."""

from __future__ import annotations

import cc3d
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class EmbeddingMeanLoss(nn.Module):
    """Object-balanced pull, inter-object push, and mean-norm regularization.

    Based on ZettaAI/DeepEM ``feature/sr-zero-pad/deepem/loss/mean.py``.
    Unlike DeepEM, average non-empty batch samples, always intersect objects
    with the validity mask, ignore non-positive IDs, and compute connected
    components inside the loss (26-connectivity by default). Components with
    the same parent ID are excluded from push, but still count in its ordered
    pair denominator. There is no background-as-object mode.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.001,
        delta_v: float = 0.0,
        delta_d: float = 1.5,
        recompute_ext: bool = True,
        connectivity: int = 26,
        pair_chunk_size: int = 1024,
        pair_checkpoint: bool = False,
    ):
        super().__init__()
        if connectivity not in (6, 18, 26):
            raise ValueError("connectivity must be 6, 18, or 26")
        if not delta_d > 0 or not delta_v >= 0:
            raise ValueError("delta_d must be positive and delta_v non-negative")
        if not isinstance(pair_chunk_size, int) or pair_chunk_size < 1:
            raise ValueError("pair_chunk_size must be an integer >= 1")
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.recompute_ext = recompute_ext
        self.connectivity = connectivity
        self.pair_chunk_size = pair_chunk_size
        self.pair_checkpoint = pair_checkpoint

    def forward(self, pred, target, mask=None, gt_seg=None):
        """Consume ``[B,D,Z,Y,X]`` embeddings and raw GT; target is unused."""
        if pred.ndim != 5:
            raise ValueError("pred must have shape [B,D,Z,Y,X]")
        if gt_seg is None:
            raise ValueError("EmbeddingMeanLoss requires data.label_transform.emit_gt_seg: true")
        ids = torch.as_tensor(gt_seg, device=pred.device).detach()
        if ids.ndim == 5 and ids.shape[1] == 1:
            ids = ids[:, 0]
        expected = (pred.shape[0], *pred.shape[2:])
        if ids.ndim != 4 or tuple(ids.shape) != expected:
            raise ValueError("gt_seg must be [B,1,Z,Y,X] or [B,Z,Y,X] matching pred")
        if mask is not None:
            mask = torch.as_tensor(mask, device=pred.device).detach().bool()
            if mask.ndim == 5:
                mask = mask.all(dim=1)
            if mask.ndim != 4 or tuple(mask.shape) != expected:
                raise ValueError("mask must be [B,C,Z,Y,X] or [B,Z,Y,X] matching pred")
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred = pred.to(torch.float64 if pred.dtype == torch.float64 else torch.float32)
            ids = (ids.round() if ids.is_floating_point() else ids).long()
            losses = []
            for b in range(pred.shape[0]):
                valid = ids[b] > 0
                if mask is not None:
                    valid = valid & mask[b]
                if not valid.any():
                    continue
                labels = ids[b][valid]
                if self.recompute_ext:
                    volume = torch.where(valid, ids[b], 0).cpu().numpy().astype(np.uint64)
                    components = cc3d.connected_components(volume, connectivity=self.connectivity)
                    labels = torch.as_tensor(components.astype(np.int64), device=pred.device)[valid]
                obj, inv = torch.unique(labels, return_inverse=True)
                n = len(obj)
                idx = valid.flatten().nonzero().squeeze(1)
                e = pred[b].reshape(pred.shape[1], -1).index_select(1, idx).T
                counts = torch.bincount(inv, minlength=n).to(pred.dtype)
                means = e.new_zeros(n, e.shape[1]).index_add_(0, inv, e) / counts[:, None]
                pull = (
                    (e - means.index_select(0, inv))
                    .abs()
                    .sum(1)
                    .sub(self.delta_v)
                    .clamp_min(0)
                    .square()
                )
                loss_int = (e.new_zeros(n).index_add_(0, inv, pull) / counts).mean()
                parents = obj
                if self.recompute_ext:
                    # Every voxel in a component has the same parent; repeated writes agree.
                    parents = torch.zeros_like(obj).scatter_(0, inv, ids[b][valid])
                loss_ext = means.sum() * 0
                if n > 1:
                    for start in range(0, n, self.pair_chunk_size):
                        end = min(start + self.pair_chunk_size, n)
                        if self.pair_checkpoint:
                            chunk = checkpoint(
                                self._pair_sum, means, parents, start, end, use_reentrant=False
                            )
                        else:
                            chunk = self._pair_sum(means, parents, start, end)
                        loss_ext = loss_ext + chunk
                    loss_ext = loss_ext / (n * (n - 1))
                loss_nrm = means.abs().sum(1).mean()
                losses.append(self.alpha * loss_int + self.beta * loss_ext + self.gamma * loss_nrm)
            return torch.stack(losses).mean() if losses else (pred * 0).sum()

    def _pair_sum(self, means, parents, start, end):
        distance = (means[start:end, None, :] - means[None, :, :]).abs().sum(-1)
        # Unique object IDs also exclude the diagonal when recompute_ext=False.
        keep = parents[start:end, None] != parents[None, :]
        return ((2 * self.delta_d - distance).clamp_min(0).square() * keep).sum()
